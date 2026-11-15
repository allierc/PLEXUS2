"""exp27 speed audit (2026-10-21), test 1: `mpm_strain[implementation: polar]`.

The viscoelastic (Maxwell) branch of `mpm_strain` pulls the principal stretches of F toward their geometric mean, with the memory kept per
substep a = exp(-dt / tau):

    F = U diag(s) V^T,   J_l = (s0 s1 s2)^(1/3),   F' = U diag(J_l + a (s - J_l)) V^T

which is, without the factorisation,

    F' = a F + (1 - a) J_l R,       R = U V^T the orthogonal polar factor of F,   J_l = |det F|^(1/3).

The default does the singular value decomposition with cuSOLVER's batched Jacobi solver on every viscoelastic point, 50 times a frame (52 ms
a call at 306k points on the RTX PRO 6000 of the cluster); `mpm_strain[warp]` does it with `wp.svd3`, which is fast (0.11 ms) but reconstructs F
only to about 1e-3 (measured against the default, 2026-10-21: worst difference of an entry of F 1.3e-3 of the largest entry). The polar factor
by Newton's iteration R <- (R + R^-T) / 2, the iteration the scatter of the same spec already runs, converges to float32 round-off, so this is
the same quantity as the default's to round-off and runs in one Warp kernel of one thread per point.
"""
from __future__ import annotations

import torch

from plexus.models.registry import register_operator
from plexus.operators.mpm_ops import HAVE_WARP, MPMStrainWarp, _const_any, _wp_launch, sub_dt

if HAVE_WARP:
    import warp as wp
    from plexus.operators.mpm_ops import polar_R

    @wp.kernel
    def strain_polar(C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
                     LIQ: wp.array(dtype=float), VIS: wp.array(dtype=float),
                     TAU: wp.array(dtype=float), OCC: wp.array(dtype=float),
                     dt: float, has_liq: int, has_vis: int, iters: int):
        p = wp.tid()
        if OCC[p] <= 0.0:                                     # dormant points do not deform
            return
        I3 = wp.identity(n=3, dtype=float)
        Fp = (I3 + dt * C[p]) * F[p]
        if has_liq == 1 and LIQ[p] > 0.0:                     # LIQUID: drop shape memory, keep volume
            Fp = I3 * wp.pow(wp.max(wp.determinant(Fp), 1.0e-6), 1.0 / 3.0)
        if has_vis == 1 and VIS[p] > 0.0:                     # VISCOELASTIC: a F + (1 - a) J_l R
            Jl = wp.pow(wp.max(wp.abs(wp.determinant(Fp)), 1.0e-6), 1.0 / 3.0)
            a = wp.exp(-dt / wp.max(TAU[p], 1.0e-6))
            Fp = a * Fp + ((1.0 - a) * Jl) * polar_R(Fp, iters)
        F[p] = Fp


@register_operator("mpm_strain", implementation="polar", family="mpm", set="particle", kind="lateral",
                   title="update material strain, Maxwell relaxation by the polar factor", equation=r"""$$\mathbf F \;\leftarrow\; (\mathbf I + \Delta t\,\mathbf C)\,\mathbf F
\qquad\text{(liquid: } \mathbf F\leftarrow J^{1/d}\mathbf I,\ J=\det\mathbf F\text{)}$$""")
class MPMStrainPolar(MPMStrainWarp):
    r"""`warp_polar` implementation of mpm_strain: the viscoelastic relaxation toward the polar factor of F in one Warp kernel; elastic and liquid points, no snow.
    particle -> particle: as mpm_strain; F <- a F + (1 - a) J^(1/3) R with R the polar factor of F by `iters` Newton steps.

        \mathbf F \;\leftarrow\; (\mathbf I + \Delta t\,\mathbf C)\,\mathbf F \qquad\text{(liquid: } \mathbf F\leftarrow J^{1/d}\mathbf I,\ J=\det\mathbf F\text{)}
        F              the deformation gradient, the identity for unstrained material
        I              the identity
        \Delta t       `dt_sub`, the substep
        C              the affine velocity gradient
        J              det F, the ratio of current to rest volume (`liquid_volume`: carried through F, or integrated as dJ/dt = J tr C)
        d              the dimension

    Reference: as `mpm_strain`; the polar decomposition by Newton iteration (Higham 1986, SIAM J. Sci. Stat. Comput. 7:1160-1174)."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"Updates each material point's deformation gradient $\mathbf F$ (the identity when unstrained) over a substep $\Delta "
        r"t$: $\mathbf F\leftarrow(\mathbf I+\Delta t\,\mathbf C)\,\mathbf F$, with $\mathbf C$ the point's affine "
        r"velocity-gradient matrix from the last gather. A viscoelastic point then relaxes toward the polar factor $\mathbf R$ "
        r"of $\mathbf F$, $\mathbf F\leftarrow a\,\mathbf F+(1-a)\,J^{1/3}\mathbf R$, with $\mathbf R$ found by `iters` Newton "
        r"steps, $J=\det\mathbf F$ and $a$ the weight kept from the old $\mathbf F$, so the cell forgets its deformation at the "
        r"rate of its relaxation time. Liquid points are reset to $\mathbf F=J^{1/d}\mathbf I$, keeping only their volume. One "
        r"Warp kernel; elastic and liquid points, no snow."
    )
    PARAM_UNITS = {'iters': 'count'}
    SYMBOLS = {'F': 'the deformation gradient, the identity for unstrained material', 'I': 'the identity', '\\Delta t': '`dt_sub`, the substep', 'C': 'the affine velocity gradient', 'J': 'det F, the ratio of current to rest volume (`liquid_volume`: carried through F, or integrated as dJ/dt = J tr C)', 'd': 'the dimension'}
    PARAM_ROLES = {'iters': 'newton_steps_of_the_polar_decomposition'}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.iters = int(params.get("iters", 8))

    def forward(self, H, mask=None):
        if not HAVE_WARP:
            raise RuntimeError("mpm_strain[polar] needs warp-lang")
        p = H.level(self.at); dev = p.state.device
        if p.F.shape[-1] != 3 or str(dev) == "cpu":
            raise RuntimeError(f"mpm_strain[polar] is 3D CUDA only (got dim={p.F.shape[-1]}, dev={dev})")
        if _const_any(self, "_c_is_snow", getattr(p, "is_snow", None)):
            raise RuntimeError("mpm_strain[polar] has no snow branch; use implementation: warp or default")
        has_vis = _const_any(self, "_c_is_visco", getattr(p, "is_visco", None))
        dt = sub_dt(H, self.dt_sub)
        liq = getattr(p, "is_liquid", None)
        if getattr(self, "_side_polar", None) is None:        # run-constant, built once
            z = torch.zeros(p.n, device=dev)
            def _f(t):
                return t.float().contiguous() if t is not None else z
            self._side_polar = (_f(liq), _f(getattr(p, "is_visco", None)),
                                p.visco_tau.contiguous() if getattr(p, "visco_tau", None) is not None else torch.ones(p.n, device=dev))
        liqf, visf, tau = self._side_polar
        occ = p.occ.contiguous() if getattr(p, "occ", None) is not None else torch.ones(p.n, device=dev)
        _wp_launch(strain_polar, int(p.n), dev,
                   [wp.from_torch(p.C, dtype=wp.mat33), wp.from_torch(p.F, dtype=wp.mat33), wp.from_torch(liqf), wp.from_torch(visf),
                    wp.from_torch(tau), wp.from_torch(occ), float(dt), 1 if liq is not None else 0, int(bool(has_vis)), self.iters])
        return {}
