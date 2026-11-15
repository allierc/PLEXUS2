"""exp27 speed audit (2026-10-21), test 5a/5b: `mpm_scatter[warp_pp]` and `mpm_gather[warp_pp]`, the Warp scatter and gather on the
PER-CELL GRID BLOCKS.

The grid of exp27 is `per_parent: cell` (see `MPMGrid`): every cell owns a block of n_grid^3 = 24^3 nodes of side `cell_box` = 3.0, re-centred
on the cell's live centroid at the first scatter of each substep. The torch bodies of `mpm_scatter` / `mpm_gather` handle that with `_pp_frame`
(the block index of every point and the block origins) and `_pp_flat` (the node index offset by the block). The Warp kernels of mpm_ops
(`p2g_atomic`, `g2p`) have no block: they index one grid and would clamp every cell's points into the same 24^3 nodes. These two take the
block index of every point (`PID`, int64) and the origin of every block (`ORG`, [n_blocks, 3]) and put each point on its own block:

    x_local = x - ORG[PID[p]]                         what `bspline(X - _po[_pi], ...)` is given
    node    = PID[p] * n_local + (gi * ny + gj) * nz + gk       what `_pp_flat` adds

`_pp_frame` stays the torch code of the default (a centroid by `index_add_` over the live points, once per substep): it is the same call, so
the block origins are the default's. Everything else of the default body is reproduced for the specification's operators: the body force and
the Stokes drag on the velocity, the fixed-corotated stress with the rotation by Newton's iteration written as the default writes it (the
cofactor form, with the determinant floor of 1e-12), the `extra_stress` of `mpm_viscosity`, snow hardening and the liquid colour as in
`p2g_atomic`; in the gather the speed cap, the position clamp to [2 dx, cell_box - 2 dx] in the block frame, and the frozen dormant points.

What is NOT done is refused loudly: `store_stress`, a grid `origin`, a periodic box, a residual-stress rest tensor, `wall_damp != 1`, 2D.
"""
import torch

from plexus.models.registry import register_operator
from plexus.operators.mpm_ops import (HAVE_WARP, MPMGather, MPMScatter, _ancestor_accel, _const_any, _hand_body_force_to_grid, _no_origin,
                                      _pp_frame, _wp_launch, sub_dt)

FIXED_SCALE = float(2 ** 40)         # the step of the fixed-point deposit under strict determinism: 2^-40 of a mass or a momentum

if HAVE_WARP:
    import warp as wp

    @wp.func
    def polar_cofactor(F: wp.mat33, iters: int) -> wp.mat33:
        """The rotation of F = R S by Newton's iteration R <- (R + R^-T) / 2, with R^-T as the cofactor matrix over the determinant, as
        `mpm_ops._polar_higham` writes it in 3D: column k of the cofactor is cross(column k+1, column k+2) of R, the determinant is
        column 0 of R dotted with column 0 of the cofactor, floored at 1e-12 in absolute value."""
        R = F
        for _ in range(iters):
            a0 = wp.vec3(R[0, 0], R[1, 0], R[2, 0])
            a1 = wp.vec3(R[0, 1], R[1, 1], R[2, 1])
            a2 = wp.vec3(R[0, 2], R[1, 2], R[2, 2])
            c0 = wp.cross(a1, a2)
            c1 = wp.cross(a2, a0)
            c2 = wp.cross(a0, a1)
            det = a0[0] * c0[0] + a0[1] * c0[1] + a0[2] * c0[2]
            if wp.abs(det) < 1.0e-12:
                det = 1.0e-12
            R = wp.mat33(0.5 * (R[0, 0] + c0[0] / det), 0.5 * (R[0, 1] + c1[0] / det), 0.5 * (R[0, 2] + c2[0] / det),
                         0.5 * (R[1, 0] + c0[1] / det), 0.5 * (R[1, 1] + c1[1] / det), 0.5 * (R[1, 2] + c2[1] / det),
                         0.5 * (R[2, 0] + c0[2] / det), 0.5 * (R[2, 1] + c1[2] / det), 0.5 * (R[2, 2] + c2[2] / det))
        return R

    @wp.func
    def bspline_w(f: float, i: int) -> float:
        """The quadratic B-spline weight of the i-th of the 3 nodes along one axis, at the fractional offset f of the point."""
        w = float(0.0)
        if i == 0:
            w = 0.5 * (1.5 - f) * (1.5 - f)
        elif i == 1:
            w = 0.75 - (f - 1.0) * (f - 1.0)
        else:
            w = 0.5 * (f - 0.5) * (f - 0.5)
        return w

    @wp.kernel
    def p2g_pp(STATE: wp.array2d(dtype=float), pa: int, va: int,
               C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
               MASS: wp.array(dtype=float), MU: wp.array(dtype=float), LA: wp.array(dtype=float), PVOL: wp.array(dtype=float),
               AEXT: wp.array(dtype=wp.vec3), JP: wp.array(dtype=float), SNW: wp.array(dtype=float), LIQ: wp.array(dtype=float),
               OCC: wp.array(dtype=float), XTR: wp.array(dtype=wp.mat33),
               PID: wp.array(dtype=wp.int64), ORG: wp.array(dtype=wp.vec3),
               GM: wp.array(dtype=float), GMV: wp.array(dtype=wp.vec3), GC: wp.array(dtype=float),
               GMI: wp.array(dtype=wp.int64), GMVI: wp.array(dtype=wp.int64), scale: float,
               ngx: int, ngy: int, ngz: int, n_local: int,
               dx: float, inv_dx: float, dt: float, drag: float, iters: int,
               has_snw: int, has_liq: int, has_xtr: int, fixed: int):
        p = wp.tid()
        if OCC[p] <= 0.0:                                  # dormant points deposit nothing (the default masks their weights)
            return
        blk = int(PID[p])
        o = ORG[blk]
        x = wp.vec3(STATE[p, pa + 0] - o[0], STATE[p, pa + 1] - o[1], STATE[p, pa + 2] - o[2])   # the point in its block's frame
        v0 = wp.vec3(STATE[p, va + 0], STATE[p, va + 1], STATE[p, va + 2])
        v = v0 + dt * (AEXT[p] - drag * v0)                # body force and Stokes drag
        mass = MASS[p]
        Fp = F[p]
        Cp = C[p]
        J = wp.determinant(Fp)
        R = polar_cofactor(Fp, iters)
        mu_p = MU[p]
        la_p = LA[p]
        if has_snw == 1 and SNW[p] > 0.0:
            h = wp.exp(wp.clamp(10.0 * (1.0 - JP[p]), -6.0, 6.0))
            mu_p = mu_p * h
            la_p = la_p * h
        S = 2.0 * mu_p * ((Fp - R) * wp.transpose(Fp)) + wp.identity(n=3, dtype=float) * (la_p * J * (J - 1.0))
        if has_xtr == 1:
            S = S + XTR[p]
        S = S * ((-dt * 4.0 * inv_dx * inv_dx) * PVOL[p])
        affine = S + Cp * mass
        base = wp.vec3(wp.floor(x[0] * inv_dx - 0.5), wp.floor(x[1] * inv_dx - 0.5), wp.floor(x[2] * inv_dx - 0.5))
        fx = wp.vec3(x[0] * inv_dx - base[0], x[1] * inv_dx - base[1], x[2] * inv_dx - base[2])
        node0 = blk * n_local
        for i in range(3):
            wi = bspline_w(fx[0], i)
            gi = wp.clamp(int(base[0]) + i, 0, ngx - 1)
            for j in range(3):
                wj = bspline_w(fx[1], j)
                gj = wp.clamp(int(base[1]) + j, 0, ngy - 1)
                for k in range(3):
                    wk = bspline_w(fx[2], k)
                    gk = wp.clamp(int(base[2]) + k, 0, ngz - 1)
                    w = wi * wj * wk
                    idx = node0 + (gi * ngy + gj) * ngz + gk
                    dpos = wp.vec3((float(i) - fx[0]) * dx, (float(j) - fx[1]) * dx, (float(k) - fx[2]) * dx)
                    mom = mass * v + affine * dpos
                    if fixed == 1:
                        # FIXED POINT: the sum of integers does not depend on the order the threads add them in, so the grid is the same
                        # on every run. Each term is rounded to a multiple of 2^-40 (about 1e-12), far below a float32 term of this scale.
                        wp.atomic_add(GMI, idx, wp.int64(wp.floor(w * mass * scale + 0.5)))
                        wm = w * mom
                        wp.atomic_add(GMVI, 3 * idx + 0, wp.int64(wp.floor(wm[0] * scale + 0.5)))
                        wp.atomic_add(GMVI, 3 * idx + 1, wp.int64(wp.floor(wm[1] * scale + 0.5)))
                        wp.atomic_add(GMVI, 3 * idx + 2, wp.int64(wp.floor(wm[2] * scale + 0.5)))
                    else:
                        wp.atomic_add(GM, idx, w * mass)
                        wp.atomic_add(GMV, idx, w * mom)
                    if has_liq == 1:
                        wp.atomic_add(GC, idx, w * mass * LIQ[p])

    @wp.kernel
    def fixed_to_float(GMI: wp.array(dtype=wp.int64), GMVI: wp.array(dtype=wp.int64), inv_scale: float,
                       GM: wp.array(dtype=float), GMV: wp.array(dtype=wp.vec3)):
        i = wp.tid()
        GM[i] = float(GMI[i]) * inv_scale
        GMV[i] = wp.vec3(float(GMVI[3 * i + 0]) * inv_scale, float(GMVI[3 * i + 1]) * inv_scale, float(GMVI[3 * i + 2]) * inv_scale)

    @wp.kernel
    def g2p_pp(STATE: wp.array2d(dtype=float), pa: int, va: int,
               C: wp.array(dtype=wp.mat33), GV: wp.array(dtype=wp.vec3), OCC: wp.array(dtype=float),
               PID: wp.array(dtype=wp.int64), ORG: wp.array(dtype=wp.vec3),
               ngx: int, ngy: int, ngz: int, n_local: int,
               inv_dx: float, dt: float, vmax: float, lo: float, hi: float):
        p = wp.tid()
        blk = int(PID[p])
        o = ORG[blk]
        x = wp.vec3(STATE[p, pa + 0] - o[0], STATE[p, pa + 1] - o[1], STATE[p, pa + 2] - o[2])   # the point in its block's frame
        if OCC[p] <= 0.0:                                  # dormant: frozen, but written back through the block frame like the default
            STATE[p, pa + 0] = x[0] + o[0]
            STATE[p, pa + 1] = x[1] + o[1]
            STATE[p, pa + 2] = x[2] + o[2]
            return
        base = wp.vec3(wp.floor(x[0] * inv_dx - 0.5), wp.floor(x[1] * inv_dx - 0.5), wp.floor(x[2] * inv_dx - 0.5))
        fx = wp.vec3(x[0] * inv_dx - base[0], x[1] * inv_dx - base[1], x[2] * inv_dx - base[2])
        newv = wp.vec3(0.0, 0.0, 0.0)
        newC = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        node0 = blk * n_local
        for i in range(3):
            wi = bspline_w(fx[0], i)
            gi = wp.clamp(int(base[0]) + i, 0, ngx - 1)
            for j in range(3):
                wj = bspline_w(fx[1], j)
                gj = wp.clamp(int(base[1]) + j, 0, ngy - 1)
                for k in range(3):
                    wk = bspline_w(fx[2], k)
                    gk = wp.clamp(int(base[2]) + k, 0, ngz - 1)
                    w = wi * wj * wk
                    gv = GV[node0 + (gi * ngy + gj) * ngz + gk]
                    dpos = wp.vec3(float(i) - fx[0], float(j) - fx[1], float(k) - fx[2])
                    newv = newv + w * gv
                    newC = newC + w * wp.outer(gv, dpos)
        newC = newC * (4.0 * inv_dx)
        for a in range(3):                                 # torch.nan_to_num
            if newv[a] != newv[a]:
                newv[a] = 0.0
            for b in range(3):
                if newC[a, b] != newC[a, b]:
                    newC[a, b] = 0.0
        sp = wp.max(wp.length(newv), 1.0e-9)
        newv = newv * (wp.min(sp, vmax) / sp)              # the speed cap (a CFL velocity)
        xn = x + dt * newv
        for a in range(3):
            if xn[a] != xn[a]:
                xn[a] = 0.5
            xn[a] = wp.clamp(xn[a], lo, hi)
        STATE[p, pa + 0] = xn[0] + o[0]
        STATE[p, pa + 1] = xn[1] + o[1]
        STATE[p, pa + 2] = xn[2] + o[2]
        STATE[p, va + 0] = newv[0]
        STATE[p, va + 1] = newv[1]
        STATE[p, va + 2] = newv[2]
        C[p] = newC


def _pp_refuse(op, g, p, H, name):
    D = p.F.shape[-1]
    if not HAVE_WARP:
        raise RuntimeError(f"{name} needs warp-lang")
    if not getattr(g, "per_parent", None):
        raise ValueError(f"{name}: the grid {getattr(g, 'name', '?')!r} is not `per_parent`; use `implementation: warp` for a single grid")
    _no_origin(op, g)
    if D != 3 or str(p.state.device) == "cpu" or bool(getattr(H, "periodic", False)):
        raise RuntimeError(f"{name} is 3D, non-periodic, CUDA only")


@register_operator("mpm_scatter", implementation="warp_pp", family="mpm", set="particle", kind="exchange",
                   title="spread point mass onto the per-cell grid blocks, one atomic kernel", equation=r"""$$\boldsymbol\sigma = 2\mu(\mathbf F-\mathbf R)\mathbf F^{\!\top} + \lambda(J-1)J\,\mathbf I,
\qquad
\mathbf Q = -\frac{4\,\Delta t}{\Delta x^{2}}\,V\,\boldsymbol\sigma + m\,\mathbf C$$""")
class MPMScatterWarpPP(MPMScatter):
    r"""`warp_pp` implementation of mpm_scatter: the particle-to-grid transfer on a `per_parent` grid as one Warp kernel with atomics.
    particle -> field: as mpm_scatter, on the per-cell grid blocks (block index = the particle's parent). CUDA, 3D; not bit-reproducible (float atomics).

        \boldsymbol\sigma = 2\mu(\mathbf F-\mathbf R)\mathbf F^{\!\top} + \lambda(J-1)J\,\mathbf I, \qquad \mathbf Q = -\frac{4\,\Delta t}{\Delta x^{2}}\,V\,\boldsymbol\sigma + m\,\mathbf C
        \sigma         the Kirchhoff stress of the fixed-corotated law
        \mu, \lambda   the particle's Lame parameters, stress
        F              the deformation gradient
        R              its rotation (polar decomposition F = R S; `polar`, `polar_iters`)
        J              det F, the volume ratio
        I              the identity
        Q              the affine momentum matrix deposited with the particle: the stress impulse plus the APIC term
        \Delta t       `dt_sub`, the substep (stable below dx / sqrt(E/rho))
        \Delta x       the grid spacing
        V              the particle's rest volume
        m              its mass
        C              its affine velocity gradient (APIC)

    Reference: as `mpm_scatter`; the per-parent grid of `mpm_grid[per_parent]`."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"The particle-to-grid transfer. Each point computes its Kirchhoff stress $\boldsymbol\sigma=2\mu(\mathbf F-\mathbf "
        r"R)\mathbf F^{\!\top}+\lambda(J-1)J\,\mathbf I$ (the fixed-corotated law: $\mu$ and $\lambda$ its Lamé parameters, "
        r"$\mathbf R$ the rotation of $\mathbf F$, $J=\det\mathbf F$) and spreads its mass $m$ and its momentum $\mathbf "
        r"Q=-\tfrac{4\,\Delta t}{\Delta x^{2}}\,V\,\boldsymbol\sigma+m\,\mathbf C$ to the grid nodes around it ($V$ the point's "
        r"volume, $\mathbf C$ its affine matrix, $\Delta x$ the grid spacing). Every cell has its own grid block (the block is "
        r"the point's parent), so cells share no mass or momentum on the grid: they meet only through the contact and adhesion "
        r"operators. One Warp kernel with atomic adds, so not bit-reproducible."
    )
    PARAM_UNITS = {'a_max': 'acceleration', 'drag': 'rate', 'dt_sub': 'time', 'polar_iters': 'count', 'body_force': 'acceleration'}
    SYMBOLS = {'\\sigma': 'the Kirchhoff stress of the fixed-corotated law', '\\mu, \\lambda': "the particle's Lame parameters, stress", 'F': 'the deformation gradient', 'R': 'its rotation (polar decomposition F = R S; `polar`, `polar_iters`)', 'J': 'det F, the volume ratio', 'I': 'the identity', 'Q': 'the affine momentum matrix deposited with the particle: the stress impulse plus the APIC term', '\\Delta t': '`dt_sub`, the substep (stable below dx / sqrt(E/rho))', '\\Delta x': 'the grid spacing', 'V': "the particle's rest volume", 'm': 'its mass', 'C': 'its affine velocity gradient (APIC)'}

    MECHANISM_TAGS = ["particle_to_grid", "fixed_corotated_stress", "per_parent_blocks", "fused_kernel"]
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False

    def forward(self, H, mask=None):
        p = H.level(self.at); g = H.field(self.grid); dev = p.state.device
        _pp_refuse(self, g, p, H, "mpm_scatter[warp_pp]")
        if getattr(self, "store_stress", False):
            raise ValueError("mpm_scatter[warp_pp]: `store_stress: true` is not implemented (no per-point stress is formed); use the default body")
        if getattr(p, "F_res_inv", None) is not None:
            raise ValueError("mpm_scatter[warp_pp]: a residual-stress rest tensor `F_res_inv` is not implemented; use the default body")
        if self.polar != "higham":
            raise ValueError("mpm_scatter[warp_pp]: only `polar: higham` (the default) is implemented; use the default body for `polar: svd`")
        if self.body_force != "particle":
            raise ValueError("mpm_scatter[warp_pp]: `body_force: grid` is not implemented; use the default body")
        dt = sub_dt(H, self.dt_sub)
        D = 3
        pa, _ = p.state_schema["pos"]; va, _ = p.state_schema["vel"]
        a_ext = _ancestor_accel(H, p, self.a_max, D, dev)
        part_accel = getattr(H, "part_accel", None)
        if part_accel is not None:
            a_ext = a_ext + part_accel
        a_ext = (a_ext + torch.nan_to_num(H.delta(p.name))).contiguous()
        has_snw = _const_any(self, "_c_snow", getattr(p, "is_snow", None))
        has_liq = _const_any(self, "_c_liquid", getattr(p, "is_liquid", None))
        if getattr(self, "_sbuf", None) is None:                         # run-constant side arrays, built once
            z = torch.zeros(p.n, device=dev)
            def _mf(t):
                return t.float().contiguous() if t is not None else z
            self._sbuf = (_mf(getattr(p, "is_snow", None)), _mf(getattr(p, "is_liquid", None)),
                          torch.ones(p.n, device=dev) if getattr(p, "occ", None) is None else None)
        snw, liq, occ1 = self._sbuf
        if getattr(self, "colour_from_face", None) is not None:        # the colour channel as the per-face weight (MPMScatter)
            liq, has_liq = self._face_colour(p, H).float().contiguous(), True
        occ = occ1 if occ1 is not None else p.occ.contiguous()
        # the additive Kirchhoff stresses: active (per set, or the global slot of this set's size) and the viscous one of `mpm_viscosity`
        xtr = getattr(p, "act_stress", None)
        gact = getattr(H, "active_stress", None)
        if gact is not None and gact.shape[0] == p.n:
            xtr = gact if xtr is None else xtr + gact
        ext = getattr(p, "extra_stress", None)
        if ext is None:
            gx = getattr(H, "extra_stress", None)
            ext = gx if (gx is not None and gx.shape[0] == p.n) else None
        if ext is not None:
            xtr = ext if xtr is None else xtr + ext
        turg = getattr(p, "turgor", None)
        if turg is not None:                                              # tau - P I, as the default: folded into the additive stress
            eye = torch.eye(3, device=dev)
            xtr = -eye * turg[:, None, None] if xtr is None else xtr - eye * turg[:, None, None]
        has_xtr = xtr is not None
        xtr = xtr.contiguous() if has_xtr else torch.zeros(1, 3, 3, device=dev)
        _recentre = getattr(self, "_zeroes_grid", True)
        pidx, org = _pp_frame(H, g, p, p.get("pos"), getattr(p, "occ", None), recentre=_recentre)
        gm, gmv, gc = g.m, g.mv, g.c
        if _recentre:
            gm.zero_(); gmv.zero_(); gc.zero_()
        # UNDER `PLEXUS_STRICT_DETERMINISM` (torch.use_deterministic_algorithms) the deposit is a fixed-point sum, which atomics make
        # order-independent; the float atomics of the default mode add the 27 x points terms of a node in whatever order the threads arrive.
        fixed = bool(torch.are_deterministic_algorithms_enabled())
        if fixed:
            if has_liq:
                raise ValueError("mpm_scatter[warp_pp]: strict determinism with a liquid colour field is not implemented")
            if not _recentre:
                raise ValueError("mpm_scatter[warp_pp]: strict determinism with a second scatter into the same grid is not implemented")
            n = g.m.numel()
            if getattr(self, "_fixed_buf", None) is None or self._fixed_buf[0].numel() != n:
                self._fixed_buf = (torch.zeros(n, dtype=torch.int64, device=dev), torch.zeros(3 * n, dtype=torch.int64, device=dev))
            gmi, gmvi = self._fixed_buf
            gmi.zero_(); gmvi.zero_()
        else:
            gmi = gmvi = torch.zeros(1, dtype=torch.int64, device=dev)
        _wp_launch(p2g_pp, int(p.n), dev,
                   [wp.from_torch(p.state), int(pa), int(va),
                    wp.from_torch(p.C.contiguous(), dtype=wp.mat33), wp.from_torch(p.F.contiguous(), dtype=wp.mat33),
                    wp.from_torch(p.mass.contiguous()), wp.from_torch(p.mu.contiguous()), wp.from_torch(p.la.contiguous()),
                    wp.from_torch(p.p_vol.contiguous()), wp.from_torch(a_ext, dtype=wp.vec3),
                    wp.from_torch(p.Jp.contiguous()), wp.from_torch(snw), wp.from_torch(liq), wp.from_torch(occ),
                    wp.from_torch(xtr, dtype=wp.mat33),
                    wp.from_torch(pidx.contiguous()), wp.from_torch(org.contiguous(), dtype=wp.vec3),
                    wp.from_torch(gm), wp.from_torch(gmv.view(-1, 3), dtype=wp.vec3), wp.from_torch(gc),
                    wp.from_torch(gmi), wp.from_torch(gmvi), float(FIXED_SCALE),
                    int(g.shape[0]), int(g.shape[1]), int(g.shape[2]), int(g.n_local),
                    float(g.dx), float(g.inv_dx), float(dt), float(self.drag), int(self.polar_iters),
                    int(has_snw), int(has_liq), int(has_xtr), int(fixed)])
        if fixed:
            _wp_launch(fixed_to_float, int(g.m.numel()), dev,
                       [wp.from_torch(gmi), wp.from_torch(gmvi), float(1.0 / FIXED_SCALE),
                        wp.from_torch(gm), wp.from_torch(gmv.view(-1, 3), dtype=wp.vec3)])
        return {}


@register_operator("mpm_gather", implementation="warp_pp", family="mpm", set="particle", kind="exchange",
                   title="move points with the per-cell grid blocks, one kernel", equation=r"""$$\mathbf v_i = \sum_g w_{ig}\,\mathbf v_g,
\qquad
\mathbf C_i = \frac{4}{\Delta x^{2}}\sum_g w_{ig}\,\mathbf v_g\,(\mathbf x_g-\mathbf x_i)^{\!\top},
\qquad
\mathbf x_i \mathrel{+}= \Delta t\,\mathbf v_i$$""")
class MPMGatherWarpPP(MPMGather):
    r"""`warp_pp` implementation of mpm_gather: the grid-to-particle transfer on a `per_parent` grid as one Warp kernel.
    field -> particle: as mpm_gather, pure reads of the particle's own grid block. CUDA, 3D.

        \mathbf v_i = \sum_g w_{ig}\,\mathbf v_g, \qquad \mathbf C_i = \frac{4}{\Delta x^{2}}\sum_g w_{ig}\,\mathbf v_g\,(\mathbf x_g-\mathbf x_i)^{\!\top}, \qquad \mathbf x_i \mathrel{+}= \Delta t\,\mathbf v_i
        v_i            the velocity of particle i
        w_{ig}         the quadratic B-spline weight of node g at particle i
        v_g            the velocity of node g
        C_i            the APIC affine velocity matrix (local velocity gradient) carried to the next scatter and to mpm_strain
        \Delta x       the grid spacing
        x_g, x_i       the node and particle positions
        \Delta t       `dt_sub`, the substep

    Reference: as `mpm_gather`; the per-parent grid of `mpm_grid[per_parent]`."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"The grid-to-particle transfer. Each point takes the grid velocities around it, $\mathbf v_i=\sum_g w_{ig}\,\mathbf "
        r"v_g$ ($w_{ig}$ the quadratic B-spline weight of node $g$ at point $i$), renews its affine matrix $\mathbf "
        r"C_i=\tfrac{4}{\Delta x^{2}}\sum_g w_{ig}\,\mathbf v_g(\mathbf x_g-\mathbf x_i)^{\!\top}$ and moves, $\mathbf "
        r"x_i\mathrel{+}=\Delta t\,\mathbf v_i$. It only reads the point's own cell's grid block. One Warp kernel."
    )
    PARAM_UNITS = {'dt_sub': 'time', 'vmax': 'velocity', 'wall_contact': 'length', 'wall_contact_cells': 'count', 'wall_damp': 'fraction'}
    SYMBOLS = {'v_i': 'the velocity of particle i', 'w_{ig}': 'the quadratic B-spline weight of node g at particle i', 'v_g': 'the velocity of node g', 'C_i': 'the APIC affine velocity matrix (local velocity gradient) carried to the next scatter and to mpm_strain', '\\Delta x': 'the grid spacing', 'x_g, x_i': 'the node and particle positions', '\\Delta t': '`dt_sub`, the substep'}

    MECHANISM_TAGS = ["grid_to_particle", "advection", "per_parent_blocks", "fused_kernel"]
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False

    def forward(self, H, mask=None):
        p = H.level(self.at); g = H.field(self.frm); dev = p.state.device
        _pp_refuse(self, g, p, H, "mpm_gather[warp_pp]")
        if self.wall_damp != 1.0:
            raise ValueError("mpm_gather[warp_pp]: `wall_damp` != 1 (inelastic wall contact) is not implemented; use the default body")
        dt = sub_dt(H, self.dt_sub)
        pa, _ = p.state_schema["pos"]; va, _ = p.state_schema["vel"]
        occ = getattr(p, "occ", None)
        occ = torch.ones(p.n, device=dev) if occ is None else occ.contiguous()
        pidx, org = _pp_frame(H, g, p, p.get("pos"), getattr(p, "occ", None))
        box = [float(b) for b in g.world_size][:3]
        dx = float(g.dx)
        vmax = min(self.vmax, 0.4 * dx / dt)
        _wp_launch(g2p_pp, int(p.n), dev,
                   [wp.from_torch(p.state), int(pa), int(va), wp.from_torch(p.C, dtype=wp.mat33),
                    wp.from_torch(g.v.view(-1, 3), dtype=wp.vec3), wp.from_torch(occ),
                    wp.from_torch(pidx.contiguous()), wp.from_torch(org.contiguous(), dtype=wp.vec3),
                    int(g.shape[0]), int(g.shape[1]), int(g.shape[2]), int(g.n_local),
                    float(g.inv_dx), float(dt), float(vmax), float(2 * dx), float(box[0] - 2 * dx)])
        return {}
