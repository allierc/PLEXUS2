"""exp01 demo (2026-10-30, Cedric: "the second demo should be about PhysGaussian"): the two plastic laws PhysGaussian adds to MLS-MPM -- METAL
(von Mises) and SAND (Drucker-Prager) -- as one Plexus operator, `mpm_plasticity[law: metal | sand]`.

Their solver (papers/PhysGaussian/mpm_solver_warp/mpm_utils.py: `von_mises_return_mapping`, `sand_return_mapping`) takes a trial deformation gradient
F = (I + dt C) F, factorises it by the singular value decomposition F = U diag(s) V^T, and returns the STRESS-FREE part of the strain to the yield
surface in the logarithmic (Hencky) strain eps = log s. Here that step is its own operator: `mpm_strain` advances F, `mpm_plasticity` projects it, and
the scatter forms the stress as it does for any solid, the fixed-corotated law of the Lame parameters of the point (PhysGaussian takes the Hencky stress of
the projected F instead; the two agree to second order in the small elastic strain a return map leaves). One material per point set, so a metal body and
a sand body of one scene are two sets, each with its own operator.
"""
from __future__ import annotations

import math

import torch

from plexus.models.registry import register_operator
from plexus.operators.mpm_ops import HAVE_WARP, Lateral, _wp_launch

if HAVE_WARP:
    import warp as wp

    @wp.kernel
    def plastic_project(F: wp.array(dtype=wp.mat33), MU: wp.array(dtype=float), LA: wp.array(dtype=float),
                        YIELD: wp.array(dtype=float), OCC: wp.array(dtype=float), model: int, alpha: float, xi: float):
        p = wp.tid()
        if OCC[p] <= 0.0:
            return
        Ft = F[p]
        U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        s = wp.vec3(0.0)
        wp.svd3(Ft, U, s, V)
        mu = MU[p]
        la = LA[p]
        if mu <= 0.0:
            return                                                      # a torn or liquid point has nothing to return
        eps = wp.vec3(wp.log(wp.max(wp.abs(s[0]), 1.0e-6)), wp.log(wp.max(wp.abs(s[1]), 1.0e-6)), wp.log(wp.max(wp.abs(s[2]), 1.0e-6)))
        tr = eps[0] + eps[1] + eps[2]
        eh = eps - wp.vec3(tr / 3.0, tr / 3.0, tr / 3.0)
        nh = wp.length(eh) + 1.0e-12
        if model == 1:                                                  # metal: von Mises, the deviatoric Kirchhoff stress 2 mu |eps_hat| against the yield
            if 2.0 * mu * nh > YIELD[p]:
                dg = nh - YIELD[p] / (2.0 * mu)
                e2 = eps - (dg / nh) * eh
                F[p] = U * wp.diag(wp.vec3(wp.exp(e2[0]), wp.exp(e2[1]), wp.exp(e2[2]))) * wp.transpose(V)
                if xi > 0.0:
                    YIELD[p] = YIELD[p] + 2.0 * mu * xi * dg            # linear isotropic hardening
        else:                                                           # sand: Drucker-Prager, a cohesionless cone, no tension
            dg = nh + (3.0 * la + 2.0 * mu) / (2.0 * mu) * tr * alpha
            if dg > 0.0:
                if tr > 0.0:
                    F[p] = U * wp.transpose(V)                          # in tension the grains part: stress-free
                else:
                    e2 = eps - eh * (dg / nh)
                    F[p] = U * wp.diag(wp.vec3(wp.exp(e2[0]), wp.exp(e2[1]), wp.exp(e2[2]))) * wp.transpose(V)


@register_operator("mpm_plasticity", family="mpm", set="particle", kind="lateral", title="return the strain to the yield surface",
                   equation=r"""$$\boldsymbol\varepsilon=\log\boldsymbol\sigma(\mathbf F),\quad \Delta\gamma=\|\hat{\boldsymbol\varepsilon}\|-\frac{\sigma_y}{2\mu}\ \ (\text{metal}),\quad \Delta\gamma=\|\hat{\boldsymbol\varepsilon}\|+\frac{3\lambda+2\mu}{2\mu}\,\mathrm{tr}\,\boldsymbol\varepsilon\,\alpha\ \ (\text{sand}),\quad \boldsymbol\varepsilon\leftarrow\boldsymbol\varepsilon-\Delta\gamma\,\frac{\hat{\boldsymbol\varepsilon}}{\|\hat{\boldsymbol\varepsilon}\|}$$""")
class MPMPlasticity(Lateral):
    r"""the plastic flow of a solid, the step PhysGaussian adds after the strain update: the deformation gradient is factorised F = U diag(s) V^T, and the part of the logarithmic strain beyond the yield surface is removed, so what was beyond it stays as a permanent shape change (a dent in metal, a heap of sand).
    particle -> particle: reads F, mu, la and the occupancy; writes F in place (and, with hardening, the point's yield stress); capturable.

        \boldsymbol\varepsilon=\log\boldsymbol\sigma(\mathbf F),\quad \Delta\gamma=\|\hat{\boldsymbol\varepsilon}\|-\frac{\sigma_y}{2\mu}\ \ (\text{metal}),\quad \Delta\gamma=\|\hat{\boldsymbol\varepsilon}\|+\frac{3\lambda+2\mu}{2\mu}\,\mathrm{tr}\,\boldsymbol\varepsilon\,\alpha\ \ (\text{sand}),\quad \boldsymbol\varepsilon\leftarrow\boldsymbol\varepsilon-\Delta\gamma\,\frac{\hat{\boldsymbol\varepsilon}}{\|\hat{\boldsymbol\varepsilon}\|}
        \boldsymbol\varepsilon   the Hencky strain, the logarithm of the singular values of F
        \hat{\boldsymbol\varepsilon} its deviatoric part, minus the mean of its three components
        \sigma_y        `yield_stress`, the von Mises yield stress of metal (stress)
        \mu, \lambda    the point's Lame parameters
        \Delta\gamma    the plastic multiplier, how far beyond the yield surface the trial strain lies
        \alpha          the friction coefficient of the sand, sqrt(2/3) 2 sin(phi) / (3 - sin(phi)) for the friction angle `friction_angle`

    Reference: Xie, T. et al. (2024). PhysGaussian: physics-integrated 3D Gaussians for generative dynamics. CVPR 2024; Klar, G. et al. (2016). Drucker-Prager elastoplasticity for sand animation. ACM Trans. Graph. 35(4):103."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"Permanent deformation. After the strain update the deformation gradient is split as $\mathbf F=\mathbf "
        r"U\,\mathrm{diag}(s)\,\mathbf V^{\top}$ and its Hencky strain $\boldsymbol\varepsilon=\log s$ is measured. With "
        r"`law: metal` (von Mises) the part of the deviatoric strain whose stress $2\mu\|\hat{\boldsymbol\varepsilon}\|$ "
        r"exceeds the yield stress `yield_stress` is removed ($\mu$ the point's shear modulus, "
        r"$\hat{\boldsymbol\varepsilon}$ the strain minus its mean of the three components); with `law: sand` "
        r"(Drucker-Prager) the strain beyond a cohesionless friction cone of angle `friction_angle` is removed, and a "
        r"point in tension is released stress-free. What was removed stays as a permanent shape change: a dent in metal, a "
        r"heap of sand."
    )
    PARAM_UNITS = {'yield_stress': 'stress', 'friction_angle': 'angle', 'hardening': 'dimensionless'}
    SYMBOLS = {'\\boldsymbol\\varepsilon': 'the Hencky strain, the logarithm of the singular values of F', '\\hat{\\boldsymbol\\varepsilon}': 'its deviatoric part, minus the mean of its three components', '\\sigma_y': '`yield_stress`, the von Mises yield stress of metal', '\\mu, \\lambda': "the point's Lame parameters", '\\Delta\\gamma': 'the plastic multiplier, how far beyond the yield surface the trial strain lies', '\\alpha': 'the friction coefficient of the sand from `friction_angle`'}
    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["law"]
    MECHANISM_TAGS = ["plasticity", "von_mises", "drucker_prager", "return_mapping"]
    PARAM_ROLES = {"law": "metal_or_sand", "yield_stress": "von_mises_yield_of_metal", "friction_angle": "sand_internal_friction_degrees",
                   "hardening": "metal_isotropic_hardening_slope"}
    REFERENCE = "Xie, T. et al. (2024). CVPR 2024; Klar, G. et al. (2016). ACM Trans. Graph. 35(4):103."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.law = str(params["law"]).lower()
        if self.law not in ("metal", "sand"):
            raise ValueError(f"mpm_plasticity: law must be metal or sand, got {self.law!r}")
        self.yield_stress = float(params.get("yield_stress", 1.0e3))
        self.hardening = float(params.get("hardening", 0.0))
        phi = math.radians(float(params.get("friction_angle", 30.0)))
        self.alpha = math.sqrt(2.0 / 3.0) * 2.0 * math.sin(phi) / (3.0 - math.sin(phi))

    def forward(self, H, mask=None):
        if not HAVE_WARP:
            raise RuntimeError("mpm_plasticity needs warp-lang")
        p = H.level(self.at); dev = p.state.device
        if str(dev) == "cpu":
            raise RuntimeError("mpm_plasticity is CUDA only")
        if getattr(self, "_yield", None) is None:
            self._yield = torch.full((p.n,), self.yield_stress, device=dev, dtype=torch.float32)
            if not p.F.is_contiguous():
                raise RuntimeError("mpm_plasticity writes F in place and needs it contiguous")
        occ = p.occ.contiguous() if getattr(p, "occ", None) is not None else torch.ones(p.n, device=dev)
        _wp_launch(plastic_project, int(p.n), dev,
                   [wp.from_torch(p.F, dtype=wp.mat33), wp.from_torch(p.mu.contiguous()), wp.from_torch(p.la.contiguous()),
                    wp.from_torch(self._yield), wp.from_torch(occ), 1 if self.law == "metal" else 2, float(self.alpha), float(self.hardening)])
        return {}
