"""deform_control[cell_flattening]: each cell's own volume-preserving flattening rate (exp29 minimal 3 x 3 rig, Cedric 2026-10-28: "inject a constant deformation with a matrix diagonal different in Z vs XY").

A variant of `deform_control` (cell_ops.py), in its own module so that no archived spec changes: the same F <- expm(-A dt) F once a frame, with A the diagonal rate (a, a, -2a) of the cell the
point belongs to (trace 0: the cell's volume is kept), read from a per-cell block instead of six numbers for the whole set.
"""
from __future__ import annotations

import numpy as np
import torch

from plexus.models.registry import register_operator
from plexus.operators.cell_ops import DeformControl


@register_operator("deform_control", implementation="cell_flattening", family="motility", set="particle", kind="lateral",
                   equation=r"""$$\mathbf F_p\leftarrow\mathbf G_c\,\mathbf F_p,\qquad \mathbf G_c=\exp(-\mathbf A_c\,\Delta t),\quad \mathbf A_c=a_c\,\big(\mathbf I-3\,\hat{\mathbf n}\hat{\mathbf n}^{\mathsf T}\big)$$""",
                   title="flatten each cell at its own rate, volume kept")
class DeformControlCellFlatten(DeformControl):
    r"""rest-shape rate that flattens a cell: the sheet plane widens at rate a_c and the normal axis narrows at 2 a_c, so the cell's volume is kept (trace of A is 0).
    particle -[containment]-> particle: reads a_c per cell (the cell block `block`, else the one number `flattening_rate`) through each point's parent and the sheet normal (the fixed `axis`, else the direction of the cell's centroid from `centre`, a vesicle, else from the cell's own virtual centre in the cell block `centre_block`, which `cell_normal` fits from the neighbours), multiplies the point's F by G_c once a frame, writes F in place.

        \mathbf F_p\leftarrow\mathbf G_c\,\mathbf F_p,\qquad \mathbf G_c=\exp(-\mathbf A_c\,\Delta t),\quad \mathbf A_c=a_c\,\big(\mathbf I-3\,\hat{\mathbf n}\hat{\mathbf n}^{\mathsf T}\big)
        F_p            the elastic deformation gradient of point p
        G_c            the rest-shape increment of the cell c that owns p: e^{-a_c dt} (I - n n^T) + e^{2 a_c dt} n n^T, the normal n on the `axis` (default z) or radial from `centre`
        a_c            the cell's flattening rate, per unit time: the log-stretch per time unit of the cell's extent in the plane (the cell's thickness along the normal narrows at 2 a_c)
        n              the unit vector of the sheet normal: `axis`, or from `centre` through the cell's centroid
        \Delta t       the frame length

    Reference: as `deform_control`; Xu, M. et al. (2025). IEEE TVCG 31(10):9140-9153."""
    PARAM_UNITS = {'rate': 'rate', 'axis': 'count', 'flattening_rate': 'rate', 'centre': 'length', 'types': 'count'}
    SYMBOLS = {'F_p': 'the elastic deformation gradient of point p', 'G_c': 'the rest-shape increment of the cell that owns p', 'a_c': "the cell's flattening rate, per unit time",
               'n': 'the unit vector of the sheet normal', '\\Delta t': 'the frame length'}
    PARAM_ROLES = {'types': 'cell_types_flattened_default_all', 'rate': 'inherited_six_number_rate_unused_the_cell_block_or_the_rate_key_is_read', 'cells': 'the_cell_set', 'block': 'cell_block_holding_each_cells_flattening_rate', 'flattening_rate': 'flattening_rate_of_every_cell', 'axis': 'sheet_normal_axis_0_1_2', 'centre': 'vesicle_centre_for_a_radial_normal', 'centre_block': 'cell_block_with_each_cells_virtual_centre', 'over': 'frames_of_action'}
    REQUIRES_PARAMS = []

    def __init__(self, params, device="cpu"):
        super().__init__({**params, "rate": params.get("rate", [0.0] * 6)}, device)      # the parent wants one of rate / field
        self.cells = str(params.get("cells", "cell"))
        self.block = params.get("block")
        self.a = None if params.get("flattening_rate") is None else float(params["flattening_rate"])
        if self.block is None and self.a is None:
            raise ValueError("deform_control[cell_flattening]: give `block` (a cell block of rates) or `flattening_rate` (one rate for every cell)")
        self.centre = None if params.get("centre") is None else [float(v) for v in params["centre"]]
        self.centre_block = params.get("centre_block")
        self.axis = int(params.get("axis", 2))
        self.types = params.get("types")                 # `types: [outer]` (exp36, 2026-10-31): flatten the cells of these types only; None = every cell
        if self.axis not in (0, 1, 2):
            raise ValueError(f"deform_control[cell_flattening]: axis is 0, 1 or 2, got {self.axis}")

    def forward(self, H, mask=None):
        p, C = H.level(self.at), H.level(self.cells)
        if self.over and int(getattr(H, "frame", 0) or 0) >= self.over:
            return {}
        if self.block is not None and self.block not in C.state_schema:
            raise ValueError(f"deform_control[cell_flattening]: {self.cells!r} has no block {self.block!r}")
        par = p.parent.long().clamp(0, C.state.shape[0] - 1)
        a_c = C.state[:, C.state_schema[self.block][0]] if self.block is not None else torch.full((C.state.shape[0],), self.a, device=p.F.device, dtype=p.F.dtype)
        a = a_c[par].to(p.F.dtype) * (p.occ > 0.5).to(p.F.dtype)                        # a dormant slot is left alone
        tm = _type_mask(C, self.types)
        if tm is not None:
            a = a * tm.to(a.dtype)[par]                                                   # a = 0: G = I, the cell is left as it is
        dt = float(getattr(H, "dt", 0.002))
        if self.centre is None and self.centre_block is None:
            d = torch.exp(-a * dt)[:, None].repeat(1, 3)                                # the plane: e^{-a dt}
            d[:, self.axis] = torch.exp(2.0 * a * dt)                                   # the normal: e^{2 a dt}
            F = d[:, :, None] * p.F                                                     # diag(d) @ F
        else:                                                                           # a vesicle or a fitted layer: the normal of a cell is the direction from its centre (`centre`, or its `centre_block`) through its centroid
            cen = C.state[:, C.state_schema["pos"][0]:C.state_schema["pos"][1]].to(p.F.dtype)
            if self.centre_block is not None:
                b0, b1 = C.state_schema[self.centre_block]
                n = cen - C.state[:, b0:b1].to(cen.dtype)                               # each cell's own (virtual) centre
            else:
                n = cen - torch.tensor(self.centre, device=cen.device, dtype=cen.dtype)
            n = (n / n.norm(dim=1, keepdim=True).clamp(min=1e-9))[par]                  # [point, 3]
            e1 = torch.exp(-a * dt)[:, None, None]
            e2 = torch.exp(2.0 * a * dt)[:, None, None]
            eye = torch.eye(3, device=n.device, dtype=n.dtype)[None]
            G = e1 * (eye - n[:, :, None] * n[:, None, :]) + e2 * (n[:, :, None] * n[:, None, :])
            F = G @ p.F
        if torch.is_grad_enabled():
            p.F = F
        else:
            p.F.copy_(F)
        return {}


from plexus.models.base import Lateral          # noqa: E402



@register_operator("deform_control", implementation="cell_inflation", family="motility", set="particle", kind="lateral",
                   equation=r"""$$\mathbf F_p\leftarrow\mathbf G_p\,\mathbf F_p,\quad \mathbf G_p=\mathrm{diag}\big(e^{g_p a_c w_x\Delta t},\,e^{g_p a_c w_y\Delta t},\,e^{g_p a_c w_z\Delta t}\big),\quad g_p=\mathrm{clip}\Big(\frac{J_{\max}-\det\mathbf F_p}{0.1},\,0,\,1\Big)$$""",
                   title="inflate the cells of the chosen types against being squashed")
class DeformControlCellInflate(DeformControl):
    r"""a fixed rest-shape growth of the cells of the chosen types, the same stretch along every axis (or weighted per axis): their volume ratio rises, the fixed-corotated pressure lambda (J - 1) J turns positive and pushes back against the contact that squashes them.
    particle -[containment]-> particle: reads a_c per cell (the cell block `block`, else the one number `inflation_rate`) through each point's parent, stretches the point's deformation gradient once a frame, and stops by itself at a volume ratio `jmax`.

        \mathbf F_p\leftarrow\mathbf G_p\,\mathbf F_p,\quad \mathbf G_p=\mathrm{diag}\big(e^{g_p a_c w_x\Delta t},\,e^{g_p a_c w_y\Delta t},\,e^{g_p a_c w_z\Delta t}\big),\quad g_p=\mathrm{clip}\Big(\frac{J_{\max}-\det\mathbf F_p}{0.1},\,0,\,1\Big)
        F_p            the elastic deformation gradient of point p
        G_p            the rest-shape increment of the point: a diagonal stretch in the world frame, switched off smoothly as p reaches the volume ratio J_max
        a_c            the inflation rate of the cell c that owns p, per unit time: the log-stretch per time unit along an axis of weight 1
        w              `aniso`, the per-axis weights (x, y, z): 1, 1, 1 inflates a cell in all directions, 1, 1, 0.5 half as fast along z (a flat tissue)
        J_max          `jmax`, the volume ratio det F at which the inflation stops (a gate 0.1 wide)
        \Delta t       the frame length

    Reference: as `deform_control`; Plexus exp35 / exp36 (this work, 2026-10-31)."""
    PARAM_UNITS = {'rate': 'rate', 'inflation_rate': 'rate', 'aniso': 'dimensionless', 'jmax': 'dimensionless', 'types': 'count'}
    SYMBOLS = {'F_p': 'the elastic deformation gradient of point p', 'G_p': 'the rest-shape increment of the point, a diagonal stretch switched off as it reaches the volume ratio J_max',
               'a_c': "the cell's inflation rate, per unit time", 'w': '`aniso`, the per-axis weights (x, y, z)', 'J_max': '`jmax`, the volume ratio at which the inflation stops', '\\Delta t': 'the frame length'}
    PARAM_ROLES = {'rate': 'inherited_six_number_rate_unused_the_cell_block_or_the_rate_key_is_read', 'cells': 'the_cell_set', 'block': 'cell_block_holding_each_cells_inflation_rate', 'inflation_rate': 'inflation_rate_of_every_cell', 'aniso': 'per_axis_weights_x_y_z',
                   'jmax': 'volume_ratio_at_which_inflation_stops', 'types': 'cell_types_inflated_default_all'}
    REQUIRES_PARAMS = []

    def __init__(self, params, device="cpu"):
        super().__init__({**params, "rate": params.get("rate", [0.0] * 6)}, device)
        self.cells = str(params.get("cells", "cell"))
        self.block = params.get("block")
        self.a = None if params.get("inflation_rate") is None else float(params["inflation_rate"])
        if self.block is None and self.a is None:
            raise ValueError("deform_control[cell_inflation]: give `block` (a cell block of rates) or `inflation_rate` (one rate for every cell)")
        self.aniso = [float(v) for v in params.get("aniso", [1.0, 1.0, 1.0])]
        if len(self.aniso) != 3:
            raise ValueError("deform_control[cell_inflation]: `aniso` is three weights (x, y, z)")
        self.jmax = float(params.get("jmax", 1.5))
        self.types = params.get("types")

    def forward(self, H, mask=None):
        p, C = H.level(self.at), H.level(self.cells)
        if self.over and int(getattr(H, "frame", 0) or 0) >= self.over:
            return {}
        if self.block is not None and self.block not in C.state_schema:
            raise ValueError(f"deform_control[cell_inflation]: {self.cells!r} has no block {self.block!r}")
        par = p.parent.long().clamp(0, C.state.shape[0] - 1)
        a_c = C.state[:, C.state_schema[self.block][0]] if self.block is not None else torch.full((C.state.shape[0],), self.a, device=p.F.device, dtype=p.F.dtype)
        a = a_c[par].to(p.F.dtype) * (p.occ > 0.5).to(p.F.dtype)
        tm = _type_mask(C, self.types)
        if tm is not None:
            a = a * tm.to(a.dtype)[par]
        J = torch.linalg.det(p.F.detach())
        g = ((self.jmax - J) / 0.1).clamp(0.0, 1.0)                                    # a smooth stop at J_max; no gradient through the gate
        w = torch.as_tensor(self.aniso, device=p.F.device, dtype=p.F.dtype)
        dt = float(getattr(H, "dt", 0.002))
        d = torch.exp((a * g)[:, None] * w[None, :] * dt)                               # [point, 3]
        F = d[:, :, None] * p.F                                                         # diag(d) @ F
        if torch.is_grad_enabled():
            p.F = F
        else:
            p.F.copy_(F)
        return {}

@register_operator("attraction_repulsion", implementation="stretched_cohesion", family="interaction", set="particle", kind="lateral",
                   equation=r"""$$f(\tilde r)=p_1\,e^{-\tilde r^{2p_2}/2\sigma^2}-p_3\,e^{-\tilde r^{2p_4}/2\sigma^2},\quad \tilde r_{ij}=\frac{\lvert(\Delta x,\Delta y,\lambda_c\Delta z)\rvert}{d\,\lambda_c^{1/3}},\quad \mathbf a_i=g\,\Big\langle f(\tilde r_{ij})\,(\mathbf x_j-\mathbf x_i)\Big\rangle_j$$""",
                   title="space a cell's points at a flattened constant distance")
class AttractionRepulsionCellAniso(Lateral):
    r"""the overdamped soft-core attraction-repulsion law (ParticleGraph `arbitrary`, Plexus `attraction_repulsion`) between the points of ONE cell, with the distance measured in a metric stretched along the sheet normal, so the points settle at a constant spacing d in the plane and d / lambda along the normal; returned as a body acceleration the MPM substep consumes (the operator never writes the state).
    particle -[containment]-> particle: reads pos and parent, returns for every point the mean pair displacement over the cell's own points within the cutoff, times `gain`, the cell's net removed.

        f(\tilde r)=p_1 e^{-\tilde r^{2p_2}/2\sigma^2}-p_3 e^{-\tilde r^{2p_4}/2\sigma^2},\quad \tilde r_{ij}=\frac{\lvert(\Delta x,\Delta y,\lambda_c\Delta z)\rvert}{d\,\lambda_c^{1/3}},\quad \mathbf a_i=g\,\Big\langle f(\tilde r_{ij})(\mathbf x_j-\mathbf x_i)\Big\rangle_j
        f              the pair law at the scaled distance, positive for attraction, zero at 1 (p_1 = p_3)
        \tilde r_{ij}  the distance of points i, j of one cell in units of the rest spacing, z (the `axis`) stretched by lambda_c
        d              `rest`, the rest spacing of the seed's points, world length
        \lambda_c      the cell's aspect ratio, plane over normal, from `z_stretch_block` (else `z_stretch`); the rest spacing is d lambda^(1/3) in the plane and d lambda^(-2/3) along the normal, so the volume per point stays d^3
        \sigma         `sigma`, the width of the two Gaussians in units of the rest spacing
        a_i            the body acceleration of point i, which the grid's Stokes drag (`drag` of mpm_scatter) turns into a velocity a_i / drag: the law is overdamped
        g              `gain`, the acceleration per unit of mean pair displacement, per time squared
        j              the other live points of the same cell within `radius` rest spacings

    Reference: D'Orsogna, M. R., Chuang, Y. L., Bertozzi, A. L. & Chayes, L. S. (2006). Phys. Rev. Lett. 96:104302."""
    PARAM_UNITS = {'rest': 'length', 'sigma': 'dimensionless', 'radius': 'dimensionless', 'gain': 'rate', 'z_stretch': 'dimensionless', 'axis': 'count'}
    SYMBOLS = {'f': 'the pair law at the scaled distance, positive for attraction', '\\tilde r_{ij}': 'the distance of two points of a cell in units of the rest spacing, the normal stretched',
               'd': 'the rest spacing of the seed points', '\\lambda_c': "the cell's aspect ratio, plane over normal", '\\sigma': 'the width of the two Gaussians in units of the rest spacing',
               'a_i': 'the body acceleration of point i', 'g': 'the acceleration per unit of mean pair displacement, per time squared', 'j': 'the other live points of the same cell within the cutoff'}
    PARAM_ROLES = {'cells': 'the_cell_set', 'rest': 'rest_spacing_of_the_seed_points', 'z_stretch': 'plane_over_normal_aspect_ratio', 'z_stretch_block': 'cell_block_with_each_cells_aspect_ratio',
                   'axis': 'sheet_normal_axis_0_1_2', 'p': '[pull_strength, pull_range, push_strength, push_range]', 'sigma': 'gaussian_width_in_rest_spacings',
                   'radius': 'cutoff_in_rest_spacings', 'gain': 'acceleration_per_mean_pair_displacement'}
    EMIT = "mpm_acceleration"                        # consumed by the MPM substep as a_ext, not engine-integrated; the state is never written
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["rest"]
    REQUIRES_TYPE_PROPS = []
    MECHANISM_TAGS = ["short_range_repulsion", "long_range_attraction", "lattice_forming", "rest_shape"]
    REFERENCE = "D'Orsogna, M. R., Chuang, Y. L., Bertozzi, A. L. & Chayes, L. S. (2006). Phys. Rev. Lett. 96:104302."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.cells = str(params.get("cells", "cell"))
        self.rest = float(params["rest"])
        self.aspect = float(params.get("z_stretch", 1.0))
        self.aspect_block = params.get("z_stretch_block")
        self.axis = int(params.get("axis", 2))
        self.p = [float(v) for v in params.get("p", [1.0, 1.0, 1.0, 2.0])]
        if len(self.p) != 4 or self.p[0] != self.p[2]:
            raise ValueError("attraction_repulsion[stretched_cohesion]: p is [p1, p2, p3, p4] with p1 == p3 (the law is zero at the rest spacing)")
        self.sigma = float(params.get("sigma", 0.7))
        self.radius = float(params.get("radius", 1.3))
        self.gain = float(params.get("gain", 100.0))
        self._B = None

    def forward(self, H, mask=None):
        P, C = H.level(self.at), H.level(self.cells)
        pos = P.get("pos")
        nC = int(C.state.shape[0])
        npt = pos.shape[0]
        if self._B is None:                                                              # the points of a cell are one contiguous block of the set
            if npt % nC or not bool((P.parent.long().view(nC, npt // nC) == torch.arange(nC, device=pos.device)[:, None]).all()):
                raise ValueError("attraction_repulsion[stretched_cohesion]: the points are not laid out as one equal block per cell")
            self._B = npt // nC
        B = self._B
        dev, dt_ = pos.device, pos.dtype
        lam = torch.full((nC,), self.aspect, device=dev, dtype=dt_)
        if self.aspect_block and self.aspect_block in C.state_schema:
            ab = C.state[:, C.state_schema[self.aspect_block][0]].to(dt_)
            lam = torch.where(ab > 0, ab, lam)
        with torch.no_grad():
            X = pos.view(nC, B, 3)
            O = (P.occ > 0.5).view(nC, B)
            scale = torch.ones(nC, 1, 1, 3, device=dev, dtype=dt_)
            scale[:, 0, 0, self.axis] = lam
            D = X[:, None, :, :] - X[:, :, None, :]                                      # [cell, i, j, 3] = x_j - x_i
            rest_c = (self.rest * lam.pow(1.0 / 3.0))[:, None, None]
            r2 = (((D * scale) ** 2).sum(-1)) / rest_c ** 2                              # scaled distance squared
            s2 = 2.0 * self.sigma ** 2
            f = self.p[0] * torch.exp(-r2.pow(self.p[1]) / s2) - self.p[2] * torch.exp(-r2.pow(self.p[3]) / s2)
            ok = O[:, :, None] & O[:, None, :] & (r2 < self.radius ** 2) & (r2 > 1e-12)
            f = torch.where(ok, f, torch.zeros_like(f))
            deg = ok.sum(-1).clamp(min=1).to(dt_)
            dpos = (f[..., None] * D).sum(2) / deg[..., None]                            # mean pair displacement, [cell, i, 3]
            dpos = dpos * O[..., None]
            dpos = dpos - dpos.sum(1, keepdim=True) / O.sum(1).clamp(min=1).to(dt_)[:, None, None]      # the cell's net displacement removed
            dpos = dpos * O[..., None]
        return {self.at: (self.gain * dpos).view(npt, 3)}


from plexus.models.base import Lateral as _Lateral      # noqa: E402,F811


def fit_surface_height(X, n, live, radius, ridge=1.0e-3, min_neighbours=6, return_all=False, pair_ok=None):
    r"""the height, along the normal n_i, of the surface fitted through the neighbours of cell i, at the cell's own place, relative to the cell's centroid (negative: the surface passes below the cell).
    A weighted least-squares quadratic h ~ c0 + c1 u + c2 v + c3 u^2 + c4 u v + c5 v^2 of the neighbours' heights h = (x_j - x_i) . n_i over their tangent coordinates (u, v), the cell itself left out,
    Gaussian weights of width radius / 2, a ridge on c1 .. c5; c0 is returned. A cell with fewer than `min_neighbours` live neighbours within `radius` takes the weighted mean height (a plane through them).
    On a sphere, or any smooth layer, c0 is about 0; on a cell sitting above or below its neighbours it is the step.
    `pair_ok` ([i, j] bool, default None = every pair): the neighbours j the fit of cell i may use (cell_normal[same_side] leaves out the cells of a facing wall)."""
    N = X.shape[0]
    ref = torch.zeros_like(n)
    ref[:, 0] = 1.0
    ref = torch.where((n[:, 0:1].abs() > 0.9), torch.tensor([0.0, 1.0, 0.0], device=n.device, dtype=n.dtype)[None].expand_as(n), ref)
    e1 = torch.linalg.cross(n, ref)
    e1 = e1 / e1.norm(dim=1, keepdim=True).clamp(min=1e-12)
    e2 = torch.linalg.cross(n, e1)
    R = X[None, :, :] - X[:, None, :]                                                   # [i, j, 3] = x_j - x_i
    sig = 0.5 * radius
    u = (R * e1[:, None, :]).sum(-1) / sig
    v = (R * e2[:, None, :]).sum(-1) / sig
    h = (R * n[:, None, :]).sum(-1)
    r2 = (R * R).sum(-1)
    eye = torch.eye(N, device=X.device, dtype=torch.bool)
    ok = live[None, :] & ~eye & (r2 < radius ** 2)
    if pair_ok is not None:
        ok = ok & pair_ok
    W = torch.exp(-r2 / (2.0 * sig ** 2)) * ok.to(X.dtype)
    nb = ok.sum(1)
    wsum = W.sum(1).clamp(min=1e-12)
    c0_plane = (W * h).sum(1) / wsum
    Phi = torch.stack([torch.ones_like(u), u, v, u * u, u * v, v * v], dim=-1)           # [i, j, 6]
    A = torch.einsum("ij,ija,ijb->iab", W, Phi, Phi)
    A = A + ridge * wsum[:, None, None] * torch.diag(torch.tensor([0.0, 1, 1, 1, 1, 1], device=X.device, dtype=X.dtype))[None] + 1e-12 * torch.eye(6, device=X.device, dtype=X.dtype)[None]
    b = torch.einsum("ij,ija,ij->ia", W, Phi, h)
    c0 = torch.linalg.solve(A, b[:, :, None])[:, 0, 0]
    sol = torch.linalg.solve(A, b[:, :, None])[:, :, 0]
    c0 = torch.where((nb >= min_neighbours) & torch.isfinite(c0), c0, c0_plane)
    if return_all:                                                                       # the six coefficients in the tangent coordinates u, v of width radius / 2, the neighbour count
        return sol, nb
    return c0.clamp(-0.5 * radius, 0.5 * radius)


@register_operator("cell_normal", family="polarity", set="cell", kind="lateral",
                   equation=r"""$$\mathbf n_i=\arg\min_{\lvert\mathbf u\rvert=1}\mathbf u^{\mathsf T}\Big(\mathbf C_i+\tau\,(\mathbf I-\mathbf n^{0}_i\mathbf n^{0\mathsf T}_i)\Big)\mathbf u,\quad \mathbf C_i=\frac{\sum_j w_{ij}(\mathbf x_j-\bar{\mathbf x}_i)(\mathbf x_j-\bar{\mathbf x}_i)^{\mathsf T}}{\sum_j w_{ij}},\quad \mathbf o_i=\mathbf x_i-L\,\mathbf n_i$$""",
                   title="fit each cell's layer normal from its neighbours")
class CellNormal(_Lateral):
    r"""the normal of the cell layer at each cell, fitted to the centroids of its neighbours, written as a virtual centre o_i far behind the cell so that every operator that reads a per-cell centre (`centre_block`) reads a normal.
    cell -[neighbour relation]-> cell: reads the cell centroids (`pos`, kept by aggregate_centroid) and the block `block`, writes `block` (width 3) and, when `mid_block` is given, the point the layer's plane passes through there (width 3), in place, a derived readout like aggregate_centroid.

        \mathbf n_i=\arg\min_{\lvert\mathbf u\rvert=1}\mathbf u^{\mathsf T}\Big(\mathbf C_i+\tau\,(\mathbf I-\mathbf n^{0}_i\mathbf n^{0\mathsf T}_i)\Big)\mathbf u,\quad \mathbf C_i=\frac{\sum_j w_{ij}(\mathbf x_j-\bar{\mathbf x}_i)(\mathbf x_j-\bar{\mathbf x}_i)^{\mathsf T}}{\sum_j w_{ij}},\quad \mathbf o_i=\mathbf x_i-L\,\mathbf n_i
        n_i            the unit normal of the layer at cell i: the direction of least spread of the neighbouring centroids, its sign chosen to agree with the previous one
        C_i            the covariance of the centroids of the live cells within `radius` of cell i (itself included), Gaussian weights w_ij of width radius / 2
        n^0_i          the previous normal (the block's own value), at the first frame `prior_axis` or the direction from `prior_centre`
        \tau           `prior`, a variance added across n^0 so that a cell with fewer than three neighbours keeps its previous normal, world length squared
        o_i            the virtual centre written to `block`: the cell's centroid minus `reach` times n_i (reach 1000: a far centre, as the 3 x 3 rig's [x, y, -1000])
        L              `reach`
        blend          the new normal is the previous one moved `blend` of the way to the fitted one
        m_i            the point of the layer's plane at cell i, written to `mid_block`: the Gaussian-weighted mean of those centroids, or with `curvature` the point of a quadratic surface fitted through the neighbours (the cell itself left out) at the cell's own place, x_i + c_0 n_i, so that a curved layer carries no bias

    Reference: plane fitting by the smallest eigenvector of a local covariance (Hoppe, H. et al. (1992). Surface reconstruction from unorganized points. SIGGRAPH 26:71-78); Plexus exp29 (this work, 2026-10)."""
    PARAM_UNITS = {'radius': 'length', 'reach': 'length', 'prior': 'length', 'blend': 'fraction', 'prior_axis': 'count', 'prior_centre': 'length', 'types': 'count'}
    SYMBOLS = {'n_i': 'the unit normal of the layer at cell i', 'C_i': 'the covariance of the neighbouring centroids', 'n^0_i': 'the previous normal', '\\tau': 'the prior variance across the previous normal',
               'o_i': 'the virtual centre written to the block', 'L': 'the reach of the virtual centre', 'blend': 'the fraction of the fitted normal moved in per frame'}
    PARAM_ROLES = {'types': 'cell_types_the_layer_is_fitted_through_default_all', 'block': 'cell_block_receiving_the_virtual_centre', 'radius': 'neighbourhood_radius', 'reach': 'distance_of_the_virtual_centre_behind_the_cell', 'prior': 'variance_added_across_the_previous_normal',
                   'blend': 'fraction_of_the_fitted_normal_per_frame', 'prior_axis': 'first_frame_normal_axis_0_1_2', 'prior_centre': 'first_frame_normal_radial_from_this_centre', 'mid_block': 'cell_block_receiving_the_point_of_the_layer_plane', 'curvature': 'plane_point_from_a_fitted_curved_surface'}
    EMIT = None
    MAY_MUTATE_INTEGRATED_STATE = True              # a derived readout of the geometry (as aggregate_centroid writes the cell centroids), not a dynamics operator
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["block"]
    MECHANISM_TAGS = ["polarity", "layer_normal", "apical_basal_axis"]
    REFERENCE = "Hoppe, H. et al. (1992). SIGGRAPH 26:71-78."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.block = str(params["block"])
        self.radius = float(params.get("radius", 2.5))
        self.reach = float(params.get("reach", 1000.0))
        self.prior = float(params.get("prior", 0.02))
        self.blend = float(params.get("blend", 0.5))
        self.prior_axis = int(params.get("prior_axis", 2))
        self.prior_centre = None if params.get("prior_centre") is None else [float(v) for v in params["prior_centre"]]
        self.mid_block = params.get("mid_block")
        self.curvature = bool(params.get("curvature", False))
        self.types = params.get("types")                 # `types: [outer]` (exp36, 2026-10-31): the layer is the cells of these types: only they are fitted, and only through each other       # the plane point from a quadratic surface through the neighbours, the cell left out (a curved layer), not their mean
        self._started = False

    def forward(self, H, mask=None):
        C = H.level(self.at)
        a0, a1 = C.state_schema["pos"]
        b0, b1 = C.state_schema[self.block]
        with torch.no_grad():
            live = C.occ > 0.5
            tm = _type_mask(C, self.types)
            if tm is not None:
                live = live & tm                                                        # the fit sees, and writes, the layer's cells only
            X = C.state[:, a0:a1].detach().double()
            if not self._started:                                                   # the first frame: the prior normal of every cell
                if self.prior_centre is not None:
                    n0 = X - torch.tensor(self.prior_centre, device=X.device, dtype=X.dtype)
                else:
                    n0 = torch.zeros_like(X)
                    n0[:, self.prior_axis] = 1.0
                self._started = True
            else:
                n0 = X - C.state[:, b0:b1].detach().double()
            n0 = n0 / n0.norm(dim=1, keepdim=True).clamp(min=1e-12)
            r2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
            W = torch.exp(-r2 / (2.0 * (0.5 * self.radius) ** 2)) * (r2 < self.radius ** 2) * live[None, :].double()
            wsum = W.sum(1, keepdim=True).clamp(min=1e-12)
            m = (W @ X) / wsum
            S2 = (W @ (X[:, :, None] * X[:, None, :]).reshape(X.shape[0], 9)) / wsum
            cov = S2.reshape(-1, 3, 3) - m[:, :, None] * m[:, None, :]
            eye = torch.eye(3, device=X.device, dtype=X.dtype)[None]
            cov = cov + self.prior * (eye - n0[:, :, None] * n0[:, None, :])
            _, V = torch.linalg.eigh(cov)
            v = V[:, :, 0]
            v = v * torch.where((v * n0).sum(1, keepdim=True) < 0, -1.0, 1.0)
            n = (1.0 - self.blend) * n0 + self.blend * v
            n = n / n.norm(dim=1, keepdim=True).clamp(min=1e-12)
            vc = (X - self.reach * n).to(C.state.dtype)
            C.state[:, b0:b1] = torch.where(live[:, None], vc, C.state[:, b0:b1])
            if self.mid_block is not None:
                m0, m1 = C.state_schema[self.mid_block]
                mp = m
                if self.curvature:
                    mp = X + fit_surface_height(X, n, live, self.radius)[:, None] * n
                C.state[:, m0:m1] = torch.where(live[:, None], mp.to(C.state.dtype), C.state[:, m0:m1])
        return {}


from plexus.models.base import Exchange as _Exchange      # noqa: E402


def _type_mask(C, types):
    """A per-cell 0/1 mask of the cells whose type is in `types` (type names of the set, or type indices); None = every cell."""
    if types is None:
        return None
    names = list(getattr(C, "type_names", []) or [])
    idx = [names.index(t) if isinstance(t, str) else int(t) for t in types]
    nt = C.node_type.long()
    m = torch.zeros_like(nt, dtype=torch.bool)
    for i in idx:
        m |= nt == i
    return m


@register_operator("active_force", model="layer_slab", family="mechanics", set="particle", kind="exchange",
                   equation=r"""$$\mathbf a_p=-k\,\mathrm{sign}(d_p)\,\big(\lvert d_p\rvert-h\big)^{+}\,\mathbf n_c,\qquad d_p=(\mathbf x_p-\mathbf m_c)\cdot\mathbf n_c$$""",
                   title="hold each cell in the plane of its layer")
class ActiveForceLayerPlane(_Exchange):
    r"""a soft slab round the plane of the cell layer: a point farther than `half_thickness` from the layer's local plane, along the normal, is pushed back along it, so a layer that grows stays one cell thick; the plane is the one `cell_normal` fits, so it follows any shape of layer.
    particle -[containment]-> particle: reads each point's cell's normal (`centre_block`, the virtual centre `cell_normal` writes) and plane point (`mid_block`), returns a body acceleration for the MPM substep; the state is never written.

        \mathbf a_p=-k\,\mathrm{sign}(d_p)\,\big(\lvert d_p\rvert-h\big)^{+}\,\mathbf n_c,\qquad d_p=(\mathbf x_p-\mathbf m_c)\cdot\mathbf n_c
        a_p            the acceleration of point p
        k              `wall_stiffness`, the stiffness of the slab walls, per time squared
        d_p            the signed distance of point p from the plane of its cell's layer, along the normal
        h              `half_thickness`, the half thickness of the slab, world length (per cell from the cell block `half_thickness_block` when given)
        n_c            the unit normal of the layer at the cell c that owns p: from the cell's virtual centre (`centre_block`) through its centroid
        m_c            the point of the layer's plane at cell c (`mid_block`)

    Reference: Plexus exp29 (this work, 2026-10); the rigid-plate version of the same idea is `plate_confine`."""
    PARAM_UNITS = {'wall_stiffness': 'rate', 'half_thickness': 'length', 'types': 'count'}
    SYMBOLS = {'a_p': 'the acceleration of point p', 'k': 'the stiffness of the slab walls', 'd_p': "the signed distance of point p from its cell's layer plane", 'h': 'the half thickness of the slab',
               'n_c': 'the unit normal of the layer at the cell', 'm_c': "the point of the layer's plane at the cell"}
    PARAM_ROLES = {'cells': 'the_cell_set', 'centre_block': 'cell_block_with_each_cells_virtual_centre', 'mid_block': 'cell_block_with_the_point_of_the_layer_plane', 'half_thickness': 'half_thickness_of_the_slab', 'wall_stiffness': 'stiffness_of_the_slab_walls',
                   'half_thickness_block': 'cell_block_with_each_cells_slab_half_thickness', 'types': 'cell_types_the_slab_acts_on_default_all'}
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["centre_block", "mid_block"]
    MECHANISM_TAGS = ["confinement", "layer_normal", "monolayer"]
    REFERENCE = "Plexus exp29 (this work, 2026-10)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.cells = str(params.get("cells", "cell"))
        self.centre_block = str(params["centre_block"])
        self.mid_block = str(params["mid_block"])
        self.half = float(params.get("half_thickness", 0.6))
        self.half_block = params.get("half_thickness_block")        # a cell block read instead of `half_thickness`: the slab thickness as a knob of the optimisation (exp34)
        self.k = float(params.get("wall_stiffness", 1000.0))
        # `types: [outer]` (exp36, 2026-10-31): the slab acts on the cells of these types only. In a FILLED tissue the layer plane fitted through an
        # interior cell's neighbours runs through the middle of the tissue, and the slab of stiffness k drives every point of that cell at k x its
        # distance: the bulk spheroid inverted particles in the first substep and went non-finite by the 24th. None = every cell (the hollow vesicle).
        self.types = params.get("types")

    def forward(self, H, mask=None):
        P, C = H.level(self.at), H.level(self.cells)
        X = P.get("pos")
        par = P.parent.long().clamp(0, C.state.shape[0] - 1)
        a0, a1 = C.state_schema["pos"]
        b0, b1 = C.state_schema[self.centre_block]
        m0, m1 = C.state_schema[self.mid_block]
        dt_ = X.dtype
        n = C.state[:, a0:a1].to(dt_) - C.state[:, b0:b1].to(dt_)
        n = n / n.norm(dim=1, keepdim=True).clamp(min=1e-9)
        d = ((X - C.state[:, m0:m1].to(dt_)[par]) * n[par]).sum(1)
        half = self.half if self.half_block is None else C.state[:, C.state_schema[self.half_block][0]].to(dt_)[par]
        excess = torch.sign(d) * (d.abs() - half).clamp(min=0.0)
        acc = -self.k * excess[:, None] * n[par] * (P.occ > 0.5).to(dt_)[:, None]
        tm = _type_mask(C, self.types)
        if tm is not None:
            acc = acc * tm.to(dt_)[par][:, None]
        if mask is not None:
            acc = acc * mask[:, None].to(dt_)
        return {self.at: acc}


# `active_force[box_slab]` -- THE SAME SOFT WALL ON THE FACES OF A FIXED BOX (exp09, 2026-11-14): a molecular landscape is a thin slab of bodies seen face on,
# and nothing held MPM points inside a world-fixed box: `layer_slab` needs a layer plane fitted through each cell's neighbours (a crowd has none),
# `plate_confine` projects positions (kind structural), which moves an MPM point without its strain F knowing, so a body pressed on a plate stays flattened,
# and the engine's `boundary: wall` clamp does the same once per frame. A variant of layer_slab's law and not an edit of it: the same excess-distance wall,
# with the plane fixed in the world and one pair of faces per axis.
@register_operator("active_force", model="box_slab", family="mechanics", set="particle", kind="exchange",
                   equation=r"""$$\mathbf a_p=k\sum_{d=1}^{3}\Big[\big(l_d-x_{p,d}\big)^{+}-\big(x_{p,d}-u_d\big)^{+}\Big]\,\hat{\mathbf e}_d$$""",
                   title="hold every point inside a fixed box with soft walls")
class ActiveForceBoxSlab(_Exchange):
    r"""soft walls on the six faces of a box fixed in the world: a point beyond a face is pushed back along that face's axis in proportion to how far beyond it is, so bodies of material points stay in a slab without their positions being edited.
    particle -[containment]-> particle: reads pos, returns a body acceleration for the MPM substep; the state is never written.

        \mathbf a_p=k\sum_{d=1}^{3}\Big[\big(l_d-x_{p,d}\big)^{+}-\big(x_{p,d}-u_d\big)^{+}\Big]\,\hat{\mathbf e}_d
        a_p            the acceleration of point p
        k              `wall_stiffness`, the stiffness of the walls, per time squared (keep sqrt(k) x substep_dt well under 2)
        x_{p,d}        the coordinate of point p along axis d, world length
        l_d, u_d       `lo`, `hi`: the lower and upper face of the box on axis d, world length (default 0 and the world size)
        e_d            the unit vector of axis d
        (.)^+          the positive part: zero for a point inside the box

    Reference: Plexus exp09 (this work, 2026-11-14); the same wall on a fitted layer plane is active_force[layer_slab], the rigid-plate version plate_confine."""
    PARAM_UNITS = {'wall_stiffness': 'rate', 'lo': 'length', 'hi': 'length'}
    SYMBOLS = {'a_p': 'the acceleration of point p', 'k': 'the stiffness of the walls, per time squared', 'x_{p,d}': 'the coordinate of point p along axis d',
               'l_d, u_d': 'the lower and upper face of the box on axis d', 'e_d': 'the unit vector of axis d', '(.)^+': 'the positive part'}
    PARAM_ROLES = {'wall_stiffness': 'stiffness_of_the_box_walls', 'lo': 'lower_faces_of_the_box_xyz', 'hi': 'upper_faces_of_the_box_xyz'}
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["confinement", "soft_wall", "box"]
    REFERENCE = "Plexus exp09 (this work, 2026-11-14)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.k = float(params.get("wall_stiffness", 400.0))
        self.lo = None if params.get("lo") is None else [float(v) for v in params["lo"]]
        self.hi = None if params.get("hi") is None else [float(v) for v in params["hi"]]

    def forward(self, H, mask=None):
        P = H.level(self.at)
        X = P.get("pos")
        dt_ = X.dtype
        lo = torch.as_tensor(self.lo if self.lo is not None else [0.0, 0.0, 0.0], device=X.device, dtype=dt_)
        hi = torch.as_tensor(self.hi if self.hi is not None else [float(w) for w in H.world_size][:3], device=X.device, dtype=dt_)
        acc = self.k * ((lo[None] - X).clamp(min=0.0) - (X - hi[None]).clamp(min=0.0)) * (P.occ > 0.5).to(dt_)[:, None]
        if mask is not None:
            acc = acc * mask[:, None].to(dt_)
        return {self.at: acc}


from plexus.operators.adhesion_ops import SeedCellGrid as _SeedCellGrid      # noqa: E402


@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="cell_grid_pulls",
                   title="lay out arrays of cube cells with several pulled cells per array")
class SeedCellGridPulls(_SeedCellGrid):
    r"""seed_positions[cell_grid] where each array may have SEVERAL pulled cells (the parent allows one per array), for a chain of cells stretched from both ends.
    pt -> pt, cell: as cell_grid; per pulled cell the block `pull` is the unit direction `cell_pull` drags it along.

    `pulls`: for every array a list of {cell: [i, k], dir: [dx, dy, dz]} (cell = the column and layer of the cube in its array), or one such list for all arrays.

    Reference: Plexus exp27, the cell adhesion testbench (this work, 2026-10); Chu, Y.-S. et al. (2004). J. Cell Biol. 167:1183-1194."""
    PARAM_ROLES = dict(_SeedCellGrid.PARAM_ROLES, pulls="per_array_list_of_pulled_cells_and_directions")

    def __init__(self, params, device="cpu"):
        pulls = params.get("pulls")
        super().__init__({k: v for k, v in params.items() if k != "pulls"}, device)
        if pulls is not None and len(pulls) and isinstance(pulls[0], dict):
            pulls = [pulls] * (self.arrays[0] * self.arrays[1])
        if pulls is not None and len(pulls) != self.arrays[0] * self.arrays[1]:
            raise ValueError(f"seed_positions[cell_grid_pulls]: {len(pulls)} pull lists for {self.arrays[0] * self.arrays[1]} arrays")
        self.pulls = pulls

    def forward(self, H, mask=None):
        out = super().forward(H, mask)
        C = H.level(self.cell_set)
        if self.pulls is None:
            return out
        for need in ("pull", "arr", "ij"):
            if need not in C.state_schema:
                raise ValueError(f"seed_positions[cell_grid_pulls]: the cell set needs the block {need!r}")
        p0, p1 = C.state_schema["pull"]
        a0, _ = C.state_schema["arr"]
        i0, _ = C.state_schema["ij"]
        st = C.state.clone()
        n_set = 0
        for a, plist in enumerate(self.pulls):
            for pl in plist:
                i, k = int(pl["cell"][0]), int(pl["cell"][1])
                d = np.array([float(v) for v in pl["dir"]])
                d = d / max(np.linalg.norm(d), 1e-12)
                hit = (st[:, a0] == float(a)) & (st[:, i0] == float(i)) & (st[:, i0 + 1] == float(k)) & (C.occ > 0.5)
                if int(hit.sum()) != 1:
                    raise ValueError(f"seed_positions[cell_grid_pulls]: array {a} has {int(hit.sum())} cells at column {i}, layer {k}")
                st[hit, p0:p1] = torch.as_tensor(d, dtype=st.dtype, device=st.device)
                n_set += 1
        C.state = st
        print(f"[seed_positions:cell_grid_pulls] {n_set} pulled cells in {len(self.pulls)} arrays", flush=True)
        return out


def _unit_quaternion_matrix(q):
    """The rotation matrix of the unit quaternion q = (w, x, y, z), float64 [3, 3]."""
    w, x, y, z = (q / q.norm()).tolist()
    return torch.tensor([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=torch.float64)


def _axis_angle_matrix(axis, deg):
    """The rotation matrix of `deg` degrees about `axis` (Rodrigues), float64 [3, 3]."""
    a = torch.tensor([float(v) for v in axis], dtype=torch.float64)
    a = a / a.norm()
    t = float(deg) * np.pi / 180.0
    K = torch.tensor([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]], dtype=torch.float64)
    return torch.eye(3, dtype=torch.float64) + float(np.sin(t)) * K + (1.0 - float(np.cos(t))) * (K @ K)


class _FrameRotation(_Lateral):
    """Shared body of `frame_rotation[apply]` and `frame_rotation[undo]`: a change of coordinates of the state the MPM substep reads and writes."""
    EMIT = None
    MAY_MUTATE_INTEGRATED_STATE = True              # a change of coordinates, not a dynamics operator: it moves no body, only the frame the next operators see
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = True                           # under a tape it rebinds instead of writing in place
    MECHANISM_TAGS = ["frame_rotation", "isotropy", "discretisation_bias"]
    PARAM_UNITS = {'fixed': 'dimensionless', 'seed': 'count'}
    PARAM_ROLES = {'positions': 'per_set_blocks_that_are_points', 'vectors': 'per_set_blocks_that_are_directions', 'strain': 'rotate_the_point_F_and_C_of_the_at_set',
                   'centre': 'point_the_frame_turns_about', 'fixed': 'test_hook_one_fixed_axis_and_angle_instead_of_a_random_rotation', 'seed': 'random_seed_of_the_rotations',
                   'live_only': 'turn_only_the_live_points_of_at_and_leave_the_dormant_slots_of_the_pool_as_they_are'}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.positions = {str(k): [str(b) for b in v] for k, v in (params.get("positions") or {self.at: ["pos"]}).items()}
        self.vectors = {str(k): [str(b) for b in v] for k, v in (params.get("vectors") or {self.at: ["vel"]}).items()}
        self.strain = bool(params.get("strain", True))
        self.centre = None if params.get("centre") is None else [float(v) for v in params["centre"]]
        self.fixed = None if params.get("fixed") is None else [float(v) for v in params["fixed"]]
        if self.fixed is not None and len(self.fixed) != 4:
            raise ValueError("frame_rotation: `fixed` is [ax, ay, az, degrees]")
        self.seed = int(params.get("seed", 0))
        self.live_only = bool(params.get("live_only", False))                 # turn only the live points of `at` (occ > 0), not the dormant slots of a big pool (no tape only)

    def _turn(self, H, R, c, cfg):
        """Move every listed block into the frame turned by R (float64 [3, 3]) about c: points c + R (x - c), directions R v, the strain F -> R F and C -> R C R^T, the frame-level delta R a."""
        p = H.level(self.at)
        if getattr(H, "part_accel", None) is not None:
            raise NotImplementedError("frame_rotation: `H.part_accel` is a world-frame field it does not turn")
        for bad in ("act_stress", "extra_stress", "F_res_inv"):
            if getattr(p, bad, None) is not None:
                raise NotImplementedError(f"frame_rotation: the point tensor `{bad}` lives in the world frame and is not turned here")
        positions, vectors, strain = cfg
        Rt = R.T
        grad = torch.is_grad_enabled()                                    # under a tape the state is REBOUND (a new tensor), never written in place

        def put(L, b0, b1, new):
            if grad:
                L.state = torch.cat([L.state[..., :b0], new.to(L.state.dtype), L.state[..., b1:]], dim=-1)
            else:
                L.state[..., b0:b1] = new.to(L.state.dtype)
        live = torch.nonzero(p.occ > 0).squeeze(1) if (self.live_only and not grad) else None      # rows of `at` that are turned; None: all of them
        with torch.set_grad_enabled(grad):
            for setname, blocks in positions.items():
                L = H.level(setname)
                for b in blocks:
                    b0, b1 = L.state_schema[b]
                    if live is not None and L is p:
                        L.state[live, b0:b1] = ((L.state[live, b0:b1].double() - c) @ Rt + c).to(L.state.dtype)
                    else:
                        put(L, b0, b1, (L.state[..., b0:b1].double() - c) @ Rt + c)
            for setname, blocks in vectors.items():
                L = H.level(setname)
                for b in blocks:
                    b0, b1 = L.state_schema[b]
                    if live is not None and L is p:
                        L.state[live, b0:b1] = (L.state[live, b0:b1].double() @ Rt).to(L.state.dtype)
                    else:
                        put(L, b0, b1, L.state[..., b0:b1].double() @ Rt)
            if strain:
                if live is not None:
                    p.F[live] = (R @ p.F[live].double()).to(p.F.dtype)
                    p.C[live] = (R @ p.C[live].double() @ Rt).to(p.C.dtype)
                else:
                    F2, C2 = (R @ p.F.double()).to(p.F.dtype), (R @ p.C.double() @ Rt).to(p.C.dtype)
                    if grad:
                        p.F, p.C = F2, C2
                    else:
                        p.F.copy_(F2)
                        p.C.copy_(C2)
            for setname in set(positions) | set(vectors):
                d = H._delta.get(setname)
                if d is not None and d.shape[-1] == 3:
                    if live is not None and H.level(setname) is p:
                        d[live] = (d[live].double() @ Rt).to(d.dtype)
                    else:
                        d2 = (d.double() @ Rt).to(d.dtype)
                        if grad:
                            H._delta[setname] = d2
                        else:
                            d.copy_(d2)


@register_operator("frame_rotation", model="apply", family="harness", set="particle", kind="lateral",
                   equation=r"""$$\mathbf x\leftarrow\mathbf c+\mathbf Q(\mathbf x-\mathbf c),\quad \mathbf v\leftarrow\mathbf Q\mathbf v,\quad \mathbf F\leftarrow\mathbf Q\mathbf F,\quad \mathbf C\leftarrow\mathbf Q\mathbf C\mathbf Q^{\mathsf T}$$""",
                   title="turn the whole body by a random rotation before the substep")
class FrameRotationApply(_FrameRotation):
    r"""turn every point, direction, strain and velocity gradient of the body by one random rotation about its centre, so the grids and lattices the next operators use see the body at a new orientation; `frame_rotation[undo]` turns it back.
    particle -[frame]-> particle: reads and writes the blocks named in `positions` and `vectors`, the point strain `F` and velocity gradient `C` of `at`, and the frame-level delta; keeps Q and c for the undo.

        \mathbf x\leftarrow\mathbf c+\mathbf Q(\mathbf x-\mathbf c),\quad \mathbf v\leftarrow\mathbf Q\mathbf v,\quad \mathbf F\leftarrow\mathbf Q\mathbf F,\quad \mathbf C\leftarrow\mathbf Q\mathbf C\mathbf Q^{\mathsf T}
        x              a point block (a position)
        v              a vector block (a velocity or direction)
        c              `centre`, or the centroid of the live points of `at`
        Q              a rotation drawn uniformly over all rotations (a uniform unit quaternion), a new one at every call; `fixed` gives one rotation instead
        F              the point's deformation gradient (the strain it carries): turned on the left, as the material law is unchanged by turning the body
        C              the point's affine velocity gradient

    Reference: the rotation augmentation of ParticleGraph, Affine_Particle (rotate the inputs, run, rotate the outputs back); Plexus exp29 (this work, 2026-10)."""
    SYMBOLS = {'x': 'a point block, a position', 'v': 'a vector block, a velocity or direction', 'c': 'the point the frame turns about', 'Q': 'a random rotation, new at every call', 'F': "the point's deformation gradient",
               'C': "the point's affine velocity gradient"}
    REFERENCE = "ParticleGraph Affine_Particle rotation augmentation; Plexus exp29 (this work, 2026-10)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self._gen = torch.Generator(device="cpu").manual_seed(self.seed)

    def forward(self, H, mask=None):
        p = H.level(self.at)
        dev = p.state.device
        if self.fixed is not None:
            R = _axis_angle_matrix(self.fixed[:3], self.fixed[3])
        else:
            R = _unit_quaternion_matrix(torch.randn(4, generator=self._gen, dtype=torch.float64))
        R = R.to(dev)
        if self.centre is not None:
            c = torch.tensor(self.centre, dtype=torch.float64, device=dev)
        else:
            a0, a1 = p.state_schema["pos"]
            w = (p.occ > 0).double()
            c = ((p.state[..., a0:a1].double() * w[:, None]).sum(0) / w.sum().clamp_min(1.0)).detach()      # a bookkeeping point, the same in the undo: it carries no gradient
        if not hasattr(H, "_frame_rotation"):
            H._frame_rotation = []
        cfg = (self.positions, self.vectors, self.strain)
        H._frame_rotation.append((R, c, cfg))
        self._turn(H, R, c, cfg)
        return {}


@register_operator("frame_rotation", model="undo", family="harness", set="particle", kind="lateral",
                   equation=r"""$$\mathbf x\leftarrow\mathbf c+\mathbf Q^{\mathsf T}(\mathbf x-\mathbf c),\quad \mathbf v\leftarrow\mathbf Q^{\mathsf T}\mathbf v,\quad \mathbf F\leftarrow\mathbf Q^{\mathsf T}\mathbf F,\quad \mathbf C\leftarrow\mathbf Q^{\mathsf T}\mathbf C\mathbf Q$$""",
                   title="turn the body back after the substep")
class FrameRotationUndo(_FrameRotation):
    r"""turn everything `frame_rotation[apply]` turned back by the inverse rotation about the same point, so what the substep did in the turned frame is carried into the world frame.
    particle -[frame]-> particle: as apply, with the Q, c and block lists that apply kept (the undo line of a spec needs only `at`).

        \mathbf x\leftarrow\mathbf c+\mathbf Q^{\mathsf T}(\mathbf x-\mathbf c),\quad \mathbf v\leftarrow\mathbf Q^{\mathsf T}\mathbf v,\quad \mathbf F\leftarrow\mathbf Q^{\mathsf T}\mathbf F,\quad \mathbf C\leftarrow\mathbf Q^{\mathsf T}\mathbf C\mathbf Q
        Q              the rotation of the matching apply
        c              the point the matching apply turned about

    Reference: as frame_rotation[apply]."""
    SYMBOLS = {'Q': 'the rotation of the matching apply', 'c': 'the point the matching apply turned about'}
    REFERENCE = "ParticleGraph Affine_Particle rotation augmentation; Plexus exp29 (this work, 2026-10)."

    def forward(self, H, mask=None):
        stack = getattr(H, "_frame_rotation", None)
        if not stack:
            raise RuntimeError("frame_rotation[undo] has no matching frame_rotation[apply] before it in the schedule")
        R, c, cfg = stack.pop()
        self._turn(H, R.T, c, cfg)
        return {}


from plexus.operators.adhesion_ops import CellContactVolumeInterfaceSparse as _ContactSparse      # noqa: E402


@register_operator("pair_potential", model="cell_contact_iso", family="interaction", set="particle", kind="exchange",
                   equation=r"""$$\nabla^{2}\phi\;\approx\;\frac{1}{30\,\Delta x^{2}}\Big(14\,\Sigma_{\rm face}+3\,\Sigma_{\rm edge}+\Sigma_{\rm corner}-128\,\phi_n\Big)$$""",
                   title="the contact tension of the sparse lattice with a rotation-symmetric Laplacian")
class CellContactVolumeInterfaceSparseIso(_ContactSparse):
    r"""pair_potential[cell_contact] with the 27-point isotropic Laplacian in place of the seven-point one, so the interface tension of a contact does not prefer the axes of the lattice.
    particle -> particle: as cell_contact; only the Laplacian of the volume fractions that gives the contact area differs.

        \nabla^{2}\phi\;\approx\;\frac{1}{30\,\Delta x^{2}}\Big(14\,\Sigma_{\rm face}+3\,\Sigma_{\rm edge}+\Sigma_{\rm corner}-128\,\phi_n\Big)
        \phi            a cell's volume fraction on the lattice
        \phi_n          its value at the node n
        \Sigma_{face}   the sum of \phi over the 6 nodes that share a face with n
        \Sigma_{edge}   the sum over the 12 nodes that share an edge with n
        \Sigma_{corner} the sum over the 8 nodes that share a corner with n
        \Delta x        the lattice spacing

    The seven-point stencil has a leading error (dx^2 / 12)(d_x^4 + d_y^4 + d_z^4) phi, which is not a function of the distance alone; with these weights the leading error is (dx^2 / 12) |grad|^4 phi, the same in every direction (Patra, M. & Karttunen, M. (2006). Numer. Methods Partial Differ. Equ. 22:936-953).

    Reference: Patra, M. & Karttunen, M. (2006). Numer. Methods Partial Differ. Equ. 22:936-953; Plexus exp29 (this work, 2026-10)."""
    PARAM_UNITS = {}
    SYMBOLS = {'\\phi': "a cell's volume fraction on the lattice", '\\phi_n': 'its value at the node n', '\\Sigma_{face}': 'the sum of phi over the 6 face neighbours of n',
               '\\Sigma_{edge}': 'the sum over the 12 edge neighbours', '\\Sigma_{corner}': 'the sum over the 8 corner neighbours', '\\Delta x': 'the lattice spacing'}
    REFERENCE = "Patra, M. & Karttunen, M. (2006). Numer. Methods Partial Differ. Equ. 22:936-953."
    _W = {1: 14.0 / 30.0, 2: 3.0 / 30.0, 3: 1.0 / 30.0}                  # the weight of a neighbour by the number of axes it is offset along

    def _lattice_laplacian(self, A, per):
        def P(B, d):                                                      # the sum of the two neighbours along axis d
            return self._shift(B, d, 1, per and d == 0) + self._shift(B, d, -1, per and d == 0)
        Px, Py, Pz = P(A, 0), P(A, 1), P(A, 2)
        Pxy = P(Px, 1)
        edge = Pxy + P(Px, 2) + P(Py, 2)
        corner = P(Pxy, 2)
        return (14.0 * (Px + Py + Pz) + 3.0 * edge + corner - 128.0 * A) / 30.0

    def _own_laplacian(self, phi_u, uk, ck, ix, iy, iz, shape, per, Nw):
        Ny, Nz = shape[1], shape[2]
        Lown = (-128.0 / 30.0) * phi_u
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for oz in (-1, 0, 1):
                    k = abs(ox) + abs(oy) + abs(oz)
                    if k == 0:
                        continue
                    nbs = []
                    for a_, o_, n_, d in ((ix, ox, shape[0], 0), (iy, oy, Ny, 1), (iz, oz, Nz, 2)):
                        nb = a_ + o_
                        nbs.append(torch.remainder(nb, n_) if (d == 0 and per) else nb.clamp(0, n_ - 1))
                    fnb = (nbs[0] * Ny + nbs[1]) * Nz + nbs[2]
                    kq = ck * Nw + fnb
                    pos = torch.searchsorted(uk, kq).clamp_(max=uk.numel() - 1)
                    Lown = Lown + self._W[k] * torch.where(uk[pos] == kq, phi_u[pos], torch.zeros_like(phi_u))
        return Lown
