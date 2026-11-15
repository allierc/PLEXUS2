"""A basement membrane (BM) shell around a spheroid of MPM (material point) cells -- exp27, 2026-10-20.

The goal paper (Tauber, Morales, Li, Cui, Stringer, Choi, Wang & Mahadevan, "Basement membrane dynamics governs salivary gland budding")
models the BM as an elastic interface that resists in-plane STRETCH (modulus mu = E_y z) and out-of-plane BENDING (kappa = E_y z^3), z the
BM thickness, with an extra stiffness kappa_e against OUTWARD bending, relaxing viscously (the interface friction zeta, coarse-graining the
cells' rearrangements) and loaded by the growth of the cells inside it. This module is that interface for a cloud of cells made of material
points, which the older spring-graph BM of exp11 (membrane_ops.py) cannot touch: its contact needs a vertex mesh.

    seed_positions[bm_sphere]   nodes on a Fibonacci sphere just outside the outermost points of a set (`radius: auto`)
    bm_shell                    the interface itself: stretch springs on a neighbour graph built once from the seeded nodes (k = sqrt(3)/2 E_y z,
                                the spring constant whose triangulated sheet has the 2-D modulus E_y z), an umbrella-operator bending term
                                (kappa = E_y z^3, the extra kappa_e on outward bending), a Maxwell relaxation of the rest lengths and of the
                                rest curvature (`tau`, 0 = elastic), bonds that break past a strain (`break_strain`, a hole); overdamped
    bm_cell_contact             exchange, every MPM substep: the nodes and the points of the cells push each other apart closer than `r0` and
                                the cells of `adh_types` are held to the nodes by a tether out to `adhesion_reach` (the cell-matrix adhesion); equal and
                                opposite -- the point side enters the MPM substep as F / m, the node side is summed into a buffer
    bm_cell_apply               lateral, once per frame after the substeps: the mean of that buffer over the substeps, divided by the friction,
                                is the nodes' velocity from the cells

WHY A PAIRWISE CONTACT AND NOT THE CELLS' LATTICE. The nodes were first laid, as one more body, on the lattice of `pair_potential[cell_contact_dense]`
(its overlap energy and its interface tension, a negative outer-membrane entry as the cell-matrix adhesion). That is stable and conserves momentum
only if the field it samples is smooth on the lattice; a membrane one to two lattice spacings thick is not, and the point and node forces differed
by a factor of two in a test of one cell pressed on it. The pairwise form is exact in momentum. Its two requirements, found by running it: the
nodes must be as close as the points (`spacing` = the point spacing 0.125, or points slip between them), and the repulsion must be on the scale of the
cells' own contact (k_rep 60, not 3: the cells' contact is 10 times stiffer and pushed points straight through).

WHY NOT bm_bond. exp11's `bm_bond` is the same spring graph, but it carries the machinery of the vertex-mesh rig (secretion, rebonding, the
`membrane_alive` and `membrane_new` hand-offs) and has no bending term; the paper's control knob z needs both stretch and bending, and the
cells here are not a mesh. This is a variant in its own module.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from plexus.models.base import Exchange, Lateral, Seed
from plexus.models.registry import register_operator

SQRT3_2 = 0.5 * math.sqrt(3.0)


def fibonacci_sphere(n, centre, radius):
    """n quasi-uniform points on a sphere (the golden-angle spiral), as a float64 array [n, 3]."""
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    th = math.pi * (1.0 + math.sqrt(5.0)) * k
    return np.asarray(centre, float) + radius * np.stack([np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)], 1)


def stretch_constant(E_y, z):
    """The spring constant of one bond of a triangulated sheet whose 2-D stretching modulus is E_y z: k = sqrt(3)/2 E_y z."""
    return SQRT3_2 * float(E_y) * float(z)


def bending_modulus(E_y, z):
    """The paper's bending modulus kappa = E_y z^3."""
    return float(E_y) * float(z) ** 3


def umbrella_stiffness(kappa, spacing):
    """The coefficient kb of the umbrella energy E = sum_i kb / 2 |du_i|^2 that carries the bending modulus kappa on a sheet of node spacing s:
    the change of curvature is 4 |du| / s^2 and a node owns the area sqrt(3)/2 s^2, so kb = 8 sqrt(3) kappa / s^2."""
    return 8.0 * math.sqrt(3.0) * float(kappa) / float(spacing) ** 2


def radius_pairs(P, N, r):
    """All pairs (i, j) with |P_i - N_j| < r, P [m, 3] and N [n, 3] on one device: the nodes are binned into cubes of side r, sorted by bin, and
    every point looks into its 27 neighbouring bins (a binary search per bin). Returns (i, j, d) with d = P_i - N_j."""
    dev = P.device
    if P.shape[0] == 0 or N.shape[0] == 0:
        e = torch.zeros(0, dtype=torch.long, device=dev)
        return e, e.clone(), torch.zeros(0, 3, dtype=P.dtype, device=dev)
    lo = N.min(0).values - r
    nb = torch.floor((N - lo) / r).long()
    mx = nb.max(0).values
    D = mx + 5
    key = lambda b: ((b[..., 0] + 2) * D[1] + (b[..., 1] + 2)) * D[2] + (b[..., 2] + 2)      # noqa: E731  (bins -2 .. max + 2 stay non-negative)
    order = torch.argsort(key(nb))
    sk = key(nb)[order]
    bp = torch.floor((P - lo) / r).long()
    ok = ((bp >= -1) & (bp <= (mx + 1))).all(1)
    pidx = ok.nonzero().squeeze(1)
    bp = bp[pidx]
    o = torch.arange(-1, 2, device=dev)
    offs = torch.stack(torch.meshgrid(o, o, o, indexing="ij"), -1).reshape(-1, 3)
    I, J = [], []
    for off in offs:
        kq = key(bp + off)
        a = torch.searchsorted(sk, kq, right=False)
        b = torch.searchsorted(sk, kq, right=True)
        cnt = b - a
        tot = int(cnt.sum())
        if tot == 0:
            continue
        rep_i = pidx.repeat_interleave(cnt)
        start = a.repeat_interleave(cnt)
        csum = torch.cumsum(cnt, 0) - cnt
        within = torch.arange(tot, device=dev) - csum.repeat_interleave(cnt)
        I.append(rep_i)
        J.append(order[start + within])
    if not I:
        e = torch.zeros(0, dtype=torch.long, device=dev)
        return e, e.clone(), torch.zeros(0, 3, dtype=P.dtype, device=dev)
    I, J = torch.cat(I), torch.cat(J)
    d = P[I] - N[J]
    near = d.norm(dim=1) < r
    return I[near], J[near], d[near]


@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="bm_sphere",
                   title="lay a basement membrane on a sphere outside a set of points", equation=r"""$$\mathbf x_i=\mathbf c+R\,\hat{\mathbf f}_i,\quad i=0\ldots n-1,\qquad R=Q_{p}\big(\lvert\mathbf x_k-\mathbf c\rvert\big)+\delta\ \ (\text{radius: auto})$$""")
class SeedBMSphere(Seed):
    r"""the nodes of a spring-graph basement membrane on a Fibonacci sphere of radius R about `centre`, R the `percentile` of the cells' point distances plus `offset` when `radius: auto`.
    bm_node -> bm_node: writes `pos` and `alive` of the first n nodes (the set must hold at least n; an error says how many are needed).

        \mathbf x_i=\mathbf c+R\,\hat{\mathbf f}_i,\quad i=0\ldots n-1,\qquad R=Q_{p}\big(\lvert\mathbf x_k-\mathbf c\rvert\big)+\delta\ \ (\text{radius: auto})
        x_i            the position of node i
        c              `centre`, the sphere's centre
        R              `radius`, or with `radius: auto` the percentile Q_p of the live points' distances from the centre plus the offset
        \hat f_i       the i-th direction of the Fibonacci sphere (equal-area points)
        Q_p            the `percentile` (default 99.5) of the distances of `from_set`'s live points
        \delta         `offset` (default 0.3, the contact distance r0 of bm_cell_contact)

    Reference: Plexus (this work); the Fibonacci sphere: Gonzalez, A. (2010). Measurement of areas on a sphere using Fibonacci and latitude-longitude lattices. Math. Geosci. 42:49-64."""
    PARAM_UNITS = {'centre': 'length', 'radius': 'length', 'offset': 'length', 'percentile': 'dimensionless', 'spacing': 'length'}
    SYMBOLS = {'x_i': 'the position of node i', 'c': "`centre`, the sphere's centre", 'R': "`radius`, or with `radius: auto` the percentile Q_p of the live points' distances from the centre plus the offset", '\\hat f_i': 'the i-th direction of the Fibonacci sphere (equal-area points)', 'Q_p': "the `percentile` (default 99.5) of the distances of `from_set`'s live points", '\\delta': '`offset` (default 0.3, the contact distance r0 of bm_cell_contact)'}
    REQUIRES_PARAMS = ["centre"]
    SUPPORTED_DIMS = [3]
    PARAM_ROLES = {"centre": "sphere_centre_world", "radius": "radius_world_or_auto", "spacing": "node_spacing_world",
                   "offset": "gap_between_the_outermost_points_and_the_nodes", "from_set": "the_set_whose_surface_it_wraps",
                   "percentile": "percentile_of_the_point_radii_taken_as_the_surface"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "bm_node")
        self.c = np.array([float(v) for v in params["centre"]])
        self.radius = params.get("radius", "auto")
        self.spacing = float(params.get("spacing", 0.25))
        self.offset = float(params.get("offset", 0.3))
        self.from_set = str(params.get("from_set", "pt"))
        self.percentile = float(params.get("percentile", 99.5))

    def forward(self, H, mask=None):
        L = H.level(self.at)
        if str(self.radius) == "auto":
            P = H.level(self.from_set)
            X = P.get("pos").detach()[(P.occ > 0.5)].double().cpu().numpy()
            R = float(np.percentile(np.linalg.norm(X - self.c, axis=1), self.percentile)) + self.offset
        else:
            R = float(self.radius)
        n = int(round(4.0 * math.pi * R * R / (SQRT3_2 * self.spacing ** 2)))
        have = int(L.get("pos").shape[0])
        if n > have:
            raise ValueError(f"seed_positions[bm_sphere]: a sphere of radius {R:.2f} at spacing {self.spacing:g} needs {n} nodes, the set has {have}")
        pos = np.tile(self.c, (have, 1))
        pos[:n] = fibonacci_sphere(n, self.c, R)
        p0, p1 = L.state_schema["pos"]
        st = L.state.clone()
        st[:, p0:p1] = torch.as_tensor(pos, dtype=st.dtype, device=st.device)
        L.state = st
        L.alive.zero_()
        L.alive[:n] = True
        print(f"[seed_positions:bm_sphere] {n} nodes on a sphere of radius {R:.2f} about {np.round(self.c, 2).tolist()} (spacing {self.spacing:g})", flush=True)
        return {}


@register_operator("bm_shell", family="mechanics", set="particle", kind="lateral", title="a stretchable, bendable basement membrane",
                   equation=r"""$$E=\sum_{(ij)}\tfrac{k}{2}\,(L_{ij}-l_{ij})^2+\sum_i\tfrac{k_b^{(i)}}{2}\,\lvert\mathbf u_i-\mathbf u_i^{0}\rvert^2,\quad k=\tfrac{\sqrt3}{2}E_yz,\ \ k_b=\tfrac{8\sqrt3\,\kappa}{s^2},\ \ \kappa=E_yz^3$$""")
class BMShell(Lateral):
    r"""the basement membrane of the budding goal paper on a node graph: stretch springs along every bond and an umbrella bending energy at every node, in the overdamped limit.
    bm_node -> bm_node: reads `pos`, emits a velocity F / `overdamped_gamma` (`emit: velocity`); the bond graph is built once from the live nodes (`max_neighbours` nearest within `cutoff` mean spacings).

        E=\sum_{(ij)}\tfrac{k}{2}\,(L_{ij}-l_{ij})^2+\sum_i\tfrac{k_b^{(i)}}{2}\,\lvert\mathbf u_i-\mathbf u_i^{0}\rvert^2,\quad k=\tfrac{\sqrt3}{2}E_yz,\ \ k_b=\tfrac{8\sqrt3\,\kappa}{s^2},\ \ \kappa=E_yz^3
        E              the shell's elastic energy
        k              the bond stiffness sqrt(3)/2 E_y z, force per length
        L_{ij}         the current length of bond ij
        l_{ij}         its rest length
        k_b^{(i)}      the bending stiffness at node i, 8 sqrt(3) (kappa + kappa_e [outward]) / s^2
        u_i            the umbrella vector at node i, the mean of its neighbours minus the node
        u_i^0          its rest value
        E_y            `E_y`, the membrane's Young modulus, stress
        z              `z`, the membrane thickness, length
        \kappa         the bending modulus E_y z^3, energy
        s              the mean node spacing

    Reference: Plexus (this work), after the module docstring's goal paper (Tauber, Wang & Mahadevan: the basement membrane as a growth-loaded interface); the umbrella bending energy of a spring-graph shell: Seung, H. S. & Nelson, D. R. (1988). Phys. Rev. A 38:1005-1018."""
    PARAM_UNITS = {'E_y': 'stress', 'z': 'length', 'kappa_e': 'energy', 'tau': 'time', 'cutoff': 'dimensionless', 'max_neighbours': 'count', 'overdamped_gamma': 'mass_per_time', 'substeps': 'count', 'break_strain': 'fraction', 'centre': 'length'}
    SYMBOLS = {'E': "the shell's elastic energy", 'k': 'the bond stiffness sqrt(3)/2 E_y z, force per length', 'L_{ij}': 'the current length of bond ij', 'l_{ij}': 'its rest length', 'k_b^{(i)}': 'the bending stiffness at node i, 8 sqrt(3) (kappa + kappa_e [outward]) / s^2', 'u_i': 'the umbrella vector at node i, the mean of its neighbours minus the node', 'u_i^0': 'its rest value', 'E_y': "`E_y`, the membrane's Young modulus, stress", 'z': '`z`, the membrane thickness, length', '\\kappa': 'the bending modulus E_y z^3, energy', 's': 'the mean node spacing'}
    EMIT = "velocity"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["centre"]
    PARAM_ROLES = {"E_y": "membrane_modulus", "z": "membrane_thickness_world", "kappa_e": "extra_stiffness_against_outward_bending",
                   "tau": "remodelling_time", "overdamped_gamma": "interface_friction", "centre": "sphere_centre_world",
                   "break_strain": "strain_at_which_a_bond_breaks", 'cutoff': 'bond_cutoff_in_mean_spacings', 'max_neighbours': 'bonds_per_node', 'substeps': 'substeps_per_frame'}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "bm_node")
        self.c = torch.tensor([float(v) for v in params["centre"]], dtype=torch.float64)
        self.E_y = float(params.get("E_y", 200.0))
        self.z = float(params.get("z", 0.25))
        self.kappa_e = float(params.get("kappa_e", 0.0))
        self.tau = float(params.get("tau", 0.0))
        self.gamma = float(params.get("overdamped_gamma", 50.0))
        self.max_nb = int(params.get("max_neighbours", 6))
        self.cutoff = float(params.get("cutoff", 1.6))
        self.break_strain = float(params.get("break_strain", 0.0))
        # `substeps` (default 1; `auto`): the overdamped relaxation of the nodes under the shell's own forces is integrated in this many explicit
        # sub-steps of the frame (the cells' load, `bm_cell_apply`, is held over the frame). One step is stable only for dt (12 k + 4 kb) / gamma < 2,
        # and the bending term kb = 8 sqrt(3) kappa / s^2 grows as z^3 and as 1 / s^2: a thick membrane of closely spaced nodes needs many.
        # `auto` picks the fewest that keep each sub-step under 1.
        self.substeps = params.get("substeps", 1)
        self.k = stretch_constant(self.E_y, self.z)
        self.kappa = bending_modulus(self.E_y, self.z)
        self.i = self.j = self.l0 = self.u0 = self.nrm = self.deg = None
        self.spacing = None

    # ---- the graph --------------------------------------------------------------------------------------------------------------
    def _build(self, pos, alive):
        from scipy.spatial import cKDTree
        idx = alive.nonzero(as_tuple=True)[0]
        x = pos[idx].detach().double().cpu().numpy()
        tree = cKDTree(x)
        nn = tree.query(x, k=2)[0][:, 1]
        s = float(np.median(nn))
        k = min(self.max_nb + 1, len(x))
        d, nb = tree.query(x, k=k, distance_upper_bound=self.cutoff * s)
        ii, jj = [], []
        for col in range(1, k):
            ok = np.isfinite(d[:, col])
            ii.append(np.arange(len(x))[ok]); jj.append(nb[ok, col])
        ii, jj = np.concatenate(ii), np.concatenate(jj)
        a, b = np.minimum(ii, jj), np.maximum(ii, jj)
        key = np.unique(a.astype(np.int64) * (len(x) + 1) + b)
        a, b = key // (len(x) + 1), key % (len(x) + 1)
        idx_np = idx.cpu().numpy()
        dev = pos.device
        self.i = torch.as_tensor(idx_np[a], device=dev)
        self.j = torch.as_tensor(idx_np[b], device=dev)
        self.spacing = s
        self.alive_bond = torch.ones(len(a), dtype=torch.bool, device=dev)
        p = pos.detach()
        self.l0 = (p[self.j] - p[self.i]).norm(dim=1)
        self.deg = torch.zeros(pos.shape[0], dtype=pos.dtype, device=dev)
        self.deg.index_add_(0, self.i, torch.ones_like(self.l0)); self.deg.index_add_(0, self.j, torch.ones_like(self.l0))
        self.u0 = self._umbrella(p)
        c = self.c.to(device=dev, dtype=pos.dtype)
        r = p - c
        self.nrm = r / r.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.kb0 = umbrella_stiffness(self.kappa, s)
        self.kb_e = umbrella_stiffness(self.kappa_e, s) if self.kappa_e > 0.0 else 0.0
        print(f"[bm_shell] {len(a)} bonds on {len(x)} nodes ({2.0 * len(a) / len(x):.1f} per node), spacing {s:.3f}, k {self.k:.3g} (E_y {self.E_y:g}, z {self.z:g}), "
              f"kappa {self.kappa:.3g} (kb {self.kb0:.3g}), kappa_e {self.kappa_e:g}, tau {self.tau:g}, gamma {self.gamma:g}", flush=True)

    def _umbrella(self, p):
        S = torch.zeros_like(p)
        mask = self.alive_bond
        i, j = self.i[mask], self.j[mask]
        S.index_add_(0, i, p[j]); S.index_add_(0, j, p[i])
        deg = torch.zeros(p.shape[0], dtype=p.dtype, device=p.device)
        deg.index_add_(0, i, torch.ones(i.shape[0], dtype=p.dtype, device=p.device)); deg.index_add_(0, j, torch.ones(j.shape[0], dtype=p.dtype, device=p.device))
        self._deg_now = deg
        return torch.where(deg[:, None] > 0, S / deg.clamp_min(1.0)[:, None] - p, torch.zeros_like(p))

    # ---- energy and forces (pure in `pos`: the tests differentiate them) -----------------------------------------------------------
    def _kb(self, du):
        if self.kb_e == 0.0:
            return torch.full((du.shape[0],), self.kb0, dtype=du.dtype, device=du.device)
        outward = ((du * self.nrm).sum(1) < 0.0).to(du.dtype)
        return self.kb0 + self.kb_e * outward

    def stretch_energy(self, pos):
        """E_S at `pos` [n, 3] (the nodes of the graph), at the present rest state."""
        m = self.alive_bond
        i, j = self.i[m], self.j[m]
        L = (pos[j] - pos[i]).norm(dim=1)
        return 0.5 * self.k * ((L - self.l0[m]) ** 2).sum()

    def bending_energy(self, pos):
        """E_B at `pos`, at the present rest state."""
        du = self._umbrella(pos) - self.u0
        return 0.5 * (self._kb(du) * (du * du).sum(1)).sum()

    def energy(self, pos):
        """E_S + E_B at `pos`."""
        return self.stretch_energy(pos) + self.bending_energy(pos)

    def forces(self, pos):
        """-dE/dx, [n, 3]."""
        m = self.alive_bond
        i, j = self.i[m], self.j[m]
        d = pos[j] - pos[i]
        L = d.norm(dim=1).clamp_min(1e-12)
        fb = (self.k * (L - self.l0[m]) / L)[:, None] * d
        F = torch.zeros_like(pos)
        F.index_add_(0, i, fb); F.index_add_(0, j, -fb)
        du = self._umbrella(pos) - self.u0
        deg = self._deg_now
        kb = self._kb(du)
        w = kb[:, None] * du / deg.clamp_min(1.0)[:, None]
        G = torch.zeros_like(pos)
        G.index_add_(0, i, w[j]); G.index_add_(0, j, w[i])
        return F + kb[:, None] * du - G

    def forward(self, H, mask=None):
        L = H.level(self.at)
        pos = L.get("pos")
        if self.i is None:
            self._build(pos, L.alive)
            dt = float(getattr(H, "dt", 1.0) or 1.0)
            ratio = dt * (12.0 * self.k + 4.0 * (self.kb0 + self.kb_e)) / self.gamma
            if str(self.substeps) == "auto":
                self.n_sub = max(1, int(math.ceil(ratio / 1.0)))
            else:
                self.n_sub = int(self.substeps)
            ratio = ratio / self.n_sub
            if ratio >= 2.0:
                raise ValueError(f"bm_shell: the stiffest mode needs dt (12 k + 4 kb) / (gamma substeps) < 2 and has {ratio:.2f} (dt {dt:g}, k {self.k:.3g}, "
                                 f"kb {self.kb0 + self.kb_e:.3g}, gamma {self.gamma:g}, substeps {self.n_sub}): raise `substeps` (or `auto`), `overdamped_gamma`, or lower E_y z")
        p = pos.detach()
        dt = float(getattr(H, "dt", 1.0) or 1.0)
        alive_f = L.alive[:, None].to(p.dtype)
        n_sub = int(getattr(self, "n_sub", 1))
        if n_sub > 1:                                         # relax under the shell's own forces in n_sub steps; hand the engine the mean velocity
            x = p.clone()
            for _ in range(n_sub):
                x = x + (dt / n_sub) * (self.forces(x) * alive_f) / self.gamma
            vel = (x - p) / dt
        else:
            vel = (self.forces(p) * alive_f) / self.gamma
        if self.tau > 0.0 or self.break_strain > 0.0:
            m = self.alive_bond
            i, j = self.i[m], self.j[m]
            Lb = (p[j] - p[i]).norm(dim=1)
            if self.tau > 0.0:
                r = min(dt / self.tau, 1.0)
                self.l0[m] = self.l0[m] + r * (Lb - self.l0[m])
                self.u0 = self.u0 + r * ((self._umbrella(p)) - self.u0)
            if self.break_strain > 0.0:
                strain = (Lb - self.l0[m]) / self.l0[m].clamp_min(1e-12)
                gone = torch.nonzero(m).squeeze(1)[strain > self.break_strain]
                if gone.numel():
                    self.alive_bond[gone] = False
                    H.bm_broken = int(getattr(H, "bm_broken", 0)) + int(gone.numel())
        return {self.at: vel.to(pos.dtype)}


@register_operator("bm_cell_contact", family="boundary", set="particle", kind="exchange", title="a basement membrane and the cells inside it",
                   equation=r"""$$\mathbf F_{ij}=\Big[k_{\rm rep}\,(r_0-d)^+ -k_{\rm adh}A\,(d-r_0)\,(1-s)^2\,[r_0<d<adhesion_reach]\Big]\hat{\mathbf d}_{ij},\quad s=\tfrac{d-r_0}{adhesion_reach-r_0},\ \ \mathbf F_{ji}=-\mathbf F_{ij}$$""")
class BMCellContact(Exchange):
    r"""contact and adhesion between the membrane nodes and the cells' material points, every MPM substep: the nodes are a wall the points cannot cross and the points a load on the nodes.
    pt -> pt (reads `bm_node`): for every point-node pair closer than `adhesion_reach`, repulsion below r0 and, for the cells whose type is in `adh_types`, adhesion between r0 and the reach; the force on the point is summed on the node (read by bm_cell_apply).

        \mathbf F_{ij}=\Big[k_{\rm rep}\,(r_0-d)^+ -k_{\rm adh}A\,(d-r_0)\,(1-s)^2\,[r_0<d<adhesion_reach]\Big]\hat{\mathbf d}_{ij},\quad s=\tfrac{d-r_0}{adhesion_reach-r_0},\ \ \mathbf F_{ji}=-\mathbf F_{ij}
        F_{ij}         the force on point i from node j, per unit mass
        k_{rep}        `k_rep`, the repulsion stiffness, acceleration per length
        r_0            `r0`, the contact distance
        d              the point-node distance
        k_{adh}        `k_adh`, the adhesion stiffness, acceleration per length
        A              the cell's adhesion level (block `adh`)
        s              the adhesion's fade, (d - r0)/(adhesion_reach - r0)
        adhesion\_reach `adhesion_reach`, how far a node reaches to adhere
        \hat d_{ij}    the unit vector from node j to point i
        F_{ji}         the equal and opposite on the node

    Reference: Plexus (this work), after the module docstring's goal paper (Tauber, Wang & Mahadevan); the catch-bond-free adhesion band is the form of exp27."""
    PARAM_UNITS = {'adhesion_reach': 'length', 'k_adh': 'acceleration', 'k_rep': 'acceleration', 'r0': 'length'}
    SYMBOLS = {'F_{ij}': 'the force on point i from node j, per unit mass', 'k_{rep}': '`k_rep`, the repulsion stiffness, acceleration per length', 'r_0': '`r0`, the contact distance', 'd': 'the point-node distance', 'k_{adh}': '`k_adh`, the adhesion stiffness, acceleration per length', 'A': "the cell's adhesion level (block `adh`)", 's': "the adhesion's fade, (d - r0)/(adhesion_reach - r0)", 'adhesion\\_reach': '`adhesion_reach`, how far a node reaches to adhere', '\\hat d_{ij}': 'the unit vector from node j to point i', 'F_{ji}': 'the equal and opposite on the node'}
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["nodes", "cells"]
    PARAM_ROLES = {"nodes": "the_membrane_node_set", "cells": "the_cell_set", "r0": "contact_distance_world", "k_rep": "repulsion_stiffness",
                   "adh": "adhesion_strength", "k_adh": "adhesion_stiffness", "adhesion_reach": "adhesion_reach_world",
                   "adh_types": "cell_type_indices_that_adhere"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.nodes = str(params["nodes"])
        self.cells = str(params["cells"])
        self.r0 = float(params.get("r0", 0.3))
        if "r_c" in params:
            raise ValueError("bm_cell_contact: the parameter `r_c` was renamed `adhesion_reach` (exp33, 2026-10-26; no alias) -- rewrite the spec")
        self.adhesion_reach = float(params.get("adhesion_reach", 0.5))
        self.k_rep = float(params.get("k_rep", 40.0))
        self.adh = float(params.get("adh", 0.0))
        self.k_adh = float(params.get("k_adh", 20.0))
        self.adh_types = [int(v) for v in params.get("adh_types", [1])]
        self.last = {}

    def pair_mag(self, dist, adherent):
        """The signed magnitude of the force on a point from a node at distance `dist` (positive = away from the node): the repulsion
        k_rep (r0 - d)^+ for every pair, less the tether k_adh A (d - r0)^+ (1 - s)^2 for the pairs flagged `adherent`, s = (d - r0) / (adhesion_reach - r0)."""
        mag = self.k_rep * (self.r0 - dist).clamp_min(0.0)
        if self.adh > 0.0:
            s = ((dist - self.r0) / max(self.adhesion_reach - self.r0, 1e-9)).clamp(0.0, 1.0)
            att = self.k_adh * self.adh * (dist - self.r0).clamp_min(0.0) * (1.0 - s) ** 2
            mag = mag - torch.where(adherent, att, torch.zeros_like(att))
        return mag

    def _forces(self, H):
        P, Nd, C = H.level(self.at), H.level(self.nodes), H.level(self.cells)
        X = P.get("pos").detach()
        live = (P.occ > 0.5)
        pi = live.nonzero().squeeze(1)
        nodes_alive = Nd.alive.nonzero().squeeze(1)
        Xn = Nd.get("pos").detach()[nodes_alive]
        I, J, d = radius_pairs(X[pi], Xn, self.adhesion_reach)
        I, J = pi[I], nodes_alive[J]
        dist = d.norm(dim=1).clamp_min(1e-9)
        u = d / dist[:, None]
        adherent = torch.zeros_like(dist, dtype=torch.bool)
        if self.adh > 0.0:
            cid = H.lift_index(self.at, self.cells)
            tp = C.node_type.to(X.device)[cid[I]]
            for t in self.adh_types:
                adherent |= (tp == t)
        return I, J, self.pair_mag(dist, adherent)[:, None] * u      # the force on the point; the node feels the opposite

    def forward(self, H, mask=None):
        P, Nd = H.level(self.at), H.level(self.nodes)
        X = P.get("pos")
        I, J, f = self._forces(H)
        F = torch.zeros_like(X)
        F.index_add_(0, I, f.to(X.dtype))
        m = P.mass.to(X.dtype) if getattr(P, "mass", None) is not None else torch.ones(X.shape[0], dtype=X.dtype, device=X.device)
        acc = F / m.clamp_min(1e-30)[:, None]
        if not hasattr(Nd, "bm_f"):
            Nd.register_buffer("bm_f", torch.zeros(Nd.get("pos").shape[0], 3, dtype=X.dtype, device=X.device))
            Nd.register_buffer("bm_n", torch.zeros(1, dtype=X.dtype, device=X.device))
        Nd.bm_f.index_add_(0, J, (-f).to(X.dtype))
        Nd.bm_n += 1.0
        self.last = {"pairs": int(I.numel())}
        H.bm_contact_calls = int(getattr(H, "bm_contact_calls", 0)) + 1
        H.bm_contact_pairs = int(I.numel())
        H.bm_contact_amax = float(acc.abs().max()) if acc.numel() else 0.0
        return {self.at: acc}


@register_operator("bm_cell_apply", family="boundary", set="particle", kind="lateral", title="the cells' load on the basement membrane", equation=r"""$$\mathbf v_j=\frac{\langle \mathbf F_j\rangle_{\text{substeps}}}{\gamma}$$""")
class BMCellApply(Lateral):
    r"""once per frame, after the MPM substeps: the node velocity from the cells, the mean over the substeps of the contact force summed on the node, divided by the shell's friction.
    bm_node -> bm_node: reads the force buffer bm_cell_contact filled, emits a velocity (`emit: velocity`) and clears the buffer.

        \mathbf v_j=\frac{\langle \mathbf F_j\rangle_{\text{substeps}}}{\gamma}
        v_j            the velocity handed to node j
        \langle F_j\rangle the mean over the frame's substeps of the contact force summed on node j
        \gamma         `overdamped_gamma`, the friction bm_shell uses, mass per time

    Reference: Plexus (this work); the overdamped node update of bm_shell."""
    PARAM_UNITS = {'overdamped_gamma': 'mass_per_time'}
    SYMBOLS = {'v_j': 'the velocity handed to node j', '\\langle F_j\\rangle': "the mean over the frame's substeps of the contact force summed on node j", '\\gamma': '`overdamped_gamma`, the friction bm_shell uses, mass per time'}
    EMIT = "velocity"
    SUPPORTED_DIMS = [3]
    PARAM_ROLES = {"overdamped_gamma": "interface_friction"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "bm_node")
        self.gamma = float(params.get("overdamped_gamma", 50.0))

    def forward(self, H, mask=None):
        Nd = H.level(self.at)
        pos = Nd.get("pos")
        if not hasattr(Nd, "bm_f") or float(Nd.bm_n) <= 0.0:
            return {self.at: torch.zeros_like(pos)}
        v = (Nd.bm_f / float(Nd.bm_n)) / self.gamma * Nd.alive[:, None].to(pos.dtype)
        Nd.bm_f.zero_()
        Nd.bm_n.zero_()
        return {self.at: v.to(pos.dtype)}
