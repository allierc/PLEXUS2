"""exp09 big scenes (2026-11-17): `pair_potential[cell_contact_warp_pp]` -- the overlap penalty of pair_potential[cell_contact] as Warp kernels on each cell's
own block of lattice nodes.

pair_potential[cell_contact] (CellContactVolumeInterfaceSparse) holds, for every point, its 27 lattice nodes at once: the node indices [points, 27, 3] int64, the
flat (cell, node) keys [points x 27] int64, the weights and their gradients [points, 27, 3], and a sort of the keys -- about 85 GB at 70 million points (exp09 b10m:
10 million bodies of 7 points ran out of the H200's 140 GB at its first frame). The overlap penalty needs no more than two fields over the world's lattice,

    S1(n) = sum_A phi_A(n),   S2(n) = sum_A phi_A(n)^2,   P(n) = (S1(n)^2 - S2(n)) / 2 = sum_{A<B} phi_A(n) phi_B(n),

and, at each point, its own cell's fraction phi_A at its 27 nodes. A cell's points lie on a few nodes: each cell gets a block of `block`^3 nodes (its origin the
lowest node any of its points touches), so the cell's fraction is stored on its block, not keyed per point and node. Four steps, each one kernel or one torch
expression over the world's lattice:

    1  deposit   per point: phi_A on its cell's block and S1 on the world, quadratic B-spline weights w x V / dx^3 (atomic adds)
    2  square    per block node: S2 += phi_A^2 (atomic adds)
    3  penalty   per world node: dU = k (P - s_0)_+
    4  force     per point: F_i = -V_i sum_n grad w_in dU(n) (S1(n) - phi_A(n)),  a_i = F_i / m_i

the same arithmetic as the default (its node factor q = dU (phi - phi_A), its force F_i = -V_i sum G_in q), summed in another order: equal to float round-off, not
bit-identical (atomics). The INTERFACE TENSION of cell_contact (a gamma matrix with any non-zero entry, its per-cell blocks) is NOT done, nor a per-cell overlap
tolerance (`contact_slack_block`), a periodic lattice or float64: each is refused loudly. CUDA, 3D, float32.
"""
import math

import torch

from plexus.models.registry import register_operator
from plexus.operators.adhesion_ops import CellContactVolumeInterfaceSparse
from plexus.operators.mpm_ops import HAVE_WARP, _wp_launch

if HAVE_WARP:
    import warp as wp

    @wp.func
    def _w(f: float, k: int) -> float:
        """The quadratic B-spline weight of stencil node k (0, 1, 2) at offset f = x / dx - base, as the default writes it."""
        if k == 0:
            return 0.5 * (1.5 - f) * (1.5 - f)
        if k == 1:
            return 0.75 - (f - 1.0) * (f - 1.0)
        return 0.5 * (f - 0.5) * (f - 0.5)

    @wp.func
    def _dw(f: float, k: int, dx: float) -> float:
        """Its derivative along the axis, per length."""
        if k == 0:
            return (f - 1.5) / dx
        if k == 1:
            return (-2.0 * (f - 1.0)) / dx
        return (f - 0.5) / dx

    @wp.func
    def _node(u: int, n: int, per: int) -> int:
        """A node index on an axis of the world lattice: wrapped onto [0, n) on a periodic axis, else clamped to it."""
        if per != 0:
            return ((u % n) + n) % n
        return wp.clamp(u, 0, n - 1)

    @wp.kernel
    def _cc_deposit(X: wp.array(dtype=wp.vec3), OCC: wp.array(dtype=float), PV: wp.array(dtype=float), CID: wp.array(dtype=wp.int32),
                    ORG: wp.array(dtype=wp.vec3i), nx: int, ny: int, nz: int, B: int, dx: float, dx3: float, per: int,
                    PHIB: wp.array2d(dtype=float), S1: wp.array(dtype=float)):
        """One thread per point: its 27 weights times V / dx^3 onto its cell's block (PHIB [cells, block^3]: Warp holds at most 2^31 entries along one axis) and onto the world's S1."""
        i = wp.tid()
        if OCC[i] <= 0.0:
            return
        x = X[i]
        gx = wp.vec3(x[0] / dx, x[1] / dx, x[2] / dx)
        bx = int(wp.floor(gx[0] - 0.5))
        by = int(wp.floor(gx[1] - 0.5))
        bz = int(wp.floor(gx[2] - 0.5))
        fx = gx[0] - float(bx)
        fy = gx[1] - float(by)
        fz = gx[2] - float(bz)
        vd = PV[i] / dx3
        c = CID[i]
        o = ORG[c]
        for a in range(3):
            ix = _node(bx + a, nx, per)
            lx = ix - o[0]            # the block's own (unwrapped) node numbers on a periodic lattice
            if per != 0:
                lx = bx + a - o[0]
            wa = _w(fx, a)
            for b in range(3):
                iy = _node(by + b, ny, per)
                ly = iy - o[1]
                if per != 0:
                    ly = by + b - o[1]
                wb = _w(fy, b)
                for d in range(3):
                    iz = _node(bz + d, nz, per)
                    lz = iz - o[2]
                    if per != 0:
                        lz = bz + d - o[2]
                    dep = wa * wb * _w(fz, d) * vd
                    wp.atomic_add(PHIB, c, (lx * B + ly) * B + lz, dep)
                    wp.atomic_add(S1, (ix * ny + iy) * nz + iz, dep)

    @wp.kernel
    def _cc_square(PHIB: wp.array2d(dtype=float), ORG: wp.array(dtype=wp.vec3i), nx: int, ny: int, nz: int, B: int, per: int, S2: wp.array(dtype=float)):
        """One thread per (cell, block node): its cell's fraction squared onto the world's S2."""
        c, r = wp.tid()
        v = PHIB[c, r]
        if v == 0.0:
            return
        lx = r // (B * B)
        ly = (r // B) - lx * B
        lz = r - (r // B) * B
        o = ORG[c]
        wp.atomic_add(S2, (_node(o[0] + lx, nx, per) * ny + _node(o[1] + ly, ny, per)) * nz + _node(o[2] + lz, nz, per), v * v)

    @wp.kernel
    def _cc_force(X: wp.array(dtype=wp.vec3), OCC: wp.array(dtype=float), PV: wp.array(dtype=float), M: wp.array(dtype=float), CID: wp.array(dtype=wp.int32),
                  ORG: wp.array(dtype=wp.vec3i), nx: int, ny: int, nz: int, B: int, dx: float, per: int,
                  PHIB: wp.array2d(dtype=float), S1: wp.array(dtype=float), DU: wp.array(dtype=float), ACC: wp.array(dtype=wp.vec3)):
        """One thread per point: a_i = -V_i sum_n grad w_in dU(n) (S1(n) - phi_A(n)) / m_i (0 for a dormant point)."""
        i = wp.tid()
        if OCC[i] <= 0.0:
            ACC[i] = wp.vec3(0.0, 0.0, 0.0)
            return
        x = X[i]
        gx = wp.vec3(x[0] / dx, x[1] / dx, x[2] / dx)
        bx = int(wp.floor(gx[0] - 0.5))
        by = int(wp.floor(gx[1] - 0.5))
        bz = int(wp.floor(gx[2] - 0.5))
        fx = gx[0] - float(bx)
        fy = gx[1] - float(by)
        fz = gx[2] - float(bz)
        c = CID[i]
        o = ORG[c]
        F = wp.vec3(0.0, 0.0, 0.0)
        for a in range(3):
            ix = _node(bx + a, nx, per)
            lx = ix - o[0]
            if per != 0:
                lx = bx + a - o[0]
            wa = _w(fx, a)
            da = _dw(fx, a, dx)
            for b in range(3):
                iy = _node(by + b, ny, per)
                ly = iy - o[1]
                if per != 0:
                    ly = by + b - o[1]
                wb = _w(fy, b)
                db = _dw(fy, b, dx)
                for d in range(3):
                    iz = _node(bz + d, nz, per)
                    lz = iz - o[2]
                    if per != 0:
                        lz = bz + d - o[2]
                    wd = _w(fz, d)
                    dd = _dw(fz, d, dx)
                    n = (ix * ny + iy) * nz + iz
                    e = DU[n] * (S1[n] - PHIB[c, (lx * B + ly) * B + lz])
                    F = F + wp.vec3(da * wb * wd, wa * db * wd, wa * wb * dd) * e
        ACC[i] = F * (-PV[i] / wp.max(M[i], 1.0e-30))


@register_operator("pair_potential", implementation="cell_contact_warp_pp", family="interaction", set="particle", kind="exchange",
                   title="keep cells apart, the overlap penalty on each cell's own block of lattice nodes, Warp kernels",
                   equation=r"""$$E_{\rm overlap}=\Delta x^3\sum_n\tfrac{k}{2}\,(P_n-s_0)_+^2,\quad P_n=\tfrac12\Big[\big(\textstyle\sum_A\phi_A(n)\big)^2-\sum_A\phi_A(n)^2\Big],\qquad \mathbf F_i=-\frac{\partial E}{\partial\mathbf x_i}$$""")
class CellContactWarpPP(CellContactVolumeInterfaceSparse):
    r"""pair_potential[cell_contact]'s overlap penalty with each cell's volume fraction kept on its own block of lattice nodes and the world holding two sums of them,
    as Warp kernels: memory in cells x block^3 + 3 x the lattice's nodes, not points x 27 x (keys, weights, gradients).
    particle -> particle: as cell_contact with no interface tension (every entry of `contact_surface_tension` 0); emits an acceleration. Equal to the default to float
    round-off (atomics). Refused: a non-zero tension, `contact_surface_tension_block`, `contact_slack_block`, `contact_edge_width_block`, `period` (x alone: `periodic` makes every axis periodic), float64, the CPU.

        E_{\rm overlap}=\Delta x^3\sum_n\tfrac{k}{2}\,(P_n-s_0)_+^2,\quad P_n=\tfrac12\Big[\big(\textstyle\sum_A\phi_A(n)\big)^2-\sum_A\phi_A(n)^2\Big],\qquad \mathbf F_i=-\frac{\partial E}{\partial\mathbf x_i}
        E_{overlap}    the overlap penalty: zero until two cells overlap by more than s_0 at a lattice node, then quadratic in the excess
        P_n            the overlap at lattice node n, the sum of the pairwise products of the cells' volume fractions there
        \phi_A(n)      cell A's volume fraction at node n: its points' quadratic B-spline weights times their volumes over dx^3
        k              `repulsion_strength`, the overlap penalty's stiffness
        s_0            `contact_slack`, the overlap tolerated at a node
        \Delta x       the lattice spacing (`dx`, else the grid's)
        F_i, x_i       the force on material point i and its position; F_i / m_i is the acceleration handed to the MPM substep

    `block`: the nodes along each axis of a cell's block (default 8), grown to the widest cell's span when a cell's points touch more nodes.

    Reference: as cell_contact; Plexus exp09 (this work, 2026-11-17: 10 million bodies of 7 points)."""
    EXPLAIN = (
        r"Keeps cells from overlapping, as `cell_contact` does when no contact has a tension. Each cell's points are smeared onto a lattice as its volume fraction "
        r"$\phi_A$; where two cells' fractions meet, their product $P_n$ is the overlap, and past the tolerated $s_0$ it costs $\tfrac{k}{2}(P_n-s_0)^2$ per node. "
        r"Here each cell keeps its fraction on its own small block of nodes and the world holds only the sum of the fractions and the sum of their squares, from which "
        r"$P_n=\tfrac12[(\sum_A\phi_A)^2-\sum_A\phi_A^2]$; four Warp kernels, so the memory follows the number of cells and not the points times their 27 nodes."
    )
    PARAM_UNITS = {"block": "count"}
    SYMBOLS = {'E_{overlap}': 'the overlap penalty', 'P_n': 'the overlap at lattice node n', '\\phi_A(n)': "cell A's volume fraction at node n",
               'k': '`repulsion_strength`, the overlap stiffness', 's_0': '`contact_slack`, the tolerated overlap', '\\Delta x': 'the lattice spacing',
               'F_i, x_i': 'the force on material point i and its position'}
    REFERENCE = "Plexus exp09 (this work, 2026-11-17)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.block = int(params.get("block", 8))
        self.periodic = bool(params.get("periodic", False))      # every axis periodic, its period the world box: the lattice wraps, positions stay unwrapped
        self._phib = None

    def _refuse(self, dt_, dev):
        why = []
        if any(float(v) != 0.0 for row in self.gamma for v in row):
            why.append("a non-zero contact_surface_tension (the interface tension is not done)")
        for name, blk in (("contact_surface_tension_block", self.ct_block), ("contact_slack_block", self.overlap_block), ("contact_edge_width_block", self.width_block)):
            if blk:
                why.append(f"`{name}`")
        if self.period > 0.0:
            why.append("a periodic lattice (`period`)")
        if dt_ != torch.float32:
            why.append(f"state of {dt_} (float32 only)")
        if not HAVE_WARP or dev.type != "cuda":
            why.append("no Warp on CUDA")
        if why:
            raise ValueError("pair_potential[cell_contact_warp_pp] does not do: " + "; ".join(why) + " -- use pair_potential[cell_contact]")

    def forward(self, H, mask=None):
        P = H.level(self.at)
        X = P.get("pos")
        dev, dt_ = X.device, X.dtype
        self._refuse(dt_, dev)
        C = H.level(self.cells)
        nC = int(C.state.shape[0])
        cid = H.lift_index(self.at, self.cells).long()
        dx = float(self.dx) if self.dx is not None else float(H.field(self.grid).dx)
        if self.periodic:                                           # the lattice holds a whole number of nodes along each period
            shape = [int(round(float(w) / dx)) for w in H.world_size]
            if any(abs(float(w) / dx - n_) > 1e-4 for w, n_ in zip(H.world_size, shape)):
                raise ValueError(f"pair_potential[cell_contact_warp_pp]: periodic, but the world {[float(w) for w in H.world_size]} is not a whole number of lattice spacings {dx}")
        else:
            shape = [int(math.ceil(float(w) / dx)) + 3 for w in H.world_size]
        nx, ny, nz = shape
        Nw = nx * ny * nz
        B = self.block
        live = P.occ > 0
        # each cell's block origin: the lowest (clamped) node any of its live points touches; its extent checked against `block`
        base = torch.floor(X.detach() / dx - 0.5).long()
        if self.periodic:                                           # a cell's block in its own unwrapped node numbers
            lo, hi = base, base + 2
        else:
            lo = torch.stack([base[:, k].clamp(0, shape[k] - 1) for k in range(3)], 1)
            hi = torch.stack([(base[:, k] + 2).clamp(0, shape[k] - 1) for k in range(3)], 1)
        big = torch.iinfo(torch.int64).max
        org = torch.full((nC, 3), big, dtype=torch.long, device=dev)                # (an unwrapped origin may be negative: `big` marks a cell with no live point)
        lc, ll, lh = cid[live], lo[live], hi[live]
        org.scatter_reduce_(0, lc[:, None].expand(-1, 3), ll, reduce="amin", include_self=True)
        org = torch.where(org == big, torch.zeros_like(org), org)
        span = int((lh - org[lc]).max()) + 1 if lc.numel() else 0
        if span > B:                                                   # the block grows to the widest cell (step 21's 23 nm bodies on a 2 nm lattice: 12 nodes)
            print(f"[cell_contact_warp_pp] a cell's points touch {span} lattice nodes along an axis: blocks of {span}^3 (`block` was {B})", flush=True)
            self.block = B = span
        nb = nC * B ** 3
        if self._phib is None or self._phib.numel() != nb or self._phib.device != dev:
            self._phib = torch.zeros((nC, B ** 3), dtype=torch.float32, device=dev)
        else:
            self._phib.zero_()
        S1 = torch.zeros(Nw, dtype=torch.float32, device=dev)
        S2 = torch.zeros(Nw, dtype=torch.float32, device=dev)
        Xc = X.detach().contiguous()
        occ = P.occ.to(torch.float32).contiguous()
        pv = (P.p_vol.to(torch.float32) if getattr(P, "p_vol", None) is not None else torch.ones_like(occ)).contiguous()
        m = (P.mass.to(torch.float32) if getattr(P, "mass", None) is not None else torch.ones_like(occ)).contiguous()
        cid32 = cid.to(torch.int32).contiguous()
        org32 = org.to(torch.int32).contiguous()
        n = int(Xc.shape[0])
        dx3 = float(torch.tensor(dx, dtype=torch.float32) ** 3)
        _wp_launch(_cc_deposit, n, dev, [wp.from_torch(Xc, dtype=wp.vec3), wp.from_torch(occ), wp.from_torch(pv), wp.from_torch(cid32), wp.from_torch(org32, dtype=wp.vec3i),
                                         nx, ny, nz, B, dx, dx3, int(self.periodic), wp.from_torch(self._phib), wp.from_torch(S1)])
        _wp_launch(_cc_square, (nC, B ** 3), dev, [wp.from_torch(self._phib), wp.from_torch(org32, dtype=wp.vec3i), nx, ny, nz, B, int(self.periodic), wp.from_torch(S2)])
        Pn = 0.5 * (S1 * S1 - S2)
        kk = self.k
        if self.stiffness_block:
            kk = C.state[:, C.state_schema[self.stiffness_block][0]][C.occ > 0.5].mean().to(torch.float32)
        dU = (kk * (Pn - float(self.c0)).clamp(min=0.0)).to(torch.float32).contiguous()
        acc = torch.zeros((n, 3), dtype=torch.float32, device=dev)
        _wp_launch(_cc_force, n, dev, [wp.from_torch(Xc, dtype=wp.vec3), wp.from_torch(occ), wp.from_torch(pv), wp.from_torch(m), wp.from_torch(cid32),
                                       wp.from_torch(org32, dtype=wp.vec3i), nx, ny, nz, B, dx, int(self.periodic), wp.from_torch(self._phib), wp.from_torch(S1), wp.from_torch(dU),
                                       wp.from_torch(acc, dtype=wp.vec3)])
        if mask is not None:
            acc = acc * mask[:, None].to(dt_)
        H.cell_contact_max = float(Pn.max()) if Pn.numel() else 0.0
        return {self.at: acc}
