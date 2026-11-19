"""exp27 speed audit (2026-10-21), test 5e: the neighbour searches of `contact_finder` and `active_force[sphere_crawl]` on the GPU.

WHY. At 320k live points the two searches were the largest CPU steps of an exp27 frame (RTX A6000 host, one thread): `contact_finder`
1.33 s per frame, of which the scipy `cKDTree` build was 0.10 s and its k = 16 query 1.08 s, and `active_force[sphere_crawl]` 0.89 s, of
which the tree build 0.10 s and `query_ball_point(return_length=True)` 0.70 s. Both are fixed-radius searches over points that the
GPU already holds, so here they run as Warp hash-grid queries (`wp.HashGrid`, as `radius_graph[warp]`) and the rest of
`contact_finder` -- which the default does in numpy with a Python dict over every pair -- runs in torch on the device.

WHAT IS KEPT, exactly (the answer of the default, not an approximation of it):

  contact_finder[hashgrid]   a live point's partner is its NEAREST point in ANOTHER cell closer than `bond_range + skin`, provided that point is
        among the `k_nn` (default 16) nearest points of all, itself included -- the rule the default gets from `tree.query(k=k_nn,
        distance_upper_bound=bond_range + skin)` and "first neighbour of another cell". Computed here in two passes over the hash-grid
        candidates: the nearest other-cell point (ties go to the lower index), then its rank, the number of candidates closer than it
        in the same (distance, index) order; the partner stands when the rank is below `k_nn`. Distances are float64 as in the default
        (the hash grid only chooses the candidates, in float32 with a margin). Everything after the partner search is the default's
        arithmetic in torch float64 (face class of each cell pair, contact area per (cell pair, face class), junction rows created in
        ascending key order into the lowest free rows, area, load, idle counter, retirement, the pair list with the row of each pair).
        The one thing that is NOT reproduced is the kd-tree's own order among points at EXACTLY the same distance, which scipy does not
        specify; here the lower point index wins. `period > 0` (a ring) is not implemented and runs the default. The sums follow numpy's order
        of addition (the direction sum per cell pair row by row, three-component sums as ((a0 + a1) + a2)) so that the face classes, the
        rows and the pair list are the default's to the last bit.
  active_force[sphere_crawl_hashgrid]   `sphere_crawl` with one method replaced, `_ball_counts`: the number of live points within `probe`
        of each probe point (`d^2 <= probe^2` in float64, as `query_ball_point`). Everything else, including the rotational-diffusion
        draws of the CPU torch generator, is the default's code.

`active_force` is selected as a MODEL here and not as `implementation: hashgrid` next to `model: sphere_crawl`: a spec line may name a model
or an implementation, not both (plexus/schema.py `resolve_op_line`), and the contract keeps one variant namespace per operator.
"""
import contextlib
import math

import numpy as np
import torch

from plexus.models.registry import register_operator
from plexus.operators.adhesion_ops import ActiveForceSphereCrawl, FACE_A, FACE_B, FACE_L, JunctionContact, _FG
from plexus.operators.mpm_ops import ActiveForcePersistentWalk

try:
    import warp as wp
    wp.init()
    HAVE_WARP = True
except Exception:
    HAVE_WARP = False

_GRID_DIM = 64                    # cells per axis of the hash grid; coordinates beyond wrap, which only adds candidates the exact test rejects


if HAVE_WARP:

    @wp.kernel
    def _jc_partner(P32: wp.array(dtype=wp.vec3), PX: wp.array(dtype=wp.vec3d), CID: wp.array(dtype=wp.int32),
                    grid: wp.uint64, rq: float, ub2: wp.float64, ub2f: float, k_nn: int,
                    PART: wp.array(dtype=wp.int32)):
        """One thread per live point: the local index of its partner, or -1."""
        i = wp.tid()
        pf = P32[i]
        px = PX[i]
        ci = CID[i]
        best_d = wp.float64(1.0e300)
        best_j = int(-1)
        j = int(0)
        q = wp.hash_grid_query(grid, pf, rq)
        while wp.hash_grid_query_next(q, j):
            if CID[j] != ci:
                df = P32[j] - pf
                if wp.dot(df, df) <= ub2f:                     # float32 prefilter, margin included in ub2f
                    pj = PX[j]
                    dx = pj[0] - px[0]
                    dy = pj[1] - px[1]
                    dz = pj[2] - px[2]
                    d2 = dx * dx + dy * dy + dz * dz
                    if d2 < ub2:
                        if (d2 < best_d) or (d2 == best_d and j < best_j):
                            best_d = d2
                            best_j = j
        if best_j < 0:
            PART[i] = -1
            return
        rank = int(0)
        q2 = wp.hash_grid_query(grid, pf, rq)
        while wp.hash_grid_query_next(q2, j):
            df = P32[j] - pf
            if wp.dot(df, df) <= ub2f:
                pj = PX[j]
                dx = pj[0] - px[0]
                dy = pj[1] - px[1]
                dz = pj[2] - px[2]
                d2 = dx * dx + dy * dy + dz * dz
                if d2 < ub2:
                    if (d2 < best_d) or (d2 == best_d and j < best_j):
                        rank = rank + 1
        if rank < k_nn:
            PART[i] = best_j
        else:
            PART[i] = -1

    @wp.kernel
    def _ball_count(P32: wp.array(dtype=wp.vec3), PX: wp.array(dtype=wp.vec3d),
                    Q32: wp.array(dtype=wp.vec3), QX: wp.array(dtype=wp.vec3d),
                    grid: wp.uint64, rq: float, r2: wp.float64, r2f: float, N: wp.array(dtype=wp.int32)):
        """One thread per probe point: the number of points within the radius, itself included if it is one of them."""
        i = wp.tid()
        qf = Q32[i]
        qx = QX[i]
        c = int(0)
        j = int(0)
        q = wp.hash_grid_query(grid, qf, rq)
        while wp.hash_grid_query_next(q, j):
            df = P32[j] - qf
            if wp.dot(df, df) <= r2f:
                pj = PX[j]
                dx = pj[0] - qx[0]
                dy = pj[1] - qx[1]
                dz = pj[2] - qx[2]
                if dx * dx + dy * dy + dz * dz <= r2:
                    c = c + 1
        N[i] = c


    @wp.kernel
    def _seg_sum3_k(V: wp.array(dtype=wp.vec3d), START: wp.array(dtype=wp.int32), CNT: wp.array(dtype=wp.int32),
                    OUT: wp.array(dtype=wp.vec3d)):
        """One thread per segment: the rows of the segment added one after the other, from 0.0, in the order they are stored."""
        c = wp.tid()
        s0 = wp.float64(0.0)
        s1 = wp.float64(0.0)
        s2 = wp.float64(0.0)
        for t in range(CNT[c]):
            v = V[START[c] + t]
            s0 = s0 + v[0]
            s1 = s1 + v[1]
            s2 = s2 + v[2]
        OUT[c] = wp.vec3d(s0, s1, s2)


@contextlib.contextmanager
def _on_torch_stream(dev):
    """Warp launches, and the hash-grid build, on PyTorch's current stream (see `mpm_ops._wp_launch`)."""
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(dev)), sync_enter=False, sync_exit=False):
        yield


def _wdev(dev):
    return f"cuda:{dev.index or 0}"


def _dot3(a, b):
    """`(a * b).sum(1)` as numpy adds the three products of a row: ((p0 + p1) + p2). Torch's own reduction order is not specified."""
    p = a * b
    return (p[:, 0] + p[:, 1]) + p[:, 2]


def _norm3(v):
    """`np.linalg.norm(v, axis=1)` for [n, 3]: the square root of ((v0^2 + v1^2) + v2^2), the order numpy sums in."""
    q = v * v
    return torch.sqrt((q[:, 0] + q[:, 1]) + q[:, 2])


def _seg_sum3(vals, seg, n_seg):
    """[n_seg, 3] float64: out[g] is the sum of the rows `vals[k]` with `seg[k] == g`, added in ascending k from 0.0 -- what
    `np.add.at(zeros, seg, vals)` does, to the last bit (a torch `index_add_` adds in an order of its own)."""
    dev = vals.device
    out = torch.zeros(n_seg, 3, dtype=torch.float64, device=dev)
    if int(vals.shape[0]) == 0:
        return out
    order = torch.sort(seg, stable=True).indices
    sv = vals[order].contiguous()
    cnt = torch.bincount(seg, minlength=n_seg)
    start = (torch.cumsum(cnt, 0) - cnt).to(torch.int32).contiguous()
    with _on_torch_stream(dev):
        wp.launch(_seg_sum3_k, dim=n_seg, device=_wdev(dev),
                  inputs=[wp.from_torch(sv, dtype=wp.vec3d), wp.from_torch(start), wp.from_torch(cnt.to(torch.int32).contiguous()),
                          wp.from_torch(out, dtype=wp.vec3d)])
    return out


def _tdiv(x, c):
    """x / c for a Python scalar c as a division of two tensors. Torch divides a tensor by a host scalar as a multiplication by the
    scalar's reciprocal, which can differ from numpy's division in the last bit; a 0-d tensor on the device is divided exactly."""
    return x / torch.as_tensor(c, dtype=x.dtype, device=x.device)


def _grid(owner, dev):
    gr = getattr(owner, "_grid", None)
    if gr is None:
        gr = owner._grid = wp.HashGrid(_GRID_DIM, _GRID_DIM, _GRID_DIM, device=_wdev(dev))
    return gr


def _partners(owner, Xs, X, cid, ub, k_nn):
    """[n] int64: for each live point the local index of its partner in another cell, -1 for none. `Xs` the positions as stored,
    `X` the same in float64, `cid` the cell of each point."""
    dev = X.device
    n = int(X.shape[0])
    p32 = Xs.float().contiguous()
    X = X.contiguous()
    cid32 = cid.to(torch.int32).contiguous()
    rq = float(ub) * (1.0 + 1.0e-4) + 1.0e-5                 # a superset of the true neighbours: the exact test is the float64 one
    part = torch.empty(n, dtype=torch.int32, device=dev)
    with _on_torch_stream(dev):
        gr = _grid(owner, dev)
        wp32 = wp.from_torch(p32, dtype=wp.vec3)
        gr.build(points=wp32, radius=rq)
        wp.launch(_jc_partner, dim=n, device=_wdev(dev),
                  inputs=[wp32, wp.from_torch(X, dtype=wp.vec3d), wp.from_torch(cid32), gr.id, rq, float(ub) * float(ub),
                          rq * rq, int(k_nn), wp.from_torch(part)])
    return part.long()


def _known(occ, rowkey):
    """The occupied junction rows sorted by key, ties in row order: (sorted keys, their rows)."""
    rows = torch.nonzero(occ).reshape(-1)
    keys = rowkey[rows]
    order = torch.argsort(keys, stable=True)
    return keys[order], rows[order]


def _lookup(sk, srow, q):
    """For each key in q the LAST occupied row carrying it (the default's `dict(zip(keys, rows))` keeps the last), -1 for none."""
    out = torch.full_like(q, -1)
    if sk.numel() == 0 or q.numel() == 0:
        return out
    pos = torch.searchsorted(sk, q, right=True) - 1
    ok = pos >= 0
    pos = pos.clamp(min=0)
    ok = ok & (sk[pos] == q)
    return torch.where(ok, srow[pos], out)


@register_operator("contact_finder", implementation="hashgrid", family="topology", set="junction", kind="rewire",
                   title="detect cell contacts, neighbour search and junction table on the GPU", equation=r"""$$A_{ab,fg}=s^{2}\,\big\lvert\{\,i\in a:\ \exists j\in b,\ \lvert\mathbf x_i-\mathbf x_j\rvert<r_b,\ (f_i,f_j)=fg\}\big\rvert$$""")
class JunctionContactHashgrid(JunctionContact):
    r"""`hashgrid` implementation of contact_finder: the partner search as a Warp hash-grid query and the bookkeeping in torch on the device.
    particle <-> particle: as contact_finder (same table, same parameters, same outputs); the neighbour search runs on the device. Not bit-reproducible (float atomics).

        A_{ab,fg}=s^{2}\,\big\lvert\{\,i\in a:\ \exists j\in b,\ \lvert\mathbf x_i-\mathbf x_j\rvert<r_b,\ (f_i,f_j)=fg\}\big\rvert
        A_{ab,fg}      the contact area of cells a and b through face-pair class fg, world length squared
        s              `spacing`, the point spacing
        i, j           live points of the two cells
        x_i, x_j       their positions
        r_b            `bond_range`, the contact distance (named `r_c`, then `bond_reach` until 2026-10-31)
        (f_i, f_j)     the faces of the two points (lateral, apical, basal), their class fg

    Reference: as `contact_finder`; NVIDIA Warp hash grid."""
    PARAM_UNITS = {'bond_range': 'length', 'k_nn': 'count', 'period': 'length', 'retire_after': 'frames', 'skin': 'length', 'spacing': 'length', 'workers': 'count'}
    SYMBOLS = {'A_{ab,fg}': 'the contact area of cells a and b through face-pair class fg, world length squared', 's': '`spacing`, the point spacing', 'i, j': 'live points of the two cells', 'x_i, x_j': 'their positions', 'r_b': '`bond_range`, the contact distance (named `r_c`, then `bond_reach` until 2026-10-31)', '(f_i, f_j)': 'the faces of the two points (lateral, apical, basal), their class fg'}

    def forward(self, H, mask=None):
        P, C, J = H.level(self.points), H.level(self.cells), H.level(self.at)
        dev, dt_ = P.state.device, P.state.dtype
        if (self._periods(H) > 0.0).any() or not HAVE_WARP or dev.type != "cuda":     # a periodic axis: the kd-tree of the CPU implementation
            return super().forward(H, mask)
        nC = int(C.state.shape[0])
        live = torch.nonzero(P.occ > 0.5).reshape(-1)
        n = int(live.numel())
        if n == 0:
            return super().forward(H, mask)
        Xs = P.get("pos").detach()[live]
        X = Xs.double()
        cid = self._cell_index(H)[live]
        H._adh_period = 0.0
        H._adh_box = np.zeros(3)
        part = _partners(self, Xs, X, cid, self.bond_reach + self.skin, self.k_nn)
        i = torch.nonzero(part >= 0).reshape(-1)
        j = part[i]
        ci, cj = cid[i], cid[j]
        a, b = torch.minimum(ci, cj), torch.maximum(ci, cj)
        # the face of a contact, decided once per cell pair (see JunctionContact.forward)
        if "pol" in C.state_schema:
            pol = C.get("pol").detach().double()
        else:
            pol = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=dev).repeat(nC, 1)
        pol = pol / _norm3(pol).clamp(min=1e-12)[:, None]
        dvec = X[j] - X[i]
        dvec = dvec / _norm3(dvec).clamp(min=1e-12)[:, None]
        dvec = torch.where((ci == a)[:, None], dvec, -dvec)                          # from cell a toward cell b
        k0 = a * nC + b
        u0, inv0 = torch.unique(k0, return_inverse=True)
        nsum = _seg_sum3(dvec, inv0, int(u0.numel()))
        nmean = nsum / _norm3(nsum).clamp(min=1e-12)[:, None]
        ua, ub = u0 // nC, u0 % nC
        cut = 1.0 / math.sqrt(2.0)
        ca_ = _dot3(nmean, pol[ua])
        cb_ = -_dot3(nmean, pol[ub])
        fa = torch.full((u0.numel(),), FACE_L, dtype=torch.long, device=dev)
        fa[ca_ > cut] = FACE_A
        fa[ca_ < -cut] = FACE_B
        fb = torch.full((u0.numel(),), FACE_L, dtype=torch.long, device=dev)
        fb[cb_ > cut] = FACE_A
        fb[cb_ < -cut] = FACE_B
        fg = torch.as_tensor(_FG, device=dev)[fa, fb][inv0]
        key = k0 * 6 + fg
        touch = _norm3(X[i] - X[j]) < self.bond_reach
        ia = torch.where(ci == a, i, j)
        tk, tp = key[touch], ia[touch]
        # contact per key: the distinct points of the LOWER cell within bond_range, x spacing^2
        if tk.numel():
            codes = torch.unique(tk * n + tp)
            uk, cnt = torch.unique(codes // n, return_counts=True)
            area_v = cnt.double() * (self.h ** 2)
        else:
            uk = torch.empty(0, dtype=torch.long, device=dev)
            area_v = torch.empty(0, dtype=torch.float64, device=dev)
        # the junction table as it is
        occJ = J.occ > 0.5
        sch = J.state_schema
        cA, cB, cF, cR, cAr, cL = (sch[k][0] for k in ("ca", "cb", "fg", "rho", "area", "load"))
        st = J.state.detach().clone()
        # the key in int64: in the state's float32 a * nC + b is exact only below 2^24, so above about 1,700 cells no row matched its own contact again and every
        # contact took a new row each frame (exp09 a100k, 100,000 cells, 2026-11-17: the table full, no bond kept)
        rowkey = (torch.round(st[:, cA]).long() * nC + torch.round(st[:, cB]).long()) * 6 + torch.round(st[:, cF]).long()
        rowkey = torch.where(occJ, rowkey, torch.full_like(rowkey, -1))
        if not isinstance(self._idle, torch.Tensor) or tuple(self._idle.shape) != (J.n,) or self._idle.device != st.device:
            self._idle = torch.zeros(J.n, dtype=torch.long, device=st.device)
        idle = self._idle
        sk, srow = _known(occJ, rowkey)
        new = uk[_lookup(sk, srow, uk) < 0]
        if new.numel():
            free = torch.nonzero(J.occ <= 0.5).reshape(-1)[: new.numel()]
            m = int(free.numel())
            if m < new.numel():
                print(f"[contact_finder] the junction buffer is FULL: {new.numel() - m} contacts get no row", flush=True)
            if m:
                kn = new[:m]
                cab = kn // 6
                st[free, cA] = (cab // nC).to(st.dtype)
                st[free, cB] = (cab % nC).to(st.dtype)
                st[free, cF] = (kn % 6).to(st.dtype)
                st[free, cR] = 0.0
                occJ[free] = True
                idle[free] = 0
                rowkey[free] = kn
                sk, srow = _known(occJ, rowkey)
        # area, load, idle frames, retirement
        st[:, cAr] = 0.0
        rows_k = _lookup(sk, srow, uk)
        okk = rows_k >= 0
        st[rows_k[okk], cAr] = area_v[okk].to(st.dtype)
        acc = getattr(H, "_adh_load_acc", None)
        nsub = max(int(getattr(H, "_adh_load_n", 0)), 1)
        if acc is not None and acc.numel() == J.n:
            ld = _tdiv(acc.detach(), nsub)
            q = ld / st[:, cAr].clamp(min=1e-12)
            st[:, cL] = torch.where(st[:, cAr] > 0, q, torch.zeros_like(q))
        H._adh_load_acc = torch.zeros(J.n, device=dev, dtype=dt_)
        H._adh_load_n = 0
        has = st[:, cAr] > 0
        self._idle = idle = torch.where(has, torch.zeros_like(idle), idle + 1)
        gone = occJ & (idle >= self.retire_after) & (st[:, cR] < 1e-3)
        # the Verlet list for the substeps: global point indices and the junction row of each pair (rows as before this call's retirement)
        lk = _lookup(sk, srow, key)
        H._adh_pairs = (live[i], live[j], lk)
        occJ = occJ & ~gone
        J.state = st
        J.occ = occJ.to(J.occ.dtype)
        H.junction_rows = int(occJ.sum())
        return {}



@register_operator("contact_finder", implementation="sites", family="topology", set="junction", kind="rewire",
                   title="detect contacts, each point counted in its site: a run of consecutive bodies of a chain", equation=r"""$$A_{a\,S(b),fg}=s^{2}\,\big\lvert\{\,i\in a:\ \exists j\in S(b),\ \lvert\mathbf x_i-\mathbf x_j\rvert<r_b,\ (f_i,f_j)=fg\}\big\rvert$$""")
class JunctionContactSites(JunctionContactHashgrid):
    r"""contact_finder[hashgrid] with every point counted in its body's SITE instead of its body: the cell block `site_block` holds, per cell, the first cell of its site --
    for a body of a chain (an mRNA's subunits, exp09), the first subunit of its run of k consecutive subunits; for any other body, itself. So all the point pairs between a
    protein and the k subunits of one site make ONE junction (one row, one contact area, one bound share in bond_binding, one count against a valence), named by the protein
    and the site's first subunit; the springs (bond_springs) act on every one of those point pairs, each through that row. Two subunits of one site are one cell to it.
    particle <-> particle: as contact_finder[hashgrid] (same table, same parameters), with `site_block`; `periodic: true` makes every axis periodic with the world box as
    its period, searched on the device when there is one (`search: cpu` keeps the kd-tree of the CPU implementation).

        A_{a\,S(b),fg}=s^{2}\,\big\lvert\{\,i\in a:\ \exists j\in S(b),\ \lvert\mathbf x_i-\mathbf x_j\rvert<r_b,\ (f_i,f_j)=fg\}\big\rvert
        A_{a S(b),fg}  the contact area of cell a and the site S(b) of cell b through face-pair class fg, world length squared
        S(b)           the cells of b's site: those whose `site_block` value is b's
        s              `spacing`, the point spacing
        r_b            `bond_range`, the contact distance

    Reference: as `contact_finder`; Plexus exp09 (this work, 2026-11-19: mRNA as chains of subunits, a binding site as a run of them)."""
    PARAM_UNITS = dict(JunctionContactHashgrid.PARAM_UNITS)
    SYMBOLS = dict(JunctionContactHashgrid.SYMBOLS, **{'A_{a S(b),fg}': 'the contact area of cell a and the site of cell b', 'S(b)': "the cells of b's site"})
    REFERENCE = "Plexus exp09 (this work, 2026-11-19)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.site_block = str(params["site_block"])
        self.periodic = bool(params.get("periodic", False))     # every axis periodic, its period the world box's size
        # `search` (default "auto"): where a PERIODIC box is searched. "auto": on the device when there is a GPU with Warp and a hash-grid cell divides the box's three
        # sides exactly (contact_periodic_warp_ops.periodic_forward: the grid's wrap is then the box's, every distance taken to the nearest image; the CPU search's
        # pair list, junction rows and areas bit for bit on exp09 rna1_3dp, tests/test_exp09_periodic.py), else the CPU kd-tree; "cpu": always the kd-tree, which is
        # what every periodic run did until 2026-11-19 (one search of rna1_3dp's 214,465 points on an RTX A6000 host: 0.71 s, against 0.008 s on the device)
        self.search = str(params.get("search", "auto")).lower()
        if self.search not in ("auto", "cpu"):
            raise ValueError(f"contact_finder[sites]: `search` is auto or cpu, not {self.search!r}")
        self._device_search = None                              # decided at the first periodic call: does a cell divide the box

    def forward(self, H, mask=None):
        if self.periodic and self.search == "auto" and HAVE_WARP and H.level(self.points).state.device.type == "cuda":
            from plexus.operators import contact_periodic_warp_ops as CP      # here, not at the top: that module imports this one
            if self._device_search is None:
                try:
                    w, dims = CP.grid_cells([float(v) for v in H.world_size], float(self.bond_reach + self.skin) * (1.0 + 1.0e-4) + 1.0e-5)
                    self._device_search = True
                    print(f"[contact_finder:sites] periodic box searched on the device: hash grid of {dims[0]} x {dims[1]} x {dims[2]} cells of {w:g}", flush=True)
                except ValueError as e:
                    self._device_search = False
                    print(f"[contact_finder:sites] periodic box searched on the CPU (kd-tree): {e}", flush=True)
            if self._device_search:
                return CP.periodic_forward(self, H, mask)
        return super().forward(H, mask)

    def _periods(self, H):
        if self.periodic:
            return np.asarray([float(v) for v in H.world_size], np.float64)
        return super()._periods(H)

    def _cell_index(self, H):
        idx = H.lift_index(self.points, self.cells)
        C = H.level(self.cells)
        if self.site_block not in C.state_schema:
            raise ValueError(f"contact_finder[sites]: the cell set `{self.cells}` has no block `{self.site_block}` (each cell's site: the first cell of its run)")
        rep = torch.round(C.state[:, C.state_schema[self.site_block][0]]).long().clamp(0, C.state.shape[0] - 1)
        return rep[idx.long()].to(idx.dtype)

@register_operator("contact_finder", implementation="differentiable", family="topology", set="junction", kind="rewire",
                   title="detect cell contacts and keep the junction density on the tape", equation=r"""$$A_{ab,fg}=s^{2}\,\big\lvert\{\,i\in a:\ \exists j\in b,\ \lvert\mathbf x_i-\mathbf x_j\rvert<r_b,\ (f_i,f_j)=fg\}\big\rvert$$""")
class JunctionContactDiff(JunctionContactHashgrid):
    r"""`hashgrid_diff` implementation of contact_finder: the hash-grid search that does not cut the bond density's graph.
    particle <-> particle: as contact_finder[hashgrid], with the junction table rebuilt from the live tensor so gradients reach the bond density rho that bond_binding integrates.

        A_{ab,fg}=s^{2}\,\big\lvert\{\,i\in a:\ \exists j\in b,\ \lvert\mathbf x_i-\mathbf x_j\rvert<r_b,\ (f_i,f_j)=fg\}\big\rvert
        A_{ab,fg}      the contact area of cells a and b through face-pair class fg, world length squared
        s              `spacing`, the point spacing
        i, j           live points of the two cells
        x_i, x_j       their positions
        r_b            `bond_range`, the contact distance (named `r_c`, then `bond_reach` until 2026-10-31)
        (f_i, f_j)     the faces of the two points (lateral, apical, basal), their class fg

    Reference: as `contact_finder`; Plexus exp30 (this work)."""
    PARAM_UNITS = {'bond_range': 'length', 'k_nn': 'count', 'period': 'length', 'retire_after': 'frames', 'skin': 'length', 'spacing': 'length', 'workers': 'count'}
    SYMBOLS = {'A_{ab,fg}': 'the contact area of cells a and b through face-pair class fg, world length squared', 's': '`spacing`, the point spacing', 'i, j': 'live points of the two cells', 'x_i, x_j': 'their positions', 'r_b': '`bond_range`, the contact distance (named `r_c`, then `bond_reach` until 2026-10-31)', '(f_i, f_j)': 'the faces of the two points (lateral, apical, basal), their class fg'}
    DIFFERENTIABLE = True

    def forward(self, H, mask=None):
        J = H.level(self.at)
        r = J.state_schema["rho"][0]
        prev = J.state[:, r]                                           # with its graph, from the last frame's integration
        out = super().forward(H, mask)
        now = J.state[:, r]
        if prev.shape == now.shape and prev.requires_grad:
            same = now == prev.detach()                                # unchanged rows keep the tape, the reset rows read the parent's 0
            st = J.state.clone()
            st[:, r] = torch.where(same, prev, now)
            J.state = st
        return out


@register_operator("active_force", "pulse_to_contraction", family="mechanics", set="particle", kind="exchange",
                   model="sphere_crawl_hashgrid", title="crawl cells over a sphere and reach into gaps, on the GPU", equation=r"""$$\mathbf a_i=f_c\,\chi_i\,\mathbf p_c+g\,e_i\,u_i\,\hat{\mathbf d}_i-\overline{(\cdot)}_c,\qquad \mathbf p_c\perp\hat{\mathbf n}_c=\frac{\mathbf x_c-\mathbf o}{\lvert\mathbf x_c-\mathbf o\rvert}$$""")
class ActiveForceSphereCrawlHashgrid(ActiveForceSphereCrawl):
    r"""`sphere_crawl_hashgrid` model of active_force: `active_force[sphere_crawl]` computed on the device.
    particle -> particle: as active_force[sphere_crawl]; the count of live points around each probe is a Warp hash-grid query and the per-cell sums run in torch on the device.

        \mathbf a_i=f_c\,\chi_i\,\mathbf p_c+g\,e_i\,u_i\,\hat{\mathbf d}_i-\overline{(\cdot)}_c,\qquad \mathbf p_c\perp\hat{\mathbf n}_c=\frac{\mathbf x_c-\mathbf o}{\lvert\mathbf x_c-\mathbf o\rvert}
        a_i            the acceleration of point i
        f_c            the drive
        \chi_i         1 in the innermost band
        p_c            the tangent-plane polarity
        g              `reach`
        e_i            the free-space measure
        u_i            the outwardness
        \hat d_i       the direction into free space
        \overline{(\cdot)}_c the per-cell mean, removed
        \hat n_c       the radial unit vector from the centre o to the cell's centroid x_c

    Reference: as `active_force[sphere_crawl]`; NVIDIA Warp hash grid."""
    PARAM_UNITS = {'centre': 'length'}
    SYMBOLS = {'a_i': 'the acceleration of point i', 'f_c': 'the drive', '\\chi_i': '1 in the innermost band', 'p_c': 'the tangent-plane polarity', 'g': '`reach`', 'e_i': 'the free-space measure', 'u_i': 'the outwardness', '\\hat d_i': 'the direction into free space', '\\overline{(\\cdot)}_c': 'the per-cell mean, removed', '\\hat n_c': "the radial unit vector from the centre o to the cell's centroid x_c"}

    def _count_dev(self, pos, alive_idx, Qt, rad):
        """float64 [m]: the live points (`pos[alive_idx]`) within `rad` of each probe point Qt [m, 3] float64, `d^2 <= rad^2` in float64."""
        dev = pos.device
        m = int(Qt.shape[0])
        if m == 0 or alive_idx.numel() == 0:
            return torch.zeros(m, dtype=torch.float64, device=dev)
        Xs = pos[alive_idx]
        X64 = Xs.double().contiguous()
        p32 = Xs.float().contiguous()
        Qt = Qt.contiguous()
        Q32 = Qt.float().contiguous()
        rq = float(rad) + 1.0e-4                           # Q is float64 and the grid sees it in float32; the exact test is float64
        cnt = torch.empty(m, dtype=torch.int32, device=dev)
        with _on_torch_stream(dev):
            gr = _grid(self, dev)
            wp32 = wp.from_torch(p32, dtype=wp.vec3)
            gr.build(points=wp32, radius=rq)
            wp.launch(_ball_count, dim=m, device=_wdev(dev),
                      inputs=[wp32, wp.from_torch(X64, dtype=wp.vec3d), wp.from_torch(Q32, dtype=wp.vec3),
                              wp.from_torch(Qt, dtype=wp.vec3d), gr.id, rq, float(rad) ** 2, rq * rq, wp.from_torch(cnt)])
        return cnt.double()

    def _ball_counts(self, lvl, X, alive, Q, rad):
        pos = lvl.get("pos").detach()
        dev = pos.device
        if not HAVE_WARP or dev.type != "cuda":
            return super()._ball_counts(lvl, X, alive, Q, rad)
        al = torch.as_tensor(np.flatnonzero(alive), device=dev)
        return self._count_dev(pos, al, torch.as_tensor(Q, device=dev, dtype=torch.float64), rad).cpu().numpy()

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        Xt = lvl.get("pos")
        dev, dt_ = Xt.device, Xt.dtype
        if not HAVE_WARP or dev.type != "cuda":
            return super().forward(H, mask)
        f64 = torch.float64
        X = Xt.detach().double()
        N = int(X.shape[0])
        alive = lvl.occ > 0.5
        nc = int(H.level(self.cell_set).state.shape[0])
        idx = H.lift_index(self.at, self.cell_set)
        sel = alive.clone()
        if mask is not None:
            sel &= (mask.detach() > 0)
        h = float(getattr(H, "dt", 1.0))
        ia, isel = idx[alive], idx[sel]
        cnt_t = torch.zeros(nc, dtype=f64, device=dev).index_add_(0, ia, torch.ones(ia.numel(), dtype=f64, device=dev))   # integers: exact
        xb_t = _seg_sum3(X[alive], ia, nc)
        xb_t = xb_t / cnt_t.clamp(min=1.0)[:, None]
        cnt, xb = cnt_t.cpu().numpy(), xb_t.cpu().numpy()
        # ---- per cell, on the host, the default's code -------------------------------------------------------------------------
        nrm = xb - self.c0
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        if self._p3 is None or self._p3.shape[0] != nc:
            ActiveForcePersistentWalk._init_cells(self, nc)
            self._p3 = self._p.detach().cpu().numpy().copy()
            fresh = True
        else:
            fresh = False
        p = self._p3
        p = p - (p * nrm).sum(1, keepdims=True) * nrm                                  # onto the tangent plane
        pn = np.linalg.norm(p, axis=1, keepdims=True)
        bad = pn[:, 0] < 1e-6
        if bad.any():
            alt = np.cross(nrm[bad], np.array([1.0, 0.3, 0.1])); p[bad] = alt; pn[bad] = np.linalg.norm(alt, axis=1, keepdims=True)
        p = p / pn
        if not fresh:
            Dc = self._Dc.detach().cpu().numpy()
            n_sub = max(1, int(math.ceil(float(Dc.max()) * h / 0.01)))
            hs = h / n_sub
            for _ in range(n_sub):
                z = torch.randn(nc, generator=self._gen, dtype=torch.float64).numpy()
                dth = np.sqrt(2.0 * Dc * hs) * z
                if self.steer > 0.0 and self._G is not None:
                    Gv, sg = self._G
                    Gh = Gv - (Gv * nrm).sum(1, keepdims=True) * nrm
                    Gh = Gh / np.maximum(np.linalg.norm(Gh, axis=1, keepdims=True), 1e-9)
                    cross = np.cross(p, Gh)
                    ang = np.arctan2((cross * nrm).sum(1), (p * Gh).sum(1))          # signed angle from p toward the gap
                    dth = dth + self.steer * sg * np.sin(ang) * hs
                p = p * np.cos(dth)[:, None] + np.cross(nrm, p) * np.sin(dth)[:, None]
            p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
        self._p3 = p
        self._p = torch.as_tensor(p, dtype=torch.float64)
        # ---- per point, on the device ------------------------------------------------------------------------------------------
        p_t = torch.as_tensor(p, dtype=f64, device=dev)
        nrm_t = torch.as_tensor(nrm, dtype=f64, device=dev)
        rr = _norm3(X - torch.as_tensor(self.c0, dtype=f64, device=dev))
        rmin = torch.full((nc,), float("inf"), dtype=f64, device=dev).scatter_reduce_(0, isel, rr[sel], reduce="amin", include_self=True)
        chi = (sel & (rr <= rmin[idx] + float(self.basal_band))).to(f64)
        n_all = torch.zeros(nc, dtype=f64, device=dev).index_add_(0, isel, torch.ones(isel.numel(), dtype=f64, device=dev))
        n_bas = torch.zeros(nc, dtype=f64, device=dev).index_add_(0, isel, chi[sel])
        chi = chi * (n_all / n_bas.clamp(min=1.0))[idx]
        amp = (self._fc * self._s).detach().to(device=dev, dtype=f64)
        zero = torch.zeros((), dtype=f64, device=dev)
        a = torch.where(sel[:, None], (amp[idx] * chi)[:, None] * p_t[idx], zero)
        gate = torch.ones(nc, dtype=f64, device=dev)
        if self.traction > 0.0:
            gate = torch.as_tensor(self._traction_gate(H, nc), dtype=f64, device=dev)
            a = a * gate[idx][:, None]
            cells_sel = torch.unique(isel)
            self.last["gate_mean"] = float(gate[cells_sel].mean()) if cells_sel.numel() else 0.0
        # free space to the side, in the tangent plane
        band = sel & (rr <= rmin[idx] + float(self.reach_band))
        bi = torch.nonzero(band).reshape(-1)
        e = torch.zeros(N, dtype=f64, device=dev)
        d = torch.zeros(N, 3, dtype=f64, device=dev)
        G_t = torch.zeros(nc, 3, dtype=f64, device=dev)
        if bi.numel():
            rad = self.probe
            ib = idx[bi]
            v = X[bi] - xb_t[ib]
            v = v - _dot3(v, nrm_t[ib])[:, None] * nrm_t[ib]
            dl = v / _norm3(v).clamp(min=1e-9)[:, None]
            Q = X[bi] + self.ahead * dl
            n = self._count_dev(Xt.detach(), torch.nonzero(alive).reshape(-1), Q, rad)
            filled = _tdiv(n, (4.0 / 3.0) * math.pi * rad ** 3 / self.h ** 3)
            free = (1.0 - filled).clamp(0.0, 1.0)
            e[bi] = _tdiv(free - self.free_floor, max(self.free_ref - self.free_floor, 1e-9)).clamp(0.0, 1.0)
            d[bi] = dl
            G_t = _seg_sum3(e[bi, None] * d[bi], ib, nc)
        n_face = (self.side / self.h) * (float(self.reach_band) / self.h)
        G = G_t.cpu().numpy()
        sg = np.minimum(1.0, np.linalg.norm(G, axis=1) / n_face)
        self._G = (G, sg)
        self.last = dict(self.last, edge_points=int((e > 0).sum()),
                         steer_mean=float(sg[cnt > 0].mean()) if (cnt > 0).any() else 0.0)
        if self.reach > 0.0:
            u = _dot3(d, p_t[idx]).clamp(min=0.0)
            push = self.reach * (e * u)[:, None] * d
            tot = _seg_sum3(push[sel], isel, nc)
            push[sel] = push[sel] - (tot / n_all.clamp(min=1.0)[:, None])[isel]
            push = torch.where(sel[:, None], push, zero)
            # the reach is gated like the walk (see ActiveForceSphereCrawl.forward)
            push = push * gate[idx][:, None]
            a = a + push
        if self.zero_net:
            mp = getattr(lvl, "mass", None)
            m = mp.detach().to(f64) if mp is not None else torch.ones(N, dtype=f64, device=dev)
            m = m * alive.to(f64)
            # the two sums over ALL points are numpy's (a sequential sum over rows, and the pairwise sum of the masses): their order is
            # part of the answer, so they are taken on the host, on 8 MB and 2.5 MB copied back
            net = (m[:, None] * a).cpu().numpy().sum(0) / max(float(m.cpu().numpy().sum()), 1e-30)
            self.last["net_acc_before"] = float(np.linalg.norm(net))
            a = a - torch.as_tensor(net, dtype=f64, device=dev)[None, :] * alive.to(f64)[:, None]
        return {self.at: a.to(dt_)}
