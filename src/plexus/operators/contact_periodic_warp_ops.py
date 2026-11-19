"""exp10 (2026-11-19): the contact search of a PERIODIC box on the device. Run by contact_finder[sites] when `periodic: true` and there is a GPU (adhesion_warp_ops).

TESTED (tests/test_exp09_periodic.py, 2026-11-19): on exp09's periodic box rna1_3dp it gives the CPU kd-tree's pair list, junction rows and areas bit for bit.

WHY. contact_finder[sites] with `periodic: true` (exp09 rna1_3dp, exp10 oo1) hands the search to the CPU kd-tree of contact_finder: every frame the positions of every
live point go to the host, a periodic tree is built and queried, and the result comes back. At exp10's 540,000 points the GPU sits at 6 to 9 % (Cedric: "we should use
the right rig, not the cpu one"). The hash grid of contact_finder[hashgrid] already wraps: a cell index is taken modulo the grid's size. So if the grid's cells divide
the box exactly along every axis, a point and its periodic images fall in the same cell, a query near a face visits the cells of the opposite face, and all that is
left to do is to measure every distance to the NEAREST image.

    periodic_forward    what contact_finder[sites] runs when `periodic: true` on a GPU: the search of a box periodic on every axis as one Warp kernel on a hash grid whose
                        cells divide the box. No operator of its own (Cedric, 2026-11-19: "we had the periodic working in exp09 ??? why do we need variants in exp10"):
                        the same operator, the same spec, the search on the device instead of on the host; the CPU kd-tree stays the fallback
"""
import math

import numpy as np
import torch

from plexus.operators import adhesion_warp_ops as AW
from plexus.operators.adhesion_ops import FACE_A, FACE_B, FACE_L, _FG
from plexus.operators.adhesion_warp_ops import HAVE_WARP

if HAVE_WARP:
    wp = AW.wp

    @wp.kernel
    def _jc_partner_periodic(P32: wp.array(dtype=wp.vec3), PX: wp.array(dtype=wp.vec3d), CID: wp.array(dtype=wp.int32),
                             grid: wp.uint64, rq: float, ub2: wp.float64, ub2f: float, k_nn: int, Lf: wp.vec3, Ld: wp.vec3d,
                             PART: wp.array(dtype=wp.int32)):
        """adhesion_warp_ops._jc_partner with every distance taken to the nearest periodic image (periods Lf, Ld): one thread per live point, the local index of its
        partner (the nearest point of another cell within the bound, if it is among the point's k_nn nearest points), or -1."""
        i = wp.tid()
        pf = P32[i]
        px = PX[i]
        ci = CID[i]
        half = wp.float64(0.5)
        best_d = wp.float64(1.0e300)
        best_j = int(-1)
        j = int(0)
        q = wp.hash_grid_query(grid, pf, rq)
        while wp.hash_grid_query_next(q, j):
            if CID[j] != ci:
                df = P32[j] - pf
                df = wp.vec3(df[0] - Lf[0] * wp.floor(df[0] / Lf[0] + 0.5), df[1] - Lf[1] * wp.floor(df[1] / Lf[1] + 0.5), df[2] - Lf[2] * wp.floor(df[2] / Lf[2] + 0.5))
                if wp.dot(df, df) <= ub2f:                     # float32 prefilter, margin included in ub2f
                    pj = PX[j]
                    dx = pj[0] - px[0]
                    dy = pj[1] - px[1]
                    dz = pj[2] - px[2]
                    dx = dx - Ld[0] * wp.floor(dx / Ld[0] + half)
                    dy = dy - Ld[1] * wp.floor(dy / Ld[1] + half)
                    dz = dz - Ld[2] * wp.floor(dz / Ld[2] + half)
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
            df = wp.vec3(df[0] - Lf[0] * wp.floor(df[0] / Lf[0] + 0.5), df[1] - Lf[1] * wp.floor(df[1] / Lf[1] + 0.5), df[2] - Lf[2] * wp.floor(df[2] / Lf[2] + 0.5))
            if wp.dot(df, df) <= ub2f:
                pj = PX[j]
                dx = pj[0] - px[0]
                dy = pj[1] - px[1]
                dz = pj[2] - px[2]
                dx = dx - Ld[0] * wp.floor(dx / Ld[0] + half)
                dy = dy - Ld[1] * wp.floor(dy / Ld[1] + half)
                dz = dz - Ld[2] * wp.floor(dz / Ld[2] + half)
                d2 = dx * dx + dy * dy + dz * dz
                if d2 < ub2:
                    if (d2 < best_d) or (d2 == best_d and j < best_j):
                        rank = rank + 1
        if rank < k_nn:
            PART[i] = best_j
        else:
            PART[i] = -1


def grid_cells(L, rq, tol=1.0e-6):
    """(cell width, (nx, ny, nz)): a cell that divides the three periods `L` exactly -- a whole number of cells along x whose width gives whole numbers of cells along y
    and z too. Of those, the smallest not smaller than the search radius `rq` (a query then visits 27 cells); if every one is smaller, the largest (a query visits more
    cells, which the hash grid allows). Cells under a quarter of `rq` are not tried; a box whose periods share no cell raises."""
    L = [float(v) for v in L]
    ok = []
    for nx in range(1, int(math.floor(L[0] / (0.25 * float(rq)))) + 1):
        w = L[0] / nx
        ny, nz = L[1] / w, L[2] / w
        if abs(ny - round(ny)) < tol * max(ny, 1.0) and abs(nz - round(nz)) < tol * max(nz, 1.0) and round(ny) >= 1 and round(nz) >= 1:
            ok.append((w, (nx, int(round(ny)), int(round(nz)))))
    if not ok:
        raise ValueError(f"contact_finder[sites], periodic: no hash-grid cell divides the periods {L} exactly; make the box's sides whole multiples of one length (they "
                         f"are multiples of the point spacing in exp09 and exp10), or use contact_finder[sites], which searches on the CPU")
    big = [q for q in ok if q[0] >= float(rq)]
    return min(big, key=lambda q: q[0]) if big else max(ok, key=lambda q: q[0])


def _partners_periodic(owner, Xw, cid, ub, k_nn, L):
    """[n] int64: adhesion_warp_ops._partners for a periodic box. `Xw` the live points' positions in float64, already wrapped into [0, L); the hash grid is built with
    the cell of `grid_cells`, so that the grid's own wrap is the box's."""
    dev = Xw.device
    n = int(Xw.shape[0])
    p32 = Xw.float().contiguous()
    Xw = Xw.contiguous()
    cid32 = cid.to(torch.int32).contiguous()
    rq = float(ub) * (1.0 + 1.0e-4) + 1.0e-5                 # a superset of the true neighbours: the exact test is the float64 one
    w, dims = grid_cells(L, rq)
    # the candidates' radius, wider than rq by what the float32 side can be off by: the grid's period along an axis (cells x the float32 cell width) against the box's,
    # and the rounding of a coordinate as large as the box (a world of 3,400 length units rounds positions to 2e-4). The exact test below it is unchanged
    w32 = float(np.float32(w))
    rq = rq + max(abs(float(L[k]) - dims[k] * w32) for k in range(3)) + 16.0 * float(np.finfo(np.float32).eps) * max(float(v) for v in L)
    part = torch.empty(n, dtype=torch.int32, device=dev)
    with AW._on_torch_stream(dev):
        gr = getattr(owner, "_pgrid", None)
        if gr is None or getattr(owner, "_pgrid_dims", None) != dims:
            gr = owner._pgrid = wp.HashGrid(dims[0], dims[1], dims[2], device=AW._wdev(dev))
            owner._pgrid_dims = dims
        wp32 = wp.from_torch(p32, dtype=wp.vec3)
        gr.build(points=wp32, radius=float(w))
        wp.launch(_jc_partner_periodic, dim=n, device=AW._wdev(dev),
                  inputs=[wp32, wp.from_torch(Xw, dtype=wp.vec3d), wp.from_torch(cid32), gr.id, rq, float(ub) * float(ub), rq * rq, int(k_nn),
                          wp.vec3(float(L[0]), float(L[1]), float(L[2])), wp.vec3d(float(L[0]), float(L[1]), float(L[2])), wp.from_torch(part)])
    return part.long()


def periodic_forward(self, H, mask=None):
    """contact_finder[sites].forward for a box periodic on every axis, on the device: adhesion_warp_ops.JunctionContactHashgrid.forward with the neighbour search on a
    hash grid whose cells divide the box (`_partners_periodic`) and its two separations taken to the nearest image. `self` is the finder (its parameters and its idle
    counters). Returns {} as the finder does; raises ValueError when no cell divides the box (the caller then searches on the CPU)."""
    P, C, J = H.level(self.points), H.level(self.cells), H.level(self.at)
    dev, dt_ = P.state.device, P.state.dtype
    nC = int(C.state.shape[0])
    live = torch.nonzero(P.occ > 0.5).reshape(-1)
    n = int(live.numel())
    if n == 0:
        return AW.JunctionContactHashgrid.forward(self, H, mask)
    Lv = np.asarray([float(v) for v in H.world_size], np.float64)
    Lt = torch.as_tensor(Lv, dtype=torch.float64, device=dev)
    X = P.get("pos").detach()[live].double()
    Xw = X - Lt * torch.floor(X / Lt)                                           # every point's image inside the box [0, L)
    Xw = torch.where(Xw >= Lt, Xw - Lt, Xw)                                     # a coordinate a rounding left AT the period
    cid = self._cell_index(H)[live]
    H._adh_period = float(Lv[0])
    H._adh_box = Lv                                                             # every axis's period, for bond_springs' nearest image

    def near(d):                                                                # a separation to the nearest image
        return d - Lt * torch.round(d / Lt)
    part = _partners_periodic(self, Xw, cid, self.bond_reach + self.skin, self.k_nn, Lv)
    i = torch.nonzero(part >= 0).reshape(-1)
    j = part[i]
    ci, cj = cid[i], cid[j]
    a, b = torch.minimum(ci, cj), torch.maximum(ci, cj)
    # from here on: adhesion_warp_ops.JunctionContactHashgrid.forward, its two separations X[j] - X[i] taken to the nearest image
    if "pol" in C.state_schema:
        pol = C.get("pol").detach().double()
    else:
        pol = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=dev).repeat(nC, 1)
    pol = pol / AW._norm3(pol).clamp(min=1e-12)[:, None]
    dpair = near(Xw[j] - Xw[i])
    dvec = dpair / AW._norm3(dpair).clamp(min=1e-12)[:, None]
    dvec = torch.where((ci == a)[:, None], dvec, -dvec)                          # from cell a toward cell b
    k0 = a * nC + b
    u0, inv0 = torch.unique(k0, return_inverse=True)
    nsum = AW._seg_sum3(dvec, inv0, int(u0.numel()))
    nmean = nsum / AW._norm3(nsum).clamp(min=1e-12)[:, None]
    ua, ub = u0 // nC, u0 % nC
    cut = 1.0 / math.sqrt(2.0)
    ca_ = AW._dot3(nmean, pol[ua])
    cb_ = -AW._dot3(nmean, pol[ub])
    fa = torch.full((u0.numel(),), FACE_L, dtype=torch.long, device=dev)
    fa[ca_ > cut] = FACE_A
    fa[ca_ < -cut] = FACE_B
    fb = torch.full((u0.numel(),), FACE_L, dtype=torch.long, device=dev)
    fb[cb_ > cut] = FACE_A
    fb[cb_ < -cut] = FACE_B
    fg = torch.as_tensor(_FG, device=dev)[fa, fb][inv0]
    key = k0 * 6 + fg
    touch = AW._norm3(dpair) < self.bond_reach
    ia = torch.where(ci == a, i, j)
    tk, tp = key[touch], ia[touch]
    if tk.numel():                                                               # contact per key: the distinct points of the LOWER cell within bond_range, x spacing^2
        codes = torch.unique(tk * n + tp)
        uk, cnt = torch.unique(codes // n, return_counts=True)
        area_v = cnt.double() * (self.h ** 2)
    else:
        uk = torch.empty(0, dtype=torch.long, device=dev)
        area_v = torch.empty(0, dtype=torch.float64, device=dev)
    occJ = J.occ > 0.5
    sch = J.state_schema
    cA, cB, cF, cR, cAr, cL = (sch[k][0] for k in ("ca", "cb", "fg", "rho", "area", "load"))
    st = J.state.detach().clone()
    rowkey = (torch.round(st[:, cA]).long() * nC + torch.round(st[:, cB]).long()) * 6 + torch.round(st[:, cF]).long()      # in int64: exact above 1,700 cells too
    rowkey = torch.where(occJ, rowkey, torch.full_like(rowkey, -1))
    if not isinstance(self._idle, torch.Tensor) or tuple(self._idle.shape) != (J.n,) or self._idle.device != st.device:
        self._idle = torch.zeros(J.n, dtype=torch.long, device=st.device)
    idle = self._idle
    sk, srow = AW._known(occJ, rowkey)
    new = uk[AW._lookup(sk, srow, uk) < 0]
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
            sk, srow = AW._known(occJ, rowkey)
    st[:, cAr] = 0.0                                                             # area, load, idle frames, retirement
    rows_k = AW._lookup(sk, srow, uk)
    okk = rows_k >= 0
    st[rows_k[okk], cAr] = area_v[okk].to(st.dtype)
    acc = getattr(H, "_adh_load_acc", None)
    nsub = max(int(getattr(H, "_adh_load_n", 0)), 1)
    if acc is not None and acc.numel() == J.n:
        ld = AW._tdiv(acc.detach(), nsub)
        q = ld / st[:, cAr].clamp(min=1e-12)
        st[:, cL] = torch.where(st[:, cAr] > 0, q, torch.zeros_like(q))
    H._adh_load_acc = torch.zeros(J.n, device=dev, dtype=dt_)
    H._adh_load_n = 0
    has = st[:, cAr] > 0
    self._idle = idle = torch.where(has, torch.zeros_like(idle), idle + 1)
    gone = occJ & (idle >= self.retire_after) & (st[:, cR] < 1e-3)
    lk = AW._lookup(sk, srow, key)                                               # the Verlet list for the substeps: global point indices and the junction row of each pair
    H._adh_pairs = (live[i], live[j], lk)
    occJ = occJ & ~gone
    J.state = st
    J.occ = occJ.to(J.occ.dtype)
    H.junction_rows = int(occJ.sum())
    return {}
