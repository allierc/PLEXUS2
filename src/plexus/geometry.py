"""Geometry helpers shared by the engine and operators: minimum-image
displacement and a blockwise radius neighbour graph.

Kept here (package level, not in an operator file) because both the edge builder
and any pairwise operator need the same boundary-aware displacement, and the
engine may want it too.
"""
from __future__ import annotations

import torch

try:
    import warp as wp
    wp.init()
    HAVE_WARP = True
except Exception:
    HAVE_WARP = False

_GRID_DIM = 64                                   # cells per axis of the Warp hash grid; coordinates beyond wrap, which only adds candidates the exact distance test rejects
_GRIDS = {}                                      # one hash grid per CUDA device, reused between calls
_WARP_ABOVE = 4e6                                # the hash grid is used from this many (query x reference) point pairs; below it a few rows of cdist are cheaper than building the grid

if HAVE_WARP:

    @wp.kernel
    def _pairs_scan(Q: wp.array(dtype=wp.vec3), R: wp.array(dtype=wp.vec3), grid: wp.uint64, r: float, same: int, count_only: int,
                    OFF: wp.array(dtype=wp.int32), CNT: wp.array(dtype=wp.int32), EJ: wp.array(dtype=wp.int32)):
        """One thread per query point, run twice: count its neighbours within r in R, then write their indices at the offset OFF[i]."""
        i = wp.tid()
        qp = Q[i]
        c = int(0)
        q = wp.hash_grid_query(grid, qp, r)
        j = int(0)
        while wp.hash_grid_query_next(q, j):
            if same == 0 or j != i:
                if wp.length(R[j] - qp) <= r:
                    if count_only == 0:
                        EJ[OFF[i] + c] = j
                    c = c + 1
        if count_only == 1:
            CNT[i] = c


def minimum_image(d: torch.Tensor, periodic: bool, world=1.0) -> torch.Tensor:
    """Wrap a displacement `d[..., D]` to its nearest periodic image (no-op when not
    periodic). `world` is the per-axis box size: a scalar keeps the legacy 2D torus
    [0,W] x [0,1] (x wraps by W, y by 1); a length-D vector wraps each axis by its own
    size -- the dimension-generic form. Used for both the edge cutoff and the force
    direction, so they agree."""
    if not periodic:
        return d
    if d.shape[-1] == 2 and not torch.is_tensor(world) and not isinstance(world, (list, tuple)):
        W = float(world)                                   # legacy 2D: [0,W] x [0,1]
        dx = d[..., 0] - W * torch.round(d[..., 0] / W)
        dy = d[..., 1] - torch.round(d[..., 1])
        return torch.stack([dx, dy], dim=-1)
    W = torch.as_tensor(world, device=d.device, dtype=d.dtype)   # [D] per-axis box size
    if bool((W == 1).all()):                                      # unit box: skip the /W and *W passes
        return d - torch.round(d)                                #   (two fewer full-tensor kernels)
    return d - W * torch.round(d / W)


def neighbour_mean(pos, occ, edge_index, periodic, world, msg_fn) -> torch.Tensor:
    """Mean over each receiver i's live neighbours j of `msg_fn(i, j, d_ij)` -> [N, D].

    `edge_index` is [2, E] (row0 receiver i, row1 neighbour j); `d_ij = pos_j - pos_i`
    is the minimum-image displacement. The shared reduction behind the boids steering
    rules (cohesion / alignment / separation). Dimension-generic: output width = D =
    pos.shape[-1].
    """
    N, D = pos.shape[0], pos.shape[-1]
    if edge_index.numel() == 0:
        return torch.zeros(N, D, device=pos.device)
    i, j = edge_index[0], edge_index[1]
    d = minimum_image(pos[j] - pos[i], periodic, world)
    msg = msg_fn(i, j, d) * occ[j, None]                    # ignore dormant neighbours
    acc = torch.zeros(N, D, device=pos.device).index_add_(0, i, msg)
    deg = torch.zeros(N, device=pos.device).index_add_(0, i, occ[j])
    return (acc / deg.clamp(min=1.0)[:, None]) * occ[:, None]


def _radius_block_mask(pos_i, pos_all, min2, max2, gi, jidx, periodic, world):
    """Per-block radius mask [B, N]: True where min2 < |pos_i - pos_all|^2 < max2 and j > i.
    Pure tensors, so a fusing backend can collapse the diff/square/reduce/compare into one
    kernel and never materialise the [B, N, D] difference; whether that happens is the
    engine's `compile:` decision, not this function's. The dynamic edge extraction
    (`.nonzero()`) stays OUTSIDE, eager."""
    d = minimum_image(pos_i[:, None, :] - pos_all[None, :, :], periodic, world)
    dist2 = (d * d).sum(-1)                                  # [B, N]
    return (dist2 > min2) & (dist2 < max2) & (jidx > gi)     # radius rule + upper triangle


def edges_radius_blockwise(
    pos: torch.Tensor,
    occ: torch.Tensor,
    r_min: float,
    r_max: float,
    periodic: bool = False,
    world_width=1.0,
    block: int = 2048,
) -> torch.Tensor:
    """Bidirectional edge_index [2, E] for live pairs with `r_min < dist < r_max`,
    built blockwise so the O(N^2) distance matrix is never materialised (scales to
    1e4-1e5 nodes). Row 0 = receiver i, row 1 = neighbour j; every pair appears in
    both directions. Ported from ParticleGraph's edges_radius_blockwise.

    The occupancy filter runs ONCE over the surviving edge list (O(E),
    `live[row] & live[col]`) rather than as two `[B, N]` boolean reductions inside
    the block loop -- the live/live mask only ever drops pairs touching a dormant
    node, so the post-loop filter gives the identical edge set at lower cost.
    Boundary handling goes through `minimum_image` (periodic 2D/3D, free; scalar or
    per-axis world all supported).
    """
    device = pos.device
    N = pos.shape[0]
    min2, max2 = r_min * r_min, r_max * r_max
    jidx = torch.arange(N, device=device)[None, :]          # [1, N]

    rows, cols = [], []
    for i0 in range(0, N, block):
        i1 = min(i0 + block, N)
        gi = torch.arange(i0, i1, device=device)[:, None]   # [B, 1] global i
        m = _radius_block_mask(pos[i0:i1], pos, min2, max2, gi, jidx,
                               periodic, world_width)       # [B, N] radius+triangle mask
        ii, jj = m.nonzero(as_tuple=True)                   # dynamic edge extraction stays eager
        rows.append(ii + i0); cols.append(jj)
        del m

    row = torch.cat(rows) if rows else torch.empty(0, dtype=torch.long, device=device)
    col = torch.cat(cols) if cols else torch.empty(0, dtype=torch.long, device=device)

    live = occ > 0                                          # drop pairs touching a dormant node
    keep = live[row] & live[col]                            # O(E), once over the edge list
    row, col = row[keep], col[keep]

    eij = torch.stack([row, col], 0)                        # [2, E_half]  (i, j) with j > i
    return torch.cat([eij, eij.flip(0)], 1).contiguous()    # mirror -> both directions


def radius_pairs(xq, xr, radius, same=False, chunk=1024):
    """All index pairs (qi, rj) with |xq[qi] - xr[rj]| <= radius, grouped by qi -> (qi [E] long, rj [E] long, cnt [Nq] long, the pairs of each query).

    `same` says xq is xr, and drops the pair of a point with itself. On a CUDA device with a large pair count the search is a Warp hash grid (cells of side `radius`, work O(N x neighbours)); otherwise `chunk` rows of
    torch.cdist at a time (work O(N^2), memory follows the chunk). Every distance is taken from coordinate differences on both branches (the hash grid always did; the `torch.cdist` branch uses compute_mode="donot_use_mm_for_euclid_dist" since 2026-11-08, before which its matrix-product distances put points near the radius on the wrong side by round-off at coordinates far from the origin). Integer indices only: no gradient."""
    nq, nr, D = xq.shape[0], xr.shape[0], xq.shape[1]
    if HAVE_WARP and xq.device.type == "cuda" and float(nq) * float(nr) > _WARP_ABOVE:
        dev = xq.device; wdev = wp.device_from_torch(dev)
        q3 = torch.zeros(nq, 3, device=dev, dtype=torch.float32); q3[:, :D] = xq.float()
        r3 = q3 if same else torch.zeros(nr, 3, device=dev, dtype=torch.float32)
        if not same:
            r3[:, :D] = xr.float()
        gr = _GRIDS.get(str(dev))
        if gr is None:
            gr = _GRIDS[str(dev)] = wp.HashGrid(_GRID_DIM, _GRID_DIM, _GRID_DIM, device=wdev)
        wq, wr = wp.from_torch(q3.contiguous(), dtype=wp.vec3), wp.from_torch(r3.contiguous(), dtype=wp.vec3)
        gr.build(points=wr, radius=float(radius))
        cnt = torch.zeros(nq, device=dev, dtype=torch.int32); zero = torch.zeros(1, device=dev, dtype=torch.int32)
        args = [wq, wr, gr.id, float(radius), 1 if same else 0]
        wp.launch(_pairs_scan, dim=nq, device=wdev, inputs=args + [1, wp.from_torch(zero), wp.from_torch(cnt), wp.from_torch(zero)])
        off = (torch.cumsum(cnt, 0, dtype=torch.int32) - cnt).contiguous(); E = int(cnt.sum().item())
        ej = torch.empty(E, device=dev, dtype=torch.int32)
        if E:
            wp.launch(_pairs_scan, dim=nq, device=wdev, inputs=args + [0, wp.from_torch(off), wp.from_torch(cnt), wp.from_torch(ej)])
        cnt = cnt.long()
        return torch.repeat_interleave(torch.arange(nq, device=dev), cnt), ej.long(), cnt
    qs, rs = [], []
    for i0 in range(0, nq, chunk):
        m = torch.cdist(xq[i0:i0 + chunk], xr, compute_mode="donot_use_mm_for_euclid_dist") <= radius        # distances from coordinate differences: the default mode takes them from a matrix product, ~0.1 % of a radius of 0.3 off at coordinates ~25
        if same:
            r_ = torch.arange(m.shape[0], device=m.device); m[r_, r_ + i0] = False
        ii, jj = m.nonzero(as_tuple=True); qs.append(ii + i0); rs.append(jj)
    qi = torch.cat(qs) if qs else torch.empty(0, dtype=torch.long, device=xq.device); rj = torch.cat(rs) if rs else qi.clone()
    return qi, rj, torch.bincount(qi, minlength=nq)


def _point_spacing(x, n=2048, chunk=256):
    """The point spacing s: the median distance from a point to its nearest neighbour in x, over a strided sample of at most n points (the median, so that a few sparse points do not set the scale of a dense skin)."""
    s = x[::max(1, x.shape[0] // n)]
    return torch.cat([torch.cdist(s[i0:i0 + chunk], x).topk(2, largest=False).values[:, 1] for i0 in range(0, s.shape[0], chunk)]).median()


def facing_pairs(xa, xb, ring=2.0, cone=3.0, reach=12.0, edge=0.25, facing=0.7, max_gap=None, slack=0.25, ahead=0.5, min_neighbours=6):
    """The part of the local gap of body a to body b that holds for a whole frame -> a dict: the front surface points `idx` of a, their outward normals `ns`, and the candidate pairs (`qi` index into idx, `jb` index into xb).

    A point of a is a surface point when the mean of its neighbours in a (all within `ring` point spacings, at least `min_neighbours` of them) lies at least `edge` point spacings away (its neighbours are on one side); its outward normal is the direction away from that mean. It is a front
    point when it faces b (cosine of the normal with the line from a's centre to b's centre above `facing`) and none of a's own points lies ahead of it inside the cone (|x_j - x_i| along the normal in (`ahead` x s, reach_len), sideways
    below cone_len; a neighbour in the same face, off the plane by less than `ahead` point spacings, is beside the point, not ahead of it). A candidate pair is a point j of b inside the cone around a front point's normal, widened by `slack` x reach_len for the motion of the points during the frame. The point spacing s is the median nearest-neighbour
    distance of a; reach_len = reach x s (at most `max_gap`, a length: a gap beyond it is not wanted) and cone_len = cone x s. The searches are `radius_pairs` (hash grid on the GPU); nothing here carries a gradient."""
    with torch.no_grad():
        Na = xa.shape[0]; sp = float(_point_spacing(xa)); cone_len = cone * sp; reach_len = reach * sp if max_gap is None else min(reach * sp, float(max_gap)); mv = slack * reach_len
        qi, rj, cnt = radius_pairs(xa, xa, ring * sp, same=True)                                  # the neighbours of every point within `ring` spacings
        off = torch.zeros_like(xa).index_add(0, qi, xa[rj] - xa[qi]) / cnt.clamp(min=1).to(xa.dtype)[:, None]       # their mean, relative to the point
        n = -off / off.norm(dim=1, keepdim=True).clamp(min=1e-12); u = xb.mean(0) - xa.mean(0); u = u / u.norm().clamp(min=1e-12)
        surf = torch.nonzero((cnt >= min_neighbours) & (off.norm(dim=1) > edge * sp) & ((n @ u) > facing)).reshape(-1)
        qs, rs, _ = radius_pairs(xa[surf], xa, (reach_len ** 2 + cone_len ** 2) ** 0.5)          # own points around each surface point: is anything ahead of it?
        Dm = xa[rs] - xa[surf][qs]; ns = n[surf]; t = (Dm * ns[qs]).sum(-1); perp = (Dm - t[:, None] * ns[qs]).norm(dim=-1)
        blocked = torch.zeros(surf.numel(), device=xa.device).index_add(0, qs, ((t > ahead * sp) & (t < reach_len) & (perp < cone_len)).float()) > 0
        idx = surf[~blocked]; ns = n[idx]
        qi, jb, _ = radius_pairs(xa[idx], xb, ((reach_len + mv) ** 2 + (cone_len + mv) ** 2) ** 0.5)       # points of b around each front point
        Dm = xb[jb] - xa[idx][qi]; t = (Dm * ns[qi]).sum(-1); perp = (Dm - t[:, None] * ns[qi]).norm(dim=-1)
        c = (t > -mv) & (t < reach_len + mv) & (perp < cone_len + mv)                              # keep the pairs inside the widened cone
    return dict(idx=idx, ns=ns, qi=qi[c], jb=jb[c], reach_len=reach_len, cone_len=cone_len, spacing=sp)


def facing_pairs_bodies(X, live, par, ids, near, ring=2.0, cone=3.0, reach=12.0, edge=0.25, facing=0.7, max_gap=None, slack=0.25, ahead=0.5, min_neighbours=6):
    """`facing_pairs` of every ordered pair of bodies in `near` in one pass -> one dict for `facing_gaps(X, X, pairs)`: the front points `idx` and normals `ns` of each (point, other body) entry, the candidate pairs (`qi` index into idx, `jb` index into X) and, per candidate, the
    `reach_len` and `cone_len` of the body that owns the front point.

    X [N, D] positions, live [N] bool, par [N] the body (parent cell) of each point, ids the sorted body ids, near the ordered pairs (a, b) of body ids to search. What does not depend on the other body is done once per body, for all bodies together: the spacing s (a loop over the bodies),
    the ring neighbours and normals, the surface points, the points blocked by an own point ahead of them. What does is done for all front points together: one radius search of the front points around all points, kept where the point belongs to a body of a near pair whose centre lies
    toward the normal (cosine above `facing`) and inside the widened cone. An entry (front point, other body) with no candidate is left out: it would have had no gap and no force. The searches are `radius_pairs`, run at the largest radius of any body, and each body's own radius is then tested again on the rows it returns.
    Nothing here carries a gradient."""
    with torch.no_grad():
        dev = X.device; L = torch.nonzero(live).reshape(-1); xl = X[L]; inv = torch.searchsorted(ids, par[L]); nb, N = ids.numel(), xl.shape[0]
        nr = torch.tensor(near, device=dev).reshape(-1, 2); pa, pb = torch.searchsorted(ids, nr[:, 0]), torch.searchsorted(ids, nr[:, 1]); P = pa.numel()
        pid = torch.full((nb, nb), -1, dtype=torch.long, device=dev); pid[pa, pb] = torch.arange(P, device=dev)                  # pid[a, b] the number of the ordered pair (a, b), -1 where the bodies are not near
        cen = torch.zeros(nb, xl.shape[1], device=dev, dtype=xl.dtype).index_add(0, inv, xl) / torch.bincount(inv, minlength=nb).clamp(min=1).to(xl.dtype)[:, None]
        u = cen[pb] - cen[pa]; u = u / u.norm(dim=1, keepdim=True).clamp(min=1e-12)
        order = torch.argsort(inv); st = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(torch.bincount(inv, minlength=nb), 0)]).tolist()
        sp = torch.ones(nb, device=dev, dtype=xl.dtype); first = torch.zeros(nb, dtype=torch.bool, device=dev); first[pa] = True
        for b in torch.unique(pa).tolist():
            sp[b] = _point_spacing(xl[order[st[b]:st[b + 1]]])
        reach_len = reach * sp if max_gap is None else (reach * sp).clamp(max=float(max_gap)); cone_len = cone * sp; mv = slack * reach_len
        Rr = ring * sp
        qi, rj, _ = radius_pairs(xl, xl, float(Rr.max()), same=True)                                                          # the neighbours of every point within `ring` spacings of its own body
        d = xl[rj] - xl[qi]; k = (inv[qi] == inv[rj]) & (d.norm(dim=1) <= Rr[inv[qi]]); qi, d = qi[k], d[k]
        cnt = torch.bincount(qi, minlength=N); off = torch.zeros_like(xl).index_add(0, qi, d) / cnt.clamp(min=1).to(xl.dtype)[:, None]
        on = off.norm(dim=1); n = -off / on.clamp(min=1e-12)[:, None]
        surf = torch.nonzero((cnt >= min_neighbours) & (on > edge * sp[inv]) & first[inv]).reshape(-1)
        qs, rs, _ = radius_pairs(xl[surf], xl, float((reach_len ** 2 + cone_len ** 2).sqrt().max()))                          # own points around each surface point: is anything ahead of it?
        bs = inv[surf][qs]; k = inv[rs] == bs; qs, rs, bs = qs[k], rs[k], bs[k]
        Dm = xl[rs] - xl[surf][qs]; ns = n[surf][qs]; t = (Dm * ns).sum(-1); perp = (Dm - t[:, None] * ns).norm(dim=-1)
        blocked = torch.zeros(surf.numel(), device=dev).index_add(0, qs, ((t > ahead * sp[bs]) & (t < reach_len[bs]) & (perp < cone_len[bs])).float()) > 0
        front = surf[~blocked]
        qq, jj, _ = radius_pairs(xl[front], xl, float(((reach_len + mv) ** 2 + (cone_len + mv) ** 2).sqrt().max()))          # points of every body around each front point
        a = inv[front][qq]; p = pid[a, inv[jj]]; Dm = xl[jj] - xl[front][qq]; nq = n[front][qq]; t = (Dm * nq).sum(-1); perp = (Dm - t[:, None] * nq).norm(dim=-1)
        c = (p >= 0) & ((nq * u[p.clamp(min=0)]).sum(-1) > facing) & (t > -mv[a]) & (t < reach_len[a] + mv[a]) & (perp < cone_len[a] + mv[a])
        qq, jj, a, p = qq[c], jj[c], a[c], p[c]
        uk, ent = torch.unique(qq * P + p, return_inverse=True); i_ent = front[uk // P]
    return dict(idx=L[i_ent], ns=n[i_ent], qi=ent, jb=L[jj], reach_len=reach_len[a], cone_len=cone_len[a], spacing=sp)


def facing_gaps(xa, xb, pairs=None, ring=2.0, cone=3.0, reach=12.0, tau=0.00005, edge=0.25, facing=0.7, max_gap=None):
    """The local gap of body a's front surface points to body b -> (idx [m], gap [m], has [m] bool, n [m, D]).

    The gap of a front point is the soft minimum (temperature `tau`, a length) of (x_j - x_i) . n over the candidate points j of b with 0 < t < reach_len and sideways distance below cone_len (t the distance along the point's outward
    normal n); `has` says whether any point of b was in the cone, and the gap is 1e3 where none was. `pairs` is the output of `facing_pairs` (the front points, normals and candidate pairs): pass the one built at the start of the
    frame to reread the gaps at the current positions without searching again, or None to build it here. `idx` indexes xa; the normals are detached, the gaps are differentiable in the positions."""
    if pairs is None:
        pairs = facing_pairs(xa, xb, ring, cone, reach, edge, facing, max_gap)
    idx, ns, qi, jb = pairs["idx"], pairs["ns"], pairs["qi"], pairs["jb"]; M = idx.numel()
    D = xb[jb] - xa[idx][qi]; nq = ns[qi]; t = (D * nq).sum(-1)                                    # distance of each candidate along the normal, with gradient
    with torch.no_grad():
        ok = (t > 0) & (t < pairs["reach_len"]) & ((D - t[:, None] * nq).norm(dim=-1) < pairs["cone_len"])
        has = torch.zeros(M, device=xa.device).index_add(0, qi, ok.float()) > 0
    tt = torch.where(ok, t, torch.full_like(t, 1e3))                                              # a candidate outside the cone sits at 1e3 and weighs exp(-1e3 / tau) = 0
    with torch.no_grad():
        m = torch.full((M,), 1e3, device=xa.device, dtype=t.dtype).scatter_reduce(0, qi, tt, "amin", include_self=True)       # the shortest candidate of each point: a shift of the soft minimum that cancels, for stability
    s = torch.zeros(M, device=xa.device, dtype=t.dtype).index_add(0, qi, torch.exp(-(tt - m[qi]) / tau))
    g = torch.where(has, m - tau * torch.log(s.clamp(min=1e-30)), torch.full_like(m, 1e3))
    return idx, g, has, ns
