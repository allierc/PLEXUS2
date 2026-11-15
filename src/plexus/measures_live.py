"""Differentiable metrics of the live state: torch scalars with a graph back to the positions, read inside the window.

Each is registered by name and takes the hierarchy H plus the keyword arguments of its `metrics:` entry in the search
spec. The two the vesicle search uses:

    set_diameter   D = 2 x mean_i | x_i - mean(x) |  over the live rows of a set (the cells' centroids)
    grid_gini      G = 1 - sum_c phi_c(n)^2, averaged over the grid nodes n that carry mass, where phi_c(n) is the
                   volume fraction of cell c at node n from a trilinear deposit of its points on a lattice of spacing h;
                   0 when every node is one cell's, rising as two cells' points interleave (the cell-integrity ruler of
                   exp29, 2026-10-23: 0.03 for a tidy vesicle, 0.10-0.14 for interleaved ones)
    grid_entropy   the same deposit, mass-weighted Shannon entropy of phi over the cells at each node
    set_count      the live rows of a set (no gradient; a monitor)
"""
from __future__ import annotations

import math

import torch

from plexus.measures import register_live


def lattice_origin(x, h, origin="data"):
    """The corner of the deposit lattice, detached. `origin="data"` (the default): two spacings below the points' smallest coordinate, so the lattice follows the cloud; then a point that
    moves the cloud's extreme shifts the lattice under EVERY point, which the gradient does not see (a uniform translation of the cloud leaves the value unchanged and has gradient != 0; moving the
    one extreme point by 1e-3 of a length unit changed grid_gini 3,000 times more than its own gradient says, exp38 2026-11-08). `origin="world"`: the same corner snapped DOWN to a multiple of h, so the
    nodes are the same world positions whatever the extent of the cloud, the value is a function of the positions alone and the gradient is its derivative."""
    lo = x.detach().min(0).values
    if origin == "world":
        lo = torch.floor(lo / h) * h
    elif origin != "data":
        raise ValueError(f"origin must be 'data' or 'world', not {origin!r}")
    return lo - 2 * h


def deposit(x, pid, h, n_cells, origin="data"):
    """Trilinear deposit of the points x [N, 3] (torch) on a lattice of spacing h: the (cell, node) keys' weights phi, the node
    index of each key, and the node count. Differentiable in x through the weights. `origin`: see `lattice_origin`."""
    lo = lattice_origin(x, h, origin)
    g = (x - lo) / h
    i0 = torch.floor(g).long()
    f = g - i0.to(x.dtype)
    dims = (i0.detach().max(0).values + 3).tolist()
    keys, ws = [], []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2]))
                node = ((i0[:, 0] + dx) * dims[1] + (i0[:, 1] + dy)) * dims[2] + (i0[:, 2] + dz)
                keys.append(pid.long() * (dims[0] * dims[1] * dims[2]) + node)
                ws.append(w)
    keys = torch.cat(keys)
    ws = torch.cat(ws)
    uk, inv = torch.unique(keys, return_inverse=True)
    phi = torch.zeros(uk.shape[0], dtype=x.dtype, device=x.device).index_add_(0, inv, ws)      # phi_c(n) per (cell, node) key
    node_of_key = uk % (dims[0] * dims[1] * dims[2])
    un, inv_n = torch.unique(node_of_key, return_inverse=True)
    return phi, inv_n, un.shape[0]


def grid_mixing(x, pid, h, n_cells=None, origin="data"):
    """(gini, entropy) of the deposited per-cell fractions, both torch scalars with gradients to x."""
    n_cells = int(pid.max().item()) + 1 if n_cells is None else n_cells
    phi, inv_n, n_nodes = deposit(x, pid, h, n_cells, origin)
    Phi = torch.zeros(n_nodes, dtype=x.dtype, device=x.device).index_add_(0, inv_n, phi)
    S2 = torch.zeros(n_nodes, dtype=x.dtype, device=x.device).index_add_(0, inv_n, phi * phi)
    gini = ((Phi * Phi - S2).sum()) / (Phi * Phi).sum().clamp(min=1e-12)
    q = phi / Phi[inv_n].clamp(min=1e-12)
    Hn = torch.zeros(n_nodes, dtype=x.dtype, device=x.device).index_add_(0, inv_n, -(q * torch.log(q.clamp(min=1e-12))))
    ent = (Phi * Hn).sum() / Phi.sum().clamp(min=1e-12)
    return gini, ent


def _live(H, set_name):
    L = H.level(set_name)
    occ = getattr(L, "occ", None)
    live = (occ > 0.5) if occ is not None else torch.ones(L.state.shape[0], dtype=torch.bool, device=L.state.device)
    return L, live


@register_live("set_diameter")
def set_diameter(H, set="cell", **_):
    L, live = _live(H, set)
    x = L.get("pos")[live]
    return 2.0 * (x - x.mean(0)).norm(dim=1).mean()


@register_live("set_roundness")
def set_roundness(H, set="cell", **_):
    """HOW ROUND THE VESICLE IS: the standard deviation of the live cells' centroid distances from the vesicle's centroid over their mean (0 a sphere of
    centres; the axis lobes of the exp29 sphere read 0.18 at 383 cells against 0.08 with the rotation wrapper). Differentiable in the cell positions."""
    L, live = _live(H, set)
    x = L.get("pos")[live]
    r = (x - x.mean(0)).norm(dim=1)
    return r.std() / r.mean()


@register_live("grid_gini")
def grid_gini(H, set="pt", parents="cell", h=0.2, origin="data", **_):
    P, live = _live(H, set)
    x = P.get("pos")[live]
    pid = P.parent.long()[live]
    return grid_mixing(x, pid, float(h), H.level(parents).state.shape[0], origin)[0]


@register_live("grid_entropy")
def grid_entropy(H, set="pt", parents="cell", h=0.2, origin="data", **_):
    P, live = _live(H, set)
    x = P.get("pos")[live]
    pid = P.parent.long()[live]
    return grid_mixing(x, pid, float(h), H.level(parents).state.shape[0], origin)[1]


@register_live("soft_degree")
def soft_degree(H, set="cell", points="pt", h=0.2, reach=1.15, width=0.1, **_):
    """COHESION: the mean number of touching neighbours per cell, softly counted. Two cells touch when their centroids are closer than
    `reach` x (r_i + r_j), r_i = (3 n_i h^3 / 4 pi)^(1/3) the radius of cell i from its n_i live points of spacing h; the count is a
    sigmoid of width `width` x that distance, so the metric is differentiable in the centroids. A confluent sheet cell has 5-6 in-plane
    neighbours; a vesicle whose cells are pushed apart (exp30 smoke 4, 2026-10-23: the diameter grew x1.20 with gaps opening between the
    cells, which the Gini mixing does not see) loses them. Used as a log-ratio term with a NEGATIVE weight, like the diameter: the loss is
    minimised, so -w log(C_end / C_start) rewards contact RISING (the first search spec carried +20 and rewarded the fall; audit C1)."""
    C, live = _live(H, set)
    P, plive = _live(H, points)
    par = P.parent.long()
    n = torch.zeros(C.state.shape[0], dtype=torch.float32, device=P.state.device).index_add_(0, par[plive], torch.ones(int(plive.sum()), device=P.state.device))
    r = (3.0 * n * float(h) ** 3 / (4.0 * math.pi)).clamp(min=1e-9).pow(1.0 / 3.0)
    x = C.get("pos")[live]
    rl = r[live]
    # NOT torch.cdist: its backward is NaN on the diagonal (two identical points, 0 / 0); the self pair is excluded below anyway
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    eye = torch.eye(x.shape[0], device=x.device, dtype=x.dtype)
    d = torch.sqrt(d2 + eye)                                   # the diagonal reads 1, never 0, and is dropped from the count
    thr = float(reach) * (rl[:, None] + rl[None, :])
    soft = torch.sigmoid((thr - d) / (float(width) * thr)) * (1.0 - eye)
    return soft.sum(1).mean()


@register_live("set_count")
def set_count(H, set="cell", **_):
    _, live = _live(H, set)
    return live.sum().to(torch.float32)


def _cell_shape(H, set, points, h):
    """Per cell slot: live point count n, radius of gyration squared rg2 about the cell's own centroid, and the ball radius R of n points at
    spacing h; plus the live-cell mask and the per-point parent and distance to the centroid (all differentiable in the point positions)."""
    C, live = _live(H, set)
    P, plive = _live(H, points)
    par = P.parent.long()[plive]
    x = P.get("pos")[plive]
    nC = C.state.shape[0]
    one = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
    n = torch.zeros(nC, dtype=x.dtype, device=x.device).index_add_(0, par, one)
    cen = torch.zeros(nC, x.shape[1], dtype=x.dtype, device=x.device).index_add_(0, par, x) / n.clamp(min=1.0)[:, None]
    r2 = ((x - cen[par]) ** 2).sum(1)
    rg2 = torch.zeros(nC, dtype=x.dtype, device=x.device).index_add_(0, par, r2) / n.clamp(min=1.0)
    R = (3.0 * n * float(h) ** 3 / (4.0 * math.pi)).clamp(min=1e-12).pow(1.0 / 3.0)
    has = live & (n > 0)
    return n, rg2, R, has, par, r2.sqrt()


@register_live("cell_stretch")
def cell_stretch(H, set="cell", points="pt", h=0.2, rho0=1.2, **_):
    """INTEGRITY BY SHAPE (2026-10-25, tested offline on 29 landed episodes with tools/exp30_integrity.py): the mean over live cells of
    relu(rho_c - rho0), rho_c = Rg_c / Rg_ball(N_c) the cell's radius of gyration over that of a solid ball of its own N_c points at spacing
    h (Rg_ball = sqrt(3/5) R, R = (3 N_c h^3 / 4 pi)^(1/3)). A round cell reads 1.0, a stretched or hollow one 1.2-1.3, a torn one 1.3-1.5
    (cell-level correlation with the pieces ruler -0.51; rho > 1.2 catches 0.93 of the torn cells and flags 0.25 of the intact, mostly
    stretched, ones). Zero when every cell is at least as compact as rho0; used with `form: hinge, ref: 0, unit: <gate width in rho>`."""
    n, rg2, R, has, _, _ = _cell_shape(H, set, points, h)
    rho = (rg2 / (0.6 * R * R).clamp(min=1e-12)).clamp(min=1e-12).sqrt()
    return torch.relu(rho[has] - float(rho0)).mean()


@register_live("cell_stray")
def cell_stray(H, set="cell", points="pt", h=0.2, reach=1.6, width=0.15, **_):
    """INTEGRITY BY PIECES (2026-10-25): the mean over live cells of the soft share of a cell's points farther than `reach` x R from its
    centroid (a sigmoid of width `width` x R; R the ball radius of its N_c points). A ball reads ~0.02, a torn cell the share of its detached
    pieces (median 0.09 over the torn cells of 29 episodes; cell-level correlation with the pieces ruler -0.65, the closest differentiable
    proxy of "every cell one piece"). Used with `form: hinge, ref: <the intact level, 0.02-0.03>, unit: <gate width>`."""
    n, rg2, R, has, par, d = _cell_shape(H, set, points, h)
    soft = torch.sigmoid((d - float(reach) * R[par]) / (float(width) * R[par]).clamp(min=1e-12))
    nC = R.shape[0]
    share = torch.zeros(nC, dtype=soft.dtype, device=soft.device).index_add_(0, par, soft) / n.clamp(min=1.0)
    return share[has].mean()


@register_live("wall_spread")
def wall_spread(H, points="pt", **_):
    """THE WALL'S THICKNESS AS A SPREAD: the standard deviation of the live points' distance from the vesicle's centroid. A one-cell-thick
    shell reads about a cell radius; a shell whose cells turn into radial columns (exp29: wall x2.7 against the vertex twin's x1.23) reads
    more. Differentiable in the point positions; used as a ratio to the warm-up's (`track_ref` against the vertex twin, exp32)."""
    P, live = _live(H, points)
    x = P.get("pos")[live]
    r = (x - x.mean(0)).norm(dim=1)
    return r.std()


@register_live("cell_radial_aspect")
def cell_radial_aspect(H, set="cell", points="pt", h=0.2, **_):
    """COLUMNAR OR FLAT: the mean over live cells of the cell's rms extent along the RADIAL direction (from the vesicle centroid through the
    cell's centroid) over its rms extent in the tangent plane, from the cell's second moments -- 1 isotropic, > 1 a radial column (exp29's
    MPM twin: 2.4), < 1 a flattened sheet cell (the vertex twin). No eigen-decomposition: the radial axis is given, the tangential variance
    is the remainder of the trace. Differentiable in the point positions (the radial direction is detached)."""
    C, live = _live(H, set)
    P, plive = _live(H, points)
    par = P.parent.long()[plive]
    x = P.get("pos")[plive]
    nC = C.state.shape[0]
    one = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
    n = torch.zeros(nC, dtype=x.dtype, device=x.device).index_add_(0, par, one)
    cen = torch.zeros(nC, x.shape[1], dtype=x.dtype, device=x.device).index_add_(0, par, x) / n.clamp(min=1.0)[:, None]
    c0 = x.mean(0)
    nrm = (cen - c0).detach()
    nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp(min=1e-9)
    d = x - cen[par]
    rad = (d * nrm[par]).sum(1)
    var_r = torch.zeros(nC, dtype=x.dtype, device=x.device).index_add_(0, par, rad * rad) / n.clamp(min=1.0)
    var_all = torch.zeros(nC, dtype=x.dtype, device=x.device).index_add_(0, par, (d * d).sum(1)) / n.clamp(min=1.0)
    var_t = ((var_all - var_r) / 2.0).clamp(min=1e-12)                   # the mean tangential variance per in-plane axis
    has = live & (n > 3)
    return (var_r[has] / var_t[has]).clamp(min=1e-12).sqrt().mean()

@register_live("contact_fraction")
def contact_fraction(H, set="cell", points="pt", h=0.4, thresh=0.1, width=0.05, **_):
    """ADHESION, geometry-free: the share of live points that sit in another cell's material. Every point deposits trilinearly on a
    lattice of spacing h -- TWICE the point spacing by default (0.4 for points at 0.2): a trilinear deposit reaches one lattice step, so
    at the point spacing two touching cells' boundary points, one spacing apart, would share no node and read as apart; at point i the OTHER cells' share of the local density is c^o_i = (Phi - phi_{c_i}) / Phi (the
    quantity `apical_constriction` reads to find a free face); the metric is the mean over points of sigmoid((c^o_i - thresh) / width).
    Unlike `soft_degree`, which compares centroid distances with SPHERE-equivalent radii and so reads a flattening sheet as losing
    neighbours (exp33, 2026-10-25: the contact term fought the diameter term on every knob for that reason), this one does not care
    about a cell's shape, only whether its points touch foreign points: an adhered sheet keeps it whatever its thickness, cells that
    come apart lose it (offline, within 1.5 h: the arm-100 control 0.14 -> 0.26 over 550 frames, the adhered fits 0.21-0.29, the
    dissociated diameter-only fit 0.015). Differentiable in the point positions through the deposit weights."""
    C, live = _live(H, set)
    P, plive = _live(H, points)
    x = P.get("pos")[plive]
    pid = P.parent.long()[plive]
    lo = x.detach().min(0).values - 2 * h
    g = (x - lo) / h
    i0 = torch.floor(g).long()
    f = g - i0.to(x.dtype)
    dims = (i0.detach().max(0).values + 3).tolist()
    nn = dims[0] * dims[1] * dims[2]
    corners, ws = [], []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2]))
                node = ((i0[:, 0] + dx) * dims[1] + (i0[:, 1] + dy)) * dims[2] + (i0[:, 2] + dz)
                corners.append(node)
                ws.append(w)
    node = torch.stack(corners, 1)                                             # [N, 8]
    w = torch.stack(ws, 1)                                                     # [N, 8]
    Phi = torch.zeros(nn, dtype=x.dtype, device=x.device).index_add_(0, node.reshape(-1), w.reshape(-1))          # all cells per node
    key = pid[:, None] * nn + node                                             # (cell, node)
    uk, inv = torch.unique(key.reshape(-1), return_inverse=True)
    phi = torch.zeros(uk.shape[0], dtype=x.dtype, device=x.device).index_add_(0, inv, w.reshape(-1))              # own cell per node
    own = (w * phi[inv].reshape(w.shape)).sum(1)
    tot = (w * Phi[node]).sum(1)
    frac_other = ((tot - own) / tot.clamp(min=1e-9)).clamp(0.0, 1.0)
    s0 = 1.0 / (1.0 + math.exp(float(thresh) / float(width)))                # the sigmoid at frac 0, taken out so a lone cell reads 0
    return ((torch.sigmoid((frac_other - float(thresh)) / float(width)) - s0) / (1.0 - s0)).mean()


@register_live("void_fraction")
def void_fraction(H, points="pt", h=0.4, lo=10.0, hi=90.0, thr=0.25, width=0.1, origin="data", **_):
    """HOLES IN THE SHEET: the share of the shell's volume that is empty. The shell is the band between the `lo`-th and `hi`-th percentile of the
    live points' distance from their centroid (detached); the points deposit trilinearly on a lattice of spacing h (twice the point spacing);
    a node of the band is empty when its density falls below `thr` of the median density of the occupied band nodes (detached), counted by a
    sigmoid of width `width` x that median, so the share is differentiable in the point positions. The vertex twin's sheet is confluent -- 0 by
    construction; the MPM vesicle starts at 0.008 and ends at 0.12-0.21 (the frozen control, the adhered fits) or 0.45 (the diameter-only fit)
    over 550 frames (offline, 2026-10-25). exp33: the adhesion measures (`soft_degree`, `contact_fraction`) read whether points touch SOME other
    cell, which a vesicle with holes keeps near its maximum; they could not see the holes the movie shows (Cedric, 2026-10-25)."""
    P, plive = _live(H, points)
    x = P.get("pos")[plive]
    c0 = x.detach().mean(0)
    r = (x.detach() - c0).norm(dim=1)
    rlo, rhi = torch.quantile(r, float(lo) / 100.0), torch.quantile(r, float(hi) / 100.0)
    org = lattice_origin(x, h, origin)
    g = (x - org) / h
    i0 = torch.floor(g).long()
    f = g - i0.to(x.dtype)
    dims = [int(v) for v in (i0.detach().max(0).values + 3).tolist()]
    nn = dims[0] * dims[1] * dims[2]
    dens = torch.zeros(nn, dtype=x.dtype, device=x.device)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2]))
                node = ((i0[:, 0] + dx) * dims[1] + (i0[:, 1] + dy)) * dims[2] + (i0[:, 2] + dz)
                dens = dens.index_add(0, node, w)
    ax = [torch.arange(n, device=x.device, dtype=x.dtype) for n in dims]
    X = torch.stack(torch.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3) * h + org
    rr = (X - c0).norm(dim=1)
    band = (rr >= rlo) & (rr <= rhi)
    d = dens[band]
    ref = d.detach()[d.detach() > 0].median()
    return torch.sigmoid((float(thr) * ref - d) / (float(width) * ref)).mean()


@register_live("cell_height")
def cell_height(H, set="cell", points="pt", **_):
    """CELL HEIGHT, apical to basal: the mean over live cells of the radial extent of the cell's points, from the radial coordinate's spread
    inside the cell (distance from the vesicle's centroid, centre detached): a uniform extent L has standard deviation L / sqrt(12), so
    L = 2 sqrt(3) std. exp33 (2026-10-25, Cedric: the vertex model's cell height is the apical-basal distance, which the MPM cell's thickness
    corresponds to): tracked against the vertex curve of V0f / A0 (tools/exp33_reference.py). Differentiable in the point positions."""
    C, live = _live(H, set)
    P, plive = _live(H, points)
    par = P.parent.long()[plive]
    x = P.get("pos")[plive]
    nC = C.state.shape[0]
    rad = (x - x.detach().mean(0)).norm(dim=1)
    one = torch.ones_like(rad)
    n = torch.zeros(nC, dtype=x.dtype, device=x.device).index_add_(0, par, one)
    m = torch.zeros(nC, dtype=x.dtype, device=x.device).index_add_(0, par, rad) / n.clamp(min=1.0)
    v = torch.zeros(nC, dtype=x.dtype, device=x.device).index_add_(0, par, (rad - m[par]) ** 2) / n.clamp(min=1.0)
    has = live & (n > 3)
    return (2.0 * math.sqrt(3.0) * torch.sqrt(v[has] + 1e-12)).mean()



def _two_types(L, live, axis, types):
    """The positions along `axis` of the live points of the two types `types` = (a, b) of a set -> (xa, xb)."""
    x = L.get("pos")[live][:, int(axis)]
    t = L.node_type[live]
    return x[t == int(types[0])], x[t == int(types[1])]


def _facing_gap(xa, xb, ring, cone, reach, tau, edge):
    """(sum of the local gaps of body a's front surface points to body b, the number of points that face b): see `plexus.geometry.facing_gaps`."""
    from plexus.geometry import facing_gaps
    _, g, has, _ = facing_gaps(xa, xb, ring=ring, cone=cone, reach=reach, tau=tau, edge=edge)
    return (g * has).sum(), has.sum()


@register_live("interface_gap", dim="length")
def interface_gap(H, set="pt", types=(0, 1), by="types", ring=2.0, cone=3.0, reach=12.0, tau=0.00005, edge=0.25, **_):
    """INTERFACE GAP between two bodies a and b, measured LOCALLY (Cedric, 2026-11-05): at every surface point of a body, along its own outward normal, the distance to the other body's points inside a narrow cone, averaged over the
    surface points that face the other body, from a to b and from b to a. Rotation-invariant, and a tilt of one body against the other does not read as a gap or an overlap. The normal of a point is the direction away from the mean of its neighbours
    within `ring` point spacings in its own body; `edge` is the offset of that mean, as a fraction of the point spacing, above which a point counts as on the surface; `by` says whether `types` are node types or parent cells; `cone` and `reach` are in point spacings, `tau` is the soft-minimum length.
    The soft minimum reads a little low (by about tau x ln of the number of points within tau of the nearest). Differentiable in the point positions."""
    L, live = _live(H, set)
    x = L.get("pos")[live]; t = (L.parent if by == "parents" else L.node_type)[live]
    xa, xb = x[t == int(types[0])], x[t == int(types[1])]
    sa, na = _facing_gap(xa, xb, float(ring), float(cone), float(reach), float(tau), float(edge))
    sb, nb = _facing_gap(xb, xa, float(ring), float(cone), float(reach), float(tau), float(edge))
    return (sa + sb) / (na + nb).clamp(min=1)


@register_live("interface_layer_gap", dim="length")
def interface_layer_gap(H, set="pt", types=(0, 1), by="types", q=0.01, **_):
    """SIGNED LAYER GAP between two bodies a and b along the line between their centres (from the current frame): the 1st percentile of b's positions along it minus the 99th percentile of a's (`q`, the fraction of points beyond each),
    so it is the distance between the two front layers, positive apart, negative when they overlap, and it stays defined and differentiable while the bodies interpenetrate (the local `interface_gap` finds no point ahead then)."""
    L, live = _live(H, set)
    x = L.get("pos")[live]; t = (L.parent if by == "parents" else L.node_type)[live]
    xa, xb = x[t == int(types[0])], x[t == int(types[1])]
    u = (xb.mean(0) - xa.mean(0)).detach(); u = u / u.norm().clamp(min=1e-12)
    return torch.quantile(xb @ u, float(q)) - torch.quantile(xa @ u, 1.0 - float(q))


@register_live("interface_mixing")
def interface_mixing(H, set="pt", types=(0, 1), by="types", width=0.0005, **_):
    """MIXING of two bodies a and b along the line between their centres (taken from the current frame, so a pair that turns together keeps its axis): the fraction of (a, b) point pairs in the wrong order (a beyond b) along it,
    as a smooth step of width `width` (a length), times 2, so 0 = sorted, about 1 = fully shuffled (half the pairs inverted), 2 = the bodies have swapped places. Differentiable in the point positions."""
    L, live = _live(H, set)
    x = L.get("pos")[live]; t = (L.parent if by == "parents" else L.node_type)[live]
    xa, xb = x[t == int(types[0])], x[t == int(types[1])]
    n = (xb.mean(0) - xa.mean(0)).detach(); n = n / n.norm().clamp(min=1e-12)
    return 2.0 * torch.sigmoid(((xa @ n)[:, None] - (xb @ n)[None, :]) / width).mean()
