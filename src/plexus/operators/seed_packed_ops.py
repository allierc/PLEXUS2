"""`seed_positions[packed]` -- A CROWD OF MOLECULES PACKED AT GENERATE TIME (exp09 / exp10, 2026-11-20).

exp09 and exp10 seeded their bodies from a table (`seed_positions[from_points_ragged]`) that `tools/exp09_landscape.py write` packed and saved beside the spec:
95 MB for exp10's 11,672 bodies, too large to ship, so a fresh PLEXUS2 clone could not run `generate` without first running the tool (Cedric, 2026-11-20: "I do not
get why the seed is not part of generate, I do not want an extra tool here, create a seed operator"). Here the spec carries what the table was made of -- each
kind's shape and point count, how many bodies of each kind, the chains, the box, the gap, the random seed -- and the operator packs the bodies when `generate`
runs: the same random sequential placement, the same table, then seeded as from_points_ragged_batched seeds it.

The packing functions are the ONE copy: tools/exp09_landscape.py imports them from here (its membrane, vesicle and coarse scenes still write a table with them).

    body_points    a body's points about its centre: the n lattice points that score lowest in its shape (ellipsoid, lobes, lobes_wide, torus, stack, block)
    chain_groups   rows of block subunits one lattice spacing apart, straight or meandering, with riders seated on them
    pack           random sequential placement: largest first, each body at a random place and turn, kept when it is `gap` clear of every point placed
    table_of       the placed bodies as the seed table's arrays: positions in micrometres (NaN rows padding to `slots`), volumes, and the per-body cell columns
"""
from __future__ import annotations

import math

import numpy as np

from plexus.models.registry import register_operator
from plexus.operators.adhesion_ops import SeedFromPointsRaggedBatched

SPACING = 0.2                  # the lattice of a body's points, length units: one point per 2 nm in exp09 (10 nm per unit), per 1 nm in exp10 (5 nm per unit)


def body_points(shape, n, spacing=SPACING):
    """[n, 3] points of one body about its own centroid, length units: the `n` points of a lattice of spacing `spacing` (SPACING) with the lowest score of the shape.
    shape = (kind, par): ("ellipsoid", (a, b, c)) the points nearest the centre in the metric of semi-axes a : b : c; ("lobes", [((x, y, z), r), ...]) the points
    nearest any lobe centre, each distance divided by its lobe's radius r (centres and radii in units of the first lobe's radius), on a lattice 19 points wide;
    "lobes_wide" the same on a lattice as wide as the lobes reach; ("block", (nx, ny, nz)) a full lattice block; ("torus", (R, r)) the points nearest a circle in the
    xy plane, centre-line radius R and tube radius r (their ratio); ("stack", (rho, ...)) layers along y from the top down, each the points within rho spacings of
    the body's axis. A block and a stack are always on SPACING."""
    kind, par = shape
    PV = spacing ** 3                                                                              # this lattice's point volume
    g = np.arange(-9, 10) * spacing
    lat = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    if kind == "ellipsoid":
        score = np.linalg.norm(lat / np.asarray(par, float)[None], axis=1)
    elif kind in ("lobes", "lobes_wide"):                                                          # lobes_wide: the same on a lattice as wide as the lobes reach (a body over 19 points long; exp10's Tudor)
        cen = np.array([c for c, _ in par], float)
        rad = np.array([r for _, r in par], float)
        cen = cen - (cen * rad[:, None] ** 3).sum(0) / (rad ** 3).sum()                         # the lobes' volume-weighted centre at the lattice's origin
        s = (n * PV / (4.0 / 3.0 * np.pi * (rad ** 3).sum())) ** (1.0 / 3.0)                    # the first lobe's radius if the lobes did not overlap; they do, so the n points fill them a little further out
        if kind == "lobes_wide":
            half = int(np.ceil(float((np.abs(s * cen) + s * rad[:, None]).max()) / spacing)) + 2
            g = np.arange(-half, half + 1) * spacing
            lat = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        score = (np.linalg.norm(lat[:, None] - s * cen[None], axis=2) / (s * rad[None])).min(1)
    elif kind == "torus":
        R, r = (float(v) for v in par)
        s = (n * PV / (2.0 * np.pi ** 2 * R * r * r)) ** (1.0 / 3.0)                               # the tube radius at which the ring's volume 2 pi^2 R r^2 holds the n points
        score = np.hypot(np.hypot(lat[:, 0], lat[:, 1]) - s * R, lat[:, 2]) / (s * r)
    elif kind == "stack":
        g = np.arange(-5, 6)
        ij = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        P = np.concatenate([np.column_stack([q[:, 0] * SPACING, np.full(len(q), -k * SPACING), q[:, 1] * SPACING])
                            for k, rho in enumerate(par) for q in [ij[(ij ** 2).sum(1) <= float(rho) ** 2 + 1e-9]]])
        if len(P) != n:
            raise ValueError(f"a stack of {par} has {len(P)} points, not {n}")
        return P - P.mean(0)
    elif kind == "block":                                                                          # a full lattice block of nx x ny x nz points
        ax = [(np.arange(k) - 0.5 * (k - 1)) * SPACING for k in par]
        P = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
        if len(P) != n:
            raise ValueError(f"a block of {par} points has {len(P)} points, not {n}")
        return P
    else:
        raise ValueError(f"unknown shape {kind!r}")
    P = lat[np.argsort(score, kind="stable")[:n]]
    return P - P.mean(0)


def _riders(grp, ch, species):
    """`ch['riders']` = dict(kind, every): a body of species `kind` seated on one subunit in `every` of the row `grp`, on alternate sides in the row's plane, its
    lowest layer of points one spacing from the subunit's outer layer and on the subunit's own lattice, so that static bonds tie the two (a ribosome on a thread:
    the pair then moves as one). A rider that would come within 0.7 of a spacing of another body of the row is left out. Returns the row with its riders appended."""
    rd = ch.get("riders")
    if not rd:
        return grp
    dims = species[ch["kind"]]["shape"][1]
    Q0 = body_points(species[rd["kind"]]["shape"], species[rd["kind"]]["n"])
    cen = np.array([P.mean(0) for _, P in grp])
    out, side = list(grp), 1.0
    for k in range(int(rd["every"]) // 2, len(grp), int(rd["every"])):
        t = cen[min(k + 1, len(grp) - 1)] - cen[max(k - 1, 0)]
        t = t / np.linalg.norm(t)
        nrm = side * np.array([-t[1], t[0], 0.0])
        Q = Q0.copy()
        for ax, nd in ((0, dims[0]), (2, dims[2])):                    # along the row and along z: on the subunit's lattice (its points sit at multiples of the spacing, or half-way for an even count)
            want = 0.0 if nd % 2 else 0.5 * SPACING
            Q[:, ax] += want - (Q[0, ax] - round((Q[0, ax] - want) / SPACING) * SPACING)
        Q[:, 1] += 0.5 * (dims[1] - 1) * SPACING + SPACING - Q[:, 1].min()          # its lowest layer one spacing outside the subunit's outer layer
        X = cen[k] + Q[:, :1] * t + Q[:, 1:2] * nrm + Q[:, 2:3] * np.array([0.0, 0.0, 1.0])
        others = np.concatenate([P for j, (_, P) in enumerate(out) if j != k])
        if np.linalg.norm(X[:, None] - others[None], axis=2).min() < 0.7 * SPACING:          # 0.7: on a turning thread the next element's corner is already 0.17 from its neighbour
            continue
        out.append((rd["kind"], X))
        side = -side
    return out


def chain_groups(chains, seed, species):
    """Chains of bodies for `pack` to place, each as one rigid piece: for every dict(kind, n, m, bend) of `chains`, `n` rows of `m` bodies of the block species `kind`
    (its entry in `species`: dict(n = points, shape)), each one lattice spacing from the next, so a row is continuous material cut into bodies.
    `bend` 0 (the default): a straight row along x on ONE lattice -- a stiff filament. `bend` > 0: a meander -- from one subunit to the next the row's heading in the
    xy plane turns by a random angle of standard deviation `bend`, radians (cut at 1.5 x bend), each subunit turned to face along the row, and the step widened by half
    a point spacing x that angle: at the largest turn of bend 0.2 (0.3 rad) the two facing layers of points are 0.17 apart on the inside of the turn and 0.29 on the
    outside (0.2 on a straight row). A row that comes back within a block and a half of itself is drawn again, and so is one taller than 0.8 of its `band`; the row
    is returned turned so that it runs along x from its first subunit to its last. Returns [[(species, points)]]."""
    rng = np.random.default_rng(int(seed) + 7919)
    out = []
    for ch in chains:
        kind, m, bend = ch["kind"], int(ch["m"]), float(ch.get("bend") or 0.0)
        sub = body_points(species[kind]["shape"], species[kind]["n"])
        w = species[kind]["shape"][1][0] * SPACING
        for _ in range(int(ch["n"])):
            if bend <= 0.0:
                out.append(_riders([(kind, sub + np.array([(k - 0.5 * (m - 1)) * w, 0.0, 0.0])) for k in range(m)], ch, species))
                continue
            for _try in range(3000):
                phi, c, grp = rng.uniform(-0.5, 0.5) * bend * math.sqrt(m), np.zeros(3), []
                for k in range(m):
                    if k:
                        d = float(np.clip(rng.normal(0.0, bend), -1.5 * bend, 1.5 * bend))
                        step = w + 0.5 * SPACING * abs(d)
                        c = c + 0.5 * step * np.array([math.cos(phi), math.sin(phi), 0.0])
                        phi += d
                        c = c + 0.5 * step * np.array([math.cos(phi), math.sin(phi), 0.0])
                    R = np.array([[math.cos(phi), -math.sin(phi), 0.0], [math.sin(phi), math.cos(phi), 0.0], [0.0, 0.0, 1.0]])
                    grp.append((kind, c + sub @ R.T))
                cen = np.array([P.mean(0) for _, P in grp])
                far = np.abs(np.arange(m)[:, None] - np.arange(m)[None]) >= 3                      # two subunits at least three apart along the row
                if (np.linalg.norm(cen[:, None] - cen[None], axis=2)[far] < 1.5 * w).any():        # ... closer than a block and a half: the row has come back on itself
                    continue
                a = -math.atan2(cen[-1, 1] - cen[0, 1], cen[-1, 0] - cen[0, 0])                   # turned so that the row runs along x from its first subunit to its last
                T = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
                grp = [(nm, P @ T.T) for nm, P in grp]
                if ch.get("band") and np.ptp(np.concatenate([P for _, P in grp])[:, 1]) + 2.0 * SPACING > 0.8 * (float(ch["band"][1]) - float(ch["band"][0])):
                    continue                                                                       # a meander that could not lie inside its band: drawn again
                break
            else:
                raise ValueError(f"chain_groups: no meander of {m} `{kind}` at bend {bend} fits {ch.get('band') or 'clear of itself'} in 3,000 draws; lower `bend` or `m`")
            grp = _riders(grp, ch, species)
            mid = np.concatenate([P for _, P in grp]).mean(0)
            out.append([(nm, P - mid) for nm, P in grp])
    return out


def chain_pieces(chains, lo, hi, seed, species):
    """A scene's chains as `pack` takes them: a chain with `rows` = [(height above the box's floor, depth), ...] is straight along x, centred in x, one at each of those
    places, and placed as given (fixed); every other chain is a group for pack to place, with its `band` and `tilt`. Returns (fixed, groups, bands, tilts)."""
    fixed = []
    for ch in chains:
        for (y, zz), grp in zip(ch.get("rows") or [], chain_groups([dict(ch, n=len(ch.get("rows") or []), bend=0.0)], seed, species)):
            fixed += [(nm, np.array([0.5 * (lo[0] + hi[0]), lo[1] + float(y), lo[2] + float(zz)]) + P) for nm, P in grp]
    free = [ch for ch in chains if not ch.get("rows")]
    return (fixed, chain_groups(free, seed, species), [ch.get("band") for ch in free for _ in range(int(ch["n"]))],
            [ch.get("tilt") for ch in free for _ in range(int(ch["n"]))])


def _rotation(rng):
    """A rotation matrix drawn uniformly over all rotations (a uniform unit quaternion)."""
    w, x, y, z = (lambda q: q / np.linalg.norm(q))(rng.normal(size=4))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pack(counts, lo, hi, gap, seed, species=None, tries=4000, fixed=(), yrange=None, groups=(), where=None, group_band=None, group_tilt=None, shapes=None, spacing=SPACING):
    """Random sequential placement: every body at a random place and orientation inside the box [lo, hi], largest first, kept when none of its points is within `gap`
    of a point already placed (tested on an occupancy lattice of 0.1 length units). Each body of species `name` is body_points of `species[name]`, or `shapes[name]`
    when given. `fixed`: bodies placed beforehand as given (a membrane); `yrange`: {species: (y_lo, y_hi)}, the part of the box that species may lie in; `where`:
    {species: rule(centre, half extents) -> bool}, a further rule on where a body of that species may sit (inside or outside a vesicle); `groups`: chains of bodies
    placed after the fixed ones, each as ONE rigid piece turned about z by a random angle (filaments and threads, lying in the slab's plane), inside its species'
    `yrange` or, when `group_band[i]` = (y0, y1) is given for chain i, between those two heights above the box's floor; with `group_tilt[i]` the random angle is within
    that many radians of the x axis (either way round) instead of anywhere in the turn.
    Returns ([(species, points [n, 3] world)], {species: bodies that found no place}, how many of the bodies are fixed ones or chain subunits: the first of the list)."""
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    h = 0.1
    shape = np.ceil((hi - lo) / h).astype(int) + 1
    taken = np.zeros(shape, bool)
    r = int(math.ceil(gap / h))
    o = np.arange(-r, r + 1)
    off = np.stack(np.meshgrid(o, o, o, indexing="ij"), -1).reshape(-1, 3)
    off = off[np.linalg.norm(off, axis=1) * h <= gap + 1e-9]
    bp = shapes or {name: body_points(species[name]["shape"], species[name]["n"]) for name in counts}             # the same for every body of a species: drawn once
    todo = [(name, bp[name]) for name, c in counts.items() for _ in range(int(c))]
    order = sorted(range(len(todo)), key=lambda i: (-len(todo[i][1]), rng.random()))
    placed, failed = [], {}
    for name, X in fixed:
        st = np.clip((np.floor((X - lo) / h).astype(int)[:, None, :] + off[None]).reshape(-1, 3), 0, shape - 1)
        taken[st[:, 0], st[:, 1], st[:, 2]] = True
        placed.append((name, X))
    for gi, grp in enumerate(groups):
        glo, ghi = lo.copy(), hi.copy()
        if group_band and group_band[gi]:
            glo[1], ghi[1] = lo[1] + float(group_band[gi][0]), lo[1] + float(group_band[gi][1])
        elif yrange and grp[0][0] in yrange:
            glo[1], ghi[1] = max(lo[1], yrange[grp[0][0]][0]), min(hi[1], yrange[grp[0][0]][1])
        P0 = np.concatenate([P for _, P in grp])
        cuts = np.cumsum([len(P) for _, P in grp])[:-1]
        for _ in range(tries):
            ang = rng.uniform(0.0, 2.0 * math.pi) if not (group_tilt and group_tilt[gi] is not None) else rng.uniform(-1.0, 1.0) * float(group_tilt[gi]) + math.pi * int(rng.integers(2))
            P = P0 @ np.array([[math.cos(ang), -math.sin(ang), 0.0], [math.sin(ang), math.cos(ang), 0.0], [0.0, 0.0, 1.0]]).T
            e = np.abs(P).max(0) + 0.5 * spacing
            if np.any(ghi - e <= glo + e):
                continue
            c = rng.uniform(glo + e, ghi - e)
            if where and grp[0][0] in where and not where[grp[0][0]](c, e):
                continue
            X = c + P
            ijk = np.floor((X - lo) / h).astype(int)
            if taken[ijk[:, 0], ijk[:, 1], ijk[:, 2]].any():
                continue
            st = np.clip((ijk[:, None, :] + off[None]).reshape(-1, 3), 0, shape - 1)
            taken[st[:, 0], st[:, 1], st[:, 2]] = True
            placed += [(nm, Xk) for (nm, _), Xk in zip(grp, np.split(X, cuts))]
            break
        else:
            failed[grp[0][0] + " chain"] = failed.get(grp[0][0] + " chain", 0) + 1
    n_struct = len(placed)                                                                         # the membrane's bodies and the chains' subunits: what static bonds may tie
    for i in order:
        name, P0 = todo[i]
        blo, bhi = lo.copy(), hi.copy()
        if yrange and name in yrange:
            blo[1], bhi[1] = max(lo[1], yrange[name][0]), min(hi[1], yrange[name][1])
        for _ in range(tries):
            P = P0 @ _rotation(rng).T
            e = np.abs(P).max(0) + 0.5 * spacing
            if np.any(bhi - e <= blo + e):
                continue
            c = rng.uniform(blo + e, bhi - e)
            if where and name in where and not where[name](c, e):
                continue
            X = c + P
            ijk = np.floor((X - lo) / h).astype(int)
            if taken[ijk[:, 0], ijk[:, 1], ijk[:, 2]].any():
                continue
            st = np.clip((ijk[:, None, :] + off[None]).reshape(-1, 3), 0, shape - 1)
            taken[st[:, 0], st[:, 1], st[:, 2]] = True
            placed.append((name, X))
            break
        else:
            failed[name] = failed.get(name, 0) + 1
    return placed, failed, n_struct


def sites_of(k, chain_m, names, got):
    """Each body's site, as a cell index: for a subunit of a chain (its kind in `chain_m`, {kind: subunits per chain}; the chains of a kind one after the other, a
    chain's subunits in order), the first subunit of its run of `k`; for any other body, itself. Bodies in the table's order: kind by kind in `names`, `got[kind]` each.
    With k at least the longest chain, a subunit's site is its chain's first subunit: the molecule it belongs to."""
    k = int(k)
    out, base = [], 0
    for nm in names:
        n = int(got[nm])
        for q in range(n):
            if nm in chain_m:
                m = int(chain_m[nm])
                out.append(base + (q // m) * m + ((q % m) // k) * k)
            else:
                out.append(base + q)
        base += n
    return np.array(out, np.float64)


def table_of(placed, n_struct, order, slots, um_per_unit, point_volume, columns=None, unbonded=(), sites_k=None, molecule=False, chain_m=None):
    """The placed bodies as a seed table: bodies kind by kind in `order` (the cell set's type order; a kind of `order` with no body is skipped), each kind's in the order
    `pack` placed them. Returns (names, {kind: bodies}, {array name: array}) with
        pos          [bodies, slots, 3] micrometres (points x um_per_unit), NaN in the rows a body does not use
        volume_um3   [bodies] points x point_volume x um_per_unit^3
        value        [bodies] 1 for a fixed body or a chain subunit (what static bonds may tie) unless its kind is `unbonded`, 0 otherwise
        cell__adh    [bodies, 6] ones: bond_binding's ceiling per face class
        cell__<c>    [bodies] for every c of `columns` ({c: {kind: value}}, 0 for a kind not named): valence, grate, drive
        cell__site, cell__molecule   [bodies] sites_of with `sites_k`, and with a site as long as any chain, when asked."""
    names = [nm for nm in order if any(q[0] == nm for q in placed)]
    bodies = [X for nm in names for q, X in placed if q == nm]
    got = {nm: sum(1 for q, _ in placed if q == nm) for nm in names}
    kinds = [nm for nm in names for q, _ in placed if q == nm]
    pos = np.full((len(bodies), int(slots), 3), np.nan)
    for k, X in enumerate(bodies):
        pos[k, :len(X)] = X * um_per_unit
    out = dict(pos=pos, volume_um3=np.array([len(X) * point_volume * um_per_unit ** 3 for X in bodies]),
               value=np.array([float(i < n_struct and nm not in (unbonded or ())) for nm in names for i, (q, _) in enumerate(placed) if q == nm]),
               cell__adh=np.ones((len(bodies), 6)))
    for c, per in (columns or {}).items():
        out["cell__" + c] = np.array([float((per or {}).get(nm, 0.0)) for nm in kinds])
    if sites_k:
        out["cell__site"] = sites_of(sites_k, chain_m or {}, names, got)
    if molecule:
        out["cell__molecule"] = sites_of(10 ** 9, chain_m or {}, names, got)                   # a chain's first subunit (a site of the whole chain); any other body, itself
    return names, got, out


class _Table(dict):
    """A seed table held in memory, read as from_points_ragged reads an npz (`files`, `table[name]`)."""

    @property
    def files(self):
        return list(self.keys())


@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="packed",
                   title="pack MPM bodies of given shapes at random into a box, then seed them",
                   equation=r"""$$\mathbf x_{k,j}=\mathbf c_k+R_k\,\mathbf q_{\tau_k,j},\qquad \min_{j,\,l\neq k,\,i}\lVert\mathbf x_{k,j}-\mathbf x_{l,i}\rVert\ge g$$""")
class SeedPacked(SeedFromPointsRaggedBatched):
    r"""Random sequential packing at generate time: the bodies of every kind (`kinds`: each kind's point count and shape), largest first, each put at a random centre and a random turn inside the box and kept when it is at least `gap` from every point placed before; the chains (`chains`) go in first, each as one rigid row of subunits turned about z. The packed bodies are then seeded as seed_positions[from_points_ragged_batched] seeds a table: the same table that tools/exp09_landscape.py used to write, made here from the spec.
    pt -> pt, cell: as from_points_ragged_batched; the per-body cell columns (`value`, `adh`, `valence`, `grate`, `drive`, `site`, `molecule`) are written into the cell blocks of those names that the cell set declares.

        \mathbf x_{k,j}=\mathbf c_k+R_k\,\mathbf q_{\tau_k,j},\qquad \min_{j,\,l\neq k,\,i}\lVert\mathbf x_{k,j}-\mathbf x_{l,i}\rVert\ge g
        x_{k,j}        the position of point j of body k, world length
        c_k            the body's centre, drawn uniformly where the whole body lies inside the box `box`
        R_k            its turn, drawn uniformly over all rotations (a chain: about z only, or within `tilt` of the x axis)
        q_{\tau,j}     point j of a body of kind tau about its centroid: the `points` lattice points of spacing `spacing` that score lowest in the kind's `shape`
        \tau_k         the body's kind
        g              `gap`: the least distance between the points of two bodies, tested on a lattice of 0.1 length units

    Reference: random sequential adsorption, Widom, B. (1966). J. Chem. Phys. 44:3888-3894; Plexus exp09 / exp10 (this work, 2026-11-20)."""
    EXPLAIN = (
        r"Builds a crowd of molecules when the run starts. Every body of every kind is a fixed set of lattice points in its kind's shape; the bodies are placed one "
        r"at a time, largest first, each at a random position and orientation, and kept only if no point comes closer than the gap $g$ to a point already placed. "
        r"Chains of subunits are placed first as rigid rows. The result is the seed table a tool used to write to disk, now made from the spec itself."
    )
    PARAM_UNITS = dict(SeedFromPointsRaggedBatched.PARAM_UNITS, box="length", gap="length", spacing="length")
    PARAM_UNITS.pop("points", None)
    SYMBOLS = {'x_{k,j}': 'the position of point j of body k, world length', 'c_k': "the body's centre, drawn uniformly where the whole body lies inside the box `box`",
               'R_k': 'its turn, drawn uniformly over all rotations (a chain: about z only, or within `tilt` of the x axis)',
               'q_{\\tau,j}': "point j of a body of kind tau about its centroid: the `points` lattice points of spacing `spacing` that score lowest in the kind's `shape`",
               '\\tau_k': "the body's kind", 'g': '`gap`: the least distance between the points of two bodies, tested on a lattice of 0.1 length units'}
    REQUIRES_PARAMS = ["cell_set", "kinds", "box", "slots"]
    PARAM_ROLES = dict({k: v for k, v in SeedFromPointsRaggedBatched.PARAM_ROLES.items() if k != "points"},
                       kinds="ordered_kinds_with_point_count_and_shape", counts="bodies_per_kind_placed_at_random", chains="rows_of_subunits_placed_first",
                       box="world_box_lo_hi_the_bodies_lie_in", gap="least_distance_between_two_bodies_points", seed="random_seed_of_the_packing",
                       slots="point_slots_per_body", spacing="lattice_spacing_of_a_bodys_points", unbonded="kinds_static_bonds_never_tie",
                       columns="per_kind_values_of_cell_blocks", sites="subunits_per_binding_site", molecule="write_each_cells_molecule")
    REFERENCE = "Widom, B. (1966). J. Chem. Phys. 44:3888-3894; Plexus exp09 / exp10 (this work, 2026-11-20)."

    def __init__(self, params, device="cpu"):
        super().__init__(dict(params, points="packed"), device)
        self.points = "the spec (packed at generate)"
        self.species = {str(k["name"]): dict(n=int(k["points"]), shape=(str(k["shape"][0]), k["shape"][1])) for k in params["kinds"]}
        self.order = [str(k["name"]) for k in params["kinds"]]
        self.counts = {str(k): int(v) for k, v in (params.get("counts") or {}).items()}
        self.chains = [dict(c) for c in (params.get("chains") or [])]
        self.lo, self.hi = (np.array([float(v) for v in b]) for b in params["box"])
        self.gap = float(params.get("gap", 0.3))
        self.seed = int(params.get("seed", 1))
        self.slots = int(params["slots"])
        self.spacing = float(params.get("spacing", SPACING))
        self.unbonded = tuple(params.get("unbonded") or ())
        self.columns = dict(params.get("columns") or {})
        self.sites = int(params["sites"]) if params.get("sites") else None             # subunits per binding site of a chain
        self.molecule = bool(params.get("molecule", False))
        for c in self.chains:
            if c["kind"] not in self.species:
                raise ValueError(f"seed_positions[packed]: a chain of `{c['kind']}`, a kind not in `kinds` {self.order}")
        for nm in self.counts:
            if nm not in self.species:
                raise ValueError(f"seed_positions[packed]: `counts` names `{nm}`, a kind not in `kinds` {self.order}")

    def packed(self):
        """(names, {kind: bodies placed}, the table): the chains first (chain_pieces), then the bodies placed at random, in the table's columns (table_of)."""
        fixed, groups, band, tilt = chain_pieces(self.chains, self.lo, self.hi, self.seed, self.species)
        placed, failed, n_struct = pack(self.counts, self.lo, self.hi, self.gap, self.seed, self.species, fixed=fixed, groups=groups, group_band=band, group_tilt=tilt)
        if failed:
            print(f"[seed_positions:packed] NO PLACE FOUND for {failed} (the box is full at this gap); they are left out", flush=True)
        return table_of(placed, n_struct, self.order, self.slots, self.u, self.spacing ** 3, columns=self.columns, unbonded=self.unbonded,
                        sites_k=self.sites, molecule=self.molecule,
                        chain_m={ch["kind"]: int(ch["m"]) for ch in self.chains})

    def _table(self, H):
        names, got, t = self.packed()
        C = H.level(self.cell_set)
        tn = list(getattr(C, "type_names", []) or [])
        want = {nm: int(v) for nm, v in zip(tn, np.bincount(C.node_type.detach().cpu().numpy()[C.occ.detach().cpu().numpy() > 0.5], minlength=len(tn)))} if tn else {}
        if tn and [n for n in tn if want.get(n)] != names:
            raise ValueError(f"seed_positions[packed]: the packed kinds {names} are not the cell set's types in order {tn}")
        if tn and any(want[nm] != got[nm] for nm in names):
            raise ValueError(f"seed_positions[packed]: packed {got}, the cell set holds {dict((n, want[n]) for n in names)}: the spec's type counts must be what the packing places")
        return _Table(t)
