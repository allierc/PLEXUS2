"""Half-edge ALGORITHMS on the closed 3D surface: rings, edge split, face division.

Moved out of `discovery_okuda/ops/topology_ops.py`, and moved to `models/` rather than `operators/`
because none of it is an operator: these are pure functions over the flat table that
`plexus.models.mesh.MeshTable` holds, and `cell_divide` / `edge_flip` are the operators that drive
them. Keeping them beside the table is what makes the table's central invariant checkable in one
place -- `rings_from_flat_3d` walks `E_face` IN TABLE ORDER and never sorts, so the ordering of the
flat table is the geometry.
"""
from __future__ import annotations
import numpy as np


# rings_from_flat_3d: Flat half-edge table -> per-face vertex rings, in table order -> list of rings.
def rings_from_flat_3d(es, et, ef, nF):
    """Flat half-edge table -> list of per-face vertex rings (ordered). build_sphere_mesh emits each
    face's half-edges contiguously in ring order, and divide preserves that, so grouping src vertices
    by face recovers the CCW ring."""
    rings = [[] for _ in range(nF)]
    for k in range(len(ef)):
        rings[int(ef[k])].append(int(es[k]))
    return rings


# flat_from_rings_3d: Per-face rings -> compact flat half-edge table and the new_face -> old_face map.
def flat_from_rings_3d(rings):
    """Compact (drop dead faces), renumber, and emit the flat half-edge table + the surviving-face
    index map (new_face -> old_face) so per-face targets can be carried over."""
    es, et, ef, keep = [], [], [], []
    for f, r in enumerate(rings):
        if r is None or len(r) < 3:
            continue
        nf = len(keep); keep.append(f)
        k = len(r)
        for i in range(k):
            es.append(r[i]); et.append(r[(i + 1) % k]); ef.append(nf)
    return (np.array(es, np.int64), np.array(et, np.int64), np.array(ef, np.int64),
            len(keep), np.array(keep, np.int64))


# _edge_face_map: Directed edge (u, v) -> the face that traverses it -> dict.
def _edge_face_map(rings):
    """Directed edge (u,v) -> face that traverses it. On a closed orientable surface the opposite
    directed edge (v,u) belongs to the one neighbouring face."""
    m = {}
    for f, r in enumerate(rings):
        if r is None or len(r) < 3:
            continue
        k = len(r)
        for i in range(k):
            m[(r[i], r[(i + 1) % k])] = f
    return m


# _emap_drop: Remove a ring's directed edges from the edge map, only entries still owned by that face.
def _emap_drop(emap, f, ring):
    """Remove `ring`'s directed edges from `emap`, but only the entries still owned by face `f`.

    The ownership test matters: a division inserts a midpoint into a NEIGHBOUR's ring, so by the time
    that neighbour is refreshed some of its old keys may already have been re-registered to another
    face. Deleting blindly would drop a live entry."""
    if not ring or len(ring) < 3:
        return
    k = len(ring)
    for i in range(k):
        key = (ring[i], ring[(i + 1) % k])
        if emap.get(key) == f:
            del emap[key]


# _emap_add: Register a face's directed edges in the edge map.
def _emap_add(emap, f, ring):
    """Register face `f`'s directed edges in `emap`."""
    if not ring or len(ring) < 3:
        return
    k = len(ring)
    for i in range(k):
        emap[(ring[i], ring[(i + 1) % k])] = f


# _insert_after: Insert vertex w between consecutive u, v of a ring, in place -> bool.
def _insert_after(ring, u, v, w):
    """Insert vertex w between consecutive u,v in ring (in place). Returns True on success."""
    k = len(ring)
    for i in range(k):
        if ring[i] == u and ring[(i + 1) % k] == v:
            ring.insert(i + 1, w)
            return True
    return False


# divide_face_3d: Divide a face by an edge-midpoint septum -> (idx_daughterB, m1, m2), or None.
def divide_face_3d(rings, pos, f, project=True, ea=None, eb=None, emap=None, births=None):
    """Divide face f by an edge-midpoint septum across edges `ea` and `eb` (pass the two edges the cell's SHORT axis crosses: the septum runs perpendicular to the long axis, Hertwig's rule, giving compact daughters; default roughly opposite edges). Mutates `rings` (f -> two daughters, neighbours gain the midpoint vertices) and appends the two new vertex positions to `pos`. Returns (idx_daughterB, m1, m2): daughter A stays at index f, B is appended; None if it cannot.

    `emap`: the edge -> face map, passed in and maintained instead of rebuilt per division (the O(E) rebuild was 80% of `cell_divide`: 19.4 of 24.2 s at frame 380 of `mesh_mpm_spheroid_nominal`, the term behind 246 ms/frame becoming 85 s/frame). A dict is updated in place for the four faces a division touches (mother, new daughter, two neighbours); None rebuilds.

    `births`: a list the vertex parentage is appended to, `(m1, (a0, a1))` and `(m2, (b0, b1))` (each new vertex is the midpoint of a named edge), so a per-vertex quantity has a value at a vertex born mid-run. None records nothing."""
    r = rings[f]
    k = len(r)
    if k < 4:                                   # need >=4 edges to split two non-adjacent ones cleanly
        return None
    if ea is None:
        ea, eb = 0, k // 2
    i0, i1 = sorted((ea % k, eb % k))
    if i1 - i0 < 2 or (i0 + k - i1) < 2:        # adjacent edges -> a daughter would be degenerate; use opposite
        i0, i1 = 0, k // 2
    a0, a1 = r[i0], r[(i0 + 1) % k]
    b0, b1 = r[i1], r[(i1 + 1) % k]
    own_map = emap is None
    if own_map:
        emap = _edge_face_map(rings)
    nbrA = emap.get((a1, a0)); nbrB = emap.get((b1, b0))
    if nbrA is None or nbrB is None or nbrA == f or nbrB == f or nbrA == nbrB:
        return None                             # both split edges must have distinct interior neighbours

    m1 = len(pos); m2 = len(pos) + 1
    p_a = 0.5 * (np.asarray(pos[a0]) + np.asarray(pos[a1]))
    p_b = 0.5 * (np.asarray(pos[b0]) + np.asarray(pos[b1]))
    if project:                                 # push the midpoint back out to the local shell radius
        ra = 0.5 * (np.linalg.norm(pos[a0]) + np.linalg.norm(pos[a1]))
        rb = 0.5 * (np.linalg.norm(pos[b0]) + np.linalg.norm(pos[b1]))
        p_a = p_a * (ra / max(np.linalg.norm(p_a), 1e-9))
        p_b = p_b * (rb / max(np.linalg.norm(p_b), 1e-9))
    pos.append(p_a); pos.append(p_b)
    if births is not None:                      # (new vertex, the two endpoints it was born between)
        births.append((m1, (int(a0), int(a1))))
        births.append((m2, (int(b0), int(b1))))

    # the neighbours' OLD rings, taken before `_insert_after` mutates them in place (`_emap_drop` needs the keys as they were)
    old_A = list(rings[nbrA]); old_B = list(rings[nbrB])

    # split the two shared edges inside the neighbouring faces (keeps every edge shared by two faces)
    if not _insert_after(rings[nbrA], a1, a0, m1):
        return None
    if not _insert_after(rings[nbrB], b1, b0, m2):
        # nbrA has already been mutated on this path, so a maintained map would describe a ring that no longer exists: repair it here, not at the caller
        if not own_map:
            _emap_drop(emap, nbrA, old_A); _emap_add(emap, nbrA, rings[nbrA])
        return None

    # daughter A: m1 -> (r[i0+1 .. i1]) -> m2, closed by the septum m2->m1
    seg_A = [m1] + [r[j % k] for j in range(i0 + 1, i1 + 1)] + [m2]
    # daughter B: m2 -> (r[i1+1 .. i0]) -> m1, closed by the septum m1->m2
    seg_B = [m2] + [r[j % k] for j in range(i1 + 1, i0 + 1 + k)] + [m1]
    rings[f] = seg_A
    rings.append(seg_B)
    if not own_map:
        # four faces changed and nothing else: the mother (now daughter A), the appended daughter B, and the two neighbours that gained a midpoint; drop first, then add, so an edge that moves between them ends up owned by the face that has it now
        _emap_drop(emap, f, r)
        _emap_drop(emap, nbrA, old_A)
        _emap_drop(emap, nbrB, old_B)
        _emap_add(emap, f, seg_A)
        _emap_add(emap, len(rings) - 1, seg_B)
        _emap_add(emap, nbrA, rings[nbrA])
        _emap_add(emap, nbrB, rings[nbrB])
    return len(rings) - 1, m1, m2


# face_collapse_3d: T2 / cell extrusion: collapse a triangular face to a point, so the sheet closes over it -> bool.
def face_collapse_3d(rings, pos, f, births=None):
    """T2 / cell extrusion: collapse triangular face `f` to a point, so the sheet closes over it. The inverse of `divide_face_3d` (the Die family of the algebra).

    Sound because collapsing a TRIANGLE of a closed trivalent surface removes 2 vertices, 3 edges and 1 face, so chi = V - E + F is unchanged and the sheet stays closed at genus 0; a face with more sides would merge k > 3 vertices into a rosette the trivalent mechanics cannot represent. So the caller sheds neighbours first (`edge_flip` walks a shrinking cell down to a triangle) and this refuses anything else.

    Mutates `rings` (f retired, every reference to its vertices rewired) and `pos` (the survivor moved to the centroid). Returns True if the collapse happened AND left a valid surface; on failure `rings` and `pos` are untouched (a half-applied topology edit is worse than a refused one).

    `births`: the same out-parameter as `divide_face_3d`, a merge being a birth with three parents (the survivor moves to the centroid, so a per-vertex quantity must travel the same way). Appended only on success, after the closure check."""
    r = rings[f]
    if r is None or len(r) != 3:
        return False
    keep, drop = int(r[0]), {int(v) for v in r[1:]}
    c = pos[list(r)].mean(0)
    # work on a copy and commit only if the surface is valid: a mesh that lost its closure does not announce itself (the genus check cannot see a surface folded through its own centre)
    trial = [None if rg is None else list(rg) for rg in rings]
    trial[f] = None
    for g, rg in enumerate(trial):
        if rg is None:
            continue
        ng = [keep if int(u) in drop else int(u) for u in rg]
        ded = [ng[i] for i in range(len(ng)) if ng[i] != ng[(i - 1) % len(ng)]]
        # a neighbour that shared TWO collapsed vertices loses a side; below three it is degenerate: refuse rather than silently delete a second cell
        if len(ded) < 3:
            return False
        trial[g] = ded
    # commit only if the surface is still the same surface: same Euler characteristic, same number of rim loops, still edge-manifold (a closed-sphere test, euler == 2, would refuse every collapse on an OPEN sheet, so a dying cell would shrink to nothing and buckle the sheet); a collapse that would pinch the rim or tear a hole is still refused
    ok0, _V, _E, _F, euler0, loops0 = _check_surface(rings)
    ok, _V, _E, _F, euler, loops = _check_surface(trial)
    if not (ok0 and ok) or euler != euler0 or loops != loops0:
        return False
    for g in range(len(rings)):
        rings[g] = None if trial[g] is None else np.asarray(trial[g], dtype=np.int64)
    pos[keep] = c
    if births is not None:                      # the survivor is reborn from all three of them
        births.append((keep, (keep,) + tuple(sorted(drop))))
    return True


# _check_surface: Validate an orientable surface with or without a rim -> (ok, V, E, F, euler, n_rim_loops).
def _check_surface(rings):
    """Validate an orientable surface WITH OR WITHOUT A RIM: every directed edge appears at most
    once, no duplicate vertices within a ring. Rim edges (a directed edge with no opposite) are
    allowed and counted; the rim is followed into loops. Returns (ok, V, E, F, euler, n_rim_loops)
    with E counting each undirected edge once, so euler = V - E + F is 2 for a closed sphere and 1
    for a disc. `_check_closed` is this test plus "no rim edge at all"."""
    seen = {}
    verts = set()
    F = 0
    for r in rings:
        if r is None or len(r) < 3:
            continue
        F += 1
        if len(set(r)) != len(r):
            return False, 0, 0, 0, 0, 0
        k = len(r)
        for i in range(k):
            e = (int(r[i]), int(r[(i + 1) % k]))
            if e in seen:
                return False, 0, 0, 0, 0, 0             # a directed edge used twice -> non-manifold
            seen[e] = True
            verts.add(e[0])
    rim = {}                                             # rim edge, oriented the way the face walks it
    for (u, v) in seen:
        if (v, u) not in seen:
            if u in rim:
                return False, 0, 0, 0, 0, 0             # two rim edges leave one vertex -> pinched rim
            rim[u] = v
    loops = 0
    left = set(rim)
    while left:
        loops += 1
        u = next(iter(left))
        while u in left:
            left.discard(u); u = rim[u]
    V = len(verts); E = (len(seen) + len(rim)) // 2
    return True, V, E, F, V - E + F, loops


# _check_closed: Validate a closed orientable surface -> (ok, V, E, F, euler).
def _check_closed(rings):
    """Validate a closed orientable surface: every directed edge appears once, its opposite exists,
    no duplicate vertices within a ring. Returns (ok, V, E, F, euler)."""
    seen = {}
    verts = set()
    F = 0
    for r in rings:
        if r is None or len(r) < 3:
            continue
        F += 1
        if len(set(r)) != len(r):
            return False, 0, 0, 0, 0
        k = len(r)
        for i in range(k):
            e = (r[i], r[(i + 1) % k])
            if e in seen:
                return False, 0, 0, 0, 0            # a directed edge used twice -> non-manifold
            seen[e] = True
            verts.add(r[i])
    for (u, v) in seen:
        if (v, u) not in seen:
            return False, 0, 0, 0, 0                # boundary edge -> not closed
    V = len(verts); E = len(seen) // 2
    return True, V, E, F, V - E + F


if __name__ == "__main__":                          # standalone soundness test (no engine)
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mesh_ops import build_sphere_mesh

    verts, es, et, ef, nF = build_sphere_mesh(120, 5.0, 0.15, 0)
    rings = rings_from_flat_3d(es, et, ef, nF)
    pos = [v for v in verts]
    ok, V, E, F, eu = _check_closed(rings)
    print(f"start: closed={ok} V={V} E={E} F={F} euler={eu}")

    rng = np.random.default_rng(0)
    ndone = 0
    for _ in range(40):
        f = int(rng.integers(0, len(rings)))
        if rings[f] is None or len(rings[f]) < 4:
            continue
        res = divide_face_3d(rings, pos, f)
        if res is not None:
            ndone += 1
    ok, V, E, F, eu = _check_closed(rings)
    print(f"after {ndone} divisions: closed={ok} V={V} E={E} F={F} euler={eu}  (want euler=2)")
    es2, et2, ef2, nF2, keep = flat_from_rings_3d(rings)
    print(f"rebuilt flat: nF={nF2} half-edges={len(es2)} (each real edge twice -> {len(es2)//2} edges)")
