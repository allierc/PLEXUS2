"""The 3D vertex model: an epithelium as a closed surface of polygonal cells, where the degrees
of freedom are the VERTICES and everything a cell does is a consequence of where they sit.

The state is a closed half-edge mesh. A cell is a FACE, a junction an EDGE, and the mechanics an
energy over faces and edges whose gradient moves the vertices. Because the representation is a
mesh rather than a lattice, the topology can change -- a cell divides by inserting a septum, dies
by contracting to nothing, and neighbours exchange by flipping an edge -- and every one of those
is an operator here rather than a special case in a solver.

In the order they appear below:

    seed_mesh        seed        the closed half-edge surface, built once at frame 0
    cell_mechanics   lateral     the shape energy, and the force that is its gradient
    cell_divide      structural  a septum through a face: one cell becomes two
    cell_die         structural  contract a cell to a point and remove it from the surface
    topo_record      structural  one recorded frame of topology per tick, so it can be rendered
    edge_flip        structural  the T1: a reversible neighbour exchange

then the models -- different hypotheses in one slot -- and the implementations, which change only
the arithmetic:

    seed_mesh[apicobasal]     the same epithelium seeded WITH its thickness
    cell_mechanics            models: monolayer, apicobasal, marinari
                              implementations: warp (default), autograd, compile
    cell_divide               models: doubler, timer
    cell_die                  models: competition, smaller, dimmer, older, crowded, lonely,
                              small, stalled, chem_low, prescribed, field

Reference: Okuda, S., Inoue, Y., Eiraku, M., Sasai, Y. & Adachi, T. (2013). Reversible network
reconnection model for simulating large deformation in dynamic tissue morphogenesis. Biomech.
Model. Mechanobiol. 12(4):627-644 -- the reconnection `edge_flip` implements; Okuda, S., Miura,
T., Inoue, Y., Adachi, T. & Eiraku, M. (2018). Combining Turing and 3D vertex models reproduces
autonomous multicellular morphogenesis of the tissue. Sci. Rep. 8:2386; the ancestor is Honda, H.,
Tanemura, M. & Nagai, T. (2004). A three-dimensional vertex dynamics cell model of space-filling
polyhedra simulating cell division. J. Theor. Biol. 226(4):439-453. The shape energy is
Farhadifar, R., Roper, J.-C., Aigouy, B., Eaton, S. & Julicher, F. (2007). The influence of cell
mechanics, cell-cell interactions, and proliferation on epithelial packing. Curr. Biol.
17:2095-2104. The mesh representation follows Tyssue.
"""
from __future__ import annotations
import contextlib
import math
import numpy as np
import torch
from scipy.spatial import SphericalVoronoi
from plexus.models.base import Lateral, Seed, Structural
from plexus.models.mesh import MeshTable, declare_vertex_carry
from plexus.models.registry import register_operator
from plexus.models.base import Rewire
from plexus.models.topology import (rings_from_flat_3d, flat_from_rings_3d,
                                    _edge_face_map, _check_closed)
from plexus.models.base import Lateral
# (was `from mesh_ops import face_geometry_3d`) -- same module now


MYO_SKIPPED: list = []


def _carry_face_state(m, keep, dt, dev):
    """Reindex any EXTRA per-face array an operator has declared, through the same `keep` map.

    `keep` maps new face -> old face, and `cell_divide` / `cell_die` already use it to carry A0, P0,
    V0f, Vbirth, divjit, age, ndiv and alive across a rebuild. That list was a literal tuple, so a
    per-face state introduced by a NEW operator was silently left indexed against faces that had
    moved -- the same defect class as per-half-edge myosin before `junction_sync`, one level up
    and with no vertex-pair key to recover from.

    `m["face_carry"]` makes the list open. An operator declares its own array once and the topology
    operators still know nothing about what is in it, which is the whole point: the alternative is
    every topology operator learning the name of every state, and the next state added edits them all
    again.

    NOTE ON SEMANTICS. `keep` COPIES the parent's value onto both daughters, so what is carried this
    way must be an INTENSIVE quantity -- a density, a concentration, an age. Carrying an extensive
    one (an amount, a mass) doubles it at every division. `medioapical_myosin` stores an areal density
    for exactly this reason.

    ONE IMPLEMENTATION, FOUR CALLERS. The body of this moved to `MeshTable.reindex_faces` so the
    carry is a property of the mesh rather than a helper each topology operator has to remember to
    call -- `edge_flip` did not, for its whole life, and dropped the medioapical myosin every time a
    flip lost a face. The semantics here are unchanged, clamp included.
    """
    if not m.get("face_carry"):
        return
    if hasattr(m, "reindex_faces"):
        m.reindex_faces(keep, dt=dt, dev=dev)
        return
    # A BARE DICT still reaches this: four operator self-tests build fake meshes as plain dicts and
    # must keep working, and every archived run predates the table.
    idx = torch.as_tensor(np.asarray(keep, np.int64), device=dev)
    for nm in sorted(m.get("face_carry") or ()):
        a = m.get(nm)
        if a is None:
            continue
        a = a if torch.is_tensor(a) else torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
        m[nm] = a.to(dev)[idx.clamp(max=max(a.shape[0] - 1, 0))].to(dt)


def _as_np(v, dtype=np.float64):
    """A per-cell array as numpy, whether it arrived as a tensor or was already an array."""
    if v is None:
        return None
    if hasattr(v, "detach"):
        return v.detach().cpu().numpy().astype(dtype)
    return np.asarray(v, dtype).copy()


def resolve_cell_set(H, at, override=None):
    """The set this mesh's faces ARE -- read from the mesh's own declaration, not from a parameter.

    DECLARED ONCE, ON THE SET. `sets.<vertex set>.cell_set:` is where the pairing lives, `schema.py`
    refuses a `mesh:` without it, and `engine._attach_mesh` records it as `Level.mesh_cell_set`.
    Every operator that crosses between the two then took the SAME pairing again as its own
    `cell_set:` parameter -- 67 lines across the 21 specs in `config/tissue`, all of them the word
    `cell` -- and defaulted it to the literal `"cell"` when absent. That is a relation stated in one
    place and re-stated in sixty-seven, and the schema's own message already says which is right:
    the pairing "must be declared once here rather than repeated as a parameter on every operator
    that crosses between them".

    TWO THINGS THE REPETITION COST, both already recorded in this tree. `edge_flip` fell back to the
    literal `"cell"` and so renumbered a set it never declared it needed. And `cell_divide`
    defaulted to `None` rather than to `"cell"` -- alone among the readers -- so on a divide line
    the parameter was LOAD-BEARING: omit it and the daughters silently stop inheriting their
    mother's cell state, with nothing to say so.

    `override` keeps a spec that still passes `cell_set:` working; no spec in `config/tissue` takes
    that path any more.
    """
    if override is not None:
        return override
    lvl = H.level(at) if at in getattr(H, "levels", {}) else None
    cs = getattr(lvl, "mesh_cell_set", None) if lvl is not None else None
    if cs is None:
        raise KeyError(
            f"no cell set for {at!r}: declare `cell_set:` on that set, beside its `mesh:`. A face "
            f"of the mesh IS a cell, and this operator crosses between the two.")
    return cs


def size_ref(m):
    """The reference volume a size rule compares against: the polyhedron median when the seed
    declared `v0_from: polyhedron` and one has been cached, the seed-time wedge median otherwise.
    Before c671fb31 every reader but `cell_divide` read the wedge one; the September 7 working
    points (cyc4_sizer, apop2_ks*) were made that way and are reproduced only that way."""
    if "v_ref_poly" in m:
        return float(m["v_ref_poly"])
    return float(m.get("v_ref", 1.0))


def settling(H, m):
    """True while the tissue is inside its seed's `ref_frame` window: the size rules hold."""
    return int(getattr(H, "frame", 0) or 0) < int(m.get("ref_frame", 0) or 0)


def reset_vbirth_at_reference(H, m, at, v, nF):
    """At the reference frame, once: every seeded cell's `Vbirth` becomes the volume it has now,
    in the convention `v` is in. Before this the block held the seed's TARGET, which on an
    apico-basal shell is 0.6 of the settled volume -- a first generation born 40% small."""
    if int(m.get("ref_frame", 0) or 0) <= 0 or m.get("vbirth_at_reference"):
        return
    # WITH THE ENGINE'S WARM-UP (2026-09-28) the reset waits for its end -- frame 0 of the record, the relaxed tissue --
    # and a cell's birth volume is the one its AGE implies: `seed_cycle` puts a cell at cycle position p at
    # v_b (1 + p), so v_b = v / (1 + p). Without a warm-up (`general.warmup: 0`) the old reset, verbatim.
    _wu = int(getattr(H, "warmup", 0) or 0)
    if _wu > 0 and int(getattr(H, "frame", 0) or 0) < _wu:
        return
    cat = resolve_cell_set(H, at)
    vb = cell_block(H, cat, "Vbirth", nF)
    if vb is not None and len(vb) == nF:
        _v = np.asarray(v[:nF], np.float64)
        if _wu > 0:
            _p = cell_block(H, cat, "cycle_progress", nF)
            if _p is not None and len(_p) == nF:
                _v = _v / (1.0 + np.clip(np.asarray(_p, np.float64), 0.0, 1.0))
        set_cell_block(H, cat, "Vbirth", _v, nF)
    m["vbirth_at_reference"] = True


def cell_size(lvl, m, nF, pos_np=None, H=None, at="vertex"):
    """A cell's volume AND the reference to compare it against, IN ONE CONVENTION.

    THE DEFECT THIS EXISTS TO END. A cell's volume has two definitions in this tree and both are
    lengths cubed, so nothing dimensional ever separated them:

        wedge        `face_geometry_3d` -- origin-referenced cones over the mid-surface ring. It is
                     what the mid-surface energy integrates, and it CANNOT SEE THICKNESS.
        polyhedron   `apicobasal_geometry_3d` -- two caps and one wall per ring edge, by the
                     divergence theorem. It is what `cell_mechanics[apicobasal]` defends.

    On the reference spheroid they read 2.5433 and 1.3508 for the same cell. Mixed, they produced:
    a G1 that ran 60% long because `cell_grow`'s checkpoint subtracted a polyhedron `Vbirth` from a
    wedge threshold; a `gate_ab_population` in which NOT ONE CELL DIVIDED IN 401 FRAMES, because the
    tissue answered growth by thickening while the wedge volume the trigger read actually FELL,
    2.32 -> 2.07; and a `cell_die` whose extrusion threshold is stated in a volume the cell does not
    have.

    `cell_divide` solved it for itself -- read the polyhedron where the run carries a separation,
    cache the matching reference -- and this is that solution made shared, so the other three size
    readers cannot drift from it. THE RUN'S GEOMETRY DECIDES, not a parameter: a set carrying `sep`
    is a set of polyhedra, and there is nothing to configure.

    Returns `(v_now, v_ref)` as float64 numpy, both in the same convention.
    """
    import numpy as _np
    _s = lvl.get("sep") if "sep" in getattr(lvl, "state_schema", {}) else None
    _Nv = int(m["Nv"])
    P = (torch.as_tensor(pos_np, dtype=torch.float32)[:_Nv] if pos_np is not None
         else lvl.get("pos")[:_Nv].detach().to(torch.float32).cpu())
    # THE GEOMETRY DECIDES: a set that carries a separation is a set of polyhedra, and every size
    # reader takes the polyhedron volume -- the volume `cell_mechanics[apicobasal]` defends. This
    # was the c671fb31 rule, withdrawn because it re-timed the size-triggered working points, and
    # it is back for a reason measured on `divide_growing_ball` (R2 of
    # notes/size_cycle/SIZE_CYCLE_PLAN.md, finding 13): with `cell_divide` reading the WEDGE the
    # shell crumples from frame 600 (thickness CV 0.67 at 801, `tools/spheroid_gauge.py`), with
    # the polyhedron it is a spheroid for the whole run. A cell whose thickness runs away divides
    # when its polyhedron doubles, and its daughters relax; the wedge cannot see thickness, so
    # under it the thick cells persist and the spread compounds. The seed's TARGETS are untouched
    # by this -- `v0_from` is a target convention, and the polyhedron target was what crumpled the
    # mechanics (finding 8); this is about what the rules READ.
    if _s is not None:
        _s = _s.detach() if hasattr(_s, "detach") else torch.as_tensor(_s)
        if int(_s.shape[0]) >= _Nv:
            vp, _, _, _ = apicobasal_geometry_3d(P.cpu(), _s[:_Nv].to(torch.float32).cpu(),
                                                 m["E_srce"].cpu(), m["E_trgt"].cpu(),
                                                 m["E_face"].cpu(), nF)
            v = vp.numpy().astype(_np.float64)
            # CACHED AT THE FIRST CALL, the way the wedge reference is fixed at seed time. The specs
            # seed their thickness at the energy's own equilibrium (`tools/equilibrium_h.py`), so
            # the first call is a rest state and needs no settling window -- see `cell_divide`,
            # where three attempts to time this measurement on a moving shell each produced a
            # different population from the same spec.
            if "v_ref_poly" not in m:
                m["v_ref_poly"] = float(_np.median(v))
            if H is not None and not m.get("vbirth_at_reference"):
                reset_vbirth_at_reference(H, m, at, v, nF)       # once; waits for the end of a warm-up
            m["size_convention"] = "polyhedron"
            return v, float(m["v_ref_poly"])
    area, _, _, vf = face_geometry_3d(P.cpu(), m["E_srce"].cpu(), m["E_trgt"].cpu(),
                                      m["E_face"].cpu(), nF, apex=wedge_apex(m, P.cpu()))
    v_ref = float(m.get("v_ref", 1.0))
    # A FLAT SHEET HAS NO WEDGE. `seed_mesh shape: disc` lays the patch in the plane of its own
    # `centre`, which is the wedge apex, so every cone over it has zero volume: V0f, `V0` and
    # `v_ref` were all seeded at exactly 0 on every `sheet_*` run. Everything stated "as a fraction
    # of v_ref" then reads as a fraction of nothing -- `cell_die`'s extrusion threshold was 0, so a
    # cell shrunk to the 1e-9 floor is never "small enough" and not one cell is ever extruded
    # from a sheet, so the patches of unremovable dead cells buckle it (`sheet_morphogen_die` is
    # the spec that shows it). On a sheet the cell's size IS its area, the
    # convention the paper's 2D model uses, and the reference is the seed-time median area,
    # cached at the first call exactly as `v_ref_poly` is. Decided by the geometry -- a seeded
    # wedge reference of zero -- and not by a parameter, like the other two conventions.
    if v_ref <= 1e-12:
        a = area.numpy().astype(_np.float64)
        if "a_ref" not in m:
            m["a_ref"] = float(_np.median(a[:nF]))
        m["size_convention"] = "area"
        return a, float(m["a_ref"])
    m["size_convention"] = "wedge"
    return vf.numpy().astype(_np.float64), v_ref


def cell_block(H, cat, name, nF):
    """The first `nF` rows of a DECLARED width-1 block on the cell set, as float64. None if absent.

    THE OTHER HOME FOR PER-CELL STATE, AND THE ONE plexus2 ACTUALLY DESCRIBES. Two stores have
    grown up side by side: `chem`, `area` and `centroid` are declared blocks on the `cell` SET, permuted
    by `Hierarchy.renumber_set`; `A0`, `V0f`, `age`, `phase` and the rest are columns on the mesh
    table of the `vertex` set, permuted by `MeshTable.reindex_faces` through `m["face_carry"]`. They
    hold the same kind of quantity -- one number per cell -- and the paper knows about only one of
    them. An operator whose output is not declared state cannot emit a delta, which is why every
    writer of the second store had to register `kind="structural"`; the two facts are one fact.

    THE THREE TOPOLOGY PATHS ALREADY AGREE, which is what makes moving a column safe rather than a
    rewrite. `cell_die` calls `reindex_faces(keep)` and `renumber_set(cat, keep, nF2)` with THE SAME
    `keep`; `edge_flip` does the same after a face drop; `cell_divide` reindexes the face columns by
    `keep` and copies the mother's whole cell-set row onto each daughter (`cst[nF + i] =
    cst[mother]`), which is the same semantics `reindex_faces` gives -- the parent's value on both
    daughters, so what moves must be INTENSIVE, exactly as before. A column therefore lands in the
    same place either way, and the byte-identity harness is what says so rather than this comment.

    RETURNS None RATHER THAN RAISING when the block is not declared, because the caller is better
    placed to say what the spec is missing: `cell_cycle` names the three blocks and the set in one
    message. A silent zero here would be the defect this move exists to remove.
    """
    lvl = H.level(cat) if cat else None
    if lvl is None or getattr(lvl, "state", None) is None or name not in lvl.state_schema:
        return None
    return lvl.get(name)[:nF, 0].detach().cpu().numpy().astype(np.float64)


def set_cell_block(H, cat, name, values, nF):
    """Write `values` (length nF) into the first nF rows of a declared width-1 cell block.

    IN PLACE, ON THE VIEW `Level.get` RETURNS, which is a slice of the set's `state`. That is
    writing integrated state from an operator and it is why `cell_cycle` is still `kind:
    structural` after this rung -- S4 turns the phase into a continuous progress variable that can
    be emitted as a delta, and only then does the write go away. Moving the STORE and changing the
    KIND are two separate claims and this rung makes only the first.
    """
    lvl = H.level(cat)
    col = lvl.get(name)
    col[:nF, 0] = torch.as_tensor(np.asarray(values), dtype=lvl.state.dtype,
                                  device=lvl.state.device)


def cell_block_t(H, cat, name, nF):
    """The live rows of a declared width-1 cell block AS A TENSOR, in the set's own dtype.

    THE FLOAT64 ROUND-TRIP IN `cell_block` IS NOT FREE. That one returns float64 numpy because its
    first callers already worked in numpy and compared against thresholds coarse enough not to
    care. The energy's targets are not like that: `A0`, `P0` and `V0f` go straight into
    `cell_mechanics`, which turns them into vertex positions, and `edge_flip` reconnects on an
    edge-length comparison -- so a single mantissa bit moves one flip and the run diverges from
    there. Staying in the state tensor's dtype is what makes moving these arrays a MOVE, testable
    by byte-identity, rather than a change that merely looks like one.
    """
    lvl = H.level(cat) if cat else None
    if lvl is None or getattr(lvl, "state", None) is None or name not in lvl.state_schema:
        return None
    return lvl.get(name)[:nF, 0]


def set_cell_block_t(H, cat, name, values, nF):
    """Write a tensor into the first nF rows of a declared width-1 cell block, no dtype detour."""
    lvl = H.level(cat)
    v = values if torch.is_tensor(values) else torch.as_tensor(np.asarray(values))
    lvl.get(name)[:nF, 0] = v.to(dtype=lvl.state.dtype, device=lvl.state.device)


def require_cell_block(H, cat, name, nF, who):
    """`cell_block`, but a missing declaration is an error naming the spec line that fixes it.

    THE DIFFERENCE BETWEEN A REFUSAL AND A SILENT ZERO is the whole reason per-cell state is being
    moved onto the set. A mesh table is an open namespace: `m.get("Vbirth")` on a run that never
    seeded it returns None, and every reader of it grew its own fallback -- `divjit`'s was a
    fixed-seed draw, which means a spec missing its seed still ran, still divided, and staggered its
    divisions with a jitter nobody asked for. A declared block cannot be missing by accident; it can
    only be missing from the spec, and then this says so.
    """
    a = cell_block(H, cat, name, nF)
    if a is None:
        raise KeyError(
            f"{who} needs the set {cat!r} to declare the width-1 block {name!r}. Add it under "
            f"`sets.{cat}.state` as `{name}: {{width: 1, record: false}}`. It is per-cell state, "
            f"so it belongs to the set whose members are cells -- where the topology operators "
            f"renumber it without having to know its name.")
    return a


def _seed_cell_blocks(H, cat, nF, dt, dev, **arrays):
    """Write a seed operator's per-cell arrays into declared blocks on the cell set.

    SEEDING IS WHERE THE DECLARATION HAS TO BE CHECKED, because it is the first write and the only
    one that happens before frame 0. A spec that forgot the block would otherwise reach its first
    `cell_divide` -- possibly hundreds of frames later, possibly in a run left overnight -- before
    anything noticed.
    """
    for k, v in arrays.items():
        if cell_block(H, cat, k, nF) is None:
            raise KeyError(
                f"seed_mesh writes the per-cell block {k!r}, which the set {cat!r} does not "
                f"declare. Add `{k}: {{width: 1, record: false}}` under `sets.{cat}.state`.")
        set_cell_block(H, cat, k, np.asarray(v, np.float64), nF)


def fib_sphere(n, r=1.0):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); theta = np.pi * (1 + 5 ** 0.5) * i
    return r * np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def build_sphere_mesh(n, r=1.0, jitter=0.0, seed=0, lattice_rotation=False):
    """Closed spherical half-edge mesh from a spherical Voronoi of a (jittered) Fibonacci sphere.
    Returns vertices [Nv,3], E_srce/E_trgt/E_face, nF. Every edge is shared by exactly two faces.

    `lattice_rotation` -- the Fibonacci lattice has its poles on z, and on the apico-basal working
    point the seeded shell elongates ALONG z by itself (aspect ratio ~1.12 by frame 150 with no
    anisotropy at all, exp 5 round 1), so any axis-dependent test on it is biased toward z. True
    rotates the lattice by a random rotation drawn from `seed`, so the lattice's own direction is
    different in every run and unrelated to any declared axis. Default False: every existing seed
    is unchanged."""
    pts = fib_sphere(n, r)
    if lattice_rotation:
        _q, _r = np.linalg.qr(np.random.default_rng(seed + 7919).normal(size=(3, 3)))
        _q = _q * np.sign(np.diag(_r))                   # a proper uniform rotation (Haar)
        if np.linalg.det(_q) < 0:
            _q[:, 0] = -_q[:, 0]
        pts = pts @ _q.T
    if jitter > 0:
        g = np.random.default_rng(seed)
        pts = pts + jitter * (g.random((n, 3)) - 0.5)
        pts = r * pts / np.linalg.norm(pts, axis=1, keepdims=True)
    sv = SphericalVoronoi(pts, radius=r, center=np.zeros(3)); sv.sort_vertices_of_regions()
    faces = [np.array(reg, np.int64) for reg in sv.regions]
    V = sv.vertices                                          # reorient every face CCW as seen from OUTSIDE
    for i, rr in enumerate(faces):                           # (Newell normal points away from the centre) so the
        P = V[rr]; N = np.cross(P, np.roll(P, -1, 0)).sum(0) # mesh is consistently oriented: each directed edge
        if float(np.dot(N, P.mean(0))) < 0:                  # appears once (needed by the topology ops) and every
            faces[i] = rr[::-1]                              # per-cell wedge volume is positive.
    es, et, ef = [], [], []
    for f, rr in enumerate(faces):
        k = len(rr)
        for i in range(k):
            es.append(int(rr[i])); et.append(int(rr[(i + 1) % k])); ef.append(f)
    return (sv.vertices.astype(np.float64), np.array(es, np.int64), np.array(et, np.int64),
            np.array(ef, np.int64), len(faces))


def build_hexagon_mesh(n=1, r=1.0, jitter=0.0, seed=0):
    """ONE regular hexagon as an open half-edge mesh: a single face, six vertices, CCW from +z.

    THE ONLY SOLID IN THIS PROMOTION WHOSE SHAPE INDEX IS KNOWN ON PAPER, and that is its whole
    reason for existing. A regular hexagonal prism of side a = 1 and height h = 1 has

        A_cap = (3 sqrt3 / 2) a^2 = 2.598076      A_lat = 6 a h = 6
        S     = 2 A_cap + A_lat = 3 sqrt3 + 6 = 11.196152
        V     = A_cap h = 3 sqrt3 / 2 = 2.598076        V^(2/3) = 1.889882
        s     = S / V^(2/3) = 5.924261377933605

    and that ONE number is only reachable if the fan triangulation, the cap centroid convention,
    both cap orientations and the two-triangle wall split are simultaneously right. It is gate row
    AB-C2, and `build_disc_mesh` cannot produce it: its lattice lays equal x- and y-pitch with
    alternate rows offset by half a step, so its Voronoi cells are hexagon-ISH and never regular.

    A VALUE ON `shape:`, NOT A MODEL, by the vocabulary's own written test -- "a flat patch and a
    closed shell are the same hypothesis about the tissue seeded into two different geometries". A
    single regular cell is the third such geometry.

    THE CIRCUMRADIUS IS THE SIDE, for a hexagon and only for a hexagon, so `radius: 1.0` gives side
    1. `n` and `jitter` are accepted and ignored: the builders share one call signature, and a
    jittered hexagon is not a hexagon.
    """
    th = np.arange(6) * (np.pi / 3.0)
    verts = np.stack([r * np.cos(th), r * np.sin(th), np.zeros(6)], 1).astype(np.float64)
    es = np.arange(6, dtype=np.int64)
    et = (np.arange(6, dtype=np.int64) + 1) % 6
    return verts, es, et, np.zeros(6, np.int64), 1


def build_prism_mesh(n, r=1.0, jitter=0.0, seed=0, height=None):
    """ONE apico-basal cell as a closed surface: an apical cap, a basal cap, and n lateral walls.

    Returns the same four things every builder here returns -- vertices [Nv,3], E_srce/E_trgt/E_face,
    nF -- so it drops into `seed_mesh` beside the sphere and the disc, and everything downstream
    (mesh_contact, the renderer, the geometry operators) works on it unchanged.

    WHY A PRISM AND NOT A MID-SURFACE WITH `sep`. Plexus's apico-basal representation is a mid-
    surface mesh plus a per-vertex half-separation, with apical at `pos + sep`, basal at `pos - sep`
    and the lateral walls REBUILT on every call rather than stored. That is the right representation
    to compute with and the wrong one to DRAW or to put material inside: a single cell in it is one
    polygon, which renders as a flat disc and encloses nothing. This builder emits the surface that
    polygon stands for -- the apical ring at +h/2, the basal ring at -h/2, one quad per side -- so a
    single cell is a closed, orientable, star-shaped surface with an inside.

        Nv = 2n     nF = n + 2       nH = 2n (the two caps) + 4n (the walls) = 6n
        V - E + F = 2n - 3n + (n + 2) = 2, as a closed surface must be

    `n` is the number of SIDES, `r` the circumradius of the caps and `height` their separation --
    the cell's own apico-basal axis, along y, which is the up axis everywhere in this library.

    Reference: the prism is the cell of Okuda, S. et al. (2013), Biomech. Model. Mechanobiol.
    12:627-644, drawn rather than integrated.
    """
    n = max(3, int(n))
    h = float(height if height is not None else 2.0 * r)
    g = np.random.default_rng(seed)
    th = 2.0 * np.pi * np.arange(n) / n
    rr = r * (1.0 + jitter * (g.random(n) - 0.5))
    apical = np.stack([rr * np.cos(th), np.full(n, 0.5 * h), rr * np.sin(th)], 1)
    basal = np.stack([rr * np.cos(th), np.full(n, -0.5 * h), rr * np.sin(th)], 1)
    V = np.concatenate([apical, basal], 0)
    rings = [list(range(n)), list(range(2 * n - 1, n - 1, -1))]          # the two caps
    for k in range(n):                                                   # one wall per side
        k1 = (k + 1) % n
        rings.append([k, k1, n + k1, n + k])
    # ORIENTED CCW SEEN FROM OUTSIDE, by the same Newell test `build_sphere_mesh` uses: a face whose
    # normal points back toward the body is reversed. That is what makes every directed edge appear
    # exactly once, which the topology operators and `mesh_contact`'s star-shape both rely on.
    c = V.mean(0)
    for i, rr_ in enumerate(rings):
        P = V[rr_]
        N = np.cross(P - c, np.roll(P, -1, 0) - c).sum(0)
        if float(np.dot(N, P.mean(0) - c)) < 0:
            rings[i] = rr_[::-1]
    es, et, ef = [], [], []
    for f, rr_ in enumerate(rings):
        k = len(rr_)
        for i in range(k):
            es.append(int(rr_[i])); et.append(int(rr_[(i + 1) % k])); ef.append(f)
    return (V.astype(np.float64), np.array(es, np.int64), np.array(et, np.int64),
            np.array(ef, np.int64), len(rings))


def build_prisms_mesh(n, r=1.0, jitter=0.0, seed=0, height=None):
    """`n` ADJACENT apico-basal cells: hexagonal prisms on the lattice that tiles the plane.

    The single-cell `build_prism_mesh` repeated over the first `n` sites of a hexagonal lattice,
    which is the packing an epithelium actually has -- flat-top hexagons of circumradius `r` sit at
    a centre-to-centre distance of sqrt(3) r, so neighbouring cells share a full wall rather than
    touching along a line. Each cell is its OWN closed surface: the shared wall is two coincident
    faces, one belonging to each cell.

    THAT IS DELIBERATE AND IT IS THE DIFFERENCE FROM `seed_mesh[apicobasal]`. A tissue in the
    apico-basal representation is ONE mid-surface whose faces are the cells, so a wall is a single
    face and the cells cannot be separated. Here each cell is a closed volume of its own, which is
    what is needed to put material INSIDE one -- a containment map wants a cell to have an inside,
    and a shared wall would make "which cell is this point in" ambiguous exactly at the wall.

        per cell   Nv = 2m,  nF = m + 2,  nH = 6m       (m = 6 sides)
        n cells    the same, n times, with no shared vertices

    Reference: Honda, H. (1978). J. Theor. Biol. 72:523-543 (the hexagonal packing of an
    epithelium); the prism itself is Okuda, S. et al. (2013), Biomech. Model. Mechanobiol.
    12:627-644.
    """
    n = max(1, int(n))
    m = 6
    d = float(np.sqrt(3.0)) * float(r)                       # centre-to-centre of adjacent hexagons
    # THE ORDER OF THE SITES IS A PICTURE DECISION AND NOTHING ELSE. A compact rhombus is the
    # tightest packing and the worst thing to look at: two of its four cells sit directly behind the
    # other two in any axis-aligned view, so a four-cell patch renders as two. This walk spreads the
    # first few sites across the plane while keeping every one of them adjacent to the previous.
    ax = [(0, 0), (1, 0), (1, -1), (2, -1), (0, 1), (-1, 1), (-1, 0), (0, -1),
          (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2), (-2, 1), (-2, 0)]
    if n > len(ax):
        raise ValueError(f"build_prisms_mesh: {n} cells asked for, {len(ax)} lattice sites known")
    centres = [(d * (q + 0.5 * rr), 0.0, d * float(np.sqrt(3.0)) / 2.0 * rr)
               for q, rr in ax[:n]]
    Vs, ess, ets, efs = [], [], [], []
    nv0, nf0 = 0, 0
    for cx, cy, cz in centres:
        V, es, et, ef, nF = build_prism_mesh(m, r, jitter, seed, height=height)
        Vs.append(V + np.array([cx, cy, cz], V.dtype))
        ess.append(es + nv0); ets.append(et + nv0); efs.append(ef + nf0)
        nv0 += V.shape[0]; nf0 += nF
    return (np.concatenate(Vs, 0), np.concatenate(ess), np.concatenate(ets),
            np.concatenate(efs), nf0)


def build_disc_mesh(n, r=1.0, jitter=0.0, seed=0):
    """A FLAT Voronoi patch as an open half-edge mesh: vertices [Nv,3] (z=0), E_srce/E_trgt/E_face, nF.

    The planar twin of `build_sphere_mesh`, and it exists for one reason: a sheet is where the
    monolayer's apical/basal offset can be judged with no curvature in the picture. On a closed
    shell a normal offset that is going unstable and a shell that is merely bumpy look alike; on a
    flat patch the mid-surface should stay in its plane, and any departure IS the instability.

    OPEN, AND DELIBERATELY SO. Only the Voronoi regions that are BOUNDED and lie wholly inside the
    disc are kept, so the patch has a rim: rim edges belong to one face, not two. Euler is not 2 and
    `_check_closed` will say so -- which is correct, this is not a closed surface. The operators that
    care refuse a rim face rather than crashing (`divide_face_3d` returns None when a split edge has
    no neighbour), so the interior grows and the rim does not.

    CCW AS SEEN FROM +z, so the Newell area vector points along +z and every wedge volume taken about
    an origin BELOW the patch is positive -- which is what makes `cell_grow` and `cell_divide` work
    on a sheet without changing them: at height h the wedge volume is (h/3)A, exactly proportional to
    the cell's area.
    """
    from scipy.spatial import Voronoi
    g = np.random.default_rng(seed)
    # a jittered hex-ish lattice covering the disc, which gives rounder cells than pure Poisson
    m = int(np.ceil(np.sqrt(n / 0.7854)))                    # disc packs pi/4 of its bounding square
    step = 2.0 * r / max(m, 1)
    xs, ys = np.meshgrid(np.arange(m + 1) * step - r, np.arange(m + 1) * step - r)
    xs = xs + (np.arange(m + 1)[:, None] % 2) * 0.5 * step   # offset alternate rows -> hexagonal
    pts = np.stack([xs.ravel(), ys.ravel()], 1)
    if jitter > 0:
        pts = pts + jitter * step * (g.random(pts.shape) - 0.5)
    pts = pts[np.linalg.norm(pts, axis=1) <= r * 1.35]       # a collar OUTSIDE the disc, so the
    #                                                          cells we keep have bounded regions
    vor = Voronoi(pts)
    keep_r = r                                               # cells whose vertices all sit inside
    faces, V = [], vor.vertices
    for ip, ir in enumerate(vor.point_region):
        reg = vor.regions[ir]
        if not reg or -1 in reg:                             # unbounded region: outside the collar
            continue
        P = V[reg]
        if np.linalg.norm(P, axis=1).max() > keep_r:
            continue
        faces.append(np.asarray(reg, np.int64))
    if not faces:
        raise ValueError(f"build_disc_mesh: no bounded cell inside r={r} for n={n}")
    used = np.unique(np.concatenate(faces))                  # drop the collar's vertices
    remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    verts = np.zeros((len(used), 3), np.float64); verts[:, :2] = V[used]
    es, et, ef = [], [], []
    for f, reg in enumerate(faces):
        rr = remap[reg]
        P = verts[rr, :2]
        c = P.mean(0)
        if np.cross(P[0] - c, P[1] - c) < 0:                 # CCW seen from +z
            rr = rr[::-1]
        k = len(rr)
        for i in range(k):
            es.append(int(rr[i])); et.append(int(rr[(i + 1) % k])); ef.append(f)
    return (verts, np.array(es, np.int64), np.array(et, np.int64),
            np.array(ef, np.int64), len(faces))


def build_disc_mesh_isotropic(n, r=1.0, jitter=0.0, seed=0):
    """`build_disc_mesh` without its two preferred axes -- for `seed_mesh[isotropic]`.

    The default lattice lays EQUAL x- and y-pitch with alternate rows offset by half a step, so each
    Voronoi cell is 2/sqrt(3) = 1.155 times longer along y than along x, and the rows run along x.
    Measured on exp08's relaxed sheet: a tissue-mean cell elongation M_E 0.07 along y, and a polarity
    that orders along x when the tissue is stretched along y but not when it is stretched along x
    (exp08 Phase 2, Finding P3). Here the row pitch is sqrt(3)/2 of the column pitch -- a true
    hexagonal lattice -- and the lattice is turned by an angle drawn from `seed` (uniform on [0, pi)),
    so no axis is preferred across seeds. Otherwise identical: the same jitter, collar and Voronoi.
    """
    from scipy.spatial import Voronoi
    g = np.random.default_rng(seed)
    th = float(g.uniform(0.0, np.pi))
    m = int(np.ceil(np.sqrt(n / 0.7854 * (np.sqrt(3) / 2))))       # the default's cell count: rows are denser
    step = 2.0 * r / max(m, 1)
    ny = int(np.ceil(2.0 * r / (step * np.sqrt(3) / 2)))
    xs, ys = np.meshgrid(np.arange(m + 1) * step - r, np.arange(ny + 1) * step * np.sqrt(3) / 2 - r)
    xs = xs + (np.arange(ny + 1)[:, None] % 2) * 0.5 * step
    pts = np.stack([xs.ravel(), ys.ravel()], 1)
    c, s_ = np.cos(th), np.sin(th)
    pts = pts @ np.array([[c, s_], [-s_, c]])
    if jitter > 0:
        pts = pts + jitter * step * (g.random(pts.shape) - 0.5)
    pts = pts[np.linalg.norm(pts, axis=1) <= r * 1.35]
    vor = Voronoi(pts)
    faces, V = [], vor.vertices
    for ip, ir in enumerate(vor.point_region):
        reg = vor.regions[ir]
        if not reg or -1 in reg:
            continue
        if np.linalg.norm(V[reg], axis=1).max() > r:
            continue
        faces.append(np.asarray(reg, np.int64))
    if not faces:
        raise ValueError(f"build_disc_mesh_isotropic: no bounded cell inside r={r} for n={n}")
    used = np.unique(np.concatenate(faces))
    remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    verts = np.zeros((len(used), 3), np.float64); verts[:, :2] = V[used]
    es, et, ef = [], [], []
    for f, reg in enumerate(faces):
        rr = remap[reg]
        P = verts[rr, :2]
        cc = P.mean(0)
        if np.cross(P[0] - cc, P[1] - cc) < 0:
            rr = rr[::-1]
        k = len(rr)
        for i in range(k):
            es.append(int(rr[i])); et.append(int(rr[(i + 1) % k])); ef.append(f)
    return (verts, np.array(es, np.int64), np.array(et, np.int64),
            np.array(ef, np.int64), len(faces))


def build_strip_mesh(n, r=1.0, jitter=0.0, seed=0, kind="plane", width=0.5, twist_R=None):
    """A Voronoi patch on a RECTANGLE, mapped onto a surface: `plane`, `ribbon` or `moebius`.

    One builder for three shapes because they are one construction. Cells are laid out in a
    parameter rectangle `(u, v)` with `u` around the strip and `v` across it, tessellated there, and
    only then placed in R^3 by the shape's own map:

        plane     (u, v, 0)                                    -- flat, two free edges, one seam cut
        ribbon    ((R cos t), (R sin t), v),        t = 2 pi u / L
        moebius   ((R + v cos(t/2)) cos t, (R + v cos(t/2)) sin t, v sin(t/2))

    THE PERIODIC KINDS ARE STITCHED, NOT CUT. A strip built by tessellating a rectangle and bending
    it has a SEAM: the cells at u = 0 and u = L are neighbours in space and strangers in the
    half-edge table, so `edge_flip` cannot pass the join and every measure that walks the ring reads
    a boundary where the surface has none. So the point set is TILED -- copies at `u - L` and
    `u + L` -- the Voronoi is taken over base plus ghosts, only cells whose SEED lies in the base
    domain are kept, and vertices that land on the same point in R^3 are merged. The join is then
    ordinary interior mesh.

    `moebius` TILES WITH A FLIP, `v -> -v`, which is the whole content of the shape: going once
    around returns you to the same place on the other side of the surface. **That makes it the one
    seed in this repo on which "apical" is not globally definable** -- a Moebius band is
    non-orientable, so a normal carried continuously around the strip comes back reversed, and any
    model that assumes a consistent outward side has to fail somewhere. `apicobasal_span_invalid_count`
    (gate row AB-B1) counts exactly that, so this shape is not decoration: it is the specimen on
    which the apico-basal representation's own orientation assumption is falsifiable.

    `width` is the strip's extent across, as a fraction of `r`. `twist_R` defaults to `r`.
    """
    from scipy.spatial import Voronoi
    if kind not in ("plane", "ribbon", "moebius"):
        raise ValueError(f"build_strip_mesh: kind must be plane|ribbon|moebius, got {kind!r}")
    g = np.random.default_rng(seed)
    R = float(r if twist_R is None else twist_R)
    w = float(width) * float(r)
    L = 2.0 * np.pi * R if kind != "plane" else 4.0 * float(r)
    step = float(np.sqrt(max(L * w, 1e-12) / max(n, 1)))
    nu = max(int(round(L / step)), 3)
    if kind != "plane":
        step = L / nu                                        # exact divisor, so the tiling matches
    nv = max(int(round(w / step)), 2)
    uu, vv = np.meshgrid((np.arange(nu) + 0.5) * step,
                         (np.arange(nv + 1) - nv / 2.0) * step)
    uu = uu + (np.arange(nv + 1)[:, None] % 2) * 0.5 * step   # alternate rows -> hexagonal
    pts = np.stack([uu.ravel(), vv.ravel()], 1)
    if jitter > 0:
        pts = pts + jitter * step * (g.random(pts.shape) - 0.5)
    base = pts.copy()
    if kind == "plane":
        # A COLLAR ON ALL FOUR SIDES, exactly as `build_disc_mesh` does: a Voronoi cell is bounded
        # only when it is surrounded, so the cells that are KEPT need neighbours that are not.
        ghosts = [base + [dx, dy] for dx in (-L, 0.0, L) for dy in (-w, 0.0, w)]
    else:
        # PERIODIC IN u, WITH THE MOEBIUS FLIP. `v -> -v` on the wrap is the half twist.
        s = -1.0 if kind == "moebius" else 1.0
        ghosts = [base,
                  np.stack([base[:, 0] - L, s * base[:, 1]], 1),
                  np.stack([base[:, 0] + L, s * base[:, 1]], 1),
                  base + [0.0, w], base + [0.0, -w]]         # collar across, where edges are free
    allp = np.concatenate(ghosts, 0)
    vor = Voronoi(allp)
    nb = len(base)
    faces = []
    for ip in range(len(allp)):
        if kind == "plane":
            if not (0.0 <= allp[ip, 0] <= L and -w / 2 <= allp[ip, 1] <= w / 2):
                continue
        elif ip >= nb:
            continue                                         # keep only seeds of the base domain
        reg = vor.regions[vor.point_region[ip]]
        if not reg or -1 in reg:
            continue
        faces.append(np.asarray(reg, np.int64))
    if not faces:
        raise ValueError(f"build_strip_mesh: no bounded cell for kind={kind!r}, n={n}")

    def to3d(P):
        u, v = P[:, 0], P[:, 1]
        if kind == "plane":
            return np.stack([u - L / 2, v, np.zeros_like(u)], 1)
        t = 2.0 * np.pi * u / L
        if kind == "ribbon":
            return np.stack([R * np.cos(t), R * np.sin(t), v], 1)
        rad = R + v * np.cos(t / 2.0)
        return np.stack([rad * np.cos(t), rad * np.sin(t), v * np.sin(t / 2.0)], 1)

    used = np.unique(np.concatenate(faces))
    P3 = to3d(vor.vertices[used])
    # MERGE WHAT THE MAP BROUGHT TOGETHER. Two parameter vertices a period apart are ONE point on a
    # ribbon; rounding to a fraction of the cell spacing identifies them without a neighbour search.
    key = np.round(P3 / (step * 1e-3)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    verts = P3[first]
    remap = -np.ones(len(vor.vertices), np.int64)
    remap[used] = inv
    es, et, ef = [], [], []
    for f, reg in enumerate(faces):
        rr = remap[reg]
        rr = np.array([rr[i] for i in range(len(rr)) if rr[i] != rr[i - 1]], np.int64)
        if len(rr) < 3:
            continue
        Q = vor.vertices[reg]
        c = Q.mean(0)
        if np.cross(Q[0] - c, Q[1] - c) < 0:                 # CCW in the PARAMETER plane
            rr = rr[::-1]
        for i in range(len(rr)):
            es.append(int(rr[i])); et.append(int(rr[(i + 1) % len(rr)])); ef.append(f)
    nF = int(max(ef) + 1) if ef else 0
    return (verts, np.array(es, np.int64), np.array(et, np.int64),
            np.array(ef, np.int64), nF)


def face_geometry_3d(pos, es, et, ef, nF, eocc=None, apex=None):
    """Per-face 3D area (Newell area-vector magnitude), perimeter, centroid, and the PER-CELL wedge
    volume v_f = (1/3)(cen_f . N_f) -- the volume of the pyramid from the sphere centre to the face.
    The lumen volume is just sum_f v_f, but keeping it per-cell lets each cell carry its own volume
    elasticity (a distributed term that resists local buckling). All differentiable in `pos`."""

    # CENTRED ON THE APEX FIRST, and not only for the wedge. The Newell area vector is a sum of
    # cross products of positions; about a box centre 58 units out each product is ~3400 and
    # they cancel to a face area of order 1, which in float32 keeps three digits of it (K_A's
    # energy moved by 1.2e-3 relative under a pure translation on the certification sphere).
    # Subtracting the apex costs one pass and makes every term below bit-identical to the same
    # tissue at the origin, up to the round-off already baked into the stored positions.
    if apex is not None:
        pos = pos - apex
        apex = None
    s = pos[es]; t = pos[et]
    length = (t - s).norm(dim=-1)
    cross = torch.cross(s, t, dim=-1)                        # consecutive-vertex cross products
    dev, dt = pos.device, pos.dtype
    if eocc is not None:                                     # reservoir: zero out dead (padding) half-edges
        length = length * eocc; cross = cross * eocc[:, None]; sw = s * eocc[:, None]; w = eocc
    else:
        sw = s; w = torch.ones_like(length)
    N = 0.5 * torch.zeros(nF, 3, device=dev, dtype=dt).index_add(0, ef, cross)   # area vector / face
    area = N.norm(dim=-1)
    perim = torch.zeros(nF, device=dev, dtype=dt).index_add(0, ef, length)
    cnt = torch.zeros(nF, device=dev, dtype=dt).index_add(0, ef, w)
    centroid = torch.zeros(nF, 3, device=dev, dtype=dt).index_add(0, ef, sw) / cnt.clamp(min=1)[:, None]
    # THE WEDGE'S APEX IS DECLARED, NOT THE WORLD ORIGIN. `apex=None` keeps the origin -- and
    # every run seeded there bit-identical, since x - 0.0 == x -- while a tissue seeded elsewhere
    # passes its own centre through `wedge_apex(m, pos)`. About the origin, a cell's wedge grows
    # with its distance from (0, 0, 0): at a box centre 43 away the far side of a shell carried
    # ~26% more wedge than the near side, so cell_grow's targets, cell_divide's trigger and the
    # calibrations' V_eq all depended on where the tissue sat. Measured: a 40-frame run at
    # [25, 25, 25] diverged from the same run at the origin by 0.42 in vertex position by frame
    # 10 with no division yet, then divided 310 cells against 319.
    cen = centroid if apex is None else centroid - apex
    vf = (1.0 / 3.0) * (cen * N).sum(dim=-1)                      # per-cell wedge volume (sum = lumen volume)
    return area, perim, centroid, vf


def wedge_apex(m, like):
    """The apex `face_geometry_3d` measures its wedges from: the tissue's declared seed centre,
    carried on the mesh table as `centre` by seed_mesh, as a tensor on `like`'s device and dtype.
    None -- the world origin, the historical formula -- when the mesh carries no centre."""
    try:
        c = m.get("centre") if m is not None else None
    except Exception:                                    # noqa: BLE001 -- a view without .get
        c = None
    if c is None:
        return None
    return torch.as_tensor([float(v) for v in c], device=like.device, dtype=like.dtype)


def _shape_energy_core(pos, es, et, ef, nF, A0, P0, V0f, alive, R0, K_A, K_P, K_V, K_R, Lam, Gam,
                       eocc, vocc, K_bend=0.0, twin_face=None, K_lumen=0.0, myo_e=None, Gam_l=0.0,
                       apex=None):
    """Explicit-arg vertex-model shape energy on a FIXED-size RESERVOIR (torch.compile-friendly: shapes never
    change, so it compiles once even under division). Dead slots are masked out: `alive` (faces),
    `eocc` (half-edges), `vocc` (vertices, for the radial term). R0 is a tensor (changes each frame);
    the K_* / Lam / Gam coefficients are compile-time constants."""
    # CENTRED ON THE APEX before any term is formed. The edge-length terms are translation-
    # invariant on paper and not in float32: the unit vector of a short Voronoi edge, both ends
    # near 58.6, kept two digits -- the line-tension gradient alone differed by 15% relative
    # between the origin and a shift of 58.6 on the certification sphere, 6.6e-3 at a shift
    # of 1.0. That was the base model's 0.12 per-vertex divergence between the same cellfix
    # tissue at the origin and at the box centre. No-op at the origin (x - 0 == x).
    if apex is not None:
        pos = pos - apex
        apex = None
    area, perim, centroid, vf = face_geometry_3d(pos, es, et, ef, nF, eocc, apex=apex)
    E = (K_A * (area - A0) ** 2 + K_P * (perim - P0) ** 2 + 0.5 * Gam * perim ** 2) * alive
    line = (pos[et] - pos[es]).norm(dim=-1) * eocc          # line tension over live half-edges only
    # PER-JUNCTION MYOSIN, when a junction operator has supplied it. `Lam` alone is one number for the
    # whole tissue, so no junction can be weaker than its neighbours and myosin cannot be recruited where
    # tension is high. `myo_e` is a per-half-edge multiplier on exactly that term -- which is where
    # actomyosin enters a vertex model -- and defaults to None, in which case this reduces to `Lam * line.sum()`
    # exactly and every existing run is bit-identical.
    E = E.sum() + (Lam * line.sum() if myo_e is None else Lam * (myo_e * line).sum())
    # A PER-JUNCTION QUADRATIC, WHICH IS NOT THE SAME TERM AS `Gam`. `Gam` above is Farhadifar's
    # cortical contractility, (Gamma/2) * PERIMETER^2, one number per CELL. `Gam_l` is Marinari's
    # (Nature 484:542, Supplementary p.1), (Gamma/2) * l^2 per JUNCTION -- they state the substitution
    # explicitly, "proportional to the square of the junction length, instead of the square of
    # perimeter length, and thus we model individual junctions as elastic springs with equilibrium
    # length l0 = -Lambda/Gamma". The two are rival hypotheses about one force, not additive
    # contributions, which is why `cell_mechanics[model: marinari]` sets one and zeroes the other
    # rather than the spec being free to set both.
    #
    # THE SUM IS OVER HALF-EDGES AND THEIR SUM IS OVER JUNCTIONS. A closed half-edge mesh carries two
    # half-edges per junction, so `line.sum()` is 2x a junction sum. The factor is NOT applied here --
    # `Lam` has always meant "per half-edge" and changing that would move every existing run -- it is
    # applied by the marinari variant when it maps the paper's parameters in. See ShapeEnergy3DMarinari.
    if Gam_l:
        E = E + 0.5 * Gam_l * ((line ** 2) * eocc).sum()
    E = E + K_V * ((vf - V0f) ** 2 * alive).sum()
    # THE RADIUS IS MEASURED FROM THE TISSUE'S OWN CENTRE, weighted by occupancy -- the same
    # correction the apico-basal and monolayer cores carry. |pos| is that radius only for a
    # vesicle at the origin, and cellfix_B_new seeded at a wall box's centre with K_R 0.4 would
    # otherwise be pulled toward the corner by this term.
    _w = vocc.to(pos.dtype).reshape(-1, 1)
    _c = (pos * _w).sum(dim=0, keepdim=True) / _w.sum().clamp(min=1.0)
    E = E + K_R * ((((pos - _c).norm(dim=1) - R0) ** 2) * vocc).sum()   # radial over live vertices only
    if K_bend > 0 and twin_face is not None:
        # DIHEDRAL bending (Wardetzky hinge): penalise the angle between a cell face's outward normal and
        # the normal of the cell across each shared edge -> smooths sharp cell-to-cell FOLDS (the hollow /
        # inverted-cap tilt the metric flags), high-frequency, WITHOUT flattening gentle whole-shell curvature
        # (unlike the radial K_R). Newell area vector per face -> unit normal; sum over half-edges (edge twice).
        s = pos[es]; t = pos[et]
        crossN = torch.cross(s, t, dim=-1) * eocc[:, None]
        Nf = 0.5 * torch.zeros(nF, 3, device=pos.device, dtype=pos.dtype).index_add(0, ef, crossN)
        nhat = Nf / (Nf.norm(dim=-1, keepdim=True) + 1e-12)
        cosang = (nhat[ef] * nhat[twin_face]).sum(dim=-1)       # cos(dihedral) per half-edge
        E = E + K_bend * ((1.0 - cosang) * eocc).sum()
    if K_lumen > 0:
        # GLOBAL LUMEN INCOMPRESSIBILITY: penalise the shell for enclosing LESS volume than a sphere of
        # its current total area. A buckled/wrinkled shell ALWAYS encloses less (isoperimetric inequality),
        # so this is the one constraint that distinguishes a sphere from a per-cell-volume-preserving
        # buckle -- which the per-cell wedge term (a pyramid from the origin) and local area/perim cannot.
        # Relative deficit -> scale-invariant. (Coral only; a tube is NOT the max-volume shape -> off there.)
        A_tot = (area * alive).sum()
        V_sphere = A_tot ** 1.5 / (6.0 * 3.14159265358979 ** 0.5)
        V_tot = (vf * alive).sum()
        E = E + K_lumen * ((V_tot - V_sphere) / (V_sphere + 1e-9)) ** 2
    return E


def _relax_subset(pos, es, et, ef, nF, A0, P0, V0f, alive, R0, mech, move_mask, iters, apex=None):
    """Bounded-Euler shape-energy descent that moves ONLY the vertices in `move_mask` (the fresh
    daughters + their one-ring after a division). Heals the just-cut caps in place so a division never
    leaves an inverted cap for the global relaxation to (fail to) fix. Reuses `_shape_energy_core`."""
    dev, dt = pos.device, pos.dtype
    eocc = torch.ones(es.shape[0], device=dev, dtype=dt); vocc = torch.ones(pos.shape[0], device=dev, dtype=dt)
    R0t = torch.as_tensor(float(R0), device=dev, dtype=dt)
    with torch.no_grad():
        cap = mech["cap_frac"] * (pos[et] - pos[es]).norm(dim=-1).mean().clamp(min=1e-6)
    mm = move_mask.to(dt)[:, None]
    x = pos.clone()
    for _ in range(iters):
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            E = _shape_energy_core(xg, es, et, ef, nF, A0, P0, V0f, alive, R0t, mech["K_A"], mech["K_P"],
                                   mech["K_V"], mech["K_R"], mech["Lambda"], mech["Gamma"], eocc, vocc, apex=apex)
            g = torch.nan_to_num(torch.autograd.grad(E, xg)[0])
        step = -mech["eta"] * g
        step = step * torch.clamp(cap / (step.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
        if mech.get("plane_axis") is not None:                    # 2D vertex model: see ShapeEnergy3D
            step[:, int(mech["plane_axis"])] = 0.0
        x = x + step * mm                                        # only the fresh region moves
    return x.detach()


def _engine_owns_clock(params, default=1):
    """D1: the ENGINE owns the operator period; operators must not gate themselves as well.

    Several operators kept a private `every`/`_k` AND were gated by the engine, so the effective
    period was the PRODUCT. A config saying `every: 2` fired once every FOUR frames, and every
    per-call quantity (min_cycle, max_div, max_div_frac) silently meant 4x what it said. That is
    the defect that re-anchored the whole archive.

    Returns 1 always -- the operator never gates.

    `every > 1` still hard-errors, because a spec written under the OLD semantics means something
    different now: its author wrote `every: 2` intending period 2 and silently got 4, so honouring
    it as 2 would change the run without telling anyone. Such a spec must be read by a human.

    BUT the engine reads THE SAME `every` key (engine.py `max(1, int(o.params.get("every", 1)))`),
    so a blanket raise also forbids the correct, engine-owned multi-rate cadence -- which is the
    only way to express "this operator runs every k frames" at all. That made every archived spec
    carrying `every: 2` permanently unloadable, including the two that generate the Turing x vertex
    videos on the site's front page.

    So the raise is now an opt-OUT, not a wall: a spec that has been migrated declares
    `engine_clock: true` next to its `every`, which asserts "this period is written for the
    engine-owned clock; do not second-guess it". Unmigrated specs still fail loudly.
    """
    e = int(params.get("every", default))
    if e > 1 and not bool(params.get("engine_clock", False)):
        raise ValueError(
            f"D1: operator-side `every={e}` is no longer supported -- the engine owns the clock. "
            f"(A private period multiplied the engine's, so `every: 2` meant once every 4 frames "
            f"and every per-call quantity meant 4x what it said.) If this spec was written for "
            f"the OLD semantics, its true period was {e}x the engine's -- convert it and set "
            f"`every: {e * e}`. Then declare `engine_clock: true` beside it to confirm the period "
            f"is now engine-owned.")
    return 1



# CANONICAL NAME `seed_mesh`, ALIAS `mesh_seed`, and the alias is not politeness. Archived
# specifications use both spellings in comparable numbers, so registering only one would make half
# the corpus fail to load -- loudly, with `operator not in registry`, but fail. The canonical form
# follows the `seed_<noun>` convention core already uses for `seed_from_segmentation`, while the
# okuda classes kept the old spelling, so 325 specs, including every `r0*` the campaign has ever
# written, could not be run at all. `seed_cell_chem` (320 specs) had the same break.
@register_operator("seed_mesh", "mesh_seed", set="vertex", kind="seed", family="seed", title="build the cell mesh", equation=r"""$$\{\mathbf c_j\}_{j=1}^{n}\ \text{on a sphere of radius }R\ \to\ \text{Voronoi faces},\qquad A^0_f=a_0\frac{4\pi R^2}{n},\quad P^0_f=p_0\sqrt{A^0_f},\quad V^0=\tfrac43\pi R^3$$""")
class SeedMesh3D(Structural):
    r"""build the surface the whole model lives on, once, at frame 0: a closed half-edge mesh whose faces are the cells.
    vertex -> vertex: writes every vertex position, and stashes the half-edge table and the per-face targets (A0, P0, V0) on the mesh.

        \{\mathbf c_j\}_{j=1}^{n}\ \text{on a sphere of radius }R\ \to\ \text{Voronoi faces},\qquad A^0_f=a_0\frac{4\pi R^2}{n},\quad P^0_f=p_0\sqrt{A^0_f},\quad V^0=\tfrac43\pi R^3
        c_j            the n seed points of the spherical Voronoi tessellation, jittered by `jitter`
        R              `radius`, the sphere radius, in world units
        n              `n_cells`, the number of cells
        A^0_f, P^0_f   the face's target area and perimeter, in world units squared and world units
        a_0            `a0_scale`, a multiplier on the target area
        p_0            `p0`, the target shape index P0/sqrt(A0), dimensionless
        V^0            the target lumen volume, in world units cubed

    Reference: Okuda, S. et al. (2013). Biomech. Model. Mechanobiol. 12(4):627-644."""
    PARAM_UNITS = {'radius': 'length', 'centre': 'length', 'height': 'length', 'width': 'length', 'jitter': 'fraction', 'p0': 'dimensionless', 'a0_scale': 'dimensionless', 'n_cells': 'count', 'seed': 'count', 'vseed_cv': 'fraction', 'age_seed': 'frames'}
    SYMBOLS = {'c_j': 'the n seed points of the spherical Voronoi tessellation, jittered by `jitter`', 'R': '`radius`, the sphere radius, in world units', 'n': '`n_cells`, the number of cells', 'A^0_f, P^0_f': "the face's target area and perimeter, in world units squared and world units", 'a_0': '`a0_scale`, a multiplier on the target area', 'p_0': '`p0`, the target shape index P0/sqrt(A0), dimensionless', 'V^0': 'the target lumen volume, in world units cubed'}
    PARAM_ROLES = {'n_cells': 'number_of_cells', 'radius': 'radius_of_the_sphere_or_disc', 'jitter': 'disorder_of_the_seed_points_as_a_fraction_of_their_spacing', 'p0': 'target_shape_index', 'seed': 'random_seed', 'centre': 'centre_of_the_tissue', 'shape': 'sphere_or_disc', 'lattice_rotation': 'turn_the_disc_lattice', 'height': 'height_of_the_strip', 'width': 'width_of_the_strip', 'type_layout': 'how_cell_types_are_laid_out', 'cell_set': 'the_cell_set', 'a0_scale': 'multiplier_on_the_target_area', 'v0_uniform': 'one_target_volume_for_every_cell', 'vseed_cv': 'spread_of_the_seeded_volumes', 'age_seed': 'spread_of_the_seeded_ages'}
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    # WHAT THE SEED'S NUMBERS ARE. `radius` and `h0` are lengths in the world box; `p0` is the
    # dimensionless shape index P/sqrt(A); `vseed_cv` is a fraction of the quantity it perturbs.
    # `n_cells` is a count. Anything not named here is UNDECLARED and therefore silent.
    # WHICH VOLUME THE SEEDED TARGET IS. `V0f` and `v_ref` come from `face_geometry_3d`, the WEDGE
    # volume: a sum of origin-referenced cones over the cell's mid-surface ring. Everything
    # downstream inherits that convention unless something re-seeds it, which is what the
    # apicobasal seed's `v0_from: polyhedron` exists to do.
    BLOCK_UNITS = {"V0f": "volume[wedge]", "Vbirth": "volume[wedge]",
                   "A0": "area[midsurface]", "P0": "length"}
    MECHANISM_TAGS = ["vesicle", "epithelial_shell", "spherical", "half_edge_mesh", "initial_condition"]
    REFERENCE = "Okuda, S. et al. (2013). Reversible network reconnection model for simulating large deformation in 3D tissues. Biomech. Model. Mechanobiol. 12:627-644; tyssue (DamCB)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.n = int(params.get("n_cells", 150)); self.R = float(params.get("radius", 5.0))
        self.jitter = float(params.get("jitter", 0.15)); self.p0 = float(params.get("p0", 3.9))
        self.seed = int(params.get("seed", 0))
        # WHERE THE SPHERE GOES. `build_sphere_mesh` returns it about the ORIGIN, and until now
        # nothing could move it: every spec that used this ran in a `free` world centred on 0, so a
        # unit box put the vesicle in a corner. Default [0,0,0], so the 461 specs that never asked
        # are byte-identical.
        self.centre = [float(v) for v in params.get("centre", [0.0, 0.0, 0.0])]
        # `shape: sphere | disc` -- THE GEOMETRY OF THE INITIAL CONDITION, and it is a value on this
        # operator rather than a `model:` because a flat patch and a closed shell are the same
        # hypothesis about the tissue seeded into two different geometries. `seed_plate` already
        # spells its geometry this way.
        #
        # A DISC IS AN OPEN MESH, which the rest of the vertex stack has never seen: `divide_face_3d`
        # refuses a face whose split edges lack two distinct interior neighbours, and every rim face
        # is one. That is the point of seeding it -- a flat sheet is where the monolayer's normal
        # offset can be tuned with no curvature in the picture -- but it means the rim does not
        # divide, and the run has to be read knowing that.
        #
        # `hexagon` IS THE THIRD VALUE, added for gate row AB-C2: one regular hexagonal cell, the
        # only solid in the apico-basal promotion whose shape index is known on paper. See
        # `build_hexagon_mesh`. It is a value for the same reason `disc` is -- the same hypothesis
        # about the tissue, seeded into a geometry chosen so a closed form is reachable.
        self.shape = str(params.get("shape", "sphere")).lower()
        self.lattice_rotation = bool(params.get("lattice_rotation", False))   # see build_sphere_mesh
        # THE APICO-BASAL AXIS OF A `prism`: the separation between its apical and its basal cap, in
        # world units. Unset, a prism is as tall as it is wide -- which is a cuboidal cell, not a
        # claim that height does not matter.
        self.height = params.get("height")
        if self.height is not None:
            self.height = float(self.height)
        # `plane`, `ribbon` and `moebius` are three more VALUES on the same axis, for the reason
        # `disc` is one: the same hypothesis about the tissue, seeded into a different geometry.
        # See `build_strip_mesh` -- and note that `moebius` is the one seed on which "apical" is not
        # globally definable, which is what makes it a test rather than a picture.
        self.strip_width = float(params.get("width", 0.5))
        if self.shape not in ("sphere", "disc", "hexagon", "plane", "ribbon", "moebius",
                              "prism", "prisms"):
            raise ValueError(f"mesh_seed: shape must be 'sphere', 'disc', 'hexagon', 'plane', "
                             f"'ribbon' or 'moebius', "
                             f"got {self.shape!r}")
        # PARTITION THE TYPES BY POSITION, NOT AT RANDOM. `type_layout` on a SET is applied at build
        # (engine.py), over that set's own coordinates -- which a mesh cell does not have yet: the
        # cells do not exist until this operator runs, so the build-time split assigns every one of
        # them to type 0 and a two-type spheroid comes out uniform. Declaring it here instead cuts
        # the shell at the equator of the axis named, tiling the declared `fraction`s along it, at
        # the moment the faces first have centroids.
        self.type_layout = str(params.get("type_layout", "random")).lower()
        self._cat = params.get("cell_set")          # override only; the SET declares the pairing
        # THE PREFERRED AREA, SET APART FROM THE GEOMETRY. `A0` was always `mean(area)` of the mesh
        # just built, so every run started exactly at its target and the ONLY direction it could go
        # was down, under whatever line tension the mechanics carried: a sweep of six specs that all
        # ended between A/A0 0.44 and 0.72 and never once tested expansion.
        #
        # It is also the missing half of the Marinari crowding transform. `cell_mechanics
        # [model: marinari]` rescales K, Lambda and Gamma by g^-2, g^-1/2 and g^-1, which leaves W
        # invariant ONLY together with A0' = g A0 -- and that line is the seeder's, so `crowding: 4`
        # weakened the area term 16x against a line term weakened 2x and collapsed the shell to
        # radius 1.4 with CV(junction length) 1.9. With this, `a0_scale: g` completes the transform,
        # and the paper's own words become the gate: a tissue "free to expand so that its total area
        # was larger by a factor g" must settle at A = g x its seeded area.
        #
        # `a0_scale: {axis: x, values: [g_0, g_1, ...]}` -- ONE g PER EQUAL-WIDTH BAND of the seeded
        # cell centroids along that axis, lowest coordinate first, so `values: [2.0, 1.0]` asks the
        # cells of the low-x half for twice the area of the high-x half. The scalar form is the
        # special case of one band and keeps its old code path, byte for byte. The scalar targets
        # that have no per-cell form (V0, v_ref, R0) take the MEAN of the values.
        _a0 = params.get("a0_scale", 1.0)
        self.a0_axis, self.a0_values = None, None
        if isinstance(_a0, dict):
            _axv = _a0.get("axis", "x")
            self.a0_axis = int(_axv) if isinstance(_axv, int) else "xyz".index(str(_axv).lower())
            self.a0_values = [float(v) for v in _a0.get("values", [])]
            if not self.a0_values:
                raise ValueError("mesh_seed: a0_scale {axis, values} needs at least one value")
            self.a0_scale = float(np.mean(self.a0_values))
        else:
            self.a0_scale = float(_a0)
        self.v0_uniform = bool(params.get("v0_uniform", False))   # one target volume for every cell
        if self.a0_scale <= 0 or any(v <= 0 for v in (self.a0_values or [])):
            raise ValueError(f"mesh_seed: a0_scale must be > 0, got {_a0}")
        self.vseed_cv = float(params.get("vseed_cv", 0.0))  # STOCHASTIC VOLUME SEED: per-cell random cell-cycle
        #   phase at t=0 (spread of the initial division threshold) -> desynchronises the FIRST division wave
        # `age_seed` -- THE OTHER WAY TO DESYNCHRONISE, AND THE ONLY ONE A CLOCK CAN USE. `vseed_cv`
        # spreads the THRESHOLD (`divjit`), which is a phase only for a rule that reads size:
        # `cell_divide[timer]` fires on `age >= cycle * jit` and every cell is seeded with age 0, so
        # a clock-driven tissue divides in ONE frame however wide `vseed_cv` is -- measured on
        # ab_06_population: 1279 of 1280 cells divided at frame 480 and on no other frame.
        #
        # AND SPREADING THE THRESHOLD IS THE WRONG KNOB WHEN THE CELLS MUST COME OUT THE SAME SIZE.
        # Paired with `cell_grow[model: timer]`, which drives every cell to one target volume in
        # `cycle_frames`, a cell that divides early divides SMALL: threshold noise buys sparseness
        # by spending size uniformity. Spreading the initial AGE instead leaves every cell the same
        # cycle length -- so they all reach the same size at division -- and merely offsets them in
        # time. Uniform volumes and sparse division are then not in tension.
        #
        # In DIVISION-CALLS, the unit `cycle`, `min_cycle` and `max_cycle` use; set it to `cycle` to
        # spread the population evenly over one generation. 0 (the default) seeds age 0 as before.
        self.age_seed = float(params.get("age_seed", 0.0))

    def forward(self, H, mask=None):
        # THE PAIRING IS READ FROM THE SET, ONCE PER CALL -- see `resolve_cell_set`.
        self.cell_set = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
        lvl = H.level(self.at); dev = lvl.state.device; dt = lvl.state.dtype
        if self.shape in ("plane", "ribbon", "moebius"):
            verts, es, et, ef, nF = build_strip_mesh(self.n, self.R, self.jitter, self.seed,
                                                     kind=self.shape, width=self.strip_width)
        else:
            _build = {"sphere": build_sphere_mesh, "disc": build_disc_mesh,
                      "hexagon": build_hexagon_mesh,
                      "prism": build_prism_mesh,
                      "prisms": build_prisms_mesh}[self.shape]
            _kw = {"height": self.height} if (self.shape in ("prism", "prisms")
                                              and self.height is not None) else {}
            if self.shape == "sphere" and self.lattice_rotation:
                _kw["lattice_rotation"] = True
            verts, es, et, ef, nF = _build(self.n, self.R, self.jitter, self.seed, **_kw)
        Nv = verts.shape[0]; Nbuf = lvl.state.shape[0]
        if Nv > Nbuf:
            raise ValueError(f"sphere mesh has {Nv} vertices but buffer n={Nbuf}")
        verts = verts + np.asarray(self.centre, verts.dtype)
        pos = torch.zeros(Nbuf, 3, dtype=dt, device=dev)
        pos[:Nv] = torch.as_tensor(verts, dtype=dt, device=dev)
        px0, px1 = lvl.state_schema["pos"]
        st = lvl.state.clone(); st[:, px0:px1] = pos; lvl.state = st
        if getattr(lvl, "occ", None) is not None:
            occ = torch.zeros(Nbuf, device=dev); occ[:Nv] = 1.0; lvl.occ = occ
        est = torch.as_tensor(es, device=dev); ett = torch.as_tensor(et, device=dev)
        eft = torch.as_tensor(ef, device=dev)
        area, perim, centroid, vf = face_geometry_3d(pos[:Nv], est, ett, eft, nF,
                                                     apex=torch.as_tensor(self.centre, device=dev, dtype=pos.dtype))
        A_seed = float(area.mean())
        A0 = A_seed * self.a0_scale; P0 = self.p0 * (A0 ** 0.5)
        # PER-CELL g, when `a0_scale` names an axis: the band of each cell's seeded centroid.
        g = None
        if self.a0_axis is not None:
            _c = centroid[:nF, self.a0_axis]
            _k = len(self.a0_values)
            _b = ((_c - _c.min()) / (_c.max() - _c.min()).clamp_min(1e-12) * _k).long().clamp(0, _k - 1)
            g = torch.as_tensor(self.a0_values, dtype=dt, device=dev)[_b]
            print(f"[mesh_seed] preferred area per cell: a0_scale {self.a0_values} x the seeded mean "
                  f"area {A_seed:.4f}, in {_k} equal-width bands along {'xyz'[self.a0_axis]} -> "
                  + "  ".join(f"g={v:g}: {int((_b == i).sum())} cells"
                              for i, v in enumerate(self.a0_values)), flush=True)
        if self.a0_scale != 1.0 and g is None:
            print(f"[mesh_seed] preferred area A0 = {self.a0_scale:g} x the seeded mean area "
                  f"{A_seed:.4f} = {A0:.4f}: every cell starts at {1.0 / self.a0_scale:.3f} of its "
                  f"target, so the shell is under-inflated and must EXPAND "
                  f"(radius x {self.a0_scale ** 0.5:.3f} if the area term wins outright)", flush=True)
        if self.vseed_cv > 0:                                    # random initial cell-cycle phase per cell
            dj = np.clip(1.0 + self.vseed_cv * np.random.default_rng(self.seed + 101).standard_normal(nF), 0.4, 1.8)
        else:
            dj = np.ones(nF)                                     # all cells born in phase (synchronised)
        # UNIFORM AND NOT GAUSSIAN, because this is a PHASE and a phase is uniform on its cycle: a
        # tissue in steady state has as many cells just born as just about to divide. A Gaussian
        # would pile the population at the middle of the cycle and still give a division wave, only
        # a rounder one. See `age_seed` in __init__ for why the phase and not the threshold.
        ag = (np.random.default_rng(self.seed + 202).random(nF) * self.age_seed
              if self.age_seed > 0 else np.zeros(nF))
        # A `MeshTable`, WHICH IS A `dict` -- see `plexus.models.mesh`. Every reader is unchanged
        # by construction: the type still passes `isinstance(m, dict)` (which the D4 acted-ledger
        # tests, and whose failure would score every mesh-only operator as inert), still iterates,
        # still takes `get`/`setdefault`/`m[k] = v` on an open namespace. What it adds is a name for
        # the thing, a place for the carry and the snapshot to live, and an engine that knows it
        # exists -- `grep -rn "_mesh" src/plexus/` returned NOTHING before this.
        #
        # FILLED, NOT REBOUND, when the set declared `mesh: half_edge`. `engine._build_mesh`
        # allocated the table during `build`, and rebinding here would hand every consumer a
        # DIFFERENT OBJECT from the one the engine allocated -- which is the whole failure mode the
        # declaration exists to end: anything that took a reference before the seed ran (the
        # recorder, a probe, the salvage) would keep writing into an orphan. So the seed writes
        # THROUGH the existing table when there is one, and only creates one for a spec that has
        # not declared it yet, which is nearly all of them.
        if self.type_layout.startswith("split_"):
            from plexus.engine import retype
            _ax = "xyz".index(self.type_layout[-1])
            _cl = H.level(self.cell_set) if self.cell_set in H.levels else None
            _fr = getattr(_cl, "_type_fracs", None) if _cl is not None else None
            if _cl is None or _fr is None or int(_fr.numel()) < 2:
                print(f"[mesh_seed] type_layout={self.type_layout} ignored: set "
                      f"{self.cell_set!r} declares fewer than two types", flush=True)
            else:
                order = torch.argsort(centroid[:, _ax])              # faces, sorted along the axis
                nt = torch.zeros(int(_cl.n), dtype=torch.long, device=dev)
                cuts = (torch.cumsum(_fr / _fr.sum(), 0) * nF).round().long().tolist()
                lo = 0
                for tid, hi in enumerate(cuts):
                    nt[order[lo:min(int(hi), nF)]] = tid
                    lo = int(hi)
                retype(_cl, nt)
                _names = list(getattr(_cl, "type_names", []) or [])
                _tally = "  ".join(f"{(_names[t] if t < len(_names) else t)}={int((nt[:nF] == t).sum())}"
                                   for t in range(int(_fr.numel())))
                print(f"[mesh_seed] type_layout={self.type_layout}: {nF} cells cut at the "
                      f"{'xyz'[_ax]} equator -> {_tally}", flush=True)
        seeded = dict(E_srce=est, E_trgt=ett, E_face=eft, nF=nF, Nv=Nv,
                         centre=tuple(float(v) for v in self.centre),   # the wedge apex, see face_geometry_3d
                         A0=(torch.full((nF,), A0, dtype=dt, device=dev) if g is None
                             else A_seed * g),
                         P0=(torch.full((nF,), P0, dtype=dt, device=dev) if g is None
                             else self.p0 * (A_seed * g).sqrt()),
                         alive=torch.ones(nF, dtype=dt, device=dev),
                         age=torch.as_tensor(ag, dtype=dt, device=dev),      # per-cell cell-cycle PHASE at t=0
                         # THE VOLUME AND RADIUS TARGETS FOLLOW `a0_scale`, on the isotropic rescale
                         # it implies: a preferred AREA g times larger is a preferred LENGTH sqrt(g)
                         # larger, hence a volume g^1.5 larger. Leaving V0f at the seeded value would
                         # set `K_V` against the very expansion `a0_scale` asks for, so the shell
                         # would stall short of its area target and the gate would fail on a
                         # constraint the paper's 2D model does not even have. `Vbirth` follows V0f
                         # for the same reason: a cell must still divide at twice its OWN target,
                         # not at twice a volume it was never meant to hold.
                         # `v0_uniform` -- EVERY CELL ASKS FOR THE SAME VOLUME, which is the
                         # control this seed could not express. `A0` and `P0` are already scalars
                         # here (`torch.full`), so the target AREA is uniform and only the target
                         # VOLUME is per-cell, taken from each cell's own seeded wedge. That
                         # asymmetry is invisible until you want to ask whether the mechanics
                         # equalises cell volume on its own: with per-cell targets it cannot, because
                         # every cell is already at the size it was asked for, and the answer would
                         # be about the seed rather than about the energy. Setting them all to the
                         # MEDIAN makes the question well posed.
                         V0f=(torch.full_like(vf, float(vf.median())) if self.v0_uniform
                              else vf.detach().clone()) * (self.a0_scale ** 1.5 if g is None else g ** 1.5),
                         V0=float(vf.sum()) * self.a0_scale ** 1.5,
                         v_ref=float(vf.median()) * self.a0_scale ** 1.5,   # REFERENCE cell volume (Okuda v_ref) -> uniform cells:
                         #   morphogen growth caps v_eq at (4/3)v_ref, cells cycle in [2/3,4/3]v_ref centred on v_ref
                         R0=float(np.linalg.norm(verts - np.asarray(self.centre, verts.dtype),
                                                axis=1).mean()) * self.a0_scale ** 0.5, verts0=verts)
        m = getattr(lvl, "_mesh", None)
        if isinstance(m, MeshTable):
            m.clear(); m.update(seeded)          # the engine's table, filled in place
        else:
            lvl._mesh = MeshTable(**seeded)      # spec has no `mesh:` declaration yet
        # THE PER-CELL BLOCKS GO ON THE CELL SET, not into `seeded`. `Vbirth` is the volume this
        # cell was born holding and `divjit` its personal multiplier on the division threshold;
        # both are one number per cell, so plexus2 puts them on the set whose members are cells,
        # where `Hierarchy.renumber_set` carries them through every topology edit without any
        # operator naming them. A spec that declares neither gets a clear refusal rather than a
        # silent zero -- which is what the mesh table gave, and what `divjit`'s fixed-seed fallback
        # was papering over.
        _seed_cell_blocks(H, self.cell_set, nF, dt, dev,
                          Vbirth=vf.detach().cpu().numpy() * (self.a0_scale ** 1.5 if g is None
                                                              else g.detach().cpu().numpy() ** 1.5),
                          divjit=np.asarray(dj, np.float64))
        # IDENTITY (R4 of notes/size_cycle/SIZE_CYCLE_PLAN.md). `cell_id` is a number a cell keeps
        # for its whole life and no other cell ever has; `parent_id` is its mother's, -1 for a
        # seeded cell. Optional blocks: a spec that declares them gets lineage that survives every
        # renumbering, and a report reads births and divisions off the ids instead of arguing
        # from index stability. A spec that does not declare them is untouched.
        if cell_block(H, self.cell_set, "cell_id", nF) is not None:
            set_cell_block(H, self.cell_set, "cell_id", np.arange(nF, dtype=np.float64), nF)
            m["next_id"] = int(nF)
        if cell_block(H, self.cell_set, "parent_id", nF) is not None:
            set_cell_block(H, self.cell_set, "parent_id", np.full(nF, -1.0), nF)
        return {}


@register_operator("seed_mesh", "mesh_seed", set="vertex", kind="seed", family="seed",
                   implementation="settle", title="build the cell mesh and settle it", equation=r"""$$\text{settling}(t)=\big[t<t_{\mathrm{ref}}\big]$$""")
class SeedMeshSettle(SeedMesh3D):
    r"""`settle` implementation of seed_mesh: the default mesh plus a settle window during which the size rules hold.
    vertex -> vertex: as the default seed_mesh; `settling(H, m)` is true for the first `ref_frame` frames, and cell_divide and cell_cycle hold while it is.

        \text{settling}(t)=\big[t<t_{\mathrm{ref}}\big]
        t              the frame
        t_{ref}        `ref_frame`, the settle window, in frames

    Reference: none -- a numerical settle window. Plexus (this work)."""
    PARAM_UNITS = {'ref_frame': 'frames'}
    SYMBOLS = {'t': 'the frame', 't_{ref}': '`ref_frame`, the settle window, in frames'}
    PARAM_ROLES = {'ref_frame': 'frames_the_tissue_relaxes_before_the_size_rules_read_it'}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.ref_frame = int(params.get("ref_frame", 0))

    def forward(self, H, mask=None):
        out = super().forward(H, mask)
        m = getattr(H.level(self.at), "_mesh", None)
        if m is not None:
            m["ref_frame"] = self.ref_frame              # read by cell_divide / cell_cycle
        return out


@register_operator("seed_mesh", "mesh_seed", set="vertex", kind="seed", family="seed",
                   model="apicobasal", title="build a thick cell mesh", equation=r"""$$\mathbf s_i=\tfrac{h_0}{2}\,\mathbf n_i,\qquad \mathbf a_i=\mathbf x_i+\mathbf s_i,\quad \mathbf b_i=\mathbf x_i-\mathbf s_i$$""")
class SeedMeshApicoBasal(SeedMesh3D):
    r"""`apicobasal` implementation of seed_mesh: the same epithelium, seeded with its thickness as a per-vertex half-separation block.
    vertex -> vertex: as the default seed_mesh, plus the per-vertex block `sep` (half the apico-basal separation along the vertex normal) and the targets measured on the doubled surface.

        \mathbf s_i=\tfrac{h_0}{2}\,\mathbf n_i,\qquad \mathbf a_i=\mathbf x_i+\mathbf s_i,\quad \mathbf b_i=\mathbf x_i-\mathbf s_i
        s_i            `sep`, the vertex's half-separation
        h_0            `h0`, the full cell thickness, in world units
        n_i            the vertex normal
        a_i, b_i       the apical and basal positions of vertex i
        x_i            the mid-surface position

    Reference: Okuda, S. et al. (2013). Biomech. Model. Mechanobiol. 12(4):627-644. Plexus apico-basal promotion R2 (this work)."""
    PARAM_UNITS = {'h0': 'length', 'ref_frame': 'frames'}
    SYMBOLS = {'s_i': "`sep`, the vertex's half-separation", 'h_0': '`h0`, the full cell thickness, in world units', 'n_i': 'the vertex normal', 'a_i, b_i': 'the apical and basal positions of vertex i', 'x_i': 'the mid-surface position'}
    PARAM_ROLES = {'h0': 'full_cell_thickness', 'sep_block': 'name_of_the_half_separation_block', 'apical': 'which_side_is_apical', 'v0_from': 'how_the_target_volume_is_measured', 'ref_frame': 'settle_window_in_frames'}
    # WHAT THIS SEED IMPOSES, overriding the mid-surface parent's wedge. A cell with a separation
    # IS a polyhedron, so its target volume is one -- which is exactly what
    # `cell_mechanics[apicobasal]` defends, and the two now agree at load.
    #
    # A CLASS ATTRIBUTE AND NOT AN INSTANCE ONE. The checker resolves an operator to its CLASS
    # before anything is constructed -- `schema.load` has no instances yet -- so a `BLOCK_UNITS`
    # assigned in `__init__` is invisible to it, which is how the first attempt at this kept
    # reporting the disagreement it had just fixed. A spec that opts back to `v0_from: wedge` for
    # an archived run therefore still reads `polyhedron` here and gets warned; that is the correct
    # outcome, because such a run genuinely IS inconsistent.
    BLOCK_UNITS = {"V0f": "volume[polyhedron]", "Vbirth": "volume[polyhedron]",
                   "A0": "area[midsurface]", "P0": "length"}

    @classmethod
    def block_units(cls, params):
        """The convention this seed imposes, WHICH THE SPEC CAN STILL CHOOSE.

        `v0_from: wedge` is a real composition and not a legacy escape: `gate_ab_sphere` is the
        mid-surface mechanics carrying a separation nothing reads, so its cells genuinely are
        wedge-targeted and warning about them would be the checker inventing a defect. The default
        is `polyhedron` because a cell with a separation normally IS one.
        """
        conv = "polyhedron" if str(params.get("v0_from", "wedge")).lower() == "polyhedron" else "wedge"
        return {"V0f": f"volume[{conv}]", "Vbirth": f"volume[{conv}]",
                "A0": "area[midsurface]", "P0": "length"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.h0 = float(params.get("h0", 0.4))                 # FULL thickness; sep is h0/2
        self.sep_block = str(params.get("sep_block", "sep"))
        # WHICH WAY THE APICAL SIDE FACES. `out` (default, every archived run): apical = pos + sep
        # with sep along the outward normal, so a closed vesicle is apical-out and its basal
        # surface faces the lumen. `in`: the sign of sep is flipped, apical faces the lumen and
        # the basal surface faces out -- the polarity of a cyst in a matrix, where a basement
        # membrane sits. The energy is unchanged: `cell_mechanics[apicobasal] surface:` names the
        # ring the belt acts on, so `apical: in` + `surface: basal` is the archived spheroid with
        # its labels swapped.
        self.apical_side = str(params.get("apical", "out")).lower()
        if self.apical_side not in ("out", "in"):
            raise ValueError(f"seed_mesh[apicobasal]: apical must be out|in, not {self.apical_side!r}")
        # WHICH VOLUME THE CELL'S TARGET IS EXPRESSED IN -- see the block at the end of `forward`.
        #
        # `polyhedron` IS THE DEFAULT NOW, AND IT IS THE ONLY SELF-CONSISTENT ONE. This shipped as
        # `wedge` to keep archived runs reading identically, and that made every apicobasal spec in
        # the tree ask its cells for a volume they cannot have: `cell_mechanics[apicobasal]` pulls
        # the POLYHEDRON volume toward `V0f` while the seed filled `V0f` with a WEDGE one -- 1.3508
        # against 2.5433 for the same cell on the reference spheroid. A spec may still write
        # `v0_from: wedge` to reproduce an archived run, and the loader will then tell it what it
        # disagrees with.
        # THE DEFAULT IS `wedge`, AND IT IS A WORKING-POINT DECISION. c671fb31 flipped it to
        # `polyhedron` and thereby re-targeted every apico-basal spec that had never named a
        # convention -- about eighty of them, cvd2_adder_tension included: V0f 2.57 -> 1.36 on
        # the reference spheroid, the shell shrank from r 4.33 to 3.65 by frame 3 and division
        # started 300 frames early. A spec that wants the polyhedron says `v0_from: polyhedron`
        # (the apop2_abfix_* family does); a default may not move a corpus.
        self.v0_from = str(params.get("v0_from", "wedge")).lower()
        # `ref_frame` -- THE FRAME AT WHICH THE SIZE RULES FIRST READ THE TISSUE. A seeded shell
        # is not at rest in its polyhedron volume: the energy's rest offset inflates every cell
        # over the opening frames (x1.4-1.7 on the reference spheroid, settling in 30-50 frames,
        # `tools/equilibrium_h.py`), and a reference taken on that ramp is a different number on
        # every spec. Until this frame `cell_divide` and `cell_cycle` hold; at it, `v_ref` is
        # cached and every seeded cell's `Vbirth` becomes the volume it actually has. Declared,
        # so it is the same frame on every run; 0 (the default) is the old behaviour.
        self.ref_frame = int(params.get("ref_frame", 0))
        if self.v0_from not in ("wedge", "polyhedron"):
            raise ValueError(f"seed_mesh[apicobasal]: v0_from must be wedge|polyhedron, "
                             f"not {self.v0_from!r}")

    def forward(self, H, mask=None):
        out = super().forward(H, mask)                         # pos, occ and the mesh table
        lvl = H.level(self.at)
        if self.sep_block not in lvl.state_schema:
            raise ValueError(
                f"seed_mesh[apicobasal] needs a `{self.sep_block}` block on the {self.at!r} set -- "
                f"declare it under `sets:` as `{self.sep_block}: {{width: 3, integration: "
                f"first_order}}`. Without it there is nowhere to put the apico-basal separation and "
                f"the run would be the mid-surface model wearing a different operator's name.")
        # THE SPATIAL SCHEMA MUST STILL BE THE SPATIAL SCHEMA. `_resolve_schema` says "the spec
        # always wins", so a `state:` block REPLACES the set's schema rather than extending it, and
        # `schema_from_spec` defaults every block to first_order/free. A spec that declares `sep`
        # and restates `pos` without `boundary: world` runs perfectly well and silently loses the
        # world clamp -- positions stop being confined and nothing says so. Checked here because
        # this operator is the only thing that knows the spec meant to keep a spatial set spatial.
        _coord = lvl.state_schema.coordinate
        if _coord is None or _coord.name != "pos":
            raise ValueError(
                f"seed_mesh[apicobasal]: the {self.at!r} set's coordinate block is "
                f"{_coord.name if _coord else None!r}, not `pos`. Declaring a `state:` block "
                f"REPLACES the schema; restate `pos` as "
                f"{{width: 3, role: coordinate, integration: coordinate, "
                f"boundary: world}} alongside `{self.sep_block}`.")
        if _coord.boundary != "world":
            raise ValueError(
                f"seed_mesh[apicobasal]: `pos` has boundary {_coord.boundary!r}, not 'world'. The "
                f"spec's `state:` block dropped the world clamp -- the run would not be confined "
                f"and nothing else would say so.")
        m = getattr(lvl, "_mesh", None)
        Nv = int(m["Nv"])
        pos = lvl.get("pos")[:Nv]
        # THE SAME NORMAL THE ENERGY USES, not a second definition of it. `monolayer_shells`
        # returns (apical, basal, vertex_normal, vertex_thickness) and its docstring is explicit
        # about why one formula with two callers beats two copies: "Two copies of 'offset along the
        # vertex normal by h/2' would agree until one of them changed, and the picture would then
        # stop being a picture of the model." The thickness argument is a dummy here -- only the
        # normal is taken -- because at R2 the separation is uniform and `h0` sets it directly.
        nF = int(m["nF"])
        _h = torch.full((nF,), self.h0, dtype=pos.dtype, device=pos.device)
        _a, _b, n, _hv = monolayer_shells(pos, m["E_srce"], m["E_trgt"], m["E_face"], nF, _h)
        c0, c1 = lvl.state_schema[self.sep_block]
        st = lvl.state.clone()
        st[:, c0:c1] = 0.0
        st[:Nv, c0:c1] = (0.5 * self.h0) * n * (-1.0 if self.apical_side == "in" else 1.0)
        lvl.state = st
        # DECLARED FOR THE CARRY, so a vertex born on a septum inherits its parents' separation
        # instead of the buffer's zero -- a cell of zero height along the seam it just grew.
        # R1(b) built `carry_vertices`; this is its first real consumer.
        declare_vertex_carry(m, self.sep_block)
        print(f"[seed_mesh:apicobasal] {Nv} vertices carry an apico-basal separation of "
              f"h0/2 = {0.5 * self.h0:g} along the vertex normal (cell thickness {self.h0:g})",
              flush=True)
        # `v0_from: polyhedron` -- SEED THE TARGET IN THE CONVENTION THE ENERGY MEASURES.
        #
        # THE DEFECT IT REMOVES. The parent seeds `V0f` and `v_ref` from `face_geometry_3d`'s WEDGE
        # volume -- a sum of origin-referenced cones over the mid-surface ring -- because that is
        # what the mid-surface model's `K_V` integrates. An apico-basal cell is not that solid: it
        # is the polyhedron of two caps and one wall per ring edge, and `cell_mechanics[apicobasal]`
        # pulls THAT volume toward `V0f` with `k_v`. The two differ by a fixed factor for the same
        # cell -- measured on the reference spheroid, wedge 2.5433 against polyhedron 1.3508 -- so
        # every cell is born asking for a volume it does not have, and the tissue inflates until the
        # polyhedron reaches a wedge-sized target.
        #
        # MEASURED, WITH NO GROWTH OPERATOR SCHEDULED AT ALL: `apop2_ab_half` runs its mean radius
        # from 5.03 to 6.06 in 300 frames, a swelling of 1.20x that no operator in the spec asks
        # for. It also masks the mechanism the spec exists to show -- killing 691 of 2000 cells
        # holds the radius at 1.02x of its seed, where the same deaths on the mid-surface sheet
        # contract it to 0.85x.
        #
        # DEFAULT False, so no archived apicobasal run changes meaning. It is a seed-time choice
        # and belongs here: `v_ref` is read by `cell_die`'s extrusion threshold and `cell_grow`'s
        # ceiling, so setting both from one convention is what makes those thresholds mean the size
        # they say. (The wider reconciliation -- one volume for growth, division, death AND the
        # energy -- is AB_R7R8_TODO section 0a and is not settled here.)
        m["v0_from"] = self.v0_from                  # read by cell_size / size_ref
        m["ref_frame"] = self.ref_frame              # read by cell_divide / cell_cycle / cell_size
        if self.v0_from == "polyhedron":
            _wedge = float(m.get("v_ref", 0.0))     # what the mid-surface model would have used
            vp, _ap, _cp, _hp = apicobasal_geometry_3d(
                lvl.get("pos")[:Nv].detach(), lvl.get(self.sep_block)[:Nv].detach(),
                m["E_srce"], m["E_trgt"], m["E_face"], nF)
            vp = vp.detach().to(device=m["V0f"].device, dtype=m["V0f"].dtype)
            m["V0f"] = (torch.full_like(vp, float(vp.median())) if self.v0_uniform else vp.clone())
            m["V0"] = float(m["V0f"].sum())
            m["v_ref"] = float(vp.median())
            # AND THE SHARED REFERENCE, from frame 0. `cell_divide` used to cache `v_ref_poly` at
            # its own first call and `cell_grow`, `cell_die` and `cell_cycle` each read the wedge
            # one; they all read this now (`cell_size`), so there is a single seed-time median in
            # the convention the energy defends and the four cannot drift apart.
            # THE POLYHEDRON REFERENCE IS NOT WRITTEN HERE WHEN THE SEED DECLARES A SETTLE WINDOW.
            # Pre-filled with the seed-time median (1.35 on the reference spheroid), it was found
            # cached by `cell_divide` and `cell_size`, which then never took theirs at `ref_frame`
            # -- every size threshold was stated against a volume the cells had already inflated
            # past (settled median 2.59): the R1b sizer divided at 1.05 x the settled median, i.e.
            # at 2 x the seed's. With `ref_frame > 0` the first reader after the window caches the
            # reference and resets `Vbirth`.
            if self.ref_frame <= 0:
                m["v_ref_poly"] = float(vp.median())
            vb = cell_block(H, resolve_cell_set(H, self.at), "Vbirth", nF)
            if vb is not None:
                set_cell_block(H, resolve_cell_set(H, self.at), "Vbirth",
                               m["V0f"].detach().cpu().numpy().astype(np.float64), nF)
            print(f"[seed_mesh:apicobasal] V0f and v_ref seeded from the POLYHEDRON volume "
                  f"(median {float(vp.median()):.4f}) rather than the wedge volume "
                  f"({_wedge:.4f}) -- the target is now in the convention `k_v` measures",
                  flush=True)
        return out


@register_operator("cell_mechanics", set="vertex", kind="lateral", family="mechanics", title="relax cell shapes",
                   equation=r"""$$E=\sum_f\Big[K_A\big(A_f-A^{0}\big)^2+K_P\big(P_f-P^{0}\big)^2+K_V\big(v_f-v^{\mathrm{eq}}_f\big)^2\Big]+\Lambda\sum_e\ell_e,\qquad \mathbf f=-\nabla E$$""")
class ShapeEnergy3D(Lateral):
    r"""the 3D vertex-model shape energy of the vesicle, relaxed by gradient descent on the vertices each frame.
    vertex -> vertex: reads pos and the per-face targets (A0, P0, V0f), emits the vertex velocity -mu grad E.

        E=\sum_f\Big[K_A\big(A_f-A^{0}\big)^2+K_P\big(P_f-P^{0}\big)^2+K_V\big(v_f-v^{\mathrm{eq}}_f\big)^2\Big]+\Lambda\sum_e\ell_e,\qquad \mathbf f=-\nabla E
        E              the shape energy
        K_A, K_P, K_V  area, perimeter and per-cell volume stiffnesses
        A_f, P_f, v_f  face f's area, perimeter and wedge volume
        A^0, P^0, v^{eq}_f their targets
        \Lambda        the line tension on every edge
        \ell_e         edge length
        f              the force on the vertices, -grad E

    Reference: Farhadifar, R. et al. (2007). Curr. Biol. 17:2095-2104 (the energy); Okuda, S. et al. (2013). Biomech. Model. Mechanobiol. 12(4):627-644."""
    PARAM_UNITS = {'K_A': 'stress', 'K_P': 'force', 'K_V': 'stress', 'K_R': 'stress', 'K_bend': 'energy', 'K_lumen': 'stress', 'Lambda': 'force', 'Gamma': 'force', 'p0': 'dimensionless', 'mu': 'dimensionless', 'dt': 'time', 'eta': 'dimensionless', 'cap_frac': 'fraction', 'relax_iters': 'count', 'smooth_iters': 'count', 'smooth_w': 'fraction', 'antiinv': 'dimensionless', 'centre': 'length'}
    SYMBOLS = {'E': 'the shape energy', 'K_A, K_P, K_V': 'area, perimeter and per-cell volume stiffnesses', 'A_f, P_f, v_f': "face f's area, perimeter and wedge volume", 'A^0, P^0, v^{eq}_f': 'their targets', '\\Lambda': 'the line tension on every edge', '\\ell_e': 'edge length', 'f': 'the force on the vertices, -grad E'}
    SUPPORTED_DIMS = [3]; EMIT = "velocity"; DIFFERENTIABLE = True
    # FORCED BY THE ENERGY, NOT CHOSEN. With E = sum_f [K_A(A-A0)^2 + K_P(P-P0)^2 + K_V(v-v_eq)^2]
    # + Lam sum_e l_e an energy (F*L): K_A L^4 = F L gives K_A = F/L^3, K_P L^2 = F L gives F/L,
    # K_V L^6 = F L gives F/L^5, and Lam L = F L gives Lam = F -- a line tension IS a force.
    # `mu` multiplies a force to make a velocity, so it is a mobility, L/(F*T).
    # THE MID-SURFACE MODEL DEFENDS THE WEDGE VOLUME, as this class's own docstring says: "K_V is a
    # PER-CELL volume elasticity on each cell's wedge volume v_f". So `V0f` here IS a wedge volume,
    # and that is consistent with the seed.
    BLOCK_UNITS = {"V0f": "volume[wedge]", "A0": "area[midsurface]", "P0": "length"}
    REQUIRES_PARAMS = ["p0"]
    INPUTS = ["vertex"]; OUTPUTS = ["vertex"]; READS = ["pos"]; WRITES = ["pos"]
    MECHANISM_TAGS = ["vertex_model", "shape_energy", "cell_volume_elasticity", "vesicle", "force_balance"]
    # THE SECOND CITATION WAS THE WRONG OKUDA 2015, AND THE PAGES WERE WRONG TOO. It read
    # "Biomech. Model. Mechanobiol. 14:413-421 (3D volume/surface)": that paper is 413-425,
    # and it is the VISCOSITY paper -- local velocity fields to make vertex dynamics Galilean
    # invariant -- which this operator does not implement. The 3D volume/surface energy is the
    # other Okuda 2015, in Biophysics and Physicobiology.
    REFERENCE = ("Farhadifar, R., Roper, J.-C., Aigouy, B., Eaton, S. & Julicher, F. (2007). "
                 "Curr. Biol. 17(24):2095-2104, doi:10.1016/j.cub.2007.11.049 -- the vertex-model "
                 "shape energy (area, perimeter, line tension) this generalises; "
                 "Okuda, S., Inoue, Y. & Adachi, T. (2015). Three-dimensional vertex model for "
                 "simulating multicellular morphogenesis. Biophys. Physicobiol. 12:13-20 -- the "
                 "3D volume/surface form. NOT an Active Vertex Model: see the class docstring.")
    PARAM_ROLES = {"p0": "target_shape_index", "K_A": "area_stiffness", "K_P": "perimeter_stiffness",
                   "Lambda": "surface_tension", "K_V": "cell_volume_elasticity", "cap_frac": "stability_cap",
                   "centre": "origin_of_the_K_R_radial_term", 'Gamma': 'cortical_contractility_on_the_perimeter_squared', 'K_R': 'radial_restoring_stiffness_toward_the_sphere', 'K_bend': 'bending_stiffness', 'K_lumen': 'global_lumen_volume_stiffness', 'antiinv': 'penalty_against_inverted_faces', 'dt': 'relaxation_time_step', 'eta': 'step_size_of_the_relaxation', 'mu': 'mobility_of_the_vertices', 'plane_axis': 'normal_axis_of_a_flat_sheet', 'relax_iters': 'relaxation_iterations_per_frame', 'smooth_iters': 'laplacian_smoothing_sweeps', 'smooth_w': 'weight_of_the_smoothing'}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.K_A = float(params.get("K_A", 1.0)); self.K_P = float(params.get("K_P", 1.0))
        self.p0 = float(params.get("p0", 3.9)); self.Lambda = float(params.get("Lambda", 0.1))
        self.K_V = float(params.get("K_V", 0.5)); self.K_R = float(params.get("K_R", 0.0))
        _c = params.get("centre", None)
        self._centre = None if _c is None else torch.tensor([float(v) for v in _c])
        # DIHEDRAL bending (Wardetzky): penalises adjacent-cell normal deviation -> smooths the local folds
        # the hollow metric flags (division-injected cap tilts). High-frequency: unlike the radial K_R it
        # does NOT flatten gentle whole-shell/tube curvature. 0 = off (default). See _shape_energy_core.
        self.K_bend = float(params.get("K_bend", 0.0))
        # GLOBAL LUMEN incompressibility (isoperimetric): penalise enclosing less volume than a sphere of
        # the current area -> distinguishes sphere from a per-cell-volume-preserving buckle. 0=off. Coral only.
        self.K_lumen = float(params.get("K_lumen", 0.0))
        self.Gamma = float(params.get("Gamma", 0.0))             # cortical contractility (1/2)Gamma*P^2 -> rounds cells
        self.Gam_l = 0.0                # per-JUNCTION (1/2)Gamma*l^2; set by model=marinari, not by a spec
        self.mu = float(params.get("mu", 1.0))
        self.dt = float(params.get("dt", 1.0)); self.relax_iters = int(params.get("relax_iters", 6))
        self.eta = float(params.get("eta", 0.08)); self.cap_frac = float(params.get("cap_frac", 0.12))
        # ANTI-INVERSION filtered step (IPC-analog, differentiable): hollow caps are inverting faces
        # (signed wedge volume v_f flips sign at the division septum). Each bounded-Euler substep, scale
        # back the move of any vertex whose incident face would drop v_f below `antiinv` x median(v_f) --
        # a move that only makes an already-inverted face WORSE is blocked, a recovering move is allowed.
        # Straight-through (scale detached) so the rollout stays differentiable. 0 = off (default).
        # `plane_axis` -- CONSTRAIN THE SOLVER TO A PLANE, which is what makes this a 2D vertex
        # model. The ENERGY is unchanged; only the descent is projected, so a flat sheet stays in
        # its plane by construction rather than by a restoring force fighting it.
        #
        # A PENALTY CANNOT DO THIS JOB. This loop takes `relax_iters` free 3D steps per frame and
        # an external penalty corrects only once afterwards, so the sheet undulates by many times
        # the declared gap in between. Raising the penalty stiffness does not help: an explicit
        # penalty is stability-limited exactly where it would start to bite, and the run diverges
        # before it becomes strong enough to hold the sheet flat.
        #
        # None (the default) leaves every existing run untouched.
        _pa = params.get("plane_axis", None)
        self.plane_axis = None if _pa is None else int(_pa)
        self.antiinv = float(params.get("antiinv", 0.0))
        # Lloyd-like tangential regularization (vertex-model analog of Turing's surface_lloyd): rounds cells
        self.smooth_iters = int(params.get("smooth_iters", 0)); self.smooth_w = float(params.get("smooth_w", 0.0))

    def _antiinv_scale(self, x, step, es, et, ef, nF, eocc, vf_ref, floor):
        """Straight-through per-vertex step scale (detached) that halves the move of any vertex whose
        incident face would drop its signed wedge volume below `floor` AND below its current value ---
        i.e. block moves that push a face toward inversion, allow recovering moves. A few backtracks."""
        scale = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        for _ in range(5):
            _, _, _, vf = face_geometry_3d(x + step * scale, es, et, ef, nF, eocc, apex=getattr(self, "_apex", None))
            bad = (vf < floor) & (vf < vf_ref)                   # per-face: inverting and getting worse
            if not bool(bad.any()):
                break
            badv = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            badv.index_add_(0, es, bad.to(x.dtype)[ef] * eocc)   # half-edge src vertices of the bad faces
            scale = torch.where((badv > 0)[:, None], scale * 0.5, scale)
        return scale.detach()

    def _grad(self, p, es, et, ef, nF, A0, P0, V0f, alive, R0t, eocc, vocc, twin_face=None,
              myo_e=None):
        """dE/d(pos), through the hand-written warp kernels when they apply, autograd otherwise.

        WARP IS THE DEFAULT. The energy is unchanged -- `vertex_warp` implements the derivative OF
        `_shape_energy_core`, checked term by term against `torch.autograd.grad` in
        `tests/test_vertex_warp.py` at a relative error of ~3.5e-07, which is float32 round-off.
        What changes is the cost: the autograd backward was the single largest entry in the frame's
        profile (25.6 s of 85.8 s on `mesh_mpm_spheroid_nominal` at frame 380), and it is
        LAUNCH-bound rather than arithmetic-bound -- 3.27 ms per call whether the mesh has 1,188
        half-edges or 71,988. The warp path is 0.40 ms at every one of those sizes: 8.1x on the
        gradient, 3.5x on the whole frame (120.5 -> 34.1 ms/frame end to end).

        IT FALLS BACK RATHER THAN REFUSING, and says so once: on CPU, in float64, without warp
        installed, or when `K_bend` / `K_lumen` is on -- those two terms are not ported, and a term
        silently dropped from a gradient is a different model, not a faster one. The fallback is the
        ONLY way the autograd route is reached; there is no operator key that forces it, because
        which route differentiates one and the same energy is a backend choice and not a model.

        NOT BIT-IDENTICAL, and neither is the thing it replaces. Both accumulate per-face sums with
        float32 atomics whose order is not fixed -- warp's `atomic_add` and torch's `index_add` --
        so two runs of the UNMODIFIED default already differ from each other.
        """
        # LOCAL AND LATE: `try_shape_energy_grad` is defined at the BOTTOM of this file, below the
        # class that calls it, so a module-level import would be a forward reference.
        from plexus.operators.vertex_ops import try_shape_energy_grad
        g = try_shape_energy_grad(self, p, es, et, ef, nF, A0, P0, V0f, alive, R0t,
                                  eocc, vocc, twin_face, myo_e)
        if g is not None:
            return g
        with torch.enable_grad():
            p = p.detach().requires_grad_(True)
            # THE RADIAL TERM'S ORIGIN, AND WHY IT HAS TO BE DECLARABLE. `K_R` penalises
            # (|pos| - R0)^2 with |pos| measured from the WORLD ORIGIN, so an energy that reads as
            # translation-invariant is not: place the tissue at [25,25,25] of a 50-unit box and every
            # vertex sits 43 units out against an R0 of 4.65, and the term tears the shell apart --
            # 685 cells instead of 6,786, measured. Every other term (areas, perimeters, volumes,
            # edge lengths) IS translation-invariant, so evaluating the whole energy on `p - centre`
            # changes only this one and leaves the gradient of the rest identical.
            #
            # DEFAULT [0, 0, 0], so the 461 specs that never asked are bit-identical.
            if self._centre is not None:
                p_e = p - self._centre.to(p.device, p.dtype)
            else:
                p_e = p
            E = _shape_energy_core(p_e, es, et, ef, nF, A0, P0, V0f, alive, R0t, self.K_A, self.K_P,
                                   self.K_V, self.K_R, self.Lambda, self.Gamma, eocc, vocc,
                                   self.K_bend, twin_face, self.K_lumen, myo_e, self.Gam_l,
                                   apex=getattr(self, "_apex", None))
            g = torch.autograd.grad(E, p)[0]
        return torch.nan_to_num(g)

    def _grad_myo(self, m, *a, **kw):
        """`_grad` with the mesh's per-junction myosin, if a junction operator has supplied one.

        Read through the MESH rather than passed down the call chain, because the operator that writes it
        (`junction_ops.junction_myosin`) runs at a different point in the schedule and the two never see
        each other. Absent -> None -> the energy reduces to the scalar-Lambda form exactly.
        """
        myo = m.get("myo") if isinstance(m, dict) else None
        if myo is not None and myo.shape[0] != a[1].shape[0]:
            # A FRAME RELAXED WITH NO MYOSIN AT ALL, which is the operator being off rather than the
            # operator being approximated -- and it used to happen in silence. It should now be
            # unreachable: `junction_sync` re-keys the array after every topology operator, so a
            # mismatch here means the sync is missing from the schedule or is placed before the operator
            # that resized the buffer. Counted and announced once rather than raised, because raising
            # would take down every archived specification that predates the sync operator.
            MYO_SKIPPED.append((int(myo.shape[0]), int(a[1].shape[0])))
            if len(MYO_SKIPPED) == 1:
                print(f"[cell_mechanics] myosin array is {myo.shape[0]} long against "
                      f"{a[1].shape[0]} half-edges -- relaxing WITHOUT myosin this frame. Schedule "
                      f"`junction_sync` after the topology operators.", flush=True)
            myo = None
        return self._grad(*a, myo_e=myo, **kw)

    @staticmethod
    def _twin_faces(es, et, ef, Nv):
        """Per half-edge -> the face on the OTHER side of that (undirected) edge, for the dihedral term.
        Twin of half-edge (u->v) is the half-edge (v->u); match by integer key. Closed mesh: always found."""
        key = es * Nv + et; twinkey = et * Nv + es
        order = torch.argsort(key); ks = key[order]
        pos = torch.searchsorted(ks, twinkey).clamp(max=key.shape[0] - 1)
        found = ks[pos] == twinkey
        return torch.where(found, ef[order[pos]], ef)          # fallback to self (no penalty) if no twin

    def forward(self, H, mask=None):
        lvl = H.level(self.at); pos_full = lvl.get("pos")
        v_full = torch.zeros_like(pos_full)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {self.at: v_full}
        Nv = int(m["Nv"]); nF = int(m["nF"]); es = m["E_srce"]; et = m["E_trgt"]; ef = m["E_face"]
        E = es.shape[0]; dev = pos_full.device; dt = pos_full.dtype
        # `plane_axis` TRAVELS WITH THE REST. `cell_divide` heals the two fresh daughters and
        # their one-ring with `_relax_subset`, a SECOND descent loop reading these numbers -- and
        # a constraint the main loop honours and that one does not is a constraint the run does
        # not have. Measured: with the main relax projected into the plane and this omitted, the
        # sheet still drifted to sd(z) = 5.5% of an edge, and the drift tracked division activity.
        m["mech"] = dict(K_A=self.K_A, K_P=self.K_P, K_V=self.K_V, K_R=self.K_R, Lambda=self.Lambda,
                         Gamma=self.Gamma, eta=self.eta, cap_frac=self.cap_frac,
                         plane_axis=self.plane_axis)   # for cell_divide local relax
        x0 = pos_full[:Nv].detach().clone()
        R0t = torch.as_tensor(float(m["R0"]), dtype=dt, device=dev)
        with torch.no_grad():
            cap = self.cap_frac * (x0[et] - x0[es]).norm(dim=-1).mean().clamp(min=1e-6)
        # Relax the LIVE vertices only: the energy is summed over the live faces and half-edges, so
        # `eocc` and `vocc` are all-ones here and the occupancy masking inside `_shape_energy_core`
        # is a no-op. (There is no padded-reservoir variant any more: whether the body below is
        # traced by a fusing compiler is the engine's `compile:` decision, not this operator's.)
        eocc = torch.ones(E, device=dev, dtype=dt); vocc = torch.ones(Nv, device=dev, dtype=dt)
        twin = self._twin_faces(es, et, ef, Nv) if self.K_bend > 0 else None   # dihedral neighbour faces
        x = x0.clone()
        self._apex = wedge_apex(m, x)                        # the wedge apex for this solve, see face_geometry_3d
        floor = None
        if self.antiinv > 0:                                 # anti-inversion floor = frac of median live wedge vol
            _, _, _, vf0 = face_geometry_3d(x, es, et, ef, nF, eocc, apex=self._apex)
            floor = self.antiinv * (vf0[vf0 > 0].median() if (vf0 > 0).any() else vf0.new_tensor(1e-9)).clamp(min=1e-9)
        for _ in range(max(1, self.relax_iters)):
            step = -(self.eta * self.mu) * self._grad_myo(m, x, es, et, ef, nF, m["A0"], m["P0"],
                                                      m["V0f"], m["alive"], R0t, eocc, vocc, twin)
            step = step * torch.clamp(cap / (step.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
            if floor is not None:                            # block any substep that drives a face toward inversion
                _, _, _, vf_cur = face_geometry_3d(x, es, et, ef, nF, eocc, apex=self._apex)
                step = step * self._antiinv_scale(x, step, es, et, ef, nF, eocc, vf_cur, floor)
            if self.plane_axis is not None:                   # 2D vertex model: no motion off the sheet
                step[:, self.plane_axis] = 0.0
            x = x + step
        if self.smooth_iters and self.smooth_w > 0:      # Lloyd-like tangential regularization -> rounder cells
            es, et = m["E_srce"], m["E_trgt"]
            ones = torch.ones(es.shape[0], device=x.device, dtype=x.dtype)
            with torch.no_grad():
                for _ in range(self.smooth_iters):
                    nbr = torch.zeros_like(x).index_add(0, es, x[et])          # sum of edge-neighbours per vertex
                    deg = torch.zeros(Nv, device=x.device, dtype=x.dtype).index_add(0, es, ones).clamp(min=1)
                    xs = (1 - self.smooth_w) * x + self.smooth_w * (nbr / deg[:, None])
                    x = xs * (x.norm(dim=1, keepdim=True) / xs.norm(dim=1, keepdim=True).clamp(min=1e-9))  # keep on shell
        # DIVIDED BY THE ENGINE'S dt, NOT BY THIS OPERATOR'S, so the two cancel BY CONSTRUCTION.
        #
        # This is a UNIT CONVERSION and nothing else: the relaxation above solved a DISPLACEMENT
        # (`x - x0`), using `eta`, `cap_frac` and `relax_iters`, none of which involve a timestep.
        # `EMIT = "velocity"` means the engine will apply `pos += v * general.dt`, so the only
        # divisor that returns the displacement actually solved is `general.dt`.
        #
        # IT USED TO DIVIDE BY `self.dt`, and the two cancelled in every spec ever written because
        # both were 1.0. The moment a spec lowered `general.dt` to co-schedule an MPM continuum --
        # 0.0032 -- and left this at 1.0, the tissue relaxed at 0.32% of the rate it had solved for:
        # a vesicle that reaches 7,814 cells reached 270. NOTHING LOOKED WRONG. The shell stayed
        # round, every shape diagnostic passed, the contact behaved; it read as a growth-rate
        # problem and it was a unit-of-time problem, and it took a spec-for-spec ablation to find.
        # A parameter that must always equal another parameter is not a parameter.
        #
        # BIT-IDENTICAL FOR THE CORPUS: 1,606 of the 1,610 operator lines that declare a `dt` in
        # `config/` already set it to their spec's `general.dt`, so the ratio was already 1. The
        # four that did not were the specs this was found in.
        _dt = float(getattr(H, "dt", self.dt) or self.dt)
        if "dt" in self.params and abs(float(self.params["dt"]) - _dt) > 1e-12 * max(_dt, 1.0) \
                and not getattr(self, "_dt_warned", False):
            self._dt_warned = True
            from plexus.paths import warn
            warn(f"[warn] cell_mechanics: `dt: {self.params['dt']}` is IGNORED -- the divisor is "
                 f"general.dt ({_dt}) so that the engine's own multiplication cancels it. Remove "
                 f"the parameter; leaving it in the spec suggests a knob that is not there.")
        v_full[:Nv] = (x - x0) / max(_dt, 1e-9)
        if mask is not None:
            v_full = v_full * mask[:, None].float()
        return {self.at: v_full}


@register_operator("cell_mechanics", implementation="shape_contact", set="vertex", kind="lateral",
                   family="mechanics", title="relax cell shapes against contacts", equation=r"""$$U=U_{\mathrm{shape}}(\mathbf x)-\sum_v\mathbf F_v\cdot\mathbf x_v+\sum_i\tfrac12 k\,p_i(\mathbf x)^2$$""")
class ShapeEnergyContact3D(ShapeEnergy3D):
    r"""the default `cell_mechanics`: the shape energy relaxed against the membrane's contacts and the matrix's load.
    vertex -> vertex (and -> each contact partner's set, a velocity): reads pos, the targets, the basal load buffer and the contact lists; emits the vertex velocity and returns each membrane node the force it pushed with.

        U=U_{\mathrm{shape}}(\mathbf x)-\sum_v\mathbf F_v\cdot\mathbf x_v+\sum_i\tfrac12 k\,p_i(\mathbf x)^2
        U_{shape}      the parent's shape energy, unchanged
        F_v            the matrix's frame-mean load on vertex v
        k              the contact stiffness
        p_i            the penetration of contact i, re-evaluated every relaxation iteration

    Reference: Okuda, S. et al. (2018). Sci. Rep. 8:2386 (the shape energy and its budding); Chen, Z. et al. (2015). Comput. Methods Appl. Mech. Engrg. 293:1-19 (the contact)."""
    PARAM_UNITS = {}
    SYMBOLS = {'U_{shape}': "the parent's shape energy, unchanged", 'F_v': "the matrix's frame-mean load on vertex v", 'k': 'the contact stiffness', 'p_i': 'the penetration of contact i, re-evaluated every relaxation iteration'}

    def _grad_myo(self, m, *a, **kw):
        g = super()._grad_myo(m, *a, **kw)
        ld = getattr(self, "_loads", None)
        if not ld:
            return g
        from plexus.operators.contact_ops import linear_contact_forces
        x = a[0]
        fext, conts = ld
        if fext is not None:
            g = g - fext.to(g.dtype)
        zero = torch.zeros_like(x)
        for C in conts:
            C["_F"], C["_fv"] = linear_contact_forces(C, x, zero, C["nF"])
            g = g - C["_fv"].to(g.dtype)
        return g

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return super().forward(H, mask)
        pos = lvl.get("pos")
        Nv = int(m["Nv"])
        _fr, _bc, fext, conts = _frame_loads(H, m, Nv, pos.device, pos.dtype)
        self._loads = (fext, conts) if (fext is not None or conts) else None
        try:
            out = super().forward(H, mask)
        finally:
            self._loads = None
        return _book_contacts(H, m, _fr, _bc, conts, Nv, pos.device, out)


def junction_spring_grad(x, es, et, ef, nF, keys, l0, stride, k):
    """dE/dx of E = sum over junctions of (k / 2) (l - l0)^2 / l0, one term per junction.

    A junction is found by its vertex pair (key = min * stride + max, as `junction_ops` keys its store);
    one with no stored rest length contributes nothing. Each undirected junction is two half-edges in
    the interior and one on a free edge, so each half-edge carries 1 / (its key's count) of the term."""
    live = ef < nF
    vi, vj = es[live].long(), et[live].long()
    key = torch.minimum(vi, vj) * stride + torch.maximum(vi, vj)
    order = torch.argsort(keys)
    ks, vs = keys[order], l0[order].to(x.dtype)
    idx = torch.searchsorted(ks, key).clamp(max=max(ks.numel() - 1, 0))
    hit = (ks.numel() > 0) & (ks[idx] == key)
    L0 = torch.where(hit, vs[idx], torch.ones_like(vs[idx]) if vs.numel() else torch.ones(key.shape, dtype=x.dtype, device=x.device))
    d = x[vj] - x[vi]
    L = d.norm(dim=1).clamp_min(1e-12)
    _, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
    T = torch.where(hit, k * (L - L0) / L0.clamp_min(1e-12), torch.zeros_like(L)) / cnt[inv].to(x.dtype)
    f = (T / L)[:, None] * d
    g = torch.zeros_like(x)
    g.index_add_(0, vj, f)
    g.index_add_(0, vi, -f)
    return g


@register_operator("cell_mechanics", model="junction_spring", set="vertex", kind="lateral",
                   family="mechanics", title="relax cell shapes with elastic junctions", equation=r"""$$E=E_{\mathrm{shape}}+\sum_j\frac{k}{2}\,\frac{(\ell_j-\ell^0_j)^2}{\ell^0_j},\qquad k=\kappa\Lambda$$""")
class ShapeEnergyJunctionSpring3D(ShapeEnergy3D):
    r"""`cell_mechanics` plus a spring on every junction whose rest length remodels.
    vertex -> vertex: as cell_mechanics, with the junction rest lengths read from `junction_myosin[rest_length]`'s store.

        E=E_{\mathrm{shape}}+\sum_j\frac{k}{2}\,\frac{(\ell_j-\ell^0_j)^2}{\ell^0_j},\qquad k=\kappa\Lambda
        E_{shape}      the parent's shape energy
        \ell_j, \ell^0_j junction j's length and remodelled rest length
        k              the spring stiffness, kappa times the line tension
        \kappa         `rl_kappa` of junction_myosin[rest_length]
        \Lambda        the line tension

    Reference: Staddon, M. F. et al. (2019). Biophys. J. 117:1739-1750; the shape energy as `cell_mechanics`."""
    PARAM_UNITS = {}
    SYMBOLS = {'E_{shape}': "the parent's shape energy", '\\ell_j, \\ell^0_j': "junction j's length and remodelled rest length", 'k': 'the spring stiffness, kappa times the line tension', '\\kappa': '`rl_kappa` of junction_myosin[rest_length]', '\\Lambda': 'the line tension'}

    def _grad_myo(self, m, *a, **kw):
        keys = m.get("rl_keys") if isinstance(m, dict) else None
        if keys is None:
            return super()._grad_myo(m, *a, **kw)
        m_shape = {k_: v for k_, v in m.items() if k_ != "myo"}
        g = super()._grad_myo(m_shape, *a, **kw)
        x, es, et, ef, nF = a[0], a[1], a[2], a[3], a[4]
        k = float(m.get("rl_kappa", 0.0)) * float(self.Lambda)
        if k == 0.0:
            return g
        return g + junction_spring_grad(x, es, et, ef, int(nF), keys.to(x.device), m["rl_vals"].to(x.device),
                                        int(m["rl_stride"]), k).to(g.dtype)


# There is ONE growth operator on this mesh, `cell_grow` in diffusion_reaction, and deliberately
# so. A second one duplicated it and contradicted it: both wrote the same mesh targets, one
# multiplicatively (V0f <- V0f * g^3) and the other by assignment from its own snapshot
# (V0f <- V0f_init * s^3). Scheduled together, whichever ran second silently overwrote the first
# every frame, so the growth rate a specification declared was not the one it got.
#
# Every call site is ported: uniform body-wide inflation is `cell_grow` with `rho = 1` and the
# gate open (`a_sw = 0`). `max_scale` capped the LINEAR scale and `vth_frac` caps per-cell
# VOLUME, so the same plateau is `vth_frac = max_scale ** 3`.


@register_operator("cell_divide", set="vertex", kind="divide", family="population", title="divide cells above a size",
                   equation=r"""$$\text{divide } j\ \text{when}\ v_j>\text{factor}\cdot v_{\mathrm{ref}},\qquad w\longrightarrow\tfrac w2,\tfrac w2$$""")
class Divide3D(Structural):
    r"""cell division: a septum through a face, so one cell becomes two and the surface gains a cell without losing its closure.
    vertex -> vertex: rebuilds the half-edge table, the vertex positions and every per-face array.

        \text{divide } j\ \text{when}\ v_j>\text{factor}\cdot v_{\mathrm{ref}},\qquad w\longrightarrow\tfrac w2,\tfrac w2
        v_j            the cell's wedge volume, in world units cubed
        v_{ref}        the seed-time median cell volume
        factor         `factor`, how many median cells a cell must reach, dimensionless
        w              the face, split by a septum between two edge midpoints

    Reference: Okuda, S. et al. (2013). Biomech. Model. Mechanobiol. 12(4):627-644. The in-surface division follows the sheet topology of Tyssue."""
    PARAM_UNITS = {'factor': 'dimensionless', 'cycle_cv': 'fraction', 'max_cells': 'count', 'min_cycle': 'count', 'max_cycle': 'count', 'p0': 'dimensionless', 'local_relax': 'count', 'size_escape': 'dimensionless', 'orient_asw': 'fraction'}
    SYMBOLS = {'v_j': "the cell's wedge volume, in world units cubed", 'v_{ref}': 'the seed-time median cell volume', 'factor': '`factor`, how many median cells a cell must reach, dimensionless', 'w': 'the face, split by a septum between two edge midpoints'}
    PARAM_ROLES = {'cell_set': 'the_cell_set', 'currency': 'volume_or_area_as_the_size_read', 'cycle_cv': 'spread_of_the_per_cell_threshold', 'factor': 'size_threshold_in_median_cells', 'local_relax': 'relaxation_iterations_around_the_septum', 'max_cells': 'ceiling_on_the_cell_count_zero_means_none', 'max_cycle': 'division_calls_after_which_a_cell_divides_anyway', 'min_cycle': 'division_calls_before_a_cell_may_divide', 'orient_asw': 'activator_switch_level_for_the_septum_orientation', 'orient_iface': 'orient_the_septum_along_the_type_interface', 'p0': 'target_shape_index_of_the_daughters', 'project': 'project_the_new_vertices_back_onto_the_surface', 'size_escape': 'size_above_which_the_cycle_gates_are_ignored', 'skip_queued': 'skip_cells_already_queued', 'skip_sentenced': 'skip_cells_marked_for_death'}
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    # MULTIPLES AND COUNTS. `factor` and `delta` are multiples of `v_ref`, not volumes -- which is
    # what makes them portable across mesh scales -- so they are dimensionless. The cycle bounds
    # are counts of CALLS, not durations, which is why they are `count` and not `time`.
    # `V0f` IS DELIBERATELY UNDECLARED HERE, and this is the clearest case in the tree for why
    # UNKNOWN had to exist. The division trigger reads the POLYHEDRON volume when the run carries a
    # separation and the WEDGE volume when it does not -- the operator prints which on every run
    # ("this run carries a separation, so the trigger reads the POLYHEDRON volume; reference 1.3550
    # (the wedge reference is 2.5462)"). One block, two conventions, selected by the composition
    # rather than by a parameter. A class attribute cannot express that, and declaring either would
    # be wrong on half the specs in `config/tissue`.
    # AND `Vbirth` IS UNDECLARED FOR THE SAME REASON. This operator writes it from whichever volume
    # it just triggered on -- polyhedron where the run carries a separation, wedge where it does
    # not -- so it propagates the run's convention rather than choosing one, exactly like `V0f`
    # above. Declaring `wedge` made it disagree with the apicobasal seed that had just been made
    # consistent, which is the checker accusing the operator that has no opinion.
    BLOCK_UNITS: dict = {}
    MECHANISM_TAGS = ["division", "cell_division", "vesicle", "proliferation", "volume_doubling"]
    REFERENCE = "Hertwig, O. (1884) (long-axis division rule); tyssue cell_division (DamCB)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        # A POPULATION CEILING, off by default so nothing archived changes. `before_frame` stops
        # division at a TIME; this stops it at a SIZE, which is what "grow to 30k cells and then
        # just let the chemistry run" needs -- the frame at which a given spec reaches a given
        # count is not knowable in advance, and guessing it either truncates the growth or wastes
        # the rest of the run.
        self.max_cells = int(params.get("max_cells", 0) or 0)
        self.factor = float(params.get("factor", 2.0))  # divide when volume >= factor x birth volume
        # `split_cv` and `g1_ramp`: WITHDRAWN (R1 of notes/size_cycle/SIZE_CYCLE_PLAN.md), not read.
        # Both perturbed the daughters' TARGETS while the septum stayed where `divide_face_3d` put
        # it. The daughters' birth volumes are now MEASURED on the rebuilt mesh (see forward), so
        # the asymmetry the septum actually produces -- CV 0.27 of the two pieces on the reference
        # spheroid -- is the asymmetry the size rules see, and there is nothing left for a
        # target perturbation to add.
        self.cycle_cv = float(params.get("cycle_cv", 0.0))
        # THE CURRENCY THE SIZE RULES READ (exp33, 2026-10-25): `volume` (the default, bit-identical to every spec already written) is the run's
        # geometry, read by `cell_size`; `mass` is the cell's own TARGET AREA A0 -- what growth adds and division halves. Under planar growth
        # (`cell_grow[planar]`, V0f ~ A0 ~ s^2, a monolayer of fixed density) the two differ only by the mechanics' lag: a cell realises
        # 41-62 % of its A0, so a trigger on the realised volume measures how far the mechanics has got, not how much the cell has grown.
        self.currency = str(params.get("currency", "volume"))
        if self.currency not in ("volume", "mass"):
            raise ValueError(f"cell_divide: currency is `volume` or `mass`, not {self.currency!r}")
        # `skip_queued: true` (exp 11 ladder L3, 2026-09-28; default false): with `skip_sentenced`, a cell waiting in
        # `cell_die[t2]`'s leaving queue does not divide in place either (on by default it stopped every in-place
        # division of a gland whose dive queue holds all its ready cells -- exp 11 Finding 136)
        self.skip_queued = bool(params.get("skip_queued", False))
        # `size_escape: k` (default 0 = off; exp 11 R2, 2026-09-30): A SIZE CHECKPOINT on the queue above. A queued cell
        # whose `area` block exceeds k x the median is NOT held back: it divides in place. Wang's surface cells keep a
        # constant size; a ready cell stretched by a growing branch while it waits to dive otherwise grows flat without
        # limit (run 460: 55 cells at 4-5x the median basal area, half of them division-ready, Finding 184).
        self.size_escape = float(params.get("size_escape", 0.0) or 0.0)
        # STOCHASTIC CELL CYCLE: Gaussian CV of each daughter's
        #   cell-cycle length (fresh division threshold). >0 keeps division waves broken up (desynchronised) as the
        #   tissue proliferates -- essential at scale so max-rate division never outruns relaxation. 0 -> uniform reset_noise.
        self.p0 = float(params.get("p0", 3.72))
        self.every = _engine_owns_clock(params, default=3); self._k = 0
        # max_div / max_div_frac / vcap: WITHDRAWN, and not read anywhere. An
        # archived spec may still carry them; they are ignored, not honoured.
        # LOCAL DAUGHTER RELAX: after the septum, run a few bounded-Euler shape-energy steps on ONLY the
        # fresh daughters + their one-ring, so a division never hands the global relaxation an inverted
        # cap (the sole source of hollow cells -- growth alone never makes one). Uses the coeffs stashed
        # on m["mech"] by cell_mechanics. 0 = off (default); the coral/tube fix sets it > 0.
        self.local_relax = int(params.get("local_relax", 0))
        # DIVISION MODEL = volume-primary + bounded DURATION (Okuda: divide at 2x volume, but the cell-cycle
        # PERIOD is constrained). min_cycle: a cell may not divide before this many division-calls since birth
        # (a fast-growing tip cell can't divide instantly -> tighter size CV); max_cycle: force division after
        # this many calls even if volume < 2x (a stalled cell still cycles). 0/inf = pure volume-doubling.
        self.min_cycle = int(params.get("min_cycle", 0)); self.max_cycle = int(params.get("max_cycle", 10 ** 9))
        self._cat = params.get("cell_set")          # override only; resolved from the set in forward()
        # ORIENTED division at the red/white interface (Okuda's tube mechanism, user issue 3): the dividing
        # plane of an ACTIVATED (red) cell is oriented so the daughters stack ALONG the bud axis (the
        # direction from the vesicle centre to the activated tip) instead of by the cell's own long axis.
        # This adds new cells NORMAL to the body surface -> builds the tube WALL and EXTENDS the protrusion,
        # rather than widening it. orient_asw = activator threshold that flags an interface/red cell. 0 = off.
        self.orient_iface = bool(params.get("orient_iface", False))
        self.orient_asw = self.tunable(params.get("orient_asw"), 1.0, tensor=True)
        # `project` -- PUSH THE TWO NEW MIDPOINTS BACK OUT TO THE LOCAL SHELL RADIUS. True is
        # right for a closed vesicle, where a septum midpoint left at the chord would dimple the
        # surface inward at every division. It is a SPHERE assumption, and on a flat sheet it is
        # simply wrong: |p| is measured from the world origin, so it lifts the midpoints off the
        # plane. On a flat disc it is the entire residual out-of-plane drift once the solver
        # itself is constrained: the sheet acquires an out-of-plane spread of several percent of an
        # edge length purely from division, even though its mechanics can no longer leave the
        # plane at all.
        self.project = bool(params.get("project", True))
        # `skip_sentenced` (default false = unchanged): a cell under a `cell_die` sentence (`apop_flag`)
        # is not divided. Without it a cell sentenced while in M (`cell_die[t2]` `rule: phase`, exp 11)
        # is cut while it shrinks and BOTH daughters inherit the sentence -- two cells lost per leaver.
        self.skip_sentenced = bool(params.get("skip_sentenced", False))

    def _size(self, lvl, m, nF, pos_np, H):
        """(size now, reference) in the operator's currency. `volume`: `cell_size` (the run's geometry decides). `mass`: the target area A0
        and its seed-time median, cached on the first call; the birth block `Vbirth` is initialised to the seed's A0 then, so a first-generation
        cell's birth mass is in the same currency as its daughters' (written from this reading at age 1)."""
        if self.currency == "mass":
            a0 = m["A0"].detach().cpu().numpy().astype(np.float64)[:nF]
            if "mass_ref" not in m:
                m["mass_ref"] = float(np.median(a0))
                set_cell_block(H, self.cell_set, "Vbirth", a0.copy(), nF)
            return a0, float(m["mass_ref"])
        return cell_size(lvl, m, nF, pos_np, H=H, at=self.at)

    def _trigger(self, v_now, v_birth, jit, age, v_ref):
        """Has this cell earned a division? THE ONLY THING A `model=` VARIANT OF cell_divide CHANGES.

        ONE "WHEN" (R4 of notes/size_cycle/SIZE_CYCLE_PLAN.md): a NEW spec states its size rule on
        `cell_cycle` -- G1 rules `sizer`, `adder`, `doubler`, `timer`, `transition_probability`,
        `inhibitor_dilution`, with `t_s = t_g2 = t_m = 0` for "divide when the rule fires" -- and
        schedules `cell_divide[model: cycle]`, which does the topology when the cycle says M. The
        `sizer` / `adder` / `doubler` / `timer` / `concerted` models of THIS operator are the same
        rules on the other contract, kept for the specs that carry them (the okuda archive) and not
        to be used by a new one. The eleven size/cycle specs say their rule on the cycle.

        THE DEFAULT IS `sizer`: an ABSOLUTE threshold, v >= factor * v_ref, where v_ref is the
        seed-time median cell volume. Ginzberg, Kafri & Kirschner (Science 2015) are explicit that
        this is what size control requires -- "both the cell's target size and its actual size
        must be evaluated on ABSOLUTE rather than relative scales" -- and it is the mechanism they
        review as the G1/S size checkpoint, where small cells are held longer so they can catch up.
        A sizer corrects a size deviation in ONE generation.

        The rule this replaced as default, `v >= factor * v_birth`, is kept as `model: doubler`.
        It is relative, so it corrects nothing: a cell born 30% small divides 30% small and passes
        the deviation on undiminished. Combined with exponential growth it does worse than nothing,
        which is what this campaign measured -- vol_cv 0.160 at seed to 0.53 by frame 900.
        """
        return v_now >= self.factor * jit * v_ref

    def _realised_volumes(self, lvl, m, nF):
        """Every cell's volume on the mesh as it stands, in `cell_size`'s convention. None on a
        mesh with no live vertices."""
        if int(m["Nv"]) <= 0:
            return None
        v, _ = cell_size(lvl, m, nF)
        return np.asarray(v, np.float64)

    def _fresh_djit(self, rng, n=1):
        """Fresh per-cell division-threshold multiplier. Gaussian CV (cycle_cv) when set -> desynchronised
        cell cycles; otherwise the legacy uniform +/-reset_noise jitter. Clamped so thresholds stay sane."""
        # There is ONE way to desynchronise the cycle, `cycle_cv`. A second jitter parameter read
        # only on the `cycle_cv == 0` branch is unreachable whenever cycle_cv is set, so a search
        # editing it would measure exactly zero change and record that as evidence about the
        # mechanism rather than about the parameter being dead.
        if self.cycle_cv > 0:
            v = np.clip(1.0 + self.cycle_cv * rng.standard_normal(n), 0.4, 1.8)
        else:
            v = np.ones(n)                     # cycle_cv = 0 means synchronous, not "jittered a bit"
        return v if n > 1 else float(v[0])

    def forward(self, H, mask=None):
        # THE PAIRING IS READ FROM THE SET, ONCE PER CALL -- see `resolve_cell_set`.
        self.cell_set = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
        from plexus.models.topology import rings_from_flat_3d, flat_from_rings_3d, divide_face_3d
        # THE VERTEX PARENTAGE, collected here and spent below. Empty `vertex_carry` -> a no-op, so
        # every existing spec is byte-identical; the list is built regardless because it costs two
        # tuples per division and a conditional would be one more thing to get wrong.
        births: list = []
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None or settling(H, m):
            return {}
        self._k += 1                    # monotonic tick only -- D1: the engine owns the period
        dev = lvl.state.device; dt = lvl.state.dtype; buf = lvl.state.shape[0]
        Nv = m["Nv"]
        pos_np = lvl.get("pos")[:Nv].detach().cpu().numpy().astype(np.float64)
        es = m["E_srce"].detach().cpu().numpy(); et = m["E_trgt"].detach().cpu().numpy()
        ef = m["E_face"].detach().cpu().numpy(); nF = int(m["nF"])
        # ONE READER (R2 of notes/size_cycle/SIZE_CYCLE_PLAN.md). This operator used to compute the
        # wedge volume here and then, whenever the run carried a separation, replace it with the
        # polyhedron volume and cache its own reference -- while `cell_cycle`, `cell_die` and
        # `cell_grow` read `cell_size`. Two conventions in one tissue: the daughters' `Vbirth`
        # was written in this operator's and read in the cycle's. `cell_size` decides for all
        # four now, so `vf`, `v_ref` and `Vbirth` are one currency.
        vf, v_ref = self._size(lvl, m, nF, pos_np, H)
        vf = np.asarray(vf, np.float64)
        rings = rings_from_flat_3d(es, et, ef, nF)
        pos = [p for p in pos_np]
        A0 = m["A0"].detach().cpu().numpy().tolist()
        V0f = m["V0f"].detach().cpu().numpy().tolist()
        Vbirth = require_cell_block(H, self.cell_set, "Vbirth", nF, "cell_divide").tolist()
        alive = m["alive"].detach().cpu().numpy().tolist()
        rng = np.random.default_rng(12345 + self._k)
        # THE FIXED-SEED FALLBACK IS GONE WITH THE MESH COLUMN. It read: if `divjit` is absent,
        # draw it here from `default_rng(777)`. That is a spec's missing seed answered by an
        # invented one -- the run divided, staggered its thresholds with a jitter nobody declared,
        # and looked entirely normal. A declared block cannot be absent by accident, so the honest
        # answer is the refusal `require_cell_block` gives.
        djit = require_cell_block(H, self.cell_set, "divjit", nF, "cell_divide").tolist()
        _cid = cell_block(H, self.cell_set, "cell_id", nF)
        cid = None if _cid is None else _cid.tolist()
        _pid = cell_block(H, self.cell_set, "parent_id", nF)
        pid = None if _pid is None else _pid.tolist()
        next_id = int(m.get("next_id", nF))
        age = m.get("age")                                       # per-cell age in division-calls since birth
        age = ([0] * nF) if (age is None or age.shape[0] != nF) else (age.detach().cpu().numpy() + 1).tolist()
        # THE BIRTH VOLUME, RE-READ ONCE THE MECHANICS HAS ANSWERED THE SEPTUM. A cell at age 1 was
        # cut at this operator's previous call; the piece the septum made is not the volume it
        # settles to (the rest offset above inflates a fresh daughter by a third within a few
        # frames), and the size rules compare against what the daughter HAS at the start of its
        # cycle. `vf` is in the trigger's own convention, so `Vbirth` and the threshold agree.
        if ndiv_seen := any(a == 1 for a in age):
            for f in range(nF):
                if age[f] == 1 and alive[f] > 0:
                    Vbirth[f] = float(vf[f])
            set_cell_block(H, self.cell_set, "Vbirth", np.asarray(Vbirth, np.float64), nF)
        # HAS THIS CELL EVER DIVIDED? `age` alone cannot answer it: it starts at 0 for every SEEDED
        # cell and is only RESET to 0 by a division, so in the opening frames a whole untouched
        # tissue looks "just divided" -- the movie flashed entirely green in p1_ph_rd_only, a run in
        # which division never fires at all. Counting divisions separates born-this-way from
        # divided-just-now; the renderer needs both.
        ndiv = m.get("ndiv")
        ndiv = ([0] * nF) if (ndiv is None or ndiv.shape[0] != nF) else ndiv.detach().cpu().numpy().tolist()
        # volume-primary + bounded duration: divide if (2x volume AND old enough) OR (past max cycle length)
        # THE REFERENCE IN THE SAME UNITS AS `vf` ABOVE -- polyhedron where the run has a
        # separation, wedge where it does not. Mixing them is the defect the block above exists for.
        # `v_ref` came from `cell_size` above, in the same convention as `vf`.
        # A MODEL MAY ANSWER FROM THE TABLE INSTEAD OF FROM THE FOUR SCALARS. `_trigger` sees
        # (v_now, v_birth, jit, age, v_ref) and that is the right interface for every rule that
        # reads size or time -- but `model: cycle` reads a PHASE another operator owns, which is not
        # among them. `_ready_mask` is the opt-in: a model that defines it gets one per-face boolean
        # computed once, and every model that does not is untouched.
        _mask = self._ready_mask(H, m, nF) if hasattr(self, "_ready_mask") else None
        _sent = None
        if self.skip_sentenced and m.get("apop_flag") is not None and len(m["apop_flag"]) == nF:
            _fl = m["apop_flag"]
            _sent = (_fl.detach().cpu().numpy() if torch.is_tensor(_fl) else np.asarray(_fl)) > 0
            # AND THE CELLS WAITING IN `cell_die[t2]`'s LEAVING QUEUE (`m["t2_waiting"]`, cell ids): a cell chosen to
            # dive but held back by the per-call cap is not free to divide in place meanwhile (exp 11, ladder L3)
            _wq = m.get("t2_waiting") if getattr(self, "skip_queued", False) else None
            if _wq:
                _ci = cell_block(H, self.cell_set, "cell_id", nF)
                if _ci is not None:
                    _ar = cell_block(H, self.cell_set, "area", nF) if self.size_escape > 0 else None
                    _sent = _sent | queued_hold(_ci, _wq, _ar, self.size_escape)

        def _ready(f):
            if rings[f] is None or len(rings[f]) < 4 or alive[f] <= 0:
                return False
            if _sent is not None and _sent[f]:
                return False
            vol_ok = bool(_mask[f]) if _mask is not None else \
                self._trigger(vf[f], Vbirth[f], djit[f], age[f], v_ref)
            return (vol_ok and age[f] >= self.min_cycle) or (age[f] >= self.max_cycle)
        cand = [f for f in range(nF) if _ready(f)]
        # THE POPULATION CEILING, and it is a HARD STOP rather than a throttle. Once the tissue has
        # `max_cells` cells nothing further divides, so the run keeps integrating -- chemistry,
        # mechanics, edge flips all continue -- on a population that has stopped changing. That is
        # the difference between this and `before_frame`: a frame number cannot say "when the
        # tissue is this big", because the frame at which a given spec reaches a given count is not
        # knowable before the run. Zero (the default) means no ceiling, so nothing archived moves.
        #
        # IT DROPS THE CANDIDATES RATHER THAN CAPPING THEM. Letting the last few through to land
        # exactly on `max_cells` would make the stop depend on how many cells happened to ripen on
        # the same tick, which is noise; refusing the whole tick makes the ceiling reproducible.
        if self.max_cells and nF >= self.max_cells:
            cand = []
        rng.shuffle(cand)                                        # unbiased when more cells are ready than the cap
        # NO THROTTLE, NO BYPASS. Every cell that is READY divides, and READY is the test two
        # lines above: current volume >= factor x jitter x its OWN birth volume, after min_cycle
        # (or past max_cycle, which is a backstop and must be set long or it becomes the rate).
        # That is P3 -- "a cell divides because it got big" -- and it is the mechanism the paper
        # is about. The pace therefore comes from how fast cell_grow inflates cells:
        # ONE number, and it is a growth rate, in the place where the biology is.
        #
        # THREE NUMERICAL SHORTCUTS USED TO OVERRIDE IT, and each in turn WAS the rate:
        #   max_div         an absolute divisions-per-call floor. cap_div = max(max_div,
        #                   frac x N), and the floor won at every realistic cell count -- 30 per
        #                   call whatever the growth was doing.
        #   max_div_frac    the same as a fraction; entirely masked by the floor above it.
        #   vcap            force-divided any cell over 1.5 x the REFERENCE volume, bypassing the
        #                   x2 trigger, the jitter and min_cycle. v_ref drifts DOWN as cells
        #                   divide and shrink, so it fired earlier and earlier.
        #
        # A volume cap on the division trigger decouples the division rate from the growth rate
        # entirely: cells divide and shrink, so the trigger fires earlier and earlier and the
        # tissue overruns its target cell count by a large factor while the rate written in the
        # specification is masked. What such a run measures is the counter, not the tissue.
        ndone = 0
        daughter_mothers = []                                    # mother face index of each appended daughter (order)
        bud_axis = None; a_cells = None; orient_thr = None       # ORIENTED interface division: bud axis = centre->red tip
        if self.orient_iface and self.cell_set is not None:
            clvl0 = H.level(self.cell_set)
            if clvl0 is not None and "chem" in clvl0.state_schema:
                ci0, _ = clvl0.state_schema["chem"]
                a_cells = clvl0.state[:nF, ci0].detach().cpu().numpy()
                # RELATIVE TO THE FIELD, for the reason `rd_interface_tension.a_sw` is: an absolute
                # threshold on a field whose scale the chemistry sets is one edit from selecting
                # nothing. `orient_asw` defaulted to 1.0 with the same (0.2, 6.0) range, and only
                # 20 of 78 campaign runs ever reached act_max > 1.0 -- so in 74% of runs
                # `cell_divide:orient_iface`, a named Okuda mechanism, could orient nothing and was
                # behaviourally `hertwig`. A `set_impl ... orient_iface` edit was a silent no-op.
                amax = float(a_cells.max()) if a_cells.size else 0.0
                thr = orient_thr = self.orient_asw * amax
                rc = [np.array([pos[v] for v in rings[f]]).mean(0) for f in range(nF)
                      if amax > 0 and a_cells[f] > thr
                      and rings[f] is not None and len(rings[f]) >= 3]
                if len(rc):
                    ba = np.mean(rc, 0); nba = float(np.linalg.norm(ba))
                    if nba > 1e-6:
                        bud_axis = ba / nba
        blocked = 0                     # divisions the RESERVOIR refused, not the biology
        # THE EDGE->FACE MAP IS BUILT ONCE FOR THE FRAME, not once per dividing cell.
        #
        # `divide_face_3d` needs two lookups out of it and used to rebuild the whole O(E) dict for
        # them. That made the operator quadratic in the tissue: both the edge count and the number
        # of cells ripening on a tick grow together, so the rebuild cost grew with the product.
        # Profiled on `mesh_mpm_spheroid_nominal` at frame 380 it was 19.4 s of `cell_divide`'s
        # 24.2 s, over 5,480 rebuilds in 96 frames -- 57 rebuilds a frame, one per division.
        # Built here and maintained inside `divide_face_3d`, it is one build a frame.
        emap = _edge_face_map(rings)
        # THE CELL BUFFER CAN FILL FIRST, and it used to crash rather than refuse: a flat apico-basal disc
        # carries ~2 vertices a cell, so with the usual vertex buffer the CELL set's rows ran out first and
        # the next `set_cell_block` raised "expanded size (20000) must match (20227)" (exp12 round 5b,
        # 2026-09-26). The same counted refusal as the vertex buffer, and nothing changes for a run that
        # never reached either limit.
        _clvl = H.level(self.cell_set) if self.cell_set else None
        cbuf = int(_clvl.state.shape[0]) if _clvl is not None and getattr(_clvl, "state", None) is not None else None
        for f in cand:
            if len(pos) + 2 > buf or (cbuf is not None and len(rings) + 1 > cbuf):
                # THE VERTEX (OR CELL) BUFFER IS FULL. Counted and reported, never silent.
                #
                # A bare `break` would be the whole story: division simply stops and nothing says
                # why. A buffer sized for the seed cell count caps the tissue at (V+4)/2 cells, and
                # a run that reaches it adds ZERO cells for every remaining frame. Such a run
                # measures a full array rather than a tissue, and every numerical check still
                # passes it -- the cell count is simply constant, which is a legal trajectory.
                #
                # A run truncated by its own reservoir is not evidence about growth, and the
                # difference between "the tissue stopped dividing" and "the tissue was not
                # ALLOWED to divide" is the difference between a result and an artefact.
                blocked = len(cand) - len(daughter_mothers)
                break
            r = rings[f]
            P = np.array([pos[v] for v in r]); c = P.mean(0)      # LONG-AXIS (Hertwig) split: septum perpendicular
            try:                                                 # to the cell's longest axis -> compact daughters
                _, _, vh = np.linalg.svd(P - c, full_matrices=False); u = vh[0]
                n = c / (np.linalg.norm(c) + 1e-9)               # outward (radial) face normal on the sphere
                # THE SAME RELATIVE THRESHOLD THAT BUILT THE AXIS. Testing
                # `a_cells[f] > self.orient_asw` -- the ABSOLUTE value -- while the axis above is
                # already relative to `amax` makes the two disagree about which cells are the bud.
                # At orient_asw 0.6 on a field peaking at 1.47 the axis takes the top 40% and this
                # took everything above 0.6, a different and larger set; on any run whose field
                # peaks below orient_asw the axis exists and NO cell passes here, which is the
                # silent-`hertwig` failure the comment above says was fixed.
                if bud_axis is not None and orient_thr is not None and f < len(a_cells) and a_cells[f] > orient_thr:
                    ut = bud_axis - float(np.dot(bud_axis, n)) * n   # bud axis in this cell's tangent plane ->
                    if np.linalg.norm(ut) > 1e-6:                    # daughters separate ALONG the protrusion (extend the
                        u = ut / np.linalg.norm(ut)                  # wall) instead of by the cell's own long axis
                w = np.cross(n, u); w = w / (np.linalg.norm(w) + 1e-9)   # short-axis direction in the tangent plane
                mids = 0.5 * (P + np.roll(P, -1, 0)); proj = (mids - c) @ w
                ea, eb = int(np.argmax(proj)), int(np.argmin(proj))
            except Exception:
                ea, eb = 0, len(r) // 2
            res = divide_face_3d(rings, pos, f, ea=ea, eb=eb, emap=emap, project=self.project,
                                 births=births)
            if res is None:
                continue
            # PROVISIONAL: the mother's targets and volume split evenly. Re-split below, once the
            # mesh is rebuilt, in proportion to what each daughter ACTUALLY encloses -- see the
            # block after `_carry_face_state`.
            half = other = 0.5 * vf[f]
            a0d = a0e = 0.5 * A0[f]; v0d = v0e = 0.5 * V0f[f]
            A0[f] = a0d; V0f[f] = v0d; Vbirth[f] = half           # daughter A (kept at index f)
            djit[f] = self._fresh_djit(rng); age[f] = 0           # fresh (desync'd) thresholds; reset cell-cycle age
            if cid is not None:                                   # both daughters are new cells; the mother's id ends here
                mother_id = cid[f]
                cid[f] = float(next_id); cid.append(float(next_id + 1)); next_id += 2
                if pid is not None:
                    pid[f] = mother_id; pid.append(mother_id)
            ndiv[f] = ndiv[f] + 1
            A0.append(a0e); V0f.append(v0e); Vbirth.append(other); alive.append(1.0)  # daughter B
            djit.append(self._fresh_djit(rng)); age.append(0); ndiv.append(ndiv[f])
            daughter_mothers.append(f)
            ndone += 1
        if ndone == 0:
            m["age"] = torch.as_tensor(np.asarray(age), dtype=dt, device=dev)   # persist ageing even without division
            m["ndiv"] = torch.as_tensor(np.asarray(ndiv), dtype=dt, device=dev)
        # RESERVOIR PRESSURE, reported every frame it bites. `div_blocked` is how many cells
        # wanted to divide this frame and were refused for want of room; `buf_full` says the
        # array is at its ceiling. Both travel with the mesh so run_one can record them.
        m["div_blocked"] = int(blocked)
        m["buf_full"] = bool(len(pos) + 2 > buf or (cbuf is not None and len(rings) + 1 > cbuf))
        if blocked:
            print(f"[cell_divide] RESERVOIR FULL: {blocked} division(s) refused for want of vertex "
                  f"buffer ({len(pos)}/{buf}) or cell buffer ({len(rings)}/{cbuf}). This run is capped by "
                  f"its array, not by its biology -- every later measurement describes the reservoir.",
                  flush=True)
            return {}
        # THE PER-VERTEX CARRY, spent here for the same reason `reindex_faces` is spent beside the
        # per-face one: a topology edit that changes who a row IS must be followed by every array
        # that is indexed by that row, in the same place, or the fourth operator does not get the
        # memo. With no `vertex_carry` declared this walks an empty name list and returns, so every
        # existing spec is byte-identical -- which is the claim R1 has to make and the twin measures.
        if hasattr(m, "carry_vertices") and births:
            m.carry_vertices(births, dt=dt, dev=dev, level=lvl)
        es2, et2, ef2, nF2, keep = flat_from_rings_3d(rings)
        A0a = np.array([A0[i] for i in keep], np.float64)
        V0fa = np.array([V0f[i] for i in keep], np.float64)
        Vba = np.array([Vbirth[i] for i in keep], np.float64)
        dja = np.array([djit[i] for i in keep], np.float64)
        cida = None if cid is None else np.array([cid[i] for i in keep], np.float64)
        pida = None if pid is None else np.array([pid[i] for i in keep], np.float64)
        m["next_id"] = next_id
        agea = np.array([age[i] for i in keep], np.float64)
        ndva = np.array([ndiv[i] for i in keep], np.float64)
        alv = np.array([alive[i] for i in keep], np.float64)
        P0a = self.p0 * np.sqrt(np.maximum(A0a, 1e-9))
        Nv2 = len(pos)
        px0, px1 = lvl.state_schema["pos"]
        st = lvl.state.clone()
        st[:Nv2, px0:px1] = torch.as_tensor(np.asarray(pos), dtype=dt, device=dev)
        lvl.state = st
        if getattr(lvl, "occ", None) is not None:
            occ = torch.zeros(buf, device=dev); occ[:Nv2] = 1.0; lvl.occ = occ
        m["E_srce"] = torch.as_tensor(es2, device=dev); m["E_trgt"] = torch.as_tensor(et2, device=dev)
        m["E_face"] = torch.as_tensor(ef2, device=dev); m["nF"] = nF2; m["Nv"] = Nv2
        m["A0"] = torch.as_tensor(A0a, dtype=dt, device=dev)
        m["P0"] = torch.as_tensor(P0a, dtype=dt, device=dev)
        m["V0f"] = torch.as_tensor(V0fa, dtype=dt, device=dev)
        m["age"] = torch.as_tensor(agea, dtype=dt, device=dev)
        m["ndiv"] = torch.as_tensor(ndva, dtype=dt, device=dev)
        m["alive"] = torch.as_tensor(alv, dtype=dt, device=dev)
        # A DAUGHTER MUST BE CARRIED FROM ITS MOTHER, AND `keep` ALONE CANNOT SAY THAT.
        #
        # `keep[j]` is the index of new face `j` in the RINGS LIST, and `divide_face_3d` APPENDS
        # daughter B at rings index `nF + i` while daughter A keeps the mother's index `f`. The
        # eight explicit targets above survive that because their python lists were extended in the
        # same order before `keep` was applied -- `A0.append(a0e)`, `age.append(0)` -- so `A0[keep]`
        # finds a real entry at `nF + i`. A `face_carry` array is NOT extended: it is still the
        # length-nF tensor the operator wrote last frame, and `reindex_faces` CLAMPS an out-of-range
        # index to `nF - 1`. So every daughter B was given THE LAST CELL'S VALUE -- not its
        # mother's, not a default, but whichever cell happened to sit at the end of the array.
        #
        # WHAT THAT COST, on `cycle_phases`: daughter B inherited a stranger's phase and time-in-
        # phase, so `cell_cycle`'s reset -- which fires on "age 0 AND phase still M" -- missed it
        # whenever that stranger was not in M, and the daughter carried on from the middle of
        # someone else's cycle. Measured at frame 13 of 120, 17 of 217 cells held a `phase_t` that
        # belonged to no ancestor of theirs; the worst read 56.5 frames into a phase it had never
        # entered against the 1.0 its mother's reset should have given it.
        #
        # `carry` is `keep` with the appended rows redirected to the mother, which is what
        # `reindex_faces`'s docstring has always CLAIMED the map does. Faces that merely moved are
        # untouched, so a spec that never divides is unaffected.
        carry = keep
        if daughter_mothers:
            born = {nF + i: mo for i, mo in enumerate(daughter_mothers)}
            carry = np.array([born.get(int(o), int(o)) for o in keep], np.int64)
        _carry_face_state(m, carry, dt, dev)
        m["n_div"] = int(m.get("n_div", 0)) + ndone
        # BIRTH VOLUME IS MEASURED, NOT BOOKKEPT. Storing the arithmetic half `vf * p` of the
        # mother makes every cell's birth volume read `v_ref` in a converged population, so the
        # adder's threshold `Vbirth + delta v_ref` collapses into the sizer's `2 v_ref` -- a slope
        # of -1 of added volume on birth volume where an adder is 0 -- while the daughters the
        # septum actually made vary by a quarter.
        #
        # `Vbirth` IS THE ACTUAL VOLUME, in `cell_size`'s convention: provisionally the piece the
        # septum made, re-read at this operator's next call (see the top of forward) once the
        # mechanics has answered the cut. THE TARGETS STAY THE MOTHER'S HALVES. Splitting them in
        # proportion to the septum's pieces was tried (R1-R2) and hands a small piece a small
        # target with the same footprint: its thickness collapses, the spread grows with every
        # division, and `divide_growing_ball` -- a spheroid for 801 frames on the branch base --
        # crumpled from frame 500 (thickness CV 0.67, `tools/spheroid_gauge.py`). The even split
        # is what keeps the shell a shell; the size rules read the measured `Vbirth`, not the
        # target, so nothing they need is lost. A flat sheet whose wedge volumes are zero has no
        # volume to measure and keeps the provisional halves.
        if daughter_mothers:
            vm = self._realised_volumes(lvl, m, nF2)
            if vm is not None and float(np.median(vm)) > 1e-12:
                pos_of = {int(o): j for j, o in enumerate(keep)}   # old index -> new index
                for i, mo in enumerate(daughter_mothers):
                    ja, jb = pos_of.get(int(mo)), pos_of.get(nF + i)
                    if ja is None or jb is None:
                        continue
                    Vba[ja], Vba[jb] = max(float(vm[ja]), 1e-9), max(float(vm[jb]), 1e-9)
        if self.local_relax > 0 and "mech" in m and Nv2 > Nv:    # heal the fresh caps in place at birth
            esT = m["E_srce"]; etT = m["E_trgt"]; efT = m["E_face"]
            newv = torch.zeros(Nv2, dtype=torch.bool, device=dev); newv[Nv:Nv2] = True   # appended septum verts
            touch = newv[esT] | newv[etT]                        # half-edges incident to a fresh vertex
            ring = newv.clone(); ring[etT[touch]] = True; ring[esT[touch]] = True         # + their one-ring
            posf = lvl.state[:Nv2, px0:px1].detach().clone()
            xr = _relax_subset(posf, esT, etT, efT, nF2, m["A0"], m["P0"], m["V0f"], m["alive"],
                               float(m["R0"]), m["mech"], ring, self.local_relax, apex=wedge_apex(m, posf))
            st2 = lvl.state.clone(); st2[:Nv2, px0:px1] = xr; lvl.state = st2
        # propagate the cell morphogen to daughters: each appended cell (new index nF+i) inherits its
        # mother's cell state so the RD pattern rides along through division (seg_A keeps the mother's).
        if self.cell_set is not None and daughter_mothers:
            clvl = H.level(self.cell_set)
            if clvl is not None and clvl.state.shape[0] >= nF2:
                cst = clvl.state.clone()
                for i, mother in enumerate(daughter_mothers):
                    cst[nF + i] = cst[mother]                    # daughter inherits all mother cell state (chem, ...)
                clvl.state = cst
                if getattr(clvl, "occ", None) is not None:
                    cocc = torch.zeros(clvl.state.shape[0], device=clvl.state.device); cocc[:nF2] = 1.0; clvl.occ = cocc
            # AND ITS TYPE, which is not part of `state` and was therefore not inherited. `node_type`
            # is its own buffer, so a daughter kept whatever type its BUFFER SLOT was assigned at
            # build -- which for the usual contiguous fraction split is type 0 for every low index.
            # Measured on a 200-cell spheroid seeded 100 soft / 100 tense: by frame 401 it was 4,104
            # soft and 100 tense, every one of the 4,004 daughters born soft, and the type predicted
            # which hemisphere a cell was in 53.8% of the time -- chance. Any per-type property
            # (myosin drive, adhesion, a clone label) silently washed out of a growing tissue, and
            # the run still looked plausible because the SHAPE change it produced early on persisted.
            nt = getattr(clvl, "node_type", None) if clvl is not None else None
            if nt is not None and int(nt.numel()) >= nF2:
                from plexus.engine import retype
                nt = nt.clone()
                for i, mother in enumerate(daughter_mothers):
                    nt[nF + i] = nt[mother]
                retype(clvl, nt)                                 # and put the derived buffers back
        # LAST, BECAUSE THE INHERITANCE ABOVE COPIES THE WHOLE ROW. `cst[nF + i] = cst[mother]` is
        # right for everything the mother and daughter share -- the morphogen, the cell type -- and
        # WRONG for the two blocks this operator has just computed per daughter: `Vbirth` is the
        # volume each daughter was actually born with (`half` and `other`, which differ whenever
        # `split_cv > 0`), and `divjit` is a FRESH draw per daughter, the whole point of which is
        # that sisters do not divide together.
        #
        # Written here rather than beside `m["A0"]` above, and the difference is measurable: with
        # the writes before the copy, every daughter inherited its mother's jitter, the division
        # stagger collapsed, and `gate_00_spheroid` finished at 6,674 cells against its pinned
        # 6,914 with `t1_total` 1,497 against 1,499. The mesh columns never had this problem
        # because nothing copies a mesh row wholesale.
        set_cell_block(H, self.cell_set, "Vbirth", Vba, nF2)
        set_cell_block(H, self.cell_set, "divjit", dja, nF2)
        if cida is not None:
            set_cell_block(H, self.cell_set, "cell_id", cida, nF2)
        if pida is not None:
            set_cell_block(H, self.cell_set, "parent_id", pida, nF2)
        return {}


@register_operator("cell_die", set="vertex", kind="die", family="population", title="remove marked cells",
                   equation=r"""$$A^0_j\leftarrow A^0_j\Big(1-\frac1\tau\Big)\ \text{while dying};\qquad \text{remove } j\ \text{when}\ A^0_j<c\,A_{\mathrm{ref}}$$""")
class Apoptosis3D(Structural):
    r"""cell elimination: a marked cell contracts its own ring to a point over several frames and is then removed from the surface.
    vertex -> vertex: shrinks the marked cells' target area and perimeter, then rebuilds the half-edge table when a ring has contracted far enough.

        A^0_j\leftarrow A^0_j\Big(1-\frac1\tau\Big)\ \text{while dying};\qquad \text{remove } j\ \text{when}\ A^0_j<c\,A_{\mathrm{ref}}
        A^0_j          the dying cell's target area
        \tau           1/`shrink_rate`, the frames a cell takes to disappear
        c              `critical_frac`, the fraction of the seed-time median size at which the cell is removed
        A_{ref}        the seed-time median cell size

    Reference: Okuda, S. et al. (2013). Biomech. Model. Mechanobiol. 12(4):627-644. Live-cell extrusion: Rosenblatt, J., Raff, M. C. & Cramer, L. P. (2001). Curr. Biol. 11:1847-1857."""
    PARAM_UNITS = {'shrink_rate': 'rate', 'critical_frac': 'fraction', 'frac': 'fraction', 'small_frac': 'fraction', 'stall_frac': 'fraction', 'stall_margin': 'fraction', 'field_frac': 'dimensionless', 'a_sw': 'fraction', 'max_mark_frac': 'fraction', 'cone_deg': 'angle', 'band_deg': 'angle', 'n_bands': 'count', 'n_max': 'count', 'n_min': 'count', 'min_age': 'count', 'after_frame': 'frames', 'p0': 'dimensionless', 'seed': 'count', 'chan': 'count'}
    SYMBOLS = {'A^0_j': "the dying cell's target area", '\\tau': '1/`shrink_rate`, the frames a cell takes to disappear', 'c': '`critical_frac`, the fraction of the seed-time median size at which the cell is removed', 'A_{ref}': 'the seed-time median cell size'}
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    # `shrink_rate` IS PER CALL AND IS DECLARED AS WHAT IT IS. It multiplies the target volume by
    # (1 - shrink_rate) once per invocation, so it is a dimensionless fraction per call and NOT a
    # rate in 1/T -- exactly the confusion S3 removed from `cell_grow`. Declaring it `rate` would
    # be a claim this code does not honour: `cell_die` is `kind: die` and has not been converted.
    # The angles are degrees, which are dimensionless.
    # `V0f` IS UNDECLARED BECAUSE DEATH PROPAGATES A CONVENTION RATHER THAN IMPOSING ONE. It reads
    # whatever the seed put there and compares it against the matching reference through
    # `cell_size`, so its convention is the run's, not its own. Declaring `wedge` here -- as this
    # did while the mismatch was being surfaced -- would now be the checker accusing the one
    # operator that has no opinion.
    BLOCK_UNITS = {"A0": "area[midsurface]"}
    MECHANISM_TAGS = ["apoptosis", "cell_elimination", "extrusion", "delamination", "die"]
    REFERENCE = ("Monier, B. et al. (2015). Apico-basal forces exerted by apoptotic cells drive "
                 "epithelium folding. Nature 518:245-248; tyssue B-Apoptosis (DamCB).")
    PARAM_ROLES = {"cells": "explicit apoptotic cell indices", "mode": "how dying cells are chosen",
                   "shrink_rate": "target-volume shrink per tick",
                   "critical_frac": "extrude below this fraction of v_ref",
                   "frac": "(band/cone/chem_low) size of the dying population",
                   "a_sw": "(chem_low) die where the activator is below this fraction of its max",
                   "max_mark_frac": "cap on the fraction of the tissue under sentence at once", 'after_frame': 'frame_before_which_nothing_is_marked', 'band_deg': 'half_width_of_each_latitude_ring', 'cell_set': 'the_cell_set', 'chan': 'chem_column_of_the_activator', 'compare': 'which_tail_of_the_field_qualifies', 'cone_deg': 'half_angle_of_the_ablated_cap', 'field_frac': 'field_threshold_as_a_multiple_of_the_median', 'min_age': 'division_calls_before_a_cell_may_be_marked', 'n_bands': 'number_of_latitude_rings', 'n_max': 'neighbour_count_at_which_a_crowded_cell_dies', 'n_min': 'neighbour_count_below_which_a_lonely_cell_dies', 'p0': 'target_shape_index', 'region': 'list_band_or_cone', 'seed': 'random_seed', 'small_frac': 'volume_threshold_in_median_cells', 'stall_frac': 'growth_threshold_as_a_fraction_of_the_neighbour_mean', 'stall_margin': 'neighbour_mean_growth_below_which_nothing_is_marked'}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex"); self._cat = params.get("cell_set")
        self.cells = [int(c) for c in (params.get("cells") or [])]
        # WHICH CELLS DIE is targeting, not a second mechanism: the shrink-shed-extrude pathway is
        # identical in every mode, so these are one operator rather than four.
        # DEFAULT `competition`, NOT `list`. The loop's `add_op` edit writes only the numeric
        # parameters an operator declares, so a non-numeric default is what every composition the
        # Proposer builds will actually run. `list` with no `cells` can never fire -- injecting it
        # would have handed the search an operator that is inert by construction, which is the
        # failure worth guarding against. `competition` is the default because it needs no
        # chemistry, so it is available on every base, and it is the canonical cell-competition
        # hypothesis.
        #
        # What makes a cell die is on the `model:` AXIS, not a `mode:` setting. It is the same kind
        # of choice as what makes a cell divide, and `cell_divide` puts that on the model axis --
        # two mechanisms for one question, in adjacent classes in this file.
        #
        # TWO OF THE THIRTEEN COLLAPSED INTO VALUES rather than becoming models, because they were
        # never distinct hypotheses:
        #   `list`/`band`/`cone` -> `model: prescribed` with `region:` -- three GEOMETRIES of one
        #     claim ("the experimenter ablates this patch"), the `mesh_seed.shape` case.
        #   `field_high`/`field_low` -> `model: field` with `compare:` -- the docstring below already
        #     says these "carry no criterion of their own", so the direction is a value on one model.
        if "mode" in params:
            raise ValueError(
                "cell_die: `mode` is gone -- write `model:`. What makes a cell die is a hypothesis, "
                "the same kind `cell_divide` carries on `model:`. `list`/`band`/`cone` are now "
                "`model: prescribed` with `region:`, and `field_high`/`field_low` are "
                "`model: field` with `compare: high|low`.")
        _death = getattr(type(self), "DEATH", "competition")
        if _death == "prescribed":
            _death = str(params.get("region", "list"))
            if _death not in ("list", "band", "cone"):
                raise ValueError(f"cell_die[prescribed]: region must be list|band|cone, "
                                 f"got {_death!r}")
        elif _death == "field":
            _cmp = str(params.get("compare", "high"))
            if _cmp not in ("high", "low"):
                raise ValueError(f"cell_die[field]: compare must be high|low, got {_cmp!r}")
            _death = "field_" + _cmp
        self.mode = _death
        #   list|band|cone|small|stalled|chem_low
        #   LOCAL (vs neighbours): competition|smaller|dimmer|older|crowded|lonely
        #   PUBLISHED FIELD: field_high|field_low -- see `_marked`. These two carry no criterion of
        #   their own; `field` names a per-cell quantity some other operator measured and published
        #   on the mesh, and `field_frac` is the multiple of its live MEDIAN that counts as
        #   qualifying. Death stops needing a new branch for every new measurement.
        self.field = str(params.get("field", "elong"))
        self.field_frac = float(params.get("field_frac", 1.5))
        self.frac = self.tunable(params.get("frac"), 0.04, tensor=True)
        self.cone_deg = float(params.get("cone_deg", 22.8))  # 10 cells across on a 2,000-cell ball
        self.band_deg = float(params.get("band_deg", 8.0))  # half-width of each ring
        self.n_bands = int(params.get("n_bands", 1))              # >1 -> that many latitude rings
        self.small_frac = float(params.get("small_frac", 0.35))  # (small) die below this x v_ref
        # THE THRESHOLD IS PER MODE, because the quantities have different dynamic ranges and one
        # number cannot serve them. The activator spans zero to its maximum across a spot
        # boundary, so a factor-of-two cut on it selects cells easily. Volume, age and growth vary
        # by a few tens of percent between neighbours, so the same factor-of-two cut on those asks
        # for a cell that essentially cannot exist in a healthy sheet, and the model then kills
        # nothing while appearing to be under test.
        # A silent rule is not a conservative rule -- it is an untested one, which this project has
        # now paid for three times (rd_interface_tension twice, chem_low once).
        _STALL_DEFAULT = {"dimmer": 0.5, "competition": 0.7, "older": 0.7,
                          "stalled": 0.8, "smaller": 0.85}
        self.stall_frac = float(params.get(
            "stall_frac", _STALL_DEFAULT.get(self.mode, 0.5)))
        #                                                          (stalled) die below this x the
        self.stall_margin = float(params.get("stall_margin", 0.02))  # population's MEDIAN growth,
        self.min_age = int(params.get("min_age", 4))              # once older than min_age, and
        self.n_max = int(params.get("n_max", 9))          # (crowded) die at this many neighbours
        self.n_min = int(params.get("n_min", 3))          # (lonely) die at this few
        #   only while the median cell has actually grown by stall_margin -- see _marked
        # WHICH SPECIES DECIDES WHO DIES: 0 by default; 2 reads the second RD system, which is
        # what lets one chemistry drive growth and another drive death in the same tissue.
        self.chan = int(params.get("chan", 0))
        self.a_sw = float(params.get("a_sw", 0.25))  # fraction of the activator's own max
        self.shrink = float(params.get("shrink_rate", 0.04))
        self.crit = float(params.get("critical_frac", 0.12))  # x v_ref, the seed-time median
        self.p0 = float(params.get("p0", 3.72))  # target shape index, for P0
        # HOW FAST, as distinct from WHO -- see forward(). 0.5% of the tissue marked at once is
        # ~37 cells on the 7,400-cell parents this has to survive, against the 1,660 that `smaller`
        # took uncapped.
        self.max_mark_frac = float(params.get("max_mark_frac", 0.005))
        self.after_frame = int(params.get("after_frame", 0))
        self.every = _engine_owns_clock(params, default=1); self._k = 0
        self.seed = int(params.get("seed", 0))

    # CARRY EACH PER-CELL ARRAY THROUGH A COLLAPSE ONCE. After a T2 the rebuilt face order is `keep`
    # (new -> old), and this operator writes `A0`, `V0f`, `P0`, `age`, `ndiv`, `alive` and `apop_flag`
    # in that new order through the mesh table -- then `Hierarchy.renumber_set` permutes the whole cell
    # set with the same `keep`. Every one of those names now LIVES on the cell set (the table serves
    # it, `MeshTable._proxied`), so it is permuted TWICE, and at each extrusion the cells beyond the
    # dead one take a neighbouring index's age and targets. Measured (exp 14, Finding 16, a 60-cell
    # shell): in ticks with a removal, 838 of 3,280 surviving cells' ages jumped; in ticks without,
    # 0 of 12,598. The note below about `Vbirth`/`divjit` states the rule; these names were missed.
    # False keeps the default bit-identical to every archived run; `implementation: t2` sets True.
    CARRY_ONCE = False

    def _to_triangle(self, rings, pos, f):
        """Is dying cell `f` ready for the T2 (`face_collapse_3d` takes triangles only)? The default
        only asks, and leaves the walk from n sides to three to `edge_flip`; `implementation: t2`
        does the walk itself (see `Apoptosis3DT2`)."""
        return len(rings[f]) == 3

    def _after_topology(self, H, m, es, et):
        """A pass over the rebuilt half-edge table before it is stored; the default changes nothing
        (`implementation: t2` reclaims the vertices the collapses orphaned)."""
        return es, et

    @staticmethod
    def _ring_neighbours(rings, f):
        """The faces sharing an EDGE with f, read off the rings before f is collapsed."""
        r = rings[f]
        if r is None:
            return []
        edges = {(int(r[i]), int(r[(i + 1) % len(r)])) for i in range(len(r))}
        edges |= {(b, a) for a, b in edges}
        out = set()
        for g, rg in enumerate(rings):
            if rg is None or g == f:
                continue
            for i in range(len(rg)):
                if (int(rg[i]), int(rg[(i + 1) % len(rg)])) in edges:
                    out.add(g); break
        return sorted(out)

    def _nb(self, m, nF, q):
        """(neighbour mean of `q`, neighbour count) on the CELL ADJACENCY GRAPH.

        Every useful death rule -- game of life, too small vs others, too slow vs others, too long
        since dividing vs others -- is a LOCAL comparison between a cell and its neighbours, and the
        rules differ only in which per-cell quantity is compared. So the graph is built once, here.

        WHY LOCAL AND NOT GLOBAL. `chem_low` compares each cell against the whole field and marks
        the whole tissue when the pattern weakens -- every cell shrinks and not one is extruded. A
        rule comparing growth against the population median marks NOBODY when every cell grows at
        the SAME slow rate, since none then sits below half the median. Uniformly slow is not the
        same as losing, and only the neighbour
        graph can tell them apart. A local rule is also self-limiting: it cannot mark a uniform
        field however extreme that field is.

        Adjacency is the face across each half-edge -- the same relation rd_interface_tension uses
        for the red/white ring, so "adjacent" means one thing in this file.
        """
        import torch as _t
        es = _t.as_tensor(m["E_srce"], dtype=_t.long)
        et = _t.as_tensor(m["E_trgt"], dtype=_t.long)
        ef = _t.as_tensor(m["E_face"], dtype=_t.long)
        twin = ShapeEnergy3D._twin_faces(es, et, ef, int(m["Nv"])).cpu().numpy()
        efn = ef.cpu().numpy()
        ok = (efn < nF) & (twin < nF) & (twin >= 0)
        ssum = np.zeros(nF); cnt = np.zeros(nF)
        np.add.at(ssum, efn[ok], q[twin[ok]])
        np.add.at(cnt, efn[ok], 1.0)
        mean = np.zeros(nF)
        hit = cnt > 0
        mean[hit] = ssum[hit] / cnt[hit]
        return mean, cnt

    def _loser(self, m, nF, q, below=True):
        """Cells whose `q` is far from their neighbours' -- the shared body of every local mode."""
        ag = m.get("age")
        nb, cnt = self._nb(m, nF, q)
        old_enough = (ag.detach().cpu().numpy()[:nF] >= self.min_age) if ag is not None \
            else np.ones(nF, bool)
        live = (cnt > 0) & old_enough
        if below:
            # the neighbours must be doing the thing, or a quiet patch culls itself from inside
            return set(np.where(live & (nb > self.stall_margin)
                                & (q < self.stall_frac * nb))[0].tolist())
        return set(np.where(live & (q > nb / max(self.stall_frac, 1e-9)))[0].tolist())

    def _q(self, m, H, nF, what):
        """The per-cell quantity a local mode compares."""
        # A CELL IS SMALL BECAUSE IT IS SMALL, NOT BECAUSE IT ASKS TO BE (R4 of
        # notes/size_cycle/SIZE_CYCLE_PLAN.md). `volume` and `growth` read the TARGET `V0f` until
        # now, and since R1 every daughter's target is the mother's half exactly, so `small` and
        # `smaller` could not see the asymmetry the septum actually produces (birth volumes CV
        # 0.27) and `competition` compared a target with a measured `Vbirth`. All three read the
        # measured volume through `cell_size` now -- one reader, the one the rules use.
        if what in ("growth", "volume"):
            lvl = H.level(self.at)
            v_now, _ = cell_size(lvl, m, nF)
            if what == "volume":
                return np.asarray(v_now, np.float64)
            vb = cell_block(H, self.cat, "Vbirth", nF)
            if vb is None:
                return None
            return np.maximum(np.asarray(v_now, np.float64) / np.maximum(vb, 1e-12) - 1.0, 0.0)
        if what == "age":
            a = m.get("age")
            return None if a is None else a.detach().cpu().numpy()[:nF]
        if what == "act":
            clvl = H.level(self.cat)
            if clvl is None or "chem" not in getattr(clvl, "state_schema", {}):
                return None
            h0, _ = clvl.state_schema["chem"]
            return clvl.state[:nF, h0 + self.chan].detach().cpu().numpy()
        # ANY FIELD PUBLISHED ON THE MESH, BY NAME. The four above are hard-coded because they are
        # the engine's own bookkeeping (volume, birth volume, age, chemistry); anything an operator
        # measures and publishes arrives here instead, so a new probe needs no branch of its own.
        # `cell_shape_probe` writes `elong`; asking for a name nothing published returns None,
        # which every caller reads as "unknown", never as zero.
        v = m.get(what)
        if v is None:
            return None
        v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        return v[:nF] if v.ndim == 1 and v.shape[0] >= nF else None

    def _admit(self, flag, want, m, H, nF):
        """Turn the mode's proposal into the marks actually TAKEN this tick.

        A DEATH RATE, NOT A DEATH SET. Every state-defined rule re-evaluates each frame, so on a
        growing tissue the marks ACCUMULATE: measured on the campaign's own best parents, `smaller`
        killed 1,660 of 7,424 cells and took the run from protr 1.513 / grip 0.228 to 1.131 /
        0.049, and every mode but `crowded` did something comparable. The thresholds were
        calibrated on a static 400-cell sheet where a population is marked ONCE; nothing in them
        bounds a flux, and a threshold on a state variable never can -- it names a set, and the set
        refills as fast as the tissue can produce members of it.

        `max_mark_frac` caps how much of the tissue may be under sentence AT ONCE, so the mode
        chooses WHO dies and this chooses HOW FAST. `crowded` is the existence proof that a gentle
        rate leaves a run intact: 104 deaths, no premise broken, grip 0.216 -> 0.116 where the
        others reached ~0.01.

        THE NAMED POPULATIONS ARE EXEMPT, because the cap bounds a flux and a set chosen once has
        none: `apopgeo_half` deliberately sentences 45% of the sheet in one act, and capping it
        would silently delete the hardest topology test this operator has.
        """
        if self.mode in ("list", "band", "cone"):
            for f in want:
                flag[f] = 1.0
            return flag
        held = np.where(flag > 0)[0]
        room = max(1, int(self.max_mark_frac * nF)) - len(held)
        if room <= 0:                       # the queue is full: let it drain before sentencing more
            return flag
        fresh = np.array(sorted(want - {int(x) for x in held}), dtype=np.int64)
        if len(fresh) > room:
            # WORST FIRST once the cap bites, so the rule still means what it says: a competition
            # rule that culls an arbitrary subset of its losers is a lottery wearing the rule's name.
            sev = self._severity(m, H, nF, fresh)
            fresh = fresh[np.argsort(sev)[:room]] if sev is not None else fresh[:room]
        flag[fresh] = 1.0
        return flag

    def _severity(self, m, H, nF, idx):
        """How badly each of `idx` qualifies -- SMALLEST DIES FIRST. Only consulted when the rate
        cap bites and the mode has named more cells than may die at once; a competition rule that
        then culls an arbitrary subset of its losers would be a lottery wearing the rule's name."""
        _Q = {"competition": "growth", "smaller": "volume", "dimmer": "act", "older": "age"}
        if self.mode in _Q:
            q = self._q(m, H, nF, _Q[self.mode])
            if q is None:
                return None
            nb = self._nb(m, nF, q)[0]
            r = q[idx] / np.maximum(nb[idx], 1e-12)
            return -r if self.mode == "older" else r        # `older` loses by being ABOVE
        if self.mode in ("small", "stalled"):
            q = self._q(m, H, nF, "volume" if self.mode == "small" else "growth")
            return None if q is None else q[idx]
        if self.mode == "chem_low":
            q = self._q(m, H, nF, "act")
            return None if q is None else q[idx]
        if self.mode == "crowded":
            return -self._nb(m, nF, np.ones(nF))[1][idx].astype(float)   # most crowded first
        if self.mode == "lonely":
            return self._nb(m, nF, np.ones(nF))[1][idx].astype(float)
        if self.mode in ("field_high", "field_low"):
            q = self._q(m, H, nF, self.field)
            if q is None:
                return None
            # SMALLEST DIES FIRST is this method's contract, so `field_high` -- where the WORST
            # offender is the largest value -- has to be negated or the cap would spare exactly
            # the cells the mode exists to remove.
            return (-q[idx]) if self.mode == "field_high" else q[idx]
        return None                                          # list/band/cone: a named population

    def _marked(self, m, H, nF):
        """The set of cell indices currently marked to die."""
        if self.mode == "list":
            return {c for c in self.cells if c < nF}
        # THE CENTROID IS A PRECONDITION OF TWO MODES, NOT OF DEATH. This guard used to stand here
        # unconditionally -- `if centroid is None: return set()` -- so every mode in the operator was
        # silently gated behind a quantity only `band` and `cone` read. A geometric precondition
        # that switches off a competition rule, or a probe-driven one, is the same defect as an
        # absolute threshold on a relative field: the operator goes quiet for a reason unrelated to
        # its own mechanism, and quiet is indistinguishable from "nothing qualified".
        u = None
        if self.mode in ("band", "cone"):
            centroid = m.get("centroid_np")                                  # per-cell centroid from the rings
            if centroid is None:
                return set()
            u = centroid / (np.linalg.norm(centroid, axis=1) + 1e-12)[:, None]
        if self.mode == "band":                                    # `n_bands` latitude rings
            lat = np.degrees(np.arcsin(np.clip(u[:, 2], -1, 1)))   # -90 .. +90
            if self.n_bands <= 1:
                return set(np.where(np.abs(lat) < self.band_deg)[0].tolist())
            # evenly spaced in latitude, endpoints excluded so no ring sits on a pole
            centres = np.linspace(-90.0, 90.0, self.n_bands + 2)[1:-1]
            hit = np.zeros(len(lat), bool)
            for c in centres:
                hit |= np.abs(lat - c) < self.band_deg
            return set(np.where(hit)[0].tolist())
        if self.mode == "cone":                                     # one contiguous cap
            ax = np.array([0.0, 0.0, 1.0])
            return set(np.where(np.degrees(np.arccos(np.clip(u @ ax, -1, 1)))
                                < self.cone_deg)[0].tolist())
        if self.mode == "small":
            # DEATH BY SIZE, and the one selection that is genuinely about the cell's own state
            # rather than where it sits. A cell whose volume has fallen below `small_frac` of
            # v_ref -- the seed-time median -- is squeezed out, which is what an epithelium does
            # with a cell it can no longer accommodate. It re-evaluates for the same reason
            # `chem_low` does: a cell arrives in this set by shrinking, not by being pushed.
            vv = self._q(m, H, nF, "volume")                           # measured, see `_q`
            if vv is None:
                return set()
            v_ref = size_ref(m)                                        # the same convention: see `cell_size`
            return set(np.where(vv < self.small_frac * v_ref)[0].tolist())
        if self.mode == "stalled":
            # CELL COMPETITION: a cell that is not growing while its neighbours are gets removed.
            #
            # RELATIVE, WHICH IS THE WHOLE POINT. `chem_low` marks every cell below a fraction of
            # the activator's maximum, so when the pattern weakens it marks the WHOLE TISSUE:
            # every cell shrinks, the activator maximum goes to zero, and not one cell is ever
            # extruded -- because death needs a cell squeezed to a triangle by neighbours that are
            # NOT shrinking too. A threshold against the population's own
            # median cannot do that: it always names a minority, by construction.
            #
            # V0f/Vbirth is growth since birth and `age` is time since birth. Both are already
            # carried across renumbering by `keep` -- in cell_divide and in this operator -- so this
            # needs no new bookkeeping, which is the part that has gone wrong twice already.
            v = m.get("V0f"); vb = cell_block(H, self.cat, "Vbirth", nF); ag = m.get("age")
            if v is None or vb is None:
                return set()
            vv = v.detach().cpu().numpy()[:nF]
            vbb = np.maximum(vb, 1e-12)
            # THE EXCESS OVER BIRTH SIZE, NOT THE RATIO -- and the first version compared ratios,
            # which cannot work. V0f/Vbirth is >= 1 for any cell that has grown at all and exactly
            # 1.0 for one that has not, so with a median near 1.4 a cut at 0.5 x median = 0.7 sits
            # BELOW the floor and marks nobody. Measured: zero deaths across three runs and
            # hundreds of frames. `g = ratio - 1` is fractional growth SINCE BIRTH -- 0 for a
            # stalled cell -- and half the median excess is the comparison that was intended. On a
            # synthetic population of 400 stalled cells among 1,600 growing ones: the ratio form
            # marks 0, the excess form marks 618.
            g = np.maximum(vv / vbb - 1.0, 0.0)
            med = float(np.median(g))
            # A GLOBALLY STALLED TISSUE IS NOT A TISSUE FULL OF LOSERS. If the median cell has not
            # grown, there is no competition to lose and this marks nobody -- otherwise the same
            # runaway that chem_low produced would return by another route.
            if med < self.stall_margin:
                return set()
            old_enough = (ag.detach().cpu().numpy()[:nF] >= self.min_age) if ag is not None \
                else np.ones(nF, bool)
            return set(np.where((g < self.stall_frac * med) & old_enough)[0].tolist())
        # ---- THE LOCAL FAMILY. All five compare a cell with the cells TOUCHING it; they differ
        # only in what is compared, which is why they share `_loser`. Each is a different
        # biological hypothesis about why a cell is eliminated, not a different way to compute one.
        if self.mode == "competition":            # grows slower than its neighbours -- Myc-style
            q = self._q(m, H, nF, "growth")       # cell competition, the loser is out-proliferated
            return set() if q is None else self._loser(m, nF, q, below=True)
        if self.mode == "smaller":                # smaller than its neighbours: squeezed out by a
            q = self._q(m, H, nF, "volume")       # tissue that can no longer accommodate it
            return set() if q is None else self._loser(m, nF, q, below=True)
        if self.mode == "dimmer":                 # less activator than its neighbours -- the LOCAL
            q = self._q(m, H, nF, "act")          # form of chem_low, which failed by being global
            return set() if q is None else self._loser(m, nF, q, below=True)
        if self.mode == "older":                  # has gone longer without dividing than its
            q = self._q(m, H, nF, "age")          # neighbours: a cell that stopped cycling
            return set() if q is None else self._loser(m, nF, q, below=False)
        if self.mode in ("crowded", "lonely"):
            # GAME OF LIFE, on the real adjacency rather than a lattice. A closed trivalent sheet
            # gives every cell about six neighbours, so `crowded` fires where division has packed a
            # region and `lonely` only after deaths have already thinned one -- which makes
            # `lonely` a rule about how a wound SPREADS or heals, and worth having for that alone.
            deg = self._nb(m, nF, np.ones(nF))[1]
            ag = m.get("age")
            old_enough = (ag.detach().cpu().numpy()[:nF] >= self.min_age) if ag is not None \
                else np.ones(nF, bool)
            if self.mode == "crowded":
                return set(np.where(old_enough & (deg >= self.n_max))[0].tolist())
            return set(np.where(old_enough & (deg > 0) & (deg <= self.n_min))[0].tolist())
        if self.mode == "chem_low":                                 # die BETWEEN the spots
            clvl = H.level(self.cat)
            if clvl is None or "chem" not in getattr(clvl, "state_schema", {}):
                return set()
            h0, _h1 = clvl.state_schema["chem"]
            a = clvl.state[:nF, h0 + self.chan].detach().cpu().numpy()
            amax = float(np.nanmax(a)) if a.size else 0.0
            if amax <= 1e-9:
                return set()
            return set(np.where(a < self.a_sw * amax)[0].tolist())
        if self.mode in ("field_high", "field_low"):
            # DEATH PLUGGED ONTO A MEASUREMENT SOMEONE ELSE MADE, which is the whole point of the
            # pair. Every other mode in this operator computes its own criterion internally --
            # eleven bespoke measurements living inside one Die -- and the only one that did not
            # was `chem_low`, which reads the `chem` another operator wrote. These two generalise
            # that: name a field, and any operator that publishes it can drive death without a
            # line changing here. `cell_shape_probe` publishes `elong`; the next probe will
            # publish something else and this will already work.
            q = self._q(m, H, nF, self.field)
            if q is None:
                # UNPUBLISHED IS NOT ZERO. No probe ran, or its precondition was absent -- either
                # way nothing is known about these cells, and killing none of them is the only
                # honest reading. Killing all of them is what a `0 > thr` comparison on a
                # zero-filled array would have done for `field_low`.
                return set()
            ok = np.isfinite(q)
            if not ok.any():
                return set()
            # RELATIVE TO THE FIELD'S OWN SPREAD, for the reason every threshold here is: `elong`
            # is a shape index near 3.7 and an aspect near 1.5, two quantities with no common
            # scale, and one `frac` has to mean the same thing on both. The reference is the
            # MEDIAN of the live cells, not the max: one runaway cell must not set the scale of
            # the quantity that exists to find it. `frac` 1.5 with mode field_high therefore reads
            # "half again the typical cell".
            med = float(np.median(q[ok]))
            if not np.isfinite(med) or abs(med) < 1e-12:
                return set()
            thr = self.field_frac * med
            hit = (q > thr) if self.mode == "field_high" else (q < thr)
            return set(np.where(ok & hit)[0].tolist())
        return set()

    def forward(self, H, mask=None):
        # THE PAIRING IS READ FROM THE SET, ONCE PER CALL -- see `resolve_cell_set`.
        self.cat = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
        # THE ACTED LEDGER IS BLIND TO THIS OPERATOR, AND TO THE WHOLE DIE FAMILY. Measured over
        # rounds r001-r012 of the live campaign: 24 runs killed cells, 6,693 deaths in total, and
        # `inert_operators` recorded `cell_die` as having acted in ZERO of them -- including
        # runs that extruded 200 cells.
        #
        # The cause is structural, in both senses. `instrument._wrap` decides `acted` from
        # `_nonzero(out) or (before != after)`, where `before`/`after` fingerprint the LEVEL STATE.
        # A Die operator returns `{}` -- it has no delta to contribute -- and does its work by
        # mutating the mesh: `nF`, the half-edge table, `alive`, and the per-face arrays. None of
        # that is in the fingerprint, so the engine cannot see it and infers "nothing happened".
        #
        # HARMLESS TODAY AND NOT SAFE TO LEAVE. It never lands in `inert_operators`, so the
        # inert-implies-inconclusive rule in `round.score` cannot misfire on it. But the ledger is
        # the only cheap evidence that an operator RAN, and for this family it says nothing at all
        # -- so a death run and a run where death was silently unreachable are indistinguishable in
        # the record. That is the failure this project has paid for with `rd_interface_tension`
        # (written off inert twice, never having fired) and `cell_chem_from_shape` (100% acted, changing
        # nothing).
        #
        # THE FIX IS FOR THE OPERATOR TO REPORT, NOT FOR THE ENGINE TO GUESS. Fingerprinting the
        # mesh would work and is the wrong shape: the engine would be inferring from a hash what
        # this operator already knows exactly. `m["n_apop"]` is incremented here on every
        # extrusion, so the operator can hand back a liveness record -- deaths this tick, cells
        # marked, cells waiting to be shed -- and the ledger can record a fact instead of a
        # deduction. That record is also what gate G13 needs: the frame a cell crosses the
        # extrusion threshold, separate from the frame it is shed, which no downstream sampling
        # stride can recover.
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        # LOCAL, as Divide3D does: topology_ops imports from this module, so a
        # module-level import here is circular.
        from plexus.models.topology import (rings_from_flat_3d, flat_from_rings_3d,
                                           face_collapse_3d)
        births: list = []                        # see Divide3D: the per-vertex carry's parentage
        self._k += 1
        dev = m["V0f"].device; dt = m["V0f"].dtype
        nF = int(m["nF"]); Nv = int(m["Nv"])
        px0, px1 = lvl.state_schema["pos"]
        pos_t = lvl.state[:Nv, px0:px1].detach().cpu().numpy().astype(np.float64)
        es = m["E_srce"].cpu().numpy(); et = m["E_trgt"].cpu().numpy(); ef = m["E_face"].cpu().numpy()
        rings = [np.asarray(r, np.int64) for r in rings_from_flat_3d(es, et, ef, nF)]
        # per-cell centroid, for the geometric modes
        m["centroid_np"] = np.array([pos_t[r].mean(0) if r is not None and len(r) else np.zeros(3)
                                for r in rings])
        # MARKED CELLS ARE A PER-FACE FLAG, NOT A LIST OF INDICES, and the first version was a
        # list. Every collapse renumbers the faces through `keep`, so `cells: [900]` pointed at a
        # DIFFERENT cell after each death: marking one cell killed seven in the first smoke run,
        # a cascade down the index space. The flag rides the same `keep` map as `alive` and `age`,
        # so a cell stays the cell it was.
        # AS NUMPY, BECAUSE THIS OPERATOR IS NUMPY. `apop_flag` was a float64 numpy array on the
        # mesh table -- `m["apop_flag"] = np.asarray(...)` a few hundred lines down -- and the whole
        # marking path is written in numpy: `np.where`, fancy indexing, set arithmetic. On the cell
        # set it is a float32 CUDA tensor like every other declared block, so it is converted once
        # here rather than each of the six places that assume an array. The values are a 0/1 flag,
        # so float64 -> float32 -> float64 is exact and the run stays byte-identical.
        flag = m.get("apop_flag")
        if flag is not None and hasattr(flag, "detach"):
            flag = flag.detach().cpu().numpy().astype(np.float64)
        if flag is None or len(flag) != nF:
            base = np.zeros(nF, np.float64)
            if flag is not None:                       # carried across a topology change already
                base[:min(len(flag), nF)] = flag[:min(len(flag), nF)]
            flag = base
        # AN INDEX IS NOT AN IDENTITY, A REGION IS -- so `list` marks ONCE and the state-defined
        # modes re-evaluate. `cells: [900]` names the cell at index 900 AT THE MOMENT IT IS READ,
        # and every collapse renumbers the faces through `keep`; re-reading it each frame therefore
        # hands the mark to a fresh victim after every death. Measured before this guard: marking
        # one cell killed three, at ticks 86, 164 and 236, each reporting exactly one marked cell.
        #
        # This is the seed-window argument of the Plexus2 paper in a second place. There, an
        # initial condition re-applied every timestep "converts it from an initial condition into a
        # moving boundary condition"; here a one-off selection re-applied every timestep converts a
        # death sentence into a death RATE. `seed` is a kind so the runtime can impose the window;
        # this operator is `structural`, so it imposes its own.
        _want = set()
        if self.mode in ("list", "band", "cone"):     # a POPULATION chosen once; see below
            if not m.get("apop_marked_once"):
                _want = {f for f in self._marked(m, H, nF) if f < nF}
                m["apop_marked_once"] = True
        else:
            # chem_low IS THE ONLY MODE THAT MAY RE-EVALUATE, and band/cone sat on this branch
            # until it was measured. A geometric region LOOKS like state and is not: as the equator
            # constricts, fresh cells rotate INTO the band and are marked, so a ring of 24 became a
            # mouth that ate the sphere -- 2,000 cells down to 497 by frame 270. A ring or a cap is
            # a POPULATION chosen at one moment, exactly like an index list. Only a chemical
            # threshold is a genuine ongoing condition: a cell whose activator has fallen should
            # now die, and no cell enters that set merely by being pushed.
            _want = {f for f in self._marked(m, H, nF) if f < nF}
        flag = self._admit(flag, _want, m, H, nF)
        m["apop_flag"] = flag
        marked = {int(f) for f in np.where(flag > 0)[0]}
        if not marked:
            return {}
        V0f = m["V0f"].detach().cpu().numpy().astype(np.float64)
        A0 = m["A0"].detach().cpu().numpy().astype(np.float64)
        # `critical_frac * v_ref` IN THE CONVENTION `V0f` IS ACTUALLY IN -- see `cell_size`. It was
        # the wedge reference on every run, so an apicobasal cell was extruded at a fraction of a
        # volume it does not have.
        # THE CRITERION IS ON A TARGET, SO ITS REFERENCE IS THE TARGET'S. `size_tgt` below is `V0f`
        # (or `A0` on a sheet), a quantity in the SEED's convention -- wedge on every apico-basal
        # spec in the tree -- and `crit` is a fraction of it. `cell_size` reads the polyhedron for
        # a set with a separation (R2b), which is right for what a cell HAS and wrong for what it
        # ASKS: against the polyhedron median apop2_ks0p1 shed 325 cells by frame 37 where the
        # working point sheds none. The reference here is the seed-time median in the target's
        # own units, which is what it was before R2b.
        _vn, _ = cell_size(lvl, m, nF, pos_t)
        _by_area = m.get("size_convention") == "area"
        v_ref = float(m.get("a_ref", 0.0)) if _by_area else float(m.get("v_ref", 1.0))
        crit = self.crit * v_ref
        # THE TARGET THAT IS SHRUNK, IN THE SAME CONVENTION AS `crit`: the volume target on a
        # shell, the area target on a flat sheet (where V0f is identically zero -- see
        # `cell_size`). Read AFTER the shrink below, through `size_tgt()`.
        _by_area = m.get("size_convention") == "area"
        # 1. SHRINK. cell_mechanics contracts the cell toward the smaller target; T1 then finds its
        #    edges short and sheds neighbours. Nothing is removed here.
        for f in marked:
            if f < nF:
                V0f[f] = max(V0f[f] * (1.0 - self.shrink), 1e-9)
                A0[f] = max(A0[f] * (1.0 - self.shrink) ** (2.0 / 3.0), 1e-9)
        m["V0f"] = torch.as_tensor(V0f, dtype=dt, device=dev)
        m["A0"] = torch.as_tensor(A0, dtype=dt, device=dev)
        size_tgt = A0 if _by_area else V0f
        # THE SHRINK HAS TO SURVIVE `cell_grow`, AND WRITING `V0f` ALONE DOES NOT. `cell_grow`
        # recomputes the target wholesale every frame -- `m["V0f"] = m["V0f_init"] * s**3` -- from
        # the cell's own growth scale, so a shrink written only to `V0f` is DISCARDED on the next
        # tick, not merely outpaced by it. `cell_die` runs every 4 frames and `cell_grow` every 1,
        # so three ticks in four erase it and the fourth is erased immediately after.
        #
        # MEASURED on gate_ab_population, 401 frames: 52 cells marked, `V0f` never reaching the
        # `critical_frac * v_ref` the collapse requires, and ZERO extrusions -- while the row that
        # exists to catch a vacuous population rung reported exactly that.
        #
        # IT NEVER SHOWED IN THE OKUDA DEATH SPECS because they gate growth on chemistry (`rho:
        # 0.1`), so a marked cell in a low-activator region grows at a tenth of the rate and the
        # shrink still wins. A spec with uniform growth (`rho: 1.0`) is the case that exposes it,
        # and this promotion's population rung is the first to run one.
        #
        # So the shrink is applied to the quantity `cell_grow` DERIVES from. The growth scale
        # itself is left alone: it is the cell's clock, and a dying cell has not stopped ageing,
        # it has stopped being big.
        for _k, _p in (("V0f_init", 1.0), ("A0_init", 2.0 / 3.0), ("P0_init", 1.0 / 3.0)):
            _a = m.get(_k)
            if _a is None:
                continue                     # no `cell_grow` in this schedule: nothing to survive
            _v = _a.detach().cpu().numpy().astype(np.float64)
            if len(_v) < nF:
                continue
            for f in marked:
                if f < nF:
                    _v[f] = max(_v[f] * (1.0 - self.shrink) ** _p, 1e-9)
            m[_k] = torch.as_tensor(_v, dtype=dt, device=dev)
        # P0 FOLLOWS A0, as it does everywhere else in this file: the target perimeter of a cell
        # with target area A is p0*sqrt(A). Leaving it behind would make the shrinking cell chase a
        # perimeter its area no longer supports, which is a shape-index error, not a size one.
        m["P0"] = torch.as_tensor(self.p0 * np.sqrt(np.maximum(A0, 1e-9)), dtype=dt, device=dev)
        # 2. EXTRUDE the ones that are now triangles AND small enough.
        # A DYING CELL'S CONTENTS GO SOMEWHERE: the sum of activity in the dying cell's vicinity
        # should not change much, by construction.
        #
        # Simply dropping the row from the cell state takes its activator and inhibitor out of the
        # tissue with it. That is a discontinuity in a conserved quantity, injected at every death
        # and accounted for nowhere -- a cell is extruded and its
        # chemistry is simply gone. Real extrusion hands the contents to the neighbours that close
        # over the gap.
        #
        # AMOUNT, NOT CONCENTRATION. The cell holds a * v; the neighbours receive that amount and
        # each converts it back to a concentration by its own volume. Distributing the
        # CONCENTRATION would create or destroy material whenever the neighbours are a different
        # size from the deceased, which is exactly the accounting error this fixes.
        clvl_c = H.level(self.cat)
        _has_chem = clvl_c is not None and "chem" in getattr(clvl_c, "state_schema", {})
        _bequest = []                       # (neighbour indices, amount per channel)
        gone = 0
        for f in sorted(marked, reverse=True):
            if f >= nF or rings[f] is None:
                continue
            if size_tgt[f] > crit:
                continue
            _kn_f = self.nb_sides_for(float(size_tgt[f]), crit) if hasattr(self, "nb_sides_for") else None
            if not (self._to_triangle(rings, pos_t, f, keep_nb_sides=_kn_f) if _kn_f is not None
                    else self._to_triangle(rings, pos_t, f)):  # the default waits for `edge_flip`
                continue
            nbrs = [int(g) for g in self._ring_neighbours(rings, f) if g < nF]
            if face_collapse_3d(rings, pos_t, f, births=births):
                gone += 1
                if _has_chem and nbrs:
                    h0, h1 = clvl_c.state_schema["chem"]
                    amt = clvl_c.state[f, h0:h1].detach().clone() * float(size_tgt[f])
                    _bequest.append((nbrs, amt))
        if gone == 0:
            return {}
        if _bequest:
            cs = clvl_c.state.clone()
            h0, h1 = clvl_c.state_schema["chem"]
            for nbrs, amt in _bequest:
                # DO NOT POUR INTO A CELL THAT IS ITSELF VANISHING. The divisor is the recipient's
                # volume, and a cell that is MARKED but cannot reach a triangle shrinks to the
                # 1e-9 floor and stays there -- so a bequest divided by that floor injects a
                # concentration many orders of magnitude too large in one step, and a few of those
                # compound. Without this guard the conservation rule above becomes an AMPLIFIER,
                # by dividing by a number the model allows to reach zero.
                live = [g for g in nbrs if float(size_tgt[g]) >= crit]
                if not live:
                    # every neighbour is on its way out too: the material leaves with the cell,
                    # which is what happened before conservation existed and is the honest
                    # alternative to inventing somewhere to put it
                    continue
                # AND THE BEQUEST MUST LEAVE THE RECIPIENT INSIDE THE INTEGRATOR'S BASIN.
                # Conservation is correct as conservation and unbounded as CONCENTRATION: the
                # increment is share/V_g, and Gray-Scott integrates stably only while its
                # activator stays in the region its own kinetics bound. A bequest that pushes a
                # neighbour outside it diverges under explicit Euler, and the divergence is not
                # gentle -- measured on tsd_cap10 and tsd_cap25, act_mean_floor reached -7.29e11
                # and -2.42e23 before going NaN at frame 1430, while tsd_cap10_fast with FIVE
                # TIMES the deaths stayed clean at 0.0465. The amount of death is not the
                # variable; the two failing runs are the SLOW ones (shrink 0.05 against 0.15), so
                # cells linger near the extrusion threshold and their neighbours receive many
                # small bequests instead of few.
                #
                # `ceil` is the field's own running maximum, which is the only scale available
                # that the chemistry itself sets -- the same argument that makes a_sw a fraction
                # rather than an absolute. Material that cannot be placed without leaving the
                # basin is DROPPED AND COUNTED, never silently injected: a conservation law that
                # is quietly violated is worse than one that reports where it failed, because the
                # first is indistinguishable from a working one.
                share = amt / float(len(live))
                # TORCH, NOT NUMPY: `cs` is the cell state and lives on the GPU. `np.nanmax` on a
                # CUDA tensor raises "can't convert cuda:0 device type tensor to numpy", which
                # killed the two runs this bound exists to fix -- so the fix's first outing failed
                # for a reason unrelated to the fix. Hoisted out of the per-neighbour loop too: it
                # is a property of the field, not of the recipient.
                # ELEMENTWISE, AND PER COLUMN. The bequest spans every chemistry channel the cell
                # carries, so `inc` and `room` are vectors: a scalar comparison on them raises
                # "Boolean value of Tensor with more than one value is ambiguous", and a single
                # ceiling taken from channel 0 would clamp the substrate against the activator's
                # maximum. Each column is bounded by its own.
                ceil = torch.nan_to_num(cs[:nF, h0:h1], nan=0.0).amax(dim=0) if nF else None
                for g in live:
                    inc = share / float(size_tgt[g])       # the recipient's size, in the run's convention
                    if ceil is None:
                        cs[g, h0:h1] += inc
                        continue
                    room = (ceil - cs[g, h0:h1]).clamp(min=0.0)
                    fits = torch.minimum(inc, room)
                    over = (inc - fits).sum()
                    if float(over) > 0:
                        m["apop_spill"] = m.get("apop_spill", 0.0) + float(over) * float(V0f[g])
                    cs[g, h0:h1] += fits
            clvl_c.state = cs
        # 3. REBUILD, exactly as cell_divide does: `keep` maps new face -> old face and carries every
        #    per-face array across, so nothing can fall out of step with the mesh.
        # THE PER-VERTEX CARRY, spent here for the same reason `reindex_faces` is spent beside the
        # per-face one: a topology edit that changes who a row IS must be followed by every array
        # that is indexed by that row, in the same place, or the fourth operator does not get the
        # memo. With no `vertex_carry` declared this walks an empty name list and returns, so every
        # existing spec is byte-identical -- which is the claim R1 has to make and the twin measures.
        if hasattr(m, "carry_vertices") and births:
            m.carry_vertices(births, dt=dt, dev=dev, level=lvl)
        es2, et2, ef2, nF2, keep = flat_from_rings_3d(rings)
        def _car(name, arr=None):
            a = (arr if arr is not None else m[name].detach().cpu().numpy().astype(np.float64))
            return torch.as_tensor(np.asarray([a[i] for i in keep], np.float64), dtype=dt, device=dev)
        # `_once`: the names served from the cell set, when this implementation carries ONCE (see
        # `CARRY_ONCE`); empty for the default, whose lines below then run exactly as they always did.
        _once = set(getattr(m, "_proxied", ()) or ()) if self.CARRY_ONCE else set()
        if "A0" not in _once:
            m["A0"] = _car("A0", A0)
        if "V0f" not in _once:
            m["V0f"] = _car("V0f", V0f)
        _a0 = (np.asarray([A0[i] for i in keep], np.float64) if "P0" not in _once and "A0" in _once
               else m["A0"].detach().cpu().numpy() if "P0" not in _once else np.asarray(A0, np.float64))
        m["P0"] = torch.as_tensor(self.p0 * np.sqrt(np.maximum(_a0, 1e-9)), dtype=dt, device=dev)
        # `Vbirth` AND `divjit` ARE NOT HERE ANY MORE, and their absence is the point: they are
        # blocks on the cell set, and `H.renumber_set(self.cat, keep, nF2)` a few lines below
        # permutes the whole of that set's state with the same `keep`. Carrying them here as well
        # would permute them twice.
        for nm in ("age", "ndiv", "alive"):
            if nm in m and nm not in _once:
                m[nm] = _car(nm)
        _carry_face_state(m, keep, dt, dev)
        m["apop_flag"] = (flag if "apop_flag" in _once
                          else np.asarray([flag[i] for i in keep], np.float64))   # identity, across renumbering
        st = lvl.state.clone()
        st[:len(pos_t), px0:px1] = torch.as_tensor(pos_t, dtype=dt, device=dev)
        lvl.state = st
        es2, et2 = self._after_topology(H, m, es2, et2)   # after the positions are written
        m["E_srce"] = torch.as_tensor(es2, device=dev); m["E_trgt"] = torch.as_tensor(et2, device=dev)
        m["E_face"] = torch.as_tensor(ef2, device=dev); m["nF"] = nF2
        # THE CELL SET FOLLOWS THE MESH. `chem` is indexed by face, so a death that compacts the
        # faces and not the chemistry silently re-assigns every activator value to a different
        # cell -- the same class of error as a per-face target left behind by division.
        clvl = H.level(self.cat)
        if clvl is not None and clvl.state.shape[0] >= nF:
            # MEASURED INNOCENT, and worth recording because it was the obvious suspect. At every
            # death min(a) BEFORE this copy equals min(a) AFTER it -- tick 34 read -0.03872 on
            # both sides. The negative activator that breaks P12 on every run where cells die
            # arrives from the reaction/diffusion step operating on a mesh a death has just
            # perturbed, not from this reindex. Ruled out by experiment as well: removing
            # cell_chem_from_shape left it at -0.0948, and quartering chi left it at -0.0689.
            # ONE ENGINE CALL: state, occupancy, the coordinate delta accumulator AND the extra
            # first-order block deltas, which the hand-rolled version below never reached (its
            # guard tests `isinstance(key, tuple)` and `_delta_blocks` is keyed by level NAME).
            if not H.renumber_set(self.cat, keep, n_new=nF2):
                # LOUD, BECAUSE THE SILENT VERSION COST A WEEK. This returned False on every call
                # for the whole promotion and both call sites discarded it; the code it replaced at
                # least printed a warning. The counter also rides `MeshTable.SCALAR_RECORD`, so a
                # gate row can assert it stayed zero instead of a human having to read a log.
                m["renumber_failed"] = int(m.get("renumber_failed", 0)) + 1
                print(f"[cell_die] renumber_set({self.cat!r}) DID NOT ACT -- the cell state, its "
                      f"occupancy and its pending deltas are now mis-indexed against the mesh, and "
                      f"the chemistry will scramble from here", flush=True)
        # THE PENDING DELTAS MUST BE RENUMBERED TOO, AND THIS IS THE BUG THAT BROKE P12 ON EVERY
        # RUN WHERE A CELL DIED.
        #
        # The engine zeroes the delta accumulator once per TICK and integrates at the END of the
        # schedule, so `cell_chem_diffuse` and `cell_chem_react` deposit their per-cell deltas early and the
        # engine applies them last. An operator that RENUMBERS the set in between leaves every one
        # of those deltas pointing at a different cell -- a large negative flux meant for one cell
        # lands on another that has almost no activator, and the concentration goes negative. That
        # is exactly what happens: the activator goes NEGATIVE on every model that kills anything,
        # decaying afterwards as the scrambled deltas diffuse away, while a no-death control never
        # leaves zero.
        #
        # `cell_divide` never hit it because appending is not renumbering: daughters go to indices
        # >= nF and every existing cell keeps its own. Removal is the first operation in this
        # engine that moves a row.
        #
        # Permuting the accumulator with the SAME `keep` map fixes it wherever the operator sits
        # in the schedule. Confirmed independently by moving apoptosis ahead of the chemistry --
        # act_min held at 0.0000 for the whole run and P12 and P4 both cleared -- but that is a
        # discipline every spec would have to remember, and this is the guarantee instead.
        # (the deltas were renumbered by `H.renumber_set` above, together with the state)
        m["n_apop"] = int(m.get("n_apop", 0)) + gone
        # LOUD, because a death is not recoverable and the count must be auditable against the
        # cell count: nF -> nF2 should differ by exactly `gone` and by nothing else.
        print(f"[cell_die] tick {self._k}: extruded {gone} cell(s), {nF} -> {nF2} faces "
              f"(marked {len(marked)}, total {m['n_apop']})", flush=True)
        return {}


@register_operator("cell_die", model="competition", set="vertex", kind="die",
                   family="population", title="remove slower-growing cells", equation=r"""$$\text{mark } j\ \text{when}\ g_j<s\,\overline{g}_{k\sim j}\ \text{and}\ \overline{g}_k>m$$""")
class Apoptosis3DCompetition(Apoptosis3D):
    r"""`competition` model of cell_die: a cell growing more slowly than its neighbours is out-proliferated and removed.
    vertex -> vertex: as cell_die; this model only marks. Only what marks a cell for death differs between the `model:` variants of cell_die; the extrusion is the base's.

        \text{mark } j\ \text{when}\ g_j<s\,\overline{g}_{k\sim j}\ \text{and}\ \overline{g}_k>m
        g_j            the cell's fractional growth since birth, V0f_j / Vbirth_j
        \overline{g}_{k\sim j} the mean over its mesh neighbours
        s              `stall_frac`, a fraction of the neighbour mean
        m              `stall_margin`, the neighbour mean below which nothing is marked

    Reference: Morata, G. & Ripoll, P. (1975). Dev. Biol. 42:211-221; de la Cova, C. et al. (2004). Cell 117:107-116."""
    PARAM_UNITS = {}
    SYMBOLS = {'g_j': "the cell's fractional growth since birth, V0f_j / Vbirth_j", '\\overline{g}_{k\\sim j}': 'the mean over its mesh neighbours', 's': '`stall_frac`, a fraction of the neighbour mean', 'm': '`stall_margin`, the neighbour mean below which nothing is marked'}
    DEATH = "competition"


@register_operator("cell_die", model="smaller", set="vertex", kind="die",
                   family="population", title="remove cells smaller than neighbours", equation=r"""$$\text{mark } j\ \text{when}\ v_j<s\,\overline{v}_{k\sim j}\ \text{and}\ \overline{v}_k>m$$""")
class Apoptosis3DSmaller(Apoptosis3D):
    r"""`smaller` model of cell_die: a cell smaller than its neighbours is squeezed out.
    vertex -> vertex: as cell_die; this model only marks. Only what marks a cell for death differs between the `model:` variants of cell_die; the extrusion is the base's.

        \text{mark } j\ \text{when}\ v_j<s\,\overline{v}_{k\sim j}\ \text{and}\ \overline{v}_k>m
        v_j            the cell's target volume V0f, in world units cubed
        \overline{v}_{k\sim j} the mean over its mesh neighbours
        s              `stall_frac`, a fraction of the neighbour mean
        m              `stall_margin`

    Reference: Marinari, E. et al. (2012). Nature 484:542-545."""
    PARAM_UNITS = {}
    SYMBOLS = {'v_j': "the cell's target volume V0f, in world units cubed", '\\overline{v}_{k\\sim j}': 'the mean over its mesh neighbours', 's': '`stall_frac`, a fraction of the neighbour mean', 'm': '`stall_margin`'}
    DEATH = "smaller"


@register_operator("cell_die", model="dimmer", set="vertex", kind="die",
                   family="population", title="remove cells with less activator", equation=r"""$$\text{mark } j\ \text{when}\ a_j<s\,\overline{a}_{k\sim j}\ \text{and}\ \overline{a}_k>m$$""")
class Apoptosis3DDimmer(Apoptosis3D):
    r"""`dimmer` model of cell_die: a cell carrying less activator than its neighbours, the local form of `chem_low`.
    vertex -> vertex: as cell_die; this model only marks. Only what marks a cell for death differs between the `model:` variants of cell_die; the extrusion is the base's.

        \text{mark } j\ \text{when}\ a_j<s\,\overline{a}_{k\sim j}\ \text{and}\ \overline{a}_k>m
        a_j            the cell's activator concentration
        \overline{a}_{k\sim j} the mean over its mesh neighbours
        s              `stall_frac`, a fraction of the neighbour mean
        m              `stall_margin`

    Reference: Morata, G. & Ripoll, P. (1975). Dev. Biol. 42:211-221."""
    PARAM_UNITS = {}
    SYMBOLS = {'a_j': "the cell's activator concentration", '\\overline{a}_{k\\sim j}': 'the mean over its mesh neighbours', 's': '`stall_frac`, a fraction of the neighbour mean', 'm': '`stall_margin`'}
    DEATH = "dimmer"


@register_operator("cell_die", model="older", set="vertex", kind="die",
                   family="population", title="remove cells older than neighbours", equation=r"""$$\text{mark } j\ \text{when}\ \mathrm{age}_j>\overline{\mathrm{age}}_{k\sim j}/s$$""")
class Apoptosis3DOlder(Apoptosis3D):
    r"""`older` model of cell_die: a cell that has gone longer without dividing than its neighbours.
    vertex -> vertex: as cell_die; this model only marks. Only what marks a cell for death differs between the `model:` variants of cell_die; the extrusion is the base's.

        \text{mark } j\ \text{when}\ \mathrm{age}_j>\overline{\mathrm{age}}_{k\sim j}/s
        age_j          division calls since the cell was born
        \overline{age}_{k\sim j} the mean over its mesh neighbours
        s              `stall_frac`, the fraction the neighbour mean is divided by

    Reference: Morata, G. & Ripoll, P. (1975). Dev. Biol. 42:211-221."""
    PARAM_UNITS = {}
    SYMBOLS = {'age_j': 'division calls since the cell was born', '\\overline{age}_{k\\sim j}': 'the mean over its mesh neighbours', 's': '`stall_frac`, the fraction the neighbour mean is divided by'}
    DEATH = "older"


@register_operator("cell_die", model="crowded", set="vertex", kind="die",
                   family="population", title="remove cells with too many neighbours", equation=r"""$$\text{mark } j\ \text{when}\ \deg(j)\ge n_{\max}\ \text{and}\ \mathrm{age}_j\ge\mathrm{age}_{\min}$$""")
class Apoptosis3DCrowded(Apoptosis3D):
    r"""`crowded` model of cell_die: a cell with more neighbours than a trivalent sheet should give it.
    vertex -> vertex: as cell_die; this model only marks. Only what marks a cell for death differs between the `model:` variants of cell_die; the extrusion is the base's.

        \text{mark } j\ \text{when}\ \deg(j)\ge n_{\max}\ \text{and}\ \mathrm{age}_j\ge\mathrm{age}_{\min}
        \deg(j)        the number of cells sharing an edge with j, about six on a closed trivalent surface
        n_{max}        `n_max`, the neighbour count at which the cell dies
        age_{min}      `min_age`, in division calls

    Reference: Eisenhoffer, G. T. et al. (2012). Nature 484:546-549."""
    PARAM_UNITS = {}
    SYMBOLS = {'\\deg(j)': 'the number of cells sharing an edge with j, about six on a closed trivalent surface', 'n_{max}': '`n_max`, the neighbour count at which the cell dies', 'age_{min}': '`min_age`, in division calls'}
    DEATH = "crowded"


@register_operator("cell_die", model="small", set="vertex", kind="die",
                   family="population", title="remove cells below a size", equation=r"""$$\text{mark } j\ \text{when}\ v_j<s\,v_{\mathrm{ref}}$$""")
class Apoptosis3DSmall(Apoptosis3D):
    r"""`small` model of cell_die: below an absolute volume threshold.
    vertex -> vertex: as cell_die; this model only marks. Only what marks a cell for death differs between the `model:` variants of cell_die; the extrusion is the base's.

        \text{mark } j\ \text{when}\ v_j<s\,v_{\mathrm{ref}}
        v_j            the cell's target volume V0f
        v_{ref}        the seed-time median cell volume
        s              `small_frac`, a fraction of a typical cell at the start

    Reference: Marinari, E. et al. (2012). Nature 484:542-545."""
    PARAM_UNITS = {}
    SYMBOLS = {'v_j': "the cell's target volume V0f", 'v_{ref}': 'the seed-time median cell volume', 's': '`small_frac`, a fraction of a typical cell at the start'}
    DEATH = "small"


@register_operator("cell_die", model="stalled", set="vertex", kind="die",
                   family="population", title="remove cells that stop growing", equation=r"""$$\text{mark } j\ \text{when}\ g_j<s$$""")
class Apoptosis3DStalled(Apoptosis3D):
    r"""`stalled` model of cell_die: below an absolute growth threshold.
    vertex -> vertex: as cell_die; this model only marks. Only what marks a cell for death differs between the `model:` variants of cell_die; the extrusion is the base's.

        \text{mark } j\ \text{when}\ g_j<s
        g_j            the cell's fractional growth since birth
        s              `stall_frac`, here an absolute fraction

    Reference: Morata, G. & Ripoll, P. (1975). Dev. Biol. 42:211-221."""
    PARAM_UNITS = {}
    SYMBOLS = {'g_j': "the cell's fractional growth since birth", 's': '`stall_frac`, here an absolute fraction'}
    DEATH = "stalled"


@register_operator("cell_die", model="chem_low", set="vertex", kind="die",
                   family="population", title="remove cells low in activator", equation=r"""$$\text{mark } j\ \text{when}\ a_j<a_{sw}\max_k a_k$$""")
class Apoptosis3DChemLow(Apoptosis3D):
    r"""`chem_low` model of cell_die: below an absolute activator threshold, so cells die between the spots of a pattern.
    vertex -> vertex: as cell_die; this model only marks. Only what marks a cell for death differs between the `model:` variants of cell_die; the extrusion is the base's.

        \text{mark } j\ \text{when}\ a_j<a_{sw}\max_k a_k
        a_j            the cell's activator concentration
        a_{sw}         `a_sw`, a fraction of the pattern's own peak
        \max_k a_k     the maximum over the whole tissue

    Reference: Plexus (this work); death keyed to a morphogen the reaction-diffusion operators write."""
    PARAM_UNITS = {}
    SYMBOLS = {'a_j': "the cell's activator concentration", 'a_{sw}': "`a_sw`, a fraction of the pattern's own peak", '\\max_k a_k': 'the maximum over the whole tissue'}
    DEATH = "chem_low"


@register_operator("cell_die", model="chem_rate", set="vertex", kind="die",
                   family="population", title="remove cells at a fate-set rate", equation=r"""$$\mathrm{fate}_j=\arg\max_k \mathrm{chem}_j[k],\qquad P(\text{mark } j\ \text{per call})=r_{\mathrm{fate}_j}\cdot\text{every},\ t>t_{\mathrm{start}}$$""")
class Apoptosis3DChemRate(Apoptosis3D):
    r"""`chem_rate` model of cell_die: a cell leaves the epithelium at a rate set by its fate.
    vertex -> vertex: as cell_die; this model only marks, with probability per call. Only what marks a cell for death differs between the `model:` variants of cell_die; the extrusion is the base's.

        \mathrm{fate}_j=\arg\max_k \mathrm{chem}_j[k],\qquad P(\text{mark } j\ \text{per call})=r_{\mathrm{fate}_j}\cdot\text{every},\ t>t_{\mathrm{start}}
        fate_j         the cell's fate, the largest of its `fate_cols` columns (none below `fate_threshold`)
        r              `fate_rates`, the leaving rate per frame of each fate
        every          `every`, calls between evaluations
        t_{start}      `t_start`, the frame the leaving begins

    Reference: Kicheva, A. et al. (2014). Science 345:1254927 (Figs. 3A, 3C, the two-phase model)."""
    PARAM_UNITS = {'fate_rates': 'rate', 'fate_threshold': 'dimensionless', 't_start': 'frames', 'every': 'count', 'seed': 'count'}
    SYMBOLS = {'fate_j': "the cell's fate, the largest of its `fate_cols` columns (none below `fate_threshold`)", 'r': '`fate_rates`, the leaving rate per frame of each fate', 'every': '`every`, calls between evaluations', 't_{start}': '`t_start`, the frame the leaving begins'}
    PARAM_ROLES = {'every': 'calls_between_evaluations', 'fate_cols': 'chem_columns_of_the_fates', 'fate_rates': 'leaving_rate_per_frame_of_each_fate', 'fate_threshold': 'chem_level_below_which_a_cell_has_no_fate', 'seed': 'random_seed', 't_start': 'frame_the_leaving_begins'}
    DEATH = "chem_rate"

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.fate_cols = [int(c) for c in (params.get("fate_cols") or [])]
        self.fate_rates = [float(r) for r in (params.get("fate_rates") or [])]
        if len(self.fate_cols) != len(self.fate_rates):
            raise ValueError("cell_die[chem_rate]: fate_cols and fate_rates must have the same length")
        self.fate_threshold = float(params.get("fate_threshold", 0.5))
        self.t_start = float(params.get("t_start", 0.0))
        self.call_every = max(int(params.get("every", 1)), 1)
        self._rng = np.random.default_rng(int(params.get("seed", 0)) + 7919)

    def _marked(self, m, H, nF):
        if not self.fate_cols or not any(r > 0 for r in self.fate_rates):
            return set()
        if float(getattr(H, "frame", 0) or 0) < self.t_start:
            return set()
        clvl = H.level(self.cat)
        if clvl is None or "chem" not in getattr(clvl, "state_schema", {}):
            return set()
        h0, _ = clvl.state_schema["chem"]
        G = clvl.state[:nF, [h0 + c for c in self.fate_cols]].detach().cpu().numpy()
        fate = np.argmax(G, axis=1)
        p = np.asarray(self.fate_rates, float)[fate] * self.call_every
        p[G.max(axis=1) < self.fate_threshold] = 0.0
        return set(np.where(self._rng.random(nF) < p)[0].tolist())


@register_operator("cell_die", model="prescribed", set="vertex", kind="die",
                   family="population", title="ablate a chosen patch", equation=r"""$$\text{mark } j\ \text{when}\ j\in\mathcal R$$""")
class Apoptosis3DPrescribed(Apoptosis3D):
    r"""`prescribed` model of cell_die: the experimenter ablates a patch; a control, and the way to ask what a wound does.
    vertex -> vertex: as cell_die; this model only marks. Only what marks a cell for death differs between the `model:` variants of cell_die; the extrusion is the base's.

        \text{mark } j\ \text{when}\ j\in\mathcal R
        \mathcal R     `region`, a list of cells, a set of latitude bands, or one cap

    Reference: Plexus (this work); laser ablation as an experimental protocol."""
    PARAM_UNITS = {}
    SYMBOLS = {'\\mathcal R': '`region`, a list of cells, a set of latitude bands, or one cap'}
    DEATH = "prescribed"


@register_operator("cell_die", model="field", set="vertex", kind="die", family="population", title="remove cells by a per-cell signal", equation=r"""$$\text{mark } j\ \text{when}\ q_j\gtrless f\,\operatorname{median}_k q_k$$""")
class Apoptosis3DField(Apoptosis3D):
    r"""`field` model of cell_die: death keyed to a per-cell quantity some other operator published.
    vertex -> vertex: as cell_die; this model only marks. Only what marks a cell for death differs between the `model:` variants of cell_die; the extrusion is the base's.

        \text{mark } j\ \text{when}\ q_j\gtrless f\,\operatorname{median}_k q_k
        q_j            the field named by `field` on the mesh
        f              `field_frac`, a multiple of the live-cell median
        \gtrless       `compare: high | low`, which tail qualifies

    Reference: Plexus (this work); the generic form of the measured criteria."""
    PARAM_UNITS = {}
    SYMBOLS = {'q_j': 'the field named by `field` on the mesh', 'f': '`field_frac`, a multiple of the live-cell median', '\\gtrless': '`compare: high | low`, which tail qualifies'}
    DEATH = "field"


@register_operator("seed_mesh", "mesh_seed", set="vertex", kind="seed", family="seed",
                   implementation="isotropic", title="build the cell mesh, no preferred axis", equation=r"""$$\text{row pitch}=\tfrac{\sqrt3}{2}\,a,\qquad \theta\sim\mathcal U(0,2\pi)$$""")
class SeedMeshIsotropic(SeedMesh3D):
    r"""`isotropic` implementation of seed_mesh: the default mesh with a true hexagonal disc lattice, turned by an angle drawn from `seed`.
    vertex -> vertex: as the default seed_mesh; the disc builder is swapped for `build_disc_mesh_isotropic` for the length of the call.

        \text{row pitch}=\tfrac{\sqrt3}{2}\,a,\qquad \theta\sim\mathcal U(0,2\pi)
        a              the lattice spacing
        \theta         the rotation of the lattice

    Reference: none -- a lattice choice. Plexus exp08 (this work)."""
    PARAM_UNITS = {}
    SYMBOLS = {'a': 'the lattice spacing', '\\theta': 'the rotation of the lattice'}

    def forward(self, H, mask=None):
        import plexus.operators.vertex_ops as _V
        orig = _V.build_disc_mesh
        _V.build_disc_mesh = build_disc_mesh_isotropic
        try:
            return super().forward(H, mask)
        finally:
            _V.build_disc_mesh = orig


def mutant_patches(m, pos, nF, pool, k, size, rng, centre=None):
    """k cells of `pool` as patches of `size` adjacent cells: ceil(k / size) patch centres drawn from the
    pool, each grown by its nearest pool cells (face centroids), the last patch cut to make k in total.

    `centre` (exp 11, default None = the draw above, unchanged): a direction [x, y, z]; then ONE patch of the k
    pool cells nearest the pool cell whose centroid direction from the tissue's centroid lies closest to it -- a
    declared site, as a declared hole has an axis."""
    es = np.asarray(m["E_srce"].detach().cpu() if hasattr(m["E_srce"], "detach") else m["E_srce"], np.int64)
    ef = np.asarray(m["E_face"].detach().cpu() if hasattr(m["E_face"], "detach") else m["E_face"], np.int64)
    P = np.asarray(pos.detach().cpu() if hasattr(pos, "detach") else pos, np.float64)
    live = ef < nF
    cen = np.zeros((nF, 3))
    np.add.at(cen, ef[live], P[es[live]])
    cen /= np.maximum(np.bincount(ef[live], minlength=nF), 1)[:, None]
    pool = np.asarray(pool, np.int64)
    if centre is not None:
        d = np.asarray(centre, np.float64)
        d = d / max(np.linalg.norm(d), 1e-12)
        u = cen[pool] - cen[pool].mean(0)
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        c = pool[int(np.argmax(u @ d))]
        return pool[np.argsort(np.linalg.norm(cen[pool] - cen[c], axis=1), kind="stable")][:k]
    free = np.ones(nF, bool)
    chosen = []
    for c in rng.choice(pool, size=int(np.ceil(k / size)), replace=False):
        if not free[c]:
            continue
        cand = pool[free[pool]]
        near = cand[np.argsort(np.linalg.norm(cen[cand] - cen[c], axis=1), kind="stable")][:size]
        free[near] = False
        chosen.extend(near.tolist())
    return np.asarray(chosen[:k], np.int64)


@register_operator("seed_mesh", "mesh_seed", implementation="lineage", set="vertex", kind="seed",
                   family="seed", title="build the cell mesh with lineage labels", equation=r"""$$\mathrm{clone}_j=j,\qquad \mathrm{mutant}_j\sim\mathrm{Bernoulli}(f_m)$$""")
class SeedMeshLineage(SeedMesh3D):
    r"""the same mesh, with every seeded cell labelled by its lineage: neutral clones.
    vertex -> vertex: as the chosen `geometry` variant of seed_mesh, plus the cell blocks `clone` (the founder's index, inherited unchanged) and `mutant`.

        \mathrm{clone}_j=j,\qquad \mathrm{mutant}_j\sim\mathrm{Bernoulli}(f_m)
        clone_j        the founder's index, copied to every descendant
        f_m            `mutant_frac`, the fraction of seeded cells marked mutant

    Reference: Clayton, E. et al. (2007). Nature 446:185-189 (neutral clones). Plexus exp14 (this work)."""
    PARAM_UNITS = {'mutant_frac': 'fraction', 'progenitor_frac': 'fraction', 'mutant_cluster': 'count', 'mutant_centre': 'length'}
    SYMBOLS = {'clone_j': "the founder's index, copied to every descendant", 'f_m': '`mutant_frac`, the fraction of seeded cells marked mutant'}
    PARAM_ROLES = {'geometry': 'which_seed_mesh_variant_builds_the_mesh', 'mutant_centre': 'centre_about_which_the_mutants_are_clustered', 'mutant_cluster': 'number_of_mutant_clusters', 'mutant_frac': 'fraction_of_cells_marked_mutant', 'mutant_in': 'which_cells_may_be_mutant', 'progenitor_frac': 'fraction_of_cells_marked_progenitor'}
    MECHANISM_TAGS = ["initial_condition", "lineage_label"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.mutant_frac = float(params.get("mutant_frac", 0.0))
        self.progenitor_frac = float(params.get("progenitor_frac", 1.0))
        # `mutant_in: progenitors` (exp 14 Phase 3): the mutant is induced in single PROGENITORS, as Colom
        # 2020 induces DN-Maml1 (Fig. 6c); `mutant_frac` is then a fraction of the progenitors. Default: any cell.
        self.mutant_in = str(params.get("mutant_in", "any"))
        # `mutant_cluster` (exp 12 Phase 2): the mutants are placed as patches of this many adjacent cells
        # -- a leader cluster, Cheung 2013's multicellular leading front -- instead of singly. Default 1:
        # the single-cell draw below, unchanged.
        self.mutant_cluster = int(params.get("mutant_cluster", 1))
        # `mutant_centre` (exp 11 Phase 2, default None = unchanged): with `mutant_cluster` > 1, ONE patch of all the
        # mutants around the cell nearest this direction from the tissue centroid -- a declared site for a hole the
        # mutant cells digest (`bm_unbond[model: release]`), as a declared hole has an axis.
        _mc = params.get("mutant_centre", None)
        self.mutant_centre = None if _mc is None else [float(v) for v in _mc]
        if self.mutant_cluster < 1:
            raise ValueError(f"seed_mesh[lineage]: mutant_cluster must be >= 1, got {self.mutant_cluster}")
        if self.mutant_in not in ("any", "progenitors"):
            raise ValueError(f"seed_mesh[lineage]: mutant_in must be any|progenitors, got {self.mutant_in!r}")
        geo = params.get("geometry")
        self._geometry = None
        if geo:
            from plexus.models.registry import get_operator
            self._geometry = get_operator("seed_mesh", variant=str(geo))(dict(params), device)

    def forward(self, H, mask=None):
        if self._geometry is None:
            out = super().forward(H, mask)
        else:
            for k in ("_cat", "_at"):                    # what the engine set on this instance
                if hasattr(self, k):
                    setattr(self._geometry, k, getattr(self, k))
            out = self._geometry.forward(H, mask)
            self.cell_set = getattr(self._geometry, "cell_set", None) or \
                resolve_cell_set(H, self.at, getattr(self, "_cat", None))
        m = getattr(H.level(self.at), "_mesh", None)
        if m is None:
            return out
        nF = int(m["nF"])
        # optional blocks, see the class docstring: a set without them gets the default mesh
        if cell_block(H, self.cell_set, "clone", nF) is not None:
            set_cell_block(H, self.cell_set, "clone", np.arange(nF, dtype=np.float64), nF)
        fate = None
        if cell_block(H, self.cell_set, "fate", nF) is not None:
            fate = np.ones(nF)
            k = int(round(self.progenitor_frac * nF))
            fate[np.random.default_rng(self.seed + 223).choice(nF, size=k, replace=False)] = 0.0
            set_cell_block(H, self.cell_set, "fate", fate, nF)
        if cell_block(H, self.cell_set, "mutant", nF) is not None:
            mut = np.zeros(nF)
            pool = np.flatnonzero(fate < 0.5) if (self.mutant_in == "progenitors" and fate is not None) \
                else np.arange(nF)
            k = int(round(self.mutant_frac * len(pool)))
            if k > 0 and self.mutant_cluster == 1:
                mut[np.random.default_rng(self.seed + 211).choice(pool, size=k, replace=False)] = 1.0
            elif k > 0:
                mut[mutant_patches(m, H.level(self.at).get("pos"), nF, pool, k, self.mutant_cluster,
                                   np.random.default_rng(self.seed + 211), centre=self.mutant_centre)] = 1.0
            set_cell_block(H, self.cell_set, "mutant", mut, nF)
        return out


def adhesion_hazard(B, b_ref, r_mit):
    """The Type I dive probability of cells ready to divide, from their grip (exp 11 Phase 3 B5, 2026-09-29):
    h_i = 1 - r_mit min(1, B_i / b_ref). A cell rounding up for mitosis lets go of the matrix (Wang 2021, Fig S2A:
    delamination visible ~45 min before anaphase; Hsu 2013: outer cells leave the membrane while dividing);
    r_mit is the share of its grip a mitotic cell keeps, and a cell that keeps it divides at the boundary instead
    (Type II). B_i the bound integrin (`bm_clutch`'s `itg_B`), b_ref its median at the first use."""
    c = np.clip(np.asarray(B, np.float64) / max(float(b_ref), 1e-12), 0.0, 1.0)
    return 1.0 - float(r_mit) * c


def adhesion_return_p(p0, lam, G, g_ref, kappa, u=1.0):
    """A returning body's per-call return probability (exp 11 Phase 3 B5, 2026-09-29), Wang 2021's
    dE = (e_cc - e_cm) S_B read as a rate: p = p0 lam u exp(-kappa (G - g_ref)), clipped to [0, 1]. The
    cell-matrix gain is the matrix at the host site times the body's own receptor: lam is `bm_clutch`'s covered
    fraction times the membrane's mass there (1 at rest; collagenase lowers it), u the body's integrin over the
    resting total, capped at 1 (a beta1 block lowers it). G is the body's E-cadherin (`cell_protein_level`), g_ref
    the surface lineage's level: a body inside long enough to raise its E-cadherin binds the core instead."""
    p = (float(p0) * np.asarray(lam, np.float64) * np.clip(np.asarray(u, np.float64), 0.0, 1.0)
         * np.exp(-float(kappa) * (np.asarray(G, np.float64) - float(g_ref))))
    return np.clip(p, 0.0, 1.0)


def newborn_pairs(cid, pid, seen, p, rng):
    """`cell_die[t2]` `rule: newborn`: the face indices of ONE daughter of each in-place division born since the last
    call -- two cell ids not in `seen` that share a parent_id (a reinsert newcomer is a lone new id and is left out) --
    each pair taken with probability p. Returns (chosen face indices, the ids now present). The first call (seen None)
    takes nothing: every cell present then is the seed."""
    ids = np.rint(np.asarray(cid, np.float64)).astype(np.int64); par = np.rint(np.asarray(pid, np.float64)).astype(np.int64)
    now = set(ids.tolist())
    if seen is None:
        return set(), now
    by = {}
    for f in range(len(ids)):
        if int(ids[f]) not in seen:
            by.setdefault(int(par[f]), []).append(f)
    out = set()
    for _p, fs in sorted(by.items()):
        if len(fs) == 2 and rng.random() < p:
            out.add(int(fs[int(rng.integers(2))]))
    return out, now


@register_operator("cell_die", implementation="t2", set="vertex", kind="die", family="population",
                   title="remove cells, collapsing them in one step", equation=r"""$$n_{\mathrm{mark}}=\max\big(0,\ N_{\mathrm{alive}}-N_{\mathrm{cap}}\big),\qquad \text{ring}_j:\ n\to3\ \text{by T1s}\ \to\ \text{T2}$$""")
class Apoptosis3DT2(Apoptosis3D):
    r"""`t2` implementation of cell_die: the same extrusion, carried out as a T2 the operator completes itself, for a tissue that turns over for many generations.
    vertex -> vertex: marks by `rule:` (any model of cell_die), flips each ready cell's own edges shortest first until it is a triangle, collapses it in the same call, and compacts the vertex set.

        n_{\mathrm{mark}}=\max\big(0,\ N_{\mathrm{alive}}-N_{\mathrm{cap}}\big),\qquad \text{ring}_j:\ n\to3\ \text{by T1s}\ \to\ \text{T2}
        n_{mark}       cells sentenced this call, the excess over the capacity, worst first
        N_{alive}      live cells not already under sentence
        N_{cap}        `capacity`, the cells the layer holds (0 = no ceiling)
        T1, T2         the edge flip and the face collapse of the vertex model

    Reference: Eisenhoffer, G. T. et al. (2012). Nature 484:546-549; the T1/T2 transitions: Farhadifar, R. et al. (2007). Curr. Biol. 17:2095-2104."""
    PARAM_UNITS = {'capacity': 'count', 'max_t1': 'count', 'newborn_p': 'fraction', 'type2_p': 'fraction', 'keep_nb_sides': 'count', 'wound_deg': 'angle', 'hazard': 'rate', 'r_mit': 'fraction', 'shed': 'fraction', 'sink': 'velocity', 'depth0': 'length', 'max_depth': 'length', 'enter_offset': 'length', 'enter_split': 'fraction', 'max_leave_per_call': 'count', 'max_leave_frac': 'fraction', 'ready_progress': 'fraction', 'contract': 'fraction', 'stuck_frac': 'fraction'}
    SYMBOLS = {'n_{mark}': 'cells sentenced this call, the excess over the capacity, worst first', 'N_{alive}': 'live cells not already under sentence', 'N_{cap}': '`capacity`, the cells the layer holds (0 = no ceiling)', 'T1, T2': 'the edge flip and the face collapse of the vertex model'}
    PARAM_ROLES = {'rule': 'which_cell_die_model_marks', 'region': 'list_band_or_cone_for_rule_prescribed', 'compare': 'which_tail_for_rule_field', 'capacity': 'cells_the_layer_holds_zero_means_none', 'max_t1': 'edge_flips_per_cell_per_call', 'newborn_p': 'probability_a_newborn_is_marked_for_rule_newborn', 'type2_p': 'probability_of_a_type_ii_exit_for_rule_phase', 'type2_mark': 'block_set_on_a_type_ii_exit', 'keep_nb_sides': 'neighbour_sides_a_collapse_must_keep', 'wound_frame': 'frame_of_the_wound', 'wound_deg': 'half_angle_of_the_wound', 'select': 'global_or_local_selection_of_the_excess', 'plane_axis': 'normal_axis_of_a_flat_sheet', 'hazard': 'constant_leaving_rate_per_frame', 'hazard_from': 'fixed_or_from_a_block', 'r_mit': 'fraction_of_mitotic_cells_that_leave', 'bound_block': 'bound_integrin_block_read_by_the_hazard', 'order': 'order_the_ready_cells_are_collapsed_in', 'resist_block': 'block_that_resists_marking', 'progenitor_fallback': 'mark_progenitors_when_no_other_cell_qualifies', 'to_set': 'particle_set_a_leaving_cell_becomes_a_member_of', 'shed': 'fraction_of_the_volume_shed_on_leaving', 'sink': 'speed_a_left_cell_sinks_at', 'depth0': 'depth_a_left_cell_starts_at', 'max_depth': 'depth_a_left_cell_stops_at', 'to_set_kinematic': 'move_the_left_cell_kinematically', 'enter_offset': 'distance_a_cell_enters_at', 'enter_divided': 'enter_as_two_daughters', 'hold_when_full': 'hold_the_leaving_when_the_target_set_is_full', 'enter_split': 'fraction_of_the_volume_given_to_the_first_daughter', 'enter_mark': 'block_set_on_an_entering_cell', 'to_children': 'child_set_of_the_target_set', 'child_volume': 'volume_per_child_point', 'max_leave_per_call': 'ceiling_on_the_cells_leaving_per_call', 'max_leave_frac': 'ceiling_on_the_fraction_leaving_per_call', 'ready_progress': 'cycle_progress_at_which_a_cell_is_ready', 'collapse_measured': 'collapse_on_the_measured_size', 'contract': 'fraction_the_ring_contracts_before_collapse', 'stuck_frac': 'fraction_of_the_target_below_which_a_stuck_cell_is_collapsed', 'isolate': 'isolate_the_dying_cell_before_the_collapse'}
    MECHANISM_TAGS = ["death", "extrusion", "t2_transition", "homeostasis", "capacity"]
    CARRY_ONCE = True               # see Apoptosis3D.CARRY_ONCE: each per-cell array permuted once

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        rule = str(params.get("rule", "crowded"))
        if rule == "prescribed":
            rule = str(params.get("region", "list"))
            if rule not in ("list", "band", "cone"):
                raise ValueError(f"cell_die[t2]: region must be list|band|cone, got {rule!r}")
        elif rule == "field":
            rule = "field_" + str(params.get("compare", "high"))
        self.mode = rule
        self.capacity = int(params.get("capacity", 0) or 0)
        self.max_t1 = int(params.get("max_t1", 12))
        self.newborn_p = self.tunable(params.get("newborn_p"), 1.0, tensor=True)  # `rule: newborn` only (see `_marked`)
        self.type2_p = float(params.get("type2_p", 0.0) or 0.0)    # `rule: phase` + Type II (see `_marked`)
        self.type2_mark = params.get("type2_mark")
        self._t2_ids = set()
        # `keep_nb_sides: k` (exp 11, 2026-09-28; default 0 = unchanged): a forced flip on the dying cell's edge must
        # leave the cell ACROSS that edge with at least k sides, or that edge is skipped this call. Each flip takes a
        # side from the cell across the edge; shortest-first, on a layer that only loses cells (exp 11's Type I dive),
        # it made triangles and squares -- 29 triangles in 370 cells at row 80 against none on the dividing control --
        # and those are strong positive curvature defects: the outermost 2 % of the tent-shaped glands were almost all
        # triangles (Finding 114). A dying cell with no edge to spare waits for the layer's own flips.
        self.keep_nb_sides = int(params.get("keep_nb_sides", 0) or 0)
        # WHERE THE EXCESS IS TAKEN. `global`: the rule's worst anywhere in the tissue. `division`:
        # among the neighbours of the cells born this call first (falling back to anywhere when none
        # qualifies) -- a loss coupled to a neighbour's division, which is what defines the
        # cell-extrinsic class of Klein & Simons 2011 (Box 2) and Colom 2020's lattice (Fig. 5a).
        # Measured with `global` (exp 14 batch 2): clone sizes nearer the well-mixed law than the
        # voter model's, the loss being decoupled from the neighbour's division.
        # A WOUND, ONCE, ON TOP OF THE RULE (exp 14's wound arm): at the first call at or after frame
        # `wound_frame` every cell whose centroid lies within `wound_deg` of the +z axis (seen from
        # the origin, the shell's centre) is sentenced, past the capacity and the rate cap -- the
        # experimenter's ablation, `model: prescribed`'s `cone` on the same line as the turnover rule.
        # A second `cell_die` line would do it too, but specs address operators by name.
        self.wound_frame = params.get("wound_frame")
        self.wound_deg = float(params.get("wound_deg", 30.0))
        self._wounded = False
        # `pair`: each DIVISION pays for itself -- its excess is taken among the neighbours of its own
        # two daughters (sisters share `parent_id`), the most crowded first, ties at random. `division`
        # pools the neighbours of every cell born in the call and takes the most crowded of the pool,
        # so a mutant dividing inside its own patch can have its excess paid by a wild-type cell beside
        # some other newborn; the voter model the gates compare with pairs each birth with a loss
        # beside IT (exp 14 round 10: selection 2.5x the voter's with `division`).
        self.select = str(params.get("select", "global"))
        if self.select not in ("global", "division", "pair"):
            raise ValueError(f"cell_die[t2]: select must be global|division|pair, got {self.select!r}")
        pa = params.get("plane_axis")
        self.plane_axis = None if pa is None else int(pa)
        # STRATIFICATION (exp 14 Phase 2). `rule: fate` makes the candidates the committed cells --
        # `fate` block 1, written by `cell_divide model: fate` -- so only B cells leave the basal
        # layer, as in Clayton et al. 2007 (Nature 446:185, Fig. 4b: post-mitotic basal cells "transfer
        # at a steady rate to the suprabasal compartment"). Who among them leaves is either a HAZARD
        # -- each candidate sentenced with probability `hazard` per call, Clayton's Gamma, which leaves
        # the basal count free to drift -- or the capacity with `select: pair`, each division pushing
        # out a committed neighbour of its own daughters: Colom et al. 2020's model, in which
        # "progenitor cell division is linked to a neighboring cell differentiating and exiting from
        # the basal layer" (Nat. Genet. 52:604, Fig. 5a).
        self.hazard = float(params.get("hazard", 0.0))
        # `hazard_from: bound` (default `fixed`, the scalar above; exp 11 Phase 3 B5, 2026-09-29): each candidate's
        # hazard is its own, `adhesion_hazard` of its bound integrin (`bound_block`, default `itg_B`) with the
        # mitotic grip `r_mit` (default 0.08) -- the Type I / Type II split follows the cell's grip, not a number. The hazard
        # branch still needs `hazard` > 0 to run (it selects the per-call draw); its value is then not used
        self.hazard_from = str(params.get("hazard_from", "fixed")).lower()
        if self.hazard_from not in ("fixed", "bound"):
            raise ValueError(f"cell_die[t2]: hazard_from is fixed or bound, got {self.hazard_from!r}")
        self.r_mit = float(params.get("r_mit", 0.08))
        self.bound_block = str(params.get("bound_block", "itg_B"))
        self._b_ref = None
        # WHICH COMMITTED CELL GOES, when the capacity takes the excess and more candidates qualify than
        # may go: `random` (the default), or `crowded` -- most neighbours first, the base's own measure of
        # crowding (`rule: crowded`'s), so that with `select: global` a committed cell delaminates where the
        # layer is most packed, anywhere in the tissue, rather than beside the division that overfilled it
        # (exp 14 round 17: the pair coupling grew clones at 0.31 basal cells a week against Clayton's 0.40).
        self.order = str(params.get("order", "random"))
        if self.order not in ("random", "crowded"):
            raise ValueError(f"cell_die[t2]: order must be random|crowded, got {self.order!r}")
        # RESISTANCE TO DIFFERENTIATION (exp 14 Phase 3): with `order: crowded`, a committed cell whose
        # `resist_block` is 1 goes only after every other candidate -- the mutant's advantage of Colom et al.
        # 2020's neighbour-constrained fitness model, where cells "with a lower likelihood of differentiation
        # will, on average, win and persist" at a clone's edge (Fig. 5a); it acts only where a mutant and a
        # wild-type cell compete for the same exit, and is neutral inside a mutant patch.
        self.resist_block = params.get("resist_block")
        # PROGENITORS MAY LEAVE TOO (exp 14 Phase 3). With `progenitor_fallback`, `rule: fate` names every
        # basal cell old enough and the order puts the committed ones first: a progenitor differentiates and
        # leaves only when too few committed cells are left to pay a division's excess. Without it a
        # fate-biased mutant patch, which makes progenitors and few committed cells, cannot be thinned: two
        # runs grew 1,100 -> 3,971-3,998 basal cells (3,690-3,780 progenitors) and filled the cell buffer
        # (round 19). Colom 2020's model has no committed pool at all -- "progenitor cell division is linked
        # to a neighboring cell differentiating" (Fig. 5a).
        self.progenitor_fallback = bool(params.get("progenitor_fallback", False))
        if self.mode == "fate" and not (self.capacity or self.hazard > 0):
            raise ValueError("cell_die[t2] rule: fate needs `hazard` or `capacity` -- with neither, "
                             "every committed cell would be sentenced at once")
        # THE CELL IS NOT LOST, IT MOVES. With `to_set: <particle set>` every cell this operator
        # removes reappears in that set, carrying every width-1 label the two sets both declare
        # (`clone`, `mutant`, `cell_id`, ...): the suprabasal compartment. There it sinks inward from
        # the shell along the direction it left by (`depth0` + `sink` x its age in calls, a fraction
        # of the shell's mean radius, at most `max_depth` -- the layers of a stratified epithelium,
        # the lumen side of an organoid) and is shed with probability `shed` per call. No mechanics
        # act on it: its position is kinematic, and nothing in the basal layer reads it.
        self.to_set = params.get("to_set")
        self.shed = float(params.get("shed", 0.0))
        self.sink = float(params.get("sink", 0.0))
        self.depth0 = float(params.get("depth0", 0.05))
        self.max_depth = float(params.get("max_depth", 0.8))
        self._sb_calls = 0
        # DIVING IN, NOT SINKING (exp 11; Wang et al. 2021 Cell 184:3702, Fig 2A-B: 92.4 % of the
        # division-ready surface cells of a salivary bud delaminate and divide in the interior).
        # `to_set_kinematic: false` (default true = the suprabasal routine above, unchanged): a leaver is
        # placed ONCE in a free slot of `to_set`, `enter_offset` (world length, default 0.5 -- the inner
        # cell's radius, `bm_contact[live, side: apical]`'s `offset`) inside the apical surface along the
        # inward normal -- its centroid + the mean `sep` of its ring (the apical point, half a layer
        # thickness in) + enter_offset x sep/|sep| -- with zero `vel`, and this operator never moves,
        # ages or sheds a live particle of `to_set` again: the interior's own mechanics moves them.
        # Without a `sep` block the inward normal is toward the tissue's centroid, from the centroid.
        self.to_set_kinematic = bool(params.get("to_set_kinematic", True))
        self.enter_offset = float(params.get("enter_offset", 0.5))
        self.enter_divided = bool(params.get("enter_divided", False))
        # `hold_when_full` (default false; exp 11 Finding 165, 2026-09-29): a cell is sentenced to leave only while
        # `to_set` has a free slot for it (two with `enter_divided`), counting the leavers already sentenced. Off, a
        # leaver with no slot is removed from the layer and only counted (`supra_overflow`): exp 11's interior filled
        # its 1,500 slots at row 220 of a 300-row run and the layer drained 1,310 -> 192 cells into nothing.
        self.hold_when_full = bool(params.get("hold_when_full", False))
        self.enter_split = float(params.get("enter_split", 0.5))
        self.enter_mark = params.get("enter_mark")
        # `to_children: <point set>` (default none): `to_set` is a set of MPM CELLS, each owning a block of
        # material points in that set -- a leaver becomes a whole body in the lumen, its block woken as a
        # uniform ball of `child_volume` (world^3; default the median volume of the live bodies) about the
        # entry point (`mpm_ops.wake_mpm_body`). exp 11, 2026-09-27: the surface cell that dives in to divide.
        self.to_children = params.get("to_children")
        self.child_volume = params.get("child_volume")
        self.max_leave_per_call = int(params.get("max_leave_per_call", 0) or 0)
        self.max_leave_frac = float(params.get("max_leave_frac", 0.0) or 0.0)
        # `rule: phase`: the candidates are the DIVISION-READY cells -- `cycle_progress` at or past
        # `ready_progress` (default 1.0, the end of the cycle = M on `cell_cycle`'s clock; `phase` >= M
        # where no `cycle_progress` is declared), each seen ONCE (by `cell_id`) as it becomes ready, so
        # `hazard` is the probability that a ready cell leaves INSTEAD of dividing in place (Wang's
        # 0.924), not a per-call rate. A sentenced cell stays in M while it shrinks out of the layer, so
        # the spec's `cell_divide[model: cycle]` must carry `skip_sentenced: true` or it cuts the leaver
        # (and both pieces inherit the sentence). Idle while the seed settles, as division is.
        # `enter_divided: true` (default false): the leaver divides AS it enters -- two particles,
        # `enter_split` (world, default 0.5) apart across the inward normal -- Wang's "delaminate and
        # divide in the interior", so a leave followed by the return of both daughters conserves the
        # division. `enter_mark: <block of to_set>` (default none): that width-1 block is set to 1 on
        # every particle this operator places (and `cell_divide[model: point]` copies it to their
        # daughters), which is what `cell_divide[model: reinsert]`'s `mark` reads -- only the progeny
        # of the cells that dived return, not the interior the rig seeded.
        self.ready_progress = float(params.get("ready_progress", 1.0))
        # `collapse_measured: true` (default false = unchanged): the T2 also waits for the cell's MEASURED
        # volume (`cell_size`, the run's convention) to fall below `critical_frac` x the measured reference,
        # not only its target. On the apico-basal shell of exp 11 the target shrinks at `shrink_rate` a call
        # while the polyhedron follows over many frames, so the target test collapsed cells still at full
        # size into a point, and the yanked neighbours crushed the layer (a 200-cell shell at 92 % leaving:
        # median cell volume 1.07 -> 0.03 in 30 frames, 22 inverted cells).
        self.collapse_measured = bool(params.get("collapse_measured", False))
        self._v_meas = None
        # `contract` (fraction per call, default 0 = none): every cell already under sentence pulls its
        # ring vertices that fraction of the way to its centroid at the start of each call -- the
        # actomyosin ring that closes over an extruding cell (Rosenblatt et al. 2001). The apico-basal
        # layer's positional bending (`k_bend`) holds a cell's footprint against a shrinking target
        # alone: measured on exp 11's rig, sentenced cells' targets fell 10x in 8 calls while their
        # polyhedron volumes stayed at 1.5 (reference 1.0).
        self.contract = float(params.get("contract", 0.0))
        # `stuck_frac` (default 0 = off): a sentenced cell whose shrinking target has fallen below stuck_frac x the
        # collapse size protects its neighbours down to `keep_nb_sides` - 1 sides (never under 4), so a dive the side
        # rule has held for many calls can finish; the others keep the full rule (all at 4 emptied the layer, R7).
        self.stuck_frac = float(params.get("stuck_frac", 0.0) or 0.0)
        # `isolate` (default false): a queued cell is sentenced only if no neighbour is under sentence or picked in the
        # same call; the others keep their turn in the queue. Adjacent dying cells stall each other's collapse: in exp 11
        # R9, 89-90 % of the sentenced cells touched another one, and they piled up as flat giants (Finding 193).
        # Wang's diving cells are scattered through the surface.
        self.isolate = bool(params.get("isolate", False))
        self._seen_ready = set()
        if self.mode == "phase" and not (self.capacity or self.hazard > 0):
            raise ValueError("cell_die[t2] rule: phase needs `hazard` (the fraction of division-ready "
                             "cells that leave) or `capacity`")

    def _severity(self, m, H, nF, idx):
        """`rule: fate` with `order: crowded`: most neighbours first; otherwise the base's order."""
        if self.mode == "fate" and self.progenitor_fallback:
            f = cell_block(H, self.cat, "fate", nF) if H is not None else None
            base = (-self._nb(m, nF, np.ones(nF))[1][idx].astype(float) if self.order == "crowded"
                    else np.zeros(len(idx)))
            if f is not None:
                base = base + 1e3 * (f[idx] < 0.5)            # committed cells first, progenitors after
            if self.resist_block:
                r = cell_block(H, self.cat, self.resist_block, nF) if H is not None else None
                if r is not None:
                    base = base + 1e6 * (r[idx] > 0.5)
            return base
        if self.mode == "fate" and self.order == "crowded":
            sev = -self._nb(m, nF, np.ones(nF))[1][idx].astype(float)
            if self.resist_block:
                r = cell_block(H, self.cat, self.resist_block, nF) if H is not None else None
                if r is not None:
                    sev = sev + 1e6 * (r[idx] > 0.5)          # after every non-resistant candidate
            return sev
        return super()._severity(m, H, nF, idx)

    def _marked(self, m, H, nF):
        """`rule: fate`: the committed (B) cells old enough to go; any other rule, the base's."""
        if self.mode == "phase":
            if settling(H, m):
                return set()
            out = self._newly_ready(H, nF)
            # `type2_p` (exp 11, 2026-09-28; default 0 = off): Wang's Type II division beside the Type I dive, in the SAME
            # operator (two `cell_die[t2]` would share `apop_flag` and shrink and remove each other's cells): of each
            # in-place division born since the last call, with probability type2_p one daughter leaves too, and wakes
            # with `type2_mark` (e.g. `ret`: it returns without dividing inside) instead of `enter_mark`.
            if self.type2_p > 0:
                cid = cell_block(H, self.cat, "cell_id", nF); pid = cell_block(H, self.cat, "parent_id", nF)
                if cid is not None and pid is not None:
                    rng = np.random.default_rng(self.seed * 6007 + 29 + int(getattr(self, "_k", 0)))
                    t2, self._nb_seen = newborn_pairs(cid, pid, getattr(self, "_nb_seen", None), self.type2_p, rng)
                    ids = np.rint(np.asarray(cid, np.float64)).astype(np.int64)
                    self._t2_ids.update(int(ids[f]) for f in t2)
                    m["n_type2"] = int(m.get("n_type2", 0)) + len(t2)
                    self._t2_now = set(t2)                # sentenced OUTSIDE the Type I cap, see `_admit_rule`
            return out
        if self.mode == "newborn":
            # `rule: newborn` (exp 11, 2026-09-28): Wang's Type II division -- of each in-place division born since the
            # last call (a PAIR of new cell ids sharing a parent_id; a reinsert newcomer is a single new id), with
            # probability `newborn_p` ONE daughter, at random, leaves the layer (with `enter_mark: ret` it goes in as a
            # body that returns without dividing). The rest of the T2 machinery is unchanged.
            if settling(H, m):
                return set()
            cid = cell_block(H, self.cat, "cell_id", nF); pid = cell_block(H, self.cat, "parent_id", nF)
            if cid is None or pid is None:
                return set()
            rng = np.random.default_rng(self.seed * 7919 + 17 + int(getattr(self, "_k", 0)))
            out, self._nb_seen = newborn_pairs(cid, pid, getattr(self, "_nb_seen", None),
                                               float(getattr(self, "newborn_p", 1.0)), rng)
            m["n_type2"] = int(m.get("n_type2", 0)) + len(out)
            return out
        if self.mode != "fate":
            return super()._marked(m, H, nF)
        fate = cell_block(H, self.cat, "fate", nF)
        if fate is None:
            return set()
        ag = m.get("age")
        old_enough = (ag.detach().cpu().numpy()[:nF] >= self.min_age) if ag is not None \
            else np.ones(nF, bool)
        who = old_enough if self.progenitor_fallback else ((fate > 0.5) & old_enough)
        return set(np.flatnonzero(who).tolist())

    def forward(self, H, mask=None):
        if self.contract > 0:
            self._contract(H)
        if self.collapse_measured:
            self._stash_measured(H)
        if not self.to_set:
            return super().forward(H, mask)
        cat = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
        m0 = getattr(H.level(self.at), "_mesh", None)
        nF0 = int(m0["nF"]) if m0 is not None else 0
        before = self._leaving_labels(H, cat, m0, nF0)
        out = super().forward(H, mask)
        m = getattr(H.level(self.at), "_mesh", None)
        nF = int(m["nF"]) if m is not None else 0
        after = cell_block(H, cat, "cell_id", nF)
        gone = []
        if before is not None and after is not None:
            left = set(before["cell_id"].tolist()) - set(after.tolist())
            gone = [i for i, c in enumerate(before["cell_id"].tolist()) if c in left]
        if self.to_set_kinematic:
            self._suprabasal(H, m, before, gone)
        else:
            self._enter(H, m, before, gone)
        return out

    def _contract(self, H):
        """`contract`: the sentenced cells' ring vertices moved `contract` of the way to its centroid."""
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        fl = m.get("apop_flag") if m is not None else None
        if fl is None:
            return
        nF, nv = int(m["nF"]), int(m["Nv"])
        fl = (fl.detach().cpu().numpy() if torch.is_tensor(fl) else np.asarray(fl))[:nF]
        dying = np.flatnonzero(fl > 0)
        if not len(dying):
            return
        ef = m["E_face"].detach().cpu().numpy()
        es = m["E_srce"].detach().cpu().numpy()
        p0, p1 = lvl.state_schema["pos"]
        x = lvl.state[:nv, p0:p1].detach().cpu().numpy().astype(np.float64)
        for f in dying.tolist():
            r = es[ef == f]
            if len(r):
                c = x[r].mean(0)
                x[r] = c + (1.0 - self.contract) * (x[r] - c)
        st = lvl.state.clone()
        st[:nv, p0:p1] = torch.as_tensor(x, dtype=st.dtype, device=st.device)
        lvl.state = st

    def _stash_measured(self, H):
        """`collapse_measured`: every cell's measured volume and reference at the start of the call."""
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        if m is None or not int(m.get("Nv", 0) or 0):
            self._v_meas = None
            return
        v, vr = cell_size(lvl, m, int(m["nF"]), H=H, at=self.at)
        self._v_meas, self._v_meas_ref = np.asarray(v, np.float64), float(vr)

    def _newly_ready(self, H, nF):
        """`rule: phase`: the division-ready cells not yet seen ready (by `cell_id`; every ready cell
        each call when no `cell_id` is declared). Idle while the seed settles, as division is."""
        p = cell_block(H, self.cat, "cycle_progress", nF)
        if p is not None:
            ready = p >= self.ready_progress - 1e-4
        else:
            ph = cell_block(H, self.cat, "phase", nF)
            if ph is None:
                return set()
            ready = ph >= CellCycle3D.M - 1e-3
        cid = cell_block(H, self.cat, "cell_id", nF)
        if cid is None:
            return set(np.flatnonzero(ready).tolist())
        ids = np.rint(cid).astype(np.int64)
        self._seen_ready &= set(ids.tolist())
        new = [int(f) for f in np.flatnonzero(ready) if int(ids[f]) not in self._seen_ready]
        self._seen_ready.update(int(ids[f]) for f in new)
        return set(new)

    def _entry_points(self, H, m, nF, cen):
        """Per cell, where a leaver enters `to_set` (`to_set_kinematic: false`) and the inward unit normal:
        the apical point (centroid + mean ring `sep`) moved `enter_offset` further along sep/|sep|."""
        vl = H.level(self.at)
        nv = int(m["Nv"])
        es = m["E_srce"].detach().cpu().numpy().astype(np.int64)
        ef = m["E_face"].detach().cpu().numpy().astype(np.int64)
        ok = (ef >= 0) & (ef < nF)
        cen = np.asarray(cen, np.float64)[:nF]
        if "sep" in vl.state_schema:
            S = vl.get("sep")[:nv].detach().cpu().numpy().astype(np.float64)
            sm = np.zeros((nF, 3)); cnt = np.zeros(nF)
            np.add.at(sm, ef[ok], S[es[ok]]); np.add.at(cnt, ef[ok], 1.0)
            sm /= np.maximum(cnt, 1.0)[:, None]
            n = sm / np.maximum(np.linalg.norm(sm, axis=1, keepdims=True), 1e-12)
            return cen + sm + self.enter_offset * n, n
        X = vl.get("pos")[:nv].detach().cpu().numpy().astype(np.float64)
        u = X.mean(0) - cen
        n = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        return cen + self.enter_offset * n, n

    def _enter(self, H, m, before, gone):
        """`to_set_kinematic: false`: this call's leavers woken in free slots of `to_set` at their entry
        points, `vel` zero, labels carried; nothing else in the set is touched."""
        if not gone or before is None or "_enter" not in before:
            return
        P = H.level(self.to_set)
        sch = P.state_schema
        if "pos" not in sch:
            raise ValueError(f"cell_die[t2]: to_set {self.to_set!r} must declare `pos`")
        dev, dt = P.state.device, P.state.dtype
        st, oc = P.state.clone(), P.occ.clone()
        free = torch.nonzero(oc < 0.5).flatten()
        per = 2 if self.enter_divided else 1
        k = min(len(gone), int(free.numel()) // per)
        if k < len(gone) and m is not None:
            m["supra_overflow"] = int(m.get("supra_overflow", 0)) + len(gone) - k
        if k == 0:
            return
        idx = np.asarray(gone[:k], np.int64)
        x = np.asarray(before["_enter"], np.float64)[idx]
        if per == 2:                                      # the two daughters, across the inward normal
            n = np.asarray(before["_inward"], np.float64)[idx]
            ax = np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
            t = np.cross(n, ax)
            t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
            x = np.concatenate([x - 0.5 * self.enter_split * t, x + 0.5 * self.enter_split * t])
            idx = np.concatenate([idx, idx])
        slot = free[:len(idx)]
        p0, p1 = sch["pos"]
        st[slot, p0:p1] = torch.as_tensor(x, device=dev, dtype=dt)
        if "vel" in sch:
            v0, v1 = sch["vel"]
            st[slot, v0:v1] = 0.0
        if "dir" in sch:                                  # outward, as the kinematic routine's
            d0, d1 = sch["dir"]
            st[slot, d0:d1] = torch.as_tensor(-before["_inward"][idx], device=dev, dtype=dt)
        if "age" in sch:
            st[slot, sch["age"][0]] = 0.0
        for name, vals in before.items():
            if name in sch and vals is not None and name != "centroid" and not name.startswith("_"):
                st[slot, sch[name][0]] = torch.as_tensor(np.asarray(vals)[idx], device=dev, dtype=dt)
        if self.enter_mark:
            if self.enter_mark not in sch:
                raise ValueError(f"cell_die[t2]: enter_mark {self.enter_mark!r} is not a block of {self.to_set!r}")
            st[slot, sch[self.enter_mark][0]] = 1.0
        if self.type2_p > 0 and self.type2_mark and self._t2_ids and before.get("cell_id") is not None:
            if self.type2_mark not in sch:
                raise ValueError(f"cell_die[t2]: type2_mark {self.type2_mark!r} is not a block of {self.to_set!r}")
            _g = np.rint(np.asarray(before["cell_id"], np.float64)[idx]).astype(np.int64)
            _is2 = torch.as_tensor(np.array([int(g) in self._t2_ids for g in _g]), device=slot.device)
            if bool(_is2.any()):
                if self.enter_mark:
                    st[slot[_is2], sch[self.enter_mark][0]] = 0.0
                st[slot[_is2], sch[self.type2_mark][0]] = 1.0
                self._t2_ids.difference_update(int(g) for g in _g[_is2.cpu().numpy()])
        oc[slot] = 1.0
        P.state, P.occ = st, oc
        if self.to_children:
            from plexus.operators.mpm_ops import mpm_blocks, wake_mpm_body
            if self.child_volume is not None:
                vol = float(self.child_volume)
            else:
                Pc = H.level(self.to_children)
                blocks, _per = mpm_blocks(H, self.to_set, self.to_children)
                lv = (oc > 0.5).nonzero(as_tuple=True)[0]
                bv = Pc.p_vol[blocks[lv]].sum(1) if getattr(Pc, "p_vol", None) is not None else None
                vol = float(bv.median()) if bv is not None and bv.numel() else 1.0
            if not hasattr(self, "_wake_gen"):
                self._wake_gen = torch.Generator(device="cpu").manual_seed(self.seed + 104723)
            wake_mpm_body(H, self.to_set, self.to_children, slot.tolist(), x, vol, gen=self._wake_gen)
        if m is not None:
            m["n_entered"] = int(m.get("n_entered", 0)) + k

    def _leaving_labels(self, H, cat, m, nF):
        """Every cell's id, centroid and shared labels at the start of the call -- what a cell that
        leaves in this call takes with it. None without a `cell_id` block (nothing to follow)."""
        cid = cell_block(H, cat, "cell_id", nF)
        if cid is None or m is None:
            return None
        clvl = H.level(cat)
        if "centroid" in clvl.state_schema:
            c0, c1 = clvl.state_schema["centroid"]
            cen = clvl.state[:nF, c0:c1].detach().cpu().numpy().astype(np.float64)
        else:
            cen = np.asarray(m.get("centroid_np"), np.float64)[:nF]
        P = H.level(self.to_set)
        if P is None:
            raise ValueError(f"cell_die[t2]: to_set {self.to_set!r} is not a set of this spec")
        names = [k for k, (a, b) in P.state_schema.items()
                 if b - a == 1 and k not in ("age",) and k in clvl.state_schema
                 and clvl.state_schema[k][1] - clvl.state_schema[k][0] == 1]
        out = {"cell_id": cid, "centroid": cen, **{k: cell_block(H, cat, k, nF) for k in names}}
        if not self.to_set_kinematic:
            out["_enter"], out["_inward"] = self._entry_points(H, m, nF, cen)
        return out

    def _suprabasal(self, H, m, before, gone):
        """Age, shed and sink the suprabasal set; put this call's leavers in its free slots."""
        P = H.level(self.to_set)
        sch = P.state_schema
        for need in ("pos", "dir", "age"):
            if need not in sch:
                raise ValueError(f"cell_die[t2]: to_set {self.to_set!r} must declare `{need}` "
                                 f"(pos 3, dir 3, age 1)")
        rng = np.random.default_rng(self.seed * 1_000_003 + 104729 + self._sb_calls)
        self._sb_calls += 1
        dev, dt = P.state.device, P.state.dtype
        st = P.state.clone()
        oc = P.occ.clone()
        a0, _a1 = sch["age"]; d0, d1 = sch["dir"]; p0, p1 = sch["pos"]
        live = oc > 0.5
        st[live, a0] += 1.0
        if self.shed > 0 and bool(live.any()):
            u = torch.as_tensor(rng.random(int(oc.numel())), device=dev)
            oc[live & (u < self.shed)] = 0.0
        vl = H.level(self.at)
        Nv = int(m["Nv"]) if m is not None else 0
        if Nv <= 0:
            P.state, P.occ = st, oc
            return
        v0, v1 = vl.state_schema["pos"]
        X = vl.state[:Nv, v0:v1].detach().to(dev, dt)
        c = X.mean(0)
        R = (X - c).norm(dim=1).mean()
        if gone:
            free = torch.nonzero(oc < 0.5).flatten()
            k = min(len(gone), int(free.numel()))
            if k < len(gone) and m is not None:
                m["supra_overflow"] = int(m.get("supra_overflow", 0)) + len(gone) - k
            idx = np.asarray(gone[:k], np.int64)
            slot = free[:k]
            cen = torch.as_tensor(before["centroid"][idx], device=dev, dtype=dt)
            u = cen - c
            st[slot, d0:d1] = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
            st[slot, a0] = 0.0
            for name, vals in before.items():
                if name in sch and vals is not None and name != "centroid":
                    b0, _b1 = sch[name]
                    st[slot, b0] = torch.as_tensor(np.asarray(vals)[idx], device=dev, dtype=dt)
            oc[slot] = 1.0
        live = oc > 0.5
        depth = (self.depth0 + self.sink * st[live, a0]).clamp(max=self.max_depth)
        st[live, p0:p1] = c + st[live, d0:d1] * (R * (1.0 - depth))[:, None]
        P.state, P.occ = st, oc

    def _admit(self, flag, want, m, H, nF):
        """The base's admission (a rate cap, worst first), with the capacity and random ties; and
        the one-off wound."""
        flag = self._admit_rule(flag, want, m, H, nF)
        if (self.wound_frame is not None and not self._wounded
                and int(getattr(H, "frame", 0) or 0) >= int(self.wound_frame)):
            self._wounded = True
            c = m.get("centroid_np")
            if c is not None and len(c) >= nF:
                u = np.asarray(c[:nF], float)
                u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
                hit = np.degrees(np.arccos(np.clip(u[:, 2], -1, 1))) < self.wound_deg
                flag[np.flatnonzero(hit)] = 1.0
                m["wound_cells"] = int(hit.sum())
        return flag

    def _room(self, H, flag):
        """With `hold_when_full`: how many more cells `to_set` can take this call -- its free slots, less the leavers
        already sentenced (`flag` > 0), per leaver 2 slots with `enter_divided`. None when the guard is off."""
        if not self.hold_when_full or not getattr(self, "to_set", None) or H is None:
            return None
        P = H.level(self.to_set)
        if P is None:
            return None
        free = int((P.occ < 0.5).sum())
        pending = int((np.asarray(flag) > 0).sum())
        per = 2 if self.enter_divided else 1
        return (free - per * pending) // per

    def _admit_rule(self, flag, want, m, H, nF):
        if self.hazard > 0:                        # each candidate at `hazard` a call, no ceiling
            rng = np.random.default_rng(self.seed * 1_000_003 + 15485863 + self._k)
            held = {int(x) for x in np.where(flag > 0)[0]}
            cand = np.array(sorted(set(want) - held), dtype=np.int64)
            hz = self.hazard
            if self.hazard_from == "bound" and len(cand):
                Bc = cell_block(H, self.cat, self.bound_block, nF) if H is not None else None
                if Bc is None:
                    raise ValueError(f"cell_die[t2]: hazard_from bound needs the width-1 block {self.bound_block!r} "
                                     f"on {self.cat!r} (written by bm_clutch)")
                if self._b_ref is None and (Bc > 0).any():
                    self._b_ref = float(np.median(Bc[Bc > 0]))
                hz = adhesion_hazard(Bc[cand], self._b_ref or 1.0, self.r_mit)
                m["hazard_mean"] = float(np.mean(hz))
            pick = cand[rng.random(len(cand)) < hz]
            # `max_leave_per_call: N` (default 0 = no ceiling, the line above unchanged): at most N new
            # sentences a call, a random N of the drawn -- the others stay candidates (a phase-ready cell
            # stays in M) and are drawn again next call. exp 11: with `hazard` 1 and no division in the
            # layer, every cell that became ready during the settling window left at once -- 111 of 200 in
            # 12 frames, the layer crushed (Finding 94); `max_mark_frac` does not reach this branch.
            _cap = self.max_leave_per_call
            if self.max_leave_frac > 0:
                # `max_leave_frac` (default 0 = off): the cap FOLLOWS THE LAYER, max(max_leave_per_call,
                # ceil(frac x nF)) a call. A fixed cap of 2 met the start of exp 11's Type I loop and throttled
                # its steady state: ~400 cells on a ~60-frame cycle are ~7 ready a frame, so the queue grew, the
                # waiting cells kept growing in M and became giant flat facets (the tent-shaped gland of mc_x_cm)
                _cap = max(_cap, int(math.ceil(self.max_leave_frac * nF)))
            if _cap > 0:
                # A QUEUE, NOT A DRAW: `rule: phase` offers each ready cell ONCE (`_ready`, by `cell_id`), so a
                # cell the cap passed over would never be offered again -- it sat in M, swelled (median volume
                # 1.15 -> 3.9) and the layer stopped growing (34 dives in 320 frames). The passed-over cells wait,
                # first come first served, while they stay live and unsentenced.
                ids = cell_block(H, self.cat, "cell_id", nF) if H is not None else None
                if ids is not None:
                    at = {int(ids[f]): int(f) for f in range(nF)}
                    queued = [at[c] for c in getattr(self, "_leave_q", []) if c in at and flag[at[c]] <= 0]
                    pool = list(dict.fromkeys(queued + [int(f) for f in pick]))
                    if self.isolate and m.get("E_face") is not None:
                        _sel, _rest = isolated_split(pool, flag, m["E_srce"], m["E_trgt"], m["E_face"], nF, _cap)
                    else:
                        _sel, _rest = pool[: _cap], pool[_cap:]
                    pick = np.asarray(_sel, dtype=np.int64)
                    self._leave_q = [int(ids[f]) for f in _rest]
                    m["t2_waiting"] = set(self._leave_q)          # read by `cell_divide` `skip_sentenced`
                elif len(pick) > _cap:
                    pick = rng.choice(pick, size=_cap, replace=False)
            room = self._room(H, flag)
            if room is not None:
                pick = pick[: max(room, 0)]
            flag[pick] = 1.0
            # Type II leavers (`type2_p`) are not queued behind the Type I cap: a boundary division sends its daughter
            # in at once (with the cap they took the Type I slots and most never left -- exp 11 Finding 134)
            _t2n = getattr(self, "_t2_now", None)
            if _t2n:
                _i = np.asarray(sorted(f for f in _t2n if f < len(flag)), dtype=np.int64)
                _r = self._room(H, flag)
                if _r is not None:
                    _i = _i[: max(_r, 0)]
                if len(_i):
                    flag[_i] = 1.0
                self._t2_now = set()
            return flag
        if self.mode in ("list", "band", "cone") or not self.capacity:
            return super()._admit(flag, want, m, H, nF)
        held = np.where(flag > 0)[0]
        room = min(max(1, int(self.max_mark_frac * nF)) - len(held), nF - len(held) - self.capacity)
        if room <= 0:
            return flag
        # TIES AT RANDOM. The worst-first sort keeps index order among equals, and with `crowded` most
        # candidates tie at seven neighbours; the lowest index would always die, and index order is
        # the seed's spatial order -- a drift in who survives that no rule declared.
        rng = np.random.default_rng(self.seed * 1_000_003 + self._k)
        cand = want - {int(x) for x in held}
        if self.select == "pair":
            picked = self._pair_victims(m, H, nF, cand, room, rng)
            flag[np.asarray(sorted(picked), dtype=np.int64)] = 1.0
            room -= len(picked)
            cand = cand - picked
            if room <= 0:
                return flag
        if self.select == "division":
            near = cand & self._beside_births(m, nF)
            cand = near or cand
        fresh = rng.permutation(np.array(sorted(cand), dtype=np.int64))
        if len(fresh) > room:
            sev = self._severity(m, H, nF, fresh)
            fresh = fresh[np.argsort(sev, kind="stable")[:room]] if sev is not None else fresh[:room]
        flag[fresh] = 1.0
        return flag

    def _pair_victims(self, m, H, nF, cand, room, rng):
        """One cell per division born this call, among the candidates beside ITS two daughters, the
        most crowded first (ties at random); divisions visited in random order, at most `room`."""
        ag = m.get("age")
        if ag is None:
            return set()
        age = ag.detach().cpu().numpy()[:nF] if hasattr(ag, "detach") else np.asarray(ag)[:nF]
        young = np.flatnonzero(age < 1)
        if not len(young):
            return set()
        par = cell_block(H, self.cat, "parent_id", nF) if H is not None else None
        groups = {}
        for f in young.tolist():                      # sisters share their mother's id
            key = int(par[f]) if par is not None and par[f] >= 0 else -1 - f
            groups.setdefault(key, []).append(f)
        import torch as _t
        es = _t.as_tensor(m["E_srce"], dtype=_t.long); et = _t.as_tensor(m["E_trgt"], dtype=_t.long)
        ef = _t.as_tensor(m["E_face"], dtype=_t.long)
        twin = ShapeEnergy3D._twin_faces(es, et, ef, int(m["Nv"])).cpu().numpy()
        efn = ef.cpu().numpy()
        deg = self._nb(m, nF, np.ones(nF))[1]
        picked = set()
        for key in rng.permutation(np.array(sorted(groups), dtype=np.int64)).tolist():
            if len(picked) >= room:
                break
            sis = set(groups[key])
            nb = {int(twin[k]) for k in np.flatnonzero(np.isin(efn, list(sis)))
                  if 0 <= twin[k] < nF} - sis
            opts = [g for g in nb if g in cand and g not in picked]
            if not opts:
                continue
            opts = rng.permutation(np.asarray(opts, dtype=np.int64))
            picked.add(int(opts[np.argsort(-deg[opts], kind="stable")[0]]))   # most crowded; ties random
        return picked

    def _beside_births(self, m, nF):
        """The cells sharing an edge with a cell born this call (age < 1), the newborns excluded."""
        ag = m.get("age")
        if ag is None:
            return set()
        age = ag.detach().cpu().numpy()[:nF] if hasattr(ag, "detach") else np.asarray(ag)[:nF]
        young = age < 1
        if not young.any():
            return set()
        import torch as _t
        es = _t.as_tensor(m["E_srce"], dtype=_t.long); et = _t.as_tensor(m["E_trgt"], dtype=_t.long)
        ef = _t.as_tensor(m["E_face"], dtype=_t.long)
        twin = ShapeEnergy3D._twin_faces(es, et, ef, int(m["Nv"])).cpu().numpy()
        efn = ef.cpu().numpy()
        ok = (efn < nF) & (twin >= 0) & (twin < nF)
        nb = twin[ok][young[efn[ok]]]
        return {int(g) for g in nb if not young[g]}

    def _to_triangle(self, rings, pos, f, keep_nb_sides=None):
        """Flip the dying cell's own edges, shortest first, until it has three sides. Each T1 on an
        edge of f takes one side from f and one from the cell across the edge, and gives one to each
        of the two cells at its ends (`t1_flip_3d`, which refuses any flip that would break the
        closed surface or fold a face). True when f is a triangle. With `collapse_measured`, False (not
        yet) while the cell's measured volume is above `critical_frac` x the measured reference."""
        if (self.collapse_measured and self._v_meas is not None and f < len(self._v_meas)
                and self._v_meas[f] > self.crit * self._v_meas_ref):
            return False
        if len(rings[f]) == 3:
            return True
        emap, vf = _edge_face_map(rings), _vertex_faces(rings)
        for _ in range(self.max_t1):
            r = [int(v) for v in rings[f]]
            if len(r) <= 3:
                break
            es = [(r[i], r[(i + 1) % len(r)]) for i in range(len(r))]
            es.sort(key=lambda e: float(np.linalg.norm(np.asarray(pos[e[0]]) - np.asarray(pos[e[1]]))))
            _kn = self.keep_nb_sides if keep_nb_sides is None else keep_nb_sides
            if _kn:
                # the cell across edge (a, b) of f holds the reversed directed edge (b, a)
                es = [e for e in es if emap.get((e[1], e[0])) is not None
                      and len(rings[emap[(e[1], e[0])]]) - 1 >= _kn]
            if not any(t1_flip_3d(rings, pos, e, emap=emap, vf=vf, plane_axis=self.plane_axis) is not None
                       for e in es):
                break
        return len(rings[f]) == 3

    def nb_sides_for(self, size_tgt, crit):
        """The side floor `_to_triangle` protects this sentenced cell's neighbours with: `keep_nb_sides`, or one less
        (never under 4) once the cell's shrinking target has fallen below `stuck_frac` x the collapse size -- a dive
        stuck that long (exp 11 R5: sentenced cells 10 -> 245 over rows 100-240, the stuck ones 9-sided, Finding 190)."""
        kn = self.keep_nb_sides
        if kn and self.stuck_frac > 0 and size_tgt < self.stuck_frac * crit:
            return max(int(kn) - 1, 4)
        return kn

    def _after_topology(self, H, m, es, et):
        """Compact the vertex set to the vertices the new half-edge table still references.

        WHY. `face_collapse_3d` merges a dying cell's last three vertices into one and leaves the
        other two allocated: nothing references them, but they stay inside the live prefix `Nv`.
        A division appends two vertices, so a tissue that turns over -- one extrusion per division
        -- grows `Nv` by two per cycle of every cell and never gives any back. Measured on exp 14's
        first batch: 402 cells on 4,000 live vertices where a closed shell needs 2F - 4 = 800, the
        reservoir full after ~1,500 divisions, and every division refused for the remaining 19,000
        frames -- a tissue that froze at its capacity (exp 14, Finding 6).

        HOW. The kept vertices, in their old order, go through `Hierarchy.renumber_set` -- the
        engine's one permutation of a set's state, occupancy and pending deltas -- so `pos`, `vel`
        and any other per-vertex state block (`sep` on an apico-basal shell) move together; the
        declared per-vertex mesh columns (`vertex_carry`) are permuted here; the half-edge table is
        remapped. Returns the remapped (E_srce, E_trgt)."""
        es, et = np.asarray(es, np.int64), np.asarray(et, np.int64)
        nv = int(m["Nv"])
        used = np.unique(np.concatenate([es, et]))
        if len(used) == nv or not len(used):
            return es, et
        if not H.renumber_set(self.at, used, n_new=len(used)):
            print("[cell_die t2] renumber_set on the vertex set DID NOT ACT -- "
                  "vertices left in place", flush=True)
            return es, et
        remap = np.full(max(nv, int(used.max()) + 1), -1, np.int64)
        remap[used] = np.arange(len(used))
        for nm in sorted(m.get("vertex_carry") or ()):
            a = m.get(nm)
            if a is not None and hasattr(a, "shape") and a.shape[0] >= nv:
                idx = torch.as_tensor(used, dtype=torch.long, device=a.device) if torch.is_tensor(a) else used
                m[nm] = a[idx]
        m["Nv"] = int(len(used))
        m["n_vertex_reclaimed"] = int(m.get("n_vertex_reclaimed", 0)) + (nv - len(used))
        return remap[es], remap[et]



@register_operator("cell_divide", model="doubler", set="vertex", kind="divide", family="population", title="divide at double birth volume", equation=r"""$$\text{divide } j\ \text{when}\ v_j\ge\text{factor}\cdot v^{\mathrm{birth}}_j$$""")
class Divide3DDoubler(Divide3D):
    r"""`doubler` model of cell_divide: divide at `factor` times this cell's own birth volume, the null the absolute-size rule has to beat.
    vertex -> vertex: as cell_divide; only the trigger differs.

        \text{divide } j\ \text{when}\ v_j\ge\text{factor}\cdot v^{\mathrm{birth}}_j
        v_j            the cell's wedge volume
        v^{birth}_j    its own volume at birth
        factor         `factor`, dimensionless

    Reference: Ginzberg, M. B., Kafri, R. & Kirschner, M. W. (2015). Science 348:1245075."""
    PARAM_UNITS = {}
    SYMBOLS = {'v_j': "the cell's wedge volume", 'v^{birth}_j': 'its own volume at birth', 'factor': '`factor`, dimensionless'}
    MECHANISM_TAGS = ["division", "volume_doubling", "relative_threshold", "no_size_control"]

    def _trigger(self, v_now, v_birth, jit, age, v_ref):
        return v_now >= self.factor * jit * v_birth


@register_operator("cell_divide", model="postmitotic", set="vertex", kind="divide", family="population",
                   title="divide cells except post-mitotic ones", equation=r"""$$\text{divide } j\ \text{when}\ v_j\ge\text{factor}\cdot v_{\mathrm{ref}}\ \text{and}\ b_j[c]\le\theta$$""")
class Divide3DPostmitotic(Divide3D):
    r"""`postmitotic` model of cell_divide: the default size rule, except that a post-mitotic cell never divides.
    vertex -> vertex: as cell_divide; a cell whose `veto_block[:, veto_col]` is above `veto_th` is vetoed. A `max_cycle` below 1e9 is refused.

        \text{divide } j\ \text{when}\ v_j\ge\text{factor}\cdot v_{\mathrm{ref}}\ \text{and}\ b_j[c]\le\theta
        v_j, v_{ref}   the cell's volume and the seed-time median
        b_j[c]         the cell's `veto_block` value at column `veto_col`
        \theta         `veto_th`, in the block's own units

    Reference: Serra, D. et al. (2019). Nature 569:66-72 (the Delta-high founder becomes the post-mitotic Paneth cell)."""
    PARAM_UNITS = {'veto_th': 'dimensionless', 'veto_below_th': 'dimensionless', 'veto_col': 'count', 'veto_below_col': 'count'}
    SYMBOLS = {'v_j, v_{ref}': "the cell's volume and the seed-time median", 'b_j[c]': "the cell's `veto_block` value at column `veto_col`", '\\theta': "`veto_th`, in the block's own units"}
    PARAM_ROLES = {'veto_block': 'block_read_for_the_veto', 'veto_col': 'its_column', 'veto_th': 'level_above_which_a_cell_is_post_mitotic', 'veto_below_col': 'second_column_that_must_be_below_its_threshold', 'veto_below_th': 'that_threshold'}
    MECHANISM_TAGS = ["division", "cell_division", "post_mitotic", "paneth", "lateral_inhibition"]
    REFERENCE = "Serra, D. et al. (2019). Nature 569:66-72."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.veto_block = str(params.get("veto_block", "chem"))
        self.veto_col = int(params.get("veto_col", 1))            # Delta of cell_chem_react[notch_delta]
        self.veto_th = float(params.get("veto_th", 0.5))  # in the block's own units
        bc, bt = params.get("veto_below_col"), params.get("veto_below_th")
        if (bc is None) != (bt is None):
            raise ValueError("cell_divide[postmitotic]: veto_below_col and veto_below_th go together")
        self.veto_below_col = None if bc is None else int(bc)
        self.veto_below_th = None if bt is None else float(bt)
        if self.max_cycle < 10 ** 9:
            raise ValueError(
                f"cell_divide[postmitotic]: max_cycle {self.max_cycle} (calls since birth) would force a "
                f"vetoed cell to divide, since Divide3D ORs the backstop in after the mask. Leave it unset.")
        self._veto_warned = False

    def _veto(self, H, nF):
        """Per-cell boolean, True = post-mitotic; None when the block or a column is not there."""
        cat = self.cell_set
        lvl = H.level(cat) if cat in H.levels else None
        need = max(self.veto_col, -1 if self.veto_below_col is None else self.veto_below_col)
        if lvl is None or getattr(lvl, "state", None) is None or self.veto_block not in lvl.state_schema \
                or lvl.get(self.veto_block).shape[1] <= need:
            if not self._veto_warned:
                print(f"[cell_divide:postmitotic] set {cat!r} has no {self.veto_block!r} block with column "
                      f"{need}: nothing is vetoed, this is the default size rule")
                self._veto_warned = True
            return None
        blk = lvl.get(self.veto_block)[:nF].detach().cpu().numpy().astype(np.float64)
        v = blk[:, self.veto_col] > self.veto_th
        if self.veto_below_col is not None:
            v &= blk[:, self.veto_below_col] < self.veto_below_th
        return v

    def _ready_mask(self, H, m, nF):
        """`_trigger` per cell with the post-mitotic cells False; None (the default path) when none is."""
        veto = self._veto(H, nF)
        if veto is None or not veto.any():
            return None
        # THE SAME FOUR SCALARS `forward` BUILDS, read the same way: `cell_size` has already run this
        # call (so its cached reference is fixed), `Vbirth` has already been re-read for age-1 cells,
        # and the age is the stored one plus this call.
        lvl = H.level(self.at)
        Nv = int(m["Nv"])
        pos_np = lvl.get("pos")[:Nv].detach().cpu().numpy().astype(np.float64)
        vf, v_ref = cell_size(lvl, m, nF, pos_np, H=H, at=self.at)
        vf = np.asarray(vf, np.float64)
        Vbirth = require_cell_block(H, self.cell_set, "Vbirth", nF, "cell_divide").tolist()
        djit = require_cell_block(H, self.cell_set, "divjit", nF, "cell_divide").tolist()
        a = m.get("age")
        age = ([0] * nF) if (a is None or a.shape[0] != nF) else (a.detach().cpu().numpy() + 1).tolist()
        ready = np.array([bool(self._trigger(vf[f], Vbirth[f], djit[f], age[f], v_ref)) for f in range(nF)],
                         dtype=bool)
        m["div_vetoed"] = int((ready & veto).sum())      # ripe cells held back this call, for the record
        return ready & ~veto


@register_operator("cell_divide", model="timer", set="vertex", kind="divide", family="population", title="divide cells on a clock", equation=r"""$$\text{divide } j\ \text{when}\ \mathrm{age}_j\ge T\,\mathrm{jit}_j$$""")
class Divide3DTimer(Divide3D):
    r"""`timer` model of cell_divide: divide on the clock, size ignored.
    vertex -> vertex: as cell_divide; only the trigger differs.

        \text{divide } j\ \text{when}\ \mathrm{age}_j\ge T\,\mathrm{jit}_j
        age_j          division calls since birth
        T              `cycle`, in division calls (as `min_cycle`), not frames
        jit_j          the per-cell jitter drawn from `cycle_cv`

    Reference: Ginzberg, M. B., Kafri, R. & Kirschner, M. W. (2015). Science 348:1245075."""
    PARAM_UNITS = {'cycle': 'count'}
    SYMBOLS = {'age_j': 'division calls since birth', 'T': '`cycle`, in division calls (as `min_cycle`), not frames', 'jit_j': 'the per-cell jitter drawn from `cycle_cv`'}
    PARAM_ROLES = {'cycle': 'division_calls_between_divisions'}
    MECHANISM_TAGS = ["division", "timer", "cell_cycle_clock", "size_independent"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.cycle = float(params.get("cycle", 8.0))

    def _trigger(self, v_now, v_birth, jit, age, v_ref):
        return age >= self.cycle * jit


@register_operator("cell_divide", model="timer_fitness", set="vertex", kind="divide", family="population",
                   title="divide on a clock, mutants faster", equation=r"""$$\text{divide } j\ \text{when}\ \mathrm{age}_j\ge\frac{T\,\mathrm{jit}_j}{1+\alpha\,\mathrm{mutant}_j}$$""")
class Divide3DTimerFitness(Divide3DTimer):
    r"""`timer_fitness` model of cell_divide: the timer, with a heritable proliferative advantage for `mutant` cells.
    vertex -> vertex: as cell_divide[timer]; the `mutant` block shortens the cycle.

        \text{divide } j\ \text{when}\ \mathrm{age}_j\ge\frac{T\,\mathrm{jit}_j}{1+\alpha\,\mathrm{mutant}_j}
        T              `cycle`, in division calls
        \alpha         `advantage`, the extra divisions per wild-type division, dimensionless
        mutant_j       the cell's `mutant` block, 0 or 1

    Reference: Morata, G. & Ripoll, P. (1975). Dev. Biol. 42:211-221 (Minute competition)."""
    PARAM_UNITS = {'advantage': 'dimensionless'}
    SYMBOLS = {'T': '`cycle`, in division calls', '\\alpha': '`advantage`, the extra divisions per wild-type division, dimensionless', 'mutant_j': "the cell's `mutant` block, 0 or 1"}
    PARAM_ROLES = {'advantage': 'proliferative_advantage_of_a_mutant'}
    MECHANISM_TAGS = ["division", "timer", "cell_cycle_clock", "fitness", "mutant"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        # A HERITABLE PROLIFERATIVE ADVANTAGE (exp 14, mutant takeover). A cell whose declared
        # `mutant` block is 1 divides at age >= cycle * jit / (1 + advantage): it cycles
        # (1 + advantage) times as often, so `advantage` is the extra divisions per wild-type cycle,
        # the per-cycle fitness difference of Watson et al. 2020 (Science 367:1449, Fig. 1C: a fit
        # clone grows at rate s relative to the normal cells). 0, the default, or no `mutant` block
        # leaves the rule exactly `age >= cycle * jit`.
        self.advantage = float(params.get("advantage", 0.0))

    def _ready_mask(self, H, m, nF):
        """The per-cell form of `_trigger` when `advantage` applies; None (use `_trigger`) otherwise.

        It needs the cell's `mutant` block, which `_trigger`'s four scalars do not carry. The age is
        the one `forward` compares -- the stored age plus this call -- and the jitter the same
        `divjit` block."""
        if self.advantage == 0.0:
            return None
        mut = cell_block(H, self.cell_set, "mutant", nF)
        if mut is None:
            return None
        a = m.get("age")
        age = np.zeros(nF) if (a is None or a.shape[0] != nF) else a.detach().cpu().numpy()[:nF] + 1
        jit = require_cell_block(H, self.cell_set, "divjit", nF, "cell_divide")
        return age >= self.cycle * jit / (1.0 + self.advantage * (mut > 0.5))


@register_operator("cell_divide", model="fate", set="vertex", kind="divide", family="population",
                   title="divide with progenitor fate choice", equation=r"""$$P(AA)=r+\beta,\quad P(BB)=r-\beta,\quad P(AB)=1-2r$$""")
class Divide3DFate(Divide3DTimerFitness):
    r"""`fate` model of cell_divide: `timer_fitness` with the single-progenitor fates of Clayton et al. 2007.
    vertex -> vertex: as cell_divide[timer_fitness]; only progenitors (A) divide, and each division draws the pair of daughter fates.

        P(AA)=r+\beta,\quad P(BB)=r-\beta,\quad P(AB)=1-2r
        r              `r_sym`, the probability of a symmetric division
        \beta          `fate_bias`, the shift toward AA
        A, B           progenitor and committed daughter

    Reference: Clayton, E. et al. (2007). Nature 446:185-189 (Fig. 3b; r = 0.08)."""
    PARAM_UNITS = {'r_sym': 'fraction', 'fate_bias': 'fraction', 'seed': 'count'}
    SYMBOLS = {'r': '`r_sym`, the probability of a symmetric division', '\\beta': '`fate_bias`, the shift toward AA', 'A, B': 'progenitor and committed daughter'}
    PARAM_ROLES = {'r_sym': 'probability_of_a_symmetric_division', 'fate_bias': 'shift_of_the_symmetric_divisions_toward_two_progenitors', 'seed': 'random_seed', 'waiting': 'timer_or_exponential_waiting'}
    MECHANISM_TAGS = ["division", "timer", "cell_fate", "asymmetric_division", "progenitor"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        # r, the symmetric fraction: P(AA) = P(BB) = r_sym, P(AB) = 1 - 2 r_sym (Clayton 2007: 0.08)
        self.r_sym = float(params.get("r_sym", 0.08))
        if not 0.0 <= self.r_sym <= 0.5:
            raise ValueError(f"cell_divide[fate]: r_sym must be in [0, 0.5], got {self.r_sym}")
        self.fate_seed = int(params.get("seed", 0))
        self._fate_calls = 0
        # A FATE-BIASED MUTANT (exp 14 Phase 3): a division whose daughters carry `mutant` 1 gives AA with
        # probability r_sym + fate_bias and BB with r_sym - fate_bias -- Colom 2020's "imbalance in division
        # outcome, so the average mutant cell division produces more progenitor than differentiating cells",
        # an advantage "even if the rate of cell division is unchanged" (Nat. Genet. 52:604, Results).
        self.fate_bias = float(params.get("fate_bias", 0.0))
        if not (0.0 <= self.r_sym - abs(self.fate_bias) and self.r_sym + abs(self.fate_bias) <= 0.5):
            raise ValueError("cell_divide[fate]: r_sym +- fate_bias must stay in [0, 0.5]")
        self.waiting = str(params.get("waiting", "timer"))
        if self.waiting not in ("timer", "exponential"):
            raise ValueError(f"cell_divide[fate]: waiting must be timer|exponential, got {self.waiting!r}")
        self._wait_calls = 0

    def _ready_mask(self, H, m, nF):
        """The timer's rule (with the advantage, if any) or a constant rate, and only for A cells."""
        fate = cell_block(H, self.cell_set, "fate", nF)
        if fate is not None and self.waiting == "exponential":
            rng = np.random.default_rng(self.fate_seed * 1_000_003 + 31337 + self._wait_calls)
            self._wait_calls += 1
            mut = cell_block(H, self.cell_set, "mutant", nF)
            gain = 1.0 + (self.advantage * (mut > 0.5) if mut is not None else 0.0)
            p = gain / max(self.cycle - self.min_cycle, 1.0)
            return (rng.random(nF) < p) & (fate < 0.5)          # `min_cycle` is applied by `Divide3D`
        base = super()._ready_mask(H, m, nF)
        if fate is None:
            return base
        if base is None:                         # the timer's own rule, `age >= cycle * jit`
            a = m.get("age")
            age = np.zeros(nF) if (a is None or a.shape[0] != nF) else a.detach().cpu().numpy()[:nF] + 1
            jit = require_cell_block(H, self.cell_set, "divjit", nF, "cell_divide")
            base = age >= self.cycle * jit
        return np.asarray(base, bool) & (fate < 0.5)

    def forward(self, H, mask=None):
        cat = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
        m0 = getattr(H.level(self.at), "_mesh", None)
        nF0 = int(m0["nF"]) if m0 is not None else 0
        cid0 = cell_block(H, cat, "cell_id", nF0)
        top = float(cid0.max(initial=-1.0)) if cid0 is not None else None
        out = super().forward(H, mask)
        m = getattr(H.level(self.at), "_mesh", None)
        if m is None or top is None:
            return out
        nF = int(m["nF"])
        fate = cell_block(H, self.cell_set, "fate", nF)
        cid = cell_block(H, self.cell_set, "cell_id", nF)
        par = cell_block(H, self.cell_set, "parent_id", nF)
        if fate is None or cid is None or par is None:
            return out
        born = np.flatnonzero(cid > top)          # both daughters carry fresh ids (`Divide3D`)
        if not len(born):
            return out
        rng = np.random.default_rng(self.fate_seed * 1_000_003 + 7919 + self._fate_calls)
        self._fate_calls += 1
        sisters = {}
        for f in born.tolist():
            sisters.setdefault(int(par[f]), []).append(f)
        mut = cell_block(H, self.cell_set, "mutant", nF) if self.fate_bias else None
        for mother in sorted(sisters):
            pair = sisters[mother]
            u = rng.random()
            b = self.fate_bias if (mut is not None and mut[pair[0]] > 0.5) else 0.0
            if u < self.r_sym + b:
                fates = [0.0] * len(pair)                          # AA: the clone gains a progenitor
            elif u < 2.0 * self.r_sym:
                fates = [1.0] * len(pair)                          # BB: the clone loses one
            else:
                fates = [0.0] + [1.0] * (len(pair) - 1)            # AB, which sister at random
                fates = rng.permutation(fates).tolist()
            fate[pair] = fates
        set_cell_block(H, self.cell_set, "fate", fate, nF)
        return out


@register_operator("seed_cycle", set="vertex", kind="seed", family="population",
                   equation=r"""$$u\sim\mathcal U(0,1),\qquad \text{phase}=p(uT),\qquad V^0_f\leftarrow V^0_f\,\frac{1+u}{1.5}$$""", title="spread cells over the cell cycle")
class SeedCycle3D(Seed):
    r"""spread a seeded population over the cell cycle, once, before frame 0.
    vertex -> cell: writes each cell's phase, its time in the phase and its target volume from one uniform draw.

        u\sim\mathcal U(0,1),\qquad \text{phase}=p(uT),\qquad V^0_f\leftarrow V^0_f\,\frac{1+u}{1.5}
        u              where the cell is through its cycle
        T              the cycle length, the sum of `t_g1`, `t_s`, `t_g2`, `t_m`, in frames
        p(\cdot)       the phase containing that time
        V^0_f          the cell's target volume, growing from v_b to 2 v_b across the cycle

    Reference: Powell, E. O. (1956). J. Gen. Microbiol. 15:492-511 (the age distribution of a growing population). Plexus (this work)."""
    PARAM_UNITS = {'t_g1': 'frames', 't_s': 'frames', 't_g2': 'frames', 't_m': 'frames', 'seed': 'count'}
    SYMBOLS = {'u': 'where the cell is through its cycle', 'T': 'the cycle length, the sum of `t_g1`, `t_s`, `t_g2`, `t_m`, in frames', 'p(\\cdot)': 'the phase containing that time', 'V^0_f': "the cell's target volume, growing from v_b to 2 v_b across the cycle"}
    PARAM_ROLES = {'cell_set': 'the_cell_set', 'rescale_volume': 'rescale_the_target_volume_with_the_phase', 'seed': 'random_seed', 't_g1': 'duration_of_g1', 't_s': 'duration_of_s', 't_g2': 'duration_of_g2', 't_m': 'duration_of_m'}
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False
    MECHANISM_TAGS = ["cell_cycle", "asynchronous_seed", "initial_condition"]
    REFERENCE = "Plexus (this work); the phase division is Howard & Pelc (1953) Heredity 6:261-273."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex"); self._cat = params.get("cell_set")
        self.t = [float(params.get("t_g1", 110.0)), float(params.get("t_s", 80.0)),
                  float(params.get("t_g2", 40.0)), float(params.get("t_m", 10.0))]
        self.rescale = bool(params.get("rescale_volume", True))
        self.seed = int(params.get("seed", 0))

    def forward(self, H, mask=None):
        cat = resolve_cell_set(H, self.at, self._cat)
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        nF = int(m["nF"]); dev = lvl.state.device; dt = lvl.state.dtype
        for blk in ("phase", "phase_t", "cyc_inhib", "cycle_progress"):
            if cell_block(H, cat, blk, nF) is None:
                raise KeyError(
                    f"seed_cycle writes the per-cell block {blk!r}, which the set {cat!r} does not "
                    f"declare. It seeds the blocks `cell_cycle` advances, so declare all of phase, "
                    f"phase_t, cyc_inhib, cyc_vprev and cycle_progress under `sets.{cat}.state`.")
        rng = np.random.default_rng(self.seed + 4242)
        u = rng.random(nF)
        tot = float(sum(self.t)) or 1.0
        cut = np.cumsum(self.t) / tot
        ph = np.searchsorted(cut, u, side="right").astype(np.float64).clip(0, 3)
        lo = np.concatenate([[0.0], cut[:-1]])[ph.astype(int)]
        pt = (u - lo) * tot
        set_cell_block(H, cat, "phase", ph, nF)
        set_cell_block(H, cat, "phase_t", pt, nF)
        set_cell_block(H, cat, "cycle_progress", lo + pt / tot, nF)
        set_cell_block(H, cat, "cyc_inhib", np.ones(nF), nF)
        if self.rescale and m.get("V0f") is not None and len(m["V0f"]) == nF:
            v0 = m["V0f"].detach().cpu().numpy().astype(np.float64)
            m["V0f"] = torch.as_tensor(v0 * (1.0 + u) / 1.5, dtype=dt, device=dev)
            vb = cell_block(H, cat, "Vbirth", nF)
            if vb is not None and len(vb) == nF:
                set_cell_block(H, cat, "Vbirth", vb / 1.5, nF)
        print(f"[seed_cycle] spread {nF} cells over the cycle: G1 {int((ph == 0).sum())}, "
              f"S {int((ph == 1).sum())}, G2 {int((ph == 2).sum())}, M {int((ph == 3).sum())}",
              flush=True)
        return {}


def cycle_size(m, nF, v_now, v_ref, size_from="measured", v0_ref=None):
    """The size `cell_cycle` reads. `measured`: `v_now` (from `cell_size`), unchanged. `target`: the cell's target
    `V0f` in the measured convention, V0f x v_ref / v0_ref, where `v0_ref` is the SEED's median target (fixed by the
    caller on its first call) -- so `g1_size` keeps its meaning (a multiple of the seed-time median cell). Not
    median(V0f_init): `cell_grow` re-baselines that to the current targets at every division, and a floating
    reference let the median cell never reach the checkpoint (exp 11 mc_e_tc_ctl, first build)."""
    if size_from not in ("target", "both"):
        return v_now
    V0 = np.asarray(m["V0f"][:nF].detach().cpu().numpy() if hasattr(m["V0f"], "detach") else m["V0f"][:nF], np.float64)
    if v0_ref is None:
        Vi = m.get("V0f_init")
        Vi = np.asarray(Vi[:nF].detach().cpu().numpy() if hasattr(Vi, "detach") else (Vi[:nF] if Vi is not None else V0), np.float64)
        v0_ref = float(np.median(Vi))
    v_t = V0 * float(v_ref) / max(float(v0_ref), 1e-12)
    # `both`: the SMALLER of the two -- a cell leaves G1 only when its target has doubled (it wants to) AND its body has
    # (it can). Reading the target alone divided squeezed cells on schedule, halving volumes they never had; the
    # layer's median volume fell 1.18 -> 0.12 in 160 rows under every membrane setting (exp 11 Finding 124).
    return np.minimum(v_t, np.asarray(v_now, np.float64)) if size_from == "both" else v_t


@register_operator("cell_cycle", set="vertex", kind="lateral", family="population", title="advance the cell cycle", equation=r"""$$\frac{dp}{dt}=\frac{1}{T},\qquad \text{phase}=\begin{cases}G1&p<t_{g1}/T\\ S&p<(t_{g1}+t_s)/T\\ G2&p<(T-t_m)/T\\ M&\text{else}\end{cases}$$""")
class CellCycle3D(Lateral):
    r"""G1 -> S -> G2 -> M as per-cell state, advanced by a declared rule; it divides nothing.
    cell -> cell: reads the cell set's `phase_t`, `cyc_inhib`, `cyc_vprev` blocks and the mesh volumes, emits d(cycle_progress)/dt; cell_divide[cycle] reads the phase.

        \frac{dp}{dt}=\frac{1}{T},\qquad \text{phase}=\begin{cases}G1&p<t_{g1}/T\\ S&p<(t_{g1}+t_s)/T\\ G2&p<(T-t_m)/T\\ M&\text{else}\end{cases}
        p              cycle progress, 0 at birth, 1 at the end of M
        T              the cycle length, the sum of the four durations
        t_{g1}, t_s, t_{g2}, t_m `t_g1`, `t_s`, `t_g2`, `t_m`, the phase durations, in frames (defaults 110/80/40/10 of 240)

    Reference: Howard, A. & Pelc, S. R. (1953). Heredity 6(suppl.):261-273 (the G1-S-G2-M division of the cycle)."""
    PARAM_UNITS = {'t_g1': 'frames', 't_s': 'frames', 't_g2': 'frames', 't_m': 'frames', 'g1_size': 'dimensionless', 'phase_cv': 'fraction', 'seed': 'count'}
    SYMBOLS = {'p': 'cycle progress, 0 at birth, 1 at the end of M', 'T': 'the cycle length, the sum of the four durations', 't_{g1}, t_s, t_{g2}, t_m': '`t_g1`, `t_s`, `t_g2`, `t_m`, the phase durations, in frames (defaults 110/80/40/10 of 240)'}
    # NOT `MAY_MUTATE_INTEGRATED_STATE` ANY MORE (R5b): every quantity this operator produces is
    # returned as a delta on a declared block and integrated by the engine, and the one write that
    # was left -- the async spread's rescale of `V0f` and `Vbirth` -- belongs to `seed_cycle`. The
    # tick-0 integration invariant therefore covers this operator.
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False
    # DURATIONS IN SIMULATION TIME AFTER S4, not counts of frames. The cycle is one continuous
    # coordinate whose rate is 1/T, so a phase of fraction f_k traversed at 1/T takes f_k*T, and
    # `t_g1`...`t_m` ARE those durations. `p_g1` is the hazard of leaving G1, a true 1/T whose
    # reciprocal is the mean waiting time. `g1_size` is a multiple of `v_ref`, not a volume.
    # `cyc_vprev` HOLDS LAST FRAME'S VOLUME AND IT IS THE WEDGE ONE, from `face_geometry_3d` in
    # `forward`. It is what the dilution model spends and what the sizer's dV/dt is built from, so
    # the convention matters even though nothing outside this operator reads the block.
    BLOCK_UNITS = {"cyc_vprev": "volume[wedge]"}
    MECHANISM_TAGS = ["cell_cycle", "G1_S_G2_M", "phase_progression", "restriction_point",
                      "size_checkpoint"]
    REFERENCE = ("Ginzberg, M.B., Kafri, R. & Kirschner, M.W. (2015). On being the right (cell) "
                 "size. Science 348:1245075 (the G1 size checkpoint); Schmoller, K.M. et al. "
                 "(2015). Dilution of the cell-cycle inhibitor Whi5 controls budding-yeast cell "
                 "size. Nature 526:268-272; Zatulovskiy, E. et al. (2020). Cell growth dilutes the "
                 "cell cycle inhibitor Rb to trigger cell division. Science 369:466-471; "
                 "Smith, J.A. & Martin, L. (1973). Do cells cycle? PNAS 70:1263-1267.")
    PARAM_ROLES = {"t_g1": "G1 duration, frames", "t_s": "S duration, frames",
                   "t_g2": "G2 duration, frames", "t_m": "M duration, frames",
                   "g1_size": "G1 exit threshold, in units of v_ref",
                   "phase_cv": "per-cell CV on the phase durations",
 'cell_set': 'the_cell_set', 'seed': 'random_seed', 'size_from': 'volume_or_area_as_the_size_read'}
    G1, S, G2, M = 0, 1, 2, 3

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex"); self._cat = params.get("cell_set")
        self.t = [float(params.get("t_g1", 110.0)), float(params.get("t_s", 80.0)),
                  float(params.get("t_g2", 40.0)), float(params.get("t_m", 10.0))]
        self.g1_size = float(params.get("g1_size", 1.6))
        self.phase_cv = float(params.get("phase_cv", 0.0))
        # `size_from: target` (exp 11, 2026-09-28; default `measured` = unchanged): the size the cycle reads -- the
        # sizer's checkpoint, its progress and the birth size -- is the cell's TARGET `V0f`, put in the measured
        # convention by the seed's ratio v_ref / median(V0f_init) (`cycle_size`). Reading the measured volume, a
        # squeezed cell never reached the checkpoint while `cell_grow` kept growing its target, and a stretched one
        # passed it early and divided with a small target that its daughters inherited: the targets spread from
        # 0.65-1.36x the median at the seed to 0.1-8.4x by row 160 on the healthy gland (exp 11 Finding 119).
        self.size_from = str(params.get("size_from", "measured")).lower()
        if self.size_from not in ("measured", "target", "both"):
            raise ValueError(f"cell_cycle: size_from is measured, target or both, got {self.size_from!r}")
        self._vb_target = False
        if "seed_async" in params:
            raise ValueError(
                "cell_cycle no longer takes `seed_async`: spreading a seeded population over the "
                "cycle is an initial condition and belongs to the `seed_cycle` operator, which goes "
                "in the spec's `seed:` block with the same phase durations. Without it every cell "
                "starts at (G1, 0), which is what `seed_async: false` meant.")
        self.every = _engine_owns_clock(params, default=1)
        self.seed = int(params.get("seed", 0))
        self._rng = np.random.default_rng(self.seed + 4242)

    # ------------------------------------------------------------------ the rule
    def _g1_rate(self, ctx):
        """dp/dt THROUGH G1. THE ONLY THING A MODEL CHANGES; the base is the size checkpoint.

        A cell leaves G1 when it is big enough, not when it is old enough: this is the restriction
        point in mammalian cells and Start in budding yeast, and it is the transition Ginzberg et al.
        review as the one that has to be read on an ABSOLUTE scale for size control to work at all.
        `g1_size` is therefore in units of `v_ref`, the seed-time median cell volume, exactly as
        `cell_divide.factor` and `.delta` are.

        AS A RATE, THE SIZER SAYS SOMETHING THE PREDICATE COULD NOT: progress through G1 IS volume
        accumulated toward the checkpoint. The cell needs to add `v* - v_birth` of volume to pass,
        it adds `dV/dt` of it per unit time, and G1 occupies the fraction `f_G1` of the cycle, so

            dp/dt = f_G1 * (dV/dt) / (v* - v_birth),      v* = g1_size * v_ref

        reaches exactly the G1/S boundary at exactly the moment the old predicate fired. A cell that
        is ALREADY past `v*` -- one born large, or one whose threshold moved -- is carried over the
        boundary within the step by the second term, which reproduces the predicate's own answer for
        that case rather than making it wait for a volume increment it does not need.

        `ctx` carries the per-cell arrays the models read: see `forward`, where it is built.
        """
        # THE RATE IS SIGNED, AND THAT IS NOT AN OVERSIGHT. A cell that SHRINKS moves back through
        # G1, because the predicate this replaces read the cell's volume NOW -- `v_now >= v*` is
        # false again the moment the cell falls back under the threshold, and progress that only
        # ever accumulated would be a ratchet the predicate never had. Measured with the ratchet in
        # place, the sizer's population reached 2,289 cells against the predicate's 1,493 over 401
        # frames, because cells banked every transient inflation and never paid it back.
        # THE DISTANCE IS THE CELL'S OWN: `v* - V_b`, its birth volume to the checkpoint. It used
        # to be `(g1_size - 1) v_ref`, the same number for every cell, because `Vbirth` was in a
        # different volume convention from `v_now`; that made G1 an ADDER of `0.6 v_ref` for any
        # cell born below `v_ref` (R1 of notes/size_cycle/SIZE_CYCLE_PLAN.md, finding 4).
        # `cell_size` and the measured `Vbirth` are in one convention now, so the sizer can say what
        # it means. A cell born at or past `v*` is carried over the boundary in this step.
        v_star = self.g1_size * ctx["v_ref"]
        need = np.maximum(v_star - ctx["v_birth"], 1e-9)
        r = ctx["f"][self.G1] * ctx["dv_dt"] / need
        past = (ctx["v_now"] >= v_star) | (ctx["v_birth"] >= v_star)
        return np.where(past, np.maximum(r, ctx["to_boundary"]), r)

    def forward(self, H, mask=None):
        # THE PAIRING IS READ FROM THE SET, ONCE PER CALL -- see `resolve_cell_set`.
        self.cat = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None or settling(H, m):
            return {}
        nF = int(m["nF"]); dev = lvl.state.device; dt = lvl.state.dtype
        _np = lambda k: (m[k].detach().cpu().numpy().astype(np.float64)     # noqa: E731
                         if k in m and m[k] is not None else None)
        # ALL FOUR PER-CELL ARRAYS LIVE ON THE CELL SET -- see `cell_block`. `phase` joined them at
        # S2b, which is the rung that also renamed its recorded key from `vertex__mesh_phase` to
        # `cell__phase` and taught both renderer paths to read it there. Nothing of this operator's
        # state is on the mesh table any more, so `face_carry` is empty for a `cell_cycle` run and
        # the daughter-inheritance question is answered in one place: `cell_divide` copies the
        # mother's whole cell-set row.
        for _b in ("phase", "phase_t", "cyc_inhib", "cyc_vprev"):
            if cell_block(H, self.cat, _b, nF) is None:
                raise KeyError(
                    f"cell_cycle needs the set {self.cat!r} to declare the width-1 block {_b!r}. "
                    f"Add it under `sets.{self.cat}.state` -- `{_b}: {{width: 1, record: false}}` "
                    f"-- for all four of phase, phase_t, cyc_inhib, cyc_vprev. `phase` is the "
                    f"one that is RECORDED and drawn (0 = G1, 1 = S, 2 = G2, 3 = M); the other "
                    f"three are the operator's own bookkeeping -- time in phase, inhibitor "
                    f"concentration, last frame's volume -- and take `record: false`. All four "
                    f"are declared because per-cell state belongs to the cell set, where the "
                    f"topology operators already renumber it.")
        ph = cell_block(H, self.cat, "phase", nF)
        pt = cell_block(H, self.cat, "phase_t", nF)
        cc = cell_block(H, self.cat, "cyc_inhib", nF)
        # A DELTA IS MEASURED AGAINST WHAT THE BLOCK HOLDS, NOT AGAINST A LOCAL COPY OF IT, and the
        # difference is the whole reason the async seed and the birth reset still work. Both of them
        # SET a value -- `seed_async` scatters the population over the cycle, division puts a
        # daughter back at (G1, 0) -- and under the old in-place write `set_cell_block` simply stored
        # it. An operator that returns rates cannot store anything, so a set has to be expressed as
        # the rate that reaches it in one step: `(target - stored) / dt`. Measured against the
        # post-seed LOCAL instead, the seed cancels itself and every cell starts at G1 with the
        # population synchronised, which is exactly what `seed_async` exists to prevent.
        ph_st, pt_st, cc_st = ph.copy(), pt.copy(), cc.copy()
        _seeded = False
        # THE SEED FIRES ON `cyc_vprev == 0`, NOT ON A MISSING `phase`. A mesh column did not exist
        # before its first write, so `phase is None` was a sound test for "this operator has not run
        # yet". A declared block exists from the moment the set is built and reads 0.0, which is a
        # LEGITIMATE phase (G1) -- so the old test would never fire and the population would start
        # every cell at (G1, phase_t 0), the synchronised culture `seed_async` exists to avoid.
        # `cyc_vprev` is a volume: 0 is not a value any live cell can hold, and this operator is the
        # only writer, so it is the one array whose zero unambiguously means "never written".
        # THE FIRST CALL INITIALISES, IT DOES NOT SEED. Spreading the population over the cycle is
        # `seed_cycle`'s job (R5b of notes/size_cycle/SIZE_CYCLE_PLAN.md) -- it runs before frame 0,
        # writes the four blocks and rescales the volumes, which is the one thing this operator used
        # to do by writing integrated state directly. What is left here is the inhibitor's starting
        # value, returned as a delta like everything else, so this operator no longer declares
        # `MAY_MUTATE_INTEGRATED_STATE`. A run with no `seed_cycle` starts every cell at (G1, 0): a
        # synchronised culture, which is a legitimate initial condition and an obvious one in the
        # movie -- the population divides in waves.
        if not np.any(cell_block(H, self.cat, "cyc_vprev", nF) > 0.0):
            _seeded = True
            if not np.any(cell_block(H, self.cat, "cycle_progress", nF) > 0.0):
                ph = np.zeros(nF); pt = np.zeros(nF)          # no seed_cycle: everyone at (G1, 0)
            cc = np.where(cc > 0.0, cc, 1.0)
        if pt is None or len(pt) != nF:
            pt = np.zeros(nF)
        if cc is None or len(cc) != nF:
            cc = np.ones(nF)
        # DECLARED ONCE, CARRIED BY THE TOPOLOGY FROM THEN ON -- see `_carry_face_state`. Only
        # `phase` is here now; the other three are blocks on the cell set and are carried by
        # `Hierarchy.renumber_set` and by `cell_divide`'s mother-row copy instead.

        # ONE CONVENTION, CHOSEN BY THE RUN'S GEOMETRY -- see `cell_size`. This read the WEDGE
        # volume unconditionally, so on an apicobasal run the sizer's checkpoint, the dilution
        # model's `cyc_vprev` and `g1_size * v_ref` were all stated in a volume the cell does not
        # have, while `cell_mechanics[apicobasal]` was defending the polyhedron.
        v_now, v_ref = cell_size(lvl, m, nF, H=H, at=self.at)
        if self.size_from in ("target", "both"):
            if getattr(self, "_v0_ref", None) is None:
                # the first call after settling, before any division: `V0f_init` is still the seed's targets
                _vi = m.get("V0f_init", m["V0f"])
                self._v0_ref = float(np.median(np.asarray(_vi[:nF].detach().cpu().numpy() if hasattr(_vi, "detach")
                                                          else _vi[:nF], np.float64)))
            v_now = cycle_size(m, nF, v_now, v_ref, self.size_from, v0_ref=self._v0_ref)
        _dt = float(getattr(H, "dt", 1.0))
        vb0 = cell_block(H, self.cat, "Vbirth", nF)
        if self.size_from in ("target", "both") and not self._vb_target:
            # the first call after settling: every cell's birth size restated as its target, once
            vb0 = v_now.copy(); set_cell_block(H, self.cat, "Vbirth", vb0, nF); self._vb_target = True
        _rate_prev = cell_block(H, self.cat, "cyc_rate", nF)
        if _rate_prev is None or len(_rate_prev) != nF:
            _rate_prev = np.zeros(nF)
        age = _np("age")
        # THE RESET: just divided = age 0 while still carrying the mother's M. See the class note.
        born = np.zeros(nF, bool)
        if age is not None and len(age) == nF:
            born = (age <= 0) & (ph >= self.M)
            ph = np.where(born, float(self.G1), ph)
            pt = np.where(born, 0.0, pt)
        if self.size_from in ("target", "both") and born.any():
            # a daughter's birth size is its own halved TARGET, not the volume `cell_divide` measured
            vb0 = (vb0.copy() if vb0 is not None and len(vb0) == nF else v_now.copy())
            vb0[born] = v_now[born]; set_cell_block(H, self.cat, "Vbirth", vb0, nF)
        # DILUTION IS APPLIED TO EVERY CELL EVERY FRAME, whatever the model, because it is a
        # statement about volume and not about the rule: a concentration in a growing cell falls.
        #
        # A ZERO IN `cyc_vprev` MEANS "NOT YET SET" AND MUST NOT DILUTE. On the mesh table the
        # column simply did not exist before the first write, and `_np` returned None, so the first
        # frame skipped the dilution. A declared block exists from the moment the set is built and
        # reads zero, and a bare `vp / v_now` would then multiply every cell's inhibitor by 0 on
        # frame 0 -- every cell instantly past its dilution checkpoint, which is the opposite of
        # the model. The per-cell `vp > 0` test reproduces the old behaviour exactly: zero only
        # ever occurs on the first frame, because a daughter inherits its mother's row and a
        # renumbered cell keeps its own.
        # THE INHIBITOR JUST BEFORE THE DILUTION, which is what `inhibitor_dilution` spends: taken
        # AFTER the birth reset, so a daughter's first frame dilutes from the 1.0 it was reset to
        # rather than from the mother's spent concentration.
        # THE INHIBITOR AT BIRTH ENCODES SIZE, or the dilution model is not one. A size-
        # independent AMOUNT `A` synthesised per cycle (Schmoller 2015, Zatulovskiy 2020) gives a
        # birth concentration `A / V_b`, higher in a small cell; resetting every daughter to 1.0
        # made G1 end at `V = V_b / inhib_thresh` for every cell -- a relative rule (finding 5 of
        # notes/size_cycle/SIZE_CYCLE_PLAN.md). With `A = v_ref` a cell born at the reference starts
        # at 1.0 and the threshold keeps its reading. And the birth frame is NOT diluted against
        # `cyc_vprev`, which is the MOTHER's volume the daughter inherited -- that doubled the
        # inhibitor of every newborn.
        if age is not None and len(age) == nF:
            vb_now = vb0 if (vb0 is not None and len(vb0) == nF) else v_now
            cc = np.where(born, v_ref / np.maximum(vb_now, 1e-12), cc)
        cc0 = cc.copy()
        vp = cell_block(H, self.cat, "cyc_vprev", nF)
        if vp is not None and len(vp) == nF:
            fac = np.where(vp > 0.0, np.clip(vp / np.maximum(v_now, 1e-12), 0.0, 4.0), 1.0)
            cc = cc * np.where(born, 1.0, fac)
        # `cyc_vprev` IS RETURNED AS A DELTA NOW rather than written here, and the arithmetic is
        # unchanged by that: the engine adds `dt * (v_now - vp)/dt` to `vp`, which is `v_now`.

        jit = cell_block(H, self.cat, "divjit", nF)
        jit = np.ones(nF) if jit is None or len(jit) != nF else jit
        cvj = (1.0 + self.phase_cv * self._rng.standard_normal(nF)).clip(0.4, 1.8) \
            if self.phase_cv > 0 else np.ones(nF)

        # ============================================================ the cycle as a rate, not a jump
        # THE CYCLE IS STATE, NOT A COUNTER. Rather than four integers and a per-phase
        # counter advanced by a predicate, and it is now ONE continuous coordinate `p` -- position
        # through the cycle, 0 at birth, 1 at the end of M -- whose RATE the models set. `phase` is
        # read off `p` by the phase boundaries and no longer holds anything `p` does not.
        #
        # WHY IT HAD TO CHANGE. plexus2's eight families are all rate laws: an operator says how
        # fast state moves and the engine integrates. A predicate that assigns `phase + 1` is not a
        # rate, has no delta form, and is exactly why this operator was `kind: structural` and had
        # to declare `MAY_MUTATE_INTEGRATED_STATE`. Every per-frame quantity below is now returned
        # as a delta on a declared block, so the engine does the updating and the tick-0 invariant
        # is free to check it.
        #
        # WHAT `p` BUYS BEYOND COMPLIANCE, which is the part worth keeping even if the algebra had
        # not asked. Under the predicate, a cell in G1 was a cell in G1 -- there was no reading of
        # HOW FAR through it was, because "far" was only ever elapsed frames and G1's exit does not
        # read elapsed frames in three of the four models. `p` is that reading, and it is defined by
        # each model's own currency: volume accumulated toward the checkpoint for the sizer,
        # inhibitor diluted away for `inhibitor_dilution`, elapsed time for `timer`, fraction of the
        # cell's own drawn waiting time for `transition_probability`. A population's spread in `p`
        # is therefore comparable across the four in a way a phase histogram never was.
        T = float(sum(self.t)) or 1.0
        f = np.asarray(self.t, np.float64) / T                  # each phase as a fraction of the cycle
        cut = np.cumsum(f)                                      # phase boundaries in `p`
        p = cell_block(H, self.cat, "cycle_progress", nF)
        if p is None:
            raise KeyError(
                f"cell_cycle needs the set {self.cat!r} to declare the width-1 block "
                f"'cycle_progress' -- `cycle_progress: {{width: 1}}` under `sets.{self.cat}.state`. "
                f"It is the cell's continuous position through the cycle (0 at birth, 1 at the end "
                f"of M) and it is what this operator integrates; `phase` is read off it.")
        p_st = p.copy()
        # SEEDED FROM THE PHASES THE ASYNC DRAW ALREADY MADE, so `seed_async` keeps meaning what it
        # meant: a cell drawn into the middle of S starts at the `p` that IS the middle of S.
        if _seeded and not np.any(p > 0.0):
            lo = np.concatenate([[0.0], cut[:-1]])[ph.astype(int)]
            p = lo + pt / T                        # `pt` frames into a phase is pt/T of the cycle
        p = np.where(born, 0.0, p) if age is not None and len(age) == nF else p
        # dV/dt AND -dc/dt ARE THE TWO CURRENCIES THE MODELS SPEND. Both come from the frame's own
        # volume change, which `cyc_vprev` already recorded for the dilution above.
        dv_dt = np.where(vp > 0.0, (v_now - vp) / max(_dt, 1e-12), 0.0) \
            if vp is not None and len(vp) == nF else np.zeros(nF)
        _rate_draw = _rate_prev.copy()
        ctx = dict(v_now=v_now, v_ref=v_ref, v_birth=(vb0 if vb0 is not None else v_now),
                   rate_prev=_rate_prev, rate_draw=_rate_draw, born=born, conc_prev=cc0,
                   dv_dt=dv_dt, conc=cc, jit=jit, cvj=cvj, f=f, T=T, p=p, dt=_dt,
                   to_boundary=np.maximum(cut[self.G1] - p, 0.0) / max(_dt, 1e-12),
                   rng=self._rng)
        # OUTSIDE G1 EVERY MODEL IS THE SAME CLOCK, and that is the design of the family rather than
        # a simplification: S, G2 and M are near-fixed in all four hypotheses, so a difference
        # between two runs has exactly one possible cause. `1/T` in cycle units per unit time is a
        # cycle of exactly `T` -- the phase fractions do the rest, because a phase of fraction f_k
        # traversed at 1/T takes f_k * T, which is `t_k`.
        rate = np.full(nF, 1.0 / T) / np.maximum(jit * cvj, 1e-9)
        g1 = p < cut[self.G1]
        if g1.any():
            rate = np.where(g1, self._g1_rate(ctx), rate)
        # M IS NOT ADVANCED. A cell leaves M by DIVIDING, which is `cell_divide[model: cycle]`'s
        # business; advancing it here would let a tissue with no divide operator cycle for ever
        # while never making a cell, which is a population model that silently is not one. Under
        # the predicate that was `min(phase + 1, M)`; here it is a rate of zero, which is the same
        # statement and is one the engine can integrate.
        rate = np.where(p >= cut[self.G2], 0.0, rate)
        # BACKWARD ONLY INSIDE G1. S, G2 and M are clocks in every model and a clock does not run
        # backwards; only the G1 rules read a quantity that can fall.
        rate = np.where((p >= cut[self.G1]) & (rate < 0.0), 0.0, rate)
        # A DAUGHTER'S CYCLE STARTS WHEN ITS BIRTH VOLUME HAS BEEN READ, not at the septum. At the
        # cut a daughter is the geometric piece the septum made (about half the mother, CV 0.27)
        # and within a frame the mechanics has pulled it to its target; `cell_divide` re-reads
        # `Vbirth` at its next call (`age` 0 -> 1). A rule that started integrating at the cut
        # counted that mechanical jump as growth: the cycle-based adder left G1 at
        # piece + delta, a fixed 1.5 v_ref, and scored as a sizer (slope -0.96 where the
        # divide-family adder on the re-read Vbirth scored -0.13). So the cycle holds at p = 0
        # until `age` is 1 -- the frames between the cut and the next divide call, four at most.
        # A CELL THAT HAS NEVER DIVIDED IS NOT UNBORN. `age` is 0 both for a daughter cut on the
        # previous call and for every SEEDED cell until the first division call increments it, so
        # the hold below caught the whole seeded population and pinned it at p = 0 -- which, during
        # a settle window where `cell_divide` does not run at all, erased the spread `seed_cycle`
        # had just drawn and started every run as a synchronised culture (finding 24 of
        # notes/size_cycle/SIZE_CYCLE_PLAN.md: 100 % of cells in G1 at frame 61 of a run seeded
        # 43/38/14/5). `ndiv` counts divisions, so `ndiv >= 1` is what "was cut" means.
        _nd = _np("ndiv")
        if age is not None and len(age) == nF:
            unborn = (age <= 0) & ((_nd >= 1.0) if (_nd is not None and len(_nd) == nF)
                                   else np.ones(nF, bool))
            rate = np.where(unborn, 0.0, rate)
            p = np.where(unborn, 0.0, p)
        # A CELL STOPS AT M'S DOORSTEP. `rate` is already zero once `p` is inside M, so the only
        # way past is the single step that crosses -- and an unbounded rate can carry it far past
        # (a `transition_probability` cell whose drawn waiting time is a fraction of a frame reached
        # p = 3.76 before this cap). `p` has no meaning beyond the start of M, because M is left by
        # dividing and not by progressing, so the crossing step lands exactly on the boundary.
        p_next = p + _dt * rate
        # A G1 EXIT LANDS ON THE G1/S BOUNDARY AND NOT PAST IT. The G1 rules return a rate with
        # no upper bound -- the sizer's `dV/dt / (v* - V_b)` for a cell born a hair under `v*`,
        # the dilution rule's `-dc/dt` for a small cell whose inhibitor is concentrated -- and
        # a single clip at M's doorstep let such a cell cross S and G2 in the same step. Measured
        # on cycle_dilution: 200 -> 2,927 cells in 300 frames with a mean cycle of 28 frames
        # against an S+G2+M clock of 130, and the median volume at zero. S and G2 are clocks and
        # take their own time from the boundary on.
        # ... AND A HAIR PAST IT. `cycle_progress` is float32 state, and a cell parked EXACTLY on
        # `cut[G1]` reads as G1 again next frame once the stored value rounds below the boundary
        # (183.3/400 does; 110/240 happened not to), so the cap put it back on the boundary every
        # frame for ever: R2's four cycle arms froze at 304 cells with 100 % of cells "in S".
        p_next = np.where(g1, np.minimum(p_next, cut[self.G1] + 1e-5), p_next)
        p_next = np.clip(p_next, 0.0, cut[self.G2])
        ph_next = np.searchsorted(cut, p_next, side="right").astype(np.float64).clip(0, self.M)
        # `phase_t` IS A READOUT NOW, NOT A DRIVER. Nothing above consults it -- the models spend
        # volume, inhibitor or a drawn waiting time, and `timer` spends `p` itself -- but it is
        # declared by every spec that runs this operator and is what a reader means by "how long has
        # this cell been in S", so it is kept and reset on a phase change like before.
        pt_next = np.where(ph_next != ph, 0.0, pt + _dt)
        return self._emit(H, nF, {
            "cycle_progress": (p_next - p_st) / max(_dt, 1e-12),
            "phase":          (ph_next - ph_st) / max(_dt, 1e-12),
            "phase_t":        (pt_next - pt_st) / max(_dt, 1e-12),
            "cyc_inhib":      (cc - cc_st) / max(_dt, 1e-12),
            "cyc_vprev":      (v_now - (vp if vp is not None else v_now)) / max(_dt, 1e-12),
            "cyc_rate":       (ctx["rate_draw"] - _rate_prev) / max(_dt, 1e-12),
        })

    def _emit(self, H, nF, blocks):
        """Per-cell deltas -> full-buffer `(set, block)` deltas the engine integrates.

        THE LIVE CELLS ARE A PREFIX OF THE BUFFER and `_integrate` adds a block delta to the whole
        of `lvl.state[:, cx0:cx1]` with no mask, so a short tensor would not broadcast and a full
        one must be zero past `nF` or the dead tail of the buffer would drift.
        """
        clvl = H.level(self.cat)
        N = int(clvl.state.shape[0])
        out = {}
        for name, d in blocks.items():
            if d is None or cell_block(H, self.cat, name, nF) is None:
                continue                       # a block this spec did not declare is not written
            z = torch.zeros(N, 1, device=clvl.state.device, dtype=clvl.state.dtype)
            z[:nF, 0] = torch.as_tensor(np.asarray(d, np.float64),
                                        device=clvl.state.device).to(clvl.state.dtype)
            out[(self.cat, name)] = z
        return out



@register_operator("cell_cycle", model="timer", set="vertex", kind="lateral", family="population", title="advance the cell cycle on a clock", equation=r"""$$\text{leave G1 at}\ t=t_{g1}$$""")
class CellCycleTimer(CellCycle3D):
    r"""`timer` model of cell_cycle: every phase on a clock, G1 included; the null a checkpoint has to beat.
    cell -> cell: as cell_cycle; only the G1 exit rule differs.

        \text{leave G1 at}\ t=t_{g1}
        t              frames in G1
        t_{g1}         `t_g1`

    Reference: Ginzberg, M. B., Kafri, R. & Kirschner, M. W. (2015). Science 348:1245075."""
    PARAM_UNITS = {}
    SYMBOLS = {'t': 'frames in G1', 't_{g1}': '`t_g1`'}
    MECHANISM_TAGS = ["cell_cycle", "G1_S_G2_M", "timer", "size_independent"]

    def _g1_rate(self, ctx):
        # THE CLOCK, WHICH IS THE SAME CLOCK THE OTHER THREE PHASES RUN ON. `1/T` in cycle units per
        # unit time makes G1 take `f_G1 * T = t_g1`, so this model is the one whose `p` is literally
        # elapsed time -- and that is the null the other three have to beat.
        return np.full(np.shape(ctx["v_now"]), 1.0 / ctx["T"]) / np.maximum(ctx["jit"] * ctx["cvj"], 1e-9)


@register_operator("cell_cycle", model="adder", set="vertex", kind="lateral", family="population", title="advance the cycle by added volume", equation=r"""$$\frac{dp}{dt}=f_{G1}\frac{dV/dt}{\Delta\,v_{\mathrm{ref}}},\qquad \text{leave G1 at}\ V=V_b+\Delta\,v_{\mathrm{ref}}$$""")
class CellCycleAdder(CellCycle3D):
    r"""`adder` model of cell_cycle: G1 ends when the cell has added `g1_delta` times `v_ref` since birth.
    cell -> cell: as cell_cycle; only the G1 exit rule differs.

        \frac{dp}{dt}=f_{G1}\frac{dV/dt}{\Delta\,v_{\mathrm{ref}}},\qquad \text{leave G1 at}\ V=V_b+\Delta\,v_{\mathrm{ref}}
        p              cycle progress
        f_{G1}         G1's share of the cycle
        V, V_b         the cell's volume and its birth volume
        \Delta         `g1_delta`, in units of the seed-time median volume v_ref

    Reference: Taheri-Araghi, S. et al. (2015). Curr. Biol. 25:385-391; Cadart, C. et al. (2018). Nat. Commun. 9:3275."""
    PARAM_UNITS = {'g1_delta': 'dimensionless'}
    SYMBOLS = {'p': 'cycle progress', 'f_{G1}': "G1's share of the cycle", 'V, V_b': "the cell's volume and its birth volume", '\\Delta': '`g1_delta`, in units of the seed-time median volume v_ref'}
    MECHANISM_TAGS = ["cell_cycle", "G1_S_G2_M", "adder", "size_control", "incremental_threshold"]
    PARAM_ROLES = {"g1_delta": "volume added through G1, in units of v_ref"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.g1_delta = float(params.get("g1_delta", 1.0))

    def _g1_rate(self, ctx):
        need = max(self.g1_delta * ctx["v_ref"], 1e-9)
        r = ctx["f"][self.G1] * ctx["dv_dt"] / need
        past = ctx["v_now"] >= ctx["v_birth"] + need
        return np.where(past, np.maximum(r, ctx["to_boundary"]), r)


@register_operator("cell_cycle", model="doubler", set="vertex", kind="lateral", family="population", title="advance the cycle at doubled volume", equation=r"""$$\frac{dp}{dt}=f_{G1}\frac{dV/dt}{(\phi-1)V_b},\qquad \text{leave G1 at}\ V=\phi V_b$$""")
class CellCycleDoubler(CellCycle3D):
    r"""`doubler` model of cell_cycle: G1 ends at `g1_factor` times this cell's own birth volume, the relative rule that corrects nothing.
    cell -> cell: as cell_cycle; only the G1 exit rule differs.

        \frac{dp}{dt}=f_{G1}\frac{dV/dt}{(\phi-1)V_b},\qquad \text{leave G1 at}\ V=\phi V_b
        p              cycle progress
        f_{G1}         G1's share of the cycle
        V, V_b         the cell's volume and its birth volume
        \phi           `g1_factor`, dimensionless

    Reference: Ginzberg, M. B., Kafri, R. & Kirschner, M. W. (2015). Science 348:1245075."""
    PARAM_UNITS = {'g1_factor': 'dimensionless'}
    SYMBOLS = {'p': 'cycle progress', 'f_{G1}': "G1's share of the cycle", 'V, V_b': "the cell's volume and its birth volume", '\\phi': '`g1_factor`, dimensionless'}
    MECHANISM_TAGS = ["cell_cycle", "G1_S_G2_M", "doubler", "relative_threshold", "no_size_control"]
    PARAM_ROLES = {"g1_factor": "multiple of the cell's own birth volume at which G1 ends"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.g1_factor = float(params.get("g1_factor", 2.0))

    def _g1_rate(self, ctx):
        need = np.maximum((self.g1_factor - 1.0) * ctx["v_birth"], 1e-9)
        r = ctx["f"][self.G1] * ctx["dv_dt"] / need
        past = ctx["v_now"] >= self.g1_factor * ctx["v_birth"]
        return np.where(past, np.maximum(r, ctx["to_boundary"]), r)


@register_operator("cell_cycle", model="transition_probability", set="vertex", kind="lateral",
                   family="population", title="start the cycle at random", equation=r"""$$P(\text{still in G1 after }k\text{ frames})=(1-p_{g1})^k,\qquad \text{mean wait}\ 1/p_{g1}$$""")
class CellCycleTransitionProbability(CellCycle3D):
    r"""`transition_probability` model of cell_cycle: G1 is a constant hazard of starting, not a duration.
    cell -> cell: as cell_cycle; only the G1 exit rule differs.

        P(\text{still in G1 after }k\text{ frames})=(1-p_{g1})^k,\qquad \text{mean wait}\ 1/p_{g1}
        p_{g1}         `p_g1`, the per-frame probability of leaving G1
        k              frames in G1

    Reference: Smith, J. A. & Martin, L. (1973). PNAS 70:1263-1267."""
    PARAM_UNITS = {'p_g1': 'fraction'}
    SYMBOLS = {'p_{g1}': '`p_g1`, the per-frame probability of leaving G1', 'k': 'frames in G1'}
    MECHANISM_TAGS = ["cell_cycle", "G1_S_G2_M", "transition_probability", "memoryless",
                      "stochastic_start"]
    REFERENCE = "Smith, J.A. & Martin, L. (1973). Do cells cycle? PNAS 70:1263-1267."
    PARAM_ROLES = {"p_g1": "per-frame probability of leaving G1; default 1/t_g1"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.p_g1 = float(params.get("p_g1", 1.0 / max(self.t[0], 1.0)))

    def _g1_rate(self, ctx):
        """THE HAZARD, WRITTEN AS A SPEED DRAWN ONCE PER CELL, which is the same distribution.

        A per-frame coin flip has no rate form -- it is a jump, and a jump is what S4 exists to
        remove -- but the QUANTITY it defines does. A constant hazard `p_g1` means the waiting time
        is exponential with mean `1/p_g1`; so drawing `w ~ Exp(1/p_g1)` when the cell enters G1 and
        crossing G1 at the constant speed `f_G1 / w` produces exactly that waiting-time
        distribution, and is memoryless in the only sense the model claims -- the draw does not read
        how long the cell has waited, or anything else about it.

        WHAT IT CHANGES, said plainly: an individual cell's exit is now decided at G1 entry rather
        than re-rolled every frame, so a run is no longer memoryless CONDITIONAL ON THE RUN -- the
        population's exponential tail is identical, one cell's own future is not. The alternative
        (rate = f_G1 * Bernoulli(p_g1)/dt) reproduces the per-frame roll and is a jump wearing a
        rate's clothes, which is the thing being retired.

        `cyc_rate` HOLDS THE DRAW so it survives to the next frame and through a renumbering; it is
        reset for the cells that just entered G1, which after S4 are exactly the ones `born` marks.
        """
        # THE DRAW IS FLOORED AT ONE STEP. An exponential has no lower bound, so a waiting time of
        # 1e-4 frames is a legal draw and gives a G1 speed of `f_G1 / 1e-4` -- which carried `p` to
        # 3.76 in the first version of this, past the end of the cycle in a single step. A cell
        # cannot pass through G1 faster than the simulation can resolve, so `dt` is the floor.
        w = np.maximum(self._rng.exponential(1.0 / max(self.p_g1, 1e-12), np.shape(ctx["v_now"])),
                       ctx["dt"])
        fresh = ctx["born"] | (ctx["rate_prev"] <= 0.0)
        ctx["rate_draw"][:] = np.where(fresh, w, ctx["rate_prev"])
        return ctx["f"][self.G1] / np.maximum(ctx["rate_draw"], 1e-9)


@register_operator("cell_cycle", model="inhibitor_dilution", set="vertex", kind="lateral",
                   family="population", title="advance the cycle as inhibitor dilutes", equation=r"""$$c\leftarrow c\,\frac{v_{\mathrm{prev}}}{v_{\mathrm{now}}}\ \text{every frame},\qquad \text{leave G1 when}\ c\le c_{\mathrm{th}}$$""")
class CellCycleInhibitorDilution(CellCycle3D):
    r"""`inhibitor_dilution` model of cell_cycle: the size checkpoint with a mechanism under it, an inhibitor diluted by growth crossing a threshold.
    cell -> cell: as cell_cycle; only the G1 exit rule differs.

        c\leftarrow c\,\frac{v_{\mathrm{prev}}}{v_{\mathrm{now}}}\ \text{every frame},\qquad \text{leave G1 when}\ c\le c_{\mathrm{th}}
        c              the inhibitor concentration, the `cyc_inhib` block
        v_{prev}, v_{now} the cell's volume last frame and now
        c_{th}         `inhib_thresh`, in the block's units

    Reference: Schmoller, K. M. et al. (2015). Nature 526:268-272 (Whi5 dilution)."""
    PARAM_UNITS = {'inhib_thresh': 'dimensionless'}
    SYMBOLS = {'c': 'the inhibitor concentration, the `cyc_inhib` block', 'v_{prev}, v_{now}': "the cell's volume last frame and now", 'c_{th}': "`inhib_thresh`, in the block's units"}
    MECHANISM_TAGS = ["cell_cycle", "G1_S_G2_M", "inhibitor_dilution", "size_control",
                      "restriction_point", "mechanistic_sizer"]
    REFERENCE = ("Schmoller, K.M., Turner, J.J., Koivomagi, M. & Skotheim, J.M. (2015). Dilution of "
                 "the cell-cycle inhibitor Whi5 controls budding-yeast cell size. Nature "
                 "526:268-272; Zatulovskiy, E., Zhang, S., Berenson, D.F., Topacio, B.R. & "
                 "Skotheim, J.M. (2020). Cell growth dilutes the cell cycle inhibitor Rb to trigger "
                 "cell division. Science 369:466-471.")
    PARAM_ROLES = {"inhib_thresh": "concentration at which G1 ends, relative to its value at birth"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.inhib_thresh = float(params.get("inhib_thresh", 0.6))

    def _g1_rate(self, ctx):
        # PROGRESS IS INHIBITOR SPENT. The cell has to dilute `c` from 1 at birth down to
        # `inhib_thresh`, and it spends `-dc/dt` of that per unit time, so G1 is traversed exactly
        # when the concentration arrives at the threshold:
        #
        #     dp/dt = f_G1 * (-dc/dt) / (1 - inhib_thresh)
        #
        # `-dc/dt` is the dilution the base class has already applied this frame, read back as the
        # difference it made. A cell that is not growing does not dilute and does not progress,
        # which is the model's own claim and was true of the predicate too.
        # SIGNED, for the reason the sizer's rate is: a cell whose volume falls RE-CONCENTRATES its
        # inhibitor, and the predicate `c <= inhib_thresh` goes false again when it does. Clamping
        # the drop at zero made progress a ratchet and let the population run to 2,532 cells against
        # the predicate's 727 over 401 frames.
        drop = (ctx["conc_prev"] - ctx["conc"]) / max(ctx["dt"], 1e-12)
        # THE SPAN IS THE CELL'S OWN: from ITS birth concentration `v_ref / V_b` down to the
        # threshold. Normalised by `1 - inhib_thresh` it was the same span for every cell, so a
        # cell born small -- concentrated, `c_b` 1.5 -- left G1 after diluting by 0.4, at `c` 1.1,
        # never reaching the threshold at all: mean cycle 129 frames against an S+G2+M clock of
        # 130 on cycle_dilution, i.e. no G1. Same defect, same fix, as the sizer's denominator.
        c_b = ctx["v_ref"] / np.maximum(ctx["v_birth"], 1e-12)
        span = np.maximum(c_b - self.inhib_thresh, 1e-9)
        r = ctx["f"][self.G1] * drop / span
        return np.where(ctx["conc"] <= self.inhib_thresh, np.maximum(r, ctx["to_boundary"]), r)


@register_operator("cell_divide", model="cycle", set="vertex", kind="divide", family="population", title="divide when the cycle reaches mitosis", equation=r"""$$\text{divide } j\ \text{when}\ \text{phase}_j=M$$""")
class Divide3DCycle(Divide3D):
    r"""`cycle` model of cell_divide: divide when cell_cycle says the cell is in M.
    vertex -> vertex: as cell_divide; the trigger reads the phase and nothing else.

        \text{divide } j\ \text{when}\ \text{phase}_j=M
        phase_j        the cell's phase written by cell_cycle

    Reference: none of its own -- the hypothesis lives in the scheduled cell_cycle model. Plexus (this work)."""
    PARAM_UNITS = {}
    SYMBOLS = {'phase_j': "the cell's phase written by cell_cycle"}
    MECHANISM_TAGS = ["division", "cell_cycle", "mitosis", "phase_gated"]

    def _trigger(self, v_now, v_birth, jit, age, v_ref):
        return False                                    # replaced wholesale in forward, below

    def _ready_mask(self, H, m, nF):
        # THE PHASE IS A BLOCK ON THE CELL SET, so this needs the hierarchy and not just the mesh
        # table. A spec that schedules this model without declaring the block, or without
        # `cell_cycle`, gets an all-False mask and divides nothing -- the honest failure named in
        # the class docstring, unchanged by where the array lives.
        ph = cell_block(H, self.cell_set, "phase", nF)
        if ph is None:
            return np.zeros(nF, bool)
        return ph >= CellCycle3D.M


@register_operator("cell_divide", model="adder", set="vertex", kind="divide", family="population", title="divide after adding a fixed volume", equation=r"""$$\text{divide } j\ \text{when}\ v_j\ge v^{\mathrm{birth}}_j+\Delta\,\mathrm{jit}_j\,v_{\mathrm{ref}}$$""")
class Divide3DAdder(Divide3D):
    r"""`adder` model of cell_divide: divide once the cell has added a constant volume since birth.
    vertex -> vertex: as cell_divide; only the trigger differs.

        \text{divide } j\ \text{when}\ v_j\ge v^{\mathrm{birth}}_j+\Delta\,\mathrm{jit}_j\,v_{\mathrm{ref}}
        v_j, v^{birth}_j the cell's volume and its birth volume
        \Delta         `delta`, in units of the seed-time median volume v_ref, as `factor` is
        jit_j          the per-cell jitter from `cycle_cv`

    Reference: Campos, M. et al. (2014). Cell 159:1433-1446; Taheri-Araghi, S. et al. (2015). Curr. Biol. 25:385-391."""
    PARAM_UNITS = {'delta': 'dimensionless'}
    SYMBOLS = {'v_j, v^{birth}_j': "the cell's volume and its birth volume", '\\Delta': '`delta`, in units of the seed-time median volume v_ref, as `factor` is', 'jit_j': 'the per-cell jitter from `cycle_cv`'}
    MECHANISM_TAGS = ["division", "adder", "size_control", "incremental_threshold",
                      "absolute_increment"]
    REFERENCE = ("Taheri-Araghi, S. et al. (2015). Cell-size control and homeostasis in bacteria. "
                 "Curr. Biol. 25:385-391; Cadart, C. et al. (2018). Size control in mammalian cells "
                 "involves modulation of both growth rate and cell cycle duration. "
                 "Nat. Commun. 9:3275.")
    PARAM_ROLES = {"delta": "volume added per cycle, in units of v_ref"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.delta = float(params.get("delta", 1.0))

    def _trigger(self, v_now, v_birth, jit, age, v_ref):
        return v_now >= v_birth + self.delta * jit * v_ref


# Moved here from `sheet_ops.py` on 2026-09-27 (experiments/INSTRUCTION.md: a variant lives beside its
# base operator). The file's docstring, kept:
#
# Division on a FLAT, OPEN sheet: `cell_divide[implementation: adder_sheet]`, in its own module.
#
#     cell_divide[adder_sheet]   divide   the `adder` rule unchanged; the septum is computed in the
#                                         sheet's plane and placed across two edges that HAVE a
#                                         neighbour, so a cell on the free edge can divide
#
# WHY IT EXISTS (exp12_tumor_invasion, 2026-09-26, round 4). The division geometry in `Divide3D.forward`
# was written for a closed shell centred on the origin, and a flat disc breaks it twice:
#
# 1. THE SEPTUM ORIENTATION IS NOISE. The short-axis direction is w = n x u with n = c / |c|, "the outward
#    (radial) face normal on the sphere". On a disc in the z = 0 plane c lies IN the plane, so n is an
#    in-plane radial vector, w points out of the plane, every edge midpoint projects to 0 on it, and
#    `ea = eb = 0`; `divide_face_3d` then falls back to edges 0 and k/2 of the ring -- an orientation set
#    by the ring's starting vertex, not by the cell's long axis (Hertwig's rule is lost).
# 2. A CELL ON THE FREE EDGE NEVER DIVIDES when that fallback pair contains a rim edge, because
#    `divide_face_3d` refuses a split edge with no neighbour (documented at `build_disc_mesh`). Over
#    300 frames it did not matter; over 900 the rim cells swelled to 65-77x the median area, broke
#    `tools/mesh_sanity.py`'s area line, and -- the nutrient uptake being per area -- drained the supply
#    where it enters, so cells starved at the tissue's EDGE.
#
# THE FIX, AND WHERE IT LIVES. The edge pair is recomputed here: the long axis u of the cell's polygon
# projected into the sheet's plane (normal n = the `plane_axis` unit vector), the short-axis direction
# w = n x u, and the edges whose midpoints lie furthest along +w and -w, not adjacent. An edge on the free
# edge may be crossed: `divide_face_open` splits it with no neighbouring ring to update, so the rim gains
# a cell (with the original's refusal a rim cell could only split through interior edges and the
# perimeter never grew). No midpoint projection: "back out to the local shell radius" means nothing on a
# plane. The choice lives inside `Divide3D.forward`'s loop, so rather than copy
# ~300 lines this implementation routes the ONE call that consumes it, `divide_face_3d`, through
# `sheet_septum` for the duration of its own forward and restores it after -- the shared file is
# untouched, and every other run's division is exactly what it was.

def sheet_split_edges(ring, pos, emap, f, axis=2, allow_open=True):
    """(ea, eb): the two edges of face `f` (edge i runs ring[i] -> ring[i+1]) the septum crosses --
    across the cell's in-plane long axis (Hertwig), not adjacent to each other -- or None when no such
    pair exists. With `allow_open` an edge on the FREE EDGE (no twin) may be crossed, which is what lets
    the rim gain cells (`divide_face_open`); without it only edges with a neighbour qualify."""
    k = len(ring)
    if k < 4:
        return None
    P = np.asarray([pos[v] for v in ring], float)
    n = np.zeros(3); n[int(axis)] = 1.0
    c = P.mean(0)
    Q = P - c
    Q[:, int(axis)] = 0.0
    try:
        _, _, vh = np.linalg.svd(Q, full_matrices=False)
        u = vh[0]
    except np.linalg.LinAlgError:
        return None
    w = np.cross(n, u)
    w /= np.linalg.norm(w) + 1e-12
    mids = 0.5 * (P + np.roll(P, -1, 0))
    proj = (mids - c) @ w
    ok = [i for i in range(k) if emap.get((ring[(i + 1) % k], ring[i])) != f
          and (allow_open or emap.get((ring[(i + 1) % k], ring[i])) is not None)]
    best, best_gap = None, -np.inf
    for a in ok:
        for b in ok:
            if a == b or (a - b) % k in (1, k - 1):
                continue
            gap = proj[a] - proj[b]
            if gap > best_gap:
                best, best_gap = (a, b), gap
    return best


def divide_face_open(rings, pos, f, ea, eb, emap, births=None):
    """`divide_face_3d` for an OPEN mesh: the same edge-midpoint septum across edges `ea` and `eb`, but
    a split edge on the free edge (no twin) is split too -- its midpoint joins the rim, with no
    neighbouring ring to update. Returns (daughter B's index, m1, m2) or None, like the original, and
    maintains `emap` and `births` the same way. No midpoint projection: a sheet has no shell radius.

    WHY IT IS NEEDED. `divide_face_3d` refuses an edge without a neighbour, so a rim cell could only
    ever divide through two interior edges, the daughter keeping the free edge kept ALL of it, and the
    perimeter never gained a cell however far the disc grew (exp12 round 5 check: 75 rim cells before
    and after 228 forced divisions)."""
    import plexus.models.topology as TOP
    r = rings[f]
    k = len(r)
    if k < 4:
        return None
    i0, i1 = sorted((ea % k, eb % k))
    if i1 - i0 < 2 or (i0 + k - i1) < 2:
        return None
    a0, a1 = r[i0], r[(i0 + 1) % k]
    b0, b1 = r[i1], r[(i1 + 1) % k]
    nbrA, nbrB = emap.get((a1, a0)), emap.get((b1, b0))
    if nbrA == f or nbrB == f or (nbrA is not None and nbrA == nbrB):
        return None
    m1, m2 = len(pos), len(pos) + 1
    pos.append(0.5 * (np.asarray(pos[a0]) + np.asarray(pos[a1])))
    pos.append(0.5 * (np.asarray(pos[b0]) + np.asarray(pos[b1])))
    if births is not None:
        births.append((m1, (int(a0), int(a1))))
        births.append((m2, (int(b0), int(b1))))
    old_A = list(rings[nbrA]) if nbrA is not None else None
    old_B = list(rings[nbrB]) if nbrB is not None else None
    if nbrA is not None and not TOP._insert_after(rings[nbrA], a1, a0, m1):
        return None
    if nbrB is not None and not TOP._insert_after(rings[nbrB], b1, b0, m2):
        if nbrA is not None:
            TOP._emap_drop(emap, nbrA, old_A); TOP._emap_add(emap, nbrA, rings[nbrA])
        return None
    seg_A = [m1] + [r[j % k] for j in range(i0 + 1, i1 + 1)] + [m2]
    seg_B = [m2] + [r[j % k] for j in range(i1 + 1, i0 + 1 + k)] + [m1]
    rings[f] = seg_A
    rings.append(seg_B)
    TOP._emap_drop(emap, f, r)
    if nbrA is not None:
        TOP._emap_drop(emap, nbrA, old_A)
    if nbrB is not None:
        TOP._emap_drop(emap, nbrB, old_B)
    TOP._emap_add(emap, f, seg_A)
    TOP._emap_add(emap, len(rings) - 1, seg_B)
    if nbrA is not None:
        TOP._emap_add(emap, nbrA, rings[nbrA])
    if nbrB is not None:
        TOP._emap_add(emap, nbrB, rings[nbrB])
    return len(rings) - 1, m1, m2


@contextlib.contextmanager
def sheet_septum(axis=2):
    """Route `plexus.models.topology.divide_face_3d` through the sheet's edge choice, and restore it on
    exit (also on an exception). `Divide3D.forward` imports the name at call time, so it sees this."""
    import plexus.models.topology as TOP
    orig = TOP.divide_face_3d

    def planar(rings, pos, f, project=True, ea=None, eb=None, emap=None, births=None):
        m = emap if emap is not None else TOP._edge_face_map(rings)
        pair = sheet_split_edges(rings[f], pos, m, f, axis)
        if pair is None:
            return None
        return divide_face_open(rings, pos, f, pair[0], pair[1], m, births)

    TOP.divide_face_3d = planar
    try:
        yield planar
    finally:
        TOP.divide_face_3d = orig


@register_operator("cell_divide", implementation="adder_sheet", set="vertex", kind="divide",
                   family="population", title="divide after adding volume, flat sheet", equation=r"""$$\text{divide } j\ \text{when}\ v_j\ge v^{\mathrm{birth}}_j+\Delta\,v_{\mathrm{ref}},\qquad \text{septum}\perp\text{in-plane long axis}$$""")
class Divide3DAdderSheet(Divide3DAdder):
    r"""`adder_sheet` implementation of cell_divide: the adder on a flat open sheet, the septum placed across the cell's in-plane long axis.
    vertex -> vertex: as cell_divide[adder]; the septum is in-plane and the seeded cells may start spread over the cycle.

        \text{divide } j\ \text{when}\ v_j\ge v^{\mathrm{birth}}_j+\Delta\,v_{\mathrm{ref}},\qquad \text{septum}\perp\text{in-plane long axis}
        v_j, v^{birth}_j the cell's volume and its birth volume
        \Delta         `delta`, in median cells
        v_{ref}        the seed-time median volume

    Reference: Hertwig, O. (1884). Jena. Z. Naturwiss. 18:274-312 (the long-axis rule); the adder as in `cell_divide[adder]`."""
    PARAM_UNITS = {'phase_spread': 'fraction', 'plane_axis': 'count'}
    SYMBOLS = {'v_j, v^{birth}_j': "the cell's volume and its birth volume", '\\Delta': '`delta`, in median cells', 'v_{ref}': 'the seed-time median volume'}
    MECHANISM_TAGS = Divide3DAdder.MECHANISM_TAGS + ["flat_sheet", "hertwig"]
    PARAM_ROLES = {**Divide3DAdder.PARAM_ROLES, "plane_axis": "axis normal to the sheet",
                   "phase_spread": "spread of the seeded cells' first-cycle progress, 0-1"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.sheet_axis = int(params.get("plane_axis", 2))
        self.phase_spread = float(params.get("phase_spread", 0.0))
        if not 0.0 <= self.phase_spread <= 1.0:
            raise ValueError(f"cell_divide[adder_sheet]: phase_spread must be in [0, 1], got {self.phase_spread}")

    def _seed_phases(self, H, m):
        """Random first-cycle progress for every live cell, once, at the first call after settling."""
        import torch
        lvl = H.level(self.at)
        nF = int(m["nF"]); Nv = int(m["Nv"])
        pos_np = lvl.get("pos")[:Nv].detach().cpu().numpy().astype(np.float64)
        v, v_ref = cell_size(lvl, m, nF, pos_np, H=H, at=self.at)        # caches the settled reference
        jit = cell_block(H, self.cell_set, "divjit", nF)
        jit = np.ones(nF) if jit is None else np.asarray(jit, np.float64)
        seed = int(getattr(getattr(H, "config", None), "seed", 0) or 0)
        u = np.random.default_rng(7919 + seed).uniform(0.0, self.phase_spread, nF)
        vb = np.asarray(v, np.float64)[:nF] - u * self.delta * jit * float(v_ref)
        set_cell_block(H, self.cell_set, "Vbirth", vb, nF)
        age = m.get("age")
        dt = m["V0f"].dtype; dev = m["V0f"].device
        m["age"] = torch.full((nF,), 2.0, dtype=dt, device=dev) if age is None else \
            torch.clamp(torch.as_tensor(age, dtype=dt, device=dev)[:nF], min=2.0)   # past the age-1 re-read
        m["phase_seeded"] = True

    def forward(self, H, mask=None):
        m = getattr(H.level(self.at), "_mesh", None)
        if self.phase_spread > 0 and m is not None and not m.get("phase_seeded") and not settling(H, m):
            self._seed_phases(H, m)
        with sheet_septum(self.sheet_axis):
            return super().forward(H, mask)


# cell_divide[point] -- DIVISION OF A CELL THAT IS ONE PARTICLE (exp 11, 2026-09-27: the inner cells of a
# stratified bud, replacing the growing-volume clock `k_core`/`core_rate` of cell_mechanics[apicobasal_contact]).
POINT_DIVIDE_TRACE: list = []       # per call: (frame, divisions, live count)


@register_operator("cell_divide", model="point", set="particle", kind="divide", family="population",
                   title="divide point cells",
                   equation=r"""$$p=e^{r\,\Delta t}-1,\qquad \mathbf x_{m,d}=\mathbf x\mp\tfrac{\delta}{2}\,\hat{\mathbf u}$$""")
class CellDividePoint(Structural):
    r"""`point` model of cell_divide: one particle per cell divides into a free slot, the daughters placed about the mother.
    particle -> particle: wakes dormant slots (every per-slot buffer copied from the mother), writes the two daughters' positions and zeroes their `vel`; no delta.

        p=e^{r\,\Delta t}-1,\qquad \mathbf x_{m,d}=\mathbf x\mp\tfrac{\delta}{2}\,\hat{\mathbf u}
        p              the per-call division probability of a live cell
        r              `rate`, the population's exponential growth rate, per unit time
        \Delta t       the frame length
        x_{m,d}        the two daughters' positions
        \delta         `offset`, the distance between the daughters, in world units
        \hat u         a uniform random unit vector

    Reference: Drasdo, D. & Hoehme, S. (2005). Phys. Biol. 2:133-147 (the centre-based cell)."""
    PARAM_UNITS = {'rate': 'rate', 'offset': 'length', 'max_cells': 'count', 'seed': 'count', 'rate_decay_T': 'frames'}
    SYMBOLS = {'p': 'the per-call division probability of a live cell', 'r': "`rate`, the population's exponential growth rate, per unit time", '\\Delta t': 'the frame length', 'x_{m,d}': "the two daughters' positions", '\\delta': '`offset`, the distance between the daughters, in world units', '\\hat u': 'a uniform random unit vector'}
    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["rate"]
    MECHANISM_TAGS = ["division", "cell_division", "proliferation", "centre_based_cells"]
    PARAM_ROLES = {"rate": "exponential_growth_rate_per_time", "offset": "daughter_separation_at_birth",
                   "max_cells": "population_ceiling", "seed": "rng_seed", 'rate_decay_T': 'frames_over_which_the_rate_decays_zero_means_none'}
    REFERENCE = ("Drasdo, D. & Hoehme, S. (2005). Phys. Biol. 2:133-147.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.rate = float(params["rate"])
        # `rate_decay_T` (default 0 = none): the rate falls as rate / (1 + t / T) -- the same hyperbolic decay as
        # `cell_grow`'s, so an interior growing with a linearly growing surface (Wang 2021 Fig 1H) keeps its ratio to it.
        self.rate_decay_T = float(params.get("rate_decay_T", 0.0))
        self.offset = float(params.get("offset", 0.5))
        self.max_cells = int(params.get("max_cells", 0) or 0)
        self.seed = int(params.get("seed", 0))
        self._gen = torch.Generator().manual_seed(self.seed)
        self._said_full = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dt = float(getattr(H, "dt", 1.0) or 1.0)
        _r = self.rate
        if self.rate_decay_T > 0:
            _r = _r / (1.0 + float(getattr(H, "frame", 0) or 0) * dt / self.rate_decay_T)
        p = math.expm1(_r * dt)
        if not 0.0 <= p < 1.0:
            raise ValueError(f"cell_divide[point]: rate {self.rate:g} x dt {dt:g} gives a division probability "
                             f"{p:.3g} per call; it must lie in [0, 1) -- lower the rate or dt")
        live = (lvl.occ > 0).nonzero(as_tuple=True)[0]
        n_live = int(live.numel())
        fr = getattr(H, "frame", None)
        fr = -1 if fr is None else int(fr)
        u = torch.rand(n_live, generator=self._gen)           # drawn for every live cell, every call
        mother = live[(u < p).to(live.device)]
        if self.max_cells > 0:
            mother = mother[: max(0, self.max_cells - n_live)]
        if mother.numel() == 0:
            POINT_DIVIDE_TRACE.append((fr, 0, n_live))
            return {}
        D = lvl.get("pos").shape[-1]
        d = torch.randn(int(mother.numel()), D, generator=self._gen)
        d = (d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(lvl.state.device, lvl.state.dtype)
        new, src = lvl.spawn(mother)
        if new.numel() < mother.numel() and not self._said_full:
            self._said_full = True
            print(f"[cell_divide/point] `{self.at}` has no free slot for {int(mother.numel() - new.numel())} "
                  f"division(s) at frame {fr} ({n_live} live of {lvl.n}); raise the set's `buffer`", flush=True)
        if new.numel():
            half = 0.5 * self.offset * d[: new.numel()]
            p0, p1 = lvl.state_schema["pos"]
            x = lvl.state[src, p0:p1].clone()
            lvl.state[src, p0:p1] = x - half
            lvl.state[new, p0:p1] = x + half
            if "vel" in lvl.state_schema:
                v0, v1 = lvl.state_schema["vel"]
                lvl.state[src, v0:v1] = 0.0
                lvl.state[new, v0:v1] = 0.0
        POINT_DIVIDE_TRACE.append((fr, int(new.numel()), n_live + int(new.numel())))
        return {}


@register_operator("cell_divide", model="reinsert", set="vertex", kind="divide", family="population",
                   title="return inner cells to the surface",
                   equation=r"""$$P(\text{return per call}\mid h_i\le b)=p_{\mathrm{ret}}$$""")
class Divide3DReinsert(Divide3D):
    r"""`reinsert` model of cell_divide: an inner cell returns to the surface layer, taken in as a new surface cell by the face-division machinery.
    vertex -> vertex (and `from_set` particle -> gone): per call, at most `max_per_call` live particles of `from_set` within `band` of the apical surface become surface cells; the host's own state is put back.

        P(\text{return per call}\mid h_i\le b)=p_{\mathrm{ret}}
        h_i            the particle's height above the apical ring, from the nearest fan triangle
        b              `band`, in world units
        p_{ret}        `p_return`, the per-call probability

    Reference: Wang, S. et al. (2021). Cell 184:3702-3716 (STAR Methods eq 20-22, 29-32)."""
    PARAM_UNITS = {'band': 'length', 'p_return': 'fraction', 'kappa_G': 'dimensionless', 'g_ref': 'dimensionless', 'max_per_call': 'count', 'new_frac': 'fraction', 'seed': 'count'}
    SYMBOLS = {'h_i': "the particle's height above the apical ring, from the nearest fan triangle", 'b': '`band`, in world units', 'p_{ret}': '`p_return`, the per-call probability'}
    PARAM_ROLES = {'from_set': 'particle_set_the_returning_cells_come_from', 'band': 'height_above_the_apical_surface_within_which_a_particle_may_return', 'host_pick': 'how_the_host_face_is_chosen', 'p_return': 'per_call_probability_of_a_return', 'p_from': 'fixed_or_from_a_block', 'kappa_G': 'gain_of_the_block_read_probability', 'g_ref': 'reference_level_of_that_block', 'ecad_block': 'block_the_probability_is_read_from', 'max_per_call': 'ceiling_on_returns_per_call', 'new_frac': 'fraction_of_the_host_given_to_the_new_cell', 'new_frac_of': 'what_new_frac_is_a_fraction_of', 'sep_block': 'name_of_the_half_separation_block', 'mark': 'block_set_on_a_returned_cell', 'from_children': 'child_set_of_from_set', 'age_layer': 'age_the_returned_cell_as_the_layer', 'seed': 'random_seed'}
    MECHANISM_TAGS = ["division", "reinsertion", "stratification", "cell_sorting"]
    REQUIRES_PARAMS = ["from_set"]
    _HOST_BLOCKS = ("Vbirth", "divjit", "cell_id", "parent_id")
    _NEW_ZERO = ("phase", "cycle_progress", "phase_t", "cyc_vprev")

    def __init__(self, params, device="cpu"):
        p = dict(params)
        p.setdefault("every", 1)                  # the base's private default (3) would refuse the spec
        super().__init__(p, device)
        self.from_set = str(params["from_set"])
        self.band = float(params.get("band", 0.6))
        self.host_pick = str(params.get("host_pick", "nearest")).lower()
        self.p_return = float(params.get("p_return", 0.05))
        if not 0.0 <= self.p_return <= 1.0:
            raise ValueError(f"cell_divide[reinsert]: p_return must lie in [0, 1], got {self.p_return}")
        # `p_from: adhesion` (default `fixed`, p_return for every body; exp 11 Phase 3 B5, 2026-09-29): each body's
        # own probability, `adhesion_return_p(p_return, lam_host, G_body, g_ref, kappa_G)` -- the matrix at the host
        # (bm_clutch's covered fraction x the membrane's mass there, by the host's cell_id) and the body's
        # E-cadherin (`ecad_block` of `from_set`, default `ecad`; `g_ref` 0.35, `kappa_G` 2.0)
        self.p_from = str(params.get("p_from", "fixed")).lower()
        if self.p_from not in ("fixed", "adhesion"):
            raise ValueError(f"cell_divide[reinsert]: p_from is fixed or adhesion, got {self.p_from!r}")
        self.kappa_G = float(params.get("kappa_G", 2.0))
        self.g_ref = float(params.get("g_ref", 0.35))
        self.ecad_block = str(params.get("ecad_block", "ecad"))
        self.max_per_call = int(params.get("max_per_call", 4))
        self.new_frac = float(params.get("new_frac", 0.5))
        self.sep_block = str(params.get("sep_block", "sep"))
        self.mark = params.get("mark")
        # `from_children: <point set>` (default none): `from_set` is a set of MPM CELLS -- a returning cell's
        # block of material points goes back to the dormant pool with it (`mpm_ops.sleep_mpm_body`).
        self.from_children = params.get("from_children")
        self.age_layer = bool(params.get("age_layer", False))
        self.new_frac_of = str(params.get("new_frac_of", "host")).lower()
        if self.new_frac_of not in ("host", "median", "reference"):
            raise ValueError(f"cell_divide[reinsert]: new_frac_of is host, median or reference, got {self.new_frac_of!r}")
        self._ref_targets = {}
        self.seed = int(params.get("seed", 0))
        self._rng = np.random.default_rng(self.seed + 7919)
        self._geo = None
        self._hosts = []

    def _trigger(self, v_now, v_birth, jit, age, v_ref):
        return False

    def _ready_mask(self, H, m, nF):
        mk = np.zeros(nF, bool)
        mk[np.asarray(self._hosts, np.int64)] = True
        return mk

    def _adhesion_p(self, H, m, P, live, face, nF):
        """Per live body, `adhesion_return_p` at the face under it: the host's matrix from bm_clutch's last call
        (`clutch_by_id`: cell_id -> covered fraction x mass; 1 where the host is unknown), the body's E-cadherin."""
        lam = np.ones(len(face))
        by = m.get("clutch_by_id")
        ids = cell_block(H, self.cell_set, "cell_id", nF)
        if by is not None and ids is not None:
            look = dict(zip(np.asarray(by[0], np.int64).tolist(), np.asarray(by[1], np.float64).tolist()))
            ff = np.clip(face, 0, nF - 1)
            hid = np.rint(np.asarray(ids, np.float64)[ff]).astype(np.int64)
            lam = np.array([look.get(int(k), 1.0) for k in hid])
        G = np.full(len(face), self.g_ref)
        if self.ecad_block in P.state_schema:
            G = P.state[live, P.state_schema[self.ecad_block][0]].detach().cpu().numpy().astype(np.float64)
            G = np.where(G > 0, G, self.g_ref)
        # the body's integrin, carried from the layer through the dive (bm_clutch's pools), over the resting U0 = 1
        u = np.ones(len(face))
        _it = [b for b in ("itg_I", "itg_A", "itg_N", "itg_B") if b in P.state_schema]
        if len(_it) == 4:
            u = sum(P.state[live, P.state_schema[b][0]].detach().cpu().numpy().astype(np.float64) for b in _it)
            u = np.where(u > 0, u, 1.0)                       # a body that never carried any: no penalty
        p = adhesion_return_p(self.p_return, lam, G, self.g_ref, self.kappa_G, u)
        m["return_p_mean"] = float(p.mean()) if len(p) else 0.0
        return p

    def _returning(self, H, m, nF):
        """(particle slots, host faces) of this call's returns, one particle per host."""
        from plexus.operators import membrane_ops as _MO
        P = H.level(self.from_set)
        if P is None:
            raise ValueError(f"cell_divide[reinsert]: from_set {self.from_set!r} is not a set of this spec")
        live = (P.occ > 0.5).nonzero(as_tuple=True)[0]
        u = self._rng.random(int(live.numel()))            # drawn for every live particle, every call
        if not live.numel():
            return [], []
        if self._geo is None:
            self._geo = _MO._basal_lookup()
        y = P.get("pos")[live]
        got = _MO._live_basal(H, self.at, self.sep_block, self._geo, y.device, y.dtype, side="apical")
        if got is None:
            return [], []
        _, c, M = got
        hit, tri, _w, _n, gap = _MO._nearest_faces(M, c, y)
        h = -gap                                            # height INTO the lumen
        face = M["ef"][tri].cpu().numpy()
        pr = self.p_return
        if self.p_from == "adhesion":
            pr = self._adhesion_p(H, m, P, live, face, nF)
        ok = (hit & (h <= self.band)).cpu().numpy() & (u < pr) & (face < nF)
        if self.mark:
            if self.mark not in P.state_schema:
                raise ValueError(f"cell_divide[reinsert]: mark {self.mark!r} is not a block of {self.from_set!r}")
            ok &= P.state[live, P.state_schema[self.mark][0]].detach().cpu().numpy() > 0.5
        ring = np.bincount(m["E_face"].cpu().numpy(), minlength=nF)
        fl = m.get("apop_flag")                            # a cell on its way out hosts no one
        if fl is not None and len(fl) == nF:
            fl = fl.detach().cpu().numpy() if torch.is_tensor(fl) else np.asarray(fl)
            ring = np.where(fl > 0, 0, ring)
        # `host_pick: largest` (exp 11 ladder L3, 2026-09-28; default `nearest` = unchanged): the newcomer enters by
        # cutting the LARGEST cell among the face under it and that face's neighbours -- Wang's "sites of easiest
        # entry"; cutting whichever cell lay under the returning body halved small and large cells alike, and in a
        # layer that grows only by returns that random halving was the volume spread (CV 0.58-0.72, Finding 139).
        _vol = _nbr = None
        if self.host_pick == "largest":
            _vol = np.asarray(cell_size(H.level(self.at), m, nF, H=H, at=self.at)[0], np.float64)
            es_, et_, ef_ = (m[k].cpu().numpy() for k in ("E_srce", "E_trgt", "E_face"))
            _key = {(int(a), int(b)): int(f) for a, b, f in zip(es_, et_, ef_)}
            _nbr = {}
            for (a, b), f in _key.items():
                g = _key.get((b, a))
                if g is not None and f < nF and g < nF:
                    _nbr.setdefault(f, set()).add(g)
        slots, hosts = [], []
        for i in np.flatnonzero(ok):
            f = int(face[i])
            if _vol is not None:
                cand = [g for g in ([f] + sorted(_nbr.get(f, ()))) if g not in hosts and ring[g] >= 4]
                if cand:
                    f = max(cand, key=lambda g: _vol[g])
            if f in hosts or ring[f] < 4:                  # one return per host; a triangle cannot be cut
                continue
            slots.append(int(live[i])); hosts.append(f)
            if len(hosts) >= self.max_per_call:
                break
        return slots, hosts

    def forward(self, H, mask=None):
        self.cell_set = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None or settling(H, m):
            return {}
        nF = int(m["nF"])
        # `age_layer: true` (default false = unchanged): THIS operator ages every cell of the layer by one per
        # call. `age` is advanced by `cell_divide`'s base body, which this model restores on the old cells so a
        # layer that also divides in place is not aged twice; with NO division in the layer (exp 11's Type I
        # loop: every division dives) nothing aged the cells, every `age` stayed 0, and `cell_cycle`'s birth
        # test (age 0 while in M) reset each cell that reached M back to (G1, 0) -- after the seeded first wave
        # no cell ever reached M again and the layer stopped growing (Finding 96).
        if self.age_layer and "age" in m:
            a = m["age"].detach().clone()
            a[:nF] = a[:nF] + 1.0
            m["age"] = a
        slots, hosts = self._returning(H, m, nF)
        if not hosts:
            return {}
        cs = self.cell_set
        cid0 = cell_block(H, cs, "cell_id", nF)
        if cid0 is None:
            raise ValueError("cell_divide[reinsert] needs a `cell_id` block on the cell set")
        keep_m = {k: m[k].detach().clone() for k in ("A0", "P0", "V0f", "age", "ndiv") if k in m}
        keep_b = {k: cell_block(H, cs, k, nF) for k in self._HOST_BLOCKS}
        keep_s = {k: m.get(k) for k in ("n_div", "div_blocked", "buf_full")}
        self._hosts = hosts
        try:
            super().forward(H, mask)
        finally:
            self._hosts = []
        m = lvl._mesh
        nF2 = int(m["nF"])
        cid = cell_block(H, cs, "cell_id", nF2)
        # WHO WAS CUT: a host's id was replaced by `next_id`, its appended piece holds `next_id + 1`
        # (division keeps every old face at its index -- `keep` is the identity -- and appends).
        by_id = {int(round(v)): j for j, v in enumerate(cid[:nF2])}
        cut = [f for f in hosts if int(round(cid[f])) != int(round(cid0[f]))]
        new = [by_id[int(round(cid[f])) + 1] for f in cut]
        for k, v in keep_m.items():                        # the old cells, as they were
            a = m[k].detach().clone()
            a[:nF] = v[:nF].to(a.dtype)
            # `new_frac_of: median` (default `host`): the newcomer's targets are `new_frac` of the LAYER'S MEDIAN
            # cell, not of its host's -- a host is often near division size, and a newcomer given its targets
            # is ready to leave the moment it arrives (exp 11: 302 of 419 layer cells waiting in M, giant facets)
            ref = None
            if k in ("A0", "V0f", "P0"):
                if self.new_frac_of == "median":
                    ref = float(v[:nF].median())
                elif self.new_frac_of == "reference":
                    # `reference`: the layer's median AT THE FIRST RETURN, cached -- a median read each call is a
                    # RATCHET: newcomers at 0.7 of it pull it down, the next ones are smaller still (0.7^k), and
                    # the layer thinned to a median volume of 0 by the end of the clip (Finding 101)
                    if k not in self._ref_targets:
                        self._ref_targets[k] = float(v[:nF].median())
                    ref = self._ref_targets[k]
            for f, j in zip(cut, new):
                if k in ("A0", "V0f"):
                    a[j] = self.new_frac * (ref if ref is not None else v[f])
                elif k == "P0":
                    a[j] = (ref if ref is not None else v[f]) * math.sqrt(self.new_frac)
                elif k == "age":
                    a[j] = 0.0
                elif k == "ndiv":
                    a[j] = 1.0
            m[k] = a
        for k, v in keep_b.items():
            if v is None:
                continue
            a = np.array(cell_block(H, cs, k, nF2), np.float64)
            a[:nF] = v[:nF]
            if k == "parent_id":
                a[new] = -1.0
            set_cell_block(H, cs, k, a, nF2)
        for k in self._NEW_ZERO:
            a = cell_block(H, cs, k, nF2)
            if a is not None and new:
                a = np.array(a, np.float64); a[new] = 0.0
                set_cell_block(H, cs, k, a, nF2)
        P = H.level(self.from_set)
        clvl = H.level(cs)
        names = [k for k, (a, b) in P.state_schema.items()
                 if b - a == 1 and k not in ("age", "cell_id", "parent_id") and k in clvl.state_schema
                 and clvl.state_schema[k][1] - clvl.state_schema[k][0] == 1]
        gone = [slots[hosts.index(f)] for f in cut]
        for k in names:
            a = np.array(cell_block(H, cs, k, nF2), np.float64)
            a[new] = P.state[gone, P.state_schema[k][0]].detach().cpu().numpy()
            set_cell_block(H, cs, k, a, nF2)
        rep = m.get("div_blocked", 0)
        for k, v in keep_s.items():
            m[k] = v if v is not None else {"n_div": 0, "div_blocked": 0, "buf_full": False}[k]
        fl = m.get("apop_flag")                            # a newcomer is not under a host's sentence
        if fl is not None and new and len(fl) == nF2:
            fl = fl.clone() if torch.is_tensor(fl) else np.array(fl, np.float64)
            fl[new] = 0.0
            m["apop_flag"] = fl
        m["reinsert_blocked"] = int(rep or 0)
        if gone:
            if self.from_children:
                from plexus.operators.mpm_ops import sleep_mpm_body
                sleep_mpm_body(H, self.from_set, self.from_children, gone)
            P.kill(torch.as_tensor(gone, dtype=torch.long, device=P.occ.device))
        m["n_reinsert"] = int(m.get("n_reinsert", 0)) + len(gone)
        return {}


# ============================================================================================
#  a colony on a lattice of SITES (exp 15, rig 1; moved from colony_ops.py 2026-09-27, unchanged)
# ============================================================================================
# The set's buffer holds every SITE of the domain, placed once by `seed_positions`; a site is
# occupied when its slot is live (`occ = 1`) and empty when it is dormant, and the strain is two
# one-hot columns of `chem` at `chan`. A division wakes one specific empty slot beside the mother, so
# nothing moves and the colony's history is written into WHERE each strain sits. `Level.spawn` is not
# used: it wakes the FIRST free slots and copies the mother's position into them, right for an
# off-lattice body and wrong for a site. Momeni et al. 2013 (eLife 2:e00230): two cooperating strains
# intermix, competitors segregate -- their individual-based fitness model, every constant as printed.

from plexus.operators.seed_ops import _strain_span   # noqa: E402  one definition, beside seed_colony


def _spacing(x):
    """Median nearest-site distance of the whole lattice (every slot, live or dormant)."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(x).query(x, k=2)
    return float(np.median(d[:, 1]))


@register_operator("topo_record", set="vertex", kind="structural", family="harness", title="record the mesh each frame", probe=True, equation=r"""$$\text{rec}_t=(\text{half-edges}_t,\ n_{\mathrm{vert},t})$$""")
class TopoSnapshot3D(Structural):
    r"""a measurement, as an operator: record the current mesh once per frame so a vesicle whose topology changes can be rendered frame by frame.
    vertex -> vertex: reads the half-edge table and the vertex count, writes nothing; appends one row per frame to the recording.

        \text{rec}_t=(\text{half-edges}_t,\ n_{\mathrm{vert},t})
        rec_t          the row recorded at frame t

    Reference: none -- a recording, not a mechanism. Plexus (this work)."""
    PARAM_UNITS = {}
    SYMBOLS = {'rec_t': 'the row recorded at frame t'}
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = False
    MECHANISM_TAGS = ["recording", "topology_history", "diagnostic"]
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device); self.at = params.get("_at", "vertex")
        self.every = _engine_owns_clock(params); self._k = 0   # engine owns the stride (D1)

    def forward(self, H, mask=None):
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        self._k += 1                    # monotonic tick only -- D1: the engine owns the period
        def cp(k):                                              # per-cell mechanical targets (for offline force/stress
            v = m.get(k)                                        # analysis) -- None-safe numpy copies
            return v.detach().cpu().numpy().copy() if v is not None and hasattr(v, "detach") else None
        m.setdefault("hist", []).append(dict(
            E_srce=m["E_srce"].detach().cpu().numpy().copy(),
            E_trgt=m["E_trgt"].detach().cpu().numpy().copy(),
            E_face=m["E_face"].detach().cpu().numpy().copy(),
            nF=int(m["nF"]), Nv=int(m["Nv"]),
            A0=cp("A0"), P0=cp("P0"), V0f=cp("V0f"),            # targets -> analyze_forces reconstructs the energy
            # `age` = division-calls since this cell was born (cell_divide resets it to 0 on
            # division). Recorded so a renderer can colour RECENTLY DIVIDED cells. The first
            # attempt inferred "just divided" from cell AREA -- a sliver test -- and it never
            # fired: a division splits a cell into two roughly equal halves, so a normal daughter
            # is ~50-70% of its neighbours while the sliver test looks below 15%. It detects
            # DEGENERATE cells, not new ones. Age is the actual event, not a proxy for it.
            age=cp("age"), ndiv=cp("ndiv"),
            # PER-JUNCTION MYOSIN, when a junction operator has written one. Recorded for the same
            # reason `age`/`ndiv` are: a renderer cannot colour by a quantity that only existed inside
            # one frame's forward pass. `cp` is None-safe, so a run without the operator records None.
            myo=cp("myo"),
            # PER-CELL GROWTH INHIBITION, when a second morphogen is switching growth off. Recorded
            # for exactly the reason `age` and `myo` are: a renderer cannot colour by a quantity
            # that only existed inside one frame's forward pass, and "where is growth stopped" is
            # the whole point of an inhibitor -- an invisible mechanism is one nobody can check.
            # None-safe, so a run without an inhibitor records None and the renderer draws nothing.
            inhib=cp("inhib_frac"),
            # THE CONSERVATION LAW'S OWN ERROR TERM. Material a dying cell could not bequeath
            # without pushing a neighbour out of the integrator's basin is dropped and counted
            # here rather than injected. It must be ~0 on a healthy run; a large value says the
            # bequest is being refused, which is the diagnostic for the -7.3e11 divergence.
            apop_spill=float(m.get("apop_spill", 0.0)),
            # THE TWO-POOL STATE, when `medioapical_myosin` is in the schedule. `myo_med` is the
            # AREAL density on each cell and `myo_amount` the AMOUNT on each half-edge; the pair is
            # what makes the conservation ledger measurable offline, since `myo` alone is normalised
            # to a tissue mean of 1 and so cannot say how much myosin there is.
            myo_med=cp("myo_med"), myo_amount=cp("myo_amount"),
            # THE RESERVOIR, PER FRAME. cell_divide sets these on the mesh and nothing carried them
            # into the history, so run_one read them and always found nothing -- a run that
            # plateaued at 98.5% of its array reported buf_full False. The flag existed, the
            # counter existed, and the one structure anybody reads afterwards did not have them.
            div_blocked=int(m.get("div_blocked") or 0),
            # CUMULATIVE DEATHS, for the same reason div_blocked is here: cell_die counts
            # every extrusion on the mesh and nothing carried it into the history, so the only
            # visible trace of a death was the cell COUNT -- which cannot distinguish "nothing
            # died" from "deaths were masked by divisions". A run whose count grows and a run
            # whose count holds can both have death running, and the count alone cannot say which
            # is which.
            n_apop=int(m.get("n_apop") or 0),
            # WHICH CELLS ARE MARKED TO DIE, recorded for the same reason `age` is: a renderer
            # cannot colour a state that only existed inside one frame's forward pass. Without it
            # a dying cell is drawn exactly like a living one, the sheet closes over the gap, and
            # the mechanism is invisible in the movie -- which is what happened: a 293-cell patch
            # dying at the north pole could not be seen even looking straight down at it.
            # EITHER TYPE. The flag is a declared block on the cell set now, so it arrives as a
            # tensor; a spec that has not moved it still has the numpy column. An `isinstance`
            # check on one of the two silently recorded None for the other.
            apop=_as_np(m.get("apop_flag")),
            buf_full=bool(m.get("buf_full"))))
        return {}


def face_polygons_3d(pos_np, mesh):
    """3D face polygons + per-face area/perimeter/shape-index (for rendering)."""
    es, et, ef, nF = mesh["E_srce"], mesh["E_trgt"], mesh["E_face"], mesh["nF"]
    rings = [[] for _ in range(nF)]
    for k in range(len(ef)):
        rings[int(ef[k])].append(int(es[k]))
    polys, area, perim = [], np.zeros(nF), np.zeros(nF)
    for f in range(nF):
        v = pos_np[rings[f]]; polys.append(v)
        N = 0.5 * np.abs(np.cross(v, np.roll(v, -1, 0)).sum(0))
        area[f] = np.linalg.norm(0.5 * np.cross(v, np.roll(v, -1, 0)).sum(0))
        perim[f] = np.linalg.norm(np.roll(v, -1, 0) - v, axis=1).sum()
    shape = np.where(area > 1e-9, perim / np.sqrt(np.maximum(area, 1e-9)), np.nan)
    return polys, area, perim, shape


def _vertex_faces(rings):
    """vertex -> set of faces incident to it (used to find the third face at u / at v)."""
    vf = {}
    for f, r in enumerate(rings):
        if r is None or len(r) < 3:
            continue
        for w in r:
            vf.setdefault(w, set()).add(f)
    return vf


def _insert_before(ring, anchor, w):
    """New ring with w inserted immediately BEFORE the first occurrence of anchor (or None)."""
    out, done = [], False
    for x in ring:
        if x == anchor and not done:
            out.append(w); done = True
        out.append(x)
    return out if done else None


def _insert_after(ring, anchor, w):
    """New ring with w inserted immediately AFTER the first occurrence of anchor (or None)."""
    out, done = [], False
    for x in ring:
        out.append(x)
        if x == anchor and not done:
            out.append(w); done = True
    return out if done else None


def _ring_ok(r):
    return r is not None and len(r) >= 3 and len(set(r)) == len(r)


# --------------------------------------------------------------------------------------------------
#  local closed-surface (manifold) check on the 4 changed faces
# --------------------------------------------------------------------------------------------------
def _boundary_de(faces_map):
    """Directed edges of a face patch; returns the set of BOUNDARY directed edges (those whose reverse
    is not in the patch), or None if any directed edge repeats inside the patch (-> non-manifold)."""
    de = set()
    for r in faces_map.values():
        k = len(r)
        for i in range(k):
            e = (r[i], r[(i + 1) % k])
            if e in de:
                return None                                  # directed edge used twice -> non-manifold
            de.add(e)
    return set(e for e in de if (e[1], e[0]) not in de)


def _local_manifold_ok(old_map, new_map):
    """A T1 only touches the four faces {A,B,C,D}; every changed directed edge stays inside that patch.
    So the mesh stays closed iff the patch mates with the (unchanged) exterior exactly as before, i.e.
    the patch's boundary directed edges are unchanged -- and no directed edge repeats in the new patch."""
    b_old = _boundary_de(old_map); b_new = _boundary_de(new_map)
    return b_new is not None and b_old is not None and b_new == b_old


# --------------------------------------------------------------------------------------------------
#  3D per-face validity: non-degenerate, outward-facing, and SIMPLE (no self-crossing)
# --------------------------------------------------------------------------------------------------
def _seg_cross(p1, p2, p3, p4):
    """Proper 2D segment intersection (touching endpoints do NOT count)."""
    ccw = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1 = ccw(p3, p4, p1); d2 = ccw(p3, p4, p2); d3 = ccw(p1, p2, p3); d4 = ccw(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _polygon_simple_2d(Q):
    """True iff the 2D polygon Q has no two NON-adjacent edges properly crossing (catches bow-ties)."""
    k = len(Q)
    for i in range(k):
        for j in range(i + 1, k):
            if j == i + 1 or (i == 0 and j == k - 1):
                continue                                     # adjacent edges share a vertex
            if _seg_cross(Q[i], Q[(i + 1) % k], Q[j], Q[(j + 1) % k]):
                return False
    return True


def _face_ok_3d(ring, getp, normal=None):
    """Face is valid: >=3 verts, non-zero area, Newell normal points OUTWARD (dot with centroid > 0),
    and the polygon is SIMPLE when projected onto its own plane. `getp(i)` returns vertex i's 3-vector.

    `normal` -- THE SHEET'S OWN NORMAL, for a flat tissue. "Outward" is a vesicle's word: the Newell
    normal is compared with the centroid, i.e. the radial from the origin. A sheet built by
    `build_sheet_mesh` lies IN the plane z = 0 through the origin, so that dot product is exactly
    zero for every face and every T1 is refused as "inward-facing" -- a `sheet_*` run flips
    nothing while it is planar and only begins to flip once it has buckled. With the
    plane declared (`cell_mechanics.plane_axis`), the orientation to keep is the one the builder
    fixed, CCW seen from +axis, and that is what `normal` compares against.

    IN PLAIN PYTHON FLOATS, NOT NUMPY. A ring has six vertices; the numpy version spent 160 us per
    call in the fixed cost of ~20 small-array operations (np.cross alone was 40 us), and `edge_flip`
    calls this seven times per candidate, 24 candidates a tick -- 6.4 ms of every frame on
    `spheroid_organelles_nucleus`. The arithmetic is the same IEEE-754 double sequence (sums in
    ring order, then the halving, the same 1e-9 and 1e-12 guards); 200,000 random and degenerate
    rings gave the same verdict as the numpy body on every one."""
    k = len(ring)
    if k < 3:
        return False
    P = []
    for i in ring:
        p = getp(i)
        P.append((float(p[0]), float(p[1]), float(p[2])))
    cx = cy = cz = 0.0
    for x, y, z in P:
        cx += x; cy += y; cz += z
    cx /= k; cy /= k; cz /= k
    Nx = Ny = Nz = 0.0                                       # Newell area vector (|N| = area)
    for i in range(k):
        x0, y0, z0 = P[i]; x1, y1, z1 = P[(i + 1) % k]
        Nx += y0 * z1 - z0 * y1; Ny += z0 * x1 - x0 * z1; Nz += x0 * y1 - y0 * x1
    Nx *= 0.5; Ny *= 0.5; Nz *= 0.5
    a = math.sqrt(Nx * Nx + Ny * Ny + Nz * Nz)
    if a < 1e-9:
        return False                                         # degenerate
    if normal is None:
        d = Nx * cx + Ny * cy + Nz * cz
    else:
        d = Nx * float(normal[0]) + Ny * float(normal[1]) + Nz * float(normal[2])
    if d <= 0.0:
        return False                                         # flipped over
    nx, ny, nz = Nx / a, Ny / a, Nz / a                      # project to the face plane, test simplicity
    def _tangent(p):
        ex, ey, ez = p[0] - cx, p[1] - cy, p[2] - cz
        t = ex * nx + ey * ny + ez * nz
        return ex - t * nx, ey - t * ny, ez - t * nz
    e1 = _tangent(P[0])
    if math.sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]) < 1e-9:
        e1 = _tangent(P[1])
    L = math.sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]) + 1e-12
    e1x, e1y, e1z = e1[0] / L, e1[1] / L, e1[2] / L
    e2x, e2y, e2z = ny * e1z - nz * e1y, nz * e1x - nx * e1z, nx * e1y - ny * e1x
    Q = [((x - cx) * e1x + (y - cy) * e1y + (z - cz) * e1z,
          (x - cx) * e2x + (y - cy) * e2y + (z - cz) * e2z) for x, y, z in P]
    return _polygon_simple_2d(Q)


# --------------------------------------------------------------------------------------------------
#  the T1 flip
# --------------------------------------------------------------------------------------------------
# Per frame: (frame, flips this frame, cumulative flips, live faces). The RATE per cell per frame is
# what discriminates a homogenising (length-keyed) myosin feedback from a destabilising (tension-keyed)
# one; junction-length CV moves the same way under both.
T1_TRACE: list = []


def t1_flip_3d(rings, pos, e_uv, new_len=None, emap=None, vf=None, plane_axis=None):
    """One surface T1 on interior edge e_uv=(u,v): rewire the four rings A,B,C,D and move u,v apart
    along the tangent-plane perpendicular (projected onto the shell). Mutates `rings` and pos[u],pos[v]
    in place and returns (u,v) on success; returns None (no-op) if the flip is impossible or would break
    an invariant: boundary edge, non-trivalent u/v, C==D, a face < 3 verts, a duplicated vertex in a
    ring, a broken closed-surface, or a non-simple/inward face for BOTH chiralities x BOTH signs.
    OPTIMISATION: pass `emap` (edge->face) and `vf` (vertex->faces) built ONCE by the caller; a successful
    flip updates them in place (only the 4 faces A,B,C,D change), avoiding an O(F) rebuild per candidate."""
    u, v = int(e_uv[0]), int(e_uv[1])
    if emap is None:
        emap = _edge_face_map(rings)
    A = emap.get((u, v)); B = emap.get((v, u))
    if A is None or B is None or A == B:
        return None                                          # boundary / degenerate interior edge
    if vf is None:
        vf = _vertex_faces(rings)
    Cs = vf.get(u, set()) - {A, B}; Ds = vf.get(v, set()) - {A, B}
    if len(Cs) != 1 or len(Ds) != 1:
        return None                                          # u or v not trivalent -> no clean T1
    C = Cs.pop(); D = Ds.pop()
    if C == D or len({A, B, C, D}) != 4:
        return None                                          # the two "far" cells must be distinct
    rA, rB, rC, rD = rings[A], rings[B], rings[C], rings[D]

    nA = [w for w in rA if w != v]                           # A loses v
    nB = [w for w in rB if w != u]                           # B loses u
    if not (_ring_ok(nA) and _ring_ok(nB)):
        return None                                          # would leave a face with < 3 verts

    # geometry: collapse u,v to their midpoint then reopen perpendicular in the local tangent plane
    pu = np.asarray(pos[u], float); pv = np.asarray(pos[v], float)
    mid = 0.5 * (pu + pv); rmid = np.linalg.norm(mid)
    d = pv - pu; L = np.linalg.norm(d)
    if L < 1e-12 or (plane_axis is None and rmid < 1e-9):
        return None
    # THE SHELL NORMAL, AND ON A SHEET IT IS NOT THE RADIAL. For a closed vesicle the outward
    # radial from the world origin is the surface normal, which is what this assumed. On a FLAT
    # patch it is not: a sheet at height 5 has a radial that tilts away from +z everywhere except
    # the axis, so reopening the junction perpendicular to it, and then re-projecting both new
    # vertices onto a sphere below, lifts them off the plane at every flip. Ablation measured it
    # exactly: with `edge_flip` removed the sheet stayed at sd(z) = 0.00e+00, with it 4.23% of an
    # edge -- the whole residual, and none of it from division or from the relaxation.
    if plane_axis is None:
        n = mid / rmid                                       # outward radial (shell normal)
    else:
        n = np.zeros(3); n[int(plane_axis)] = 1.0            # the sheet's own normal
    perp = np.cross(n, d / L); pn = np.linalg.norm(perp)
    if pn < 1e-9:                                            # edge is radial -> perpendicular undefined
        return None
    perp = perp / pn
    half = 0.5 * (new_len if new_len is not None else L)     # reopen at ~ new_len (default: same length)
    rm = 0.5 * (np.linalg.norm(pu) + np.linalg.norm(pv))     # target shell radius for both split verts

    old_map = {A: rA, B: rB, C: rC, D: rD}
    #   two chiralities: insert v/u BEFORE (v->u in C) or AFTER (u->v in C) -- both are closed T1s;
    #   geometry (simple + outward) picks the correct, non-folding one.
    for insert in (_insert_before, _insert_after):
        nC = insert(rC, u, v)                                # C gains v next to u
        nD = insert(rD, v, u)                                # D gains u next to v
        if not (_ring_ok(nC) and _ring_ok(nD)):
            continue
        new_map = {A: nA, B: nB, C: nC, D: nD}
        if not _local_manifold_ok(old_map, new_map):
            continue                                         # would break the closed surface
        for sign in (+1.0, -1.0):
            nu = mid - sign * perp * half; nv = mid + sign * perp * half
            if plane_axis is None:
                nu = nu * (rm / (np.linalg.norm(nu) + 1e-12))  # back onto the shell
                nv = nv * (rm / (np.linalg.norm(nv) + 1e-12))
            else:                                              # back onto the SHEET
                nu[int(plane_axis)] = mid[int(plane_axis)]
                nv[int(plane_axis)] = mid[int(plane_axis)]
            getp = lambda i: (nu if i == u else nv if i == v else pos[i])
            if all(_face_ok_3d(r, getp, normal=None if plane_axis is None else n) for r in (nA, nB, nC, nD)):
                for fid, ro, rn in ((A, rA, nA), (B, rB, nB), (C, rC, nC), (D, rD, nD)):
                    for i in range(len(ro)):                # keep the passed maps in sync: only A,B,C,D changed
                        emap.pop((ro[i], ro[(i + 1) % len(ro)]), None)
                    for i in range(len(rn)):
                        emap[(rn[i], rn[(i + 1) % len(rn)])] = fid
                    for w in ro:
                        s = vf.get(w)
                        if s is not None:
                            s.discard(fid)
                    for w in rn:
                        vf.setdefault(w, set()).add(fid)
                rings[A] = nA; rings[B] = nB; rings[C] = nC; rings[D] = nD
                pos[u] = nu; pos[v] = nv
                return (u, v)
    return None


# --------------------------------------------------------------------------------------------------
#  plexus operator
# --------------------------------------------------------------------------------------------------
@register_operator("edge_flip", set="vertex", kind="rewire", family="topology", title="swap neighbours",
                   equation=r"""$$\ell_e<\ell_{\mathrm{th}}\ \Rightarrow\ \text{reconnect: } (a,b)\ \longleftrightarrow\ (c,d)$$""")
class ReconnectT1_3D(Rewire):
    r"""the T1 transition: a reversible neighbour exchange, the ingredient that makes this a dynamic vertex model rather than a re-tessellated one.
    vertex -> vertex: reads pos, flips short interior edges and repositions their two vertices; vertex, edge and face counts are preserved.

        \ell_e<\ell_{\mathrm{th}}\ \Rightarrow\ \text{reconnect: } (a,b)\ \longleftrightarrow\ (c,d)
        \ell_e         the edge length
        \ell_{th}      `l_th`, the threshold length in world units (or `l_th_frac` of the median edge)
        (a,b), (c,d)   the two cells that stop and the two that start being neighbours

    Reference: Okuda, S., Inoue, Y., Eiraku, M., Sasai, Y. & Adachi, T. (2013). Biomech. Model. Mechanobiol. 12:627-644."""
    PARAM_UNITS = {'l_th': 'length', 'l_th_frac': 'fraction', 'max_flips': 'count'}
    SYMBOLS = {'\\ell_e': 'the edge length', '\\ell_{th}': '`l_th`, the threshold length in world units (or `l_th_frac` of the median edge)', '(a,b), (c,d)': 'the two cells that stop and the two that start being neighbours'}
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["T1_transition", "reversible_network_reconnection", "intercalation",
                      "neighbour_exchange", "vertex_model", "vesicle", "surface"]
    PARAM_ROLES = {"l_th": "reconnection_threshold_length (absolute; 0 -> use l_th_frac)",
                   "l_th_frac": "threshold as fraction of the mean edge length",
                   "max_flips": "cap on reconnections per call", "every": "call period (ticks)", 'cell_set': 'the_cell_set'}

    # THE FLIP ITSELF, AS AN ATTRIBUTE, so a variant can widen which edges may flip without copying the
    # loop that chooses them (`edge_flip[boundary]` below). The default is `t1_flip_3d`, unchanged.
    _t1 = staticmethod(lambda *a, **k: t1_flip_3d(*a, **k))

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self._cat = params.get("cell_set")          # override only; the SET declares the pairing
        self.l_th = float(params.get("l_th", 0.0))           # absolute; <=0 -> l_th_frac x mean edge
        self.l_th_frac = float(params.get("l_th_frac", 0.15))
        self.max_flips = int(params.get("max_flips", 20))
        # (was `from mesh_ops import _engine_owns_clock`) -- same module now
        self.every = _engine_owns_clock(params); self._k = 0

    def forward(self, H, mask=None):
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        self._k += 1                    # monotonic tick only -- D1: the engine owns the period
        dev = lvl.state.device; dt = lvl.state.dtype
        Nv = int(m["Nv"])
        pos_np = lvl.get("pos")[:Nv].detach().cpu().numpy().astype(np.float64)
        es = m["E_srce"].detach().cpu().numpy(); et = m["E_trgt"].detach().cpu().numpy()
        ef = m["E_face"].detach().cpu().numpy(); nF = int(m["nF"])
        rings = rings_from_flat_3d(es, et, ef, nF)
        # READ OFF `mech`, NOT DECLARED AGAIN HERE. `cell_mechanics` stashes the constraint it is
        # solving under, so a spec says `plane_axis` once and every topology operator that moves a
        # vertex honours the same plane. Repeating it per operator is how three of them end up
        # disagreeing about which surface the tissue lives on.
        _plane = (m.get("mech") or {}).get("plane_axis")
        emap = _edge_face_map(rings); vf = _vertex_faces(rings)   # build the adjacency maps ONCE; t1_flip_3d
        #   updates them incrementally on each successful flip (was rebuilt O(F) per candidate = the hot spot)
        pos = [p.copy() for p in pos_np]
        length = np.linalg.norm(pos_np[et] - pos_np[es], axis=1)
        thr = self.l_th if self.l_th > 0 else self.l_th_frac * float(length.mean())
        order = np.argsort(length)                           # shortest interior edges first
        seen = set(); used = set(); ndone = 0                # `used` verts -> one flip per vertex / call
        moved: set = set()               # the vertices a flip repositioned: their `sep` is re-aimed below
        for k in order:
            if ndone >= self.max_flips or length[k] >= thr:
                break
            a, b = int(es[k]), int(et[k]); key = (min(a, b), max(a, b))
            if key in seen or a in used or b in used:
                continue
            seen.add(key)
            if self._t1(rings, pos, (a, b), new_len=thr, emap=emap, vf=vf,
                          plane_axis=_plane) is not None:
                used.add(a); used.add(b); ndone += 1
                moved.add(a); moved.add(b)
        # TRACED PER FRAME, because the RATE is the observable that separates a length-keyed myosin
        # feedback from a tension-keyed one. A length feedback homogenises junction lengths and should
        # SUPPRESS T1s; a tension feedback is the destabilising one the germband-extension literature
        # describes and should PRODUCE them. Both move junction-length CV the same way, so CV cannot
        # tell them apart -- this can. `n_t1` was already accumulated; only the per-frame series was
        # missing, and without it the discriminating measurement was unavailable.
        T1_TRACE.append((int(getattr(H, "frame", -1) or -1), ndone, int(m.get("n_t1", 0)) + ndone,
                         int(nF)))
        if ndone == 0:
            return {}
        es2, et2, ef2, nF2, keep = flat_from_rings_3d(rings)
        # "T1 DROPS NO FACE" WAS AN ASSUMPTION, NOT AN INVARIANT, and it held only while nothing
        # shrank a cell to nothing. `flat_from_rings_3d` discards any ring that falls below three
        # vertices; a flip on a cell that is already a triangle can do exactly that. This line
        # threw `keep` away, so when it happened T1 lost a face and left EVERY per-face array --
        # A0, P0, V0f, Vbirth, divjit, age, ndiv, alive -- shifted by one against the mesh.
        #
        # Found by cell_die, which is the first operator that shrinks a cell far enough to
        # reach the case: marking ONE cell for death produced three deaths at ticks 86, 164 and
        # 236, each reporting exactly one marked cell, because the shift kept handing the flag to
        # a new victim. The control with T1 and no apoptosis holds at 2000 cells, which is why
        # this went eighteen rounds unseen.
        if nF2 != nF:
            print(f"[edge_flip] a flip left {nF - nF2} face(s) below three sides -- "
                  f"reindexing {nF} -> {nF2}. Per-face targets follow the mesh; a cell was lost "
                  f"to topology, not to biology.", flush=True)
            # `Vbirth` and `divjit` are gone from this list for the reason `cell_die`'s note
            # gives: they are cell-set blocks and `renumber_set` below already permutes them.
            for _nm in ("A0", "P0", "V0f", "age", "ndiv", "alive"):
                if _nm in m:
                    _a = m[_nm].detach().cpu().numpy()
                    m[_nm] = torch.as_tensor(np.asarray([_a[i] for i in keep]),
                                             dtype=m[_nm].dtype, device=m[_nm].device)
            # ONLY WHILE IT IS STILL A MESH COLUMN. Declared on the cell set it is permuted by
            # `H.renumber_set` below, with the same `keep`; carrying it here as well would permute
            # it twice.
            if isinstance(dict.get(m, "apop_flag", None), np.ndarray):
                m["apop_flag"] = np.asarray([m["apop_flag"][i] for i in keep], np.float64)
            # AND THE OPEN NAMES, which this branch has never carried. The tuple above is the
            # CLOSED list every topology operator knows; `face_carry` is the open one an operator
            # declares for itself, and `cell_divide` and `cell_die` have routed it through
            # `_carry_face_state` since it existed. This one did not, so a flip that lost a face
            # left `medioapical_myosin`'s per-face density indexed against faces that had moved --
            # the same defect class as per-half-edge myosin before `junction_sync`, one level up.
            #
            # UNREACHABLE IN THE GATED RUN, and asserted so: this branch needs `nF2 != nF`, which
            # `_ring_ok` refuses to produce (it rejects any new ring below three sides). The fix is
            # covered by `tools/test_mesh_carry.py` instead, which constructs the condition.
            _carry_face_state(m, keep, m["A0"].dtype if "A0" in m else torch.float32,
                                        m["E_srce"].device)
            # THE CELL STATE AND THE PENDING DELTAS FOLLOW TOO, and leaving them behind is the
            # same defect cell_die had: `chem` is indexed by face, and the engine zeroes its
            # delta accumulator once per TICK and integrates at the END of the schedule, so
            # cell_chem_diffuse and cell_chem_react deposit per-cell deltas that are applied after this runs.
            # Renumber the faces without renumbering those and every activator value, and every
            # pending flux, lands on a different cell.
            #
            # It stayed hidden because a flip only drops a face when a cell is ALREADY a triangle,
            # which needs something to have shrunk it -- so nothing reached this branch until
            # apoptosis existed. It then fails immediately and enormously -- the activator goes
            # many orders of magnitude negative when division and death run in the same tick,
            # where the same operator on a non-growing tissue holds it at exactly zero.
            # edge_flip declares no `cell_set`, so the cell level is looked up by name and
            # a model without one simply skips this -- H.level raises rather than returning None.
            # ONE ENGINE CALL, for the same reason `cell_die` makes it: state, occupancy, the
            # coordinate delta accumulator and the extra first-order block deltas are four stores
            # the engine keeps per set, and a renumber that forgets one scrambles it silently. The
            # version this replaces forgot the fourth -- its guard tested `isinstance(key, tuple)`
            # while `_delta_blocks` is keyed by level NAME, so the branch could never run.
            #
            # `edge_flip` declares no `cell_set`, so the cell level is looked up by name and a model
            # without one simply skips: `renumber_set` returns False rather than raising.
            # NOT `getattr(self, "cat", None) or "cell"`. This operator has no `cat`, so that
            # expression WAS the literal `"cell"` every time -- `edge_flip` renumbering a set it
            # never declared it needed, which is the case `resolve_cell_set`'s note names.
            _cat = resolve_cell_set(H, self.at, self._cat)
            if not H.renumber_set(_cat, keep, n_new=nF2):
                m["renumber_failed"] = int(m.get("renumber_failed", 0)) + 1
                print(f"[edge_flip] renumber_set({_cat!r}) DID NOT ACT after a face drop -- the "
                      f"cell state is now mis-indexed against the mesh", flush=True)
        m["E_srce"] = torch.as_tensor(es2, device=dev)
        m["E_trgt"] = torch.as_tensor(et2, device=dev)
        m["E_face"] = torch.as_tensor(ef2, device=dev)
        m["nF"] = nF2
        px0, px1 = lvl.state_schema["pos"]                   # write the two moved verts back (like cell_divide)
        st = lvl.state.clone()
        st[:Nv, px0:px1] = torch.as_tensor(np.asarray(pos), dtype=dt, device=dev)
        # A FLIP MOVES TWO VERTICES AND MUST RE-AIM THEIR THICKNESS VECTORS (R3 of
        # notes/size_cycle/SIZE_CYCLE_PLAN.md, finding 16). `t1_flip_3d` places `u` and `v` on the
        # rotated junction and back on the shell; their `sep` stayed what it was at the old
        # positions, so after every flip two prisms carried a thickness vector aimed for somewhere
        # else -- a trapezoid apiece, which the mechanics never straightened. Measured on the
        # sweep's base arm: trapezoid fraction 0 at frame 0, 0.105 by frame 10 with 23 flips and
        # not one division, climbing with the flip count for the whole run. Each moved vertex
        # keeps its thickness and takes the direction of its ring neighbours' vectors.
        if moved and "sep" in lvl.state_schema:
            sx0, sx1 = lvl.state_schema["sep"]
            sep = st[:Nv, sx0:sx1].detach().cpu().numpy().astype(np.float64)
            es_np = es if isinstance(es, np.ndarray) else np.asarray(es)
            et_np = et if isinstance(et, np.ndarray) else np.asarray(et)
            for w in moved:
                nb = et_np[es_np == w]
                nb = nb[nb != w]
                if nb.size == 0:
                    continue
                d = sep[nb].mean(axis=0); nrm = float(np.linalg.norm(d))
                if nrm < 1e-12:
                    continue
                sep[w] = d / nrm * float(np.linalg.norm(sep[w]))
            st[:Nv, sx0:sx1] = torch.as_tensor(sep, dtype=dt, device=dev)
        lvl.state = st
        m["n_t1"] = int(m.get("n_t1", 0)) + ndone
        return {}


def _commit_rings(rings, emap, vf, old_map, new_map):
    """Write the changed faces back and keep the caller's edge->face and vertex->faces maps in sync."""
    for fid, ro in old_map.items():
        rn = new_map[fid]
        for i in range(len(ro)):
            emap.pop((ro[i], ro[(i + 1) % len(ro)]), None)
        for w in ro:
            s_ = vf.get(w)
            if s_ is not None:
                s_.discard(fid)
    for fid, rn in new_map.items():
        for i in range(len(rn)):
            emap[(rn[i], rn[(i + 1) % len(rn)])] = fid
        for w in rn:
            vf.setdefault(w, set()).add(fid)
        rings[fid] = rn


def _mates_kept(emap, old_map, new_map):
    """No directed edge repeats in the new patch, and every edge the patch shared with a face OUTSIDE it
    is still there -- the closed-surface test without its "rim unchanged" half, which a boundary T1
    changes by design."""
    b_new = _boundary_de(new_map)
    b_old = _boundary_de(old_map)
    if b_new is None or b_old is None:
        return False
    patch = set(old_map)
    for (x, y) in b_old:
        f = emap.get((y, x))
        if f is not None and f not in patch and (x, y) not in b_new:
            return False
    return True


def t1_flip_boundary(rings, pos, e_uv, new_len=None, emap=None, vf=None, plane_axis=None):
    """A T1 on an OPEN tissue: the interior T1 where it applies, and the two moves at the free rim that
    `t1_flip_3d` refuses (a rim vertex has two faces, an interior T1 needs three).

    SPOKE (an edge u-v from an interior vertex u to a rim vertex v, between rim cells A and B, with C the
    third cell at u): A and B stop touching, C reaches the rim between them --
        A loses v,  B loses u,  C gains v just before u.
    This is the interior T1 with the outside medium as its fourth cell.
    RIM (an edge v->u of cell C on the rim, u shared with A, v with B): the inverse -- C leaves the rim
    there, A and B meet --
        C loses v,  A gains v just after u,  B gains u just after v.
    The geometry is the interior T1's: u and v collapse to their midpoint and reopen perpendicular to the
    old edge, at `new_len`, both signs tried, kept only if every changed face stays simple and keeps its
    orientation. Returns (u, v) on success, None when nothing valid exists."""
    r = t1_flip_3d(rings, pos, e_uv, new_len=new_len, emap=emap, vf=vf, plane_axis=plane_axis)
    if r is not None:
        return r
    if emap is None:
        emap = _edge_face_map(rings)
    if vf is None:
        vf = _vertex_faces(rings)
    a0, b0 = int(e_uv[0]), int(e_uv[1])
    fa, fb = emap.get((a0, b0)), emap.get((b0, a0))
    new_map = old_map = None
    if fa is not None and fb is not None and fa != fb:                          # SPOKE candidate
        ca, cb = vf.get(a0, set()) - {fa, fb}, vf.get(b0, set()) - {fa, fb}
        if len(ca) == 1 and len(cb) == 0 and vf.get(b0, set()) == {fa, fb}:
            u, v = a0, b0
        elif len(cb) == 1 and len(ca) == 0 and vf.get(a0, set()) == {fa, fb}:
            u, v = b0, a0
        else:
            return None
        A, B = emap.get((u, v)), emap.get((v, u))
        C = (vf[u] - {A, B}).pop()
        old_map = {A: rings[A], B: rings[B], C: rings[C]}
        new_map = {A: [w for w in rings[A] if w != v], B: [w for w in rings[B] if w != u],
                   C: _insert_before(rings[C], u, v)}
    elif (fa is None) != (fb is None):                                          # RIM candidate
        v, u = (a0, b0) if fa is not None else (b0, a0)                         # C holds v -> u
        C = emap[(v, u)]
        Au, Bv = vf.get(u, set()) - {C}, vf.get(v, set()) - {C}
        if len(Au) != 1 or len(Bv) != 1:
            return None
        A, B = Au.pop(), Bv.pop()
        if A == B:
            return None
        old_map = {A: rings[A], B: rings[B], C: rings[C]}
        new_map = {A: _insert_after(rings[A], u, v), B: _insert_after(rings[B], v, u),
                   C: [w for w in rings[C] if w != v]}
    else:
        return None
    if not all(_ring_ok(rn) for rn in new_map.values()) or not _mates_kept(emap, old_map, new_map):
        return None
    pu = np.asarray(pos[u], float); pv = np.asarray(pos[v], float)
    mid = 0.5 * (pu + pv); d = pv - pu; L = np.linalg.norm(d)
    if L < 1e-12:
        return None
    if plane_axis is None:
        rmid = np.linalg.norm(mid)
        if rmid < 1e-9:
            return None
        n = mid / rmid
    else:
        n = np.zeros(3); n[int(plane_axis)] = 1.0
    perp = np.cross(n, d / L); pn = np.linalg.norm(perp)
    if pn < 1e-9:
        return None
    perp = perp / pn
    half = 0.5 * (new_len if new_len is not None else L)
    for sign in (+1.0, -1.0):
        nu = mid - sign * perp * half; nv = mid + sign * perp * half
        if plane_axis is not None:
            nu[int(plane_axis)] = mid[int(plane_axis)]; nv[int(plane_axis)] = mid[int(plane_axis)]
        getp = lambda i: (nu if i == u else nv if i == v else pos[i])          # noqa: E731
        if all(_face_ok_3d(rn, getp, normal=None if plane_axis is None else n) for rn in new_map.values()):
            _commit_rings(rings, emap, vf, old_map, new_map)
            pos[u] = nu; pos[v] = nv
            return (u, v)
    return None


@register_operator("edge_flip", model="boundary", set="vertex", kind="rewire", family="topology",
                   title="swap neighbours, tissue edge included", equation=r"""$$\ell_e<\ell_{\mathrm{th}}\ \Rightarrow\ \text{reconnect, the medium as the fourth cell at the rim}$$""")
class ReconnectT1Boundary(ReconnectT1_3D):
    r"""`boundary` implementation of edge_flip: the interior T1 plus the two moves at the free rim of an open tissue.
    vertex -> vertex: as edge_flip, plus `t1_flip_boundary` so a cell can reach the boundary or leave it.

        \ell_e<\ell_{\mathrm{th}}\ \Rightarrow\ \text{reconnect, the medium as the fourth cell at the rim}
        \ell_e         the edge length
        \ell_{th}      `l_th`

    Reference: Okuda, S. et al. (2013). Biomech. Model. Mechanobiol. 12:627-644; the rim moves are the same reconnection with the outside medium as the fourth cell."""
    PARAM_UNITS = {}
    SYMBOLS = {'\\ell_e': 'the edge length', '\\ell_{th}': '`l_th`'}
    _t1 = staticmethod(t1_flip_boundary)


# --------------------------------------------------------------------------------------------------
#  standalone self-test (no engine): build a jittered vesicle, run many T1 flips, assert it stays
#  CLOSED with Euler=2 and that a T1 keeps V,E,F constant.
# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # (was `from mesh_ops import build_sphere_mesh`) -- same module now

    verts, es, et, ef, nF = build_sphere_mesh(150, 5.0, 0.15, 0)
    rings = rings_from_flat_3d(es, et, ef, nF)
    pos = [v.copy() for v in verts]
    ok0, V0, E0, F0, eu0 = _check_closed(rings)
    print(f"start:  closed={ok0} V={V0} E={E0} F={F0} euler={eu0}")

    def undirected_edges(rr):
        s = set()
        for r in rr:
            if r is None or len(r) < 3:
                continue
            k = len(r)
            for i in range(k):
                a, b = r[i], r[(i + 1) % k]; s.add((min(a, b), max(a, b)))
        return s

    ndone = 0
    for sweep in range(6):                                   # several sweeps: each flip reopens the edge
        E = sorted(undirected_edges(rings),                  #   at ~mean length so it exits the short set
                   key=lambda ab: np.linalg.norm(pos[ab[1]] - pos[ab[0]]))
        me = float(np.mean([np.linalg.norm(pos[b] - pos[a]) for a, b in E]))
        thr = 0.9 * me                                       # target the shorter ~half of the edges
        used = set(); fired = 0
        for (a, b) in E:
            if np.linalg.norm(pos[b] - pos[a]) >= thr:
                break
            if a in used or b in used:
                continue
            if t1_flip_3d(rings, pos, (a, b), new_len=me) is not None:  # reopen at ~mean length
                used.add(a); used.add(b); fired += 1
        ndone += fired
        ok, V, E_, F, eu = _check_closed(rings)
        print(f"sweep {sweep}: fired {fired:3d}  closed={ok} V={V} E={E_} F={F} euler={eu}")
        assert ok and eu == 2, "mesh broke the closed-surface invariant"

    ok, V, E, F, eu = _check_closed(rings)
    print(f"after {ndone} T1 flips:  closed={ok} V={V} E={E} F={F} euler={eu}  (want euler=2)")
    assert ok and eu == 2
    assert (V, E, F) == (V0, E0, F0), "a T1 must keep V,E,F constant"    # dV=dE=dF=0
    es2, et2, ef2, nF2, keep = flat_from_rings_3d(rings)
    print(f"rebuilt flat: nF={nF2} half-edges={len(es2)} (each real edge twice -> {len(es2)//2} edges)")
    assert ndone >= 20, "expected a meaningful number of flips"
    print(f"SELF-TEST OK  ({ndone} flips, closed, euler=2, V/E/F unchanged)")


def apical_basal_shells(pos, es, et, ef, nF, h_cell):
    """Apical (outer) and basal (inner) vertex positions a_i, b_i = x_i +/- (H_i/2) n_i, for RENDERING
    the monolayer as two offset shells with a visible thickness. Same offset the energy uses."""
    dev, dt = pos.device, pos.dtype
    Nv = pos.shape[0]
    s, t = pos[es], pos[et]
    Nf = 0.5 * torch.zeros(nF, 3, device=dev, dtype=dt).index_add(0, ef, torch.cross(s, t, dim=-1))
    vn = torch.zeros(Nv, 3, device=dev, dtype=dt).index_add(0, es, Nf[ef])
    n = vn / (vn.norm(dim=-1, keepdim=True) + 1e-12)
    cnt = torch.zeros(Nv, device=dev, dtype=dt).index_add(0, es, torch.ones(es.shape[0], device=dev, dtype=dt))
    hv = torch.zeros(Nv, device=dev, dtype=dt).index_add(0, es, h_cell[ef]) / cnt.clamp(min=1e-9)
    return pos + 0.5 * hv[:, None] * n, pos - 0.5 * hv[:, None] * n


def monolayer_shells(pos, es, et, ef, nF, h_cell, eocc=None):
    """The apical and basal surfaces of a monolayer, from its MID-surface and a per-cell thickness.

    ONE DEFINITION, TWO CALLERS. `monolayer_geometry_3d` needs these to build the prism the energy
    is written on, and the renderer needs them to draw the epithelium as something with a thickness
    rather than as the mid-surface it stores. Two copies of "offset along the vertex normal by h/2"
    would agree until one of them changed, and the picture would then stop being a picture of the
    model -- so the formula lives here and both call it.

    The normal at a vertex is the normalised sum of the area vectors of its incident faces, and the
    thickness at a vertex is the mean over its incident cells; the shell straddles the mid-surface
    symmetrically, apical OUTWARD. Returns (apical, basal, vertex_normal, vertex_thickness).
    """
    dev, dt = pos.device, pos.dtype
    Nv = pos.shape[0]
    s, t = pos[es], pos[et]
    ones_e = torch.ones(es.shape[0], device=dev, dtype=dt) if eocc is None else eocc
    # mid-surface face area vectors -> vertex normals (sum of incident face area vectors)
    crossm = torch.cross(s, t, dim=-1) * ones_e[:, None]
    Nf = 0.5 * torch.zeros(nF, 3, device=dev, dtype=dt).index_add(0, ef, crossm)
    vn = torch.zeros(Nv, 3, device=dev, dtype=dt).index_add(0, es, Nf[ef] * ones_e[:, None])
    n = vn / (vn.norm(dim=-1, keepdim=True) + 1e-12)
    # thickness at a vertex = mean thickness of incident cells
    cnt_v = torch.zeros(Nv, device=dev, dtype=dt).index_add(0, es, ones_e)
    hv = torch.zeros(Nv, device=dev, dtype=dt).index_add(0, es, h_cell[ef] * ones_e) / cnt_v.clamp(min=1e-9)
    return pos + 0.5 * hv[:, None] * n, pos - 0.5 * hv[:, None] * n, n, hv


def monolayer_geometry_3d(pos, es, et, ef, nF, h_cell, eocc=None):
    """Per-cell prism volume v_f and surface s_f (apical+basal+lateral), plus the apical/basal cap areas.
    All differentiable in `pos`. h_cell is per-cell thickness [nF]. eocc masks dead half-edges (or None)."""
    dev, dt = pos.device, pos.dtype
    ones_e = torch.ones(es.shape[0], device=dev, dtype=dt) if eocc is None else eocc
    crossm = torch.cross(pos[es], pos[et], dim=-1) * ones_e[:, None]
    Nf = 0.5 * torch.zeros(nF, 3, device=dev, dtype=dt).index_add(0, ef, crossm)
    a, b, _, _ = monolayer_shells(pos, es, et, ef, nF, h_cell, eocc)
    a_s, a_t, b_s, b_t = a[es], a[et], b[es], b[et]
    # cell VOLUME = mid-surface area x thickness (v_j = A_mid*h_j). Exact for a flat cell, first-order in
    # curvature (the O((h/R)^2) prism correction ~0.3% is dropped); ALWAYS positive & differentiable, and
    # -- the key point -- bending resistance comes from the SURFACE term below (apical!=basal area under
    # curvature), NOT from the volume, so A_mid*h is the physically correct choice, not just the simple one.
    v_f = Nf.norm(dim=-1) * h_cell
    # apical / basal cap areas (Newell magnitude, origin-independent)
    Na = 0.5 * torch.zeros(nF, 3, device=dev, dtype=dt).index_add(0, ef, torch.cross(a_s, a_t, dim=-1) * ones_e[:, None])
    Nb = 0.5 * torch.zeros(nF, 3, device=dev, dtype=dt).index_add(0, ef, torch.cross(b_s, b_t, dim=-1) * ones_e[:, None])
    A_ap, A_ba = Na.norm(dim=-1), Nb.norm(dim=-1)
    # lateral quad area per edge = tri(a_s,a_t,b_t) + tri(a_s,b_t,b_s)
    la = (0.5 * torch.cross(a_t - a_s, b_t - a_s, dim=-1).norm(dim=-1)
          + 0.5 * torch.cross(b_t - a_s, b_s - a_s, dim=-1).norm(dim=-1)) * ones_e
    A_lat = torch.zeros(nF, device=dev, dtype=dt).index_add(0, ef, la)
    s_f = A_ap + A_ba + A_lat
    return v_f, s_f, A_ap, A_ba


def isolated_split(pool, flag, es, et, ef, nF, cap):
    """Walk `pool` (face indices, queue order) and keep, up to `cap`, the faces with no neighbour already under
    sentence (`flag` > 0) or kept earlier in this walk; return (kept, the rest in their order). Neighbours are read
    from the half-edge twins (s -> t on face f, t -> s on its neighbour)."""
    es = np.asarray(es.cpu() if hasattr(es, "cpu") else es, np.int64)
    et = np.asarray(et.cpu() if hasattr(et, "cpu") else et, np.int64)
    ef = np.asarray(ef.cpu() if hasattr(ef, "cpu") else ef, np.int64)
    ok = ef < nF
    es, et, ef = es[ok], et[ok], ef[ok]
    face_of = {(int(a), int(b)): int(f) for a, b, f in zip(es, et, ef)}
    want = set(int(f) for f in pool)
    nbrs = {f: set() for f in want}
    for a, b, f in zip(es, et, ef):
        f = int(f)
        if f in want:
            g = face_of.get((int(b), int(a)))
            if g is not None and g != f:
                nbrs[f].add(g)
    fl = np.asarray(flag)
    busy = set(int(g) for g in np.flatnonzero(fl[:nF] > 0))
    kept, rest = [], []
    for f in pool:
        f = int(f)
        if len(kept) < cap and not (nbrs.get(f, set()) & busy):
            kept.append(f); busy.add(f)
        else:
            rest.append(f)
    return kept, rest


def queued_hold(cell_id, waiting, area=None, escape=0.0):
    """Per face, True where the cell waits in `cell_die[t2]`'s leaving queue (`waiting`, cell ids) and is therefore
    held back from dividing in place (`cell_divide` `skip_queued`) -- unless `escape` > 0 and its `area` exceeds
    escape x the median area of all faces (the size checkpoint, `size_escape`)."""
    ci = np.rint(np.asarray(cell_id, np.float64)).astype(np.int64)
    hold = np.isin(ci, np.asarray(sorted(waiting), np.int64))
    if escape > 0 and area is not None:
        a = np.asarray(area, np.float64).reshape(-1)[: ci.size]
        hold &= ~(a > escape * float(np.median(a)))
    return hold


def apicobasal_geometry_3d(pos, sep, es, et, ef, nF, eocc=None):
    """Per-cell TRUE polyhedron volume `v_f` and surface `s_f` from the DOUBLED degree-of-freedom set.

    The twin of `monolayer_geometry_3d`, and the difference between them is the whole promotion.
    There the cell's two surfaces are `pos +/- (h/2) n` -- a kinematic identity that ties them to the
    mid-surface and to each other -- and the volume is `A_mid * h`, the flat-cell answer with the
    `O((h/R)^2)` prism correction dropped. Here apical `= pos + sep` and basal `= pos - sep` with
    `sep` a free per-vertex vector, and the volume is the volume OF THE POLYHEDRON:

        V_f = (1/6) sum over the cell's triangles of (p-o) . ((q-o) x (r-o))     [divergence theorem]

          apical cap    the ring on `a`, fanned from the cap's own centroid, outward winding
          basal cap     the ring on `b`, REVERSED, because it faces the other way
          lateral wall  one quad per ring edge, (a_s, b_s, b_t) + (a_s, b_t, a_t)

    THE WINDING IS THE SAME AS `gate_measures._cell_polyhedron_volume`, term for term, and that is
    deliberate: a gate row that disagreed with the energy would be measuring a different solid than
    the one the tissue relaxed. The measure was written first and caught its own wall wound INWARD
    -- a hexagonal prism of +2.598 reading -0.866, with the closure row still green, because an
    inward-wound wall is still closed. Both now read (a_s, b_s, b_t), (a_s, b_t, a_t).

    `o` IS THE CELL'S OWN MID-RING CENTROID, NOT THE WORLD ORIGIN. The volume of a closed surface is
    origin-independent in exact arithmetic and this state is float32, so the choice is a numerical
    one and it is the load-bearing kind: about the world origin every term is `O(R^3)` and they
    cancel down to `O(h*A)`, losing three or four significant figures on a shell at R = 5. About the
    cell's own centroid every term is already the size of the answer.

    Returns (v_f, s_f, A_ap, A_ba). All differentiable in BOTH `pos` and `sep`.
    """
    dev, dt = pos.device, pos.dtype
    ones_e = torch.ones(es.shape[0], device=dev, dtype=dt) if eocc is None else eocc
    a, b = pos + sep, pos - sep
    a_s, a_t, b_s, b_t = a[es], a[et], b[es], b[et]
    z3 = lambda: torch.zeros(nF, 3, device=dev, dtype=dt)                        # noqa: E731
    cnt = torch.zeros(nF, device=dev, dtype=dt).index_add(0, ef, ones_e).clamp(min=1e-9)
    centroid = z3().index_add(0, ef, pos[es] * ones_e[:, None]) / cnt[:, None]        # mid-ring centroid
    ca = z3().index_add(0, ef, a_s * ones_e[:, None]) / cnt[:, None]             # apical cap centroid
    cb = z3().index_add(0, ef, b_s * ones_e[:, None]) / cnt[:, None]             # basal cap centroid
    o = centroid[ef]

    def tri(p, q, r):
        return torch.einsum("ij,ij->i", p - o, torch.cross(q - o, r - o, dim=-1)) * ones_e

    v6 = torch.zeros(nF, device=dev, dtype=dt)
    v6 = v6.index_add(0, ef, tri(ca[ef], a_s, a_t))                              # apical cap
    v6 = v6.index_add(0, ef, tri(cb[ef], b_t, b_s))                              # basal cap, reversed
    v6 = v6.index_add(0, ef, tri(a_s, b_s, b_t))                                 # wall, triangle 1
    v6 = v6.index_add(0, ef, tri(a_s, b_t, a_t))                                 # wall, triangle 2
    # THE MAGNITUDE, NOT THE SIGNED SUM. The triangulation is oriented with the apical cap
    # outward and the basal cap reversed, so a tissue seeded apical-in (`seed_mesh apical: in`,
    # the polarity of a cyst in a matrix) returned every volume NEGATIVE (-1.3515 against +1.3515
    # on the reference spheroid), the volume term pulled each cell toward a volume of the wrong
    # sign and the shell relaxed to r 4.85 instead of 4.53. The polarity is the TISSUE's, so the
    # sign is taken once, from the sum over the tissue, and applied to every cell: a single
    # inverted cell (an extruding one in the apoptosis runs) keeps its own negative volume, as
    # it always did, and the K_V term keeps seeing it. Bit-identical for every apical-out run
    # (multiplying by +1.0) and the whole tissue flips for an apical-in one.
    v_f = v6 / 6.0
    _sgn = torch.sign(v_f.detach().sum())
    v_f = v_f * torch.where(_sgn == 0, torch.ones_like(_sgn), _sgn)
    # THE CAP'S AREA IS MEASURED ON THE SURFACE ITS VOLUME IS MEASURED ON -- the SAME centroid fan,
    # summed as true triangle areas, NOT the Newell magnitude ||1/2 sum a_s x a_t||. The Newell
    # form makes the caps the same arithmetic as the monolayer's on a right prism, so AB-C1 comes
    # out an identity rather than a near miss -- but IT BUYS THAT IDENTITY WITH A NULL SPACE, and
    # the null space is the whole failure mode of the free separation.
    #
    # The Newell magnitude is the area of the ring's PLANAR PROJECTION, so a crumpled cap and a flat
    # one of the same outline measure the SAME, and the area is stationary to first order when one
    # vertex moves normal to the cap. The volume, meanwhile, was already the fan -- so the energy was
    # pricing one surface and enclosing another. With nothing but the lateral wall resisting, `sep`
    # relaxes downhill into a checkerboard: measured on `gate_ab_thickshell` extended to 80 frames
    # at sep_mu 1, neighbouring vertices anti-correlate across an edge (Pearson r -0.42 by frame 20),
    # the thickness varies five times more WITHIN a cell's ring (cv 1.19) than between cells (0.24),
    # the median thickness collapses 1.428 -> 0.237 while a tail reaches 7.47, and spans start
    # inverting at frame 40. Quartering the step to eta 0.02 over 120 iterations reproduces it and
    # collapses slightly faster, so it is this energy's own descent direction and not the stepping.
    #
    # THE IDENTITIES SURVIVE, WHICH IS WHY THIS IS A FIX AND NOT A CHANGE OF MODEL. For a planar
    # convex ring fanned from its own centroid the triangle areas sum to the polygon area exactly, so
    # AB-C1 (flat patch, caps exactly planar) and AB-C2 (regular hexagonal prism) are unchanged to
    # the last bit. What moves is any curved shell, where the ring is not planar and the cap's true
    # area is the larger of the two -- which is the case the null space lived in.
    A_ap = torch.zeros(nF, device=dev, dtype=dt).index_add(
        0, ef, 0.5 * torch.cross(a_s - ca[ef], a_t - ca[ef], dim=-1).norm(dim=-1) * ones_e)
    A_ba = torch.zeros(nF, device=dev, dtype=dt).index_add(
        0, ef, 0.5 * torch.cross(b_s - cb[ef], b_t - cb[ef], dim=-1).norm(dim=-1) * ones_e)
    la = (0.5 * torch.cross(a_t - a_s, b_t - a_s, dim=-1).norm(dim=-1)
          + 0.5 * torch.cross(b_t - a_s, b_s - a_s, dim=-1).norm(dim=-1)) * ones_e
    A_lat = torch.zeros(nF, device=dev, dtype=dt).index_add(0, ef, la)
    return v_f, A_ap + A_ba + A_lat, A_ap, A_ba


def _apicobasal_energy_core(pos, sep, es, et, ef, nF, V_eq, alive, k_v, kappa_s, Lam, K_R, R0,
                            eocc, vocc, gamma=0.0, surface="apical", kappa_h=0.0, lam_e=None):
    """The monolayer's energy on the polyhedron: same functional, different geometry.

        U = sum_j [ 1/2 k_v (V_j - V_eq_j)^2 + kappa_s S_j + 1/2 gamma P_j^2 ]
            + Lam sum_e l_e + K_R sum_i (|x_i| - R0)^2

    THE FUNCTIONAL IS UNCHANGED ON PURPOSE. The hypothesis under test is that a cell is a polyhedron
    with two independent caps, not that it obeys some new constitutive law -- so every coefficient
    keeps its meaning and a spec can be moved between the two `model:` values without retuning. If
    this energy also changed form, a difference between the arms would have two possible causes and
    would settle neither.

    `surface:` -- WHICH RING THE PERIMETER, THE LINE TENSION AND THE RADIAL SPRING ACT ON, and the
    default is `apical` BECAUSE THE MID-SURFACE IS NOT A MEMBRANE. `pos` is a coordinate choice, not
    an object: the design sets apical = pos + sep and basal = pos - sep, so `pos` is IDENTICALLY the
    midpoint of the two caps (measured on gate_ab_curved: max |pos - (apical+basal)/2| = 0.000e+00
    over every vertex) and carries no information the caps do not. It is kept as the coordinate
    block for compatibility -- it is the surface the incumbent `monolayer` model integrates, which
    is what makes the R3 reduction a bit-level identity rather than a comparison -- and that is a
    statement about the SOLVER, not about the cell.

    So writing `P_j`, the line tension and the radial spring on it put three forces on the average
    of two membranes, which is an object no epithelium has: a junctional belt in this model ran
    along a surface that is not there. `surface: apical` puts them on a real one. `mid` is kept as
    a value so a run made before this change can be reproduced exactly, and `basal` because which
    surface a belt is on is a claim about the tissue and belongs in the spec.

    WHAT THIS DOES AND DOES NOT MOVE. On a FLAT patch with a uniform frozen `sep` the apical ring is
    the mid ring translated by one constant vector, so every edge length and every perimeter is
    unchanged and AB-C1 stays exact to the bit. On a curved shell the apical ring is larger by the
    offset, so any spec with a non-zero `gamma` or `Lambda` moves -- which is the point: it was
    being evaluated in the wrong place. `K_R` is a declared suppression either way
    (`graphs_data/mesh_mpm/README.md`) and every apicobasal gate runs it at 0.

    The volume and the total surface `S_j` are untouched: they are properties of the whole
    polyhedron and were never mid-surface quantities.
    """
    v_f, s_f, _, _ = apicobasal_geometry_3d(pos, sep, es, et, ef, nF, eocc)
    E = (0.5 * k_v * (v_f - V_eq) ** 2 * alive).sum() + kappa_s * (s_f * alive).sum()
    ring = {"apical": pos + sep, "basal": pos - sep, "mid": pos}[surface]
    if gamma != 0.0:
        perim = torch.zeros(nF, device=pos.device, dtype=pos.dtype).index_add(
            0, ef, (ring[et] - ring[es]).norm(dim=-1) * eocc)
        E = E + 0.5 * gamma * (perim ** 2 * alive).sum()
    if Lam != 0.0:
        # `lam_e` -- A PER-JUNCTION MULTIPLIER on the line tension, the channel the mid-surface
        # energy has always read (`m["myo"]`, written by `junction_myosin`). Absent, the term is
        # the scalar form exactly. It is what lets an in-surface ANISOTROPY -- circumferential
        # junctions pulling harder than meridional ones, the "molecular corset" -- act on the
        # apico-basal tissue (exp 5; M6 of notes/EPITHELIUM_BM_ECM_ROADMAP.md).
        _l = (ring[et] - ring[es]).norm(dim=-1)
        if lam_e is not None:
            _l = _l * lam_e
        E = E + Lam * (_l * eocc).sum()
    if K_R != 0.0:
        # THE RADIUS IS MEASURED FROM THE TISSUE'S OWN CENTRE, weighted by occupancy so dead
        # buffer rows do not pull it. |x| is the same thing only for a vesicle at the origin;
        # seeded at a wall box's centre the spring pulled every vertex toward the corner. See
        # ApicoBasalShapeEnergy3D._calibrate for the measurement.
        _w = vocc.to(ring.dtype).reshape(-1, 1)
        _c = (ring * _w).sum(dim=0, keepdim=True) / _w.sum().clamp(min=1.0)
        E = E + K_R * ((((ring - _c).norm(dim=1) - R0) ** 2) * vocc).sum()
    # `kappa_h` -- A STIFFNESS ON THE THICKNESS FIELD (R3d of notes/size_cycle/SIZE_CYCLE_PLAN.md;
    # `ab_sep_dirichlet` is its warp kernel, checked against this expression by
    # `tests/test_vertex_warp_apicobasal.py`).
    # Nothing above prefers a prism to a frustum: a cell that flares its apical cap and pinches its
    # basal one keeps its volume and nearly its surface, so every T1 kick and every septum walked
    # the caps apart and the mechanics never walked them back (half the cells trapezoids by frame
    # 400 on every dividing arm, finding 15). This is the Dirichlet energy of `sep` along the
    # ring edges, 1/2 kappa_h sum_e |sep_s - sep_t|^2: neighbouring thickness vectors want the
    # same length and the same direction, which is a prism, while a slow variation across the
    # shell -- its curvature -- costs almost nothing. kappa_h L^2 = F L, so kappa_h is a tension,
    # like kappa_s. 0 (the default) is byte-identical to every run before it.
    if kappa_h != 0.0:
        E = E + 0.5 * kappa_h * (((sep[es] - sep[et]) ** 2).sum(dim=-1) * eocc).sum()
    return E


def _monolayer_energy_core(pos, es, et, ef, nF, h_cell, V_eq, alive, k_v, kappa_s, Lam, K_R, R0, eocc, vocc, gamma=0.0):
    """U = sum_j [ 1/2 k_v (v_j - v_eq_j)^2 + kappa_s s_j + 1/2 gamma P_j^2 ] + Lam*sum_e l_e + K_R*sum_i (|x_i|-R0)^2 .
    gamma is a cortical CONTRACTILITY (perimeter^2) that rounds cells and resists shear -- a cell-shape
    regularizer standing in for the RNR/T1 remeshing Okuda relies on (without it the bare volume+surface
    energy shears/spikes under large deformation). Lam/K_R are optional dials (both default 0 in the op)."""
    v_f, s_f, _, _ = monolayer_geometry_3d(pos, es, et, ef, nF, h_cell, eocc)
    E = (0.5 * k_v * (v_f - V_eq) ** 2 * alive).sum() + kappa_s * (s_f * alive).sum()
    if gamma != 0.0:
        perim = torch.zeros(nF, device=pos.device, dtype=pos.dtype).index_add(0, ef, (pos[et] - pos[es]).norm(dim=-1) * eocc)
        E = E + 0.5 * gamma * (perim ** 2 * alive).sum()
    if Lam != 0.0:
        E = E + Lam * ((pos[et] - pos[es]).norm(dim=-1) * eocc).sum()
    if K_R != 0.0:
        # from the tissue's own centre, as in _apicobasal_energy_core above
        _w = vocc.to(pos.dtype).reshape(-1, 1)
        _c = (pos * _w).sum(dim=0, keepdim=True) / _w.sum().clamp(min=1.0)
        E = E + K_R * ((((pos - _c).norm(dim=1) - R0) ** 2) * vocc).sum()
    return E


@register_operator("cell_mechanics", model="monolayer", set="vertex", kind="lateral", family="mechanics", title="relax monolayer cell shapes", equation=r"""$$E=\sum_f\Big[\tfrac{k_v}{2}\big(V_f-V^0_f\big)^2+\kappa_s S_f\Big]+\tfrac\gamma2\sum_f P_f^2+\Lambda\sum_e\ell_e$$""")
class MonolayerShapeEnergy3D(Lateral):
    r"""`monolayer` model of cell_mechanics: the per-cell apical, basal and lateral surfaces of a sheet of thickness h0 built kinematically from the mid-surface.
    vertex -> vertex: reads pos and the targets; emits the vertex velocity -mu grad E on the mid-surface, the apical and basal surfaces at pos +/- (h0/2) n.

        E=\sum_f\Big[\tfrac{k_v}{2}\big(V_f-V^0_f\big)^2+\kappa_s S_f\Big]+\tfrac\gamma2\sum_f P_f^2+\Lambda\sum_e\ell_e
        V_f, V^0_f     cell f's prism volume and its target
        k_v            `k_v`, the volume stiffness
        S_f            the cell's total surface, apical + basal + lateral
        \kappa_s       `kappa_s`, the surface tension
        \gamma         `gamma`, the cortical contractility
        P_f, \ell_e    perimeter and edge length
        \Lambda        `Lambda`, the line tension

    Reference: Okuda, S. et al. (2018). Sci. Rep. 8:2386, Eq. 3."""
    PARAM_UNITS = {'k_v': 'stress', 'kappa_s': 'energy_per_area', 'h0': 'length', 'gamma': 'force', 'Lambda': 'force', 'K_R': 'stress', 'mu': 'dimensionless', 'dt': 'time', 'relax_iters': 'count', 'eta': 'dimensionless', 'cap_frac': 'fraction', 'mono_k': 'stress'}
    SYMBOLS = {'V_f, V^0_f': "cell f's prism volume and its target", 'k_v': '`k_v`, the volume stiffness', 'S_f': "the cell's total surface, apical + basal + lateral", '\\kappa_s': '`kappa_s`, the surface tension', '\\gamma': '`gamma`, the cortical contractility', 'P_f, \\ell_e': 'perimeter and edge length', '\\Lambda': '`Lambda`, the line tension'}
    SUPPORTED_DIMS = [3]; EMIT = "velocity"; DIFFERENTIABLE = True
    INPUTS = ["vertex"]; OUTPUTS = ["vertex"]; READS = ["pos"]; WRITES = ["pos"]
    MECHANISM_TAGS = ["vertex_model", "monolayer", "cell_3d_volume", "surface_tension", "emergent_bending", "force_balance"]
    REFERENCE = "Okuda, S. et al. (2018). Sci. Rep. 8:2386 (monolayer 3D vertex model, Eq. 3)."
    PARAM_ROLES = {"k_v": "cell_volume_elasticity", "kappa_s": "surface_tension", "h0": "cell_thickness", 'K_R': 'radial_restoring_stiffness', 'Lambda': 'line_tension', 'cap_frac': 'cap_on_the_vertex_step_as_a_fraction_of_the_edge_length', 'dt': 'relaxation_time_step', 'eta': 'step_size_of_the_relaxation', 'gamma': 'cortical_contractility', 'mono_k': 'stiffness_of_the_monolayer_regulariser', 'mu': 'mobility_of_the_vertices', 'plane_axis': 'normal_axis_of_a_flat_sheet', 'relax_iters': 'relaxation_iterations_per_frame', 'rest_calibration': 'how_the_rest_state_is_calibrated'}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.k_v = float(params.get("k_v", 4.0)); self.kappa_s = float(params.get("kappa_s", 0.2))
        self.h0 = float(params.get("h0", 0.4))                    # uniform cell thickness (v1: fixed field)
        self.gamma = float(params.get("gamma", 0.0))             # cortical contractility (cell-shape regularizer)
        self.Lambda = float(params.get("Lambda", 0.0)); self.K_R = float(params.get("K_R", 0.0))
        self.mu = float(params.get("mu", 1.0)); self.dt = float(params.get("dt", 1.0))
        self.relax_iters = int(params.get("relax_iters", 30)); self.eta = float(params.get("eta", 0.08))
        self.cap_frac = float(params.get("cap_frac", 0.12))
        # REST STATE. "force_balance" makes the seeded vesicle an equilibrium; "volume_only" is the
        # original behaviour, kept so the collapse can be reproduced deliberately. See _rest_offset.
        self.rest_calibration = str(params.get("rest_calibration", "force_balance"))
        # THE VOLUME TARGET'S SCALE, DECLARABLE INSTEAD OF INFERRED. `V_eq = mono_k * V0f` converts
        # the wedge target `cell_grow` scales into this model's cell volume, and `mono_k` is
        # normally calibrated once from the seeded mesh -- median rest volume over median wedge.
        # That is right for a run and wrong for a CONTROLLED COMPARISON: two models that agree on
        # the volume to float32 still calibrate to constants differing in the last bits, and at the
        # seeded state `v - V_eq` is a difference of two nearly equal numbers, so those last bits
        # are a large RELATIVE perturbation of the volume force exactly where it is smallest.
        # Measured on gate_ab_flat: 3e-7 in `mono_k` opened the two arms to 1.1e-2 of an edge within
        # one frame, decaying back to 1.2e-4 once the cells had left their targets. Declaring it
        # holds the calibration fixed ACROSS arms, which is what makes AB-C1 a comparison of two
        # energies rather than of two calibrations. Default None: every existing spec calibrates
        # exactly as before.
        _mk = params.get("mono_k", None)
        self.mono_k = None if _mk is None else float(_mk)
        # THE SAME PLANAR CONSTRAINT THE DEFAULT IMPLEMENTATION TAKES. This class does not derive
        # from `ShapeEnergy3D` -- it is a different energy on the same contract -- so it parses
        # and applies `plane_axis` itself. Without this the key is accepted, ignored, and a flat
        # monolayer buckles for a reason that has nothing to do with its thickness.
        _pa = params.get("plane_axis", None)
        self.plane_axis = None if _pa is None else int(_pa)

    def _rest_offset(self, x0, es, et, ef, nF, h, V_eq0, alive, R0t, eocc, vocc):
        """The constant to add to every cell's target volume so the SEEDED SHELL IS AT REST.

        WHY THIS EXISTS. The energy is  U = 1/2 k_v (v-V_eq)^2 + kappa_s s + 1/2 gamma P^2.  Only the
        first term can push outward; the other two always pull the surface in. Calibrating V_eq to
        the rest volume alone -- which is what the `mono_k` line below does -- balances the volume
        term against nothing, so a freshly seeded ball is NOT in equilibrium and collapses under its
        own tension. Measured: radius 5.00 -> 1.80 in 20 frames at the shipped settings, and still
        falling; with gamma=0 it is 2.95, with gamma=0 AND kappa_s=0 it is exactly 5.00. Runs that
        looked healthy avoided this only by loading a pre-relaxed checkpoint (round_40_mc8 starts at
        radius 6.14) rather than seeding, so the collapse was invisible for the whole campaign.

        WHY A CONSTANT, NOT A FACTOR. At force balance
            V_eq - v = (2 kappa_s + 1/2 gamma p0^2) / (k_v h0),
        because both tension terms contribute a SIZE-INDEPENDENT amount to dU/dA (P = p0 sqrt(A), so
        1/2 gamma P^2 = 1/2 gamma p0^2 A, whose derivative carries no A). The offset must therefore
        survive unchanged when a cell grows or divides -- a multiplicative correction would shrink
        with V0f and quietly reintroduce the collapse after the first division.

        WHY SOLVED, NOT TYPED IN. The loop sweeps k_v, kappa_s, gamma, h0, the seed radius and the
        cell count; a tension that happens to balance at one setting collapses at the next. The
        offset enters the force exactly linearly --
            g(delta) = g(0) - k_v delta grad(V_total)
        because U_vol is quadratic in (v - V_eq) -- so one gradient pair gives the exact root. What
        is zeroed is the RADIAL component: tangential forces are the cell-shape relaxation we want
        to keep, and only the radial resultant inflates or deflates the vesicle.

        WHY IT ITERATES. One solve is not enough in practice: it zeroes the radial force on the
        SEEDED mesh, but the first thing the relaxation does is even out cell shapes tangentially,
        and that moves the balance. Measured with a single solve, the defaults held to x1.014 but
        gamma=0 still drifted to x0.80 and gamma=0.3 to x1.10. So alternate -- relax the shape, put
        it back on the target sphere, re-solve -- until the shape stops changing. The rescaling is
        what pins the answer: it asks for the offset whose equilibrium radius IS the seed radius,
        rather than whatever radius the relaxation happens to wander to.
        """
        # ABOUT THE VESICLE'S OWN CENTRE, NOT THE ORIGIN. Every radius here was |x|, which is the
        # same thing only for a vesicle seeded at (0, 0, 0). Seeded at the centre of a wall box, a
        # 5.0-radius shell sat 43 from the origin: R_target became 43, `u` pointed along (1, 1, 1)
        # for every vertex, and the rescale at the end of the loop squeezed the whole tissue along
        # that diagonal while dragging it toward the corner -- measured as a mid-radius of 7.16
        # against 5.00 from the same seed, and a centroid drifting 0.35 a frame, before the first
        # frame was drawn. The centroid is weighted by occupancy so dead buffer rows do not pull it.
        def _ctr(y):
            w = vocc.to(y.dtype).reshape(-1, 1)
            return (y * w).sum(dim=0, keepdim=True) / w.sum().clamp(min=1.0)
        R_target = (x0 - _ctr(x0)).norm(dim=1).mean().clamp(min=1e-9)
        delta = torch.zeros((), dtype=x0.dtype, device=x0.device)
        x = x0.clone()
        for _ in range(8):
            V_eq = (V_eq0 + delta).clamp(min=1e-9)
            u = (x - _ctr(x)) / (x - _ctr(x)).norm(dim=1, keepdim=True).clamp(min=1e-9)         # outward radial direction
            g0 = self._grad(x, es, et, ef, nF, h, V_eq, alive, R0t, eocc, vocc)
            with torch.enable_grad():                                   # grad of TOTAL cell volume
                xg = x.detach().requires_grad_(True)
                vf, _, _, _ = monolayer_geometry_3d(xg, es, et, ef, nF, h, eocc)
                gV = torch.autograd.grad((vf * alive).sum(), xg)[0]
            den = self.k_v * (u * torch.nan_to_num(gV)).sum()
            if not torch.isfinite(den) or den.abs() < 1e-12:
                break
            step = ((u * g0).sum() / den).detach()
            if not torch.isfinite(step):
                break
            delta = delta + step
            # let the shape relax under the new offset, then put it back on the target sphere so the
            # next solve is asked about the radius we actually want
            V_eq = (V_eq0 + delta).clamp(min=1e-9)
            with torch.no_grad():
                cap = self.cap_frac * (x[et] - x[es]).norm(dim=-1).mean().clamp(min=1e-6)
            for _ in range(max(1, self.relax_iters)):
                s = -(self.eta * self.mu) * self._grad(x, es, et, ef, nF, h, V_eq, alive,
                                                       R0t, eocc, vocc)
                x = x + s * torch.clamp(cap / (s.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
            c = _ctr(x)
            x = c + (x - c) * (R_target / (x - c).norm(dim=1).mean().clamp(min=1e-9))
        return delta.detach()

    def _grad(self, x, es, et, ef, nF, h, V_eq, alive, R0t, eocc, vocc):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            E = _monolayer_energy_core(x, es, et, ef, nF, h, V_eq, alive, self.k_v, self.kappa_s,
                                       self.Lambda, self.K_R, R0t, eocc, vocc, self.gamma)
            g = torch.autograd.grad(E, x)[0]
        return torch.nan_to_num(g)

    def forward(self, H, mask=None):
        lvl = H.level(self.at); pos_full = lvl.get("pos"); v_full = torch.zeros_like(pos_full)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {self.at: v_full}
        Nv = int(m["Nv"]); nF = int(m["nF"]); es, et, ef = m["E_srce"], m["E_trgt"], m["E_face"]
        E = es.shape[0]; dev, dt = pos_full.device, pos_full.dtype
        x0 = pos_full[:Nv].detach().clone()
        eocc = torch.ones(E, device=dev, dtype=dt); vocc = torch.ones(Nv, device=dev, dtype=dt)
        R0t = torch.as_tensor(float(m["R0"]), dtype=dt, device=dev)
        h_cell = torch.full((nF,), self.h0, dtype=dt, device=dev)   # v1: uniform fixed thickness
        # PUBLISHED FOR THE RENDERER. The mesh this operator works on is the MID-surface, so a
        # picture of it is a picture of a sheet with no thickness -- which is the one thing the
        # monolayer model adds. Recording the per-cell thickness (a per-face column, so it rides
        # `mesh_face_offsets` into the trajectory like `A0` does) lets the cross section rebuild
        # the apical and basal shells through `monolayer_shells`, the same function the energy
        # below is written on.
        m["mono_h"] = float(self.h0)
        # target monolayer volume: calibrate ONCE so V_eq matches the rest prism volume, then track the
        # growth op's scaling of the wedge target V0f (cell_grow scales V0f per cell) -> reuse it.
        v_rest, _, _, _ = monolayer_geometry_3d(x0, es, et, ef, nF, h_cell, eocc)
        if "mono_k" not in m:
            if self.mono_k is not None:
                m["mono_k"] = self.mono_k                     # declared: see __init__
            else:
                wedge = face_geometry_3d(x0, es, et, ef, nF, eocc, apex=wedge_apex(m, x0))[3]
                m["mono_k"] = float((v_rest.median() / wedge.median().clamp(min=1e-9)).item())
        V_eq = (m["mono_k"] * m["V0f"]).clamp(min=1e-9)
        if self.rest_calibration == "force_balance":
            if "mono_delta" not in m:
                m["mono_delta"] = self._rest_offset(x0, es, et, ef, nF, h_cell, V_eq,
                                                    m["alive"], R0t, eocc, vocc)
            V_eq = (V_eq + m["mono_delta"]).clamp(min=1e-9)
        with torch.no_grad():
            cap = self.cap_frac * (x0[et] - x0[es]).norm(dim=-1).mean().clamp(min=1e-6)
        # STASHED FOR THE TOPOLOGY OPERATORS, exactly as the default implementation does it:
        # `edge_flip` reads the plane off `m["mech"]`, so a monolayer sheet keeps its T1s in plane.
        m["mech"] = dict(K_A=0.0, K_P=0.0, K_V=self.k_v, K_R=self.K_R, Lambda=self.Lambda,
                         Gamma=self.gamma, eta=self.eta, cap_frac=self.cap_frac,
                         plane_axis=self.plane_axis)
        x = x0.clone()
        for _ in range(max(1, self.relax_iters)):
            step = -(self.eta * self.mu) * self._grad(x, es, et, ef, nF, h_cell, V_eq, m["alive"], R0t, eocc, vocc)
            step = step * torch.clamp(cap / (step.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
            if self.plane_axis is not None:
                step[:, self.plane_axis] = 0.0
            x = x + step
        # THE DIVISOR IS `general.dt`, NOT A DECLARED ONE -- the same by-construction fix as the
        # default implementation (see the long note at ShapeEnergy3D). This operator emits a
        # VELOCITY that the engine immediately multiplies by `general.dt`, so the two must cancel;
        # a declared `dt` that differs from the harness's silently rescales the relaxation rate and
        # nothing looks wrong. Both implementations of the same contract now read the harness.
        _dt = float(getattr(H, "dt", self.dt) or self.dt)
        if "dt" in self.params and abs(float(self.params["dt"]) - _dt) > 1e-12 * max(_dt, 1.0) \
                and not getattr(self, "_dt_warned", False):
            self._dt_warned = True
            from plexus.paths import warn
            warn(f"[warn] cell_mechanics[monolayer]: `dt: {self.params['dt']}` is IGNORED -- the "
                 f"divisor is general.dt ({_dt}) so that the engine's own multiplication cancels "
                 f"it. Remove the parameter; leaving it in the spec suggests a knob that is not there.")
        v_full[:Nv] = (x - x0) / max(_dt, 1e-9)
        if mask is not None:
            v_full = v_full * mask[:, None].float()
        return {self.at: v_full}


@register_operator("cell_mechanics", model="apicobasal", set="vertex", kind="lateral",
                   family="mechanics", title="relax thick epithelial cell shapes", equation=r"""$$E(\mathbf x,\mathbf s)=E_{\mathrm{mono}}(\mathbf x+\mathbf s,\ \mathbf x-\mathbf s)+\tfrac{\kappa_h}{2}\sum_e\lvert\mathbf s_{\mathrm{src}(e)}-\mathbf s_{\mathrm{trg}(e)}\rvert^2,\qquad \dot{\mathbf s}=-\mu_s\mu\,\nabla_{\mathbf s}E$$""")
class ApicoBasalShapeEnergy3D(Lateral):
    r"""`apicobasal` model of cell_mechanics: the monolayer's energy written on the doubled degree-of-freedom set (pos, sep).
    vertex -> vertex: reads pos and the `sep` block, emits the velocity of both; the energy is the monolayer's term for term, evaluated on apical = pos + sep, basal = pos - sep.

        E(\mathbf x,\mathbf s)=E_{\mathrm{mono}}(\mathbf x+\mathbf s,\ \mathbf x-\mathbf s)+\tfrac{\kappa_h}{2}\sum_e\lvert\mathbf s_{\mathrm{src}(e)}-\mathbf s_{\mathrm{trg}(e)}\rvert^2,\qquad \dot{\mathbf s}=-\mu_s\mu\,\nabla_{\mathbf s}E
        x, s           the mid-surface position and the half-separation
        E_{mono}       the monolayer energy
        \kappa_h       `kappa_h`, the thickness-field stiffness
        \mu_s          `sep_mu`, the separation's mobility relative to the mid-surface's `mu` (0 = the monolayer's fixed thickness)

    Reference: Okuda, S. et al. (2018). Sci. Rep. 8:2386, Eq. 3. Plexus apico-basal promotion R3 (this work)."""
    PARAM_UNITS = {'k_v': 'stress', 'kappa_s': 'energy_per_area', 'kappa_h': 'force', 'gamma': 'force', 'Lambda': 'force', 'K_R': 'stress', 'mu': 'dimensionless', 'sep_mu': 'dimensionless', 'dt': 'time', 'relax_iters': 'count', 'eta': 'dimensionless', 'cap_frac': 'fraction', 'mono_k': 'stress'}
    SYMBOLS = {'x, s': 'the mid-surface position and the half-separation', 'E_{mono}': 'the monolayer energy', '\\kappa_h': '`kappa_h`, the thickness-field stiffness', '\\mu_s': "`sep_mu`, the separation's mobility relative to the mid-surface's `mu` (0 = the monolayer's fixed thickness)"}
    SUPPORTED_DIMS = [3]; EMIT = "velocity"; DIFFERENTIABLE = True
    # THE SAME FUNCTIONAL, SO THE SAME DIMENSIONS -- see `_apicobasal_energy_core`'s equation.
    # `kappa_s` multiplies the cell's whole polyhedron SURFACE, so kappa_s * L^2 = F*L and
    # kappa_s = F/L: it IS a surface tension, which is why raising it holds a dying cell's ring
    # open against `k_v` and stalled 59 of the extrusions on `apop2_ab_half`.
    # AND THIS ONE DEFENDS THE POLYHEDRON, WHICH IS THE WHOLE FINDING. `_apicobasal_energy_core`
    # evaluates `1/2 k_v (V_j - V_eq_j)^2` with `V_j` from `apicobasal_geometry_3d` -- two caps and
    # one wall per ring edge, by the divergence theorem -- while `V_eq_j` is the `V0f` the seed
    # filled with a WEDGE volume. Measured on the reference spheroid those two read 1.3508 and
    # 2.5433 for the same cell.
    #
    # DECLARING IT IS NOT FIXING IT. The tissue stays half-converted; what changes is that the
    # loader now says so on every apicobasal spec instead of the fact living in one gate's prose
    # (`gate_ab_population`'s doubling-time row). Reconciling the two -- one volume for growth,
    # division, death AND the energy -- is AB_R7R8_TODO section 0a and is deliberately out of scope.
    BLOCK_UNITS = {"V0f": "volume[polyhedron]"}
    INPUTS = ["vertex"]; OUTPUTS = ["vertex"]; READS = ["pos", "sep"]; WRITES = ["pos", "sep"]
    MECHANISM_TAGS = ["vertex_model", "apicobasal", "cell_polyhedron", "cell_3d_volume",
                      "surface_tension", "emergent_bending", "force_balance"]
    REFERENCE = ("Okuda, S. et al. (2013). Biomech. Model. Mechanobiol. 12:627-644 (3D vertex model "
                 "with independent apical and basal surfaces); Okuda, S. et al. (2018). Sci. Rep. "
                 "8:2386 (the monolayer reduction this generalises).")
    PARAM_ROLES = {"k_v": "cell_volume_elasticity", "kappa_s": "surface_tension",
                   "kappa_h": "thickness_field_stiffness", "sep_mu": "apicobasal_mobility",
                   "surface": "which surface the ring terms act on", 'K_R': 'radial_restoring_stiffness', 'Lambda': 'line_tension', 'cap_frac': 'cap_on_the_vertex_step', 'dt': 'relaxation_time_step', 'eta': 'step_size_of_the_relaxation', 'gamma': 'cortical_contractility', 'mono_k': 'stiffness_of_the_monolayer_regulariser', 'mu': 'mobility_of_the_vertices', 'plane_axis': 'normal_axis_of_a_flat_sheet', 'relax_iters': 'relaxation_iterations_per_frame', 'rest_calibration': 'how_the_rest_state_is_calibrated', 'sep_block': 'name_of_the_half_separation_block'}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.sep_block = str(params.get("sep_block", "sep"))
        self.k_v = float(params.get("k_v", 4.0)); self.kappa_s = float(params.get("kappa_s", 0.2))
        self.kappa_h = float(params.get("kappa_h", 0.0))       # thickness-field stiffness; see the energy core
        # `p0` IS NOT ON THIS CONTRACT, AND A SPEC THAT SETS IT IS REFUSED. The mid-surface model
        # reads a target shape index; this energy has no perimeter term keyed to one -- its
        # perimeter enters only through `gamma`, which is stated directly. Silently ignored, a
        # `p0` written here makes a sweep arm identical to its own baseline to four decimals.
        if "p0" in params:
            raise ValueError(
                "cell_mechanics[apicobasal] does not read `p0`: this energy has no perimeter term "
                "keyed to a target shape index, so the value would do nothing. Use `gamma` for the "
                "cortical contractility. (`p0` on `cell_divide` and `cell_die` is a different "
                "parameter -- it sets a fresh cell's P0 from its A0 -- and is read.)")
        self.gamma = float(params.get("gamma", 0.0))
        self.Lambda = float(params.get("Lambda", 0.0)); self.K_R = float(params.get("K_R", 0.0))
        self.mu = float(params.get("mu", 1.0)); self.dt = float(params.get("dt", 1.0))
        self.relax_iters = int(params.get("relax_iters", 30)); self.eta = float(params.get("eta", 0.08))
        self.cap_frac = float(params.get("cap_frac", 0.12))
        self.sep_mu = float(params.get("sep_mu", 1.0))
        # WHICH SURFACE THE RING TERMS ACT ON -- see `_apicobasal_energy_core`. `apical` by default,
        # because the mid-surface is a coordinate choice and not a membrane, so a perimeter, a line
        # tension or a radial spring written on it is a force on the average of two surfaces.
        self.surface = str(params.get("surface", "apical")).lower()
        if self.surface not in ("apical", "basal", "mid"):
            raise ValueError(f"cell_mechanics[apicobasal]: surface must be apical|basal|mid, "
                             f"got {self.surface!r}")
        self.rest_calibration = str(params.get("rest_calibration", "force_balance"))
        _mk = params.get("mono_k", None)                  # see MonolayerShapeEnergy3D.__init__
        self.mono_k = None if _mk is None else float(_mk)
        _pa = params.get("plane_axis", None)
        self.plane_axis = None if _pa is None else int(_pa)

    # ---------------------------------------------------------------- gradients
    def _grad(self, x, s, es, et, ef, nF, V_eq, alive, R0t, eocc, vocc, want_sep):
        """(dU/dx, dU/ds). ONE autograd pass over both, because they are one energy.

        `want_sep` is False at `sep_mu: 0`, where the second gradient would be computed, multiplied
        by zero and thrown away -- and it is not free: it doubles the graph the relaxation walks
        thirty times a frame.

        THE WARP KERNELS FIRST, AUTOGRAD WHERE THEY DO NOT APPLY -- the arrangement the default
        model has (`try_shape_energy_grad`). `apicobasal_energy_grad_warp` is the hand-derived
        gradient of the same energy in both degrees of freedom; on CUDA/float32 without
        PLEXUS_STRICT_DETERMINISM it is the route, and it is why this operator went from 227 ms
        a frame to a few on `spheroid_organelles_nucleus`.
        """
        _lam_e = getattr(self, "_lam_e", None)
        # THE WARP KERNELS CARRY A SCALAR LINE TENSION ONLY: with a per-junction multiplier in
        # play the autograd path serves the gradient (as it did for `kappa_h` before R5a).
        _g = (try_apicobasal_grad(self, x, s, es, et, ef, nF, V_eq, alive, R0t, eocc, vocc)
              if _lam_e is None else None)
        if _g is not None:
            return _g[0], (_g[1] if want_sep else None)
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            s = s.detach().requires_grad_(want_sep)
            E = _apicobasal_energy_core(x, s, es, et, ef, nF, V_eq, alive, self.k_v, self.kappa_s,
                                        self.Lambda, self.K_R, R0t, eocc, vocc, self.gamma,
                                        self.surface, kappa_h=self.kappa_h, lam_e=_lam_e)
            if want_sep:
                gx, gs = torch.autograd.grad(E, (x, s))
                return torch.nan_to_num(gx), torch.nan_to_num(gs)
            gx = torch.autograd.grad(E, x)[0]
        return torch.nan_to_num(gx), None

    def _rest_offset(self, x0, s0, es, et, ef, nF, V_eq0, alive, R0t, eocc, vocc):
        """The constant added to every cell's target volume so the SEEDED SHELL IS AT REST.

        THE SAME SOLVE THE MONOLAYER DOES, on this geometry -- see `MonolayerShapeEnergy3D._rest_offset`
        for why it exists at all (only the volume term can push outward; calibrating `V_eq` to the
        rest volume balances it against nothing and a freshly seeded ball collapses under its own
        tension, radius 5.00 -> 1.80 in 20 frames), why the correction is a CONSTANT and not a factor,
        why it is solved rather than typed in, and why it iterates. Repeated here rather than
        inherited because the gradient is this energy's, and a shared solve that called the wrong
        `_grad` would calibrate one model against the other's rest state.

        `sep` IS HELD FIXED THROUGHOUT THE SOLVE, at every `sep_mu`. The question the offset answers
        is which target volume puts the MID-SURFACE at radius R; letting the thickness relax inside
        the calibration would answer it with a different tissue than the one that then runs.
        """
        # ABOUT THE VESICLE'S OWN CENTRE, NOT THE ORIGIN. Every radius here was |x|, which is the
        # same thing only for a vesicle seeded at (0, 0, 0). Seeded at the centre of a wall box, a
        # 5.0-radius shell sat 43 from the origin: R_target became 43, `u` pointed along (1, 1, 1)
        # for every vertex, and the rescale at the end of the loop squeezed the whole tissue along
        # that diagonal while dragging it toward the corner -- measured as a mid-radius of 7.16
        # against 5.00 from the same seed, and a centroid drifting 0.35 a frame, before the first
        # frame was drawn. The centroid is weighted by occupancy so dead buffer rows do not pull it.
        def _ctr(y):
            w = vocc.to(y.dtype).reshape(-1, 1)
            return (y * w).sum(dim=0, keepdim=True) / w.sum().clamp(min=1.0)
        R_target = (x0 - _ctr(x0)).norm(dim=1).mean().clamp(min=1e-9)
        delta = torch.zeros((), dtype=x0.dtype, device=x0.device)
        x = x0.clone()
        for _ in range(8):
            V_eq = (V_eq0 + delta).clamp(min=1e-9)
            u = (x - _ctr(x)) / (x - _ctr(x)).norm(dim=1, keepdim=True).clamp(min=1e-9)
            g0, _ = self._grad(x, s0, es, et, ef, nF, V_eq, alive, R0t, eocc, vocc, False)
            with torch.enable_grad():
                xg = x.detach().requires_grad_(True)
                vf, _, _, _ = apicobasal_geometry_3d(xg, s0, es, et, ef, nF, eocc)
                gV = torch.autograd.grad((vf * alive).sum(), xg)[0]
            den = self.k_v * (u * torch.nan_to_num(gV)).sum()
            if not torch.isfinite(den) or den.abs() < 1e-12:
                break
            step = ((u * g0).sum() / den).detach()
            if not torch.isfinite(step):
                break
            delta = delta + step
            V_eq = (V_eq0 + delta).clamp(min=1e-9)
            with torch.no_grad():
                cap = self.cap_frac * (x[et] - x[es]).norm(dim=-1).mean().clamp(min=1e-6)
            for _ in range(max(1, self.relax_iters)):
                gx, _ = self._grad(x, s0, es, et, ef, nF, V_eq, alive, R0t, eocc, vocc, False)
                st = -(self.eta * self.mu) * gx
                x = x + st * torch.clamp(cap / (st.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
            c = _ctr(x)
            x = c + (x - c) * (R_target / (x - c).norm(dim=1).mean().clamp(min=1e-9))
        return delta.detach()

    # ---------------------------------------------------------------- forward
    def forward(self, H, mask=None):
        lvl = H.level(self.at); pos_full = lvl.get("pos"); v_full = torch.zeros_like(pos_full)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {self.at: v_full}
        if self.sep_block not in lvl.state_schema:
            raise ValueError(
                f"cell_mechanics[apicobasal] needs a `{self.sep_block}` block on the {self.at!r} "
                f"set -- seed it with `{{op: seed_mesh, implementation: apicobasal}}`. Without it "
                f"there is no second surface and the run would be the mid-surface model wearing "
                f"this operator's name.")
        Nv = int(m["Nv"]); nF = int(m["nF"]); es, et, ef = m["E_srce"], m["E_trgt"], m["E_face"]
        E = es.shape[0]; dev, dt = pos_full.device, pos_full.dtype
        x0 = pos_full[:Nv].detach().clone()
        s0 = lvl.get(self.sep_block)[:Nv].detach().clone()
        eocc = torch.ones(E, device=dev, dtype=dt); vocc = torch.ones(Nv, device=dev, dtype=dt)
        R0t = torch.as_tensor(float(m["R0"]), dtype=dt, device=dev)
        self._R0 = float(m["R0"])                        # the same number, sync-free for the kernels
        # PUBLISHED FOR THE RENDERER, and it is now a MEASUREMENT rather than a declared constant:
        # the monolayer writes its `h0` here because its thickness is one, and this model's is a
        # per-vertex vector. The cross section wants one number, so it gets the mean cell thickness
        # |a - b| = 2|sep|, which is what `h0` meant on the other model.
        m["mono_h"] = float(2.0 * s0.norm(dim=1).mean())
        # THE SAME V_eq THE MONOLAYER GETS, and that is what makes AB-C1 a controlled comparison:
        # both arms calibrate the wedge target `V0f` -- the quantity `cell_grow` scales -- against
        # their own rest volume, and on a flat patch those rest volumes are the same number.
        if "mono_k" not in m:
            if self.mono_k is not None:
                m["mono_k"] = self.mono_k                     # declared: see the monolayer's __init__
            elif str(m.get("v0_from", "wedge")).lower() == "polyhedron":
                # `mono_k` CONVERTS A WEDGE-UNIT TARGET INTO THE POLYHEDRON VOLUME THIS ENERGY
                # DEFENDS. A seed that declared `v0_from: polyhedron` wrote `V0f` in that volume
                # already; scaling it by v_rest/wedge (0.53 on the reference spheroid) a second
                # time put the target at half the seeded volume, the rest solve then added the
                # difference back as `mono_delta`, and every cell inflated x1.7 over the opening
                # frames while the size readers cached a reference from the un-inflated seed
                # (R1 of notes/size_cycle/SIZE_CYCLE_PLAN.md, finding 7).
                m["mono_k"] = 1.0
            else:
                # measured ONCE, here: it was computed on every frame and read on this one
                v_rest, _, _, _ = apicobasal_geometry_3d(x0, s0, es, et, ef, nF, eocc)
                wedge = face_geometry_3d(x0, es, et, ef, nF, eocc, apex=wedge_apex(m, x0))[3]
                m["mono_k"] = float((v_rest.median() / wedge.median().clamp(min=1e-9)).item())
        V_eq = (m["mono_k"] * m["V0f"]).clamp(min=1e-9)
        if self.rest_calibration == "force_balance":
            if "mono_delta" not in m:
                m["mono_delta"] = self._rest_offset(x0, s0, es, et, ef, nF, V_eq, m["alive"],
                                                    R0t, eocc, vocc)
            V_eq = (V_eq + m["mono_delta"]).clamp(min=1e-9)
        with torch.no_grad():
            cap = self.cap_frac * (x0[et] - x0[es]).norm(dim=-1).mean().clamp(min=1e-6)
            # THE SEPARATION'S CAP IS IN THE SEPARATION'S OWN UNIT -- a fraction of the mean cell
            # HALF-thickness, not of the mean junction length. A cap borrowed from the mid-surface
            # would be several times the whole thickness on a thin epithelium, which is not a cap.
            cap_s = self.cap_frac * s0.norm(dim=1).mean().clamp(min=1e-6)
        m["mech"] = dict(K_A=0.0, K_P=0.0, K_V=self.k_v, K_R=self.K_R, Lambda=self.Lambda,
                         Gamma=self.gamma, eta=self.eta, cap_frac=self.cap_frac,
                         plane_axis=self.plane_axis)
        move_sep = self.sep_mu != 0.0
        # THE PER-JUNCTION LINE-TENSION MULTIPLIER, when an operator has written one for THIS
        # topology (`junction_myosin`, e.g. `model: oriented`). A mis-sized array is not used: the
        # frame relaxes with the scalar line tension and says so once.
        _myo = m.get("myo")
        self._lam_e = None
        if torch.is_tensor(_myo) and self.Lambda != 0.0:
            if _myo.shape[0] == E:
                self._lam_e = _myo.detach().to(device=dev, dtype=dt)
            elif not getattr(self, "_myo_warned", False):
                self._myo_warned = True
                print(f"[cell_mechanics/apicobasal] m['myo'] has {_myo.shape[0]} entries for {E} "
                      f"half-edges -- relaxing with the scalar line tension", flush=True)
        if apicobasal_warp_unavailable(x0) is None and self._lam_e is None:
            # THE WHOLE LOOP ON THE WARP PATH: state stays in the kernels' buffers, one gradient
            # and one `ab_step` per iteration, the same arithmetic as the torch loop below.
            x, s = relax_apicobasal_warp(self, x0, s0, es, et, ef, nF, V_eq, m["alive"], self._R0,
                                         eocc, vocc, cap, cap_s, move_sep)
            return self._emit(H, pos_full, x0, s0, x, s, Nv, move_sep, mask)
        x = x0.clone(); s = s0.clone()
        for _ in range(max(1, self.relax_iters)):
            gx, gs = self._grad(x, s, es, et, ef, nF, V_eq, m["alive"], R0t, eocc, vocc, move_sep)
            step = -(self.eta * self.mu) * gx
            step = step * torch.clamp(cap / (step.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
            if self.plane_axis is not None:
                step[:, self.plane_axis] = 0.0
            x = x + step
            if move_sep:
                ds = -(self.eta * self.mu * self.sep_mu) * gs
                ds = ds * torch.clamp(cap_s / (ds.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
                s = s + ds
        return self._emit(H, pos_full, x0, s0, x, s, Nv, move_sep, mask)

    def _emit(self, H, pos_full, x0, s0, x, s, Nv, move_sep, mask):
        """The relaxed state as the velocities the engine integrates -- shared by both loops."""
        v_full = torch.zeros_like(pos_full)
        # THE DIVISOR IS `general.dt`, NOT A DECLARED ONE -- the same by-construction fix both other
        # implementations of this contract carry: the engine multiplies an emitted velocity by
        # `general.dt`, so the two must cancel or the relaxation rate is silently rescaled.
        _dt = float(getattr(H, "dt", self.dt) or self.dt)
        if "dt" in self.params and abs(float(self.params["dt"]) - _dt) > 1e-12 * max(_dt, 1.0) \
                and not getattr(self, "_dt_warned", False):
            self._dt_warned = True
            from plexus.paths import warn
            warn(f"[warn] cell_mechanics[apicobasal]: `dt: {self.params['dt']}` is IGNORED -- the "
                 f"divisor is general.dt ({_dt}).")
        v_full[:Nv] = (x - x0) / max(_dt, 1e-9)
        if mask is not None:
            v_full = v_full * mask[:, None].float()
        out = {self.at: v_full}
        if move_sep:
            # THE SECOND DELTA, ROUTED BY `(set, block)` -- what R1(a) was built for. Without that
            # key the engine folds every delta for a set into its COORDINATE block, so a separation
            # velocity would be added to `pos` and the tissue would translate instead of thicken.
            vs = torch.zeros_like(pos_full)
            vs[:Nv] = (s - s0) / max(_dt, 1e-9)
            if mask is not None:
                vs = vs * mask[:, None].float()
            out[(self.at, self.sep_block)] = vs
        return out


def _frame_loads(H, m, Nv, dev, dt):
    """This frame's matrix load (`f_basal`, or None) and contact lists, for a tissue relaxation."""
    _fr = getattr(H, "frame", None)
    _fr = None if _fr is None else int(_fr)
    _fb = m.get("f_basal")
    fext = (_fb.to(device=dev, dtype=dt)
            if (torch.is_tensor(_fb) and _fr is not None and m.get("f_basal_frame") == _fr
                and _fb.shape[0] == Nv) else None)
    _bc = m.get("basal_contacts")
    conts = [C for C in (_bc.values() if isinstance(_bc, dict) else ())
             if C.get("frame") == _fr and int(C.get("nv", -1)) == Nv]
    return _fr, _bc, fext, conts


def _book_contacts(H, m, _fr, _bc, conts, Nv, dev, out):
    """Book each list's last-iteration pair in the ledger and hand each membrane node the velocity
    F / gamma of the force the tissue relaxed against -- the same F, so the pair that acted is the
    pair booked. A frame with a live contact but no list still opens the ledger, so the row records
    zero rather than an earlier frame's numbers."""
    from plexus.operators.contact_ops import basal_load, record_interface
    if _fr is not None and (conts or isinstance(_bc, dict)):
        basal_load(m, _fr, Nv, dev)
    for C in conts:
        F = C.get("_F")
        if F is None or F.shape[0] == 0:
            continue
        record_interface(m, F, C["_fv"])
        npos = H.level(C["set"]).get("pos")
        v = torch.zeros_like(npos)
        v[C["node"]] = (F / float(C["gamma"])).to(npos.dtype)
        out[C["set"]] = out[C["set"]] + v if C["set"] in out else v
        C["_F"] = C["_fv"] = None
    return out


@register_operator("cell_mechanics", implementation="apicobasal_contact", set="vertex",
                   kind="lateral", family="mechanics", title="relax thick cells against surroundings", equation=r"""$$U=U_{ab}(\mathbf x,\mathbf s)-\sum_v\mathbf F_v\cdot(\mathbf x_v-\mathbf s_v)+\sum_i\tfrac12 k p_i^2-e_{cm}\sum_f c_fA_f+\tfrac{k_c}{2}\big(V_L-V^*\big)^2+\tfrac{k_h}{2}\sum_v(2|\mathbf s_v|-h^*)^2+\tfrac{k_b}{2}\sum_v|(L\mathbf x)_v|^2$$""")
class ApicoBasalContactShapeEnergy3D(ApicoBasalShapeEnergy3D):
    r"""`apicobasal_contact` model of cell_mechanics: the apico-basal energy relaxed against what surrounds the tissue, with cell-matrix adhesion, an interior core, a columnar height and bending as optional terms.
    vertex -> vertex (and -> each contact partner's set, a velocity): reads pos, sep, the basal load buffer and the contact lists; emits the velocity of both groups and returns each membrane node the force it relaxed against.

        U=U_{ab}(\mathbf x,\mathbf s)-\sum_v\mathbf F_v\cdot(\mathbf x_v-\mathbf s_v)+\sum_i\tfrac12 k p_i^2-e_{cm}\sum_f c_fA_f+\tfrac{k_c}{2}\big(V_L-V^*\big)^2+\tfrac{k_h}{2}\sum_v(2|\mathbf s_v|-h^*)^2+\tfrac{k_b}{2}\sum_v|(L\mathbf x)_v|^2
        U_{ab}         the parent's apico-basal energy
        F_v, k, p_i    the matrix load, the contact stiffness and penetration
        e_{cm}         `e_cm`, the cell-matrix adhesion energy per area
        c_f, A_f       how much membrane cell f holds (0..1) and its basal cap area
        k_c, V_L, V^*  `k_core`, the enclosed interior volume and its target (clock, `core_alpha` or `core_from`)
        k_h, h^*       `k_height` and `height_target`
        k_b, L         `k_bend` and the combinatorial Laplacian of the mid-surface

    Reference: Okuda, S. et al. (2013). Biomech. Model. Mechanobiol. 12:627-644; Chen, Z. et al. (2015). Comput. Methods Appl. Mech. Engrg. 293:1-19; Wang, S. et al. (2021). Cell 184:3702-3716."""
    PARAM_UNITS = {'e_cm': 'energy_per_area', 'k_core': 'stress', 'core_rate': 'rate', 'core_alpha': 'dimensionless', 'k_height': 'force', 'height_target': 'length', 'k_bend': 'force', 'k_bend_cg': 'count', 'kappa_h_jacobi': 'count'}
    SYMBOLS = {'U_{ab}': "the parent's apico-basal energy", 'F_v, k, p_i': 'the matrix load, the contact stiffness and penetration', 'e_{cm}': '`e_cm`, the cell-matrix adhesion energy per area', 'c_f, A_f': 'how much membrane cell f holds (0..1) and its basal cap area', 'k_c, V_L, V^*': '`k_core`, the enclosed interior volume and its target (clock, `core_alpha` or `core_from`)', 'k_h, h^*': '`k_height` and `height_target`', 'k_b, L': '`k_bend` and the combinatorial Laplacian of the mid-surface'}
    PARAM_ROLES = {'apical_sep': 'full_or_half_apical_separation_for_the_contact', 'bound_block': 'bound_integrin_block_for_e_cm_from_bound', 'cell_set': 'the_cell_set', 'core_alpha': 'interior_gain_over_the_layer_gain', 'core_bulk': 'interior_as_a_bulk_material', 'core_from': 'set_whose_cell_count_sets_the_interior_target', 'core_rate': 'growth_rate_of_the_interior_clock_target', 'e_cm': 'cell_matrix_adhesion_energy_per_area', 'e_cm_from': 'ligand_or_bound', 'height_implicit': 'integrate_the_height_spring_exactly', 'height_target': 'target_cell_height', 'k_bend': 'positional_bending_stiffness', 'k_bend_cg': 'conjugate_gradient_iterations_of_the_bending_solve', 'k_core': 'stiffness_of_the_interior_volume', 'k_height': 'stiffness_of_the_columnar_height', 'kappa_h_implicit': 'integrate_the_thickness_field_implicitly', 'kappa_h_jacobi': 'jacobi_sweeps_of_the_implicit_thickness_solve'}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.e_cm = float(params.get("e_cm", 0.0))
        self.e_cm_from = str(params.get("e_cm_from", "ligand")).lower()
        if self.e_cm_from not in ("ligand", "bound"):
            raise ValueError(f"cell_mechanics[apicobasal_contact]: e_cm_from is ligand or bound, got {self.e_cm_from!r}")
        self.bound_block = str(params.get("bound_block", "itg_B"))
        self.cm_cell_set = str(params.get("cell_set", "cell"))
        self._b_ref = None
        self.apical_sep = str(params.get("apical_sep", "full")).lower()
        if self.e_cm and self.e_cm >= self.kappa_s:
            raise ValueError(f"cell_mechanics[apicobasal_contact]: e_cm {self.e_cm:g} must be below "
                             f"kappa_s {self.kappa_s:g}, the basal tension it offsets")
        self.k_core = float(params.get("k_core", 0.0))
        self.core_rate = self.tunable(params.get("core_rate"), 0.0, tensor=True)
        _ca = params.get("core_alpha", None)
        self.core_alpha = None if _ca is None else float(_ca)
        self.core_from = params.get("core_from", None)
        if self.core_from is not None and self.core_alpha is not None:
            raise ValueError("cell_mechanics[apicobasal_contact]: core_from and core_alpha are two targets; give one")
        self.k_height = float(params.get("k_height", 0.0))
        self.core_bulk = bool(params.get("core_bulk", False))
        self.k_bend = float(params.get("k_bend", 0.0))
        self.k_bend_cg = int(params.get("k_bend_cg", 20))
        self.height_implicit = bool(params.get("height_implicit", False))
        self.height_target = float(params.get("height_target", 1.0))
        self.kappa_h_implicit = bool(params.get("kappa_h_implicit", False))
        self.kappa_h_jacobi = int(params.get("kappa_h_jacobi", 10))

    @staticmethod
    def _kappa_h_backward_euler(r, z, es, et, eocc, tau, sweeps):
        """(I + tau L) s_new = r by `sweeps` Jacobi sweeps started from z, L the ring half-edge Laplacian."""
        w = eocc.to(r.dtype)[:, None]
        deg = torch.zeros(r.shape[0], 1, dtype=r.dtype, device=r.device)
        deg = deg.index_add(0, es, w).index_add(0, et, w)
        for _ in range(max(1, sweeps)):
            nb = torch.zeros_like(r).index_add(0, es, w * z[et]).index_add(0, et, w * z[es])
            z = (r + tau * nb) / (1.0 + tau * deg)
        return z

    @staticmethod
    def _bend_solve(r, x0, es, et, eocc, tau, iters):
        """(I + tau L^2) x = r by conjugate gradient from x0, L the combinatorial ring-edge Laplacian (symmetric)."""
        w = eocc.to(r.dtype)[:, None]
        deg = torch.zeros(r.shape[0], 1, dtype=r.dtype, device=r.device).index_add(0, es, w).index_add(0, et, w)

        def lap(v):
            nb = torch.zeros_like(v).index_add(0, es, w * v[et]).index_add(0, et, w * v[es])
            return deg * v - nb

        def A(v):
            return v + tau * lap(lap(v))
        x = x0.clone()
        res = r - A(x)
        p = res.clone()
        rr = (res * res).sum(0)
        for _ in range(max(1, iters)):
            Ap = A(p)
            a = rr / (p * Ap).sum(0).clamp_min(1e-30)
            x = x + a * p
            res = res - a * Ap
            rr2 = (res * res).sum(0)
            if float(rr2.max()) < 1e-20:
                break
            p = res + (rr2 / rr.clamp_min(1e-30)) * p
            rr = rr2
        return x

    @staticmethod
    def _core_grad(x, s, es, et, ef, nF, eocc, core):
        """d/d(x, s) of 1/2 k_core (V_L - V_target)^2 by autograd, and V_L."""
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            sg = s.detach().requires_grad_(True)
            V = enclosed_ring_volume(xg + core["sgn"] * sg, es, et, ef, nF, eocc, core["o"])
            E = 0.5 * core["k"] * (V - core["V"]) ** 2
            gx, gs = torch.autograd.grad(E, (xg, sg))
        return gx.detach(), gs.detach(), float(V.detach())

    def _core_target(self, H, m, x0, s0, es, et, ef, nF, eocc):
        """This frame's inner-mass term, or None when `k_core` is 0. The ring and V_L(0) are fixed once."""
        if self.k_core <= 0:
            return None
        o = x0.mean(dim=0).detach()
        va = float(enclosed_ring_volume(x0 + s0, es, et, ef, nF, eocc, o))
        vb = float(enclosed_ring_volume(x0 - s0, es, et, ef, nF, eocc, o))
        n_in = None
        if self.core_from is not None:
            L = H.level(self.core_from) if H is not None else None
            if L is None:
                raise ValueError(f"cell_mechanics[apicobasal_contact]: core_from set {self.core_from!r} not found")
            n_in = float((L.occ > 0.5).sum())
        if "core" not in m:
            m["core"] = dict(sgn=1.0 if va <= vb else -1.0, V_first=min(va, vb), VS_first=abs(vb - va),
                             N_first=n_in)
        _fr = getattr(H, "frame", None)
        t = 0.0 if _fr is None else float(int(_fr))
        C = m["core"]
        if n_in is not None:
            V = C["V_first"] * n_in / max(C["N_first"], 1.0)
        elif self.core_alpha is not None:
            V = C["V_first"] + self.core_alpha * (abs(vb - va) - C["VS_first"])
        else:
            V = C["V_first"] * (1.0 + self.core_rate * t)
        m["core_target"] = V
        k = self.k_core * (C["V_first"] / max(V, 1e-9)) if self.core_bulk else self.k_core
        return dict(k=k, sgn=C["sgn"], o=o, V=V)

    @staticmethod
    def _adhesion_grad(x, s, es, et, ef, nF, cov, e_cm):
        """d/d(x, s) of -e_cm sum_f cov_f A_f(x - s), by autograd on the basal caps' fan areas."""
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            sg = s.detach().requires_grad_(True)
            b = xg - sg
            cnt = torch.bincount(ef, minlength=nF).clamp_min(1).to(b.dtype)
            cen = torch.zeros(nF, 3, dtype=b.dtype, device=b.device).index_add(0, ef, b[es]) / cnt[:, None]
            tri = 0.5 * torch.cross(b[es] - cen[ef], b[et] - cen[ef], dim=1).norm(dim=1)
            E = -e_cm * (cov[ef] * tri).sum()
            gx, gs = torch.autograd.grad(E, (xg, sg))
        return gx.detach(), gs.detach()

    def forward(self, H, mask=None):
        lvl = H.level(self.at); pos_full = lvl.get("pos"); v_full = torch.zeros_like(pos_full)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {self.at: v_full}
        if self.sep_block not in lvl.state_schema:
            raise ValueError(f"cell_mechanics[apicobasal_contact] needs a `{self.sep_block}` block "
                             f"on the {self.at!r} set -- seed it with the apicobasal seed_mesh")
        Nv = int(m["Nv"]); nF = int(m["nF"]); es, et, ef = m["E_srce"], m["E_trgt"], m["E_face"]
        E = es.shape[0]; dev, dt = pos_full.device, pos_full.dtype
        x0 = pos_full[:Nv].detach().clone()
        s0 = lvl.get(self.sep_block)[:Nv].detach().clone()
        eocc = torch.ones(E, device=dev, dtype=dt); vocc = torch.ones(Nv, device=dev, dtype=dt)
        R0t = torch.as_tensor(float(m["R0"]), dtype=dt, device=dev)
        self._R0 = float(m["R0"])
        self._lam_e = None
        m["mono_h"] = float(2.0 * s0.norm(dim=1).mean())
        # THE PARENT'S TARGET VOLUME, step for step (see ApicoBasalShapeEnergy3D.forward).
        if "mono_k" not in m:
            if self.mono_k is not None:
                m["mono_k"] = self.mono_k
            elif str(m.get("v0_from", "wedge")).lower() == "polyhedron":
                m["mono_k"] = 1.0
            else:
                v_rest, _, _, _ = apicobasal_geometry_3d(x0, s0, es, et, ef, nF, eocc)
                wedge = face_geometry_3d(x0, es, et, ef, nF, eocc, apex=wedge_apex(m, x0))[3]
                m["mono_k"] = float((v_rest.median() / wedge.median().clamp(min=1e-9)).item())
        V_eq = (m["mono_k"] * m["V0f"]).clamp(min=1e-9)
        if self.rest_calibration == "force_balance":
            if "mono_delta" not in m:
                m["mono_delta"] = self._rest_offset(x0, s0, es, et, ef, nF, V_eq, m["alive"],
                                                    R0t, eocc, vocc)
            V_eq = (V_eq + m["mono_delta"]).clamp(min=1e-9)
        with torch.no_grad():
            cap = self.cap_frac * (x0[et] - x0[es]).norm(dim=-1).mean().clamp(min=1e-6)
            cap_s = self.cap_frac * s0.norm(dim=1).mean().clamp(min=1e-6)
        m["mech"] = dict(K_A=0.0, K_P=0.0, K_V=self.k_v, K_R=self.K_R, Lambda=self.Lambda,
                         Gamma=self.gamma, eta=self.eta, cap_frac=self.cap_frac,
                         plane_axis=self.plane_axis)
        move_sep = self.sep_mu != 0.0
        # THIS FRAME'S LOADS ONLY: a buffer or a list from an earlier frame is never applied.
        _fr, _bc, fext, conts = _frame_loads(H, m, Nv, dev, dt)
        from plexus.operators.contact_ops import linear_contact_forces
        cov = None
        if self.e_cm > 0 and self.e_cm_from == "bound":
            _b = cell_block_t(H, self.cm_cell_set, self.bound_block, nF)
            if _b is None:
                raise ValueError(f"cell_mechanics[apicobasal_contact]: e_cm_from bound needs the width-1 block "
                                 f"{self.bound_block!r} on the set {self.cm_cell_set!r} (written by bm_clutch)")
            _b = _b.to(device=dev, dtype=dt)
            if self._b_ref is None and bool((_b > 0).any()):
                self._b_ref = float(_b[_b > 0].median())
            if self._b_ref:
                cov = (_b / self._b_ref).clamp(0.0, 1.0)
        elif self.e_cm > 0:
            _lig = m.get("bm_ligand")
            if torch.is_tensor(_lig) and m.get("bm_ligand_frame") == _fr and _lig.shape[0] == nF:
                cov = (_lig / _lig.median().clamp_min(1e-12)).clamp(0.0, 1.0).to(device=dev, dtype=dt)
        core = self._core_target(H, m, x0, s0, es, et, ef, nF, eocc)
        h_star = None
        if self.k_height > 0 and move_sep:
            if "h_star" not in m:
                m["h_star"] = self.height_target * float(2.0 * s0.norm(dim=1).median())
            h_star = m["h_star"]
        x = x0.clone(); s = s0.clone()
        kh_imp = self.kappa_h_implicit and move_sep and self.kappa_h != 0.0
        for _ in range(max(1, self.relax_iters)):
            # `kappa_h_implicit`: THE GRADIENT AT kappa_h 0 -- `_grad` and the warp kernels both read self.kappa_h
            _kh = self.kappa_h
            if kh_imp:
                self.kappa_h = 0.0
            try:
                gx, gs = self._grad(x, s, es, et, ef, nF, V_eq, m["alive"], R0t, eocc, vocc, move_sep)
            finally:
                self.kappa_h = _kh
            if cov is not None:
                gxa, gsa = self._adhesion_grad(x, s, es, et, ef, nF, cov, self.e_cm)
                gx = gx + gxa
                if gs is not None:
                    gs = gs + gsa
            if h_star is not None and not self.height_implicit:
                sn = s.norm(dim=1, keepdim=True).clamp_min(1e-12)
                gs = gs + self.k_height * (2.0 * sn - h_star) * 2.0 * s / sn
            if core is not None:
                gxc, gsc, m["core_volume"] = self._core_grad(x, s, es, et, ef, nF, eocc, core)
                gx = gx + gxc
                if gs is not None:
                    gs = gs + gsc
            if fext is not None:
                gx = gx - fext
                if gs is not None:
                    gs = gs + fext
            for C in conts:
                C["_F"], C["_fv"] = linear_contact_forces(C, x, s, C["nF"])
                _f = C["_fv"].to(dt)
                gx = gx - _f
                if gs is not None:
                    # a force f on the basal point x - s: -f on x, +f on s; on the APICAL point x + s
                    # (`bm_contact[live]` with `side: apical`, an inner cell mass): -f on both
                    if C.get("side") == "apical" and self.apical_sep == "along":
                        # `apical_sep: along` (default `full` = the line below): the separation takes only the
                        # part of the reaction ALONG itself -- it may be squeezed, not turned; the sideways part
                        # moves the mid-surface alone. With 21,760 MPM points on the apical side the full reaction
                        # turned the layer's prisms 10 -> 44 degrees off the normal in ten frames (the point
                        # interior: 3-6), and the sheared prisms read as inverted cells (Finding 104).
                        _sh = s / s.norm(dim=1, keepdim=True).clamp_min(1e-12)
                        gs = gs - (_f * _sh).sum(1, keepdim=True) * _sh
                    else:
                        gs = (gs - _f) if C.get("side") == "apical" else (gs + _f)
            step = -(self.eta * self.mu) * gx
            if self.k_bend > 0:             # IMEX positional bending, before the cap (docstring)
                step = self._bend_solve(x + step, x, es, et, eocc, self.eta * self.mu * self.k_bend,
                                        self.k_bend_cg) - x
            step = step * torch.clamp(cap / (step.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
            if self.plane_axis is not None:
                step[:, self.plane_axis] = 0.0
            x = x + step
            if move_sep:
                ds = -(self.eta * self.mu * self.sep_mu) * gs
                if kh_imp:                  # IMEX: the kappa_h term by backward Euler, BEFORE the cap (docstring)
                    ds = self._kappa_h_backward_euler(s + ds, s, es, et, eocc,
                                                      self.eta * self.mu * self.sep_mu * self.kappa_h,
                                                      self.kappa_h_jacobi) - s
                ds = ds * torch.clamp(cap_s / (ds.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
                s = s + ds
                if h_star is not None and self.height_implicit:
                    sn = s.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    u = (2.0 * sn - h_star) * math.exp(-4.0 * self.eta * self.mu * self.sep_mu * self.k_height)
                    s = s / sn * (0.5 * (h_star + u))
        out = self._emit(H, pos_full, x0, s0, x, s, Nv, move_sep, mask)
        return _book_contacts(H, m, _fr, _bc, conts, Nv, dev, out)


# ==================================================================================================
# cell_mechanics[implementation: apicobasal_region] -- region tensions, a held lumen, region swelling
# (exp 10; folded in from crypt_ops.py 2026-09-27, per experiments/INSTRUCTION.md "Code: reuse the
# operator, add a variant, land it where it belongs"). Registration and code unchanged.
#
# Region tensions and a lumen, on the apico-basal shell (exp 10, organoid crypt).
#
#     cell_mechanics[implementation: apicobasal_region]
#         the apico-basal energy, plus a per-cell apical and basal surface tension read off the cell's
#         FATE, plus a lumen volume held at a target
#
# WHY IT EXISTS. Yang et al. 2021 (Nat Cell Biol 23:733) model the intestinal organoid as a closed
# monolayer whose cells carry apical, basal and lateral surface tensions set by their REGION, crypt or
# villus, around a lumen of fixed volume (Supplementary Note, Eq. 1 and 7); crypt bulging is the crypt's
# apical tension rising to m sigma_0 with every other tension held (SN section 4). The apico-basal
# energy (`vertex_ops._apicobasal_energy_core`) has one tension, `kappa_s`, on every surface of every
# cell, and no lumen; exp 10's audit (experiments/exp10_organoid_crypt.md, "The model") found neither
# in `organ_mechanics`, `mesh_inside` or `medioapical_myosin`. This variant adds the two, and leaves the
# parent untouched: with both off it IS the parent, frame for frame, warp path included.
#
#     U = U_apicobasal(x, s; V_eq_j g_j)
#         + sum_j ( sig_a_j A_ap_j + sig_b_j A_ba_j )            region tensions
#         + 1/2 k_lumen (V_L - lumen_target V_L(0))^2             the lumen
#
# g_j is the REGION'S TARGET-VOLUME FACTOR, `v_swell: [villus, crypt]`, ramped like the tensions:
# g_j = 1 + ramp(t) (v[0] + (v[1] - v[0]) f_j - 1). Yang 2021 find the lumen shrinks because villus
# cells SWELL with its water (Fig. 7f, hypothesis iii, the most efficient for budding): single-cell
# volumes bulged -> budded, median 265 -> 414 um^3 in the villus (x1.56) and 234 -> 181 in the crypt
# (x0.77), Extended Data Fig. 8b (source-data sheet ExtendedDataFigure8B_Left, n = 946/769 and
# 385/551 cells). Without it this model's lumen can only shrink by the whole organoid contracting.
#
# A_ap_j, A_ba_j are cell j's apical and basal cap areas, the same fan areas the parent's volume is
# measured on (`apicobasal_geometry_3d`), so no null space opens between the surface priced and the
# surface enclosed. sig_j = ramp(t) (s[0] + (s[1] - s[0]) f_j), f_j the cell's fate (`fate_block`
# column `fate_col`, clipped to [0, 1]) and ramp(t) = min(1, t / sigma_ramp): the SN's "linear increase
# of the crypt apical tension with time", which fits the six bulging organoids of Yang Fig. 2c. V_L is the
# volume the INNER surface encloses (the ring facing the lumen, chosen once), V_L(0) its value on the
# first frame.
#
# FRAME 0 IS THE PARENT'S. The rest calibration (`_rest_offset`) and the first relaxation run with both
# terms off, so a crypt arm, its control and its hold arm open on the same calibrated shell; the lumen's
# V_L(0) is read from that frame's seeded positions. From frame 1, whenever a term is live, the frame
# relaxes in the parent's torch loop with autograd's gradient -- the warp kernels carry neither term.
#
# Reference: Yang, Q. et al. (2021). Cell fate coordinates mechano-osmotic forces in intestinal crypt
# formation. Nat. Cell Biol. 23:733-744 (Supplementary Note, Eq. 1, 7; section 4); Okuda, S. et al.
# (2013). Biomech. Model. Mechanobiol. 12:627-644 (the apico-basal vertex model).
# ==================================================================================================
def enclosed_ring_volume(ring, es, et, ef, nF, eocc, o):
    """|volume| enclosed by the closed surface the cell rings of `ring` [Nv, 3] tile, each ring fanned
    from its own centroid, by the divergence theorem about the fixed point `o` [3]:

        V = | (1/6) sum over half-edges (c_f - o) . ((r_s - o) x (r_t - o)) |

    The fan and the winding are `apicobasal_geometry_3d`'s caps. `o` is held fixed (detached) by the
    caller: a closed surface's volume does not depend on it. The magnitude is taken with the DETACHED
    sign of the sum, so the gradient is that of |V| and an apical-in and an apical-out tissue read the
    same positive volume."""
    cnt = torch.zeros(nF, device=ring.device, dtype=ring.dtype).index_add(0, ef, eocc).clamp(min=1e-9)
    cen = torch.zeros(nF, 3, device=ring.device, dtype=ring.dtype).index_add(
        0, ef, ring[es] * eocc[:, None]) / cnt[:, None]
    a, b, c = cen[ef] - o, ring[es] - o, ring[et] - o
    v = (torch.einsum("ij,ij->i", a, torch.cross(b, c, dim=-1)) * eocc).sum() / 6.0
    sg = torch.sign(v.detach())
    return v * torch.where(sg == 0, torch.ones_like(sg), sg)


def region_lumen_energy(pos, sep, es, et, ef, nF, alive, eocc, sig_a=None, sig_b=None, lumen=None):
    """The two added terms (module docstring), differentiable in `pos` and `sep`. 0 when both are off."""
    E = pos.new_zeros(())
    if sig_a is not None or sig_b is not None:
        _, _, A_ap, A_ba = apicobasal_geometry_3d(pos, sep, es, et, ef, nF, eocc)
        if sig_a is not None:
            E = E + (sig_a * A_ap * alive).sum()
        if sig_b is not None:
            E = E + (sig_b * A_ba * alive).sum()
    if lumen is not None:
        V_L = enclosed_ring_volume(pos + lumen["sgn"] * sep, es, et, ef, nF, eocc, lumen["o"])
        if lumen.get("V0") is not None:
            E = E + 0.5 * lumen["k"] * (V_L - lumen["V0"]) ** 2
        if lumen.get("frac") is not None:
            # THE OSMOTIC LUMEN OF A GROWING ORGANOID: 1/2 k_f V_out (V_L / V_out - f)^2, V_out the volume
            # inside the outer surface this frame (held fixed in the gradient). Its pressure on the lumen is
            # -dE/dV_L = k_f (f - V_L / V_out): k_f, a pressure, times the lumen fraction's shortfall.
            V_out = lumen["V_out"]
            E = E + 0.5 * lumen["k_f"] * V_out * (V_L / V_out - lumen["frac"]) ** 2
    return E


@register_operator("cell_mechanics", implementation="apicobasal_region", set="vertex",
                   kind="lateral", family="mechanics", title="relax thick cells with region tensions", equation=r"""$$U=U_{ab}+\sum_f\big(\sigma_a(r_f)A^{ap}_f+\sigma_b(r_f)A^{ba}_f\big)+\tfrac{k_\ell}{2}\big(V_{\mathrm{lumen}}-V^*_{\mathrm{lumen}}\big)^2$$""")
class ApicoBasalRegionShapeEnergy3D(ApicoBasalShapeEnergy3D):
    r"""`apicobasal_region` model of cell_mechanics: the apico-basal energy with region-dependent apical and basal tensions and a held lumen.
    vertex -> vertex: as cell_mechanics[apicobasal], with the extra tensions read per cell off `fate_block` column `fate_col` and a lumen pressure.

        U=U_{ab}+\sum_f\big(\sigma_a(r_f)A^{ap}_f+\sigma_b(r_f)A^{ba}_f\big)+\tfrac{k_\ell}{2}\big(V_{\mathrm{lumen}}-V^*_{\mathrm{lumen}}\big)^2
        U_{ab}         the parent's apico-basal energy
        \sigma_a, \sigma_b `sigma_a`, `sigma_b`, the extra apical and basal tensions over `kappa_s`, per region
        r_f            cell f's region, read off `fate_block`
        A^{ap}_f, A^{ba}_f the cell's apical and basal areas
        k_\ell, V^*_{lumen} `k_lumen` and the lumen target (`lumen_target`, or `lumen_frac` of the organoid volume)

    Reference: Yang, Q. et al. (2021). Nat. Cell Biol. 23:733-744 (region tensions of the intestinal organoid)."""
    PARAM_UNITS = {'sigma_a': 'energy_per_area', 'sigma_b': 'energy_per_area', 'sigma_ramp': 'frames', 'k_lumen': 'stress', 'k_lumen_frac': 'stress', 'lumen_frac': 'fraction', 'lumen_target': 'volume', 'lumen_ramp': 'frames', 'lumen_until': 'frames', 'v_swell': 'dimensionless', 'fate_col': 'count'}
    SYMBOLS = {'U_{ab}': "the parent's apico-basal energy", '\\sigma_a, \\sigma_b': '`sigma_a`, `sigma_b`, the extra apical and basal tensions over `kappa_s`, per region', 'r_f': "cell f's region, read off `fate_block`", 'A^{ap}_f, A^{ba}_f': "the cell's apical and basal areas", 'k_\\ell, V^*_{lumen}': '`k_lumen` and the lumen target (`lumen_target`, or `lumen_frac` of the organoid volume)'}
    PARAM_ROLES = {'cell_set': 'the_cell_set', 'fate_block': 'block_the_region_is_read_from', 'fate_col': 'its_column', 'k_lumen': 'stiffness_of_the_lumen_volume', 'k_lumen_frac': 'pressure_toward_the_lumen_fraction', 'lumen_frac': 'lumen_target_as_a_fraction_of_the_organoid_volume', 'lumen_ramp': 'frames_the_lumen_target_ramps_over', 'lumen_target': 'lumen_target_volume', 'lumen_until': 'frame_the_lumen_hold_ends', 'sigma_a': 'extra_apical_tension_per_region', 'sigma_b': 'extra_basal_tension_per_region', 'sigma_ramp': 'frames_the_tensions_ramp_over', 'v_swell': 'target_volume_multiplier_per_region'}
    MECHANISM_TAGS = ApicoBasalShapeEnergy3D.MECHANISM_TAGS + ["region_tension", "spontaneous_curvature",
                                                               "lumen_volume"]
    REFERENCE = ("Yang, Q. et al. (2021). Nat. Cell Biol. 23:733-744 (Supplementary Note, Eq. 1, 7); "
                 + ApicoBasalShapeEnergy3D.REFERENCE)

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)

        def _pair(v):
            if v is None:
                return None
            v = [float(x) for x in (v if isinstance(v, (list, tuple)) else [v, v])]
            if len(v) != 2:
                raise ValueError(f"cell_mechanics[apicobasal_region]: sigma_a/sigma_b take one value "
                                 f"or [villus, crypt], got {v}")
            return v
        self.sigma_a = _pair(params.get("sigma_a", None))
        self.sigma_b = _pair(params.get("sigma_b", None))
        self.fate_block = str(params.get("fate_block", "chem"))
        self.fate_col = int(params.get("fate_col", 0))
        self.cell_set = str(params.get("cell_set", "cell"))
        self.sigma_ramp = float(params.get("sigma_ramp", 0.0))
        self.k_lumen = float(params.get("k_lumen", 0.0))
        self.lumen_target = float(params.get("lumen_target", 1.0))
        _lf = params.get("lumen_frac", 0.0)
        # ONE NUMBER, OR [start, end] RAMPED LINEARLY over the frames `lumen_ramp: [from, to]`: a real
        # organoid's lumen is ~0.07 of it at 20 cells (LSTree 002-Budding) and ~0.26 before its crypt
        # bulges (Yang 2021 Fig. 1f), so a growing organoid's target must move between them.
        # A LIST of targets through the frames `lumen_ramp` (the same length) also works -- Yang's lumen
        # INFLATES and then DEFLATES (0.26 -> 0.12, Fig. 1f), which one [start, end] pair cannot say.
        self.lumen_frac_pair = ([float(x) for x in _lf] if isinstance(_lf, (list, tuple)) else [float(_lf)] * 2)
        self.lumen_frac = max(self.lumen_frac_pair)
        _lr = params.get("lumen_ramp", None)
        self.lumen_ramp = None if _lr is None else [float(x) for x in _lr]
        if self.lumen_ramp is not None and len(self.lumen_ramp) != len(self.lumen_frac_pair):
            raise ValueError(f"cell_mechanics[apicobasal_region]: lumen_ramp has {len(self.lumen_ramp)} frames, "
                             f"lumen_frac {len(self.lumen_frac_pair)} values -- one frame per value")
        self.k_lumen_frac = float(params.get("k_lumen_frac", 0.0))
        _lu = params.get("lumen_until", None)
        self.lumen_until = None if _lu is None else float(_lu)
        self.v_swell = _pair(params.get("v_swell", None))
        self._sig_a = self._sig_b = self._lumen = None

    def _extra_terms(self):
        return self._sig_a is not None or self._sig_b is not None or self._lumen is not None

    def _ramp(self, H):
        _fr = getattr(H, "frame", None)
        _fr = 0 if _fr is None else int(_fr)
        return 1.0 if self.sigma_ramp <= 0 else min(1.0, _fr / self.sigma_ramp)

    def _fate(self, H, nF, dev, dt):
        if self.cell_set not in H.levels or self.fate_block not in H.level(self.cell_set).state_schema:
            raise ValueError(
                f"cell_mechanics[apicobasal_region]: sigma_a/sigma_b/v_swell read the fate from the "
                f"`{self.fate_block}` block of the `{self.cell_set}` set, which is not declared. "
                f"Declare it (and seed it, e.g. seed_cell_chem[model: cones]).")
        blk = H.level(self.cell_set).get(self.fate_block)
        return blk[:nF, self.fate_col].to(device=dev, dtype=dt).clamp(0.0, 1.0)

    def _swell(self, H, nF, dev, dt):
        """This frame's per-cell target-volume factor g_j, or None when `v_swell` is off or ramp is 0."""
        ramp = self._ramp(H)
        if self.v_swell is None or ramp <= 0:
            return None
        f = self._fate(H, nF, dev, dt)
        return 1.0 + ramp * (self.v_swell[0] + (self.v_swell[1] - self.v_swell[0]) * f - 1.0)

    def _region_terms(self, H, m, nF, x0, s0, es, et, ef, eocc, dev, dt):
        """This frame's per-cell (sig_a, sig_b) and lumen dict, or None each when off."""
        ramp = self._ramp(H)
        sig_a = sig_b = None
        if (self.sigma_a is not None or self.sigma_b is not None) and ramp > 0:
            f = self._fate(H, nF, dev, dt)
            if self.sigma_a is not None:
                sig_a = ramp * (self.sigma_a[0] + (self.sigma_a[1] - self.sigma_a[0]) * f)
            if self.sigma_b is not None:
                sig_b = ramp * (self.sigma_b[0] + (self.sigma_b[1] - self.sigma_b[0]) * f)
        lumen = None
        _fr = getattr(H, "frame", None)
        frac_on = (self.k_lumen_frac > 0 and self.lumen_frac > 0
                   and (self.lumen_until is None or (0 if _fr is None else int(_fr)) < self.lumen_until))
        if self.k_lumen > 0 or frac_on:
            if "ab_lumen" not in m:
                # ONCE: which ring faces the lumen (the one enclosing less) and its volume now.
                o = x0.mean(dim=0).detach()
                va = float(enclosed_ring_volume(x0 + s0, es, et, ef, nF, eocc, o))
                vb = float(enclosed_ring_volume(x0 - s0, es, et, ef, nF, eocc, o))
                m["ab_lumen"] = dict(sgn=1.0 if va <= vb else -1.0, V_first=min(va, vb))
            L = m["ab_lumen"]
            o = x0.mean(dim=0).detach()
            lumen = dict(k=self.k_lumen, V0=self.lumen_target * L["V_first"] if self.k_lumen > 0 else None,
                         sgn=L["sgn"], o=o, frac=None)
            if frac_on:
                V_out = float(enclosed_ring_volume(x0 - L["sgn"] * s0, es, et, ef, nF, eocc, o))
                vals = self.lumen_frac_pair
                if self.lumen_ramp is None or len(set(vals)) == 1:
                    frac = vals[-1]
                else:
                    # PIECEWISE LINEAR through (lumen_ramp[k], lumen_frac[k]), held flat outside.
                    frac = float(np.interp(0 if _fr is None else int(_fr), self.lumen_ramp, vals))
                lumen.update(frac=frac, k_f=self.k_lumen_frac, V_out=max(V_out, 1e-12))
        return sig_a, sig_b, lumen

    def _grad(self, x, s, es, et, ef, nF, V_eq, alive, R0t, eocc, vocc, want_sep):
        """The parent's gradient when no term is live; otherwise autograd of the whole energy."""
        if not self._extra_terms():
            return super()._grad(x, s, es, et, ef, nF, V_eq, alive, R0t, eocc, vocc, want_sep)
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            s = s.detach().requires_grad_(want_sep)
            E = (_apicobasal_energy_core(x, s, es, et, ef, nF, V_eq, alive, self.k_v, self.kappa_s,
                                         self.Lambda, self.K_R, R0t, eocc, vocc, self.gamma,
                                         self.surface, kappa_h=self.kappa_h,
                                         lam_e=getattr(self, "_lam_e", None))
                 + region_lumen_energy(x, s, es, et, ef, nF, alive, eocc,
                                       self._sig_a, self._sig_b, self._lumen))
            if want_sep:
                gx, gs = torch.autograd.grad(E, (x, s))
                return torch.nan_to_num(gx), torch.nan_to_num(gs)
            gx = torch.autograd.grad(E, x)[0]
        return torch.nan_to_num(gx), None

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        self._sig_a = self._sig_b = self._lumen = None
        if m is None or self.sep_block not in lvl.state_schema or (
                self.sigma_a is None and self.sigma_b is None and self.k_lumen <= 0
                and self.v_swell is None and not (self.k_lumen_frac > 0 and self.lumen_frac > 0)):
            return super().forward(H, mask)
        pos_full = lvl.get("pos")
        Nv = int(m["Nv"]); nF = int(m["nF"]); es, et, ef = m["E_srce"], m["E_trgt"], m["E_face"]
        E = es.shape[0]; dev, dt = pos_full.device, pos_full.dtype
        x0 = pos_full[:Nv].detach().clone()
        s0 = lvl.get(self.sep_block)[:Nv].detach().clone()
        eocc = torch.ones(E, device=dev, dtype=dt); vocc = torch.ones(Nv, device=dev, dtype=dt)
        terms = self._region_terms(H, m, nF, x0, s0, es, et, ef, eocc, dev, dt)
        if "mono_delta" not in m and self.rest_calibration == "force_balance":
            return super().forward(H, mask)                   # frame 0: the parent's (module docstring)
        g = self._swell(H, nF, dev, dt)
        if all(t is None for t in terms) and g is None:
            return super().forward(H, mask)
        if m.get("basal_contacts") or torch.is_tensor(m.get("f_basal")):
            raise ValueError("cell_mechanics[apicobasal_region] does not carry the basal load or the "
                             "membrane contacts -- use implementation: apicobasal_contact for those.")
        self._sig_a, self._sig_b, self._lumen = terms
        # THE PARENT'S FORWARD, TORCH LOOP, from here: the same target volume, caps and step.
        R0t = torch.as_tensor(float(m["R0"]), dtype=dt, device=dev)
        self._R0 = float(m["R0"])
        m["mono_h"] = float(2.0 * s0.norm(dim=1).mean())
        V_eq = (m["mono_k"] * m["V0f"]).clamp(min=1e-9)
        if self.rest_calibration == "force_balance":
            V_eq = (V_eq + m["mono_delta"]).clamp(min=1e-9)
        if g is not None:
            V_eq = V_eq * g
        with torch.no_grad():
            cap = self.cap_frac * (x0[et] - x0[es]).norm(dim=-1).mean().clamp(min=1e-6)
            cap_s = self.cap_frac * s0.norm(dim=1).mean().clamp(min=1e-6)
        m["mech"] = dict(K_A=0.0, K_P=0.0, K_V=self.k_v, K_R=self.K_R, Lambda=self.Lambda,
                         Gamma=self.gamma, eta=self.eta, cap_frac=self.cap_frac,
                         plane_axis=self.plane_axis)
        move_sep = self.sep_mu != 0.0
        _myo = m.get("myo")
        self._lam_e = (_myo.detach().to(device=dev, dtype=dt)
                       if torch.is_tensor(_myo) and self.Lambda != 0.0 and _myo.shape[0] == E else None)
        x = x0.clone(); s = s0.clone()
        for _ in range(max(1, self.relax_iters)):
            gx, gs = self._grad(x, s, es, et, ef, nF, V_eq, m["alive"], R0t, eocc, vocc, move_sep)
            step = -(self.eta * self.mu) * gx
            step = step * torch.clamp(cap / (step.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
            if self.plane_axis is not None:
                step[:, self.plane_axis] = 0.0
            x = x + step
            if move_sep:
                ds = -(self.eta * self.mu * self.sep_mu) * gs
                ds = ds * torch.clamp(cap_s / (ds.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
                s = s + ds
        return self._emit(H, pos_full, x0, s0, x, s, Nv, move_sep, mask)


# ----------------------------------------------------------------------------------------------
# Every variant of `cell_mechanics` lives in this file -- they are one contract, and reading them
# side by side is how anyone can tell which one a specification is getting.
#
# `HAVE_WARP` and `_wp_launch` come from `mpm_ops`, and that is a smell worth naming rather than
# hiding: the Warp launch helper is shared infrastructure that happens to live in an MPM module, so
# a vertex operator imports from an MPM one to get it. Moving it is a separate change with its own
# blast radius.
#
# NVIDIA Warp gradient for the vertex-model shape energy: `cell_mechanics[implementation: warp]`.
# ==========================================================================================================
from plexus.operators.mpm_ops import HAVE_WARP, _wp_launch      # noqa: E402  see the note above

if HAVE_WARP:
    import warp as wp

    @wp.kernel
    def face_accum(POS: wp.array(dtype=wp.vec3), ES: wp.array(dtype=wp.int32),
                   ET: wp.array(dtype=wp.int32), EF: wp.array(dtype=wp.int32),
                   EOCC: wp.array(dtype=float),
                   N: wp.array(dtype=wp.vec3), P: wp.array(dtype=float),
                   CNT: wp.array(dtype=float), CSUM: wp.array(dtype=wp.vec3)):
        """Per-face area vector, perimeter, live-edge count and centroid sum -- `face_geometry_3d`.

        The half-sum in N_f = (1/2) sum (s x t) is NOT applied here: it is applied once per face in
        `face_scalars`, so this kernel is exactly the accumulation and the factor lives in one place.
        """
        e = wp.tid()
        w = EOCC[e]
        if w <= 0.0:                                # dead reservoir slot: contributes to nothing
            return
        f = EF[e]
        s = POS[ES[e]]
        t = POS[ET[e]]
        wp.atomic_add(N, f, wp.cross(s, t) * w)
        wp.atomic_add(P, f, wp.length(t - s) * w)
        wp.atomic_add(CNT, f, w)
        wp.atomic_add(CSUM, f, s * w)

    @wp.kernel
    def face_scalars(N: wp.array(dtype=wp.vec3), P: wp.array(dtype=float),
                     CNT: wp.array(dtype=float), CSUM: wp.array(dtype=wp.vec3),
                     A0: wp.array(dtype=float), P0: wp.array(dtype=float),
                     V0F: wp.array(dtype=float), ALIVE: wp.array(dtype=float),
                     K_A: float, K_P: float, K_V: float, Gam: float, APEX: wp.vec3,
                     G: wp.array(dtype=wp.vec3), DP: wp.array(dtype=float),
                     CG: wp.array(dtype=wp.vec3)):
        """The three per-face derivatives, folded into what the half-edge pass actually needs:
        `G` (the covector on N_f, already halved), `DP` (dE/dP_f) and `CG` (the centroid route)."""
        f = wp.tid()
        a = ALIVE[f]
        Nf = N[f] * 0.5                             # the half-sum, applied once
        area = wp.length(Nf)
        cnt = wp.max(CNT[f], 1.0)
        # THE WEDGE APEX IS THE DECLARED CENTRE, NOT THE ORIGIN -- the same `apex` that
        # `face_geometry_3d` takes. About the origin, a tissue seeded at a box centre 58 units
        # away had wedge volumes ~12x its own, and K_V crushed it from r 5.0 to 3.2 in one frame
        # while dragging it toward (0, 0, 0); the torch body, which had the apex, held r 4.66.
        centroid = CSUM[f] / cnt - APEX
        vf = wp.dot(centroid, Nf) / 3.0

        dA = 2.0 * K_A * (area - A0[f]) * a
        DP[f] = (2.0 * K_P * (P[f] - P0[f]) + Gam * P[f]) * a
        dv = 2.0 * K_V * (vf - V0F[f]) * a

        nhat = wp.vec3(0.0, 0.0, 0.0)
        if area > 1.0e-20:
            nhat = Nf / area
        # HALVED HERE, not in the half-edge kernel: dN_f/d(cross_e) = 1/2, and folding it in once
        # per face keeps the scatter pass to the three cross products it is really doing.
        G[f] = (nhat * dA + centroid * (dv / 3.0)) * 0.5
        CG[f] = Nf * (dv / (3.0 * cnt))

    @wp.kernel
    def edge_grad(POS: wp.array(dtype=wp.vec3), ES: wp.array(dtype=wp.int32),
                  ET: wp.array(dtype=wp.int32), EF: wp.array(dtype=wp.int32),
                  EOCC: wp.array(dtype=float), MYO: wp.array(dtype=float),
                  G: wp.array(dtype=wp.vec3), DP: wp.array(dtype=float),
                  CG: wp.array(dtype=wp.vec3),
                  Lam: float, Gam_l: float, has_myo: int,
                  GRAD: wp.array(dtype=wp.vec3)):
        """Scatter every half-edge's contribution onto its two endpoints."""
        e = wp.tid()
        w = EOCC[e]
        if w <= 0.0:
            return
        f = EF[e]
        i = ES[e]
        j = ET[e]
        s = POS[i]
        t = POS[j]

        # --- through the area vector and the wedge volume (both act on N_f) ---------------- #
        Gf = G[f]
        wp.atomic_add(GRAD, i, wp.cross(t, Gf) * w)
        wp.atomic_add(GRAD, j, wp.cross(Gf, s) * w)

        # --- through the centroid: cen_f averages the SOURCE vertices only ------------------ #
        wp.atomic_add(GRAD, i, CG[f] * w)

        # --- through the edge length: perimeter + line tension + junction spring ------------ #
        d = t - s
        L = wp.length(d)
        if L > 1.0e-20:
            u = d / L
            m = 1.0
            if has_myo == 1:
                m = MYO[e]
            c = (DP[f] + Lam * m + Gam_l * L) * w
            wp.atomic_add(GRAD, j, u * c)
            wp.atomic_add(GRAD, i, u * (-c))

    @wp.kernel
    def vertex_radial(POS: wp.array(dtype=wp.vec3), VOCC: wp.array(dtype=float),
                      R0: float, K_R: float, C: wp.vec3, GRAD: wp.array(dtype=wp.vec3)):
        """K_R (|x - c| - R0)^2 over live vertices, `c` the live occupancy-weighted centroid --
        the one term that is not a sum over faces, and the same reference `_shape_energy_core`
        uses (about the origin it pinned a tissue to the world corner)."""
        v = wp.tid()
        w = VOCC[v]
        if w <= 0.0:
            return
        x = POS[v] - C
        r = wp.length(x)
        if r > 1.0e-20:
            wp.atomic_add(GRAD, v, x * (2.0 * K_R * (r - R0) * w / r))


_WARNED = set()


def _warn_once(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        from plexus.paths import warn
        warn(msg)


def shape_energy_grad_warp(pos, es, et, ef, nF, A0, P0, V0f, alive, R0, K_A, K_P, K_V, K_R,
                           Lam, Gam, eocc, vocc, myo_e=None, Gam_l=0.0, buffers=None,
                           apex=None, rcentre=None):
    """dE/d(pos) for `_shape_energy_core`, in four warp kernels instead of an autograd backward.

    `buffers` -- an optional dict reused across the relax loop's iterations, so the per-face and
    per-vertex scratch is allocated once per frame rather than once per descent step. The arrays are
    zeroed here, which is a torch `zero_` per buffer and cheaper than reallocating them.
    """
    dev = pos.device
    Nv = pos.shape[0]
    E = es.shape[0]
    b = buffers if buffers is not None else {}
    if b.get("_nF") != nF or b.get("_Nv") != Nv:
        b.clear()
        b["_nF"] = nF; b["_Nv"] = Nv
        z = lambda *shape: torch.zeros(*shape, device=dev, dtype=torch.float32)  # noqa: E731
        b["N"] = z(nF, 3); b["P"] = z(nF); b["CNT"] = z(nF); b["CSUM"] = z(nF, 3)
        b["G"] = z(nF, 3); b["DP"] = z(nF); b["CG"] = z(nF, 3); b["GRAD"] = z(Nv, 3)
    for k in ("N", "P", "CNT", "CSUM", "GRAD"):
        b[k].zero_()

    wpos = wp.from_torch(pos.contiguous(), dtype=wp.vec3)
    wes = wp.from_torch(es.to(torch.int32).contiguous())
    wet = wp.from_torch(et.to(torch.int32).contiguous())
    wef = wp.from_torch(ef.to(torch.int32).contiguous())
    weo = wp.from_torch(eocc.contiguous())
    wN = wp.from_torch(b["N"], dtype=wp.vec3); wP = wp.from_torch(b["P"])
    wC = wp.from_torch(b["CNT"]); wCS = wp.from_torch(b["CSUM"], dtype=wp.vec3)
    wG = wp.from_torch(b["G"], dtype=wp.vec3); wDP = wp.from_torch(b["DP"])
    wCG = wp.from_torch(b["CG"], dtype=wp.vec3)
    wGRAD = wp.from_torch(b["GRAD"], dtype=wp.vec3)
    has_myo = 1 if myo_e is not None else 0
    wmyo = wp.from_torch((myo_e if myo_e is not None else eocc).contiguous())
    # `apex` -- the wedge apex (None: the origin, as face_geometry_3d). `rcentre` -- the point the
    # radial term is measured from (None: the live occupancy-weighted centroid, as the torch body).
    _apex = wp.vec3(0.0, 0.0, 0.0) if apex is None else wp.vec3(*[float(v) for v in apex])
    rcentre_coupled = rcentre is None                     # a live centroid, not a declared point
    if rcentre is None:
        _w = vocc.to(pos.dtype).reshape(-1, 1)
        rcentre = ((pos * _w).sum(dim=0) / _w.sum().clamp(min=1.0)).tolist()
    _rc = wp.vec3(*[float(v) for v in rcentre])

    _wp_launch(face_accum, E, dev, [wpos, wes, wet, wef, weo, wN, wP, wC, wCS])
    _wp_launch(face_scalars, nF, dev,
               [wN, wP, wC, wCS, wp.from_torch(A0.contiguous()), wp.from_torch(P0.contiguous()),
                wp.from_torch(V0f.contiguous()), wp.from_torch(alive.contiguous()),
                float(K_A), float(K_P), float(K_V), float(Gam), _apex, wG, wDP, wCG])
    _wp_launch(edge_grad, E, dev,
               [wpos, wes, wet, wef, weo, wmyo, wG, wDP, wCG,
                float(Lam), float(Gam_l), int(has_myo), wGRAD])
    _wp_launch(vertex_radial, Nv, dev,
               [wpos, wp.from_torch(vocc.contiguous()), float(R0), float(K_R), _rc, wGRAD])
    if rcentre_coupled and K_R != 0.0:
        # THE CENTROID MOVES WITH THE VERTICES. `_shape_energy_core` measures the radial term
        # from the live occupancy-weighted centroid c(x), and autograd differentiates through
        # c as well: dE/dx_i gains -(w_i / W) * sum_j g_j, the mean radial force, which is
        # zero only on a perfectly balanced shell. The kernel holds c fixed, so the term is
        # added here in torch (Nv vectors, one reduction); without it the K_R-only gradient
        # disagreed with autograd by 2.6e-3 relative on the certification sphere.
        _w = vocc.to(pos.dtype).reshape(-1, 1)
        _x = pos - torch.as_tensor(rcentre, device=dev, dtype=pos.dtype)
        _r = _x.norm(dim=1, keepdim=True).clamp_min(1e-20)
        _g = (2.0 * float(K_R)) * (_r - float(R0)) * _w * _x / _r
        b["GRAD"] -= (_w / _w.sum().clamp(min=1.0)) * _g.sum(dim=0, keepdim=True)
    return b["GRAD"]


def try_shape_energy_grad(op, p, es, et, ef, nF, A0, P0, V0f, alive, R0t, eocc, vocc,
                          twin_face=None, myo_e=None):
    """The gradient for `op` in warp, or None if this run is not one the kernels can serve.

    CALLED BY `ShapeEnergy3D._grad`, WHICH IS WHY IT IS A FUNCTION AND NOT A SUBCLASS. Warp is the
    default backend, and the default has to reach `cell_mechanics`, `cell_mechanics[model:
    marinari]` and anything else that inherits that gradient. A subclass reaches exactly one of them
    -- `marinari` derives from `ShapeEnergy3D`, so a `ShapeEnergy3DWarp` sibling would have left the
    Marinari sweep on autograd while reporting that warp was the default.

    RETURNS None RATHER THAN RAISING, so the caller simply carries on into autograd. The reason is
    printed once per distinct cause: a fallback that is invisible is a performance cliff nobody can
    find, and one that is printed per frame is noise in a 500-frame log.
    """
    why = None
    from plexus.engine import STRICT_DETERMINISM
    if STRICT_DETERMINISM:
        # STRICT MEANS STRICT, AND THESE KERNELS CANNOT HONOUR IT. `PLEXUS_STRICT_DETERMINISM` sets
        # `torch.use_deterministic_algorithms(True, warn_only=False)` precisely so that a kernel
        # with no deterministic implementation FAILS rather than quietly running the
        # nondeterministic path -- see the note at engine.py:58. `wp.atomic_add` is outside torch's
        # machinery altogether, so it would not fail and would not be deterministic either: the run
        # would claim reproducibility it does not have. That is the one outcome the flag exists to
        # prevent, so the warp path steps aside whenever it is set.
        #
        # It is also what makes a bit-identity comparison against an older commit possible at all:
        # with the flag set on both sides, both take the autograd path, where otherwise one side
        # would compare a Warp gradient against an autograd one and no row could match.
        why = "PLEXUS_STRICT_DETERMINISM is set, and wp.atomic_add is outside torch's guarantee"
    elif not HAVE_WARP:
        why = "warp is not installed"
    elif p.device.type != "cuda":
        why = f"the run is on {p.device.type}, and these kernels are CUDA-only"
    elif p.dtype != torch.float32:
        why = f"positions are {p.dtype}; the kernels are float32"
    elif getattr(op, "K_bend", 0.0) > 0:
        why = "K_bend > 0 -- the dihedral hinge is not ported"
    elif getattr(op, "K_lumen", 0.0) > 0:
        why = "K_lumen > 0 -- the global isoperimetric term is not ported"
    if why is not None:
        _warn_once(why, f"[warn] cell_mechanics: the warp gradient is unavailable because {why}, so "
                        f"this run uses autograd. The physics is identical; the speed is not.")
        return None
    if op._centre is not None:
        # EVERY TERM BUT THE RADIAL ONE IS TRANSLATION-INVARIANT, and the caller evaluates the whole
        # energy on `p - centre`; d/dp and d/d(p-centre) are the same map, so shifting the positions
        # here reproduces it exactly.
        p = p - op._centre.to(p.device, p.dtype)
    _ap = getattr(op, "_apex", None)
    if _ap is not None:                                   # centred first, as _shape_energy_core is
        p = p - _ap.to(p.device, p.dtype)
    if not hasattr(op, "_wbuf"):
        op._wbuf = {}
    g = shape_energy_grad_warp(p, es, et, ef, nF, A0, P0, V0f, alive, float(R0t),
                               op.K_A, op.K_P, op.K_V, op.K_R, op.Lambda, op.Gamma,
                               eocc, vocc, myo_e=myo_e, Gam_l=op.Gam_l, buffers=op._wbuf)
    return torch.nan_to_num(g)


# ==========================================================================================================
# NVIDIA Warp gradient for the APICO-BASAL energy: `cell_mechanics[model: apicobasal]`.
#
# WHY IT EXISTS. `spheroid_organelles_nucleus` (200 -> 702 cells, at most 1,400 live vertices) ran at
# 250 ms a frame, and 227 of those were `ApicoBasalShapeEnergy3D._grad`: thirty autograd passes over
# `_apicobasal_energy_core`, each ~780 CUDA kernels of a few microseconds' work. Profiled, the GPU was
# busy 24 ms of the 250; the rest was launching. The default model's gradient had already been
# hand-derived into four kernels (`shape_energy_grad_warp`, above); this is the same treatment for
# the polyhedron energy, in both of its degrees of freedom.
#
# THE DERIVATIVE, TERM BY TERM (`apicobasal_geometry_3d` is the function being differentiated):
#   apical = pos + sep, basal = pos - sep, so dE/dpos = dE/da + dE/db and dE/dsep = dE/da - dE/db.
#   Every term is a sum over half-edges (s -> t of face f) of a function of a_s, a_t, b_s, b_t, the
#   two cap centroids ca_f, cb_f (means of a_s, b_s over the face) and the mid-ring centroid o_f.
#   Signed volume: v6 = sum of T(p, q, r) = (p-o).((q-o)x(r-o)) over the four triangles per edge;
#     dT/dp = (q-o)x(r-o), dT/dq = (r-o)x(p-o), dT/dr = (p-o)x(q-o).
#     THE ROUTE THROUGH `o` IS NOT COMPUTED: the polyhedron is closed (two caps and a wall), and the
#     sum of a closed surface's triple products is independent of the reference point exactly, so
#     autograd's o-route sums to float32 rounding noise. The certification test measures the size of
#     that omission -- it is the reason the tolerance is relative and not zero.
#   Triangle area A = |u x v| / 2:  dA/du = (v x n)/2, dA/dv = (n x u)/2 with n the unit normal, and
#     the vertex both edges leave from gets minus their sum.
#   The cap centroids: ca_f = sum_e w_e a_s / cnt_f, so a gradient landing on ca_f is spread back to
#     every source vertex of the ring at w_e / cnt_f -- a second per-edge pass, after the first has
#     finished accumulating it (`ab_centroid_route`).
#   Perimeter, line tension and the radial spring act on the ring `surface:` names (apical / basal /
#   mid), routed to (pos, sep) by the same identity. The radial term and its centroid coupling stay
#   in torch, as they do for the default model: they are one reduction over Nv vectors.
# ==========================================================================================================
if HAVE_WARP:

    @wp.func
    def _dA_du(u: wp.vec3, v: wp.vec3):
        n = wp.cross(u, v)
        L = wp.length(n)
        if L > 1.0e-20:
            return wp.cross(v, n / L) * 0.5
        return wp.vec3(0.0, 0.0, 0.0)

    @wp.func
    def _dA_dv(u: wp.vec3, v: wp.vec3):
        n = wp.cross(u, v)
        L = wp.length(n)
        if L > 1.0e-20:
            return wp.cross(n / L, u) * 0.5
        return wp.vec3(0.0, 0.0, 0.0)

    @wp.func
    def _tri6(o: wp.vec3, p: wp.vec3, q: wp.vec3, r: wp.vec3):
        return wp.dot(p - o, wp.cross(q - o, r - o))

    @wp.func
    def _tri_area(u: wp.vec3, v: wp.vec3):
        return 0.5 * wp.length(wp.cross(u, v))

    @wp.kernel
    def ab_face_accum(POS: wp.array(dtype=wp.vec3), SEP: wp.array(dtype=wp.vec3),
                      ES: wp.array(dtype=wp.int32), EF: wp.array(dtype=wp.int32),
                      EOCC: wp.array(dtype=float),
                      CNT: wp.array(dtype=float), CSUM: wp.array(dtype=wp.vec3),
                      CA: wp.array(dtype=wp.vec3), CB: wp.array(dtype=wp.vec3)):
        """Per-face live-edge count and the three centroid sums (mid ring, apical cap, basal cap)."""
        e = wp.tid()
        w = EOCC[e]
        if w <= 0.0:
            return
        f = EF[e]
        p = POS[ES[e]]
        s = SEP[ES[e]]
        wp.atomic_add(CNT, f, w)
        wp.atomic_add(CSUM, f, p * w)
        wp.atomic_add(CA, f, (p + s) * w)
        wp.atomic_add(CB, f, (p - s) * w)

    @wp.kernel
    def ab_face_measure(POS: wp.array(dtype=wp.vec3), SEP: wp.array(dtype=wp.vec3),
                        ES: wp.array(dtype=wp.int32), ET: wp.array(dtype=wp.int32),
                        EF: wp.array(dtype=wp.int32), EOCC: wp.array(dtype=float),
                        CNT: wp.array(dtype=float), CSUM: wp.array(dtype=wp.vec3),
                        CA: wp.array(dtype=wp.vec3), CB: wp.array(dtype=wp.vec3), ring: int,
                        V6: wp.array(dtype=float), S: wp.array(dtype=float),
                        PER: wp.array(dtype=float)):
        """Six times the signed polyhedron volume, the total surface and the ring perimeter, per
        face -- the forward pass of `apicobasal_geometry_3d`, term for term."""
        e = wp.tid()
        w = EOCC[e]
        if w <= 0.0:
            return
        f = EF[e]
        i = ES[e]
        j = ET[e]
        cnt = wp.max(CNT[f], 1.0e-9)
        o = CSUM[f] / cnt
        ca = CA[f] / cnt
        cb = CB[f] / cnt
        ps = POS[i]
        pt = POS[j]
        ss = SEP[i]
        st = SEP[j]
        a_s = ps + ss
        a_t = pt + st
        b_s = ps - ss
        b_t = pt - st
        v6 = (_tri6(o, ca, a_s, a_t) + _tri6(o, cb, b_t, b_s)
              + _tri6(o, a_s, b_s, b_t) + _tri6(o, a_s, b_t, a_t))
        wp.atomic_add(V6, f, v6 * w)
        area = (_tri_area(a_s - ca, a_t - ca) + _tri_area(b_s - cb, b_t - cb)
                + _tri_area(a_t - a_s, b_t - a_s) + _tri_area(b_t - a_s, b_s - a_s))
        wp.atomic_add(S, f, area * w)
        rs = ps
        rt = pt
        if ring == 0:
            rs = a_s
            rt = a_t
        elif ring == 1:
            rs = b_s
            rt = b_t
        wp.atomic_add(PER, f, wp.length(rt - rs) * w)

    @wp.kernel
    def ab_face_scalars(V6: wp.array(dtype=float), PER: wp.array(dtype=float),
                        V_EQ: wp.array(dtype=float), ALIVE: wp.array(dtype=float),
                        SGN: wp.array(dtype=float), k_v: float, kappa_s: float, gamma: float,
                        DV6: wp.array(dtype=float), DS: wp.array(dtype=float),
                        DP: wp.array(dtype=float)):
        """The per-face covectors: dE/d(v6_f), dE/d(S_f), dE/d(perimeter_f). `SGN` is the tissue's
        polarity sign, `sign(sum_f v6_f)` with 0 -> 1, exactly as the torch body takes it."""
        f = wp.tid()
        a = ALIVE[f]
        sg = SGN[0]
        v = V6[f] / 6.0 * sg
        DV6[f] = k_v * (v - V_EQ[f]) * a * sg / 6.0
        DS[f] = kappa_s * a
        DP[f] = gamma * PER[f] * a

    @wp.kernel
    def ab_edge_grad(POS: wp.array(dtype=wp.vec3), SEP: wp.array(dtype=wp.vec3),
                     ES: wp.array(dtype=wp.int32), ET: wp.array(dtype=wp.int32),
                     EF: wp.array(dtype=wp.int32), EOCC: wp.array(dtype=float),
                     CNT: wp.array(dtype=float), CSUM: wp.array(dtype=wp.vec3),
                     CA: wp.array(dtype=wp.vec3), CB: wp.array(dtype=wp.vec3),
                     DV6: wp.array(dtype=float), DS: wp.array(dtype=float),
                     DP: wp.array(dtype=float), Lam: float, ring: int,
                     GA: wp.array(dtype=wp.vec3), GB: wp.array(dtype=wp.vec3),
                     GM: wp.array(dtype=wp.vec3),
                     GCA: wp.array(dtype=wp.vec3), GCB: wp.array(dtype=wp.vec3)):
        """Every half-edge's contribution to dE/d(apical), dE/d(basal), dE/d(mid ring) at its two
        endpoints, and to dE/d(cap centroid) of its face."""
        e = wp.tid()
        w = EOCC[e]
        if w <= 0.0:
            return
        f = EF[e]
        i = ES[e]
        j = ET[e]
        cnt = wp.max(CNT[f], 1.0e-9)
        o = CSUM[f] / cnt
        ca = CA[f] / cnt
        cb = CB[f] / cnt
        ps = POS[i]
        pt = POS[j]
        ss = SEP[i]
        st = SEP[j]
        a_s = ps + ss
        a_t = pt + st
        b_s = ps - ss
        b_t = pt - st
        cv = DV6[f] * w
        cs = DS[f] * w
        # --- volume: four triangles, three derivatives each ------------------------------- #
        cao = ca - o
        cbo = cb - o
        aso = a_s - o
        ato = a_t - o
        bso = b_s - o
        bto = b_t - o
        gca = wp.cross(aso, ato) * cv                                 # T(ca, a_s, a_t)
        gas = wp.cross(ato, cao) * cv
        gat = wp.cross(cao, aso) * cv
        gcb = wp.cross(bto, bso) * cv                                 # T(cb, b_t, b_s)
        gbt = wp.cross(bso, cbo) * cv
        gbs = wp.cross(cbo, bto) * cv
        gas = gas + wp.cross(bso, bto) * cv                           # T(a_s, b_s, b_t)
        gbs = gbs + wp.cross(bto, aso) * cv
        gbt = gbt + wp.cross(aso, bso) * cv
        gas = gas + wp.cross(bto, ato) * cv                           # T(a_s, b_t, a_t)
        gbt = gbt + wp.cross(ato, aso) * cv
        gat = gat + wp.cross(aso, bto) * cv
        # --- surface: two cap fans and two wall triangles ---------------------------------- #
        u = a_s - ca
        v = a_t - ca
        du = _dA_du(u, v) * cs
        dv = _dA_dv(u, v) * cs
        gas = gas + du
        gat = gat + dv
        gca = gca - (du + dv)
        u = b_s - cb
        v = b_t - cb
        du = _dA_du(u, v) * cs
        dv = _dA_dv(u, v) * cs
        gbs = gbs + du
        gbt = gbt + dv
        gcb = gcb - (du + dv)
        u = a_t - a_s
        v = b_t - a_s
        du = _dA_du(u, v) * cs
        dv = _dA_dv(u, v) * cs
        gat = gat + du
        gbt = gbt + dv
        gas = gas - (du + dv)
        u = b_t - a_s
        v = b_s - a_s
        du = _dA_du(u, v) * cs
        dv = _dA_dv(u, v) * cs
        gbt = gbt + du
        gbs = gbs + dv
        gas = gas - (du + dv)
        # --- perimeter + line tension on the named ring ------------------------------------ #
        c = (DP[f] + Lam) * w
        if c != 0.0:
            rs = ps
            rt = pt
            if ring == 0:
                rs = a_s
                rt = a_t
            elif ring == 1:
                rs = b_s
                rt = b_t
            d = rt - rs
            L = wp.length(d)
            if L > 1.0e-20:
                ud = d * (c / L)
                if ring == 0:
                    gat = gat + ud
                    gas = gas - ud
                elif ring == 1:
                    gbt = gbt + ud
                    gbs = gbs - ud
                else:
                    wp.atomic_add(GM, j, ud)
                    wp.atomic_add(GM, i, -ud)
        wp.atomic_add(GA, i, gas)
        wp.atomic_add(GA, j, gat)
        wp.atomic_add(GB, i, gbs)
        wp.atomic_add(GB, j, gbt)
        wp.atomic_add(GCA, f, gca)
        wp.atomic_add(GCB, f, gcb)

    @wp.kernel
    def ab_centroid_route(ES: wp.array(dtype=wp.int32), EF: wp.array(dtype=wp.int32),
                          EOCC: wp.array(dtype=float), CNT: wp.array(dtype=float),
                          GCA: wp.array(dtype=wp.vec3), GCB: wp.array(dtype=wp.vec3),
                          GA: wp.array(dtype=wp.vec3), GB: wp.array(dtype=wp.vec3)):
        """The cap centroid is the mean of the ring's SOURCE vertices: spread its gradient back."""
        e = wp.tid()
        w = EOCC[e]
        if w <= 0.0:
            return
        f = EF[e]
        k = w / wp.max(CNT[f], 1.0e-9)
        wp.atomic_add(GA, ES[e], GCA[f] * k)
        wp.atomic_add(GB, ES[e], GCB[f] * k)

    @wp.kernel
    def ab_radial_accum(POS: wp.array(dtype=wp.vec3), SEP: wp.array(dtype=wp.vec3),
                        VOCC: wp.array(dtype=float), ring: int,
                        CSUM: wp.array(dtype=wp.vec3), W: wp.array(dtype=float)):
        """The live occupancy-weighted centroid of the named ring: numerator and weight."""
        v = wp.tid()
        w = VOCC[v]
        if w <= 0.0:
            return
        r = POS[v]
        if ring == 0:
            r = POS[v] + SEP[v]
        elif ring == 1:
            r = POS[v] - SEP[v]
        wp.atomic_add(CSUM, 0, r * w)
        wp.atomic_add(W, 0, w)

    @wp.kernel
    def ab_radial_grad(POS: wp.array(dtype=wp.vec3), SEP: wp.array(dtype=wp.vec3),
                       VOCC: wp.array(dtype=float), ring: int, K_R: float, R0: float,
                       CSUM: wp.array(dtype=wp.vec3), W: wp.array(dtype=float),
                       GR: wp.array(dtype=wp.vec3), GSUM: wp.array(dtype=wp.vec3)):
        """K_R (|r - c| - R0)^2 per live vertex of the ring, and the sum of those forces, which the
        route through the centroid `c` hands back to every vertex (see `ab_combine`)."""
        v = wp.tid()
        w = VOCC[v]
        if w <= 0.0:
            GR[v] = wp.vec3(0.0, 0.0, 0.0)
            return
        r = POS[v]
        if ring == 0:
            r = POS[v] + SEP[v]
        elif ring == 1:
            r = POS[v] - SEP[v]
        c = CSUM[0] / wp.max(W[0], 1.0)
        x = r - c
        n = wp.max(wp.length(x), 1.0e-20)
        g = x * (2.0 * K_R * (n - R0) * w / n)
        GR[v] = g
        wp.atomic_add(GSUM, 0, g)

    @wp.kernel
    def ab_combine(GA: wp.array(dtype=wp.vec3), GB: wp.array(dtype=wp.vec3),
                   GM: wp.array(dtype=wp.vec3), VOCC: wp.array(dtype=float), ring: int,
                   has_radial: int, GR: wp.array(dtype=wp.vec3), GSUM: wp.array(dtype=wp.vec3),
                   W: wp.array(dtype=float),
                   GX: wp.array(dtype=wp.vec3), GS: wp.array(dtype=wp.vec3)):
        """apical = pos + sep, basal = pos - sep: dE/dpos = ga + gb (+ the mid ring's own terms),
        dE/dsep = ga - gb; then the radial spring on the named ring, with its centroid coupling
        -(w_i / W) * sum_j g_j -- autograd differentiates through `c` too."""
        v = wp.tid()
        gx = GA[v] + GB[v] + GM[v]
        gs = GA[v] - GB[v]
        if has_radial == 1:
            g = GR[v] - GSUM[0] * (VOCC[v] / wp.max(W[0], 1.0))
            gx = gx + g
            if ring == 0:
                gs = gs + g
            elif ring == 1:
                gs = gs - g
        GX[v] = gx
        GS[v] = gs

    @wp.kernel
    def ab_sep_dirichlet(SEP: wp.array(dtype=wp.vec3), ES: wp.array(dtype=wp.int32),
                         ET: wp.array(dtype=wp.int32), EOCC: wp.array(dtype=float),
                         kappa_h: float, GS: wp.array(dtype=wp.vec3)):
        """The thickness field's stiffness, 1/2 kappa_h sum_e |sep_s - sep_t|^2, added to dE/dsep.

        One half-edge, one difference: d/d(sep_s) = +kappa_h w (sep_s - sep_t) and the opposite at
        the other end. Accumulated ATOMICALLY into `GS` after `ab_combine` has written it, which is
        why this launch is last -- `ab_combine` assigns, everything after it adds. The term touches
        `sep` only; `pos` is untouched, so `GX` is not read here.
        """
        e = wp.tid()
        w = EOCC[e]
        if w <= 0.0:
            return
        i = ES[e]
        j = ET[e]
        g = (SEP[i] - SEP[j]) * (kappa_h * w)
        wp.atomic_add(GS, i, g)
        wp.atomic_add(GS, j, -g)

    @wp.func
    def _nan_to_num(v: wp.vec3):
        """`torch.nan_to_num` on a vec3: NaN -> 0, +-inf -> +-float32 max."""
        out = wp.vec3(0.0, 0.0, 0.0)
        for k in range(3):
            c = v[k]
            if wp.isnan(c):
                c = 0.0
            elif wp.isinf(c):
                if c > 0.0:
                    c = 3.4028234663852886e38
                else:
                    c = -3.4028234663852886e38
            out[k] = c
        return out

    @wp.kernel
    def ab_step(GX: wp.array(dtype=wp.vec3), GS: wp.array(dtype=wp.vec3),
                eta_mu: float, eta_mu_sep: float, CAP: wp.array(dtype=float),
                CAPS: wp.array(dtype=float), plane_axis: int, move_sep: int,
                X: wp.array(dtype=wp.vec3), S: wp.array(dtype=wp.vec3)):
        """One bounded overdamped step on both degrees of freedom, the same arithmetic as the
        torch loop in `ApicoBasalShapeEnergy3D.forward`: step = -(eta mu) g, its length capped at
        `cap`, the in-plane component zeroed when `plane_axis` is declared."""
        v = wp.tid()
        st = _nan_to_num(GX[v]) * (-eta_mu)
        st = st * wp.min(CAP[0] / (wp.length(st) + 1.0e-12), 1.0)
        if plane_axis >= 0:
            st[plane_axis] = 0.0
        X[v] = X[v] + st
        if move_sep == 1:
            ds = _nan_to_num(GS[v]) * (-eta_mu_sep)
            ds = ds * wp.min(CAPS[0] / (wp.length(ds) + 1.0e-12), 1.0)
            S[v] = S[v] + ds


def _ab_buffers(b, pos, sep, es, et, ef, nF, V_eq, alive, eocc, vocc, R0, k_v, kappa_s, Lam, K_R,
                gamma, ring, kappa_h=0.0):
    """The scratch, the static inputs and the RECORDED LAUNCHES for one topology.

    Everything the six gradient kernels read is held in buffers this dict owns, so each launch can
    be recorded once (`wp.launch(..., record_cmd=True)`) and replayed per descent step for a few
    microseconds -- `wp.launch` itself packs ~20 arguments through Python on every call and cost
    100 us a kernel, 18 ms a frame, six times the GPU time of the whole relaxation. The launches
    are re-recorded whenever a topology, a coefficient or an input tensor's identity changes.
    """
    dev = pos.device
    Nv, E = pos.shape[0], es.shape[0]
    key = (nF, Nv, E, id(es), id(et), id(ef),
           float(R0), float(k_v), float(kappa_s), float(Lam), float(K_R), float(gamma), int(ring),
           float(kappa_h))
    if b.get("_key") == key:
        # THE VALUES ARE COPIED IN, THE TABLES ARE NOT: `V_eq` is a fresh tensor every frame
        # (`mono_k * V0f + mono_delta`), so keying on its identity rebuilt everything -- eight
        # recordings and a dozen allocations -- 200 times a run for the same topology.
        b["EOCC"].copy_(eocc); b["VOCC"].copy_(vocc); b["V_EQ"].copy_(V_eq); b["ALIVE"].copy_(alive)
        return b
    b.clear()
    b["_key"] = key
    b["_pin"] = (es, et, ef)                                # keeps the ids above meaningful
    b["_nF"], b["_Nv"], b["_E"] = nF, Nv, E
    f32 = dict(device=dev, dtype=torch.float32)
    # ONE FLAT SCRATCH TENSOR, so the per-step clear is one kernel and not seventeen.
    lay = (("CNT", nF, 1), ("V6", nF, 1), ("S", nF, 1), ("PER", nF, 1), ("W", 1, 1),
           ("CSUM", nF, 3), ("CA", nF, 3), ("CB", nF, 3), ("GCA", nF, 3), ("GCB", nF, 3),
           ("GA", Nv, 3), ("GB", Nv, 3), ("GM", Nv, 3), ("RC", 1, 3), ("GSUM", 1, 3))
    acc = torch.zeros(sum(n * d for _, n, d in lay), **f32)
    b["acc"] = acc
    k = 0
    for name, n, d in lay:
        t = acc[k:k + n * d].view(n, d) if d == 3 else acc[k:k + n]
        b[name] = t
        b["w" + name] = wp.from_torch(t, dtype=wp.vec3) if d == 3 else wp.from_torch(t)
        k += n * d
    for name, n in (("DV6", nF), ("DS", nF), ("DP", nF)):
        b[name] = torch.zeros(n, **f32); b["w" + name] = wp.from_torch(b[name])
    for name in ("GX", "GS", "GR", "POS", "SEP"):
        b[name] = torch.zeros(Nv, 3, **f32); b["w" + name] = wp.from_torch(b[name], dtype=wp.vec3)
    b["SGN"] = torch.ones(1, **f32); b["wSGN"] = wp.from_torch(b["SGN"])
    b["CAP"] = torch.zeros(1, **f32); b["wCAP"] = wp.from_torch(b["CAP"])
    b["CAPS"] = torch.zeros(1, **f32); b["wCAPS"] = wp.from_torch(b["CAPS"])
    b["wES"] = wp.from_torch(es.to(torch.int32).contiguous())
    b["wET"] = wp.from_torch(et.to(torch.int32).contiguous())
    b["wEF"] = wp.from_torch(ef.to(torch.int32).contiguous())
    for name, src in (("EOCC", eocc), ("VOCC", vocc), ("V_EQ", V_eq), ("ALIVE", alive)):
        b[name] = src.detach().clone().contiguous().to(torch.float32)
        b["w" + name] = wp.from_torch(b[name])
    wdev = f"cuda:{dev.index or 0}"
    rec = lambda kern, dim, inputs: wp.launch(kern, dim=dim, device=wdev, inputs=inputs,  # noqa: E731
                                              record_cmd=True)
    has_r = 1 if K_R != 0.0 else 0
    P, Sp = b["wPOS"], b["wSEP"]
    b["launches"] = [
        rec(ab_face_accum, E, [P, Sp, b["wES"], b["wEF"], b["wEOCC"],
                               b["wCNT"], b["wCSUM"], b["wCA"], b["wCB"]]),
        rec(ab_face_measure, E, [P, Sp, b["wES"], b["wET"], b["wEF"], b["wEOCC"], b["wCNT"],
                                 b["wCSUM"], b["wCA"], b["wCB"], int(ring), b["wV6"], b["wS"],
                                 b["wPER"]]),
        # `ab_face_scalars` needs the polarity sign, taken in torch between these two -- see below
        rec(ab_face_scalars, nF, [b["wV6"], b["wPER"], b["wV_EQ"], b["wALIVE"], b["wSGN"],
                                  float(k_v), float(kappa_s), float(gamma),
                                  b["wDV6"], b["wDS"], b["wDP"]]),
        rec(ab_edge_grad, E, [P, Sp, b["wES"], b["wET"], b["wEF"], b["wEOCC"], b["wCNT"],
                              b["wCSUM"], b["wCA"], b["wCB"], b["wDV6"], b["wDS"], b["wDP"],
                              float(Lam), int(ring), b["wGA"], b["wGB"], b["wGM"],
                              b["wGCA"], b["wGCB"]]),
        rec(ab_centroid_route, E, [b["wES"], b["wEF"], b["wEOCC"], b["wCNT"], b["wGCA"],
                                   b["wGCB"], b["wGA"], b["wGB"]]),
        rec(ab_radial_accum, Nv, [P, Sp, b["wVOCC"], int(ring), b["wRC"], b["wW"]]),
        rec(ab_radial_grad, Nv, [P, Sp, b["wVOCC"], int(ring), float(K_R), float(R0),
                                 b["wRC"], b["wW"], b["wGR"], b["wGSUM"]]),
        rec(ab_combine, Nv, [b["wGA"], b["wGB"], b["wGM"], b["wVOCC"], int(ring), has_r,
                             b["wGR"], b["wGSUM"], b["wW"], b["wGX"], b["wGS"]]),
        rec(ab_sep_dirichlet, E, [Sp, b["wES"], b["wET"], b["wEOCC"], float(kappa_h), b["wGS"]]),
    ]
    b["has_radial"] = has_r
    b["has_kappa_h"] = 1 if kappa_h != 0.0 else 0
    return b


def _ab_stream(b, dev):
    """Warp's handle on TORCH'S CURRENT STREAM, cached per stream -- the same rule `_wp_launch`
    enforces (a launch on any other stream is invisible to a CUDA-graph capture)."""
    ts = torch.cuda.current_stream(dev)
    if b.get("_ts") != ts.cuda_stream:
        b["_ts"] = ts.cuda_stream
        b["_ws"] = wp.stream_from_torch(ts)
    return b["_ws"]


def _ab_grad_into(b, stream):
    """The gradient of the energy at the state held in `b["POS"]`, `b["SEP"]`, written to
    `b["GX"]`, `b["GS"]`. Six recorded launches, one clear, and the polarity sign in between."""
    b["acc"].zero_()
    L = b["launches"]
    L[0].launch(stream); L[1].launch(stream)
    # the tissue's polarity sign, taken once from the sum -- see `apicobasal_geometry_3d`
    _sg = torch.sign(b["V6"].sum())
    b["SGN"].copy_(torch.where(_sg == 0, torch.ones_like(_sg), _sg).reshape(1))
    L[2].launch(stream); L[3].launch(stream); L[4].launch(stream)
    if b["has_radial"]:
        L[5].launch(stream); L[6].launch(stream)
    L[7].launch(stream)
    if b["has_kappa_h"]:
        L[8].launch(stream)                      # adds into GS, so it goes after ab_combine writes it


def apicobasal_energy_grad_warp(pos, sep, es, et, ef, nF, V_eq, alive, R0, k_v, kappa_s, Lam,
                                K_R, gamma, eocc, vocc, surface="apical", buffers=None,
                                kappa_h=0.0):
    """(dE/dpos, dE/dsep) for `_apicobasal_energy_core`, in warp kernels instead of an autograd
    backward. `buffers` is reused across calls: scratch, converted index tables and recorded
    launches live there, rebuilt only when the topology or a coefficient changes."""
    ring = {"apical": 0, "basal": 1, "mid": 2}[surface]
    b = _ab_buffers(buffers if buffers is not None else {}, pos, sep, es, et, ef, nF, V_eq, alive,
                    eocc, vocc, R0, k_v, kappa_s, Lam, K_R, gamma, ring, kappa_h)
    b["POS"].copy_(pos); b["SEP"].copy_(sep)
    _ab_grad_into(b, _ab_stream(b, pos.device))
    return b["GX"].clone(), b["GS"].clone()


def relax_apicobasal_warp(op, x0, s0, es, et, ef, nF, V_eq, alive, R0, eocc, vocc, cap, cap_s,
                          move_sep):
    """The whole relaxation of `ApicoBasalShapeEnergy3D.forward` on the warp path: `relax_iters`
    bounded overdamped steps, each one gradient (`_ab_grad_into`) and one `ab_step`, on state that
    never leaves the buffers. Returns the relaxed (x, s)."""
    ring = {"apical": 0, "basal": 1, "mid": 2}[op.surface]
    if not hasattr(op, "_wbuf"):
        op._wbuf = {}
    b = _ab_buffers(op._wbuf, x0, s0, es, et, ef, nF, V_eq, alive, eocc, vocc, R0, op.k_v,
                    op.kappa_s, op.Lambda, op.K_R, op.gamma, ring, float(getattr(op, "kappa_h", 0.0)))
    stream = _ab_stream(b, x0.device)
    b["POS"].copy_(x0); b["SEP"].copy_(s0)
    b["CAP"].copy_(cap.reshape(1)); b["CAPS"].copy_(cap_s.reshape(1))
    if b.get("_step") is None:
        pa = -1 if op.plane_axis is None else int(op.plane_axis)
        b["_step"] = wp.launch(ab_step, dim=b["_Nv"], device=f"cuda:{x0.device.index or 0}",
                               record_cmd=True,
                               inputs=[b["wGX"], b["wGS"], float(op.eta * op.mu),
                                       float(op.eta * op.mu * op.sep_mu), b["wCAP"], b["wCAPS"],
                                       pa, 1 if move_sep else 0, b["wPOS"], b["wSEP"]])
    for _ in range(max(1, op.relax_iters)):
        _ab_grad_into(b, stream)
        b["_step"].launch(stream)
    return b["POS"].clone(), b["SEP"].clone()


def apicobasal_warp_unavailable(x):
    """Why the warp path cannot serve this run, or None -- the same reasons `try_shape_energy_grad`
    steps aside for, said once."""
    why = None
    from plexus.engine import STRICT_DETERMINISM
    if STRICT_DETERMINISM:
        why = "PLEXUS_STRICT_DETERMINISM is set, and wp.atomic_add is outside torch's guarantee"
    elif not HAVE_WARP:
        why = "warp is not installed"
    elif x.device.type != "cuda":
        why = f"the run is on {x.device.type}, and these kernels are CUDA-only"
    elif x.dtype != torch.float32:
        why = f"positions are {x.dtype}; the kernels are float32"
    if why is not None:
        _warn_once(why, f"[warn] cell_mechanics[apicobasal]: the warp gradient is unavailable "
                        f"because {why}, so this run uses autograd. The physics is identical; the "
                        f"speed is not.")
    return why


def try_apicobasal_grad(op, x, s, es, et, ef, nF, V_eq, alive, R0t, eocc, vocc):
    """(dE/dpos, dE/dsep) in warp, or None if this run is not one the kernels can serve -- the
    same contract as `try_shape_energy_grad`."""
    if apicobasal_warp_unavailable(x) is not None:
        return None
    if not hasattr(op, "_wbuf"):
        op._wbuf = {}
    # `R0t` is a 0-d CUDA tensor; `float()` of it is a device sync per call. The operator
    # publishes the same number as `_R0` before it starts a frame, so the sync is paid once.
    R0 = getattr(op, "_R0", None)
    if R0 is None:
        R0 = float(R0t)
    gx, gs = apicobasal_energy_grad_warp(x, s, es, et, ef, nF, V_eq, alive, R0, op.k_v,
                                         op.kappa_s, op.Lambda, op.K_R, op.gamma, eocc, vocc,
                                         surface=op.surface, buffers=op._wbuf,
                                         kappa_h=float(getattr(op, "kappa_h", 0.0)))
    return gx, gs


# `cell_divide[model: mpm]` -- REGISTERED HERE, AFTER `cell_divide`'s default body, and not in mpm_ops.py beside the
# helpers it calls: mpm_ops is imported first, and the first registration of a name becomes its DEFAULT, so living
# there it made every `cell_divide` without a `model:` resolve to this class (exp 11, 2026-09-27, caught by
# tests/test_bm_contact_live.py).
from plexus.operators.mpm_ops import mpm_blocks, _mpm_reset_points, _mpm_label_points  # noqa: E402

@register_operator("cell_divide", model="mpm", set="particle", kind="divide", family="population",
                   title="divide cells of material points",
                   equation=r"""$$\xi_i=(\mathbf x_i-\bar{\mathbf x}_c)\cdot\mathbf u_c\ \gtrless\ \tilde\xi_c,\qquad v_i=\frac{V_c}{2\,n},\qquad P(\text{divide per call})=p_{\rm div}$$""")
class CellDivideMPM(Structural):
    r"""`mpm` model of cell_divide: a cell made of material points divides, its points cut by the plane through their centroid normal to its long axis.
    particle cell -> two particle cells: reads the cell's point block (`points`, the containment map); writes both daughters' blocks, the second daughter in a free slot of this set.

        \xi_i=(\mathbf x_i-\bar{\mathbf x}_c)\cdot\mathbf u_c\ \gtrless\ \tilde\xi_c,\qquad v_i=\frac{V_c}{2\,n},\qquad P(\text{divide per call})=p_{\rm div}
        \xi_i          point i's position along the cell's long axis u_c from the centroid
        \tilde\xi_c    the median of xi, so each half holds half the points
        v_i            the point volume after the cut, the cell's volume over the points per daughter
        V_c, n         the cell's volume and its point count
        p_{div}        `p_div`, the per-call division probability of a marked cell

    Reference: Wang, S. et al. (2021). Cell 184:3702-3716 (Fig 2A-B)."""
    PARAM_UNITS = {'p_div': 'fraction', 'max_per_call': 'count', 'jitter': 'fraction', 'gap': 'length', 'daughter_frac': 'fraction', 'seed': 'count'}
    SYMBOLS = {'\\xi_i': "point i's position along the cell's long axis u_c from the centroid", '\\tilde\\xi_c': 'the median of xi, so each half holds half the points', 'v_i': "the point volume after the cut, the cell's volume over the points per daughter", 'V_c, n': "the cell's volume and its point count", 'p_{div}': '`p_div`, the per-call division probability of a marked cell'}
    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["points"]
    MECHANISM_TAGS = ["division", "mpm_body", "cytokinesis"]
    PARAM_ROLES = {"points": "the_material_point_set", "mark": "cell_block_that_arms_division",
                   "p_div": "per_call_probability", "daughter_mark": "cell_block_set_on_daughters",
                   "max_per_call": "cap", "jitter": "copy_offset_world", "seed": "rng_seed",
                   "daughter_frac": "daughter_volume_over_mother", 'clear_mark': 'clear_mark_on_both_daughters', 'gap': 'distance_between_the_daughters'}
    REFERENCE = "Wang, S. et al. (2021). Cell 184:3702-3716."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.points = str(params["points"])
        self.mark = params.get("mark")
        self.p_div = float(params.get("p_div", 0.1))
        self.daughter_mark = params.get("daughter_mark")
        self.clear_mark = bool(params.get("clear_mark", True))
        self.max_per_call = int(params.get("max_per_call", 8))
        self.jitter = float(params.get("jitter", 0.03))
        # `gap` (world, default 0.25 -- about the contact distance of the points of two cells): the two
        # daughters are drawn apart by gap/2 each along the axis, so no point of one starts inside the
        # other's contact range -- the halves of a cut touch across the plane, and a contact force
        # that meets two points 0.01 apart throws them (measured: one daughter's point 2.4 units out)
        self.gap = float(params.get("gap", 0.25))
        # `daughter_frac` (default 0.5: a division halves the body): each daughter's volume as a fraction of
        # the mother's. 1.0 is Wang's CONSTANT CELL SIZE -- the daughter has grown back to its mother's size by
        # its next cycle, so the interior's volume follows its cell count, where the dividing cells are. At 0.5
        # an interior dividing on its own shrank its bodies instead: exp 11 W-L, radius of gyration 0.46 -> 0.17
        # over 200 rows while the count went 430 -> 2,309, total material nearly flat (Finding 180).
        self.daughter_frac = float(params.get("daughter_frac", 0.5))
        self.seed = int(params.get("seed", 0))
        self._gen = torch.Generator(device="cpu").manual_seed(self.seed + 60013)

    def forward(self, H, mask=None):
        C, P = H.level(self.at), H.level(self.points)
        blocks, per = mpm_blocks(H, self.at, self.points)
        live = (C.occ > 0.5).detach().cpu().numpy()
        ok = live.copy()
        if self.mark:
            if self.mark not in C.state_schema:
                raise ValueError(f"cell_divide[mpm]: mark {self.mark!r} is not a block of {self.at!r}")
            ok &= C.state[:, C.state_schema[self.mark][0]].detach().cpu().numpy() > 0.5
        cand = np.flatnonzero(ok)
        if cand.size == 0:
            return {}
        u = torch.rand(cand.size, generator=self._gen, dtype=torch.float64).numpy()
        go = cand[u < self.p_div][: self.max_per_call]
        free = np.flatnonzero(~live)
        go = go[: free.size]
        if go.size == 0:
            if cand.size and free.size == 0:
                H.mpm_div_blocked = int(getattr(H, "mpm_div_blocked", 0)) + 1
            return {}
        p0, p1 = P.state_schema["pos"]
        D = p1 - p0
        for j, c in enumerate(go.tolist()):
            d = int(free[j])
            idx = blocks[c]
            X = P.state[idx, p0:p1].detach().double()
            vol = float(P.p_vol[idx].sum()) if getattr(P, "p_vol", None) is not None else 1.0
            xb = X.mean(0)
            G = ((X - xb)[:, :, None] * (X - xb)[:, None, :]).mean(0)
            w, V = torch.linalg.eigh(G)
            ax = V[:, -1]
            vbar = None
            if "vel" in P.state_schema:
                v0, v1 = P.state_schema["vel"]
                vbar = P.state[idx, v0:v1].detach().mean(0)
            # MITOTIC ROUNDING: each daughter is born a BALL of half the mother's volume, the two on the
            # mother's long axis (Hertwig's rule), gap/2 clear of the plane between them -- a daughter
            # built from the mother's half keeps a half-ball's rest shape, and every later division
            # flattens it further (measured: median aspect 1.3 -> 2.1 in two generations)
            v_d = self.daughter_frac * vol
            r_d = (3.0 * v_d / (4.0 * math.pi)) ** (1.0 / 3.0)
            new = []
            for slot, sgn in ((c, 1.0), (d, -1.0)):
                u = torch.randn(per, D, generator=self._gen, dtype=torch.float64)
                u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
                r = torch.rand(per, 1, generator=self._gen, dtype=torch.float64) ** (1.0 / D)
                cen = xb + sgn * (r_d + 0.5 * self.gap) * ax
                new.append((slot, None, cen + (u * r).to(device=cen.device, dtype=cen.dtype) * r_d))
            for slot, _pick, pos in new:
                tgt = blocks[slot]
                _mpm_reset_points(P, tgt, pos.to(device=P.state.device, dtype=P.state.dtype),
                                  torch.full((per,), v_d / per, dtype=P.state.dtype, device=P.state.device))
                if vbar is not None:
                    st = P.state.clone()
                    st[tgt, v0:v1] = vbar
                    P.state = st
                _mpm_label_points(P, tgt, slot, "pid", "cellc")
                oc = P.occ.clone(); oc[tgt] = 1.0; P.occ = oc
            # the cell level: the second daughter woken, labels copied, marks moved to the daughters
            st, oc = C.state.clone(), C.occ.clone()
            for k, (a, b) in C.state_schema.items():
                if b - a == 1 or k in ("pos",):
                    st[d, a:b] = st[c, a:b]
            c0, c1 = C.state_schema["pos"]
            for slot, pick, pos in new:
                st[slot, c0:c1] = pos.mean(0).to(st.dtype)
            if self.mark and self.clear_mark:
                st[[c, d], C.state_schema[self.mark][0]] = 0.0
            if self.daughter_mark:
                if self.daughter_mark not in C.state_schema:
                    raise ValueError(f"cell_divide[mpm]: daughter_mark {self.daughter_mark!r} is not a block of {self.at!r}")
                st[[c, d], C.state_schema[self.daughter_mark][0]] = 1.0
            oc[d] = 1.0
            C.state, C.occ = st, oc
        H.mpm_divisions = int(getattr(H, "mpm_divisions", 0)) + int(go.size)
        return {}


@register_operator("cell_divide", model="mpm_planar", set="particle", kind="divide", family="population",
                   title="divide cells in the sheet plane",
                   equation=r"""$$\mathbf u_c=\arg\max_{\mathbf u\perp\mathbf e_n}\ \mathbf u^{\!\top}G_c\,\mathbf u,\qquad \xi_i=(\mathbf x_i-\bar{\mathbf x}_c)\cdot\mathbf u_c\ \gtrless\ 0,\qquad v_i=\frac{f\,V_c}{n}$$""")
class CellDivideMPMPlanar(CellDivideMPM):
    r"""`mpm_planar` model of cell_divide: a cell of a monolayer made of material points divides in the plane of the sheet.
    particle cell -> two particle cells: as cell_divide[mpm], with the long axis taken within the sheet's plane and each daughter keeping its column.

        \mathbf u_c=\arg\max_{\mathbf u\perp\mathbf e_n}\ \mathbf u^{\!\top}G_c\,\mathbf u,\qquad \xi_i=(\mathbf x_i-\bar{\mathbf x}_c)\cdot\mathbf u_c\ \gtrless\ 0,\qquad v_i=\frac{f\,V_c}{n}
        u_c            the cell's long axis within the plane normal to `plane_axis`
        G_c            the gyration tensor of the cell's points
        e_n            the sheet normal
        \xi_i          point i's position along u_c from the centroid
        v_i, f, V_c, n the point volume, the daughter fraction, the cell volume and point count

    Reference: Aw, W. Y. et al. (2016). Curr. Biol. 26:2090-2100."""
    PARAM_UNITS = {'plane_axis': 'count'}
    SYMBOLS = {'u_c': "the cell's long axis within the plane normal to `plane_axis`", 'G_c': "the gyration tensor of the cell's points", 'e_n': 'the sheet normal', '\\xi_i': "point i's position along u_c from the centroid", 'v_i, f, V_c, n': 'the point volume, the daughter fraction, the cell volume and point count'}
    PARAM_ROLES = dict(CellDivideMPM.PARAM_ROLES, plane_axis="sheet_normal_axis",
                       nucleus="label_stiffness_radius_frac_of_each_daughters_nucleus")
    REFERENCE = "Aw, W. Y. et al. (2016). Curr. Biol. 26:2090-2100."

    def __init__(self, params, device="cpu"):
        super().__init__({k: v for k, v in params.items() if k not in ("plane_axis", "nucleus")}, device)
        self.plane_axis = int(params.get("plane_axis", 2))
        self.nucleus = params.get("nucleus")

    def forward(self, H, mask=None):
        C, P = H.level(self.at), H.level(self.points)
        blocks, per = mpm_blocks(H, self.at, self.points)
        live = (C.occ > 0.5).detach().cpu().numpy()
        ok = live.copy()
        if self.mark:
            if self.mark not in C.state_schema:
                raise ValueError(f"cell_divide[mpm_planar]: mark {self.mark!r} is not a block of {self.at!r}")
            ok &= C.state[:, C.state_schema[self.mark][0]].detach().cpu().numpy() > 0.5
        cand = np.flatnonzero(ok)
        if cand.size == 0:
            return {}
        u = torch.rand(cand.size, generator=self._gen, dtype=torch.float64).numpy()
        go = cand[u < self.p_div][: self.max_per_call]
        free = np.flatnonzero(~live)
        go = go[: free.size]
        if go.size == 0:
            if cand.size and free.size == 0:
                H.mpm_div_blocked = int(getattr(H, "mpm_div_blocked", 0)) + 1
            return {}
        p0, p1 = P.state_schema["pos"]
        D = p1 - p0
        ax_in = [a for a in range(D) if a != self.plane_axis]
        nl = self.nucleus or {}
        nlab = nl.get("label")
        for j, c in enumerate(go.tolist()):
            d = int(free[j])
            idx = blocks[c]
            X = P.state[idx, p0:p1].detach().double()
            vol = float(P.p_vol[idx].sum()) if getattr(P, "p_vol", None) is not None else 1.0
            xb = X.mean(0)
            Y = (X - xb)[:, ax_in]
            w, V = torch.linalg.eigh((Y[:, :, None] * Y[:, None, :]).mean(0))
            ax = torch.zeros(D, dtype=X.dtype, device=X.device)
            ax[ax_in] = V[:, -1]
            xi = (X - xb) @ ax
            vbar = None
            if "vel" in P.state_schema:
                v0, v1 = P.state_schema["vel"]
                vbar = P.state[idx, v0:v1].detach().mean(0)
            mu0 = la0 = None
            n_nuc = 0
            if getattr(P, "mu", None) is not None:
                if nlab and nlab in P.state_schema:
                    cyto = P.state[idx, P.state_schema[nlab][0]] < 0.5
                    n_nuc = int((~cyto).sum())
                    mu0 = float(P.mu[idx][cyto].median()) if bool(cyto.any()) else float(P.mu[idx].min())
                    la0 = float(P.la[idx][cyto].median()) if bool(cyto.any()) else float(P.la[idx].min())
                src_mu, src_la = P.mu[idx].clone(), P.la[idx].clone()
            v_d = self.daughter_frac * vol
            halves = []
            for slot, side in ((c, 1.0), (d, -1.0)):
                own = torch.nonzero(side * xi > 0).flatten()
                if own.numel() == 0:
                    own = torch.nonzero(side * xi >= 0).flatten()
                k = int(own.numel())
                reps = own[torch.arange(per, device=own.device) % k]
                pos = X[reps].clone()
                extra = torch.arange(per, device=own.device) >= k
                if bool(extra.any()):
                    jit = (torch.randn(int(extra.sum()), D, generator=self._gen, dtype=torch.float64)
                           * self.jitter).to(pos.device)
                    q = pos[extra] + jit
                    s_q = (q - xb) @ ax
                    q = q - torch.where(side * s_q < 0, s_q, torch.zeros_like(s_q))[:, None] * ax   # back on its side
                    pos[extra] = q
                pos = pos + side * 0.5 * self.gap * ax
                halves.append((slot, reps, pos))
            for slot, reps, pos in halves:
                tgt = blocks[slot]
                _mpm_reset_points(P, tgt, pos.to(device=P.state.device, dtype=P.state.dtype),
                                  torch.full((per,), v_d / per, dtype=P.state.dtype, device=P.state.device))
                if vbar is not None:
                    st = P.state.clone()
                    st[tgt, v0:v1] = vbar
                    P.state = st
                if getattr(P, "mu", None) is not None:
                    if mu0 is not None:
                        Xd = pos.to(P.state.dtype)
                        cen = Xd.mean(0)
                        rr = (Xd - cen).norm(dim=1)
                        # A WHOLE NUCLEUS EACH: the mother's count of nuclear points, the ones nearest the daughter's
                        # centre (`radius_frac` x the rms radius when the mother had none). A radius alone found no
                        # point in a narrow half-column of 48 (tests/test_exp21_mpm_sheet.py)
                        k_n = n_nuc if n_nuc > 0 else int((rr < float(nl.get("radius_frac", 0.45))
                                                           * rr.pow(2).mean().sqrt()).sum())
                        nuc = torch.zeros(per, dtype=torch.bool, device=rr.device)
                        nuc[torch.argsort(rr)[:max(1, k_n)]] = True
                        s = float(nl.get("stiffness", 1.0))
                        P.mu[tgt] = torch.where(nuc.to(P.mu.device), mu0 * s, mu0).to(P.mu.dtype)
                        P.la[tgt] = torch.where(nuc.to(P.la.device), la0 * s, la0).to(P.la.dtype)
                        st = P.state.clone()
                        st[tgt, P.state_schema[nlab][0]] = nuc.to(st.dtype).to(st.device)
                        P.state = st
                    else:
                        P.mu[tgt] = src_mu[reps].to(P.mu.dtype)
                        P.la[tgt] = src_la[reps].to(P.la.dtype)
                _mpm_label_points(P, tgt, slot, "pid", "cellc")
                oc = P.occ.clone(); oc[tgt] = 1.0; P.occ = oc
            st, oc = C.state.clone(), C.occ.clone()
            for k_, (a, b) in C.state_schema.items():
                if b - a == 1 or k_ in ("pos",):
                    st[d, a:b] = st[c, a:b]
            c0, c1 = C.state_schema["pos"]
            for slot, reps, pos in halves:
                st[slot, c0:c1] = pos.mean(0).to(st.dtype)
            if self.mark and self.clear_mark:
                st[[c, d], C.state_schema[self.mark][0]] = 0.0
            if self.daughter_mark:
                st[[c, d], C.state_schema[self.daughter_mark][0]] = 1.0
            oc[d] = 1.0
            C.state, C.occ = st, oc
        H.mpm_divisions = int(getattr(H, "mpm_divisions", 0)) + int(go.size)
        return {}


@register_operator("cell_cycle", model="mpm", set="cell", kind="lateral", family="population", title="advance the cycle of point cells", equation=r"""$$\frac{dp_c}{dt}=\frac1{T_c},\qquad T_c=(t_{g1}+t_s+t_{g2}+t_m)\,e^{\sigma g_c-\sigma^2/2},\ g_c\sim\mathcal N(0,1)$$""")
class CellCycleMPM(Lateral):
    r"""`mpm` model of cell_cycle: G1 -> S -> G2 -> M as per-cell state of a set of cells made of material points, arming cell_divide[mpm] at the end of M.
    cell -> cell (reads the points of each cell for the size checkpoint): writes in place the width-1 cell blocks `phase`, `cycle_progress`, `size0` and `mark`.

        \frac{dp_c}{dt}=\frac1{T_c},\qquad T_c=(t_{g1}+t_s+t_{g2}+t_m)\,e^{\sigma g_c-\sigma^2/2},\ g_c\sim\mathcal N(0,1)
        p_c            the cell's cycle progress, 0 at birth, 1 at the end of M
        T_c            its cycle length, in frames
        \sigma         `phase_cv`, the spread of the cycle lengths
        t_{g1}, t_s, t_{g2}, t_m the phase durations, in frames

    Reference: Howard, A. & Pelc, S. R. (1953). Heredity 6(suppl.):261-273; Powell, E. O. (1956). J. Gen. Microbiol. 15:492-511."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"Moves each cell through G1, S, G2 and M. With the default `timer` rule the cycle progress $p_c$ grows as "
        r"$\dfrac{dp_c}{dt}=\dfrac1{T_c}$, with the cycle length $T_c=(t_{g1}+t_s+t_{g2}+t_m)\,e^{\sigma g_c-\sigma^2/2}$, "
        r"$g_c\sim\mathcal N(0,1)$ drawn per cell and $\sigma$ = `phase_cv` the spread of the cycle lengths. With `rule: "
        r"doubler_mass` a cell is ready when its mass $m_c$ (in points, from cell_grow) reaches `factor` times its birth mass, "
        r"each cell's threshold carrying a per-cell error `factor_cv`. A ready cell is marked so that cell_divide acts on it. It "
        r"writes the cell blocks phase, cycle_progress, size0 and mark."
    )
    PARAM_UNITS = {'t_g1': 'frames', 't_s': 'frames', 't_g2': 'frames', 't_m': 'frames', 'phase_cv': 'fraction', 'factor': 'dimensionless', 'factor_cv': 'fraction', 'g1_gate': 'fraction', 'warmup': 'frames', 'seed': 'count'}
    SYMBOLS = {'p_c': "the cell's cycle progress, 0 at birth, 1 at the end of M", 'T_c': 'its cycle length, in frames', '\\sigma': '`phase_cv`, the spread of the cycle lengths', 't_{g1}, t_s, t_{g2}, t_m': 'the phase durations, in frames'}
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["cell_cycle", "G1_S_G2_M", "phase_progression", "size_checkpoint", "mpm_body"]
    PARAM_ROLES = {"t_g1": "G1 duration, frames", "t_s": "S duration, frames", "t_g2": "G2 duration, frames",
                   "t_m": "M duration, frames", "phase_cv": "per-cell CV on the cycle duration",
                   "mark": "cell_block_set_when_ready", "born": "cell_block_the_division_sets_on_daughters",
                   "points": "the_material_point_set", "g1_gate": "G1 exit size, fraction of the target",
                   "warmup": "frames before the clock runs", "seed": "rng_seed",
                   "rule": "timer_or_doubler_mass", "factor": "division_at_this_multiple_of_the_birth_mass", "factor_cv": "per_cell_error_on_the_threshold",
                   "mass_block": "cell_block_with_the_mass_in_points", "mass0_block": "cell_block_for_the_birth_mass"}
    REFERENCE = "Howard, A. & Pelc, S. R. (1953). Heredity 6(suppl.):261-273; Powell, E. O. (1956). J. Gen. Microbiol. 15:492-511."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.dur = [float(params.get(k, d)) for k, d in (("t_g1", 110.0), ("t_s", 80.0), ("t_g2", 40.0), ("t_m", 10.0))]
        if min(self.dur) < 0 or sum(self.dur) <= 0:
            raise ValueError("cell_cycle[mpm]: phase durations are >= 0 and sum above 0")
        self.T = sum(self.dur)
        self.bounds = np.cumsum(self.dur)[:3] / self.T
        self.cv = float(params.get("phase_cv", 0.0))
        self.mark = str(params.get("mark", "div_ready"))
        self.born = str(params.get("born", "born"))
        self.points = params.get("points")
        self.g1_gate = float(params.get("g1_gate", 0.0))
        self.warmup = float(params.get("warmup", 0.0))
        self._gen = torch.Generator(device="cpu").manual_seed(int(params.get("seed", 0)) + 911)
        self._Tc = None
        self._t = 0.0
        # `rule: doubler_mass` (exp33, 2026-10-25; default `timer` = the clock above, unchanged): a cell is READY when its MASS (the `mass_block` cell block
        # `cell_grow[mpm]` grows, in points) reaches `factor` x its own birth mass x a per-cell error -- `cell_cycle[doubler]`'s rule on the vertex set
        self.rule = str(params.get("rule", "timer"))
        if self.rule not in ("timer", "doubler_mass"):
            raise ValueError("cell_cycle[mpm]: rule is timer or doubler_mass")
        self.factor = float(params.get("factor", 2.0)); self.factor_cv = float(params.get("factor_cv", 0.0))
        self.mass_block = str(params.get("mass_block", "mass")); self.mass0_block = str(params.get("mass0_block", "mass0"))
        self._seed = int(params.get("seed", 0)); self._nb = None; self._eps = None

    def _frozen_normal(self, slot, count):
        """A standard normal that is a FUNCTION of (seed, cell slot, birth count): counter-based, so a rollout replayed from the same state
        draws the same errors (the reparameterisation trick: the noise is an input, the threshold factor x (1 + cv eps) a function of it)."""
        return float(np.random.default_rng([self._seed, int(slot), int(count)]).standard_normal())

    def _jit(self, eps):
        """The threshold multiplier of a cell: 1 + cv eps clipped to [0.4, 1.8], the vertex `cell_divide`'s `_fresh_djit` law."""
        return float(np.clip(1.0 + self.factor_cv * eps, 0.4, 1.8))

    def _forward_mass(self, H):
        C = H.level(self.at)
        sch = C.state_schema
        for b in (self.mark, self.born, self.mass_block, self.mass0_block):
            if b not in sch:
                raise ValueError(f"cell_cycle[mpm]: {self.at!r} needs a width-1 state block {b!r}")
        cm, c0, ck, cb = (sch[b][0] for b in (self.mass_block, self.mass0_block, self.mark, self.born))
        st = C.state.clone()                                            # FUNCTIONAL: a detach here cut the knobs (cell blocks) from every metric read after it (exp33 R1)
        live = (C.occ > 0.5).detach().cpu().numpy()
        n = C.state.shape[0]
        if self._nb is None:
            self._nb = np.zeros(n, np.int64)
            self._eps = np.array([self._frozen_normal(i, 0) for i in range(n)])
        mass = st[:, cm].detach().double().cpu().numpy(); m0 = st[:, c0].detach().double().cpu().numpy()
        born = (st[:, cb].detach() > 0.5).cpu().numpy() & live
        for i in np.flatnonzero(born):                                  # a daughter: half the mother's mass (the divide copied it whole), a new birth, a new error
            mass[i] *= 0.5; m0[i] = mass[i]; self._nb[i] += 1; self._eps[i] = self._frozen_normal(i, self._nb[i])
        seed_new = live & (m0 <= 0) & (mass > 0)                         # the seeded cells: the first generation's birth mass is the mass they start with
        m0[seed_new] = mass[seed_new]
        ready = live & (m0 > 0) & (mass >= self.factor * np.array([self._jit(e) for e in self._eps]) * m0)
        dev = st.device
        st[:, cm] = torch.as_tensor(mass, dtype=st.dtype, device=dev); st[:, c0] = torch.as_tensor(m0, dtype=st.dtype, device=dev)
        st[:, cb] = torch.where(torch.as_tensor(born, device=dev), torch.zeros_like(st[:, 0]), st[:, cb])
        st[:, ck] = torch.where(torch.as_tensor(ready, device=dev), torch.ones_like(st[:, 0]), st[:, ck])
        C.state = st
        return {}

    def _draw_T(self, n):
        g = torch.randn(n, generator=self._gen, dtype=torch.float64)
        return self.T * torch.exp(self.cv * g - 0.5 * self.cv * self.cv)

    def forward(self, H, mask=None):
        if self.rule == "doubler_mass":
            return self._forward_mass(H)
        C = H.level(self.at)
        sch = C.state_schema
        for b in ("cycle_progress", "phase", "size0", self.mark, self.born):
            if b not in sch:
                raise ValueError(f"cell_cycle[mpm]: {self.at!r} needs a width-1 state block {b!r}")
        col = {b: sch[b][0] for b in ("cycle_progress", "phase", "size0", self.mark, self.born)}
        st = C.state.detach().clone()
        dev = st.device
        live = (C.occ > 0.5).detach().cpu().numpy()
        p = st[:, col["cycle_progress"]].double().cpu()
        s0 = st[:, col["size0"]].double().cpu()
        born = (st[:, col[self.born]] > 0.5).cpu().numpy() & live
        if self._Tc is None:
            # THE SEED: positions in the cycle from the growing population's age density, sizes to match
            self._Tc = self._draw_T(C.n)
            u = torch.rand(C.n, generator=self._gen, dtype=torch.float64)
            p0 = -torch.log2(1.0 - 0.5 * u)
            lv = torch.as_tensor(live)
            p = torch.where(lv, p0, p)
            m = float((2.0 ** p0[lv]).mean()) if bool(lv.any()) else 1.0
            s0 = torch.where(lv, torch.full_like(s0, 1.0 / m), s0)
        if born.any():
            b = torch.as_tensor(born)
            p = torch.where(b, torch.zeros_like(p), p)
            self._Tc[b] = self._draw_T(int(b.sum()))
        dt = float(getattr(H, "dt", 1.0))
        if self._t >= self.warmup:
            rate = dt / self._Tc
            if self.g1_gate > 0.0 and self.points:
                P = H.level(self.points)
                par = P.parent.reshape(-1).long()
                pl = (P.occ > 0.5).to(torch.float64)
                J = torch.linalg.det(P.F.double()) if getattr(P, "F", None) is not None else torch.ones(P.n, dtype=torch.float64, device=P.state.device)
                pv = P.p_vol.double() * pl
                V = torch.zeros(C.n, dtype=torch.float64, device=pv.device).index_add_(0, par, pv * J).cpu()
                V0 = torch.zeros(C.n, dtype=torch.float64, device=pv.device).index_add_(0, par, pv).cpu()
                tgt = s0 * (2.0 ** p) * V0
                stuck = (p < self.bounds[0]) & (V < self.g1_gate * tgt)
                rate = torch.where(stuck, torch.zeros_like(rate), rate)
            p = torch.where(torch.as_tensor(live), (p + rate).clamp(max=1.0), p)
        ph = torch.full_like(p, 3.0)
        for k in (2, 1, 0):
            ph = torch.where(p < float(self.bounds[k]), torch.full_like(p, float(k)), ph)
        st[:, col["cycle_progress"]] = p.to(st.dtype).to(dev)
        st[:, col["phase"]] = ph.to(st.dtype).to(dev)
        st[:, col["size0"]] = s0.to(st.dtype).to(dev)
        st[:, col[self.born]] = torch.where(torch.as_tensor(born).to(dev), torch.zeros_like(st[:, 0]), st[:, col[self.born]])
        ready = torch.as_tensor(live).to(dev) & (p.to(dev) >= 1.0)
        st[:, col[self.mark]] = torch.where(ready, torch.ones_like(st[:, 0]), st[:, col[self.mark]])
        C.state = st
        self._t += dt
        return {}


# `cell_die[model: mpm_chem_low]` -- REGISTERED HERE, AFTER `cell_die`'s default body and its `chem_low` model (the first
# registration of a name becomes its DEFAULT, TRAPS T-55), beside `cell_divide[mpm]` and `cell_cycle[mpm]`.
from plexus.operators.mpm_ops import sleep_mpm_body  # noqa: E402


@register_operator("cell_die", model="mpm_chem_low", set="cell", kind="die", family="population",
                   title="remove cells of material points that are low in activator",
                   equation=r"""$$\text{mark }c:\ a_c<a_{sw}\max_k a_k;\qquad N_c\leftarrow\lfloor(1-\sigma)N_c\rfloor,\ \ \mathbf F_i\leftarrow(N_c/N_c')^{1/3}\mathbf F_i\ \ \text{while marked};\qquad\text{remove }c\text{ at }N_c\le\rho\,\bar N_0$$""")
class CellDieMPMChemLow(Structural):
    r"""`mpm_chem_low` model of cell_die: a cell made of material points dies where its activator is low, shedding its points call by call.
    cell -> its points and the cell set: reads the cell block `chem` at column `chan` and the point occupancy; marks, sheds points from the marked cells, and retires a cell whose points fall below the critical count.

        \text{mark }c:\ a_c<a_{sw}\max_k a_k;\qquad N_c\leftarrow\lfloor(1-\sigma)N_c\rfloor,\ \ \mathbf F_i\leftarrow(N_c/N_c')^{1/3}\mathbf F_i\ \ \text{while marked};\qquad\text{remove }c\text{ at }N_c\le\rho\,\bar N_0
        a_c            the cell's activator
        a_{sw}         `a_sw`, a fraction of the field's running maximum (absolute with `a_sw_rel: false`)
        N_c            the cell's point count
        \sigma         `shrink_rate`, the fraction of points shed per call
        F_i            the remaining points' deformation gradient, scaled to keep the volume
        \rho, \bar N_0 `critical_frac` and the median seed-time point count

    Reference: Monier, B. et al. (2015). Nature 518:245-248; the rule is Plexus's (`cell_die[chem_low]`)."""
    PARAM_UNITS = {'a_sw': 'fraction', 'shrink_rate': 'fraction', 'critical_frac': 'fraction', 'max_mark_frac': 'fraction', 'min_age': 'count', 'chan': 'count', 'seed': 'count', 'swell': 'dimensionless'}
    SYMBOLS = {'a_c': "the cell's activator", 'a_{sw}': "`a_sw`, a fraction of the field's running maximum (absolute with `a_sw_rel: false`)", 'N_c': "the cell's point count", '\\sigma': '`shrink_rate`, the fraction of points shed per call', 'F_i': "the remaining points' deformation gradient, scaled to keep the volume", '\\rho, \\bar N_0': '`critical_frac` and the median seed-time point count'}
    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["points"]
    MECHANISM_TAGS = ["apoptosis", "cell_elimination", "mpm_body", "die"]
    PARAM_ROLES = {"points": "the_material_point_set", "chan": "chem_column_of_the_activator",
                   "a_sw": "die_below_this_fraction_of_the_activator_maximum", "a_sw_rel": "fraction_of_the_maximum_or_absolute",
                   "max_mark_frac": "cap_on_the_fraction_of_live_cells_marked_at_once",
                   "min_age": "calls_before_a_new_cell_can_be_marked", "shrink_rate": "fraction_of_the_points_shed_per_call",
                   "critical_frac": "retire_below_this_fraction_of_the_seed_median_point_count",
                   "swell": "stretch_the_remaining_points_so_the_body_pulls_in", "bequeath": "hand_the_chemistry_to_the_neighbours",
                   "seed": "rng_seed", "block": "cell_block_of_the_signal_default_chem", "above": "mark_cells_above_the_threshold_instead_of_below"}
    REFERENCE = "Monier, B. et al. (2015). Nature 518:245-248; Plexus (discovery_okuda cell_die[chem_low])."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        if "mode" in params:
            raise ValueError("cell_die[mpm_chem_low]: `mode` is gone -- the rule is the model (`chem_low` marks cells low "
                             "in activator; this model is its body of material points)")
        self.at = params.get("_at", "cell")
        self.points = str(params["points"])
        self.chan = int(params.get("chan", 0))
        self.a_sw = float(params.get("a_sw", 0.25))
        self.a_rel = bool(params.get("a_sw_rel", True))
        self.max_mark_frac = float(params.get("max_mark_frac", 0.005))
        self.min_age = int(params.get("min_age", 4))
        self.shrink = float(params.get("shrink_rate", 0.04))
        self.crit = float(params.get("critical_frac", 0.12))
        self.swell = bool(params.get("swell", True))
        self.bequeath = bool(params.get("bequeath", True))
        # exp06 (2026-11-14), both off by default: `block` reads the signal from another cell block than `chem` (a survival factor beside a Gierer-Meinhardt pair, whose
        # width fixes `chem` at 2); `above: true` marks the cells ABOVE the threshold instead of below (death where a signal is HIGH: holes at the activator spots)
        self.block = str(params.get("block", "chem"))
        self.above = bool(params.get("above", False))
        if not 0.0 < self.shrink < 1.0:
            raise ValueError("cell_die[mpm_chem_low]: shrink_rate is a fraction in (0, 1)")
        self._rng = np.random.default_rng(int(params.get("seed", 0)) + 4409)
        self._flag = None
        self._age = None
        self._last = None
        self._nref = None

    def forward(self, H, mask=None):
        C, P = H.level(self.at), H.level(self.points)
        sch = C.state_schema
        if self.block not in sch:
            return {}
        live = (C.occ > 0.5).detach().cpu().numpy()
        nC = live.size
        occ = (P.occ > 0.5).detach().cpu().numpy()
        par = P.parent.detach().cpu().numpy()
        counts = np.bincount(par[occ], minlength=nC)
        if self._flag is None or len(self._flag) != nC:
            self._flag = np.zeros(nC, bool)
            self._age = np.full(nC, float(self.min_age))
            self._last = counts.copy()
            self._was = live.copy()
        if self._nref is None:
            self._nref = float(np.median(counts[live])) if live.any() else 0.0
        # ages: a cell born into a free slot, or one that just divided (its point count fell by a fifth), starts again
        born = live & ~self._was
        halved = live & ~self._flag & (counts < 0.8 * self._last)
        self._age[born | halved] = 0.0
        self._age[live] += 1.0
        self._was = live.copy()
        self._flag &= live
        # mark
        h0, _ = sch[self.block]
        a = C.state[:, h0 + self.chan].detach().cpu().numpy().astype(np.float64)
        n_live = int(live.sum())
        if n_live:
            amax = float(np.nanmax(a[live]))
            thr = self.a_sw * amax if self.a_rel else self.a_sw
            if not self.a_rel or amax > 1e-9:
                want = live & ~self._flag & ((a > thr) if self.above else (a < thr)) & (self._age >= self.min_age)
                room = max(1, int(self.max_mark_frac * n_live)) - int(self._flag.sum())
                fresh = np.flatnonzero(want)
                if room > 0 and fresh.size:
                    self._flag[fresh[np.argsort(-a[fresh] if self.above else a[fresh])][:room]] = True
        # shed, then retire
        D = P.state_schema["pos"][1] - P.state_schema["pos"][0]
        retire, dropped, scale = [], [], {}
        for c in np.flatnonzero(self._flag & live):
            n = int(counts[c])
            n_new = min(n - 1, int(n * (1.0 - self.shrink)))
            if n_new <= self.crit * self._nref:
                retire.append(int(c))
                continue
            mem = np.flatnonzero(occ & (par == c))
            dropped.append(self._rng.choice(mem, n - n_new, replace=False))
            scale[int(c)] = (n / n_new) ** (1.0 / D)
        dev = P.state.device
        if dropped:
            drop = torch.as_tensor(np.concatenate(dropped), device=dev, dtype=torch.long)
            p0, p1 = P.state_schema["pos"]
            c0, c1 = sch["pos"]
            st = P.state.clone()
            st[drop, p0:p1] = C.state[P.parent[drop].long(), c0:c1].to(st.dtype)
            if "vel" in P.state_schema:
                v0, v1 = P.state_schema["vel"]
                st[drop, v0:v1] = 0.0
            P.state = st
            oc = P.occ.clone(); oc[drop] = 0.0; P.occ = oc
            if getattr(P, "F", None) is not None:
                if self.swell:
                    fac = torch.ones(P.state.shape[0], device=P.F.device, dtype=P.F.dtype)
                    gone = np.zeros(P.state.shape[0], bool); gone[drop.cpu().numpy()] = True
                    for c, s in scale.items():
                        fac[torch.as_tensor(np.flatnonzero(occ & (par == c) & ~gone), device=P.F.device)] = s
                    P.F.mul_(fac[:, None, None])
                P.F[drop] = torch.eye(D, device=P.F.device, dtype=P.F.dtype)
            if getattr(P, "C", None) is not None:
                P.C[drop] = 0.0
            H.mpm_shed = int(getattr(H, "mpm_shed", 0)) + int(drop.numel())
        if retire:
            if self.bequeath and getattr(C, "edge_index", None) is not None and C.edge_index.numel():
                ei = C.edge_index.detach().cpu().numpy()
                st = C.state.clone()
                for c in retire:
                    nb = np.unique(ei[1][ei[0] == c])
                    nb = nb[live[nb] & ~np.isin(nb, retire)]
                    if nb.size and "chem" in C.state_schema:
                        c0_, c1_ = C.state_schema["chem"]
                        amount = st[c, c0_:c1_] * float(counts[c])
                        for k in nb.tolist():
                            st[k, c0_:c1_] += amount / (nb.size * max(float(counts[k]), 1.0))
                C.state = st
            sleep_mpm_body(H, self.at, self.points, retire)
            C.kill(torch.as_tensor(retire, dtype=torch.long, device=C.occ.device))
            self._flag[retire] = False
            self._was[retire] = False                    # a slot `cell_divide` refills this frame is a newborn at the next call
            H.mpm_deaths = int(getattr(H, "mpm_deaths", 0)) + len(retire)
        # bookkeeping blocks and the counts the next call compares with
        occ2 = (P.occ > 0.5).detach().cpu().numpy()
        counts2 = np.bincount(par[occ2], minlength=nC)
        if "npts" in sch or "dying" in sch:
            st = C.state.clone()
            if "npts" in sch:
                st[:, sch["npts"][0]] = torch.as_tensor(counts2, dtype=st.dtype, device=st.device)
            if "dying" in sch:
                st[:, sch["dying"][0]] = torch.as_tensor(self._flag, dtype=st.dtype, device=st.device)
            C.state = st
        self._last = counts2.copy()
        return {}
