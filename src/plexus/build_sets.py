"""Build one set of the spec into a `Level`: the build-time helpers `engine.build` calls per set.

    _assign_types       node_type [n] and the per-type parameter tables
    _default_reserve    dormant slots per parent block
    _build_edge_set     a set whose elements are connections between two endpoint sets
    _build_mesh         an empty half-edge table on a set that declares `mesh:`
    _link_mesh_maps     mesh -> half-edge set -> cell set, once every set exists
    retype              rewrite node_type and refresh the buffers derived from it
    expand_batch        a leading batch axis on every set's state, [N, W] -> [B, N, W]
"""
from __future__ import annotations

import numpy as np
import torch

from plexus.io.edges import _edges_from_file
from plexus.models.base import Hierarchy, Level
from plexus.models.registry import _ENTITY_REGISTRY
from plexus.models.state import _entity_meta, _resolve_schema


# MESH_CELL_STATE: Per-cell names the cell set may serve itself; a name not declared there stays a mesh column.
MESH_CELL_STATE = ("A0", "P0", "V0f", "alive", "age", "ndiv",
                   "Vbirth", "divjit", "phase", "phase_t", "cyc_inhib", "cyc_vprev",
                   "mg_scale", "A0_init", "P0_init", "V0f_init",
                   "apop_flag", "inhib_frac", "elong")


# _default_reserve: Dormant slots per parent block -> int.
def _default_reserve(s: dict, sname: str, per: int) -> int:
    """Dormant slots per parent block: the spec's `grow_reserve`, else the entity's
    `reserve_factor` times `per_parent`, else zero. One function for the allocation and for the
    per-block type assignment, so the two cannot disagree about the block length."""
    if "grow_reserve" in s:
        return int(s["grow_reserve"])
    _ecls = _ENTITY_REGISTRY.get(str(s.get("entity") or sname))
    _rf = float(getattr(_ecls, "RESERVE_FACTOR", 0.0) or 0.0) if _ecls is not None else 0.0
    return int(round(_rf * int(per)))


# _assign_types: Assign node_type [n] and the per-type parameter tables of a set.
def _assign_types(lvl: Level, s: dict, H: Hierarchy, device: str) -> None:
    """Assign `node_type` [n] over the buffer from the per-type `fraction` / `count`, and build the
    per-type parameter table `type_params` [T, P] that operators index (the inverse-problem target)."""
    types = s.get("types")
    if not types:
        return
    lvl.type_names = list(types.keys())
    node_type = torch.zeros(lvl.n, dtype=torch.long, device=device)
    # layout: random permutation (default); `split_x` / `split_y` tile the types along x / y (a spatial
    # partition to test a sorting force against); `ordered` gives the i-th type to the i-th element
    layout = s.get("type_layout", "random")
    type_list = list(types.values())
    if layout == "split_x":
        # live cells only: an argsort over the whole buffer would sort the dead slots (x = 0) first
        perm = torch.nonzero(lvl.occ > 0, as_tuple=False).flatten()
        perm = perm[torch.argsort(lvl.state[perm, 0])]
        total = int(perm.numel())
    elif layout == "split_y":
        # as split_x, along y; same live-index guard
        perm = torch.nonzero(lvl.occ > 0, as_tuple=False).flatten()
        perm = perm[torch.argsort(lvl.state[perm, 1])]
        total = int(perm.numel())
    elif layout == "ordered":
        # i-th type to the i-th element in declared order, reserve stays type 0 (for individually
        # placed bodies: `start:` lists one centre per body, a random permutation would decouple them)
        perm = torch.nonzero(lvl.occ > 0, as_tuple=False).flatten()
        total = int(perm.numel())
    else:
        perm = torch.randperm(lvl.n, generator=H.rng, device=device)
        total = lvl.n
    # `count:` states an exact inventory (421 membrane pieces) instead of a share; used as is, not rounded
    counted = all("count" in t for t in type_list)
    # a count on a contained set is PER PARENT: tiled per block (contiguous, in parent order, block b =
    # rows [b*per, (b+1)*per)) and shuffled inside each block, so every cell gets the same inventory
    _per = s.get("per_parent")
    if counted and "parent" in s and _per is not None and not isinstance(_per, dict):
        # block = per_parent + reserve (the reserve may come from the entity's `reserve_factor`)
        per = int(_per) + _default_reserve(s, lvl.name, int(_per))
        nblk = lvl.n // per
        if nblk * per != lvl.n:
            raise ValueError(f"{lvl.name}: {lvl.n} elements do not divide into blocks of {per}")
        pat = torch.cat([torch.full((int(t["count"]),), tid, dtype=torch.long, device=device)
                         for tid, t in enumerate(type_list)])
        # shuffle only the live head [:live_n]; the dormant reserve tail stays type 0
        live_n = min(int(pat.numel()), per)
        if layout == "ordered":
            # declared order in every block (a Dale-signed circuit reads its sign vector off it)
            head = pat[:live_n][None, :].expand(nblk, live_n)
        else:
            order = torch.argsort(torch.rand(nblk, live_n, generator=H.rng, device=device), dim=1)
            head = pat[:live_n][order]                                    # [nblk, live_n]
        tail = torch.zeros(nblk, per - live_n, dtype=torch.long, device=device)
        node_type = torch.cat([head, tail], 1).reshape(-1)
    else:
        start = 0
        for tid, t in enumerate(type_list):
            if counted:
                k = int(t["count"])
            else:
                # last type absorbs the remainder, so per-type rounding never leaves nodes unassigned
                k = (total - start) if tid == len(type_list) - 1 else int(round(t["fraction"] * total))
            node_type[perm[start:start + k]] = tid; start += k
    lvl.register_buffer("node_type", node_type)
    if all("p" in t for t in types.values()):
        P = torch.tensor([list(t["p"]) for t in types.values()], dtype=torch.float32, device=device)
        lvl.register_buffer("type_params", P)
    # broadcast per-type scalar props (move_speed, ...) to per-node buffers [n], so operators read `lvl.<prop>`
    scalar_keys = {k for t in types.values() for k, v in t.items()
                   if k not in ("fraction", "p", "core", "layers", "color") and isinstance(v, (int, float))}
    # keep the per-type table so `retype` can refresh the derived buffers when a node changes type
    lvl._type_scalars = {}
    for k in sorted(scalar_keys):
        tab = torch.tensor([float(t.get(k, 0.0)) for t in type_list], device=device)
        lvl.register_buffer(k, tab[node_type.clamp(min=0, max=tab.numel() - 1)])
        lvl._type_scalars[k] = tab
    # the raw table too, for string properties (`region: basal`)
    lvl._type_table = [dict(t) for t in type_list]
    lvl._type_fracs = torch.tensor([float(t.get("fraction", 1.0 / max(len(type_list), 1)))
                                    for t in type_list], device=device)


# retype: Rewrite a set's node_type and refresh the buffers derived from it.
def retype(lvl, node_type) -> None:
    """Rewrite a set's `node_type` and refresh the per-node buffers derived from it (`_type_scalars`),
    which are broadcast once at build and would otherwise keep the old type's values."""
    lvl.node_type = node_type
    for k, tab in (getattr(lvl, "_type_scalars", None) or {}).items():
        getattr(lvl, k)[:] = tab[node_type.clamp(min=0, max=tab.numel() - 1)]


# _build_edge_set: Build a set whose elements are connections between two endpoint sets.
def _build_edge_set(H, sname: str, s: dict, device: str) -> None:
    """Build an edge-set: a set whose elements are connections, joined to endpoint sets by
    `pre` / `post` index maps [E].

    Edges come from an inline `edges: [[pre, post(, w)], ...]` or `edges_file: <path.npz>`. The set
    is contained in `parent` (slot 0, a single network) and has its own `state:` schema (usually
    non-spatial); `weights` fill its `w` block [E, w1 - w0]."""
    if s.get("edges_file"):
        pre_a, post_a, w_a = _edges_from_file(s["edges_file"])
        E = int(pre_a.shape[0])
        pre = torch.as_tensor(np.ascontiguousarray(pre_a), dtype=torch.long, device=device)
        post = torch.as_tensor(np.ascontiguousarray(post_a), dtype=torch.long, device=device)
        weights = None if w_a is None else np.ascontiguousarray(w_a, dtype=np.float32)
    else:
        edges = s["edges"]
        E = len(edges)
        pre = torch.tensor([int(e[0]) for e in edges], dtype=torch.long, device=device)
        post = torch.tensor([int(e[1]) for e in edges], dtype=torch.long, device=device)
        # weights: a parallel `weights: [...]` list, or a 3rd element of each edge `[pre, post, w]`
        weights = s.get("weights")
        if weights is None and all(len(e) >= 3 for e in edges):
            weights = [e[2] for e in edges]
    schema = _resolve_schema(s, H.dim, sname)
    state = torch.zeros(E, schema.dim, device=device)           # [E, schema.dim]
    if weights is not None and "w" in schema:
        w0, w1 = schema["w"]
        state[:, w0:w1] = torch.as_tensor(
            np.asarray(weights, dtype=np.float32), device=device).reshape(E, w1 - w0)
    occ = torch.ones(E, device=device)
    parent_idx = torch.zeros(E, dtype=torch.long, device=device)
    _, render, depth = _entity_meta(sname, H.dim)
    lvl = Level(sname, depth=depth, state=state, occ=occ, state_schema=schema,
                parent=parent_idx, parent_name=s["parent"],
                pre=pre, post=post, pre_name=s["pre"], post_name=s["post"], role=s.get("role"))
    lvl.render = render
    _assign_types(lvl, s, H, device)
    H.add_level(lvl)


# _build_mesh: Allocate an empty half-edge table on a set that declares mesh.
def _build_mesh(lvl, s: dict, device: str) -> None:
    """Allocate an empty half-edge table on the set if it declares `mesh: half_edge`.

    Build only allocates the container (`mesh_seed` fills it), so `H.level('vertex').mesh` exists
    and answers `n_faces == 0` before any operator runs. `mesh_set` names the half-edge set whose
    `maps:` give the topology; `mesh_cell_set` is derived from it in `_link_mesh_maps`."""
    kind = s.get("mesh")
    if kind is None:
        return
    from plexus.models.mesh import MeshTable
    z = torch.empty(0, dtype=torch.long, device=device)
    lvl.mesh = MeshTable(E_srce=z, E_trgt=z.clone(), E_face=z.clone(), nF=0, Nv=0)
    lvl.mesh_set = kind
    lvl.mesh_cell_set = None            # filled by `_link_mesh_maps` once every set exists


# _link_mesh_maps: Resolve mesh -> half-edge set -> cell set, once every set exists.
def _link_mesh_maps(H, sim) -> None:
    """Second build pass: resolve `mesh:` -> half-edge set -> `maps.face` -> cell set.

    Runs after the last `add_level` because a map needs both ends. Sets `lvl.mesh_vertex_set` and
    `lvl.mesh_cell_set` (None for a 1D mesh: a curve, no `face`), and binds the per-cell state the
    cell set declares (`MESH_CELL_STATE`) so it is served from there instead of the mesh table."""
    for sname, lvl in H.levels.items():
        hs = getattr(lvl, "mesh_set", None)
        if hs is None:
            continue
        he = H.levels[hs] if hs in H.levels else None
        maps = dict(getattr(he, "maps", {}) or {}) if he is not None else {}
        if not maps:
            raise ValueError(
                f"set {sname!r} declares `mesh: {hs}` but {hs!r} has no `maps:`. The half-edge set "
                f"is what states the topology: `srce`/`trgt` into the vertex set, `face` into the "
                f"cell set.")
        lvl.mesh_vertex_set = maps["srce"]
        lvl.mesh_cell_set = maps.get("face")
        if lvl.mesh_cell_set is not None:
            cl = H.levels.get(maps["face"]) if hasattr(H.levels, "get") else (
                H.levels[maps["face"]] if maps["face"] in H.levels else None)
            m = getattr(lvl, "mesh", None)
            if cl is not None and m is not None and hasattr(m, "bind_cell_state"):
                bound = m.bind_cell_state(cl, MESH_CELL_STATE)
                if bound:
                    print(f"[build] {sname}: {len(bound)} per-cell block(s) served from "
                          f"{maps['face']!r}: {', '.join(bound)}", flush=True)
        if maps["srce"] != sname or maps["trgt"] != sname:
            raise ValueError(
                f"set {hs!r} is {sname!r}'s mesh, so its `srce`/`trgt` maps must land in {sname!r}, "
                f"not in {maps['srce']!r}/{maps['trgt']!r}. A half-edge runs between two vertices "
                f"OF THE MESH IT BELONGS TO.")


# expand_batch: Add a leading batch axis to every set's state -> B.
def expand_batch(H, B: int) -> int:
    """Give every set a leading batch axis: state [N, W] -> [B, N, W]. Returns B.

    Done after build and seed so no seeding path changes: the world is seeded once, then copied
    (`repeat`, not `expand`, which would share storage across trials). A batch is B runs of ONE world,
    so it appears nowhere in the spec language and `H.dim` is unchanged."""
    if B <= 1:
        return 1
    for lvl in H.levels.values():
        lvl.state = lvl.state.unsqueeze(0).repeat(B, *([1] * (lvl.state.dim())))
    H.batch = int(B)
    H.zero_delta()
    return int(B)
