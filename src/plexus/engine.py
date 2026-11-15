"""The generic engine: the interpreter of the spec language.

Three phases, in order:
    build(sim)     allocate every set and field the spec declares (placement, velocity, type and
                   edge keys of `sets:` are data the engine interprets, not operators)
    seed(H, sim)   run `sim.seed_ops` (the `seed:` section) exactly once: x_0 = S(theta_S)
    run(sim)       build, seed, then iterate the schedule: x_{t+1} = Phi(x_t; theta_D)

Every dynamics operator is dispatched by name; the engine never special-cases a kind inside the
schedule loop. By what they touch, the kinds are:
    lateral / aggregate / broadcast / exchange   return a per-set delta, summed and integrated once
                                                 per tick (1st vs 2nd order from the operator's `EMIT`)
    field        mutates a field in place
    rewire       rebuilds a relation
    structural   changes the entity set
    seed         not dispatched by `run`: see `seed()` (a `LATE_SEED` one: `seed_late()`, after the operators exist)
Invariant: only `_integrate` writes pos/vel, unless an operator declares
`MAY_MUTATE_INTEGRATED_STATE` (seed / structural / derived-readout); checked on tick 0.

Legacy: a `seed`-kind operator declared under `operators:` and scheduled is still accepted and
confined to the opening frames by `_seed_window` / `_resolve_gates` (deprecated, see
notes/campaigns/SEED_MIGRATION.md). New specs use `seed:`.
"""
from __future__ import annotations

import contextlib
import time
import os
import math
import numpy as np
import torch
from torch import nn

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# STRICT_DETERMINISM: Opt-in bit-reproducible kernels (PLEXUS_STRICT_DETERMINISM=1, set by the promotion gate); costs 4.4x on si_waterfall's warp frame (245 vs 55 ms, RTX A6000).
STRICT_DETERMINISM = os.environ.get("PLEXUS_STRICT_DETERMINISM", "") not in ("", "0", "false")
if STRICT_DETERMINISM:
    torch.use_deterministic_algorithms(True, warn_only=False)

from plexus.models.base import Hierarchy, Level
from plexus.models.state import spatial_schema, schema_from_spec, StateSchema, BOUNDARY_WORLD
from plexus.models.state import _entity_schema, _entity_meta, _resolve_schema   # moved to models/state.py; re-exported for callers of `engine._resolve_schema`
from plexus.models.registry import _ENTITY_REGISTRY, get_operator, get_entity, get_field, _entity_class, \
    _resolve_emit, _resolve_default_impl, _resolve_gates, _live_ids, _live_instances, MPM_WARP_DEFAULT, SEED_MAX
import plexus.operators        # noqa: F401  self-registers the operator library
import plexus.models.entities  # noqa: F401  self-registers entity state-schemas
from plexus.models.entities import DEFAULT_STATE_SCHEMA, DEFAULT_RENDER
from plexus.schema import Spec, Selector, _selector_mask
from plexus.paths import warn
from plexus import utils
from plexus.utils import quiet, _print_run_summary
from plexus.build_sets import (MESH_CELL_STATE, _default_reserve, _assign_types, retype,
                              _build_edge_set, _build_mesh, _link_mesh_maps,
                              expand_batch)
from plexus.io.edges import _edges_from_file
from plexus.run_record import _setup_recording, _assemble, record_levels, record_fields
from plexus.placement import _orbit_groups, _init_velocity, _measured_centers, _start_centers
from plexus.CUDA_graph_capture import SubstepState, run_substep_block



# build: Construct the Hierarchy (levels and fields) from a validated spec -> Hierarchy.
def build(sim: Spec, device: str = "cpu") -> Hierarchy:
    """Construct the Hierarchy (levels + fields) from a validated `sim`, in three passes.

    1. Top-level sets (no parent): place particles over the world box (`spawn` / `start` / uniform
       fill), assign per-type properties, set the initial velocity (`vel_init`).
    2. Contained sets (children): map each to its parent, scatter inside it, let the entity
       `provision` its own per-node buffers (e.g. mpm_particle's F/C/mass).
    3. Continuous fields: a grid bound to one set (one channel per coupled type); its dynamics live
       in the deposit / diffuse / decay / sense operators.

    Allocate-and-place only: `spawn` / `start` / `vel_init` / `types` are declarative data on `sets:`.
    A `seed:` operator (`seed()`) is the reusable mechanism for anything heavier (mesh construction,
    segmentation import, reaction-diffusion patterning)."""
    H = Hierarchy()
    H.config = sim
    H.rng = torch.Generator(device=device).manual_seed(sim.seed)
    H.dim = int(getattr(sim, "dim", 2))                    # the dimension contract
    H.world_size = torch.tensor([float(w) for w in getattr(sim, "world_size", [sim.world, 1.0])],
                                device=device)             # per-axis box [w0 .. w_{D-1}]
    H.world_width = float(H.world_size[0])                 # legacy scalar (axis-0 width)
    H.boundary = sim.boundary                              # 'periodic' (wrap) | 'wall' (clamp) | 'free'/'none'/'open' (unbounded)
    H.periodic = (sim.boundary == "periodic")
    H.dt = float(sim.dt)                        # per-FRAME sim time; substeps read H.sub_dt
    # frame number as a device tensor: a CUDA graph bakes in python ints, so frame-dependent operators read this
    H.frame_t = torch.zeros((), device=device)
    H.obstacles = list(getattr(sim, "obstacles", []) or [])   # wall rects/discs for the `bounce` op
    # names of the scheduled operators, so an operator can ask what else is scheduled (e.g. `cell_grow`'s
    # `vth_frac` ceiling must stand down once `cell_divide` is scheduled)
    H.scheduled_ops = frozenset(o.op for o in (getattr(sim, "operators", None) or [])
                                if getattr(o, "op", None))
    # sets a seed places in space (see `_resolve_schema(spatial=)`)
    _spatial = {o.on.set for o in list(getattr(sim, "seed_ops", None) or []) + list(sim.operators or [])
                if getattr(o, "op", None) == "seed_positions"}

    # pass 1: top-level sets (no parent) -- positions seeded across the domain.
    for sname, s in sim.sets.items():
        _frame_meta = None            # set by `frame:`; hung on the Level below
        if "parent" in s:
            continue
        n = int(s["n"])
        D = H.dim
        buffer = int(s.get("buffer", n))               # allocated slots (occupancy marks live subset)
        _, render, depth = _entity_meta(sname, D, s.get("entity"))   # render + depth from the registry
        schema = _resolve_schema(s, D, sname, spatial=sname in _spatial)          # StateSchema: `state:` block, else the entity's, else pos/vel
        dim = schema.dim
        state = torch.zeros(buffer, dim, device=device)
        has_pos = "pos" in schema                      # spatial sets place positions; a non-spatial set (voltage,...) does not
        head = None
        gid = None                                     # group id per particle (only the pair spawns)
        grot = None
        if has_pos:
            px0, px1 = schema["pos"]
            # positions come from the `seed_positions` operator (its `_spawn*` rules live in operators/seed_ops.py); old `spawn:` keys are refused
            _legacy = [k for k in s if k == "spawn" or str(k).startswith("spawn_")]
            if _legacy:
                raise ValueError(
                    f"sets.{sname}: {', '.join(_legacy)} -- `spawn` is no longer a property of a "
                    f"set. Where a set's entities start is a seed operator:\n"
                    f"  seed:\n"
                    f"    - {{op: seed_positions, at: {sname}, mode: {s.get('spawn', '...')!r}, "
                    f"radius: {s.get('spawn_radius', '...')}}}\n"
                    f"(`two_disks` takes separation / offset / tilt / thickness / arms; the set keeps "
                    f"`n`, `state`, `types`, `vel_init`.)")
            if s.get("frame") is not None and "start" not in s:
                pos, _frame_meta = _measured_centers(sname, s, n, device)
            elif "start" in s:
                pos = _start_centers(s["start"], n, H.rng, device)  # known locations (e.g. an MPM blob)
            else:
                pos = torch.rand(n, D, generator=H.rng, device=device) * H.world_size   # uniform in the box
            state[:n, px0:px1] = pos
        vinit = s.get("vel_init", 0.0)                      # random initial speed (e.g. boids start moving)
        if not isinstance(vinit, dict) and float(vinit or 0.0) > 0 and "vel" in schema:
            vx0, vx1 = schema["vel"]
            state[:n, vx0:vx1] = (torch.rand(n, D, generator=H.rng, device=device) - 0.5) * (2 * float(vinit))
        occ = torch.zeros(buffer, device=device); occ[:n] = 1.0
        # runtime set: state [buffer, W] (pos|vel columns per the schema), live mask `occ` [buffer];
        # operators read and write it via H.level(name)
        lvl = Level(sname, depth=depth, state=state, occ=occ, state_schema=schema)
        lvl.frame = _frame_meta                                 # None unless the set declared `frame:`
        lvl.render = render
        lvl.vmax = float(s["vmax"]) if "vmax" in s else None    # optional per-tick cell speed cap
        lvl.maps = dict(s.get("maps") or {})    # declared functions OUT of this set (role -> set)
        _build_mesh(lvl, s, device)                             # `mesh: <set>` -> an empty table to fill
        if head is not None:
            # heading: unit vector [buffer, D] in every dimension (read by glide / bounce / sense)
            hbuf = torch.zeros(buffer, head.shape[1], device=device)
            hbuf[:n] = head
            lvl.register_buffer("heading", hbuf)
        if gid is not None:
            # which disc each particle came from, and the plane it was placed in (read by `vel_init`)
            gbuf = torch.zeros(buffer, dtype=torch.long, device=device)
            gbuf[:n] = gid
            lvl.register_buffer("spawn_group", gbuf)
            lvl.spawn_group_rot = grot
        _assign_types(lvl, s, H, device)
        lvl.types_raw = s.get("types")          # raw per-type config (layers/material/block) for child provisioning
        # an entity may provision a ROOT set too (mpm_particle gets F, C, mass, mu, la, p_vol there);
        # `parent=None` is passed explicitly, so an entity that needs a parent says so
        _ent_r = _entity_class(sname, s.get("entity"))
        _prov_r = getattr(_ent_r, "provision", None) if _ent_r is not None else None
        if _prov_r is not None:
            _prov_r(lvl, None, s, H, device)
        # a computed velocity IC reads x_0, final only after the seeds have run: stash it, `run()`
        # applies it right after `seed()`
        lvl._vel_init = vinit if (isinstance(vinit, dict) and "vel" in schema) else None
        H.add_level(lvl)

    # pass 2: contained sets. Each child is mapped to its parent (`parent` index + `parent_name`) and
    # scattered within it; one parent may have many child sets (membrane, cytoplasm, nucleus, ...)
    for sname, s in sim.sets.items():
        if "parent" not in s:
            continue
        pname = s["parent"]
        if pname not in H.levels:
            raise ValueError(f"set {sname!r} has parent {pname!r}, which is not a declared set")
        parent = H.level(pname)
        if s.get("edge_set"):                                     # an edge-set: elements are connections (pre/post), not scattered in space
            _build_edge_set(H, sname, s, device)
            continue
        # `per_parent`: an int (every parent holds the same count) or a {parent type: count} mapping
        # (unequal parents, e.g. a thin membrane patch vs a nucleus)
        per_map = s["per_parent"]
        if isinstance(per_map, dict):
            _tn = list(getattr(parent, "type_names", []) or [])
            if not _tn:
                raise ValueError(
                    f"set {sname!r} gives `per_parent` as a mapping, but its parent {pname!r} "
                    f"declares no `types:` -- there are no parent kinds to key it by.")
            _missing = [t for t in _tn if t not in per_map]
            _extra = [t for t in per_map if t not in _tn]
            if _missing or _extra:
                raise ValueError(
                    f"set {sname!r} `per_parent` mapping does not match {pname!r}'s types: "
                    f"missing {_missing or '-'}, unknown {_extra or '-'}. Every parent must be "
                    f"told how many children it holds.")
            if int(s.get("grow_reserve", 0)):
                raise ValueError(
                    f"set {sname!r} combines a `per_parent` mapping with `grow_reserve`. The "
                    f"reserve is a fixed dormant tail on every parent's block, and with unequal "
                    f"blocks there is no single tail length -- declare one or the other.")
            per_vec = torch.tensor([int(per_map[_tn[int(t)]]) for t in parent.node_type.tolist()],
                                   dtype=torch.long, device=device)
            per = None                                       # no single count; `per_vec` is the answer
        else:
            per = int(per_map)
            per_vec = None
        radius = float(s.get("radius", 0.02))
        # `radius_inner` makes a shell (membrane) instead of a ball; 0 = solid ball
        r_in = float(s.get("radius_inner", 0.0))
        # dormant slots per parent: `grow_reserve`, else the entity's `reserve_factor` x `per_parent`, else 0
        reserve = _default_reserve(s, sname, per) if per is not None else int(s.get("grow_reserve", 0))
        per_tot = None if per is None else per + reserve
        _, render, depth = _entity_meta(sname, H.dim, s.get("entity"))  # render + depth from the registry
        schema = _resolve_schema(s, H.dim, sname, spatial=sname in _spatial)      # StateSchema: `state:` block, else the entity's, else pos/vel
        dim = schema.dim
        has_pos = "pos" in schema                                 # spatial child: scatter in space; non-spatial (voltage,...) child: no placement
        if per_vec is None:
            Np = parent.n * per_tot                               # `per` live + `reserve` dormant per parent slot
            parent_idx = torch.arange(parent.n, device=device).repeat_interleave(per_tot)
        else:
            # unequal but contiguous blocks in parent order: parent p's children are rows [offset[p], offset[p] + per_vec[p])
            Np = int(per_vec.sum())
            parent_idx = torch.arange(parent.n, device=device).repeat_interleave(per_vec)
            lvl_per_particle = per_vec[parent_idx]                 # this child's block size, per child
            s = {**s, "_per_vec": per_vec, "_per_particle": lvl_per_particle}
        state = torch.zeros(Np, dim, device=device)
        D = H.dim                                                # the child's pos dimension (the global dim contract)
        if has_pos:
            px0, px1 = schema["pos"]
            # `parent_pos: <block>` names the parent block children scatter about (default `pos`; a vertex-tissue cell has `centroid`)
            _pp = str(s.get("parent_pos", "pos"))
            if _pp not in parent.state_schema:
                raise ValueError(f"set {sname!r}: parent {pname!r} has no {_pp!r} block to scatter about; "
                                 f"declare `parent_pos: <block>` on the child set")
            ppos = parent.get(_pp)[parent_idx][:, :D]                # parent position, projected to the child's dim
            # scatter uniformly in a ball (2D: annulus) of `radius` about the parent; the 2D polar path
            # is kept for bit-identical MPM seeding
            if D == 2:
                # annulus: sample sqrt(u) between the squared radii for a flat density
                _u = torch.rand(Np, generator=H.rng, device=device)
                r = torch.sqrt(r_in ** 2 + _u * (radius ** 2 - r_in ** 2)) if r_in > 0 \
                    else torch.sqrt(_u) * radius
                th = torch.rand(Np, generator=H.rng, device=device) * 2 * math.pi
                offset = torch.stack([r * torch.cos(th), r * torch.sin(th)], 1)
            else:
                d = torch.randn(Np, D, generator=H.rng, device=device)        # isotropic direction
                d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-9)
                # D-ball / D-annulus: the D-th root of a uniform draw between radii^D is flat in volume
                _u = torch.rand(Np, generator=H.rng, device=device)
                r = (r_in ** D + _u * (radius ** D - r_in ** D)).pow(1.0 / D) if r_in > 0 \
                    else _u.pow(1.0 / D) * radius
                offset = d * r[:, None]
            state[:, px0:px1] = ppos + offset
            # `vel_init` on a contained set: one coherent launch velocity per parent body, shared by its
            # particles (aliases `vel_init_cell`, `vel_init_cube`)
            _vi = s.get("vel_init", s.get("vel_init_cell", s.get("vel_init_cube", 0.0)))
            # `vel_init: {mode: cone, speed, axis, spread_deg, speed_jitter}`: each parent launches at `speed`
            # along `axis`, tilted up to `spread_deg` about a random azimuth (a spray of jets)
            _cone = isinstance(_vi, dict) and str(_vi.get("mode", "")).lower() == "cone"
            vcell = 0.0 if isinstance(_vi, dict) else float(_vi)
            if (_cone or vcell > 0) and "vel" in schema:
                vx0, vx1 = schema["vel"]
                vc = getattr(parent, "_launch_v", None)
                if vc is None or vc.shape != (parent.n, D):
                    if _cone:
                        ax = int(_vi.get("axis", 1))
                        sp = float(_vi.get("speed", 1.0))
                        half = math.radians(float(_vi.get("spread_deg", 30.0)))
                        jit = float(_vi.get("speed_jitter", 0.0))
                        lat = [i for i in range(D) if i != ax]
                        th = torch.rand(parent.n, generator=H.rng, device=device) * half
                        vc = torch.zeros(parent.n, D, device=device)
                        vc[:, ax] = torch.cos(th)
                        if len(lat) == 1:                               # 2-D: one side or the other
                            sgn = torch.where(torch.rand(parent.n, generator=H.rng, device=device) < 0.5, -1.0, 1.0)
                            vc[:, lat[0]] = torch.sin(th) * sgn
                        else:                                           # 3-D: a random azimuth
                            phi = torch.rand(parent.n, generator=H.rng, device=device) * (2 * math.pi)
                            vc[:, lat[0]] = torch.sin(th) * torch.cos(phi)
                            vc[:, lat[1]] = torch.sin(th) * torch.sin(phi)
                        mag = sp * (1.0 + jit * (torch.rand(parent.n, generator=H.rng, device=device) - 0.5) * 2.0)
                        vc = vc * mag[:, None]
                    else:
                        vc = (torch.rand(parent.n, D, generator=H.rng, device=device) - 0.5) * (2 * vcell)
                        _axes = s.get("vel_init_axes")
                        if _axes is not None:
                            keep = torch.zeros(D, device=device)
                            for _a in _axes:
                                keep[int(_a)] = 1.0
                            vc = vc * keep
                    # `types: [..]` launches only those bodies; `per_type: {t: {speed, spread_deg}}` gives one cone per type
                    _pt = (_vi.get("per_type") if isinstance(_vi, dict) else None) or {}
                    if _pt and _cone:
                        tn = list(getattr(parent, "type_names", []) or [])
                        pnt = getattr(parent, "node_type", None)
                        for t, kw in _pt.items():
                            if t not in tn or pnt is None:
                                continue
                            sel = (pnt == tn.index(t))
                            if not bool(sel.any()):
                                continue
                            sp_t = float(kw.get("speed", sp))
                            half_t = math.radians(float(kw.get("spread_deg", float(_vi.get("spread_deg", 30.0)))))
                            th_t = torch.rand(int(sel.sum()), generator=H.rng, device=device) * half_t
                            d = torch.zeros(int(sel.sum()), D, device=device)
                            d[:, ax] = torch.cos(th_t)
                            if len(lat) == 1:
                                sg = torch.where(torch.rand(int(sel.sum()), generator=H.rng, device=device) < 0.5, -1.0, 1.0)
                                d[:, lat[0]] = torch.sin(th_t) * sg
                            else:
                                ph = torch.rand(int(sel.sum()), generator=H.rng, device=device) * (2 * math.pi)
                                d[:, lat[0]] = torch.sin(th_t) * torch.cos(ph)
                                d[:, lat[1]] = torch.sin(th_t) * torch.sin(ph)
                            mg = sp_t * (1.0 + jit * (torch.rand(int(sel.sum()), generator=H.rng, device=device) - 0.5) * 2.0)
                            vc[sel] = d * mg[:, None]
                    _only = (_vi.get("types") if isinstance(_vi, dict) else None) or s.get("vel_init_types")
                    if _only:
                        tn = list(getattr(parent, "type_names", []) or [])
                        want = torch.zeros(parent.n, dtype=torch.bool, device=device)
                        pnt = getattr(parent, "node_type", None)
                        for t in _only:
                            if t in tn and pnt is not None:
                                want |= (pnt == tn.index(t))
                        vc = vc * want[:, None].to(vc.dtype)
                    parent._launch_v = vc
                state[:, vx0:vx1] = vc[parent_idx]
        occ = parent.occ[parent_idx].clone()                      # a child is live iff its parent is
        if reserve > 0:                                           # the `reserve` tail of each parent block starts DORMANT
            block_pos = torch.arange(Np, device=device) % per_tot
            is_reserve = block_pos >= per
            occ[is_reserve] = 0.0
            if has_pos:
                state[is_reserve, px0:px1] = ppos[is_reserve]     # park the dormant pool at the parent centre
        lvl = Level(sname, depth=depth, state=state, occ=occ, state_schema=schema,
                    parent=parent_idx, parent_name=pname, role=s.get("role"))
        lvl.render = render
        lvl.maps = dict(s.get("maps") or {})
        _build_mesh(lvl, s, device)                # a CONTAINED set may carry the surface too
        _assign_types(lvl, s, H, device)
        # a contained set publishes its raw type table too, so a child's `provision` can read `youngs`,
        # `material`, `density` through a three-level chain (cell -> compartment -> mpm_particle)
        lvl.types_raw = s.get("types")
        # the entity may provision domain-specific per-node buffers (e.g. mpm_particle's F/C/mass/mu/la/p_vol)
        ent = _entity_class(sname, s.get("entity"))
        provision = getattr(ent, "provision", None) if ent is not None else None
        if provision is not None:
            provision(lvl, parent, s, H, device)
        H.add_level(lvl)

    # pass 2b: resolve every mesh's maps, now that every set exists.
    _link_mesh_maps(H, sim)

    # pass 3: continuous fields, each bound to one set (one channel per type of the coupled set unless
    # `components` is given); the operators do all the dynamics
    for fname, f in sim.fields.items():
        cls = get_field(f.get("frame", "grid"))
        couples = f.get("couples_to")
        # `title` is a display name for the watcher, never a constructor argument
        fcfg = {k: v for k, v in f.items() if k not in ("frame", "title")}
        # components default to the coupled set's type count; a prescribed field (`source`) defines its own
        if "components" not in fcfg and "source" not in fcfg:
            ntypes = len(getattr(H.level(couples), "type_names", []) or []) if couples in H.levels else 0
            fcfg["components"] = ntypes or 1
        import inspect
        if "dim" in inspect.signature(cls.__init__).parameters and "dim" not in fcfg:
            fcfg["dim"] = H.dim                                       # N-D grid field follows the dimension contract
        # the per-axis box (MPMGrid's cell size must follow the world); passed only if the signature accepts it
        if "world_size" in inspect.signature(cls.__init__).parameters and "world_size" not in fcfg:
            fcfg["world_size"] = [float(w) for w in H.world_size]
        fld = cls(fname, width=H.world_width, device=device, **fcfg)   # name positional; rest by keyword
        if hasattr(fld, "periodic"):
            fld.periodic = H.periodic                                   # torus field iff the world wraps
        H.add_field(fld)

    return H


# seed: Run the seed: section exactly once, x_0 = S(theta_S).
def seed(H: Hierarchy, sim: Spec, device: str = "cpu") -> None:
    """Run `sim.seed_ops` (the `seed:` section) exactly once: x_0 = S(theta_S).

    No loop, no frame gate (`H.frame` is 0 meanwhile, which a seed may read). Each seed operator
    runs one forward(), same contract as a dynamics operator (`forward(H, mask) -> {level: delta}`);
    the usual one writes state directly and returns `{}`, a returned delta is integrated once.
    A no-op for specs without a `seed:` section (the legacy `_seed_window` handles those)."""
    if not sim.seed_ops:
        return
    H.frame = 0
    H.zero_delta()
    for o in sim.seed_ops:
        cls = get_operator(o.op, variant=o.impl)
        if getattr(cls, "KIND", None) != "seed":
            # also checked here: `seed()` is public and can be called without going through load()
            raise RuntimeError(
                f"seed: {o.op!r} resolved to kind={getattr(cls, 'KIND', None)!r}, not "
                f"\"seed\" -- refusing to run a dynamics operator as a seed.")
        if getattr(cls, "LATE_SEED", False):
            continue                                 # runs in `seed_late`, once the operators exist (it reads their learned constants)
        op = cls(o.ctor_params(), device)
        H._active_op = o.op                          # see `Hierarchy.record_maps`
        deltas = op(H, _selector_mask(H, o.mask_selector()))
        # same `(set, block)` key as the tick loop, so a seed with a second dynamical block means the same thing
        default_block = getattr(op, "INTEGRAND", None)
        for key, d in deltas.items():
            lvlname, blk = key if isinstance(key, tuple) else (key, default_block)
            H.add_delta(lvlname, d, blk)
    if any(torch.count_nonzero(d) for d in H._delta.values()):
        _integrate(H, sim.dt)


# seed_late: Run the seeds flagged `LATE_SEED` once, after the operators exist and the caller's `on_ready` has run.
def seed_late(H: Hierarchy, sim: Spec, device: str = "cpu", grad: bool = False) -> None:
    """Run the `seed:` operators whose class sets `LATE_SEED = True`, once, right after `on_ready`.

    x_0 for a state that is INFERRED from the operators' own constants. `seed()` runs before the operators exist and so cannot
    read them; a late seed sees `H.operators` with the caller's parameters already in place (`on_ready` is where a caller sets them) and runs under `grad` as the ticks do, so the gradient of the fit reaches the constants through the start state. Same contract as a
    seed otherwise: write the state in place and return `{}` (nothing integrates here, so a returned delta is refused). A no-op for a spec with
    no late seed."""
    late = [o for o in sim.seed_ops if getattr(get_operator(o.op, variant=o.impl), "LATE_SEED", False)]
    if not late:
        return
    with (contextlib.nullcontext() if grad else torch.no_grad()):
        for o in late:
            op = get_operator(o.op, variant=o.impl)(o.ctor_params(), device)
            H._active_op = o.op
            if op(H, _selector_mask(H, o.mask_selector())):
                raise RuntimeError(f"seed: {o.op!r} is a LATE_SEED and returned a delta; a late seed writes its state in place and returns {{}}")


# _integrate: Turn each set's accumulated delta into its next state, once per tick.
def _integrate(H: Hierarchy, dt: float) -> None:
    """Turn each set's accumulated delta into its next state, once per tick, in the order of `H.emit_order`:
    `velocity` reads the delta as a velocity (x += dt dpos); `acceleration` as an acceleration
    (v += dt acc; x += dt v). Sets with no engine-integrated operator hold still (an MPM body is advected
    by the MPM substep). Friction is the `drag` operator, not a knob here."""
    box = H.world_size                                     # [D] per-axis box size
    for name, emit in H.emit_order.items():
        lvl = H.levels[name]
        out = H.delta(name)
        schema = lvl.state_schema
        coord = schema.coordinate                          # the position-like integrated block (pos, or voltage, ...)
        if coord is None:
            continue                                       # no engine-integrated state
        rate = schema.rate                                 # the rate block of a 2nd-order set (vel), or None (1st-order)
        cx0, cx1 = schema.slice(coord.name)
        x = lvl.state[..., cx0:cx1]
        new = lvl.state.clone()
        if rate is not None:
            # inertial (2nd-order) set -- the pos/vel path, byte-identical to before:
            #   EMIT=velocity     -> the rate block IS the delta (overdamped)
            #   EMIT=acceleration -> integrate the delta into the rate block
            vx0, vx1 = schema.slice(rate.name)
            v = lvl.state[..., vx0:vx1]
            v = out if emit == "velocity" else v + dt * out
            vmax = getattr(lvl, "vmax", None)              # optional speed clamp (anti-overshoot)
            if vmax:
                sp = v.norm(dim=-1, keepdim=True)
                v = v * (sp.clamp(max=vmax) / sp.clamp(min=1e-9))
            x = x + dt * v
            new[..., vx0:vx1] = v
        else:
            # first-order (overdamped) set -- the delta is dx/dt directly (voltage, gating, conc)
            x = x + dt * out
        if coord.boundary == BOUNDARY_WORLD:               # a spatial coordinate is clamped/wrapped to the box;
            b = getattr(H, "boundary", "wall")             # a `free` block (voltage) has no box and is left alone
            if b == "periodic":
                x = torch.remainder(x, box)                # torus: wrap each axis by its size
            elif b in ("free", "none", "open"):
                pass                                       # unbounded: particles drift in open space
            else:
                x = torch.minimum(x.clamp(min=0.0), box)   # wall: clamp each axis to [0, w_k]
        new[..., cx0:cx1] = x
        lvl.state = new

    # extra dynamical blocks (chem, a0, ...): first-order x += dt * delta, unbounded; one set can carry several independently integrated blocks
    for name, blocks in H._delta_blocks.items():
        lvl = H.levels[name]; schema = lvl.state_schema
        new = lvl.state.clone()
        for bname, d in blocks.items():
            cx0, cx1 = schema.slice(bname)
            new[..., cx0:cx1] = new[..., cx0:cx1] + dt * d
        lvl.state = new




# apply_learnable_blocks: Lift each learnable block into an nn.Parameter held on the spec -> list of parameters.
def apply_learnable_blocks(H, sim) -> list:
    """`learnable: {block: w, of: recurrent}` -> an nn.Parameter the loss can reach. Returns the parameters, in spec order.

    The block's current columns are lifted into a parameter held on the Level and written back into the
    state each run (functionally: clone, assign, publish), so a loss on the final frame differentiates
    back to it. It starts at the spec's own value (e.g. a connectome's measured synapse areas), so a fit
    reads as movement away from the measurement."""
    made = []
    held = getattr(sim, "fitted", None)
    if held is None:
        held = sim.fitted = {}
    for e in getattr(sim, "learnable", []):
        if "block" not in e:
            # refuse, don't skip: an entry without `block:` (e.g. an operator substitution) would run unchanged and say nothing
            raise ValueError(
                f"learnable: {e} has no `block:`. Only {{block:, of:}} is fitted; a law "
                f"substitution ({{replaces:, with:}}) is not implemented.")
        lvl = H.level(e["of"])
        blk = e["block"]
        if blk not in lvl.state_schema:
            raise ValueError(
                f"learnable: block {blk!r} is not a block of set {e['of']!r} "
                f"(it has {sorted(lvl.state_schema)}).")
        a, b = lvl.state_schema[blk]
        # held on the Spec: `run` rebuilds the Hierarchy every call, so a parameter made inside one rollout would not reach the next
        key = f"{e['of']}.{blk}"
        par = held.get(key)
        if par is None:
            par = held[key] = nn.Parameter(lvl.state[:, a:b].detach().clone())
        if not hasattr(lvl, "_fitted"):
            lvl._fitted = nn.ParameterDict()
        lvl._fitted[blk] = par
        st = lvl.state.clone()
        st[:, a:b] = par
        lvl.state = st
        made.append(par)
        if getattr(sim, "_learnable_announced", False) is False:
            print(f"[learnable] {e['of']}.{blk}: {par.numel()} value(s) fitted, "
                  f"starting at the spec's own", flush=True)
    sim._learnable_announced = True        # once per spec, not once per rollout
    return made


# run: Forward-simulate a spec -> (Hierarchy, recorded trajectory).
def run(sim: Spec, out_path: str | None = None, device: str = "cpu",
        on_frame=None, progress: bool = False,
        grad: bool = False, batch: int = 1, on_seeded=None,
        on_ready=None) -> tuple[Hierarchy, dict]:
    """Forward-simulate `sim`. Returns (Hierarchy, recorded trajectory).

    Order: build the Hierarchy, seed x_0, instantiate the operators once, then per tick run the
    schedule (substeps, frame gates, optional CUDA graph / torch.compile), integrate, record.

    Hooks (a caller's write at a fixed moment):
      on_seeded(H)  once, after seeding and before the batch axis exists -- the moment to write x_0.
                    `on_frame` fires at the END of a tick, so a value written there is one step late.
      on_ready(H)   once, after the operators exist (`H.operators`) and before tick 0 -- the moment
                    to set operator parameters.
      on_frame(H, tick)  after every tick.

    `grad=True` keeps the autograd tape over the whole rollout, so a loss on the FINAL state
    differentiates back to x_0 and to tensor-valued operator parameters. Opt-in: the tape costs
    memory forward generation does not need. The recorded trajectory is always detached, and a
    structural operator must write functionally (clone, write, publish) or it cuts the tape.
    """
    _picked = _resolve_default_impl(sim, device, grad)
    if _picked:
        print(f"[engine] implementation: warp for {', '.join(_picked)} "
              f"(unset in the spec; `implementation: default` forces the torch body)", flush=True)
    elif not any(o.impl for o in sim.operators):    # no variant anywhere: every operator runs its default body
        print("[engine] no warp variant picked: every operator runs its default body", flush=True)
    H = build(sim, device)                    # 1) Hierarchy: every set (level) + field, from the spec
    seed(H, sim, device)                      # 1.5) x_0 = S(theta_S), exactly once
    for lvl in H.levels.values():             # 1.6) v_0 from x_0 (stashed by `build`)
        vinit = getattr(lvl, "_vel_init", None)
        if vinit:
            vx0, vx1 = lvl.state_schema["vel"]
            vel = _init_velocity(vinit, lvl, H.world_size, H.rng, device)   # [n, D]
            st = lvl.state.clone(); st[:vel.shape[0], vx0:vx1] = vel; lvl.state = st
            lvl._vel_init = None
    apply_learnable_blocks(H, sim)            # 1.7) `learnable: {block:, of:}` -> tensor leaves
    if on_seeded is not None:                 # 1.75) caller's write into x_0
        on_seeded(H)
    expand_batch(H, int(batch))               # 1.8) state [N, W] -> [B, N, W]
    H.emit_order = _resolve_emit(sim, H)      # 2) per-set integration order (1st-order velocity / 2nd-order acceleration)
    # 3) one instance per operator: (op name, live instance, selector, (after, before, every))
    inst = [(o.op,
             get_operator(o.op, variant=o.impl)(o.ctor_params(), device),
             o.mask_selector(),
             gate)                                       # multi-rate: runs only when tick % every == 0
            for o, gate in zip(sim.operators, _resolve_gates(sim))]

    # ModuleList, in spec order: H.parameters() finds operator tensor leaves; names line up by index
    H.operators = nn.ModuleList([ob for _nm, ob, _sel, _g in inst])
    H.operator_names = [nm for nm, _ob, _sel, _g in inst]
    if on_ready is not None:                  # caller's write into operator parameters
        on_ready(H)
    seed_late(H, sim, device, grad)           # 3.5) the seeds that read the operators' (learned) constants

    # the first scatter into a shared mpm_grid zeroes it, later ones add (fixed by spec order, so
    # it is a constant a replayed CUDA graph can bake in)
    _first_into = set()
    for _nm, _ob, _sel, _g in inst:
        if _nm not in ("mpm_scatter", "p2g"):
            continue
        _grid = getattr(_ob, "grid", "mpm_grid")
        _ob._zeroes_grid = _grid not in _first_into
        _first_into.add(_grid)

    rec_index, rec_sets, occ_sets, rec_state, rec_mesh, fstride, rec_fields = _setup_recording(sim, H)
    # live buffers on H so an interrupted run can be assembled (`_assemble`). `_rec` is a positional
    # contract with run_one.py (`_assemble(Hf, *Hf._rec, n_rows=...)`): rec_mesh stays out of it
    H._rec = (sim, rec_sets, occ_sets, rec_state, rec_fields)
    H._rec_mesh = rec_mesh
    H._rec_index = rec_index                  # tick -> row, so a salvage can truncate
    _print_run_summary(sim, H)

    # The i-th schedule entry naming an operator runs the i-th LIVE instance of that name (declaration
    # order, filtered by frame window). Binding by name ran every instance at every entry, and binding by
    # declaration order alone never ran the second of two windowed instances.
    _seen_this_tick = {}

    # warm-up: `Spec.warmup` = W ticks run before the recorded frames; frame gates and strides
    # read the recorded-frame clock f = tick - W (f < 0 reads as frame 0). W is set from the spec below.
    W = 0
    _inst_names = [_nm for _nm, _ob, _sel, _g in inst]
    _inst_gates = [_g for _nm, _ob, _sel, _g in inst]

    _all_by_name = {}                                     # operator name -> every declared instance index
    for _i, _nm in enumerate(_inst_names):
        _all_by_name.setdefault(_nm, []).append(_i)
    _live_by_name = _live_instances(_inst_names, _inst_gates, -W)    # ... and only those whose window is open now

    # warn (never fail) when the schedule names an operator fewer times than it is declared: the extra
    # instances never run, and an unwindowed one is a set left out of the physics
    try:
        from collections import Counter as _C
        _named = _C()
        for _st in (sim.schedule or []):
            if isinstance(_st, dict):
                for _t in (_st.get("steps") or []):
                    _named[_t] += 1
            else:
                _named[str(_st)] += 1
        for _op_name, _all_i in _all_by_name.items():
            _n_named = _named.get(_op_name, 0)
            if _n_named < len(_all_i):
                _sets = ", ".join(str(inst[_j][0]) if not isinstance(inst[_j][0], str)
                                  else str(getattr(inst[_j][1], "at", "?"))
                                  for _j in _all_i[_n_named:])
                warn(f"[warn] schedule names {_op_name!r} {_n_named} time(s) but {len(_all_i)} are "
                      f"declared: {len(_all_i) - _n_named} instance(s) will never run"
                      + (f" (at: {_sets})" if _sets else "")
                      + ". If those are not gated by a window, that set is out of the physics.")
    except Exception:                                                # noqa: BLE001
        pass

    # the substep blocks' persistent delta buffers and CUDA-graph bookkeeping (see CUDA_graph_capture.SubstepState)
    _substep_state = SubstepState()

    # ---- torch.compile, opt-in: `{substep_dt: X, steps: [...], compile: true}` on the substep block.
    # Not bit-identical to eager (fused, reordered float ops), so a spec with it can never be a
    # promotion twin. Installed after tick 0, so every operator's run-constant caches
    # (`bool(t.any())` predicates) are resolved and the traced body has no graph breaks.
    _compile_done = [False]

    def _install_compile():
        if _compile_done[0]:
            return
        _compile_done[0] = True
        _compiled = []
        for _step in sim.schedule:
            if not (isinstance(_step, dict) and _step.get("compile")):
                continue
            _mode = str(_step.get("compile_mode", "default"))
            # dynamo settings; left at defaults compiling was SLOWER than eager (164 vs 156 ms/frame)
            #  - capture_scalar_outputs stays off: it breaks inductor (mpm_gather caches the box instead)
            #  - allow_unspec_int_on_nn_module: H.micro advances every substep, not a constant
            #  - recompile_limit: the polar decomposition's mask length is data-dependent
            import torch._dynamo as _dyn
            _dyn.config.allow_unspec_int_on_nn_module = True
            _dyn.config.recompile_limit = max(int(getattr(_dyn.config, "recompile_limit", 8)),
                                              int(_step.get("compile_recompile_limit", 64)))
            if _step.get("compile_verbose"):
                _dyn.config.verbose = True                        # full graph-break provenance
            else:
                # mute the dynamo / inductor stack warnings; `compile_verbose: true` brings them back
                import logging
                for _lg in ("torch._dynamo", "torch._inductor", "torch._dynamo.convert_frame"):
                    logging.getLogger(_lg).setLevel(logging.ERROR)
                import warnings as _w
                _w.filterwarnings("ignore", message=".*TensorFloat32.*")
                _w.filterwarnings("ignore", message=".*fp32_precision.*")
            for _op_name in _step.get("steps", []):
                for _j in _live_by_name.get(_op_name, []):
                    if _j in [c[0] for c in _compiled]:
                        continue
                    _ob = inst[_j][1]
                    try:
                        # compile the bound method, not the module (which would return a new
                        # OptimizedModule and break the getattr(ob, EMIT / INTEGRAND ...) below)
                        _ob.forward = torch.compile(_ob.forward, mode=_mode, dynamic=False)
                        _compiled.append((_j, _op_name))
                    except Exception as _e:                      # no triton, unsupported backend, ...
                        warn(f"torch.compile unavailable for {_op_name!r} ({type(_e).__name__}: {_e}); "
                             f"running it eager")
        if _compiled:
            warn(f"torch.compile ON for {len(_compiled)} operator instance(s): "
                 f"{', '.join(t for _i, t in _compiled)} (mode={_mode}). Compiled arithmetic is fused "
                 f"and reordered, so this run is NOT bit-identical to the same spec uncompiled -- it "
                 f"cannot be used as a promotion twin. The first few frames pay compilation.")

    def _run_scheduled(op_name, tick):
        """Run the operator instance this schedule position names, enforcing the first-tick
        integration-invariant guard on non-opted-out operators."""
        _n = _seen_this_tick.get(op_name, 0)
        _seen_this_tick[op_name] = _n + 1
        _idx = _live_by_name.get(op_name) or []                 # the LIVE instances, in declaration order
        _all = _all_by_name.get(op_name) or []             # every declared instance of that name
        # which instances this occurrence runs:
        #   _n < len(_idx)  the i-th occurrence takes the i-th live instance (normal)
        #   _n < len(_all)  named as often as declared, but the rest are gated off this tick: run nothing
        #   otherwise       named more often than declared (one operator repeated): every live instance
        _want = ({_idx[_n]} if _n < len(_idx)
                 else set() if _n < len(_all)
                 else set(_idx))
        for _j, (nm, ob, sel, (after_frame, before_frame, every)) in enumerate(inst):
            if nm != op_name or _j not in _want:
                continue
            # tick 0 runs every kind, recorders included, so the positions and topology series both start there
            if not (after_frame <= max(tick - W, 0) < before_frame):
                continue                                 # engine-level frame gate (skip = no delta, no RNG drawn)
            if every > 1 and (tick - W) % every != 0:
                continue                                 # multi-rate: run this operator only every `every` ticks
            snap = ({n: l.state.clone() for n, l in H.levels.items()}
                    if tick == 0 and not getattr(ob, "MAY_MUTATE_INTEGRATED_STATE", False) else None)
            H._active_op = nm                        # so `H.gather` can record who walked a relation
            deltas = ob(H, _selector_mask(H, sel))   # call the operator: forward() runs, returns {set: delta} (or {})
            # a returned key is `set` or `(set, block)`; a bare set takes the operator's INTEGRAND block
            # (None = the set's coordinate block). A delta for a set no operator is declared `at:` is never integrated
            default_block = getattr(ob, "INTEGRAND", None)
            for key, d in deltas.items():            # break here to inspect `deltas`; empty for EMIT=None ops
                if isinstance(key, tuple):
                    lvlname, blk = key               # explicit per-delta block
                else:
                    lvlname, blk = key, default_block
                H.add_delta(lvlname, d, blk)         # record each returned delta for end-of-tick integration

            if snap is not None:
                for n, before in snap.items():
                    # NaN-aware: torch.equal is False for an unchanged state holding a NaN
                    after = H.levels[n].state
                    if before.shape != after.shape or not bool(((before == after)
                                                                | (before.isnan() & after.isnan())).all()):
                        raise RuntimeError(
                            f"operator {nm!r} wrote the integrated state of set {n!r} "
                            f"directly. A dynamics operator must RETURN a delta (the engine "
                            f"integrates it); only structural / derived-readout operators "
                            f"(MAY_MUTATE_INTEGRATED_STATE) may write state. (integration invariant)")

    # warm-up length W: `general.warmup`, else the longest settle window a seed declared
    # (`_mesh["ref_frame"]`), else 0. Recorded keys move by W, so row 0 is tick W
    _wd = getattr(sim, "warmup", None)
    if _wd is None:
        _wd = max([int((getattr(_l, "_mesh", None) or {}).get("ref_frame", 0) or 0) for _l in H.levels.values()] + [0])
    W = max(int(_wd), 0)
    H.warmup = W
    if W > 0:
        _shifted = {t + W: i for t, i in rec_index.items()}
        rec_index.clear(); rec_index.update(_shifted)
        print(f"[engine] warm-up: {W} tick(s) before frame 0, not recorded -- the settle window runs with the "
              f"size rules idle and `cell_grow` holding (general.warmup: 0 restores the recorded window)", flush=True)
    ticks = range(W + sim.n_frames + 1)
    if progress:                                     # live progress bar over the simulated frames
        try:
            from tqdm import tqdm
            ticks = tqdm(ticks, desc=f"[generate] {sim.name}", unit="frame", ncols=150, leave=True,
                         bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                                    "[{elapsed}<{remaining}{postfix}]")
        except ImportError:
            pass
    frame_ms: list[float] = []                       # wall ms per tick, carried into the trajectory
    _dt_of = None                                    # per-tick frame duration (`general.dt_schedule`); None keeps H.dt == sim.dt
    if getattr(sim, "dt_schedule", None):
        _dt_of = [float(d) for f, d in sim.dt_schedule for _ in range(int(f))]
    with (contextlib.nullcontext() if grad else torch.no_grad()):     # tape only when asked (`grad`)
        for tick in ticks:                           # one tick = one pass of the schedule + integrate
            # ===== 1. PREAMBLE: clocks and per-tick state, and which operator instances are live in this frame
            _t_tick = time.perf_counter()            # wall time of the whole tick, recorder included
            if tick == 1:
                _install_compile()                   # after tick 0 has warmed every run-constant cache
            if progress and tick and hasattr(ticks, "set_postfix_str") and tick % 8 == 0:
                _r = ticks.format_dict.get("rate")   # tqdm's recent rate, not elapsed/tick (which folds in startup)
                if _r:
                    ticks.set_postfix_str(f"{1000.0 / _r:.1f} ms/frame  {_r:.2f} fps",
                                          refresh=False)
            H.frame = tick                           # current tick (read by prescribed fields, e.g. playback)
            H.frame_t.fill_(float(tick))             # the same, reachable from inside a CUDA graph
            if _dt_of is not None:                   # `general.dt_schedule`: this frame's own duration
                H.dt = _dt_of[min(tick, len(_dt_of) - 1)]
            # H.micro counts micro-steps (ticks AND substeps): a shared grid clears on the first
            # scatter of a new micro-step and accumulates on the later ones
            H.micro = getattr(H, "micro", 0) + 1
            H.zero_delta()
            _seen_this_tick.clear()                  # mention of a name -> instance, per tick
            _live_by_name.clear()
            _live_by_name.update(_live_instances(_inst_names, _inst_gates, tick - W))     # ... and only instances whose window is open
            # ===== 2. SCHEDULE: each step runs its operators, which add their output into H's per-set deltas;
            #        a plain step runs once, a substep block (`substep_dt`) runs `count` times per frame
            for step in sim.schedule:                # operators accumulate per-set deltas
                # substep block `{substep_dt: <dt_sub>, steps: [...]}`: run `steps` round(H.dt / dt_sub)
                # times per frame at dt_sub (e.g. MPM strain -> P2G -> grid -> G2P). Deltas from the
                # outer schedule (gravity) persist across it
                if isinstance(step, dict) and "substep_dt" in step:
                    run_substep_block(H, sim, step, inst, tick, device, _live_ids(_inst_gates, tick - W),
                                      _run_scheduled, _seen_this_tick, _substep_state)
                    continue
                # plain step: run each named operator once
                for op_name in (step if isinstance(step, list) else [step]):
                    _run_scheduled(op_name, tick)
            # ===== 3. INTEGRATE: turn each set's summed delta into its next state, once per tick
            _integrate(H, H.dt)                      # integrate each set once, at end of tick (H.dt == sim.dt unless dt_schedule)
            # ===== 4. RECORD: positions, state blocks, meshes and fields of recorded frames; the on_frame callback; the tick time
            ri = rec_index.get(tick)                 # None on un-recorded ticks (strided long runs)
            if ri is not None:
                record_levels(H, ri, rec_sets, occ_sets, rec_state, rec_mesh)
            if H.fields and tick >= W and ((tick - W) % fstride == 0 or tick - W == sim.n_frames):
                record_fields(H, rec_fields)
            if on_frame is not None and tick >= W:
                on_frame(H, tick - W)
            if tick >= W:
                frame_ms.append((time.perf_counter() - _t_tick) * 1000.0)

    out = _assemble(H, sim, rec_sets, occ_sets, rec_state, rec_fields, rec_mesh=rec_mesh)
    out["frame_ms"] = np.asarray(frame_ms, np.float32)   # one wall-clock reading per simulated tick

    if out_path is not None:
        import zarr

        def _put(g, key, data):
            """Write one array on zarr 2 or zarr 3 (`create_array` is the zarr-3 name)."""
            data = np.asarray(data)
            try:
                a = g.create_array(name=key, shape=data.shape, dtype=data.dtype)
                a[...] = data
                return
            except (AttributeError, TypeError):
                pass
            try:
                g.create_dataset(key, data=data)
                return
            except TypeError:
                g.create_dataset(key, shape=data.shape, dtype=data.dtype)[...] = data

        root = zarr.open_group(out_path, mode="w")
        for name in H.levels:
            g = root.create_group(name)
            if out["sets"][name]["pos"] is not None:            # spatial: the pos trajectory
                _put(g, "pos", out["sets"][name]["pos"])
            _put(g, "occ", out["sets"][name]["occ"])
            if out["sets"][name]["state"] is not None:          # non-spatial state blocks -> state/<block>
                sg = g.create_group("state")
                for bname, arr in out["sets"][name]["state"].items():
                    _put(sg, bname, arr)
            ms = out["sets"][name].get("mesh")
            if ms:                                              # the ragged half-edge history
                mg = g.create_group("mesh")
                _put(mg, "offsets", np.cumsum([0] + [len(m["E_srce"]) for m in ms]))
                _put(mg, "face_offsets", np.cumsum([0] + [int(m["nF"]) for m in ms]))
                _put(mg, "nF", np.asarray([m["nF"] for m in ms], np.int64))
                _put(mg, "Nv", np.asarray([m["Nv"] for m in ms], np.int64))
                for col in ("E_srce", "E_trgt", "E_face"):
                    _put(mg, col, np.concatenate([m[col] for m in ms]).astype(np.int64))
                from plexus.models.mesh import mesh_row_columns
                _scal, _edge, _face, _fill = mesh_row_columns(ms)   # union of columns over rows, zero-filled
                for col in _scal:
                    _put(mg, col, np.asarray([_fill(m, col) for m in ms], np.float64))
                for col in _edge:
                    _vals = [_fill(m, col) for m in ms]
                    _put(mg, col, np.concatenate(_vals).astype(np.float32))
                    _put(mg, col + "_offsets",
                         np.cumsum([0] + [len(v) for v in _vals]).astype(np.int64))
                for col in _face:
                    _put(mg, col, np.concatenate([_fill(m, col) for m in ms]).astype(np.float32))
        for fn, fd in out["fields"].items():
            g = root.create_group(fn)
            _put(g, "grid", fd["grid"])
            _put(g, "colors", fd["colors"])
        root.attrs.update(name=sim.name, seed=sim.seed, world=H.world_width)
    return H, out
