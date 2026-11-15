"""The container and the contract layer: the vocabulary every Plexus simulation is built from.

Engine-agnostic: it says what a set, a field and an operator ARE, not how a run is stepped (`engine.py`).

Entities
  Level      a set of like nodes of one type, stored as flat batched tensors (`state` [N, W], optional
             `embedding`), never one object per node. Containment is a parent map to another set
             (`parent` indices + `parent_name`); many sets may share one parent, so the container is a
             typed containment graph, not a scale ladder. A Level has a fixed buffer size and a per-node
             occupancy `occ` in [0, 1] marking the live subset, so divide / die fit a constant shape.
  Field      a continuum f: Omega -> R^c on its own discretization frame, bound to one Level by
             `couples_to`; subclasses supply the frame and the transfer kernels (scatter / gather / step).
  Hierarchy  the ordered Levels + flat Fields + the run-time scratch the engine attaches (config, rng,
             delta accumulators, world geometry); an nn.Module, so `.to(device)` moves it all.

Operators
  Contract:  __init__(self, params: dict, device="cpu");  forward(self, H, mask=None) -> {set: delta}
  The returned deltas are time-derivative contributions; the engine sums them per set and integrates
  each set once at the end of the tick (order from the operator's `EMIT`). An operator never writes the
  integrated state (pos / vel) directly (the engine checks on frame 0); everything else (relations,
  occupancy, fields, auxiliary buffers) it mutates in place and returns `{}`.

  Kinds: seven are dynamics, dispatched by the relation the operator acts on; `seed` is a separate
  lifecycle phase (x_0 = S(theta_S), once, via `engine.seed()`, then x_{t+1} = Phi(x_t; theta_D)).
    lateral     within a set               signalling, boids, MPM particle forces
    aggregate   children -> parent         particles -> cell centroid
    broadcast   parent -> children         cell decision -> particle force
    exchange    set <-> field / set        P2G / G2P, secrete / sense, reaction
    field       field -> field             diffuse, decay, react, playback
    structural  changes |S_k|, membership  divide, duplicate, die
    rewire      rebuilds E (edge_index)    membrane ring, neighbour graph
    seed        writes x_0, once           mesh / tissue placement, field IC

  `params` is the operator's spec line merged with its field refs (`to` / `from`); `mask` is the live
  boolean selection of its `at:` selector, recomputed every tick.

Capability contract: operators declare what they need so the validator fails before the run:
  REQUIRES_PARAMS (keys the spec line provides), REQUIRES_TYPE_PROPS (per-type properties, resolved
  along the containment chain). `KIND` and `SET` are stamped by `@register_operator`; do not set them."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from plexus.models.state import StateSchema


# KINDS: The recognised operator kinds (dispatch tags): the dynamics kinds, and `seed`, which writes x_0 once before any dynamics.
KINDS = ("lateral", "aggregate", "broadcast", "exchange", "rewire", "divide", "die", "seed",
         "field", "structural")


# EMITS: What a returned delta IS, shared by `Operator.EMIT` and the spec `emit:`: velocity, acceleration, or mpm_acceleration (routed to the MPM substep, not engine-integrated); None = no set delta.
EMITS = ("velocity", "acceleration", "mpm_acceleration")


# Level: A set S_k of like nodes at one scale, stored as flat batched tensors.
class Level(nn.Module):
    """A set `S_k` of like nodes at one scale, state [N, W] (W from the StateSchema).

    Allocated at a fixed buffer size; `occ` (occupancy in [0, 1], default ones) marks the live subset so
    divide / die can wake or retire slots without resizing; `active` is the boolean live mask. Domain
    operators register their own per-node buffers on the Level (MPM's `mass` / `F` / `C`, `node_type`);
    the contract mandates only `state`, `occ` and, for a contained set, `parent`."""

    def __init__(
        self,
        name: str,
        depth: int = 0,                             # hierarchy DEPTH (0 = leaf); a scale hint only.
                                                    # (was `level`, renamed to free the word "level" from the set / depth overload; `level=` is still accepted)
        state: torch.Tensor = None,                 # [N, d]   dynamic state
        embedding: Optional[torch.Tensor] = None,   # [N, e]   learnable a_i
        parent: Optional[torch.Tensor] = None,      # [N]      index into the PARENT SET (the map pi_{this->parent})
        edge_index: Optional[torch.Tensor] = None,  # [2, E]   lateral relation
        occ: Optional[torch.Tensor] = None,         # [N]      occupancy in [0,1] (default ones)
        state_schema=None,                          # StateSchema (the fifth primitive), or a legacy {block:(c0,c1)} dict
        parent_name: Optional[str] = None,          # name of the set that contains this one (a containment EDGE)
        role: Optional[str] = None,                 # this set's role inside its parent (membrane / cytoplasm / nucleus...)
        pre: Optional[torch.Tensor] = None,         # [E] INCIDENCE map to the pre-endpoint set (edge-set only)
        post: Optional[torch.Tensor] = None,        # [E] INCIDENCE map to the post-endpoint set (edge-set only)
        pre_name: Optional[str] = None,             # the set `pre` indexes into (e.g. neuron)
        post_name: Optional[str] = None,            # the set `post` indexes into (e.g. neuron)
        mesh=None,                                  # a MeshTable, when the set declares `mesh: half_edge`
    ):
        super().__init__()
        self.name = name
        # `depth` is a scale hint only; containment is by `parent_name`, not by integer.
        self.depth = depth
        self.parent_name = parent_name              # which set contains this one (None for a top-level set)
        self.role = role
        # incidence maps: an edge-set's elements are connections sent to their endpoints by `pre` / `post` (index buffers like `parent`); empty for an ordinary set
        self.pre_name = pre_name
        self.post_name = post_name
        # the half-edge table, when the set declares one (faces, per-face targets and an ordering that is the geometry; see `plexus.models.mesh`)
        self.mesh = mesh
        # state is first-class: a legacy {block: (c0, c1)} dict is normalised into a StateSchema (still dict-indexable)
        # `is not None`, not `or`: an EMPTY schema (a relation like `half_edge`) is falsy and would be replaced by the pos/vel default, seeding a cloud of particles
        self.state_schema = StateSchema.normalize(
            state_schema if state_schema is not None else {"pos": (0, 2), "vel": (2, 4)})
        N = state.shape[0]
        self.register_buffer("state", state)
        self.embedding = nn.Parameter(embedding) if embedding is not None else None
        self.register_buffer(
            "parent",
            parent if parent is not None else torch.empty(0, dtype=torch.long, device=state.device),
        )
        self.register_buffer(
            "edge_index",
            edge_index if edge_index is not None else torch.empty(2, 0, dtype=torch.long, device=state.device),
        )
        self.register_buffer(
            "pre",
            pre if pre is not None else torch.empty(0, dtype=torch.long, device=state.device),
        )
        self.register_buffer(
            "post",
            post if post is not None else torch.empty(0, dtype=torch.long, device=state.device),
        )
        self.register_buffer(
            "occ",
            occ if occ is not None else torch.ones(N, device=state.device),
        )
        # lineage for structural operators: `birth` = occupancy a node was born with (drives "mass doubled -> divide"), `lineage` = the slot it split from (-1 = founder)
        self.register_buffer("birth", self.occ.clone())
        self.register_buffer("lineage", torch.full((N,), -1, dtype=torch.long, device=state.device))

    @property
    def n(self) -> int:
        """Buffer size (allocated slots, live or dormant). Read from the right: a batched run carries `state` as [B, N, W], so `shape[0]` would report B."""
        return self.state.shape[-2]

    def get(self, block: str) -> torch.Tensor:
        """A view of a named state block ('pos', 'vel', ...) per the schema, indexed from the right so it works for [N, W] and for batched [B, N, W] state."""
        a, b = self.state_schema[block]
        return self.state[..., a:b]

    @property
    def is_edge_set(self) -> bool:
        """True if this set carries incidence maps -- its elements are connections
        (a synapse/junction/vessel), sent to endpoints by `pre`/`post`."""
        return self.pre_name is not None or self.post_name is not None

    def incidence(self, role: str) -> torch.Tensor:
        """The incidence-map index buffer for `role` ('pre' or 'post')."""
        return getattr(self, role)

    def incidence_name(self, role: str) -> str:
        """The endpoint set name for `role`."""
        return getattr(self, f"{role}_name")

    @property
    def active(self) -> torch.Tensor:
        """Boolean mask of live nodes (`occ > 0`)."""
        return self.occ > 0

    @property
    def n_active(self) -> int:
        return int(self.active.sum())

    def __repr__(self):
        return (f"Level({self.name!r}, depth={self.depth}, n={self.n}, "
                f"active={self.n_active}, d={self.state.shape[-1]})")

    # cardinality primitives (spawn / kill / free_slots) that structural operators use instead of scanning occupancy by hand; shapes stay constant (fixed buffer, `occ` marks the live subset)
    # `_mesh` is the legacy name of `mesh` (about forty readers: operators, recorder, renderers), kept as an alias
    @property
    def _mesh(self):
        return self.mesh

    @_mesh.setter
    def _mesh(self, value):
        self.mesh = value

    def per_node_buffers(self):
        """Yield (name, tensor) for every registered buffer indexed by node, so a
        structural op touches them all uniformly. Excludes the relation
        `edge_index` (shaped [2, E], not per-node) and the immutable `birth` /
        `lineage` / `occ` which `spawn`/`kill` manage explicitly."""
        for name, b in self.named_buffers(recurse=False):
            if name in ("edge_index", "occ", "birth", "lineage"):
                continue
            if b.dim() >= 1 and b.shape[0] == self.n:
                yield name, b

    def free_slots(self, k: int) -> torch.Tensor:
        """Up to `k` dormant slot indices (`occ == 0`), fewer if the buffer is
        nearly full. An empty result means the buffer is exhausted -- the op
        should stop dividing rather than error (the proven 'buffer full' guard)."""
        dormant = (self.occ == 0).nonzero(as_tuple=True)[0]
        return dormant[:k]

    def spawn(self, src_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Wake free slots as clones of `src_idx`: copy EVERY per-node buffer (state, parent, node_type, domain buffers like MPM mass / F / C / mu / la), set `occ = 1`, record lineage. The caller overrides only what must differ (nudged position, zeroed velocity, reset F). Returns (new_idx, src_used), truncated to the free slots available."""
        dst = self.free_slots(int(src_idx.numel()))
        src = src_idx[: dst.numel()]
        for _, b in self.per_node_buffers():
            b[dst] = b[src]
        self.occ[dst] = 1.0
        self.birth[dst] = self.birth[src]          # default: inherit baseline (divide overrides per-daughter)
        self.lineage[dst] = src
        return dst, src

    def kill(self, idx: torch.Tensor, park: Optional[torch.Tensor] = None) -> None:
        """Retire live slots (the efflux/death boundary): `occ = 0`, drop physical
        `mass` to 0 if present so the node stops contributing to MPM scatter /
        mass-weighted aggregation, and optionally park its position off-domain at
        `park` so it is neither drawn nor sensed."""
        self.occ[idx] = 0.0
        if "mass" in dict(self.named_buffers(recurse=False)):
            self.get_buffer("mass")[idx] = 0.0
        if park is not None:
            self.state[idx, : park.numel()] = park


# Field: A continuum f: Omega -> R^c on its own frame, bound to one Level.
class Field(nn.Module):
    """A continuum `f: Omega -> R^c` on its own frame, bound to one Level.

    Subclasses (registered via `@register_field`) supply the discretization and
    the transfer kernels. `scatter` writes object state into the field; `gather`
    reads the field back onto objects (a delta); `step` advances the field's own
    PDE one tick (Laplacian diffusion + decay). P2G/G2P, secrete/sense and
    morphogen sampling are all (scatter, gather) pairs over a Field.
    """

    COUPLES_TO: Optional[str] = None    # stamped by @register_field
    FRAME: Optional[str] = None

    def __init__(self, name: str, couples_to: Optional[str] = None):
        super().__init__()
        self.name = name
        # the Level whose types this field's channels mirror (slime: one channel per species); optional, a prescribed field (a video) binds to no set
        self.couples_to = couples_to

    def scatter(self, level: "Level") -> None:          # object -> field
        raise NotImplementedError

    def gather(self, level: "Level") -> torch.Tensor:   # field -> object (delta)
        raise NotImplementedError

    def step(self) -> None:                             # advance the field's own PDE
        raise NotImplementedError

    # spatial field math on the grid (Axis B)
    def grad_at(self, pos, channel=0, periodic=False):
        """Gradient [N, D] of a grid channel, bilinear-sampled at world positions `pos` [N, D]. Central difference: wraps under a periodic world, replicates under a wall. `channel=None` sums all channels. 2D grids (`grid` is [C, nx, ny]): operators using it declare SUPPORTED_DIMS=[2]."""
        import torch.nn.functional as F
        g = self.grid.sum(0) if channel is None else self.grid[int(channel)]      # [nx, ny]
        W = self.width
        mode = "circular" if periodic else "replicate"
        gp = F.pad(g[None, None], (1, 1, 1, 1), mode=mode)[0, 0]                   # [nx+2, ny+2]
        gx = (gp[2:, 1:-1] - gp[:-2, 1:-1]) * 0.5 * (self.nx / W)                  # d/dx per world unit
        gy = (gp[1:-1, 2:] - gp[1:-1, :-2]) * 0.5 * self.ny                        # d/dy per world unit
        grad = torch.stack([gx, gy], 0)[None]                                     # [1, 2, nx, ny]
        gxn = (pos[:, 0] / W) * 2 - 1
        gyn = (pos[:, 1] / 1.0) * 2 - 1
        grid = torch.stack([gyn, gxn], -1)[None, None]                            # grid_sample: x=ny, y=nx
        return F.grid_sample(grad, grid, mode="bilinear", padding_mode="border",
                             align_corners=True)[0, :, 0].t()                     # [N, D]

    def sample(self, pos, channel=None):
        """Bilinear read of the grid at world positions `pos` [N, D]: [N, C] for `channel=None`, [N] for one channel. Border padding; same convention as `grad_at`. 2D grids."""
        import torch.nn.functional as F
        data = self.grid if channel is None else self.grid[int(channel):int(channel) + 1]   # [C, nx, ny]
        W = self.width
        gxn = (pos[:, 0] / W) * 2 - 1
        gyn = (pos[:, 1] / 1.0) * 2 - 1
        g = torch.stack([gyn, gxn], -1)[None, None]
        out = F.grid_sample(data[None], g, mode="bilinear", padding_mode="border",
                            align_corners=True)[0, :, 0].t()                      # [N, C']
        return out[:, 0] if channel is not None else out


# Hierarchy: The model of one run: its Levels (state), Fields, operators (laws) and world settings in one nn.Module -> Hierarchy.
class Hierarchy(nn.Module):
    """The model of one run: Levels (ordered bottom-up) and Fields hold the state, `operators` the laws, and the
    world settings (`config`, `rng`, `dim`, `world_size`, `boundary`, `dt`) the box they act in. One `nn.Module`,
    so `parameters()` and `state_dict()` reach the state and the learnable laws together.

    `build()` creates the levels, fields and world settings; `run()` then instantiates the operators and
    fills `operators`, `operator_names` and `emit_order` (empty until then). The engine's own scratch has
    a leading underscore (`_delta`, `_rec`, ...). Operators are pure and return per-set deltas; the engine
    sums them here and integrates each set once at the end of the tick. Use `zero_delta` / `add_delta` /
    `delta` so operators and engine share one convention (a delta is a velocity or an acceleration, per the
    operator's `EMIT`)."""

    def __init__(self):
        super().__init__()
        self.levels = nn.ModuleDict()         # name -> Level (insertion order = bottom-up)
        self.fields = nn.ModuleDict()         # name -> Field
        self.operators = nn.ModuleList()      # live operator instances in spec order (filled by run())
        self.operator_names: list[str] = []   # their names, same order as `operators` (filled by run())
        self.emit_order: dict[str, str] = {}  # set -> "velocity" | "acceleration": what its summed delta means (filled by run())
        self.scheduled_ops: frozenset = frozenset()   # names of the scheduled operators (set by build())
        self._delta: dict[str, torch.Tensor] = {}          # per-set COORDINATE-block delta (MPM reads this via delta())
        self._delta_blocks: dict[str, dict] = {}           # per-set EXTRA first-order block deltas: set -> {block: tensor}
        self.dim = 2                          # spatial dimensions (set by the engine from the spec)

    # structure
    def add_level(self, lvl: Level) -> Level:
        self.levels[lvl.name] = lvl
        return lvl

    def add_field(self, fld: Field) -> Field:
        self.fields[fld.name] = fld
        return fld

    def level(self, name: str) -> Level:
        return self.levels[name]

    def field(self, name: str) -> Field:
        return self.fields[name]

    # the typed containment graph (parent maps by name, not a linear ladder)
    def children(self, parent_name: str) -> list[str]:
        """The child sets contained by `parent_name`. A parent may have MANY
        children of different roles (membrane / cytoplasm / nucleus / molecule),
        so a parent entity is a *bundle of fibres*, one per child set."""
        return [n for n, l in self.levels.items() if getattr(l, "parent_name", None) == parent_name]

    def parent_of(self, name: str) -> Optional[str]:
        """The set that contains `name` (None for a top-level set)."""
        return getattr(self.levels[name], "parent_name", None)

    def ancestors(self, name: str) -> list[str]:
        """Every set that contains `name`, however far up: nearest first, root last.

        The chain is not two levels deep (`organism -> tissue -> cell -> compartment -> mpm_particle` builds), and readers spelled `lvl.parent_name` reach one hop only. The walk is bounded, because a loop inside a substep is a hang, not an error."""
        out, seen, cur = [], {name}, self.parent_of(name)
        while cur is not None and cur not in seen and len(out) < len(self.levels):
            out.append(cur); seen.add(cur)
            cur = self.parent_of(cur)
        return out

    def lift_index(self, name: str, ancestor: str) -> torch.Tensor:
        """`[n]` index into `ancestor`, one row per element of `name`: the composition of the containment maps between them, so an Aggregate or Broadcast across k levels needs no new primitive (`H.delta(anc)[H.lift_index(child, anc)]` is a Broadcast, `index_add_(0, H.lift_index(child, anc), x)` an Aggregate).

        Cached, because it is read inside the CUDA-graph-captured MPM substep; invalidated by the identity of the `parent` buffers."""
        # containment is a relation too: record it under the `parent` role (beside `pre` / `post`), so a learnable replacing an operator cannot claim to walk nothing
        self._note_map(f"{name}->{ancestor}", "parent")
        chain = []
        cur = name
        while cur != ancestor:
            nxt = self.parent_of(cur)
            if nxt is None:
                raise ValueError(f"{ancestor!r} does not contain {name!r}: the containment chain "
                                 f"above it is {self.ancestors(name) or '(none)'}")
            chain.append(self.levels[cur].parent)
            cur = nxt
        if not chain:
            raise ValueError(f"lift_index({name!r}, {ancestor!r}): a set does not contain itself")
        key = (name, ancestor)
        sig = tuple(t.data_ptr() for t in chain)
        cache = getattr(self, "_lift_cache", None)
        if cache is None:
            cache = self._lift_cache = {}
        hit = cache.get(key)
        if hit is not None and hit[0] == sig:
            return hit[1]
        idx = chain[0]
        for nxt in chain[1:]:
            idx = nxt[idx]
        idx = idx.contiguous()
        cache[key] = (sig, idx)
        return idx

    # which relations an operator walks, MEASURED: `gather` and `scatter_along` are the only ways to traverse a relation, so recording them cannot drift from the code (a learnable may replace an operator only if it walks the same relations); off by default, one attribute test per gather
    def record_maps(self, on: bool = True) -> None:
        self._maps_log = {} if on else None

    def _note_map(self, edge_set: str, role: str) -> None:
        log = getattr(self, "_maps_log", None)
        if log is None:
            return
        log.setdefault(getattr(self, "_active_op", "?"), set()).add((edge_set, role))

    def measured_maps(self) -> dict:
        """{operator name: sorted [(edge_set, role), ...]} since `record_maps(True)`."""
        log = getattr(self, "_maps_log", None) or {}
        return {k: sorted(v) for k, v in log.items()}

    # incidence maps: gather / scatter along a named map (pre / post), not containment
    def gather(self, edge_set: str, role: str, block: str) -> torch.Tensor:
        """Gather an endpoint set's `block` state onto each edge along its `role`
        incidence map -- a lift along an incidence map. Returns `[E, w]`: the pre/post
        endpoint's block value per edge (e.g. `v_pre`, `v_post` for a synapse_ode)."""
        self._note_map(edge_set, role)
        es = self.level(edge_set)
        idx = es.incidence(role)                          # [E] endpoint index per edge
        ep = self.level(es.incidence_name(role))
        return ep.get(block)[..., idx, :]                 # [..., E, w]; batch-transparent

    def scatter_along(self, edge_set: str, role: str, values: torch.Tensor) -> torch.Tensor:
        """Sum per-edge `values` `[E, w]` onto the endpoint set along the `role`
        incidence map -- an Aggregate along an incidence map (e.g. synaptic current onto
        the post neuron). Occupancy-weighted, so dormant edges contribute nothing.
        Returns `[N_endpoint, w]`."""
        self._note_map(edge_set, role)
        es = self.level(edge_set)
        idx = es.incidence(role)
        ep = self.level(es.incidence_name(role))
        lead = values.shape[:-2]                          # () at B=1, (B,) when batched
        out = torch.zeros(*lead, ep.n, values.shape[-1], device=values.device,
                          dtype=values.dtype)
        out.index_add_(-2, idx, values * es.occ[:, None])
        return out

    # per-set delta accumulators (the integration scratch)
    def _delta_dim(self, lvl: "Level") -> int:
        """Width of a level's delta = its integrated coordinate block's width. For a
        spatial set that is `pos` (== self.dim, so byte-identical to the old
        `H.dim`-sized accumulator); for a neuron it is voltage's width, etc."""
        coord = lvl.state_schema.coordinate
        return coord.width if coord is not None else self.dim

    def zero_delta(self, dim: int = None) -> None:
        """Reset every level's delta accumulator to zeros (called once per tick). Each
        level's delta is sized to its coordinate block (pos for a spatial set, voltage
        for a neuron), not a global spatial dim."""
        if not self.levels:
            # a world of fields only (voxels are the state) has no set to integrate
            self._delta, self._delta_blocks = {}, {}
            return
        dev = next(iter(self.levels.values())).state.device
        # a delta is shaped like the state it integrates into, batch axis included, or `add_delta` would sum the batch away
        lead = {name: tuple(l.state.shape[:-2]) for name, l in self.levels.items()}
        self._delta = {name: torch.zeros(*lead[name], l.n,
                                         self._delta_dim(l) if dim is None else dim, device=dev)
                       for name, l in self.levels.items()}
        self._delta_blocks = {}                            # extra (non-coordinate) block deltas, filled lazily

    def add_delta(self, level_name: str, delta: torch.Tensor, block: str = None) -> None:
        """Add an operator's returned delta into its level's accumulator. `block` is the
        state block the delta integrates into (an operator's `INTEGRAND`); `None` or the
        coordinate block -> the COORDINATE accumulator (unchanged, what MPM reads via
        `delta()`); any OTHER dynamical block -> its own accumulator, so one set can carry
        several independently-integrated blocks (pos + chem + a0)."""
        coord = self.levels[level_name].state_schema.coordinate
        coord_name = coord.name if coord is not None else None
        if block is None or block == coord_name:
            if level_name not in self._delta:
                self._delta[level_name] = delta.clone()
            else:
                self._delta[level_name] = self._delta[level_name] + delta
        else:
            db = self._delta_blocks.setdefault(level_name, {})
            db[block] = delta.clone() if block not in db else db[block] + delta

    def renumber_set(self, level_name: str, keep, n_new: int | None = None) -> bool:
        """Permute a set's rows through `keep` (new index -> old index); True if it acted.

        The engine owns this because removal is the first operation that MOVES a row, and every per-set store must follow: the state, the occupancy, the coordinate delta accumulator and the extra first-order block deltas (`_delta_blocks`, keyed by set name). The accumulator is zeroed per tick and integrated at the end, so an operator renumbering in between would leave every delta pointing at another cell. The tail of `state` is left as it was (callers clone, write `[:n]`, and `occ` says those rows are not there): zeroing it would change bits."""
        import numpy as _np
        # `nn.ModuleDict` has no `.get`: test membership instead (a `.get` made this method return False on every call, silently skipping the permutes)
        lvl = self.levels[level_name] if level_name in self.levels else None
        if lvl is None or getattr(lvl, "state", None) is None:
            return False
        n = int(len(keep) if n_new is None else n_new)
        idx = torch.as_tensor(_np.asarray(keep), dtype=torch.long, device=lvl.state.device)
        if lvl.state.shape[0] < n:
            return False
        st = lvl.state.clone()
        st[:n] = lvl.state[idx]
        lvl.state = st
        if getattr(lvl, "occ", None) is not None:
            occ = torch.zeros(lvl.state.shape[0], device=lvl.state.device)
            occ[:n] = 1.0
            lvl.occ = occ
        d = self._delta.get(level_name)
        if d is not None:
            k = idx.to(d.device)
            d[:n] = d[k]
            d[n:] = 0.0
        for _b, t in (self._delta_blocks.get(level_name) or {}).items():
            if t is None:
                continue
            k = idx.to(t.device)
            t[:n] = t[k]
            t[n:] = 0.0
        return True

    def block_deltas(self, level_name: str) -> dict:
        """Accumulated deltas for a set's EXTRA (non-coordinate) dynamical blocks: {block: tensor}."""
        return self._delta_blocks.get(level_name, {})

    def delta(self, level_name: str) -> torch.Tensor:
        """The accumulated delta for a level (zeros if nothing wrote it)."""
        if level_name not in self._delta:
            lvl = self.levels[level_name]
            self._delta[level_name] = torch.zeros(*lvl.state.shape[:-2], lvl.n,
                                                  self._delta_dim(lvl), device=lvl.state.device)
        return self._delta[level_name]


# Operator: Base operator: a unit of dynamics with a typed signature, dispatched by the relation it acts on.
class Operator(nn.Module):
    """Base operator. Contract: `__init__(params, device)` reads tunables from the spec line; `forward(H, mask) -> {set: delta}` returns per-set time-derivative contributions (or `{}` if it mutates `H`).

    `KIND` and `SET` are stamped by `@register_operator`; `REQUIRES_PARAMS` / `REQUIRES_TYPE_PROPS` are the capability contract the validator checks. The subclasses below are thin tags, one per kind. Plain torch is the norm (index_add, pairwise)."""

    KIND: Optional[str] = None
    # the set this operator acts on (its primary input), stamped by `@register_operator(set=...)`; a set name, never a hierarchy depth
    SET: Optional[str] = None
    # typed signature (Plexus2 sec. 2.1): an operator is a morphism between sets; declarative metadata read by the validator, atlas and docs, not the engine; defaults describe a single-set morphism on SET
    # (a `MAPS` list of traversed relations was removed: declared on 23 of ~150 operators, so it could not be trusted; `H.gather` / `H.scatter_along` walk the spec's `sets.<name>.maps` at runtime)
    INPUTS: list = []                   # input sets (empty => [SET])
    OUTPUTS: list = []                  # output sets (empty => [SET])
    READS: list = []                    # state blocks consumed, by name, e.g. ["voltage", "w"]
    WRITES: list = []                   # state blocks produced, by name, e.g. ["voltage"]
    # which state BLOCK the returned delta integrates into; None = the set's coordinate block, a non-coordinate block (`chem`, `a0`) lets one set carry several independently integrated blocks
    INTEGRAND: Optional[str] = None
    # what the returned delta IS, one vocabulary shared with the spec `emit:` (see EMITS): "velocity" x += dt*delta; "acceleration" v += dt*delta, x += dt*v; "mpm_acceleration" an acceleration routed to the MPM substep; None = no set delta
    EMIT: Optional[str] = None
    REQUIRES_PARAMS: list = []          # param keys this operator must be given
    REQUIRES_TYPE_PROPS: list = []      # per-type node properties it reads (e.g. "youngs")
    # spatial dimensions supported; an operator declares which D it implements so the schema rejects an incompatible spec before the run (default 2D; 2D-specific kernels such as MLS-MPM stay [2])
    SUPPORTED_DIMS: list = [2]
    # per-implementation capability: whether gradients flow through this forward (a Neural-ODE vs a black-box solver), so an inverse loop can filter for the differentiable ones
    DIFFERENTIABLE: bool = True
    # opt-out of the frame-0 integration-invariant guard, for operators that legitimately write a set's `state` (structural, derived-state readouts)
    MAY_MUTATE_INTEGRATED_STATE: bool = False
    # world-model ledger metadata (plexus.tex Part IV): the mechanism this operator embodies, the morphologies it tends to produce, what each parameter means
    MECHANISM_TAGS: list = []           # e.g. ["long_range_attraction", "coarsening"]
    PARAM_ROLES: dict = {}              # e.g. {"sigma": "interaction_length", "gain": "field_sensitivity"}

    # the transitional fence (plexus.tex Part IV): a `TRANSITIONAL` operator wraps a mature multi-mechanism subsystem (e.g. the MLS-MPM solver) and may break the contract only while fenced: explicit, enumerated in `ARCHITECTURAL_DEBT`, isolated, scheduled for decomposition
    TRANSITIONAL: bool = False
    ARCHITECTURAL_DEBT: list = []       # human-readable list of the contract clauses it breaks
    # engine-provisioned requirements: per-node Level buffers read besides `state`, checked at forward() entry (the schema cannot see them, so a missing one must fail loudly)
    REQUIRES_BUFFERS: list = []         # per-node Level buffers, e.g. ["C", "F", "mass", "mu", "la", "p_vol"]
    REQUIRES_HSTATE: list = []          # Hierarchy scratch, e.g. ["cell_accel"] (parent->child broadcast)

    def __init__(self, params: Optional[dict] = None, device: str = "cpu"):
        super().__init__()
        self.params = params or {}
        self.device = device

    def tunable(self, value, default=None, tensor: bool = False):
        """Read a knob so that a LEARNABLE one survives construction.

        Every operator in the library writes `self.rate = float(params.get("rate", 0.0))`, and
        that `float()` is where the inverse half of Plexus dies: a spec may hand in a tensor with
        `requires_grad`, and the constructor silently casts the tape away before `forward` ever
        sees it. Two independent instruments (`operators/diff/audit.py` past the constructor,
        `operators/diff/certify.py` through a real spec) disagreed on exactly this set of knobs,
        and the gap between them WAS the coercion.

        The reference states the rule outright -- *store as an array anything you want to learn* --
        so: pass a tensor through untouched (moved to the device, tape intact), and coerce
        anything else to a float exactly as before. A forward run cannot tell the difference,
        which is why this is safe to apply to an operator whose numbers are already validated.

            self.epsilon = self.tunable(params.get("epsilon"), 1.0)
            self.epsilon = self.tunable(params.get("epsilon"), 1.0, tensor=True)    # a 0-dim float64 tensor on the device: any of them can be learned

        `tensor=True` (the curation of 2026-10-27, docs/CURATION_PLAN.md step 6) stores a Python number as a 0-dim FLOAT64 tensor on `self.device`, so the trainer's
        `{op, param}` learnable (trainer.py: `nn.Parameter(v.detach().clone())`) finds a tensor on every torch-path constant. float64 because that is what a Python float
        IS: `k * x` with a 0-dim float64 tensor k follows the same promotion as with the Python float (the dimensioned tensor's dtype wins, k is cast to it first), so a float32
        pipeline gets the same bits and a float64 pipeline keeps the full constant -- a float32 0-dim tensor would round the constant (0.0175 -> 0.017499999, 7e-10 off in
        tests/test_cell_chem_source_ptch.py). Exponents are excluded (`x ** k` picks another kernel for a tensor k than for a scalar); the bit-identical suite proves the rest per
        operator file. A constant a warp / triton launch reads (`float(self.x)` per substep) or that a CUDA graph captures stays a float: a tensor there would sync the device.
        """
        v = default if value is None else value
        if isinstance(v, torch.Tensor):
            return v.to(self.device)
        if tensor:
            return torch.tensor(float(v), dtype=torch.float64, device=self.device)
        return float(v)

    def forward(self, H: Hierarchy, mask: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @classmethod
    def signature(cls) -> dict:
        """The operator's typed signature (Plexus2 sec. 2.1): the sets it maps between and the state it reads / writes; `inputs` / `outputs` default to the acting set `SET`. Read by the atlas and the OperatorContract, never by the engine."""
        one = [cls.SET] if cls.SET else []
        return {
            "inputs":  list(cls.INPUTS) or one,
            "outputs": list(cls.OUTPUTS) or one,
            "reads":   list(cls.READS),
            "writes":  list(cls.WRITES),
        }


# Lateral: Within-set dynamics on E subset S_k x S_k (interaction, discrete Laplacian).
class Lateral(Operator):
    """Within-set dynamics on `E subset S_k x S_k` (interaction + discrete Laplacian)."""
    KIND = "lateral"


# Aggregate: children -> parent reduction over the partition `Level.parent`, occupancy-weighted.
class Aggregate(Operator):
    """children -> parent reduction over the partition `Level.parent` (occupancy-weighted)."""
    KIND = "aggregate"


# Broadcast: parent -> children lift along the partition `Level.parent`.
class Broadcast(Operator):
    """parent -> children lift along the partition `Level.parent`."""
    KIND = "broadcast"


# Exchange: set <-> field (or set <-> set) scatter / gather: P2G / G2P, secrete / sense, chemotaxis.
class Exchange(Operator):
    """set <-> field (or set <-> set) scatter/gather. Unifies P2G/G2P, secrete/sense,
    chemotaxis. Mutates a Field (or auxiliary set state) in place; returns `{}` -- OR,
    for a field->set coupling, returns a velocity/accel delta the engine integrates."""
    KIND = "exchange"


# FieldUpdate: A field's own dynamics, field -> field (diffuse, decay, react, playback); no set involved.
class FieldUpdate(Operator):
    """A field's OWN dynamics: field -> field (diffuse, decay, react, or playback of
    prescribed data). Mutates the Field's grid in place and returns `{}`. Distinct from
    `exchange`, which couples a SET to a field; here no set is involved."""
    KIND = "field"


# Structural: Changes cardinality or membership (divide, duplicate, die) on a fixed buffer, during the dynamics phase.
class Structural(Operator):
    """Changes cardinality / membership (divide, duplicate, die) on a fixed buffer
    via occupancy, DURING the dynamics phase. May emit per-node deltas during a
    gradual transition (e.g. mitosis) and only relabel membership at completion;
    returns `{}` otherwise. A dynamics kind: appears in `schedule:`, runs every
    tick it is scheduled for, for the length of the run."""
    KIND = "structural"
    MAY_MUTATE_INTEGRATED_STATE = True             # waking/retiring slots rewrites the state buffer


# Seed: Writes the initial state once, before any dynamics; not a dynamics kind and not a Structural.
class Seed(Operator):
    """Writes the initial state, once, before any dynamics runs. Not a dynamics kind and not a `Structural`.

    A Seed and a Structural both write buffers directly (both may call `Level.spawn`, both set `MAY_MUTATE_INTEGRATED_STATE`), but differ in WHEN: a Structural changes membership throughout the run, a Seed establishes the state the run starts from and never runs again. Merging them by inheritance is what let `cell_rd_seed mode: tip` re-stamp an activation cap every frame. A Seed is never scheduled: `engine.seed()` runs it once, and the schema rejects a spec that schedules one."""
    KIND = "seed"
    MAY_MUTATE_INTEGRATED_STATE = True             # establishing x_0 IS writing the state buffer


# Rewire: Rebuilds a relation E (`edge_index`) each tick, e.g. the membrane ring or a neighbour graph; emits no delta.
class Rewire(Operator):
    """Rebuilds a relation `E` (`edge_index`) -- e.g. the membrane ring or a
    neighbour graph -- each tick, so the relation tracks growth/division. Stores
    the new edges on `H`; emits no delta."""
    KIND = "rewire"
