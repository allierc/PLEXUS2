"""State: the fifth Plexus primitive, made first-class in code.

A set (`Level`) carries a `StateSchema`, a typed record of named `Block`s (`pos` / `vel` is one schema; `voltage` / `calcium`, a synaptic `g`, a metabolite `conc` are others). The engine sizes, integrates and records state by reading the schema, never by hard-coding `pos` / `vel`. Integration is declared per block (`Block.integration`):

  none                     not integrated: a parameter, readout, or frozen feature
  first_order              a state variable x with no rate block:  x += dt * delta
  coordinate               the block the set's operators move, with a rate block beside it (`pos`)
  rate                     the rate block of that coordinate (`vel`)

The schema only says which block is the coordinate and which is its rate; the ORDER of the motion is the operators' `emit`, resolved per set by `_resolve_emit`. `EMIT=velocity`: the delta is the velocity (a set with a rate block stores it there, `v = delta; x += dt * v`; one without, `x += dt * delta`). `EMIT=acceleration`: `v += dt * delta; x += dt * v`, which needs a rate block. `Block.boundary` says whether the block lives in the world box (`world`: clamp / wrap) or is unbounded (`free`, e.g. a voltage). The old spellings `second_order_coordinate` / `second_order_rate` are still read, as `coordinate` / `rate`.

A legacy `{block: (c0, c1)}` dict still works: `StateSchema.normalize` converts it, and a schema is dict-indexable (`schema['pos'] == (c0, c1)`, `'pos' in schema`)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# NONE, FIRST_ORDER, COORDINATE, RATE, INTEGRATIONS: The integration kinds of `Block.integration`.
NONE = "none"
FIRST_ORDER = "first_order"
COORDINATE = "coordinate"
RATE = "rate"
INTEGRATIONS = (NONE, FIRST_ORDER, COORDINATE, RATE)

# _LEGACY_INTEGRATION: The old spellings of the two kinds, still read from saved specs -> their new names (delete once no spec writes them).
_LEGACY_INTEGRATION = {"second_order_coordinate": COORDINATE, "second_order_rate": RATE}

# BOUNDARY_WORLD, BOUNDARY_FREE, BOUNDARIES: The boundary kinds of `Block.boundary`: a spatial coordinate clamped / wrapped to the world box, or unbounded (voltage, concentration, gating).
BOUNDARY_WORLD = "world"    # clamp / wrap to the world box (a spatial coordinate)
BOUNDARY_FREE = "free"      # unbounded (voltage, concentration, gating)
BOUNDARIES = (BOUNDARY_WORLD, BOUNDARY_FREE, None)


# Block: One named, contiguous chunk of a set's per-element state.
@dataclass(frozen=True)
class Block:
    """One named, contiguous chunk of a set's per-element state."""
    name: str
    width: int
    role: Optional[str] = None          # semantic label (coordinate/rate/voltage/gating/...) for docs + selectors
    integration: str = NONE             # one of INTEGRATIONS -- how the engine advances it in time
    boundary: Optional[str] = None      # BOUNDARY_WORLD | BOUNDARY_FREE | None
    record: bool = True                 # store this block in the trajectory
    # what the block's numbers are: a `plexus.units` quantity string ("volume", "1/T", "volume[polyhedron]"); None = UNDECLARED, silent (no claim, not a claim of dimensionlessness)
    unit: Optional[str] = None
    # what the block is, in words ("position in the world box", "membrane voltage"), shown in the watcher's plexus pane and the deck; an entity may override it per block with `state_notes=`
    describe: str = ""

    def __post_init__(self):
        object.__setattr__(self, "integration", _LEGACY_INTEGRATION.get(self.integration, self.integration))   # frozen: written through object
        if self.integration not in INTEGRATIONS:
            raise ValueError(f"block {self.name!r}: integration {self.integration!r} not in {INTEGRATIONS}")
        if self.boundary not in BOUNDARIES:
            raise ValueError(f"block {self.name!r}: boundary {self.boundary!r} not in {BOUNDARIES}")
        if self.width < 1:
            raise ValueError(f"block {self.name!r}: width must be >= 1, got {self.width}")


# StateSchema: An ordered list of Blocks with contiguous column offsets, dict-compatible (schema['pos'] == (c0, c1)).
class StateSchema:
    """An ordered list of ``Block``s with contiguous column offsets.

    Dict-compatible: ``schema['pos']`` returns the ``(c0, c1)`` slice, ``'pos' in
    schema`` works, and iteration yields block names -- so every legacy call site
    that treated the schema as a ``{name: (c0, c1)}`` dict keeps working unchanged.
    """

    def __init__(self, blocks):
        self.blocks: list[Block] = list(blocks)
        self._slices: dict[str, tuple[int, int]] = {}
        self._by_name: dict[str, Block] = {}
        c = 0
        for b in self.blocks:
            if b.name in self._slices:
                raise ValueError(f"duplicate block name {b.name!r} in StateSchema")
            self._slices[b.name] = (c, c + b.width)
            self._by_name[b.name] = b
            c += b.width
        self.dim = c                     # total state width

    # dict compatibility (legacy `state_schema['pos'] == (c0, c1)`)
    def __getitem__(self, name): return self._slices[name]
    def __contains__(self, name): return name in self._slices
    def __iter__(self): return iter(self._slices)
    def __len__(self): return len(self._slices)
    def keys(self): return self._slices.keys()
    def items(self): return self._slices.items()
    def values(self): return self._slices.values()
    def get(self, name, default=None): return self._slices.get(name, default)

    def block(self, name) -> Block: return self._by_name[name]
    def slice(self, name) -> tuple[int, int]: return self._slices[name]

    # the integration roles the engine's integrator reads
    @property
    def coordinate(self) -> Optional[Block]:
        """The block the engine advances: the ``coordinate`` block if the set has one (it has
        a rate block beside it), else the first ``first_order`` block, else ``None`` (a set
        with no engine-integrated state)."""
        for b in self.blocks:
            if b.integration == COORDINATE:
                return b
        for b in self.blocks:
            if b.integration == FIRST_ORDER:
                return b
        return None

    @property
    def rate(self) -> Optional[Block]:
        """The rate block of the coordinate, or ``None`` for a set with no rate block
        (first-order or non-integrated)."""
        for b in self.blocks:
            if b.integration == RATE:
                return b
        return None

    @property
    def recorded(self) -> list[Block]:
        return [b for b in self.blocks if b.record]

    def as_dict(self) -> dict[str, tuple[int, int]]:
        return dict(self._slices)

    def __repr__(self):
        body = ", ".join(f"{b.name}[{b.width},{b.integration}]" for b in self.blocks)
        return f"StateSchema({body})"

    # the compatibility shim
    @classmethod
    def normalize(cls, schema) -> "StateSchema":
        """Return ``schema`` if it is already a ``StateSchema``; otherwise treat it as
        a legacy ``{block: (c0, c1)}`` dict and normalize it.

        The legacy dict is the spatial ``pos``/``vel`` layout every current spec uses;
        it MUST normalize to a schema that integrates byte-identically to the old
        hard-coded path: ``pos`` -> ``coordinate`` in the ``world`` box,
        ``vel`` -> ``rate``. Any other legacy block is carried through as
        a non-integrated (``none``) block so nothing is silently integrated.
        """
        if isinstance(schema, StateSchema):
            return schema
        items = sorted(schema.items(), key=lambda kv: kv[1][0])   # contiguous by start column
        blocks = []
        for name, (c0, c1) in items:
            w = c1 - c0
            if name == "pos":
                blocks.append(Block("pos", w, role="coordinate", unit="length",
                                    integration=COORDINATE, boundary=BOUNDARY_WORLD))
            elif name == "vel":
                blocks.append(Block("vel", w, role="rate", unit="velocity",
                                    integration=RATE))
            else:
                blocks.append(Block(name, w, integration=NONE))
        return cls(blocks)


# spatial_schema: The dimension-aware pos / vel schema, state width 2*dim -> StateSchema.
def spatial_schema(dim: int) -> StateSchema:
    """The dimension-aware ``pos``/``vel`` schema (state width ``2*dim``) -- the spatial
    default every current set uses. ``pos`` is the coordinate clamped/wrapped to the
    world box; ``vel`` is its rate. Byte-identical replacement for the old
    ``{'pos': (0, D), 'vel': (D, 2D)}`` dict from ``engine._dim_schema``."""
    return StateSchema([
        Block("pos", dim, role="coordinate", unit="length",
              integration=COORDINATE, boundary=BOUNDARY_WORLD,
              describe="position in the world box"),
        # vel is not recorded: the trajectory has always stored pos only, and recording vel would break byte-identical output
        Block("vel", dim, role="rate", unit="velocity", integration=RATE, record=False,
              describe="velocity: how the position changes per unit time"),
    ])


# schema_from_spec: Build a StateSchema from a spec `state:` block -> StateSchema.
def schema_from_spec(state_decl: dict) -> StateSchema:
    """Build a ``StateSchema`` from a spec ``state:`` block. Two accepted forms::

        state: {voltage: 1, calcium: 1}                      # width only
        state: {voltage: {width: 1, integration: first_order, boundary: free}}

    A bare integer is shorthand for a ``first_order``, ``free``, recorded block of
    that width (the common non-spatial case: a scalar that integrates its own ODE and
    lives in no world box). The explicit-dict form overrides any field.
    """
    blocks = []
    for name, decl in state_decl.items():
        if isinstance(decl, dict):
            w = int(decl.get("width", 1))
            role = decl.get("role")
            blocks.append(Block(
                name, w,
                role=role,
                integration=decl.get("integration", FIRST_ORDER),
                boundary=decl.get("boundary", BOUNDARY_FREE),
                record=bool(decl.get("record", True)),
                unit=decl.get("unit", _unit_from_role(role)),
                describe=str(decl.get("describe") or decl.get("title") or ""),       # the spec's own words for the block
            ))
        else:
            blocks.append(Block(name, int(decl), integration=FIRST_ORDER, boundary=BOUNDARY_FREE))
    return StateSchema(blocks)


# _ROLE_UNITS: The two roles that already say their own dimension (coordinate = length, rate = velocity), because the engine reads both to integrate; nothing else is guessed from a block's NAME (`area` is a count on one set).
_ROLE_UNITS = {"coordinate": "length", "rate": "velocity"}


# _unit_from_role: The unit a block's role implies -> str | None.
def _unit_from_role(role):
    return _ROLE_UNITS.get(role)


# _entity_schema: An entity's declared STATE_SCHEMA resolved for this run's dim -> StateSchema | None.
def _entity_schema(decl, dim: int) -> StateSchema | None:
    """An entity's declared `STATE_SCHEMA` resolved for this run's `dim`, or None.

    A `StateSchema` is fixed; a callable `dim -> StateSchema` is resolved here (spatial entities use
    `spatial_schema`); the legacy `{block: (c0, c1)}` dict returns None -- those hard-code 2D, and
    honouring one would silently truncate a 3D set's `pos`/`vel` to (x, y)."""
    if decl is None:
        return None
    if isinstance(decl, StateSchema):
        return decl
    if callable(decl):
        return decl(dim)
    return None                                    # legacy {block: (c0, c1)} dict: a hint only


# _entity_meta: State schema, render style and depth of a set from the entity registry -> (StateSchema | None, render, depth).
def _entity_meta(sname: str, dim: int = 2, entity: str | None = None) -> tuple[StateSchema | None, dict, int]:
    """(state_schema, render, depth) of a set from the entity registry.

    `schema` is None when the name is unregistered or declares no honourable schema (the caller
    falls back to `spatial_schema(dim)`); `depth` is the hierarchy depth. `entity` overrides the
    lookup by name: a set's name need not be the entity that provides its state (fifteen organelle
    sets can all be `mpm_particle`)."""
    from plexus.models.registry import get_entity       # lazy: entities.py imports this module
    from plexus.models.entities import DEFAULT_RENDER
    try:
        ent = get_entity(entity or sname)
        schema = _entity_schema(getattr(ent, "STATE_SCHEMA", None), dim)
        render = getattr(ent, "RENDER", None) or DEFAULT_RENDER
        depth = getattr(ent, "DEPTH", None)
        depth = depth if depth is not None else 0
    except KeyError:
        schema, render, depth = None, DEFAULT_RENDER, 0
    return schema, render, depth


# _resolve_schema: The StateSchema of a set: own state:, relation, entity, else spatial default -> StateSchema.
def _resolve_schema(s: dict, D: int, sname: str | None = None, spatial: bool = False) -> StateSchema:
    """The set's StateSchema, first match wins:
        1. the set's own `state:` block (extends the spatial default if the set is `spatial` and
           declares no `pos`)
        2. an empty schema for a relation (`maps:` declared)
        3. the entity's declared schema (`models/entities.py`)
        4. `spatial_schema(D)`: pos/vel for dimension D"""
    if "state" in s:
        declared = schema_from_spec(s["state"])
        # a spatial set (placed by `seed_positions`) keeps pos/vel and gains the declared blocks
        if spatial and "pos" not in declared:
            return StateSchema(list(spatial_schema(D).blocks) + list(declared.blocks))
        return declared
    # a relation (e.g. a half-edge set) has no position: no `pos` block, so nothing is seeded or recorded
    if s.get("maps"):
        return StateSchema([])
    if sname is not None:
        ent_schema, _render, _depth = _entity_meta(sname, D, s.get("entity"))
        if ent_schema is not None:
            return ent_schema
    return spatial_schema(D)
