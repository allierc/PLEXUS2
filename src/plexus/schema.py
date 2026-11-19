"""The spec schema: load a `spec.yaml` and VALIDATE it, the contract gatekeeper.

A run is one declarative spec (`general`, `sets`, `fields`, optional `seed`, `operators` with
selectors, `schedule`, `plotting`). `load` parses it into typed objects (`Spec`, `OpSpec`,
`Selector`) and fails loudly on anything off: an unregistered operator, an unknown set / field
reference, a malformed selector, type fractions not summing to one, a required property missing
along the containment chain, a schedule name that matches no declared operator. A spec that loads here is runnable by
the engine.

It imports the registry (operator names and capability contracts) but never the engine: pure
validation. The operator modules must already be imported so the registry is populated."""

from __future__ import annotations

from plexus.paths import warn

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import yaml

if TYPE_CHECKING:                  # annotations only
    import torch
    from plexus.models.base import Hierarchy

from plexus.units import Units, parse as parse_units

# importing plexus populates base + registry; operator modules register themselves
from plexus.models import registry
from plexus.models.base import KINDS, EMITS
from plexus.models.state import INTEGRATIONS, _LEGACY_INTEGRATION

# _SELECTOR_RE: A selector, `set`, `set.block` or `set[attr=value]` (also `set.block[attr=value]`); the value is any label up to the bracket (`ciliary band`).
_SELECTOR_RE = re.compile(r"^(?P<set>\w+)(?:\.(?P<block>\w+))?(?:\[(?P<attr>\w+)=(?P<val>[^\]]+)\])?$")


# Selector: What an operator writes: `set` (every node), `set[attr=val]` (a subset), and optionally the state block of it, `set.block`.
@dataclass
class Selector:
    """`set` (every node) or `set[attr=val]` (a subset, re-checked each tick)."""
    set: str
    attr: Optional[str] = None
    val: Optional[str] = None
    block: Optional[str] = None     # the state block written, for an operator that writes one block of its set (`at: cell.chem`)

    @classmethod
    def parse(cls, s: str) -> "Selector":
        m = _SELECTOR_RE.match(str(s).strip())
        if not m:
            raise ValueError(f"bad selector {s!r} (expected 'set', 'set.block' or 'set[attr=val]')")
        v = m["val"]
        return cls(m["set"], m["attr"], v.strip() if v is not None else None, m["block"])


# _coerce: A selector value as int or float when numeric, else the string.
def _coerce(s: str):
    """Selector value as int / float when numeric (`done=0` compares to 0), else the string."""
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


# _selector_mask: The live boolean mask an operator acts on, from its at: selector -> bool [n] | None.
def _selector_mask(H: Hierarchy, sel: Selector) -> torch.Tensor:
    """Live boolean mask [n] an operator acts on, from its `at:` selector; recomputed each tick.
      at: <field>      None (field-internal operator: no per-node mask)
      at: set          lvl.active (live nodes, occ > 0)
      at: set[type=a]  live & node_type == a
      at: set[attr=v]  live & lvl.attr == v (any per-node buffer, e.g. cell[done=0])"""
    if sel.set not in H.levels:                        # a field-internal operator (at: <field>)
        return None                                    # has no per-node mask
    lvl = H.level(sel.set)
    if sel.attr is None:
        return lvl.active                              # all live nodes
    if sel.attr == "type":                             # type name -> node_type index
        # a contained set wears its parent's types (mpm_particle has no `types:`; the bodies are typed on the cell)
        _nt, _names = getattr(lvl, "node_type", None), list(getattr(lvl, "type_names", []) or [])
        if _nt is None or not _names:
            _p = getattr(lvl, "parent_name", None)
            _par = getattr(lvl, "parent", None)
            while _p in H.levels and _par is not None:
                _pl = H.level(_p)
                if getattr(_pl, "node_type", None) is not None and (getattr(_pl, "type_names", []) or []):
                    _nt, _names = _pl.node_type[_par], list(_pl.type_names)
                    break
                _par, _p = _pl.parent[_par] if getattr(_pl, "parent", None) is not None else None, getattr(_pl, "parent_name", None)
        if _nt is None or sel.val not in _names:
            raise ValueError(f"selector {sel.set!r}[type={sel.val!r}]: neither the set nor its parents "
                             f"declare that type (they have {_names or 'none'})")
        return lvl.active & (_nt == _names.index(sel.val))
    # general set[attr=val]: match a per-node buffer (e.g. cell[done=0] -> lvl.done == 0)
    if not hasattr(lvl, sel.attr):
        raise ValueError(f"selector {sel.set!r}[{sel.attr}={sel.val}] has no per-node "
                         f"buffer {sel.attr!r} on the set (set it via an operator first).")
    return lvl.active & (getattr(lvl, sel.attr) == _coerce(sel.val))


# OpSpec: One operator line of a spec: `op`, selector `on` (what it writes), what it reads (`frm`, `via`), `params`.
@dataclass
class OpSpec:
    op: str
    on: Selector                    # `at:`: the one set or field the operator WRITES
    frm: Optional[object] = None    # `from:`: what it reads: a field or set name, a `set.block`, or a {role: ref} mapping (see `split_ref`)
    via: Optional[object] = None    # the edge set that carries the relation: its name, or a {role: edge set} mapping
    impl: Optional[str] = None      # which VARIANT of the contract: a `model` (another hypothesis) or an `implementation` (the same biology computed differently)
    params: dict = field(default_factory=dict)
    mask: Optional[Selector] = None  # the rows the operator acts on, when `at:` names a FIELD: the set it reads (`from: cell[done=0]`)

    def mask_selector(self) -> Selector:
        """The selector the row mask is built from: `at:`, or, for an operator that writes a field from a set (a particle-to-grid transfer), the set it reads."""
        return self.mask or self.on

    def ctor_params(self) -> dict:
        """The dict an operator class is constructed with: the line's parameters plus its endpoints (`from`, `via`) and the name of the set or field it writes (`_at`)."""
        return {**self.params, "from": self.frm, "via": self.via, "_at": self.on.set, "_at_block": self.on.block}


# Spec: A validated run: general scalars, sets, fields, operators, seed operators, schedule, plotting.
@dataclass
class Spec:
    name: str
    seed: int                       # RNG seed (`general.seed`); not the `seed:` section below
    n_frames: int
    dt: float
    sets: dict
    fields: dict
    operators: list[OpSpec]
    schedule: list
    # `title` is what the run is CALLED in a caption, slide or paper; `name` is the terse machine-shaped id of its folder, trajectory and movie
    title: str = ""
    seed_ops: list[OpSpec] = field(default_factory=list)   # the `seed:` section (x_0, run once by `engine.seed()`, never scheduled)
    obstacles: list = field(default_factory=list)   # wall rectangles [x0,y0,x1,y1] or discs [cx,cy,r]
    boundary: str = "wall"                           # 'wall' (clamp) or 'periodic' (torus)
    world: float = 1.0                               # domain width (= world_size[0]); legacy 2D scalar
    dim: int = 2                                     # spatial dimensions (the global dimension contract)
    world_size: list = field(default_factory=lambda: [1.0, 1.0])   # per-axis box [w0 .. w_{D-1}]
    plotting: dict = field(default_factory=dict)     # render STYLE (colormap, point_size, ...) — read by plexus.plot
    # which driver runs the schedule: `default` is plexus.engine.run, a name N resolves to plexus.<N>_engine.run (a mode of running, not other physics)
    engine: str = "default"
    # what is fitted, validated in `_parse_learnable`: [{block: w, of: recurrent}] makes a state block a tensor leaf; empty on every forward spec
    learnable: list = field(default_factory=list)
    record_cap: int = 10000                          # max recorded SET (position) frames; the trajectory is strided if n_frames exceeds it
    field_record_cap: int = 256                      # max recorded FIELD (grid) frames — fields are large, so a tighter cap
    # physical scale from `general.units:` (length_um, time_s, force_nN; never inferred); absent = dimensionless; `dt` is in seconds
    units: "Units" = field(default_factory=lambda: Units(declared=False))
    # save the trajectory: True every frame, False none, None = the legacy `record_cap` striding
    save_data: bool = None
    # warm-up ticks before frame 0, not recorded: None = the seed's settle window (`ref_frame`), 0 = none
    warmup: int = None
    # wall-clock minutes between partial saves of `trajectory.npz` during a run (0 = only at the end): a long run can be scored while it goes
    save_every_min: float = 0.0
    # every that many frames the rows recorded since the last chunk are written to `chunks/rows_<first>_<last>.npz` and NO whole trajectory is
    # written at the end (0 = off): a run too large for one file at its end is read, and drawn, while it goes
    save_chunk_frames: int = 0
    # with save_chunk_frames: at every recorded row the live members of one set whose parent's centre lies inside a box are kept as well, {set, box: [[x0, y0], [x1, y1]],
    # margin}: the window of a world too large to record whole (window__pos, window__parent, window__offsets in each chunk)
    save_chunk_window: dict = None
    # per-segment frame duration `[{frames: N, dt: X}, ...]`, frames summing to n_frames (a speed ramp); None = every frame lasts `dt`
    dt_schedule: list = None


# split_ref: an operator line's ENDPOINTS, in one syntax. `at:` is the one set or field the operator WRITES; everything it reads is `from:`, and the edge set that carries a relation is `via:`. There is no `to:`:
#   at:   cell                         the set or field that receives the operator's result
#   from: pt.vel                       one source: a set and one of its state blocks; a bare name, no dot, is a whole field or a whole set (`from: mpm_grid`, `from: mpm_particle`)
#   via:  junction                     the edge set that carries it; its `pre` / `post` legs give the direction, so `from:` must be on `pre` and `at:` on `post`
#   from: {chem: cell.chem, forcing: stimulus.u}          an operator with several inputs: a mapping of the ROLE names it declares to a ref (or a list of refs)
#   via:  {chem: junction}             a mapping of the same role names to the edge set that crosses (only for the roles that cross a relation)
def split_ref(ref):
    """'set.block' -> (set, block); a bare name -> (None, name) (a whole field or set)."""
    sname, dot, bname = str(ref).partition(".")
    return (sname, bname) if dot else (None, sname)


def _check_refs(name, o, raw):
    """Load-time checks of an operator line's `from:` / `via:`: every named set, block, field and edge set exists, and a relation is read from its `pre` set and written to its `post` set (the set at `at:`)."""
    if "to" in o:
        raise ValueError(f"operator {name!r}: `to:` no longer exists. The set or field an operator writes is `at:`, and what it reads is `from:` "
                         f"(`at: mpm_grid` with `from: mpm_particle` is the particle-to-grid transfer).")
    def one(ref, key):
        if "[" in str(ref):                            # a set with a row selector: `from: cell[done=0]`
            ref = Selector.parse(ref).set
        sname, bname = split_ref(ref)
        if sname is None:
            if bname not in raw["fields"] and bname not in raw["sets"]:
                raise ValueError(f"operator {name!r} references unknown field or set {bname!r} in `{key}:`")
            return
        if sname not in raw["sets"]:
            raise ValueError(f"operator {name!r}: `{key}: {ref}` names the unknown set {sname!r}")
        st = raw["sets"][sname].get("state")
        if isinstance(st, dict) and bname not in st:
            raise ValueError(f"operator {name!r}: `{key}: {ref}`: the set {sname!r} has no state block {bname!r} (it has: {', '.join(st)})")

    v = o.get("from")
    for ref in ((v.values() if isinstance(v, dict) else [v]) if v is not None else []):
        for r in (ref if isinstance(ref, list) else [ref]):                        # a role may name several refs: `index: [dn_cells, dn_tissue]`
            one(r, "from")
    via = o.get("via")
    if via is None:
        return
    edges = dict(via) if isinstance(via, dict) else {None: via}
    for role, e in edges.items():
        es = raw["sets"].get(e)
        if es is None or not es.get("edge_set"):
            raise ValueError(f"operator {name!r}: `via: {e}` is not a declared edge set (a set with `edge_set: true`)")
        v = o.get("from")
        ref = (v.get(role) if isinstance(v, dict) else (v if role is None else None)) if v is not None else None     # a single `from:` is checked against a single `via:`; with roles, only the role's own
        if isinstance(ref, str) and "." in ref and split_ref(ref)[0] != es["pre"]:
            raise ValueError(f"operator {name!r}: `from: {ref}` is on the set {split_ref(ref)[0]!r}, but the edge set {e!r} has pre: {es['pre']!r}")
        at = Selector.parse(o["at"]).set
        if not isinstance(via, dict) and at != es["post"]:                          # one relation: `at:` is the set it writes; with several roles each operator names what it does with them
            raise ValueError(f"operator {name!r}: `at: {at}`, but the edge set {e!r} has post: {es['post']!r}; an operator that reads across a relation writes its `post` set")


# _RESERVED: Operator-line keys that are not operator parameters.
_RESERVED = {"op", "at", "from", "via", "implementation", "model"}
# _BUILTIN_STEPS: Schedule builtins: none, every name in the schedule must resolve to a declared operator.
_BUILTIN_STEPS: set = set()


# _parse_dt_schedule: Parse `general.dt_schedule` -> [(frames, dt), ...] | None.
def _parse_dt_schedule(raw, n_frames):
    """`general.dt_schedule` -> [(frames, dt), ...] or None. See `Spec.dt_schedule`."""
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError("general.dt_schedule must be a non-empty list of {frames: N, dt: X} segments")
    out = []
    for seg in raw:
        if not isinstance(seg, dict) or set(seg) != {"frames", "dt"}:
            raise ValueError(f"general.dt_schedule segment {seg!r} must be exactly {{frames: N, dt: X}}")
        f, d = int(seg["frames"]), float(seg["dt"])
        if f <= 0 or d <= 0:
            raise ValueError(f"general.dt_schedule segment {seg!r}: frames and dt must be positive")
        out.append((f, d))
    if sum(f for f, _ in out) != n_frames:
        raise ValueError(f"general.dt_schedule covers {sum(f for f, _ in out)} frames but n_frames is "
                         f"{n_frames}; the segments must add up to the run")
    return out


# _parse_learnable: Validate `learnable:` -> list of {block, of} dicts (nothing is built here).
def _parse_learnable(block, raw):
    """Validate `learnable:` and return it as a list of dicts; nothing is built here.

    One form: `{block: w, of: recurrent}` makes a state block a tensor leaf (parameter recovery: fit
    synaptic weights, time constants, conductances). The law form `{replaces:, with:}` and unknown
    keys are REFUSED, not ignored: nothing instantiates them, so a run would silently use the
    original operator."""
    if not block:
        return []
    if not isinstance(block, list):
        raise ValueError("`learnable:` is a LIST of entries, one per thing fitted")
    sets = raw.get("sets", {})
    out = []
    for i, e in enumerate(block):
        if not isinstance(e, dict):
            raise ValueError(f"learnable[{i}] is not a mapping")
        if "replaces" in e:
            raise ValueError(
                f"learnable[{i}] is a law substitution ({{replaces:, with:}}), which nothing runs: "
                f"the engine never instantiated one, and the package meant to was deleted. Only "
                f"{{block:, of:}} is fitted today. A fitted law will be a variant of the operator "
                f"it replaces, registered beside its `model:` and `implementation:`.")
        for k in ("block", "of"):
            if k not in e:
                raise ValueError(f"learnable[{i}] needs `{k}:`")
        unread = sorted(set(e) - {"block", "of"})
        if unread:
            raise ValueError(
                f"learnable[{i}] has key(s) {unread} that nothing reads. A block is fitted as one "
                f"free value per element; a key that changes that would be silently dropped.")
        if e["of"] not in sets:
            raise ValueError(
                f"learnable[{i}] fits block {e['block']!r} of {e['of']!r}, which is not a "
                f"declared set. Declared: {sorted(sets)}")
        out.append(dict(e))
    return out


# load: Parse and validate a spec file -> Spec (a spec that loads is runnable by the engine).
def load(path: str) -> Spec:
    with open(path) as f:
        raw = yaml.safe_load(f)

    # run / world scalars live under `general:`; fall back to the top level (back-compat)
    g = raw.get("general", {})
    def gv(key, default=None):
        return g.get(key, raw.get(key, default))

    for key in ("sets", "fields", "operators", "schedule"):
        if key not in raw:
            raise ValueError(f"simulation missing required key: {key!r}")
    if gv("name") is None:
        raise ValueError("simulation missing required key: 'name' (under general:)")

    # --- the dimension contract: dim (default 2) + per-axis world box -------- #
    dim = int(gv("dim", 2))
    world_raw = gv("world", 1.0)
    if isinstance(world_raw, (list, tuple)):
        world_size = [float(x) for x in world_raw]
    else:
        world_size = [float(world_raw)] + [1.0] * (dim - 1)   # scalar: axis-0 = width, rest = 1
    if len(world_size) != dim:
        raise ValueError(f"general.world has {len(world_size)} entries but dim={dim}; "
                         f"give a length-{dim} box e.g. world: {[1.0] * dim}")

    # --- sets: type fractions sum to 1; buffer (if given) >= n -------------- #
    for sname, s in raw["sets"].items():
        types = s.get("types", {})
        if types:
            # `fraction` (share) or `count` (number of elements), exactly one per set; counts convert to fractions here so every other reader is unchanged
            n_counted = sum(1 for t in types.values() if "count" in t)
            n_frac = sum(1 for t in types.values() if "fraction" in t)
            if n_counted and n_counted != len(types):
                missing = [k for k, t in types.items() if "count" not in t]
                raise ValueError(
                    f"set {sname!r}: {n_counted} of {len(types)} types declare `count:` -- "
                    f"missing on {missing}. A partition is stated one way for the whole set.")
            if n_counted and n_frac:
                both = [k for k, t in types.items() if "count" in t and "fraction" in t]
                raise ValueError(
                    f"set {sname!r} types {both} declare BOTH `count:` and `fraction:`. They are "
                    f"two statements of one partition -- give the count, or give the share.")
            if n_counted:
                # `count` is checked against what the set holds: `n` for a root set, `per_parent` for a contained one
                live = s.get("n", s.get("per_parent"))
                if isinstance(live, dict):
                    raise ValueError(
                        f"set {sname!r} uses `count:` on its types and a per-type `per_parent:` "
                        f"mapping. `count` counts THIS set's elements and the mapping counts its "
                        f"CHILDREN's -- put the mapping on the child set, where it belongs.")
                if live is None:
                    raise ValueError(f"set {sname!r} uses `count:` but declares neither `n:` nor "
                                     f"`per_parent:`, so there is nothing for the counts to add up to")
                tot = sum(int(t["count"]) for t in types.values())
                if tot != int(live):
                    raise ValueError(
                        f"set {sname!r} type counts sum to {tot:,}, but the set holds "
                        f"{int(live):,}. Every element belongs to exactly one type.")
                for t in types.values():
                    t["fraction"] = int(t["count"]) / float(live)
            else:
                total = sum(t["fraction"] for t in types.values())
                if abs(total - 1.0) > 1e-6:
                    raise ValueError(f"set {sname!r} type fractions sum to {total}, must be 1.0")
        # `buffer` = allocated slots of a cardinality-changing set (`occ` marks the live subset); it must hold the initial set
        buf = s.get("buffer")
        if buf is not None:
            live0 = s.get("n", s.get("per_parent"))
            if isinstance(live0, dict):
                live0 = max(int(v) for v in live0.values())   # the largest block a parent may hold
            if live0 is not None and int(buf) < int(live0):
                raise ValueError(
                    f"set {sname!r} buffer={buf} is smaller than its initial size {live0}; "
                    f"a structural (divide/duplicate) run needs buffer >= initial.")
        # `mesh:` names the half-edge SET of the surface; that set's `maps:` give the topology (`srce` / `trgt` into this set, `face` into the cell set)
        # validated loudly: a typo (`half-edge`) or `mesh:` on the wrong set would otherwise parse clean and allocate nothing
        mk = s.get("mesh")
        if mk is not None:
            if s.get("pre") is not None or s.get("post") is not None:
                raise ValueError(
                    f"set {sname!r} declares `mesh:` and is an EDGE-SET (it has pre/post). An "
                    f"edge-set's elements are connections; a mesh's are vertices.")
            if int(raw.get("general", {}).get("dim", 2)) != 3:
                raise ValueError(
                    f"set {sname!r} declares `mesh: {mk}` but the model is "
                    f"{raw.get('general', {}).get('dim')}D -- every mesh operator declares "
                    f"SUPPORTED_DIMS = [3].")
            if mk not in raw["sets"]:
                raise ValueError(
                    f"set {sname!r} declares `mesh: {mk}`, which is not a set in this spec "
                    f"(have: {', '.join(sorted(raw['sets']))}). `mesh:` names the half-edge SET "
                    f"whose `maps:` say what the topology is; it is no longer a kind.")
            if s.get("cell_set") is not None:
                raise ValueError(
                    f"set {sname!r} still declares `cell_set: {s['cell_set']}`. That key retired: "
                    f"a face of the mesh IS a cell, and the pairing is now `maps.face` on the "
                    f"{mk!r} set. Delete it from here and from every operator line.")
            # `face` is optional: `srce` / `trgt` make it a mesh, `face` makes it a cellular surface; no `face` = a 1D curve (nF = 0, e.g. a cilium), which needs a declared table because row order cannot say a nine-doublet ring
            hs = raw["sets"][mk]
            for role in ("srce", "trgt"):
                if role not in (hs.get("maps") or {}):
                    raise ValueError(
                        f"set {mk!r} is used as {sname!r}'s mesh but declares no `maps.{role}:`. An "
                        f"element of a mesh set runs BETWEEN two elements of {sname!r}, so both "
                        f"ends are functions out of it and both must be declared. (`maps.face:` is "
                        f"the third, and it is optional: with it the mesh is a closed surface whose "
                        f"faces are cells, without it the mesh is a curve -- a filament, a chain, "
                        f"an axoneme -- and `nF` stays 0.)")

        # `maps:` = named functions OUT of this set, one row to one row of the codomain
        mp = s.get("maps")
        if mp is not None:
            if not isinstance(mp, dict) or not mp:
                raise ValueError(f"set {sname!r}: `maps:` must be a mapping of role -> set name")
            for role, tgt in mp.items():
                if tgt not in raw["sets"]:
                    raise ValueError(
                        f"set {sname!r} map {role!r} lands in {tgt!r}, which is not a set in this "
                        f"spec (have: {', '.join(sorted(raw['sets']))})")
                if tgt == sname:
                    raise ValueError(
                        f"set {sname!r} map {role!r} lands in itself. A map goes OUT of a set.")
            if s.get("pre") is not None or s.get("post") is not None:
                raise ValueError(
                    f"set {sname!r} declares both `maps:` and `pre`/`post`. They are the same idea "
                    f"with different lifetimes -- pre/post are a static edge list, `maps` are "
                    f"owned by the topology operators -- so a set has one or the other.")

        # `entity:` names the registered kind that provides the set's state when its own name is not that kind; validated here because an unregistered name silently falls back to pos/vel and an MPM set dies later on a missing `F`
        ent = s.get("entity")
        if ent is not None:
            from plexus.models.registry import _ENTITY_REGISTRY
            if ent not in _ENTITY_REGISTRY:
                raise ValueError(
                    f"set {sname!r} declares `entity: {ent}`, which is not a registered entity "
                    f"(have: {', '.join(sorted(_ENTITY_REGISTRY))})")

        # optional `state:` = the set's StateSchema (absent = spatial pos/vel); each entry is a width or {width, integration, boundary, role, record, unit}
        # `unit:` is not validated: the units layer is warn-only, a typo must never stop a run
        st = s.get("state")
        if st is not None:
            if not isinstance(st, dict) or not st:
                raise ValueError(f"set {sname!r} `state:` must be a non-empty mapping of block -> width|dict")
            for bname, decl in st.items():
                if isinstance(decl, dict):
                    if int(decl.get("width", 1)) < 1:
                        raise ValueError(f"set {sname!r} state block {bname!r}: width must be >= 1")
                    integ = decl.get("integration", "first_order")
                    if _LEGACY_INTEGRATION.get(integ, integ) not in INTEGRATIONS:       # the old spellings still load
                        raise ValueError(f"set {sname!r} state block {bname!r}: integration {integ!r} "
                                         f"not one of {INTEGRATIONS}")
                    bnd = decl.get("boundary", "free")
                    if bnd not in ("free", "world"):
                        raise ValueError(f"set {sname!r} state block {bname!r}: boundary {bnd!r} "
                                         f"must be 'free' or 'world'")
                elif int(decl) < 1:
                    raise ValueError(f"set {sname!r} state block {bname!r}: width must be >= 1")
        # `edge_set`: elements are connections joined to endpoint sets by `pre` / `post`; needs a containing `parent`, both endpoint sets, and `edges` or `edges_file`
        if s.get("edge_set"):
            if "parent" not in s:
                raise ValueError(f"edge-set {sname!r} needs a `parent:` (its containing set, e.g. network)")
            for role in ("pre", "post"):
                if role not in s:
                    raise ValueError(f"edge-set {sname!r} needs a `{role}:` endpoint set")
                if s[role] not in raw["sets"]:
                    raise ValueError(f"edge-set {sname!r} `{role}: {s[role]}` is not a declared set")
            # connections: an inline `edges` list or an `edges_file` (.npz) for connectome size; exactly one
            has_inline, has_file = "edges" in s, bool(s.get("edges_file"))
            if has_inline and has_file:
                raise ValueError(
                    f"edge-set {sname!r} gives both `edges:` and `edges_file:` -- name one. "
                    f"The file is the connectome; an inline list beside it is a second answer "
                    f"to the same question.")
            if not has_inline and not has_file:
                raise ValueError(f"edge-set {sname!r} needs `edges: [[pre, post], ...]` or "
                                 f"`edges_file: <path.npz>` (edge_index [2, E] + weights [E])")
            if has_file:
                if "weights" in s:
                    raise ValueError(
                        f"edge-set {sname!r} sets `weights:` beside `edges_file:` -- the weights "
                        f"belong in the npz, next to the edges they weight.")
            else:
                if not isinstance(s["edges"], list) or not s["edges"]:
                    raise ValueError(f"edge-set {sname!r} needs a non-empty inline `edges: [[pre, post], ...]` list")
                for e in s["edges"]:
                    if not (isinstance(e, (list, tuple)) and len(e) in (2, 3)):
                        raise ValueError(f"edge-set {sname!r}: each edge must be [pre, post] or "
                                         f"[pre, post, weight], got {e!r}")

    # operator-line resolution shared by `operators:` and `seed:`: names registered, valid KIND, selectors and fields exist, capability contract met -> (cls, kind, OpSpec)
    def types_in_chain(set_name):
        # a required property may sit on the set's types or be inherited from a parent set (mpm acts on particles but reads `youngs` off the cell type)
        seen = set()
        while set_name and set_name not in seen:
            seen.add(set_name)
            ts = raw["sets"][set_name].get("types")
            if ts:
                return set_name, ts
            set_name = raw["sets"][set_name].get("parent")
        return None, {}

    def resolve_op_line(o):
        name = o["op"]
        # `model:` and `implementation:` are separate keys and swapping them is refused (Gray-Scott and Brusselator are not two ways to compute one reaction)
        impl = o.get("implementation")
        modl = o.get("model")
        if impl is not None and modl is not None:
            raise ValueError(
                f"operator {o['op']!r} names both `model: {modl}` and "
                f"`implementation: {impl}`; a variant is one or the other")
        # resolve the selected implementation; capability checks run against the one that will execute
        try:
            cls = registry.get_operator(name, impl, modl)
        except KeyError:
            renamed = registry.renamed_operator_message(name)
            if renamed:
                raise ValueError(renamed) from None
            registry.check_renamed_variant(name, modl or impl)
            try:
                contract = registry.get_contract(name)
            except KeyError:
                raise ValueError(
                    f"operator {name!r} not in registry. Available: "
                    f"{sorted(registry._OPERATOR_REGISTRY)}")
            raise ValueError(
                f"operator {name!r} has no variant {(modl or impl)!r}; "
                f"models: {contract.models() or '-'}  "
                f"implementations: {contract.impls() or '-'}")
        registry.check_renamed_params(name, modl or impl, o)
        kind = getattr(cls, "KIND", None)
        if kind not in KINDS:
            raise ValueError(
                f"operator {name!r} has unrecognised kind {kind!r}; expected one of {KINDS}.")
        supported = getattr(cls, "SUPPORTED_DIMS", [2])
        if dim not in supported:
            raise ValueError(
                f"operator {name!r} supports dims {supported}, not dim={dim} "
                f"(set general.dim, or use a dimension-generic / *_3d operator).")
        # integration-order vocabulary: the class default and the optional spec override must name a recognised state
        cls_emit = getattr(cls, "EMIT", None)
        if cls_emit is not None and cls_emit not in EMITS:
            raise ValueError(
                f"operator {name!r} has invalid EMIT {cls_emit!r}; "
                f"expected one of {EMITS} or None.")
        emit = o.get("emit")
        if emit is not None and emit not in EMITS:
            raise ValueError(
                f"operator {name!r} sets emit: {emit!r}, not one of {EMITS}.")
        sel = Selector.parse(o["at"])
        # `at:` names a SET (set / exchange operators) or a FIELD (field-internal operators like diffuse / decay)
        if sel.set not in raw["sets"] and sel.set not in raw["fields"]:
            raise ValueError(f"operator {name!r} acts on unknown set or field {sel.set!r}")
        if sel.block is not None:
            st = (raw["sets"].get(sel.set) or {}).get("state")
            if sel.set not in raw["sets"] or (isinstance(st, dict) and sel.block not in st):
                raise ValueError(f"operator {name!r}: `at: {o['at']}`: the set {sel.set!r} has no state block {sel.block!r}"
                                 + (f" (it has: {', '.join(st)})" if isinstance(st, dict) else ""))
        _check_refs(name, o, raw)
        _src = o.get("from")
        _mask = None
        if sel.set in raw["fields"] and sel.set not in raw["sets"] and isinstance(_src, str):
            _m = Selector.parse(_src)                   # a field written from a set: the rows come from the set read
            _mask = _m if _m.set in raw["sets"] else None
        params = {k: v for k, v in o.items() if k not in _RESERVED}
        if getattr(cls, "USES_VIA_FILES", False) and isinstance(o.get("via"), dict):
            # an operator that builds its own graph from declared edge sets (e.g. a short / mid / long range graph) is handed each one's file, by role
            files = {r: raw["sets"][e]["edges_file"] for r, e in o["via"].items() if raw["sets"].get(e, {}).get("edges_file")}
            if files:
                params["_via_files"] = files

        # capability contract: what the operator declares it requires
        have = set(o.keys())
        for req in getattr(cls, "REQUIRES_PARAMS", []):
            if req not in have:
                raise ValueError(
                    f"operator {name!r} requires param {req!r} (declared in "
                    f"{cls.__name__}.REQUIRES_PARAMS); add it to the operator line.")
        for prop in getattr(cls, "REQUIRES_TYPE_PROPS", []):
            owner, set_types = types_in_chain((_mask or sel).set)
            if not set_types:
                raise ValueError(
                    f"operator {name!r} requires per-type property {prop!r}, but neither "
                    f"{sel.set!r} nor its parents declare `types`.")
            # one property, several spellings: `youngs` / `bulk_modulus` reach the same Lame pair, so a type satisfies any alternative the operator lists
            alts = getattr(cls, "TYPE_PROP_ALTERNATIVES", {}).get(prop, ())
            for tname, t in set_types.items():
                if prop not in t and not any(a in t for a in alts):
                    also = f" (or {' / '.join(alts)})" if alts else ""
                    raise ValueError(
                        f"operator {name!r} requires property {prop!r}{also} on every type of "
                        f"{owner!r}; missing on type {tname!r}. "
                        f"(declared in {cls.__name__}.REQUIRES_TYPE_PROPS)")
        opspec = OpSpec(op=name, on=sel, frm=o.get("from"), via=o.get("via"),
                        impl=(modl or impl), params=params, mask=_mask)
        return cls, kind, opspec

    # operators: names registered, valid KIND, selectors and fields exist
    ops = []
    _legacy_seed_ops = set()          # kind="seed" operators declared the old way -- see below
    for o in raw["operators"]:
        cls, kind, opspec = resolve_op_line(o)
        ops.append(opspec)
        if kind == "seed":
            _legacy_seed_ops.add(opspec.op)

    # `seed:` section: x_0, once, before any dynamics. Every line must be a `kind="seed"` operator, may not carry `every` / `before_frame` / `after_frame`, and may not also appear in `schedule:`
    _SEED_ONLY_FORBIDDEN = ("every", "before_frame", "after_frame")
    seed_ops = []
    for o in raw.get("seed", []):
        cls, kind, opspec = resolve_op_line(o)
        if kind != "seed":
            raise ValueError(
                f"seed: {o['op']!r} has kind {kind!r}, not \"seed\" -- only operators that "
                f"establish x_0 may be declared in the seed: section. A dynamics operator "
                f"belongs in operators:/schedule:.")
        bad = [k for k in _SEED_ONLY_FORBIDDEN if k in o]
        if bad:
            raise ValueError(
                f"seed: {o['op']!r} sets {bad}, a dynamic scheduling control -- seed has no "
                f"schedule, it runs exactly once. Remove {bad} from this seed: line.")
        seed_ops.append(opspec)

    # a seed-kind operator under `operators:` is the deprecated spelling (notes/campaigns/SEED_MIGRATION.md): accepted with a warning
    if _legacy_seed_ops:
        warn(f"[warn] deprecated: {sorted(_legacy_seed_ops)} declared in operators: with "
              f"kind=\"seed\" -- move to the seed: section (a `before_frame` wider than 1 cannot "
              f"move: `seed:` has no window). Still "
              f"accepted for now via the legacy engine seed-window path.")

    # warn about per-type properties no operator reads (typo guard)
    used_props = set()
    for o in raw["operators"]:
        cls = registry.get_operator(o["op"])
        used_props |= set(getattr(cls, "REQUIRES_TYPE_PROPS", []))
        used_props |= set(getattr(cls, "OPTIONAL_TYPE_PROPS", []))   # read only in some modes (e.g. alignment per_type)
    # read outside operators: core / layers / block, `material` / `density` / `tau` (entity provision hook, per particle type), `bulk_modulus` (TYPE_PROP_ALTERNATIVES), `count` (schema and `_assign_types`)
    _KNOWN_TYPE_KEYS = {"fraction", "count", "core", "layers", "block",
                        "material", "density", "tau", "bulk_modulus", "shape",
                        "eta",                        # per-type dynamic viscosity (mpm_viscosity)
                        "fill",                       # a block's point layout (entities: random | lattice)
                        "sign", "role", "shape", "form", "aspect", "axis", "hollow", "rotate", "repeat", "pitch", "scatter", "ring", "shell", "orient", "fill", "obj", "place"} | used_props  # a neuron type's Dale sign and circuit role (neural_panel)
    for sname, s in raw["sets"].items():
        for tname, t in s.get("types", {}).items():
            for k in t:
                if k not in _KNOWN_TYPE_KEYS:
                    warn(f"[warn] property {k!r} on {sname}.{tname} is read by no operator "
                          f"(known: {sorted(_KNOWN_TYPE_KEYS)})")

    # schedule: every token resolves to an operator; a step is a token, a list of tokens, or a substep micro-loop `{substep_dt: dt, steps: [...]}` run round(general.dt / dt) times (e.g. the MPM strain -> P2G -> grid -> G2P cycle)
    op_names = {o.op for o in ops}
    seed_op_names = {o.op for o in seed_ops}       # new-style seed:, excluded from schedule
    for step in raw["schedule"]:
        if isinstance(step, dict) and "substep" in step:
            raise ValueError("the `{substep: N, dt: X}` schedule form was removed; write "
                             "`{substep_dt: X, steps: [...]}` and set `general.dt` to the "
                             "per-frame sim time (substeps = round(general.dt / X)).")
        if isinstance(step, dict) and "substep_dt" in step:
            if not isinstance(step.get("steps"), list):
                raise ValueError("a `{substep_dt: …}` schedule step needs a `steps:` list")
            # the key set is closed, so `complie: true` is an error and not a silently ignored performance flag
            _ok = {"substep_dt", "steps", "capture", "compile", "compile_mode",
                   "compile_recompile_limit", "compile_verbose"}
            _bad = set(step) - _ok
            if _bad:
                raise ValueError(f"unknown key(s) {sorted(_bad)} on a `{{substep_dt: …}}` step; "
                                 f"allowed: {sorted(_ok)}")
            tokens = step["steps"]
        else:
            tokens = step if isinstance(step, list) else [step]
        for tok in tokens:
            if tok in _BUILTIN_STEPS:
                continue
            # a `seed:` operator is run once by engine.seed(), never by the schedule loop; scheduling it too would re-run it on the schedule's terms
            if tok in seed_op_names:
                raise ValueError(
                    f"schedule step {tok!r} is a seed: operator (kind=\"seed\") -- seed runs "
                    f"exactly once, before the schedule, via engine.seed(). Remove it from "
                    f"schedule: (and from operators:, if declared there too).")
            if tok not in op_names:
                raise ValueError(f"schedule step {tok!r} is not a declared operator or builtin")

    # a scatter without a grid solve is a dead material (momentum never divided by mass, bodies stand still): refuse a schedule that names `mpm_scatter` and never `mpm_grid_update`
    _named = []
    for step in raw["schedule"]:
        _named += list(step.get("steps", [])) if isinstance(step, dict) else \
            (list(step) if isinstance(step, list) else [step])
    if any(t in ("mpm_scatter", "p2g") for t in _named) and "mpm_grid_update" not in _named:
        raise ValueError(
            "schedule names mpm_scatter (particle -> grid) but never mpm_grid_update (the grid "
            "solve): momentum would be deposited and never divided by mass, every gather would read "
            "zero, and every MPM body would stand still under any force. Schedule the cycle as "
            "mpm_strain, mpm_scatter (one per set), mpm_grid_update (once), mpm_gather (one per set).")

    _spec = Spec(
        name=gv("name"),
        title=str(gv("title", "") or ""),
        seed=int(gv("seed", 0)),
        n_frames=int(gv("n_frames", 200)),
        dt=float(gv("dt", 0.05)),
        sets=raw["sets"],
        fields=raw["fields"],
        operators=ops,
        schedule=raw["schedule"],
        seed_ops=seed_ops,
        obstacles=gv("obstacles", []),
        boundary=gv("boundary", "wall"),
        world=world_size[0],
        dim=dim,
        world_size=world_size,
        plotting=raw.get("plotting", {}),
        learnable=_parse_learnable(raw.get("learnable"), raw),
        engine=str(gv("engine", "default")).lower(),
        record_cap=int(gv("record_cap", 10000)),
        field_record_cap=int(gv("field_record_cap", 256)),
        units=parse_units(gv("units", None)),
        save_data=gv("save_data", None),
        warmup=(None if gv("warmup", None) is None else int(gv("warmup", None))),
        save_every_min=float(gv("save_every_min", 0.0)),
        save_chunk_frames=int(gv("save_chunk_frames", 0)),
        save_chunk_window=gv("save_chunk_window", None),
        dt_schedule=_parse_dt_schedule(gv("dt_schedule", None), int(gv("n_frames", 200))),
    )
    if _spec.dt_schedule:
        _spec.dt = float(_spec.dt_schedule[0][1])
    # units check: one warning-only pass, wrapped so a defect in the checker cannot stop the load (plexus/units_check.py)
    try:
        from plexus.units_check import check as _units_check
        # the variant map, not `get_operator`: `OpSpec.impl` is `(model or implementation)` and `contract.implementations` is keyed by variant name across both axes
        _classes = {}
        for _o in list(ops) + list(seed_ops):
            try:
                _c = registry.get_contract(_o.op)
                _classes[_o.op] = (_c.implementations.get(_o.impl or _c.default),
                                   dict(_o.params or {}))
            except Exception:                    # noqa: BLE001
                pass
        _units_check(_spec, _classes)
    except Exception as _e:                      # noqa: BLE001
        print(f"[units] check skipped ({type(_e).__name__}: {_e})", flush=True)
    return _spec
