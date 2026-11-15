"""Registries for Plexus: decorator-based, one per kind of thing.

  ENTITY    node kinds (a Level's state schema): "cell", "particle", "molecule"
  OPERATOR  the dynamics, tagged with the set it acts on and its kind (`base.KINDS`), so a schedule or
            the LLM agentic loop can enumerate "everything that can run on set S"
  FIELD     continuum discretizations: "grid", "mesh", "implicit"

Usage:
    @register_operator("signal", set="neuron", kind="lateral")
    class SignalOperator(Operator): ...

    op_cls = get_operator("signal")
    at_neuron = operators_at_set("neuron")       # -> {name: cls, ...}

Also here: operator contracts and variants, renamed-operator errors, seed discovery, the operator
families, and the functions that bind operators to a run (default implementation, integration order,
seed window)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:                  # annotations only: schema.py imports this module
    from plexus.schema import Spec


# _warn: Print a warning through `plexus.paths.warn`, imported lazily so the registry has no import-time dependency.
def _warn(msg: str) -> None:
    """`plexus.paths.warn`, imported lazily so the registry keeps no import-time dependency."""
    from plexus.paths import warn
    warn(msg)

_ENTITY_REGISTRY: dict[str, type] = {}
_OPERATOR_REGISTRY: dict[str, type] = {}    # name -> DEFAULT implementation class (enumeration / back-compat)
_FIELD_REGISTRY: dict[str, type] = {}


# _make_decorator: Build the registering decorator of one registry; `tag_defaults` are the tags every entry carries -> a `register(*names, **tags)` decorator factory.
def _make_decorator(registry: dict, label: str, **tag_defaults):
    def register(*names: str, **tags):
        def decorator(cls):
            meta = {**tag_defaults, **tags}
            for k, v in meta.items():
                setattr(cls, k.upper(), v)
            cls.REGISTERED_NAMES = list(names)
            for name in names:
                if name in registry:
                    raise ValueError(
                        f"{label} name '{name}' already registered to "
                        f"{registry[name].__name__}"
                    )
                registry[name] = cls
            return cls
        return decorator
    return register


# register_entity, register_field: Registering decorators; an entity carries `depth` (0 = leaf), a field `couples_to` and `frame`.
register_entity = _make_decorator(_ENTITY_REGISTRY, "Entity", depth=None)
register_field = _make_decorator(_FIELD_REGISTRY, "Field", couples_to=None, frame=None)


# OperatorContract: A biological operator: a fixed typed signature with one or more interchangeable numerical implementations.
@dataclass
class OperatorContract:
    """A biological operator: a FIXED typed signature (its biology) with one or more interchangeable numerical IMPLEMENTATIONS (their numerics).

    The contract owns name / kind / family / signature; each implementation owns only how the delta is computed and its capabilities (supported dims, differentiability). Choosing an implementation is a numerical choice, never a biological one (`{op: neuron_voltage, implementation: hodgkin_huxley}` vs `implementation: gnn` is the same contract). The default is the sole / first-registered implementation, so single-implementation specs are unaffected."""
    name: str
    kind: str | None = None
    family: str | None = None
    set: str | None = None
    signature: dict = field(default_factory=dict)         # the DEFAULT variant's, kept for callers
    # one signature per variant: variants of one contract do not read the same things (`cell_chem_diffuse[interface_weighted]` also reads the vertex set and `pos`), but `signature` came from the first registration only; it stays, for the audit tools and the atlas
    signatures: dict = field(default_factory=dict)        # variant name -> that variant's signature
    implementations: dict = field(default_factory=dict)   # variant name -> class (models + impls)
    default: str | None = None
    # which axis each variant varies: `model` = another biological hypothesis, `implementation` = the same biology computed differently (conflating them made findings recorded against `cell_chem_react` really about Gray-Scott)
    axis: dict = field(default_factory=dict)              # variant name -> "model" | "implementation"

    def get(self, implementation: str | None = None, model: str | None = None,
            variant: str | None = None) -> type:
        """The class for a named variant. `model=` / `implementation=` are the spec's keys and are checked against the axis the variant was registered on (naming one where the other is meant is refused); `variant=` is the axis-agnostic lookup the engine uses after the schema has made that check."""
        if model is not None and implementation is not None:
            raise KeyError(f"operator {self.name!r}: name a model or an implementation, not both")
        if variant is not None:
            if variant not in self.implementations:
                raise KeyError(f"operator {self.name!r} has no variant {variant!r}; "
                               f"models: {self.models() or '-'}  "
                               f"implementations: {self.impls() or '-'}")
            return self.implementations[variant]
        want = model if model is not None else implementation
        variant = want or self.default
        if variant not in self.implementations:
            kinds = {"model": sorted(v for v, a in self.axis.items() if a == "model"),
                     "implementation": sorted(v for v, a in self.axis.items()
                                              if a == "implementation")}
            raise KeyError(
                f"operator {self.name!r} has no variant {variant!r}; "
                f"models: {kinds['model'] or '-'}  implementations: {kinds['implementation'] or '-'}")
        got = self.axis.get(variant, "implementation")
        if want is not None:
            asked = "model" if model is not None else "implementation"
            if got != asked:
                raise KeyError(
                    f"operator {self.name!r}: {variant!r} is a {got}, not a{'n' if asked[0] == 'i' else ''}"
                    f" {asked}. Write `{got}: {variant}`. "
                    + ("A model is a different biological hypothesis at this slot, so swapping it "
                       "is an experiment; an implementation is the same biology computed "
                       "differently." if got == "model" else
                       "An implementation is the same biology computed differently; a result that "
                       "changes under the swap is about discretisation, not about the tissue."))
        return self.implementations[variant]

    def models(self) -> list:
        return sorted(v for v, a in self.axis.items() if a == "model")

    def impls(self) -> list:
        return sorted(v for v, a in self.axis.items() if a == "implementation")

    def capabilities(self) -> dict:
        """Per-implementation capability table: supported dims + differentiability."""
        return {i: {"dims": list(getattr(c, "SUPPORTED_DIMS", [])),
                    "differentiable": bool(getattr(c, "DIFFERENTIABLE", True))}
                for i, c in self.implementations.items()}


_OP_CONTRACTS: dict[str, OperatorContract] = {}   # name -> contract (signature + all implementations)


# register_operator: Register an operator implementation; the first registration of a name creates its contract -> the class decorator.
def register_operator(*names: str, implementation: str | None = None,
                      model: str | None = None, **tags):
    """Register an operator implementation. The FIRST registration of a name creates its
    contract (from the class's typed `signature()`); a later registration of the SAME name
    with a different `implementation=` adds an interchangeable implementation to that same
    contract. `set=`/`kind=`/`family=` and any other tag are stamped as UPPER-case class
    attributes (SET / KIND / FAMILY ...). One decorator may list alias names, which each
    resolve to the same class."""
    tags.setdefault("set", None)
    tags.setdefault("kind", None)
    if implementation is not None and model is not None:
        raise ValueError(f"operator {names[0]!r}: a variant is a model OR an implementation")
    def decorator(cls):
        for k, v in tags.items():
            setattr(cls, k.upper(), v)
        cls.REGISTERED_NAMES = list(names)
        # `family="seed"` claims that the operator establishes x_0, so declaring it with another kind (structural, exchange, lateral) is a masquerade that would skip the seed lifecycle; the reverse (`kind="seed"` with a domain family such as "anatomy") is the intended orthogonal use; warn, not raise, so prototypes outside the tree still import
        if tags.get("family") == "seed" and tags.get("kind") != "seed":
            _warn(
                f"[warn] operator {names[0]!r}: family=\"seed\" claims this establishes "
                f"the initial state, but kind={tags.get('kind')!r} -- a seed masquerading "
                f"as a dynamics kind. Should be registered with kind=\"seed\".")
        impl = model or implementation or "default"
        axis = "model" if model is not None else "implementation"
        cls.IMPLEMENTATION = impl
        cls.VARIANT_AXIS = axis
        for name in names:
            contract = _OP_CONTRACTS.get(name)
            if contract is None:
                contract = OperatorContract(
                    name=name, kind=getattr(cls, "KIND", None),
                    family=getattr(cls, "FAMILY", None), set=getattr(cls, "SET", None),
                    signature=cls.signature() if hasattr(cls, "signature") else {},
                    default=impl)
                _OP_CONTRACTS[name] = contract
                _OPERATOR_REGISTRY[name] = cls          # default impl: enumeration + back-compat
            else:
                if impl in contract.implementations:
                    raise ValueError(
                        f"operator {name!r} already has variant {impl!r} "
                        f"({contract.implementations[impl].__name__})")
                # variants of one contract must share its KIND (a weak guard: the `model` / `implementation` axis carries the real distinction)
                if getattr(cls, "KIND", None) != contract.kind:
                    raise ValueError(
                        f"operator {name!r} variant {impl!r} has kind "
                        f"{getattr(cls, 'KIND', None)!r}, but the contract's kind is "
                        f"{contract.kind!r}; a variant may not change the kind.")
            contract.implementations[impl] = cls
            contract.axis[impl] = axis
            # every variant including the first, so `signatures` is complete and `signatures[contract.default]` is the object `signature` holds
            contract.signatures[impl] = cls.signature() if hasattr(cls, "signature") else {}
        return cls
    return decorator


# get_entity: The registered entity class for a name -> class (KeyError if unknown).
def get_entity(name: str) -> type:
    return _ENTITY_REGISTRY[name]


# _entity_class: The registered entity class of a set -> class | None.
def _entity_class(sname: str, entity: str | None = None):
    """The registered entity class of a set, or None. It may define `provision(lvl, parent, s, H,
    device)` to allocate domain-specific per-node buffers at build (e.g. mpm_particle's F/C/mass)."""
    try:
        return get_entity(entity or sname)
    except KeyError:
        return None


# _RENAMED: Old operator name -> (new `op:` text, since when). A renamed operator is an ERROR naming the new spelling, not an alias (docs/CURATION_PLAN.md decision 4); record folders' specs are rewritten by tools/rename_ops.py at merge time only.
_RENAMED: dict[str, tuple[str, str]] = {}            # old name -> (new `op:` text, when)


# renamed_operator: Declare that an operator is now spelled differently; the old name becomes an error.
def renamed_operator(old: str, new: str, since: str) -> None:
    """Declare that operator `old` is now spelled `new` (`new` may carry a model: "junction[contract]" reads `op: junction, model: contract`)."""
    if old in _OP_CONTRACTS:
        raise ValueError(f"operator {old!r} is still registered; a rename removes the old registration")
    _RENAMED[old] = (new, since)


# renamed_operator_message: The error text for a renamed operator name -> str | None.
def renamed_operator_message(name: str) -> str | None:
    """The error text for an operator name that a rename replaced, or None if `name` was never renamed."""
    if name in _RENAMED:
        new, since = _RENAMED[name]
        return f"operator {name!r} was renamed {new!r} ({since}); no alias is kept -- rewrite the spec"
    new = _plain_names()["operators"].get(name)
    if new is not None:
        return f"operator {name!r} was renamed {new!r} ({_PLAIN_NAMES_SINCE}); no alias is kept -- rewrite the spec"
    return None


# _check_renamed: Raise KeyError, naming the new spelling, for an operator name a rename replaced.
def _check_renamed(name: str) -> None:
    msg = renamed_operator_message(name)
    if msg:
        raise KeyError(msg)


# _PLAIN_NAMES_SINCE: Date of the plain-names rename (docs/NAME_MAP.yaml, the list `tools/rename_names.py` also reads); the map is read only when a spec names an old name, to turn "unknown variant" or a silently ignored key into an error naming the new one.
_PLAIN_NAMES_SINCE = "2026-10-31 and 2026-11-09"
_PLAIN: dict | None = None


# _plain_names: The plain-names rename map -> {operators, variants, params}.
def _plain_names() -> dict:
    """{operators: {old: new}, variants: {op: {new: old}}, params: {(op, variant): {old_key: new_key}}}; the variant is the NEW name, or None for every variant of the operator."""
    global _PLAIN
    if _PLAIN is None:
        import os
        import yaml
        _PLAIN = {"operators": {}, "variants": {}, "renamed_variants": {}, "params": {}}
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "docs", "NAME_MAP.yaml")
        try:
            m = yaml.safe_load(open(path))
        except (OSError, yaml.YAMLError):
            return _PLAIN
        _PLAIN["operators"] = dict(m["operators"])
        for op, d in m["variants"].items():
            _PLAIN["renamed_variants"][op] = dict(d["rename"])                                   # op -> {old variant: new variant}
            _PLAIN["variants"][op] = {v: k for k, v in d["rename"].items()}                      # op -> {new variant: old variant}
        for p in m["parameters"]:
            new_of = (m["variants"].get(p["op"]) or {}).get("rename", {})
            for v in p["variants"]:
                key = (m["operators"].get(p["op"], p["op"]), None if v == "*" else new_of.get(v, v))
                _PLAIN["params"].setdefault(key, {}).update(p["rename"])
    return _PLAIN


# check_renamed_variant: Raise ValueError for a variant the plain-names rename replaced.
def check_renamed_variant(op: str, variant: str) -> None:
    """Raise ValueError for a variant of `op` that the plain-names rename of 2026-10-31 replaced (`op` is the name the spec used; an old operator name is refused first)."""
    new = _plain_names()["renamed_variants"].get(op, {}).get(variant)
    if new is not None:
        raise ValueError(f"operator {op!r}: variant {variant!r} was renamed {new!r} ({_PLAIN_NAMES_SINCE}); no alias is kept -- rewrite the spec")


# check_renamed_params: Raise ValueError for a parameter key the plain-names rename replaced.
def check_renamed_params(op: str, variant: str | None, keys) -> None:
    """Raise ValueError if a spec line of operator `op` (variant `variant`, the NEW name or None) still carries a parameter key the 2026-10-31 rename replaced.
    An optional key left under its old name would otherwise be ignored and the default used, a different run with no error."""
    nm = _plain_names()["params"]
    rules = dict(nm.get((op, None), {}))
    rules.update(nm.get((op, variant), {}))
    stale = {k: rules[k] for k in keys if k in rules}
    if stale:
        raise ValueError(f"operator {op!r}: " + ", ".join(f"`{o}` was renamed `{n}`" for o, n in stale.items())
                         + f" ({_PLAIN_NAMES_SINCE}); no alias is kept -- rewrite the spec")


# get_operator: The class of an operator (default variant, or a named `model` / `implementation`) -> class.
def get_operator(name: str, implementation: str | None = None,
                 model: str | None = None, variant: str | None = None) -> type:
    """The class for operator `name` -- the default variant, or a named `model` /
    `implementation`. Raises KeyError if the operator or the variant is unknown, or if the
    variant exists on the other axis (naming a model an implementation is refused)."""
    if name not in _OP_CONTRACTS:
        _check_renamed(name)
    return _OP_CONTRACTS[name].get(implementation, model, variant)


# get_contract: The full contract of an operator (signature + every implementation) -> OperatorContract.
def get_contract(name: str) -> OperatorContract:
    """The full operator contract (signature + every registered implementation)."""
    if name not in _OP_CONTRACTS:
        _check_renamed(name)
    return _OP_CONTRACTS[name]


# get_field: The registered field class for a name -> class.
def get_field(name: str) -> type:
    return _FIELD_REGISTRY[name]


# operators_at_set: All operators registered to act on a set -> {name: class}.
def operators_at_set(set_name: str) -> dict[str, type]:
    """All operators registered to act on `set_name` (the LLM's action set for that
    biological set). `set_name` is a set like "cell"/"neuron", not a depth."""
    return {n: c for n, c in _OPERATOR_REGISTRY.items() if getattr(c, "SET", None) == set_name}


# operators_at_level: Deprecated alias of `operators_at_set` -> {name: class}.
def operators_at_level(level: str) -> dict[str, type]:
    """Deprecated alias for `operators_at_set` (the arg names a set, never a depth)."""
    return operators_at_set(level)


# operators_of_kind: All operators of one kind -> {name: class}.
def operators_of_kind(kind: str) -> dict[str, type]:
    return {n: c for n, c in _OPERATOR_REGISTRY.items() if getattr(c, "KIND", None) == kind}


# is_seed: Whether the operator registered under a name establishes x_0 (kind == "seed") -> bool.
def is_seed(name: str) -> bool:
    """Whether the operator registered as `name` establishes x_0 (`kind == "seed"`)."""
    return getattr(_OPERATOR_REGISTRY.get(name), "KIND", None) == "seed"


# seed_operators: All operators that establish x_0 -> {name: class} (`operators_of_kind("seed")` under a name that says what the query is for).
def seed_operators() -> dict[str, type]:
    """All operators that establish x_0 -- `operators_of_kind("seed")` under a name
    that says what the query is for."""
    return operators_of_kind("seed")


# seed_contracts: The full contracts of every seed operator -> {name: OperatorContract}.
def seed_contracts() -> dict[str, OperatorContract]:
    """The full contracts (signature + every implementation) of every seed operator,
    for the discovery loop to query "what can establish x_0?" without walking the
    default-implementation registry by hand."""
    return {n: c for n, c in _OP_CONTRACTS.items() if c.kind == "seed"}


# OPERATOR_FAMILIES: The closed set of operator families, a taxonomy of what an operator is FOR (`kind` says how it moves data, and 62 of 157 are `lateral`); built from a census of what is registered; `tools/audit_operator_registry.py` fails on a missing or unknown family.
OPERATOR_FAMILIES = {
    "seed",          # establish initial state on a set
    "population",    # the set gains or loses members: divide, die, grow, enter, be eaten
    "mechanics",     # internal force, stress, elasticity and relaxation of a body
    "interaction",   # neighbour and pairwise forces between members of a set
    "motion",        # self-propulsion, kinematics, and the decision rules that drive them
    "polarity",      # heading and orientation
    "fields",        # scalar/vector field evolution -- diffusion, decay, advection, sources
    "signalling",    # secretion and sensing between an entity and its surroundings
    "topology",      # rewiring the graph or the mesh
    "hierarchy",     # parent<->child plumbing and cross-scale readouts
    "coupling",      # transfer between two substrates
    "boundary",      # confinement, contact, absorption, slip at a surface
    "mpm",           # the MLS-MPM substep machinery
    "harness",       # bookkeeping and scaffolding that is not biology
}


# operators_by_family: All canonical operators of a family (aliases excluded) -> {name: class}.
def operators_by_family(family: str) -> dict[str, type]:
    """All CANONICAL operators in `family` (aliases excluded)."""
    return {n: c for n, c in _OPERATOR_REGISTRY.items()
            if getattr(c, "FAMILY", None) == family
            and getattr(c, "REGISTERED_NAMES", [n])[0] == n}


# catalog_summary: Human-readable table of everything registered, for the docs and the CLI -> str.
def catalog_summary() -> str:
    """Human-readable table of everything registered (docs / CLI). Each operator also shows its MECHANISM_TAGS (a capability index) and PARAM_ROLES (what each tunable means): this is the one consumer that makes that metadata live, feeding the operator catalog the docs and the mechanism-search layer read."""
    def tag(c, name):
        return str(getattr(c, name, None))
    lines = ["# entities"]
    for n, c in sorted(_ENTITY_REGISTRY.items()):
        lines.append(f"  {n:18s} depth={tag(c, 'DEPTH')}")
    lines.append("# operators (by family; canonical names, aliases in parens)")
    canon = [(n, c) for n, c in _OPERATOR_REGISTRY.items()
             if getattr(c, "REGISTERED_NAMES", [n])[0] == n]        # skip aliases
    for fam in sorted(OPERATOR_FAMILIES):
        ops = sorted((n, c) for n, c in canon if getattr(c, "FAMILY", None) == fam)
        if not ops:
            continue
        lines.append(f"## {fam}")
        for n, c in ops:
            al = getattr(c, "REGISTERED_NAMES", [n])[1:]
            alias = f"  (alias {', '.join(al)})" if al else ""
            lines.append(f"  {n:20s} set={tag(c, 'SET'):10s} kind={tag(c, 'KIND'):10s} emit={tag(c, 'EMIT')}{alias}")
            tags = getattr(c, "MECHANISM_TAGS", None)
            if tags:
                lines.append(f"      tags:  {', '.join(tags)}")
            roles = getattr(c, "PARAM_ROLES", None)
            if roles:
                for p, role in roles.items():
                    lines.append(f"      · {p:11s} {role}")
    orphan = sorted(n for n, c in canon if getattr(c, "FAMILY", None) not in OPERATOR_FAMILIES)
    if orphan:
        lines.append(f"## (NO/UNKNOWN FAMILY -- audit will fail): {', '.join(orphan)}")
    lines.append("# fields")
    for n, c in sorted(_FIELD_REGISTRY.items()):
        lines.append(f"  {n:18s} couples_to={tag(c, 'COUPLES_TO'):10s} frame={tag(c, 'FRAME')}")
    return "\n".join(lines)




# MPM_WARP_DEFAULT: MPM operators that default to implementation: warp.
MPM_WARP_DEFAULT = ("mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather")


# _resolve_default_impl: Write implementation: warp onto unspecified MPM operators and `squared_law all_pairs` when warp is safe -> names of the operators changed.
def _resolve_default_impl(sim, device: str, grad: bool = False) -> list[str]:
    """Write `implementation: warp` onto the operators a spec left unspecified; returns their names.

    Measured on si_waterfall (570,760 particles, 96^3): 973.8 ms/frame on torch, 31.8 ms with the MPM
    cycle on warp. Falls back to the torch body (so shipped specs keep running) when warp cannot:
      dim != 3, cpu device, periodic boundary (warp gather clamps), grad=True (no backward),
      legacy CSF (`cell_surface_tension` without `csf_rho`), `csf_smooth`, `plate_axis`,
      `mpm_scatter[store_stress]`.
    `implementation: default` forces the torch body. Warp is not bit-identical to torch (6 ulp after
    14 substeps), so the pick is written back onto the spec and printed: it is part of a run's identity."""
    picked = []
    # `implementation: default` is an escape hatch, not a variant: normalise it to None (the torch
    # body) and remember it, or the warp pass below would hand it straight back to warp
    _forced = set()
    for o in sim.operators:
        if str(o.impl or "").lower() == "default":
            o.impl = None
            _forced.add(id(o))
    _cuda = str(device).startswith("cuda")
    _bad = grad or str(getattr(sim, "boundary", "wall")).lower() == "periodic"
    # `squared_law all_pairs` first, before the 3D gate: its warp variant is 2D and 3D, and the torch
    # body materialises [N, N] separations (9.35 GB, 147 ms/step at 25,000 particles vs 0.04 GB, 4.9 ms)
    if _cuda and not _bad:
        for o in sim.operators:
            if (o.op == "squared_law" and o.impl is None and id(o) not in _forced
                    and bool(o.params.get("all_pairs", False))):
                o.impl = "warp"
                picked.append(o.op)
    if int(getattr(sim, "dim", 2)) != 3 or not _cuda:
        return picked
    if _bad:
        return picked
    for o in sim.operators:
        if o.op not in MPM_WARP_DEFAULT or o.impl is not None or id(o) in _forced:
            continue
        # `store_stress` needs the per-particle Cauchy stress, which the warp scatter never forms
        if o.op == "mpm_scatter" and bool(o.params.get("store_stress", False)):
            continue
        if o.op == "mpm_grid_update":
            q = o.params
            _st = float(q.get("cell_surface_tension", 0.0) or 0.0)
            if (_st > 0.0 and float(q.get("csf_rho", 0.0) or 0.0) <= 0.0
                    or int(q.get("csf_smooth", 0) or 0)
                    or q.get("plate_axis") is not None):
                continue
        o.impl = "warp"
        picked.append(o.op)
    return picked


# _resolve_emit: Integration order of each set's coordinate block -> {set: velocity | acceleration}.
def _resolve_emit(sim: Spec, H: "Hierarchy" = None) -> dict:
    """set -> integration order (`velocity` | `acceleration`) of its coordinate block.

    Read from each force-emitting operator's `emit:` (spec) else its class `EMIT`; operators of one
    set must agree. Skipped: `None` emit, `mpm_acceleration` (routed to the MPM substep), and an
    operator whose `INTEGRAND` is a non-coordinate block (it integrates its own first-order block)."""
    modes: dict[str, str] = {}
    for o in sim.operators:
        cls = get_operator(o.op, variant=o.impl)
        # `extra_emit_sets(params) -> {set: order}`: a second set the operator integrates (in no `at:`)
        extra = getattr(cls, "extra_emit_sets", None)
        for s2, m2 in ((extra(o.params) or {}).items() if callable(extra) else ()):
            if s2 in modes and modes[s2] != m2:
                raise ValueError(f"set {s2!r} has operators with conflicting integration order ({modes[s2]} vs {m2} from {o.op!r})")
            modes[s2] = m2
        emit = o.params.get("emit") or getattr(cls, "EMIT", None)
        if emit not in ("velocity", "acceleration"):
            continue                                   # None / mpm_acceleration: no engine-integrated set delta
        s = o.on.set
        integrand = getattr(cls, "INTEGRAND", None)    # which block this op integrates (None = coordinate)
        if integrand is not None and H is not None and s in H.levels:
            coord = H.levels[s].state_schema.coordinate
            if coord is not None and integrand != coord.name:
                continue                               # a non-coordinate (first-order) integrand: doesn't set the order
        if s in modes and modes[s] != emit:
            raise ValueError(
                f"set {s!r} has operators with conflicting integration order "
                f"({modes[s]} vs {emit} from {o.op!r}); a set integrates as one order.")
        modes[s] = emit
    return modes


# SEED_MAX: Most opening frames a seed may run; it can never span the run.
SEED_MAX = 10


# _seed_window: Opening frames a legacy seed-kind operator may run -> int.
def _seed_window(sim):
    """Opening frames a legacy `kind="seed"` operator (under `operators:` / `schedule:`, deprecated) may run.

    The largest `before_frame` any such seed declares (1 if none), capped at SEED_MAX: a model may need a
    few frames to settle, but an initial condition cannot run for the whole trajectory. Operators of the
    `seed:` section go through `seed()` and never reach here."""
    declared = [int(o.params["before_frame"]) for o in sim.operators
                if getattr(get_operator(o.op, variant=o.impl), "KIND", None) == "seed"
                and "before_frame" in o.params]
    return min(max(declared) if declared else 1, SEED_MAX)


# _resolve_gates: Each operator's frame window (after, before, every), in spec order -> list of (int, int, int).
def _resolve_gates(sim) -> list[tuple[int, int, int]]:
    """An operator runs at frame f when after <= f < before and, if every > 1, f % every == 0 (multi-rate).

    Defaults: after 0, before 2^30, every 1. A legacy `kind="seed"` operator gets (0, min(before, seed window), 1):
    the engine, not the spec, confines seeds to the start."""
    seed_window = _seed_window(sim)
    gates = []
    for o in sim.operators:
        after = int(o.params.get("after_frame", 0))
        before = int(o.params.get("before_frame", 1 << 30))
        every = max(1, int(o.params.get("every", 1)))
        if getattr(get_operator(o.op, variant=o.impl), "KIND", None) == "seed":
            gates.append((0, min(before, seed_window), 1))
        else:
            gates.append((after, before, every))
    return gates


# _live_ids: The operator instances whose frame gate is open at recorded frame f, in declaration order -> tuple of instance indices.
def _live_ids(gates, f: int) -> tuple:
    """`gates` is the (after, before, every) window of each instance, as `_resolve_gates` returns it; `f` the recorded-frame clock
    (tick minus the warm-up length). A warm-up tick (f < 0) reads as frame 0 for the window, but keeps f for the stride."""
    fw = max(f, 0)
    return tuple(j for j, (after, before, every) in enumerate(gates)
                 if after <= fw < before and (every <= 1 or f % every == 0))


# _live_instances: The live instance indices at recorded frame f, grouped by operator name -> {name: [index, ...]}.
def _live_instances(names, gates, f: int) -> dict:
    """The i-th mention of a name in the schedule runs the i-th index of its list, so the order is declaration order."""
    by = {}
    for j in _live_ids(gates, f):
        by.setdefault(names[j], []).append(j)
    return by
