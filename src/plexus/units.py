"""units: the physical scale of a model, declared once and never inferred.

A spec used to be entirely dimensionless, fine for a mechanism and fatal for a claim (a basement membrane `2e-3` box units thick was ~24x too thick against the ~0.1 um it is; a modulus `E = 400` was quoted against Candiello's 0.4-3 MPa; a turnover `tau_bm = 40` was 4-14 h only under an unstated 9-18 minutes per frame).

The contract is deliberately minimal, THREE base scales, because mechanics needs three:

    length_um   micrometres per simulation length unit   (the world box is 1 unit by convention)
    time_s      seconds per simulation time unit         (default 1.0: `dt` is in SECONDS)
    force_nN    nanonewtons per simulation force unit    (optional; without it only force RATIOS mean anything)

plus an optional `amount` for field concentrations. Everything else is DERIVED and never declared separately (a second declaration is a second chance to disagree):

    position, radius, l0     L          length_um
    area, volume             L^2, L^3   length_um^2, length_um^3
    velocity                 L/T        length_um / time_s
    rate, 1/tau              1/T        1 / time_s
    force                    F          force_nN
    stress, Young's modulus  F/L^2      force_nN / length_um^2       (-> Pa)
    2D (membrane) modulus    F/L        force_nN / length_um         (-> N/m)
    energy                   F L        force_nN * length_um
    mobility 1/gamma         L/(F T)    length_um / (force_nN * time_s)
    areal mass density       M/L^2      not derivable (declare `amount`)

Deliberately not provided: automatic conversion of state buffers and unit checking of operator ARITHMETIC (a false comfort). Operators compute in simulation units; `Units` is a declaration attached to the run so what comes OUT can be reported in micrometres and seconds. A model with no `units:` block is dimensionless and no statement about it may carry a unit (the loader warns, not fails).

What IS checked is the BOUNDARY: what a spec hands an operator and what the plot reports back, one comparison at load (`Dim` and `Quantity` below; notes/campaigns/UNITS_REFACTOR.md). They never touch a tensor, so the zero-copy path into warp is untouched."""
from __future__ import annotations

from dataclasses import dataclass


# Units: The three base scales of a run (length_um, time_s, force_nN) plus an optional amount; every conversion is derived.
@dataclass(frozen=True)
class Units:
    """The three base scales of a run, plus an optional amount. All conversions are DERIVED."""

    length_um: float = 1.0       # micrometres per simulation length unit
    time_s: float = 1.0          # seconds per simulation time unit (default: dt IS seconds)
    force_nN: float | None = None    # nanonewtons per simulation force unit; None = ratios only
    amount: str | None = None    # free-text label for a field's amount ("nM", "molecules", ...)
    declared: bool = False       # False when no `units:` block was given: the run is dimensionless

    # derived: exactly one way to compute each
    @property
    def area_um2(self):
        return self.length_um ** 2

    @property
    def volume_um3(self):
        return self.length_um ** 3

    @property
    def velocity_um_per_s(self):
        return self.length_um / self.time_s

    @property
    def rate_per_s(self):
        return 1.0 / self.time_s

    @property
    def stress_Pa(self):
        """force/length^2. nN/um^2 = 1e-9 N / 1e-12 m^2 = 1e3 Pa, hence the factor."""
        return None if self.force_nN is None else 1.0e3 * self.force_nN / self.length_um ** 2

    @property
    def tension_N_per_m(self):
        """force/length, the 2D modulus a membrane actually has. nN/um = 1e-9 N / 1e-6 m = 1e-3 N/m."""
        return None if self.force_nN is None else 1.0e-3 * self.force_nN / self.length_um

    @property
    def energy_aJ(self):
        """force*length. nN*um = 1e-9 N * 1e-6 m = 1e-15 J = 1000 aJ."""
        return None if self.force_nN is None else 1.0e3 * self.force_nN * self.length_um

    def seconds(self, sim_time):
        return sim_time * self.time_s

    def hours(self, sim_time):
        return sim_time * self.time_s / 3600.0

    def um(self, sim_length):
        return sim_length * self.length_um

    def describe(self):
        if not self.declared:
            return ("units: NONE DECLARED -- this run is dimensionless and no result from it may be "
                    "quoted with a unit")
        f = "ratios only" if self.force_nN is None else f"{self.force_nN:g} nN"
        s = "" if self.stress_Pa is None else f", 1 stress unit = {self.stress_Pa:.4g} Pa"
        return (f"units: 1 length unit = {self.length_um:g} um, 1 time unit = {self.time_s:g} s "
                f"({self.time_s / 3600.0:.4g} h), 1 force unit = {f}{s}"
                + (f", amount in {self.amount}" if self.amount else ""))


# parse: Build Units from a spec's `general.units:` block -> Units (absent = dimensionless, declared=False).
def parse(raw):
    """Build `Units` from a spec's `general.units:` block. Absent -> dimensionless, `declared=False`. Only the three base scales (and `amount`) are accepted; declaring a derived scale is an ERROR (`area_um2: 4` beside `length_um: 3` is a contradiction the loader would otherwise resolve silently)."""
    if raw is None:
        return Units(declared=False)
    if not isinstance(raw, dict):
        raise ValueError("general.units must be a mapping, e.g. {length_um: 1176, time_s: 600}")
    allowed = {"length_um", "time_s", "force_nN", "amount"}
    extra = set(raw) - allowed
    if extra:
        raise ValueError(
            f"general.units has undeclarable entries {sorted(extra)}. Only {sorted(allowed)} are "
            f"base scales; area, velocity, stress, rate and energy are DERIVED from them (see "
            f"plexus/units.py) and declaring one is a second chance to disagree with the first.")
    for k in ("length_um", "time_s", "force_nN"):
        if k in raw and raw[k] is not None and float(raw[k]) <= 0:
            raise ValueError(f"general.units.{k} must be positive, got {raw[k]}")
    return Units(length_um=float(raw.get("length_um", 1.0)),
                 time_s=float(raw.get("time_s", 1.0)),
                 force_nN=(None if raw.get("force_nN") is None else float(raw["force_nN"])),
                 amount=(None if raw.get("amount") is None else str(raw["amount"])),
                 declared=True)


# the quantity vocabulary: a quantity carries a `Dim` AND an optional CONVENTION. A dimension catches `cell_grow.rate` meaning "per FRAME" while the engine integrated per unit time, and the volume panel's un-multiplied µm³; it cannot catch `v_star - Vbirth`, a WEDGE volume minus a POLYHEDRON volume (2.5433 vs 1.3508 for one cell, both lengths cubed, which made G1 run 60% long): the convention does. No dependency (not `astropy` / `pint`): they wrap values, and these are torch tensors that `warp.from_torch` takes zero-copy; nothing below touches a tensor.

from typing import NamedTuple, Optional          # noqa: E402  (grouped with its own section)

# _SUP, _UNSUP: Digit <-> superscript maps; the parser accepts both spellings (`L^3` and `L³`), so a round trip through `str` holds.
_SUP = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻"}
_UNSUP = {v: k for k, v in _SUP.items()}


# sup: An exponent as superscript text (1 -> the empty string) -> str.
def sup(n: int) -> str:
    """3 -> the superscript three; 1 -> the empty string, because a superscript one is noise."""
    return "" if n == 1 else "".join(_SUP[c] for c in str(n))


# _desup: Superscript digits back to ASCII with a caret, so the parser reads one spelling -> str.
def _desup(s: str) -> str:
    """Superscript digits back to ASCII with a caret, so the parser reads a single spelling."""
    out: list = []
    for c in s:
        if c in _UNSUP:
            if out and out[-1] not in "^-0123456789":
                out.append("^")
            out.append(_UNSUP[c])
        else:
            out.append(c)
    return "".join(out)


# Dim: Integer exponents of the three base scales plus the optional amount (a NamedTuple: one tuple compare).
class Dim(NamedTuple):
    """Integer exponents of the three base scales plus the optional amount: `L` (length_um), `T` (time_s), `F` (force_nN) and `A` for a field's concentration. A tuple of four integers, not a dict: a dimension is compared far more often than built, and a `NamedTuple` gives equality and hashing free."""
    L: int = 0
    T: int = 0
    F: int = 0
    A: int = 0

    def __mul__(self, other):
        return Dim(self.L + other.L, self.T + other.T, self.F + other.F, self.A + other.A)

    def __truediv__(self, other):
        return Dim(self.L - other.L, self.T - other.T, self.F - other.F, self.A - other.A)

    def __pow__(self, n: int):
        return Dim(self.L * n, self.T * n, self.F * n, self.A * n)

    def __str__(self) -> str:
        """`L³`, `1/T`, `F/L²`, `1` -- a spelling `parse_dim` accepts, so a round trip holds."""
        if self == DIMENSIONLESS:
            return "1"
        num = [s + sup(e) for s, e in zip("LTFA", self) if e > 0]
        den = [s + sup(-e) for s, e in zip("LTFA", self) if e < 0]
        top = "*".join(num) if num else "1"
        if not den:
            return top
        bot = "*".join(den)
        return f"{top}/{bot}" if len(den) == 1 else f"{top}/({bot})"


# DIMENSIONLESS, LENGTH, TIME, FORCE, AMOUNT: The base dimensions.
DIMENSIONLESS = Dim()
LENGTH = Dim(L=1)
TIME = Dim(T=1)
FORCE = Dim(F=1)
AMOUNT = Dim(A=1)

# NAMED: This file's table of derived quantities (see the module docstring), transcribed, so an operator writes "volume" and cannot disagree with it; a name is preferred to a formula because only the name stays right if the table is revised.
NAMED = {
    "dimensionless": DIMENSIONLESS,
    "fraction": DIMENSIONLESS,           # a ratio that happens to lie in [0, 1]
    "count": DIMENSIONLESS,              # a number of entities
    "angle": DIMENSIONLESS,              # radians or degrees: a ratio of two lengths either way
    "length": LENGTH,                    # position, radius, rest length, thickness
    "position": LENGTH, "radius": LENGTH, "thickness": LENGTH,
    "area": Dim(L=2),
    "volume": Dim(L=3),
    "time": TIME,
    "velocity": Dim(L=1, T=-1),
    "rate": Dim(T=-1),                   # 1/tau, a per-unit-time frequency
    "force": FORCE,
    "stress": Dim(L=-2, F=1),            # also Young's modulus -> Pa
    "modulus": Dim(L=-2, F=1),
    "membrane_modulus": Dim(L=-1, F=1),  # the 2D modulus Y2 = E*T -> N/m
    "tension": Dim(L=-1, F=1),           # a surface tension: energy per area IS force per length
    "line_tension": FORCE,               # energy per length IS a force
    "energy": Dim(L=1, F=1),
    "mobility": Dim(L=1, T=-1, F=-1),    # 1/gamma
    "amount": AMOUNT,
    "concentration": Dim(L=-3, A=1),
    # electricity without a fourth scale: charge is COUNTED in elementary charges, so a voltage is an energy per charge (F L, through the constant e) and a current a count per time (1/T, through e); added for the proton economy of the flagellar motor (builder/exp_02_bacterium)
    "voltage": Dim(L=1, F=1),            # e psi: the energy one elementary charge gains
    "current": Dim(T=-1),                # elementary charges per unit time
    # the same three dimensions at a single channel's scale (experiments/exp04): tens of pA (30,000 on an fA axis), a nanometre pore, and a gating tension quoted in mN/m (= nN/um relabelled); only the reporting factor and label differ (ELECTRIC below)
    "current_pA": Dim(T=-1),
    "length_nm": LENGTH,
    "tension_mN_m": Dim(L=-1, F=1),
}

# E_CHARGE_C: The elementary charge in coulombs.
E_CHARGE_C = 1.602176634e-19            # coulomb per elementary charge (exact, SI 2019)
# ELECTRIC: name -> (factor applied AFTER the three scales, label): an energy in nN um is 1e-15 J, over e is volts, x1e3 is millivolts; a count per second times e is amperes, x1e15 is fA.
ELECTRIC = {"voltage": (1e-15 / E_CHARGE_C * 1e3, "mV"),
            "current": (E_CHARGE_C * 1e15, "fA"),
            "current_pA": (E_CHARGE_C * 1e12, "pA"),
            "length_nm": (1e3, "nm"),                # um -> nm
            "tension_mN_m": (1.0, "mN/m")}           # nN/um IS mN/m

# UNKNOWN: The absence of a claim, deliberately not a `Dim` (DIMENSIONLESS says "a pure ratio", UNKNOWN says nobody has said yet): treating undeclared as dimensionless would warn on every one of the ~145 unannotated operators and be ignored within a day, so UNKNOWN is compatible with everything, silently.
UNKNOWN = "unknown"


# parse_dim: A quantity string (a name, `L³`, `L^3`, `1/T`, `F/L²`) -> Dim, or UNKNOWN if unreadable.
def parse_dim(s):
    """`"volume"`, `"L³"` or `"L^3"`, `"1/T"`, `"F/L²"`, `"L/(F*T)"`, `"1"` -> a `Dim`. A name first, a formula second (`NAMED` is the authority). Returns `UNKNOWN` rather than raising on anything unreadable: this feeds a WARN-ONLY checker, and a typo in a unit annotation must not stop a run."""
    if s is None or s is UNKNOWN:
        return UNKNOWN
    if isinstance(s, Dim):
        return s
    t = _desup(str(s).strip())
    if not t:
        return UNKNOWN
    if t in NAMED:                       # exact first: `current_pA`, `tension_mN_m` keep their case
        return NAMED[t]
    if t.lower() in NAMED:
        return NAMED[t.lower()]
    try:
        num, _, den = t.partition("/")
        d = _parse_product(num)
        if den:
            d = d / _parse_product(den)
        return d
    except (ValueError, KeyError):
        return UNKNOWN


# _parse_product: A product of dimension factors (`L^3`, `F*L`, `(F*T)`, `1`) -> Dim; raises on anything else.
def _parse_product(s):
    """`L^3`, `F*L`, `(F*T)`, `1` -> a `Dim`. Raises on anything else; `parse_dim` catches."""
    t = s.strip().strip("()").strip()
    if t in ("", "1"):
        return DIMENSIONLESS
    out = DIMENSIONLESS
    for term in t.split("*"):
        base, _, exp = term.strip().partition("^")
        base = base.strip().upper()
        if base not in ("L", "T", "F", "A"):
            raise ValueError(f"unknown base scale {base!r}")
        out = out * (Dim(**{base: 1}) ** (int(exp) if exp.strip() else 1))
    return out


# Quantity: A dimension plus an optional convention: what it is and which way it was measured.
class Quantity(NamedTuple):
    """A dimension plus an optional CONVENTION: what it is, and which way it was measured. The convention is a bare string, not an enum (the set is not knowable in advance). Two exist: `volume[wedge]` (`face_geometry_3d`: origin-referenced cones over the mid-surface ring) and `volume[polyhedron]` (`apicobasal_geometry_3d`: divergence theorem over two caps and one wall per ring edge). `None` = no convention distinction exists for this quantity (a length is a length), which is not the same as `UNKNOWN`."""
    dim: object                          # a `Dim`, or `UNKNOWN`
    convention: Optional[str] = None

    def __str__(self) -> str:
        d = "unknown" if self.dim is UNKNOWN else str(self.dim)
        return f"{d}[{self.convention}]" if self.convention else d


# quantity: A spec-facing quantity string such as `volume[polyhedron]` -> Quantity.
def quantity(s):
    """`"volume[polyhedron]"` -> `Quantity(Dim(L=3), "polyhedron")`, the spec-facing entry point. Not named `parse`, which in this module already builds the run's `Units`."""
    if s is None:
        return Quantity(UNKNOWN)
    if isinstance(s, Quantity):
        return s
    t = str(s).strip()
    conv = None
    if t.endswith("]") and "[" in t:
        t, _, rest = t.partition("[")
        conv = rest[:-1].strip() or None
    return Quantity(parse_dim(t), conv)


# compatible: Whether two quantities describe the same kind of number -> bool.
def compatible(a, b):
    """Do these two describe the same kind of number? Two independent tests, both permissive on absence: dimension (equal, or either `UNKNOWN`: an undeclared quantity accuses nobody) and convention (equal, or either `None`: a plain `volume` is compatible with `volume[wedge]`). Only two different STATED conventions are a finding, which is exactly the `v_star - Vbirth` case."""
    a, b = quantity(a), quantity(b)
    if a.dim is not UNKNOWN and b.dim is not UNKNOWN and a.dim != b.dim:
        return False
    if a.convention and b.convention and a.convention != b.convention:
        return False
    return True


# why_incompatible: One short clause naming the disagreement -> str | None (None if compatible).
def why_incompatible(a, b):
    """One short clause naming the disagreement, for the checker's single line. `None` if ok."""
    a, b = quantity(a), quantity(b)
    if a.dim is not UNKNOWN and b.dim is not UNKNOWN and a.dim != b.dim:
        return f"{a.dim} vs {b.dim}"
    if a.convention and b.convention and a.convention != b.convention:
        return f"{a.convention} vs {b.convention} (same dimension, different measurement)"
    return None


# to_physical: Simulation units -> physical units, the only converter in the tree -> float | None.
def to_physical(value, dim, units):
    """Simulation units -> physical units, the ONLY function in the tree allowed to convert.

        physical = value * length_um**L * time_s**T * force_nN**F

    `units` is this module's `Units`. Returns `None` ("cannot be quoted") when the run is undeclared or a base scale the dimension needs was not given (without `force_nN` only force RATIOS are meaningful); the caller then prints the bare simulation number with no unit. State buffers and operators stay in simulation units: this exists for the reporting boundary (scale bar, curve axes, panel readouts, colour legend)."""
    d = dim if isinstance(dim, Dim) else parse_dim(dim)
    if d is UNKNOWN or value is None or units is None or not getattr(units, "declared", False):
        return None
    if d.A != 0:                         # `amount` is a LABEL, not a number: nothing to scale by
        return None
    scale = 1.0
    for exp, s in ((d.L, units.length_um), (d.T, units.time_s), (d.F, units.force_nN)):
        if exp == 0:
            continue
        if s is None:
            return None                  # an undeclared scale is not a scale of 1
        scale *= float(s) ** exp
    _name = str(dim).strip().split("[")[0] if isinstance(dim, str) else None
    if _name in ELECTRIC:                # through the elementary charge: mV, fA
        scale *= ELECTRIC[_name][0]
    return value * scale


# unit_label: The unit label of a quantity (`µm`, `µm²`, `s`, `nN`, ...) -> str | None.
def unit_label(dim, units):
    """`µm`, `µm²`, `µm³`, `s`, `µm/s`, `nN` ... or `None` when the run cannot quote one. Built from the same declaration as `to_physical`, so a label can never appear over a number that was not converted (the movie's volume panel once printed `µm³` beside a raw value, wrong by 1000 on every tissue run)."""
    d = dim if isinstance(dim, Dim) else parse_dim(dim)
    if d is UNKNOWN or d == DIMENSIONLESS or units is None or not getattr(units, "declared", False):
        return None
    _name = str(dim).strip().split("[")[0] if isinstance(dim, str) else None
    if _name in ELECTRIC:
        if (d.L and units.length_um is None) or (d.F and units.force_nN is None) or (d.T and units.time_s is None):
            return None
        return ELECTRIC[_name][1]
    num, den = [], []
    for exp, s, sym in ((d.L, units.length_um, "µm"), (d.T, units.time_s, "s"),
                        (d.F, units.force_nN, "nN")):
        if exp == 0:
            continue
        if s is None:
            return None                  # cannot label what cannot be converted
        (num if exp > 0 else den).append(sym + sup(abs(exp)))
    if not num and not den:
        return None
    top = "·".join(num) if num else "1"
    return top if not den else top + "/" + "·".join(den)
