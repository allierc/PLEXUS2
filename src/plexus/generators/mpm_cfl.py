"""Grid-dt CFL auto-correction for MPM specs, applied to the YAML, not in code.

The MLS-MPM substep is explicit, so it obeys the Courant condition: an elastic wave may not cross more
than ~`cfl` of a grid cell per substep,

    dt_sub <= cfl * dx / c ,   c = sqrt((lambda + 2*mu) / rho)   (elastic P-wave speed),   dx = 1/n_grid

with c set by the STIFFEST material (liquids have mu = 0 and are slow). When the micro-timestep exceeds
the limit this warns and rewrites the spec.yaml in place, lowering it to the stable value while PRESERVING
the per-frame simulated time (idempotent: a corrected spec is unchanged on re-run).

A liquid with `cell_surface_tension` also has a CAPILLARY limit, independent of the elastic one and often
much smaller (Brackbill / Kothe / Zemach 1992):

    dt_cap <= cfl * sqrt(rho_liquid * dx^D / (2*pi*sigma))

It is computed only when `csf_rho` is set (otherwise `cell_surface_tension` is a tension divided by a cell
mass, not a physical tension) and is +inf otherwise. Calibrated on material_3d_water_drop (290k particles,
n_grid 96): the peak node mass reads 1.8 / 10.7 / 60.8 full liquid cells at dt / dt_cap_raw = 0.50 / 0.80 /
1.00, so the same `cfl` factor (default 0.4) is applied to it, leaving 1.25x margin on the measured edge.

Both MPM spec forms are handled: the ORACLE op `mls_mpm_mechanics` (lowers `dt_sub`, raises `substeps`) and
the DECOMPOSED cycle (a schedule block `{substep_dt: X, steps: [mpm_strain, mpm_scatter, mpm_grid_update,
mpm_gather]}`: lowers only `substep_dt`, the count auto-rises). Only the relevant tokens are edited, so
comments and layout survive."""
from __future__ import annotations

import math
import os
import re

import yaml

from plexus.paths import warn

# _NU: Poisson ratio used to turn `youngs` into the Lame pair (matches the mpm_particle entity).
_NU = 0.2                                    # Poisson ratio (matches the mpm_particle entity)


# _lame: The Lame pair of Young's modulus E -> (mu, lambda).
def _lame(E, nu=_NU):
    mu = E / (2 * (1 + nu))
    la = E * nu / ((1 + nu) * (1 - 2 * nu))
    return mu, la


# _max_wave_speed: Largest elastic P-wave speed over every set's materials -> float (mu = 0 for a liquid).
def _max_wave_speed(spec, rho_default=1.0) -> float:
    """Largest elastic P-wave speed over EVERY set's materials (mu = 0 for liquid).

    Walks all sets, not `sets["cell"]`: in a composed body the materials live on the CHILD sets (nucleus at 300, membrane at 1500), and reading a soft placeholder on the parent makes c too SMALL, so the limit too LOOSE and the guard silently passes a substep that should be tightened. Density is per set, overridable per type, as in `models/entities.py`."""
    # density and material can sit on different sets (27 water blocks declare `youngs` / `material` on the parent `cell` and `density` on the child `mpm_particle`): pair them, for every set that declares a density, with the sets whose types describe ITS material (itself and its parent), or a rho 0.25 spec was evaluated at 1.0 and passed as stable
    sets = spec.get("sets") or {}
    # declared densities first, defaults only where nothing was declared (a default 1.0 on the parent won the min(), worth sqrt(1000) in the wave speed of an SI water spec)
    rho_decl = {sn: float(st["density"]) for sn, st in sets.items()
                if isinstance(st, dict) and "density" in st}
    rho_for = {}
    for sname, st in sets.items():
        if not isinstance(st, dict):
            continue
        rho_for[sname] = rho_decl.get(sname, rho_default)
        par = st.get("parent")
        if par in sets and sname in rho_decl:
            # the child's particles carry the child's density and take their material from the parent's type table: attribute the lower declared density to the parent so a heavy child is not missed, but never let an undeclared parent's default win
            cand = [rho_decl[sname]] + ([rho_decl[par]] if par in rho_decl else [])
            rho_for[par] = min(cand)
    cmax = 0.0
    for sname, st in sets.items():
        if not isinstance(st, dict):
            continue
        rho_set = rho_for.get(sname, float(st.get("density", rho_default)))
        for t in (st.get("types") or {}).values():
            if not isinstance(t, dict):
                continue
            rho = float(t.get("density", rho_set))          # per-type density wins (light/heavy cytosol)
            layers = list(t.get("layers") or [{"youngs": t.get("youngs"),
                                               "bulk_modulus": t.get("bulk_modulus"),
                                               "material": t.get("material", "elastic")}])
            if t.get("core"):
                layers.append(t["core"])
            for L in layers:
                mat = L.get("material", t.get("material", "elastic"))
                # `bulk_modulus` IS lambda for a liquid (mu = 0); without it the pass fell back to the `youngs` default of 100 and reported a speed for a material the spec never declared
                K = L.get("bulk_modulus", t.get("bulk_modulus"))
                if K is not None and mat == "liquid":
                    mu, la = 0.0, float(K)
                else:
                    E = L.get("youngs", t.get("youngs"))
                    mu, la = _lame(float(E if E is not None else 100.0))
                    if mat == "liquid":
                        mu = 0.0             # liquid carries no shear modulus
                cmax = max(cmax, math.sqrt(max(la + 2 * mu, 0.0) / max(rho, 1e-12)))
    return cmax


# _similarity: Report the dimensionless groups of an MPM scene (Mach, Reynolds, Bond, Weber, ...) at build.
def _similarity(spec, dx: float, dim: int) -> list:
    """The dimensionless groups of an MPM scene, reported at build.

    A report, not a check: past diagnoses were all arithmetic on numbers the spec already contains (the corpus "water" impacts at Mach 0.449 with 10.1% volumetric strain; a drop sampled at 283 particles per cell against a convention of 8; 64 mL over a 0.3 m floor lands 0.34 of one cell deep). Printing them turns a wasted cluster run into a line of output. Each group is emitted only when the spec declares what it is built from (Reynolds needs a viscosity, Bond and Weber a surface tension, Mach a sound speed): a group computed from a default is a number about the default."""
    import math as _m
    out = []
    sets = spec.get("sets") or {}
    gen = spec.get("general") or {}
    w = gen.get("world", 1.0)
    box = [float(x) for x in w] if isinstance(w, (list, tuple)) else [float(w)] * dim
    H = box[1] if len(box) > 1 else box[0]
    g = 0.0
    for o in (spec.get("operators") or []):
        if isinstance(o, dict) and o.get("op") == "gravity":
            g = abs(float(o.get("g", 0.0) or 0.0))
    rho = 1.0
    for st in sets.values():
        if isinstance(st, dict) and "density" in st:
            rho = float(st["density"])
    K, R = _liquid_scale(spec, dim)
    if not (K < float("inf")):
        # not a liquid but still a wave: a solid or snow body has lambda + 2*mu, and Mach says whether it is simulated or merely deformed (si_freefall's snowball impacts at Mach 1.85); only the surface-tension groups need a liquid
        c_s = _max_wave_speed(spec)
        if c_s <= 0:
            return out
        # `_max_wave_speed` returns a SPEED: K = rho*c^2 (using it as K read snow's 19.7 m/s as 0.986 and its Mach as 37.6)
        K, R = rho * c_s * c_s, float("inf")
    c = _m.sqrt(K / max(rho, 1e-30))
    # the fall and the body's own height, from a `block` if written, else from the DERIVED volume (a `particle_mass` spec has no block; reading the box reports the room's height: self-weight strain 25.2% instead of the body's 10.1%)
    y0, hgt = None, None
    for st in sets.values():
        if not isinstance(st, dict):
            continue
        for t in ((st.get("types")) or {}).values():
            b = (t or {}).get("block")
            if b and len(b) >= 2 * dim:
                y0 = float(b[1]) if y0 is None else min(y0, float(b[1]))
                _h = abs(float(b[dim + 1]) - float(b[1]))
                hgt = _h if hgt is None else max(hgt, _h)
    if y0 is None:                                    # derived volume: N * m_p / rho, on `start`
        for sn, st in sets.items():
            if not isinstance(st, dict) or "particle_mass" not in st:
                continue
            _n = int(st.get("per_parent", 0) or 0)
            _mp = float(st["particle_mass"]); _r = float(st.get("density", 1.0) or 1.0)
            _side = (_n * _mp / _r) ** (1.0 / dim)
            _par = sets.get(st.get("parent")) or {}
            _st = (_par.get("start") or [[0.0] * dim])[0]
            y0 = float(_st[1]) - _side / 2.0
            hgt = _side
    fall = max((y0 or 0.0) - 2.0 * dx, 0.0)
    U = _m.sqrt(2.0 * g * fall) if g > 0 else 0.0
    L = R if R < float("inf") else (hgt if hgt else H)
    Hb = hgt if hgt else H                            # the BODY's height, for the self-weight strain
    out.append(f"c = sqrt(K/rho) = {c:.4g}  (K {K:.4g}, rho {rho:g})")
    if U > 0:
        out.append(f"Mach {U / c:.3f}  ->  drho/rho ~ {100 * (U / c) ** 2:.2f}%"
                   + ("   <- WARN: above 0.2, this is a compressible foam, not a liquid"
                      if U / c > 0.2 else ""))
        out.append(f"Froude {U / _m.sqrt(g * L):.3f}   (U {U:.4g}, L {L:.4g})")
    if g > 0:
        out.append(f"self-weight volumetric strain rho*g*H/K = {100 * rho * g * Hb / K:.3f}%"
                   f"   (H = the body's own height, {Hb:.4g})")
    sig, eta, sig_legacy = 0.0, 0.0, 0.0
    for o in (spec.get("operators") or []):
        if not isinstance(o, dict):
            continue
        if o.get("op") == "mpm_grid_update":
            _s = float(o.get("cell_surface_tension", 0.0) or 0.0)
            if o.get("csf_rho"):
                sig = _s
            elif _s > 0:
                sig_legacy = _s
        if o.get("op") == "mpm_viscosity":
            eta = float(o.get("eta", 0.0) or 0.0)
    if eta > 0 and U > 0:
        out.append(f"Reynolds rho*U*L/eta = {rho * U * L / eta:.4g}")
    if sig_legacy > 0:
        # without `csf_rho` the colour field is a MASS per node (~rho*dx^D), so the effective tension is sigma * rho * dx^D (1.13e-6 at rho 1, n_grid 96, which is why `cell_surface_tension: 60` moves nothing) and it changes meaning with n_grid; Bond is reported for these 54 legacy specs too, to say so
        _se = sig_legacy * rho * dx ** dim
        out.append(f"cell_surface_tension {sig_legacy:g} is UNNORMALISED (no csf_rho): the effective "
                   f"tension is sigma*rho*dx^{dim} = {_se:.4g}, {sig_legacy / max(_se, 1e-300):.3g}x "
                   f"smaller, and it RESCALES WITH n_grid")
        if g > 0 and _se > 0:
            out.append(f"Bond rho*g*L^2/sigma_effective = {rho * g * L * L / _se:.4g}"
                       + ("   <- WARN: above 100, gravity beats surface tension by orders of "
                          "magnitude and this spec's surface tension does nothing"
                          if rho * g * L * L / _se > 100 else ""))
    if sig > 0:
        if g > 0:
            out.append(f"Bond rho*g*L^2/sigma = {rho * g * L * L / sig:.4g}")
            lc = _m.sqrt(sig / (rho * g))
            out.append(f"capillary length sqrt(sigma/rho g) = {lc:.4g} = {lc / dx:.1f} cells"
                       + ("   <- WARN: under 2 cells, nothing shaped by surface tension is resolved"
                          if lc / dx < 2.0 else ""))
        if U > 0:
            out.append(f"Weber rho*U^2*L/sigma = {rho * U * U * L / sig:.4g}")
    return out


# _cell_size: The background cell size world_size[1] / n_grid, as MPMGrid computes it -> float.
def _cell_size(spec, n_grid: int, dim: int) -> float:
    """The background cell size, `world_size[1] / n_grid`, matching MPMGrid. `dx` enters the guard as `dt <= cfl * dx / c`, so a dx too large (the old `1.0 / n_grid` assumed a unit-width world) blesses a substep too large. Reads the same `general.world` as schema.py."""
    w = (spec.get("general") or {}).get("world", 1.0)
    if isinstance(w, (list, tuple)):
        box = [float(x) for x in w]
    else:
        box = [float(w)] + [1.0] * (dim - 1)         # scalar: axis-0 width, the rest unit
    return box[1] / float(n_grid) if len(box) > 1 else box[0] / float(n_grid)


# _capillary_limit: The Brackbill / Kothe / Zemach capillary-wave limit -> (dt_cap_raw, sigma, rho), (inf, 0, 0) when it does not apply.
def _capillary_limit(spec, dx: float, dim: int):
    """The Brackbill / Kothe / Zemach capillary-wave limit, or (inf, 0, 0) when it does not apply.

        dt_cap_raw = sqrt(rho_liquid * dx^D / (2*pi*sigma))

    is the period of the shortest capillary wave the mesh holds (wavelength 2*dx) over 2*pi; stepping past it grows the wave. It is the liquid's own limit, independent of `youngs`, so it bites hardest on a pure liquid. Only when `csf_rho` is set: otherwise the colour is a mass per node and the yaml's tension is 1e6 too small at n_grid 96, and this would re-time every legacy spec for a force that is not there. rho is `csf_rho` (the liquid density by definition); with several `mpm_grid_update` operators the SMALLEST limit wins. Using rho_liquid rather than Brackbill's rho_avg = rho_liquid / 2 is looser by sqrt(2), well inside the 3.5x of the `cfl` factor."""
    dt_cap, sig_at, rho_at = float("inf"), 0.0, 0.0
    for o in (spec.get("operators") or []):
        if not isinstance(o, dict):
            continue
        try:
            sigma = float(o.get("cell_surface_tension", 0.0) or 0.0)
            rho_l = float(o.get("csf_rho", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if sigma <= 0.0 or rho_l <= 0.0:
            continue                                 # legacy / unnormalised / tension off
        d = math.sqrt(rho_l * dx ** dim / (2.0 * math.pi * sigma))
        if d < dt_cap:
            dt_cap, sig_at, rho_at = d, sigma, rho_l
    return dt_cap, sig_at, rho_at


# _liquid_scale: (softest liquid bulk modulus K, smallest liquid body radius R) -> (inf, inf) if no liquid.
def _liquid_scale(spec, dim: int):
    """(softest liquid bulk modulus K, smallest liquid body radius R), (inf, inf) if no liquid.

    A liquid has mu = 0, so its bulk modulus IS lambda = E*nu/((1+nu)(1-2nu)): at nu = 0.2, `youngs: 200` gives K = 55.6, not 200. The softest liquid and smallest body are the worst case for the ratio (least resistance, largest curvature). R is the equivalent radius of the declared extent (`block` or `radius`), the body as written before it falls and spreads, so the report is a floor on the real number."""
    sets = spec.get("sets") or {}
    K, R = float("inf"), float("inf")
    for sname, st in sets.items():
        if not isinstance(st, dict):
            continue
        rad_default = float(st.get("radius", 0.05) or 0.05)
        for t in (st.get("types") or {}).values():
            if not isinstance(t, dict):
                continue
            layers = list(t.get("layers") or [{"youngs": t.get("youngs"),
                                               "bulk_modulus": t.get("bulk_modulus"),
                                               "material": t.get("material", "elastic")}])
            if t.get("core"):
                layers.append(t["core"])
            liq = [L for L in layers
                   if L.get("material", t.get("material", "elastic")) == "liquid"]
            if not liq:
                continue
            for L in liq:
                # `bulk_modulus` says it outright; `youngs` reaches the same number through a modulus a liquid lacks (mu = 0, so K = lambda either way)
                _K = L.get("bulk_modulus", t.get("bulk_modulus"))
                if _K is None:
                    _E = L.get("youngs", t.get("youngs"))
                    _mu, _K = _lame(float(_E if _E is not None else 100.0))
                K = min(K, float(_K))
            b = t.get("block")
            if b and len(b) >= 2 * dim:
                v = 1.0
                for k in range(dim):
                    v *= abs(float(b[dim + k]) - float(b[k]))
                r = (math.sqrt(v / math.pi) if dim == 2
                     else (3.0 * v / (4.0 * math.pi)) ** (1.0 / 3.0))
            else:
                r = float(t.get("radius", rad_default))
            R = min(R, r) if r > 0 else R
    return K, R


# _snap_to_frame: The largest step <= dt_target that divides dt_frame exactly -> float.
def _snap_to_frame(dt_target: float, dt_frame: float) -> float:
    """The largest step <= `dt_target` that divides `dt_frame` EXACTLY. Both constraints at once: `dt_frame / round(dt_frame / dt_target)` divides exactly but can land ABOVE the stability limit (material_3d_multimaterial: 1.231e-04 against 1.226e-04); taking the ceiling of the substep count can only make the step smaller."""
    if not dt_frame or dt_target <= 0.0:
        return dt_target
    n = max(1, int(math.ceil(dt_frame / dt_target - 1e-9)))
    return dt_frame / n


# Courant_Friedrichs_Lewy_condition: Check, and if needed correct in place, the grid-dt CFL of an MPM spec -> (changed, info | None).
def Courant_Friedrichs_Lewy_condition(yaml_path: str, write: bool = True, verbose: bool = False):
    """Check (and, if needed, correct in place) the grid-dt CFL of an MPM spec.

    Prints exactly ONE `[grid-CFL]` line in the terminal (the verdict, and the correction when one was made); `verbose=True` also prints the
    `[similarity]` lines and the Laplace/bulk note. Warnings (`warn`) are separate and stay.

    Returns (changed: bool, info: dict|None). A spec with no `mls_mpm_mechanics`
    operator is left untouched (returns (False, None))."""
    # read the raw YAML text: the correction is a targeted text rewrite so comments and formatting survive; `spec` is the parsed view
    text = open(yaml_path).read()
    spec = yaml.safe_load(text)
    # `general.cfl_lock: true` marks a deliberately chosen substep count (Studio steps/frame field, or by hand): report the verdict, do not rewrite it (substeps are a linear multiplier on frame time, and correcting a cheap preview back up would remove the one knob that makes previewing viable)
    _lock = bool((spec.get("general") or {}).get("cfl_lock", False))
    _say = (lambda line: print(line, flush=True)) if verbose else (lambda line: None)     # the detail lines, only when asked for
    _fixnote = ""                                                                         # appended to the one verdict line when the frame-clock fix was applied

    # the MPM micro-timestep lives in one of two forms: (1) the ORACLE op `mls_mpm_mechanics` (`dt_sub` + explicit `substeps`), (2) the DECOMPOSED cycle, a schedule block `{substep_dt: X, steps: [...]}` whose count is implicit, round(general.dt / substep_dt)
    MPM_STEPS = {"mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather",
                 "p2g", "g2p", "agent_scatter", "agent_gather"}        # incl. transitional aliases
    op = next((o for o in spec.get("operators", []) if o.get("op") == "mls_mpm_mechanics"), None)
    if op is not None:
        token = "dt_sub"                                              # which YAML key we rewrite
        micro_dt = float(op.get("dt_sub", 2e-4))                      # the explicit MICRO-timestep
        substeps = int(op.get("substeps", 10))                       # explicit; per-frame time = substeps*dt_sub
        n_grid = int(op.get("n_grid", 128))
        cfl = float(op.get("cfl", 0.4))
    else:
        blk = next((s for s in spec.get("schedule", [])              # the decomposed substep micro-loop
                    if isinstance(s, dict) and "substep_dt" in s
                    and MPM_STEPS & set(s.get("steps", []))), None)
        if blk is None:
            # a flat MPM schedule (cycle at the frame timestep, no substep block) still gets its report, and is told nothing bounds its step; there is no key to rewrite, so this warns
            _ops = {o.get("op") for o in (spec.get("operators") or []) if isinstance(o, dict)}
            if MPM_STEPS & _ops:
                _nm = spec.get("general", {}).get("name", os.path.basename(yaml_path))
                _d = int(spec.get("general", {}).get("dim", 3))
                _ng = next((int(fc["n_grid"]) for fc in spec.get("fields", {}).values()
                            if isinstance(fc, dict) and "n_grid" in fc), 128)
                for _ln in _similarity(spec, _cell_size(spec, _ng, _d), _d):
                    _say(f"[similarity] {_nm}: {_ln}")
                print(f"[grid-CFL] {_nm}: NO substep block -- the MPM cycle runs once per frame at "
                      f"general.dt={spec.get('general', {}).get('dt')}, so nothing here checks it "
                      f"against the elastic or capillary limit. Wrap the mpm_* steps in "
                      f"{{substep_dt: ..., steps: [...]}} to get the guard.", flush=True)
            else:
                print(f"[grid-CFL] {spec.get('general', {}).get('name', os.path.basename(yaml_path))}: no MPM step in this spec, nothing to check.", flush=True)
            return False, None                                       # nothing to correct either way
        token = "substep_dt"
        micro_dt = float(blk.get("substep_dt", 2e-4))
        substeps = None                                              # implicit: round(general.dt / substep_dt)
        cfl = float(blk.get("cfl", 0.4))
        # grid resolution lives on the mpm_grid FIELD (`fields: {mpm_grid: {n_grid: 128}}`)
        n_grid = next((int(fc["n_grid"]) for fc in spec.get("fields", {}).values()
                       if isinstance(fc, dict) and "n_grid" in fc), 128)

    # the frame clock: `engine.run` takes round(general.dt / substep_dt) steps of substep_dt, so the frame advances `count * substep_dt`, equal to general.dt only when the division is exact (substep_dt 2.0e-4 against dt 1.5e-3 advances 1.6e-3, 6.7% too much, and reads like integrator inaccuracy)
    _dtf = float((spec.get("general") or {}).get("dt", 0.0)) if substeps is None else None
    if _dtf:
        _n = max(1, round(_dtf / micro_dt))
        _err = abs(_n * micro_dt - _dtf) / _dtf
        # 1e-5, not 1e-6, and six digits on the way out: the corrected value is written as a decimal, so four digits left a 1.2e-05 residual and this branch re-fired on its own output every run
        _divide_fix = _snap_to_frame(micro_dt, _dtf) if _err > 1e-5 else None

    # the stability limit (same physics either form)
    dim = int((spec.get("general") or {}).get("dim", 2))
    dx = _cell_size(spec, n_grid, dim)           # grid CELL SIZE, from the WORLD BOX
    cmax = _max_wave_speed(spec)                 # FASTEST elastic P-wave over EVERY set, c = sqrt((la+2mu)/rho)
    # two waves, two limits, the smaller wins: elastic (P-wave) and capillary (shortest surface wave, only where `csf_rho` normalises the colour); the same `cfl` factor applies to both, since at the bare capillary limit a drop piles 60.8 full cells of mass on one node against 1.8 clean
    cap_raw, sigma, rho_liq = _capillary_limit(spec, dx, dim)

    _nm = (spec.get("general") or {}).get("name", os.path.basename(yaml_path))
    for _line in _similarity(spec, dx, dim):
        if "WARN" in _line:
            warn(f"[similarity] {_nm}: {_line}")
        else:
            _say(f"[similarity] {_nm}: {_line}")
    dt_cap = cfl * cap_raw
    if cmax <= 0.0 and dt_cap == float("inf"):
        print(f"[grid-CFL] {_nm}: no elastic material and no surface tension, nothing bounds the step.", flush=True)
        return False, None                       # no elastic material AND no tension -> nothing to bound
    # Courant limit: a wave may cross at most `cfl` of a cell per micro-step; a larger `micro_dt` makes the explicit substep unstable
    dt_el = cfl * dx / cmax if cmax > 0.0 else float("inf")
    dt_cfl = min(dt_el, dt_cap)                  # THE binding limit
    binds = "capillary" if dt_cap < dt_el else "elastic"
    # by what margin one limit beats the other -- 1.0 means they coincide
    _ratio = (max(dt_el, dt_cap) / min(dt_el, dt_cap)
              if min(dt_el, dt_cap) > 0 and math.isfinite(max(dt_el, dt_cap)) else float("inf"))
    _why = ""
    if math.isfinite(dt_cap):
        _why = (f"; {binds} binds"
                + (f" by {_ratio:.2f}x over " + ("elastic" if binds == "capillary" else "capillary")
                   if math.isfinite(_ratio) else "")
                + f" [elastic {dt_el:.2e} (c_max={cmax:.1f}), capillary {dt_cap:.2e} "
                  f"(sigma={sigma:g}, rho_liq={rho_liq:g})]")
    name = spec.get("general", {}).get("name", yaml_path)

    # the other capillary limit has no dt in it: La = 2*sigma / (R * K) (1*sigma / R in 2D), the Laplace pressure over the liquid's bulk modulus K = lambda (mu = 0), is the fraction of its own volume the tension asks for
    # measured on material_3d_water_drop: La 0.19-0.27 clean, 0.38 degraded, 0.47 collapse (mean J 1e-6); doubling `youngs` cured a sigma-2.8 collapse, shrinking the substep 3x or changing the particle count did not
    # REPORTED, never corrected: every cure (stiffer liquid, weaker tension, bigger drop, `csf_smooth`, a wider `csf_band`) changes what the run IS
    _La = 0.0
    if sigma > 0.0 and rho_liq > 0.0:
        _K, _R = _liquid_scale(spec, dim)
        if math.isfinite(_K) and math.isfinite(_R) and _K > 0 and _R > 0:
            _La = (dim - 1) * sigma / (_R * _K)      # 2*sigma/(R*K) in 3D, sigma/(R*K) in 2D
            if _La > 0.35:
                warn(f"{name}: surface tension asks for more compression than this liquid can "
                     f"refuse -- Laplace/bulk La = {(dim - 1)}*sigma/(R*K) = {_La:.2f} "
                     f"(sigma={sigma:g}, R={_R:.3g}, K=lambda={_K:.4g}, from youngs at nu={_NU}). "
                     f"Measured on material_3d_water_drop at n_grid 96: La 0.38 costs 24-27% of "
                     f"the drop's birth volume and La 0.47 collapses it to a point (mean det F = "
                     f"1e-6, 13,000 full cells of mass on one node). LOWERING substep_dt DOES NOT "
                     f"HELP -- five steps from 2.0e-4 to 6.4e-5 collapsed alike, and so did 0.5x "
                     f"and 2x the particle count. Raise `youngs` (K scales with it), lower "
                     f"`cell_surface_tension`, or mollify the curvature: `csf_smooth: 4` took that same "
                     f"collapsing run to mean J 0.875, and `csf_band: 0.35` to mean J 0.878.")
            elif _La > 0.15:
                _say(f"[grid-CFL] {name}: Laplace/bulk La = {_La:.2f} (sigma={sigma:g}, "
                     f"R={_R:.3g}, K={_K:.4g}) -- the tension asks for {_La * 100:.0f}% of the "
                     f"liquid's volume at first order; measured 6.3% loss of mean det F at "
                     f"La 0.19, 9.8% at La 0.27, 24% at La 0.38, total collapse at La 0.47.")

    # already stable? leave it: the 1% slack makes this idempotent (it absorbs the 3-significant-figure rounding of a value written last run). The clock fix is APPLIED, not announced, and can only shrink the step, so it cannot break the stability bound below
    if _dtf and _divide_fix is not None:
        _n_new = max(1, round(_dtf / _divide_fix))
        _fixnote = (f"; {token} {micro_dt:.3e} did not divide general.dt={_dtf:.3e} "
                    f"({_err * 100:.1f}% of a frame lost per frame), corrected -> {_divide_fix:.4e} for exactly {_n_new} substeps")
        micro_dt = _divide_fix
        if write and not _lock:
            text = re.sub(rf"(\b{token}:\s*)[0-9.eE+\-]+",
                          lambda m: f"{m.group(1)}{_divide_fix:.6e}", text, count=1)
            open(yaml_path, "w").write(text)

    if micro_dt <= dt_cfl * 1.01:
        # stable, and says by how much (substeps are a linear multiplier on frame time, so unreported headroom is unclaimed speed: cell_02 and cell_03 ran at 3.7x); reported only, never acted on (CFL bounds stability, not accuracy)
        head = dt_cfl / micro_dt
        # one line, always printed, at any headroom (water_st560 sits 1.13x under the capillary limit, inside the noise of "silent", and 1.97x OVER it at 5x the tension)
        # substeps: explicit on the oracle form, implicit as round(general.dt / substep_dt)
        dt_frame = (substeps * micro_dt if substeps is not None
                    else float((spec.get("general") or {}).get("dt", 0.0)))
        now = round(dt_frame / micro_dt) if dt_frame else None
        could = max(1, round(dt_frame / dt_cfl)) if dt_frame else None
        extra = (f"; {now} substeps/frame where {could} would be stable ({now / could:.1f}x)"
                 if now and could and head >= 1.5 else "")
        print(f"[grid-CFL] {name}: {token}={micro_dt:.2e} is {head:.2f}x BELOW the stability "
              f"limit {dt_cfl:.2e} (c_max={cmax:.1f}){extra}{_why}{_fixnote}.", flush=True)
        return False, {"name": name, "cmax": cmax, "micro_dt": micro_dt, "dt_cfl": dt_cfl,
                       "ok": True, "headroom": head, "La": _La, "binds": binds, "dt_elastic": dt_el,
                       "dt_capillary": dt_cap, "sigma": sigma, "csf_rho": rho_liq}

    # too big: shrink the micro-timestep to the limit, KEEPING per-frame time. Oracle: per-frame time = substeps*dt_sub, so raise substeps as dt_sub falls. Decomposed: per-frame time = general.dt and the count is implicit, so only `substep_dt` shrinks, snapped to a step that divides general.dt so the correction does not create the clock error
    dt_new = dt_cfl if substeps is not None else _snap_to_frame(dt_cfl, _dtf)
    new_sub = int(math.ceil(substeps * micro_dt / dt_cfl)) if substeps is not None else None
    info = {"name": name, "cmax": cmax, "token": token, "dt_old": micro_dt, "dt_new": dt_new,
            "sub_old": substeps, "sub_new": new_sub, "La": _La, "binds": binds, "dt_elastic": dt_el,
            "dt_capillary": dt_cap, "sigma": sigma, "csf_rho": rho_liq,
            "over_by": micro_dt / dt_cfl}
    sub_note = (f", substeps {substeps}->{new_sub}" if new_sub is not None
                else f" ({round(_dtf / dt_new)} substeps, dividing general.dt exactly)" if _dtf
                else " (substeps auto = round(dt/substep_dt))")
    _msg = (f"[grid-CFL] {name}: {token}={micro_dt:.2e} > limit {dt_cfl:.2e} "
          f"({micro_dt / dt_cfl:.2f}x OVER) (c_max={cmax:.1f}, dx={dx:.2e}, cfl={cfl})"
          f"{_why}; correcting spec -> "
          f"{token}={dt_new:.4e}{sub_note} (per-frame time preserved)."
          if not _lock else
          f"[grid-CFL] {name}: {token}={micro_dt:.2e} > limit {dt_cfl:.2e} "
          f"({micro_dt / dt_cfl:.2f}x OVER) (c_max={cmax:.1f}, dx={dx:.2e}, cfl={cfl}){_why}; "
          f"NOT corrected -- general.cfl_lock is set, so this substep_dt is the spec's own choice. "
          f"The run may be unstable; {round(_dtf / dt_new) if _dtf else new_sub} substeps/frame "
          f"would be stable.")
    print(_msg + _fixnote, flush=True)
    if write and not _lock:
        # Rewrite ONLY the relevant token(s) in the raw text (regex, count=1) so comments/layout survive.
        text = re.sub(rf"(\b{token}:\s*)[0-9.eE+\-]+", lambda m: f"{m.group(1)}{dt_new:.6e}", text, count=1)
        if new_sub is not None:                  # oracle form only: also bump the explicit substeps
            text = re.sub(r"(\bsubsteps:\s*)\d+", lambda m: f"{m.group(1)}{new_sub}", text, count=1)
        open(yaml_path, "w").write(text)
    return True, info


# PPC_TARGET: Particles per grid cell the material points should have: 2 per axis (8 in 3D, 4 in 2D, set from `dim`); with too few a cell cannot determine the local deformation and the body loses stiffness and fractures numerically (a sampling problem that reads as a material one). Reported, never acted on.
PPC_TARGET = 8.0                 # 2 particles per axis in 3D (4.0 in 2D; set below from `dim`)
# PPC_FLOOR: Fraction of PPC_TARGET below which the material is reported as under-sampled.
PPC_FLOOR = 0.5                  # fraction of target below which the material is under-sampled


# _per_parent_mean: The average children per parent (an int `per_parent`, or a per-type mapping weighted by parent count) -> float.
def _per_parent_mean(sets: dict, sv: dict) -> float:
    """The AVERAGE children per parent. `per_parent` is one integer, or a mapping from the parent's
    type name to a count -- in which case the average is weighted by how many parents carry each
    type, since a 50-point membrane patch and a 50,000-point nuclear patch are both parents."""
    pp = sv.get("per_parent", 0)
    if not isinstance(pp, dict):
        return float(pp or 0)
    ptypes = ((sets.get(sv.get("parent")) or {}).get("types") or {})
    if not ptypes:
        return float(sum(pp.values())) / max(len(pp), 1)
    w = {tn: float(t.get("count", t.get("fraction", 0.0)) or 0.0) for tn, t in ptypes.items()}
    tot = sum(w.values()) or float(len(w))
    return sum(float(pp.get(tn, 0)) * w.get(tn, 0.0) for tn in ptypes) / tot


# _set_size: How many elements a declared set holds, following the containment chain -> int.
def _set_size(sets: dict, name: str | None) -> int:
    """How many elements a declared set holds: `n` at the root, else per_parent x the parent's
    size, up the containment chain. Bounded, because a malformed spec must not hang a diagnostic."""
    n, seen = 1.0, set()
    while name and name in sets and name not in seen:
        seen.add(name)
        sv = sets[name] or {}
        if "per_parent" in sv:
            n *= _per_parent_mean(sets, sv); name = sv.get("parent")
        else:
            return int(n * int(sv.get("n", 1)))
    return int(n)


# _body_volumes: (name, volume) of every declared body type, from particle_mass, block or radius -> list.
def _body_volumes(spec, dim):
    """(name, volume) for every declared body type: from `particle_mass`, `block`, or `radius`.

    The order matches models/entities.py, which is the only place the answer is authoritative.
    """
    out = []
    sets = spec.get("sets") or {}
    # only the sets an MPM operator acts on: every contained set declares `per_parent`, but an intermediate set of piece centres touched by no solver is not a body (it would be reported "0.01 particles per cell, UNDER-SAMPLED")
    _mpm = {str(o.get("at")) for o in (spec.get("operators") or [])
            if isinstance(o, dict) and str(o.get("op", "")).startswith(("mpm_", "p2g", "g2p",
                                                                       "mls_mpm"))}
    for sname, sv in sets.items():
        if not isinstance(sv, dict) or "per_parent" not in sv:
            continue
        if _mpm and sname not in _mpm:
            continue
        parent = sets.get(sv.get("parent"), {}) or {}
        types = parent.get("types") or {}
        rad_default = float(sv.get("radius", parent.get("radius", 0.05)) or 0.05)
        # how many bodies there are: for a contained parent that is `per_parent`, not `n` (reading 1 warned all nine compartment types of a three-level model as ABSENT)
        n_par = int(_set_size(sets, sv.get("parent")))
        # the mass-derived body: `models/entities.py` sizes a body from `particle_mass` (V = per_parent * mass / density), else `block`, else `radius`; skipping the first read `radius` (a fraction) as a length and reported si_gate at 0.0 particles per cell
        _pm = sv.get("particle_mass")
        _rho = float(sv.get("density", 1.0) or 1.0)
        _pp = sv.get("per_parent", 0)
        _v_mass = ((_per_parent_mean(sets, sv) * float(_pm) / _rho)
                   if _pm and _rho and not isinstance(_pp, dict) else None)
        if not types:
            v = _v_mass if _v_mass else (math.pi * rad_default ** 2 if dim == 2
                                         else (4.0 / 3.0) * math.pi * rad_default ** 3)
            out.append((sname, v, int(round(_per_parent_mean(sets, sv))), n_par))
            continue
        # the engine's exact allocation: the LAST type gets the remainder, so applying round() to every type reported `snow` absent
        _names = list(types)
        # `count:` is exact and not rounded (as in `_assign_types`), so the round-to-zero warning cannot fire on it
        _counted = all("count" in types[_tn] for _tn in _names)
        _alloc, _start = {}, 0
        for _i, _tn in enumerate(_names):
            if _counted:
                _k = int(types[_tn]["count"])
            else:
                _k = ((n_par - _start) if _i == len(_names) - 1
                      else int(round(float(types[_tn].get("fraction", 1.0 / len(_names))) * n_par)))
            _alloc[_tn] = max(_k, 0); _start += _k
        for tn, t in types.items():
            b = t.get("block")
            if b:
                v = 1.0
                for k in range(dim):
                    v *= abs(float(b[dim + k]) - float(b[k]))
            elif _v_mass:
                v = _v_mass                       # sized by particle_mass, as entities.py does
            else:
                r = float(t.get("radius", rad_default))
                v = math.pi * r ** 2 if dim == 2 else (4.0 / 3.0) * math.pi * r ** 3
            # not scaled by `fraction`: it picks how many PARENT CELLS take this type, and each gets the FULL `per_parent` (scaling reported 3.85 p/cell for a body with 38.5). A type that rounds to zero cells is an ABSENT body, warned about separately below
            frac = float(t.get("fraction", 1.0 / max(len(types), 1)))
            if _alloc.get(tn, 0) == 0 and n_par > 0:
                warn(f"{(spec.get('general') or {}).get('name', '?')}: type {tn!r} has "
                     f"fraction {frac} of {n_par} parent cells -> round({frac * n_par:.2f}) = ZERO "
                     f"cells, so this material is ABSENT from the run. Fractions quantise to "
                     f"1/{n_par} here; use a multiple of {1.0 / n_par:.3f} or raise the parent "
                     f"count. (The LAST type absorbs whatever the others round away, so it gets "
                     f"more cells than its fraction asks for.)")
                continue
            # per type when `per_parent` is a mapping: this compartment's own point budget, not an average that describes none of them
            _ppt = int(_pp[tn]) if isinstance(_pp, dict) else int(_pp)
            out.append((f"{sname}/{tn} x{_alloc[tn]}", v, _ppt, n_par))
    return out


# particles_per_grid_cell: Warn when a spec's grid and particle count disagree about how finely it is sampled.
def particles_per_grid_cell(yaml_path: str) -> list:
    """Warn when a spec's grid and particle count disagree about how finely it is sampled."""
    try:
        spec = yaml.safe_load(open(yaml_path))
    except Exception:
        return []
    if not isinstance(spec, dict):
        return []
    dim = int((spec.get("general") or {}).get("dim", 2))
    n_grid = next((int(fc["n_grid"]) for fc in (spec.get("fields") or {}).values()
                   if isinstance(fc, dict) and "n_grid" in fc), None)
    if n_grid is None:
        return []
    target = PPC_TARGET if dim == 3 else 4.0
    name = (spec.get("general") or {}).get("name", os.path.basename(yaml_path))
    rows = []
    _dupes = []
    # deduped: 27 identical blobs would emit 27 identical warnings
    seen = {}
    # cells from the cell size, not n_grid^dim (they agree only on a unit box: a 1e-3 m^3 body on a 0.1 m box holds 884,736 cells, and `vol * n_grid**dim` reports 884)
    _dx_ppc = _cell_size(spec, n_grid, dim)
    for label, vol, n_per_parent, _n_par in _body_volumes(spec, dim):
        cells = vol / (_dx_ppc ** dim)
        if cells <= 0:
            continue
        ppc = n_per_parent / cells
        rows.append((label, ppc, cells, n_per_parent))
        key = (round(ppc, 3), round(cells))
        if key in seen:
            seen[key][1] += 1
            continue
        seen[key] = [label, 1]
        if ppc < target * PPC_FLOOR:
            # the suggested `n_grid` for the target: n_grid = W * (N / (target*vol))^(1/dim), with the world width W in it (dropping it is right only on a unit box: si_gate at W = 0.1 m suggested 368 instead of 37)
            _W = float((spec.get("general") or {}).get("world", [1.0])[0] or 1.0)
            want_grid = max(1, int(round(_W * (n_per_parent / (target * vol)) ** (1.0 / dim))))
            want_n = int(target * cells)
            warn(f"{name}: {label} has {ppc:.2f} particles per grid cell "
                 f"({n_per_parent:,.0f} particles over {cells:,.0f} cells at n_grid={n_grid}) -- "
                 f"UNDER-SAMPLED, MPM wants ~{target:.0f}. The body will lose stiffness and may "
                 f"fracture numerically. Either n_grid={want_grid} or "
                 f"{want_n:,} particles for that body.")
            _dupes.append(key)
        elif ppc > target * 16:
            print(f"[grid-ppc] {name}: {label} has {ppc:.0f} particles per cell "
                  f"(~{target:.0f} is the convention) -- the grid cannot resolve detail the "
                  f"particles carry, so this is cost without fidelity; n_grid="
                  f"{int(round((n_per_parent / (target * vol)) ** (1.0 / dim)))} would use them.",
                  flush=True)
    for k in _dupes:                      # "... and 26 more just like it", once
        if seen.get(k, [None, 1])[1] > 1:
            print(f"[grid-ppc] {name}: ... and {seen[k][1] - 1} further bodies identical to "
                  f"{seen[k][0]}.", flush=True)
    return rows
