"""exp29 adhesion rosette arrays: one spec per swept parameter of the three pieces of cell-cell adhesion (Cedric, 2026-10-31).

Each spec is 3 x 3 = nine ARRAYS in the x-z plane; an array is ONE rounded inner cell (type inner, blue) inside a ring of SEVEN squarish outer cells (type outer, red), all in one plane. The
inner cell grows (cell_grow / cell_populate on the inner cells only) and presses on the ring; the ring is held together by the junction trio (contact_finder finds the contacts,
bond_binding binds each junction toward the cells' adhesion maximum, bond_springs puts tethers between bonded points). ONE parameter is swept per step, one value per array, printed
over each array's inner cell. The outer cells' Young's modulus is a sweep too (the outer type is then split into nine types, one per array).

  python tools/exp29_rosette.py list
  python tools/exp29_rosette.py write [--only adh kappa ...]   # -> config/tissue/mpm_rosette_<param>.yaml
"""
import argparse, os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT = "config/tissue/seeds/rosette_adh010_t75.npz"   # the relaxed ring at t = 75 min of the first adhesion sweep, array 2 (adhesion 0.1): the seed of every array
GROW = 0.0075                       # the inner cell's growth rate (mass grows e^{3 r t}); outer cells 0. 0.0025 until 2026-10-31 (Cedric: not enough to break the outer ring); x3
RESERVE = 2000                      # dormant point slots per cell for the inner cell to wake: up to (125 + 2000) / 125 = 17 x its mass (900, 8.2 x, would stop a x3 rate at frame 93)
BASE = dict(adh=1.0, kappa=3.0, k_off=0.01, k_on=1.0, sigma_beta=1.0, zeta=0.05, r_eq=0.19, bond_reach=0.38, youngs=20.0)

# one entry per swept parameter: the piece of adhesion it belongs to, the operator parameter, nine values (log or linear spaced), how it is laid on the arrays
SWEEPS = {
    "adh":        dict(piece="bond_binding", what="adhesion maximum rho* (LL, bound density)", values=[0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0], how="adhesion"),
    "kappa":      dict(piece="bond_springs", what="bond stiffness (bond_stiffness, the tether spring constant)", values=[0.3, 0.6, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0], how="block:kappa:bond_stiffness_block@bond_springs"),
    "k_off":      dict(piece="bond_binding", what="unbinding rate (unbinding_rate)", values=[0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0], how="block:koff:unbinding_rate_block@bond_binding"),
    "sigma_beta": dict(piece="bond_binding", what="bond strength (bond_strength, the Bell force scale)", values=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0], how="block:sigb:bond_strength_block@bond_binding"),
    "k_on":       dict(piece="bond_binding", what="binding rate (binding_rate)", values=[0.03, 0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0], how="block:kon:binding_rate_block@bond_binding"),
    "bond_reach": dict(piece="bond_springs", what="bond range (bond_range, the longest bonded distance)", values=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7], how="block:reach:bond_range_block@bond_springs"),
    "zeta":       dict(piece="bond_springs", what="bond damping (bond_damping)", values=[0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0], how="block:zeta:bond_damping_block@bond_springs"),
    "r_eq":       dict(piece="bond_springs", what="bond rest length (bond_rest_length)", values=[0.0, 0.05, 0.1, 0.15, 0.19, 0.25, 0.3, 0.4, 0.5], how="block:req:bond_rest_length_block@bond_springs"),
    "youngs":     dict(piece="the outer cells", what="outer cell stiffness (Young's modulus)", values=[5.0, 10.0, 20.0, 40.0, 60.0, 100.0, 150.0, 250.0, 400.0], how="types"),
}


SNAP_POINTS, SNAP_H = 282, 0.2       # points per cell and point spacing of the relaxed ring in SNAPSHOT (a cell's volume is 282 x 0.2^3 = 2.256 length^3)
LENGTH_KEYS = ("r_eq", "bond_reach")  # the sweeps and base values that are lengths tied to the point lattice: they scale with its spacing


def lattice(points, vol=None):
    """The point lattice of a rosette run with `points` points per cell at the start (None = the snapshot's own 282): its spacing s = (cell volume / points)^(1/3) and its ratio f = s / 0.2 to the lattice
    every length of the adhesion pieces was set for (bond_rest_length 0.19, bond_range 0.38 and the two spacing parameters read the junction area count x s^2)."""
    if points is None:
        return SNAP_H, 1.0
    s = ((vol or SNAP_POINTS * SNAP_H ** 3) / points) ** (1.0 / 3.0)
    return s, s / SNAP_H


def scaled(param, points=None, vol=None):
    """(sweep entry, base values, lattice spacing) with the lengths of the point lattice (bond_rest_length, bond_range) rescaled to `points` per cell: 'the same test on a finer lattice' keeps every length in
    units of the point spacing (Cedric, 2026-11-01: redo the adhesion tests at 1000 points per cell)."""
    s, f = lattice(points, vol)
    sw = dict(SWEEPS[param]); p = dict(BASE)
    if points is not None:
        for k in LENGTH_KEYS:
            p[k] = round(p[k] * f, 3)
        if param in LENGTH_KEYS:
            sw["values"] = [round(v * f, 3) for v in sw["values"]]
    return sw, p, s


def refined_snapshot(points):
    """SNAPSHOT with `points` points in every cell (config/tissue/seeds/rosette_adh010_t75_<points>.npz): the same relaxed ring on a regular lattice of spacing s = (2.256 / points)^(1/3), each cell the `points`
    lattice sites nearest to ITS old points among the sites whose nearest old point is its own (so the cells touch and never overlap), a small jitter on top; an even spread, which random points are not."""
    import numpy as np
    from scipy.spatial import cKDTree
    z = np.load(os.path.join(ROOT, SNAPSHOT), allow_pickle=True); rel, co = z["rel_pos"].astype(np.float64), z["cell_of"]
    s, _ = lattice(points); rng = np.random.default_rng(1)
    lo, hi = rel.min(0) - 0.8, rel.max(0) + 0.8
    g = [np.arange(lo[a], hi[a], s) + rng.uniform(0, s) for a in range(3)]
    sites = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
    d, j = cKDTree(rel).query(sites); own = co[j]
    out, cell_of = [], []
    for k in range(int(co.max()) + 1):
        idx = np.flatnonzero(own == k); idx = idx[np.argsort(d[idx], kind="stable")][:points]
        if len(idx) < points:
            raise ValueError(f"refined_snapshot: cell {k} has {len(idx)} candidate sites, {points} wanted")
        out.append(sites[idx] + rng.uniform(-0.15 * s, 0.15 * s, (points, 3))); cell_of.append(np.full(points, k))
    path = os.path.join(ROOT, "config", "tissue", "seeds", f"rosette_adh010_t75_{points}.npz")
    np.savez_compressed(path, rel_pos=np.concatenate(out).astype(np.float32), cell_of=np.concatenate(cell_of).astype(np.int16), source=f"{SNAPSHOT} refined to {points} points per cell, spacing {s:.4f}")
    allp = np.concatenate(out); dd = cKDTree(allp).query(allp, k=2)[0][:, 1]
    print(f"wrote {path}: {points} points in each of {len(out)} cells, spacing {s:.4f}, nearest-neighbour distance median {np.median(dd):.4f} min {dd.min():.4f}")
    return os.path.relpath(path, ROOT)


def build(param, divide=True, points=None, values=None, vol=None):
    sw, p, sp_ = scaled(param, points, vol); sw = dict(sw, values=list(values) if values is not None else sw["values"]); vals = sw["values"]; A = len(vals)
    reach_max = max(vals) if param == "bond_reach" else p["bond_reach"]
    types_out = ({f"outer_{a}": {"count": 7, "shape": "ball", "material": "viscoelastic", "youngs": float(vals[a]), "tau": 2.0, "density": 1.0} for a in range(A)} if param == "youngs"
                 else {"outer": {"count": 7 * A, "shape": "ball", "material": "viscoelastic", "youngs": 20.0, "tau": 2.0, "density": 1.0}})
    types = {"inner": {"count": A, "shape": "ball", "material": "viscoelastic", "youngs": 20.0, "tau": 2.0, "density": 1.0}, **types_out}
    T = len(types)
    blocks = {"pos": {"width": 3, "role": "coordinate", "integration": "coordinate", "boundary": "world"},
              "vel": {"width": 3, "role": "rate", "integration": "rate", "record": False},
              "pol": {"width": 3}, "adh": {"width": 6}, "arr": {"width": 1}, "ij": {"width": 2}, "tone": {"width": 1}, "lab": {"width": 1},
              "mass": {"width": 1}, "mass0": {"width": 1}, "born": {"width": 1}, "div_ready": {"width": 1}, "grow": {"width": 1}}
    kin = {"binding_rate": 1.0, "unbinding_rate": 0.01, "bond_strength": 1.0}
    adh_op = {"op": "bond_springs", "at": "pt", "junctions": "junction", "bond_stiffness": p["kappa"], "bond_damping": p["zeta"], "spacing": round(sp_, 4), "bond_rest_length": p["r_eq"], "bond_range": reach_max}
    seed = {"op": "seed_positions", "model": "rosette", "at": "pt", "cell_set": "cell", "centre": [17.0, 5.0, 17.0], "arrays": [3, 3], "array_spacing": [8.0, 8.0], "side": 1.0,
            "point_spacing": round(sp_, 4), "inner_spacing": 0.2, "n_outer": 7, "live_per_cell": points or 125, "snapshot": SNAPSHOT if points is None else (refined_snapshot(points) if vol is None else SNAPSHOT),
            "adhesion": [{"LL": float(vals[a]) if param == "adh" else p["adh"]} for a in range(A)], "labels": [float(v) for v in vals],
            "inner_values": {"grow": GROW}, "outer_values": {"grow": 0.0}}
    array_values, block_opt = {}, None
    if sw["how"].startswith("block:"):                                           # "block:<cell block>:<operator option>@<operator>": the swept value lives in a per-cell block the operator reads
        _, blk, rest = sw["how"].split(":")
        opt, opname = rest.split("@")
        blocks[blk] = {"width": 1}
        array_values[blk] = [float(v) for v in vals]
        block_opt = (opname, opt, blk)
        if opname == "bond_springs":
            adh_op[opt] = blk
    if array_values:
        seed["array_values"] = array_values
    jk = {"op": "bond_binding", "at": "junction", "cells": "cell", "implementation": "immediate", **kin, "bell": "bond", "min_area": 0.1, "type_matrix": [[1.0] * T for _ in range(T)]}
    if block_opt and block_opt[0] == "bond_binding":
        jk[block_opt[1]] = block_opt[2]
    ops = [
        {"op": "contact_finder", "at": "junction", "points": "pt", "cells": "cell", "spacing": round(sp_, 4), "bond_range": reach_max, "implementation": "hashgrid"}, jk,
        {"op": "pair_potential", "model": "cell_contact", "at": "pt", "cells": "cell", "repulsion_strength": 600.0, "contact_slack": 0.3, "contact_surface_tension": [[0.0] * T for _ in range(T)], "contact_edge_width": 0.2},
        adh_op,
        {"op": "mpm_strain", "at": "pt", "implementation": "polar"},
        {"op": "mpm_scatter", "at": "pt", "to": "mpm_grid", "drag": 1.0, "implementation": "warp_pp"},
        {"op": "mpm_grid_update", "at": "mpm_grid", "implementation": "warp_pp_balanced", "cell_surface_tension": 0.15, "csf_rho": 1.0, "csf_band": 0.1, "csf_smooth": 1, "tension_on": "all"},
        {"op": "mpm_gather", "at": "pt", "from": "mpm_grid", "implementation": "warp_pp"},
        {"op": "aggregate_centroid", "at": "cell", "child": "pt"},
        {"op": "cell_grow", "model": "mpm", "at": "cell", "points": "pt", "rate": 0.0, "rate_block": "grow", "dims": 3.0, "rho": 1.0, "after_frame": 0},
        {"op": "cell_populate", "model": "mpm", "at": "cell", "points": "pt", "seat": "fill", "jitter": 0.3, "swell": True, "seed": 1},
    ]
    sched = ["contact_finder", "bond_binding",
             {"steps": ["pair_potential", "bond_springs", "mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"], "substep_dt": 0.01, "capture": False},
             "aggregate_centroid", "cell_grow", "cell_populate"]
    grow_txt = (f"cell_grow on the inner type only, the outer cells keep their size, mass doubles every {DOUBLING:g} frames = {DOUBLING:g} min, and cell_divide cuts a blue cell at twice its birth mass "
                f"along the layer (division mass spread {DIV_CV:g}, daughter gap {DIV_GAP:g} units)" if divide else
                f"cell_grow on the inner type only, the outer cells keep their size, mass x{2.718281828 ** (3 * GROW * 240):.0f} over 240 frames if it had room, x{(125 + RESERVE) / 125:.0f} at most with its point reserve")
    title = (f"Nine arrays (3 x 3), each started from the SAME relaxed geometry (the ring of the first adhesion sweep, array 2, at t = 75 min: seven red OUTER cells touching one another round one blue INNER cell, "
             f"all in one plane, every cell {points or 282} points); only the blue inner cell grows ({grow_txt}) and presses on the ring, which the junction trio holds together (contact_finder finds the contacts, "
             f"bond_binding binds each junction toward the adhesion maximum, bond_springs tethers bonded points). SWEPT in this step, one value per array printed over the inner cell: "
             f"{sw['piece']}: {sw['what']}, {vals[0]:g} to {vals[-1]:g}; the others at adh {p['adh']:g}, bond_stiffness {p['kappa']:g}, binding_rate {p['k_on']:g}, unbinding_rate {p['k_off']:g}, bond_strength {p['sigma_beta']:g}, "
             f"bond_damping {p['zeta']:g}, bond_rest_length {p['r_eq']:g}, bond_range {p['bond_reach']:g}, outer Young's modulus 20")
    spec = {
        "general": {"name": f"mpm_rosette_{param}" if points is None else f"mpm_rosette{points}_{param}", "headline": f"adhesion rosettes, sweep of {sw['what']}", "title": title, "seed": 1, "n_frames": 240, "dt": 1.0, "record_cap": 121,
                    "boundary": "free", "dim": 3, "world": [34.0, 10.0, 34.0], "units": {"length_um": 10.0, "time_s": 60.0}},
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": 15, "per_parent": "cell", "cell_box": 3.0}},
        "sets": {"cell": {"n": A * 8, "type_layout": "ordered", "types": types, "state": blocks},
                 "pt": {"entity": "mpm_particle", "parent": "cell", "per_parent": points or 125, "grow_reserve": RESERVE, "density": 1.0, "radius": 0.5,
                        "state": {"pos": {"width": 3, "role": "coordinate", "integration": "coordinate", "boundary": "world"},
                                  "vel": {"width": 3, "role": "rate", "integration": "rate", "record": False},
                                  "pid": {"width": 1}, "cellc": {"width": 1}, "face": {"width": 1}}},
                 "junction": {"n": 0, "buffer": 4096, "state": {"ca": {"width": 1, "integration": "none"}, "cb": {"width": 1, "integration": "none"}, "fg": {"width": 1, "integration": "none"},
                                                                  "rho": {"width": 1, "integration": "first_order"}, "area": {"width": 1, "integration": "none"}, "load": {"width": 1, "integration": "none"}}}},
        "operators": ops, "schedule": sched, "seed": [seed],
        "plotting": {"renderer": "vtk_points", "subject": "pt", "cell_colours": "type", "colors": {k: ("#3b6fd4" if k == "inner" else "#e63946") for k in types}, "type_opacity": {"inner": INNER_OPACITY}, "cell_labels": {"block": "lab", "value_only": True, "font": 14},
                     "junction_panel": True, "render_3d": "contour",
                     "contour_bodies": {"mode": "partition", "iso": 0.55, "sigma": 2.0, "reach": 0.25, "smooth_method": "taubin", "smooth": 40, "pass_band": 0.02, "coat": 0.0,
                                        "color_by": {"block": "cellc", "channel": 0, "cmap": "YlOrBr", "range": [0.0, 19.0], "floor": 0.15}},
                     "surface_opacity": 1.0, "surface_roughness": 0.5, "surface_diffuse": 0.9, "surface_ambient": 0.25, "background": "#8d99a6", "foreground": "white", "box_frame": False,
                     "camera": {"elev": 5, "azim": -90}, "zoom": 1.0, "duration_s": 10, "slow_motion": 1},
    }
    if divide:
        _divide_arrays(spec, points)
    if points is not None:
        spec["sets"]["junction"]["buffer"] = 65536                 # contacts scale with the points on a cell's surface: 16 x the 4096 of 282-point cells
        spec["plotting"]["point_volume"] = round(SNAP_POINTS * SNAP_H ** 3 / points, 6)
        spec["plotting"].pop("type_opacity", None)                  # the 1000-point sweeps draw the blue cells solid (Cedric, 2026-11-01: "no transparency on the blue cells"); the 282-point ones used INNER_OPACITY
    return spec


def _divide_arrays(spec, points=None):
    """The 3 x 3 sweeps with the inner cells GROWING AND DIVIDING (version 2 of the one-rosette test, which Cedric chose, 2026-10-31): sheet in the xy plane (normal z), the camera of movie 1, 350 frames."""
    import math
    spec["general"].update(n_frames=350, world=[34.0, 34.0, 10.0])
    seed = spec["seed"][0]
    seed.update(plane="xy", centre=[17.0, 5.0, 17.0], inner_values={"grow": math.log(2.0) / (3.0 * DOUBLING)})
    spec["sets"]["cell"]["buffer"] = 200                              # cell slots for the daughters (9 arrays x up to ~10 blue cells + 7 red)
    spec["sets"]["pt"]["grow_reserve"] = 200
    if points is not None:
        spec["sets"]["cell"]["buffer"] = 300                          # the 282-point sweeps stopped at exactly 200 cells, the cap; 63 red + 9 arrays x up to ~20 blue is ~250
        spec["sets"]["pt"]["grow_reserve"] = 800                      # dormant point slots per cell slot, pooled: 300 x (points + 800) is 540,000 at 1000 points, ~2x the ~250 cells x 1.3 x points the run ends with
    i = next(k for k, o in enumerate(spec["operators"]) if o["op"] == "cell_grow")
    spec["operators"].insert(i, {"op": "cell_cycle", "model": "mpm", "at": "cell", "rule": "doubler_mass", "factor": 2.0, "factor_cv": DIV_CV, "mark": "div_ready", "born": "born", "seed": 1})
    spec["operators"].append({"op": "cell_divide", "model": "cleave_tangent", "at": "cell", "points": "pt", "yolk_weight": 0.0, "p_div": 1.0, "max_per_call": 9, "mark": "div_ready", "daughter_mark": "born",
                              "split": "median", "gap": DIV_GAP, "centre": [17.0, 17.0, -1000.0], "seed": 1})
    sched = spec["schedule"]
    sched.insert(sched.index("cell_grow"), "cell_cycle"); sched.append("cell_divide")
    spec["plotting"]["camera"] = {"elev": 55, "azim": -60}; spec["plotting"]["zoom"] = 1.0


INNER_OPACITY = 0.1                  # opacity of the blue inner cells in the movies (1 = solid): the red cells' shape, pushed by the growing blue ones, is what the movie is about (Cedric, 2026-11-01)
DIV_CV = 0.3                         # spread (a fraction of 2.0) of the mass at which an inner cell divides: 0.1 made all four blue cells divide in one frame and the crowded ring threw them out (the one-rosette test, 2026-10-31)
DIV_GAP = 0.3                        # the gap (length units) the two daughters' points are pulled apart by at a division; 0.1 overlapped them with their neighbours
DOUBLING = 80.0                      # frames (1 frame = 1 min) for the inner cell's mass to double; the inner cells then divide at twice their birth mass (cell_cycle[mpm] doubler_mass) instead of growing past their 3.0-unit grid box


def build_divide_test(name="mpm_rosette_divide_test2", factor_cv=0.3, gap=0.3):
    """ONE rosette (1 inner cell + 7 outer, 8 cells) whose inner cell grows at a doubling time of DOUBLING frames and DIVIDES at twice its birth mass, the spindle in the layer (cell_divide[cleave_tangent], centre far below
    the sheet so its normal is the layer's up axis y). The swell-in-place growth of the 3 x 3 sweeps ended with every blue cell packed against the 3.0-unit wall of its own grid (2.2 units across, 11 x the density it began with)."""
    import math
    spec = build("adh", divide=False)
    spec["general"].update(name=name, headline="adhesion rosette, the inner cell grows and divides",
                           title=f"One rosette: one blue inner cell inside seven red outer cells, started from the relaxed ring (282 points per cell); only the blue cell grows (mass doubles every {DOUBLING:g} frames = {DOUBLING:g} min) "
                                 f"and divides at twice its birth mass, the spindle along the layer, so each blue cell stays within its 3.0-unit grid box; all adhesion parameters at their defaults")
    seed = spec["seed"][0]
    seed.update(arrays=[1, 1], adhesion=[{"LL": BASE["adh"]}], labels=[BASE["adh"]], inner_values={"grow": math.log(2.0) / (3.0 * DOUBLING)})
    seed.pop("array_values", None)
    spec["sets"]["cell"].update(n=8, buffer=40)
    spec["sets"]["cell"]["types"]["inner"]["count"] = 1; spec["sets"]["cell"]["types"]["outer"]["count"] = 7
    spec["sets"]["pt"]["grow_reserve"] = 300
    i = next(k for k, o in enumerate(spec["operators"]) if o["op"] == "cell_grow")
    spec["operators"].insert(i, {"op": "cell_cycle", "model": "mpm", "at": "cell", "rule": "doubler_mass", "factor": 2.0, "factor_cv": factor_cv, "mark": "div_ready", "born": "born", "seed": 1})
    spec["operators"].append({"op": "cell_divide", "model": "cleave_tangent", "at": "cell", "points": "pt", "yolk_weight": 0.0, "p_div": 1.0, "max_per_call": 4, "mark": "div_ready", "daughter_mark": "born",
                              "split": "median", "gap": gap, "centre": [17.0, -1000.0, 17.0], "seed": 1})
    sched = spec["schedule"]
    sched.insert(sched.index("cell_grow"), "cell_cycle"); sched.append("cell_divide")
    # the sheet lies in the xy plane with its normal along z and the camera is that of movie 1 (exp29 step 0057, mpm_grid3x3_cycles: world 20^3, elev 55, azim -60, zoom 1.0); the array sits 2 units from the
    # world's middle toward the camera (horizontal direction (cos -60, sin -60)), which is DOWN the picture
    spec["general"]["world"] = [20.0, 20.0, 20.0]
    seed.update(plane="xy", centre=[10.0 + 2.0 * math.cos(math.radians(-60.0)), 10.0, 10.0 + 2.0 * math.sin(math.radians(-60.0))])
    spec["operators"][-1]["centre"] = [10.0, 10.0, -1000.0]
    spec["plotting"]["camera"] = {"elev": 55, "azim": -60}; spec["plotting"]["zoom"] = 1.0
    spec["plotting"].pop("cell_labels", None)                       # movie 1 prints no value over its cells, and one rosette has no value to tell apart
    return spec


NUCLEUS_STIFFNESS = 4.0               # Young's modulus of a cell's nucleus over its cytoplasm's (exp21's ratio; Guilak et al. 2000 measure a nucleus several times stiffer)
NUCLEUS_RADIUS_FRAC = 0.8             # nucleus radius as a fraction of the cell's rms point radius (0.8 x 0.775 = 0.62 of the cell's outer radius, a quarter of its volume)


def build_sphere_twin(name="mpm_sphere_twin", n_red=20, points=125, nucleus=False, buffer=120):
    """The twin of the one-rosette test (movie 24) in THREE dimensions: ONE blue inner cell at the centre of a closed one-cell-thick sphere of `n_red` red cells (seed_positions[vesicle], the blue cell its `core`), the blue cell
    growing (mass doubles every DOUBLING frames) and dividing (cell_divide[cleave_typed], the longest axis, there is no layer to cut across inside a lumen); the red cells neither grow nor divide. Same adhesion
    parameters, division settings, camera and blue opacity as the rosettes. `points` = material points per cell at the start (125 = movie 34; 1000 = its eight-fold denser twin, Cedric 2026-11-01): the cell keeps its 1.0 unit^3 volume, so
    each point carries 1/points of it, the seed lattice is laid at 0.625/points^(1/3) (a 4x oversample of the points kept, as 0.125 is for 125) and the reserve of dormant slots scales with it. `nucleus`: every cell starts with one nucleus, the points within NUCLEUS_RADIUS_FRAC of its centre made NUCLEUS_STIFFNESS times stiffer (set_material[nucleus], once at x_0,
    marked in the point block `nuc`) and drawn as a dark body inside the cell; `buffer` = the cell slots allocated (each carries per_parent + grow_reserve point slots; movie 34's run ends at 36 cells, so 64 is ample for the 4000-point twin whose 120 slots would be 1.2 million point slots); cell_grow adds ordinary points and a division hands each daughter the nuclear points on its side of the cut."""
    import math
    spec = build("adh", divide=False)
    spec["sets"]["pt"]["per_parent"] = points
    spec["general"].update(name=name, headline="a sphere of red cells round one blue cell that grows and divides", n_frames=350, world=[20.0, 20.0, 20.0],
                           title=f"One blue inner cell at the centre of a closed sphere of {n_red} red cells (one cell thick), every cell {points} points; only the blue cell grows (mass doubles every {DOUBLING:g} frames = {DOUBLING:g} min) "
                                 f"and divides at twice its birth mass along its longest axis (division mass spread {DIV_CV:g}, daughter gap {DIV_GAP:g} units) and presses on the shell, which the junction trio holds together "
                                 f"(contact_finder, bond_binding, bond_springs at their defaults); left panel the whole sphere, right panel the same frame cut through the middle (the cells on the camera's side hidden)")
    spec["sets"]["cell"].update(n=1 + n_red, buffer=buffer)
    spec["sets"]["cell"]["types"]["inner"]["count"] = 1; spec["sets"]["cell"]["types"]["outer"]["count"] = n_red
    spec["sets"]["pt"]["grow_reserve"] = round(200 * points / 125)
    spec["sets"]["junction"]["buffer"] = max(4096, round(4096 * points / 125 / 2))
    h = round(0.625 / points ** (1.0 / 3.0), 4)
    g = math.log(2.0) / (3.0 * DOUBLING)
    c = [10.0, 10.0, 10.0]
    spec["seed"] = [{"op": "seed_positions", "model": "vesicle", "at": "pt", "cell_set": "cell", "centre": c, "spacing": 1.05, "inner_extent": 0.5, "outer_extent": 0.5, "point_spacing": h,
                     "live_per_cell": points, "adh": BASE["adh"], "seed": 1,
                     "vesicles": [{"centre": c, "core": True, "core_radius": 0.62, "polar": False, "values": {"grow": g}}, {"centre": c, "polar": False}]}]
    i = next(k for k, o in enumerate(spec["operators"]) if o["op"] == "cell_grow")
    spec["operators"].insert(i, {"op": "cell_cycle", "model": "mpm", "at": "cell", "rule": "doubler_mass", "factor": 2.0, "factor_cv": DIV_CV, "mark": "div_ready", "born": "born", "seed": 1})
    spec["operators"].append({"op": "cell_divide", "model": "cleave_typed", "at": "cell", "points": "pt", "yolk_weight": 0.0, "p_div": 1.0, "max_per_call": 4, "mark": "div_ready", "daughter_mark": "born",
                              "gap": DIV_GAP, "seed": 1})
    sched = spec["schedule"]
    sched.insert(sched.index("cell_grow"), "cell_cycle"); sched.append("cell_divide")
    pl = spec["plotting"]
    if nucleus:
        spec["sets"]["pt"]["state"]["nuc"] = {"width": 1}
        spec["seed"].append({"op": "set_material", "model": "nucleus", "at": "pt", "stiffness": NUCLEUS_STIFFNESS, "radius_frac": NUCLEUS_RADIUS_FRAC, "label": "nuc"})
        pl["contour_bodies"] = {"nucleus": "nuc", "nucleus_color": "#1b1464"}
        spec["general"]["title"] += (f"; every cell has one nucleus at the start (the points within {NUCLEUS_RADIUS_FRAC:g} of the cell's rms point radius of its centre, {NUCLEUS_STIFFNESS:g}x stiffer, set_material[nucleus]), drawn dark")
    pl["camera"] = {"elev": 55, "azim": -60}; pl["zoom"] = 1.0
    pl.pop("cell_labels", None)
    pl["surface_iso_frac"] = 0.3               # the contour level: 0.5 (the rosette movies') broke the shell into shards once its cells stretched over the growing sphere; 0.3 keeps it closed (Cedric chose it from the montage, 2026-11-01)
    pl["point_volume"] = round(1.0 / points, 6)  # one point's true volume (a cell's 1.0 unit^3 over its `points` points; 0.008 for 125); the studio guessed 0.125^3 from the seed's point_spacing, a 4th of it, and drew shards
    return spec


N_RED = 20                           # red cells in each sphere of the sphere sweeps
SPHERE_SPACING = 8.0                 # length units between the centres of neighbouring spheres (a sphere is about 4.4 across once its blue cell has grown)


PAIRS = {   # two parameters swept at once on the 3 x 3 arrays: array a takes the first parameter's value a % 3 (column) and the second's value a // 3 (row), the label over it "first / second"
    "adh_kappa":  dict(p1="adh", v1=[0.2, 1.0, 3.0], p2="kappa", v2=[1.0, 3.0, 10.0]),
    "kon_koff":   dict(p1="k_on", v1=[0.3, 1.0, 3.0], p2="k_off", v2=[0.003, 0.03, 0.3]),
}


def build_pair(key, points=1000):
    """One run of nine arrays on a 3 x 3 GRID OF TWO PARAMETERS (Cedric, 2026-11-02): the planar sweep of the first parameter with its values repeated down the rows, the second's per-array settings merged in.
    The label of an array prints both values, "first / second"."""
    pr = PAIRS[key]; p1, p2 = pr["p1"], pr["p2"]
    a1 = [pr["v1"][a % 3] for a in range(9)]; a2 = [pr["v2"][a // 3] for a in range(9)]
    spec = build(p1, points=points, values=a1); sb = build(p2, points=points, values=a2)
    sa_, sb_ = spec["seed"][0], sb["seed"][0]
    if p2 == "adh":
        sa_["adhesion"] = sb_["adhesion"]
    for blk, v in (sb_.get("array_values") or {}).items():
        sa_.setdefault("array_values", {})[blk] = v
        spec["sets"]["cell"]["state"][blk] = {"width": 1}
    for ob in sb["operators"]:
        for oa in spec["operators"]:
            if oa["op"] == ob["op"] and oa.get("model") == ob.get("model"):
                oa.update({k: v for k, v in ob.items() if k.endswith("_block")})
    sa_["array_values"]["lab2"] = [float(v) for v in a2]
    spec["sets"]["cell"]["state"]["lab2"] = {"width": 1}
    spec["plotting"]["cell_labels"] = {"block": "lab", "block2": "lab2", "value_only": True, "font": 14}
    w1, w2 = SWEEPS[p1]["what"], SWEEPS[p2]["what"]
    spec["general"].update(name=f"mpm_rosette{points}_{key}", headline=f"adhesion rosettes, grid of {p1} x {p2}",
                           title=(f"Nine arrays on a 3 x 3 GRID OF TWO PARAMETERS, {points} points per cell: {p1} ({w1}) takes {pr['v1']} along the columns and {p2} ({w2}) takes {pr['v2']} down the rows; "
                                  f"each array's label prints both values, first / second; everything else as the single sweeps: blue inner cell grows and divides, seven red outer cells do not"))
    return spec


def build_sphere_sweep(param, points=None):
    """The TWIN of a planar rosette sweep with spheres (Cedric, 2026-11-01): the 3 x 3 arrays are 3 x 3 SPHERES, each one blue cell at the centre of a closed shell of N_RED red cells
    (seed_positions[vesicle], the blue cell its `core`), one value of the swept parameter per sphere; the blue cell grows (mass doubles every DOUBLING frames) and divides along its longest axis.
    Every sphere has its own pair of cell types (inner_a, outer_a) so that the seed can give it its own centre; the other operators and settings are the planar sweep's."""
    import math
    sw = scaled(param, points, 1.0)[0]; vals = sw["values"]; A = len(vals)
    spec = build(param, divide=False, points=points, vol=1.0)               # a sphere cell is 1.0 unit^3: its point spacing is (1 / points)^(1/3)
    g = math.log(2.0) / (3.0 * DOUBLING)
    mk = lambda youngs: {"count": 1, "shape": "ball", "material": "viscoelastic", "youngs": youngs, "tau": 2.0, "density": 1.0}
    types = {}
    for a in range(A):
        types[f"inner_{a}"] = mk(20.0)
        types[f"outer_{a}"] = {**mk(float(vals[a]) if param == "youngs" else 20.0), "count": N_RED}
    T = len(types)
    spec["sets"]["cell"].update(n=A * (1 + N_RED), buffer=450, types=types)
    spec["sets"]["pt"]["grow_reserve"] = round(200 * (points or 125) / 125)
    for o in spec["operators"]:
        if o["op"] == "bond_binding":
            o["type_matrix"] = [[1.0] * T for _ in range(T)]
        if o["op"] == "pair_potential":
            o["contact_surface_tension"] = [[0.0] * T for _ in range(T)]
    vesicles = []
    for a in range(A):
        c = [17.0 + (a % 3 - 1) * SPHERE_SPACING, 17.0 + (a // 3 - 1) * SPHERE_SPACING, 5.0]
        common = {"arr": float(a)}
        if sw["how"].startswith("block:"):
            common[sw["how"].split(":")[1]] = float(vals[a])
        extra = {"adh_by_class": {"LL": float(vals[a])}} if param == "adh" else {}
        vesicles.append({"centre": c, "core": True, "core_radius": 0.62, "polar": False, "values": {**common, "grow": g, "lab": float(vals[a])}, **extra})
        vesicles.append({"centre": c, "polar": False, "values": {**common, "lab": float("nan")}, **extra})
    spec["seed"] = [{"op": "seed_positions", "model": "vesicle", "at": "pt", "cell_set": "cell", "centre": [17.0, 17.0, 5.0], "spacing": 1.05, "inner_extent": 0.5, "outer_extent": 0.5,
                     "point_spacing": round(0.625 / (points or 125) ** (1.0 / 3.0), 4), "live_per_cell": points or 125, "adh": BASE["adh"], "seed": 1, "vesicles": vesicles}]
    i = next(k for k, o in enumerate(spec["operators"]) if o["op"] == "cell_grow")
    spec["operators"].insert(i, {"op": "cell_cycle", "model": "mpm", "at": "cell", "rule": "doubler_mass", "factor": 2.0, "factor_cv": DIV_CV, "mark": "div_ready", "born": "born", "seed": 1})
    spec["operators"].append({"op": "cell_divide", "model": "cleave_typed", "at": "cell", "points": "pt", "yolk_weight": 0.0, "p_div": 1.0, "max_per_call": 12, "mark": "div_ready", "daughter_mark": "born",
                              "gap": DIV_GAP, "seed": 1})
    sched = spec["schedule"]
    sched.insert(sched.index("cell_grow"), "cell_cycle"); sched.append("cell_divide")
    name = f"mpm_sphere_{param}" if points is None else f"mpm_sphere{points}_{param}"
    spec["general"].update(name=name, headline=f"spheres of red cells round one blue cell that grows and divides, sweep of {sw['what']}", n_frames=350, world=[34.0, 34.0, 10.0])
    spec["general"]["title"] = (f"Nine spheres (3 x 3), each one blue inner cell at the centre of a closed one-cell-thick sphere of {N_RED} red cells ({points or 125} points per cell); only the blue cell grows (mass doubles every {DOUBLING:g} frames = "
                                f"{DOUBLING:g} min) and divides at twice its birth mass along its longest axis (division mass spread {DIV_CV:g}, daughter gap {DIV_GAP:g} units) and presses on the shell, which the junction trio holds "
                                f"together. SWEPT in this step, one value per sphere printed over its blue cell: {sw['piece']}: {sw['what']}, {vals[0]:g} to {vals[-1]:g}; the others at their defaults")
    pl = spec["plotting"]
    pl["colors"] = {k: ("#3b6fd4" if k.startswith("inner") else "#e63946") for k in types}
    if points is None:
        pl["type_opacity"] = {k: INNER_OPACITY for k in types if k.startswith("inner")}
    pl["camera"] = {"elev": 55, "azim": -60}; pl["zoom"] = 1.0
    pl["surface_iso_frac"] = 0.3; pl["point_volume"] = round(1.0 / (points or 125), 6)
    return spec


def post_pair(key, points=1000):
    """Render a finished two-parameter run (graphs_data/tissue/mpm_rosette<points>_<key>) as a new step of the exp29 record, the grid explained in its top-left text."""
    import re, subprocess, sys
    pr = PAIRS[key]; name = f"mpm_rosette{points}_{key}"
    rec = os.path.join(ROOT, "experiments", "exp29_cell_tissue_2")
    env = {**os.environ, "PYTHONPATH": f"{ROOT}/src:{ROOT}/tools", "MPLBACKEND": "Agg", "PLEXUS_BUILDER": rec}
    spec = yaml.safe_load(open(os.path.join(ROOT, "config", "tissue", f"{name}.yaml")))
    sp_path = os.path.join(ROOT, "graphs_data", "tissue", name, "spec.yaml")
    saved = yaml.safe_load(open(sp_path))
    if saved["plotting"].pop("type_opacity", None) is not None:
        yaml.safe_dump(saved, open(sp_path, "w"), sort_keys=False, width=1000)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "exp29_studio.py"), f"tissue/{name}", "--why", f"TWO-PARAMETER GRID {pr['p1']} x {pr['p2']} (Cedric, 2026-11-02): " + spec["general"]["title"]],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    m = re.findall(r"'index': (\d+)", r.stdout + r.stderr)
    if not m:
        print("render failed:", (r.stdout + r.stderr)[-600:]); return 1
    n = int(m[-1]); k = f"{n:04d}"
    tpath = os.path.join(rec, "data", "overlay_texts.yaml")
    t = yaml.safe_load(open(tpath))
    t["steps"][k] = {"lines": [f"{pr['p1']} (columns) x {pr['p2']} (rows), label = first / second", f"{pr['p1']}: {', '.join(f'{v:g}' for v in pr['v1'])};  {pr['p2']}: {', '.join(f'{v:g}' for v in pr['v2'])}",
                               f"{points} points per cell; blue cell grows and divides, red cells do not"]}
    yaml.safe_dump(t, open(tpath, "w"), sort_keys=False, width=1000)
    r2 = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "exp29_text_overlay.py"), str(n), "--from-current"], capture_output=True, text=True, env=env, cwd=ROOT)
    print(f"{key}: step {k} rendered; overlay rc {r2.returncode}")
    return r2.returncode


def snapshot(run, row, array, out):
    """The geometry of ONE array of a recorded rosette run at one recorded row, as a seed file: the points of its 8 cells relative to the inner cell's centroid (cell 0 = inner, 1..7 = the ring
    in angle order), so every array of a new run starts as that relaxed, touching ring with its grown inner cell (Cedric, 2026-10-31: 'collect the geometry of movie 23 param 2 frame 75')."""
    import numpy as np
    z = np.load(os.path.join(ROOT, "graphs_data", "tissue", run, "trajectory.npz"))
    A = 9; n_out = 7
    occ = z["pt__occ"][row] > 0.5; pid = np.rint(z["pt__pid"][row, :, 0]).astype(int); X = z["pt__pos"][row]
    slots = [array] + [A + n_out * array + j for j in range(n_out)]
    ci = X[occ & (pid == slots[0])].mean(0)
    rel, cell_of = [], []
    for k, c in enumerate(slots):
        m = occ & (pid == c); rel.append(X[m] - ci); cell_of.append(np.full(int(m.sum()), k))
    np.savez_compressed(out, rel_pos=np.concatenate(rel).astype(np.float32), cell_of=np.concatenate(cell_of).astype(np.int16), source=f"{run} row {row} array {array}")
    n = [len(r) for r in rel]
    print(f"wrote {out}: {run} row {row} array {array}: points per cell {n}; ring centroid distances {[round(float(np.linalg.norm(r.mean(0))), 3) for r in rel[1:]]}")


def post(param, sphere=False, points=None):
    """Render a finished sweep (graphs_data/tissue/mpm_rosette_<param>) into the exp29 record as a new step with exp29_studio, add its top-left text (the operator and the swept parameter, the
    range, what grows) to data/overlay_texts.yaml under the step's number, and composite it with tools/exp29_text_overlay.py (no re-render of the cells for the text)."""
    import re, subprocess, sys
    sw = scaled(param, points, 1.0 if sphere else None)[0]; vals = sw["values"]
    rec = os.path.join(ROOT, "experiments", "exp29_cell_tissue_2")
    env = {**os.environ, "PYTHONPATH": f"{ROOT}/src:{ROOT}/tools", "MPLBACKEND": "Agg", "PLEXUS_BUILDER": rec}
    pre = ("mpm_sphere" if points is None else f"mpm_sphere{points}") if sphere else ("mpm_rosette" if points is None else f"mpm_rosette{points}")
    spec = yaml.safe_load(open(os.path.join(ROOT, "config", "tissue", f"{pre}_{param}.yaml")))
    why = (f"SPHERES OF RED CELLS, SWEEP OF {sw['what'].upper()} (Cedric, 2026-11-01): " if sphere else f"ADHESION ROSETTES AT {points or 282} POINTS PER CELL, SWEEP OF {sw['what'].upper()} (Cedric, {'2026-11-01' if points else '2026-10-31'}): ") + spec["general"]["title"]
    if points is not None and not sphere:                           # the run's saved spec is what the studio reads: the blue cells solid, whatever it was started with
        sp_path = os.path.join(ROOT, "graphs_data", "tissue", f"{pre}_{param}", "spec.yaml")
        saved = yaml.safe_load(open(sp_path))
        if saved["plotting"].pop("type_opacity", None) is not None:
            yaml.safe_dump(saved, open(sp_path, "w"), sort_keys=False, width=1000)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "exp29_studio.py"), f"tissue/{pre}_{param}", "--why", why], capture_output=True, text=True, env=env, cwd=ROOT)
    m = re.findall(r"'index': (\d+)", r.stdout + r.stderr)
    if not m:
        print("render failed:", (r.stdout + r.stderr)[-600:]); return 1
    n = int(m[-1]); key = f"{n:04d}"
    tpath = os.path.join(rec, "data", "overlay_texts.yaml")
    t = yaml.safe_load(open(tpath))
    t["steps"][key] = {"lines": [f"{sw['piece']}  {sw['what']}", f"{vals[0]:g} to {vals[-1]:g}, one value per {'sphere' if sphere else 'array'} (labels)",
                                 (f"blue cell grows and divides{'' if points else f' (opacity {INNER_OPACITY:g})'} in a sphere of {N_RED} red cells; other parameters at defaults" if sphere else
                                  f"{points} points per cell; blue cell grows and divides, red cells do not; other parameters at defaults" if points else
                                  f"blue cell grows and divides (opacity {INNER_OPACITY:g}), red cells do not; other parameters at defaults")]}
    yaml.safe_dump(t, open(tpath, "w"), sort_keys=False, width=1000)
    r2 = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "exp29_text_overlay.py"), str(n), "--from-current"], capture_output=True, text=True, env=env, cwd=ROOT)
    print(f"{param}: step {key} rendered; overlay rc {r2.returncode} {(r2.stdout + r2.stderr)[-200:].strip()}")
    return r2.returncode


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["list", "write", "snapshot", "post", "divide_test", "sphere_twin", "sphere_sweeps", "pairs", "post_pair"]); ap.add_argument("--only", nargs="*"); ap.add_argument("--sphere", action="store_true"); ap.add_argument("--points", type=int, default=125); ap.add_argument("--nucleus", action="store_true"); ap.add_argument("--buffer", type=int, default=120)
    ap.add_argument("--run"); ap.add_argument("--row", type=int); ap.add_argument("--array", type=int); ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "post":
        raise SystemExit(max(post(k, a.sphere, a.points if a.points != 125 else None) for k in (a.only or SWEEPS)))
    if a.cmd == "pairs":
        for k in (a.only or PAIRS):
            out = os.path.join(ROOT, "config", "tissue", f"mpm_rosette{a.points}_{k}.yaml")
            yaml.safe_dump(build_pair(k, a.points), open(out, "w"), sort_keys=False, width=1000); print("wrote", out)
        raise SystemExit(0)
    if a.cmd == "post_pair":
        raise SystemExit(max(post_pair(k, a.points) for k in (a.only or PAIRS)))
    if a.cmd == "sphere_sweeps":
        for k in (a.only or SWEEPS):
            pts = a.points if a.points != 125 else None
            out = os.path.join(ROOT, "config", "tissue", f"mpm_sphere_{k}.yaml" if pts is None else f"mpm_sphere{pts}_{k}.yaml")
            yaml.safe_dump(build_sphere_sweep(k, pts), open(out, "w"), sort_keys=False, width=1000); print("wrote", out)
        raise SystemExit(0)
    if a.cmd == "divide_test":
        out = os.path.join(ROOT, "config", "tissue", "mpm_rosette_divide_test2.yaml")
        yaml.safe_dump(build_divide_test(), open(out, "w"), sort_keys=False, width=1000); print("wrote", out); raise SystemExit(0)
    if a.cmd == "sphere_twin":
        out = os.path.join(ROOT, "config", "tissue", "mpm_sphere_twin.yaml")
        kw = {} if a.points == 125 else {"name": f"mpm_sphere_twin_{a.points}", "points": a.points}
        if a.buffer != 120 and kw:
            kw["buffer"] = a.buffer
        if a.nucleus:
            kw.update(name=f"mpm_sphere_twin_{a.points}_nucleus", points=a.points, nucleus=True)
        if kw:
            out = os.path.join(ROOT, "config", "tissue", f"{kw['name']}.yaml")
        yaml.safe_dump(build_sphere_twin(**kw), open(out, "w"), sort_keys=False, width=1000); print("wrote", out); raise SystemExit(0)
    if a.cmd == "snapshot":
        snapshot(a.run, a.row, a.array, a.out); raise SystemExit(0)
    for k, v in SWEEPS.items():
        if a.only and k not in a.only:
            continue
        if a.cmd == "list":
            print(f"{k:11s} {v['piece']:18s} {v['what']:44s} {v['values']}")
        else:
            pts = a.points if a.points != 125 else None
            out = os.path.join(ROOT, "config", "tissue", f"mpm_rosette_{k}.yaml" if pts is None else f"mpm_rosette{pts}_{k}.yaml")
            yaml.safe_dump(build(k, points=pts), open(out, "w"), sort_keys=False, width=1000); print("wrote", out)
