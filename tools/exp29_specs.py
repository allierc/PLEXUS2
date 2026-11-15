"""exp29: write the MPM twin of a minisite clip, with its chemistry, growth and division read from the PUBLISHED spec.

    python tools/exp29_specs.py grow_divide r021_12 r005_06              # config/tissue/exp29_m_<clip>.yaml
    python tools/exp29_specs.py r021_12 --cells 100 --frames 400 --name smoke_a --tag a

The clip's chemistry (`seed_cell_chem`, `cell_chem_diffuse`, `cell_chem_react`, `cell_chem_from_shape`), its growth (`cell_grow`) and its
division (`cell_divide`) are READ from `experiments/exp29_cell_tissue_2/data/minisite_specs/<clip>.yaml` (extracted from the minisite by
`tools/exp29_minisite_specs.py`) and written into the twin with the same numbers, so a parameter is never typed twice (the one rule). What the
twin does NOT take is the vertex body: `cell_mechanics` (K_A, K_P, K_V, Gamma, Lambda, K_bend, K_lumen), `edge_flip`, `topo_record`,
`cell_geometry` and `seed_mesh` have no meaning for a body of points; the body is exp27's cell on exp27's fast operator set (`stack_b` of
exp27's performance specs), with its own material, contact and adhesion. `interface_tension` (M3) and `cell_die` (M1) belong to the
death-tension fork and raise here until they are carried.

THE CLOCK. One MPM frame is one vertex frame, and the published rates (per time unit) are the vertex twin's per frame: the chemistry and the growth
integrate the same increment per frame (`--clock`, default 1: `general.dt` 1.0, a published rate means the same per frame, an `after_frame` of the clip is
the twin's). One time unit is a minute of exp27's body (E 20, tau 2). `--clock S` compresses the BODY's time: `dt` = 1 / S and every rate per time
unit x S (the Gray-Scott rate, the diffusivities, the curvature rate, the growth rate), so each frame carries the vertex twin's increments but 1 / S of
the mechanics' time -- 500 / S substeps a frame instead of 500, the cost of a frame divided by S (the grid is solved for every cell slot of the buffer
at every substep: 12 s a frame at 1,500 slots and 500 substeps). The gap between the body's relaxation (tau 2) and the growth (a doubling in 267 frames
= 267 / S time units) is the price: 267 at S 1, 53 at S 5 (exp27's gland: 90).

THE GROWTH. `cell_grow` raises a cell's linear scale, ds/dt = s r (rho + Hill(a)), volume s^3; a cell of points adds points at its own density, so
the point-count rate is 3 r (rho + Hill(a)) (`cell_populate[mpm_points_chem]`, `dims` 3). Division at `factor` x the birth volume is a point count of
`factor` x the seed's points per cell; `cycle_cv` 0.15 of the vertex `cell_divide` is a spread of the threshold, see `TARGET_JITTER`. The cut is
at the MEDIAN of the points along the division axis (`split: median`), so the two daughters hold n // 2 and n - n // 2 points of the mother's n,
as the vertex divide gives two EQUAL daughters of half the volume.

THE POOL. Points are drawn from the dormant points of the whole set, so the pool is `buffer` x `per_parent` = 1500 x 512 = 768,000 points: a vesicle of
`cells` cells growing exponentially holds 1.44 x 512 points per cell on average (a cell grows from 512 to 1,024 and halves). The vertex twins run 950
frames from 200 cells and reach about 795 cells (growth from frame 100 at 3 x 0.000866 per frame, a doubling in 267 frames); 800 cells hold about
590,000 points, a `pool_fill` of 0.77, under the 0.98 at which `G-sanity` caps the card.
"""
import argparse
import math
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "experiments", "exp29_cell_tissue_2", "data", "minisite_specs")
GD = os.path.join(os.environ.get("PLEXUS_OUTPUT_ROOT") or os.environ.get("GNN_OUTPUT_ROOT") or ROOT, "graphs_data")   # the runs' data root (default: the repository's own graphs_data/)
PER_PARENT = 512                     # points per cell at the seed: a 10 um cube at spacing 0.125 (exp27)
POINT_SPACING = 0.125
CELL_SPACING = 1.05                  # centre spacing of exp27's cells
# A division threshold N* (1 + j u), u uniform in (-1, 1), spreads the cycle time (the time to grow from N0 to N*) by
# d t / t = j u / ln 2 at the first order, a standard deviation of j / (sqrt 3 ln 2): `cycle_cv` c of the vertex cell_divide is j = c sqrt(3) ln 2.
TARGET_JITTER = lambda cv: cv * math.sqrt(3.0) * math.log(2.0)          # noqa: E731


def published(clip):
    d = yaml.safe_load(open(os.path.join(SRC, f"{clip}.yaml")))
    ops = {}
    for o in d["operators"]:
        key = {"mesh_seed": "seed_mesh", "cell_chem_seed": "seed_cell_chem"}.get(o["op"], o["op"])
        ops.setdefault(key, []).append(o)
    return d, ops


def body_operators(centre, pair_stiffness=600.0, surface_tension=0.15, contact_tension=-0.05, face_tension=None):
    """exp27's cell on its fast operator set (`stack_b`), one cell type, no crawling, no basement membrane: the cell-cell part of the sph5 / bm2c specs."""
    return [
        {"op": "contact_finder", "at": "junction", "points": "pt", "cells": "cell", "spacing": POINT_SPACING, "bond_range": 0.3,
         "implementation": "hashgrid"},
        {"op": "bond_binding", "at": "junction", "cells": "cell", "binding_rate": 1.0, "unbinding_rate": 0.01, "bond_strength": 1.0, "bell": "bond",
         "min_area": 0.1, "type_matrix": [[1.0]]},
        {"op": "pair_potential", "model": "cell_contact", "at": "pt", "cells": "cell", "repulsion_strength": pair_stiffness,
         "contact_slack": 0.25, "contact_surface_tension": [[contact_tension]], "contact_edge_width": 0.2},
        {"op": "bond_springs", "at": "pt", "junctions": "junction", "bond_stiffness": 3.0, "bond_damping": 0.05, "spacing": POINT_SPACING, "bond_rest_length": 0.15, "bond_range": 0.3},
        {"op": "mpm_viscosity", "at": "pt", "eta": 1.0},
        {"op": "mpm_strain", "at": "pt", "implementation": "polar"},
        {"op": "mpm_scatter", "at": "pt", "to": "mpm_grid", "drag": 1.0, "implementation": "warp_pp",
         **({"colour_from_face": [1.0, float(face_tension), float(face_tension)]} if face_tension is not None else {})},
        {"op": "mpm_grid_update", "at": "mpm_grid", "implementation": "warp_pp_balanced", "cell_surface_tension": surface_tension, "csf_rho": 1.0,
         "csf_band": 0.1, "csf_smooth": 1, "tension_on": "face" if face_tension is not None else "all"},
        {"op": "mpm_gather", "at": "pt", "from": "mpm_grid", "implementation": "warp_pp"},
        {"op": "aggregate_centroid", "at": "cell", "child": "pt"},
    ]


def build(clip, name, cells=200, buffer=1500, frames=950, record_cap=201, seed=1, centre=None, world=20.0, contact=0.3,
          dump_every=30, per_parent=PER_PARENT, drop=(), grow_scale=1.0, after_frame=None, no_growth=False, clock=1.0, surface_tension=0.15, contact_tension=-0.05, rho=None, a_sw=None, tau=2.0, apical_area=0.0, face_tension=None, adh_lateral=None, adh_other=None):
    pub, P = published(clip)
    for unsupported in ("interface_tension", "cell_die"):
        if unsupported in P and unsupported not in drop:
            raise SystemExit(f"{clip}: `{unsupported}` is not carried by this fork (M3 / M1); pass --drop {unsupported} to leave it out")
    centre = list(centre or [world / 2.0] * 3)
    chem = "cell_chem_react" in P
    grow = P["cell_grow"][0]
    div = P["cell_divide"][0]
    cs = {"width": 2, "integration": "first_order"}
    cell_state = {"pos": {"width": 3, "role": "coordinate", "integration": "coordinate", "boundary": "world"},
                  "vel": {"width": 3, "role": "rate", "integration": "rate", "record": False},
                  "pol": {"width": 3}, "adh": {"width": 6}, "ready": {"width": 1}, "npts": {"width": 1}}
    if chem:
        cell_state["chem"] = cs
    spec = {
        "general": {"name": name, "seed": seed, "n_frames": frames, "dt": 1.0 / clock, "record_cap": record_cap, "boundary": "free", "dim": 3,
                    "world": [world] * 3, "units": {"length_um": 10.0, "time_s": 60.0}},
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": 24, "per_parent": "cell", "cell_box": 3.0}},
        "sets": {
            "cell": {"n": cells, "buffer": buffer, "type_layout": "ordered",
                     "types": {"epi": {"count": cells, "shape": "ball", "material": "viscoelastic", "youngs": 20.0, "tau": float(tau), "density": 1.0}},
                     "state": cell_state},
            "pt": {"entity": "mpm_particle", "parent": "cell", "per_parent": per_parent, "density": 1.0, "radius": 0.5,
                   "state": {"pos": {"width": 3, "role": "coordinate", "integration": "coordinate", "boundary": "world"},
                             "vel": {"width": 3, "role": "rate", "integration": "rate", "record": False},
                             "pid": {"width": 1}, "cellc": {"width": 1}, "tcol": {"width": 1},
                             **({"face": {"width": 1}} if face_tension is not None else {})}},
            "junction": {"n": 0, "buffer": 32768,
                         "state": {"ca": {"width": 1, "integration": "none"}, "cb": {"width": 1, "integration": "none"},
                                   "fg": {"width": 1, "integration": "none"}, "rho": {"width": 1, "integration": "first_order"},
                                   "area": {"width": 1, "integration": "none"}, "load": {"width": 1, "integration": "none"}}},
        },
    }
    ops = body_operators(centre, surface_tension=surface_tension, contact_tension=contact_tension, face_tension=face_tension)
    if apical_area:
        ops.append({"op": "active_force", "model": "footprint_spreading", "at": "pt", "cell_set": "cell", "centre": centre, "spreading_strength": float(apical_area),
                    "growth_law_exponent": 0.6667})
    # the footprint drive runs INSIDE the substep block (as `bond_springs`, another mpm_acceleration operator, does): its deficit is re-read
    # every substep, so a strong k does not overshoot the way a force held for a whole frame of 500 substeps did (probe of 2026-10-23: at k 100
    # outside the block the cells collapsed within one frame; at k 2 the drive did nothing)
    schedule = ["contact_finder", "bond_binding",
                {"steps": (["active_force"] if apical_area else []) + ["pair_potential", "bond_springs", "mpm_viscosity", "mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"],
                 "substep_dt": 0.002, "capture": False},
                "aggregate_centroid"]
    seeds = [{"op": "seed_positions", "model": "vesicle", "at": "pt", "cell_set": "cell", "centre": centre, "spacing": CELL_SPACING,
              "inner_extent": 0.5, "outer_extent": 0.5, "point_spacing": POINT_SPACING, "seed": seed,
              **({"adh_by_class": {"LL": float(adh_lateral), **{k: float(adh_other) for k in ("AA", "BB", "AB", "LA", "LB")}}}
                 if adh_lateral is not None else {})}]
    if chem:
        sc = P["seed_cell_chem"][0]
        seeds.append({"op": "seed_cell_chem", "at": "cell", "seed": int(sc.get("seed", 0)) + seed, "seed_frac": sc.get("seed_frac", 0.06)})
        dif, rea, shp = P["cell_chem_diffuse"][0], P["cell_chem_react"][0], P["cell_chem_from_shape"][0]
        ops.append({"op": "cell_neighbours", "model": "point_contact_warp", "at": "cell", "points": "pt", "contact": contact})
        ops.append({"op": "cell_chem_diffuse", "implementation": dif["implementation"], "at": "cell", "d_a": dif["d_a"] * clock, "d_h": dif["d_h"] * clock,
                    "chi": dif["chi"]})
        ops.append({"op": "cell_chem_react", "model": rea["model"], "at": "cell", "rate": rea["rate"] * clock, "F": rea["F"], "kk": rea["kk"]})
        ops.append({"op": "cell_chem_from_shape", "model": "curvature_cells", "at": "cell", "beta": shp["beta"], "F0": shp["F0"],
                    "rate": shp["rate"] * clock, "centre": centre})
        schedule += ["cell_neighbours", "cell_chem_diffuse", "cell_chem_react", "cell_chem_from_shape"]
    g = {"op": "cell_populate", "model": "mpm_points_chem", "at": "cell", "points": "pt", "rate": grow["rate"] * grow_scale * clock, "rho": grow["rho"] if rho is None else float(rho),
         "a_sw": grow["a_sw"] if a_sw is None else float(a_sw), "hill": grow["hill"], "chan": 0, "dims": 3, "target": int(round(div["factor"] * per_parent)),
         "target_jitter": round(TARGET_JITTER(div["cycle_cv"]), 4), "mark": "ready", "npts_block": "npts", "jitter": 0.3, "seed": seed,
         "seat": "fill", "swell": True}
    af = int(grow.get("after_frame", 0)) if after_frame is None else int(after_frame)
    if af:
        g["after_frame"] = af
    d = {"op": "cell_divide", "model": "cleave_tangent", "at": "cell", "points": "pt", "yolk_weight": 0.0, "p_div": 1.0, "max_per_call": 8,
         "mark": "ready", "gap": 0.1, "split": "median", "centre": centre, "seed": seed}
    if chem:
        d["inherit"] = ["chem"]
    if not no_growth:
        ops.append(g)
        ops.append(d)
    ops.append({"op": "state_dump", "at": "pt", "cells": "cell", "dir": f"{GD}/tissue/{name}", "every": dump_every})
    schedule += (["cell_populate", "cell_divide"] if not no_growth else []) + ["state_dump"]
    spec["operators"] = ops
    spec["schedule"] = schedule
    spec["seed"] = seeds
    spec["plotting"] = {
        "renderer": "vtk_points", "subject": "pt", "render_3d": "contour",
        "contour_bodies": {"mode": "partition", "iso": 0.55, "sigma": 2.0, "reach": 0.25, "smooth_method": "taubin", "smooth": 40,
                           "pass_band": 0.02, "coat": 0.0,
                           "color_by": ({"block": "chem", "channel": 0, "cmap": "Reds", "range": [0.0, 0.5], "floor": 0.0} if chem else
                                        {"block": "cellc", "channel": 0, "cmap": "YlOrBr", "range": [0.0, 19.0], "floor": 0.15})},
        "surface_opacity": 1.0, "surface_roughness": 0.5, "surface_diffuse": 0.9, "surface_ambient": 0.25, "background": "#8d99a6",
        "foreground": "white", "box_frame": False, "camera": {"elev": 20, "azim": 90}, "zoom": 1.0, "duration_s": 10, "slow_motion": 1}
    return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--cells", type=int, default=200)
    ap.add_argument("--buffer", type=int, default=1500)
    ap.add_argument("--frames", type=int, default=950)
    ap.add_argument("--record-cap", type=int, default=201)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", default=None, help="the run name (one clip only); default exp29_m_<clip>")
    ap.add_argument("--tag", default="", help="appended to the default run name")
    ap.add_argument("--per-parent", type=int, default=PER_PARENT)
    ap.add_argument("--contact", type=float, default=0.3, help="reach of cell_neighbours[point_contact_warp], the one calibrated number of the neighbour graph: 0.2 gives mean degree 6.5 on the one-cell-thick vesicle (the vertex twin: 6.0), 0.3 gives 6.9")
    ap.add_argument("--grow-scale", type=float, default=1.0, help="a PROBE: the growth rate times this (a smoke that must divide in few frames); 1 for a twin")
    ap.add_argument("--after-frame", type=int, default=None, help="a PROBE: growth starts at this frame instead of the clip's")
    ap.add_argument("--clock", type=float, default=1.0, help="TIME COMPRESSION of the body: one frame is 1/clock time units and every rate per time unit is multiplied by clock, "
                    "so the per-frame increments of the chemistry and the growth (and the frame counts) are the vertex twin's, with 1/clock of the mechanics' time per frame")
    ap.add_argument("--surface-tension", type=float, default=0.15, help="the continuum surface force coefficient of `mpm_grid_update`, acting on the boundary of each cell's own grid block (area minimisation of the cell): 0.15 is exp27's")
    ap.add_argument("--contact-tension", type=float, default=-0.05, help="the contact tension of `pair_potential[cell_contact]`, an area tension on the face two cells share, added to the cells' own surface tensions (negative = adhesion): exp27's pi5 used -0.1, -0.2 froze its sheet, -0.3 tore it")
    ap.add_argument("--rho", type=float, default=None, help="a GROWTH TEST: the growth with no activator, as a fraction of the full rate, in place of the clip's (rho 1 with --a-sw 0 is uniform growth, every cell at the same rate, the documented mode of both `cell_grow` and `cell_populate[mpm_points_chem]`)")
    ap.add_argument("--a-sw", type=float, default=None, help="a GROWTH TEST: the activator at the half-switch of the growth gate, in place of the clip's")
    ap.add_argument("--tau", type=float, default=2.0, help="the cells' Maxwell relaxation time (frames): the time a cell takes to forget its shape; 2 is exp27's body")
    ap.add_argument("--apical-area", type=float, default=0.0, help="the FOOTPRINT DRIVE: `active_force[footprint_spreading]` with this spreading acceleration k (0 = off): each cell is pushed in the plane of the sheet toward a target footprint a0 N^(2/3), the vertex twin's apical-area target")
    ap.add_argument("--face-tension", type=float, default=None, help="POLARISED CORTEX: the apical and basal faces' surface tension as a fraction of the lateral one (SimuCell3D's default is 1/20): `face` labels by radial thirds, `mpm_scatter colour_from_face`, `mpm_grid_update tension_on: face`")
    ap.add_argument("--adh-lateral", type=float, default=None, help="POLARISED ADHESION: the lateral-lateral adhesion maximum (with --adh-other for the five other face-pair classes); unset = one value for all")
    ap.add_argument("--adh-other", type=float, default=0.1)
    ap.add_argument("--no-growth", action="store_true", help="a PROBE: no `cell_populate`, no `cell_divide` (the chemistry on a fixed vesicle)")
    ap.add_argument("--drop", action="append", default=[], help="a published operator the twin leaves out (interface_tension, cell_die)")
    a = ap.parse_args()
    for clip in a.clips:
        name = a.name if (a.name and len(a.clips) == 1) else f"exp29_m_{clip}{('_' + a.tag) if a.tag else ''}"
        spec = build(clip, name, cells=a.cells, buffer=a.buffer, frames=a.frames, record_cap=a.record_cap, seed=a.seed,
                     per_parent=a.per_parent, drop=tuple(a.drop), contact=a.contact, grow_scale=a.grow_scale, after_frame=a.after_frame, no_growth=a.no_growth, clock=a.clock, surface_tension=a.surface_tension, contact_tension=a.contact_tension, rho=a.rho, a_sw=a.a_sw, tau=a.tau, apical_area=a.apical_area, face_tension=a.face_tension, adh_lateral=a.adh_lateral, adh_other=a.adh_other)
        out = os.path.join(ROOT, "config", "tissue", f"{name}.yaml")
        with open(out, "w") as f:
            yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=None, width=140)
        print(f"{os.path.relpath(out, ROOT)}  ({a.cells} cells, buffer {a.buffer}, {a.frames} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
