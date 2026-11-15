"""exp29: a TWO-CELL PATCH at 0.2 um point spacing (Cedric, 2026-11-02): two cubic cells of 1.0 length unit (10 um), 125,000 material points each (spacing 0.02 unit = 0.2 um), touching along one face, joined by the
cadherin junction (bond_binding[population] weighted by the molecular rig's table, bond_springs[population]). Every length of the adhesion pieces is the step-23 value scaled by the spacing ratio 0.02 / 0.2, the grid is
48 per 1.6-unit box (dx 0.033 unit, about 4.5 points per grid cell) and the MPM substep is 0.002 for that grid.

  python tools/exp29_patch.py write [--spacing 0.02]      # -> config/tissue/mpm_patch_<spacing*1000>.yaml
"""
import argparse, os, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(ROOT, "experiments", "exp37_cadherin_molecule", "junction_table_smooth.json")


def build(h=0.02, frames=60, table=TABLE):
    f = h / 0.2                                                      # every adhesion length of step 23 (set for a point spacing of 0.2) scaled to this spacing
    m = round(1.0 / h); npts = m ** 3
    box, ng = 1.6, int(round(1.6 / (1.5 * h)))
    types = {"cad": {"count": 2, "shape": "ball", "material": "viscoelastic", "youngs": 20.0, "tau": 2.0, "density": 1.0}}
    cell_state = {"pos": {"width": 3, "role": "coordinate", "integration": "coordinate", "boundary": "world"}, "vel": {"width": 3, "role": "rate", "integration": "rate", "record": False},
                  "pol": {"width": 3}, "adh": {"width": 6}, "arr": {"width": 1}, "ij": {"width": 2}, "tone": {"width": 1}, "pull": {"width": 3}}
    ops = [
        {"op": "contact_finder", "at": "junction", "points": "pt", "cells": "cell", "spacing": h, "bond_range": round(0.38 * f, 4), "implementation": "hashgrid"},
        {"op": "bond_binding", "implementation": "population", "at": "junction", "cells": "cell", "points": "pt", "bond_set": "bond", "spacing": h, "binding_rate": 1.0, "unbinding_rate": 0.01, "bond_strength": 1.0, "bell": "bond",
         "min_area": round(0.1 * f * f, 6), "bond_range": round(0.38 * f, 4), "gap_table": table, "r_open": round(0.19 * f, 4), "seed": 1, "type_matrix": [[1.0, 1.0], [1.0, 1.0]][:1]},
        {"op": "pair_potential", "model": "cell_contact", "at": "pt", "cells": "cell", "repulsion_strength": 600.0, "contact_slack": 0.3, "contact_surface_tension": [[0.0]], "contact_edge_width": round(0.2 * f * 5, 4)},
        {"op": "bond_springs", "implementation": "population", "at": "pt", "junctions": "junction", "bond_stiffness": 3.0, "bond_damping": 0.05, "spacing": h, "bond_rest_length": round(0.19 * f, 4), "bond_range": round(0.38 * f, 4)},
        {"op": "mpm_strain", "at": "pt", "implementation": "polar"},
        {"op": "mpm_scatter", "at": "pt", "to": "mpm_grid", "drag": 1.0, "implementation": "warp_pp"},
        {"op": "mpm_grid_update", "at": "mpm_grid", "implementation": "warp_pp_balanced", "cell_surface_tension": 0.15, "csf_rho": 1.0, "csf_band": 0.1, "csf_smooth": 1, "tension_on": "all"},
        {"op": "mpm_gather", "at": "pt", "from": "mpm_grid", "implementation": "warp_pp"},
        {"op": "aggregate_centroid", "at": "cell", "child": "pt"}]
    return {
        "general": {"name": f"mpm_patch_{int(round(h * 1000)):03d}", "headline": "two cells in contact at 0.2 um point spacing, the cadherin junction",
                    "title": f"Two cubic cells of 10 um, {npts:,} material points each (spacing {h * 10:g} um), touching along a face; the junction kept bond by bond, weighted by the cadherin rig's bound fraction against the cleft",
                    "seed": 1, "n_frames": frames, "dt": 1.0, "record_cap": 61, "boundary": "free", "dim": 3, "world": [6.0, 6.0, 6.0], "units": {"length_um": 10.0, "time_s": 60.0}},
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": ng, "per_parent": "cell", "cell_box": box}},
        "sets": {"cell": {"n": 2, "type_layout": "ordered", "types": types, "state": cell_state},
                 "pt": {"entity": "mpm_particle", "parent": "cell", "per_parent": npts, "density": 1.0, "radius": 0.5,
                        "state": {"pos": {"width": 3, "role": "coordinate", "integration": "coordinate", "boundary": "world"}, "vel": {"width": 3, "role": "rate", "integration": "rate", "record": False},
                                  "pid": {"width": 1}, "cellc": {"width": 1}, "face": {"width": 1}}},
                 "junction": {"n": 0, "buffer": 256, "state": {k: {"width": 1, "integration": ("first_order" if k == "rho" else "none")} for k in ("ca", "cb", "fg", "rho", "area", "load")}},
                 "bond": {"n": 0, "buffer": 40000, "state": {"ia": {"width": 1, "integration": "none"}, "ib": {"width": 1, "integration": "none"}, "row": {"width": 1, "integration": "none"}, "pos": {"width": 3, "integration": "none"}}}},
        "operators": ops,
        "schedule": ["contact_finder", "bond_binding", {"steps": ["pair_potential", "bond_springs", "mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"], "substep_dt": 0.002, "capture": False}, "aggregate_centroid"],
        "seed": [{"op": "seed_positions", "model": "cell_grid", "at": "pt", "cell_set": "cell", "centre": [3.0, 3.0, 3.0], "side": 1.0, "point_spacing": h, "cells": [2, 1, 1], "arrays": [1, 1], "gap": 0.0, "live_per_cell": npts,
                  "adhesion": {"LL": 1.0}}],
        "plotting": {"renderer": "vtk_points", "subject": "pt", "cell_colours": "id", "junction_panel": True, "point_volume": h ** 3, "surface_iso_frac": 0.3, "camera": {"elev": 20, "azim": -60}, "zoom": 1.0, "duration_s": 8, "slow_motion": 1,
                     "background": "#8d99a6", "foreground": "white", "box_frame": False}}



def build_rods(h=0.05, per_face=8, frames=8, scale=6.5, dx=None, slack=None, spacer_k=None):
    """The two-cell patch with CADHERIN RODS on the facing surfaces instead of the bond ODE (Cedric, 2026-11-02): the rod of exp37 (one MPM body of 1512 points, EC domains as a label, a soft linker) enlarged x6.5 so that
    two of them tip to tip span the gap between the cells (23 nm -> 0.15 um = 0.015 length units; the 44 nm best cleft -> 0.29 um), `per_face` rods on each face, each standing by a spring on the cell point nearest its site;
    rods and cells are two MPM sets with a grid each, and the rods run in their own block of small substeps (they equilibrate far faster than a cell moves), their reaction on the cell points averaged over the block.
    Rod stiffness (in the cells' units) is set so that a rod's axial stiffness E A / L is about the junction's old kappa (3), its thermal energy so that E I / kT is 900 rod lengths, the ratio of the exp37 rod rig (E 2100, I 9.9 nm^4, kT 1, L 23 nm), whose soft linker then gives the persistence length of 5 rod lengths."""
    sp = build(h, frames)
    k_ = scale / 6.5                                                          # the rod of exp37 enlarged `scale` times (6.5: 23 nm -> 0.15 um)
    s_pt, across, length, dl, ll = 0.00036 * k_, 6, 0.015 * k_, 0.0030 * k_, 0.00145 * k_
    ny = int(round(length / s_pt)); npts = across * across * ny; A = (across * s_pt) ** 2; I = (across * s_pt) ** 4 / 12.0
    E_r = 3.0 * length / A; rho_r = 1.0e6; kT = E_r * I / (900.0 * length); m_pt = rho_r * s_pt ** 3
    n_rod = 2 * per_face; box_r, ng_r = 0.022 * k_, 30
    for o in sp["operators"]:
        if o["op"] == "pair_potential":
            if dx is not None: o["dx"] = dx
            if slack is not None: o["contact_slack"] = slack
    for k in ("junction", "bond"):
        sp["sets"].pop(k, None)
    sp["sets"]["rod"] = {"n": n_rod, "type_layout": "ordered", "types": {"cad": {"count": n_rod, "shape": "ball", "material": "elastic", "youngs": E_r, "density": rho_r}},
                         "state": {"pos": {"width": 3, "role": "coordinate", "integration": "coordinate", "boundary": "world"}, "vel": {"width": 3, "role": "rate", "integration": "rate", "record": False},
                                   "anchor": {"width": 1}, "side": {"width": 1}}}
    sp["sets"]["pt_rod"] = {"entity": "mpm_particle", "parent": "rod", "per_parent": npts, "density": rho_r, "radius": 0.5,
                            "state": {"pos": {"width": 3, "role": "coordinate", "integration": "coordinate", "boundary": "world"}, "vel": {"width": 3, "role": "rate", "integration": "rate", "record": False},
                                      "pid": {"width": 1}, "ec": {"width": 1}, "lk": {"width": 1}}}
    sp["fields"]["mpm_grid_rod"] = {"frame": "mpm_grid", "n_grid": ng_r, "per_parent": "rod", "cell_box": box_r}
    sp["sets"]["pt"]["state"]["pid"] = {"width": 1}
    cell_ops = [o for o in sp["operators"] if o["op"] in ("pair_potential", "mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather", "aggregate_centroid")]
    rod_ops = [{"op": "rod_foot", "at": "pt_rod", "rods": "rod", "cells_points": "pt", "k_foot": 1.0, "foot_length": 0.001, "carry": True},
               {"op": "tip_well", "implementation": "mpm", "at": "pt_rod", "cells": "rod", "depth": 8.0 * kT, "width": 0.00078 * k_},
               {"op": "thermal_bath", "implementation": "langevin", "at": "pt_rod", "friction": 20.0, "kT": kT, "substep_dt": 0.0005, "seed": 3},
               {"op": "mpm_strain", "at": "pt_rod", "implementation": "polar"},
               {"op": "mpm_scatter", "at": "pt_rod", "to": "mpm_grid_rod", "drag": 0.0, "implementation": "warp_pp"},
               {"op": "mpm_grid_update", "at": "mpm_grid_rod", "implementation": "warp_pp_balanced"},
               {"op": "mpm_gather", "at": "pt_rod", "from": "mpm_grid_rod", "implementation": "warp_pp"},
               {"op": "aggregate_centroid", "at": "rod", "child": "pt_rod"}]
    if spacer_k:                                                             # hold the facing layers where the EC1 domains of two rods (each `length` less half a domain from its anchor) just overlap
        g0 = 2.0 * (length - 0.5 * dl) + 0.1 * dl
        cell_ops.insert(0, {"op": "surface_spacer", "at": "pt", "cells": "cell", "gap": round(g0, 5), "k": spacer_k, "reach": round(3.0 * g0, 5), "lateral": round(1.2 * h, 4), "layer": round(0.4 * h, 4)})
    sp["operators"] = cell_ops + rod_ops                                      # cell operators first: the k-th occurrence of a name in the schedule is the k-th declared instance
    sp["schedule"] = [{"steps": (["surface_spacer"] if spacer_k else []) + ["pair_potential", "rod_foot", "tip_well", "thermal_bath", "mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather", "mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"],
                       "substep_dt": 0.0005, "capture": False}, "aggregate_centroid", "aggregate_centroid"]
    sp["seed"].append({"op": "seed_positions", "model": "rods_on_cells", "at": "pt_rod", "rods": "rod", "cells_points": "pt", "n_rods": per_face, "length": length, "across": across, "spacing": s_pt,
                       "domain_length": dl, "linker_length": ll, "ratio": 0.001})
    sp["general"].update(name=f"mpm_patch_rods_{int(round(h * 1000)):03d}", headline="two cells with cadherin MPM rods on the facing surfaces",
                         title=f"Two cubic cells (point spacing {h * 10:g} um) with {per_face} cadherin rods on each facing surface: the rod of exp37 enlarged x6.5 (0.15 um), 1512 points each, standing by a spring on a cell point, tips attracting; no bond ODE")
    sp["general"]["world"] = [6.0, 6.0, 6.0]; sp["plotting"].pop("junction_panel", None)
    return sp


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["write", "rods"]); ap.add_argument("--per-face", type=int, default=8); ap.add_argument("--spacing", type=float, default=0.02); ap.add_argument("--frames", type=int, default=60); ap.add_argument("--scale", type=float, default=6.5); ap.add_argument("--dx", type=float, default=None); ap.add_argument("--slack", type=float, default=None); ap.add_argument("--name", default=None); ap.add_argument("--spacer-k", type=float, default=None); a = ap.parse_args()
    sp = build_rods(a.spacing, a.per_face, a.frames, a.scale, a.dx, a.slack, a.spacer_k) if a.cmd == "rods" else build(a.spacing, a.frames)
    if a.name: sp["general"]["name"] = a.name
    out = os.path.join(ROOT, "config", "tissue", sp["general"]["name"] + ".yaml")
    yaml.safe_dump(sp, open(out, "w"), sort_keys=False, width=1000); print("wrote", out, sp["sets"]["pt"]["per_parent"], "points per cell, grid", sp["fields"]["mpm_grid"]["n_grid"])
