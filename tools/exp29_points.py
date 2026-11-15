"""exp29: a first-step spec (config/tissue/mpm_grid3x3_*.yaml, cells of 125 points on a cell_grid) rewritten at 1000 points per cell (Cedric, 2026-11-02: "rerun everything at 1000 points").

  python tools/exp29_points.py            # -> config/tissue/<name>_p1000.yaml for every first-step run

A cell keeps its volume (1.0 length^3): the lattice spacing halves (0.2 -> 0.1: 5^3 -> 10^3 points), each point carries 1/1000 of the cell, the dormant reserve scales 8x, and a spec key `spacing`
(a point spacing read by the contact operators) halves with it. Nothing else changes. `mpm_grid3x3_flatten_one10` (one cell dividing ten times, 1500 cell slots, a million points) is left out.
"""
import os, sys
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = ["cycles", "overlap_sweep", "overlap_sweep2_s59", "overlap_sweep2", "width_sweep3", "width_sweep", "tension_sweep", "overlap_sweep3", "drive_sweep", "drive_sweep2", "flatten_sweep", "flatten_nodiv",
        "flatten_f", "flatten_pair", "flatten_f2", "flatten_pair2", "flatten_pair3", "flatten_div", "flatten_div5", "flatten_div5_near", "flatten_tens"]
POINTS_OLD, POINTS = 125, 1000


def to_points(spec, name):
    k = (POINTS_OLD / POINTS) ** (1.0 / 3.0)                                   # 0.5: the spacing ratio at a fixed cell volume
    spec["general"]["name"] = name
    spec["general"]["title"] = str(spec["general"].get("title", "")) + f" -- {POINTS} points per cell (Cedric, 2026-11-02)"
    for s in spec["sets"].values():
        if s.get("entity") == "mpm_particle" and s.get("per_parent") == POINTS_OLD:
            s["per_parent"] = POINTS
            s["grow_reserve"] = round(s.get("grow_reserve", 0) * POINTS / POINTS_OLD)
    for o in spec["seed"]:
        if o.get("op") == "seed_positions" and "point_spacing" in o:
            o["point_spacing"] = round(o["point_spacing"] * k, 4)
            if o.get("live_per_cell"):
                o["live_per_cell"] = POINTS
    for o in spec["operators"]:
        if "spacing" in o:
            o["spacing"] = round(o["spacing"] * k, 4)
    spec["plotting"]["point_volume"] = round(1.0 / POINTS, 6)
    return spec


if __name__ == "__main__":
    for r in RUNS:
        src = os.path.join(ROOT, "config", "tissue", f"mpm_grid3x3_{r}.yaml")
        out = os.path.join(ROOT, "config", "tissue", f"mpm_grid3x3_{r}_p1000.yaml")
        yaml.safe_dump(to_points(yaml.safe_load(open(src)), f"mpm_grid3x3_{r}_p1000"), open(out, "w"), sort_keys=False, width=1000)
        print("wrote", os.path.relpath(out, ROOT))
