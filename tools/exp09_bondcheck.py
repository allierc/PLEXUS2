"""exp09's bond check of a recorded run (2026-11-19, after the int64 fix of the contact finders' junction key; written for the 554,500-body run of step 28): at every
stored row, every junction row must name two real cells (an index in 0 .. cells - 1), and every held bond (a live junction row of rho > 0.5) must join two bodies whose
centres are under 30 nm apart (two bodies of up to 23 nm across: a held bond any longer joins bodies that do not touch, the mark of a wrong index).

    python tools/exp09_bondcheck.py [run] [--output_root ROOT]      run default exp09_land_s554k_movie; reads <ROOT>/graphs_data/tissue/<run>/trajectory.npz

Prints one line per stored row, then BONDCHECK OK or BONDCHECK FAILED (over all rows), and exits 0 or 1.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src")]
from plexus.paths import graphs_data_path, set_data_root  # noqa: E402

NM_PER_UNIT = 10.0             # exp09's length unit: 10 nm
REACH_NM = 30.0                # the longest centre distance a held bond may span


def check(path):
    """(bad indices, held bonds over REACH_NM), summed over every stored row of the trajectory at `path`; prints each row's counts."""
    z = np.load(path, mmap_mode="r")
    nC = z["cell__occ"].shape[1]
    bad_all, far_all = 0, 0
    for r in range(z["cell__occ"].shape[0]):
        cp = np.asarray(z["cell__pos"][r], np.float64)
        jo = np.asarray(z["junction__occ"][r]).astype(bool)
        rho = np.asarray(z["junction__rho"][r])[:, 0]
        ca = np.rint(np.asarray(z["junction__ca"][r])[:, 0]).astype(np.int64)
        cb = np.rint(np.asarray(z["junction__cb"][r])[:, 0]).astype(np.int64)
        bad = int(((ca[jo] < 0) | (ca[jo] >= nC) | (cb[jo] < 0) | (cb[jo] >= nC)).sum())
        m = jo & (rho > 0.5) & (ca >= 0) & (cb >= 0) & (ca < nC) & (cb < nC)
        d = np.linalg.norm(cp[ca[m]] - cp[cb[m]], axis=1) * NM_PER_UNIT if m.any() else np.zeros(1)
        far = int((d > REACH_NM).sum())
        bad_all, far_all = bad_all + bad, far_all + far
        print(f"row {r}: {int(jo.sum()):,} junction rows, {bad:,} name a cell outside 0..{nC - 1:,}; {int(m.sum()):,} held bonds, centre distance median "
              f"{np.median(d):.1f} nm, max {d.max():.1f} nm, {far:,} over {REACH_NM:.0f} nm", flush=True)
    return bad_all, far_all


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="exp09_land_s554k_movie")
    ap.add_argument("--output_root", default=None, help="the root of graphs_data/, as Plexus_Main's (default $PLEXUS_OUTPUT_ROOT)")
    a = ap.parse_args()
    if a.output_root:
        set_data_root(a.output_root)
    bad, far = check(graphs_data_path("tissue", a.run, "trajectory.npz"))
    ok = bad == 0 and far == 0
    print("BONDCHECK", "OK" if ok else f"FAILED ({bad:,} junction rows name no cell, {far:,} held bonds over {REACH_NM:.0f} nm, all rows)")
    sys.exit(0 if ok else 1)
