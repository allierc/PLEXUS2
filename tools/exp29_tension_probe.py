"""exp29 M3: what the interface-tension arms do to a body with a planted activator domain, read from the recorded trajectory.

    python tools/exp29_tension_probe.py exp29_t_tension_off exp29_t_tension_a exp29_t_tension_a2 exp29_t_tension_b [--row n]

Per run, at the first and at the last (or `--row`) recorded row of `graphs_data/tissue/<run>/trajectory.npz`; the activator is
static in these runs (a random 25 % of the cells seeded high), so the high set is the same at every row and any change is
the tension's.

    n_high          cells with a > 0.6 max_live a (the gate of `interface_tension` and of `contact_retype[chem_threshold]`)
    bulge           (mean radius of the high cells - mean radius of the low cells) / mean radius of all live cells, the
                    radius of a cell its centroid's distance from the mean centroid of the live cells
    corr_a_r        Spearman correlation over the live cells of the activator against that radius (the vertex campaign's
                    `corr_act_rad`)
    hetero          the contact area between a high and a low cell over the total contact area, summed over the live
                    `junction` rows (one row per touching cell pair and face-pair class, `area` in world units^2)
    high_exposed    the share of the contact area of the high cells that is with low cells
    cells, points   live cells and live points
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GD = os.path.join(ROOT, "graphs_data", "tissue")


def _rank(x):
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x)); r[o] = np.arange(len(x))
    return r


def spearman(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def read(run, row, a_sw=0.6, chan=0):
    z = np.load(os.path.join(GD, run, "trajectory.npz"))
    n = z["cell__occ"].shape[0]
    t = n - 1 if row is None else (row if row >= 0 else n + row)
    live = z["cell__occ"][t]
    a = z["cell__chem"][t][:, chan].astype(np.float64)
    amax = float(a[live].max())
    high = live & (a > a_sw * amax)
    x = z["cell__pos"][t].astype(np.float64)
    c0 = x[live].mean(0)
    r = np.linalg.norm(x - c0, axis=1)
    jl = z["junction__occ"][t]
    ca = z["junction__ca"][t][:, 0].astype(int)[jl]
    cb = z["junction__cb"][t][:, 0].astype(int)[jl]
    ar = z["junction__area"][t][:, 0].astype(np.float64)[jl]
    ok = live[ca] & live[cb] & (ar > 0)
    ca, cb, ar = ca[ok], cb[ok], ar[ok]
    het = high[ca] != high[cb]
    anyhigh = high[ca] | high[cb]
    return dict(row=int(t), n_high=int(high.sum()), cells=int(live.sum()), points=int(z["pt__occ"][t].sum()),
                bulge=float((r[high].mean() - r[live & ~high].mean()) / r[live].mean()) if high.any() else float("nan"),
                corr_a_r=spearman(a[live], r[live]),
                hetero=float(ar[het].sum() / ar.sum()) if ar.sum() else float("nan"),
                high_exposed=float(ar[het].sum() / ar[anyhigh].sum()) if ar[anyhigh].sum() else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--row", type=int, default=None)
    a = ap.parse_args()
    keys = ("row", "cells", "points", "n_high", "bulge", "corr_a_r", "hetero", "high_exposed")
    print(f"{'run':<24}{'when':<7}" + "".join(f"{k:>14}" for k in keys))
    for run in a.runs:
        for when, row in (("first", 0), ("last", a.row)):
            d = read(run, row)
            print(f"{run:<24}{when:<7}" + "".join(f"{d[k]:>14.4f}" if isinstance(d[k], float) else f"{d[k]:>14}" for k in keys))
    return 0


if __name__ == "__main__":
    sys.exit(main())
