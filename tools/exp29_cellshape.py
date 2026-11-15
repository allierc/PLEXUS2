"""exp29: the SHAPE of each MPM cell, from its material points -- the detector of spiky cells.

    python tools/exp29_cellshape.py <group>/<run> [<group>/<run> ...] [--row -1]

Per live cell, from its own points x_i (the points of one cell are those with the same `pt__pid`):

    r_i      = |x_i - c|                       distance of a point from the cell's centroid c (mean of its points)
    spike    = max_i r_i / rms_i(r_i)          how far the farthest point sits, in units of the cell's rms radius;
                                               1.0 .. 1.7 for a ball or cube, above 2 for a cell with a tip
    elong    = sqrt(lambda_1 / lambda_3)       the ratio of the longest and shortest principal axes of the points'
                                               covariance (1 for a ball)

and the same two numbers split by the cell's activator (channel 0 of `cell__chem`): `high` is a > a_sw 0.6 times the
run's maximum, `low` the rest. Printed for the first stored row and the chosen row (default the last).

WHY HERE. exp29 record step 0008 (candidate (b), constant-force apical constriction) drew its red cells with spikes; the
geometry metrics of the twin comparison (hole fraction, radius curve, protrusion of the centroids) are blind to a spike on a
cell, which is a property of one body, not of the tissue. This is the per-cell shape measure; the vertex twin has no
counterpart because a polyhedron has no tip of this kind (its shape index is the nearest measure).
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GD = os.path.join(ROOT, "graphs_data")


def cell_shape(pos, pid, occ):
    """(spike, elong, cell index) of every cell holding at least 8 live points in this row."""
    live = occ > 0.5
    p, c = pos[live], pid[live].astype(np.int64)
    order = np.argsort(c, kind="stable")
    p, c = p[order], c[order]
    cells, start, count = np.unique(c, return_index=True, return_counts=True)
    spike, elong, ids = [], [], []
    for cid, s, n in zip(cells, start, count):
        if n < 8:
            continue
        x = p[s:s + n].astype(np.float64)
        d = x - x.mean(0)
        r = np.linalg.norm(d, axis=1)
        w = np.clip(np.linalg.eigvalsh(d.T @ d / n), 1e-12, None)
        spike.append(r.max() / np.sqrt((r ** 2).mean()))
        elong.append(np.sqrt(w[-1] / w[0]))
        ids.append(int(cid))
    return np.array(spike), np.array(elong), np.array(ids)


def summarise(run, row=-1, a_sw=0.6):
    z = np.load(os.path.join(GD, run, "trajectory.npz"), allow_pickle=True)
    rows = z["pt__pos"].shape[0]
    out = {}
    amax = float(np.nanmax(z["cell__chem"][..., 0])) if "cell__chem" in z.files else 0.0
    for name, t in (("first", 0), ("row", row % rows)):
        sp, el, ids = cell_shape(z["pt__pos"][t], z["pt__pid"][t][..., 0] if z["pt__pid"][t].ndim > 1 else z["pt__pid"][t],
                                 z["pt__occ"][t] if z["pt__occ"][t].ndim == 1 else z["pt__occ"][t][..., 0])
        a = z["cell__chem"][t][ids, 0] if amax > 0 else np.zeros(len(ids))
        hi = a > a_sw * amax if amax > 0 else np.zeros(len(ids), bool)
        out[name] = dict(t=t, n=len(ids), spike_median=float(np.median(sp)), spike_p95=float(np.percentile(sp, 95)),
                         elong_median=float(np.median(el)), elong_p95=float(np.percentile(el, 95)),
                         n_high=int(hi.sum()),
                         spike_high=float(np.median(sp[hi])) if hi.any() else float("nan"),
                         spike_low=float(np.median(sp[~hi])) if (~hi).any() else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--row", type=int, default=-1)
    ap.add_argument("--a-sw", type=float, default=0.6)
    a = ap.parse_args()
    print(f"{'run':36s} {'row':>4s} {'cells':>5s} {'spike med':>9s} {'spike p95':>9s} {'elong med':>9s} {'elong p95':>9s} {'high':>4s} {'spike high':>10s} {'spike low':>9s}")
    for run in a.runs:
        s = summarise(run, a.row, a.a_sw)
        for k in ("first", "row"):
            v = s[k]
            print(f"{run:36s} {v['t']:4d} {v['n']:5d} {v['spike_median']:9.2f} {v['spike_p95']:9.2f} {v['elong_median']:9.2f} "
                  f"{v['elong_p95']:9.2f} {v['n_high']:4d} {v['spike_high']:10.2f} {v['spike_low']:9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
