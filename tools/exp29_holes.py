"""exp29: how much of the sheet's surface is EMPTY -- the hole detector of the death clips, read from an MPM run's trajectory.

    python tools/exp29_holes.py exp29_t_die [--layer outer] [--every 10]

Definition (the vertex-twins fork's, 2026-10-21): fit a sphere to the live cell centroids, put 20,000 Fibonacci points on it,
d_k = the distance from point k to the nearest centroid, s = the median nearest-neighbour spacing of the centroids,

    hole_fraction = the share of the points with d_k > 1.0 s        (a hex-packed confluent tiling has d_k <= 0.58 s everywhere)
    max_hole      = max_k d_k / s

`--layer outer|inner|all` (default all) keeps the cells of one type of the seeded two-layer spheroid (`cell__node_type`, static),
for a body that is not one sheet. `--every n` prints every n-th recorded row.
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GD = os.path.join(ROOT, "graphs_data", "tissue")


def fibonacci(n):
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    th = np.pi * (1.0 + 5 ** 0.5) * k
    return np.stack([np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)], 1)


def fit_sphere(x):
    """Least-squares centre and radius: |x|^2 = 2 c.x + (r^2 - |c|^2), linear in (c, r^2 - |c|^2)."""
    A = np.c_[2.0 * x, np.ones(len(x))]
    b = (x ** 2).sum(1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    c = sol[:3]
    return c, float(np.sqrt(max(sol[3] + c @ c, 1e-12)))


def hole_stats(x, n_points=20000):
    """(hole_fraction, max_hole, s, radius) of the centroids `x` [n, 3]."""
    if len(x) < 8:
        return float("nan"), float("nan"), float("nan"), float("nan")
    c, r = fit_sphere(x)
    tree = cKDTree(x)
    s = float(np.median(tree.query(x, k=2)[0][:, 1]))
    d = tree.query(c + r * fibonacci(n_points), k=1)[0]
    return float((d > s).mean()), float(d.max() / s), s, r


def series(run, layer="all", every=10):
    z = np.load(os.path.join(GD, run, "trajectory.npz"))
    occ, pos = z["cell__occ"], z["cell__pos"].astype(np.float64)
    nt = z["cell__node_type"] if "cell__node_type" in z.files else np.zeros(occ.shape[1], int)
    names = [str(s) for s in z["cell__type_names"]] if "cell__type_names" in z.files else []
    keep = np.ones(occ.shape[1], bool)
    if layer != "all":
        keep = nt == (names.index(layer) if layer in names else {"inner": 0, "outer": 1}[layer])
    rows = sorted(set(range(0, occ.shape[0], every)) | {occ.shape[0] - 1})
    out = []
    for t in rows:
        m = occ[t] & keep
        hf, mh, s, r = hole_stats(pos[t][m])
        out.append(dict(row=t, cells=int(m.sum()), hole_fraction=hf, max_hole=mh, s=s, radius=r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--layer", default="all")
    ap.add_argument("--every", type=int, default=10)
    a = ap.parse_args()
    for run in a.runs:
        print(f"{run}  (layer {a.layer})")
        print(f"  {'row':>4}{'cells':>7}{'hole_frac':>11}{'max_hole':>10}{'s':>8}{'radius':>8}")
        for d in series(run, a.layer, a.every):
            print(f"  {d['row']:>4}{d['cells']:>7}{d['hole_fraction']:>11.3f}{d['max_hole']:>10.2f}{d['s']:>8.3f}{d['radius']:>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
