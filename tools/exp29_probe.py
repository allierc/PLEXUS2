"""exp29: what a run of an MPM twin did, from its trajectory -- cells, points, chemistry, shape, cost -- one line per recorded row.

    python tools/exp29_probe.py <run> [--rows 8]            # run = tissue/<name> or <name>

Read-only and local (an analysis, not a job). Per sampled row: live cells, live points and their share of the point pool, points per cell (median,
5th and 95th percentile), the centroids' radius about their mean (median, and the 5th-95th spread), the activator of the live cells (mean, max, the
fraction above 0.35, the clips' own `a_sw`), the connected high-activator groups on the contact graph of the centroids (cells closer than 1.6 median
neighbour spacings), and a finiteness check of every float block; then the median frame time. Nothing is typed: every number is read from the run.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GD = os.path.join(ROOT, "graphs_data")


def load(run):
    run = run if "/" in run else f"tissue/{run}"
    return run, np.load(os.path.join(GD, run, "trajectory.npz"))


def groups(X, high, link):
    """Number of connected groups of `high` cells linked when closer than `link`, and the size of each."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    idx = np.flatnonzero(high)
    if len(idx) == 0:
        return []
    pr = cKDTree(X[idx]).query_pairs(link, output_type="ndarray")
    n = len(idx)
    A = csr_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n)) if len(pr) else csr_matrix((n, n))
    k, lab = connected_components(A, directed=False)
    return sorted(np.bincount(lab).tolist(), reverse=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--a-sw", type=float, default=0.35)
    a = ap.parse_args()
    run, z = load(a.run)
    cocc = z["cell__occ"] > 0.5
    pocc = z["pt__occ"] > 0.5
    rows = z["cell__pos"].shape[0]
    sel = sorted(set(np.linspace(0, rows - 1, min(a.rows, rows)).astype(int).tolist()))
    pool = pocc.shape[1]
    chem = z["cell__chem"] if "cell__chem" in z.files else None
    par = z["pt__parent"] if "pt__parent" in z.files else None
    print(f"{run}: {rows} rows, {cocc.shape[1]} cell slots, {pool:,} point slots")
    for r in sel:
        live = np.flatnonzero(cocc[r])
        X = z["cell__pos"][r][live]
        cen = X.mean(0)
        rad = np.linalg.norm(X - cen, axis=1)
        pid = z["pt__pid"][r][:, 0][pocc[r]].astype(int)
        cnt = np.bincount(pid, minlength=cocc.shape[1])[live]
        line = (f"row {r:4d}: cells {len(live):4d}  points {int(pocc[r].sum()):7,d} ({pocc[r].mean():.2f} of pool)  per cell median "
                f"{np.median(cnt):.0f} [{np.percentile(cnt, 5):.0f}, {np.percentile(cnt, 95):.0f}]  radius {np.median(rad):.2f} "
                f"[{np.percentile(rad, 5):.2f}, {np.percentile(rad, 95):.2f}]")
        if chem is not None:
            act = chem[r][live][:, 0]
            sub = chem[r][live][:, 1]
            high = act > a.a_sw
            from scipy.spatial import cKDTree
            nn = np.median(cKDTree(X).query(X, k=2)[0][:, 1])
            g = groups(X, high, 1.6 * nn)
            line += (f"\n          activator mean {act.mean():.3f} max {act.max():.3f}  above {a.a_sw}: {high.mean():.2f} of cells in {len(g)} groups "
                     f"{g[:6]}; substrate mean {sub.mean():.3f} min {sub.min():.3f}")
        print(line)
    # the divisions: a slot that becomes live between two recorded rows is a daughter; its mother is the cell whose point count fell by about the
    # daughter's count; equal daughters (`split: median`) hold n // 2 and n - n // 2 of the mother's n
    diffs, mothers_n = [], []
    for r in range(1, rows):
        new = np.flatnonzero(cocc[r] & ~cocc[r - 1])
        if len(new) == 0:
            continue
        pid_prev = z["pt__pid"][r - 1][:, 0][pocc[r - 1]].astype(int)
        pid_now = z["pt__pid"][r][:, 0][pocc[r]].astype(int)
        c_prev = np.bincount(pid_prev, minlength=cocc.shape[1]); c_now = np.bincount(pid_now, minlength=cocc.shape[1])
        for d in new:
            fell = np.flatnonzero(cocc[r - 1] & (c_now < c_prev - 0.3 * c_now[d]))
            if len(fell) == 0:
                continue
            m = fell[np.argmin(np.abs(c_prev[fell] - c_now[fell] - c_now[d]))]
            diffs.append(abs(int(c_now[m]) - int(c_now[d]))); mothers_n.append(int(c_prev[m]))
    if diffs:
        dd = np.asarray(diffs)
        print(f"divisions read from the rows: {len(dd)}; |daughter A - daughter B| in points: median {np.median(dd):.0f}, "
              f"within 2: {np.mean(dd <= 2):.2f}, 90th percentile {np.percentile(dd, 90):.0f}; mother's points before: median {np.median(mothers_n):.0f}")
    bad = [k for k in z.files if z[k].dtype.kind == "f" and not np.isfinite(z[k]).all()]
    fm = z["frame_ms"] if "frame_ms" in z.files else None
    print("non-finite blocks:", bad or "none")
    if fm is not None:
        print(f"frame_ms median {np.median(fm):.0f} ms, total {fm.sum() / 1000:.0f} s over {len(fm)} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
