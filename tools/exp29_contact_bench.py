"""exp29 (M5): what `cell_neighbours[point_contact]` costs at the sizes of the MPM vesicle, against `[point_contact_warp]`, on this machine's GPU.

    python tools/exp29_contact_bench.py [N_CELLS ...]        # default 200 and 600 cells of 512 points: 102,400 and 307,200 points

The body is the vesicle's: n cells on a sphere of radius 1.05 sqrt(n / 4 pi), 512 points per cell uniform in the shell R +- 0.5 (density ~ the exp27 cell),
each point to its nearest centre. The median of five calls after one warm-up, in milliseconds, and whether the two graphs are identical.
"""
import math
import sys
import time
import types

import numpy as np
import torch

import plexus.operators  # noqa: F401
from plexus.models.registry import get_contract


def body(n, per=512, seed=0):
    k = np.arange(n) + 0.5
    ph = np.arccos(1 - 2 * k / n); th = math.pi * (1 + math.sqrt(5)) * k
    R = 1.05 * math.sqrt(n / (4 * math.pi))
    cen = R * np.stack([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)], 1)
    g = np.random.default_rng(seed)
    m = n * per
    v = g.normal(size=(m, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = np.cbrt(g.uniform((R - 0.5) ** 3, (R + 0.5) ** 3, m))
    X = v * r[:, None]
    from scipy.spatial import cKDTree
    return X, cKDTree(cen).query(X)[1]


def one(model, X, par, n, dev, contact=0.3):
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    pt = types.SimpleNamespace(occ=torch.ones(len(X), device=dev), get=lambda k: Xt)
    pc = torch.as_tensor(par, device=dev)
    op = get_contract("cell_neighbours").implementations[model]({"_at": "cell", "points": "pt", "contact": contact})
    ts, cel = [], None
    for i in range(6):
        cel = types.SimpleNamespace(state=torch.zeros(n, 3, device=dev), edge_index=None)
        H = types.SimpleNamespace(level=lambda nm: pt if nm == "pt" else cel, lift_index=lambda a, b: pc)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        op._n = 0
        op.forward(H)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.time() - t) * 1000.0)
    return float(np.median(ts[1:])), cel.edge_index.cpu(), cel.edge_weight.cpu()


def main():
    sizes = [int(a) for a in sys.argv[1:]] or [200, 600]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""), flush=True)
    for n in sizes:
        X, par = body(n)
        t1, e1, w1 = one("point_contact", X, par, n, dev)
        t2, e2, w2 = one("point_contact_warp", X, par, n, dev)
        same = bool(torch.equal(e1, e2) and torch.equal(w1, w2))
        print(f"{n} cells, {len(X):,} points: point_contact {t1:.1f} ms, point_contact_warp {t2:.1f} ms ({t1 / max(t2, 1e-9):.1f}x), "
              f"{e1.shape[1] // 2} cell pairs, identical {same}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
