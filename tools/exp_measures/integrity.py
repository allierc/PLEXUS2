"""CELL INTEGRITY of material-point cells: do a cell's points stay one body, or do they mix with the neighbours' into a chaos of
isosurfaces? (exp29 / exp30, Cedric 2026-10-23: "an entropy loss looking at particle distance with the cell id injected".)

Four readings, from the cheap report to the differentiable loss; the cell id enters only through which cell a neighbour, or a
deposit, counts toward.

    knn_purity        for point i, the share of its k nearest neighbours in its own cell; the mean over points (1 = every neighbour
                      kin). Discrete in the neighbour set: a report, not a loss.
    soft entropy      the same neighbourhood weighted by a Gaussian at the point spacing h, w_ij = exp(-|x_i - x_j|^2 / 2 h^2):
                      p_ic = sum_{j in c} w_ij / sum_j w_ij, H_i = -sum_c p_ic log p_ic; the mean over points, in nats (0 = pure,
                      ln 2 = two cells sharing the ball equally).
    grid mixing       the MPM's own view (what `pair_potential[cell_contact_dense]` penalises): every point deposits on the 8 nodes of
                      its lattice cell (spacing h) with trilinear weights, phi_c(n) = sum_{i in c} w_in the cell's fraction at node
                      n, Phi_n = sum_c phi_c(n). GINI  G = sum_n (Phi_n^2 - sum_c phi_c^2) / sum_n Phi_n^2 (the share of the
                      node pair-mass from DIFFERENT cells); ENTROPY  H = sum_n Phi_n H_n / sum_n Phi_n, H_n = -sum_c (phi_c /
                      Phi_n) log(phi_c / Phi_n). Both DIFFERENTIABLE in every position (torch; the weights are piecewise linear),
                      both read the intermediates the sparse contact already forms. At the seed two intact cells still share their
                      boundary nodes, so neither is zero on a tidy sheet: read against the seed's value.
    pieces            per cell, the share of its points in its largest connected piece (edges below 1.5 h); a cell is ONE when
                      that share is at least 0.95 (the sanity gate of exp29's gates.yaml). Discrete: the hard gate.
    tv_sphericity     per cell, from its deposited field: area A = sum_n |grad phi_c| h^2 (the total variation), volume
                      V = sum_n phi_c h^3, Psi = pi^(1/3) (6 V)^(2/3) / A (1 a ball; a ragged outline or a tongue lowers it).

    PYTHONPATH=src:tools python tools/exp_measures/integrity.py tissue/<run> [--h 0.2] [--frames 20 140] [--arms]   # per arm when the
                                                                                     cells carry types (a rig), else the whole run
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]


def knn_purity(p, pid, h, k=12):
    """(purity, soft entropy in nats) over the points p [N, 3] with cell ids pid [N]."""
    from scipy.spatial import cKDTree
    d, j = cKDTree(p).query(p, k=k + 1)
    d, j = d[:, 1:], j[:, 1:]
    ids = pid[j]
    purity = float((ids == pid[:, None]).mean())
    w = np.exp(-(d ** 2) / (2 * h * h))
    H = np.zeros(len(p))
    for n in range(len(p)):                                     # per point: the weighted label distribution of its k neighbours
        u, inv = np.unique(ids[n], return_inverse=True)
        pc = np.bincount(inv, weights=w[n]); pc = pc / max(pc.sum(), 1e-12); pc = pc[pc > 0]
        H[n] = -(pc * np.log(pc)).sum()
    return purity, float(H.mean())

# THE DIFFERENTIABLE CORE LIVES IN THE PACKAGE (src/plexus/measures_live.py, 2026-10-23): the refiner reads it inside the
# window, this ruler reads it after the run. One source; the names here are kept for the callers of 2026-10-23.
from plexus.measures_live import deposit as _deposit, grid_mixing  # noqa: E402,F401


def pieces(p, pid, h):
    """Per live cell id: the share of its points in its largest connected piece (edges below 1.5 h). Returns (ids, share)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pairs = cKDTree(p).query_pairs(1.5 * h, output_type="ndarray")
    pr = pairs[pid[pairs[:, 0]] == pid[pairs[:, 1]]]
    A = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(p), len(p)))
    _, lab = connected_components(A, directed=False)
    ids = np.unique(pid)
    share = np.array([np.bincount(lab[pid == c]).max() / (pid == c).sum() for c in ids])
    return ids, share


def tv_sphericity(p, pid, h):
    """Per live cell id: Psi = pi^(1/3) (6 V)^(2/3) / A from the cell's trilinear field on its own box (numpy). Returns (ids, Psi)."""
    ids = np.unique(pid)
    out = np.zeros(len(ids))
    for k, c in enumerate(ids):
        q = p[pid == c]
        lo = q.min(0) - 2 * h
        g = (q - lo) / h
        i0 = np.floor(g).astype(int); f = g - i0
        dims = i0.max(0) + 3
        F = np.zeros(dims)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                    np.add.at(F, (i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz), w)
        gx, gy, gz = np.gradient(F, h)
        A = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2).sum() * h ** 3           # sum |grad phi| dV = the area of the unit-fraction body
        V = F.sum() * h ** 3
        out[k] = (math.pi ** (1 / 3)) * (6 * V) ** (2 / 3) / max(A, 1e-12)
    return ids, out


def loss(x, pid, h, w_tv=0.0):
    """The differentiable integrity loss: the grid Gini impurity (+ w_tv x the mean deficit of sphericity, when asked; the TV term is
    numpy here and enters only as a report until it is needed as a gradient)."""
    gini, _ = grid_mixing(x, pid, h)
    return gini


def table(run, h, frames, per_arm):
    from exp_measures import exp29 as m
    c = m.cells_of(m.open_run(run))
    ntt = np.asarray(c.z["cell__node_type_t"]) if ("cell__node_type_t" in c.z.files and per_arm) else None
    for fr in frames:
        t = min(int(round(fr / c.every)), c.rows - 1)
        pos = np.asarray(c.z["pt__pos"][t]); occ = np.asarray(c.z["pt__occ"][t]) > 0; pid = np.asarray(c.z["pt__pid"][t]).reshape(-1).astype(int)
        groups = [(None, np.ones(occ.sum(), bool))] if ntt is None else [(g, ntt[t][pid[occ]] == g) for g in range(int(ntt[t].max()) + 1)]
        print(f"\n{run} frame {c.frame(t):.0f}" + ("  (per arm)" if ntt is not None else ""))
        print(f"{'arm':>4s} {'cells':>5s} {'kNN purity':>10s} {'soft entropy':>12s} {'grid gini':>9s} {'grid entropy':>12s} {'one piece':>9s} {'piece med':>9s} {'sphericity':>10s}")
        for g, msk in groups:
            p = pos[occ][msk]; ip = pid[occ][msk]
            if len(p) < 50:
                continue
            pur, sent = knn_purity(p, ip, h)
            xt = torch.as_tensor(p, dtype=torch.float64); it = torch.as_tensor(ip)
            gini, ent = grid_mixing(xt, it, h)
            ids, share = pieces(p, ip, h)
            _, psi = tv_sphericity(p, ip, h)
            print(f"{('all' if g is None else str(g)):>4s} {len(ids):5d} {pur:10.3f} {sent:12.3f} {float(gini):9.3f} {float(ent):12.3f} {np.mean(share >= 0.95):9.2f} {np.median(share):9.3f} {np.median(psi):10.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--h", type=float, default=0.2)
    ap.add_argument("--frames", type=float, nargs="+", default=[20, 140])
    ap.add_argument("--arms", action="store_true")
    a = ap.parse_args()
    table(a.run, a.h, a.frames, a.arms)
