"""Read a set's connections from a file: the `edges_file:` key of an edge-set.

A file, not an inline `edges:` list, because a connectome-sized list is unreadable and undiffable.
Used by `engine._build_edge_set`.
"""
from __future__ import annotations

import os

import numpy as np


# _edges_from_file: Read connections from an .npz -> (pre [E], post [E], weights [E] | None).
def _edges_from_file(path: str):
    """(pre [E], post [E], weights [E] | None) from an `.npz` with `edge_index` [2, E] and optional
    `weights` [E]. A file, because a connectome-sized `edges:` literal is unreadable and undiffable;
    resolved through `graphs_data_path` like `PrescribedField`'s `source:`."""
    from plexus.paths import graphs_data_path
    p = path if os.path.isabs(path) else graphs_data_path(path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"edges_file {path!r} -> {p} does not exist")
    z = np.load(p)
    if "edge_index" not in z.files:
        raise ValueError(f"edges_file {p} has no 'edge_index' array (has: {z.files})")
    ei = np.asarray(z["edge_index"])
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edges_file {p}: edge_index must be [2, E], got {ei.shape}")
    w = np.asarray(z["weights"]).reshape(-1) if "weights" in z.files else None
    if w is not None and w.shape[0] != ei.shape[1]:
        raise ValueError(f"edges_file {p}: {w.shape[0]} weights for {ei.shape[1]} edges")
    return ei[0], ei[1], w
