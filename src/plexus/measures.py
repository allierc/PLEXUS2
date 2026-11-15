"""One entry point for every quantity read off a run.

Three consumers used to compute the same numbers separately: the movie's curve panel (`live_movie._curve_row`, seven quantities on a live or replayed level), the gates (`tools/gate_measures.MEASURES`, 61 series over a trajectory reader) and the regression fingerprints (`tools/regression_lib.checkpoint_metrics`, 20 metrics at a frame): a cell count was coded three times and a shell radius twice, with two definitions (mean, median). This module holds every one, moved verbatim.

  readers    `Traj` (the eleven accessors every row measure is written against) and its implementations over the core `trajectory.npz` (`CoreTraj`, `ParticleTraj`) and the archived okuda `traj.npz` (`OkudaTraj`); `open_traj` picks one from a path
  curve      `curve_row(H, lvl, q, ntype, nt, cell_cols)`: one row [nt, 2] of (mean, sd) for a curve quantity on a level, live (engine tensors) or replayed (`_ReplayLevel`); the body is the renderer's, `count:<set>` the one addition
  frame      `checkpoint_metrics(d, fr)`: the fingerprint's metrics at one recorded frame
  registry   `MEASURES`: name -> Measure(kind, fn, dim); `kind` is "curve" (one row on a level), "row" (a series over a Traj, the gate form), "frame" (a dict at one frame) or "live" (a torch scalar of the live state WITH a graph back to the positions, `fn(H, **kw)`: the loss terms of `plexus.ucb_diff`, in `measures_live.py`); `dim` is the quantity's dimension in the vocabulary of `plexus.units` ("count", "length", "area", "volume", "fraction", or None), so a consumer converts at display time and never inside a measure

Bit-identity is the contract of the move: `tools/quantities_baseline.py` captures what the three consumers compute on the registered working points before and after, and the two captures must be equal in every bit. Each consumer keeps its own dtypes (the curve panel reads float32 off the engine, the gates float64 off the file), which is why the readers are adapters and the bodies untouched."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


# Measure: One registered quantity: name, kind (curve / row / frame / live), function and dimension.
@dataclass(frozen=True)
class Measure:
    name: str
    kind: str                    # "curve" | "row" | "frame" | "live"
    fn: Optional[Callable]       # None for a curve quantity, which `curve_row` dispatches on by name
    dim: Optional[str] = None    # plexus.units dimension, or None when the quantity has none
    doc: str = ""


MEASURES: dict[str, Measure] = {}


# register: Register a measure under a name (a different function under a taken name is refused) -> None.
def register(name: str, kind: str, fn=None, dim=None, doc=""):
    if name in MEASURES and MEASURES[name].fn is not fn:
        raise ValueError(f"measure {name!r} is already registered ({MEASURES[name].kind})")
    MEASURES[name] = Measure(name, kind, fn, dim, doc or (getattr(fn, "__doc__", "") or "").strip().split("\n")[0])
    return fn


# register_live: Decorator registering `fn(H, **kw) -> torch scalar` (with a graph to the state) as a live measure -> fn.
def register_live(name: str, dim=None):
    def wrap(fn):
        register(name, "live", fn, dim)
        return fn
    return wrap


# live_fn: The function of the live measure `name`, for a search spec's `metrics:` entry -> callable.
def live_fn(name: str):
    m = MEASURES.get(name)
    if m is None or m.kind != "live":
        raise KeyError(f"no live measure named {name!r}; registered: {sorted(n for n, x in MEASURES.items() if x.kind == 'live')}")
    return m.fn


# register_rows: Register a gate-style table {name: fn(T, **kw) -> series} as row measures -> None.
def register_rows(table: dict, dims: dict | None = None):
    """Register a gate-style table {name: fn(T, **kw) -> series} as row measures."""
    dims = dims or {}
    for n, f in table.items():
        register(n, "row", f, dims.get(n))


# dims_of: The declared dimension of each measure (of one kind, if given) -> {name: dim}.
def dims_of(kind: str | None = None) -> dict[str, str]:
    return {n: m.dim for n, m in MEASURES.items() if m.dim and (kind is None or m.kind == kind)}


# to_physical: A measured number in physical units through the run's `units:` block -> float.
def to_physical(name: str, value, units):
    """A measured number in physical units, through the run's declared `units:` block; the value
    itself when the measure declares no dimension or the run declares no scale."""
    m = MEASURES.get(name)
    if m is None or m.dim is None or not units:
        return value
    from plexus.units import to_physical as _tp
    s = _tp(1.0, m.dim, units)
    return value if s is None else value * float(s)


# _Lazy: A dict over an NpzFile that reads each array once.
class _Lazy:
    """A dict over an NpzFile that reads each array ONCE: `np.load` decompresses the whole array on every `z[key]`, so `ParticleTraj.pos(t)` written as `z["mpm_particle__pos"][t]` re-read all 778 MB of a 720-row MPM trajectory to take one row (five measures x 720 rows = 2,800 GB of decompression, a grading that ran past ten minutes)."""

    def __init__(self, z):
        self._z, self._c = z, {}

    def __contains__(self, k): return k in self._z.files

    @property
    def files(self): return self._z.files

    def __getitem__(self, k):
        if k not in self._c:
            self._c[k] = self._z[k]
        return self._c[k]


# Traj: The accessors every row measure is written against, implemented over a trajectory layout.
class Traj:
    """Seven accessors. Everything else in this file is written against these."""

    def n_rows(self) -> int: raise NotImplementedError
    def nF(self, t) -> int: raise NotImplementedError
    def nV(self, t) -> int: raise NotImplementedError
    def pos(self, t): raise NotImplementedError          # [nV, 3] live prefix
    def half_edges(self, t): raise NotImplementedError   # (E_srce, E_trgt, E_face) for row t
    def face_col(self, name, t): raise NotImplementedError   # [nF] or None
    def state(self, block, t): raise NotImplementedError     # [nF, width] or None
    # per-vertex, separate from `state` on purpose: `state` reads the CELL set cropped to nF, while a per-vertex block has another length (gate_00's last row: 13,824 vertices against 6,914 cells), so reusing `state` would return the first 6,914 rows of a 13,824-row array, which looks like data; added for the apico-basal separation `sep`
    def vertex_block(self, name, t): return None             # [nV, width] or None
    def occ(self, set_name, t): raise NotImplementedError    # [buffer] bool
    def scalar(self, name, t): return None                   # an operator's own counter, or None
    def edge_col(self, name, t): return None                 # per-HALF-EDGE state, or None


# CoreTraj: A Traj over the core's `trajectory.npz`, mesh included.
class CoreTraj(Traj):
    """The core's `trajectory.npz`."""

    def __init__(self, path, set_name=None, cell_set=None):
        self.z = _Lazy(np.load(path))
        f = self.z.files
        if set_name is None:
            c = [k[: -len("__mesh_nF")] for k in f if k.endswith("__mesh_nF")]
            set_name = c[0] if c else None
        if set_name is None:
            raise ValueError(f"{path} has no mesh set -- this gate's measures need one")
        self.s = set_name
        if cell_set is None:
            c = [k[: -len("__occ")] for k in f
                 if k.endswith("__occ") and not k.startswith(set_name)]
            cell_set = c[0] if c else None
        self.c = cell_set
        self._nF = self.z[f"{set_name}__mesh_nF"]
        self._Nv = self.z[f"{set_name}__mesh_Nv"]
        self._off = self.z[f"{set_name}__mesh_offsets"]
        self._foff = self.z[f"{set_name}__mesh_face_offsets"]
        self.path = path

    def n_rows(self): return int(len(self._nF))
    def nF(self, t): return int(self._nF[t])
    def nV(self, t): return int(self._Nv[t])
    def pos(self, t): return np.asarray(self.z[f"{self.s}__pos"][t][: self.nV(t)], float)

    def half_edges(self, t):
        a, b = int(self._off[t]), int(self._off[t + 1])
        return (self.z[f"{self.s}__mesh_E_srce"][a:b],
                self.z[f"{self.s}__mesh_E_trgt"][a:b],
                self.z[f"{self.s}__mesh_E_face"][a:b])

    def face_col(self, name, t):
        k = f"{self.s}__mesh_{name}"
        if k not in self.z.files:
            return None
        a, b = int(self._foff[t]), int(self._foff[t + 1])
        return np.asarray(self.z[k][a:b], float)

    def state(self, block, t):
        k = f"{self.c}__{block}"
        if self.c is None or k not in self.z.files:
            return None
        return np.asarray(self.z[k][t][: self.nF(t)], float)

    def vertex_block(self, name, t):
        k = f"{self.s}__{name}"
        if k not in self.z.files:
            return None
        a = np.asarray(self.z[k][t][: self.nV(t)], float)
        return a[:, None] if a.ndim == 1 else a

    def occ(self, set_name, t):
        k = f"{set_name}__occ"
        return np.asarray(self.z[k][t], bool) if k in self.z.files else None

    def scalar(self, name, t):
        k = f"{self.s}__mesh_scalar_{name}"
        return float(self.z[k][t]) if k in self.z.files else None

    def edge_col(self, name, t):
        """A per-HALF-EDGE column, sliced with ITS OWN offsets: a third ragged length that must not share `mesh_offsets`, because `junction_myosin` writes `myo` for the arrays as they were BEFORE the frame's topology operators, so on a frame with a division or flip it is a few entries short of the half-edge table and shared offsets would slice the next frame's myosin into this one. `myosin_array_aligned_with_half_edges` asserts the lengths agree; this accessor must not paper over the case it tests."""
        k = f"{self.s}__mesh_e_{name}"
        ko = k + "_offsets"
        if k not in self.z.files or ko not in self.z.files:
            return None
        o = self.z[ko]
        return np.asarray(self.z[k][int(o[t]):int(o[t + 1])], float)


# ParticleTraj: A Traj over a core trajectory with no mesh (an MPM set and its parent: gates 02 and 03).
class ParticleTraj(Traj):
    """A core trajectory with NO MESH: an MPM set and its parent (gates 02 and 03). `CoreTraj` raises on a file with no `__mesh_nF`, rightly (eleven of gate 00's rows would silently become uncomputable); a particle gate needs a different reader, not a `CoreTraj` whose mesh accessors return None, because "the mesh is empty" and "there is no mesh" are different facts and only one is a defect."""

    def __init__(self, path, set_name=None):
        self.z = _Lazy(np.load(path))
        if set_name is None:
            c = [k[: -len("__pos")] for k in self.z.files if k.endswith("__pos")]
            # the biggest positional set is the material; the parent is a single centroid
            set_name = max(c, key=lambda k: self.z[f"{k}__pos"].shape[1]) if c else None
        if set_name is None:
            raise ValueError(f"{path} has no positional set")
        self.s = set_name
        self.parent = next((k[: -len("__pos")] for k in self.z.files
                            if k.endswith("__pos") and k[: -len("__pos")] != set_name), None)
        self.path = path

    def n_rows(self): return int(self.z[f"{self.s}__pos"].shape[0])
    def nF(self, t): return int(self.occ(self.s, t).sum())
    def nV(self, t): return self.nF(t)

    def pos(self, t):
        o = self.occ(self.s, t)
        p = self.z[f"{self.s}__pos"][t]
        return np.asarray(p[o] if o is not None else p, float)

    def half_edges(self, t):
        raise KeyError("this gate's trajectory has no mesh -- a topology row does not belong in it")

    def face_col(self, name, t): return None
    def state(self, block, t):
        k = f"{self.s}__{block}"
        return np.asarray(self.z[k][t], float) if k in self.z.files else None

    def occ(self, set_name, t):
        k = f"{set_name}__occ"
        return np.asarray(self.z[k][t], bool) if k in self.z.files else None

    def scalar(self, name, t): return None

    def parent_pos(self, t):
        if self.parent is None:
            return self.pos(t).mean(0)
        return np.asarray(self.z[f"{self.parent}__pos"][t][0], float)


# OkudaTraj: A Traj over the archived okuda `traj.npz` (decimated: about 60 rows).
class OkudaTraj(Traj):
    """okuda's `traj.npz`: `pos_i`, a pickled `mesh_i`, `act_i` and `ticks`. It is decimated (about 60 of the run's rows, the ones its movie draws), so a `window:` between kept rows cannot be answered: `row_of(tick)` raises rather than picking the nearest, because a per-tick ledger evaluated across a 31-frame gap is a green row that tested nothing."""

    def __init__(self, path):
        self.z = np.load(path, allow_pickle=True)
        self.ticks = np.asarray(self.z["ticks"]).tolist() if "ticks" in self.z.files else None
        self._n = sum(1 for k in self.z.files if k.startswith("pos_"))
        self._m = [None] * self._n
        self.path = path

    def _mesh(self, t):
        if self._m[t] is None:
            v = self.z[f"mesh_{t}"]
            self._m[t] = v.item() if hasattr(v, "item") else v
        return self._m[t]

    def n_rows(self): return self._n
    def nF(self, t): return int(self._mesh(t)["nF"])
    def nV(self, t): return int(self._mesh(t)["Nv"])
    def pos(self, t): return np.asarray(self.z[f"pos_{t}"], float)

    def half_edges(self, t):
        m = self._mesh(t)
        return (np.asarray(m["E_srce"]), np.asarray(m["E_trgt"]), np.asarray(m["E_face"]))

    def face_col(self, name, t):
        v = self._mesh(t).get(name)
        return None if v is None else np.asarray(v, float).ravel()[: self.nF(t)]

    def state(self, block, t):
        if block == "chem" and f"act_{t}" in self.z.files:
            return np.asarray(self.z[f"act_{t}"], float)[:, None]
        return None

    def occ(self, set_name, t):
        return None                                # okuda records the live prefix, not the mask

    def scalar(self, name, t):
        v = self._mesh(t).get(name)
        return None if v is None or np.ndim(v) != 0 else float(v)

    def edge_col(self, name, t):
        v = self._mesh(t).get(name)
        return None if v is None or np.ndim(v) == 0 else np.asarray(v, float).ravel()

    def row_of(self, tick):
        if self.ticks is None:
            return int(tick)
        if int(tick) not in self.ticks:
            raise KeyError(f"tick {tick} is not one of the {len(self.ticks)} rows okuda kept "
                           f"(nearest {min(self.ticks, key=lambda x: abs(x - tick))}); a per-tick "
                           f"measure cannot be evaluated across the gap")
        return self.ticks.index(int(tick))


# _core: `CoreTraj` if the file carries a mesh, else `ParticleTraj` -> Traj.
def _core(path):
    """`CoreTraj` if the file carries a mesh, `ParticleTraj` if it does not."""
    z = np.load(path)
    return (CoreTraj(path) if any(k.endswith("__mesh_nF") for k in z.files)
            else ParticleTraj(path))


# open_traj: A Traj over whichever layout is at this path -> Traj.
def open_traj(path_or_dir):
    """A `Traj` over whichever layout is at this path."""
    p = path_or_dir
    if os.path.isdir(p):
        if os.path.exists(os.path.join(p, "traj.npz")):
            return OkudaTraj(os.path.join(p, "traj.npz"))
        for root, _d, files in os.walk(p):
            if "trajectory.npz" in files:
                return _core(os.path.join(root, "trajectory.npz"))
        raise FileNotFoundError(f"no trajectory under {p}")
    return OkudaTraj(p) if os.path.basename(p) != "trajectory.npz" else _core(p)




# _MeshView: A read-only mesh table with the cell set's per-cell blocks overlaid.
class _MeshView:
    """A read-only mesh table with the cell set's per-cell blocks overlaid. A view, not a dict copy: the live path's `m` is a `MeshTable` of CUDA tensors and the replay path's a plain dict, both read as `m["nF"]` and `m.get(name)` and never written, so wrapping keeps one code path and copies nothing. The mesh wins every name clash: an overlay is consulted only for a name the table lacks, so a cell-set block called `area` (there is one) can never shadow a face column or `nF`."""

    __slots__ = ("_m", "_c")

    def __init__(self, m, cell_cols):
        self._m = m
        self._c = cell_cols or {}

    def __getitem__(self, k):
        try:
            return self._m[k]
        except KeyError:
            return self._c[k]

    def __contains__(self, k):
        return k in self._m or k in self._c

    def get(self, k, default=None):
        v = self._m.get(k, None)
        return self._c.get(k, default) if v is None else v

    def __getattr__(self, a):                 # `reindex_faces`, `snapshot`, ... stay reachable
        return getattr(self._m, a)


# CURVE_DIMS, CURVE_QUANTITIES: The curve quantities and their dimensions, so the panel can convert them (the volume panel once appended `um^3` to a raw simulation number, wrong by `length_um ** 3`); `myosin` has none (a per-junction activity prints bare).
CURVE_DIMS = {"area": "area", "volume": "volume", "radius": "length",
              "cells": "count", "phase": "fraction", "cycle_progress": "fraction"}
CURVE_QUANTITIES = ("cells", "area", "volume", "radius", "myosin", "phase", "cycle_progress",
                    "shape_index")
for _q in CURVE_QUANTITIES:
    register(_q, "curve", None, CURVE_DIMS.get(_q))
register("count", "curve", None, "count", "the live count of a named set: `count:<set>`")
register("species", "curve", None, "count", "the live count of each species of a set, a line per `chem` column in the `plotting.species` colours -- the twin of `phase`: `species:<set>`")
register("block", "curve", None, None, "mean and sd of a named state block over a set's live elements: `block:<set>:<block>`")
register("jacobian", "curve", None, None, "an MPM set's volume relative to rest, mean and sd of det F over its live particles: `jacobian:<set>`")
register("cell_volume", "curve", None, "volume", "the volume of each cell of material points, mean and sd over the live cells: the convex hull of each cell's points, grouped by their `pid` block (else the set's `parent`): `cell_volume:<point set>`")
register("cell_area", "curve", None, "area", "the footprint of each cell of material points in the sheet's plane (normal z), mean and sd over the live cells: the 2D convex hull of each cell's points: `cell_area:<point set>`")
register("strain", "curve", None, None, "an MPM set's shape change: rms of the Green strain |F^T F - I| over its live particles, rotation-free: `strain:<set>`")
register("rate", "curve", None, None, "a set's rotation rate about the world z axis through the box centre, from its velocities, in turns per unit time: `rate:<set>`")
register("clones", "curve", None, None, "a Muller plot of an integer label block on the mesh's cells (a clone, a mutant flag): each of the `top` labels with the largest peak share is a band between its cumulative limits, in % of the live cells, the rest one grey band -- replay only: `clones:<block>`")


# label_rgb: The colour of integer label k (a clone, a mutant flag) as (r, g, b) in 0..1 -> tuple.
def label_rgb(k, pal=None):
    """The colour of integer label `k` (a clone, a mutant flag) as (r, g, b) in 0..1: ONE rule for the mesh's faces (`live_movie`, `render_vtk`) and the `clones:` curve, so a clone is the same colour on the cell and on its band. `pal` (hex strings) colours 0, 1, ... in order; any other label takes a hue from splitmix64 of its number (a golden-ratio step would give neighbours on the seed's Fibonacci sphere, whose indices differ by Fibonacci numbers, near-equal hues)."""
    import colorsys
    if pal and 0 <= int(k) < len(pal):
        import matplotlib.colors as _mc
        return tuple(_mc.to_rgb(pal[int(k)]))
    x = (int(k) + 1) * 0x9E3779B97F4A7C15 % 2 ** 64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) % 2 ** 64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) % 2 ** 64
    x ^= x >> 31
    return colorsys.hsv_to_rgb((x & 0xFFFF) / 65536.0, 0.45 + 0.5 * ((x >> 16) & 0xFF) / 255.0,
                               0.65 + 0.33 * ((x >> 24) & 0xFF) / 255.0)


# muller_bands: A Muller plot's bands from per-frame integer labels -> (S [T, top + 1, 2], ids, nsurv).
def muller_bands(frames, top=12):
    """A Muller plot's bands from per-frame integer labels (one array of live cells' labels per frame,
    None where unrecorded). Returns (S, ids, nsurv): S [T, top + 1, 2] of (centre, half-width) in % of
    the live cells -- band j spans centre -/+ half-width, the `top` labels with the largest PEAK share
    over the clip stacked bottom-up in that order, the last row every other label together -- `ids` the
    labels of the first rows, `nsurv` [T] the number of distinct labels present. A (centre, half-width)
    pair is what the curve panel's +-SD band draws, so the Muller plot needs no chart of its own."""
    peak = {}
    for a in frames:
        if a is None or not len(a):
            continue
        u, c = np.unique(a, return_counts=True)
        for i, f in zip(u.tolist(), (c / len(a)).tolist()):
            if f > peak.get(i, 0.0):
                peak[i] = f
    ids = sorted(peak, key=lambda i: (-peak[i], i))[:int(top)]
    T = len(frames)
    S = np.full((T, len(ids) + 1, 2), np.nan)
    nsurv = np.zeros(T, int)
    for t, a in enumerate(frames):
        if a is None or not len(a):
            continue
        nsurv[t] = len(np.unique(a))
        lo = 0.0
        for j, i in enumerate(ids):
            f = 100.0 * float(np.mean(a == i))
            S[t, j] = (lo + f / 2, f / 2)
            lo += f
        S[t, -1] = ((lo + 100.0) / 2, (100.0 - lo) / 2)
    return S, ids, nsurv


# curve_row: One row [nt, 2] of (mean, sd) for a curve quantity at a level's current frame, live or replayed -> array.
def curve_row(H, lvl, q, ntype, nt, cell_cols):
    """[nt, 2] of (mean, sd) for `q` at the level's CURRENT frame: one row of `_curve_series`. The same body serves the replay (which sets `lvl.t` and calls this per recorded frame with numpy arrays behind `lvl.get`) and the live pass (once per frame, CUDA tensors behind the same accessors), so everything feeding numpy or a CPU geometry call goes through `_np`, a no-op on an array and `.detach().cpu().numpy()` on a tensor."""
    import torch
    def _np(v):
        return v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
    row = np.full((nt, 2), np.nan)
    if isinstance(q, str) and q.startswith("jacobian:"):
        # an MPM part's volume relative to its rest: det F per particle, F being the deformation gradient the strain operator keeps on the level (never in the trajectory, so a replay reads nan)
        try:
            lv = H.level(q[len("jacobian:"):])
            F = getattr(lv, "F", None)
            if F is None:
                return row
            J = np.linalg.det(_np(F).reshape(-1, F.shape[-2], F.shape[-1]).astype(np.float64))
            occ = getattr(lv, "occ", None)
            live = _np(occ).astype(bool) if occ is not None else np.ones(J.shape[0], bool)
            v = J[live[: J.shape[0]]]
            if v.size:
                row[0] = (float(v.mean()), float(v.std()))
        except Exception:                                # noqa: BLE001
            return row
        return row
    if isinstance(q, str) and q.startswith(("cell_volume:", "cell_area:")):
        # the size of each cell of material points (exp 21: does division halve the cells, does the sheet squeeze them): a cell is the points sharing a parent id (the `pid` block, rewritten by a division and recorded every row, else the set's `parent`), its size the convex hull of those points, as volume or as the footprint area in the x-y plane (the sheet's); positions only, so the replay draws what the live pass draws (the hull is a little inside the drawn body)
        kind, name = q.split(":", 1)
        try:
            from scipy.spatial import ConvexHull
            lv = H.level(name)
            if hasattr(lv, "_pos"):                      # a replay level: follow the frame the series loop is on
                lv.t = int(getattr(lvl, "t", 0))
            X = _np(lv.get("pos")).astype(np.float64)
            pid = lv.get("pid") if "pid" in (getattr(lv, "state_schema", None) or getattr(lv, "_blocks", {})) else None
            pid = _np(pid).reshape(-1) if pid is not None else _np(lv.parent).reshape(-1)
            occ = getattr(lv, "occ", None)
            live = _np(occ).reshape(-1).astype(bool) if occ is not None else np.ones(X.shape[0], bool)
            live &= np.isfinite(X).all(1)
            X, pid = X[live], np.rint(pid[: live.shape[0]][live]).astype(np.int64)
            order = np.argsort(pid, kind="stable")
            X, pid = X[order], pid[order]
            cuts = np.flatnonzero(np.diff(pid)) + 1
            vals = []
            for blk in np.split(np.arange(pid.size), cuts):
                P = X[blk] if kind == "cell_volume" else X[blk][:, :2]
                if P.shape[0] < P.shape[1] + 2:
                    continue
                try:
                    vals.append(ConvexHull(P).volume)       # in 2D, scipy's `volume` is the area
                except Exception:                        # noqa: BLE001 -- a flat or degenerate cell
                    continue
            if vals:
                row[0] = (float(np.mean(vals)), float(np.std(vals)))
        except Exception:                                # noqa: BLE001
            return row
        return row
    if isinstance(q, str) and q.startswith("strain:"):
        # shape change without the rotation: E = (F^T F - I) / 2 is zero for any rigid motion, so its Frobenius norm per particle, averaged, is the deformation a turning part actually suffers (whether a rotor's parts deform or only look as if)
        try:
            lv = H.level(q[len("strain:"):])
            F = getattr(lv, "F", None)
            if F is None:
                return row
            Fn = _np(F).reshape(-1, F.shape[-2], F.shape[-1]).astype(np.float64)
            Cg = np.einsum("nji,njk->nik", Fn, Fn)
            E = 0.5 * (Cg - np.eye(Fn.shape[-1])[None])
            e = np.sqrt((E ** 2).sum(axis=(1, 2)))
            occ = getattr(lv, "occ", None)
            live = _np(occ).astype(bool) if occ is not None else np.ones(e.shape[0], bool)
            v = e[live[: e.shape[0]]]
            if v.size:
                row[0] = (float(np.sqrt((v ** 2).mean())), float(v.std()))
        except Exception:                                # noqa: BLE001
            return row
        return row
    if isinstance(q, str) and q.startswith("rate:"):
        # the rotation rate of a set about the world z axis through the box centre, from its own velocities: mean of (r x v).z / r^2 over live points farther than a few cells from the axis, in turns per unit time (the units table converts 1/time to Hz through `time_s`)
        try:
            lv = H.level(q[len("rate:"):])
            _t0 = getattr(lv, "t", None)                 # a replay level: read it at the curves' frame
            if _t0 is not None and hasattr(lv, "_pos") and lv is not lvl:
                lv.t = int(getattr(lvl, "t", _t0))
            try:
                X = _np(lv.get("pos")).astype(np.float64); V = _np(lv.get("vel")).astype(np.float64)
            finally:
                if _t0 is not None and hasattr(lv, "_pos") and lv is not lvl:
                    lv.t = _t0
            occ = getattr(lv, "occ", None)
            live = _np(occ).astype(bool) if occ is not None else np.ones(X.shape[0], bool)
            c = np.asarray(getattr(H, "world", [1.0, 1.0, 1.0]), float)[:2] * 0.5 if hasattr(H, "world") else np.array([0.5, 0.5])
            rx, ry = X[:, 0] - c[0], X[:, 1] - c[1]
            r2 = rx * rx + ry * ry
            keep = live & (r2 > 1e-4)
            w = (rx[keep] * V[keep, 1] - ry[keep] * V[keep, 0]) / r2[keep]
            if w.size:
                row[0] = (float(w.mean()) / (2.0 * np.pi), float(w.std()) / (2.0 * np.pi))
        except Exception:                                # noqa: BLE001
            return row
        return row
    if isinstance(q, str) and q.startswith(("block:", "total:")):
        # `total:<set>:<block>` is the same block SUMMED over the live elements instead of averaged (how much of a protein the whole tissue holds, exp 11), with no spread; `block:cell:psi` is the level of a state block, the mean and sd of its first column over the set's live elements at the current frame (a membrane potential, a proton count, a stator's stretch as a curve without registering a quantity for each); bare units, a spec puts a `factor:` on the curve to draw mV or protons
        _, name, block = q.split(":", 2)
        # `total:<set>:<block>@<weight>`: a density times the element's own size (`total:cell:itg_A@area` is an integrin density per unit area, summed as an AMOUNT)
        block, _, wblock = block.partition("@")
        if name not in getattr(H, "levels", {}):
            # the cell set on a replay is not a level of its own: its width-1 blocks reach the renderer as the mesh's cell columns (`cell_cols`, as `phase` does), one value per live face
            _m = getattr(lvl, "mesh", None)
            _nF = int(_m.get("nF", 0) or 0) if _m is not None else 0
            _cc = cell_cols(H, lvl, _nF) if _nF else {}
            if block not in _cc:
                return row
            v = np.asarray(_np(_cc[block]), float).reshape(-1)[:_nF]
            if wblock:
                if wblock not in _cc:
                    return row
                v = v * np.asarray(_np(_cc[wblock]), float).reshape(-1)[:_nF]
            if q.startswith("total:"):
                row[0] = (float(np.nansum(v)), 0.0)
            elif v.size:
                row[0] = (float(np.nanmean(v)), float(np.nanstd(v)))
            return row
        try:
            lv = H.level(name)
            # on the replay, follow the frame the curves are at: the series loop advances `t` on the level the curves read (`lvl`) and no other, so a replay level asked for its block served row 0 forever (the movie's membrane-potential panel was a flat 150 mV while the trajectory held 150 -> 143, exp_02 step 0045); same rule as `count:` below
            _t0 = getattr(lv, "t", None)
            if _t0 is not None and hasattr(lv, "_pos") and lv is not lvl:
                lv.t = int(getattr(lvl, "t", _t0))
            try:
                val = _np(lv.get(block))
                wval = _np(lv.get(wblock)) if wblock else None
            finally:
                if _t0 is not None and hasattr(lv, "_pos") and lv is not lvl:
                    lv.t = _t0
        except Exception:                                # noqa: BLE001
            return row
        if val.ndim == 3:                                # a replay keeps [T, n, w]; the level's t picks the row
            val = val[int(getattr(lvl, "t", getattr(lv, "t", 0)))]
        occ = getattr(lv, "_occ", None)
        if occ is not None and getattr(occ, "ndim", 1) == 2:
            occ = occ[int(getattr(lvl, "t", getattr(lv, "t", 0)))]
        if occ is None:
            occ = getattr(lv, "occ", None)
        live = _np(occ).astype(bool) if occ is not None else np.ones(val.shape[0], bool)
        v = np.asarray(val, float)[live][:, 0] if val.ndim == 2 else np.asarray(val, float)[live]
        if wval is not None:
            if wval.ndim == 3:
                wval = wval[int(getattr(lvl, "t", getattr(lv, "t", 0)))]
            v = v * (np.asarray(wval, float)[live][:, 0] if wval.ndim == 2 else np.asarray(wval, float)[live])
        if q.startswith("total:"):
            row[0] = (float(np.nansum(v)), 0.0)
        elif v.size:
            row[0] = (float(v.mean()), float(v.std()))
        return row
    if isinstance(q, str) and q.startswith("species:"):
        # the twin of `phase`, for strains: `quantity: species:cell` draws one line per column of the set's `chem` block, counting the live elements whose largest column is that one, in the spec's `plotting.species` colours, so panel and picture name the strains alike (`phase` partitions a tissue by cycle stage and needs a mesh; this partitions any set, a mesh or a lattice of sites, by strain: exp 15's colonies)
        try:
            lv = H.level(q[len("species:"):])
            _t0 = getattr(lv, "t", None)                 # a replay level: read it at the curves' frame
            if _t0 is not None and hasattr(lv, "_pos") and lv is not lvl:
                lv.t = int(getattr(lvl, "t", _t0))
            try:
                val = _np(lv.get("chem"))
            finally:
                if _t0 is not None and hasattr(lv, "_pos") and lv is not lvl:
                    lv.t = _t0
        except Exception:                                # noqa: BLE001
            return row
        if val.ndim == 3:
            val = val[int(getattr(lvl, "t", getattr(lv, "t", 0)))]
        occ = getattr(lv, "_occ", None)
        if occ is not None and getattr(occ, "ndim", 1) == 2:
            occ = occ[int(getattr(lvl, "t", getattr(lv, "t", 0)))]
        if occ is None:
            occ = getattr(lv, "occ", None)
        live = _np(occ).astype(bool) if occ is not None else np.ones(val.shape[0], bool)
        val = np.asarray(val, float).reshape(val.shape[0], -1)[live[: val.shape[0]]]
        lab = np.argmax(val, axis=1) if val.size else np.zeros(0, int)
        for j in range(min(nt, val.shape[1])):
            row[j] = (float((lab == j).sum()), 0.0)
        return row
    if isinstance(q, str) and q.startswith("count:"):
        # the live count of a named set, mesh or not: `quantity: count:integrin` (live, `occ` is the engine's occupancy vector; on replay `_ReplayLevel._occ` is [T, n] and `t` picks the row)
        name = q[len("count:"):]
        species = None
        if ":" in name:                                  # `count:protein:integrin` -- one species of a typed set
            name, species = name.split(":", 1)
        try:
            lv = H.level(name)
        except Exception:                                # noqa: BLE001
            return row
        occ = getattr(lv, "_occ", None)
        frame = int(getattr(lvl, "t", getattr(lv, "t", 0)))
        if occ is not None and getattr(occ, "ndim", 1) == 2:
            # the replay advances `t` on the level the curves read (the mesh set); a counted set follows that frame, not its own `t`, which the series loop never touches
            occ = occ[frame]
        if occ is None:
            occ = getattr(lv, "occ", None)
        if occ is None:
            n = float(getattr(lv, "n", 0))
        else:
            live = _np(occ).astype(bool)
            if species is not None:
                names = list(getattr(lv, "type_names", None) or [])
                if species not in names:
                    raise ValueError(f"count:{name}:{species}: {name!r} has no species {species!r} (types: {names})")
                # the type column at the same frame as the occupancy: a reserve slot is typed 0 until an operator wakes it with its species, and `lv.node_type` reads the level's OWN `t`, which the series loop never advances, so every slot born after frame 0 was counted under type 0 (7,910 "nuclei" for 670 cells); use the per-frame column, when the archive kept one, at `frame`
                ntt = getattr(lv, "_node_type_t", None)
                nt = ntt[frame] if ntt is not None else getattr(lv, "node_type", None)
                if nt is None:
                    return row
                live = live & (_np(nt).astype(int) == names.index(species))
            n = float(live.sum())
        row[0] = (n, 0.0)
        return row
    m = getattr(lvl, "mesh", None)
    if q == "phase" and (m is None or not int(m.get("nF", 0) or 0)):
        # cells without a mesh (exp 21's material-point cells, `cell_cycle[model: mpm]`): the `phase` block of the drawn set's parent, the cell set, at the series loop's frame, as a percentage of its live cells in G1 / S / G2 / M, the same four series as the mesh path
        try:
            lv = H.level(getattr(lvl, "parent_name", None) or "cell")
        except Exception:                                # noqa: BLE001
            return row
        if hasattr(lv, "_pos"):
            lv.t = int(getattr(lvl, "t", 0))
        v = lv.get("phase")
        if v is None:
            return row
        a = np.rint(_np(v).reshape(-1)).astype(int)
        occ = getattr(lv, "occ", None)
        if occ is not None:
            a = a[_np(occ).reshape(-1)[: a.size].astype(bool)]
        for j in range(min(nt, 4)):
            row[j] = (100.0 * float(np.mean(a == j)) if a.size else np.nan, 0.0)
        return row
    if m is None or not int(m.get("nF", 0) or 0):
        return row
    _es, _et, _ef = (torch.as_tensor(_np(m[k])) for k in ("E_srce", "E_trgt", "E_face"))
    nF = int(m["nF"])
    # the cell set's blocks on the curve path too: `phase` is a block on the cell set, not a face column, and this is the only reader of it outside `_mesh_face_rgb`; without the overlay the panel that shows what fraction of the tissue is in G1 / S / G2 / M would draw four empty series and look like a run with no cycle
    m = _MeshView(m, cell_cols(H, lvl, nF))
    k = (np.zeros(nF, int) if ntype is None
         else np.asarray(ntype)[np.clip(np.arange(nF), 0, len(ntype) - 1)].astype(int))
    if q == "cells":
        for j in range(nt):
            row[j] = (float((k == j).sum()), 0.0)
        return row
    if q == "phase":
        # percent of the population in each phase, four series in one panel: a cycle is a DISTRIBUTION (a tissue cycling steadily and one frozen in G1 both just grow), and the four fractions sum to 100 at every frame, so the panel is its own check; it ignores `ntype` (splitting phases by cell type as well would be sixteen series)
        v = m.get("phase")
        if v is None:
            return row
        a = np.rint(np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach")
                               else v, float).ravel()[:nF]).astype(int)
        for j in range(min(nt, 4)):
            row[j] = (100.0 * float(np.mean(a == j)) if a.size else np.nan, 0.0)
        return row
    if q == "cycle_progress":
        v = m.get("cycle_progress")
        if v is None:
            return row
        vv = np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v,
                        float).ravel()[:nF]
        for j in range(nt):
            sel = k == j
            if sel.any():
                row[j] = (float(np.nanmean(vv[sel])), float(np.nanstd(vv[sel])))
        return row
    if q == "myosin":
        v = m.get("e_myo")
        if v is None:
            return row
        ef = _np(m["E_face"]); live = ef < nF
        vv = np.asarray(v, float)[live]
        ke = k[np.clip(ef[live].astype(int), 0, nF - 1)]
        for j in range(nt):
            sel = ke == j
            if sel.any():
                row[j] = (float(np.nanmean(vv[sel])), float(np.nanstd(vv[sel])))
        return row
    import torch
    if q == "radius":
        nv = int(m["Nv"])
        P = _np(lvl.get("pos")[:nv])
        r = np.linalg.norm(P - P.mean(0), axis=1)
        for j in range(nt):                   # per VERTEX, so the type split is by face
            row[j] = (float(np.mean(r)), float(np.std(r)))
        return row
    from plexus.operators.vertex_ops import face_geometry_3d, wedge_apex
    nv = int(m["Nv"])
    pos = torch.as_tensor(_np(lvl.get("pos")[:nv]), dtype=torch.float64)
    a, _p, _c, _v = face_geometry_3d(pos, _es, _et, _ef, nF, apex=wedge_apex(m, pos))
    a = a.numpy()
    # `shape_index`: how round a cell is, perimeter / sqrt(area), dimensionless, of the same mid-surface polygon `area` reads (circle 3.545, regular hexagon 3.722, pentagon 3.812, square 4.0; lower = rounder): the vertex model's `p0` targets it, so the panel shows the energy's own shape variable relaxing
    if q == "shape_index":
        a = _p.numpy() / np.sqrt(np.maximum(a, 1e-12))
    # `area` follows `volume`'s rule, for the same reason: `face_geometry_3d` returns the MID-surface area, which on an apico-basal run bounds nothing (the cell's surface is the polyhedron's: two caps and one wall per ring edge, the `S` that `kappa_s` integrates), so a panel labelled "cell area" would show the area of a surface the cell does not have; falls back to the mid-surface when the run carries no `sep`, where it IS the cell
    if q == "area":
        _sa = lvl.get("sep")
        if _sa is not None and int(_sa.shape[0]) >= nv:
            from plexus.operators.vertex_ops import apicobasal_geometry_3d
            _ss = torch.as_tensor(np.asarray(_sa[:nv].detach().cpu()
                                             if hasattr(_sa, "detach") else _sa[:nv]),
                                  dtype=torch.float64)
            a = apicobasal_geometry_3d(pos, _ss, _es, _et, _ef, nF)[1].numpy()
    if q == "volume":
        # the volume the energy defends, which on an apico-basal run is not `_v`: `face_geometry_3d` returns the origin-referenced WEDGE volume (the cone from the world origin to the mid-surface ring), what `cell_grow` scales and `cell_divide` triggers on, but it rises when the SHELL's radius rises with the cell unchanged (a tissue dividing without growing); `cell_mechanics[model: apicobasal]` defends the polyhedron (two caps and one wall per ring edge, by the divergence theorem), which is what a plot labelled "cell volume" must show; falls back to the wedge when the run carries no `sep`
        try:
            _sp = lvl.get("sep")
        except Exception:                                # noqa: BLE001
            _sp = None
        if _sp is not None and int(_sp.shape[0]) >= nv:
            from plexus.operators.vertex_ops import apicobasal_geometry_3d
            _s = torch.as_tensor(np.asarray(_sp[:nv].detach().cpu()
                                            if hasattr(_sp, "detach") else _sp[:nv]),
                                 dtype=torch.float64)
            a = apicobasal_geometry_3d(pos, _s, _es, _et, _ef, nF)[0].numpy()
        else:
            a = _v.numpy()
    for j in range(nt):
        sel = k == j
        if sel.any():
            row[j] = (float(np.nanmean(a[sel])), float(np.nanstd(a[sel])))
    return row


# _occ: The live mask of a set at frame fr, or None -> bool [n] | None.
def _occ(d, key, fr):
    o = d.get(key)
    return None if o is None else o[fr].astype(bool)


# checkpoint_metrics: Everything the fingerprint records at one frame (keys absent from the run are absent) -> dict.
def checkpoint_metrics(d, fr: int) -> dict:
    """Everything the fingerprint records at one frame. Keys absent from the run are absent here."""
    m: dict = {}
    files = set(d.files)
    # a vertex tissue
    if "vertex__pos" in files:
        occ = _occ(d, "vertex__occ", fr)
        P = d["vertex__pos"][fr]
        P = P[occ] if occ is not None else P
        if P.shape[0]:
            c = P.mean(0)
            r = np.linalg.norm(P - c, axis=1)
            m["r_med"] = float(np.median(r))
            m["r_p10"] = float(np.percentile(r, 10))
            m["r_p90"] = float(np.percentile(r, 90))
            m["roughness"] = float(r.std() / max(r.mean(), 1e-12))
            m["centroid"] = [float(v) for v in c]
            m["vertices"] = int(P.shape[0])
            # out-of-plane spread, the sheet's own number; on a shell it is just the radius
            m["z_sd"] = float(P[:, -1].std())
        if "cell__occ" in files:
            m["cells"] = int(d["cell__occ"][fr].sum())
        if "vertex__mesh_offsets" in files:
            off = d["vertex__mesh_offsets"]
            if fr + 1 < off.shape[0]:
                m["half_edges"] = int(off[fr + 1] - off[fr])   # live half-edges, from the mesh table
        if "cell__area" in files and "cell__occ" in files:
            a = d["cell__area"][fr].reshape(-1)
            co = d["cell__occ"][fr].astype(bool)
            a = a[co] if a.shape[0] == co.shape[0] else a
            if a.size:
                m["area_mean"] = float(a.mean())
        for k in ("cell__ndiv", "cell__age"):
            if k in files and "cell__occ" in files:
                v = d[k][fr].reshape(-1)
                co = d["cell__occ"][fr].astype(bool)
                v = v[co] if v.shape[0] == co.shape[0] else v
                if k == "cell__ndiv" and v.size:
                    m["ndiv_sum"] = int(v.sum())   # sum of per-cell generation counts, monotone
        for k in ("cell__chem", "vertex__chem"):
            if k in files:
                v = d[k][fr]
                oc = _occ(d, k.split("__")[0] + "__occ", fr)
                v = v[oc] if oc is not None and v.shape[0] == oc.shape[0] else v
                if v.size and float(np.abs(v).max()) > 0.0:   # an all-zero chem block is unused
                    m["chem_max"] = float(v[:, 0].max())
                    m["chem_min"] = float(v[:, 0].min())
    # an MPM body
    mp = [k[:-5] for k in files if k.endswith("__pos") and k != "vertex__pos"]
    if mp:
        mset = "mpm_particle" if "mpm_particle" in mp else sorted(mp)[0]
        m["mpm_set"] = mset
        occ = _occ(d, mset + "__occ", fr)
        X = d[mset + "__pos"][fr]
        X = X[occ] if occ is not None else X
        if X.shape[0]:
            m["mpm_n"] = int(X.shape[0])
            m["mpm_spread"] = float(X.std(0).mean())
            m["mpm_bbox"] = float((X.max(0) - X.min(0)).mean())
            m["mpm_centroid"] = [float(v) for v in X.mean(0)]
            if "vertex__pos" in files and "r_med" in m:
                c = np.asarray(m["centroid"])
                m["mpm_inside"] = int((np.linalg.norm(X - c, axis=1) < m["r_med"]).sum())
    return m


# deaths_by: Cells that have left the live set since frame 0, when the run has a cell set -> int | None.
def deaths_by(d, fr: int) -> int | None:
    """Cells that have left the live set since frame 0, when the run has a cell set."""
    if "cell__occ" not in d.files:
        return None
    return int(max(0, int(d["cell__occ"][0].sum()) - int(d["cell__occ"][fr].sum())))

for _k in ("cells", "half_edges", "vertices", "r_med", "r_p10", "r_p90", "roughness", "z_sd", "area_mean",
           "ndiv_sum", "deaths_or_net_loss", "chem_max", "chem_min", "mpm_n", "mpm_spread", "mpm_bbox", "mpm_inside"):
    _dim = {"cells": "count", "half_edges": "count", "vertices": "count", "r_med": "length", "r_p10": "length",
            "r_p90": "length", "z_sd": "length", "area_mean": "area", "mpm_n": "count", "mpm_inside": "count",
            "ndiv_sum": "count", "deaths_or_net_loss": "count", "mpm_bbox": "length", "mpm_spread": "length"}.get(_k)
    if "fp:" + _k not in MEASURES:
        register("fp:" + _k, "frame", checkpoint_metrics, _dim, "fingerprint checkpoint metric")


# the gate rows register themselves on import of the package's copy, so the registry is complete whichever consumer imports this module first
from plexus import measures_rows as _rows  # noqa: E402,F401
from plexus import measures_live as _live  # noqa: E402,F401  (the differentiable measures of the live state, `kind="live"`)
