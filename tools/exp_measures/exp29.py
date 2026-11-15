"""exp29 cell_tissue_2 -- the rulers: does an MPM twin show what its vertex original shows?

    exp29.health     intact / finite / pool_fill of one run (a vertex run or an MPM run)
    exp29.growth     doubling_norm: time for the live cell count to double from the start of growth, in units of the
                     time the spec's own growth law takes to double a cell's volume at its full rate; `doubling_gap`
    exp29.chem       spots per 100 cells and the activator-high fraction, on the cell-contact graph; `pattern_gap`
    exp29.gate       division rate of activator-high cells over that of low cells; `div_contrast_gap`
    exp29.shape      protr_final (95th-percentile cell radius over the median) and the activator-radius correlation;
                     `excursion_gap`, `corr_act_rad_abs`
    exp29.regime     persistence of the activator-high set from the cell count's first growth to the end;
                     `persistence_order` against the vertex pair
    exp29.tension    protr_final with the interface tension on against off; `sign_match`
    exp29.death      the share of dead cells whose activator was below the clip's own threshold; `low_activator_gap`

ONE RULER READS TWO BODIES. A vertex run (`CoreTraj`: faces are cells, a half-edge table says who touches whom) and an MPM run
(a `cell` set of slots that own blocks of material points) both record the cell set as `cell__occ`, `cell__chem` and a position
(`cell__centroid` for the mesh, `cell__pos` for the points). Every number below is computed from those, the live mask, and a
contact graph (shared faces on the mesh; points within the contact distance, else the Delaunay of the centroids, for points), so a
gap between the two bodies is a gap in the biology and not in two rulers.

THE REFERENCE IS THE VERTEX TWIN, RE-RUN. `experiments/exp29_cell_tissue_2/data/vertex_references.json` is WRITTEN BY THIS
MODULE (`build_references`, `tools/exp29_references.py`) from the seeds of `tissue/exp29_v_<clip>_s{0,1,2}`: per clip and per
quantity, the seed values, their mean and their spread; and the `own_gap` of the quantity (the largest gap of one vertex seed to
the mean of the seeds, in the same normalised units the gap measures use), which sets the gates' bands. A `*_gap` key is the
run's quantity against that mean. Nothing in this file is a typed target.

THE CLIP IS IN THE RUN'S NAME (`exp29_<v|m>_<clip>_s<seed>`); the threshold `a_sw`, the death channel and the growth rate come
from the run's own `spec.yaml`.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import math
import os
import re

import numpy as np

from .common import ROOT, finite, open_run, register_run

EXP = os.path.join(ROOT, "experiments", "exp29_cell_tissue_2")
REFS = os.path.join(EXP, "data", "vertex_references.json")
CLIPS = "grow_divide r021_12 r005_06 r013_05 r016_01 r017_00_ctrl r020_01 r020_00_ctrl r021_06 tsd_max sc_antiphase".split()
A_SW = 0.35                      # the clips' own growth threshold (cell_grow.a_sw), the spot threshold
MIN_BIRTHS = 10                  # a division contrast needs at least this many divisions behind it
WINDOW = 0.2                     # the last fifth of the rows is "the end of the run"
N0_FACTOR = 1.5                  # persistence is read from the row where the cell count first reaches 1.5 x its start
TENSION_PAIRS = {"r013_05": "r017_00_ctrl", "r016_01": "r017_00_ctrl"}    # on -> off (the clips' captions)
REGIME_PARTNER = {"r020_01": "r021_06", "r021_06": "r020_01"}


# ===================================================================================== the run, as numbers
def name_of(T) -> str:
    return os.path.basename(T.dir.rstrip("/"))


def parse(name: str):
    """(body 'v' or 'm', clip, seed) from `exp29_<v|m>_<clip>_s<seed>`, else (None, None, None)."""
    m = re.match(r"exp29_([vm])_(.+?)_s(\d+)$", name)
    return (m.group(1), m.group(2), int(m.group(3))) if m else (None, None, None)


def _op(spec: dict, *names: str) -> dict:
    for o in (spec or {}).get("operators") or []:
        if isinstance(o, dict) and o.get("op") in names:
            return o
    return {}


class Cells:
    """The recorded cell set of one run, as arrays. Rows are the recorded frames; slots the buffer."""

    def __init__(self, T):
        self.T = T
        z = T.z
        self.z = z
        self.spec = getattr(T, "spec", {}) or {}
        self.body = "v" if "vertex__mesh_nF" in z.files else "m"
        self.occ = np.asarray(z["cell__occ"], bool)
        self.chem = (np.asarray(z["cell__chem"], np.float32) if "cell__chem" in z.files
                     else np.zeros(self.occ.shape + (2,), np.float32))        # a run with no chemistry: activator 0 everywhere
        self.cen = np.asarray(z["cell__centroid"] if "cell__centroid" in z.files else z["cell__pos"], np.float32)
        self.rows = self.occ.shape[0]
        g = self.spec.get("general") or {}
        nf, cap = int(g.get("n_frames") or 0), int(g.get("record_cap") or 0)
        # frames between two recorded rows: the engine's own stride (`engine.py`, `sstride`), not the spec's `record_every`,
        # which the engine never reads and an MPM spec leaves unset (a 5-frame stride was read as 1: births_rate 5x too high)
        self.every = max(1, (nf + cap) // cap) if nf and cap else int(g.get("record_every", 1) or 1)
        self.dt = float(g.get("dt", 1.0) or 1.0)
        self._cache = {}

    # ---- time
    def frame(self, t) -> float:
        return float(t * self.every)

    def time(self, t) -> float:
        return self.frame(t) * self.dt

    def n(self) -> np.ndarray:
        return self.occ.sum(1)

    def window(self) -> range:
        k = max(3, int(round(WINDOW * self.rows)))
        return range(self.rows - k, self.rows)

    # ---- one row
    def live(self, t) -> np.ndarray:
        return np.flatnonzero(self.occ[t])

    def activator(self, t, slot=None, chan=0):
        a = self.chem[t][:, chan] if chan < self.chem.shape[2] else np.zeros(self.occ.shape[1], np.float32)
        return a if slot is None else a[slot]

    def graph(self, t):
        """(slot [n], pairs [m, 2] into slot): which live cells touch."""
        if t in self._cache:
            return self._cache[t]
        slot = self.live(t)
        pairs = np.zeros((0, 2), np.int64)
        if len(slot) >= 4:
            if self.body == "v":
                from .common import cells as _cells, neighbour_pairs
                c = _cells(self.T, t)
                slot = c.slot
                pairs = neighbour_pairs(self.T, t, c)
            else:
                pairs = self._points_graph(t, slot)
        self._cache[t] = (slot, pairs)
        return slot, pairs

    def _points_graph(self, t, slot):
        z = self.z
        if "pt__pos" in z.files and "pt__pid" in z.files and "pt__occ" in z.files:
            from scipy.spatial import cKDTree
            occ = np.asarray(z["pt__occ"][t], bool)
            x = np.asarray(z["pt__pos"][t], np.float64)[occ]
            pid = np.asarray(z["pt__pid"][t])[occ].ravel().astype(np.int64)
            rc = float((_op(self.spec, "cell_neighbours") or {}).get("contact", 0.3))
            pr = cKDTree(x).query_pairs(rc, output_type="ndarray")
            if len(pr):
                a, b = pid[pr[:, 0]], pid[pr[:, 1]]
                m = a != b
                e = np.unique(np.sort(np.stack([a[m], b[m]], 1), axis=1), axis=0)
                row = -np.ones(self.occ.shape[1], np.int64)
                row[slot] = np.arange(len(slot))
                e = row[e]
                return e[(e >= 0).all(1)]
        from scipy.spatial import Delaunay
        x = self.cen[t][slot].astype(np.float64)
        tri = Delaunay(x)
        s = tri.simplices
        e = np.concatenate([s[:, [i, j]] for i in range(s.shape[1]) for j in range(i + 1, s.shape[1])])
        e = np.unique(np.sort(e, 1), axis=0)
        L = np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1)
        return e[L <= 1.8 * np.median(L)]


def cells_of(T) -> Cells:
    c = getattr(T, "_exp29_cells", None)
    if c is None:
        c = T._exp29_cells = Cells(T)
    return c


# ===================================================================================== primitives (pure numpy, tested)
def spots(high: np.ndarray, pairs: np.ndarray, min_size: int = 2) -> int:
    """Connected sets of at least `min_size` activator-high cells on the contact graph."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(high)
    if n == 0 or not high.any():
        return 0
    keep = high[pairs[:, 0]] & high[pairs[:, 1]] if len(pairs) else np.zeros(0, bool)
    e = pairs[keep] if len(pairs) else np.zeros((0, 2), np.int64)
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    size = np.bincount(lab[high])
    return int((size >= min_size).sum())


def protrusion(x: np.ndarray) -> float:
    """95th-percentile radius over the median, from the centroid of the cells (the campaign's own definition)."""
    r = np.linalg.norm(x - x.mean(0), axis=1)
    return float(np.percentile(r, 95) / (np.median(r) + 1e-9)) if r.size else 1.0


def _rank(a):
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    r[o] = np.arange(len(a))
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4 or np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return 0.0
    ra, rb = _rank(a), _rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def persistence(x0: np.ndarray, high0: np.ndarray, x1: np.ndarray, high1: np.ndarray) -> float:
    """How much of the early pattern the late pattern keeps: P(high now | the nearest early cell was high) - P(high now | low).

    Cells are matched by DIRECTION from each tissue's centroid (growth is isotropic, so a late cell sits over its ancestors),
    nearest early cell on the unit sphere; no lineage record is needed, so the same function reads both bodies."""
    d0 = x0 - x0.mean(0)
    d1 = x1 - x1.mean(0)
    d0 = d0 / (np.linalg.norm(d0, axis=1, keepdims=True) + 1e-12)
    d1 = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + 1e-12)
    from scipy.spatial import cKDTree
    _, j = cKDTree(d0).query(d1)
    anc = high0[j]
    if anc.all() or (~anc).all():
        return 0.0
    return float(high1[anc].mean() - high1[~anc].mean())


def sign_match(d, dv, spread) -> float:
    """1 when the twin's (on - off) difference has the vertex pair's sign AND is larger than the vertex seeds' own spread."""
    return 1.0 if (np.sign(d) == np.sign(dv) and abs(d) > spread) else 0.0


def normalised_gap(x, ref, floor=1e-3):
    if x is None or ref is None:
        return None
    return float(abs(x - ref) / max(abs(ref), floor))


# ===================================================================================== per-run statistics
def end_mean(c: Cells, fn):
    vals = [fn(t) for t in c.window()]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def stat_chem(c: Cells, a_sw=A_SW):
    def sp(t):
        slot, pairs = c.graph(t)
        if len(slot) < 4:
            return None
        high = c.activator(t, slot) > a_sw
        return 100.0 * spots(high, pairs) / len(slot)

    def hf(t):
        slot = c.live(t)
        return float((c.activator(t, slot) > a_sw).mean()) if len(slot) else None
    return {"spots_per_100": end_mean(c, sp), "high_fraction": end_mean(c, hf), "n_cells_end": int(c.n()[-1])}


def growth_law(spec: dict):
    """(rate g per unit time at which a cell's VOLUME grows at the full rate, declared by the spec) or None."""
    grow = _op(spec, "cell_grow")
    has_chem = bool(_op(spec, "cell_chem_react"))
    if grow:                                    # vertex: ds/dt = s rate (rho + Hill(a)), volume = s^3
        return 3.0 * float(grow["rate"]) * (float(grow.get("rho", 0.0)) + (1.0 if has_chem else 0.0)), int(grow.get("after_frame") or 0)
    cg = _op(spec, "cell_populate")
    if cg:                                      # MPM: points grow as exp(g t) with the cell type's rate
        rates = cg.get("rates")
        g = max(float(v) for v in rates.values()) if isinstance(rates, dict) else float(cg.get("rate", 0.0))
        return g, int(cg.get("after_frame") or 0)
    return None


def stat_growth(c: Cells):
    n = c.n().astype(float)
    law = growth_law(c.spec)
    after = law[1] if law else 0
    t0 = next((t for t in range(c.rows) if c.frame(t) >= after), 0)
    target = 2.0 * n[t0]
    t1 = next((t for t in range(t0, c.rows) if n[t] >= target), None)
    out = {"n_start": float(n[t0]), "n_end": float(n[-1]), "doublings": float(np.log2(n[-1] / max(n[t0], 1)))}
    if t1 is None:
        out.update(doubling_time=None, doubling_norm=None)
        return out
    dtm = c.time(t1) - c.time(t0)
    out["doubling_time"] = float(dtm)
    out["doubling_norm"] = float(dtm / (math.log(2) / law[0])) if law and law[0] > 0 else None
    return out


def births(c: Cells):
    """[(row, mother_slot, mother_row)] -- a slot occupied at row t that was not at t-1, with its mother at t-1."""
    out = []
    ids = c.z["cell__cell_id"] if "cell__cell_id" in c.z.files else None
    pid = c.z["cell__parent_id"] if "cell__parent_id" in c.z.files else None
    for t in range(1, c.rows):
        i0, i1 = _ids(c, t - 1), _ids(c, t)
        arrived = c.occ[t] & ~c.occ[t - 1]
        if i0 is not None:
            arrived = arrived | (c.occ[t] & c.occ[t - 1] & (i0 != i1))
        new = np.flatnonzero(arrived)
        if not len(new):
            continue
        prev = c.live(t - 1)
        if not len(prev):
            continue
        if ids is not None and pid is not None:
            cid = np.asarray(ids[t - 1]).ravel()
            look = {int(cid[s]): int(s) for s in prev}
            for s in new:
                m = look.get(int(np.asarray(pid[t]).ravel()[s]))
                if m is not None:
                    out.append((t, m, t - 1))
        else:
            from scipy.spatial import cKDTree
            d, j = cKDTree(c.cen[t - 1][prev]).query(c.cen[t][new])
            out += [(t, int(prev[k]), t - 1) for k in j]
    return out


def stat_gate(c: Cells, a_sw=A_SW):
    law = growth_law(c.spec)
    after = law[1] if law else 0
    t0 = next((t for t in range(c.rows) if c.frame(t) >= after), 0)
    b = [(t, s, tm) for (t, s, tm) in births(c) if t >= t0]
    hi_b = sum(1 for (_, s, tm) in b if c.activator(tm)[s] > a_sw)
    lo_b = len(b) - hi_b
    hi_cf = lo_cf = 0.0
    for t in range(max(t0, 1), c.rows):
        slot = c.live(t - 1)
        h = c.activator(t - 1, slot) > a_sw
        dtr = c.frame(t) - c.frame(t - 1)
        hi_cf += h.sum() * dtr
        lo_cf += (~h).sum() * dtr
    if hi_cf <= 0 or lo_cf <= 0 or hi_b + lo_b < MIN_BIRTHS:      # with no births the smoothed ratio only reads the cell-frames
        return {"div_contrast": None, "births_high": hi_b, "births_low": lo_b}
    return {"div_contrast": float(((hi_b + 0.5) / hi_cf) / ((lo_b + 0.5) / lo_cf)), "births_high": hi_b, "births_low": lo_b,
            "high_cell_frames": float(hi_cf), "low_cell_frames": float(lo_cf)}


def stat_shape(c: Cells):
    def prot(t):
        slot = c.live(t)
        return protrusion(c.cen[t][slot].astype(float)) if len(slot) >= 4 else None

    def cor(t):
        slot = c.live(t)
        if len(slot) < 4:
            return None
        x = c.cen[t][slot].astype(float)
        return spearman(c.activator(t, slot).astype(float), np.linalg.norm(x - x.mean(0), axis=1))
    return {"protr": end_mean(c, prot), "corr_act_rad": end_mean(c, cor)}


def stat_persistence(c: Cells, a_sw=A_SW):
    n = c.n().astype(float)
    t0 = next((t for t in range(c.rows) if n[t] >= N0_FACTOR * n[0]), None)
    if t0 is None:
        return {"persistence": None}
    s0, s1 = c.live(t0), c.live(c.rows - 1)
    p = persistence(c.cen[t0][s0].astype(float), c.activator(t0, s0) > a_sw, c.cen[c.rows - 1][s1].astype(float), c.activator(c.rows - 1, s1) > a_sw)
    return {"persistence": p, "row_start": t0}


def die_params(spec):
    d = _op(spec, "cell_die")
    if not d:
        return None
    return {"a_sw": float(d.get("a_sw", 0.25)), "chan": int(d.get("chan", 0) or 0), "after": int(d.get("after_frame") or 0)}


def _ids(c: Cells, t):
    return np.asarray(c.z["cell__cell_id"][t]).ravel() if "cell__cell_id" in c.z.files else None


def departed(c: Cells, t) -> np.ndarray:
    """Slots live at row t-1 whose cell is gone at row t: the slot is empty, or (when cells carry an id) holds another cell."""
    prev, cur = c.occ[t - 1], c.occ[t]
    i0, i1 = _ids(c, t - 1), _ids(c, t)
    gone = prev & ~cur
    if i0 is not None:
        gone = gone | (prev & cur & (i0 != i1))
    return np.flatnonzero(gone)


def stat_death(c: Cells):
    p = die_params(c.spec)
    if p is None:
        return {"n_deaths": 0, "low_activator_fraction": None}
    n_dead = low = 0
    for t in range(1, c.rows):
        gone = departed(c, t)
        if not len(gone):
            continue
        slot = c.live(t - 1)
        a = c.activator(t - 1, slot, p["chan"])
        thr = p["a_sw"] * float(a.max())
        n_dead += len(gone)
        low += int((c.activator(t - 1, gone, p["chan"]) < thr).sum())
    return {"n_deaths": n_dead, "low_activator_fraction": (low / n_dead) if n_dead else None,
            "death_fraction": n_dead / max(float(c.occ[0].sum()), 1.0)}


# ===================================================================================== references
def load_refs():
    return json.load(open(REFS)) if os.path.exists(REFS) else None


def ref(clip, group, key):
    r = load_refs()
    try:
        return r["clips"][clip][group][key]
    except (TypeError, KeyError):
        return None


def stat_all(T):
    """Every raw statistic of one run, grouped as the references are."""
    c = cells_of(T)
    body, clip, _ = parse(name_of(T))
    a_sw = float((_op(c.spec, "cell_grow") or {}).get("a_sw", A_SW) or A_SW) or A_SW
    return {"chem": stat_chem(c, a_sw), "growth": stat_growth(c), "gate": stat_gate(c, a_sw), "shape": stat_shape(c),
            "regime": stat_persistence(c, a_sw), "death": stat_death(c)}


GAP_RULES = {                                   # (group, key) -> how a gap is normalised
    ("chem", "spots_per_100"): lambda x, r: normalised_gap(x, r, 0.5),
    ("chem", "high_fraction"): lambda x, r: normalised_gap(x, r, 0.02),
    ("growth", "doubling_norm"): lambda x, r: normalised_gap(x, r, 0.05),
    ("gate", "div_contrast"): lambda x, r: normalised_gap(x, r, 0.5),
    ("shape", "protr"): lambda x, r: None if x is None or r is None else float(abs(x - r) / max(r - 1.0, 0.05)),
    ("shape", "corr_act_rad"): lambda x, r: None if x is None or r is None else float(abs(x - r) / max(abs(r), 0.2)),
    ("death", "low_activator_fraction"): lambda x, r: normalised_gap(x, r, 0.05),
}


def gap(clip, group, key, x):
    r = ref(clip, group, key)
    rule = GAP_RULES[(group, key)]
    return None if r is None else rule(x, r["mean"])


def build_references(seeds=(0, 1, 2), write=True):
    """Write `vertex_references.json` from the vertex twins on disk. Returns the dict."""
    out = {"source": "tools/exp_measures/exp29.py build_references over tissue/exp29_v_<clip>_s<seed>", "window": WINDOW,
           "a_sw": A_SW, "n0_factor": N0_FACTOR, "clips": {}}
    for clip in CLIPS:
        runs, stats = [], []
        for s in seeds:
            spec = f"tissue/exp29_v_{clip}_s{s}"
            try:
                from .common import landed_file
                if not os.path.exists(landed_file(spec)):
                    continue
                stats.append(stat_all(open_run(spec)))
                runs.append(spec)
            except Exception as e:                                          # noqa: BLE001
                out.setdefault("errors", {})[spec] = f"{type(e).__name__}: {e}"
        if not stats:
            continue
        d = {"runs": runs}
        for (group, key), rule in GAP_RULES.items():
            vals = [s[group].get(key) for s in stats]
            vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
            if not vals:
                continue
            m = float(np.mean(vals))
            own = max(rule(v, m) or 0.0 for v in vals) if len(vals) > 1 else None
            d.setdefault(group, {})[key] = {"values": vals, "mean": m, "spread": float(np.ptp(vals)) if len(vals) > 1 else None,
                                            "own_gap": own}
        for k in ("persistence",):
            vals = [s["regime"].get(k) for s in stats if s["regime"].get(k) is not None]
            if vals:
                d.setdefault("regime", {})[k] = {"values": vals, "mean": float(np.mean(vals)),
                                                 "spread": float(np.ptp(vals)) if len(vals) > 1 else None,
                                                 "own_gap": float(max(abs(v - np.mean(vals)) for v in vals)) if len(vals) > 1 else None}
        d["n_deaths"] = [s["death"]["n_deaths"] for s in stats]
        d["n_cells_end"] = [s["chem"]["n_cells_end"] for s in stats]
        out["clips"][clip] = d
    if write:
        os.makedirs(os.path.dirname(REFS), exist_ok=True)
        json.dump(out, open(REFS, "w"), indent=1)
    return out


# ===================================================================================== the registered measures
def _clip(T):
    return parse(name_of(T))[1]


def health(T, **_):
    z = T.z
    body = "v" if "vertex__mesh_nF" in z.files else "m"
    c = cells_of(T)
    rows = sorted(set(list(range(0, c.rows, max(1, c.rows // 20))) + [c.rows - 1]))
    ok = True
    for k in z.files:
        if k.split("__")[0] in ("cell", "vertex", "pt", "bm_node", "junction") and not k.endswith(("occ", "type_names", "parent_name")):
            a = z[k]
            if getattr(a, "dtype", None) is not None and a.dtype.kind == "f" and a.ndim >= 2 and a.shape[0] == c.rows:
                if not all(np.isfinite(np.asarray(a[t])).all() for t in rows):
                    ok = False
                    break
    fills = [float(c.occ.sum(1).max()) / c.occ.shape[1]]
    if body == "v" and "vertex__occ" in z.files:
        fills.append(float(np.asarray(z["vertex__occ"]).sum(1).max()) / z["vertex__occ"].shape[1])
    if "pt__occ" in z.files:
        fills.append(float(np.asarray(z["pt__occ"]).sum(1).max()) / z["pt__occ"].shape[1])
    pool = max(fills)
    pieces = None
    if "pt__pos" in z.files and "pt__pid" in z.files:
        pieces = _pieces(z, c.rows - 1, c.spec)
    intact = bool(ok and pool < 0.9 and (pieces is None or pieces >= 0.95) and int(c.n()[-1]) > 0)
    return {"intact": 1.0 if intact else 0.0, "finite": 1.0 if ok else 0.0, "pool_fill": finite(pool), "piece_fraction": finite(pieces),
            "n_cells_end": int(c.n()[-1]), "rows": c.rows}


def _pieces(z, t, spec):
    """Share of cells with at least 95 % of their points in one connected piece (points closer than 2.4 x their spacing)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    occ = np.asarray(z["pt__occ"][t], bool)
    x = np.asarray(z["pt__pos"][t], np.float64)[occ]
    pid = np.asarray(z["pt__pid"][t])[occ].ravel().astype(np.int64)
    seed = next((o for o in (spec.get("seed") or []) if isinstance(o, dict) and o.get("point_spacing")), {})
    sp = float(seed.get("point_spacing", 0.125))
    good = tot = 0
    for p in np.unique(pid):
        xs = x[pid == p]
        if len(xs) < 2:
            continue
        pr = cKDTree(xs).query_pairs(2.4 * sp, output_type="ndarray")
        g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(xs), len(xs)))
        _, lab = connected_components(g, directed=False)
        tot += 1
        good += int(np.bincount(lab).max() >= 0.95 * len(xs))
    return good / tot if tot else None


def growth(T, **_):
    s = stat_growth(cells_of(T))
    out = {k: finite(v) for k, v in s.items()}
    out["doubling_gap"] = finite(gap(_clip(T), "growth", "doubling_norm", s.get("doubling_norm")))
    return out


def chem(T, **_):
    s = stat_chem(cells_of(T))
    clip = _clip(T)
    out = {k: finite(v) for k, v in s.items()}
    g = [gap(clip, "chem", k, s[k]) for k in ("spots_per_100", "high_fraction")]
    out["spots_gap"], out["high_fraction_gap"] = finite(g[0]), finite(g[1])
    out["pattern_gap"] = finite(np.mean(g)) if all(v is not None for v in g) else None
    return out


def gate(T, **_):
    s = stat_gate(cells_of(T))
    out = {k: finite(v) for k, v in s.items()}
    out["div_contrast_gap"] = finite(gap(_clip(T), "gate", "div_contrast", s.get("div_contrast")))
    return out


def shape(T, **_):
    s = stat_shape(cells_of(T))
    clip = _clip(T)
    out = {"protr_final": finite(s["protr"]), "corr_act_rad": finite(s["corr_act_rad"]),
           "corr_act_rad_abs": finite(abs(s["corr_act_rad"])) if s["corr_act_rad"] is not None else None}
    g = [gap(clip, "shape", "protr", s["protr"]), gap(clip, "shape", "corr_act_rad", s["corr_act_rad"])]
    out["protr_gap"], out["corr_gap"] = finite(g[0]), finite(g[1])
    out["excursion_gap"] = finite(np.mean(g)) if all(v is not None for v in g) else None
    return out


def regime(T, **_):
    s = stat_persistence(cells_of(T))
    clip = _clip(T)
    out = {"persistence": finite(s.get("persistence"))}
    partner = REGIME_PARTNER.get(clip)
    me, other = ref(clip, "regime", "persistence"), ref(partner, "regime", "persistence") if partner else None
    if s.get("persistence") is not None and me and other:
        out["persistence_order"] = 1.0 if np.sign(s["persistence"] - other["mean"]) == np.sign(me["mean"] - other["mean"]) else 0.0
    else:
        out["persistence_order"] = None
    return out


def tension(T, partner=None, **_):
    """protr of this run against its partner's (same body, same seed) and the sign of (tension on) - (tension off)."""
    clip = _clip(T)
    body, _, seed = parse(name_of(T))
    mine = stat_shape(cells_of(T))["protr"]
    on, off = (clip, TENSION_PAIRS[clip]) if clip in TENSION_PAIRS else (next((k for k, v in TENSION_PAIRS.items() if v == clip), None), clip)
    other_clip = off if clip == on else on
    other = None
    pname = f"tissue/exp29_{body}_{other_clip}_s{seed}"
    from .common import landed_file
    try:
        if os.path.exists(landed_file(pname)):
            other = stat_shape(cells_of(open_run(pname)))["protr"]
    except Exception:                                                        # noqa: BLE001
        other = None
    out = {"protr_final": finite(mine), "protr_partner": finite(other), "on": on, "off": off}
    if other is None or mine is None or on is None:
        out.update(protr_on_minus_off=None, sign_match=None)
        return out
    d = (mine - other) if clip == on else (other - mine)
    r_on, r_off = ref(on, "shape", "protr"), ref(off, "shape", "protr")
    out["protr_on_minus_off"] = finite(d)
    if r_on and r_off:
        dv = r_on["mean"] - r_off["mean"]
        spread = max(r_on.get("spread") or 0.0, r_off.get("spread") or 0.0)
        out["vertex_on_minus_off"] = finite(dv)
        out["sign_match"] = sign_match(d, dv, spread)
    else:
        out["sign_match"] = None
    return out


def death(T, **_):
    s = stat_death(cells_of(T))
    clip = _clip(T)
    out = {k: finite(v) for k, v in s.items()}
    out["low_activator_gap"] = finite(gap(clip, "death", "low_activator_fraction", s.get("low_activator_fraction"))) if clip else None
    return out


register_run("exp29.health", health, None, "intact / finite / pool fill of one run")
register_run("exp29.growth", growth, None, "doubling time of the cell count, in units of the spec's own full-rate doubling time")
register_run("exp29.chem", chem, None, "spots per 100 cells and the activator-high fraction on the contact graph, against the vertex twin")
register_run("exp29.gate", gate, None, "division rate of activator-high cells over low cells, against the vertex twin")
register_run("exp29.shape", shape, None, "protrusion and the activator-radius correlation, against the vertex twin")
register_run("exp29.regime", regime, None, "persistence of the activator-high set, and its order against the vertex pair")
register_run("exp29.tension", tension, None, "protrusion with the interface tension on against off, sign against the vertex pair")
register_run("exp29.death", death, None, "the share of dead cells low in activator, against the vertex twin")


# ======================================================================================================================
# THE COMPARISON BANK (2026-10-21, after Cedric saw exp29_t_die's cells NOT fill the space of the dead cells while the vertex
# twin's do): every nominal vertex twin is compared with its MPM twin on the same four families of numbers --
#     NUMBER        how many cells, and how that number moves
#     DISTRIBUTION  how cell size, contact degree, spacing and activator are spread over the cells
#     GEOMETRY      where the cells sit: radius, protrusion, sphericity, HOLES in the sheet
#     ORGANISATION  how the cells are arranged and the chemistry on them: graph, spots, correlations, divisions, deaths
# Every metric reads ONLY the `Cells` facade (centroid, activator, volume, alive, ids, contact graph), so a gap between two
# bodies is a gap in the tissue and not in two rulers. `late` is the last third of the rows. Curves are sampled at 50 points
# spread over the run (the same fraction of the run in both bodies: BY FRAME when both runs have the same frame count).
#
#     gap of a scalar     |m_MPM - mean_V| / max(sd_V, 0.10 |mean_V|, abs_floor)       sd_V over the vertex seeds
#     gap of a curve      the run's mean |ln(c / c_V)| / max(the vertex seeds' own, CURVE_FLOOR)
#     gap of a distribution   W1(pooled late MPM values, pooled late vertex values) / median_V  /  max(the vertex seeds' own
#                             seed-to-seed W1 (same normalisation), W1_FLOOR)
#     family score        mean over the family's metrics of min(gap, GAP_CAP); a metric that does not apply (no chemistry in
#                         the vertex twin, no births) is skipped, never zero
# FLAG: gap > FLAG_GAP. The floors (eps 0.10 and the absolute floors below) are model choices: below them a difference is
# not a difference (hole_fraction 0.02: one point in fifty of the sphere; max_hole 0.15: a sixth of a cell spacing).
# ======================================================================================================================
GAP_CAP, FLAG_GAP, EPS_REL, CURVE_FLOOR, W1_FLOOR = 6.0, 3.0, 0.10, 0.05, 0.05
N_CURVE, N_QUANT, N_SPHERE_POINTS = 50, 101, 20000
HOLE_CUT = 1.5                                   # (sphere_holes) a direction farther than 1.5 x the nearest cell's own angular radius from it is in a hole
HOLE_NEIGHBOURS = 1.4                            # (holes) cells i and j touch when their centroids are within 1.4 x (rho_i + rho_j) of each other
HOLE_GAP_DEG = 150.0                             # (holes) a cell whose neighbours leave a 150 degree gap borders a hole
ABS_FLOOR = {"hole_fraction": 0.02, "max_hole": 0.15, "moran_I": 0.05, "high_fraction": 0.02, "spots_per_100": 0.5,
             "spot_size": 0.5, "corr_act_rad": 0.10, "div_contrast": 0.5, "death_low_fraction": 0.05, "ln_ratio": 0.05,
             "births_rate": 1e-5, "deaths_rate": 1e-5, "clustering": 0.02, "degree_mean": 0.2, "sphericity": 0.02,
             "protrusion": 0.02, "hole_recovery": 5.0, "doubling_time": 5.0, "spike_p95": 0.05}
FAMILIES = {
    "number": {"scalars": ["ln_ratio", "doubling_time", "births_rate", "deaths_rate"], "curves": ["N"], "dists": []},
    "distribution": {"scalars": ["spike_p95"], "curves": [], "dists": ["volume_rel", "degree", "nn_dist_rel", "activator", "spike", "elong"]},
    "geometry": {"scalars": ["protrusion", "sphericity", "hole_fraction", "max_hole"], "curves": ["R"], "dists": []},
    "organisation": {"scalars": ["degree_mean", "clustering", "moran_I", "high_fraction", "spots_per_100", "spot_size",
                                 "corr_act_rad", "div_contrast", "death_low_fraction", "hole_recovery"], "curves": [], "dists": []},
}
CURVE_NAMES = {"N": "count_curve", "R": "radius_curve"}
SKIP_IF_MISSING = ("spike_p95",)
HOLE_METRICS = ("hole_fraction", "max_hole")
ACTIVATOR_METRICS = ("activator", "moran_I", "high_fraction", "spots_per_100", "spot_size", "corr_act_rad", "div_contrast",
                     "death_low_fraction")


def late_rows(c: Cells, at_most: int = 24):
    """The last third of the rows, thinned to at most `at_most` evenly spread rows (every metric is a mean or a pool over them)."""
    r = list(range((2 * c.rows) // 3, c.rows))
    return r if len(r) <= at_most else [r[int(i)] for i in np.linspace(0, len(r) - 1, at_most).round()]


def fib_sphere_unit(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def sphere_holes(x: np.ndarray, area: np.ndarray | None = None, n_points: int = N_SPHERE_POINTS):
    """(hole_fraction, max_hole) of a sheet of cells on a closed STAR-SHAPED surface (information only: a collapsed or budded vesicle is not).

    Directions: each cell's centroid seen from the sheet's centre (the mean centroid), and `n_points` Fibonacci directions on the
    unit sphere. For each direction, the ANGLE to the nearest cell, d_k, in units of that cell's own angular radius
    theta_i = sqrt(A_i / pi) / r_i (the disc of the cell's footprint A_i -- a face's area on a mesh, V^(2/3) of a point body --
    at its distance r_i from the centre). hole_fraction = share of directions with d_k > HOLE_CUT theta_nearest; max_hole =
    max d_k / theta. Only the surface cells count (at least 0.8 x the 90th-percentile radius out: all of a vesicle, the outer
    layer of a spheroid). Using each cell's OWN size is what lets it see a gap that survivors did not grow into: against the
    survivors' own centroid spacing the gap IS the spacing, and a sheet that lost 40 % of its cells read 0.03. Using angles keeps
    a bulge or a dimple from reading as a hole (a sphere fitted to a budded vesicle misses the bud). A tiling of equal hexagons
    has d_k <= 1.10 theta; irregular and square tilings, and the vertex twins' Voronoi-like cells, stay under 1.5, which is
    HOLE_CUT (a model choice). With no `area` the nearest-neighbour angle stands in for the cell size."""
    from scipy.spatial import cKDTree
    if len(x) < 8:
        return None, None
    c = x.mean(0)
    r = np.linalg.norm(x - c, axis=1)
    sel = r >= 0.8 * np.percentile(r, 90)
    xs, rs = x[sel], r[sel]
    if len(xs) < 8:
        return None, None
    u = (xs - c) / np.maximum(rs[:, None], 1e-12)
    tree = cKDTree(u)
    if area is None or float(np.median(np.asarray(area, float)[sel])) < 1e-9:
        nn = tree.query(u, k=2)[0][:, 1]
        theta = 2 * np.arcsin(np.clip(nn / 2, 0, 1)) / 2.0 * np.ones(len(xs))
    else:
        theta = np.sqrt(np.maximum(np.asarray(area, float)[sel], 1e-12) / np.pi) / np.maximum(rs, 1e-12)
    chord, j = tree.query(fib_sphere_unit(n_points))
    ang = 2 * np.arcsin(np.clip(chord / 2, 0, 1))
    q = ang / theta[j]
    return float((q > HOLE_CUT).mean()), float(q.max())


def is_solid(c: Cells) -> bool:
    """True for a SOLID body (an MPM spheroid: inner cells under an outer layer); a vesicle (a mesh, or a shell of cells) is all surface."""
    if c.body != "m":
        return False
    return any(isinstance(o, dict) and str(o.get("model", "")).startswith(("spheroid", "clump")) for o in (c.spec.get("seed") or []))


def holes(x: np.ndarray, area: np.ndarray | None = None, rho_nb: float = HOLE_NEIGHBOURS, cut_deg: float = HOLE_GAP_DEG, solid: bool = False):
    """(hole_fraction, max_hole) of a sheet of cells, read LOCALLY: a cell that borders a hole has no neighbour on one side.

    For each surface cell (all cells of a vesicle; of a solid spheroid, those at least 0.8 x the 90th-percentile radius out) take
    the cells that touch it -- centroids within `rho_nb` x (rho_i + rho_j), rho = sqrt(A / pi), A a cell's footprint (a face's area
    on a mesh, V^(2/3) of a point body), so a small cell beside a large one still finds it --, project them on the tangent plane of the cell (the plane of its 8 nearest cells, found by PCA, so a bud or a
    collapsed vesicle is no problem), and find the largest angular gap between consecutive neighbours, g_i; a cell with fewer than
    two neighbours has g_i = 360 degrees. A confluent sheet has g_i near 60 degrees (hexagons) and rarely above 120; a cell on the
    edge of a hole has a gap of 180 or more. hole_fraction = share of surface cells with g_i > `cut_deg` (150); max_hole = the
    95th percentile of g_i in units of 60 degrees. Every cell is read against ITS OWN size, so a gap that the survivors did not
    grow into shows (against the survivors' centroid spacing it could not: the gap is the spacing), and against the angles of its
    own neighbours, not a fitted sphere (which reads a budded vesicle as full of holes: 0.15-0.37 on confluent vertex vesicles)."""
    from scipy.spatial import cKDTree
    if len(x) < 10:
        return None, None
    c = x.mean(0)
    r = np.linalg.norm(x - c, axis=1)
    sel = (r >= 0.8 * np.percentile(r, 90)) if solid else np.ones(len(x), bool)      # a vesicle is all surface (a lobed one has a radius spread)
    xs = x[sel]
    if len(xs) < 10:
        return None, None
    tree = cKDTree(xs)
    if area is None or float(np.median(np.asarray(area, float)[sel])) < 1e-9:
        rho = tree.query(xs, k=2)[0][:, 1] / 2.0
    else:
        rho = np.sqrt(np.maximum(np.asarray(area, float)[sel], 1e-12) / np.pi)
    _, idx = tree.query(xs, k=min(9, len(xs)))
    rho_max = float(rho.max())
    gaps = np.empty(len(xs))
    for i in range(len(xs)):
        P = xs[idx[i]] - xs[idx[i]].mean(0)
        _, v = np.linalg.eigh(P.T @ P)
        e1, e2 = v[:, 2], v[:, 1]
        cand = tree.query_ball_point(xs[i], rho_nb * (rho[i] + rho_max))
        nb = [j for j in cand if j != i and np.linalg.norm(xs[j] - xs[i]) <= rho_nb * (rho[i] + rho[j])]
        if len(nb) < 2:
            gaps[i] = 2 * np.pi
            continue
        d = xs[nb] - xs[i]
        ang = np.sort(np.arctan2(d @ e2, d @ e1))
        gaps[i] = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]])).max()
    return float((gaps > np.deg2rad(cut_deg)).mean()), float(np.percentile(gaps, 95) / (np.pi / 3))


def clustering_coefficient(n: int, pairs: np.ndarray) -> float:
    from scipy.sparse import coo_matrix
    if n < 3 or len(pairs) == 0:
        return 0.0
    A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float)
    deg = np.asarray(A.sum(1)).ravel()
    tri = np.asarray(A.multiply(A @ A).sum(1)).ravel() / 2.0
    ok = deg >= 2
    return float((tri[ok] / (deg[ok] * (deg[ok] - 1) / 2.0)).mean()) if ok.any() else 0.0


def morans_i(a: np.ndarray, n: int, pairs: np.ndarray) -> float:
    """Moran's I of a per-cell value on the contact graph (binary weights): +1 clustered, 0 random, negative alternating."""
    if n < 4 or len(pairs) == 0 or np.ptp(a) < 1e-9:
        return 0.0
    z = a - a.mean()
    num = 2.0 * float((z[pairs[:, 0]] * z[pairs[:, 1]]).sum())
    W = 2.0 * len(pairs)
    return float((n / W) * num / float((z * z).sum()))


def spot_sizes(high: np.ndarray, pairs: np.ndarray, min_size: int = 2) -> list:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(high)
    if n == 0 or not high.any():
        return []
    keep = high[pairs[:, 0]] & high[pairs[:, 1]] if len(pairs) else np.zeros(0, bool)
    e = pairs[keep] if len(pairs) else np.zeros((0, 2), np.int64)
    _, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)
    s = np.bincount(lab[high])
    return [int(v) for v in s if v >= min_size]


def wedge_volumes(pos, es, et, ef, nF):
    """Per-face wedge volume (1/3) c_f . N_f about the mean vertex: the vertex model's own cell volume (the studio's)."""
    live = ef < nF
    es, et, ef = es[live], et[live], ef[live]
    p = pos - pos.mean(0)
    N = np.zeros((nF, 3))
    np.add.at(N, ef, 0.5 * np.cross(p[es], p[et]))
    cen = np.zeros((nF, 3))
    np.add.at(cen, ef, p[es])
    cen /= np.maximum(np.bincount(ef, minlength=nF).astype(np.float64), 1)[:, None]
    return np.abs((cen * N).sum(1)) / 3.0


def point_volume(spec: dict) -> float:
    """The volume of one material point in world units: (point_spacing)^3 of the spec's seed."""
    for o in (spec.get("seed") or []):
        if isinstance(o, dict) and o.get("point_spacing"):
            return float(o["point_spacing"]) ** 3
    return 0.125 ** 3


def cell_footprints(c: Cells, t, slot) -> np.ndarray | None:
    """Per live cell, its footprint on the sheet: a face's area on a mesh, V^(2/3) of a point body (a cube of volume V)."""
    if c.body == "v":
        es, et, ef = (np.asarray(a, np.int64) for a in c.T.half_edges(t))
        pos, nF = c.T.pos(t), c.T.nF(t)
        live = ef < nF
        es, et, ef = es[live], et[live], ef[live]
        cen = np.zeros((nF, 3))
        np.add.at(cen, ef, pos[es])
        cen /= np.maximum(np.bincount(ef, minlength=nF).astype(np.float64), 1)[:, None]
        N = np.zeros((nF, 3))
        np.add.at(N, ef, 0.5 * np.cross(pos[es] - cen[ef], pos[et] - cen[ef]))
        return np.linalg.norm(N, axis=1)[slot]
    if "cell__npts" not in c.z.files:
        return None
    npts = np.asarray(c.z["cell__npts"][t], float).reshape(-1)[slot]
    return (npts * point_volume(c.spec)) ** (2.0 / 3.0)


def cell_volumes(c: Cells, t, slot) -> np.ndarray:
    """Per live cell: the wedge volume on a mesh, the live point count on a point body (volume of a point is constant)."""
    if c.body == "v":
        es, et, ef = (np.asarray(a, np.int64) for a in c.T.half_edges(t))
        v = wedge_volumes(c.T.pos(t), es, et, ef, c.T.nF(t))
        return v[slot]
    npts = c.z["cell__npts"][t] if "cell__npts" in c.z.files else None
    return np.asarray(npts, float).reshape(-1)[slot] if npts is not None else np.ones(len(slot))


def cell_shape_row(c: Cells, t, slot):
    """(spike, elong) per live cell with enough points, or None when the run has no points/mesh to read them from (D5, CELL SHAPE).

    spike = max_i |x_i - c| / rms_i |x_i - c| over the cell's own points; elong = sqrt(lambda_1 / lambda_3) of their covariance
    (`tools/exp29_cellshape.py::cell_shape`, imported, not rewritten). A POINT body: the cell's material points (`pt__pid`).
    A MESH body: the cell's own mesh vertices, each weighted equally (a face's half-edges visit each vertex once); a face is
    FLAT (a few vertices on a surface), so lambda_3 is the roughness of a polygon and sqrt(lambda_1 / lambda_3) would be
    meaningless: its `elong` is sqrt(lambda_1 / lambda_2), the in-plane aspect, and its n >= 4 vertices replace the points' n >= 8."""
    import exp29_cellshape as CS
    if c.body == "m":
        z = c.z
        if not all(k in z.files for k in ("pt__pos", "pt__pid", "pt__occ")):
            return None
        pid = np.asarray(z["pt__pid"][t])
        occ = np.asarray(z["pt__occ"][t])
        sp, el, ids = CS.cell_shape(np.asarray(z["pt__pos"][t]), pid.reshape(len(pid), -1)[:, 0] if pid.ndim > 1 else pid,
                                    occ.reshape(len(occ), -1)[:, 0] if occ.ndim > 1 else occ)
        keep = np.isin(ids, slot)
        return sp[keep], el[keep]
    es, et, ef = (np.asarray(a, np.int64) for a in c.T.half_edges(t))
    pos = c.T.pos(t)
    order = np.argsort(ef, kind="stable")
    es_s, ef_s = es[order], ef[order]
    faces, start, count = np.unique(ef_s, return_index=True, return_counts=True)
    want = set(int(s) for s in slot)
    spike, elong = [], []
    for f, s0, n in zip(faces, start, count):
        if int(f) not in want or n < 4:
            continue
        x = pos[es_s[s0:s0 + n]].astype(np.float64)
        d = x - x.mean(0)
        r = np.linalg.norm(d, axis=1)
        w = np.clip(np.linalg.eigvalsh(d.T @ d / n), 1e-12, None)
        spike.append(r.max() / np.sqrt((r ** 2).mean()))
        elong.append(np.sqrt(w[-1] / w[-2]))
    return np.array(spike), np.array(elong)


def _frame_stats(c: Cells, t, a_sw=A_SW) -> dict:
    slot, pairs = c.graph(t)
    x = c.cen[t][slot].astype(float)
    n = len(slot)
    a = c.activator(t, slot).astype(float)
    out = {"n": n, "x": x, "a": a}
    if n < 6:
        return out
    from scipy.spatial import cKDTree
    deg = np.bincount(pairs.ravel(), minlength=n).astype(float) if len(pairs) else np.zeros(n)
    nn = cKDTree(x).query(x, k=2)[0][:, 1]
    vol = cell_volumes(c, t, slot)
    high = a > a_sw
    sz = spot_sizes(high, pairs)
    r = np.linalg.norm(x - x.mean(0), axis=1)
    ev = np.sort(np.linalg.eigvalsh(np.cov((x - x.mean(0)).T)))
    hf, mh = holes(x, cell_footprints(c, t, slot), solid=is_solid(c))
    shp = cell_shape_row(c, t, slot)
    if shp is not None and len(shp[0]):
        out.update(spike=shp[0], elong=shp[1], spike_p95=float(np.percentile(shp[0], 95)))
    out.update(
        degree=deg, volume_rel=vol / max(np.median(vol), 1e-12), nn_dist_rel=nn / max(np.median(nn), 1e-12),
        degree_mean=float(deg.mean()), clustering=clustering_coefficient(n, pairs), moran_I=morans_i(a, n, pairs),
        high_fraction=float(high.mean()), spots_per_100=100.0 * len(sz) / n, spot_size=float(np.mean(sz)) if sz else 0.0,
        corr_act_rad=spearman(a, r), protrusion=protrusion(x), sphericity=float(ev[0] / max(ev[-1], 1e-12)),
        hole_fraction=hf, max_hole=mh, R=float(np.median(r)))
    return out


def run_bank(T) -> dict:
    """Every comparison metric of ONE run, ready to be pooled over seeds: scalars (late mean), distributions (late pooled
    quantiles), curves (50 points), and what the death and division metrics need."""
    c = cells_of(T)
    a_sw = float((_op(c.spec, "cell_grow") or {}).get("a_sw", A_SW) or A_SW) or A_SW
    late = list(late_rows(c))
    fs = [_frame_stats(c, t, a_sw) for t in late]
    fs = [f for f in fs if "degree" in f]
    sc, dist = {}, {}

    def lm(key):
        v = [f[key] for f in fs if f.get(key) is not None]
        return float(np.mean(v)) if v else None
    for k in ("degree_mean", "clustering", "moran_I", "high_fraction", "spots_per_100", "spot_size", "corr_act_rad", "protrusion",
              "sphericity", "hole_fraction"):
        sc[k] = lm(k)
    mh = [f["max_hole"] for f in fs if f.get("max_hole") is not None]
    sc["max_hole"] = float(np.mean(mh)) if mh else None
    sc["spike_p95"] = lm("spike_p95")
    for k in ("volume_rel", "degree", "nn_dist_rel", "spike", "elong"):
        pooled = np.concatenate([f[k] for f in fs if k in f]) if any(k in f for f in fs) else np.zeros(0)
        dist[k] = np.quantile(pooled, np.linspace(0, 1, N_QUANT)).tolist() if pooled.size else None
    pooled = np.concatenate([f["a"] for f in fs]) if fs else np.zeros(0)
    dist["activator"] = np.quantile(pooled, np.linspace(0, 1, N_QUANT)).tolist() if pooled.size else None
    n = c.n().astype(float)
    sc["ln_ratio"] = float(np.log(max(n[-1], 1) / max(n[0], 1)))
    g = stat_growth(c)
    sc["doubling_time"] = g.get("doubling_time")
    bs = births(c)
    de = [len(departed(c, t)) for t in range(1, c.rows)]
    cf = float(sum(n[t - 1] * (c.frame(t) - c.frame(t - 1)) for t in range(1, c.rows)))
    sc["births_rate"] = (len(bs) / cf) if cf > 0 else None
    sc["deaths_rate"] = (sum(de) / cf) if cf > 0 else None
    sc["div_contrast"] = stat_gate(c, a_sw).get("div_contrast")
    sc["death_low_fraction"] = stat_death(c).get("low_activator_fraction")
    sc["hole_recovery"] = hole_recovery(c, de)
    idx = np.linspace(0, c.rows - 1, N_CURVE).round().astype(int)
    curves = {"N": [float(n[i]) for i in idx]}
    r0 = None
    R = []
    for i in idx:
        s = c.live(i)
        x = c.cen[i][s].astype(float)
        R.append(float(np.median(np.linalg.norm(x - x.mean(0), axis=1))) if len(s) >= 4 else None)
    r0 = R[0] or 1.0
    curves["R"] = [None if v is None else v / r0 for v in R]
    return {"scalars": sc, "dists": dist, "curves": curves,
            "meta": {"rows": c.rows, "frames": float(c.frame(c.rows - 1)), "n_start": float(n[0]), "n_end": float(n[-1]),
                     "body": c.body, "chem_active": bool(len(fs) and max(f["a"].max() for f in fs) > 0.05)}}


def hole_recovery(c: Cells, deaths_per_row: list):
    """Frames from the first death until the sheet's hole_fraction is back within 2 sd of its value before any death (None when
    nothing died or it never returns)."""
    first = next((t + 1 for t, d in enumerate(deaths_per_row) if d > 0), None)
    if first is None or first < 4:
        return None
    hf = []
    for t in range(c.rows):
        s = c.live(t)
        h = holes(c.cen[t][s].astype(float), cell_footprints(c, t, s), solid=is_solid(c))[0] if (len(s) >= 10 and t % 3 == 0) else None
        hf.append(h)
    pre = [h for h in hf[:first] if h is not None]
    if len(pre) < 3:
        return None
    thr = float(np.mean(pre) + 2 * max(np.std(pre), 0.01))
    last_death = max(t + 1 for t, d in enumerate(deaths_per_row) if d > 0)
    for t in range(last_death, c.rows):
        if hf[t] is not None and hf[t] <= thr:
            return float(c.frame(t) - c.frame(first))
    return None


# ---------------------------------------------------------------------------------------------------- comparing
def _w1_q(qa, qb) -> float:
    """1-D Wasserstein distance between two empirical distributions given by their quantile functions on one grid."""
    return float(np.mean(np.abs(np.asarray(qa, float) - np.asarray(qb, float))))


def _curve_gap_raw(ca, cb) -> float | None:
    a, b = np.asarray([np.nan if v is None else v for v in ca], float), np.asarray([np.nan if v is None else v for v in cb], float)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return float(np.mean(np.abs(np.log(a[ok] / b[ok])))) if ok.any() else None


def pool_references(banks: list) -> dict:
    """The vertex seeds' banks of one clip -> mean, sd, and the seed-to-seed spreads every gap is divided by."""
    ref = {"scalars": {}, "dists": {}, "curves": {}, "meta": {"seeds": len(banks), "chem_active": any(b["meta"]["chem_active"] for b in banks),
                                                              "n_end": [b["meta"]["n_end"] for b in banks], "frames": banks[0]["meta"]["frames"]}}
    for k in set().union(*[b["scalars"].keys() for b in banks]):
        v = [b["scalars"].get(k) for b in banks]
        vv = [float(x) for x in v if x is not None and np.isfinite(x)]
        ref["scalars"][k] = {"values": v, "mean": float(np.mean(vv)) if vv else None,
                             "sd": float(np.std(vv, ddof=1)) if len(vv) > 1 else 0.0}
    for k in set().union(*[b["dists"].keys() for b in banks]):
        qs = [b["dists"].get(k) for b in banks if b["dists"].get(k) is not None]
        if not qs:
            ref["dists"][k] = None
            continue
        pooled = np.mean(qs, axis=0)                              # the pooled distribution's quantiles (equal weights)
        med = float(np.median(pooled)) or 1.0
        pairs = [_w1_q(qs[i], qs[j]) / abs(med) for i in range(len(qs)) for j in range(i + 1, len(qs))]
        ref["dists"][k] = {"quantiles": pooled.tolist(), "median": med, "w1_seed": float(np.mean(pairs)) if pairs else 0.0,
                           "per_seed": qs}
    for k in set().union(*[b["curves"].keys() for b in banks]):
        cs = [b["curves"][k] for b in banks]
        arr = np.array([[np.nan if v is None else v for v in c_] for c_ in cs], float)
        mean = np.nanmean(arr, axis=0)
        pairs = [x for i in range(len(cs)) for j in range(i + 1, len(cs)) if (x := _curve_gap_raw(cs[i], cs[j])) is not None]
        ref["curves"][k] = {"mean": [None if not np.isfinite(v) else float(v) for v in mean],
                            "seed_gap": float(np.mean(pairs)) if pairs else 0.0}
    return ref


def compare_bank(bank: dict, ref: dict) -> dict:
    """The table: one row per metric (family, name, MPM value, vertex mean, vertex sd, gap, flag) and the family scores."""
    rows = []
    chem_ref = ref["meta"]["chem_active"]
    for fam, spec in FAMILIES.items():
        for k in spec["scalars"]:
            m = bank["scalars"].get(k)
            r = ref["scalars"].get(k)
            if r is None or r["mean"] is None:
                continue                                                # the vertex twin has no value: not applicable
            if k in ACTIVATOR_METRICS and not chem_ref:
                continue
            mv = 0.0 if m is None and k in ("spots_per_100", "spot_size", "high_fraction", "moran_I", "corr_act_rad") else m
            if mv is None and k in SKIP_IF_MISSING:
                continue                                                # a run that records no points cannot be read for a tip
            if mv is None:
                rows.append(dict(family=fam, metric=k, mpm=None, vertex_mean=r["mean"], vertex_sd=r["sd"], gap=GAP_CAP, flag=True,
                                 note="the MPM twin has no value where the vertex twin has one"))
                continue
            den = max(r["sd"], EPS_REL * abs(r["mean"]), ABS_FLOOR.get(k, 1e-6))
            g = abs(mv - r["mean"]) / den
            rows.append(dict(family=fam, metric=k, mpm=mv, vertex_mean=r["mean"], vertex_sd=r["sd"], gap=float(g), flag=bool(g > FLAG_GAP)))
        for k in spec["curves"]:
            r = ref["curves"].get(k)
            raw = _curve_gap_raw(bank["curves"][k], r["mean"]) if r else None
            if raw is None:
                continue
            g = raw / max(r["seed_gap"], CURVE_FLOOR)
            rows.append(dict(family=fam, metric=CURVE_NAMES[k], mpm=raw, vertex_mean=0.0, vertex_sd=r["seed_gap"], gap=float(g), flag=bool(g > FLAG_GAP)))
        for k in spec["dists"]:
            r = ref["dists"].get(k)
            q = bank["dists"].get(k)
            if r is None or (k in ACTIVATOR_METRICS and not chem_ref):
                continue
            if q is None:
                continue
            w1 = _w1_q(q, r["quantiles"]) / abs(r["median"] if abs(r["median"]) > 1e-9 else 1.0)
            g = w1 / max(r["w1_seed"], W1_FLOOR)
            rows.append(dict(family=fam, metric=k, mpm=w1, vertex_mean=0.0, vertex_sd=r["w1_seed"], gap=float(g), flag=bool(g > FLAG_GAP)))
    fam_score = {}
    for fam in FAMILIES:
        gs = [min(r["gap"], GAP_CAP) for r in rows if r["family"] == fam]
        if fam == "geometry":
            hole = [min(r["gap"], GAP_CAP) for r in rows if r["family"] == fam and r["metric"] in HOLE_METRICS]
            rest = [min(r["gap"], GAP_CAP) for r in rows if r["family"] == fam and r["metric"] not in HOLE_METRICS]
            fam_score[fam] = (2.0 / 3.0 * float(np.mean(rest)) if rest else 0.0) + (1.0 / 3.0 * float(np.mean(hole)) if hole else 0.0) if (rest or hole) else None
        else:
            fam_score[fam] = float(np.mean(gs)) if gs else None
    hole_gap = max([r["gap"] for r in rows if r["metric"] in HOLE_METRICS] or [0.0])
    return {"rows": rows, "family": fam_score, "hole_gap": float(hole_gap), "flags": [r["metric"] for r in rows if r["flag"]]}


def bank_references(seeds=(0, 1, 2), write=True, only=None):
    """Add the bank to `vertex_references.json`: per clip, the vertex seeds' metrics pooled (mean, sd, seed-to-seed spreads)."""
    from .common import landed_file
    out = load_refs() or {"clips": {}}
    bank = {}
    for clip in (only or CLIPS):
        banks = []
        for s in seeds:
            spec = f"tissue/exp29_v_{clip}_s{s}"
            try:
                if not os.path.exists(landed_file(spec)):
                    continue
                banks.append(run_bank(open_run(spec)))
            except Exception as e:                                          # noqa: BLE001
                out.setdefault("bank_errors", {})[spec] = f"{type(e).__name__}: {e}"
        if banks:
            bank[clip] = pool_references(banks)
            bank[clip]["runs"] = [f"tissue/exp29_v_{clip}_s{s}" for s in seeds][:len(banks)]
    out.setdefault("bank", {}).update(bank)
    out["bank_spec"] = {"gap_cap": GAP_CAP, "flag_gap": FLAG_GAP, "eps_rel": EPS_REL, "curve_floor": CURVE_FLOOR, "w1_floor": W1_FLOOR,
                        "late": "last third of the rows", "hole_cut": HOLE_CUT, "abs_floor": ABS_FLOOR}
    if write:
        os.makedirs(os.path.dirname(REFS), exist_ok=True)
        json.dump(out, open(REFS, "w"), indent=1)
    return out


def compare(T, clip=None, refs_path=None, **_) -> dict:
    """A run against the vertex twin of its clip (from its name unless `clip`): the table, the family scores, the flags.
    `refs_path`: another references file (the 200-cell set, `vertex_references_n200.json`, for a run of that size)."""
    clip = clip or _clip(T)
    refs = json.load(open(refs_path)) if refs_path else load_refs()
    if not refs or "bank" not in refs or clip not in refs["bank"]:
        raise ValueError(f"no vertex references for clip {clip!r} in {REFS}: run tools/exp29_references.py --bank")
    b = run_bank(T)
    cmp = compare_bank(b, refs["bank"][clip])
    return {"bank": b, "compare": cmp, "clip": clip}


def compare_measure(T, **_):
    r = compare(T)
    f = r["compare"]["family"]
    out = {"number_score": finite(f["number"]) if f["number"] is not None else None,
           "distribution_score": finite(f["distribution"]) if f["distribution"] is not None else None,
           "geometry_score": finite(f["geometry"]) if f["geometry"] is not None else None,
           "organisation_score": finite(f["organisation"]) if f["organisation"] is not None else None,
           "hole_gap": finite(r["compare"]["hole_gap"]), "n_flags": len(r["compare"]["flags"]),
           "flags": ",".join(r["compare"]["flags"])}
    for row in r["compare"]["rows"]:
        out[f"gap.{row['metric']}"] = finite(row["gap"])
    return out


register_run("exp29.compare", compare_measure, None, "the comparison bank: number, distribution, geometry, organisation gaps against the vertex twin")
