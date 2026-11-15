"""The VTK renderer, promoted: a z-buffered, GPU, lit picture of a mesh set.

Why VTK: `mpl_toolkits.mplot3d` has no depth buffer (it sorts polygons by mean z), which is wrong for a closed cellular body (on b_star's end frame 6,124 of 12,272 apical faces face away from the camera at azimuth 310 and are drawn anyway, so one surface is drawn two ways in a single rotation). VTK discards a fragment behind another per pixel, is ~29x faster on this mesh (0.32 s a frame against 9.33) and lights it properly (28.9% of pixels lit against 4.5%).

Every function that makes a picture (`mesh_of`, `_marks`, `add`, `aim`, `_plotter`) is carried across character for character from `discovery_okuda/vtk_render.py`, so the two versions stay comparable. New: `frames_of` reads EITHER trajectory layout (okuda `traj.npz`: pos_i, mesh_i, act_i, ticks; core `trajectory.npz`: vertex__pos, vertex__occ, vertex__mesh_* ragged without pickle, cell__chem), and a run is named by its DIRECTORY.

Shading is smooth unless asked otherwise (flat gives each cell one normal, so a curved arm reads as a faceted cone):
    mesh     smooth surface WITH the cell outlines: for questions about cells (who divided, how big, how many across a tube)
    nomesh   the same surface, no outlines: for questions about SHAPE
    flat     one normal per cell, no stroke: for a STILL tiled at ~190 px, where a 0.4 px outline is a grey wash

Colour semantics, each a claim about the data:
    white -> red   the activator (the measurement)
    magenta        non-finite: not a cell any more
    green          a cell that has just divided (`age` <= 4 AND `ndiv` > 0; `age` alone starts at 0 for every seeded cell)
    blue           where the SECOND field ACTED (marked a cell to die, or switched its growth off), not how much of it there is (the trajectory carries one activator channel)

Three of the four colours read per-face state that lives on the mesh only, so `MeshTable.FACE_RECORD` is what `snapshot` keeps beyond topology; every name in it is optional."""
from __future__ import annotations

import os
import time

import numpy as np

# CAM, SIZE, FPS, KB_SECONDS, KB_ZOOM, EV_FPS: Default camera (elevation, azimuth), frame size in px, frame rate, Ken Burns revolution seconds and zoom, and the `evolve` frame rate (~60 frames -> a 5 s clip).
CAM = dict(elev=18.0, azim=30.0)
SIZE = 896
FPS = 25
KB_SECONDS = 18.0          # one revolution; the length IS the speed
KB_ZOOM = 0.55
EV_FPS = 12                # ~60 recorded frames -> a 5 s clip


# MAX_EVOLVE_FRAMES: Most frames an `evolve` clip draws (`linspace`, first and last always in): okuda keeps ~60 rows but the core keeps every one, so uncapped the two sides of a promotion pair give a 5 s clip and a 2.5-minute one of the same run, which cannot be compared frame for frame.
MAX_EVOLVE_FRAMES = 60

# SEQUENCES, LOOP_SEQ: (kind, style) per sequence number, so a caller asks for a job and not for four commands; the sequence the loop uses.
SEQUENCES = {
    0: [("kburns", "nomesh")],
    1: [("evolve", "mesh")],
    2: [("evolve", "mesh"), ("kburns", "mesh")],
    3: [("evolve", "nomesh"), ("kburns", "nomesh"), ("evolve", "mesh"), ("kburns", "mesh")],
}
LOOP_SEQ = 0

# DIVIDED, GREEN, GREEN_A, BLUE: The division mark: a cell is "just divided" while age <= DIVIDED (division CALLS), tinted GREEN keeping GREEN_A of its own colour; BLUE tints where the second field acted.
DIVIDED = 4
GREEN = (44, 160, 44)
GREEN_A = 0.5              # how much of the cell's own colour the division mark keeps
BLUE = (31, 119, 180)

# KILL: Electric blue for a cell sentenced to die, painted outright, not blended: `BLUE` is also exactly `MOTHER`, so the same blue meant two opposite things, and a pale blend vanished at the silhouette where most of the dying patch is (apop_patch_big).
KILL = (0, 153, 255)       # electric blue: a cell sentenced to die

# BODY_GREY: The body when there is no chemistry to show: an all-zero activator is not missing (`cell__chem` exists, identically 0, on every apoptosis scene) and the colormap would paint every cell white, "activator at the bottom of its range", instead of "none here".
BODY_GREY = (176, 176, 176)

# MOTHER, DAUGHTER, PAIR_A: The division pair as two colours (mother blue, daughter red): one green tint cannot say WHICH TWO cells a division produced, and a wrong area share or septum looks healthy until the halves are seen apart. `cell_divide` keeps the mother at her index and APPENDS the daughter at `nF`, so a cell whose index is at or beyond an earlier frame's face count is new: `evolve` threads `prev_nF` through, and `kburns` / `still` (one held frame, nothing to be new relative to) do not draw the pair.
MOTHER = (31, 119, 180)    # blue
DAUGHTER = (214, 39, 40)   # red
PAIR_A = 0.62              # how much of the pair colour is laid over the cell's own
# PAIR_TICKS: The window a mother stays marked: DIVIDED (4) division calls x `cell_divide.every` (4) ticks.
PAIR_TICKS = 16


# _frames_ticks: Module cell set by `frames_of` to the recorded tick numbers, when the writer kept them.
class _frames_ticks:
    """Set by `frames_of` to the recorded tick numbers, when the writer kept them: a module cell, not a return value, since two of its three callers do not want it; needed only to convert the pair window from TICKS into rows (stride 1 needs no conversion)."""
    value = None


# _pair_reference: The face count `window_ticks` ago, as the daughter threshold for row t -> int | None.
def _pair_reference(t, nFs, ticks, window_ticks):
    """The face count `window_ticks` ago, as a daughter threshold for row `t`.

    Returns None at the start of a run, where there is no `before` and nothing should be drawn.
    """
    if t == 0:
        return None
    if ticks is None:                       # stride 1: a row IS a tick
        return nFs[max(0, t - window_ticks)]
    cut = ticks[t] - window_ticks
    j = 0
    for k in range(t, -1, -1):
        if ticks[k] <= cut:
            j = k
            break
    return nFs[j]


# _okuda_frames: okuda's traj.npz (pos_i / mesh_i / act_i, the mesh a pickled dict) -> list of (pos, mesh, act, chem).
def _okuda_frames(path):
    """okuda's `traj.npz`: pos_i / mesh_i / act_i, the mesh a pickled dict."""
    z = np.load(path, allow_pickle=True)
    out = []
    for t in range(sum(1 for k in z.files if k.startswith("pos_"))):
        mt = z[f"mesh_{t}"]
        mt = mt.item() if hasattr(mt, "item") else mt
        act = z[f"act_{t}"] if f"act_{t}" in z.files else None
        # okuda's traj.npz stores the activator column only, so the fourth slot is None and `mesh_of` falls back to white -> red
        out.append((np.asarray(z[f"pos_{t}"], float), mt, act, None))
    return out


# _core_frames: The core's `trajectory.npz` unpacked into (pos, mesh, act, chem) frames -> list.
def _core_frames(path, set_name=None, cell_set=None, chan=0):
    """The core's `trajectory.npz`, unpacked into the same (pos, mesh, act) triples. Two offset arrays, and mixing them slices the wrong rows: half-edge columns are indexed by HALF-EDGE (`mesh_offsets`), per-face state by FACE (`mesh_face_offsets`), about six half-edges per face."""
    z = np.load(path)
    if set_name is None:
        cand = [k[:-len("__mesh_nF")] for k in z.files if k.endswith("__mesh_nF")]
        if not cand:
            return None
        set_name = cand[0]
    if cell_set is None:
        cc = [k[:-len("__chem")] for k in z.files if k.endswith("__chem")]
        # a sheet with no chemistry still has cells: the cell set is also found by carrying `age` and having no `pos` (the mesh's face set), or `age`, `ndiv` and `apop_flag` went unread and the mother and the dying cell were never drawn
        if not cc:
            cc = [k[:-len("__age")] for k in z.files
                  if k.endswith("__age") and f"{k[:-len('__age')]}__pos" not in z.files]
        cell_set = cc[0] if cc else None
    # every array is read once: `np.load` returns an NpzFile whose `z[key]` decompresses the WHOLE array on every access, so reading one row inside a loop over N rows decompresses the column N times (terabytes on a large trajectory; the defect `gate_measures._Lazy` exists for)
    pos = z[f"{set_name}__pos"]
    # the apical surface when the run has one: with `cell_mechanics[model: apicobasal]`, apical = pos + sep and basal = pos - sep, and drawing the MID-surface hides the one thing the model adds (the run would be unfalsifiable by eye); `pos + sep`, not an average, which is a surface the model does not have; a run with no `sep` column is untouched, byte-identical
    _sep = z[f"{set_name}__sep"] if f"{set_name}__sep" in z.files else None
    nF, Nv = z[f"{set_name}__mesh_nF"], z[f"{set_name}__mesh_Nv"]
    off, foff = z[f"{set_name}__mesh_offsets"], z[f"{set_name}__mesh_face_offsets"]
    E = {c: z[f"{set_name}__mesh_E_{c}"] for c in ("srce", "trgt", "face")}
    face_cols = {k.split("__mesh_")[1]: z[k] for k in z.files
                 if k.startswith(set_name + "__mesh_")
                 and k.split("__mesh_")[1] not in ("E_srce", "E_trgt", "E_face", "nF", "Nv",
                                                   "offsets", "face_offsets")
                 and not k.split("__mesh_")[1].startswith(("scalar_", "e_"))}
    # the per-half-edge columns, sliced by the HALF-EDGE offsets: `e_myo` is recorded every frame and was dropped here, so no junction could be coloured by its myosin (it is excluded from `face_cols` because `foff` would slice the wrong rows)
    edge_cols = {k.split("__mesh_")[1]: (z[k], z[f"{k}_offsets"] if f"{k}_offsets" in z.files else off)
                 for k in z.files
                 if k.startswith(set_name + "__mesh_e_") and not k.endswith("_offsets")}
    chem = z[f"{cell_set}__chem"] if cell_set and f"{cell_set}__chem" in z.files else None
    # per-cell blocks from the cell set, on the same footing as the mesh's face columns (a face IS a cell, so a width-1 block there is a per-face quantity; `age` and `ndiv` live there now): found by ASKING the file which blocks exist, not by naming them (`e_myo` was lost once to a list that did not mention it)
    cell_cols = {}
    if cell_set:
        _skip = ("pos", "occ", "node_type", "parent", "parent_name", "chem")
        for k in z.files:
            if not k.startswith(f"{cell_set}__") or "__mesh_" in k:
                continue
            b = k[len(cell_set) + 2:]
            if b in _skip:
                continue
            a_ = np.asarray(z[k])
            if a_.ndim == 3 and a_.shape[2] == 1:
                cell_cols[b] = a_
    # the cell set's type column for `plotting.mesh_color_by: node_type` (exp 9: LEP / MEP faces, receivers retyped by contact): `node_type` is not a state block, so the loop above never sees it; `<cell>__node_type_t` is the per-row column the engine keeps when a type changed, `<cell>__node_type` the final one otherwise
    _cnt = None
    if cell_set:
        for _k in (f"{cell_set}__node_type_t", f"{cell_set}__node_type"):
            if _k in z.files:
                _cnt = np.asarray(z[_k])
                break
    out = []
    for t in range(len(nF)):
        a, b = int(off[t]), int(off[t + 1])
        fa, fb = int(foff[t]), int(foff[t + 1])
        mt = {"E_srce": E["srce"][a:b], "E_trgt": E["trgt"][a:b], "E_face": E["face"][a:b],
              "nF": int(nF[t]), "Nv": int(Nv[t])}
        for c, arr in face_cols.items():
            mt[c] = arr[fa:fb]
        for c, arr in cell_cols.items():               # cut to the frame's live faces
            mt.setdefault(c, np.asarray(arr[t])[:int(nF[t]), 0])
        if _cnt is not None:
            mt.setdefault("node_type", np.asarray(_cnt[min(t, len(_cnt) - 1)] if _cnt.ndim == 2
                                                  else _cnt)[:int(nF[t])])
        for c, (arr, eoff) in edge_cols.items():       # each column's OWN offsets -- see live_movie
            ea, eb = int(eoff[t]), int(eoff[t + 1])
            if eb > ea:
                mt[c] = arr[ea:eb]
        act = None if chem is None else np.asarray(chem[t][:int(nF[t]), chan], float)
        # the whole chem row travels with the frame, not only the column `chan` names: a three-species run (May-Leonard u, v, w) has no single activator; `act` stays the scalar the range and metrics are taken from, and the row is what `mesh_of` colours from when the spec declares a LUT
        rows = None if chem is None else np.asarray(chem[t][:int(nF[t])], float)
        _p = np.asarray(pos[t][:int(Nv[t])], float)
        if _sep is not None:
            _s = np.asarray(_sep[t][:int(Nv[t])], float)
            if _s.shape == _p.shape and np.isfinite(_s).all():
                _p = _p + _s
        out.append((_p, mt, act, rows))
    return out


# style_of: (lut, blend, cutaway) from the run's own spec.yaml, all None without a `plotting` block -> tuple.
def style_of(run_dir):
    """`(lut, blend, cutaway)` from the run's own `spec.yaml`: the colour table is part of the MODEL. All three are None when the spec declares no `plotting` block (every Gray-Scott run), which keeps the white -> red activator map and draws the whole closed surface."""
    for nm in ("spec.yaml", "spec_run.yaml"):
        f = os.path.join(run_dir, nm)
        if not os.path.exists(f):
            continue
        try:
            import yaml
            pl = (yaml.safe_load(open(f)) or {}).get("plotting") or {}
        except Exception:
            return None, None
        lut = pl.get("species")
        return (list(lut) if lut else None), pl.get("blend"), pl.get("cutaway")
    return None, None, None


_PLOT_OVERRIDE: dict | None = None


# plot_style: The `plotting:` block for the render flags that are not colour tables -> dict.
def plot_style(run_dir) -> dict:
    """The `plotting:` block for the render flags that are not colour tables. The live config wins over the run's frozen copy (`spec.yaml` is written at GENERATE time, so editing a flag and re-running `-o plot` silently used the old one): `plot_dataset` hands down the freshly loaded spec, and the run's copy stays the fallback for callers with no spec in hand. `style_of` predates it and returns a fixed tuple; this returns the block so a new flag does not add an element every caller must unpack."""
    if _PLOT_OVERRIDE is not None:
        return dict(_PLOT_OVERRIDE)
    for nm in ("spec.yaml", "spec_run.yaml"):
        f = os.path.join(run_dir, nm)
        if os.path.exists(f):
            try:
                import yaml
                return dict((yaml.safe_load(open(f)) or {}).get("plotting") or {})
            except Exception:
                return {}
    return {}


# edges_of: The junctions as a line mesh with per-edge RGB (`plotting.edge_color`) -> PolyData.
def edges_of(pos, mt, mode, ntype=None, rng=None, colors=None, lut="inferno"):
    """The junctions as a line mesh with per-edge RGB: `plotting.edge_color`. A second actor, not a face colour: myosin lives on the JUNCTION, a face colour can only say something about a cell, and painting a cell by the mean of its edges hides what a contractile belt does (differ from edge to edge around one cell).

      `myosin`  the value `junction_myosin` wrote, through a fixed range so a colour means the same number in every frame; `plotting.edge_lut` names the matplotlib colormap (default `inferno`: dark-to-bright reads as "how much" on black)
      `type`    the owning cell's node_type, as flat categorical colours"""
    import pyvista as pv
    from matplotlib import colormaps
    nF = int(mt["nF"])
    es, et, ef = (np.asarray(mt[k]) for k in ("E_srce", "E_trgt", "E_face"))
    live = ef < nF
    if not live.any():
        return None
    i, j, f = es[live].astype(int), et[live].astype(int), ef[live].astype(int)
    if mode == "type":
        if ntype is None:
            return None
        k = np.asarray(ntype)[np.clip(f, 0, len(ntype) - 1)].astype(int)
        pal = np.asarray(colors if colors is not None else
                         [(0.35, 0.60, 1.00), (1.00, 0.35, 0.25), (0.45, 0.95, 0.55),
                          (1.00, 0.85, 0.30)], float)
        rgb = (np.clip(pal[k % len(pal)], 0, 1) * 255).astype(np.uint8)
    else:
        v = mt.get("e_myo")
        if v is None:
            return None
        v = np.asarray(v, float)[live]
        lo, hi = (rng if rng else (float(np.nanmin(v)), float(np.nanmax(v))))
        x = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
        rgb = (np.asarray(colormaps[lut](x))[:, :3] * 255).astype(np.uint8)
    lines = np.empty((len(i), 3), np.int64)
    lines[:, 0] = 2; lines[:, 1] = i; lines[:, 2] = j
    m = pv.PolyData(np.asarray(pos, float), lines=lines.ravel())
    m["rgb"] = rgb
    return m


# frames_of: Every recorded (pos, mesh, act, chem) of a run, from whichever writer produced it -> list.
def frames_of(run_dir, traj=None):
    """Every recorded (pos, mesh, act) of a run, from whichever writer produced it.

    `traj` overrides the file, so a MID-RUN snapshot written in the same format renders through
    exactly this path and needs no special case.
    """
    if traj:
        return (_okuda_frames(traj) if os.path.basename(traj) != "trajectory.npz"
                else _core_frames(traj))
    p = os.path.join(run_dir, "traj.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        _frames_ticks.value = (np.asarray(z["ticks"]).tolist() if "ticks" in z.files else None)
        return _okuda_frames(p)
    _frames_ticks.value = None                    # a core trajectory records every tick
    for root, _d, files in os.walk(run_dir):
        if "trajectory.npz" in files:
            return _core_frames(os.path.join(root, "trajectory.npz"))
    return None


# box_of: The run's own fixed camera box, shared with every other picture of it.
def box_of(run_dir, fr):
    """The run's own fixed camera box, shared with every other picture of it. Fixed, not per-frame: a camera that refits each frame makes a growing body look static (the same defect as renormalising the colour scale every frame)."""
    import json
    # a declared box wins: `plotting.camera_lbox` is the half-width of the view in world units for a SERIES of runs read against one another (a ladder where each slide adds one operator); fitted per run, a sheet that later grows would put every earlier slide's identical first frame at a different scale
    _L = (_PLOT_OVERRIDE or {}).get("camera_lbox")
    if _L:
        return float(_L)
    dj = os.path.join(run_dir, "diag.json")
    if os.path.exists(dj):
        try:
            L = (json.load(open(dj)).get("summary") or {}).get("camera_lbox")
            if L:
                return float(L)
        except Exception:
            pass
    return float(max(np.abs(np.asarray(p)).max() for p, _m, _a, _c in fr)) * 1.12


# _cmap: The white -> red activator colormap.
def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("wr", ["white", "#d62728"])


# _marks: (mother, daughter, kills, suppresses) masks over the drawn faces, None where unrecorded -> tuple.
def _marks(mt, idx, nF, prev_nF=None):
    """(mother, daughter, kills, suppresses) masks over the drawn faces, None where unrecorded.

    `mother` / `daughter` split what was one `divided` mask: a face at or beyond `prev_nF` was appended since that reference and is a daughter, the rest of the just-divided set are mothers. With `prev_nF=None` (a single held frame: `kburns`, `still`) both are None and nothing is drawn, a pair needing a before. `prev_nF` is the face count from far enough back to cover the SAME window the mothers are drawn over, not the previous frame's: `age <= DIVIDED` counts division CALLS and `cell_divide` runs every 4 ticks, so a mother stays blue ~16 ticks while "appended since the previous frame" is one row (three rows in four drew 372 mothers and zero daughters).

    The second field is two different marks: `apop` is rare and is the event, so it takes the colour outright; `inhib` is nearly the whole tissue (4,222 of 4,232 cells in `sc_inh_soft`), so painting it outright would erase the activator pattern, and it is drawn only where the activator is LOW, which is where "does not grow" means something."""
    def col(k):
        """One per-cell state over the drawn faces, PADDED rather than dropped when short: a state is recorded before the frame's divisions are applied, so on dividing frames it is a few entries shorter than `nF` (3,867 against 3,884), and dropping the whole mask made the blue vanish and return every second frame of `sc_inh_soft`. The cells it does not cover are the just-created ones, and an unmarked new cell is the truthful default."""
        v = mt.get(k)
        if v is None:
            return None
        a = np.asarray(v).ravel()
        if a.size < nF:
            a = np.concatenate([a, np.zeros(nF - a.size, a.dtype)])
        return a[:nF][idx]
    age, ndiv = col("age"), col("ndiv")
    div = None if age is None or ndiv is None else ((age <= DIVIDED) & (ndiv > 0))
    mother = daughter = None
    if prev_nF is not None:
        # a daughter is identified by its index alone: `div & new` drew ZERO daughters, since `age` and `ndiv` are written BEFORE the frame's divisions are applied, so on a dividing frame the arrays are `nF_prev` long and `ndiv > 0` is False for exactly the cells just created (the `div` set is the MOTHERS). `index >= prev_nF` is what `cell_divide` guarantees by appending; over a decimated archive it means "born since the previous drawn frame"
        new = np.asarray(idx, int) >= int(prev_nF)
        daughter = new
        mother = (div & ~new) if div is not None else None
    # the declared block name first, the retired `FACE_ALIAS` spelling second -- see live_movie
    kills = col("apop_flag") if col("apop_flag") is not None else col("apop")
    sup = col("inhib_frac") if col("inhib_frac") is not None else col("inhib")
    return (mother, daughter,
            (None if kills is None else kills > 0), (None if sup is None else sup > 0))


# mesh_color_values: The per-face values `plotting.mesh_color_by` names, or None -> array | None.
def mesh_color_values(mt, lab):
    """The per-face values `plotting.mesh_color_by: <lab>` names, or None. A block of that name first;
    `node_type` / `type` fall back to the cell set's type column (`_core_frames`, `live_movie`)."""
    if not lab:
        return None
    v = mt.get(lab)
    if v is None and str(lab).lower() in ("node_type", "type"):
        v = mt.get("node_type")
    return v


# mesh_scale_continuous: Whether `mesh_color_by` is drawn through a colormap rather than as categories -> bool.
def mesh_scale_continuous(v, st) -> bool:
    """Whether `mesh_color_by` is drawn through a colormap rather than as categories.

    `plotting.mesh_color_scale`: absent or `categorical` -> categories, the law every existing movie
    was drawn with (a float block is rounded to its nearest integer); `continuous` -> the colormap;
    `auto` -> the colormap when any finite value is not an integer, categories otherwise.
    """
    s = str((st or {}).get("mesh_color_scale", "") or "").lower()
    if s == "continuous":
        return True
    if s == "auto":
        a = np.asarray(v, float).ravel()
        f = np.isfinite(a)
        return bool(np.any(a[f] != np.rint(a[f])))
    return False


# mesh_block_continuous: Per-face values through `plotting.mesh_cmap` -> uint8 [n, 3].
def mesh_block_continuous(v, st, rng=None):
    """uint8 [n, 3]: per-face values `v` through `plotting.mesh_cmap` (default viridis). The range is fixed, not the frame's: `plotting.mesh_color_range: [lo, hi]` when declared, else the caller's `rng` (the run's in `evolve`; widened-only live), else this frame's own min / max. A non-finite value is magenta, the renderer's "not a cell any more"."""
    from matplotlib import colormaps
    st = st or {}
    a = np.asarray(v, float).ravel()
    fin = np.isfinite(a)
    r = st.get("mesh_color_range") or rng
    if r is not None and len(r) == 2:
        lo, hi = float(r[0]), float(r[1])
    elif fin.any():
        lo, hi = float(a[fin].min()), float(a[fin].max())
    else:
        lo, hi = 0.0, 1.0
    x = np.clip((np.where(fin, a, lo) - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    rgb = (np.asarray(colormaps[str(st.get("mesh_cmap", "viridis"))](x))[:, :3] * 255).astype(np.uint8)
    rgb[~fin] = (255, 26, 217)
    return rgb


# mesh_color_rng: (lo, hi) of the `mesh_color_by` block over every frame when drawn continuous, else None -> tuple | None.
def mesh_color_rng(fr):
    """(lo, hi) of the `mesh_color_by` block over EVERY frame of `fr`, when it is drawn continuous;
    None otherwise -- and None keeps `mesh_of` on the categorical law. One range for the whole clip,
    for the reason the activator has one: a per-frame range makes a strengthening pattern look static."""
    st = _PLOT_OVERRIDE or {}
    lab = str(st.get("mesh_color_by", "") or "")
    scale = str(st.get("mesh_color_scale", "") or "").lower()
    if not lab or lab.lower() in ("phase", "cycle_progress") or scale not in ("continuous", "auto"):
        return None
    vals = []
    for f in fr or []:
        v = mesh_color_values(f[1], lab)
        if v is not None:
            vals.append(np.asarray(v, float).ravel()[:int(f[1]["nF"])])
    if not vals:
        return None
    a = np.concatenate(vals)
    fin = np.isfinite(a)
    if scale == "auto" and not bool(np.any(a[fin] != np.rint(a[fin]))):
        return None
    r = st.get("mesh_color_range")
    if r is not None and len(r) == 2:
        return float(r[0]), float(r[1])
    return (float(a[fin].min()), float(a[fin].max())) if fin.any() else (0.0, 1.0)


# mesh_of: The apical shell of a frame as PolyData with per-cell RGB, rebuilt per frame (cells divide) -> PolyData.
def mesh_of(pos, mt, act, lo=None, hi=None, show_div=True, prev_nF=None, chem=None, lut=None,
            cutaway=None,
            blend=None, vmax=None, color_rng=None):
    """The apical shell as PolyData with per-cell RGB. Rebuilt per frame: cells divide."""
    import pyvista as pv
    from plexus.models.topology import rings_from_flat_3d
    nF = int(mt["nF"])
    es, et, ef = (np.asarray(mt[k]) for k in ("E_srce", "E_trgt", "E_face"))
    live = ef < nF
    rings = rings_from_flat_3d(es[live], et[live], ef[live], nF)
    # the cutaway is a WEDGE OF FACES DROPPED, not a clipping plane: a vtkClipPolyData tears faces into half-drawn, half-coloured cells, while dropping each face entirely by the side its centroid falls on keeps complete cells and a jagged opening along cell boundaries. `cutaway: [ax, ax]` removes the POSITIVE octant of the named axes (["x","y"] takes the quarter where both exceed the body's centroid, opening a quadrant onto the lumen and inner wall; one axis halves, three take an octant)
    _keep_face = None
    if cutaway:
        _ax = {"x": 0, "y": 1, "z": 2}
        _sel = [_ax[a.lower().lstrip("+-")] for a in cutaway if a.lower().lstrip("+-") in _ax]
        _neg = [a.startswith("-") for a in cutaway]
        if _sel:
            _c = np.asarray(pos, float).mean(0)
            _cen = np.array([np.asarray(pos, float)[np.asarray(r, int)].mean(0)
                             if (r is not None and len(r) >= 3) else _c for r in rings])
            _in = np.ones(len(rings), bool)
            for _k, _ax_i in enumerate(_sel):
                _side = _cen[:, _ax_i] < _c[_ax_i] if (_k < len(_neg) and _neg[_k]) \
                    else _cen[:, _ax_i] > _c[_ax_i]
                _in &= _side
            _keep_face = ~_in                       # drop the wedge, keep everything else
    faces, idx = [], []
    for f, r in enumerate(rings):
        if r is None or len(r) < 3:
            continue
        if _keep_face is not None and not _keep_face[f]:
            continue
        faces.append(len(r)); faces.extend(int(v) for v in r); idx.append(f)
    if not idx:
        return None
    m = pv.PolyData(pos, faces=np.asarray(faces, np.int64))
    # a flat activator is not an activator: `act` is non-None on every mesh run (the `cell` set always carries `chem`), so apoptosis scenes, whose `cell__chem` is identically 0 over 601 frames, were colour-mapped to one value (see `BODY_GREY`); an all-NaN frame is not flat and must still reach the colormap branch, where `rgb[~ok] = magenta` marks "not a cell any more", so "flat" requires finite values that are all zero
    _a = None if act is None else np.asarray(act, float)[:nF][idx]
    _fin = None if _a is None else np.isfinite(_a)
    flat = _a is not None and bool(_fin.any()) and not bool(np.any(_a[_fin] != 0.0))
    # with a declared LUT, flat means flat in the columns it draws: `act` is one column (chan 0 by default), and a spec whose LUT draws other columns (exp 7's gene circuit in columns 3-5 beside a morphogen in column 0) was painted BODY_GREY on every no-source control; no LUT: the test above stands
    if flat and chem is not None and lut:
        _c = np.asarray(chem, float)[:nF][idx]
        _drawn = [k for k, h in enumerate(lut) if h is not None and k < _c.shape[1]]
        if _drawn:
            _v = _c[:, _drawn]
            _vf = np.isfinite(_v)
            flat = bool(_vf.any()) and not bool(np.any(_v[_vf] != 0.0))
    # a declared LUT wins: `plotting.species` is a column-to-colour table and a property of the MODEL (Gray-Scott draws its activator and hides its substrate, May-Leonard's partition wants red / green / blue); the 2D path always honoured it, the mesh path did not
    rgb = x = None
    if chem is not None and lut and not flat:
        from plexus.live import chem_rgb
        # `vmax` is `plotting.chem_max`, the spec's fixed colour scale, which the live renderer honoured and this replay path dropped (a knocked-out gene at ~1e-9 was normalised by its own maximum and drawn at full colour: exp 7, the Pax6-/- movies); None keeps the per-frame law
        _cols, _ = chem_rgb(np.asarray(chem, float)[:nF][idx], lut=lut, blend=blend, vmax=vmax)
        if _cols is not None:
            rgb = (np.clip(_cols, 0, 1) * 255).astype(np.uint8)
            # `x` is what the suppression tint weighs itself by, so it must stay the activator's own 0..1 position, not something derived from the colour
            _lo = float(np.nanmin(_a)) if lo is None else lo
            _hi = float(np.nanmax(_a)) if hi is None else hi
            x = np.clip((_a - _lo) / (_hi - _lo + 1e-9), 0, 1)
    if rgb is None and (act is None or flat):
        # `plotting.mesh_color` overrides the grey (the live movie's key for the same thing); a spec asking for white faces has decided the marks read against white
        _body = (_PLOT_OVERRIDE or {}).get("mesh_color")
        from matplotlib.colors import to_rgb as _to_rgb
        _bc = BODY_GREY if not _body else tuple(int(round(255 * c)) for c in _to_rgb(_body))
        rgb = np.full((len(idx), 3), _bc, np.uint8)
        x = np.zeros(len(idx))
    if rgb is None and act is not None and not flat:
        a = np.asarray(act, float)[:nF][idx]
        ok = np.isfinite(a)
        # the range is the run's, not the frame's, on a movie: otherwise every frame renormalises and a strengthening pattern looks static (the same defect as the camera autofit that hid growth)
        _lo = float(np.nanmin(a)) if lo is None else lo
        _hi = float(np.nanmax(a)) if hi is None else hi
        x = np.clip((a - _lo) / (_hi - _lo + 1e-9), 0, 1)
        rgb = (np.asarray(_cmap()(x))[:, :3] * 255).astype(np.uint8)
        rgb[~ok] = (255, 26, 217)                 # magenta: not a cell any more
    # `plotting.mesh_color_by: <label block>` gives a category per cell (exp 14: `clone`, the founder a cell descends from, paints the mosaic of lineages; `mutant` the two populations); the same rule as `live_movie._mesh_face_rgb`: `label_colors` colours values 0, 1, ... in order, any other value takes a hue hashed from its number (not stepped by the golden ratio: the Fibonacci sphere puts indices a Fibonacci number apart side by side, which step to near-equal hues), so a clone keeps its colour in every frame; it replaces the body colour and the division and death marks still draw over it
    _lab = str((_PLOT_OVERRIDE or {}).get("mesh_color_by", "") or "")
    # `node_type` (alias `type`) is the cell set's type column, per frame (see `_core_frames`); a float block drawn through a colormap instead of as categories is opt-in, `mesh_color_scale` (see `mesh_block_continuous`)
    _lv = mesh_color_values(mt, _lab)
    if _lab and _lab.lower() not in ("phase", "cycle_progress") and _lv is not None \
            and (color_rng is not None
                 or mesh_scale_continuous(np.asarray(_lv, float)[:nF][idx], _PLOT_OVERRIDE or {})):
        rgb = mesh_block_continuous(np.asarray(_lv, float)[:nF][idx], _PLOT_OVERRIDE or {}, color_rng)
    elif _lab and _lab.lower() not in ("phase", "cycle_progress") and _lv is not None:
        from plexus.measures import label_rgb
        _q = np.rint(np.asarray(_lv, float)[:nF][idx]).astype(np.int64)
        _pal = (_PLOT_OVERRIDE or {}).get("label_colors") or []
        _c = np.empty((len(idx), 3))
        for k in np.unique(_q):
            _c[_q == k] = label_rgb(int(k), _pal)
        rgb = (_c * 255).astype(np.uint8)
    mother, daughter, kills, sup = _marks(mt, idx, nF, prev_nF)
    # order matters: suppression is the background, the division pair next, death last and wins outright (the rarer and more consequential event, and blue is also the mother's colour, so a dying cell that had just divided would say "mother" where it means "about to be extruded")
    if sup is not None and sup.any():
        # a blend, not a threshold: `x < 0.5` on a smooth activator toggles every cell near the line, so whole regions flipped blue and back between frames; weighting by how LOW the activator is makes a half-way cell half blue and nothing jumps
        w = (sup.astype(float) if act is None else sup * (1.0 - x)) [:, None]
        rgb = ((1.0 - w) * rgb.astype(float) + w * np.asarray(BLUE, float)).astype(np.uint8)
    # the pair, before death, blended (at PAIR_A) rather than painted, as the green tint was: solid colour throws away what the cell WAS (its activator level, or the blue of the second field) to say one bit
    for mask, col_ in ((mother, MOTHER), (daughter, DAUGHTER)):
        if show_div and mask is not None and mask.any():
            rgb[mask] = ((1.0 - PAIR_A) * rgb[mask].astype(float)
                         + PAIR_A * np.asarray(col_, float)).astype(np.uint8)
    # death follows division's rule, for its reason: both name ONE CELL, which must be visible (on the smooth surface a blue patch with no outlines marks a cell nobody can see); suppression tints a whole REGION, so it stays in both styles
    if show_div and kills is not None and kills.any():
        rgb[kills] = KILL
    # green only where both its axes exist: it marks ONE CELL that divided in the last four calls, so it needs a before and an after (`kburns` is one frame held still) and a visible cell (`nomesh` gives none), hence `evolve` + `mesh` only; blue marks a REGION and reads on a smooth surface in a single frame. The single-colour green tint this replaced stays in `GREEN` / `GREEN_A` for a caller wanting "a division happened" without the pairing
    m.cell_data["rgb"] = rgb
    return m


# add: Add the shell and return its actor, so the lights survive (`Plotter.clear()` removes them) -> actor.
def add(p, m, style):
    """Add the shell and RETURN THE ACTOR, which is the whole reason this is not `p.clear()`.

    `Plotter.clear()` removes every actor AND every light: measured 5 lights before, 0 after. So
    `evolve` would render its first frame lit and every other one with no light source at all --
    mean brightness 195.6 -> 238.9 and shading contrast 47.5 -> 33.5, a washed-out white body, with
    "the first frame is better shaded than the last" as the signature. Removing the one actor we added
    leaves the lighting rig alone.
    """
    return p.add_mesh(m, scalars="rgb", rgb=True, lighting=True,
                      # `flat` (added 16 August, asked for by name): one normal per cell, so each cell is a facet and the mesh reads through the shading with no stroke (an outline is a constant 0.4 px whatever the cell's size, a grey wash at montage scale); a curved arm reads as a faceted cone, and `nomesh` stays the default for the shape question
                      smooth_shading=(style != "flat"),
                      show_edges=(style == "mesh"), edge_color="black", line_width=0.4,
                      ambient=0.35, diffuse=0.75, specular=0.12, specular_power=18)


# aim: Point the camera at the run's box; `fill` < 1 pulls it back.
def aim(p, L, azim=None, elev=None, fill=1.0):
    """`fill` < 1 pulls the camera back: the run's box then occupies that fraction of the frame. `L` is set by the widest thing the run ever reaches, so a star's box is its ARMS and its body looks small, while a run whose body is the widest thing (a sphere that only grows, a vesicle that only shrinks) fills the card edge to edge as a close-up; the fraction makes two such cards the same size on a page."""
    # `plotting.camera: {elev, azim}` in degrees is the spec's own view of the run (a flat disc in z = 0 was always drawn from 18 degrees above its plane, as a sliver; a sheet wants a top view, elev 90); the explicit arguments (the turntable's sweep) win over the spec, which wins over CAM
    _cam = (_PLOT_OVERRIDE or {}).get("camera") or {}
    elev = float(_cam.get("elev", CAM["elev"])) if elev is None else elev
    azim = float(_cam.get("azim", CAM["azim"])) if azim is None else azim
    e = np.radians(elev)
    a = np.radians(azim)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    p.camera.position = tuple(d * L * 3.2)
    p.camera.focal_point = (0, 0, 0)
    # looking straight down, +z is the view direction and cannot also be "up" (VTK picks an arbitrary roll and the sheet spins between frames): within 5 degrees of the pole screen-up is world +y, so a top view is a map, x to the right and y up
    p.camera.up = (0, 1, 0) if abs(elev) > 85.0 else (0, 0, 1)
    p.camera.parallel_projection = True
    p.camera.parallel_scale = L * 1.05 / max(fill, 1e-3)


# _ease: Easing curve of a 0..1 parameter -> float.
def _ease(u):
    return u * u * (3.0 - 2.0 * u)


# offscreen: Make VTK stop trying to reach an X server, and stop it printing when it fails -> None.
def offscreen():
    """Make VTK stop trying to talk to an X server, and stop it printing when it fails. The noise is Xlib's, not VTK's (filtering python warnings never silenced it): a VS Code remote session exports `DISPLAY=:11` with no Xauthority the container can read, so every render window printed "Authorization required..." and "vtkXOpenGLRenderWindow: bad X server connection", twice per plotter, from C straight to fd 2. Unsetting DISPLAY means the attempt is never made (VTK goes to EGL / OSMesa directly); nothing here wants an on-screen window (`OFF_SCREEN` is set unconditionally). Idempotent, before or after `import pyvista`."""
    os.environ.pop("DISPLAY", None)
    try:
        import vtkmodules.vtkCommonCore as _vcc          # keep VTK's own warnings off the terminal
        _vcc.vtkObject.GlobalWarningDisplayOff()
    except Exception:
        pass



# open_movie: Open an mp4 that is readable while it is being written, and after a run is killed -> writer.
def open_movie(p, out, framerate, quality=8):
    """Open an mp4 that is READABLE WHILE IT IS BEING WRITTEN, and after a run is killed. A plain mp4 puts its index (the `moov` atom) at the END, so a run that dies or runs out of memory leaves `ftyp` + `mdat`, good H.264 that no player opens (si_fill lost an 86 MB movie of a 2M-particle run this way). `frag_keyframe+empty_moov` writes a self-contained index in front and one per fragment, so the file is valid from the first frame; `-flush_packets 1` is the easy-to-miss part (ffmpeg buffers, and an unflushed fragmented file sits at 36 bytes after 40 frames; measured readable at 232 frames while the writer was at 280). `-g 1` makes every frame a keyframe: larger, but cuttable anywhere."""
    p.open_movie(out, framerate=max(1, int(round(framerate))), quality=quality,
                 output_params=["-movflags", "frag_keyframe+empty_moov+default_base_moof",
                                "-g", "1", "-flush_packets", "1"])


# _plotter: An off-screen PyVista plotter with the run's background and lighting -> Plotter.
def _plotter(size=None):
    offscreen()
    import pyvista as pv
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=(size or SIZE, size or SIZE))
    p.set_background("black")
    p.enable_anti_aliasing("msaa", multi_samples=8)
    return p


# kburns: The finished specimen, turned once and zoomed in, geometry fixed and camera moving -> path of the mp4.
def kburns(run_dir, style, out, fill=1.0, label=None):
    """The finished specimen, turned once and zoomed in. Geometry fixed, camera moving."""
    fr = frames_of(run_dir)
    _lut, _blend, _cut = style_of(run_dir)
    _pl = plot_style(run_dir)
    _ec = str(_pl.get("edge_color", "") or "").lower()          # myosin | type | ""
    # division marks off when something else is being shown: `show_div` paints mothers and daughters over the face colour, right for watching a tissue grow and wrong while the picture is about myosin (two unrelated quantities on one surface); declared, so a spec says what its picture is about
    _div = bool(_pl.get("show_division", True))
    if not fr:
        return "no trajectory"
    L0 = box_of(run_dir, fr)
    name = label or os.path.basename(run_dir.rstrip("/"))
    pos, mt, act, _chem = fr[-1]
    m = mesh_of(pos, mt, act, show_div=False, chem=_chem, lut=_lut, blend=_blend,
                cutaway=_cut, vmax=_pl.get("chem_max"), color_rng=mesh_color_rng(fr))
    n = int(KB_SECONDS * FPS)
    p = _plotter(); add(p, m, style)
    p.add_text(f"{name}  {style}", position="upper_left", font_size=11, color="white")
    open_movie(p, out, FPS)
    for i in range(n):
        u = i / (n - 1)
        aim(p, L0 * (1.0 - (1.0 - KB_ZOOM) * _ease(u)),
            azim=CAM["azim"] + 360.0 * u,                 # a FULL turn, so the clip loops
            elev=CAM["elev"] + 12.0 * np.sin(np.pi * u), fill=fill)
        p.write_frame()
    p.close()
    return f"{n} frames"


# evolve: The run through time, camera nailed down -> path of the mp4.
def evolve(run_dir, style, out, fill=1.0, label=None, max_frames=None):
    """The run through time, camera nailed down."""
    fr = frames_of(run_dir)
    _lut, _blend, _cut = style_of(run_dir)
    _pl = plot_style(run_dir)
    _ec = str(_pl.get("edge_color", "") or "").lower()          # myosin | type | ""
    # DIVISION MARKS OFF WHEN SOMETHING ELSE IS BEING SHOWN -- see `evolve` for why.
    _div = bool(_pl.get("show_division", True))
    if not fr:
        return "no trajectory"
    ticks_all = _frames_ticks.value
    cap = MAX_EVOLVE_FRAMES if max_frames is None else int(max_frames)
    if cap and len(fr) > cap:
        keep = np.unique(np.linspace(0, len(fr) - 1, cap).astype(int))
        fr = [fr[i] for i in keep]
        _frames_ticks.value = ([ticks_all[i] for i in keep] if ticks_all is not None
                               else keep.tolist())
    L = box_of(run_dir, fr)
    name = label or os.path.basename(run_dir.rstrip("/"))
    # one activator range for the whole clip, over every recorded frame (per-frame normalisation makes a strengthening pattern look constant)
    vals = [np.asarray(a, float) for _p, _m, a, _c in fr if a is not None]
    lo = float(min(np.nanmin(v) for v in vals)) if vals else 0.0
    hi = float(max(np.nanmax(v) for v in vals)) if vals else 1.0
    p = _plotter()
    open_movie(p, out, EV_FPS)
    # the pair window, in ticks, matched to how long a mother stays marked: `age <= DIVIDED` is four division CALLS and `cell_divide` runs every four ticks, so about sixteen ticks; the daughter threshold is the face count from sixteen ticks ago, not from the previous frame (three rows in four of a stride-1 trajectory drew hundreds of mothers and no daughters)
    ticks = getattr(_frames_ticks, "value", None)
    nFs = [int(m_["nF"]) for _p, m_, _a, _c in fr]
    actor = txt = eactor = None
    # one range for the whole clip, as for the activator: a per-frame myosin range would make a tightening belt look constant
    _erng = None
    if _ec == "myosin":
        _vals = [np.asarray(m_["e_myo"], float)[np.asarray(m_["E_face"]) < int(m_["nF"])]
                 for _p, m_, _a, _c in fr if m_.get("e_myo") is not None]
        if _vals:
            _erng = (float(min(np.nanmin(v) for v in _vals)),
                     float(max(np.nanmax(v) for v in _vals)))
    _crng = mesh_color_rng(fr)                    # None unless `mesh_color_scale` asks for a colormap
    for t, (pos, mt, act, _chem) in enumerate(fr):
        back = _pair_reference(t, nFs, ticks, PAIR_TICKS)
        m = mesh_of(pos, mt, act, lo, hi, show_div=(style == "mesh" and _div), prev_nF=back,
                    chem=_chem, lut=_lut, blend=_blend, cutaway=_cut, vmax=_pl.get("chem_max"),
                    color_rng=_crng)
        if m is None:
            continue
        if actor is not None:
            p.remove_actor(actor)                  # NOT p.clear(): that removes the lights too
        if txt is not None:
            p.remove_actor(txt)
        if eactor is not None:
            p.remove_actor(eactor)
        actor = add(p, m, style)
        if _ec:
            em = edges_of(pos, mt, _ec, ntype=None, rng=_erng,
                          colors=list((_pl.get("colors") or {}).values()) or None,
                          lut=str(_pl.get("edge_lut", "inferno")))
            eactor = None if em is None else p.add_mesh(
                em, scalars="rgb", rgb=True, line_width=_pl.get("edge_width", 3.0),
                lighting=False, render_lines_as_tubes=True)
        txt = p.add_text(f"{name}  {style}   frame {t + 1}/{len(fr)}   {int(mt['nF'])} cells",
                         position="upper_left", font_size=11, color="white")
        aim(p, L, fill=fill)
        p.write_frame()
    p.close()
    return f"{len(fr)} frames"


# still: One frame as an image, the VTK successor to `3d.png` -> path.
def still(run_dir, style="flat", out=None, fill=1.0, frame=-1, label=True, traj=None, name=None):
    """One frame as an image: the VTK successor to `3d.png`. Flat by default here and only here: `nomesh` stays the choice for movies (the silhouette question, where a faceted cone loses the curvature), but a still tiled at ~190 px is another question: a 0.4 px outline is a grey wash and smooth shading a smooth blob, while facets scale with the cells they belong to."""
    fr = frames_of(run_dir, traj)
    if not fr:
        return "no trajectory"
    L = box_of(run_dir, fr)
    nm = name or os.path.basename(run_dir.rstrip("/"))
    pos, mt, act, _chem = fr[frame][:4]
    m = mesh_of(pos, mt, act, show_div=(style == "mesh"), chem=_chem,
                lut=style_of(run_dir)[0], blend=style_of(run_dir)[1],
                cutaway=style_of(run_dir)[2], vmax=plot_style(run_dir).get("chem_max"),
                color_rng=mesh_color_rng(fr))
    p = _plotter()
    add(p, m, style)
    if label:
        p.add_text(f"{nm}  frame {len(fr) - 1 if frame == -1 else frame}",
                   position="upper_left", font_size=11, color="white")
    aim(p, L, fill=fill)
    p.screenshot(out or os.path.join(run_dir, "3d.png"))
    p.close()
    return f"{len(fr)} frames, drew {frame}"


# compare: A and B side by side in one clip, stepped together -> path of the mp4.
def compare(dir_a, dir_b, out, style="mesh", fill=1.0, labels=("A", "B"), title="",
            max_frames=None):
    """A and B SIDE BY SIDE in one clip, stepped together: the comparison artefact. Two directories each holding a movie is two movies, not a comparison (scrub both to the same moment and hold one in your head); a digest says whether they agree, this says WHERE and HOW they differ.

    Both panels draw the same frame index, subsampled identically, on a SHARED camera box (the max of the two: a per-side box would silently rescale one panel and hide a size difference). If the sides record different row counts (okuda ~60, the core every tick) each is subsampled to the same count first, so panel k of the left is panel k of the right in RUN time, not row index."""
    import pyvista as pv
    fa, ta = frames_of(dir_a), _frames_ticks.value
    fb, tb = frames_of(dir_b), _frames_ticks.value
    if not fa or not fb:
        return "no trajectory on " + ("A" if not fa else "B")
    n = min(MAX_EVOLVE_FRAMES if max_frames is None else int(max_frames), len(fa), len(fb))
    ia = np.unique(np.linspace(0, len(fa) - 1, n).astype(int))
    ib = np.unique(np.linspace(0, len(fb) - 1, n).astype(int))
    n = min(len(ia), len(ib))
    L = max(box_of(dir_a, fa), box_of(dir_b, fb))
    # one activator range across BOTH sides, as for the camera box (a per-side range would normalise away the difference the picture shows); FOUR elements, not three: `frames_of` yields (pos, mesh, activator, colour) and a three-way unpack raised on every call, was swallowed by the caller's try/except as "comparison render skipped", and the twin suite produced compare.png but never compare.mp4
    vals = [np.asarray(a, float) for f in (fa, fb) for _p, _m, a, _c in f if a is not None]
    lo = float(min(np.nanmin(v) for v in vals)) if vals else 0.0
    hi = float(max(np.nanmax(v) for v in vals)) if vals else 1.0

    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=(2 * SIZE, SIZE), shape=(1, 2), border=False)
    p.set_background("black")
    p.enable_anti_aliasing("msaa", multi_samples=8)
    open_movie(p, out, EV_FPS)
    # one label for both panels, but each side keeps ITS OWN ticks for `_pair_reference`: "what time is this frame" is shared (the panels are aligned), "what was nF a few ticks ago on THIS side" is per-side, and the other side's ticks would mis-date the just-divided highlight
    _lab_tk, _lab_idx = (ta, ia) if _has_ticks(ta) else (tb, ib)
    nFa = [int(m["nF"]) for _q, m, _a, _c in fa]
    nFb = [int(m["nF"]) for _q, m, _a, _c in fb]
    keep = [None, None]
    for k in range(n):
        when = _when(_lab_tk, int(_lab_idx[k]), max(len(fa), len(fb)))
        for col, (fr, idx, nFs, tk, lab) in enumerate(
                ((fa, ia, nFa, ta, labels[0]), (fb, ib, nFb, tb, labels[1]))):
            t = int(idx[k])
            p.subplot(0, col)
            if keep[col] is not None:
                p.remove_actor(keep[col])
            pos, mt, act, _chem = fr[t]
            m = mesh_of(pos, mt, act, lo, hi, show_div=(style == "mesh"),
                        prev_nF=_pair_reference(t, nFs, tk, PAIR_TICKS))
            if m is None:
                continue
            keep[col] = add(p, m, style)
            p.add_text(f"{lab}   {when}   {int(mt['nF'])} cells",
                       position="upper_left", font_size=10, color="white", name=f"t{col}")
            aim(p, L, fill=fill)
        if title:
            p.subplot(0, 0)
            p.add_text(title, position="lower_left", font_size=9, color="#9a9a9a", name="ttl")
        p.write_frame()
    p.close()
    return f"{n} paired frames"


# _when: When a panel is, in simulation time rather than row index -> str.
def _when(tk, t, n_rows):
    """WHEN this panel is, in SIMULATION TIME rather than row index. The sides do not record at the same rate (okuda's `RunArchive` keeps ~60 rows, gate 00 ticks [0, 6, 13, ... 394, 401]; the core keeps all 402), and a caption printing the row index made the two panels of a byte-identical run read "frame 60" beside "frame 402", inviting the very conclusion the picture exists to rule out (both were tick 401).

    One side's ticks label BOTH panels: okuda writes a `ticks` array, the core's `trajectory.npz` does not, and the panels are aligned by construction (the comparison selects core rows through okuda's own `ticks`), so `compare` / `compare_still` pass the one that exists. Falls back to the row index, and says so, only when NEITHER side recorded ticks."""
    try:
        return f"tick {int(np.asarray(tk).ravel()[t])}"
    except Exception:
        return f"row {t + 1}/{n_rows} (no ticks)"


# _has_ticks: Whether a side recorded a tick array at all -> bool.
def _has_ticks(t):
    """Whether a side recorded a tick array at all -- see `_when`."""
    try:
        return bool(np.asarray(t).ravel().size)
    except Exception:
        return False


# compare_still: The last frame of both sides, side by side -> path.
def compare_still(dir_a, dir_b, out, style="flat", fill=1.0, labels=("A", "B"), title=""):
    """The last frame of both sides, side by side. The still the movie is scrubbed from."""
    import pyvista as pv
    fa, ta = frames_of(dir_a), _frames_ticks.value
    fb, tb = frames_of(dir_b), _frames_ticks.value
    if not fa or not fb:
        return "no trajectory on " + ("A" if not fa else "B")
    L = max(box_of(dir_a, fa), box_of(dir_b, fb))
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=(2 * SIZE, SIZE), shape=(1, 2), border=False)
    p.set_background("black")
    p.enable_anti_aliasing("msaa", multi_samples=8)
    _lab_tk, _lab_n = (ta, len(fa)) if _has_ticks(ta) else (tb, len(fb))
    when = _when(_lab_tk, _lab_n - 1, _lab_n)
    for col, (fr, lab) in enumerate(((fa, labels[0]), (fb, labels[1]))):
        p.subplot(0, col)
        pos, mt, act, _chem = fr[-1]
        add(p, mesh_of(pos, mt, act, show_div=False), style)
        p.add_text(f"{lab}   {when}   {int(mt['nF'])} cells",
                   position="upper_left", font_size=10, color="white")
        aim(p, L, fill=fill)
    if title:
        p.subplot(0, 0)
        p.add_text(title, position="lower_left", font_size=9, color="#9a9a9a")
    p.screenshot(out)
    p.close()
    return "ok"


# use_plotting: Install the live spec's `plotting:` block for this process -> None.
def use_plotting(pl: dict | None) -> None:
    """Install the live spec's `plotting:` block for this process -- see `plot_style`."""
    global _PLOT_OVERRIDE
    _PLOT_OVERRIDE = None if pl is None else dict(pl)


# render_all: Run one named sequence for one finished run directory -> {name: path}.
def render_all(run_dir, seq=LOOP_SEQ, size=None, quiet=False, fill=1.0, name=None):
    """Run one named sequence for one finished run directory; {} if there is nothing to render. Sequence 3 (four clips, 1,020 frames) takes ~20 s on b_star's 12,272 cells against a 30-40 MINUTE GPU run, which makes the full set affordable from inside a generation."""
    global SIZE
    if size:
        SIZE = size
    if frames_of(run_dir) is None:
        return {}
    nm = name or os.path.basename(run_dir.rstrip("/"))
    took = {}
    for kind, st in SEQUENCES[int(seq)]:
        # the default pair is `movie.mp4` + `movie_kburns.mp4`, the names the point renderer writes, so a run folder looks the same whichever renderer made it (they were `vtk_evolve_mesh.mp4` / `vtk_kburns_mesh.mp4`, and `movie.mp4` was whatever the point renderer drew of a mesh set: a few hundred vertices in a corner); the style suffix survives only where it distinguishes (sequence 3 writes both the `mesh` and `nomesh` cuts)
        out = os.path.join(run_dir, {("evolve", "mesh"): "movie.mp4",
                                     ("kburns", "mesh"): "movie_kburns.mp4"}
                           .get((kind, st), f"movie_{kind}_{st}.mp4"))
        t0 = time.perf_counter()
        msg = (kburns if kind == "kburns" else evolve)(run_dir, st, out, fill=fill, label=nm)
        dt = time.perf_counter() - t0
        if msg.startswith("no "):
            return {}
        took[os.path.basename(out)] = dt
        if not quiet:
            print(f"  {kind:7s} {st:7s} {msg:12s} {dt:7.1f} s -> {out}", flush=True)
    return took


# available: Whether a VTK render is possible here -> bool.
def available():
    """Is a VTK render possible here? `plot.py` falls back to matplotlib when it is not -- a
    headless node with no GL is a real configuration and it must not take a generation down."""
    try:
        import pyvista                                    # noqa: F401
        return True
    except Exception:
        return False
