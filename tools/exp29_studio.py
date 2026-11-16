"""exp29 (cell_tissue_2): the INK-LOOK movie of one run, white cells that turn red with their morphogen, archived as a step
of the experiment's record (the watcher at http://127.0.0.1:8863/watch).

    python tools/exp29_studio.py <group>/<run> --why "..." [--seconds 10] [--stride n] [--cam elev,azim]
                                 [--cut] [--no-archive] [--replace N] [--vs RUN2] [--fake-act radial] [--out-dir /tmp]
    python tools/exp29_studio.py log/okuda/r013_05 --why "..."        # a vertex run: any folder holding traj.npz / trajectory.npz
    python tools/exp29_studio.py --self-check

<run> is `graphs_data/<group>/<run>` (or any folder with a `trajectory.npz` or an okuda `traj.npz`, relative to the repo or
absolute). The record is TOOL-WRITTEN through gui_drive's numbering (PLEXUS_BUILDER defaults to
experiments/exp29_cell_tissue_2): never run tools/exp_record.py 29 (TRAPS T-79).

WHAT IS DRAWN. Flat fills with dark ink outlines (src/plexus/gui/flat_ink.py, the look of the watcher's 3-D view) on the
flat grey ground #8d99a6, white overlays. Each CELL is white at activator 0 and red (#E63946) at the top of the scale, the
colour mixed linearly: rgb = (1 - s) white + s red, s = clip(a / a_top, 0, 1). a is the cell's activator, channel 0 of the
cell set's `chem` block (a Gray-Scott system's `a`); a cell of a run with no `chem` block is white. a_top is a statistic of
the WHOLE run, the 99.5th percentile of a over every live cell of every stored row, taken before the first frame is drawn
(INSTRUCTION.md, "Render after the full run"), so two frames are comparable. The legend under the cells prints a_top.

    MPM run      each cell (`pt__pid`) its own surface, the 0.5 iso-surface of its blurred point volume fraction
                 (tools/exp23_studio.py `_cell_surface`); the cell's volume is its live point count x the volume of one
                 point, (point_spacing)^3 from the spec's seed, times (length_um)^3 when the spec has units
    vertex run   each cell (a face of the recorded closed mesh) its polygons, fanned round the face's centroid and shrunk
                 toward it (the gap shows the ground, so the ink draws the wall); the cell's volume is the model's own, the wedge v_f = (1/3) c_f . N_f, the pyramid
                 from the tissue's centre (the mean vertex) to the face (c_f the face's centroid, N_f its area vector;
                 vertex_ops.face_geometry_3d), in model units^3 (a vertex clip's spec has no length unit)

THREE INSETS on the right (the user's, 2026-10-21), each axis range taken from the whole run, a dot / cursor at the frame:
    cells        the number of live cells against time (a cell is live when it owns a live point / is a live face)
    morphogen    the DISTRIBUTION of the activator over the cells against time: x time, y activator from 0 to a_top, each
                 time column a histogram of the live cells in 24 bins (fractions, columns sum to 1), drawn in the activator's
                 own colour (white to red) with a strength of log10(1 + 99 f / f_max) / 2, f the bin's fraction and f_max the
                 run's largest one (a bin with 1 % of f_max shows at 0.15, one with 10 % at 0.52); the dark line is the median
                 activator
    cell volume  the mean volume of a live cell (thick line) with the 10th to 90th percentile band, y from 0 to 1.05 x the
                 largest 90th percentile
Time is in the spec's unit (`units.time_s` 60: min, 3600: h) or frames when the spec has none.

LIMITS. A vertex run without `V0f` or with a mesh row whose half-edges name a face >= nF has those half-edges dropped (as
cell_chem_from_shape does). The old okuda `traj.npz` keeps about 60 rows, so its insets have 60 points.
"""
import argparse
import math
import os
import shutil
import sys
import time

os.environ.pop("DISPLAY", None)                      # VTK straight to EGL/OSMesa
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)
os.environ.setdefault("PLEXUS_BUILDER", os.path.join(REPO, "experiments", "exp29_cell_tissue_2"))
GD = os.path.join(REPO, "graphs_data")

WHITE = np.array([255.0, 255.0, 255.0])
RED = np.array([0xE6, 0x39, 0x46], float)            # flat_ink.PALETTE[0]
GROUND = "#8d99a6"
TILE_BG = np.array([0x76, 0x82, 0x8e], np.float32)   # the ink look's darker grey (exp23_studio LOOK["ink"]["inset_bg"])
N_BINS = 24
SIDE_W = 480                                         # the insets' column, px
TILE = (460, 300)                                    # one inset, px
AXIS_FONT = 10                                       # point size of an inset's axis names (its tick numbers are 1 smaller); `--axis-font`, 16 for the deck (Cedric, 2026-11-12: "larger fontsize for the panel plot axis")
CUT_KEEP = []                                        # `--cut-keep TYPE`: with `--cut`, the cells of these types are never clipped: drawn whole when the plane reaches them or they lie behind it, left out when they lie wholly on the camera's side
TYPE_OPACITY = {}                                    # `--type-opacity TYPE=OPACITY`: the opacity of the cells of a type, over the spec's `plotting.type_opacity` (1 = solid)
COUNT_BY_TYPE = False                                # `--count-by-type`: the inset of live cells per cell type (red, blue) even when the spec's plotting has no `count_panel: by_type`
CLOCK_CORNER = "bottom-right"                        # where the "N cells   t = T min" text sits; `--clock-corner top-left` (Cedric, 2026-11-12) puts it where the sweeps' text overlay goes


# ---------------------------------------------------------------------------------------------------------- the colour
WHITE_INK = {"on": False}                            # `--white-ink` (Cedric, 2026-11-14, exp06: "render all steps with white instead of black"): every outline and inset line white
DOTS = {"factor": None, "r": None}                   # `--dots F` (Cedric, 2026-11-14, exp06: "the render style of the 4D panel dot style, largest dot size"): every material point a flat round dot as the watcher's 4D view draws it (bio_view._dots_update), F x its radius (1.7 = the largest dot button)
WHITE_PLOT = {"on": False}                            # `--white-plot` (Cedric, 2026-11-14, exp06: "the text in white and the axis and text of the plot in white, not the cells"): the insets' median line white, the cells' ink untouched
ACT_LABEL = {"text": "activator a"}                     # `--activator-label`: what the cell block chem[:, 0] is, in the legend and the middle inset
CELL_COLOURS = {"on": False}                         # `--cell-colours`, or `plotting: {cell_colours: id}` in the spec (Cedric, 2026-10-28): every MPM cell its own colour, by its slot id
def cell_rgb(c):
    """RGB (0-255) of cell slot `c`: the watcher's own vivid palette (gui/flat_ink.py, as the 3D+t viewer uses it), keyed on the id modulo its 10 entries, so the movie and the 3D+t viewer colour a cell alike (Cedric, 2026-10-28)."""
    from plexus.gui import flat_ink
    # BYTES, NOT FLOATS (2026-10-31): a float colour array is read by VTK as 0-1 and multiplied by 255 into a byte, so 0-255 floats wrapped round to their complements (red drawn cyan) in
    # every id-coloured render made since this palette replaced the old one on 2026-10-28 17:36; uint8 arrays are taken as they are
    return np.rint(flat_ink.palette([int(c)])[0] * 255.0).astype(np.uint8)

def cell_rgb_R(R, c):
    """`cell_rgb` unless the run colours by cell TYPE (`plotting: {cell_colours: type, colors: {inner: '#3b6fd4', outer: '#e63946'}}`, exp29 rosette arrays, 2026-10-31): then the type's colour."""
    t = getattr(R, "type_rgb", None)
    if t is not None and int(c) < len(t):
        return np.rint(np.asarray(t[int(c)], float)).astype(np.uint8)
    return cell_rgb(c)


PULL_PANEL = {"on": False}                           # `--pull-panel`, or `plotting: {pull_panel: true}`: an inset, the force of the cell_pull springs on each array against time
JUNCTION_PANEL = {"on": False}                       # `--junction-panel`, or `plotting: {junction_panel: true}` in the spec (Cedric, 2026-10-28): an inset, the bound density of the junctions of each array against time
HEIGHT_PANEL = {"on": False}                         # `--height-panel`, or `plotting: {height_panel: true}` in the spec (Cedric, 2026-10-28): a 4th inset, the mean cell height of each family against time
CELL_LABELS = {"block": None, "value_only": True, "font": 24}   # `--label-block B`, or `plotting: {cell_labels: {block: B, value_only: true}}` in the spec (Cedric, 2026-10-28)


def label_spec(R):
    """(block, value_only, font) of the family labels, or None: the flag wins, else the spec's `plotting.cell_labels`."""
    if CELL_LABELS["block"]:
        return CELL_LABELS["block"], CELL_LABELS["value_only"], CELL_LABELS["font"], None
    c = ((R.spec or {}).get("plotting") or {}).get("cell_labels")
    if not c:
        return None
    return str(c["block"]), bool(c.get("value_only", True)), int(c.get("font", CELL_LABELS["font"])), (str(c["block2"]) if c.get("block2") else None)


def label_anchors(R, ids, centres):
    """One FIXED anchor per family (Cedric, 2026-10-28, 'the labels are flickering'): the old anchor was the family's centroid and top in THIS frame, which moves by 0.5 um from frame to frame (max 2 um) and read as a flicker.
    Now: x, y = the median over about 30 sampled rows of the family's mean position, z = the highest point of the family over those rows plus 0.5 length units. Cached on R."""
    cache = getattr(R, "_lab_anchor", None)
    if cache is not None:
        return cache
    rows = R.rows
    ks = sorted(set(range(0, rows, max(1, rows // 30))) | {rows - 1})
    xy = {f: [] for f in range(len(ids))}
    top = {f: -1e9 for f in range(len(ids))}
    for k in ks:
        X = R.positions(k)
        live = R.occ[k]
        for c in np.flatnonzero(R.live[k]).tolist():
            m = live & (R.pid[k] == c)
            if not m.any():
                continue
            f = int(np.argmin(((X[m].mean(0) - centres) ** 2).sum(1)))                  # in 3-D: chains in one column at different heights share x and y
            xy[f].append(X[m].mean(0))
            top[f] = max(top[f], float(X[m][:, 2].max()))
    cache = {}
    for f in range(len(ids)):
        if xy[f]:
            q = np.median(np.array(xy[f]), axis=0)
            q[2] = top[f] + 0.5
            cache[f] = q
    R._lab_anchor = cache
    return cache


def draw_family_labels(pl, R, k, X, live, spec):
    """One white, unboxed, non-bold text per FAMILY (the cells that descend from one cell of frame 0), at a FIXED anchor above the family (`label_anchors`): the value of the per-cell block `spec[0]` at frame 0
    (a daughter keeps its mother's), alone or after `id <slot>`; a NaN value carries no label."""
    block, value_only, font, block2 = spec                  # `block2`: a second per-cell block, the label then reads "first / second" (a grid of two swept parameters)
    z = R.z
    occ0 = z["cell__occ"][0] > 0.5
    centres = z["cell__pos"][0][occ0]
    ids = np.flatnonzero(occ0)
    vals = np.arange(1, len(ids) + 1, dtype=float) if block == "family" else z[f"cell__{block}"][0][occ0][:, 0]     # block `family`: no block to read, the family's number 1..n in slot order (a baseline run has no swept value to print)
    anchors = label_anchors(R, ids, centres)
    pts, labs = [], []
    # `--label-once` / `plotting.cell_labels.once: true` (exp39, 2026-11-07, Cedric: "too many labels"): ONE label per ARRAY (the cell block `arr`), above the array, instead of one per family: a pair of 13-cell spheroids
    # has 26 families that all carry the swept value
    once = bool(CELL_LABELS.get("once")) or bool((((R.spec or {}).get("plotting") or {}).get("cell_labels") or {}).get("once", False))
    groups = [[f] for f in range(len(ids))]
    if once and "cell__arr" in z.files:
        arr = np.rint(z["cell__arr"][0][occ0][:, 0]).astype(int)
        groups = [np.flatnonzero(arr == a).tolist() for a in np.unique(arr)]
    for g in groups:
        fs = [f for f in g if f in anchors]
        f0 = next((f for f in fs if np.isfinite(vals[f])), None)
        if f0 is None:
            continue
        pts.append(anchors[f0] if len(g) == 1 else np.array([np.mean([anchors[f][0] for f in fs]), np.mean([anchors[f][1] for f in fs]), max(anchors[f][2] for f in fs)]))
        labs.append((f"{vals[f0]:g}" if value_only else f"id {ids[f0]}   {block} {vals[f0]:g}") + (f" / {z[f'cell__{block2}'][0][occ0][:, 0][f0]:g}" if block2 else ""))
    if pts:
        pl.add_point_labels(np.array(pts), labs, font_size=font, text_color="white", shape=None, bold=False, show_points=False, always_visible=True, name="family_labels")


def junction_on(R) -> bool:
    return bool(JUNCTION_PANEL["on"]) or bool((((R.spec or {}).get("plotting") or {}).get("junction_panel", False)))


def array_label_values(R):
    """The per-cell block whose first column is each array's printed label: the SWEEP value (the spec's `plotting.cell_labels.block`, the
    number the movie writes on each family -- 5 to 400 for the rosette stiffness), else the adhesion maximum `adh` as before. Printing `adh`
    gave every array the same "1" in a sweep that does not vary the adhesion (Cedric, 2026-11-01)."""
    z = R.z
    ls = label_spec(R)
    if ls is not None and f"cell__{ls[0]}" in z.files:
        return z[f"cell__{ls[0]}"][0]
    return z["cell__adh"][0] if "cell__adh" in z.files else None


def junction_series(R):
    """The BOUND DENSITY of the junctions of each ARRAY against time: for every row the sum of rho over the live junction rows whose first cell belongs to the array (the cell block `arr`), divided by the number of
    contacts a chain of the array's cells has (its cells - 1). Returns (S [rows, narr], colours, labels): the colour of an array is its first cell's, its label the lateral-lateral adhesion maximum `adh[0]` of that cell."""
    z = R.z
    rows = R.rows
    rho = z["junction__rho"][:rows].reshape(rows, -1)
    jca = np.rint(z["junction__ca"][:rows].reshape(rows, -1)).astype(int)
    jocc = z["junction__occ"][:rows].reshape(rows, -1) > 0.5
    arr = np.rint(z["cell__arr"][:rows].reshape(rows, -1)).astype(int)
    occ0 = z["cell__occ"][0] > 0.5
    ids = np.flatnonzero(occ0)
    arrs = sorted(set(arr[0][ids].tolist()))
    first = {a: int(ids[arr[0][ids] == a].min()) for a in arrs}
    n_in = {a: int((arr[0][ids] == a).sum()) for a in arrs}
    S = np.zeros((rows, len(arrs)))
    for t in range(rows):
        live = np.flatnonzero(jocc[t])
        if live.size:
            np.add.at(S[t], [arrs.index(a) for a in arr[t][jca[t][live]]], rho[t][live])
    S = S / np.array([max(n_in[a] - 1, 1) for a in arrs])[None, :]
    adh = array_label_values(R)
    return (S,) + array_style(arrs, first, adh)


def array_style(arrs, first, adh):
    """Colours and labels of the arrays: with up to 12 arrays each in its first cell's palette colour and labelled with its adhesion maximum; with more, a sequential colour (viridis) by the rank of the maximum and a label on
    about six of them (the others ''), so that a sweep of dozens of arrays stays readable."""
    labs = [f"{adh[first[a], 0]:g}" for a in arrs] if adh is not None else None
    if len(arrs) <= 12 or adh is None:
        return [tuple(cell_rgb(first[a]) / 255.0) for a in arrs], labs
    import matplotlib
    order = np.argsort(np.argsort([adh[first[a], 0] for a in arrs]))
    cmap = matplotlib.colormaps["viridis"]
    step = max(1, len(arrs) // 6)
    return [tuple(cmap(float(o) / max(len(arrs) - 1, 1))[:3]) for o in order], [lb if (int(o) % step == 0 or int(o) == len(arrs) - 1) else "" for lb, o in zip(labs, order)]


def pull_on(R) -> bool:
    return bool(PULL_PANEL["on"]) or bool((((R.spec or {}).get("plotting") or {}).get("pull_panel", False)))


def pull_series(R):
    """The FORCE the pull springs of `cell_pull` put on each ARRAY against time: for a pulled cell the spring force along its pull direction d, k (c0 + s(t) d - x(t)) . d, with c0 the centroid at the frame `delay`,
    s = min(speed (t - delay), max_disp) (the operator's own law, read from the spec; the force is recoverable from the trajectory), averaged over the array's pulled cells. Its peak is the force the chain carried before it let go.
    Returns (F [rows, narr], colours, labels, peaks)."""
    z = R.z
    rows = R.rows
    op = next(o for o in R.spec["operators"] if o["op"] == "cell_pull")
    k_, speed, delay, mx = float(op.get("k", 50.0)), float(op.get("speed", 0.1)), float(op.get("delay", 10)), float(op.get("max_disp", 3.0))
    gen = R.spec["general"]
    dt = float(gen.get("dt", 1.0))
    frame = np.linspace(0.0, float(gen["n_frames"]), rows)
    arr = np.rint(z["cell__arr"][0].reshape(-1)).astype(int)
    pull = z["cell__pull"][0]
    ids = np.flatnonzero(z["cell__occ"][0] > 0.5)
    arrs = sorted(set(arr[ids].tolist()))
    first = {a: int(ids[arr[ids] == a].min()) for a in arrs}
    t0 = int(np.argmax(frame >= delay))
    c0 = R.centroids(t0)
    s_t = np.minimum(speed * np.maximum(frame - delay, 0.0) * dt, mx)
    F = np.zeros((rows, len(arrs)))
    n_p = np.zeros(len(arrs))
    cen = [R.centroids(t) for t in range(rows)]
    for c in ids:
        d = pull[c]
        if np.linalg.norm(d) < 0.5:
            continue
        a = arrs.index(int(arr[c]))
        n_p[a] += 1
        for t in range(t0, rows):
            F[t, a] += k_ * float(((c0[c] + s_t[t] * d - cen[t][c]) @ d))
    F = F / np.maximum(n_p, 1)[None, :]
    adh = array_label_values(R)
    cols, labs = array_style(arrs, first, adh)
    return F, cols, labs, F.max(0)


def height_on(R) -> bool:
    return bool(HEIGHT_PANEL["on"]) or bool((((R.spec or {}).get("plotting") or {}).get("height_panel", False)))


def family_height(R):
    """The mean HEIGHT of the cells of each FAMILY (the cells that descend from one cell of frame 0) against time. A cell's height is the extent of its live points along z, the normal of the array
    (max z - min z), in world units; a family's value at a row is the mean over its live cells at that row (NaN while it has none). Returns (h [rows, nfam], the founder cell's slot id of each
    family). A cell belongs to the family whose frame-0 centre is nearest to its centroid in xy."""
    z = R.z
    occ0 = z["cell__occ"][0] > 0.5
    centres = z["cell__pos"][0][occ0][:, :2]
    ids = np.flatnonzero(occ0)
    nfam, ncell = len(ids), R.live.shape[1]
    h = np.full((R.rows, nfam), np.nan)
    for t in range(R.rows):
        m = R.occ[t]
        pid, Z = R.pid[t][m], R.positions(t)[m][:, 2]
        zmax = np.full(ncell, -np.inf); zmin = np.full(ncell, np.inf)
        np.maximum.at(zmax, pid, Z); np.minimum.at(zmin, pid, Z)
        cells = np.flatnonzero(R.live[t])
        fam = np.argmin(((R.centroids(t)[cells][:, None, :2] - centres[None]) ** 2).sum(-1), axis=1)
        for f in range(nfam):
            c = cells[fam == f]
            if len(c):
                h[t, f] = float(np.mean(zmax[c] - zmin[c]))
    return h, ids


def by_id(R) -> bool:
    """Colour every cell by its own id instead of by activator: the `--cell-colours` flag or the spec's `plotting.cell_colours: id`."""
    return bool(CELL_COLOURS["on"]) or str((((R.spec or {}).get("plotting") or {}).get("cell_colours", ""))).lower() in ("id", "type")


def _spec_by_id(spec_path) -> bool:
    import yaml
    try:
        return str(((yaml.safe_load(open(spec_path)) or {}).get("plotting") or {}).get("cell_colours", "")).lower() in ("id", "type")
    except Exception:                                                                                  # noqa: BLE001
        return False


ACT_CMAP = {"name": None, "lo": 0.25, "hi": 1.0}       # `--act-cmap viridis` (Cedric, 2026-11-14, exp05: "plot the activity of cells, from blue not active to yellow active"): the activator through this matplotlib map from `lo` to `hi` of it (viridis 0.25 = blue #3b528b, 1.0 = yellow #fde725) instead of white to red


def act_rgb(a, a_top):
    """[n] activator -> [n, 3] uint8: white at 0, RED at a_top and above, linear between; with `--act-cmap` the map's colour at lo + s (hi - lo), s = clip(a / a_top, 0, 1)."""
    s = np.clip(np.asarray(a, np.float64) / max(float(a_top), 1e-12), 0.0, 1.0)
    if ACT_CMAP["name"]:
        import matplotlib
        return np.clip(np.rint(np.asarray(matplotlib.colormaps[ACT_CMAP["name"]](ACT_CMAP["lo"] + s * (ACT_CMAP["hi"] - ACT_CMAP["lo"])))[..., :3] * 255.0), 0, 255).astype(np.uint8).reshape(-1, 3)
    return np.clip(np.rint((1.0 - s)[:, None] * WHITE + s[:, None] * RED), 0, 255).astype(np.uint8)


def top_of_scale(a_rows):
    """The 99.5th percentile of the activator over every live cell of every row (`a_rows`: list of 1-D arrays), at least 1e-9."""
    vals = np.concatenate([np.asarray(a, np.float64).ravel() for a in a_rows if len(a)]) if a_rows else np.zeros(1)
    return max(float(np.percentile(vals, 99.5)), 1e-9) if vals.size else 1e-9


def density_image(a_rows, a_top, n_bins=N_BINS):
    """The morphogen inset's data: [n_bins, rows] fractions of the live cells per activator bin per row (each column sums to
    1, activator clipped into [0, a_top]; row 0 of the array is the LOWEST bin), and the per-row median."""
    H = np.zeros((n_bins, len(a_rows)))
    med = np.zeros(len(a_rows))
    for j, a in enumerate(a_rows):
        a = np.clip(np.asarray(a, np.float64), 0.0, a_top)
        if len(a):
            H[:, j] = np.bincount(np.minimum((a / a_top * n_bins).astype(int), n_bins - 1), minlength=n_bins) / len(a)
            med[j] = float(np.median(a))
    return H, med


# ---------------------------------------------------------------------------------------------------------- the runs
def resolve(run):
    """(directory, kind): kind 'mpm' (trajectory.npz with a pt set), 'vertex' (a mesh: trajectory.npz with
    `*__mesh_nF`, or an okuda traj.npz)."""
    cands = [os.path.join(GD, run), run, os.path.join(REPO, run)]
    for c in cands:
        if os.path.isdir(c):
            if os.path.exists(os.path.join(c, "traj.npz")):
                return c, "vertex"
            tp = os.path.join(c, "trajectory.npz")
            if os.path.exists(tp):
                import zipfile
                names = zipfile.ZipFile(tp).namelist()
                if any(n.endswith("__mesh_nF.npy") for n in names):
                    return c, "vertex"
                if "pt__pos.npy" in names:
                    return c, "mpm"
                if "cell__pos.npy" in names:                 # point cells (exp05, 2026-11-14): one centre per cell, no material points
                    return c, "point"
                raise SystemExit(f"{tp}: neither an MPM run (pt set), a point-cell run (cell set) nor a mesh run")
    raise SystemExit(f"no run folder with a trajectory for {run!r} (tried {cands})")


def read_spec(d):
    for n in ("spec.yaml", "spec_run.yaml"):
        if os.path.exists(os.path.join(d, n)):
            return yaml.safe_load(open(os.path.join(d, n))), os.path.join(d, n)
    rel = os.path.relpath(d, GD)
    p = os.path.join(REPO, "config", rel + ".yaml")
    if os.path.exists(p):
        return yaml.safe_load(open(p)), p
    return {"general": {}}, None


def _time_unit(spec):
    """(unit, scale): the clock's unit and the factor from the spec's time unit to it. `units.time_s` 60: min, 3600: h (scale 1); any other
    (exp05: 600 s, ten minutes) in hours when the run lasts 10 h or more, else in minutes; no unit: frames."""
    g = spec.get("general", {})
    ts = float((g.get("units") or spec.get("units") or {}).get("time_s", 0) or 0)
    if abs(ts - 60.0) < 1e-6:
        return "min", 1.0
    if abs(ts - 3600.0) < 1e-6:
        return "h", 1.0
    if ts > 0:
        T_s = float(g.get("n_frames", 0)) * float(g.get("dt", 1.0)) * ts
        return ("h", ts / 3600.0) if T_s >= 36000.0 else ("min", ts / 60.0)
    return "frames", 1.0


def time_axis(spec, ticks_or_rows):
    """(t [rows], unit): the spec's own time unit (see _time_unit); frames when it names none."""
    g = spec.get("general", {})
    unit, scale = _time_unit(spec)
    dt = float(g.get("dt", 1.0)) * scale
    if isinstance(ticks_or_rows, int):
        rows = ticks_or_rows
        stride = int(g.get("record_every") or 0) or math.ceil((int(g.get("n_frames", rows)) + 1) / rows)
        fr = np.array([k_ * stride + 1 for k_ in range(rows)], float)
    else:
        fr = np.asarray(ticks_or_rows, float)
    return (fr * dt if unit != "frames" else fr), unit


POINT_VOLUME = {"v": None}                           # `--point-volume V`: over the spec's own (exp05's MPM cells: a ball of volume 1 seeded with 125 points by voronoi_parent, 0.008, and their spec names no point_spacing)
SECTION_SLAB = {"h": None}                           # `--section-slab H` (Cedric, 2026-11-15, exp05: "render from now two panels full view and cross section"): the second panel of `--section` shows only the cells within H length units of the plane through the tissue's centre, seen face-on (a thin slice), instead of the whole far half
SECTION_VIEW = {"cam": None}                         # `--section-view elev,azim`: the camera of the cross-section panel (default: the main camera's)
NO_INSETS = {"on": False}                            # `--no-insets` (same message: "remove plot curves"): no inset column; the frame is the 3-D panel(s) alone, with the clock
MARK_COLOURS = {}                                    # `--mark-colours NAME=HEX ...` (exp05 routes, 2026-11-15): with --dots, the cells of these types are drawn in that colour and every other cell by its activity: a mark a cell keeps (cell_fate) seen over the signal
CHEM_RGB = {"cols": None}                            # `--chem-rgb HEX HEX HEX` (exp05 rock-paper-scissors, 2026-11-15): a point cell is drawn in the blend of these three colours by the shares of its first three chem channels
TYPE_COLOURS = {}                                    # `--type-colours NAME=HEX ...`: with `--cell-colours type`, the colour of a cell type over the spec's `plotting.colors`


def point_volume(spec):
    pv_ = POINT_VOLUME["v"] or (spec.get("plotting") or {}).get("point_volume")      # `plotting.point_volume`: the true volume of one point where the seed's `point_spacing` is not it (the vesicle seed subsamples a 0.125 lattice to 125 points: 0.008, not 0.125^3)
    if pv_:
        return float(pv_)
    for s in spec.get("seed", []):
        if s.get("point_spacing"):
            return float(s["point_spacing"]) ** 3
    return 0.125 ** 3


class MPMRun:
    """An MPM run read once: the points of the rows drawn, and per row the live points per cell slot (the cells' size and count)."""

    def __init__(self, d, spec, fake=None, npz=None):
        z = np.load(npz or os.path.join(d, "trajectory.npz"))
        self.spec = spec
        occ_all = z["pt__occ"].astype(bool)
        pid_all = np.rint(z["pt__pid"][..., 0]).astype(np.int64)
        self.rows = occ_all.shape[0]
        ncell = z["cell__occ"].shape[1]
        self.npts = np.zeros((self.rows, ncell), np.int64)          # live points per cell slot per row
        for t in range(self.rows):
            self.npts[t] = np.bincount(pid_all[t][occ_all[t]], minlength=ncell)[:ncell]
        self.live = self.npts > 0
        self.chem = z["cell__chem"][..., 0].astype(np.float64) if "cell__chem" in z.files else None
        self.cpos = z["cell__pos"].astype(np.float64) if "cell__pos" in z.files else None
        self.z, self.occ, self.pid = z, occ_all, pid_all
        self.type_rgb = None                                         # per cell slot RGB (0-255) when `plotting.cell_colours: type`, else None
        self.type_alpha = None                                       # per cell slot opacity (0-1) from `plotting.type_opacity`, else None
        _pl = (spec or {}).get("plotting") or {}
        if str(_pl.get("cell_colours", "")).lower() == "type" and _pl.get("colors") and "cell__node_type" in z.files and "cell__type_names" in z.files:
            import matplotlib.colors as _mc
            _names = [str(x) for x in z["cell__type_names"]]
            _tc = {str(k): np.asarray(_mc.to_rgb(v)) * 255.0 for k, v in _pl["colors"].items()}
            _nt = z["cell__node_type"]
            _nt = np.rint(_nt[-1] if _nt.ndim == 2 else _nt).astype(np.int64)
            self.type_rgb = np.stack([_tc.get(_names[t], np.array([200.0, 200.0, 200.0])) if 0 <= t < len(_names) else np.array([200.0, 200.0, 200.0]) for t in _nt])
            # TYPES THAT CHANGE DURING THE RUN (exp06 phase C, 2026-11-14: `cell_fate` switches a cell's type from its signal): the recorder keeps the type column per row in
            # `cell__node_type_t` [rows, n] when it ever differed from the final one; then each drawn row takes its own column (draw_cells sets type_rgb from it), else the final column for every row
            self.type_rgb_t = None
            if "cell__node_type_t" in z.files:
                _ntt = np.rint(z["cell__node_type_t"]).astype(np.int64)
                _lut = np.stack([_tc.get(nm, np.array([200.0, 200.0, 200.0])) for nm in _names] + [np.array([200.0, 200.0, 200.0])])
                self.type_rgb_t = _lut[np.where((_ntt >= 0) & (_ntt < len(_names)), _ntt, len(_names))]
            # `plotting.type_opacity: {inner: 0.1}` (Cedric, 2026-11-01: "a transparency option on top of the cell colour"): the opacity (1 = solid, 0 = invisible) of the cells of a type, on top of the type's colour
            _to = {str(k): float(v) for k, v in (_pl.get("type_opacity") or {}).items()}
            _to.update(TYPE_OPACITY)                                 # `--type-opacity inner=1`: over the spec's own
            self.type_alpha = np.array([_to.get(_names[t], 1.0) if 0 <= t < len(_names) else 1.0 for t in _nt], float)
        # per-cell types (exp35, 2026-10-31: "plot only the outer layer"): `draw_types`, a set of type indices, limits draw_cells to those cells; None = every cell
        nt = z["cell__node_type"] if "cell__node_type" in z.files else None
        self.node_type = None if nt is None else (np.rint(nt[-1] if nt.ndim == 2 else nt).astype(np.int64))
        self.node_type_t = np.rint(z["cell__node_type_t"]).astype(np.int64) if "cell__node_type_t" in z.files else None    # [rows, n] when types changed in the run
        self.type_names = [str(x) for x in z["cell__type_names"]] if "cell__type_names" in z.files else []
        self.draw_types = None
        self.v = point_volume(spec)
        um = float((spec["general"].get("units") or {}).get("length_um", 0) or 0)
        self.um = um
        self.vol_scale = self.v * (um ** 3 if um > 0 else 1.0)
        self.vol_unit = "µm³" if um > 0 else "model units³"
        self.fake = fake
        self.t, self.t_unit = time_axis(spec, self.rows)
        self._X = None

    def positions(self, rows):
        if self._X is None:
            self._X = self.z["pt__pos"]
        return self._X[rows].astype(np.float64)

    def activator(self, t):
        """[ncell] activator of row t (zeros where the run has no chemistry); `--fake-act radial` fabricates one for a test."""
        if self.fake == "radial":
            c = self.centroids(t)
            r = np.linalg.norm(c - np.nanmedian(c[self.live[t]], axis=0), axis=1)
            R = max(float(np.nanpercentile(r[self.live[t]], 98)), 1e-9)
            return np.clip(r / R, 0, 1) * (0.35 + 0.65 * t / max(self.rows - 1, 1))
        if self.chem is None:
            return np.zeros(self.live.shape[1])
        return self.chem[t]

    def centroids(self, t):
        if self.cpos is not None:
            return self.cpos[t]
        X = self.positions(t)
        m = self.occ[t]
        n = self.live.shape[1]
        s = np.stack([np.bincount(self.pid[t][m], weights=X[m, k], minlength=n)[:n] for k in range(3)], 1)
        return s / np.maximum(self.npts[t], 1)[:, None]

    def series(self):
        a_rows = [self.activator(t)[self.live[t]] for t in range(self.rows)]
        vol = [self.npts[t][self.live[t]] * self.vol_scale for t in range(self.rows)]
        return a_rows, vol, self.live[:self.rows].sum(1)


class PointRun:
    """A run of POINT cells (exp05, 2026-11-14): one centre per cell (`cell__pos`), no material points. Each live cell is drawn as a sphere of
    the spec's `plotting.dot_radius` (one value, or one per type name; 0.5 length units when unset), the body the model's soft core gives it.
    Its volume in the inset is its `size` block (1 at birth, 2 at division, cell_divide[point_adhesion]) times the volume of that sphere,
    (4/3) pi r^3, in um^3 with the spec's `units.length_um`: the drawn sphere does not grow, the size is the model's growth variable."""

    def __init__(self, d, spec, fake=None):
        z = np.load(os.path.join(d, "trajectory.npz"))
        self.spec, self.z, self.fake = spec, z, fake
        self.occ = z["cell__occ"] > 0.5                              # [rows, n]
        self.rows = self.occ.shape[0]
        self.live = self.occ
        self.cpos = z["cell__pos"].astype(np.float64)
        self.chem = z["cell__chem"][..., 0].astype(np.float64) if "cell__chem" in z.files else None
        self.size = z["cell__size"][..., 0].astype(np.float64) if "cell__size" in z.files else np.ones(self.occ.shape)
        self.type_names = [str(x) for x in z["cell__type_names"]] if "cell__type_names" in z.files else []
        nt = z["cell__node_type"] if "cell__node_type" in z.files else np.zeros(self.occ.shape[1])
        self.node_type = np.rint(nt[-1] if nt.ndim == 2 else nt).astype(np.int64)
        self.node_type_t = np.rint(z["cell__node_type_t"]).astype(np.int64) if "cell__node_type_t" in z.files else None
        pl_ = (spec or {}).get("plotting") or {}
        dr = pl_.get("dot_radius", 0.5)
        rt = np.array([float(dr.get(nm, 0.5)) if isinstance(dr, dict) else float(dr) for nm in self.type_names] or [0.5 if isinstance(dr, dict) else float(dr)])
        self.radius = rt[np.clip(self.node_type, 0, len(rt) - 1)]   # per cell slot, from its final type
        self.um = float((spec["general"].get("units") or {}).get("length_um", 0) or 0)
        # FOR `--dots` (the 4D view's look, which `_draw_dots` draws from a point's owner `pid` and the spacing v^(1/3)): every cell owns itself, and the
        # spacing is the median centre distance from a clustered cell to its nearest neighbour, pooled over 20 rows of the run -- one value for the
        # whole movie (the 4D view measures it frame by frame, and a frame of lone cells would draw them as dots many cells wide)
        self.pid = np.broadcast_to(np.arange(self.occ.shape[1])[None], self.occ.shape)
        from scipy.spatial import cKDTree
        nn = []
        for t in np.unique(np.linspace(0, self.rows - 1, 20).astype(int)):
            Xt = self.cpos[t][self.occ[t]]
            if len(Xt) > 1:
                d = cKDTree(Xt).query(Xt, k=2)[0][:, 1]
                nn.append(d[d < 1.5 * 2.0 * float(rt.max())])
        nn = np.concatenate(nn) if nn else np.zeros(0)
        self.v = float(np.median(nn) if len(nn) else 2.0 * float(rt.max())) ** 3
        self.vol_scale = 4.0 / 3.0 * math.pi * float(rt.max()) ** 3 * (self.um ** 3 if self.um > 0 else 1.0)
        self.vol_unit = "µm³" if self.um > 0 else "model units³"
        self.type_rgb = self.type_rgb_t = self.type_alpha = None
        self.draw_types = None
        if CELL_COLOURS.get("mode") == "type" or str(pl_.get("cell_colours", "")).lower() == "type":     # every cell in its TYPE's colour, per row when types change (cell_fate)
            import matplotlib.colors as _mc
            cols = {**{str(k): v for k, v in (pl_.get("colors") or {}).items()}, **TYPE_COLOURS}
            n_ = len(self.type_names)
            lut = np.stack([np.asarray(_mc.to_rgb(cols.get(nm, "#c8c8c8"))) * 255.0 for nm in self.type_names] + [np.array([200.0, 200.0, 200.0])])
            self.type_rgb = lut[np.where((self.node_type >= 0) & (self.node_type < n_), self.node_type, n_)]
            if self.node_type_t is not None:
                self.type_rgb_t = lut[np.where((self.node_type_t >= 0) & (self.node_type_t < n_), self.node_type_t, n_)]
        if CHEM_RGB["cols"] and "cell__chem" in z.files and z["cell__chem"].shape[-1] >= 3:      # three chemicals, one colour each: a cell's colour is their blend by share, row by row
            import matplotlib.colors as _mc
            c3 = np.clip(z["cell__chem"][..., :3].astype(np.float32), 0.0, None)
            pal = np.stack([np.asarray(_mc.to_rgb(h), np.float32) * 255.0 for h in CHEM_RGB["cols"]])
            self.type_rgb_t = (c3 / np.maximum(c3.sum(-1, keepdims=True), 1e-9)) @ pal            # [rows, n, 3]
            self.type_rgb = self.type_rgb_t[-1]
        self.t, self.t_unit = time_axis(spec, self.rows)

    def positions(self, rows):
        return self.cpos[rows]

    def activator(self, t):
        if self.chem is None:
            return np.zeros(self.live.shape[1])
        return self.chem[t]

    def centroids(self, t):
        return self.cpos[t]

    def series(self):
        a_rows = [self.activator(t)[self.live[t]] for t in range(self.rows)]
        vol = [self.size[t][self.live[t]] * self.vol_scale for t in range(self.rows)]
        return a_rows, vol, self.live[:self.rows].sum(1)


_SPHERE = {}


def _draw_point_cells(pl, R, k, a_top):
    """Row k of a point-cell run: one sphere per live cell (a template sphere scaled to the cell's radius and moved to its centre), one mesh."""
    import pyvista as pv
    cells = np.flatnonzero(R.live[k])
    if not len(cells):
        return
    if "t" not in _SPHERE:
        _SPHERE["t"] = pv.Sphere(radius=1.0, theta_resolution=28, phi_resolution=18)
    tpl = _SPHERE["t"]
    nv = tpl.n_points
    P = (R.positions(k)[cells][:, None, :] + R.radius[cells][:, None, None] * np.asarray(tpl.points)[None]).reshape(-1, 3)
    f = np.asarray(tpl.faces).reshape(-1, 4)
    F = np.repeat(f[None], len(cells), axis=0)
    F[..., 1:] += (np.arange(len(cells)) * nv)[:, None, None]
    surf = pv.PolyData(P, F.reshape(-1))
    a = R.activator(k)
    cols = np.stack([cell_rgb_R(R, c) for c in cells.tolist()]) if by_id(R) else act_rgb(a[cells], a_top)
    surf["rgb"] = np.repeat(cols.astype(np.uint8), nv, axis=0)
    pl.add_mesh(surf, scalars="rgb", rgb=True, smooth_shading=True, pbr=False, specular=0.0,
                diffuse=0.15, ambient=0.85, name="cell_all", show_scalar_bar=False, opacity=1.0)


def wedge_volumes(pos, es, et, ef, nF):
    """Per-face wedge volume v_f = (1/3) c_f . N_f about the mean vertex (numpy twin of vertex_ops.face_geometry_3d), nF faces."""
    live = ef < nF
    es, et, ef = es[live], et[live], ef[live]
    p = pos - pos.mean(0)
    cross = np.cross(p[es], p[et])
    N = np.zeros((nF, 3)); np.add.at(N, ef, 0.5 * cross)
    cen = np.zeros((nF, 3)); np.add.at(cen, ef, p[es])
    cnt = np.bincount(ef, minlength=nF).astype(np.float64)
    cen /= np.maximum(cnt, 1)[:, None]
    return np.abs((cen * N).sum(1)) / 3.0


class VertexRun:
    """A mesh run read through plexus.measures.open_traj (the new trajectory.npz and the old okuda traj.npz alike)."""

    def __init__(self, d, spec, fake=None):
        from plexus.measures import open_traj
        self.tr = open_traj(d)
        self.spec = spec
        self.rows = self.tr.n_rows()
        self.fake = fake
        ticks = getattr(self.tr, "ticks", None)
        self.t, self.t_unit = time_axis(spec, ticks if ticks else self.rows)
        self.vol_unit = "model units³"
        # micrometres per model length unit, from the spec's `general.units.length_um` (0 = undeclared: no scale bar)
        self.um = float(((spec or {}).get("general", {}).get("units", {}) or {}).get("length_um", 0.0) or 0.0) if isinstance(spec, dict) else 0.0
        self._geo = {}

    def mesh(self, t):
        es, et, ef = self.tr.half_edges(t)
        return np.asarray(self.tr.pos(t), np.float64), np.asarray(es, np.int64), np.asarray(et, np.int64), np.asarray(ef, np.int64), self.tr.nF(t)

    def activator(self, t):
        nF = self.tr.nF(t)
        if self.fake == "radial":
            pos, es, et, ef, _ = self.mesh(t)
            c = np.zeros((nF, 3)); np.add.at(c, ef[ef < nF], pos[es[ef < nF]])
            c /= np.maximum(np.bincount(ef[ef < nF], minlength=nF), 1)[:, None]
            r = np.linalg.norm(c - c.mean(0), axis=1)
            return np.clip(r / max(float(r.max()), 1e-9), 0, 1)
        s = self.tr.state("chem", t)
        if s is None:
            return np.zeros(nF)
        return np.asarray(s, np.float64)[:nF, 0]

    def live_mask(self, t):
        nF = self.tr.nF(t)
        o = self.tr.occ("cell", t)
        m = np.ones(nF, bool)
        if o is not None:
            m = np.asarray(o, bool)[:nF]
        return m

    def centroids(self, t):
        """[nF, 3] the centroid of every face (cell) of row t: the mean of its vertices."""
        pos, es, et, ef, nF = self.mesh(t)
        m = ef < nF
        cen = np.zeros((nF, 3)); np.add.at(cen, ef[m], pos[es[m]])
        return cen / np.maximum(np.bincount(ef[m], minlength=nF), 1)[:, None]

    def volumes(self, t):
        if t not in self._geo:
            pos, es, et, ef, nF = self.mesh(t)
            self._geo[t] = wedge_volumes(pos, es, et, ef, nF)
        return self._geo[t]

    def series(self):
        a_rows, vol, n = [], [], []
        for t in range(self.rows):
            m = self.live_mask(t)
            a_rows.append(self.activator(t)[m])
            vol.append(self.volumes(t)[m])
            n.append(int(m.sum()))
        return a_rows, vol, np.array(n)


# ---------------------------------------------------------------------------------------------------------- the insets
def _rect(right=0.96):
    """An inset's axes rectangle [left, bottom, width, height], in fractions of the inset's own width and height: the left and bottom margins that hold the axis names and the tick numbers grow with AXIS_FONT (10 gives the original 0.17 / 0.25)."""
    f = AXIS_FONT / 10.0
    return [round(0.17 * f, 4), round(0.25 * f, 4), round(right - 0.17 * f, 4), round(0.91 - 0.25 * f, 4)]


def _ylabel(ax, text):
    """The axis name of an inset; above the default AXIS_FONT it is broken into lines of what the axes' height holds (about half an em per character), so that a long name is not cut off at the inset's edge."""
    if AXIS_FONT > 10:
        import textwrap
        text = textwrap.fill(text, max(8, int(ax.get_position().height * TILE[1] / (0.5 * AXIS_FONT * 100 / 72))))
    ax.set_ylabel(text, color="white", fontsize=AXIS_FONT)


def _fig(size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    return plt, fig


def _finish(plt, fig, ax, colour="white"):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(colour)
    ax.tick_params(colors=colour, labelsize=AXIS_FONT - 1, length=3)
    fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :4].copy()
    plt.close(fig)
    return buf


def tile_count(t, n, k, unit, ylim_top):
    plt, fig = _fig(TILE)
    ax = fig.add_axes(_rect())
    ax.plot(t, n, color="white", lw=0.8, alpha=0.45)
    ax.plot(t[: k + 1], n[: k + 1], color="white", lw=2.0)
    ax.plot([t[k]], [n[k]], "o", color="white", ms=5)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, ylim_top)
    ax.set_xlabel(f"time ({unit})", color="white", fontsize=AXIS_FONT)
    _ylabel(ax, "live cells (number)")
    return _finish(plt, fig, ax)


def tile_density(t, H, med, a_top, k, unit, hmax, label):
    """Time x activator density: pixel colour = the activator's own colour at that height, mixed over the tile's ground by
    log10(1 + 99 f / hmax) / 2; the dark line is the median; a white cursor at the frame."""
    plt, fig = _fig(TILE)
    ax = fig.add_axes(_rect())
    nb, nt = H.shape
    ycol = act_rgb((np.arange(nb) + 0.5) / nb * a_top, a_top).astype(np.float32)          # [nb, 3]
    w = (np.log10(1.0 + 99.0 * np.clip(H / max(hmax, 1e-12), 0, 1)) / 2.0)[..., None]
    img = (1 - w) * TILE_BG + w * ycol[:, None, :]
    ax.imshow(np.clip(img, 0, 255).astype(np.uint8), origin="lower", aspect="auto", interpolation="nearest",
              extent=[t[0], t[-1], 0, a_top])
    ax.plot(t, med, color="white" if (WHITE_INK["on"] or WHITE_PLOT["on"]) else "#2b333b", lw=1.6)
    ax.axvline(t[k], color="white", lw=1.2)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, a_top)
    ax.set_xlabel(f"time ({unit})", color="white", fontsize=AXIS_FONT)
    _ylabel(ax, label)
    return _finish(plt, fig, ax)


def tile_height(t, h, cols, labs, k, unit, ylim_top, h_unit):
    """One line per family, in its founder cell's own colour: the whole run faint, the run so far thick, a dot at the frame; the family's label (its block value) at the right end."""
    plt, fig = _fig(TILE)
    ax = fig.add_axes(_rect(0.91))
    for f in range(h.shape[1]):
        c = cols[f]
        ax.plot(t, h[:, f], color=c, lw=0.8, alpha=0.35)
        ax.plot(t[: k + 1], h[: k + 1, f], color=c, lw=1.8)
        if np.isfinite(h[k, f]):
            ax.plot([t[k]], [h[k, f]], "o", color=c, ms=4)
    if labs is not None:                                         # the end labels, pushed apart by at least 5 % of the axis so that no two overlap
        order = [f for f in np.argsort(h[-1]) if np.isfinite(h[-1, f])]
        ys = [float(h[-1, f]) for f in order]
        gap = 0.05 * ylim_top
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + gap)
        for f, y in zip(order, ys):
            ax.text(t[-1], y, " " + labs[f], color="white", fontsize=7, va="center")
    ax.axvline(t[k], color="white", lw=0.6, alpha=0.5)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, ylim_top)
    ax.set_xlabel(f"time ({unit})", color="white", fontsize=AXIS_FONT)
    _ylabel(ax, f"mean cell height ({h_unit})")
    return _finish(plt, fig, ax)


def tile_junction(t, S, cols, labs, k, unit, ylim_top, ylabel="bound density of the junctions"):
    """One line per array, in its first cell's colour: the whole run faint, the run so far thick, a dot at the frame; the array's label (its adhesion maximum) at the right end, pushed apart so that none overlap."""
    plt, fig = _fig(TILE)
    ax = fig.add_axes(_rect(0.91))
    for f in range(S.shape[1]):
        c = cols[f]
        ax.plot(t, S[:, f], color=c, lw=0.8, alpha=0.35)
        ax.plot(t[: k + 1], S[: k + 1, f], color=c, lw=1.8)
        ax.plot([t[k]], [S[k, f]], "o", color=c, ms=4)
    if labs is not None:
        order = list(np.argsort(S[-1]))
        ys = [float(S[-1, f]) for f in order]
        gap = 0.05 * ylim_top
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + gap)
        for f, y in zip(order, ys):
            if labs[f]:
                ax.text(t[-1], y, " " + labs[f], color="white", fontsize=7, va="center")
    ax.axvline(t[k], color="white", lw=0.6, alpha=0.5)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, ylim_top)
    ax.set_xlabel(f"time ({unit})", color="white", fontsize=AXIS_FONT)
    _ylabel(ax, ylabel)
    return _finish(plt, fig, ax)


def tile_count_by_type(t, series, k, unit, ylim_top):
    """Live cells against time, one curve per cell type in its own colour (`plotting.count_panel: by_type`, Cedric, 2026-11-02): `series` is [(colour, counts per row)], a cursor at the frame."""
    plt, fig = _fig(TILE)
    ax = fig.add_axes(_rect())
    for col, n in series:
        ax.plot(t, n, color=col, lw=0.8, alpha=0.35)
        ax.plot(t[: k + 1], n[: k + 1], color=col, lw=1.8)
        ax.plot([t[k]], [n[k]], "o", color=col, ms=4)
    ax.axvline(t[k], color="white", lw=0.6, alpha=0.5)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, ylim_top)
    ax.set_xlabel(f"time ({unit})", color="white", fontsize=AXIS_FONT)
    _ylabel(ax, "live cells (number)")
    return _finish(plt, fig, ax)


def tile_volume(t, mean, p10, p90, k, unit, ylim_top, vol_unit):
    plt, fig = _fig(TILE)
    ax = fig.add_axes(_rect())
    ax.fill_between(t, p10, p90, color="white", alpha=0.22, lw=0)
    ax.plot(t, mean, color="white", lw=0.8, alpha=0.45)
    ax.plot(t[: k + 1], mean[: k + 1], color="white", lw=2.0)
    ax.plot([t[k]], [mean[k]], "o", color="white", ms=5)
    ax.axvline(t[k], color="white", lw=0.6, alpha=0.5)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, ylim_top)
    ax.set_xlabel(f"time ({unit})", color="white", fontsize=AXIS_FONT)
    _ylabel(ax, f"cell volume ({vol_unit})")
    return _finish(plt, fig, ax)


def legend(frame, a_top, label, x0=28, y0=None):
    """The colour bar under the cells: white at 0 to red at a_top, the two numbers and what the colour is OF."""
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    im = Image.fromarray(frame); dr = ImageDraw.Draw(im)
    font = ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"), 18)
    H = frame.shape[0]
    y0 = H - 92 if y0 is None else y0
    W = 260
    bar = act_rgb(np.linspace(0, a_top, W), a_top)
    im.paste(Image.fromarray(np.repeat(bar[None], 14, axis=0)), (x0, y0 + 26))
    dr.rectangle([x0 - 1, y0 + 25, x0 + W, y0 + 40], outline=(255, 255, 255))
    dr.text((x0, y0), label, fill=(255, 255, 255), font=font)
    dr.text((x0, y0 + 44), "0", fill=(255, 255, 255), font=font)
    top = f"{a_top:.3g}"
    dr.text((x0 + W - dr.textlength(top, font=font), y0 + 44), top, fill=(255, 255, 255), font=font)
    return np.array(im)


# ---------------------------------------------------------------------------------------------------------- the camera
def camera_for(pl, pts, view_angle, size, view, up, dist_extra=0.0):
    """Aim at the middle of `pts`, back off until the points take 84 % of the window's height or 90 % of its width."""
    w, h = size
    side = np.cross(up, view)
    c = pts.mean(0)
    p = pts - c
    su, ss, sv = p @ up, p @ side, p @ view
    mid = c + side * 0.5 * (ss.max() + ss.min()) + up * 0.5 * (su.max() + su.min())
    hu, hs = 0.5 * (su.max() - su.min()), 0.5 * (ss.max() - ss.min())
    tn = math.tan(math.radians(view_angle / 2))
    dist = max(hu / (0.84 * tn), hs / (0.90 * tn * w / h)) + float(sv.max() - 0.5 * (sv.max() + sv.min()))
    pl.camera.position = tuple(mid + dist * view)
    pl.camera.focal_point = tuple(mid)
    pl.camera.up = tuple(up)
    pl.camera.view_angle = view_angle
    return dist, mid


def view_vectors(elev, azim):
    el, az = math.radians(elev), math.radians(azim)
    view = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    up = np.array([0.0, 0.0, 1.0]) - view * view[2]
    if np.linalg.norm(up) < 1e-6:            # looking straight down (elevation 90): `up` degenerated to 0, so camera_for saw an empty picture and every caller hand-set a zoom; the in-plane up is the direction away from the azimuth
        return view, np.array([-math.cos(az), -math.sin(az), 0.0])
    return view, up / max(np.linalg.norm(up), 1e-9)


def _add_lights(pl, c0, R0, view, up):
    import pyvista as pv
    side = np.cross(up, view)
    pl.remove_all_lights()
    for pos, inten in ((c0 + 6 * R0 * (view + 0.55 * up - 0.45 * side), 0.9),
                       (c0 + 6 * R0 * (view - 0.3 * up + 0.6 * side), 0.5),
                       (c0 + 6 * R0 * view, 0.45)):
        L = pv.Light(position=tuple(pos), focal_point=tuple(c0), intensity=inten)
        L.positional = False
        pl.add_light(L)


# ---------------------------------------------------------------------------------------------------------- the render
CUT_PRISM_FACTOR = 1.7          # the drawn height of a cut vertex cell in mean edge lengths (a drawing: the vertex model has no height)


def _draw_vertex_cut(pl, pos, es, et, ef, nF, cen, lm, a, a_top, c0, view, sep=None):
    """The vertex twin CUT OPEN like the MPM twin, as a cross-section: a mesh cell is a polygon with no height, so each cell on the far side of the
    cutting plane (its centroid behind the plane through c0) is drawn as a PRISM -- the polygon carried radially inward by its own width (the
    MPM twin's cells are cubes as high as they are wide) -- with the inner face in the cell's colour (the far wall seen from inside) and its
    side walls darker; the cells at the plane show their side walls as the cut ring. The height is a drawing, the vertex model has none."""
    import pyvista as pv
    ctr = cen[lm].mean(0)
    far = lm & (((cen - c0) @ view) <= 0.0)
    hf = far[ef]
    s, t_, f = es[hf], et[hf], ef[hf]
    if len(f) == 0:
        return
    Ps, Pt = pos[s], pos[t_]
    edge = np.linalg.norm(Pt - Ps, axis=1)
    mean_edge = np.bincount(f, weights=edge, minlength=nF) / np.maximum(np.bincount(f, minlength=nF), 1)
    h = (CUT_PRISM_FACTOR * mean_edge[f])[:, None]                       # 1.7: a regular hexagon is 1.73 edges wide (a cell as high as wide);
                                                                         # exp32 draws the vertex twin at a quarter of that (Cedric, 2026-10-25:
                                                                         # "the vertex reference cells too thick at start, divide by four")
    unit = lambda v: v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    if sep is not None:
        # AN APICO-BASAL RUN HAS ITS HEIGHT: the apical vertex is pos + sep, the basal one pos - sep (`seed_mesh[apicobasal]`, apical side out), so
        # the prism is the model's own, not a drawing factor (exp33, 2026-10-25: "I do not want to hack the drawing")
        Ps, Pt = Ps + sep[s], Pt + sep[t_]
        Qs, Qt = Ps - 2.0 * sep[s], Pt - 2.0 * sep[t_]
        cnt = np.maximum(np.bincount(f, minlength=nF), 1)[:, None]
        Qc = (np.stack([np.bincount(f, weights=Qs[:, i], minlength=nF) for i in range(3)], 1) / cnt)[f]
    else:
        Qs, Qt = Ps - h * unit(Ps - ctr), Pt - h * unit(Pt - ctr)
        Qc = cen[f] - h * unit(cen[f] - ctr)
    g = 0.92
    n = len(f)
    pts = np.concatenate([Qc, Qc + g * (Qs - Qc), Qc + g * (Qt - Qc)])
    idx = np.arange(n)
    faces = np.hstack([np.full((n, 1), 3), np.stack([idx, n + idx, 2 * n + idx], 1)]).ravel()
    inner = pv.PolyData(pts, faces)
    col = act_rgb(a[f], a_top)
    inner.cell_data["rgb"] = col
    pl.add_mesh(inner, scalars="rgb", rgb=True, preference="cell", smooth_shading=False, specular=0.0, diffuse=0.15, ambient=0.85,
                name="cell_all", show_scalar_bar=False, opacity=1.0)
    wp = np.concatenate([Ps, Pt, Qt, Ps, Qt, Qs])
    wf = np.hstack([np.full((2 * n, 1), 3), np.stack([np.arange(2 * n) + 0, np.arange(2 * n) + 0, np.arange(2 * n) + 0], 1)]).ravel()
    tri = np.arange(2 * n)
    wf = np.hstack([np.full((2 * n, 1), 3), np.stack([np.r_[idx, 3 * n + idx], np.r_[n + idx, 4 * n + idx], np.r_[2 * n + idx, 5 * n + idx]], 1)]).ravel()
    wall = pv.PolyData(wp, wf)
    wall.cell_data["rgb"] = (np.tile(col, (2, 1)).astype(np.float64) * 0.55).astype(np.uint8)
    pl.add_mesh(wall, scalars="rgb", rgb=True, preference="cell", smooth_shading=False, specular=0.0, diffuse=0.15, ambient=0.85,
                name="wall_all", show_scalar_bar=False, opacity=1.0)


SECTION_ALPHA = {}                                   # `--section-alpha outer=0.4`: the second panel of `--section` is the SAME whole tissue with these cell types at these opacities, not a cross section (Cedric, 2026-11-01)


def _draw_dots(pl, R, k, X, live, a, a_top, cells, cut, c0, view):
    """THE 4D VIEW'S DOTS (bio_view._dots_update, Cedric 2026-11-14): every live material point of the drawn cells a flat round dot -- a sphere sprite, unlit, so it is a disc
    of one colour that still writes a sphere's depth for the ink -- of world radius r = max(half the point spacing, 1.8 px) x `--dots`, coloured by its cell (its own colour,
    its type's, or its activity through the activator scale), the farthest dots mixed 30 % toward the ground (bio_view's 'vivid' depth cue)."""
    import pyvista as pv
    sel = live & np.isin(R.pid[k], cells)
    P = X[sel]
    owner = np.rint(R.pid[k][sel]).astype(np.int64)
    if cut:
        dv = (P - np.asarray(c0)) @ np.asarray(view)
        keep = (np.abs(dv) <= float(SECTION_SLAB["h"])) if SECTION_SLAB["h"] else (dv <= 0.0)       # a thin slice through the centre, or the far half
        nt = getattr(R, "node_type", None)
        if CUT_KEEP and nt is not None:                     # `--cut-keep TYPE` as the surfaces have it (2026-11-15, deck: the dots twin of step 0133): a cell of these types keeps all its points, unless every one lies on the camera's side
            kt = [R.type_names.index(nm) for nm in CUT_KEEP if nm in R.type_names]
            ok = owner < len(nt)
            for c in np.unique(owner[ok][np.isin(nt[owner[ok]], kt)]).tolist():
                m_ = owner == c
                keep[m_] = bool((dv[m_] < 0.0).any())
        P, owner = P[keep], owner[keep]
    if len(P) == 0:
        return
    uc = np.unique(owner)
    tab = np.zeros((int(uc.max()) + 1, 3), np.float64)
    for c in uc.tolist():
        tab[c] = cell_rgb_R(R, c) if by_id(R) else act_rgb([a[c]], a_top)[0]
    if MARK_COLOURS and getattr(R, "node_type", None) is not None:          # `--mark-colours`: the cells of these types in their own colour, over the activity colours
        import matplotlib.colors as _mcm
        ntt = getattr(R, "node_type_t", None)
        ntk = ntt[min(int(k), len(ntt) - 1)] if ntt is not None else R.node_type
        for tn, hx in MARK_COLOURS.items():
            if tn in R.type_names:
                sel_c = uc[ntk[uc] == R.type_names.index(tn)]
                tab[sel_c] = np.asarray(_mcm.to_rgb(hx)) * 255.0
    rgb = tab[owner]
    cam = pl.camera
    pos, foc = np.asarray(cam.position, float), np.asarray(cam.focal_point, float)
    n = (pos - foc) / max(float(np.linalg.norm(pos - foc)), 1e-12)
    d = (P - foc) @ n
    span = float(d.max() - d.min()) or 1.0
    bg = np.array([int(GROUND[i:i + 2], 16) for i in (1, 3, 5)], np.float64)
    w = 0.3 * ((d.max() - d) / span)[:, None]
    rgb = rgb * (1.0 - w) + bg[None, :] * w
    wpp = 2.0 * float(np.linalg.norm(pos - foc)) * math.tan(math.radians(float(cam.view_angle) / 2.0)) / float(pl.window_size[1])
    r = max(0.5 * float(R.v) ** (1.0 / 3.0), 1.8 * wpp) * float(DOTS["factor"])
    DOTS["r"] = r
    cloud = pv.PolyData(np.ascontiguousarray(P, np.float32))
    cloud["rgb"] = np.clip(rgb, 0, 255).astype(np.uint8)
    pl.add_mesh(cloud, name="cell_dots", scalars="rgb", rgb=True, style="points", point_size=2.0 * r / wpp, render_points_as_spheres=True, lighting=False,
                ambient=1.0, diffuse=0.0, specular=0.0, show_scalar_bar=False, reset_camera=False)


def _dots_ink():
    """The extra arguments of the ink for a frame of dots (bio_view's: the dot radius, the ground as the colour origin), else none."""
    if not DOTS["factor"] or not DOTS["r"]:
        return {}
    return {"dot_r": float(DOTS["r"]), "colour_origin": tuple(float(int(GROUND[i:i + 2], 16)) for i in (1, 3, 5))}


def draw_cells(pl, kind, R, k, a_top, cut, c0, R0, view, h_vox, alpha_over=None):
    """Row k of a run into the plotter: every cell its own surface (an MPM cell: the iso-surface of its points; a vertex cell: its polygons, shrunk,
    over an opaque wall), white at activator 0 to red at a_top. Replaces the previous row's actors."""
    import pyvista as pv
    import exp23_studio as e23
    if getattr(R, "type_rgb_t", None) is not None:                 # types that change in the run: this row's colours
        R.type_rgb = R.type_rgb_t[min(int(k), len(R.type_rgb_t) - 1)]
    for nm in list(pl.actors):
        if nm.startswith("cell_") or nm.startswith("wall"):
            pl.remove_actor(nm)
    if kind == "point":
        if DOTS["factor"]:                                          # the 4D view's dots: one dot per point cell
            Xk = R.positions(k)
            ck = Xk[R.occ[k]].mean(0) if (cut and R.occ[k].any()) else c0     # a cross-section is cut through the tissue's centre at THIS row (the tissue drifts and grows)
            _draw_dots(pl, R, k, Xk, R.occ[k], R.activator(k), a_top, np.flatnonzero(R.live[k]), cut, ck, view)
            return
        _draw_point_cells(pl, R, k, a_top)
        return
    if kind == "mpm":
        X = R.positions(k)
        live = R.occ[k]
        a = R.activator(k)
        cells = np.flatnonzero(R.live[k])
        if getattr(R, "draw_types", None) is not None and getattr(R, "node_type", None) is not None:
            cells = cells[np.isin(R.node_type[cells], list(R.draw_types))]
        if DOTS["factor"]:
            _draw_dots(pl, R, k, X, live, a, a_top, cells, cut, c0, view)
            return
        groups = {}
        # `plotting.contour_bodies: {nucleus: nuc, nucleus_color: ...}` (live_movie's keys): the points with block `nuc` > 0.5 of each cell are contoured on their own, as an opaque body inside the cell's glass (set_material[nucleus] marks them)
        _cb = ((R.spec or {}).get("plotting") or {}).get("contour_bodies") or {}
        if not hasattr(R, "_nuc"):                                   # read once: R.z is a compressed npz, every access decompresses
            R._nuc = R.z[f"pt__{_cb['nucleus']}"][..., 0] if _cb.get("nucleus") and f"pt__{_cb['nucleus']}" in R.z.files else None
        nuc_blk = R._nuc
        nuc_rgb = np.rint(np.asarray(__import__("matplotlib.colors").colors.to_rgb(str(_cb.get("nucleus_color", "#1b1464")))) * 255.0).astype(np.uint8)   # BYTES, as cell_rgb: floats are read as 0-1 and drew the nucleus pale
        iso = float(((R.spec or {}).get("plotting") or {}).get("surface_iso_frac", 0.5))
        nuclei = []
        for c in cells.tolist():
            sel = live & (R.pid[k] == c)
            P = X[sel]
            if len(P) < 16:
                continue
            surf = e23._cell_surface(P, R.v, h_vox, iso=float(((R.spec or {}).get("plotting") or {}).get("surface_iso_frac", 0.5)))   # the run's own contour level (live_movie's key); 0.5 unless said
            if surf is None:
                continue
            if cut and CUT_KEEP and getattr(R, "node_type", None) is not None and int(c) < len(R.node_type) and 0 <= int(R.node_type[c]) < len(R.type_names) and R.type_names[int(R.node_type[c])] in CUT_KEEP:
                if float(((surf.points - np.asarray(c0)) @ np.asarray(view)).min()) >= 0.0:      # wholly on the camera's side: the section has removed it
                    continue
            elif cut:                                        # THE SECTION: every surface clipped at the plane through c0, the half on the camera's side removed. A whole cell was hidden when its centroid was on that
                surf = surf.clip(normal=view, origin=c0)     # side, so the cell at the very centre (a lone blue cell in a sphere) vanished for the frames its centroid drifted across the plane (movie 23, 2026-11-01)
                if surf.n_points == 0:
                    continue
            col = cell_rgb_R(R, c) if by_id(R) else act_rgb([a[c]], a_top)[0]
            surf["rgb"] = np.tile(col, (surf.n_points, 1))
            alpha = float(R.type_alpha[c]) if getattr(R, "type_alpha", None) is not None and int(c) < len(R.type_alpha) else 1.0
            if alpha_over and getattr(R, "node_type", None) is not None and int(c) < len(R.node_type) and 0 <= int(R.node_type[c]) < len(R.type_names):
                alpha = alpha_over.get(R.type_names[int(R.node_type[c])], alpha)          # this panel's own opacity for the cell's type
            groups.setdefault(alpha, []).append(surf)
            if nuc_blk is not None:
                Pn = X[sel & (nuc_blk[k] > 0.5)]
                ns = e23._cell_surface(Pn, R.v, h_vox, iso=iso) if len(Pn) >= 16 else None
                if ns is not None and cut:
                    ns = ns.clip(normal=view, origin=c0)
                if ns is not None and ns.n_points:
                    ns["rgb"] = np.tile(nuc_rgb, (ns.n_points, 1)); nuclei.append(ns)
        for alpha in sorted(groups, reverse=True):                   # the opaque cells first, then each translucent group over them
            batch = groups[alpha]
            merged = batch[0] if len(batch) == 1 else pv.merge(batch, merge_points=False)
            pl.add_mesh(merged, scalars="rgb", rgb=True, smooth_shading=True, pbr=False, specular=0.0,
                        diffuse=0.15, ambient=0.85, name="cell_all" if alpha >= 1.0 else f"cell_all_{alpha:g}", show_scalar_bar=False, opacity=alpha)
        if nuclei:
            pl.add_mesh(nuclei[0] if len(nuclei) == 1 else pv.merge(nuclei, merge_points=False), scalars="rgb", rgb=True, smooth_shading=True, pbr=False, specular=0.0,
                        diffuse=0.15, ambient=0.85, name="cell_nuclei", show_scalar_bar=False, opacity=1.0)
        ls_ = label_spec(R)
        if ls_ is not None:
            draw_family_labels(pl, R, k, X, live, ls_)
    else:
        pos, es, et, ef, nF = R.mesh(k)
        m = ef < nF
        es, et, ef = es[m], et[m], ef[m]
        a = R.activator(k)
        lm = R.live_mask(k)
        cen = np.zeros((nF, 3)); np.add.at(cen, ef, pos[es])
        cen /= np.maximum(np.bincount(ef, minlength=nF), 1)[:, None]
        sep = R.tr.vertex_block("sep", k) if hasattr(R, "tr") else None
        sep = None if sep is None else np.asarray(sep, np.float64)[: len(pos)]
        keep = lm[ef]
        if cut:
            _draw_vertex_cut(pl, pos, es, et, ef, nF, cen, lm, a, a_top, c0, view, sep=sep)
            return
        if sep is not None:                                   # the full view of an apico-basal run shows its APICAL surface (pos + sep)
            pos = pos + sep
            cen = np.zeros((nF, 3)); np.add.at(cen, ef, pos[es])
            cen /= np.maximum(np.bincount(ef, minlength=nF), 1)[:, None]
        # each cell's polygons fanned round its centroid and SHRUNK toward it (0.90, 0.94 above 3000 cells): the gap
        # between two cells shows the ground behind it, which is a depth jump, so the ink draws every cell's wall
        g = 0.90 if int(lm.sum()) <= 3000 else 0.94
        c_ = cen[ef[keep]]
        n_ = len(c_)
        pts = np.concatenate([c_, c_ + g * (pos[es[keep]] - c_), c_ + g * (pos[et[keep]] - c_)])
        idx = np.arange(n_)
        faces = np.hstack([np.full((n_, 1), 3), np.stack([idx, n_ + idx, 2 * n_ + idx], 1)]).ravel()
        surf = pv.PolyData(pts, faces)
        surf.cell_data["rgb"] = act_rgb(a[ef[keep]], a_top)
        pl.add_mesh(surf, scalars="rgb", rgb=True, preference="cell", smooth_shading=False, specular=0.0,
                    diffuse=0.15, ambient=0.85, name="cell_all", show_scalar_bar=False, opacity=1.0)
        # THE WALL UNDER THE GAPS (Cedric, 2026-10-21: "we see through the empty space between the polygons"): the shrunk polygons
        # leave a gap, and a closed sheet seen through its gaps shows the far side of the vesicle. Under every cell goes the
        # FULL-SIZE polygon, in the cell's colour darkened to 45 %, pushed 1.5 % of the tissue radius toward the tissue's centre:
        # the gap is now a dark opaque wall, nothing behind it shows, and the ink draws the step between wall and cell.
        ctr = cen[lm].mean(0)
        full = np.concatenate([c_, pos[es[keep]], pos[et[keep]]])
        d_ = full - ctr
        # BEHIND the cell as the camera sees it: toward the centre when the outside is seen, AWAY from it in the cut view (the far half is seen
        # from inside, and a wall pushed inward would stand in front of the cell and hide its colour)
        full = full + (0.015 if cut else -0.015) * R0 * d_ / np.maximum(np.linalg.norm(d_, axis=1, keepdims=True), 1e-9)
        wall = pv.PolyData(full, faces)
        wall.cell_data["rgb"] = (act_rgb(a[ef[keep]], a_top).astype(np.float64) * 0.45).astype(np.uint8)
        pl.add_mesh(wall, scalars="rgb", rgb=True, preference="cell", smooth_shading=False, specular=0.0,
                    diffuse=0.15, ambient=0.85, name="wall_all", show_scalar_bar=False, opacity=1.0)


def render(run, out_mp4, out_png, seconds=10.0, stride=None, cam=None, cut=False, fake=None, npz=None, max_frames=120,
           width3d=960, height=1088, rows_max=None, no_activator=False, fps=None, section=False, until=None, frame_stride=None):
    import pyvista as pv
    import imageio.v2 as iio
    import exp23_studio as e23
    from plexus.gui import flat_ink
    e23._LOOK.clear(); e23._LOOK.update(e23.LOOK["ink"])
    e23._LOOK["text"] = "white"; e23._LOOK["border"] = (255, 255, 255)
    d, kind = resolve(run)
    spec, spec_path = read_spec(d)
    R = MPMRun(d, spec, fake=fake, npz=npz) if kind == "mpm" else (PointRun(d, spec, fake=fake) if kind == "point" else VertexRun(d, spec, fake=fake))
    if frame_stride:                                         # `--frame-stride N`: a partial trajectory.npz (general.save_every_min) holds the first rows of a longer run, N frames apart: time axis = row x N frames, not the whole run stretched over the rows
        R.t = (np.arange(R.rows) * float(frame_stride) + 1.0) * (float((spec.get('general') or {}).get('dt', 1.0)) * _time_unit(spec)[1] if R.t_unit != 'frames' else 1.0)      # the same law as time_axis (row x stride + 1 frames, times dt) with the stride of the full run
    if until is not None:                                    # `--until-frame N`: the run as if it ended at its last stored row of frame N or earlier: the insets' time axes and scales and the camera framing come from those rows only
        n_until = int((R.t <= float(until) * (float((spec.get("general") or {}).get("dt", 1.0)) * _time_unit(spec)[1] if R.t_unit != "frames" else 1.0)).sum())
        R.rows, R.t = n_until, R.t[:n_until]
    # THE WHOLE RUN FIRST: every scale of every inset and of the colouring, before a frame is drawn
    a_rows, vol_rows, n_live = R.series()
    a_top = top_of_scale(a_rows)
    no_chem = a_top <= 1e-9                                 # a run with no activator: every cell white, a scale of 1
    if no_chem:
        a_top = 1.0
    H, med = density_image(a_rows, a_top)
    hmax = float(H.max())
    vmean = np.array([float(np.mean(v)) if len(v) else 0.0 for v in vol_rows])
    p10 = np.array([float(np.percentile(v, 10)) if len(v) else 0.0 for v in vol_rows])
    p90 = np.array([float(np.percentile(v, 90)) if len(v) else 0.0 for v in vol_rows])
    n_top = 1.1 * max(4, int(n_live.max()))
    v_top = 1.05 * max(float(p90.max()), 1e-12)
    t = R.t
    rows = R.rows
    hgt = None
    if kind == "mpm" and height_on(R):
        hf, fids = family_height(R)
        hs = hf * (R.um if R.um > 0 else 1.0)
        fcols = [tuple(cell_rgb_R(R, c) / 255.0) if by_id(R) else "white" for c in fids]
        ls = label_spec(R)
        flabs = [f"{v:g}" for v in R.z[f"cell__{ls[0]}"][0][R.z["cell__occ"][0] > 0.5][:, 0]] if ls else None
        hgt = (hs, fcols, flabs, 1.05 * float(np.nanmax(hs)), "µm" if R.um > 0 else "model units")
    jun = None
    if kind == "mpm" and junction_on(R):
        Sj, jcols, jlabs = junction_series(R)
        jun = (Sj, jcols, jlabs, 1.05 * max(float(Sj.max()), 1e-9))
    pul = None
    if kind == "mpm" and pull_on(R):
        Fp, pcols, plabs, ppeak = pull_series(R)
        pul = (Fp, pcols, plabs, 1.05 * max(float(Fp.max()), 1e-9))
        print("PULL PEAKS (force of the pull springs per array, peak over time):", [round(float(v), 2) for v in ppeak], flush=True)
    ks = list(range(rows))
    if stride is None:
        stride = max(1, math.ceil(rows / max_frames))
    ks = sorted(set(range(0, rows, stride)) | {rows - 1})
    if rows_max is not None:                                 # `rows_max`: draw (and frame the camera on) rows 0 .. rows_max only; default every row
        ks = [k for k in ks if k <= int(rows_max)]
    scales = dict(a_top=a_top, hist_max=hmax, n_top=n_top, volume_top=v_top, rows=rows, drawn=len(ks), stride=stride,
                  cells_first=int(n_live[0]), cells_last=int(n_live[-1]), t_end=float(t[-1]), t_unit=R.t_unit,
                  vol_unit=R.vol_unit, volume_mean_first=float(vmean[0]), volume_mean_last=float(vmean[-1]),
                  kind=kind, fake=fake, no_chem=bool(no_chem))
    # THE VIEW
    pcam = (spec.get("plotting") or {}).get("camera") or {}
    el, az = cam if cam else (float(pcam.get("elev", 20.0)), float(pcam.get("azim", 90.0)))
    view, up = view_vectors(el, az)
    # points that frame the scene: every drawn row
    if kind in ("mpm", "point"):
        allp = np.concatenate([R.positions(k)[R.occ[k]][:: max(1, int(R.occ[k].sum()) // 3000)] for k in ks])
    else:
        allp = np.concatenate([R.mesh(k)[0][:: max(1, R.tr.nV(k) // 3000)] for k in ks])
    c0 = np.median(allp, axis=0)
    R0 = float(np.percentile(np.linalg.norm(allp - c0, axis=1), 99))
    if section:                                              # `--section` (Cedric, 2026-11-01): TWO 3-D panels, the whole tissue and, beside it, the same frame with the cells on the camera's side hidden (a cross section)
        width3d = 720
    pl = pv.Plotter(off_screen=True, window_size=(width3d, height))
    pl.set_background(GROUND)
    _add_lights(pl, c0, R0, view, up)
    va = 30.0
    dist, mid = camera_for(pl, allp, va, (width3d, height), view, up)
    pl2 = None
    if section:
        pl2 = pv.Plotter(off_screen=True, window_size=(width3d, height))
        pl2.set_background(GROUND)
        # `--section-view elev,azim`: the cross-section panel has its OWN camera, and the slice is perpendicular to it (a ring seen obliquely is cut at two
        # points by a slice across the main camera; from above the slice holds the whole ring)
        view2, up2 = view_vectors(*SECTION_VIEW["cam"]) if SECTION_VIEW["cam"] else (view, up)
        _add_lights(pl2, c0, R0, view2, up2)
        dist2, _ = camera_for(pl2, allp, va, (width3d, height), view2, up2)
    px_per_unit = height / (2.0 * dist * math.tan(math.radians(va / 2)))
    bar = None
    if R.um > 0:
        tgt = 0.2 * width3d / px_per_unit * R.um
        p10_ = 10 ** math.floor(math.log10(tgt))
        L_um = max(f * p10_ for f in (1, 2, 5) if f * p10_ <= tgt)
        bar = (L_um, L_um / R.um * px_per_unit, int(0.90 * height))
    fps = int(fps) if fps else max(1, round(len(ks) / seconds))                      # `--fps` (Cedric, 2026-10-29, 'the movie looks jerky'): 16 by default, 4 times the 4 fps of a 40-frame movie
    writer = iio.get_writer(out_mp4, fps=fps, quality=8, macro_block_size=16)
    x_side = width3d * (2 if section else 1)                 # where the insets' column starts
    W_total = x_side + (0 if NO_INSETS["on"] else SIDE_W)
    frame = None
    h_vox = 0.55 * (R.v ** (1.0 / 3.0)) if kind == "mpm" else 0.0
    for j, k in enumerate(ks):
        draw_cells(pl, kind, R, k, a_top, cut, c0, R0, view, h_vox)
        frame3d = e23._ink_frame(pl, dist, va, height, False, **_dots_ink())
        if bar is not None:
            frame3d = e23._scale_bar(frame3d, bar, "white")
        if not by_id(R):
            frame3d = legend(frame3d, a_top, ACT_LABEL["text"] + " per cell" + (" (fake, test)" if fake else " (none in this run)" if no_chem else ""), y0=height - 92)
        frame = np.empty((height, W_total, 3), np.uint8)
        frame[:] = np.array([int(GROUND[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
        frame[:, :width3d] = frame3d
        if pl2 is not None:
            draw_cells(pl2, kind, R, k, a_top, not SECTION_ALPHA, c0, R0, view2, h_vox, alpha_over=SECTION_ALPHA or None)
            frame[:, width3d:2 * width3d] = e23._ink_frame(pl2, dist2, va, height, False, **_dots_ink())
        tiles = [tile_count(t, n_live.astype(float), k, R.t_unit, n_top),
                 tile_density(t, H, med, a_top, k, R.t_unit, hmax, ACT_LABEL["text"] + " (per cell)"),
                 tile_volume(t, vmean, p10, p90, k, R.t_unit, v_top, R.vol_unit)]
        if no_activator:                                                                      # `--no-activator` (Cedric, 2026-10-28): drop the middle inset
            tiles = [tiles[0], tiles[2]]
        if hgt is not None:
            tiles.append(tile_height(t, hgt[0], hgt[1], hgt[2], k, R.t_unit, hgt[3], hgt[4]))
        if jun is not None or pul is not None:                                              # the chain insets replace the count and the volume insets
            tiles = tiles[:0] + ([tile_height(t, hgt[0], hgt[1], hgt[2], k, R.t_unit, hgt[3], hgt[4])] if hgt is not None else [])
        if jun is not None:
            tiles.append(tile_junction(t, jun[0], jun[1], jun[2], k, R.t_unit, jun[3]))
        if (COUNT_BY_TYPE or str((spec.get("plotting") or {}).get("count_panel", "")) == "by_type") and getattr(R, "node_type", None) is not None:
            ser = [(TYPE_COLOURS.get(nm) or (((spec.get("plotting") or {}).get("colors") or {}).get(nm)) or "white",
                    np.array([int((R.live[r][(R.node_type_t[min(r, len(R.node_type_t) - 1)] if getattr(R, "node_type_t", None) is not None else R.node_type) == ti]).sum()) for r in range(R.rows)], float)) for ti, nm in enumerate(R.type_names)]
            tiles.append(tile_count_by_type(t, ser, k, R.t_unit, 1.1 * max(float(n.max()) for _, n in ser)))
        if pul is not None:
            tiles.append(tile_junction(t, pul[0], pul[1], pul[2], k, R.t_unit, pul[3], ylabel="force of the pull springs (per chain)"))
        for i_, tile in enumerate([] if NO_INSETS["on"] else tiles):
            e23._blend(frame, tile, x_side + (SIDE_W - tile.shape[1]) // 2, 16 + i_ * (tile.shape[0] + 14))
        frame = e23._clock(frame, f"{int(n_live[k])} cells   t = {t[k]:.0f} {R.t_unit}", "white", corner=CLOCK_CORNER)
        writer.append_data(frame[: height // 2 * 2, : W_total // 2 * 2])
        if (j + 1) % 10 == 0:
            print(f"  frame {j + 1}/{len(ks)}", flush=True)
    writer.close()
    iio.imwrite(out_png, frame)
    pl.close()
    if pl2 is not None:
        pl2.close()
    scales["spec"] = spec_path
    return scales


# ---------------------------------------------------------------------------------------------------------- the PAIR (one movie per decision)
INK_DARK = "#101418"                                    # the MPM twin's line in the pair's insets (the vertex twin's is white)


def _scene(run, width, height, cam=None):
    """One run read, framed and lit for a panel of width x height: everything render() sets up before its frame loop."""
    import pyvista as pv
    d, kind = resolve(run)
    spec, spec_path = read_spec(d)
    R = MPMRun(d, spec) if kind == "mpm" else VertexRun(d, spec)
    a_rows, vol_rows, n_live = R.series()
    pcam = (spec.get("plotting") or {}).get("camera") or {}
    el, az = cam if cam else (float(pcam.get("elev", 20.0)), float(pcam.get("azim", 90.0)))
    view, up = view_vectors(el, az)
    ks = list(range(R.rows))
    if kind == "mpm":
        allp = np.concatenate([R.positions(k)[R.occ[k]][:: max(1, int(R.occ[k].sum()) // 3000)] for k in ks])
    else:
        allp = np.concatenate([R.mesh(k)[0][:: max(1, R.tr.nV(k) // 3000)] for k in ks])
    c0 = np.median(allp, axis=0)
    R0 = float(np.percentile(np.linalg.norm(allp - c0, axis=1), 99))
    pl = pv.Plotter(off_screen=True, window_size=(width, height))
    pl.set_background(GROUND)
    _add_lights(pl, c0, R0, view, up)
    va = 30.0
    dist, mid = camera_for(pl, allp, va, (width, height), view, up)
    q = lambda f, v: np.array([float(np.percentile(x, f)) if len(x) else 0.0 for x in v])

    def _diam(k):                                            # the vesicle's diameter: 2 x the mean distance of the live cells' centroids from their mean
        c = R.centroids(k)[R.live[k] if kind == "mpm" else R.live_mask(k)]
        return 2.0 * float(np.linalg.norm(c - c.mean(0), axis=1).mean()) if len(c) else 0.0
    diam = np.array([_diam(k) for k in ks])
    return dict(run=run, kind=kind, R=R, spec=spec, pl=pl, dist=dist, va=va, c0=c0, R0=R0, view=view, diam=diam,
                h_vox=0.55 * (R.v ** (1.0 / 3.0)) if kind == "mpm" else 0.0, a_rows=a_rows, n_live=n_live,
                a_med=q(50, a_rows), a_p10=q(10, a_rows), a_p90=q(90, a_rows),
                v_mean=np.array([float(np.mean(x)) if len(x) else 0.0 for x in vol_rows]), v_p10=q(10, vol_rows), v_p90=q(90, vol_rows),
                n_frames=int((spec.get("general") or {}).get("n_frames") or R.rows))


def tile_pair(x, series, kx, ylabel, ytop, ylog=False, size=TILE, ybot=0.0):
    """One inset of the pair: for each run (label, colour, y, lo, hi) a band lo..hi, the thin whole curve and the thick curve up to the cursor kx."""
    plt, fig = _fig(size)
    ax = fig.add_axes([0.19, 0.25, 0.77, 0.66])
    for lab, col, xs, y, lo, hi, kc in series:
        if lo is not None:
            ax.fill_between(xs, lo, hi, color=col, alpha=0.20, lw=0)
        ax.plot(xs, y, color=col, lw=0.8, alpha=0.45)
        ax.plot(xs[: kc + 1], y[: kc + 1], color=col, lw=2.0, label=lab)
        ax.plot([xs[kc]], [y[kc]], "o", color=col, ms=5)
    ax.axvline(kx, color="white", lw=0.6, alpha=0.5)
    ax.set_xlim(0, x); ax.set_ylim(ybot if not ylog else None, ytop)
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel("frame (same clock)", color="white", fontsize=10)
    ax.set_ylabel(ylabel, color="white", fontsize=10)
    ax.legend(loc="upper left", fontsize=9, frameon=False, labelcolor="linecolor")
    return _finish(plt, fig, ax)


def render_pair(run_l, run_r, out_mp4, out_png, seconds=10.0, frames=60, cam=None, cut=False, w3d=720, h3d=720, until=None):
    """TWO runs in one movie, the vertex twin on the left and the MPM twin on the right, drawn at the SAME FRACTION of their runs (the twins
    share one clock: both run n_frames of their spec), white-to-red on ONE activator scale (the larger of the two runs' 99.5th percentiles),
    each tissue framed to its own panel. Under them three insets with both runs on the same axes, every range from the two whole runs:
    cells / cells at the start, the activator (median, 10th-90th percentile), the cell volume / mean cell volume at the start.

    `until` (frames): draw frames 0 .. until of BOTH runs on one shared clock, a run longer than `until` is cut there (a 100-frame MPM arm
    against the first 100 frames of a 300-frame vertex run), in place of the same fraction of each whole run."""
    import imageio.v2 as iio
    import exp23_studio as e23
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    e23._LOOK.clear(); e23._LOOK.update(e23.LOOK["ink"])
    e23._LOOK["text"] = "white"; e23._LOOK["border"] = (255, 255, 255)
    S = [_scene(run_l, w3d, h3d, cam), _scene(run_r, w3d, h3d, cam)]
    a_top = max(top_of_scale(s["a_rows"]) for s in S)
    no_chem = a_top <= 1e-9
    a_top = 1.0 if no_chem else a_top
    nF = max(s["n_frames"] for s in S)
    names = ["vertex twin" if s["kind"] == "vertex" else "MPM twin" for s in S]
    cols = ["white", INK_DARK]
    xs = [np.linspace(0, s["n_frames"], s["R"].rows) for s in S]
    n_ratio = [s["n_live"] / max(float(s["n_live"][0]), 1.0) for s in S]
    v_ratio = [(s["v_mean"] / max(float(s["v_mean"][0]), 1e-12), s["v_p10"] / max(float(s["v_mean"][0]), 1e-12), s["v_p90"] / max(float(s["v_mean"][0]), 1e-12)) for s in S]
    n_top = 1.1 * max(float(r.max()) for r in n_ratio)
    d_ratio = [s["diam"] / max(float(s["diam"][0]), 1e-12) for s in S]
    d_top, d_bot = 1.05 * max(float(r.max()) for r in d_ratio), 0.95 * min(float(r.min()) for r in d_ratio)
    T4 = (2 * w3d // 4, TILE[1])                                                      # four insets across the two panels
    v_top = 1.05 * max(float(v[2].max()) for v in v_ratio)
    a_top_axis = 1.05 * max(float(s["a_p90"].max()) for s in S) if not no_chem else 1.0
    font = ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"), 22)
    W, H = 2 * w3d, h3d + TILE[1] + 10
    fps = max(1, round(frames / seconds))
    writer = iio.get_writer(out_mp4, fps=fps, quality=8, macro_block_size=16)
    frame = None
    for j in range(frames):
        f = j / max(frames - 1, 1)
        span = float(until) if until else None
        ks = [int(round((f * span / s["n_frames"] if span else f) * (s["R"].rows - 1))) for s in S]
        panels = []
        for s, k in zip(S, ks):
            draw_cells(s["pl"], s["kind"], s["R"], k, a_top, cut, s["c0"], s["R0"], s["view"], s["h_vox"])
            panels.append(e23._ink_frame(s["pl"], s["dist"], s["va"], h3d, False))
        frame = np.empty((H, W, 3), np.uint8)
        frame[:] = np.array([int(GROUND[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
        for i_, (img, s, k) in enumerate(zip(panels, S, ks)):
            frame[:h3d, i_ * w3d:(i_ + 1) * w3d] = img
        im = Image.fromarray(frame); dr = ImageDraw.Draw(im)
        for i_, (s, k) in enumerate(zip(S, ks)):
            dr.text((i_ * w3d + 18, 14), f"{names[i_]}: {int(s['n_live'][k])} cells", fill=(255, 255, 255), font=font)
        frame = np.array(im)
        frame = legend(frame, a_top, "activator a per cell, one scale for both" + (" (none in these runs)" if no_chem else ""), x0=18, y0=h3d - 78)
        kx = f * (span if span else nF)
        tiles = [
            tile_pair(nF, [(names[i], cols[i], xs[i], n_ratio[i], None, None, ks[i]) for i in range(2)], kx, "cells / cells at start", n_top, size=T4),
            tile_pair(nF, [(names[i], cols[i], xs[i], S[i]["a_med"], S[i]["a_p10"], S[i]["a_p90"], ks[i]) for i in range(2)], kx,
                      "activator a (median, 10-90 %)", a_top_axis, size=T4),
            tile_pair(nF, [(names[i], cols[i], xs[i], v_ratio[i][0], v_ratio[i][1], v_ratio[i][2], ks[i]) for i in range(2)], kx,
                      "cell volume / mean at start", v_top, size=T4),
            tile_pair(nF, [(names[i], cols[i], xs[i], d_ratio[i], None, None, ks[i]) for i in range(2)], kx,
                      "vesicle diameter / at start", d_top, size=T4, ybot=d_bot)]
        for i_, tile in enumerate(tiles):
            e23._blend(frame, tile, (W - 4 * T4[0]) // 2 + i_ * T4[0], h3d + 5)
        frame = e23._clock(frame, f"frame {int(round(kx))} of {nF}", "white")
        writer.append_data(frame[: H // 2 * 2, : W // 2 * 2])
        if (j + 1) % 10 == 0:
            print(f"  pair frame {j + 1}/{frames}", flush=True)
    writer.close()
    iio.imwrite(out_png, frame)
    for s in S:
        s["pl"].close()
    return dict(kind="pair", names=names, runs=[run_l, run_r], rows=[s["R"].rows for s in S], drawn=frames, a_top=a_top,
                cells_first=[int(s["n_live"][0]) for s in S], cells_last=[int(s["n_live"][-1]) for s in S], n_frames=nF,
                diam_first=[round(float(s["diam"][0]), 3) for s in S], diam_last=[round(float(s["diam"][-1]), 3) for s in S])


# ---------------------------------------------------------------------------------------------------------- self-check
def self_check():
    """Colours, the density map and the wedge volume on planted inputs; prints what it found, exits 1 on a failure."""
    ok = True

    def check(name, cond, info=None):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info is not None else ""))
        ok &= bool(cond)
    c = act_rgb([0.0, 0.5, 1.0, 2.0], 1.0)
    check("white at 0", tuple(c[0]) == (255, 255, 255), c[0])
    check("red at the top and above", tuple(c[2]) == tuple(c[3]) == (0xE6, 0x39, 0x46), (c[2], c[3]))
    check("half way is the mix", all(abs(int(c[1][i]) - round((255 + v) / 2)) <= 1 for i, v in enumerate((0xE6, 0x39, 0x46))), c[1])
    rows = [np.linspace(0, 1, 101), np.linspace(0, 10, 101), np.zeros(0)]
    top = top_of_scale(rows)
    check("a_top is the 99.5th percentile of the whole run", abs(top - np.percentile(np.concatenate(rows[:2]), 99.5)) < 1e-9, f"a_top {top:.4f}")
    Hh, med = density_image([np.full(10, 0.1), np.r_[np.zeros(5), np.full(5, 5.0)]], 5.0)
    check("each density column sums to 1", np.allclose(Hh.sum(0), 1.0), Hh.sum(0))
    check("median of the planted columns", abs(med[0] - 0.1) < 1e-9 and abs(med[1] - 2.5) < 1e-9, med)
    # wedge volume of a unit cube's six faces about its centre: each pyramid is 1/6
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    es, et, ef = [], [], []
    for f, q in enumerate(quads):
        for i in range(4):
            es.append(q[i]); et.append(q[(i + 1) % 4]); ef.append(f)
    w = wedge_volumes(v, np.array(es), np.array(et), np.array(ef), 6)
    check("wedge volume of a unit cube's faces is 1/6 each", np.allclose(w, 1 / 6), w)
    t, u = time_axis({"general": {"dt": 0.1, "n_frames": 100, "units": {"time_s": 60.0}}}, 11)
    check("time axis in minutes", u == "min" and abs(t[1] - 1.1) < 1e-9, (t[:3], u))
    t, u = time_axis({"general": {"dt": 1.0, "n_frames": 100}}, 11)
    check("time axis in frames when the spec has no unit", u == "frames", u)
    print("SELF-CHECK", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main():
    global GROUND, AXIS_FONT, CLOCK_CORNER, COUNT_BY_TYPE, TILE
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", nargs="?")
    ap.add_argument("--why")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=None, help="draw every n-th stored row (default: at most --max-frames rows)")
    ap.add_argument("--max-frames", type=int, default=160, help="at most this many of the stored rows are drawn (every row when there are fewer)")
    ap.add_argument("--fps", type=int, default=16, help="frames per second of the movie (16: a 160-frame movie lasts 10 s)")
    ap.add_argument("--cam", default=None, help="elevation,azimuth in degrees, replacing the spec's camera")
    ap.add_argument("--cut", action="store_true", help="hide the cells on the camera's side of the centre: looks into a vesicle")
    ap.add_argument("--fake-act", default=None, choices=["radial"], help="TEST ONLY: fabricate an activator (0 at the centre to 1 at the rim, growing with time)")
    ap.add_argument("--npz", default=None, help="an MPM run's one-row live_state.npz instead of its trajectory")
    ap.add_argument("--out-dir", default="/tmp")
    ap.add_argument("--no-archive", action="store_true", help="render to --out-dir only")
    ap.add_argument("--replace", type=int, default=None, help="archive INTO this existing step (its movie, still and why-file are overwritten, no new step is made)")
    ap.add_argument("--section-alpha", nargs="*", default=[], metavar="TYPE=OPACITY", help="with --section: the second panel is the whole tissue again, with these cell types at these opacities (outer=0.4), instead of the cross section")
    ap.add_argument("--section", action="store_true", help="two 3-D panels: the whole tissue and the same frame cut through the middle (the cells on the camera's side hidden)")
    ap.add_argument("--vs", default=None, help="a SECOND run: one movie with RUN on the left and --vs on the right, drawn at the same fraction of their runs, one colour scale, shared insets")
    ap.add_argument("--frame-stride", type=int, default=None, help="frames between two stored rows of a PARTIAL trajectory.npz (a run still going): the time axis is row x this, not the run's n_frames stretched over the rows present")
    ap.add_argument("--until-frame", type=int, default=None, help="draw frames 0 to this frame only (a single run: the run is cut there, its insets end there too); with --vs: of BOTH runs on one shared clock (the longer run is cut there), not the same fraction of each whole run")
    ap.add_argument("--white-ink", action="store_true", help="every line white: the cells' ink outlines (gui/flat_ink.LINE_RGB) and the median of the activator inset")
    ap.add_argument("--dots", type=float, default=None, metavar="F", help="draw every material point as the watcher's 4D view draws its dots, F x their radius (1.7 = the largest dot button)")
    ap.add_argument("--white-plot", action="store_true", help="the insets' lines white (the median of the activator inset); the cells keep their own darkened ink")
    ap.add_argument("--activator-label", default="activator a", help="the name of the cell block chem[:, 0] in the legend and the middle inset")
    ap.add_argument("--section-slab", type=float, default=None, metavar="H", help="with --section: the second panel is a thin slice, the cells within H length units of the plane through the tissue's centre, seen face-on")
    ap.add_argument("--section-view", default=None, metavar="ELEV,AZIM", help="with --section: the cross-section panel's own camera; the slice is perpendicular to it (90,0 = from above, for a ring round z)")
    ap.add_argument("--no-insets", action="store_true", help="no plot curves: the frame is the 3-D panel(s) and the clock")
    ap.add_argument("--mark-colours", nargs="*", default=[], metavar="TYPE=HEX", help="with --dots: the cells of these types in this colour, every other cell by its activity")
    ap.add_argument("--act-cmap", default=None, help="colour the activator through this matplotlib map (viridis: blue = not active, yellow = active) instead of white to red")
    ap.add_argument("--point-volume", type=float, default=None, help="the volume of one MPM point, model units^3, over the spec's (its seed's point_spacing^3 or plotting.point_volume)")
    ap.add_argument("--type-colours", nargs="*", default=[], metavar="TYPE=HEX", help="with --cell-colours type: the colour of a cell type over the spec's plotting.colors")
    ap.add_argument("--chem-rgb", nargs=3, default=None, metavar="HEX", help="point cells: colour every cell by the shares of its first three chem channels, one colour per channel (rock-paper-scissors); no activity legend")
    ap.add_argument("--cell-colours", nargs="?", const="id", default=None, choices=["id", "type"], help="type: every cell in its type's colour (plotting.colors, per row when cell_fate changes it); id (the bare flag): every MPM cell its own colour (by its slot id; a daughter gets a new one) and no activator legend; the spec's plotting.cell_colours: id does the same")
    ap.add_argument("--label-once", action="store_true", help="one label per ARRAY (the cell block `arr`), above the array, instead of one per family")
    ap.add_argument("--label-block", default=None, help="label every family (the cells descending from one cell of frame 0) with this per-cell block's value, white and unboxed; the spec's plotting.cell_labels does the same")
    ap.add_argument("--label-with-id", action="store_true", help="with --label-block: `id <slot>   <block> <value>` instead of the value alone")
    ap.add_argument("--pull-panel", action="store_true", help="add an inset: the force of the cell_pull springs on each array against time (its peak = the force the chain carried before it let go)")
    ap.add_argument("--junction-panel", action="store_true", help="add an inset: the bound density of the junctions of each array against time, one line per array in its first cell's colour, labelled with its adhesion maximum")
    ap.add_argument("--height-panel", action="store_true", help="add an inset: the mean cell height (extent of a cell's points along z) of each family against time, one line per family in its founder cell's colour")
    ap.add_argument("--no-activator", action="store_true", help="leave out the middle inset (the activator distribution): a run with no chemistry has nothing to show in it")
    ap.add_argument("--ground", default=None, help="the background colour of the 3-D panel and the movie, hex (default %s; #000000 for the deck's black slides)" % GROUND)
    ap.add_argument("--axis-font", type=int, default=None, help="point size of the insets' axis names, their tick numbers 1 smaller (default %d)" % AXIS_FONT)
    ap.add_argument("--count-by-type", action="store_true", help="add the inset of the live cells per cell type, as the specs of the sphere twins with `plotting.count_panel: by_type` have it")
    ap.add_argument("--type-opacity", nargs="*", default=[], metavar="TYPE=OPACITY", help="the opacity of the cells of a type (1 = solid), over the spec's plotting.type_opacity: inner=1 draws the blue cell solid")
    ap.add_argument("--cut-keep", nargs="*", default=[], metavar="TYPE", help="with --cut: these cell types are not clipped at the plane: drawn whole when the plane reaches them or they lie behind it, left out when wholly on the camera's side")
    ap.add_argument("--clock-corner", default=CLOCK_CORNER, choices=["bottom-right", "top-left"], help="where the 'N cells   t = T min' text is drawn")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    GROUND, AXIS_FONT, CLOCK_CORNER, COUNT_BY_TYPE = a.ground or GROUND, a.axis_font or AXIS_FONT, a.clock_corner, a.count_by_type
    CUT_KEEP.extend(a.cut_keep)
    TYPE_OPACITY.update({kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.type_opacity})
    TILE = (TILE[0], round(TILE[1] * AXIS_FONT / 10))                                     # a taller inset keeps the plot area as large when the axis names grow
    if a.self_check:
        sys.exit(self_check())
    if not a.run or not a.why:
        ap.error("run and --why are required")
    tag = a.run.strip("/").replace("/", "__") + ("_cut" if a.cut else "") + ("_fake" if a.fake_act else "")
    if a.vs:
        tmp_mp4, tmp_png = f"{a.out_dir}/{tag}__vs__{a.vs.strip('/').replace('/', '__')}_exp29.mp4", f"{a.out_dir}/{tag}__vs__{a.vs.strip('/').replace('/', '__')}_exp29.png"
        t0 = time.time()
        info = render_pair(a.run, a.vs, tmp_mp4, tmp_png, seconds=a.seconds, frames=a.max_frames if a.max_frames < 120 else 60,
                           cam=None if not a.cam else tuple(float(v) for v in a.cam.split(",")), cut=a.cut, until=a.until_frame)
        info["wall_s"] = round(time.time() - t0)
        print("PAIR", info)
        if a.no_archive:
            print({"mp4": tmp_mp4, "png": tmp_png})
            return
        import gui_drive as gd
        i = a.replace if a.replace else gd._next_index()
        shutil.copy(tmp_png, gd._path("png", i)); shutil.copy(tmp_mp4, gd._path("mp4", i))
        _, sp = read_spec(resolve(a.vs)[0])
        if sp:
            shutil.copy(sp, gd._path("spec", i))
        with open(gd._path("why", i), "w") as f:
            f.write(f"time   {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"pair   LEFT {a.run} ({info['names'][0]}), RIGHT {a.vs} ({info['names'][1]}); one clock (both n_frames {info['n_frames']}), one activator scale 0..{info['a_top']:.4g}\n")
            f.write(f"cells  left {info['cells_first'][0]} -> {info['cells_last'][0]}, right {info['cells_first'][1]} -> {info['cells_last'][1]}\n")
            f.write(f"diam   vesicle diameter (2 x mean centroid distance from their mean), first -> last row: left {info['diam_first'][0]} -> {info['diam_last'][0]} "
                    f"(x{info['diam_last'][0] / max(info['diam_first'][0], 1e-9):.3f}), right {info['diam_first'][1]} -> {info['diam_last'][1]} (x{info['diam_last'][1] / max(info['diam_first'][1], 1e-9):.3f})\n")
            f.write(f"why    {a.why}\n")
        gd.note(f"  exp29_studio {i:04d}  PAIR {a.run} | {a.vs}  -- {'REDRAWN IN PLACE: ' if a.replace else ''}{a.why}")
        print({"index": i, "wall_s": info["wall_s"]})
        return
    CELL_COLOURS["on"] = bool(a.cell_colours) or bool(a.chem_rgb); CELL_COLOURS["mode"] = a.cell_colours
    CHEM_RGB["cols"] = a.chem_rgb
    POINT_VOLUME["v"] = a.point_volume; ACT_CMAP["name"] = a.act_cmap
    MARK_COLOURS.update({kv.split("=")[0]: kv.split("=")[1] for kv in a.mark_colours})
    SECTION_SLAB["h"] = a.section_slab; NO_INSETS["on"] = bool(a.no_insets)
    SECTION_VIEW["cam"] = tuple(float(v) for v in a.section_view.split(",")) if a.section_view else None
    TYPE_COLOURS.update({kv.split("=")[0]: kv.split("=")[1] for kv in a.type_colours})
    DOTS["factor"] = a.dots
    WHITE_INK["on"] = bool(a.white_ink); WHITE_PLOT["on"] = bool(a.white_plot); ACT_LABEL["text"] = a.activator_label
    if a.white_ink:
        from plexus.gui import flat_ink
        flat_ink.LINE_RGB = (255, 255, 255)
    HEIGHT_PANEL["on"] = bool(a.height_panel)
    JUNCTION_PANEL["on"] = bool(a.junction_panel)
    PULL_PANEL["on"] = bool(a.pull_panel)
    CELL_LABELS["block"], CELL_LABELS["value_only"] = a.label_block, not a.label_with_id
    CELL_LABELS["once"] = bool(a.label_once)
    SECTION_ALPHA.update({kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.section_alpha})
    tmp_mp4, tmp_png = f"{a.out_dir}/{tag}_exp29.mp4", f"{a.out_dir}/{tag}_exp29.png"
    t0 = time.time()
    info = render(a.run, tmp_mp4, tmp_png, seconds=a.seconds, stride=a.stride, max_frames=a.max_frames,
                  cam=None if not a.cam else tuple(float(v) for v in a.cam.split(",")), cut=a.cut, fake=a.fake_act, npz=a.npz, no_activator=a.no_activator, fps=a.fps, section=a.section, until=a.until_frame, frame_stride=a.frame_stride)
    info["wall_s"] = round(time.time() - t0)
    print("SCALES USED", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in info.items()})
    if a.no_archive:
        print({"mp4": tmp_mp4, "png": tmp_png})
        return
    import gui_drive as gd                                   # the record's own numbering and paths
    i = a.replace if a.replace else gd._next_index()           # --replace N: the step is redrawn in place, the record gets no new step
    shutil.copy(tmp_png, gd._path("png", i))
    shutil.copy(tmp_mp4, gd._path("mp4", i))
    d, _ = resolve(a.run)
    spec, sp = read_spec(d)
    if sp:
        shutil.copy(sp, gd._path("spec", i))
    with open(gd._path("why", i), "w") as f:
        f.write(f"time   {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"run    {a.run} ({info['kind']}; ink look, {'every cell its own colour, by id' if (CELL_COLOURS['on'] or _spec_by_id(info.get('spec'))) else 'cells white to red by activator'}, tools/exp29_studio.py)\n")
        f.write(f"drawn  {info['drawn']} of {info['rows']} stored rows; activator scale 0..{info['a_top']:.4g} (99.5th percentile of the whole run)"
                f"{' -- FABRICATED activator, a renderer test' if a.fake_act else ''}\n")
        f.write(f"insets cells {info['cells_first']} -> {info['cells_last']}; {'' if a.no_activator else 'activator distribution; '}cell volume ({info['vol_unit']}) "
                f"mean {info['volume_mean_first']:.4g} -> {info['volume_mean_last']:.4g}, y to {info['volume_top']:.4g}\n")
        f.write(f"why    {a.why}\n")
    gd.note(f"  exp29_studio {i:04d}  {a.run}  -- {'REDRAWN IN PLACE: ' if a.replace else ''}{a.why}")
    out = {"index": i, **{k: info[k] for k in ("rows", "drawn", "a_top", "wall_s")}}
    print(out)


if __name__ == "__main__":
    main()
