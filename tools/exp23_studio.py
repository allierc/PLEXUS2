"""exp23 (frog cleavage): the MACRO-PHOTO render of one run, archived as a step of the experiment's record.

    PLEXUS_BUILDER=experiments/<record> \\
        python tools/exp23_studio.py <group>/<run> --why "..." [--seconds 10] [--caption]

The live movie colours each cell by ONE value (its mean yolk, `contour_bodies.color_by`), so the 1-cell egg is a
single tan. The human's two films (experiments/exp23_frog_cleavage/data/reference_sequence/) are a macro photograph:
a golden-brown pigmented animal half over a pale cream vegetal half, every blastomere a rounded dome, dark grooves
between them, a teal-grey background. This draws that FROM THE RECORDED TRAJECTORY:

    cells   each cell (`pt__pid`) its own surface: its points splatted on a voxel grid (0.55 x their spacing),
            blurred over 1.6 voxels, the 0.5 iso-surface of their volume fraction, Taubin-smoothed -- two cells
            pressed together meet in a groove, which ambient occlusion darkens as in the films
    colour  per vertex, from the yolk of its 6 nearest points (`pt__yolk`, 0 animal .. 1 vegetal): brown above a
            yolk of ~0.4, cream below ~0.6 (the pigment boundary of a Xenopus egg lies near the equator), with a
            speckle fixed to each material point (a pigment granule moves with its cytoplasm)
    view    from above the animal pole, tilted 25 degrees, as screencast_1
    inset   bottom-left, a cross-section through the embryo's centre along the animal-vegetal axis (`_section_image`):
            the cells' interiors, the yolk gradient, dark cell walls; majority cell over 16 points per pixel,
            rendered at 2x and averaged down (Cedric, 2026-10-11: the first was "too big, grid-like, speckled")

The movie and its last frame are written into the record through tools/gui_drive.py's own step numbering.
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
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
ROOT = os.path.join(os.environ.get("PLEXUS_OUTPUT_ROOT") or os.environ.get("GNN_OUTPUT_ROOT") or os.path.join(HERE, ".."), "graphs_data")   # the runs' data root (default: the repository's own graphs_data/)

# yolk -> colour stops, read off screencast_1_sheet.png (animal pole, granular brown; equator; vegetal cream)
_STOPS = [(0.00, (0x6a, 0x46, 0x20)), (0.38, (0x9e, 0x76, 0x3a)), (0.50, (0xc2, 0x9c, 0x5a)),
          (0.62, (0xe6, 0xd6, 0x9a)), (1.00, (0xf5, 0xec, 0xc4))]
# FLAT (Cedric, 2026-10-11: the macro look "looks like wood at best"; wanted "a more flat rendering like movie 6",
# every overlay black): the live movie's own palette (YlOrBr reversed, orange animal -> cream vegetal), no granule
# speckle, a matte material under the plain light kit, black text, a black-framed inset and a black cell-count plot.
LOOK = {"macro": dict(stops=_STOPS, speckle=0.12, dark=0.4, light=(1.1, 0.35, 0.25), pbr=True, specular=0.14,
                      specular_power=14, diffuse=0.85, ambient=0.22, ssao=True, text="#e8eef0", border=(0xd8, 0xe2, 0xe6),
                      plot=False),
        "flat": dict(stops=[(0.00, (0xcc, 0x6e, 0x1e)), (0.38, (0xe0, 0x8c, 0x33)), (0.50, (0xe9, 0xaf, 0x5c)),
                            (0.62, (0xf1, 0xd3, 0x95)), (1.00, (0xf8, 0xee, 0xc8))],
                     speckle=0.0, dark=0.0, light=(0.9, 0.5, 0.45), pbr=False, specular=0.0, specular_power=1,
                     diffuse=0.6, ambient=0.4, ssao=False, text="#000000", border=(0, 0, 0), plot=True),
        # INK (Cedric, 2026-10-13: the watcher's 3-D view in flat fills with ink outlines, "use [it] in watcher main
        # display"): gui/flat_ink.py's per-cell palette (keyed on the cell id, so a cell keeps its colour), its
        # flat material (ambient 0.85, diffuse 0.15, no specular, no SSAO) and its ink drawn from the depth buffer of
        # each frame; a flat grey ground (the live movie's #8d99a6), white overlays, the inset in the same colours
        "ink": dict(stops=_STOPS, speckle=0.0, dark=0.0, light=(0.9, 0.5, 0.45), pbr=False, specular=0.0,
                    specular_power=1, diffuse=0.15, ambient=0.85, ssao=False, text="#ffffff", border=(255, 255, 255),
                    plot=True, ink=True, bg="#8d99a6", inset_bg=(0x76, 0x82, 0x8e))}
_LOOK = LOOK["macro"]


def _yolk_rgb(y, speck):
    y = np.clip(y, 0.0, 1.0)
    st = _LOOK["stops"]
    xs = np.array([q[0] for q in st])
    rgb = np.stack([np.interp(y, xs, [q[1][c] for q in st]) for c in range(3)], 1)
    rgb *= ((1.0 + _LOOK["speckle"] * (speck - 0.5) * (1.0 - y))[:, None]
            * (1.0 - _LOOK["dark"] * np.clip(0.4 - y, 0, None))[:, None])
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _cell_surface(P, v, h, blur=1.6, iso=0.5, pad=4):
    """The 0.5 iso-surface of the blurred volume fraction of one cell's points P (each of volume v)."""
    import pyvista as pv
    from scipy.ndimage import gaussian_filter
    while True:
        lo = P.min(0) - pad * h
        shape = np.ceil((P.max(0) + pad * h - lo) / h).astype(int) + 1
        ijk = np.floor((P - lo) / h).astype(int)
        vol = np.zeros(shape, np.float32)
        np.add.at(vol, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), v / h ** 3)
        vol = gaussian_filter(vol, blur)
        # A CELL PACKED DENSER THAN ITS ISO LEVEL ASSUMES (exp29 rosettes, 2026-10-31: the growing inner cell holds 5x the 0.5 volume fraction a normal cell's median voxel has) is
        # still above `iso` on the faces of a box padded by 4 voxels, so the contour was cut by the box and the cube drew open. Pad until the blurred field has fallen below `iso` on every face.
        face = max(vol[0].max(), vol[-1].max(), vol[:, 0].max(), vol[:, -1].max(), vol[:, :, 0].max(), vol[:, :, -1].max())
        if face < iso or pad >= 40:
            break
        pad = min(40, pad + max(2, int(np.ceil(blur * np.sqrt(2.0 * np.log(face / iso))))))
    grid = pv.ImageData(dimensions=tuple(int(s) for s in shape), spacing=(h, h, h), origin=tuple(lo))
    grid.point_data["f"] = vol.ravel(order="F")
    surf = grid.contour([iso], scalars="f", method="flying_edges")
    if surf.n_points == 0:
        return None
    return surf.smooth_taubin(n_iter=25, pass_band=0.05).compute_normals(auto_orient_normals=False,
                                                                          split_vertices=False)


def render_live(run, npz, out_png, cut=False):
    """One still of a running job's dumped state (`live_state.npz`) in the ink look, white overlay -- called by the watcher's
    long-lived render worker, which imports everything once (a fresh process spends 10 s importing pyvista, vtk and warp)."""
    from matplotlib.colors import to_rgb
    _LOOK.clear(); _LOOK.update(LOOK["ink"])
    _LOOK["text"] = "white"
    _LOOK["border"] = tuple(int(255 * c) for c in to_rgb("white"))
    tmp = out_png + ".tmp.mp4"
    render(run, tmp, out_png, rows_sel=[0], section=False, cut=cut, npz=npz)
    if os.path.exists(tmp):
        os.remove(tmp)


def _section_frame(X, occ, pid, c0):
    """The section plane's in-plane horizontal axis e and normal n: the plane holds the animal axis (z) and the
    FIRST cleavage's spindle (the line joining the two first daughters' centres) turned 45 degrees, so it crosses
    both first daughters and all four quarters after the second cut. A plane fixed at x = centre can coincide with the first cut, whose azimuth the lattice picks: both
    daughters' points then fall in the slab and alternate pixel by pixel (v10's test, 2026-10-11)."""
    for k in range(X.shape[0]):
        ids = np.unique(pid[k][occ[k]])
        if len(ids) >= 2:
            ca = X[k][occ[k] & (pid[k] == ids[0])].mean(0); cb = X[k][occ[k] & (pid[k] == ids[1])].mean(0)
            e = np.array([cb[0] - ca[0], cb[1] - ca[1], 0.0])
            if np.linalg.norm(e) > 1e-6:
                e /= np.linalg.norm(e)
                # 45 degrees off the first spindle: the second cut is perpendicular to the first, so a plane
                # holding the first spindle coincides with the second cut; the diagonal crosses all four quarters
                c, s_ = np.cos(np.pi / 4), np.sin(np.pi / 4)
                e = np.array([c * e[0] - s_ * e[1], s_ * e[0] + c * e[1], 0.0])
                return e, np.array([-e[1], e[0], 0.0])
    return np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])


def _section_image(P, Yk, pid, c0, R0, speck, e, nrm, out=230, slab=0.3, k=16, ss=2, ink=False, marked=None):
    """A CROSS-SECTION of the embryo (Cedric, 2026-10-11: an inset to see inside): the points within `slab` of the
    vertical plane through the centre spanned by e (horizontal, in the plane) and z (the animal axis, up), drawn on
    an `out` x `out` tile. Every pixel takes the MAJORITY cell of its `k` nearest slab points (one nearest point
    alone flips between two cells wherever their points interleave, which speckled the first version) and the mean
    yolk of those points; a dark line runs where the majority cell changes. Rendered at `ss` x the size and averaged
    down, so the disc's edge, the walls and the colour steps are antialiased. `ink`: each cell in its own colour of
    the ink look's palette (the 3-D cells' colours), walls its colour darkened 60 %, on the look's darker grey."""
    from scipy.spatial import cKDTree
    from scipy.stats import mode
    from scipy.ndimage import binary_dilation
    D = P - c0
    sel = np.abs(D @ nrm) < slab
    half = 1.15 * R0
    n = out * ss
    px = 2 * half / n
    ys = -half + (np.arange(n) + 0.5) * px                                # along e, in the plane
    zs = half - (np.arange(n) + 0.5) * px                                 # row 0 at the top = animal pole up
    gy, gz = np.meshgrid(ys, zs)                                         # [n, n]
    bg = np.array(_LOOK.get("inset_bg", (0x3f, 0x62, 0x6c)), np.float32)
    img = np.broadcast_to(bg, (n, n, 3)).astype(np.float32).copy()
    if sel.sum() >= k:
        Q = np.stack([D[sel] @ e, D[sel][:, 2]], 1)
        d, nn = cKDTree(Q).query(np.stack([gy.ravel(), gz.ravel()], 1), k=k)
        inside = (d[:, 3] <= 1.5 * slab).reshape(n, n)                # the 4th-nearest point: a smooth silhouette
        lab = mode(pid[sel][nn], axis=1, keepdims=False).mode.reshape(n, n)
        lab = np.where(inside, lab, -1)
        if ink:
            from plexus.gui import flat_ink
            rgb = (flat_ink.palette(np.where(lab >= 0, lab, 0).ravel()) * 255).reshape(n, n, 3).astype(np.float32)
        else:
            rgb = _yolk_rgb(Yk[sel][nn].mean(1), np.full(nn.shape[0], 0.5)).reshape(n, n, 3).astype(np.float32)
        if marked is not None and len(marked):
            # the BOTTLE cells (exp25: cell block `apical` > 0 or tension > 1) in dark red, so the lip they form reads
            # in the section (Cedric, 2026-10-13: "a cross section ... to see gastrulation")
            mk = np.isin(lab, marked)
            rgb = np.where(mk[..., None], np.array([0x9e, 0x2a, 0x2b], np.float32), rgb)
        edge = np.zeros((n, n), bool)
        for ax in (0, 1):
            r = np.roll(lab, 1, axis=ax)
            edge |= (lab != r) & inside & (r != -1)
        edge = binary_dilation(edge, iterations=max(1, ss // 2)) & inside
        rgb = np.where(edge[..., None], rgb * (0.4 if ink else 0.25), rgb)
        img = np.where(inside[..., None], rgb, img)
    img = img.reshape(out, ss, out, ss, 3).mean(axis=(1, 3))             # the antialiasing average
    return np.clip(img, 0, 255).astype(np.uint8)


def _count_plot(t_all, n_all, t_now, size=(440, 280), colour="black"):
    """The cell count against time, drawn by matplotlib on a transparent background in one colour (the live
    movie's panel, in black as Cedric asked): the whole curve, with a dot at the current time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    ax = fig.add_axes([0.2, 0.22, 0.76, 0.7])
    ax.step(t_all, n_all, where="post", color=colour, lw=1.4)
    k = int(np.searchsorted(t_all, t_now, side="right")) - 1
    ax.plot([t_now], [n_all[max(k, 0)]], "o", color=colour, ms=4)
    ax.set_xlabel("time (min)", color=colour, fontsize=12); ax.set_ylabel("cells", color=colour, fontsize=12)
    ax.set_xlim(0, t_all[-1]); ax.set_ylim(0, max(4, 1.1 * max(n_all)))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(colour)
    ax.tick_params(colors=colour, labelsize=10)
    fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :4].copy()
    plt.close(fig)
    return buf


def _sphere_curves(d, um):
    """Three curves of a spheroid run over its stored rows, from the whole trajectory (never the slice): the live cells of each type, the
    share of the live outer cells that touch an inner cell (a junction row between them with area above 0.25: their grip on the layer below),
    and the median displacement (micrometres) of the outer cells alive at row 0 about the centre of the inner cells alive then, from row 0 (a median: one cell flung off the sphere would swing a mean)."""
    nt = d["cell__node_type"].astype(int)
    n = len(nt)
    occ = d["pt__occ"].astype(bool)
    pid = np.rint(d["pt__pid"][..., 0]).astype(int)
    T = occ.shape[0]
    live = np.zeros((T, n), bool)
    for t in range(T):
        live[t] = np.bincount(pid[t][occ[t]], minlength=n)[:n] > 0
    cells = [(live & (nt == 0)).sum(1), (live & (nt == 1)).sum(1)]
    ca, cb = d["junction__ca"][..., 0], d["junction__cb"][..., 0]
    ar, oc = d["junction__area"][..., 0], d["junction__occ"].astype(bool)
    grip = np.zeros(T)
    for t in range(T):
        m = oc[t] & (ar[t] > 0.25)
        a, b = np.rint(ca[t][m]).astype(int), np.rint(cb[t][m]).astype(int)
        mixed = nt[a] != nt[b]
        has = np.zeros(n, bool)
        has[np.where(nt[a] == 1, a, b)[mixed]] = True
        o = (nt == 1) & live[t]
        grip[t] = float(has[o].mean()) if o.any() else 0.0
    c = d["cell__pos"].astype(np.float64)
    ctr = c[:, (nt == 0) & live[0]].mean(1)
    P = (c - ctr[:, None]) * um
    ids = (nt == 1) & live[0]
    disp = np.median(np.linalg.norm(P[:, ids] - P[0, ids], axis=2), axis=1)
    return {"cells": cells, "grip": grip, "disp": disp}


def _curve_tile(t_all, series, t_now, ylabel, ylim, colour="white", xlabel=None, size=(270, 180)):
    """One small curve panel on a transparent background: `series` is [(values, colour)], the whole curves drawn thin and the part up to
    `t_now` thick, a dot at the current time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    ax = fig.add_axes([0.24, 0.24 if xlabel else 0.16, 0.72, 0.62 if xlabel else 0.70])
    k = int(np.searchsorted(t_all, t_now, side="right")) - 1
    k = max(k, 0)
    for v, col in series:
        ax.plot(t_all, v, color=col, lw=0.8, alpha=0.45)
        ax.plot(t_all[: k + 1], np.asarray(v)[: k + 1], color=col, lw=1.8)
        ax.plot([t_all[k]], [np.asarray(v)[k]], "o", color=col, ms=4)
    ax.set_xlim(0, t_all[-1]); ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, color=colour, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=colour, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(colour)
    ax.tick_params(colors=colour, labelsize=8, length=3)
    fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :4].copy()
    plt.close(fig)
    return buf


def _blend(frame, rgba, x0, y0):
    """Alpha-blend an RGBA tile onto the frame at (x0, y0)."""
    h, w = rgba.shape[:2]
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    frame[y0:y0 + h, x0:x0 + w] = (a * rgba[..., :3] + (1 - a) * frame[y0:y0 + h, x0:x0 + w]).astype(np.uint8)
    return frame


def _scale_bar(frame, bar, colour):
    """A flat scale bar under the cells, its label upright beneath it (Cedric, 2026-10-13: the 3D label lay in the
    xy plane and was hard to read)."""
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    from matplotlib.colors import to_rgb
    L_um, L_px, y = bar
    col = tuple(int(255 * c) for c in to_rgb(colour))
    im = Image.fromarray(frame); dr = ImageDraw.Draw(im)
    W = frame.shape[1]
    font = ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"), 22)

    def _lab(L):
        return f"{L / 1000:g} mm" if L >= 1000 else f"{L:g} \u00b5m"
    if _LOOK.get("ink"):
        # THE BAR AS WIDE AS ITS LABEL (flat_ink.round_length, the watcher's rule): `L_px / L_um` pixels per micrometre
        from plexus.gui import flat_ink
        ppu = L_px / L_um
        L_um = flat_ink.round_length(ppu, lambda L: dr.textlength(_lab(L), font=font))
        L_px = L_um * ppu
    x0 = int(W / 2 - L_px / 2)
    dr.rectangle([x0, y, x0 + int(L_px), y + 4], fill=col)
    lab = _lab(L_um)
    tw = dr.textlength(lab, font=font)
    dr.text((W / 2 - tw / 2, y + 9), lab, fill=col, font=font)
    return np.array(im)


def _clock(frame, text, colour, margin=24, corner="bottom-right"):
    """The cell count and the time at the frame's bottom-right (`corner` "top-left": its top-left), `margin` px in, drawn like the scale bar's label (PIL,
    DejaVu Sans 22, antialiased) -- Cedric, 2026-10-13: the VTK text actor it replaces was "not white, not neat", a
    jagged other font that the ink outlined as a silhouette."""
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    from matplotlib.colors import to_rgb
    col = tuple(int(255 * c) for c in to_rgb(colour))
    im = Image.fromarray(frame); dr = ImageDraw.Draw(im)
    font = ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"), 22)
    l, t, r, b = dr.textbbox((0, 0), text, font=font)
    x, y = (margin - l, margin - t) if corner == "top-left" else (frame.shape[1] - margin - (r - l) - l, frame.shape[0] - margin - (b - t) - t)
    dr.text((x, y), text, fill=col, font=font)
    return np.array(im)


def _paste(frame, inset, margin=24, border=1, colour=(0xd8, 0xe2, 0xe6)):
    """The inset at the frame's bottom-left, with a thin light border."""
    h, w = inset.shape[:2]
    H, W = frame.shape[:2]
    y0, x0 = H - margin - h, margin
    frame[y0 - border:y0 + h + border, x0 - border:x0 + w + border] = colour
    frame[y0:y0 + h, x0:x0 + w] = inset
    return frame


def _ink_frame(pl, dist, va, height, nuclei, r_cell=0.0, dot_r=None, colour_origin=(0.0, 0.0, 0.0)):
    """One frame in the ink look: the screenshot with gui/flat_ink.py's outlines drawn from the depth buffer.

    Opaque cells: the frame's own depth and colours -- each cell's silhouette, the walls between cells (a change of
    hue) and the folds of a surface. Translucent cells (`--nuclei`): a translucent surface writes no depth, and the
    walls deep inside the embryo, seen through it, must not be inked as noise, so the lines of the cells come from ONE
    EXTRA RENDER with every cell opaque -- the silhouettes and the walls on the embryo's visible surface only -- and
    each nucleus's outline from a SECOND one with the nuclei alone (read off the translucent frame's own depth, the
    rings came out hatched). `r_cell`: a cell's mean equivalent radius, world."""
    from plexus.gui import flat_ink
    wpp = 2.0 * dist * math.tan(math.radians(va / 2)) / height    # world per pixel at the focal plane

    def _shot():
        # RENDER FIRST: screenshot() alone returned the previous render, and the --nuclei frame came out with every
        # cell opaque (the extra pass's picture) and the nuclei hidden behind them
        pl.render()
        return np.ascontiguousarray(pl.screenshot(return_img=True)[..., :3]), flat_ink.zbuffer(pl.render_window)

    def _scene(zb):                                              # drawn, and not at the near plane
        return (zb < 1.0 - 1e-7) & (flat_ink.depth(zb, pl.camera) > 0.2 * dist)
    if not nuclei:
        frame, zb = _shot()
        if zb is None or zb.shape != frame.shape[:2]:
            return frame
        # `dot_r` / `colour_origin` (exp06, 2026-11-14): a frame of DOTS is inked as the watcher's 4D view inks its dots (bio_view: the dot radius, the ground as the colour origin)
        return flat_ink.ink(frame, flat_ink.depth(zb, pl.camera), _scene(zb), wpp, dot_r=dot_r, colour_origin=colour_origin)
    cells = [a for nm, a in pl.actors.items() if nm.startswith("cell_")]
    nucs = [a for nm, a in pl.actors.items() if nm.startswith("nuc_")]
    ops = [a.GetProperty().GetOpacity() for a in cells]
    for a in cells:
        a.GetProperty().SetOpacity(1.0)
    for a in nucs:
        a.SetVisibility(False)
    img_op, zb_op = _shot()                                      # pass 1: the cells' front surfaces, opaque
    for a in cells:
        a.SetVisibility(False)
    for a in nucs:
        a.SetVisibility(True)
    _, zb_n = _shot()                                            # pass 2: the nuclei alone, clean silhouettes
    for a, o in zip(cells, ops):
        a.SetVisibility(True)
        a.GetProperty().SetOpacity(o)
    frame, _ = _shot()                                           # the frame: translucent cells, nuclei inside
    if zb_n is None or zb_op is None or zb_n.shape != frame.shape[:2]:
        return frame
    sc_op = _scene(zb_op)
    _, cell_l, hard_l = flat_ink.ink_masks(img_op, flat_ink.depth(zb_op, pl.camera), sc_op, wpp)
    nuc = _scene(zb_n)
    Dn = flat_ink.depth(zb_n, pl.camera)
    rim, _, rim_h = flat_ink.ink_masks(frame, Dn, nuc, wpp, contacts=False)
    rim = (rim | rim_h) & nuc                                    # a nucleus's edge: against its cell or another nucleus
    # A NUCLEUS SEEN THROUGH MORE THAN ITS OWN CELL is outlined at the light weight: deeper than `r_cell` behind the
    # embryo's visible surface, its fill is faint through two translucent layers, and a full-weight ring around it read
    # as an empty circle (exp23_v8, 16 cells, 2026-10-13)
    near = nuc & ((Dn - flat_ink.depth(zb_op, pl.camera)) < r_cell)
    return flat_ink.apply(frame, rim & ~near, cell_l | (rim & near), hard_l)


def _nucleus_law(spec):
    """The cut's nucleus as `cell_divide[cleave]` places it: (yolk_weight a, yolk_law) from the spec, else (0, linear)."""
    for o in spec.get("operators", []):
        if o.get("op") == "cell_divide" and o.get("model") == "cleave":
            return float(o.get("yolk_weight", 1.0)), str(o.get("yolk_law", "linear"))
    return 0.0, "linear"


def render(run, out_mp4, out_png, seconds=10.0, width=1280, height=1088, rows_sel=None, section=True, nuclei=False,
           section_px=230, ring=False, cam_override=None, cut=False, npz=None, curves=True):
    import pyvista as pv
    from scipy.spatial import cKDTree
    d = np.load(npz or os.path.join(ROOT, run, "trajectory.npz"))          # `npz`: a one-row state a running job dumped
    sp = os.path.join(ROOT, run, "spec.yaml")
    if not os.path.exists(sp):                       # a cluster job copies its spec at the very end
        sp = os.path.join(HERE, "..", "config", run + ".yaml")
    spec = yaml.safe_load(open(sp))
    X, occ = d["pt__pos"].astype(np.float64), d["pt__occ"]
    pid = np.rint(d["pt__pid"][..., 0]).astype(np.int64)
    rows, N = X.shape[0], X.shape[1]
    # A RIBBON JOINED INTO A RING (exp27, `contact_finder period`): positions are unwrapped along x. `ring` draws x as an
    # angle round a circle of circumference `period` (cells straddling the seam stay whole; the cells' widths are drawn
    # stretched toward the outer edge, by (R + y)/R); otherwise each cell is moved by whole periods to put its centroid
    # in the window of one period centred on the layout.
    _per = next((float(o["period"]) for o in spec["operators"] if o.get("op") == "contact_finder" and o.get("period")), 0.0)
    if _per > 0.0:
        _cg = next(q for q in spec["seed"] if q.get("model") == "cell_grid")
        _x0, _yc = float(_cg["centre"][0]) - 0.5 * _per, float(_cg["centre"][1])
        if ring in ("tube", "tube_in"):
            # ROLLED OUT OF THE PLANE: the ribbon's thickness (z) becomes the radius and its width (y) the tube's axis,
            # so a layer keeps its width and a layer further out is stretched by only (R + dz) / R; circumference
            # `period` at the substrate's mid-plane (z = floor + 0.5 side), the top layer outside (`tube`) or inside
            _th = 2.0 * math.pi * (X[..., 0] - _x0) / _per
            _s = 1.0 if ring == "tube" else -1.0
            _rho = _per / (2.0 * math.pi) + _s * (X[..., 2] - float(_cg["centre"][2]) + 0.5)
            X[..., 0] = _rho * np.cos(_th); X[..., 2] = _rho * np.sin(_th) + float(_cg["centre"][2])
        elif ring:
            _th = 2.0 * math.pi * (X[..., 0] - _x0) / _per
            _r = _per / (2.0 * math.pi) + (X[..., 1] - _yc)
            X[..., 0] = _r * np.cos(_th); X[..., 1] = _r * np.sin(_th)
        else:
            _nc = int(pid.max()) + 1
            for _k in range(rows):
                _m = occ[_k].astype(bool)
                _n = np.bincount(pid[_k][_m], minlength=_nc)
                _cx = np.bincount(pid[_k][_m], weights=X[_k][_m, 0], minlength=_nc) / np.maximum(_n, 1)
                X[_k, :, 0] += (-_per * np.floor((_cx - _x0) / _per))[pid[_k]]
    Y = d["pt__yolk"][..., 0].astype(np.float64) if "pt__yolk" in d.files else np.zeros((rows, N))
    bm_all = None                                         # the basement membrane's nodes per row (a list of [n, 3]), when the run has one
    if "bm_node__pos" in d.files:                         # a trajectory: the node set's rows (the reserve is parked at the sphere's centre)
        _B = d["bm_node__pos"]
        _cB = np.median(_B[0], axis=0)
        bm_all = [b[np.linalg.norm(b - _cB, axis=1) > 1e-3] for b in _B]
    elif "bm__pos" in d.files:                            # a live state: the live nodes
        bm_all = [d["bm__pos"][k] for k in range(d["bm__pos"].shape[0])]
    seeds = {s.get("model"): s for s in spec["seed"]}
    if "spheroid_coat" in seeds:                          # a spheroid with a coat (basement membrane tiles or mesenchyme cells) outside it
        seeds["spheroid"] = seeds["spheroid_coat"]
    mat = None
    type_rank = None
    curves_data = None
    slab_z = None                                         # (centre z, half thickness) of the slice, when the run is cut
    if "spheroid" in seeds:                               # colour by identity: inner cells blue, outer cells red, a shade per cell
        type_rank = d["cell__node_type"].astype(int)
        if curves and d["pt__occ"].shape[0] > 1 and "junction__area" in d.files:
            curves_data = _sphere_curves(d, float((spec["general"].get("units") or {}).get("length_um", 1.0)))
        if cut:                                            # a SLAB (Cedric, 2026-10-18: "a slice is a truncation of the scene with a
            # rectangular volume"; 2026-10-20: the slice of the watcher's 3-D view, not whole cells): the POINTS within 0.2 x the spheroid's
            # radius of the centre plane normal to z are kept, so a sphere becomes a thick disc whose cells are cut flat at its faces and
            # seen from above do not overlap each other
            occ = occ.astype(bool).copy()
            _x0 = X[0][occ[0]]
            _c0 = np.median(_x0, axis=0)
            _half = 0.2 * float(np.percentile(np.linalg.norm(_x0 - _c0, axis=1), 98))
            for _k in range(rows):
                occ[_k] &= np.abs(X[_k][:, 2] - _c0[2]) <= _half
            slab_z = (float(_c0[2]), float(_half))
    view_override = None
    zone0 = np.rint(d["pt__zone"][0, :, 0]).astype(np.int64) if "pt__zone" in d.files else None
    if "blastula" in seeds:                             # a seeded blastula (exp25): the embryo, its blastocoel a body
        seed = seeds["blastula"]
        R0, c0 = float(seed["radius"]), np.array(seed["centre"], float)
        v = 4.0 / 3.0 * math.pi * R0 ** 3 / max(int(occ[0].sum()), 1)
    elif "egg" in seeds:                                # one egg (exp23): framed on the egg
        seed = seeds["egg"]
        R0, c0 = float(seed["radius"]), np.array(seed["centre"], float)
        v = 4.0 / 3.0 * math.pi * R0 ** 3 / N
    elif "collide" in seeds:                            # pairs launched at each other (exp25 collision probes)
        seed = seeds["collide"]
        r_c = float(seed["radius"]); c0 = np.array(seed["centre"], float)
        per = spec["sets"]["pt"]["per_parent"]
        per = int(next(iter(per.values()))) if isinstance(per, dict) else int(per)
        n_rows = int(len(d["cell__node_type"]) // 2) if not seed.get("pairs") else 1 + max(int(q[1]) for q in seed["pairs"])
        n_cols = 1 if not seed.get("pairs") else 1 + max(int(q[0]) for q in seed["pairs"])
        R0 = 0.5 * max(n_rows * float(seed.get("row_spacing", 3.0)),
                       n_cols * float(seed.get("column_spacing", 12.0)) if n_cols > 1 else 2 * r_c + float(seed.get("gap", 1.0)) + 4)
        R0 *= 0.75                                      # the 5.6 R0 lens distance fits an egg; a tall column needs less margin
        v = 4.0 / 3.0 * math.pi * r_c ** 3 / per
        section = False
        nt = d["cell__node_type"]
        mat = np.broadcast_to((nt / max(int(nt.max()), 1))[None, :], (rows, len(nt)))   # colour = the type's rank
    elif "cell_grid" in seeds or "clump" in seeds or "spheroid" in seeds:      # exp27 testbench / clump / spheroid: framed on the cells
        seed = seeds.get("cell_grid") or seeds.get("clump") or seeds.get("spheroid")
        hp = float(seed.get("point_spacing", 0.125 if ("cell_grid" in seeds or "spheroid" in seeds) else 0.2))
        v = hp ** 3
        both = np.concatenate([X[k][occ[k]] for k in (0, rows - 1)])
        c0 = both.mean(0)
        R0 = 0.5 * float(np.ptp(both, axis=0).max())
        if "spheroid" in seeds:                           # frame on the spheroid, not on cells flung out of it: the median point and a ball
            c0 = np.median(X[0][occ[0]], axis=0)          # about 1.3 x the starting radius
            R0 = 1.3 * float(np.percentile(np.linalg.norm(X[0][occ[0]] - c0, axis=1), 98))
        section = False
        # THE SPEC'S OWN CAMERA (exp27, 2026-10-14): the arrays lie in the x-z plane, one cell deep in y, so the
        # studio's egg view (down the animal axis, tilted 25 deg) would see only their top faces
        cam = (spec.get("plotting") or {}).get("camera") or {}
        view_override = (float(cam.get("elev", 20.0)), float(cam.get("azim", -90.0)))
        if cam_override is not None:
            view_override = cam_override
        elif cut and "spheroid" in seeds:                  # a slab seen from above and a little aside: its face and its thickness
            view_override = (55.0, -70.0)
    else:                                               # a grid of cells, one material each (exp25 division probe)
        seed = seeds["cell_row"]
        r_c = float(seed["radius"]); c0 = np.array(seed["centre"], float)
        ncol = int(seed.get("columns", 5)); sp_c = float(seed.get("spacing", 6.0))
        n_live0 = int(occ[0].sum()) // int(spec["sets"]["pt"]["per_parent"])
        nrow = int(math.ceil(n_live0 / ncol))
        R0 = 0.5 * max(ncol, nrow) * sp_c                 # framing radius
        v = 4.0 / 3.0 * math.pi * r_c ** 3 / int(spec["sets"]["pt"]["per_parent"])
        section = False                                   # no embryo to cut through
        if "cell__mat" in d.files:
            mat = d["cell__mat"][..., 0]
    h = 0.55 * v ** (1.0 / 3.0)
    speck = np.random.default_rng(23).random(N)
    g = spec["general"]
    stride = math.ceil((int(g["n_frames"]) + 1) / rows)
    t_min = [(k * stride + 1) * float(g["dt"]) for k in range(rows)]
    if npz and "frame" in d.files:                       # a live state knows its own frame
        t_min = [float(d["frame"]) * float(g["dt"]) * float((g.get("units") or {}).get("time_s", 60.0)) / 60.0]
    ks = list(range(rows)) if rows_sel is None else rows_sel
    sec_e, sec_n = _section_frame(X, occ, pid, c0)
    marked_cells = None
    if "blastula" in seeds:
        # THE SAGITTAL PLANE OF THE LIP: the animal-vegetal axis (z) and the direction from the centre to the bottle
        # cells (cell block `apical` > 0 or `tension` > 1; +x, the seed's dorsal side, when none is marked). The
        # cleavage default -- 45 deg off the first spindle -- has no spindle here and fell back to the y-z plane, 90
        # deg away from the dorsal lip.
        mk = np.zeros(0, bool)
        for blk, thr in (("cell__apical", 0.0), ("cell__tension", 1.0)):
            if blk in d.files:
                mk = d[blk][0, :, 0] > thr if not mk.size else (mk | (d[blk][0, :, 0] > thr))
        marked_cells = np.flatnonzero(mk) if mk.size else np.zeros(0, int)
        dor = np.array([1.0, 0.0, 0.0])
        if len(marked_cells):
            cm = np.mean([X[0][occ[0] & (pid[0] == c_)].mean(0) for c_ in marked_cells], 0) - c0
            if np.hypot(cm[0], cm[1]) > 1e-6:
                dor = np.array([cm[0], cm[1], 0.0]) / np.hypot(cm[0], cm[1])
        sec_e, sec_n = dor, np.array([-dor[1], dor[0], 0.0])
    a_y, law_y = _nucleus_law(spec)
    pl = pv.Plotter(off_screen=True, window_size=(width, height))
    if _LOOK.get("ink"):
        pl.set_background(_LOOK["bg"])                  # flat: the ink's silhouette test needs one ground colour
    else:
        pl.set_background("#557f8a", top="#8aaab2")
    pl.remove_all_lights()
    tilt = math.radians(25.0)
    view = np.array([0.0, -math.sin(tilt), math.cos(tilt)])
    up = np.array([0.0, math.cos(tilt), math.sin(tilt)])
    if view_override is not None:                       # elevation / azimuth in degrees, the live movie's convention
        el, az = (math.radians(a_) for a_ in view_override)
        view = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        up = np.array([0.0, 0.0, 1.0]) - view * view[2]
        up = up / max(np.linalg.norm(up), 1e-9)
    side = np.cross(up, view)
    _lk = _LOOK["light"]
    for pos, inten in ((c0 + 6 * R0 * (view + 0.55 * up - 0.45 * side), _lk[0]),  # key, upper left of the lens
                       (c0 + 6 * R0 * (view - 0.3 * up + 0.6 * side), _lk[1]),    # fill
                       (c0 + 6 * R0 * view, _lk[2])):                             # on-axis, the ring light
        L = pv.Light(position=tuple(pos), focal_point=tuple(c0), intensity=inten)
        L.positional = False
        pl.add_light(L)
    # FILL THE FRAME (Cedric, 2026-10-13: "too many movies have a lot of empty space"): every live point of every
    # row, projected on the screen's side and up axes; the camera aims at the middle of that box and backs off just
    # enough for the box to take 80 % of the frame height or 90 % of its width, whichever binds, plus the depth.
    va = 30.0
    allp = np.concatenate([X[k][occ[k]][:: max(1, int(occ[k].sum()) // 4000)] for k in range(rows)]) - c0
    if type_rank is not None:                             # a spheroid: only the points within 1.25 R0 of the centre set the frame
        allp = allp[np.linalg.norm(allp, axis=1) < 1.25 * R0]
    su, ss, sv = allp @ up, allp @ side, allp @ view
    mid = c0 + side * 0.5 * (ss.max() + ss.min()) + up * 0.5 * (su.max() + su.min())
    hu, hs = 0.5 * (su.max() - su.min()), 0.5 * (ss.max() - ss.min())
    tn_ = math.tan(math.radians(va / 2))
    # the cells take the band between 24 % (under the count plot) and 86 % (above the scale bar) of the frame height
    dist = max(hu / (0.62 * tn_), hs / (0.88 * tn_ * width / height)) + float(sv.max() - (sv.max() + sv.min()) / 2)
    mid = mid + up * (0.10 * dist * tn_)              # the band's centre lies 5 % of the height below the frame's
    pl.camera.position = tuple(mid + dist * view)
    pl.camera.focal_point = tuple(mid)
    pl.camera.up = tuple(up)
    pl.camera.view_angle = va
    # the SCALE BAR, drawn flat on the frame after each render (upright text, below the cells): pixels per world
    # unit at the focal plane; a 1/2/5 x 10^n micrometre length near a fifth of the visible width
    px_per_unit = height / (2.0 * dist * tn_)
    um = float((spec["general"].get("units") or {}).get("length_um", 0) or 0)
    bar = None
    if um > 0:
        tgt = 0.2 * width / px_per_unit * um
        p10 = 10 ** math.floor(math.log10(tgt))
        L_um = max(f * p10 for f in (1, 2, 5) if f * p10 <= tgt)
        bar = (L_um, L_um / um * px_per_unit, int(0.89 * height))   # under the cells' band (which ends at 86 %)
    if _LOOK["ssao"] and not nuclei:
        pl.enable_ssao(radius=0.35 * R0, bias=0.01, kernel_size=128, blur=True)
    if nuclei:
        pl.enable_depth_peeling(8)
    n_all = ([len(np.unique(pid[k][occ[k]])) for k in range(rows)] if _LOOK["plot"] and view_override is None
             else None)                                 # exp27: the cell count never changes, no panel
    if not _LOOK.get("ink"):
        # NOT IN THE INK LOOK: EGL/OSMesa VTK swaps FXAA for SSAA, which renders off the window -- the window's
        # z-buffer then reads all 1 and the ink found no edge at all (2026-10-13); the outlines cover the stair-steps
        pl.enable_anti_aliasing("fxaa")
    fps = max(1, round(len(ks) / seconds))
    import imageio.v2 as iio
    writer = iio.get_writer(out_mp4, fps=fps, quality=8, macro_block_size=16)
    frame = None
    for k in ks:
        for nm in list(pl.actors):
            if nm.startswith("cell_") or nm.startswith("nuc_"):
                pl.remove_actor(nm)
        live = occ[k]
        cells = np.unique(pid[k][live])
        batch = []
        for c in cells.tolist():
            sel = live & (pid[k] == c)
            P = X[k][sel]
            if len(P) < 16:
                continue
            surf = _cell_surface(P, v, h)
            if surf is None:
                continue
            if zone0 is not None and int(np.bincount(zone0[pid[k] == c]).argmax()) == 3:     # the blastocoel: fluid
                pl.add_mesh(surf, color="#cfe3ee", opacity=0.18 if nuclei else 0.35, smooth_shading=True,
                            name=f"cell_{c}")
                continue
            _, nn = cKDTree(P).query(surf.points, k=6)
            if _LOOK.get("ink") and type_rank is not None:  # spheroid: the identity's colour, a shade per cell
                from plexus.gui import flat_ink
                base = flat_ink.palette([5 if int(type_rank[c]) == 0 else (2 if int(type_rank[c]) >= 2 else 0)])[0]      # inner blue, outer red, a coat (basement membrane or mesenchyme) yellow
                col_c = np.clip(base * (0.88 + 0.24 * ((c * 37) % 11) / 10.0), 0.0, 1.0)
                surf["rgb"] = np.tile((col_c * 255).astype(np.uint8), (surf.n_points, 1))
            elif _LOOK.get("ink"):                          # the cell's own colour, keyed on its id
                from plexus.gui import flat_ink
                col_c = flat_ink.palette([c])[0]
                surf["rgb"] = np.tile((col_c * 255).astype(np.uint8), (surf.n_points, 1))
            elif mat is not None:                           # one colour per material (YlOrBr, pale -> brown)
                import matplotlib.pyplot as _plt
                col = np.array(_plt.get_cmap("YlOrBr")(0.15 + 0.85 * float(mat[k][c]))[:3]) * 255
                surf["rgb"] = np.tile(col.astype(np.uint8), (surf.n_points, 1))
            else:
                surf["rgb"] = _yolk_rgb(Y[k][sel][nn].mean(1), speck[sel][nn[:, 0]])     # granule: the nearest point's
            if not nuclei:                                  # ONE mesh for all the cells, added once after the loop: 600 add_mesh /
                batch.append(surf)                           # remove_actor calls cost 24 of a sphere render's 41 s in pyvista's actor lists
            else:
                pl.add_mesh(surf, scalars="rgb", rgb=True, smooth_shading=True, pbr=False,
                            specular=_LOOK["specular"], specular_power=_LOOK["specular_power"],
                            diffuse=_LOOK["diffuse"], ambient=_LOOK["ambient"], name=f"cell_{c}", show_scalar_bar=False,
                            opacity=0.35)
            if nuclei:
                # THE CUT'S NUCLEUS (Cedric, 2026-10-13): no nucleus is simulated -- `cell_divide[cleave]` places the
                # cut through the centre of the cell's yolk-free cytoplasm; drawn here as a dark sphere of 0.28 x the
                # cell's equivalent radius at exactly that point, inside a translucent cell
                yk = Y[k][sel]
                w = np.exp(-a_y * yk) if law_y == "exp" else np.clip(1.0 - a_y * yk, 0.0, None)
                nuc = (w[:, None] * P).sum(0) / max(float(w.sum()), 1e-12)
                r_n = 0.28 * (3.0 * len(P) * v / (4.0 * math.pi)) ** (1.0 / 3.0)
                if _LOOK.get("ink"):
                    # THE NUCLEUS IN ITS CELL'S OWN HUE, saturated 1.4x and at 0.6 of its value: a darker, stronger
                    # version of the cell around it, flat like the cells, outlined by the ink
                    import colorsys
                    hh, ss_, vv = colorsys.rgb_to_hsv(*col_c)
                    pl.add_mesh(pv.Sphere(radius=r_n, center=tuple(nuc), theta_resolution=20, phi_resolution=20),
                                color=colorsys.hsv_to_rgb(hh, min(1.0, 1.4 * ss_), 0.6 * vv), smooth_shading=True,
                                diffuse=_LOOK["diffuse"], ambient=_LOOK["ambient"], specular=0.0, name=f"nuc_{c}")
                else:
                    pl.add_mesh(pv.Sphere(radius=r_n, center=tuple(nuc), theta_resolution=20, phi_resolution=20),
                                color="#3a2414", smooth_shading=True, diffuse=0.8, ambient=0.3, name=f"nuc_{c}")
        if batch:
            merged = batch[0] if len(batch) == 1 else pv.merge(batch, merge_points=False)
            pl.add_mesh(merged, scalars="rgb", rgb=True, smooth_shading=True, pbr=False,
                        specular=_LOOK["specular"], specular_power=_LOOK["specular_power"],
                        diffuse=_LOOK["diffuse"], ambient=_LOOK["ambient"], name="cell_all", show_scalar_bar=False, opacity=1.0)
        for _nm in list(pl.actors):
            if _nm == "bm_nodes":
                pl.remove_actor(_nm)
        if bm_all is not None and _LOOK.get("ink"):
            nodes = bm_all[min(k, len(bm_all) - 1)]
            if slab_z is not None:
                nodes = nodes[np.abs(nodes[:, 2] - slab_z[0]) <= slab_z[1]]
            if len(nodes):
                from plexus.gui import flat_ink
                g_ = pv.PolyData(nodes.astype(np.float32)).glyph(scale=False, geom=pv.Icosphere(radius=0.075, nsub=1))
                pl.add_mesh(g_, color=tuple(float(v) for v in flat_ink.palette([2])[0]), smooth_shading=False, specular=0.0,
                            diffuse=_LOOK["diffuse"], ambient=_LOOK["ambient"], opacity=1.0 if slab_z is not None else 0.35, name="bm_nodes")
        if _LOOK.get("ink"):
            _rc = float(np.mean([(3.0 * int((live & (pid[k] == c_)).sum()) * v / (4.0 * math.pi)) ** (1.0 / 3.0)
                                 for c_ in cells.tolist()])) if len(cells) else 0.0
            frame = _ink_frame(pl, dist, va, height, nuclei, r_cell=_rc)
        else:
            frame = np.ascontiguousarray(pl.screenshot(return_img=True)[..., :3])
        if section:
            frame = _paste(frame, _section_image(X[k][live], Y[k][live], pid[k][live], c0, R0, speck[live], sec_e, sec_n,
                                                 out=section_px, ink=bool(_LOOK.get("ink")), marked=marked_cells),
                           colour=_LOOK["border"])
        if bar is not None:
            frame = _scale_bar(frame, bar, _LOOK["text"])
        frame = _clock(frame, f"{len(cells)} cell{'s' if len(cells) > 1 else ''}   t = {t_min[k]:.0f} min", _LOOK["text"])
        if curves_data is not None:
            tt = np.array(t_min)
            cd = curves_data
            tiles = [("cells", [(cd["cells"][1], "#e63946"), (cd["cells"][0], "#4d96ff")], (0, 1.1 * max(1, float(max(cd["cells"][0].max(), cd["cells"][1].max()))))),
                     ("on the inner layer", [(cd["grip"], "#e63946")], (0, 1.05)),
                     ("displacement (µm)", [(cd["disp"], "#e63946")], (0, 1.1 * max(1e-6, float(cd["disp"].max()))))]
            for j_, (yl_, ser_, lim_) in enumerate(tiles):
                tile = _curve_tile(tt, ser_, t_min[k], yl_, lim_, colour=_LOOK["text"], xlabel=("time (min)" if j_ == len(tiles) - 1 else None))
                frame = _blend(frame, tile, frame.shape[1] - tile.shape[1] - 16, 16 + j_ * (tile.shape[0] + 6))
        if n_all is not None:
            tile = _count_plot(np.array(t_min), np.array(n_all), t_min[k], colour=_LOOK["text"])
            frame = _blend(frame, tile, frame.shape[1] - tile.shape[1] - 24, 24)
        writer.append_data(frame)
    writer.close()
    iio.imwrite(out_png, frame)
    pl.close()
    return {"rows": len(ks), "fps": fps, "cells_last": int(len(cells)), "voxel": round(h, 4),
            "section": (("sagittal plane through the bottle cells (animal axis + dorsal direction), bottle cells dark red"
                         if marked_cells is not None else "vertical plane through the centre, 45 deg off the first spindle")
                        + f", bottom-left, {section_px} px") if section else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--why", required=True)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--stride", type=int, default=1, help="draw every n-th recorded row")
    ap.add_argument("--rows", default=None, help="comma-separated rows only (test stills; implies --no-archive)")
    ap.add_argument("--no-section", action="store_true", help="no cross-section inset")
    ap.add_argument("--section-px", type=int, default=230, help="the inset's side in pixels (Cedric 2026-10-13: smaller, 170)")
    # flat stays the default (Cedric, 2026-10-13: shown the ink look on the record movies, "not convinced -- keep [the
    # current look] for the main renderer"; the ink look is the watcher's 3D view only, bio_view.py)
    ap.add_argument("--look", default="flat", choices=sorted(LOOK),
                    help="flat (default), ink (per-cell colours with ink outlines, as the watcher's 3D view), or macro")
    ap.add_argument("--nuclei", action="store_true", help="translucent cells with each cut's nucleus drawn inside")
    ap.add_argument("--overlay", default=None, help="overlay colour (clock, plot, inset frame), e.g. white; default the look's")
    ap.add_argument("--out-dir", default="/tmp", help="where the mp4/png are written (a shared path on the cluster)")
    ap.add_argument("--caption", action="store_true", help="caption with the local VLM (loads a 12B model on a GPU)")
    ap.add_argument("--ring", nargs="?", const="annulus", default=None, choices=["annulus", "tube", "tube_in"],
                    help="a periodic ribbon (contact_finder period) drawn closed: annulus (in its own plane), tube "
                         "(rolled out of the plane, top layer outside) or tube_in (top layer inside)")
    ap.add_argument("--npz", default=None, help="render this one-row state file (a running job's live_state.npz) instead of the run's trajectory")
    ap.add_argument("--no-curves", action="store_true", help="a spheroid run: leave out the three curve panels on the right (cells, share of outer cells on the inner layer, displacement)")
    ap.add_argument("--cut", action="store_true", help="a spheroid cut in half: cells past the centre plane (y) are hidden")
    ap.add_argument("--cam", default=None, help="elevation,azimuth in degrees, replacing the spec's camera")
    ap.add_argument("--no-archive", action="store_true", help="render to /tmp only")
    ap.add_argument("--reuse", action="store_true", help="archive the /tmp render of an earlier --no-archive call")
    a = ap.parse_args()
    _LOOK.clear(); _LOOK.update(LOOK[a.look])
    if a.overlay:                                   # Cedric, 2026-10-13: white overlays on the nuclei renders
        from matplotlib.colors import to_rgb as _rgb
        _LOOK["text"] = a.overlay
        _LOOK["border"] = tuple(int(255 * c) for c in _rgb(a.overlay))
    tag = a.run.replace("/", "__") + ("" if a.look == "macro" else f"_{a.look}")
    tag = tag + ("_nuclei" if a.nuclei else "") + (f"_{a.ring}" if a.ring else "") + ("_cut" if a.cut else "")
    tmp_mp4, tmp_png = f"{a.out_dir}/{tag}_studio.mp4", f"{a.out_dir}/{tag}_studio.png"
    t0 = time.time()
    if a.reuse and os.path.exists(tmp_mp4) and os.path.exists(tmp_png):
        info = {"reused": tmp_mp4, "stride": a.stride}
    else:
        _rows = np.load(a.npz or os.path.join(ROOT, a.run, "trajectory.npz"))["pt__occ"].shape[0]
        if a.rows:
            a.no_archive = True
            sel = [int(x) for x in a.rows.split(",")]
        else:
            sel = None if a.stride <= 1 else sorted(set(range(0, _rows, a.stride)) | {_rows - 1})
        info = render(a.run, tmp_mp4, tmp_png, seconds=a.seconds, rows_sel=sel, section=not a.no_section, nuclei=a.nuclei,
                      section_px=a.section_px, ring=a.ring,
                      cam_override=None if not a.cam else tuple(float(v) for v in a.cam.split(",")), cut=a.cut, npz=a.npz, curves=not a.no_curves)
        info["wall_s"] = round(time.time() - t0)
    if a.no_archive:
        print({"mp4": tmp_mp4, "png": tmp_png, **info})
        sys.exit(0)
    import gui_drive as gd                               # the record's own numbering and paths
    i = gd._next_index()
    shutil.copy(tmp_png, gd._path("png", i))
    shutil.copy(tmp_mp4, gd._path("mp4", i))
    _sp = os.path.join(ROOT, a.run, "spec.yaml")
    shutil.copy(_sp if os.path.exists(_sp) else os.path.join(HERE, "..", "config", a.run + ".yaml"), gd._path("spec", i))
    with open(gd._path("why", i), "w") as f:
        f.write(f"time   {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"spec   {a.run} (macro-photo render of its recorded trajectory, tools/exp23_studio.py)\n")
        f.write(f"render {info}\n")
        f.write(f"why    {a.why}\n")
    gd.note(f"  studio {i:04d}  {a.run}  -- {a.why}")
    out = {"index": i, **info}
    if a.caption:
        out["caption"] = gd.caption(gd._path("mp4", i), index=i).get("caption", "")[:800]
    print(out)
