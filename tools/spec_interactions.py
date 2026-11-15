#!/usr/bin/env python
"""The material points of one cell and its neighbours, and the pairs of points that interact between them, at the first and last stored row of a run.

    PYTHONPATH=src python tools/spec_interactions.py <spec.yaml> --why <why.txt> -o out.png [--cell SLOT] [--rows 0,-1]

The companion of `tools/spec_hierarchy.py`: there a cell is a circle holding its points as a number; here the points are drawn, at their real
positions (the run's `pt__pos`) projected on the plane in which the cell and its neighbours spread most (their first two principal
axes), each cell's points inside the circle round its centroid, coloured as the hierarchy colours it (`plotting.colors`).

The lines are the pairs the spec's operators act on, one colour per operator, named under the figure by the registry's own title:

    contact_finder   each live point's PARTNER: the nearest point of ANOTHER cell among its `k_nn` nearest within `bond_range + skin`
                     (the rule of `JunctionContact.forward`, repeated here on the stored positions with the spec's own parameters).
                     The pair list is what the operators below read; a junction row's contact area is a count of these pairs.
    bond_springs     the TETHER on those pairs: F = kappa * rho_J * (r - r_eq)+ along the pair, for the pairs closer than `bond_range`
                     whose junction has bound density rho_J > 0 (line width follows the tension; the viscous term needs the velocities,
                     which a run does not store, so it is not drawn).

`pair_potential[cell_contact]` (the overlap and the contact tension) acts through the lattice nodes the cells deposit on, not through point
pairs, so it has no line here. A pair's junction row is the largest-area row of its two cells (the face class `fg` is not stored per pair).

    PYTHONPATH=src python tools/spec_interactions.py <spec.yaml> --why <why.txt> --movie interface|cells --cell SLOT --neighbour SLOT -o pair.mp4

`--movie` is rendered in VTK on a black background (the strain movies' frame), in the 4D view's iso-surface look: each cell a flat, inked body (the
contour of its points' density), cut through the middle of the pair so that the half nearer the camera is gone, each pair a tube between its two
points -- yellow for a tether, white and thinner for a contact pair without one; a tether's radius is proportional to the bound density of its
junction. The matplotlib figure (`-o out.png`) is unchanged.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys

import yaml

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BG, FG, DIM = "#0f1319", "#dfe5ea", "#7d8a96"
TUBE = 0.25                                                       # a tether tube's radius in median point spacings per unit of the junction's bound density
CONTACT, TETHER = "#9aa5b1", "#ffb347"


def _hier():
    sp = importlib.util.spec_from_file_location("spec_hierarchy", os.path.join(ROOT, "tools", "spec_hierarchy.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def _op(raw: dict, name: str) -> dict | None:
    return next((o for o in (raw.get("operators") or []) if isinstance(o, dict) and o.get("op") == name), None)


def params_of(raw: dict) -> dict:
    """The parameters of the two operators with lines, with the registry's defaults (spacing 0.125; skin half a spacing; k_nn 16; the
    tether's rest length one spacing and reach two spacings)."""
    cf, bs = _op(raw, "contact_finder") or {}, _op(raw, "bond_springs") or {}
    h = float(cf.get("spacing", 0.125))
    hb = float(bs.get("spacing", 0.125))
    return {"h": h, "reach": float(cf.get("bond_range", 2.0 * h)), "skin": float(cf.get("skin", 0.5 * h)), "k_nn": int(cf.get("k_nn", 16)),
            "kappa": float(bs.get("bond_stiffness", 1.0)), "r_eq": float(bs.get("bond_rest_length", hb)),
            "bond_reach": float(bs.get("bond_range", 2.0 * hb)), "zeta": float(bs.get("bond_damping", 0.0)),
            "blocks": [k for k in bs if k.endswith("_block")], "has_tether": bool(bs), "has_finder": bool(cf)}


def read_rows(run_dir: str, rows) -> list:
    """Per stored row: points (slot in the point buffer, position, cell), cells (type, live), junction rows (cell a, cell b, rho, area)."""
    import numpy as np
    z = np.load(os.path.join(run_dir, "trajectory.npz"))
    files = set(z.files)
    n = z["cell__occ"].shape[0]
    names = [str(x) for x in z["cell__type_names"]] if "cell__type_names" in files else None
    pos_all, pid_all, pocc_all = z["pt__pos"], z["pt__pid"], z["pt__occ"]
    cocc = z["cell__occ"]
    tt_all = z["cell__node_type_t"] if "cell__node_type_t" in files else None
    st_ = z["cell__node_type"] if "cell__node_type" in files else None
    out = []
    for r in rows:
        k = r % n
        m = pocc_all[k]
        tt = tt_all[k] if tt_all is not None else (st_ if st_ is not None else np.zeros(cocc.shape[1], int))
        d = {"row": k, "rows": n, "slot": np.flatnonzero(m), "x": pos_all[k][m].astype(np.float64), "cid": np.rint(pid_all[k].reshape(-1)[m]).astype(int),
             "live": cocc[k], "type": [(names[int(t)] if names and int(t) < len(names) else str(int(t))) for t in tt]}
        if "pt__skin" in files:                                                     # a division rig flags the points its division woke as the skin of the new faces
            d["skin"] = z["pt__skin"][k].reshape(-1)[m] > 0.5
        jm = z["junction__occ"][k]
        d["ja"] = np.rint(z["junction__ca"][k].reshape(-1)[jm]).astype(int)
        d["jb"] = np.rint(z["junction__cb"][k].reshape(-1)[jm]).astype(int)
        d["jrho"] = z["junction__rho"][k].reshape(-1)[jm]
        d["jarea"] = z["junction__area"][k].reshape(-1)[jm]
        out.append(d)
    return out


def pairs_of(d: dict, p: dict) -> dict:
    """{'i','j': point indices, 'r': distances, 'rho': bound density of the pair's junction, 'tension': tether tension} for the pair list
    of one row (see the module docstring for the rule)."""
    import numpy as np
    from scipy.spatial import cKDTree
    X, cid = d["x"], d["cid"]
    tree = cKDTree(X)
    dd, nn = tree.query(X, k=p["k_nn"], distance_upper_bound=p["reach"] + p["skin"])
    ok = np.isfinite(dd)
    nn_c = np.where(ok, nn, 0)
    other = ok & (cid[nn_c] != cid[:, None])
    has = other.any(1)
    first = np.argmax(other, axis=1)
    i = np.flatnonzero(has)
    j = nn_c[i, first[i]]
    r = np.linalg.norm(X[j] - X[i], axis=1)
    best = {}                                                       # a pair's junction row: of its two cells, the one of largest area
    for a, b, rho, ar in zip(d["ja"], d["jb"], d["jrho"], d["jarea"]):
        key = (min(a, b), max(a, b))
        if key not in best or ar > best[key][1]:
            best[key] = (float(rho), float(ar))
    rho = np.array([best.get((min(cid[a], cid[b]), max(cid[a], cid[b])), (0.0, 0.0))[0] for a, b in zip(i, j)])
    tension = p["kappa"] * rho * np.clip(r - p["r_eq"], 0.0, None) * (r < p["bond_reach"])
    return {"i": i, "j": j, "r": r, "rho": rho, "tension": tension, "bonded": (rho > 0) & (r < p["bond_reach"])}


def pick_focus(d_last: dict, d_first: dict) -> int:
    """The cell with the most contacts among the cells that are not of the first type of the cell set (the `inner` core), alive at both rows."""
    import numpy as np
    count = {}
    for a, b, ar in zip(d_last["ja"], d_last["jb"], d_last["jarea"]):
        if ar > 0:
            count[a] = count.get(a, 0) + 1
            count[b] = count.get(b, 0) + 1
    cand = [c for c in count if d_last["live"][c] and d_first["live"][c] and d_last["type"][c] != "inner"]
    return max(cand, key=lambda c: count[c]) if cand else int(np.flatnonzero(d_last["live"])[0])


def partners_of(d: dict, P: dict, focus: int) -> dict:
    """{neighbour cell: number of pairs it has with the focus cell} in one row."""
    import numpy as np
    cid = d["cid"]
    ci, cj = cid[P["i"]], cid[P["j"]]
    out = {}
    for a, b in zip(ci, cj):
        if a == focus and b != focus:
            out[int(b)] = out.get(int(b), 0) + 1
        elif b == focus and a != focus:
            out[int(a)] = out.get(int(a), 0) + 1
    return out


def geometry(d: dict, p: dict, focus: int, other: int, vprev=None, alone=None) -> dict:
    """The two cells of one row in the side view: positions Y [points, 2] (the line through the two centroids horizontal, from `focus` to
    `other`; the direction the points spread in beyond that line vertical, its sign kept from the previous row `vprev` so a movie does
    not flip), the pairs between the two cells, the points of each cell and the points that belong to a pair. A row in which `other` does not exist yet (a cell before its division) is drawn in the frame `alone` = (u, v) of the
    pair it becomes, about the focus cell's own centroid."""
    import numpy as np
    P = pairs_of(d, p)
    X, cid = d["x"], d["cid"]
    ci, cj = cid[P["i"]], cid[P["j"]]
    sel = np.flatnonzero(((ci == focus) & (cj == other)) | ((ci == other) & (cj == focus)))
    mf, mo = np.flatnonzero(cid == focus), np.flatnonzero(cid == other)
    if mo.size == 0:
        u, v = alone; mid = X[mf].mean(0)
    else:
        cF, cO = X[mf].mean(0), X[mo].mean(0)
        u = (cO - cF) / np.linalg.norm(cO - cF)
        mid = 0.5 * (cF + cO)
        both = X[np.concatenate([mf, mo])] - mid
        perp = both - np.outer(both @ u, u)
        v = np.linalg.svd(perp - perp.mean(0), full_matrices=False)[2][0]
        if vprev is not None and float(v @ vprev) < 0:
            v = -v
    Y = np.stack([(X - mid) @ u, (X - mid) @ v], 1)
    return {"P": P, "sel": sel, "mf": mf, "mo": mo, "Y": Y, "W": (X - mid) @ np.cross(u, v), "u": u, "v": v,
            "inpair": set(P["i"][sel].tolist()) | set(P["j"][sel].tolist())}


def cell_rgb(colors: dict, t: str):
    """A cell's colour from `plotting.colors` by its type name (the type, else inner / outer, else outer_0), else a grey-blue."""
    from matplotlib.colors import to_rgb
    for k in (t, "inner" if t == "inner" else "outer", "outer_0"):
        if k in colors:
            return to_rgb(colors[k])
    return (0.62, 0.68, 0.74)


def panel(ax, d: dict, p: dict, focus: int, other: int, colors: dict, when: str, time: str, zoom: bool = False, geo: dict | None = None,
          limits=None):
    """One row: two touching cells drawn as hulls holding their material points at their real positions, seen from the side (see
    `geometry`). The points that belong to a pair are bright; the lines are the pairs. `zoom` frames the interface alone (the box round
    the points of the pairs) instead of the two cells; `limits` ((x0, y0), (x1, y1)) fixes the frame (a movie). The second cell is drawn in
    a paler tint of its colour, so the points of each can be told apart when the two have the same type."""
    import numpy as np
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle, Polygon
    from scipy.spatial import ConvexHull
    ax.set_facecolor(BG)
    ax.set_axis_off()
    g = geo or geometry(d, p, focus, other)
    P, sel, mf, mo, Y, inpair = g["P"], g["sel"], g["mf"], g["mo"], g["Y"], g["inpair"]
    cid, h = d["cid"], p["h"]
    from scipy.spatial import cKDTree
    # a dot is 0.30 / 0.42 x the operator's nominal spacing `h` in radius, shrunk so that a quiet dot (diameter 0.6 h) is no wider than the
    # cell's own median nearest-neighbour distance between points (the 4000-point cells, whose dots would otherwise merge into one disc)
    dot = min(1.0, float(np.median(cKDTree(d["x"][g["mf"]]).query(d["x"][g["mf"]], k=2)[0][:, 1])) / (0.6 * h))

    def colour(c):
        return cell_rgb(colors, d["type"][c])

    for c, m in ((focus, mf), (other, mo)):
        col = np.array(colour(c))
        if c == other:
            col = 0.5 * col + 0.5
        col = tuple(col)
        hull = Y[m][ConvexHull(Y[m]).vertices]
        ax.add_patch(Polygon(hull, closed=True, facecolor=col, edgecolor=col, lw=1.0, alpha=0.18))
        quiet = [i_ for i_ in m if i_ not in inpair]
        loud = [i_ for i_ in m if i_ in inpair]
        ax.add_collection(PatchCollection([Circle(Y[i_], 0.30 * h * dot) for i_ in quiet], facecolor=col, edgecolor="none", alpha=0.45))
        ax.add_collection(PatchCollection([Circle(Y[i_], 0.42 * h * dot) for i_ in loud], facecolor=col, edgecolor="none", alpha=1.0, zorder=3))
        if not zoom:
            top = hull[np.argmax(hull[:, 1])]
            ax.text(top[0], top[1] + 0.25 * h, f"cell {c}" + (f" ({d['type'][c]})" if d["type"][c] else "") + f"  {len(m)} points", color=col,
                    fontsize=8, ha="center", va="bottom")
    tmax = max(float(P["tension"][sel].max()) if sel.size else 0.0, 1e-9)
    nt = 0
    for q in sel:
        a, b = Y[P["i"][q]], Y[P["j"][q]]
        if P["bonded"][q]:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=TETHER, lw=0.6 + 2.2 * P["tension"][q] / tmax, alpha=0.9, zorder=5, solid_capstyle="round")
            nt += 1
        else:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=CONTACT, lw=0.7, alpha=0.8, zorder=4)
    if limits is not None:
        (x0, y0), (x1, y1) = limits
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    elif zoom and sel.size:
        pts = Y[np.array(sorted(inpair))]
        lo, hi = pts.min(0) - 0.8 * h, pts.max(0) + 0.8 * h
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
    else:
        allc = np.concatenate([Y[mf], Y[mo]])
        lo, hi = allc.min(0) - h, allc.max(0) + h
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1] + 0.26 * (hi[1] - lo[1]))
    ax.set_aspect("equal")
    r = P["r"][sel]
    touch = partners_of(d, P, focus)
    if not zoom:
        ax.text(0.01, 0.995, f"{when}  {time}\ncell {focus} touches {len(touch)} cells; with cell {other}: {len(sel)} contact pairs, {nt} tethered"
                + (f"\npair distance {r.min():.2f} to {r.max():.2f} (tether rest length {p['r_eq']:g}, reach {p['bond_reach']:g})" if len(sel) else ""),
                color=FG, fontsize=8.5, ha="left", va="top", transform=ax.transAxes, linespacing=1.3)
    else:
        ax.text(0.01, 0.995, "the interface, enlarged", color=DIM, fontsize=7.5, ha="left", va="top", transform=ax.transAxes)
    return {"focus": int(focus), "other": int(other), "pairs": int(len(sel)), "tethered": nt, "neighbours": len(touch)}


def figure_png(raw: dict, run_dir: str, rows=(0, -1), focus: int | None = None, other: int | None = None, dpi: int = 100):
    import io
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    H = _hier()
    p = params_of(raw)
    if not p["has_finder"]:
        raise ValueError("this spec has no contact_finder: no pair list to draw")
    if p["blocks"]:                         # per-cell tether values (exp29 rosette arrays) are not read: the scalars stand in for them
        print(f"[spec_interactions] bond_springs reads per-cell blocks {p['blocks']}; the scalars bond_stiffness / bond_rest_length / bond_range are used", flush=True)
    ds = read_rows(run_dir, rows)
    if focus is None:
        focus = pick_focus(ds[-1], ds[0])
    colors = (raw.get("plotting") or {}).get("colors") or {}
    if other is None:                       # the neighbour with the most pairs at the last row, among those the focus also touches at the first
        last, first = partners_of(ds[-1], pairs_of(ds[-1], p), focus), partners_of(ds[0], pairs_of(ds[0], p), focus)
        both = {c: n for c, n in last.items() if c in first} or last
        other = max(both, key=both.get)
    fig = Figure(figsize=(13.0, 10.2), facecolor=BG)
    FigureCanvasAgg(fig)
    info = []
    for i, (d, when) in enumerate(zip(ds, ("start", "end"))):
        t = H.time_label(raw, d["row"], d["rows"])
        info.append(panel(fig.add_axes([0.005 + 0.5 * i, 0.395, 0.49, 0.6]), d, p, focus, other, colors, when, t))
        panel(fig.add_axes([0.005 + 0.5 * i, 0.105, 0.49, 0.28]), d, p, focus, other, colors, when, t, zoom=True)
    key = fig.add_axes([0.005, 0.005, 0.99, 0.09])
    key.set_axis_off()
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    titles = {}
    try:
        for r in H._summary().read_operators(raw):
            titles.setdefault(r["name"], r["title"])
    except Exception:                                               # noqa: BLE001
        pass
    key.plot([0.01, 0.04], [0.82, 0.82], color=CONTACT, lw=1.2)
    key.text(0.05, 0.82, f"contact_finder ({titles.get('contact_finder', '')}): each point's partner is the nearest point of another cell within {p['reach']:g} + {p['skin']:g}\n"
             f"(bond_range + skin) among its {p['k_nn']} nearest. Grey: a pair the tether does not act on (no bound density, or farther than {p['bond_reach']:g}).",
             color=FG, fontsize=7.5, va="center", linespacing=1.3)
    key.plot([0.01, 0.04], [0.30, 0.30], color=TETHER, lw=2.2)
    key.text(0.05, 0.30, f"bond_springs ({titles.get('bond_springs', '')}): tether kappa * rho * (r - r_eq)+ on a pair closer than {p['bond_reach']:g} whose junction has bound density rho > 0;\n"
             f"kappa {p['kappa']:g}, r_eq {p['r_eq']:g}; wider = more tension. pair_potential[cell_contact] acts through lattice nodes, not point pairs, so it has no line.",
             color=FG, fontsize=7.5, va="center", linespacing=1.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=BG)
    return buf.getvalue(), info


BODY = dict(dx=0.8, sigma=0.9, iso_frac=0.65)                   # a body's voxel in median point spacings, the blur in voxels, the contour in means of the occupied density


def body_surface(P, spacing: float, dx: float = BODY["dx"], sigma: float = BODY["sigma"], iso_frac: float = BODY["iso_frac"], w=None):
    """One cell's body in the 4D view's iso-surface look: the contour of its points' density on voxels of side `dx` x `spacing`, blurred over
    `sigma` voxels, at `iso_frac` of the mean density of the occupied voxels (`morph.reconstruct_contour`, the contour the watcher draws; its
    defaults of 0.95 voxel and 0.3 swell a body past its outermost points, which closes the gap between two cells); None when it has no surface."""
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from plexus.morph import reconstruct_contour
    d = dx * spacing
    lo = P.min(0) - 3.0 * d
    box = float((P.max(0) - lo).max()) + 3.0 * d
    surf = reconstruct_contour(P - lo, box, int(math.ceil(box / d)), sigma=sigma, iso_frac=iso_frac, w=w)
    return None if surf is None else surf.translate(lo, inplace=False)


def point_volumes(raw: dict, n: int, skin=None):
    """The volume weight of each of one cell's `n` live points when the cell is a coarse bulk with a dense skin (the exp38 rig: `cell_grid` lays `live_per_cell` points at `point_spacing`, `face_layers` wakes the skin
    at `pitch` after them), the bulk's spacing for the body's voxels -> (weights [n], the bulk spacing); (None, None) for any other rig. In the division rig (`cell_divide[cleave_faces]`) the skin is woken at the division in slots
    that follow no order, so the row's own flag `skin` [n] (the point block `pt__skin`) tells it from the cloud: the cloud's points share the box of the mother's seed, a skin point has the volume pitch^3."""
    import numpy as np
    seeds = [x for x in (raw.get("seed") or []) if x.get("op") == "seed_positions"]
    fl = next((x for x in seeds if x.get("model") == "face_layers"), None); cg = next((x for x in seeds if x.get("model") == "cell_grid"), None)
    dv = next((o for o in (raw.get("operators") or []) if o.get("op") == "cell_divide" and o.get("model") == "cleave_faces"), None)
    if dv is not None and cg and "live_per_cell" in cg and skin is not None:
        vb = float(np.prod(cg["box"])) / int(cg["live_per_cell"]); return np.where(skin, float(dv["pitch"]) ** 3, vb), vb ** (1.0 / 3.0)
    if not fl or not cg or "live_per_cell" not in cg:
        return None, None
    nb = int(cg["live_per_cell"]); vb = float(cg["point_spacing"]) ** 3
    if cg.get("cloud"):                                                 # a cloud body: the points share the volume of the cube minus the slab of its faces that touch a neighbour (one slab: the pair's)
        side = float(cg.get("side", 1.0)); vb = side * side * (side - float(cg.get("cloud_clear", 0.0))) / nb
    w = np.full(n, float(fl["pitch"]) ** 3); w[:nb] = vb
    return w, vb ** (1.0 / 3.0)


def _bar_png(width: int, height: int, px_per_um: float, corner: str = "bl", font_px: int = 20, margin: int = 14):
    """A transparent `width` x `height` picture holding a scale bar in a corner, `px_per_um` pixels per micrometre, as long as the shortest round length at least as wide
    as its own label (flat_ink.round_length), the label under it; returns (RGBA array, the bar's length in micrometres)."""
    import matplotlib
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from plexus.gui import flat_ink
    im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"), font_px)

    def lab(L):
        return f"{L / 1000:g} mm" if L >= 1000 else f"{L:g} \u00b5m" if L >= 1 else f"{L * 1000:g} nm"          # a bar shorter than a micrometre is labelled in nanometres: 100 nm, not 0.1 \u00b5m
    L = flat_ink.round_length(px_per_um, lambda x: dr.textlength(lab(x), font=font))
    Lp = int(round(L * px_per_um))
    y = height - margin - font_px - 10
    x0 = margin if corner == "bl" else width - margin - max(Lp, int(dr.textlength(lab(L), font=font)))
    dr.rectangle([x0, y, x0 + Lp, y + max(3, font_px // 7)], fill=(255, 255, 255, 255))      # the bar's thickness follows the label: 3 px up to a 27-px font, then a seventh of the font size
    dr.text((x0, y + 8), lab(L), fill=(255, 255, 255, 255), font=font)
    return np.asarray(im), L


def pair_rows(run_dir: str, p: dict, focus: int, other: int, before: bool = False):
    """(the stored rows a movie draws, the index of the first row in which both cells live, the frame (u, v) of the pair for the rows before it). Rows in which `other` does not exist are left out, with `before` they are drawn
    with the focus cell alone (a cell before its division)."""
    import numpy as np
    n = int(np.load(os.path.join(run_dir, "trajectory.npz"))["cell__occ"].shape[0])
    ds = [d for d in read_rows(run_dir, range(n)) if d["live"][focus] and (before or d["live"][other])]
    i0 = next(i for i, d in enumerate(ds) if d["live"][other]); g0 = geometry(ds[i0], p, focus, other) if i0 else None
    return ds, i0, ((g0["u"], g0["v"]) if g0 else None)


def frame_scale(raw: dict, run_dir: str, focus: int, other: int, zoom: bool = False, height: int = 900, before: bool = False, tilt: float = 0.0) -> float:
    """Micrometres per pixel of the frame `movie` draws for this pair (the union of the pair's boxes over every row, fitted to `height` pixels), without drawing it."""
    import numpy as np
    p = params_of(raw)
    ds, _, alone = pair_rows(run_dir, p, focus, other, before)
    vprev, lo, hi = None, None, None
    for d in ds:
        g = geometry(d, p, focus, other, vprev, alone)
        vprev = g["v"]
        if tilt:                                                        # the side view turned about the axis between the cells by `tilt` degrees: rows of a regular skin that a view along it stacks exactly now fall at different heights
            c_, s_ = math.cos(math.radians(tilt)), math.sin(math.radians(tilt)); y0, w0 = g["Y"][:, 1].copy(), g["W"].copy()
            g["Y"] = np.column_stack([g["Y"][:, 0], y0 * c_ + w0 * s_]); g["W"] = -y0 * s_ + w0 * c_
        if zoom and not len(g["mo"]):                                                   # a cell alone (before its division) does not frame the close-up of the pair it becomes
            continue
        Y = g["Y"]
        pts = Y[np.array(sorted(g["inpair"]))] if (zoom and g["inpair"]) else np.concatenate([Y[g["mf"]], Y[g["mo"]]])
        l, u_ = pts.min(0) - (0.8 if zoom else 1.0) * p["h"], pts.max(0) + (0.8 if zoom else 1.0) * p["h"]
        lo, hi = (l, u_) if lo is None else (np.minimum(lo, l), np.maximum(hi, u_))
    Hh = height // 2 * 2
    W = int(min(max(Hh * (hi[0] - lo[0]) / (hi[1] - lo[1]), 0.62 * Hh), 1.55 * Hh)) // 2 * 2
    half_h = max(0.5 * (hi[1] - lo[1]), 0.5 * (hi[0] - lo[0]) * Hh / W)
    um = float(((raw.get("general") or {}).get("units") or {}).get("length_um", 0) or 0)
    return 2.0 * half_h / Hh * um


def movie(raw: dict, run_dir: str, out: str, focus: int, other: int, zoom: bool = False, fps: float = 12.0, height: int = 900, graph: bool = False, tag: str = "", body: dict | None = None, contacts: bool = True, window: float | None = None, slab: float | None = None, node_scale: float = 1.0, tilt: float = 0.0, before: bool = False, label_only: bool = False, follow: bool = False, row_stride: int | None = None) -> str:
    """The pair of cells over every stored row of the run, as a VTK mp4 on a black background: each cell's body as the 4D view's flat, inked
    iso-surface (the focus cell in its colour, the other in a paler tint), cut through the plane of the two centroids' mid-point so that the
    half nearer the camera is gone and the pairs inside are seen, each pair a tube between its two points (yellow: the junction tether of
    bond_springs; white and thinner: a contact pair without a tether). `zoom` frames the interface alone (the box round the points of the
    pairs) instead of the two cells. The frame is fixed from every row (the union of the boxes), the side view keeps its orientation from row
    to row, and rows in which either cell does not exist are left out.

    `tag` replaces the title and the legend by that one text in large type (a sweep's value in a montage tile). `graph` draws the same frame as the graph Plexus holds at the level of material points, flat in the plane of the side view: a node per
    point (the focus cell dark, the other pale, a point that belongs to a pair full size and the rest small and dim) and an edge per pair, a
    yellow line for a tether, so that the two movies, played side by side, are one graph and its render. With `before` the rows in which the other cell does not exist yet are drawn too, the focus cell alone in the frame of the
    pair it becomes."""
    import subprocess
    import imageio_ffmpeg
    import numpy as np
    from scipy.spatial import cKDTree
    SI = importlib.util.spec_from_file_location("spec_inside", os.path.join(ROOT, "tools", "spec_inside.py"))
    SI_ = importlib.util.module_from_spec(SI)
    SI.loader.exec_module(SI_)
    pv = SI_._setup_vtk()
    ink = SI_._flat_ink()
    H = _hier()
    p = params_of(raw)
    colors = (raw.get("plotting") or {}).get("colors") or {}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    ds, i0, alone = pair_rows(run_dir, p, focus, other, before)
    geos, vprev, lo, hi = [], None, None, None
    for d in ds:
        g = geometry(d, p, focus, other, vprev, alone)
        vprev = g["v"]
        if tilt:                                                        # the side view turned about the axis between the cells by `tilt` degrees: rows of a regular skin that a view along it stacks exactly now fall at different heights (this block stood in frame_scale only until 2026-11-07: movie() never turned the view)
            c_, s_ = math.cos(math.radians(tilt)), math.sin(math.radians(tilt)); y0, w0 = g["Y"][:, 1].copy(), g["W"].copy()
            g["Y"] = np.column_stack([g["Y"][:, 0], y0 * c_ + w0 * s_]); g["W"] = -y0 * s_ + w0 * c_
        if not contacts:                                                # tethers only: a contact pair without a tether is not drawn, nor are the points that belong to no tether drawn as paired (Cedric, 2026-11-07)
            g["sel"] = g["sel"][g["P"]["bonded"][g["sel"]]]
            g["inpair"] = set(g["P"]["i"][g["sel"]].tolist()) | set(g["P"]["j"][g["sel"]].tolist())
        geos.append(g)
        if zoom and not len(g["mo"]):                                                   # a cell alone (before its division) does not frame the close-up of the pair it becomes
            continue
        Y = g["Y"]
        pts = Y[np.array(sorted(g["inpair"]))] if (zoom and g["inpair"]) else np.concatenate([Y[g["mf"]], Y[g["mo"]]])
        l, u_ = pts.min(0) - (0.8 if zoom else 1.0) * p["h"], pts.max(0) + (0.8 if zoom else 1.0) * p["h"]
        lo, hi = (l, u_) if lo is None else (np.minimum(lo, l), np.maximum(hi, u_))
    um_w0 = float(((raw.get("general") or {}).get("units") or {}).get("length_um", 0) or 0)
    if window is not None:                                                                  # a fixed square field of `window` um about the centre of the pair's points at the first row (a close-up of the junction)
        c0 = geos[i0]["Y"][np.array(sorted(geos[i0]["inpair"]))].mean(0); half = 0.5 * float(window) / um_w0
        lo, hi = c0 - half, c0 + half
    half_slab = None if slab is None else 0.5 * float(slab) / um_w0                          # `slab` um: only the points within that thickness of the section plane are drawn (a thin section of the junction)
    x0 = ds[i0]["x"][geos[i0]["mf"]]
    spacing = float(np.median(cKDTree(x0).query(x0, k=2)[0][:, 1]))                         # the focus cell's median nearest-neighbour distance at the first row
    Hh = height // 2 * 2
    W = int(min(max(Hh * (hi[0] - lo[0]) / (hi[1] - lo[1]), 0.62 * Hh), 1.55 * Hh)) // 2 * 2
    half_h = max(0.5 * (hi[1] - lo[1]), 0.5 * (hi[0] - lo[0]) * Hh / W)                        # the frame fits the box whichever way it is wider
    dist = 8.0 * half_h                                                                     # a long lens: nearly the flat side view of the 2D figures
    pl = pv.Plotter(off_screen=True, window_size=(W, Hh), border=False)
    pl.set_background(SI_.MOVIE_BG)
    pl.camera.position, pl.camera.focal_point, pl.camera.up = (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), dist), (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.0), (0, 1, 0)
    pl.camera.view_angle = 2.0 * math.degrees(math.atan(half_h / dist))
    wpp = 2.0 * half_h / Hh                                                                 # world length of one pixel at the focal plane
    zs = min(1.0, half_h / 0.5)                                                             # the nodes of a graph sit +-0.02 in front of and behind its edges to keep them apart: in a close-up that offset is a large part of the field and the lens' perspective shifts the nodes off their edges, so it scales with the field
    um_w = float(((raw.get("general") or {}).get("units") or {}).get("length_um", 0) or 0)
    bar = _bar_png(W, Hh, 1.0 / (wpp * um_w), corner="br", font_px=22, margin=22)[0] if (um_w > 0 and not tag) else None      # a scale bar on a whole-frame movie; a montage tile gets its own
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}",
                             "-framerate", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out], stdin=subprocess.PIPE)
    legend = ("node: MPM point (full size: in a pair)\nyellow edge: tether" if graph else "yellow tube: tether, radius ~ bound density") + ("\nwhite tube: contact pair, no tether" if any((~g["P"]["bonded"][g["sel"]]).any() for g in geos) else "")
    for d, g in zip(ds, geos):
        P3 = np.column_stack([g["Y"], g["W"]])
        if follow and window is not None and g["inpair"]:                                    # the field follows the junction: centred on this row's pair points, the camera's distance and view angle unchanged
            c_ = g["Y"][np.array(sorted(g["inpair"]))].mean(0)
            pl.camera.position, pl.camera.focal_point = (c_[0], c_[1], dist), (c_[0], c_[1], 0.0)
        if graph:
            bgc = np.array([int(SI_.MOVIE_BG[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
            Pg = np.column_stack([g["Y"], np.zeros(len(g["Y"]))])
            for k, (c, m) in enumerate(((focus, g["mf"]), (other, g["mo"]))):
                col = np.array(cell_rgb(colors, d["type"][c]))
                col = col if k == 0 else 0.5 * col + 0.5
                paired = np.array([i_ in g["inpair"] for i_ in m], bool)
                inslab = np.ones(len(m), bool) if half_slab is None else np.abs(g["W"][m]) < half_slab
                for kind, keep, shade, diam, depth_ in (("quiet", ~paired & inslab, 0.5 * col + 0.5 * bgc, 0.22, -0.02), ("pair", paired & inslab, col, 0.45, 0.02)):
                    X_ = Pg[m][keep] + np.array([0.0, 0.0, depth_ * zs])
                    if len(X_) == 0:
                        pl.remove_actor(f"node{k}{kind}")
                        continue
                    cloud = pv.PolyData(np.ascontiguousarray(X_, np.float32))
                    cloud["rgb"] = np.tile((shade * 255).astype(np.uint8), (len(X_), 1))
                    pl.add_mesh(cloud, name=f"node{k}{kind}", scalars="rgb", rgb=True, style="points", point_size=max(3.0, node_scale * diam * spacing / wpp), render_points_as_spheres=True,
                                lighting=False, show_scalar_bar=False, reset_camera=False)
            P3 = Pg
        else:
            for k, (c, m) in enumerate(((focus, g["mf"]), (other, g["mo"]))):
                col = np.array(cell_rgb(colors, d["type"][c]))
                if len(m) == 0:                                                                  # the other cell does not exist yet
                    pl.remove_actor(f"body{k}")
                    continue
                wts, sp_b = point_volumes(raw, len(m), d["skin"][m] if "skin" in d else None)
                surf = body_surface(P3[m], spacing, **(body or {})) if wts is None else body_surface(P3[m], sp_b, w=wts, **(body or {}))       # a coarse bulk with a dense skin: voxels at the bulk's spacing, each point weighted by its volume
                surf = surf.clip(normal=(0, 0, 1), origin=(0, 0, 0), invert=True) if surf is not None else None   # keep the half away from the camera
                if surf is not None and wts is not None and len(g["mo"] if k == 0 else g["mf"]):             # a dense skin: the density contour swells ~1 um past the points, past a 40-70 nm gap, so each body is cut at the plane of its own front layer
                    m_o = g["mo"] if k == 0 else g["mf"]; u_ = P3[m_o].mean(0) - P3[m].mean(0); u_ = u_ / np.linalg.norm(u_)
                    front = P3[m].mean(0) + u_ * np.percentile((P3[m] - P3[m].mean(0)) @ u_, 99.5)
                    surf = surf.clip(normal=tuple(u_), origin=tuple(front), invert=True)
                    surf = surf if surf.n_points else None
                if surf is None or surf.n_points == 0:
                    pl.remove_actor(f"body{k}")
                    continue
                pl.add_mesh(surf, color=tuple(col if k == 0 else 0.5 * col + 0.5), name=f"body{k}", smooth_shading=True, ambient=ink.FLAT["ambient"],
                            diffuse=ink.FLAT["diffuse"], specular=ink.FLAT["specular"], show_scalar_bar=False, reset_camera=False)
        sel = g["sel"]
        if half_slab is not None:
            sel = sel[(np.abs(g["W"][g["P"]["i"][sel]]) < half_slab) & (np.abs(g["W"][g["P"]["j"][sel]]) < half_slab)]
        for name, pick, colour in (("tether", sel[g["P"]["bonded"][sel]], "#f2c230"), ("contact", sel[~g["P"]["bonded"][sel]], "#f4f5f6")):
            tube = None
            if len(pick):
                a, b = P3[g["P"]["i"][pick]], P3[g["P"]["j"][pick]]
                line = pv.PolyData(np.concatenate([a, b]))
                line.lines = np.column_stack([np.full(len(pick), 2), np.arange(len(pick)), len(pick) + np.arange(len(pick))]).ravel()
                rad = spacing * np.maximum((0.3 if graph else 1.0) * TUBE * g["P"]["rho"][pick], 0.03 if graph else 0.05)   # a tether's radius is proportional to its junction's bound density
                line.point_data["r"] = np.concatenate([rad, rad])
                tube = line.tube(scalars="r", absolute=True, n_sides=10)
                if not graph:
                    tube = tube.clip(normal=(0, 0, 1), origin=(0, 0, 0), invert=True)
            if tube is None or tube.n_points == 0:
                pl.remove_actor(name)
            else:
                pl.add_mesh(tube, color=colour, name=name, lighting=False, show_scalar_bar=False, reset_camera=False)
        if tag:
            head = tag
        elif label_only:                                                                    # the time and the pair's tethers, nothing else printed (no title, no legend)
            ts_ = float((((raw.get('general') or {}).get('units') or {}).get('time_s', 0)) or 0); t_txt = (f"t = {d['row'] * row_stride * float((raw.get('general') or {}).get('dt', 1.0)) * ts_ / 60.0:g} min" if (row_stride and ts_) else H.time_label(raw, d['row'], d['rows']))      # a partial trajectory: row x row_stride frames
            head = f"{t_txt}   junctions = {int(g['P']['bonded'][g['sel']].sum())}"
        elif graph:
            head = (f"MPM point graph: cell {focus} (dark), cell {other} (pale)\n{H.time_label(raw, d['row'], d['rows'])}  \u00b7  {len(g['mf']) + len(g['mo']):,} nodes"
                    f"  \u00b7  {int(g['P']['bonded'][g['sel']].sum()):,} edges")
        else:
            head = f"cell {focus} (dark), cell {other} (pale), section\n{H.time_label(raw, d['row'], d['rows'])}"
        pl.add_text(head, position=(12, Hh - 14 - (78 if tag else 21) * (head.count("\n") + 1)), font_size=48 if tag else 11, color=SI_.MOVIE_FG, name="head")
        if not tag and not label_only:
            pl.add_text(legend, position=(12, 12), font_size=11, color=SI_.MOVIE_FG, name="legend")
        img = pl.screenshot(return_img=True)[:, :, :3]
        zb = None if graph else ink.zbuffer(pl.render_window)
        if zb is not None and zb.shape == img.shape[:2]:
            D = ink.depth(zb, pl.camera)
            scene = (zb < 1.0 - 1e-7) & (np.abs(D - dist) < 1.6 * half_h * max(1.0, W / Hh))
            img = ink.ink(img, D, scene, wpp, dot_r=None, colour_origin=(0.0, 0.0, 0.0))
        if bar is not None:
            al = bar[..., 3:4].astype(np.float32) / 255.0
            img = (img * (1 - al) + bar[..., :3] * al).astype(np.uint8)
        proc.stdin.write(np.ascontiguousarray(img).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--why", required=True, help="the step's why file naming the run")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--cell", type=int, default=None, help="the focus cell's slot (default: the non-core cell with the most contacts at the end)")
    ap.add_argument("--neighbour", type=int, default=None, help="the neighbour cell's slot (default: the one with the most pairs at the end)")
    ap.add_argument("--rows", default="0,-1")
    ap.add_argument("--whole", action="store_true", help="with --movie graph, frame the two whole cells (as `cells` does) instead of their interface")
    ap.add_argument("--iso", type=float, default=None, help="with --movie cells or interface, the surface threshold of the bodies, in means of the occupied density (default %g): higher draws the surface closer to the points" % BODY["iso_frac"])
    ap.add_argument("--sigma", type=float, default=None, help="the blur of the body density, in voxels (default %g)" % BODY["sigma"])
    ap.add_argument("--dx", type=float, default=None, help="the voxel of the body density, in point spacings (default %g)" % BODY["dx"])
    ap.add_argument("--tethers-only", dest="tethers_only", action="store_true", help="with --movie, draw the tethers only: the contact pairs without a tether (white) and their points are left out")
    ap.add_argument("--window", type=float, default=None, help="with --movie, a fixed square field of this many um about the centre of the pair's points (a close-up of the junction)")
    ap.add_argument("--slab", type=float, default=None, help="with --movie graph, draw only the points and tethers within this many um of the section plane")
    ap.add_argument("--node-scale", dest="node_scale", type=float, default=1.0, help="with --movie graph, the size of the nodes times this (a close-up of 40 nm tethers wants nodes smaller than the tether is long)")
    ap.add_argument("--tilt", type=float, default=0.0, help="with --movie, turn the view about the axis between the cells by this many degrees (a regular skin seen along its rows stacks them)")
    ap.add_argument("--row-stride", dest="row_stride", type=int, default=None, help="with --movie and --label-only, frames between two stored rows of a PARTIAL trajectory.npz (the label reads row x this)")
    ap.add_argument("--follow", action="store_true", help="with --movie and --window, centre the field on the pair's points of each row (the junction stays in view while the cells move)")
    ap.add_argument("--label-only", dest="label_only", action="store_true", help="with --movie, print only the time and the number of tethers of the pair (`t = 350 min   junctions = 203`): no title, no legend")
    ap.add_argument("--tag", default="", help="with --movie, this one text in large type in place of the title and the legend")
    ap.add_argument("--before", action="store_true", help="with --movie, draw the rows before the neighbour exists too: the cell alone, in the frame of the pair it becomes (a cell before its division)")
    ap.add_argument("--movie", choices=("cells", "interface", "graph"), default=None,
                    help="write the pair of cells over every stored row to -o (an .mp4, VTK on a grey background, cut at the interface): `cells` the whole-cell view, `interface` its enlargement, `graph` the same frame as the graph of material points (a node per point, an edge per pair)")
    a = ap.parse_args()
    raw = yaml.safe_load(open(a.spec))
    run = _hier().run_dir_of(a.why)
    if not run:
        sys.exit("no run with a stored trajectory is named by " + a.why)
    if a.movie:
        ds = read_rows(run, (0, -1))
        p = params_of(raw)
        focus = a.cell if a.cell is not None else pick_focus(ds[-1], ds[0])
        other = a.neighbour
        if other is None:
            last, first = partners_of(ds[-1], pairs_of(ds[-1], p), focus), partners_of(ds[0], pairs_of(ds[0], p), focus)
            both = {c: n for c, n in last.items() if c in first} or last
            other = max(both, key=both.get)
        print("wrote", movie(raw, run, a.out, focus, other, zoom=a.movie in ("interface", "graph") and not a.whole, graph=a.movie == "graph", tag=a.tag, body={k: v for k, v in dict(iso_frac=a.iso, sigma=a.sigma, dx=a.dx).items() if v is not None}, contacts=not a.tethers_only, window=a.window, slab=a.slab, node_scale=a.node_scale, tilt=a.tilt, before=a.before, label_only=a.label_only, follow=a.follow, row_stride=a.row_stride), f"(cell {focus} and cell {other})")
        return
    png, info = figure_png(raw, run, tuple(int(x) for x in a.rows.split(",")), a.cell, a.neighbour)
    open(a.out, "wb").write(png)
    print("wrote", a.out, info)


if __name__ == "__main__":
    main()
