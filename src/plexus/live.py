"""The live snapshot: `3d.png`, rewritten every 10% of the run while it is still going.

A 1,800-frame vertex model takes 20-40 minutes and, until it finishes, writes nothing anyone can look at, so two runs of a promotion pair are indistinguishable from two runs of the WRONG THING for the whole time and a mistake is first seen half an hour after it was made. This writes a picture of the current state ten times over the run, in place, with the frame number printed top left, on by default when generating.

Written atomically (render to `3d.png.tmp`, then `os.replace`, atomic on one filesystem) so a viewer never sees half a PNG. Overwritten by the final render on purpose (the movie's own code path, the run's final camera); nothing downstream needs to know which one it is reading. The matplotlib import is inside the function: `graph_data_generator` never imports matplotlib at module level (a plot switch in the generator is the ParticleGraph anti-pattern)."""
from __future__ import annotations

import os

import numpy as np


# every_n: The stride that gives 1/frac snapshots over the run, at least 1 -> int.
def every_n(n_frames: int, frac: float = 0.1) -> int:
    """The stride that gives `1/frac` snapshots over the run, at least 1."""
    return max(1, int(round(n_frames * float(frac))))


# _live_pos: Positions of the live elements of the set to draw, and its mesh table or None -> (pos, mesh).
def _live_pos(H, sname):
    """(positions of the live elements, the mesh table or None) for the set to draw."""
    lvl = H.level(sname)
    if "pos" not in lvl.state_schema:
        return None, None
    m = getattr(lvl, "mesh", None)
    pos = lvl.get("pos").detach().cpu().numpy()
    # a mesh set is drawn to `Nv`, not to `occ`: `cell_die` never touches `Nv`, so orphaned vertices are still inside every `state[:Nv]` slice, and the picture must draw the subset the mechanics acts on
    if m and int(m.get("Nv", 0)) > 0:
        return pos[:int(m["Nv"])], m
    return pos[lvl.active.detach().cpu().numpy()], None


# cutaway_mask: Which elements to keep, True everywhere unless `cutaway` names a wedge to drop -> bool [n].
def cutaway_mask(pos, cutaway):
    """Which elements to KEEP: True everywhere unless `cutaway` names a wedge to drop. The particle twin of the mesh cutaway in `render_vtk.mesh_of`: a filled body has no faces to drop, so its interior is shown by removing the points in front of it. `cutaway: ["x", "y"]` drops the quadrant where both coordinates exceed the cloud's own CENTROID (not the world origin: a growing aggregate drifts); one axis halves it, two take a quadrant, three an octant; a leading "-" flips that axis."""
    q = np.asarray(pos, float)
    if not cutaway or q.ndim != 2 or q.shape[1] < 2:
        return None
    ax = {"x": 0, "y": 1, "z": 2}
    sel = [(ax[a.lower().lstrip("+-")], a.startswith("-"))
           for a in cutaway if a.lower().lstrip("+-") in ax and ax[a.lower().lstrip("+-")] < q.shape[1]]
    if not sel:
        return None
    c = q.mean(0)
    inside = np.ones(len(q), bool)
    for i, neg in sel:
        inside &= (q[:, i] < c[i]) if neg else (q[:, i] > c[i])
    return ~inside

# dot_area_pt2: Scatter area `s` (points^2) such that a dot spans `fill` of the local spacing -> float.
def dot_area_pt2(pos, span_data, fig_px, dpi, fill=0.9, cap=(1.0, 400.0)):
    """Scatter `s` (an AREA in points^2) such that a dot spans `fill` of the LOCAL SPACING. A constant `point_size` made the number that renders a 4,000-cell disc as a sheet render 1,000 cells as dinner plates and 40,000 as dust: dot size is relative to the spacing, which changes with the population.

    Measured, not estimated from density: the MEDIAN NEAREST-NEIGHBOUR DISTANCE needs no hull, is right for non-convex layouts and shrugs off edge cells. The conversion is exact (data -> px is `fig_px / span_data`, px -> pt is `72 / dpi`, `s` is an area so squared), so the apparent dot is invariant to figure size, dpi and n, which is what makes the snapshot and the movie agree. `fill` = 1.0 is exactly touching; 0.9 leaves a hairline so dots stay countable."""
    q = np.asarray(pos, float)[:, :2]
    if len(q) < 2 or span_data <= 0:
        return 12.0
    # measured on a sample: the median converges long before the full set, and a cKDTree over every point took 1.3 s at 1 M, 7.6 s at 4 M, ~3 min at 100 M (every 5% of frames: ~80 minutes of a 100 M run that looked like a hang); 20,000 points is the cap `plexus.live_movie._dot_px` uses
    if len(q) > 20000:
        q = q[np.random.default_rng(0).choice(len(q), 20000, replace=False)]
    try:
        from scipy.spatial import cKDTree
        # k=2 because the nearest neighbour of a point is itself at distance 0
        nn = cKDTree(q).query(q, k=2)[0][:, 1]
    except Exception:
        i = np.random.default_rng(0).choice(len(q), size=min(len(q), 1500), replace=False)
        dd = np.linalg.norm(q[i][:, None, :] - q[i][None, :, :], axis=-1)
        np.fill_diagonal(dd, np.inf)
        nn = dd.min(1)
    sp = float(np.median(nn[np.isfinite(nn)])) if np.isfinite(nn).any() else 0.0
    if sp <= 0:
        return 12.0
    pt = sp * (float(fig_px) / float(span_data)) * (72.0 / float(dpi))
    return float(np.clip((fill * pt) ** 2, *cap))


# chem_rgb: The one colour law for reaction-diffusion output -> (rgb [n, 3], max per drawn column).
def chem_rgb(chem, nF=None, lut=None, blend=None, background="black", vmax=None):
    """(rgb [n, 3], [max per drawn column]) from a `chem` array: THE ONE COLOUR LAW for RD output, imported by `plexus.plot` so the live snapshot and the movie cannot show different pictures of the same numbers (they once did: `movie_cell.mp4` never read `chem`). Pure numpy, no matplotlib.

    `lut` is the column-to-colour table and belongs in the SPEC, since which columns are drawn is a property of the model: positional, one entry per chem column, `None` = declared and deliberately not drawn (Gray-Scott draws its activator, May-Leonard's partition wants red / green / blue).

    `blend` depends on what the columns mean: `subtractive` (white is quiet, each species takes light away: Gray-Scott's sparse activator on a quiet ground) or `additive` (black is quiet, each species adds its colour: a conserved partition like May-Leonard, where the barycentric mix is the honest picture). Default follows `background`, so existing plots are unchanged. Each drawn column is normalised by its OWN maximum."""
    a = np.asarray(chem, float)
    if a.ndim != 2 or a.shape[1] < 1:
        return None, []
    n = a.shape[0] if nF is None else int(nF)
    a = a[:n]
    ncol = a.shape[1]

    if lut is None:                        # the historical default: activators of up to two pairs
        lut = [None] * ncol
        for c, col in ((0, "#ff2b2b"), (2, "#2b6bff")):
            if ncol > c:
                lut[c] = col
    lut = list(lut) + [None] * max(0, ncol - len(lut))
    blend = (blend or "subtractive").lower()

    def _rgb(h):
        h = str(h).lstrip("#")
        return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])

    drawn = [(k, _rgb(lut[k])) for k in range(ncol) if lut[k] is not None]
    if not drawn:
        return None, []
    hi, norm = [], []
    # `vmax`: a FIXED scale when the spec gives one (`plotting.chem_max`, a scalar for every column or a positional list); by default each species is normalised by its own per-frame maximum, which is wrong for anything that decays (a field diffusing to a uniform 0.076 renders every cell at 94-100% and the disc goes solid red when the concentration is lowest)
    _vm = (list(vmax) if isinstance(vmax, (list, tuple)) else [vmax] * ncol) if vmax is not None else None
    for k, _c in drawn:
        v = np.where(np.isfinite(a[:, k]), a[:, k], 0.0)
        top = float(v.max())
        hi.append(top)
        ref = float(_vm[k]) if (_vm is not None and k < len(_vm) and _vm[k]) else top
        norm.append(np.clip(v / ref, 0, 1) if ref > 1e-9 else np.zeros(n))

    if blend == "additive":
        cols = np.zeros((n, 3))
        for (k, c), w in zip(drawn, norm):
            cols += w[:, None] * c[None, :]
    else:
        # subtractive: start white, each species removes the light its own colour lacks; the floor stops short of 0 so a cell carrying every species stays a dark dot, not a hole in the canvas
        f = 0.86
        cols = np.ones((n, 3))
        for (k, c), w in zip(drawn, norm):
            cols = cols - f * w[:, None] * (1.0 - c)[None, :]
    return np.clip(cols, 0, 1), hi


# _face_colours: chem_rgb for the first set that carries a `chem` block, with the spec's own LUT -> (rgb, clim).
def _face_colours(H, nF, style=None):
    """`chem_rgb` for the first set that carries a `chem` block, using the spec's own LUT."""
    style = style or {}
    for _s, l in H.levels.items():
        if "chem" not in getattr(l, "state_schema", {}):
            continue
        c = l.get("chem")
        if c is None or c.shape[0] < nF:
            continue
        return chem_rgb(c.detach().cpu().numpy(), nF,
                        lut=style.get("species"), blend=style.get("blend"),
                        background=style.get("background", "black"),
                        vmax=style.get("chem_max"))
    return None, []


# snapshot: Draw the current state to `<out_dir>/3d.png` (`2d.png` for a 2D run); never raises -> None.
def snapshot(H, tick, n_frames, out_dir, name="", sname=None, style=None):
    """Draw the current state to `<out_dir>/3d.png`. Never raises: a failed picture must not take
    down a run that is otherwise fine, which is exactly what a 3D-axes `text` signature once did to
    sixteen runs at once."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if sname is None:                        # the biggest spatial set is the interesting one
            cand = [(int(getattr(l, "n", 0)), s) for s, l in H.levels.items()
                    if "pos" in l.state_schema]
            if not cand:
                return None
            sname = max(cand)[1]
        pos, m = _live_pos(H, sname)
        _clim = []
        if pos is None or len(pos) == 0:
            return None
        D = pos.shape[1]
        os.makedirs(out_dir, exist_ok=True)
        fig = plt.figure(figsize=(6.2, 6.2), facecolor="black")
        if D >= 3:
            ax = fig.add_subplot(111, projection="3d", facecolor="black")
            # framed on the cloud, not the origin: a vertex mesh is centred on 0, but an MPM domain is [0, 1]^3 and `abs(pos).max()` would put the body in a corner at a third of the frame
            c3 = pos[:, :3].mean(0)
            r = float(np.abs(pos[:, :3] - c3).max()) * 1.08 or 1.0
            if m is not None:
                # the surface, not a point cloud: the ring of each face drawn as a polygon (20,000 vertices read as a fuzzy ball whatever the tissue does)
                from plexus.models.topology import rings_from_flat_3d
                es, et, ef = (np.asarray(_np(m[k])) for k in ("E_srce", "E_trgt", "E_face"))
                rings = rings_from_flat_3d(es, et, ef, int(m["nF"]))
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                keep = [i for i, rg in enumerate(rings) if len(rg) >= 3]
                polys = [pos[np.asarray(rings[i], int), :3] for i in keep]
                fc, _clim = _face_colours(H, int(m["nF"]), style)
                pc = Poly3DCollection(
                    polys, facecolor=(fc[keep] if fc is not None else "#dcdcdc"),
                    edgecolor="#5a5a5a", linewidths=0.15, alpha=1.0)
                ax.add_collection3d(pc)
            else:
                _k = cutaway_mask(pos[:, :3], (style or {}).get("cutaway"))
                _p = pos if _k is None else pos[_k]
                _fc, _clim = _face_colours(H, len(pos), style)
                if _fc is not None and _k is not None:
                    _fc = _fc[_k]
                ax.scatter(_p[:, 0], _p[:, 1], _p[:, 2], depthshade=False,
                           s=dot_area_pt2(_p[:, :3], 2.0 * r or 1.0, 6.2 * 110, 110),
                           c=(_fc if _fc is not None else "#dcdcdc"), linewidths=0)
            ax.set_xlim(c3[0] - r, c3[0] + r); ax.set_ylim(c3[1] - r, c3[1] + r)
            ax.set_zlim(c3[2] - r, c3[2] + r)
            ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))
            put = ax.text2D
        else:
            ax = fig.add_subplot(111, facecolor="black")
            # the 2D branch carries the chemistry too: a flat RD run has nothing to show but its chemistry (cells never move, so a grey scatter is the same picture at frame 1 and frame 6000)
            fc, _clim = _face_colours(H, len(pos), style)
            # sized from the measured spacing, not the count (`22 * 4000/n` assumed a fixed area): `dot_area_pt2` keeps dots nearly touching whatever n, world box or canvas
            _span = 2.0 * float(np.abs(pos[:, :2] - pos[:, :2].mean(0)).max()) * 1.08 or 1.0
            ax.scatter(pos[:, 0], pos[:, 1],
                       s=dot_area_pt2(pos, _span, 6.2 * 110, 110),
                       c=(fc if fc is not None else "#dcdcdc"), linewidths=0)
            ax.set_aspect("equal"); ax.set_axis_off()
            put = ax.text
        n_live = (int(m["nF"]) if m is not None else len(pos))
        unit = "cells" if m is not None else "elements"
        put(0.02, 0.97, f"{name}  frame {tick} / {n_frames}   {n_live} {unit}   [live]",
            transform=ax.transAxes, color="white", fontsize=11, va="top")
        if _clim:
            # the scale, stated: colours are normalised per frame (see `_face_colours`), so the picture means nothing without the number it was divided by
            put(0.02, 0.94, "  ".join(
                # not `'AB'[i]`: that raised IndexError on the first three-species run, which the catch-all turned into no picture
                f"{chr(65 + i) if i < 26 else i} max {v:.4g}" for i, v in enumerate(_clim)),
                transform=ax.transAxes, color="#b0b0b0", fontsize=8, va="top")
        fig.subplots_adjust(0, 0, 1, 1)
        # `.tmp` last would pick the format from the extension and matplotlib refuses "tmp", so the temporary name keeps the `.png` suffix; named for the dimension it draws (a 2D run once wrote a flat scatter called `3d.png`)
        stem = "3d" if D >= 3 else "2d"
        tmp = os.path.join(out_dir, f".{stem}.partial.png")
        fig.savefig(tmp, dpi=110, facecolor="black")
        plt.close(fig)
        os.replace(tmp, os.path.join(out_dir, f"{stem}.png"))   # atomic: never a half-written PNG
        return os.path.join(out_dir, f"{stem}.png")
    except Exception as e:
        print(f"[live] snapshot at frame {tick} skipped ({type(e).__name__}: {str(e)[:70]})",
              flush=True)
        return None


# _np: A torch tensor or array-like -> numpy array, detached and on the CPU.
def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
