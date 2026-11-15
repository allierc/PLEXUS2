"""exp29 RIG: eight small vesicles in ONE box, each with its own parameters, as exp27's movie 2 puts eight small arrays side by side (Cedric,
2026-10-23: "to test more rigs faster, better").

    python tools/exp29_rig.py --write                 # config/tissue/exp29_rig_a.yaml (overlap 0.25), exp29_rig_b.yaml (overlap 0.05), exp29_rig_v.yaml (vertex twin)
    python tools/exp29_rig.py --measure RUN [RUN..]   # the per-vesicle table and the cross-section sheet of a landed rig (rig_<run>.png in the run folder)

A vesicle = the cells of one TYPE (`seed_positions[vesicle]` with `vesicles:`), 40 cells on a sphere about its own centre; the per-vesicle knobs are
the cell type (E, tau), the type-pair contact tension (the diagonal of `contact_tension`), the per-cell surface-tension factor (`tension` block, read
by `mpm_grid_update cell_surface_tension_block`), the polarity (face labels by radial thirds, `colour_from_face` weights 1 : 0.05 : 0.05: a non-polar vesicle is all
lateral, weight 1), the adhesion row by face-pair class, and the footprint drive's k per type (`active_force[footprint_spreading]`, `spreading_strength` a list, `centre_block`).
The overlap tolerance of the contact is one number per run: rig A runs it at 0.25, rig B at 0.05, the same eight vesicles.

KNOWN CAVEATS (Cedric: "launch with known caveats"): COARSE bodies (125 points a cell at spacing 0.2 on a 0.2 grid, against 512 at 0.125) so the run
takes minutes; 40-cell vesicles (radius 1.9 against 6.6): their curvature makes the cells widen the sphere by themselves (the 100-cell probe, no drive:
diameter x1.22 where the 500-cell sphere stayed at x1.03), so every reading is against the control vesicles (0 and 7) and the vertex twin of the SAME
size (rig v: 40 cells, radius 0.707); growth x4 on the clip's rate from frame 20 (a doubling in 67 frames), 150 frames; one seed.
"""
import argparse
import math
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]
os.environ.setdefault("PLEXUS_BUILDER", os.path.join(ROOT, "experiments", "exp29_cell_tissue_2"))   # the record this tool writes into (as exp29_studio)
import exp29_specs as S  # noqa: E402

NV, NC = 8, 40                                   # vesicles, cells each
H, PP, NGRID = 0.2, 125, 15                      # point spacing, points per cell, grid nodes along a cell's 3.0 box (dx 0.2)
SPACING = 9.0                                    # between vesicle centres (6.0 let the grown vesicles touch by frame ~120); arm 0 at the largest x: the view puts it on the LEFT
CENTRES = [[5.0 + SPACING * (3 - g % 4), 7.0 + SPACING * (g // 4), 10.0] for g in range(NV)]
FRAMES, AFTER, SCALE = 150, 20, 4.0
BASE_TENSION, BASE_CONTACT = 0.15, -0.05
POLAR_ADH = {"LL": 1.0, "AA": 0.1, "BB": 0.1, "AB": 0.1, "LA": 0.1, "LB": 0.1}
#       name                                   surface tension  contact   polar  drive k  tau
ARMS = [("0 control",                            0.15,           -0.05,    False,   0.0,    2.0),
        ("1 drive k 100",                        0.15,           -0.05,    False, 100.0,    2.0),
        ("2 polar cortex 0.5 / 0.025",           0.50,           -0.05,    True,    0.0,    2.0),
        ("3 polar cortex + drive k 100",         0.50,           -0.05,    True,  100.0,    2.0),
        ("4 tension 0.5, contact -0.3",          0.50,           -0.30,    False,   0.0,    2.0),
        ("5 tau 20",                             0.15,           -0.05,    False,   0.0,   20.0),
        ("6 contact -0.1 + drive k 100",         0.15,           -0.10,    False, 100.0,    2.0),
        ("7 control (replicate)",                0.15,           -0.05,    False,   0.0,    2.0)]


def rig_spec(name, overlap):
    sp = S.build("r021_12", name, cells=NV * NC, buffer=int(NV * NC * 4.5), frames=FRAMES, record_cap=FRAMES // 2 + 1, contact=0.2, per_parent=PP,
                 grow_scale=SCALE, after_frame=AFTER, rho=1.0, a_sw=0.0, tau=2.0, apical_area=1.0, face_tension=0.05,
                 surface_tension=BASE_TENSION, contact_tension=BASE_CONTACT, centre=CENTRES[0])
    g = sp["general"]
    g["world"] = [40.0, 26.0, 20.0]
    sp["fields"]["mpm_grid"]["n_grid"] = NGRID
    cell = sp["sets"]["cell"]
    ty = {}
    for i, (nm, st, ct, pol, k, tau) in enumerate(ARMS):
        ty[f"v{i}"] = {"count": NC, "shape": "ball", "material": "viscoelastic", "youngs": 20.0, "tau": float(tau), "density": 1.0}
    cell["types"] = ty
    cell["state"]["tension"] = {"width": 1}
    cell["state"]["vcentre"] = {"width": 3}
    for o in sp["operators"]:
        op = o["op"]
        if op == "mpm_grid_update":
            o["cell_surface_tension_block"] = "tension"
        elif op == "pair_potential":
            m = np.zeros((NV, NV))
            for i, a in enumerate(ARMS):
                m[i, i] = a[2]
            o["contact_surface_tension"] = m.tolist()
            o["contact_slack"] = float(overlap)
        elif op == "bond_binding":
            o["type_matrix"] = np.ones((NV, NV)).tolist()
        elif op == "contact_finder" or op == "bond_springs":
            o["spacing"] = H
        elif op == "active_force":
            o["spreading_strength"] = [float(a[4]) for a in ARMS]
            o["centre_block"] = "vcentre"
        elif op == "cell_divide":
            o["centre_block"] = "vcentre"
            o["inherit"] = sorted(set(o.get("inherit", [])) | {"adh", "pol", "vcentre"})
    seed = sp["seed"][0]
    seed["point_spacing"] = H
    seed.pop("adh_by_class", None)
    seed["vesicles"] = [{"centre": CENTRES[i], "tension": a[1] / BASE_TENSION, "polar": a[3], **({"adh_by_class": POLAR_ADH} if a[3] else {})}
                        for i, a in enumerate(ARMS)]
    sp["plotting"]["duration_s"] = 8
    return sp


def vertex_spec(name):
    sp = yaml.safe_load(open(os.path.join(ROOT, "config", "tissue", "exp29_v_grow_u.yaml")))
    sp["general"].update(name=name, n_frames=FRAMES, record_cap=FRAMES // 2 + 1, record_every=2)
    for o in sp["operators"]:
        if o["op"] == "seed_mesh":
            o.update(n_cells=NC, radius=round(2.5 * math.sqrt(NC / 500.0), 4))
        elif o["op"] == "cell_grow":
            o.update(rate=round(0.000866 * SCALE, 6), after_frame=AFTER)
    target = 4 * NC
    cn = int(1.3 * target) + 4
    sp["_run"] = dict(sp["_run"], seed_cells=NC, target_cells=target)
    sp["sets"]["cell"]["n"], sp["sets"]["vertex"]["n"] = cn, 2 * (cn - 4) + 4
    sp["sets"]["half_edge"]["n"] = 8 * cn
    return sp


def write():
    for nm, ov in (("exp29_rig_a9", 0.25), ("exp29_rig_b9", 0.05)):
        sp = rig_spec(nm, ov)
        with open(os.path.join(ROOT, "config", "tissue", f"{nm}.yaml"), "w") as f:
            yaml.safe_dump(sp, f, sort_keys=False, default_flow_style=None, width=140)
        print(f"config/tissue/{nm}.yaml  ({NV} vesicles of {NC} cells, overlap {ov}, centres {SPACING:g} apart)")
    sp = vertex_spec("exp29_rig_v")
    with open(os.path.join(ROOT, "config", "tissue", "exp29_rig_v.yaml"), "w") as f:
        yaml.safe_dump(sp, f, sort_keys=False, default_flow_style=None, width=140)
    print("config/tissue/exp29_rig_v.yaml  (vertex twin, 40 cells)")


def measure(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from exp_measures import exp29 as m
    for run in runs:
        c = m.cells_of(m.open_run(f"tissue/{run}"))
        ntt = np.asarray(c.z["cell__node_type_t"]) if "cell__node_type_t" in c.z.files else None
        if ntt is None:
            print(f"{run}: no cell__node_type_t in the trajectory"); continue
        rows = c.rows
        t0 = min(int(round(AFTER / c.every)), rows - 1)
        print(f"\n{run}: {rows} rows every {c.every} frames; ratios to frame {AFTER}; the arm's vesicle = the cells of its type")
        print(f"{'arm':34s} {'cells':>5s} {'diam':>6s} {'footprint':>9s} {'span':>6s} {'span/h':>6s} {'degree':>6s} {'wall':>5s} {'close':>5s}")
        fig, axs = plt.subplots(2, 4, figsize=(16, 8.4)); axs = axs.ravel()
        res = {}
        for g in range(NV):
            ref = None
            for t in (t0, rows - 1):
                live = c.live(t)
                cells = np.flatnonzero(c.occ[t] & (ntt[t] == g))
                if len(cells) < 4:
                    continue
                x = c.cen[t][cells].astype(float); ctr = x.mean(0); D = 2 * np.linalg.norm(x - ctr, axis=1).mean()
                pos = np.asarray(c.z["pt__pos"][t]); occ = np.asarray(c.z["pt__occ"][t]) > 0; pid = np.asarray(c.z["pt__pid"][t]).reshape(-1).astype(int)
                sel = occ & np.isin(pid, cells)
                p, ip = pos[sel], pid[sel]
                order = np.argsort(ip); ps = p[order]; ids = ip[order]; cuts = np.flatnonzero(np.diff(ids)) + 1
                A, sp_, hv = [], [], []
                for gpts in np.split(ps, cuts):
                    if len(gpts) < 20:
                        continue
                    cen = gpts.mean(0); u = (cen - ctr) / np.linalg.norm(cen - ctr); r = (gpts - cen) @ u; rt = (gpts - cen) - np.outer(r, u)
                    a = 2 * np.pi * (rt ** 2).sum(1).mean(); A.append(a); sp_.append(np.percentile(r, 90) - np.percentile(r, 10)); hv.append(len(gpts) * H ** 3 / a)
                rp = np.linalg.norm(p - ctr, axis=1); wall = np.percentile(rp, 95) - np.percentile(rp, 5)
                from scipy.spatial import cKDTree
                tr = cKDTree(p); d, j = tr.query(p, k=12); oth = ip[j[:, 1:]] != ip[:, None]; dd = np.where(oth, d[:, 1:], np.inf).min(1)
                fr_ = c.frame(t)
                f = {"cells": len(cells), "D": D, "A": float(np.median(A)), "span": float(np.median(sp_)), "sh": float(np.median(np.array(sp_) / np.array(hv))),
                     "wall": float(wall), "close": float(np.mean(dd[np.isfinite(dd)] < 0.5 * H))}
                if ref is None:
                    ref = f
                res[g] = (f, ref, cells, ctr, t)
            if g in res:
                f, ref, cells, ctr, t = res[g]
                st = f"{ARMS[g][0]}"
                print(f"{st:34s} {f['cells']:5d} {f['D'] / ref['D']:6.3f} {f['A'] / ref['A']:9.2f} {f['span'] / ref['span']:6.2f} {f['sh']:6.2f} "
                      f"{'':>6s} {f['wall']:5.2f} {f['close']:5.3f}")
                # the cross-section: points within 0.12 of the vesicle's equatorial plane, coloured by cell
                pos = np.asarray(c.z["pt__pos"][t]); occ = np.asarray(c.z["pt__occ"][t]) > 0; pid = np.asarray(c.z["pt__pid"][t]).reshape(-1).astype(int)
                sel = occ & np.isin(pid, cells) & (np.abs(pos[:, 2] - ctr[2]) < 0.12)
                axs[g].scatter(pos[sel, 0] - ctr[0], pos[sel, 1] - ctr[1], c=(pid[sel] % 20), cmap="tab20", s=7, linewidths=0)
                axs[g].set_aspect("equal"); axs[g].set_xlim(-3.6, 3.6); axs[g].set_ylim(-3.6, 3.6); axs[g].axis("off")
                axs[g].set_title(f"{ARMS[g][0]}\n{f['cells']} cells, diameter x{f['D'] / ref['D']:.2f}", fontsize=9)
        fig.tight_layout()
        out = os.path.join(ROOT, "graphs_data", "tissue", run, f"rig_{run}.png")
        fig.savefig(out, dpi=90); plt.close(fig)
        print(f"sheet: {out}")


def movie(run, vertex_run=None, seconds=8.0, archive=True, replace=None):
    """The rig in the studio's INK LOOK (Cedric, 2026-10-23: "I expected ink style 3d view, the usual gray background, white text"): the eight vesicles seen
    from above (camera 88 degrees up), every cell white to red by its activator on one scale, each vesicle CUT at its equator (the camera side of the
    plane through the rig's centre is hidden: all eight centres lie at z = 10, so the plane is every vesicle's equator), a title over each vesicle (the
    arm, its cell count and its diameter against frame AFTER), the vertex twin's diameter ratio in the clock line. Archived as the next record step, or
    INTO step `replace`."""
    import math
    import shutil
    import time
    import pyvista as pv
    import imageio.v2 as iio
    import vtk
    import exp23_studio as e23
    import exp29_studio as ST
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    from exp_measures import exp29 as m
    e23._LOOK.clear(); e23._LOOK.update(e23.LOOK["ink"])
    e23._LOOK["text"] = "white"; e23._LOOK["border"] = (255, 255, 255)
    W, Hh = 1600, 800
    d, kind = ST.resolve(f"tissue/{run}")
    spec, spec_path = ST.read_spec(d)
    R = ST.MPMRun(d, spec)
    a_rows, _vol, n_live = R.series()
    a_top = ST.top_of_scale(a_rows)
    c = m.cells_of(m.open_run(f"tissue/{run}"))
    ntt = np.asarray(c.z["cell__node_type_t"])
    rows = R.rows
    t0 = min(int(round(AFTER / c.every)), rows - 1)
    cnt = np.zeros((rows, NV), int); dia = np.full((rows, NV), np.nan)
    for t in range(rows):
        live = c.live(t)
        for g in range(NV):
            cells = np.flatnonzero(c.occ[t] & (ntt[t] == g))
            cnt[t, g] = len(cells)
            if len(cells) > 3:
                x = c.cen[t][cells].astype(float); dia[t, g] = 2 * np.linalg.norm(x - x.mean(0), axis=1).mean()
    vd = None
    if vertex_run:
        cv = m.cells_of(m.open_run(f"tissue/{vertex_run}"))
        vd = []
        for t in range(cv.rows):
            sl = cv.live(t); x = cv.cen[t][sl].astype(float); vd.append(2 * np.linalg.norm(x - x.mean(0), axis=1).mean())
        vd = np.array(vd) / vd[min(int(round(AFTER / cv.every)), cv.rows - 1)]
    el, az = 88.0, 90.0
    view, up = ST.view_vectors(el, az)
    ks = list(range(rows))
    allp = np.concatenate([R.positions(k)[R.occ[k]][:: max(1, int(R.occ[k].sum()) // 3000)] for k in ks])
    c0 = np.median(allp, axis=0)
    R0 = float(np.percentile(np.linalg.norm(allp - c0, axis=1), 99))
    pl = pv.Plotter(off_screen=True, window_size=(W, Hh))
    pl.set_background(ST.GROUND)
    ST._add_lights(pl, c0, R0, view, up)
    va = 30.0
    dist, mid = ST.camera_for(pl, allp, va, (W, Hh), view, up)
    px_per_unit = Hh / (2.0 * dist * math.tan(math.radians(va / 2)))
    tgt = 0.2 * W / px_per_unit * R.um
    p10_ = 10 ** math.floor(math.log10(tgt))
    L_um = max(f * p10_ for f in (1, 2, 5) if f * p10_ <= tgt)
    bar = (L_um, L_um / R.um * px_per_unit, int(0.93 * Hh))
    font = ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"), 19)
    h_vox = 0.55 * (R.v ** (1.0 / 3.0))
    fps = max(1, round(rows / seconds))
    tmp_mp4 = f"/tmp/exp29_rig_{run}_ink.mp4"; tmp_png = f"/tmp/exp29_rig_{run}_ink.png"
    writer = iio.get_writer(tmp_mp4, fps=fps, quality=8, macro_block_size=16)
    coord = vtk.vtkCoordinate(); coord.SetCoordinateSystemToWorld()

    def to_screen(pt):
        coord.SetValue(float(pt[0]), float(pt[1]), float(pt[2]))
        x, y = coord.GetComputedDoubleDisplayValue(pl.renderer)
        return x, Hh - y
    frame = None
    for k in ks:
        ST.draw_cells(pl, "mpm", R, k, a_top, True, c0, R0, view, h_vox)
        f3 = e23._ink_frame(pl, dist, va, Hh, False)
        f3 = e23._scale_bar(f3, bar, "white")
        f3 = ST.legend(f3, a_top, "activator a per cell", y0=Hh - 92)
        im = Image.fromarray(f3); dr = ImageDraw.Draw(im)
        big = ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf"), 40)
        small = ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"), 15)
        ov = str(next(o for o in spec["operators"] if o["op"] == "pair_potential")["overlap"])
        dr.text((18, 14), f"contact overlap tolerance {ov}", fill=(255, 255, 255), font=font)
        if vd is not None and k >= t0:
            dr.text((18, 40), f"vertex twin (40 cells): diameter x{vd[min(k, len(vd) - 1)]:.2f}", fill=(255, 255, 255), font=small)
        for g in range(NV):
            rat = f", diameter x{dia[k, g] / dia[t0, g]:.2f}" if (k >= t0 and np.isfinite(dia[k, g]) and np.isfinite(dia[t0, g])) else ""
            y0 = 90 + g * 50
            dr.text((18, y0), ARMS[g][0], fill=(255, 255, 255), font=small)
            dr.text((30, y0 + 19), f"{cnt[k, g]} cells{rat}", fill=(255, 255, 255), font=small)
        f3 = np.array(im)
        foot = f"frame {c.frame(k):.0f} of {FRAMES}"
        frame = e23._clock(f3, foot, "white")
        writer.append_data(frame[: Hh // 2 * 2, : W // 2 * 2])
    writer.close()
    iio.imwrite(tmp_png, frame)
    pl.close()
    print(f"movie {tmp_mp4} ({rows} frames at {fps} fps)")
    if archive:
        import gui_drive as gd
        i = replace if replace else gd._next_index()
        shutil.copy(tmp_png, gd._path("png", i)); shutil.copy(tmp_mp4, gd._path("mp4", i))
        if spec_path:
            shutil.copy(spec_path, gd._path("spec", i))
        why = (f"EXP29 RIG {run}: eight 40-cell vesicles in one box, one growth test each (growth x4 on the clip's rate from frame {AFTER}, {FRAMES} frames, "
               f"coarse bodies: 125 points a cell on a 0.2 grid), seen from above, each cut at its equator, white to red by activator. Arms: "
               + "; ".join(a[0] for a in ARMS) + ". Contact overlap tolerance " + str(next(o for o in spec["operators"] if o["op"] == "pair_potential")["overlap"]) + "; vesicle centres 9 units apart, arm 0 at the left, arms 0-3 on the top row, 4-7 below"
               + ". Known caveats: coarse bodies, 40-cell vesicles widen by curvature alone (read against controls 0 and 7 and the vertex twin of the same size), one seed.")
        with open(gd._path("why", i), "w") as f:
            f.write(f"time   {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"spec   tissue/{run} (cluster run)\n")
            f.write(f"why    {why}\n")
        gd.note(f"  exp29_rig {i:04d}  {run}  -- {'REDRAWN IN PLACE: ' if replace else ''}{why}")
        print({"index": i})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--measure", nargs="+")
    ap.add_argument("--movie", help="draw the cross-section movie of this rig run and archive it as the next record step")
    ap.add_argument("--vertex", help="the vertex twin run for the movie's footer")
    ap.add_argument("--replace", type=int, default=None, help="archive INTO this existing record step (no new step)")
    a = ap.parse_args()
    if a.write:
        write()
    if a.measure:
        measure(a.measure)
    if a.movie:
        movie(a.movie, a.vertex, replace=a.replace)
