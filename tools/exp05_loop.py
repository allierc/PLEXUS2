"""exp05: a biologist's description run in Plexus, one iteration at a time (Cedric, 2026-11-14).

    python tools/exp05_loop.py measure NAME             # the measures of a landed run
    python tools/exp05_loop.py file NAME --why "..."    # file a landed run as the next watcher step of experiments/exp05_spheroid_from_words
    python tools/exp05_loop.py restyle STEP --mp4 M --png P --why "..."   # a redrawn movie into an existing step (spec and why kept, a relook line added)
    python tools/exp05_loop.py rerun PLAN.json [--streams 3] [--only 1 2]  # rerun steps on the LOCAL GPU, each re-filed in place

The measures, per stored frame, from the run's trajectory: live cells; clusters = groups of cells linked by touching (centre distance under
`touch` of the spec's cell_divide line); a SPHEROID = a cluster of at least 8 cells; for the largest cluster its cell count and its roundness
(smallest over largest principal axis of its cell centres, 1 = round); the share of cells in a cluster of 2 or more; the share of cells
growing (touching at least min_contacts others); the median distance from a clustered cell to its nearest neighbour, in cell diameters (below 1 = the two cells overlap).
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(ROOT, "experiments", "exp05_spheroid_from_words")
os.environ.setdefault("PLEXUS_BUILDER", REC)
LOG = os.path.join(ROOT, "log", "experiments", "exp05")
SPHEROID_MIN = 8                       # cells: the smallest cluster called a spheroid


def _clusters(X, touch):
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    import scipy.sparse as sp
    pairs = cKDTree(X).query_pairs(touch, output_type="ndarray")
    n = len(X)
    A = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else sp.coo_matrix((n, n))
    k, lab = connected_components(A, directed=False)
    return lab, np.bincount(lab), pairs


def _euler_cubical(a):
    """Euler characteristic V - E + F - C of the union of the CLOSED unit cubes of the True voxels of `a` (3-D bool)."""
    a = np.pad(a.astype(bool), 1)
    V = (a[:-1, :-1, :-1] | a[1:, :-1, :-1] | a[:-1, 1:, :-1] | a[:-1, :-1, 1:] | a[1:, 1:, :-1] | a[1:, :-1, 1:] | a[:-1, 1:, 1:] | a[1:, 1:, 1:]).sum()
    E = ((a[:, :-1, :-1] | a[:, 1:, :-1] | a[:, :-1, 1:] | a[:, 1:, 1:]).sum() + (a[:-1, :, :-1] | a[1:, :, :-1] | a[:-1, :, 1:] | a[1:, :, 1:]).sum()
         + (a[:-1, :-1, :] | a[1:, :-1, :] | a[:-1, 1:, :] | a[1:, 1:, :]).sum())
    F = (a[:-1, :, :] | a[1:, :, :]).sum() + (a[:, :-1, :] | a[:, 1:, :]).sum() + (a[:, :, :-1] | a[:, :, 1:]).sum()
    return int(V) - int(E) + int(F) - int(a.sum())


def voxel_topology(X, sigma=0.5, level=0.5, h=0.2):
    """(b0, b1, b2) of the body the cells make: b0 pieces, b1 tunnels through the body (a torus or an open tube 1), b2 closed cavities
    inside it (a hollow cyst 1). The body is where the cell density f(x) = sum_j exp(-|x - x_j|^2 / (2 sigma^2)) is at least `level`, on a
    grid of spacing `h` (world length; one cell diameter = 1): a lone cell is a ball of radius 0.59, two cells up to 1.7 apart are joined,
    the gaps between packed cells are filled, and a lumen wider than about 1.5 cell diameters stays open.
    Voxels are closed cubes (26-connected body, 6-connected outside); b2 = outside components - 1, b1 = b0 + b2 - Euler characteristic."""
    from scipy import ndimage
    X = np.asarray(X, float)
    if len(X) == 0:
        return 0, 0, 0
    pad = 4.0 * sigma
    lo = X.min(0) - pad
    shape = np.ceil((X.max(0) + pad - lo) / h).astype(int) + 1
    idx = np.clip(np.rint((X - lo) / h).astype(int), 0, shape - 1)
    cnt = np.zeros(tuple(shape))
    np.add.at(cnt, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    f = ndimage.gaussian_filter(cnt, sigma / h, mode="constant", truncate=4.0) * (2.0 * math.pi * sigma * sigma) ** 1.5 / h ** 3
    img = f >= level
    b0 = ndimage.label(img, structure=np.ones((3, 3, 3)))[1]
    b2 = ndimage.label(~np.pad(img, 1))[1] - 1
    chi = _euler_cubical(img)
    return int(b0), int(b0 + b2 - chi), int(b2)


def body_topology(X, touch, min_cells=SPHEROID_MIN):
    """Per cluster of at least `min_cells` cells (cells linked by touching): (cells, b1, b2, roundness), largest first."""
    lab, sizes, _ = _clusters(X, touch)
    out = []
    for c in np.flatnonzero(sizes >= min_cells):
        Y = X[lab == c]
        _, b1, b2 = voxel_topology(Y)
        ev = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov((Y - Y.mean(0)).T)), 0))
        out.append((int(len(Y)), b1, b2, round(float(ev[0] / max(ev[-1], 1e-9)), 2)))
    return sorted(out, reverse=True)


def measure(name):
    import yaml
    run = os.path.join(ROOT, "graphs_data", "tissue", name)
    spec = yaml.safe_load(open(os.path.join(run, "spec.yaml")))
    div = next(o for o in spec["operators"] if o.get("op") == "cell_divide")
    # point cells: `touch` of the division line; cells of material points (MPM): centroids closer than 1.5 units (15 um; a ball of volume 1 is 12.4 um across)
    touch, nmin = float(div.get("touch", 1.5)), int(div.get("min_contacts", 1))
    gate = next((o for o in spec["operators"] if o.get("op") == "cell_grow" and "threshold" in o), None)
    tf = float(spec["general"].get("units", {}).get("time_s", 600.0)) / 3600.0 * float(spec["general"].get("dt", 1.0))
    f = np.load(os.path.join(run, "trajectory.npz"))
    pos, occ = f["cell__pos"], f["cell__occ"]
    rows = []
    for k in range(pos.shape[0]):
        X = pos[k][occ[k].reshape(-1) > 0.5]
        lab, sizes, pairs = _clusters(X, touch)
        deg = np.bincount(pairs.reshape(-1), minlength=len(X)) if len(pairs) else np.zeros(len(X), int)
        if gate is not None and f"cell__{gate.get('adh_block', 'chem')}" in f.files:    # MPM: growing = adhesion level at or above the gate's threshold
            A = f[f"cell__{gate.get('adh_block', 'chem')}"][k][occ[k].reshape(-1) > 0.5].reshape(len(X), -1)[:, 0]
            deg = np.where(A >= float(gate["threshold"]), nmin, 0)
        big = int(np.argmax(sizes))
        Y = X[lab == big]
        if len(Y) >= 4:
            ev = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov((Y - Y.mean(0)).T)), 0))
            rnd = float(ev[0] / max(ev[-1], 1e-9))
        else:
            rnd = float("nan")
        from scipy.spatial import cKDTree
        nn = cKDTree(X).query(X, k=2)[0][:, 1] if len(X) > 1 else np.array([np.inf])
        nn = nn[sizes[lab] >= 2]
        rows.append(dict(row=k, nn_median=float(np.median(nn)) if len(nn) else float("nan"), cells=int(len(X)), clusters=int(len(sizes)), spheroids=int((sizes >= SPHEROID_MIN).sum()),
                         largest=int(sizes.max()), roundness=rnd, in_clusters=float((sizes[lab] >= 2).mean()),
                         growing=float((deg >= nmin).mean())))
    n_frames = int(spec["general"]["n_frames"])
    stride = n_frames / max(pos.shape[0] - 1, 1)
    last = rows[-1]
    first_sph = next((r["row"] for r in rows if r["spheroids"] > 0), None)
    out = dict(name=name, hours=round(n_frames * tf, 1), last=last,
               first_spheroid_h=None if first_sph is None else round(first_sph * stride * tf, 1),
               sizes_last=sorted(np.bincount(_clusters(pos[-1][occ[-1].reshape(-1) > 0.5], touch)[0]).tolist(), reverse=True)[:12])
    if "pt__pos" not in f.files:                                            # point cells: the topology of every body of 8+ cells at the end, the deaths seen
        bt = body_topology(pos[-1][occ[-1].reshape(-1) > 0.5], touch)
        out["bodies"] = bt[:8]
        out["tunnels"], out["cavities"] = int(sum(b[1] for b in bt)), int(sum(b[2] for b in bt))
        o_ = occ.reshape(occ.shape[0], -1) > 0.5
        out["deaths_seen"] = int((o_[:-1] & ~o_[1:]).sum())                 # slots live in one stored row and dormant in the next: a lower bound
        tt = []
        for k in np.linspace(0, len(pos) - 1, 11).astype(int)[1:]:
            bb = body_topology(pos[k][occ[k].reshape(-1) > 0.5], touch)
            tt.append((round(float(k * stride * tf), 1), int(sum(b[1] for b in bb)), int(sum(b[2] for b in bb))))
        out["topology_t"] = tt
    if "cell__chem" in f.files and div.get("signal_block"):                 # signalling runs: the activator of the live cells at the end
        a = f["cell__chem"][-1][occ[-1].reshape(-1) > 0.5][:, int(div.get("signal_channel", 0))]
        out["activator_last"] = dict(p10=round(float(np.percentile(a, 10)), 3), median=round(float(np.median(a)), 3),
                                     p90=round(float(np.percentile(a, 90)), 3), active=round(float((a >= float(div.get("signal_min", 0))).mean()), 3))
    json.dump(dict(summary=out, rows=rows), open(os.path.join(run, "exp05_measures.json"), "w"), indent=1)
    print("MEASURE " + json.dumps(out), flush=True)
    return out


def studio_files(name):
    """(mp4, png) of the studio movie the job drew for run `name`, or None."""
    b = os.path.join(LOG, name, "studio", f"tissue__{name}_exp29")
    return (b + ".mp4", b + ".png") if os.path.exists(b + ".mp4") else None


def title(name):
    """The 1-3 lines of LOG/<name>/title.yaml top-left on the studio movie and still of run `name`, in place, drawn by tools/exp29_text_overlay.py
    (VTK text, the record's font sizes and margin); a text box that meets a cell in any frame is not written (TITLE NOT WRITTEN is printed)."""
    import yaml
    import exp29_text_overlay as T
    lines = yaml.safe_load(open(os.path.join(LOG, name, "title.yaml")))["lines"]
    font_px, margin, _ = T.load_texts(REC)
    mp4, png = studio_files(name)
    r = T.render_text(lines, font_px)
    x0, y0 = T.placement(r, margin)
    try:
        T.overlay_movie(mp4, mp4, r["alpha"], x0, y0)
    except BaseException as e:                                # noqa: BLE001 -- SystemExit when the box meets a cell
        print(f"TITLE NOT WRITTEN {name}: {e}", flush=True)
        return
    T.overlay_still(png, png, r["alpha"], x0, y0)
    print(f"TITLE {name}: {lines}", flush=True)


def file_step(name, why, replace=None, studio=False):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import gui_drive as gd
    run = os.path.join(ROOT, "graphs_data", "tissue", name)
    i = gd._next_index() if replace is None else int(replace)
    old_why = open(gd._path("why", i)).read() if replace is not None and os.path.exists(gd._path("why", i)) else None
    mp4 = os.path.join(run, "movie.mp4")
    png = os.path.join(run, "3d.png")
    if studio:                                                # the studio movie of the job (exp29 movie 1's look) instead of the live movie
        mp4, png = studio_files(name)
    if os.path.exists(png):
        shutil.copy(png, gd._path("png", i))
    else:
        import imageio.v3 as iio
        iio.imwrite(gd._path("png", i), iio.imread(mp4, index=0, plugin="pyav"))
    shutil.copy(mp4, gd._path("mp4", i))
    shutil.copy(os.path.join(run, "spec.yaml"), gd._path("spec", i))
    m = json.load(open(os.path.join(run, "exp05_measures.json")))["summary"]
    L = m["last"]
    score = (f"score  after {m['hours']} h: {L['cells']} cells (from 20), {L['clusters']} clusters, {L['spheroids']} spheroids of 8 cells or more, "
             f"largest {L['largest']} cells (roundness {L['roundness']:.2f}, smallest over largest principal axis, 1 = round), "
             f"{L['in_clusters']:.2f} of the cells in a cluster, nearest-neighbour centre distance of clustered cells {L['nn_median']:.2f} cell diameters (1 = touching, below 1 = overlapping), "
             f" {L['growing']:.2f} growing; first spheroid at "
             f"{m['first_spheroid_h']} h; cluster sizes {m['sizes_last']}"
             + (f"; activator at the end: 10th percentile {m['activator_last']['p10']}, median {m['activator_last']['median']}, 90th {m['activator_last']['p90']}, "
                f"share of cells at or above the growth gate {m['activator_last']['active']}" if 'activator_last' in m else "")
             + (f"; TOPOLOGY of the bodies of 8+ cells (cells, tunnels b1, cavities b2, roundness): {m['bodies']}, tunnels {m['tunnels']}, cavities {m['cavities']} in all; "
                f"tunnels and cavities over time (h, b1, b2): {m['topology_t']}; deaths seen between stored rows {m['deaths_seen']}" if 'bodies' in m else ""))
    if old_why is not None:                                   # a re-render in place: the step's own text kept, one line added
        with open(gd._path("why", i), "w") as f:
            f.write(old_why.rstrip("\n") + "\n" + f"relook  {time.strftime('%Y-%m-%d %H:%M')}: {why}; " + score[len("score  "):] + "\n")
        gd.note(f"  exp05 {i:04d}  tissue/{name}  -- RE-RENDERED IN PLACE: {why}")
        print({"index": i, "score": score})
        return i
    with open(gd._path("why", i), "w") as f:
        f.write(f"time   {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"run    tissue/{name} (Plexus_Main.py -o generate)\n")
        f.write(score + "\n")
        f.write(f"why    {why}\n")
    gd.note(f"  exp05 {i:04d}  tissue/{name}  -- {why}")
    print({"index": i, "score": score})
    return i


def restyle(step, mp4, png, why):
    """File a redrawn movie (tools/exp29_studio.py --no-archive) INTO step `step`: its mp4 and png replaced, its spec and why text kept, one
    `relook` line appended to the why (Cedric, 2026-11-14: "rerender all steps in style of exp29 movie01")."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import gui_drive as gd
    i = int(step)
    shutil.copy(png, gd._path("png", i))
    shutil.copy(mp4, gd._path("mp4", i))
    w = gd._path("why", i)
    old = open(w).read().rstrip("\n") if os.path.exists(w) else ""
    with open(w, "w") as f:
        f.write(old + "\n" + f"relook  {time.strftime('%Y-%m-%d %H:%M')}: {why}\n")
    gd.note(f"  exp05 {i:04d}  -- REDRAWN IN PLACE: {why}")
    print({"index": i})


def rerun_local(item, device="cuda:0"):
    """One step of a rerun plan on the LOCAL GPU (Cedric, 2026-11-14: "use local cuda:0 to rerun first 30 steps over 1000 frames ... with a
    reservoir of 1000 cells"): generate, measure, studio movie, title, then the step re-filed in place (its why kept, a relook line added with
    what changed in the spec and the new score). `item`: {step, name, studio, title}. Returns (step, name, status)."""
    import shlex
    import yaml
    name, step = item["name"], int(item["step"])
    d = os.path.join(LOG, name)
    os.makedirs(os.path.join(d, "studio"), exist_ok=True)
    yaml.safe_dump({"lines": list(item["title"])}, open(os.path.join(d, "title.yaml"), "w"))
    env = dict(os.environ, PYTHONPATH=f"{ROOT}/src:{ROOT}/tools", MPLBACKEND="Agg", OMP_NUM_THREADS="8")
    env.pop("DISPLAY", None)
    py = sys.executable
    log = open(os.path.join(d, "local.log"), "a")
    stages = [("generate", [py, "-u", "Plexus_Main.py", "-o", "generate", f"tissue/{name}", "--device", device, "--force"], 14 * 3600),
              ("measure", [py, "-u", "tools/exp05_loop.py", "measure", name], 1800),
              ("studio", [py, "-u", "tools/exp29_studio.py", f"tissue/{name}", "--why", "exp05", "--no-archive", "--out-dir", os.path.join(d, "studio")]
               + shlex.split(item["studio"]), 4 * 3600),
              ("title", [py, "-u", "tools/exp05_loop.py", "title", name], 1800)]
    t0 = time.time()
    for nm, cmd, lim in stages:
        log.write(f"\n=== {nm} {time.strftime('%H:%M:%S')}: {' '.join(cmd)}\n"); log.flush()
        try:
            r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=lim)
        except subprocess.TimeoutExpired:
            return step, name, f"{nm} TIMEOUT after {lim} s"
        if r.returncode != 0:
            return step, name, f"{nm} FAILED (exit {r.returncode}, see {os.path.relpath(log.name, ROOT)})"
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import gui_drive as gd
    old = yaml.safe_load(open(gd._path("spec", step)))
    new = yaml.safe_load(open(os.path.join(ROOT, "config", "tissue", name + ".yaml")))
    og, ng = old["general"], new["general"]
    ch = [f"n_frames {og['n_frames']} -> {ng['n_frames']}", f"stored frames {og.get('record_cap')} -> {ng.get('record_cap')}",
          f"cell slots (sets.cell.buffer) {old['sets']['cell'].get('buffer')} -> {new['sets']['cell'].get('buffer')}"]
    if "junction" in new["sets"]:
        ch.append(f"junction slots {old['sets']['junction'].get('buffer')} -> {new['sets']['junction'].get('buffer')}")
    why = (f"RERUN on local {device} over at least 1000 frames with a reservoir of 1000 cells, drawn as the 4D panel draws its dots at the largest dot size (exp29_studio --dots 1.7) (Cedric, 2026-11-14), run tissue/{name}: "
           + ", ".join(ch) + f"; every other line as before; {round(time.time() - t0)} s")
    file_step(name, why, replace=step, studio=True)
    return step, name, "filed"


def rerun(plan_path, streams=3, device="cuda:0", only=None):
    """Every step of a rerun plan (json list of {step, name, mpm, n_frames, studio, title}) through rerun_local, `streams` at a time on one
    local GPU: the longest MPM run, then the point-cell runs in step order, then the other MPM runs, longest first."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    plan = json.load(open(plan_path))
    if only:
        plan = [p for p in plan if int(p["step"]) in only]
    mpm = sorted([p for p in plan if p["mpm"]], key=lambda p: -int(p["n_frames"]))
    pts = sorted([p for p in plan if not p["mpm"]], key=lambda p: int(p["step"]))
    plan = mpm[:1] + pts + mpm[1:]                            # the longest MPM run starts at once; the point cells (minutes each) next, so movies land early
    with ThreadPoolExecutor(max_workers=streams) as ex:
        futs = [ex.submit(rerun_local, p, device) for p in plan]
        for f in as_completed(futs):
            try:
                st, nm, status = f.result()
            except Exception as e:                            # noqa: BLE001
                st, nm, status = -1, "?", f"CRASH {e!r}"
            print(f"RERUN step {st} {nm}: {status}  ({time.strftime('%H:%M:%S')})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["measure", "file", "restyle", "title", "rerun"])
    ap.add_argument("--streams", type=int, default=3, help="rerun: runs at once on the local GPU")
    ap.add_argument("--device", default="cuda:0", help="rerun: the local device")
    ap.add_argument("--only", type=int, nargs="*", default=None, help="rerun: these steps only")
    ap.add_argument("--mp4", default=None, help="restyle: the redrawn movie")
    ap.add_argument("--png", default=None, help="restyle: its still")
    ap.add_argument("name")
    ap.add_argument("--why", default="")
    ap.add_argument("--replace", type=int, default=None, help="re-file into this existing step (a re-render) instead of a new one")
    a = ap.parse_args()
    if a.cmd == "measure":
        measure(a.name)
    elif a.cmd == "file":
        file_step(a.name, a.why, a.replace, studio=studio_files(a.name) is not None)
    elif a.cmd == "title":
        title(a.name)
    elif a.cmd == "rerun":                                    # NAME is the plan json here
        rerun(a.name, a.streams, a.device, a.only)
    elif a.cmd == "restyle":                                  # NAME is the step number here
        restyle(a.name, a.mp4, a.png, a.why)
    return 0


if __name__ == "__main__":
    sys.exit(main())
