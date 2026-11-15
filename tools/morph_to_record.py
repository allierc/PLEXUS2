#!/usr/bin/env python
"""A finished morph run (plexus.morph: graphs_data/<group>/<name>/morph.npz) as a RECORD STEP drawn by the watcher.

    PLEXUS_GUI=http://127.0.0.1:8835 PLEXUS_BUILDER=<record folder> \\
        python tools/morph_to_record.py <group>/<name> --why "..." [--zoom 3.0] [--elev 20 --azim 30]

`plexus.morph` ends in its own glass renderer; the record shows every other run in the 3D view's look
(tools/view_movie.py), so this lifts the morph's frames into the engine's own run layout and archives it:

    graphs_data/<group>/<name>/trajectory.npz   mpm_particle__pos [T, N, 3], __occ [T, N], world_size -- STORED, not
                                                compressed, so the 3D view can memory-map the rows it keeps
    graphs_data/<group>/<name>/spec.yaml        the morph's own spec (plexus.morph.spec) with a plotting block: the
                                                record look, a camera and a zoom that fill the frame with the shape (the
                                                morph's box is 0.35 wide and its shape 0.14: framed on the box the
                                                shape is a fifth of the picture)
    config/<group>/<name>.yaml                  the same spec, where the watcher finds a step's spec by name
    <record>/{spec,png,mp4,why,caption}/NNNN.*  the next step, the movie drawn by tools/view_movie.py
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)

GD = os.path.join(REPO, "graphs_data")
REC = os.environ.get("PLEXUS_BUILDER", os.path.join(REPO, "experiments", "exp01_minisite"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="<group>/<name> under graphs_data")
    ap.add_argument("--why", required=True)
    ap.add_argument("--zoom", type=float, default=3.0, help="x the box framing (the shape is ~0.4 of the box)")
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=30.0)
    ap.add_argument("--color", default="0.80,0.55,0.40", help="the body's colour, r,g,b")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--no-movie", action="store_true", help="convert and archive the still only")
    a = ap.parse_args()

    from plexus import morph
    group, name = a.run.split("/")
    d = os.path.join(GD, group, name)
    z = np.load(os.path.join(d, "morph.npz"))
    frames, box = z["frames"], float(z["box"])
    meta = json.loads(str(z["meta"]))
    T, N, _ = frames.shape
    print(f"{a.run}: {T} frames of {N:,} points, grid {meta['n_grid']}, stages {meta.get('stages')}", flush=True)

    sp = morph.spec(N, box, int(meta["n_grid"]), T - 1, 0.002, 3.4e-4, differentiable=False)
    sp["general"].update(name=name, save_data=True)
    col = [float(v) for v in a.color.split(",")]
    sp["plotting"] = dict(renderer="vtk_points", background="#8d99a6", foreground="white", up_axis=1,
                          camera=dict(elev=a.elev, azim=a.azim), zoom=a.zoom, box_frame=True, render_3d="dots",
                          dot_size=1.0, colors={"cyto": col}, duration_s=a.seconds, slow_motion=1, movie_look="dots")
    yaml.safe_dump(sp, open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
    cfg = os.path.join(REPO, "config", group, name + ".yaml")
    os.makedirs(os.path.dirname(cfg), exist_ok=True)
    yaml.safe_dump(sp, open(cfg, "w"), sort_keys=False)

    # node_type + type_names: without them the replay (`-o plot`) finds no type column and colours the cloud by its
    # height at t = 0 instead of `plotting.colors` (the first green armadillo came out as a ramp, 2026-10-27).
    np.savez(os.path.join(d, "trajectory.npz"), world_size=np.array([box] * 3, np.float64),
             mpm_particle__pos=frames.astype(np.float32), mpm_particle__occ=np.ones((T, N), np.float32),
             mpm_particle__node_type=np.zeros(N, np.int64), mpm_particle__type_names=np.array(["cyto"]))
    print("trajectory.npz written (stored)", flush=True)

    # the next step of the record: its own files, then the movie by the 3D view
    idx = len([f for f in os.listdir(os.path.join(REC, "png")) if f.endswith(".png")]) + 1
    s = f"{idx:04d}"
    import shutil
    for k, ext in (("spec", ".yaml"),):
        shutil.copy(os.path.join(d, "spec.yaml"), os.path.join(REC, k, s + ext))
    with open(os.path.join(REC, "why", s + ".txt"), "w") as f:
        f.write(f"time   {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}\nspec   {name}\nrun    graphs_data/{a.run} "
                f"(plexus.morph on the cluster)\nwhy    {a.why}\n")
    open(os.path.join(REC, "png", s + ".png"), "wb").close()
    open(os.path.join(REC, "mp4", s + ".mp4"), "wb").close()
    with open(os.path.join(REC, "journal.txt"), "a") as f:
        f.write(f"{__import__('time').strftime('%H:%M:%S')}  morph  step {s} {name}: {T} frames of {N:,} points archived "
                f"(tools/morph_to_record.py); the movie follows from -o plot (the engine's renderer)\n")
    print(f"step {s} created in {REC}", flush=True)
    if a.no_movie:
        return 0
    # THE ENGINE'S OWN MOVIE RENDERER, not the 3D view's: `-o plot` replays trajectory.npz through live_movie.replay
    # (vtk_points, dots, the box frame, white overlays) -- the look of every generated step (step 6, the bench), which
    # Cedric asked the morph to share (2026-10-27). view_movie.py drew it through the 3D view instead (ink, shading).
    r = subprocess.run([sys.executable, os.path.join(REPO, "Plexus_Main.py"), "-o", "plot", f"{group}/{name}",
                        "--device", "cpu", "--no-describe"])
    if r.returncode:
        print("-o plot failed", r.returncode); return r.returncode
    shutil.copy(os.path.join(d, "movie.mp4"), os.path.join(REC, "mp4", s + ".mp4"))
    shutil.copy(os.path.join(d, "3d.png"), os.path.join(REC, "png", s + ".png"))
    sys.path.insert(0, HERE)
    import gui_drive
    os.environ["PLEXUS_BUILDER"] = REC
    gui_drive.HISTORY = REC
    print((gui_drive.caption(os.path.join(REC, "mp4", s + ".mp4"), index=idx).get("caption") or "")[:300])
    return 0


if __name__ == "__main__":
    sys.exit(main())
