"""Archive a finished cluster run (graphs_data/<group>/<name>) as the NEXT step of a builder record, the way
`gui_drive.py opencycle` archives a local one: its movie, still and spec, the why, the journal line, and then the
config's own render option -- unless the spec says `plotting.movie_look: dots`, the movie and still are redrawn by
tools/view_movie.py through the watcher's 3D renderer (Cedric, 2026-10-28: "render option should be in config file
and used by the watcher render"; the first r1 archive copied the live renderer's dots for specs that ask for the 3D
look). `plotting.view_3d` picks what the view contours; `movie_look` is read here and in gui_drive, nowhere else.

    PLEXUS_BUILDER=<record> python tools/archive_run.py si_material/<name> --why "..." [--job LSF_ID] [--no-caption]
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="<group>/<name> under graphs_data")
    ap.add_argument("--why", required=True)
    ap.add_argument("--job", default="", help="LSF job id, for the why file")
    ap.add_argument("--no-caption", action="store_true")
    ap.add_argument("--replace", type=int, default=0, help="overwrite this existing step in place instead of appending a new one")
    a = ap.parse_args()
    rec = os.environ["PLEXUS_BUILDER"]
    group, name = a.run.split("/")
    sys.path.insert(0, REPO)
    from plexus.paths import graphs_data_path
    d = os.path.join(graphs_data_path(), group, name)
    # A RUN OF FIELDS ALONE has no point movie: its picture is `movie_<field>.mp4` and `fig_<field>_final.png`, and when there is one field those are the step's.
    movie, still = "movie.mp4", "3d.png"
    fm = sorted(f for f in os.listdir(d) if f.startswith("movie_") and f.endswith(".mp4"))
    if not os.path.isfile(os.path.join(d, movie)) and len(fm) == 1:
        movie, still = fm[0], "fig_" + fm[0][len("movie_"):-4] + "_final.png"
    for f in (still, movie):
        if not os.path.isfile(os.path.join(d, f)):
            print(f"archive_run: {d}/{f} is missing"); return 1
    idx = a.replace or len([f for f in os.listdir(os.path.join(rec, "png")) if f.endswith(".png")]) + 1
    s = f"{idx:04d}"
    if a.replace:                                                       # the old step's story and caption describe the old run
        for sub in ("story", "caption"):
            for f in os.listdir(os.path.join(rec, sub)):
                if f.startswith(s):
                    os.remove(os.path.join(rec, sub, f))
    shutil.copy(os.path.join(d, still), os.path.join(rec, "png", s + ".png"))
    shutil.copy(os.path.join(d, movie), os.path.join(rec, "mp4", s + ".mp4"))
    spec = os.path.join(REPO, "config", group, name + ".yaml")
    shutil.copy(spec, os.path.join(rec, "spec", s + ".yaml"))
    run = f"graphs_data/{a.run}" + (f" (cluster, LSF {a.job})" if a.job else "")
    with open(os.path.join(rec, "why", s + ".txt"), "w") as f:
        f.write(f"time   {time.strftime('%Y-%m-%d %H:%M:%S')}\nspec   {name}\nrun    {run}\nwhy    {a.why}\n")
    pl = (yaml.safe_load(open(spec)).get("plotting") or {})
    line = f"{time.strftime('%H:%M:%S')}  land  step {s} {name} archived (tools/archive_run.py)"
    with open(os.path.join(rec, "journal.txt"), "a") as f:
        f.write(line + "\n")
    print(line, flush=True)
    if str(pl.get("movie_look", "ink")).lower() != "dots":
        r = subprocess.run([sys.executable, os.path.join(HERE, "view_movie.py"), str(idx)],
                           env=dict(os.environ, PLEXUS_BUILDER=rec))
        if r.returncode:
            print("view_movie failed", r.returncode); return r.returncode
    if not a.no_caption:
        sys.path.insert(0, HERE)
        import gui_drive
        gui_drive.HISTORY = rec
        print((gui_drive.caption(os.path.join(rec, "mp4", s + ".mp4"), index=idx).get("caption") or "")[:300])
    return 0


if __name__ == "__main__":
    sys.exit(main())
