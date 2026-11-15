#!/usr/bin/env python
"""The record movie of a step, drawn by the 3D view's own renderer -- so the main viewer and the 3D+t view show
the SAME picture (Cedric, 2026-10-27: "use the same default style in the main viewer as in the 3D default, so that
when we switch to 3D it does not rerender").

    PLEXUS_GUI=http://127.0.0.1:8835 python tools/view_movie.py <step>            # e.g. 3, the record's 0003
    PLEXUS_GUI=... python tools/view_movie.py <step> --seconds 10 --px 992 --dry   # render to /tmp only

The server opens the step's own spec in 3D (`/api/watch/open3d?i=`, seeded, the run's trajectory strided to the
view's budget), then every kept frame is rendered at the movie's own camera through `/api/watch/render` in the
`dots` look (the spec's type colours, flat fills, ink outlines, the depth shading) and encoded at
n_kept / seconds fps. The step's mp4 and png are replaced; the journal gets a line. `plotting.movie_look: dots`
in the spec means "keep the live renderer's dots" and this tool refuses the step.
"""
import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GUI = os.environ.get("PLEXUS_GUI", "http://127.0.0.1:8835")
REC = os.environ.get("PLEXUS_BUILDER", os.path.join(REPO, "experiments", "exp01_minisite"))


def _get(path):
    with urllib.request.urlopen(GUI + path, timeout=1800) as r:
        return r.read()


def _pos(step, rec_dir):
    """The watcher's `i=` is the POSITION of a step among the record's pictures, which is step - 1 only while
    the numbering has no gaps; a record with deleted steps (exp02: 0001-0002) would open the wrong scene."""
    names = sorted(f for f in os.listdir(os.path.join(rec_dir, "png")) if f.endswith(".png"))
    f = f"{step:04d}.png"
    return names.index(f) if f in names else step - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", type=int)
    ap.add_argument("--seconds", type=float, default=None, help="movie length (default the spec's duration_s, else 10)")
    ap.add_argument("--px", type=int, default=992)
    ap.add_argument("--rec", default=None, help="record folder as the watcher's rec= (experiments/<name>)")
    ap.add_argument("--dry", action="store_true", help="render to /tmp only, replace nothing")
    ap.add_argument("--right", type=int, default=None,
                    help="TWO PANELS: the record step whose frames go on the right of this step's, drawn at this step's camera "
                         "(a step that shows a part of the same scene, e.g. one compartment). The header stays on the left "
                         "and the scale bar on the right, so the pair carries one of each.")
    ap.add_argument("--zoom", type=float, default=None,
                    help="camera zoom for every frame, in place of the step's own (the 3D view's camera is fixed on the whole box, "
                         "so a small body is small in it); a re-render, no re-run of the simulation")
    ap.add_argument("--crop", default=None,
                    help="W,H pixels: cut each panel to its central W x H before it is saved (the empty margins go); "
                         "the header and the scale bar are kept, the header on the left panel and the bar on the right one")
    ap.add_argument("--look", default=None, choices=["dots", "surface"],
                    help="the 3D view's look: dots (the spec colours, ink, shading) or surface (the isosurface by type; "
                         "the watcher's dots / surface button). Default: the spec's own `plotting.movie_look` "
                         "(`surface` -> surface, anything else -> dots). Cedric, 2026-10-28: the debris flow as a surface.")
    a = ap.parse_args()
    import yaml
    s = f"{a.step:04d}"
    spec = yaml.safe_load(open(os.path.join(REC, "spec", s + ".yaml")))
    pl = spec.get("plotting") or {}
    _ml = str(pl.get("movie_look", "ink")).lower()
    a.look = a.look or ("surface" if _ml == "surface" else "dots")
    if _ml == "dots" and a.look == "dots":
        # an asked-for SURFACE is a different picture from the live renderer's dots, so it goes through
        print(f"step {s}: plotting.movie_look is dots -- the live renderer's movie stays; nothing done"); return 2
    seconds = a.seconds or float(pl.get("duration_s") or 10.0)
    rec = f"&rec={a.rec}" if a.rec else ""
    t0 = time.time()
    j = json.loads(_get(f"/api/watch/open3d?i={_pos(a.step, REC)}&full=1{rec}"))
    if j.get("error"):
        print("open3d failed:", j["error"]); return 1
    name, nk, cam = j["name"], int(j.get("n_kept") or 1), j.get("camera") or {}
    print(f"step {s} = {name}: scene ready in {j.get('seconds')} s, {nk} kept frames, camera {cam}", flush=True)
    az, el, zoom, roll = cam.get("azim", 20), cam.get("elev", 8), cam.get("zoom", 1.0), cam.get("roll", 0)
    if a.zoom:
        zoom = a.zoom
    tmp = tempfile.mkdtemp(prefix=f"view_movie_{s}_")

    def grab(step, name, nk, tag):
        files = []
        for k in range(nk):
            url = f"/api/watch/render?name={name}&azim={az}&elev={el}&zoom={zoom}&roll={roll}&frame={k}&cut=0&glass=0&look={a.look}&t={k}"
            for attempt in range(4):
                try:
                    png = _get(url); break
                except Exception as e:                               # noqa: BLE001
                    # THE VIEW IS SHARED: a click on another step in the watcher re-opens the server's one 3D view
                    # under this render (the first chain died at frame 141 with "no attribute _actors", 2026-10-27).
                    # Re-open this step and go on from the same frame.
                    print(f"  frame {k}: {type(e).__name__} {str(e)[:80]} -- re-opening the step (attempt {attempt + 1})", flush=True)
                    time.sleep(3)
                    jj = json.loads(_get(f"/api/watch/open3d?i={_pos(step, REC)}&full=1{rec}"))
                    if jj.get("error"):
                        raise SystemExit("open3d failed: " + jj["error"])
            else:
                raise SystemExit(f"frame {k} failed four times; giving up")
            f = os.path.join(tmp, f"{tag}{k:05d}.png"); open(f, "wb").write(png); files.append(f)
            if k % 50 == 0:
                print(f"  frame {k}/{nk}  {time.time() - t0:.0f} s", flush=True)
        return files

    frames = grab(a.step, name, nk, "L")
    import numpy as np
    from PIL import Image

    def panel(img, head, bar):
        """The picture cut to --crop around its centre, with the header patch pasted back at its top-left and the
        scale-bar patch at its bottom-right when asked: the margins are background, the two overlays are not."""
        h, w = img.shape[:2]
        ph, pw = int(0.07 * h), int(0.50 * w)
        bh, bw = h - int(0.88 * h), w - int(0.80 * w)
        hp, bp = img[:ph, :pw].copy(), img[h - bh:, w - bw:].copy()
        if a.crop:
            cw, ch = (int(v) for v in a.crop.split(","))
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            img = img[y0:y0 + ch, x0:x0 + cw].copy()
        bg = img[img.shape[0] // 2, 3].copy()
        # ONLY THE OVERLAY'S OWN (WHITE) PIXELS GO BACK, not the patch's rectangle: the rectangle also carries whatever else
        # lay under the overlay in the uncropped frame -- a corner of the box's wire frame -- and a cell that reaches the corner
        # of the crop would lose its edge to the rectangle.
        def put(dst, src):
            m = (src > 200).all(axis=2)
            dst[m] = src[m]
        if head:
            put(img[:ph, :pw], hp)
        elif not a.crop:
            img[:ph, :pw] = bg
        if bar:
            put(img[-bh:, -bw:], bp)
        elif not a.crop:
            img[-bh:, -bw:] = bg
        return img

    right = None
    if a.right:
        jr = json.loads(_get(f"/api/watch/open3d?i={_pos(a.right, REC)}&full=1{rec}"))
        if jr.get("error"):
            print("open3d failed (right):", jr["error"]); return 1
        print(f"right panel = step {a.right:04d} {jr['name']}: {jr.get('n_kept')} kept frames", flush=True)
        right = grab(a.right, jr["name"], int(jr.get("n_kept") or 1), "R")
    both = []
    for k, fl in enumerate(frames):
        L = np.array(Image.open(fl).convert("RGB"))
        if right is None:
            img = panel(L, True, True) if a.crop else L
        else:
            R = np.array(Image.open(right[min(k, len(right) - 1)]).convert("RGB"))
            img = np.concatenate([panel(L, True, False), panel(R, False, True)], axis=1)
        f = os.path.join(tmp, f"{k:05d}.png"); Image.fromarray(img).save(f); both.append(f)
    frames = both
    import imageio_ffmpeg, subprocess
    # `plotting.fps` is the render config's own frame rate (Cedric, 2026-10-29: "fps should be a param in the render config"); without it the
    # movie lasts `duration_s`. With `plotting.max_frames` (drawn frames kept) and `snap_budget_gb` it sets how smooth the 3D view's movie is.
    fps = float(pl["fps"]) if pl.get("fps") else max(1.0, nk / seconds)
    seconds = nk / fps
    out = os.path.join(tmp, "movie.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-framerate", f"{fps:.4f}", "-i",
                    os.path.join(tmp, "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-movflags", "+faststart", out], check=True)   # index (moov) before the picture data: the browser starts playing at once, not after a read to the end of the file
    print(f"movie: {nk} frames at {fps:.2f} fps = {seconds:g} s -> {out}  ({time.time() - t0:.0f} s)", flush=True)
    if a.dry:
        return 0
    import shutil
    shutil.copy(out, os.path.join(REC, "mp4", s + ".mp4")); shutil.copy(frames[-1], os.path.join(REC, "png", s + ".png"))
    with open(os.path.join(REC, "journal.txt"), "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')}  rerender  step {s} {name}: movie and still redrawn by the 3D view's renderer "
                f"(tools/view_movie.py, {nk} frames at {fps:.1f} fps, look {a.look}{', right panel step %04d' % a.right if a.right else ''})\n")
    print(f"replaced {REC}/mp4/{s}.mp4 and png/{s}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
