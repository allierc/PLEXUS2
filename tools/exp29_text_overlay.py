#!/usr/bin/env python
"""exp29_text_overlay: the sweep text at the top-left of a finished exp29 movie, WITHOUT re-rendering its cells.

Cedric, 2026-10-31: "add a text top left with the operator-parameter to the render of the exp29 sweeps ... WITHOUT re-rendering: make a VTK
scene with the text, crop it and add it to the existing movie so that we do not see any difference with a new rendering".

HOW
  1. The text is drawn by VTK, as the studio draws its family labels (exp29_studio.draw_family_labels): a white, non-bold, unshadowed Arial text
     actor in an off-screen pyvista Plotter, at the movie's own pixel scale (1 text pixel = 1 movie pixel). pyvista's `add_text` doubles its
     `font_size` while `add_point_labels` does not, so the actor is given FONT_PX / 2 and its glyphs equal the labels' bit for bit (checked in
     `self_check`: the label "0.0035" at 24 px of the studio and the actor give 0 differing pixels).
  2. The scene is rendered TWICE with the same actors, once on black and once on the studio's ground (#8d99a6). The black render is the
     coverage of the glyph antialiasing, alpha = pixel / 255 (the text is white); the ground render is the check: ground + alpha x (white - ground)
     must give it back (`self_check` reports the largest difference).
  3. The alpha is cropped to its tight bounding box and composited over every frame of the finished movie: out = frame + alpha x (255 - frame).
     Over the uniform ground this is the pixel the studio would have written; over a frame's own pixels it is an ordinary blend, no halo, no box.
  4. The cell frames are decoded from the old mp4, the text added, and the movie written again with the studio's own writer settings
     (imageio ffmpeg, h264, yuv420p, quality 8, macro_block_size 16, the old fps and frame size). The still png gets the same alpha on its own
     (lossless) pixels. Both are replaced atomically (temp file + rename), so the watcher never reads half a file.

TEXT AND PLACEMENT
  `<record>/data/overlay_texts.yaml`: `font_px` (the first line, the other lines), `margin_px`, and per step `lines` (1 to 3) and an optional `x`
  (the ink's left edge, default the margin). VTK's "arial" has no square brackets (it draws [ ] as ( )), so the lines say operator(model). Line 1 is the first line's capital height `margin_px` below the frame's top edge and its ink starts
  `margin_px` from the left edge; the other lines follow at `LINE_GAP` baseline pitches. Before writing, the text's box (plus `PAD` px) is tested
  against EVERY frame of the old movie (a pixel more than `INK_TOL` levels off the ground in the 3-D panel counts as content): any hit aborts the
  step and nothing is replaced (`--allow-overlap` overrides).

CLEAN SOURCE (so that a second run never writes the text twice)
  The text is always added to a CLEAN movie: `data/pre_unify_backup_2026-10-31/redrawn_no_text/{mp4,png}/NNNN.mp4` (the 21 steps redrawn by
  exp29_studio on 2026-10-31 with every cell its own colour and the family labels, as the studio wrote them, before any text), else the
  original in `data/pre_unify_backup_2026-10-31/{mp4,png}/` (0059, whose run no longer exists). After a fresh `exp29_studio ... --replace N`
  the record's own file is the clean one: pass `--from-current`. `data/overlay_applied.json` records the sha256 of
  every movie this tool wrote; `--from-current` refuses a file whose hash is in it (the text is already on it).

USAGE (from the repository root; PYTHONPATH=src:tools MPLBACKEND=Agg; the neural-graph-linux python)
  python tools/exp29_text_overlay.py 61                 # one step, clean source from the backup
  python tools/exp29_text_overlay.py --all              # every step in the yaml
  python tools/exp29_text_overlay.py 61 --from-current  # after `exp29_studio ... --replace 61`
  python tools/exp29_text_overlay.py 61 --preview /tmp  # a png of the first frame with the text, nothing replaced
  python tools/exp29_text_overlay.py --self-check
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

os.environ.pop("DISPLAY", None)                      # VTK straight to EGL/OSMesa, as the studio
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RECORD = os.environ.get("PLEXUS_BUILDER", os.path.join(REPO, "experiments", "exp29_cell_tissue_2"))
GROUND = (0x8D, 0x99, 0xA6)                          # exp29_studio.GROUND, the ink look's grey
GROUND_HEX = "#8d99a6"
PANEL_W = 960                                        # exp29_studio render(): the 3-D panel; the insets' column is to its right
BACKUP = "pre_unify_backup_2026-10-31"
LINE_GAP = (29, 25, 25)                              # baseline pitch, px: line 1 -> 2, line 2 -> 3, line 3 -> 4
PAD = 6                                              # px kept free of content around the text's box
INK_TOL = 10                                         # levels (max over channels) off the ground that count as content
MAX_LINES = 4


# ------------------------------------------------------------------------------------------------------------------ the text, drawn by VTK
def _plotter(bg, size):
    import pyvista as pv
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background(bg)
    return pl


def _grab(pl):
    pl.render()
    im = np.ascontiguousarray(pl.screenshot(return_img=True)[..., :3])
    pl.close()
    return im


def _text(pl, s, x, y_from_bottom, font_px):
    """One white text actor, the studio label's look: Arial, not bold, no shadow. pyvista doubles `font_size`, so FONT_PX / 2."""
    assert font_px % 2 == 0, font_px
    pl.add_text(s, position=(int(x), int(y_from_bottom)), font_size=font_px // 2, color="white", font="arial", shadow=False)


def _metrics(font_px):
    """(baseline rise, capital height) in px of the VTK text at `font_px`: the ink of "H" put at y = 100 from the window's bottom."""
    pl = _plotter("black", (200, 200))
    _text(pl, "H", 50, 100, font_px)
    im = _grab(pl)
    ys = np.nonzero(im.max(2) > 0)[0]
    bottom, top = im.shape[0] - 1 - ys.max(), im.shape[0] - 1 - ys.min()
    return int(bottom - 100), int(top - bottom + 1)  # the ink's bottom row sits `rise` px above the actor's y; cap height in rows


def render_text(lines, font_px, size=(1100, 150)):
    """The alpha mask (float32 [h, w], 0 to 1) of the lines, cropped to its tight box, with the row of the crop's top and the row of line 1's
    capital top in the window (`crop_top`, `cap_top`, from which `placement` finds where the crop goes), and the two checks of the method:
    `channel_spread` (the black render's R, G, B agree: the coverage is one number per pixel) and `ground_err` (levels: the ground render
    rebuilt from the alpha against the real one)."""
    W, H = size
    fpx = [font_px[0]] + [font_px[1]] * (len(lines) - 1)
    met = {p: _metrics(p) for p in set(fpx)}
    rise1, cap1 = met[fpx[0]]
    base = [10 + cap1]                                # baseline row (from the window's top) of line 1: its capitals start at row 10
    for g in LINE_GAP[: len(lines) - 1]:
        base.append(base[-1] + g)
    assert base[-1] + 10 < H, (base, H)
    imgs = {}
    for bg in ("black", GROUND_HEX):
        pl = _plotter(bg, size)
        for s, p, b in zip(lines, fpx, base):
            rise = met[p][0]
            _text(pl, s, 20, H - 1 - b - rise + 1, p)   # the ink's bottom row (the baseline row) lands on row `b`
        imgs[bg] = _grab(pl)
    alpha_all = imgs["black"].astype(np.float32) / 255.0
    spread = float(np.abs(alpha_all[..., 0] - alpha_all[..., 1]).max() + np.abs(alpha_all[..., 1] - alpha_all[..., 2]).max())
    a = alpha_all.mean(2)
    ys, xs = np.nonzero(a > 0)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    alpha = np.ascontiguousarray(a[y0:y1 + 1, x0:x1 + 1])
    g = np.array(GROUND, np.float32)
    back = g + alpha[..., None] * (255.0 - g)                       # the ground render rebuilt from the alpha
    err = float(np.abs(back - imgs[GROUND_HEX][y0:y1 + 1, x0:x1 + 1].astype(np.float32)).max())
    off_ground = (np.abs(imgs[GROUND_HEX].astype(int) - np.array(GROUND)).max(2) > 0)       # every pixel of the ground render that is not the ground
    outside = int(off_ground.sum() - off_ground[y0:y1 + 1, x0:x1 + 1].sum())               # ... and lies outside the crop: must be 0
    return dict(alpha=alpha, cap_top=10, crop_top=int(y0), crop_left=int(x0), channel_spread=spread, ground_err=err, ink_outside_crop=outside,
                baselines=base)


# ------------------------------------------------------------------------------------------------------------------ compositing
def paste(frame, alpha, x0, y0):
    """frame (uint8 [H, W, 3]) in place: white text of coverage `alpha` with its top-left at (x0, y0)."""
    h, w = alpha.shape
    reg = frame[y0:y0 + h, x0:x0 + w].astype(np.float32)
    frame[y0:y0 + h, x0:x0 + w] = np.rint(reg + alpha[..., None] * (255.0 - reg)).astype(np.uint8)
    return frame


def placement(r, margin, x_ink=None):
    """(x0, y0) of the crop in the frame: its ink's left edge at `x_ink` (default `margin`), line 1's capitals `margin` px below the top."""
    x0 = margin if x_ink is None else int(x_ink)
    y0 = margin + (r["crop_top"] - r["cap_top"])
    return x0, y0


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def overlay_still(src_png, out_png, alpha, x0, y0):
    from PIL import Image
    im = np.array(Image.open(src_png).convert("RGB"))
    paste(im, alpha, x0, y0)
    tmp = out_png + ".tmp.png"
    Image.fromarray(im).save(tmp)
    os.replace(tmp, out_png)


def overlay_movie(src_mp4, out_mp4, alpha, x0, y0, allow_overlap=False, quality=8):
    """Every frame of `src_mp4` with the text, to `out_mp4` (atomic). Returns the report; writes nothing if the box meets content in any frame."""
    import imageio.v2 as iio
    r = iio.get_reader(src_mp4)
    meta = r.get_meta_data()
    fps, size = meta["fps"], tuple(meta["size"])
    h, w = alpha.shape
    g = np.array(GROUND, np.int16)
    ya, yb, xa, xb = max(y0 - PAD, 0), y0 + h + PAD, max(x0 - PAD, 0), x0 + w + PAD
    tmp = out_mp4 + ".tmp.mp4"
    wr = iio.get_writer(tmp, fps=int(round(fps)), quality=quality, macro_block_size=16)
    n, hits = 0, []
    try:
        for i, f in enumerate(r):
            f = np.ascontiguousarray(f)
            n_hit = int((np.abs(f[ya:yb, xa:xb].astype(np.int16) - g).max(2) > INK_TOL).sum())
            if n_hit:
                hits.append((i, n_hit))
            paste(f, alpha, x0, y0)
            wr.append_data(f)
            n += 1
    finally:
        wr.close()
        r.close()
    if hits and not allow_overlap:
        os.remove(tmp)
        raise SystemExit(f"{src_mp4}: the text box ({x0}, {y0}, {w} x {h}, pad {PAD}) meets content in {len(hits)} frames, first {hits[:3]} "
                         f"(frame, px): nothing written. Move it with `x:` in the yaml or shorten the text; --allow-overlap forces it.")
    os.replace(tmp, out_mp4)
    r2 = iio.get_reader(out_mp4)
    m2 = r2.get_meta_data()
    n2 = r2.count_frames()
    r2.close()
    return dict(frames=n, frames_out=n2, fps=fps, fps_out=m2["fps"], size=size, size_out=tuple(m2["size"]), codec=m2.get("codec"),
                overlap_frames=len(hits))


# ------------------------------------------------------------------------------------------------------------------ the record
def load_texts(record):
    p = os.path.join(record, "data", "overlay_texts.yaml")
    d = yaml.safe_load(open(p))
    font_px = tuple(int(v) for v in d.get("font_px", (24, 20)))
    margin = int(d.get("margin_px", 14))
    steps = {str(k).zfill(4): v for k, v in (d.get("steps") or {}).items()}
    for k, v in steps.items():
        ls = v["lines"]
        assert 1 <= len(ls) <= MAX_LINES, (k, ls)
        bad = [t for t in ls if "[" in t or "]" in t]
        if bad:                                         # the VTK "arial" has no square brackets: it draws [ ] as ( ), so the yaml must say what is drawn
            raise SystemExit(f"step {k}: square brackets in {bad}; write operator(model), VTK's font draws [ ] as ( ) anyway")
    return font_px, margin, steps


def sources(record, step, from_current):
    n = f"{int(step):04d}"
    cur_mp4, cur_png = (os.path.join(record, "mp4", n + ".mp4"), os.path.join(record, "png", n + ".png"))
    if from_current:
        return cur_mp4, cur_png
    b = os.path.join(record, "data", BACKUP)
    for sub in (os.path.join(b, "redrawn_no_text"), b):
        m, p = os.path.join(sub, "mp4", n + ".mp4"), os.path.join(sub, "png", n + ".png")
        if os.path.exists(m) and os.path.exists(p):
            return m, p
    raise SystemExit(f"no clean source for step {n} under {b}")


def apply_step(record, step, font_px, margin, spec, from_current=False, preview=None, allow_overlap=False, quality=8):
    n = f"{int(step):04d}"
    src_mp4, src_png = sources(record, n, from_current)
    man_path = os.path.join(record, "data", "overlay_applied.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}
    if from_current and preview is None and _sha(src_mp4) in {v["sha256_mp4"] for v in man.values()}:
        raise SystemExit(f"{src_mp4} is a movie this tool wrote (the text is on it already); redraw it with exp29_studio first, or drop --from-current")
    r = render_text(spec["lines"], font_px)
    x0, y0 = placement(r, margin, spec.get("x"))
    a = r["alpha"]
    info = dict(step=n, lines=spec["lines"], x=x0, y=y0, w=int(a.shape[1]), h=int(a.shape[0]), channel_spread=round(r["channel_spread"], 4),
                ground_err=round(r["ground_err"], 3), ink_outside_crop=r["ink_outside_crop"], source=os.path.relpath(src_mp4, record))
    if preview is not None:
        import imageio.v2 as iio
        rd = iio.get_reader(src_mp4)
        f = np.ascontiguousarray(rd.get_data(0))
        rd.close()
        paste(f, a, x0, y0)
        from PIL import Image
        out = os.path.join(preview, f"{n}_overlay_preview.png")
        Image.fromarray(f).save(out)
        info["preview"] = out
        return info
    out_mp4, out_png = os.path.join(record, "mp4", n + ".mp4"), os.path.join(record, "png", n + ".png")
    rep = overlay_movie(src_mp4, out_mp4, a, x0, y0, allow_overlap=allow_overlap, quality=quality)   # raises (writes nothing) when the box meets cells
    overlay_still(src_png, out_png, a, x0, y0)
    man[n] = dict(info, sha256_mp4=_sha(out_mp4), sha256_png=_sha(out_png), movie=rep)
    json.dump(man, open(man_path, "w"), indent=1, default=str)
    info["movie"] = rep
    return info


def self_check():
    """The text actor's glyphs equal the studio's family label's, and the alpha rebuilds the ground render."""
    ok = True
    pl = _plotter(GROUND_HEX, (960, 400))
    pl.add_point_labels(np.array([[0.0, 0.0, 0.0]]), ["0.0035"], font_size=24, text_color="white", shape=None, bold=False, show_points=False,
                        always_visible=True, name="family_labels")
    lab = _grab(pl)
    pl = _plotter(GROUND_HEX, (960, 400))
    _text(pl, "0.0035", 100, 100, 24)
    txt = _grab(pl)

    def crop(im):
        ys, xs = np.nonzero(np.abs(im.astype(int) - np.array(GROUND)).max(2) > 0)
        return im[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(int)
    a, b = crop(lab), crop(txt)
    same = a.shape == b.shape and int(np.abs(a - b).max()) == 0
    print("label vs text actor (0.0035, 24 px): shapes", a.shape, b.shape, "identical" if same else "DIFFERENT")
    ok &= same
    r = render_text(["pair_potential[cell_contact]  width", "-0.8 to 0.8", "fixed: overlap 2"], (24, 20))
    print("alpha: channel spread", round(r["channel_spread"], 4), "| ground rebuilt from alpha, max |diff| in levels", round(r["ground_err"], 3),
          "| ink outside the crop", r["ink_outside_crop"])
    ok &= r["ground_err"] <= 1.01 and r["ink_outside_crop"] == 0 and r["channel_spread"] < 1e-6
    print("SELF-CHECK", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("steps", nargs="*", help="step numbers (57 or 0057)")
    ap.add_argument("--all", action="store_true", help="every step of data/overlay_texts.yaml")
    ap.add_argument("--record", default=RECORD)
    ap.add_argument("--from-current", action="store_true", help="the record's own mp4/png are the clean source (a fresh exp29_studio --replace)")
    ap.add_argument("--preview", default=None, help="write NNNN_overlay_preview.png (first frame) to this folder and replace nothing")
    ap.add_argument("--allow-overlap", action="store_true", help="write even when the text box meets content")
    ap.add_argument("--quality", type=int, default=8, help="imageio quality of the written mp4 (the studio's own is 8)")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        sys.exit(self_check())
    font_px, margin, steps = load_texts(a.record)
    todo = sorted(steps) if a.all else [f"{int(s):04d}" for s in a.steps]
    if not todo:
        ap.error("give step numbers or --all")
    for n in todo:
        if n not in steps:
            raise SystemExit(f"step {n} has no entry in {a.record}/data/overlay_texts.yaml")
        info = apply_step(a.record, n, font_px, margin, steps[n], from_current=a.from_current, preview=a.preview, allow_overlap=a.allow_overlap,
                          quality=a.quality)
        print(json.dumps(info, default=str), flush=True)


if __name__ == "__main__":
    main()
