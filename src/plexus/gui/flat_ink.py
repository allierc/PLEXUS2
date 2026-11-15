"""The flat-ink look, shared by the watcher's 3-D view (gui/bio_view.py) and the record movies (tools/exp23_studio.py).

Cedric, 2026-10-13, on the lit sphere dots: "looks really metallic, could we have more of a bio look" -- flat fills,
one colour per object, and thin ink outlines in the object's own colour darkened, never black. The pieces:

    palette(ids)          a colour per cell id, the same id the same colour in every frame
    zbuffer / depth       the depth of the frame just rendered (no second render pass)
    ink(...)              the outlines, drawn on the screenshot from that depth and from the colours' hue
    round_length(...)     a scale bar's round length, the shortest whose bar is as wide as its own label
"""
import numpy as np

# VIVID, one per cell id (Cedric, 2026-10-13, his own set, after candy pastels, then muted tissue tones -- "too
# pastels I found" -- then deep natural tones, and the pastels again with "the choice of colour is too pastel"):
# red, orange, gold, teal, green, blue, violet, pink, cyan, lime
PALETTE = ("#E63946", "#F4A261", "#E9C46A", "#2A9D8F", "#43AA8B", "#4D96FF", "#9B5DE5", "#F15BB5",
           "#00BBF9", "#8AC926")
FLAT = dict(ambient=0.85, diffuse=0.15, specular=0.0)  # lit at most to the fill colour: no metal, no highlight
# A TOUCH OF SHADING FROM THE DEPTH BUFFER (Cedric, 2026-10-27, exp01 minisite 3D view: "add little bit of shadowing").
# The fills are flat, so a cube's three faces read as one colour; VTK's own lights cannot help a point cloud (each
# sprite is lit on its own). `shade` reads the screen-space normal of the smoothed depth and lights it from the upper
# left: faces turned up gain up to SHADE x 0.4 of their colour, faces turned away lose up to SHADE x 0.8; a face toward
# the camera is unchanged, so the flat look stays the baseline. 0 switches it off.
SHADE = 0.25
# ONE COLOUR FOR EVERY LINE (Cedric, 2026-11-14, exp06: "I do not like some black text and black on the plot ... render all steps with white
# instead of black"): an RGB tuple 0-255 paints the outlines in it; None (the default) keeps each pixel's own colour darkened.
LINE_RGB = None
LIGHT =np.asarray([-0.3, 0.85, 0.45], np.float64) / np.linalg.norm([-0.3, 0.85, 0.45])


def palette(ids) -> np.ndarray:
    """[n, 3] colours in [0, 1], one per cell id (`ids` integers), keyed on the id modulo the palette's 10 entries."""
    from matplotlib.colors import to_rgb
    pal = np.array([to_rgb(c) for c in PALETTE])
    return pal[np.mod(np.asarray(ids, np.int64), len(pal))]


def zbuffer(render_window):
    """The last render's z-buffer as an [H, W] array in [0, 1] (1 = nothing drawn), top row first like a
    screenshot, or None. Translucent surfaces write no depth here; opaque ones do."""
    try:
        from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
        from vtkmodules.util.numpy_support import vtk_to_numpy
        f = vtkWindowToImageFilter()
        f.SetInput(render_window); f.ReadFrontBufferOff(); f.SetInputBufferTypeToZBuffer(); f.Update()
        o = f.GetOutput()
        w, h, _ = o.GetDimensions()
        return vtk_to_numpy(o.GetPointData().GetScalars()).reshape(h, w)[::-1].astype(np.float64)
    except Exception:                                            # noqa: BLE001
        return None


def depth(zb, camera):
    """World distance from the camera along the view, per pixel, from a z-buffer and the camera that drew it."""
    near, far = camera.clipping_range
    if camera.parallel_projection:
        return near + zb * (far - near)
    return 2.0 * near * far / (far + near - (2.0 * zb - 1.0) * (far - near))


def ink(img, D, scene, wpp, dot_r=None, colour_origin=(0.0, 0.0, 0.0), contacts=True, per_object=False):
    """`img` shaded by `shade`, with the outlines of `ink_masks` drawn on it. THE MASKS READ THE UNSHADED IMAGE:
    the contact pass inks a colour step between neighbouring pixels, and the hillshade of a sprite cloud's bumpy
    depth puts a small colour step on every dot -- read after shading, every dot got a black speck (Cedric,
    2026-10-27: "these black dots were not there before")."""
    masks = ink_masks(img, D, scene, wpp, dot_r=dot_r, colour_origin=colour_origin, contacts=contacts, per_object=per_object)
    return apply(shade(img, D, scene, wpp, dot_r=dot_r), *masks)


def shade(img, D, scene, wpp, dot_r=None):
    """The fills shaded by the depth buffer's normal (see SHADE): `D` the per-pixel depth in world units, `scene` the
    pixels that belong to drawn objects, `wpp` world units per pixel, `dot_r` the dots' world radius (the depth of a
    sprite cloud is smoothed over one dot so the normal is the cloud's, not each sphere's).

    EACH SURFACE IS SHADED FROM ITS OWN DEPTH (Cedric, 2026-11-07: "light yellow and light red regions that are wrong"). One blur over the whole
    frame mixed the depth of a ball with that of the ball in front of it, so the band of the far ball next to the contact got a normal that
    belongs to the step between them and was lit light. The scene is cut into patches at the depth steps (a jump over 3 dot radii between
    neighbours) and at the colour steps (the direction of the colour, as the ink tells objects apart); each patch is blurred and differentiated
    on its own. Within a blur of a patch's edge the normal is the edge's, not a face's: those pixels take the factor of the nearest pixel of the same patch whose normal is
    trusted, the pixels on the cuts that of the nearest patch, so no light ring is left between the ink and a darkened fill. A line drawn in front (the box
    edge, a scale bar) is 1-2 px wide: a 5 x 5 median of the depth drops it, so it cuts a ball in two patches and tilts no normal."""
    if SHADE <= 0.0 or not scene.any():
        return img
    from scipy import ndimage as ndi
    sig = max(1.5, float(dot_r) / float(wpp)) if dot_r else 2.0
    near = ndi.distance_transform_edt(~scene, return_distances=False, return_indices=True)       # the background takes the depth of the nearest drawn pixel, so the median does not pull a silhouette toward 0
    Dm = ndi.median_filter(D[near[0], near[1]], size=5, mode="nearest")
    r = float(dot_r) if dot_r else 2.0 * float(wpp)
    tstep = max(3.0 * r, 4.0 * float(wpp))
    rgb = img[..., :3].astype(np.float64)
    un = np.linalg.norm(rgb, axis=-1, keepdims=True)
    u = np.where(un > 12.0, rgb / np.maximum(un, 1e-9), 0.0)
    cut = np.zeros(scene.shape, bool)
    for ax in (0, 1):
        e = scene & np.roll(scene, 1, ax) & ((np.abs(Dm - np.roll(Dm, 1, ax)) > tstep) | ((u * np.roll(u, 1, ax)).sum(-1) < 0.97))
        cut |= e | np.roll(e, -1, ax)                            # both pixels of a pair across a step
    lab, n = ndi.label(scene & ~cut)
    f = np.ones(scene.shape, np.float64)
    ok = np.zeros(scene.shape, bool)
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        pad = tuple(slice(max(0, k.start - 3 * int(sig + 1)), k.stop + 3 * int(sig + 1)) for k in sl)    # room for the blur beyond the patch
        m = lab[pad] == i
        if m.sum() < 30:
            continue
        w = ndi.gaussian_filter(m.astype(np.float64), sig)
        Ds = ndi.gaussian_filter(np.where(m, Dm[pad], 0.0), sig) / np.maximum(w, 1e-9)
        gy, gx = np.gradient(Ds)                                 # depth change per pixel, in world units
        gx = gx / float(wpp); gy = gy / float(wpp)               # image x to the right, image y DOWN the rows
        nrm = np.stack([-gx, -gy, np.ones_like(gx)], -1)         # the surface normal; up = -rows, so a floor whose depth FALLS down the image (gy < 0) points up
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        fi = np.clip(1.0 + SHADE * ((nrm * LIGHT).sum(-1) - LIGHT[2]), 1.0 - 0.8 * SHADE, 1.0 + 0.4 * SHADE)    # a face toward the camera: exactly 1
        good = m & (w > 0.9)
        if not good.any():
            continue
        _, ix = ndi.distance_transform_edt(~good, return_distances=True, return_indices=True)
        f[pad][m] = fi[ix[0], ix[1]][m]                          # the whole patch, its rim included, takes the factor of its nearest trusted pixel
        ok[pad] |= m
    if not ok.any():
        return img
    dist, idx = ndi.distance_transform_edt(~ok, return_distances=True, return_indices=True)       # the pixels on the cuts: the nearest patch's factor
    use = scene & (dist <= 4.0)
    fe = f[idx[0], idx[1]]
    out = img.copy()
    out[..., :3][use] = np.clip(rgb[use] * fe[use][:, None], 0, 255).astype(img.dtype)
    return out


def apply(img, thin, cell, hard):
    """The three line weights of `ink_masks` drawn on `img`: rims and folds darkened 35 %, contacts between cells
    60 %, silhouettes 70 % (each the pixel's own colour, never black). With LINE_RGB set, every line is that one colour instead."""
    out = img.copy()
    if LINE_RGB is not None:
        out[..., :3][thin | cell | hard] = np.asarray(LINE_RGB, img.dtype)
        return out
    out[..., :3][thin] = (img[..., :3][thin] * 0.65).astype(img.dtype)
    out[..., :3][cell] = (img[..., :3][cell] * 0.4).astype(img.dtype)
    out[..., :3][hard] = (img[..., :3][hard] * 0.3).astype(img.dtype)
    return out


def ink_masks(img, D, scene, wpp, dot_r=None, colour_origin=(0.0, 0.0, 0.0), contacts=True, per_object=False):
    """THE INK's three line masks (thin, cell, hard) on a screenshot `img`, from the world depth `D` of its pixels and the mask `scene` of pixels
    that belong to the drawn objects (False: the background, the panels and text, a scale bar in front).

    Three weights of line, each the pixel's own colour darkened, never black: a dot's rim over the dot behind it
    (a 4-neighbour farther by more than 0.8 of a dot's radius `dot_r`, drawn only on dots at least 6 px wide; on a
    surface, `dot_r` None, a fold farther by more than 2 px of world, `wpp` world units per pixel) darkened 35 %; the
    contact between two cells (a neighbour of another cell's colour, below; off with `contacts=False`) 60 %; the
    silhouette against the background 70 % and two pixels wide. A silhouette taken from any depth jump over 4 dot
    radii, and rims on 4 px dots, inked exp25_divide's lattice of points -- seen along its rows, every gap between two
    rows looks deep into the cell -- as solid dark stripes.

    A CELL'S COLOUR, WHATEVER ITS SHADE: the direction of the pixel's colour measured from `colour_origin` (0-255):
    the background for dots (the atmospheric cue only slides a dot toward the background), black for a surface (its
    0.15 diffuse only scales the fill). Comparing raw colours inked every depth step of one cell's dots as a contact.

    `per_object` (`plotting.ink_per_object`, Cedric, 2026-11-06: "the ink contour is global over the three balls"): every object gets
    its OWN outline. The contact between two objects is then drawn as the silhouette is, 70 % and two pixels wide, on the nearer
    object, instead of the 60 %, one-pixel `cell` line, so a ball in front of another is circled whole, not only the union of the
    balls. Objects are told apart by colour, as the contact line already does, so two touching objects of one colour still read as one."""
    r = float(dot_r) if dot_r else 2.0 * wpp
    t1 = max(0.8 * r, 1.5 * wpp)
    thin_on = dot_r is None or r / wpp >= 3.0                    # a rim on a dot under 6 px wide is all rim
    rgb = img[..., :3].astype(np.float64)
    u = rgb - np.asarray(colour_origin, float)[:3]
    un = np.linalg.norm(u, axis=-1, keepdims=True)
    u = np.where(un > 12.0, u / np.maximum(un, 1e-9), 0.0)      # within 12 of the origin: no colour to compare
    thin = np.zeros(scene.shape, bool)
    cell = np.zeros(scene.shape, bool)
    hard = np.zeros(scene.shape, bool)
    Dn = np.where(scene, D, np.inf)                              # the background is infinitely far
    for k in (1, 2):
        for ax, sh in ((0, k), (0, -k), (1, k), (1, -k)):
            nb_D = np.roll(Dn, sh, axis=ax)
            jump = nb_D - Dn                                     # > 0: the neighbour is farther
            if k == 1 or thin_on:                                # two pixels wide only on dots of 6 px and up
                hard |= scene & ~np.isfinite(nb_D)               # the background next to it: the silhouette
            if k == 1 and thin_on:
                thin |= scene & (jump > t1)
            if contacts and (thin_on or per_object) and (k == 1 or thin_on):      # per_object: on dots under 6 px too (an MPM body is a cloud of 3.6 px dots, thin_on False); as wide as the silhouette, 2 px from 6 px dots up, else 1 px (it was 2 px on every dot: a contact thicker than the outline of the ball against the background)
                nb_u = np.roll(u, sh, axis=ax)
                touch = scene & np.roll(scene, sh, axis=ax) & ((u * nb_u).sum(-1) < 0.97) & (jump >= 0)
                if per_object:
                    hard |= touch
                else:
                    cell |= touch
    return thin, cell, hard


def round_length(px_per_unit, label_px):
    """THE BAR AS WIDE AS ITS LABEL (Cedric, 2026-10-13: a 1 mm bar 430 px wide under a 60 px label "1 mm" reads as a
    rule, not a scale): the shortest of 1, 2, 2.5, 5 x 10^k units whose bar (`px_per_unit` pixels per unit) is at least
    as wide as its own label, `label_px(length)` pixels."""
    k0 = int(np.floor(np.log10(max(20.0 / max(px_per_unit, 1e-30), 1e-30))))   # 20 px: under any label's width
    for k in range(k0, k0 + 6):
        for f in (1.0, 2.0, 2.5, 5.0):
            if f * 10.0 ** k * px_per_unit >= label_px(f * 10.0 ** k):
                return f * 10.0 ** k
    return 10.0 ** (k0 + 6)
