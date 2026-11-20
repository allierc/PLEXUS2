"""exp09_landscape.render's surfaces built in batches, and its trajectory read a row at a time: the same picture, drawn faster (2026-11-19).

`render` draws a stored row by calling tools/exp23_studio._cell_surface once per body -- the body's points counted on a voxel grid, the counts blurred, the
iso-surface of the blur (VTK flying edges), smoothed (VTK windowed sinc, "Taubin"), its normals: four pyvista calls per body, each a new VTK filter and a new
wrapped mesh -- keeps every body's mesh in a list, and merges them. Measured (experiments/exp10_oocyte_posterior/data/session/render_report.md): 92 s of a
97 s frame of exp10's 11,672 bodies; and a body costs more the more meshes are already kept (1.8 ms for the first 2,500 bodies of that frame, 10.6 ms for
bodies 7,500 to 10,000; 1.4 ms throughout when no mesh is kept). Here the SAME numbers are computed with nothing kept per body but its vertices (6.4 s for
that frame):

    the voxel counts     every body at once (numpy), a body's grid placed and sized as _cell_surface does it
    the blur             scipy's gaussian_filter itself, the bodies of one grid size stacked along a 4th axis that is not blurred (line by line the same sums)
    the iso-surface      ONE vtkFlyingEdges3D on ONE vtkImageData reused for every body (pyvista's `contour` builds a filter and wraps a mesh for each)
    smooth and normals   pyvista's smooth_taubin and compute_normals with _cell_surface's arguments, on a few million vertices of many bodies at a time
    the merge            the bodies' arrays joined in the order pyvista's merge leaves them in (its first mesh last before pyvista 0.46: asked of it, `_merge_order`)

Nothing is approximated: every vertex, normal and triangle is what _cell_surface gives, bit for bit, and `frame` checks it on each run before it is trusted --
the first row a process draws, CHECK bodies are rebuilt with _cell_surface and compared; on any difference, or any error, it says so and returns None, and
`render` draws the old way.

`Run` is what `render` reads of exp29_studio.MPMRun, with the three big arrays (pt__occ, pt__pid, pt__pos) memory-mapped inside the .npz and converted one row at
a time: MPMRun reads all of them (exp10: 16.6 GB read and 23 GB held by each of the 8 processes of a movie; here 3.7 GB held).

Used by exp09_landscape.render when exp09_landscape.FAST_DRAW is set (the default). The time of a frame is now the surfaces' own arithmetic, in proportion to the
bodies: 2.3 s for 5,545 bodies, 6.4 s for 11,672, 223 s for 554,500 (94 million vertices, 27 GB held).
"""
import os
import struct
import time
import traceback
import zipfile

import numpy as np

CHECK = 16                    # bodies rebuilt the old way and compared, on the first row a process draws
MAX_VOXELS = 48_000_000       # voxels blurred at a time (float32: 192 MB)
MAX_VERTICES = 4_000_000      # vertices smoothed at a time (the bodies whole)
LAST = {}                     # seconds per part of the last `frame` call, and its counts (the profile reads it)
_STATE = {"checked": False, "ok": True, "vtk": None, "order": None}


# ---- the trajectory, a row at a time ------------------------------------------------------------------------------------------------------------------------------
def _member(path, name):
    """Array `name` of an .npz as a read-only memory map of the file, or None when it cannot be mapped (a compressed archive, an object array)."""
    from numpy.lib import format as nf
    with zipfile.ZipFile(path) as z:
        try:
            info = z.getinfo(name + ".npy")
        except KeyError:
            return None
    if info.compress_type != zipfile.ZIP_STORED:
        return None
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        head = f.read(30)                                          # the member's local header: its name and extra field lengths at bytes 26 and 28
        if head[:4] != b"PK\x03\x04":
            return None
        n_name, n_extra = struct.unpack("<HH", head[26:30])
        f.seek(info.header_offset + 30 + n_name + n_extra)        # the .npy file starts here
        ver = nf.read_magic(f)
        if ver == (1, 0):
            shape, fortran, dtype = nf.read_array_header_1_0(f)
        elif ver == (2, 0):
            shape, fortran, dtype = nf.read_array_header_2_0(f)
        else:
            return None
        if dtype.hasobject:
            return None
        return np.memmap(path, dtype=dtype, mode="r", offset=f.tell(), shape=shape, order="F" if fortran else "C")


class _Rows:
    """An array [rows, ...] of the trajectory read a row at a time: `a[k]` is row k through `conv`; the last row asked is kept."""

    def __init__(self, arr, conv):
        self.arr, self.conv, self._k, self._v = arr, conv, None, None

    def __getitem__(self, k):
        k = int(k)
        if k != self._k:
            self._k, self._v = k, self.conv(np.asarray(self.arr[k]))
        return self._v


class _Shape:
    """Stands for an array `render` only asks the shape of (MPMRun.live, [rows, cell slots])."""

    def __init__(self, shape):
        self.shape = tuple(shape)


class Run:
    """What exp09_landscape.render reads of exp29_studio.MPMRun -- rows, live.shape, type_rgb, v, occ[k], pid[k], positions(k) -- with the same values, the point
    arrays read from the file a row at a time. For a finished run: the file must not be rewritten while it is drawn."""

    def __init__(self, d, spec, npz=None):
        import exp29_studio as S
        path = npz or os.path.join(d, "trajectory.npz")
        z = np.load(path)
        self.z, self.spec = z, spec

        def arr(n):                                                  # not mappable: read whole, as MPMRun does
            m = _member(path, n)
            return z[n] if m is None else m
        occ, pid, pos = arr("pt__occ"), arr("pt__pid"), arr("pt__pos")
        self.rows = occ.shape[0]
        self.live = _Shape((self.rows, arr("cell__occ").shape[1]))
        self.occ = _Rows(occ, lambda a: a.astype(bool))
        self.pid = _Rows(pid, lambda a: np.rint(a[..., 0]).astype(np.int64))
        self._pos = pos
        self.type_rgb = None                                         # per cell slot RGB (0-255) when `plotting.cell_colours: type`: MPMRun.__init__'s own lines
        _pl = (spec or {}).get("plotting") or {}
        if str(_pl.get("cell_colours", "")).lower() == "type" and _pl.get("colors") and "cell__node_type" in z.files and "cell__type_names" in z.files:
            import matplotlib.colors as _mc
            _names = [str(x) for x in z["cell__type_names"]]
            _tc = {str(k): np.asarray(_mc.to_rgb(v)) * 255.0 for k, v in _pl["colors"].items()}
            _nt = z["cell__node_type"]
            _nt = np.rint(_nt[-1] if _nt.ndim == 2 else _nt).astype(np.int64)
            self.type_rgb = np.stack([_tc.get(_names[t], np.array([200.0, 200.0, 200.0])) if 0 <= t < len(_names) else np.array([200.0, 200.0, 200.0]) for t in _nt])
        self.v = S.point_volume(spec)

    def positions(self, k):
        return np.asarray(self._pos[k]).astype(np.float64)


# ---- the bodies of a row, as render's loop takes them ---------------------------------------------------------------------------------------------------------------
def bodies(X, pid_row, idx, cut, ncell, min_points, period=None, sec=None):
    """The bodies `render`'s loop draws of one row, in its order: (slot ids [nb], points per body [nb], their points [sum, 3] one body after the other, centres
    [nb, 3]). `idx` the live points sorted by body, `cut` where a new body starts in it; `period` (lo, box size) of a periodic box (a body is shifted by whole
    periods, its centre inside the box); `sec` (z0, z1): only the bodies whose centre lies in that slice. None when the row has no live point."""
    if len(idx) == 0:
        return None
    Xs = X[idx]
    start = np.concatenate([[0], np.asarray(cut, np.int64)]).astype(np.int64)
    n = np.diff(np.concatenate([start, [len(idx)]]))
    cid = pid_row[idx[start]]
    ok = np.flatnonzero((n >= int(min_points)) & (cid < ncell))
    cz = np.empty((len(ok), 3))
    for j, b in enumerate(ok):                                   # a body's centre exactly as the loop's X[part].mean(0) (a reduceat sums pairwise: other last bits)
        cz[j] = Xs[start[b]:start[b] + n[b]].mean(0)
    sh = None
    if period is not None:
        p_lo, p_len = period
        sh = -p_len * np.floor((cz - p_lo) / p_len)
        cz = cz + sh
    if sec:
        keep = (float(sec[0]) <= cz[:, 2]) & (cz[:, 2] <= float(sec[1]))
        ok, cz = ok[keep], cz[keep]
        sh = None if sh is None else sh[keep]
    nk = n[ok]
    first = np.cumsum(nk) - nk
    P = Xs[np.repeat(start[ok] - first, nk) + np.arange(int(nk.sum()))]
    if sh is not None:
        P = P + np.repeat(sh, nk, axis=0)
    return cid[ok], nk, P, cz


# ---- the surfaces ---------------------------------------------------------------------------------------------------------------------------------------------------
def _table(c, kmax):
    """T[n]: what _cell_surface's np.add.at leaves in a float32 voxel that n points fell in, made by that very call."""
    T = np.zeros(kmax + 1, np.float32)
    for n in range(1, kmax + 1):
        np.add.at(T, np.arange(n, kmax + 1), c)
    return T


def _volume_slow(P, v, h, blur, iso, pad=4):
    """_cell_surface's own loop for one body, up to its blurred volume: (vol [sx, sy, sz] float32, lo, shape). For the body whose blur has not fallen below `iso`
    on a face of the box padded by 4 voxels (the loop then pads more)."""
    from scipy.ndimage import gaussian_filter
    while True:
        lo = P.min(0) - pad * h
        shape = np.ceil((P.max(0) + pad * h - lo) / h).astype(int) + 1
        ijk = np.floor((P - lo) / h).astype(int)
        vol = np.zeros(shape, np.float32)
        np.add.at(vol, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), v / h ** 3)
        vol = gaussian_filter(vol, blur)
        face = max(vol[0].max(), vol[-1].max(), vol[:, 0].max(), vol[:, -1].max(), vol[:, :, 0].max(), vol[:, :, -1].max())
        if face < iso or pad >= 40:
            break
        pad = min(40, pad + max(2, int(np.ceil(blur * np.sqrt(2.0 * np.log(face / iso))))))
    return vol, lo, shape


def _vtk(iso):
    """The reused VTK objects: an image and flying edges on it, set as pyvista's `contour(method="flying_edges")` sets its own."""
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D
    img = vtkImageData()
    fe = vtkFlyingEdges3D()
    fe.SetInputData(img)
    fe.SetComputeNormals(False)
    fe.SetComputeGradients(False)
    fe.SetComputeScalars(True)
    fe.SetInputArrayToProcess(0, 0, 0, 0, "f")                    # point data "f"
    fe.SetNumberOfContours(1)
    fe.SetValue(0, float(iso))
    return dict(img=img, fe=fe, iso=float(iso))


def _tris(polys):
    """[n, 3] point ids of a vtkCellArray of triangles (a copy)."""
    from vtkmodules.util.numpy_support import vtk_to_numpy
    if hasattr(polys, "GetConnectivityArray"):
        return vtk_to_numpy(polys.GetConnectivityArray()).reshape(-1, 3).astype(np.int64)
    return vtk_to_numpy(polys.GetData()).reshape(-1, 4)[:, 1:].astype(np.int64)      # VTK 8: [3, a, b, c] per triangle


def _poly(points, tris):
    """A pyvista mesh of triangles [M, 3] on `points`."""
    import pyvista as pv
    if hasattr(pv.PolyData, "from_regular_faces"):
        return pv.PolyData.from_regular_faces(points, tris)
    faces = np.empty((len(tris), 4), np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = tris
    return pv.PolyData(points, faces.ravel())


def _merge_order():
    """Where pyvista's merge puts its first mesh: "last" (pyvista before 0.46 appends the others, then the first), "first", or None (neither: not known here).
    Asked of the installed pyvista with three one-triangle meshes."""
    import pyvista as pv
    if _STATE["order"] is None:
        tri = lambda x: _poly(np.array([[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0], [x, 1.0, 0.0]], np.float32), np.array([[0, 1, 2]], np.int64))     # noqa: E731
        got = np.rint(np.asarray(pv.merge([tri(0.0), tri(10.0), tri(20.0)], merge_points=False).points)[::3, 0] / 10.0).astype(int).tolist()
        _STATE["order"] = {(0, 1, 2): "first", (1, 2, 0): "last"}.get(tuple(got), "unknown")
    return None if _STATE["order"] == "unknown" else _STATE["order"]


def surfaces(nk, P, v, h, iso=0.5, blur=1.6, pad=4):
    """The surfaces _cell_surface(P_b, v, h, iso=iso, blur=blur) gives for each body b (nk[b] points, one body after the other in P), merged as `render` merges
    them: dict(points [N, 3] float32, normals [N, 3] float32, tris [M, 3] point ids, npts [nb] vertices per body (0: no surface), ntri [nb], order: the bodies
    with a surface in the mesh's order, off [nb] and toff [nb]: a body's first vertex and first triangle)."""
    from scipy.ndimage import gaussian_filter
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    T0 = time.perf_counter()
    tm = dict(volumes=0.0, blur=0.0, contour=0.0, smooth=0.0, normals=0.0)
    nk = np.asarray(nk, np.int64)
    nb = len(nk)
    first = np.cumsum(nk) - nk
    lo = np.minimum.reduceat(P, first, axis=0) - pad * h
    shape = np.ceil((np.maximum.reduceat(P, first, axis=0) + pad * h - lo) / h).astype(int) + 1
    ijk = np.floor((P - np.repeat(lo, nk, axis=0)) / h).astype(int)
    size = shape.prod(1).astype(np.int64)
    order = np.lexsort((shape[:, 2], shape[:, 1], shape[:, 0]))             # the bodies by grid size: those of one size are blurred as one stack
    edge = np.flatnonzero(np.any(np.diff(shape[order], axis=0) != 0, axis=1)) + 1
    if _STATE["vtk"] is None or _STATE["vtk"]["iso"] != float(iso):
        _STATE["vtk"] = _vtk(iso)
    img, fe = _STATE["vtk"]["img"], _STATE["vtk"]["fe"]
    img.SetSpacing(h, h, h)
    pts, tri = [None] * nb, [None] * nb

    def contour(b, volF, lo_b, shape_b):
        """Body b's iso-surface from its blurred volume raveled x fastest (as _cell_surface's vol.ravel(order="F"))."""
        a = numpy_to_vtk(volF, deep=False)
        a.SetName("f")
        img.SetDimensions(int(shape_b[0]), int(shape_b[1]), int(shape_b[2]))
        img.SetOrigin(float(lo_b[0]), float(lo_b[1]), float(lo_b[2]))
        img.GetPointData().SetScalars(a)
        img.Modified()
        fe.Update()
        o = fe.GetOutput()
        if o.GetNumberOfPoints():
            pts[b] = vtk_to_numpy(o.GetPoints().GetData()).copy()
            tri[b] = _tris(o.GetPolys())

    c = v / h ** 3
    a0 = 0
    cum = np.cumsum(size[order])
    while a0 < nb:                                                        # a batch: bodies order[a0:a1], at most MAX_VOXELS voxels (one body at least)
        a1 = max(a0 + 1, int(np.searchsorted(cum, (cum[a0 - 1] if a0 else 0) + MAX_VOXELS, side="right")))
        ob = order[a0:a1]
        t = time.perf_counter()
        base = np.zeros(nb, np.int64)
        base[ob] = np.cumsum(size[ob]) - size[ob]
        nq = nk[ob]
        q = np.repeat(first[ob] - (np.cumsum(nq) - nq), nq) + np.arange(int(nq.sum()))      # the points of the batch's bodies
        bq = np.repeat(ob, nq)
        flat = base[bq] + (ijk[q, 0] * shape[bq, 1] + ijk[q, 1]) * shape[bq, 2] + ijk[q, 2]
        u, cnt = np.unique(flat, return_counts=True)
        big = np.zeros(int(size[ob].sum()), np.float32)
        big[u] = _table(c, int(cnt.max()))[cnt]
        tm["volumes"] += time.perf_counter() - t
        g0 = 0
        for g1 in list(edge[(edge > a0) & (edge < a1)] - a0) + [a1 - a0]:      # the bodies of one grid size: ob[g0:g1]
            og = ob[g0:g1]
            sx, sy, sz = (int(s_) for s_ in shape[og[0]])
            t = time.perf_counter()
            G = gaussian_filter(big[base[og[0]]:base[og[0]] + len(og) * sx * sy * sz].reshape(len(og), sx, sy, sz), (0.0, blur, blur, blur))
            face = np.maximum.reduce([G[:, 0].reshape(len(og), -1).max(1), G[:, -1].reshape(len(og), -1).max(1), G[:, :, 0].reshape(len(og), -1).max(1),
                                      G[:, :, -1].reshape(len(og), -1).max(1), G[:, :, :, 0].reshape(len(og), -1).max(1), G[:, :, :, -1].reshape(len(og), -1).max(1)])
            GF = np.ascontiguousarray(G.transpose(0, 3, 2, 1)).reshape(len(og), -1)
            tm["blur"] += time.perf_counter() - t
            t = time.perf_counter()
            for j, b in enumerate(og):
                if face[j] < iso or pad >= 40:
                    contour(b, GF[j], lo[b], shape[b])
                else:                                                     # the blur reaches the box's faces: _cell_surface's own loop, which pads more
                    vol, lo_b, shape_b = _volume_slow(P[first[b]:first[b] + nk[b]], v, h, blur, iso, pad)
                    contour(b, vol.ravel(order="F"), lo_b, shape_b)
            tm["contour"] += time.perf_counter() - t
            g0 = g1
        a0 = a1
    npts = np.array([0 if p is None else len(p) for p in pts], np.int64)
    ntri = np.array([0 if q_ is None else len(q_) for q_ in tri], np.int64)
    have = np.flatnonzero(npts > 0)
    out = np.concatenate([have[1:], have[:1]]) if _merge_order() == "last" else have       # the mesh's order: pyvista's merge of the bodies in the loop's order
    off, toff = np.zeros(nb, np.int64), np.zeros(nb, np.int64)
    off[out] = np.cumsum(npts[out]) - npts[out]
    toff[out] = np.cumsum(ntri[out]) - ntri[out]
    points = np.empty((int(npts.sum()), 3), np.float32)
    normals = np.empty((int(npts.sum()), 3), np.float32)
    tris = np.empty((int(ntri.sum()), 3), np.int64)
    a0 = 0
    cum = np.cumsum(npts[out])
    while a0 < len(out):                                                  # smoothing and normals, MAX_VERTICES at a time: bodies out[a0:a1], whole
        a1 = max(a0 + 1, int(np.searchsorted(cum, (cum[a0 - 1] if a0 else 0) + MAX_VERTICES, side="right")))
        ob = out[a0:a1]
        p0, t0 = int(off[ob[0]]), int(toff[ob[0]])
        t = time.perf_counter()
        m = _poly(np.concatenate([pts[b] for b in ob]), np.concatenate([tri[b] for b in ob]) + np.repeat(off[ob] - p0, ntri[ob])[:, None])
        for b in ob:
            pts[b] = tri[b] = None
        m = m.smooth_taubin(n_iter=25, pass_band=0.05)
        tm["smooth"] += time.perf_counter() - t
        t = time.perf_counter()
        m = m.compute_normals(auto_orient_normals=False, split_vertices=False)
        n_p, n_t = int(npts[ob].sum()), int(ntri[ob].sum())
        if m.n_points != n_p or m.n_cells != n_t:
            raise ValueError(f"smoothing and normals changed the mesh: {m.n_points} vertices and {m.n_cells} triangles of {n_p} and {n_t}")
        points[p0:p0 + n_p] = m.points
        normals[p0:p0 + n_p] = m.point_data["Normals"]
        tris[t0:t0 + n_t] = np.asarray(m.faces).reshape(-1, 4)[:, 1:] + p0
        tm["normals"] += time.perf_counter() - t
        a0 = a1
    LAST.update(tm, surfaces=time.perf_counter() - T0, vertices=int(len(points)), triangles=int(len(tris)), grid_sizes=int(len(edge) + 1), voxels=int(size.sum()),
                merge_first=_STATE["order"])
    return dict(points=points, normals=normals, tris=tris, npts=npts, ntri=ntri, order=out, off=off, toff=toff)


def check(nk, P, v, h, iso, blur, s, n=CHECK):
    """Are `n` bodies of the mesh `surfaces` built (the fewest and the most points, the rest spread evenly) bit for bit what exp23_studio._cell_surface gives?
    (ok, text)."""
    import exp23_studio as e23
    nk = np.asarray(nk, np.int64)
    first = np.cumsum(nk) - nk
    pick = sorted({int(np.argmin(nk)), int(np.argmax(nk)), *np.linspace(0, len(nk) - 1, max(2, n)).astype(int).tolist()})
    bad = []
    for b in pick:
        old = e23._cell_surface(P[first[b]:first[b] + nk[b]], v, h, iso=iso, blur=blur)
        n_p, n_t, p0, t0 = int(s["npts"][b]), int(s["ntri"][b]), int(s["off"][b]), int(s["toff"][b])
        if old is None:
            if n_p:
                bad.append(f"body {b}: no surface the old way, {n_p} vertices here")
            continue
        for what, a_, b_ in (("vertices", np.asarray(old.points), s["points"][p0:p0 + n_p]), ("normals", np.asarray(old.point_data["Normals"]), s["normals"][p0:p0 + n_p]),
                             ("triangles", np.asarray(old.faces).reshape(-1, 4)[:, 1:], s["tris"][t0:t0 + n_t] - p0)):
            if a_.shape != b_.shape:
                bad.append(f"body {b} ({nk[b]} points): {what} {a_.shape} the old way, {b_.shape} here")
            elif not np.array_equal(a_, b_):
                bad.append(f"body {b} ({nk[b]} points): {what} differ, at most {float(np.abs(a_.astype(np.float64) - b_.astype(np.float64)).max()):.3g}")
    return (not bad), (f"{len(pick)} bodies rebuilt by _cell_surface: identical" if not bad else f"{len(bad)} differences in {len(pick)} bodies: " + "; ".join(bad[:6]))


def frame(R, k, X, idx, cut, spec, sec, lo, hi, ncell, min_points, fill, idc, f_body, gval, gcol, h_vox, iso, blur):
    """One row for exp09_landscape.render, in place of its loop over the bodies and its merge: (the merged mesh with point data `rgb` and `idrgb` and its normals,
    {slot: centre}, the number of bodies drawn) -- or None, and `render` draws the row its own way: when the row has no body with a surface, when pyvista's merge
    order is not known here, when the check of the first row this process draws found a surface that is not _cell_surface's, or on any error (said once, then the
    process draws the old way)."""
    if not _STATE["ok"]:
        return None
    try:
        t0 = time.perf_counter()
        plo = spec.get("plotting") or {}
        got = bodies(X, R.pid[k], idx, cut, ncell, int(plo.get("min_points", min_points)), period=(lo, hi - lo) if plo.get("periodic") else None, sec=sec)
        if got is None or not len(got[0]):
            return None
        if _merge_order() is None:
            raise ValueError("pyvista's merge puts its first mesh neither first nor last")
        cid, nk, P, cz = got
        t1 = time.perf_counter()
        s = surfaces(nk, P, R.v, h_vox, iso=iso, blur=blur)
        if not _STATE["checked"]:
            _STATE["checked"] = True
            t = time.perf_counter()
            ok, text = check(nk, P, R.v, h_vox, iso, blur, s)
            LAST["check"] = time.perf_counter() - t
            print(f"  fast draw, row {k}: {text} ({LAST['check']:.1f} s)", flush=True)
            if not ok:
                raise ValueError("the surfaces are not _cell_surface's")
        t2 = time.perf_counter()
        out, npts = s["order"], s["npts"]
        if not len(out):
            return None
        merged = _poly(s["points"], s["tris"])
        merged.point_data["Normals"] = s["normals"]
        merged.GetPointData().SetActiveNormals("Normals")
        c_out = cid[out]
        if gval is None or k >= gval.shape[0]:                       # the fills: render's own three cases
            col = fill[c_out]
        elif gval.ndim == 3:
            col = np.stack([np.clip(np.rint((gval[k, c][:, None] * gcol).sum(0) * f_body[c]), 0, 255).astype(np.uint8) for c in c_out])
        else:
            col = np.stack([np.rint(fill[c] * (1.0 - gval[k, c]) + gcol * gval[k, c]).astype(np.uint8) for c in c_out])
        merged.point_data["rgb"] = np.repeat(col, npts[out], axis=0)
        merged.point_data["idrgb"] = np.repeat(idc[c_out], npts[out], axis=0)
        centre = {int(c): z_ for c, z_ in zip(cid, cz)}
        LAST.update(bodies=t1 - t0, assemble=time.perf_counter() - t2, frame=time.perf_counter() - t0, drawn=int(len(out)))
        return merged, centre, range(len(out))
    except Exception:                                             # noqa: BLE001
        _STATE["ok"] = False
        print(f"  fast draw, row {k}: NOT USED, this process draws the old way:\n{traceback.format_exc()}", flush=True)
        return None
