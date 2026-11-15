"""Reaction-diffusion ON THE CELL GRAPH, and the two couplings between chemistry and shape.

The diffusion is not on a grid. The CELLS are the nodes, `cell_neighbours` is the graph, and the
Laplacian runs over shared faces -- so the domain grows and rewires as the tissue divides, which a
fixed lattice cannot do. That is why these are `set=cell` operators rather than `field` ones.

In the order they appear below:

    cell_geometry         aggregate  vertex mesh -> per-cell centroid, area, perimeter, volume
    cell_neighbours       rewire     the cell adjacency the Laplacian runs on
    seed_cell_chem        seed       the initial morphogen field
    cell_chem_diffuse     lateral    morphogen exchange between neighbouring cells
    cell_chem_react       lateral    the local reaction: the pattern-forming nonlinearity
    cell_grow             lateral    morphogen -> growth: the chemistry-to-shape coupling
    interface_tension     lateral    a purse-string line tension on the activator interface
    interface_push        lateral    the term that is NOT physics, kept separate on purpose
    cell_chem_from_shape  lateral    shape -> chemistry: the other half of the loop
    cell_shape_probe      lateral    one shape scalar per cell, published for a discriminator

then the models -- different hypotheses in one slot, not different arithmetic:

    seed_cell_chem       scatter (default), noise, patch, cones, simplex, uniform
    cell_chem_diffuse    graph_laplacian (default implementation), interface_weighted (a model)
    cell_chem_react      gray_scott, brusselator, gierer_meinhardt, rock_paper_scissor,
                         gray_scott_coupled, source_decay (a
                         sustained source), balaskas (a gene circuit reading a signal)
    cell_grow            default, sizer, balance, timer, stretch
    cell_chem_from_shape apical_area, curvature, pressure, tension
    cell_shape_probe     shape_index, aspect

`interface_tension` and `interface_push` are two operators and must stay two. Written as one, they
carried a purse-string line tension -- ordinary vertex-model physics -- MINUS an energy that falls
as activator-high cells move outward. The second term does not model a force: it pays the tissue to
produce the morphology a search is looking for, so a run carrying it can only ever be a control.
One name over both makes that impossible to see in a schedule.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch
from plexus.models.base import Aggregate, Lateral, Rewire, Structural
from plexus.models.registry import register_operator
from plexus.models.base import Lateral
from plexus.operators.vertex_ops import face_geometry_3d, resolve_cell_set, wedge_apex


def _chan(params, who, n_species=2):
    """The FIRST COLUMN of the contiguous species span this instance owns.

    `chan` is a COLUMN index, not a species index, though it reads like one. A reaction model occupies
    `n_species` ADJACENT columns of `chem` -- Gray-Scott two (a, u), May-Leonard three (u, v, w),
    a coupled pair of Gray-Scott systems four -- so the second two-species system starts at
    `chan: 2`, not `chan: 1`. `chan: 1` is the natural thing to write and would have put the second
    system's activator ON THE FIRST SYSTEM'S SUBSTRATE: both would run, both would look alive, and
    they would be driving one shared column through a coupling nobody wrote.

    AN EVEN-ONLY RULE ON `chan` IS WRONG the moment a three-species model exists: May-Leonard
    tiles at 0, 3, 6, and an even-only guard rejects a correct spec while accepting `chan: 2` for a
    three-species system, which overlaps. The rule is "a multiple of this model's own width", which
    is the actual tiling, and it reduces to the even rule for two-species models.

    The BOUNDS check cannot happen here -- `chem`'s width is not known until forward -- so it is
    `_span` that raises on a span running off the end of the buffer.
    """
    c = int(params.get("chan", 0))
    n = int(n_species)
    if c < 0 or (n > 0 and c % n):
        ok = [i * n for i in range(4)]
        raise ValueError(
            f"{who}: chan={c} does not start a {n}-species span. `chan` is a COLUMN index, not a "
            f"species index -- this model owns {n} adjacent columns of `chem`, so its systems tile "
            f"at {ok}... A chan that is not a multiple of {n} overlaps the neighbouring system and "
            f"silently couples the two through a shared column.")
    return c


def cap_target_rate(ds, s_prev, V0f, V0f_init, cap_v, dt, dims=3):
    """The growth rate on the linear scale `ds`, gated so the TARGET `V0f` stops at `cap_v`.

    The target's own rate is dV0f/dt = dims V0f_init s^(dims-1) ds (the chain rule `cell_grow` emits), so the
    largest ds that keeps V0f <= cap_v this step is (cap_v - V0f) / (dt dims V0f_init s^(dims-1)), and never below
    zero: a gate, not a pull -- a cell already over the cap holds, it is not shrunk."""
    dvds = (dims * V0f_init * s_prev ** (dims - 1)).clamp(min=1e-12)
    room = (cap_v - V0f).clamp(min=0.0) / max(float(dt), 1e-12) / dvds
    return torch.minimum(ds, room)


def _span(chem, chan, n, who):
    """The `n` columns starting at `chan`, or a loud failure -- the bounds half of `_chan`."""
    if chan + n > chem.shape[1]:
        raise ValueError(
            f"{who}: needs columns {chan}..{chan + n - 1} of `chem` but the block is only "
            f"{chem.shape[1]} wide. Widen `sets.<set>.state.chem.width` to at least {chan + n}.")
    return [chem[:, chan + k] for k in range(n)]


def _emit(chem, chan, terms, rate, occ):
    """A delta that is ZERO IN EVERY COLUMN THIS INSTANCE DOES NOT OWN.

    That is what makes two reaction instances additive rather than mutually overwriting: the engine
    sums operator deltas, so an instance which wrote its own columns and left the others UNSET
    would be fine, but one that wrote zeros over another's work would silently erase it. Building
    the full-width zero tensor and filling only this span is the cheap way to be certain.
    """
    out = torch.zeros_like(chem)
    for k, t in enumerate(terms):
        out[:, chan + k] = rate * t
    return out * occ


@register_operator("cell_geometry", set="cell", kind="aggregate", family="hierarchy", title="measure cell geometry",
                   equation=r"""$$\mathbf{cen}_f=\frac{1}{n_f}\sum_{e\in f}\mathbf x_{\mathrm{srce}(e)},\qquad A_f=\tfrac12\left\lVert\sum_{e\in f}\mathbf x_{\mathrm{srce}(e)}\times\mathbf x_{\mathrm{trgt}(e)}\right\rVert$$""")
class CellGeometry3D(Aggregate):
    r"""aggregates the 3D vertex mesh into per-cell scalars, the cross-scale readout the reaction-diffusion runs on: each cell's centroid and area by scatter-add over its half-edges.
    vertex -> cell: reads the half-edge table stashed on the vertex set, writes the cell set's `centroid` and area.

        \mathbf{cen}_f=\frac{1}{n_f}\sum_{e\in f}\mathbf x_{\mathrm{srce}(e)},\qquad A_f=\tfrac12\left\lVert\sum_{e\in f}\mathbf x_{\mathrm{srce}(e)}\times\mathbf x_{\mathrm{trgt}(e)}\right\rVert
        cen_f          the centroid of face (cell) f
        n_f            its number of half-edges
        x_{srce(e)}, x_{trgt(e)} the positions of half-edge e's two vertices
        A_f            the face's area, half the magnitude of the summed cross products round the ring

    Reference: none: a geometric readout, not a mechanism. Plexus (this work)."""
    PARAM_UNITS = {}
    SYMBOLS = {'cen_f': 'the centroid of face (cell) f', 'n_f': 'its number of half-edges', 'x_{srce(e)}, x_{trgt(e)}': "the positions of half-edge e's two vertices", 'A_f': "the face's area, half the magnitude of the summed cross products round the ring"}
    PARAM_ROLES = {'vertex_set': 'the_vertex_set_that_carries_the_mesh'}
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    INPUTS = ["vertex"]; OUTPUTS = ["cell"]; READS = ["pos"]; WRITES = ["area", "centroid", "volume"]
    MECHANISM_TAGS = ["aggregate", "cell_geometry", "cross_scale"]
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.vat = params.get("vertex_set", "vertex")

    def forward(self, H, mask=None):
        from plexus.operators.vertex_ops import face_geometry_3d
        clvl = H.level(self.at); vlvl = H.level(self.vat); m = getattr(vlvl, "_mesh", None)
        if m is None:
            return {}
        pos = vlvl.get("pos")[:m["Nv"]]
        area, _, centroid, vol = face_geometry_3d(pos, m["E_srce"], m["E_trgt"], m["E_face"], m["nF"])
        nF = m["nF"]; st = clvl.state.clone(); sch = clvl.state_schema
        # `volume` -- THE CELL'S SIZE AS THE SIZE RULES READ IT, written only when the set declares
        # the block, so every existing spec is unchanged. The convention is `cell_size`'s: the
        # polyhedron between the apical and basal caps whenever the vertices carry a separation
        # `sep`, the origin-referenced wedge otherwise. Computed here rather than through
        # `cell_size`, which caches the run's reference volume as a side effect. Recording it is
        # what lets a ruler read a run's cell sizes instead of rebuilding every cell's polyhedron at
        # every frame -- minutes per run on CPU (exp 3, 2026-09-25).
        if "volume" in sch:
            if "sep" in getattr(vlvl, "state_schema", {}):
                from plexus.operators.vertex_ops import apicobasal_geometry_3d
                vol = apicobasal_geometry_3d(pos, vlvl.get("sep")[:m["Nv"]], m["E_srce"], m["E_trgt"],
                                             m["E_face"], nF)[0]
            i0, i1 = sch["volume"]; st[:nF, i0:i1] = vol.detach().to(st.dtype)[:, None]
        if "centroid" in sch:
            i0, i1 = sch["centroid"]; st[:nF, i0:i1] = centroid.detach()
        if "area" in sch:
            i0, i1 = sch["area"]; st[:nF, i0:i1] = area.detach()[:, None]
        clvl.state = st
        if getattr(clvl, "occ", None) is not None:
            occ = torch.zeros(clvl.state.shape[0], device=clvl.state.device); occ[:nF] = 1.0; clvl.occ = occ
        return {}


@register_operator("cell_neighbours", set="cell", kind="rewire", family="topology", title="find neighbouring cells",
                   equation=r"""$$\mathrm{edges}=\big\{(f,g)\ :\ \text{some mesh edge is shared by faces } f \text{ and } g\big\}$$""")
class CellAdjacency(Rewire):
    r"""the relation the reaction-diffusion runs on: two cells are neighbours if and only if they share a mesh edge, rebuilt every call so it follows a T1 or a division the moment one happens.
    vertex -> cell: reads the half-edge table, writes the cell set's `edge_index`.

        \mathrm{edges}=\big\{(f,g)\ :\ \text{some mesh edge is shared by faces } f \text{ and } g\big\}
        \mathrm{edges}  the edge set of the cell graph
        f, g           two faces (cells) sharing a mesh edge

    Reference: none: the adjacency of a mesh, not a mechanism. Plexus (this work)."""
    PARAM_UNITS = {}
    SYMBOLS = {'\\mathrm{edges}': 'the edge set of the cell graph', 'f, g': 'two faces (cells) sharing a mesh edge'}
    PARAM_ROLES = {'vertex_set': 'the_vertex_set_that_carries_the_mesh'}
    SUPPORTED_DIMS = [2, 3]; DIFFERENTIABLE = False
    MECHANISM_TAGS = ["cell_neighbours", "neighbour_graph"]
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.vat = params.get("vertex_set", "vertex")

    def forward(self, H, mask=None):
        clvl = H.level(self.at); vlvl = H.level(self.vat); m = getattr(vlvl, "_mesh", None)
        dev = clvl.state.device
        if m is None:
            clvl.edge_index = torch.zeros(2, 0, dtype=torch.long, device=dev); return {}
        es = m["E_srce"].cpu().numpy(); et = m["E_trgt"].cpu().numpy(); ef = m["E_face"].cpu().numpy()
        byedge = defaultdict(list)
        for k in range(len(ef)):
            byedge[(min(int(es[k]), int(et[k])), max(int(es[k]), int(et[k])))].append(int(ef[k]))
        pairs = set()
        for faces in byedge.values():
            for a in range(len(faces)):
                for b in range(a + 1, len(faces)):
                    x, y = faces[a], faces[b]
                    if x != y:
                        pairs.add((min(x, y), max(x, y)))
        if not pairs:
            clvl.edge_index = torch.zeros(2, 0, dtype=torch.long, device=dev); return {}
        e = np.array(sorted(pairs)).T
        ei = np.concatenate([e, e[::-1]], axis=1)                # symmetric (both directions)
        clvl.edge_index = torch.as_tensor(ei, dtype=torch.long, device=dev)
        return {}


@register_operator("cell_neighbours", set="cell", kind="rewire", family="topology", model="point_contact",
                   title="find neighbouring cells by point contact",
                   equation=r"""$$\mathrm{edges}=\big\{(c,d)\ :\ \exists\, i\in c,\ j\in d,\ |\mathbf x_i-\mathbf x_j|<r_c\big\},\qquad w_{cd}=\#\{(i,j)\}$$""")
class CellAdjacencyPointContact(Rewire):
    r"""the `point_contact` model of cell_neighbours: the cells are bodies of material points, so two cells touch when some point of one lies within `contact` of some point of the other (a kd-tree on the CPU).
    particle -[containment]-> cell: reads the point set's positions and its parent map (`points`), writes the cell set's `edge_index` (both directions) and `edge_weight` (the touching pair count), every `every` frames.

        \mathrm{edges}=\big\{(c,d)\ :\ \exists\, i\in c,\ j\in d,\ |\mathbf x_i-\mathbf x_j|<r_c\big\},\qquad w_{cd}=\#\{(i,j)\}
        \mathrm{edges}  the edge set
        (c, d)         two cells
        i, j           points of c and d
        x_i, x_j       their positions
        r_c            `contact`, the reach of the contact law the cells meet through, world length
        w_{cd}         the number of touching pairs

    Reference: none: the adjacency of touching bodies, not a mechanism. Plexus (this work)."""
    PARAM_UNITS = {'contact': 'length', 'every': 'frames'}
    SYMBOLS = {'\\mathrm{edges}': 'the edge set', '(c, d)': 'two cells', 'i, j': 'points of c and d', 'x_i, x_j': 'their positions', 'r_c': '`contact`, the reach of the contact law the cells meet through, world length', 'w_{cd}': 'the number of touching pairs'}
    SUPPORTED_DIMS = [2, 3]; DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["points", "contact"]
    MECHANISM_TAGS = ["cell_neighbours", "neighbour_graph", "contact"]
    PARAM_ROLES = {"points": "the_material_point_set", "contact": "touch_distance_world", "every": "calls_between_rebuilds"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.points = str(params["points"])
        self.contact = float(params["contact"])
        self.every = max(1, int(params.get("every", 1)))
        self._n = 0

    def forward(self, H, mask=None):
        self._n += 1
        clvl = H.level(self.at)
        if (self._n - 1) % self.every and getattr(clvl, "edge_index", None) is not None:
            return {}
        P = H.level(self.points)
        X = P.get("pos").detach()
        par = H.lift_index(self.points, self.at).long()
        live = (P.occ > 0.5) & torch.isfinite(X).all(1)
        X, par = X[live], par[live]
        dev = clvl.state.device
        from scipy.spatial import cKDTree
        pr = cKDTree(X.cpu().double().numpy()).query_pairs(self.contact, output_type="ndarray")
        if len(pr) == 0:
            clvl.edge_index = torch.zeros(2, 0, dtype=torch.long, device=dev)
            clvl.edge_weight = torch.zeros(0, device=dev)
            return {}
        pc = par.cpu().numpy()
        a, b = pc[pr[:, 0]], pc[pr[:, 1]]
        m = a != b
        e = np.sort(np.stack([a[m], b[m]], 1), axis=1)
        if e.shape[0] == 0:
            clvl.edge_index = torch.zeros(2, 0, dtype=torch.long, device=dev)
            clvl.edge_weight = torch.zeros(0, device=dev)
            return {}
        u, c = np.unique(e, axis=0, return_counts=True)
        ei = np.concatenate([u.T, u.T[::-1]], axis=1)
        clvl.edge_index = torch.as_tensor(ei, dtype=torch.long, device=dev)
        clvl.edge_weight = torch.as_tensor(np.concatenate([c, c]).astype(np.float32), device=dev)
        return {}


_PC_KERNEL = {}


def _point_contact_kernel():
    """The Warp kernel of `cell_neighbours[point_contact_warp]`, built on first use (importing this module must not need Warp)."""
    if "k" in _PC_KERNEL:
        return _PC_KERNEL["k"]
    import warp as wp

    @wp.kernel
    def _pc_count(P32: wp.array(dtype=wp.vec3), PX: wp.array(dtype=wp.vec3d), CID: wp.array(dtype=wp.int32),
                  grid: wp.uint64, rq: float, r2: wp.float64, r2f: float, nC: int, CNT: wp.array(dtype=wp.int32)):
        """One thread per point i: every point j > i within the contact distance, in another cell, adds one to the pair's count."""
        i = wp.tid()
        pf = P32[i]
        px = PX[i]
        ci = CID[i]
        j = int(0)
        q = wp.hash_grid_query(grid, pf, rq)
        while wp.hash_grid_query_next(q, j):
            if j > i:
                cj = CID[j]
                if cj != ci:
                    df = P32[j] - pf
                    if wp.dot(df, df) <= r2f:                          # float32 prefilter, margin included in r2f
                        pj = PX[j]
                        dx = pj[0] - px[0]
                        dy = pj[1] - px[1]
                        dz = pj[2] - px[2]
                        if dx * dx + dy * dy + dz * dz <= r2:          # the exact float64 test, as the tree's `query_pairs`
                            a = wp.min(ci, cj)
                            b = wp.max(ci, cj)
                            wp.atomic_add(CNT, a * nC + b, 1)

    _PC_KERNEL["k"] = (wp, _pc_count)
    return _PC_KERNEL["k"]


@register_operator("cell_neighbours", set="cell", kind="rewire", family="topology", model="point_contact_warp",
                   title="find neighbouring cells by point contact, on the GPU",
                   equation=r"""$$\mathrm{edges}=\big\{(c,d)\ :\ \exists\, i\in c,\ j\in d,\ |\mathbf x_i-\mathbf x_j|<r_c\big\},\qquad w_{cd}=\#\{(i,j)\}$$""")
class CellAdjacencyPointContactWarp(CellAdjacencyPointContact):
    r"""point_contact with the pair search on the GPU (a Warp hash grid): the same edges, the same weights, in the same order, without the CPU round trip that was the slowest step of a 300k-point frame.
    particle -[containment]-> cell: as point_contact.

        \mathrm{edges}=\big\{(c,d)\ :\ \exists\, i\in c,\ j\in d,\ |\mathbf x_i-\mathbf x_j|<r_c\big\},\qquad w_{cd}=\#\{(i,j)\}
        \mathrm{edges}  the edge set
        (c, d)         two cells
        i, j           points of c and d
        x_i, x_j       their positions
        r_c            `contact`, the reach of the contact law the cells meet through, world length
        w_{cd}         the number of touching pairs

    Reference: as point_contact; NVIDIA Warp; Plexus exp29 (2026-10-21)."""
    PARAM_UNITS = {}
    SYMBOLS = {'\\mathrm{edges}': 'the edge set', '(c, d)': 'two cells', 'i, j': 'points of c and d', 'x_i, x_j': 'their positions', 'r_c': '`contact`, the reach of the contact law the cells meet through, world length', 'w_{cd}': 'the number of touching pairs'}
    _GRID_DIM = 64

    def forward(self, H, mask=None):
        clvl = H.level(self.at)
        P = H.level(self.points)
        X = P.get("pos").detach()
        if X.device.type != "cuda":
            return super().forward(H, mask)
        try:
            wp, kern = _point_contact_kernel()
        except Exception:
            return super().forward(H, mask)
        self._n += 1
        if (self._n - 1) % self.every and getattr(clvl, "edge_index", None) is not None:
            return {}
        from plexus.operators.adhesion_warp_ops import _on_torch_stream, _wdev
        dev = X.device
        nC = int(clvl.state.shape[0])
        par = H.lift_index(self.points, self.at).long()
        live = (P.occ > 0.5) & torch.isfinite(X).all(1)
        Xl, pl = X[live], par[live]
        n = int(Xl.shape[0])
        cdev = clvl.state.device
        if n == 0:
            clvl.edge_index = torch.zeros(2, 0, dtype=torch.long, device=cdev)
            clvl.edge_weight = torch.zeros(0, device=cdev)
            return {}
        p32 = Xl.float().contiguous()
        x64 = Xl.double().contiguous()
        cid = pl.to(torch.int32).contiguous()
        cnt = torch.zeros(nC * nC, dtype=torch.int32, device=dev)
        rq = float(self.contact) * (1.0 + 1.0e-4) + 1.0e-5
        gr = getattr(self, "_grid", None)
        with _on_torch_stream(dev):
            if gr is None:
                gr = self._grid = wp.HashGrid(self._GRID_DIM, self._GRID_DIM, self._GRID_DIM, device=_wdev(dev))
            w32 = wp.from_torch(p32, dtype=wp.vec3)
            gr.build(points=w32, radius=rq)
            wp.launch(kern, dim=n, device=_wdev(dev),
                      inputs=[w32, wp.from_torch(x64, dtype=wp.vec3d), wp.from_torch(cid), gr.id, rq,
                              float(self.contact) * float(self.contact), rq * rq, nC, wp.from_torch(cnt)])
        nz = torch.nonzero(cnt).reshape(-1)
        if nz.numel() == 0:
            clvl.edge_index = torch.zeros(2, 0, dtype=torch.long, device=cdev)
            clvl.edge_weight = torch.zeros(0, device=cdev)
            return {}
        a, b = nz // nC, nz % nC
        c = cnt[nz].float()
        u = torch.stack([a, b])
        clvl.edge_index = torch.cat([u, u.flip(0)], dim=1).to(cdev)
        clvl.edge_weight = torch.cat([c, c]).to(cdev)
        return {}


@register_operator("cell_neighbours", set="cell", kind="rewire", family="topology", model="label_image",
                   title="find neighbouring cells in a segmentation",
                   equation=r"""$$\mathrm{edges}=\big\{(i,j)\ :\ \text{a pixel of label } i+1 \text{ shares an edge with a pixel of label } j+1\big\}$$""")
class CellAdjacencyLabelImage(Rewire):
    r"""the `label_image` model of cell_neighbours: the cells are a measured segmentation, so who touches whom is read from the label map itself, once (the labels do not move).
    label_image field -> cell: reads the integer label map `from`, writes the cell set's `edge_index` (and the shared pixel-edge count as `edge_weight`).

        \mathrm{edges}=\big\{(i,j)\ :\ \text{a pixel of label } i+1 \text{ shares an edge with a pixel of label } j+1\big\}
        \mathrm{edges}  the edge set
        (i, j)         two cells; cell i is label i + 1, label 0 is background

    Reference: none: the adjacency of a measured segmentation, not a mechanism. Plexus (this work)."""
    PARAM_UNITS = {}
    SYMBOLS = {'\\mathrm{edges}': 'the edge set', '(i, j)': 'two cells; cell i is label i + 1, label 0 is background'}
    SUPPORTED_DIMS = [2]; DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["from"]
    MECHANISM_TAGS = ["cell_neighbours", "neighbour_graph", "instance_segmentation"]
    PARAM_ROLES = {"from": "label_field"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.field_name = params["from"]
        self._ei = None

    @staticmethod
    def pairs(grid, counts=False):
        """[m, 2] unique label pairs (a < b) that share a pixel edge; background 0 excluded. With
        `counts`, also [m] the number of pixel edges each pair shares -- its contact length in pixels."""
        out = []
        for a, b in ((grid[:-1, :], grid[1:, :]), (grid[:, :-1], grid[:, 1:])):
            m = (a > 0) & (b > 0) & (a != b)
            out.append(torch.stack([torch.minimum(a[m], b[m]), torch.maximum(a[m], b[m])], 1))
        u, c = torch.unique(torch.cat(out, 0), dim=0, return_counts=True)
        return (u, c) if counts else u

    def forward(self, H, mask=None):
        clvl = H.level(self.at)
        if self._ei is None:
            grid = H.fields[self.field_name].grid[0]
            p, c = self.pairs(grid, counts=True)
            p = p - 1                                                    # label -> cell index
            keep = (p < clvl.n).all(1)
            p, c = p[keep], c[keep]
            e = p.T.to(torch.long)
            self._ei = torch.cat([e, e.flip(0)], 1).to(clvl.state.device)
            # the contact length of each junction, in pixels, aligned with edge_index (both directions);
            # read by `cell_chem_diffuse[model: contact_weighted]`, ignored by every other reader
            self._ew = torch.cat([c, c]).to(clvl.state.device, torch.get_default_dtype())
        clvl.edge_index = self._ei
        clvl.edge_weight = self._ew
        return {}


# CANONICAL `seed_cell_chem`, ALIAS `cell_chem_seed` -- see `mesh_ops.SeedMesh3D` for why both
# spellings must resolve: 320 specs use the first and the rest use the second.
@register_operator("seed_cell_chem", "cell_chem_seed", set="cell", kind="seed", family="seed", title="set the initial morphogen",
                   species=(("a", "activator"), ("u", "substrate")),
                   equation=r"""$$u_j=1\ \ \text{everywhere},\qquad a_j=0\ \ \text{but on the seeded patch}$$""")
class CellRDSeed(Structural):
    r"""the initial morphogen field on the cell set, written once at the opening of the trajectory: the Gray-Scott initial condition, substrate full everywhere and the activator absent but on a seeded spot.
    cell -> cell: writes the `chem` block, once (a seed, confined to the opening frames by its kind).

        u_j=1\ \ \text{everywhere},\qquad a_j=0\ \ \text{but on the seeded patch}
        u_j            the substrate of cell j, 1 everywhere
        a_j            the activator, 0 but on the seeded patch where (a, u) = (0.5, 0.25)

    Reference: Plexus (this work); cone seeding after Okuda, S. et al. (2018). Combining Turing and 3D vertex models reproduces autonomous multicellular morphogenesis of the tissue. Sci. Rep. 8:2386."""
    PARAM_UNITS = {'A': 'dimensionless', 'B': 'dimensionless', 'cone_deg': 'angle', 'n_species': 'count', 'n_spots': 'count', 'noise': 'fraction', 'p0': 'dimensionless', 'patch_z': 'length', 'seed': 'count', 'seed_frac': 'fraction'}
    SYMBOLS = {'u_j': 'the substrate of cell j, 1 everywhere', 'a_j': 'the activator, 0 but on the seeded patch where (a, u) = (0.5, 0.25)'}
    PARAM_ROLES = {'A': 'brusselator_feed_for_the_steady_state_seed', 'B': 'brusselator_conversion_for_the_steady_state_seed', 'cone_deg': 'half_angle_of_a_cone', 'mode': 'scatter_or_another_layout', 'n_species': 'chem_columns_seeded', 'n_spots': 'number_of_spots', 'noise': 'relative_noise', 'p0': 'initial_level', 'patch_z': 'height_of_the_patch', 'seed': 'random_seed', 'seed_dir': 'direction_of_the_patch', 'seed_frac': 'fraction_of_cells_seeded', 'vertex_set': 'the_vertex_set'}

    N_SPECIES = 2
    SUPPORTED_DIMS = [2, 3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["initial_condition", "gray_scott"]
    MODES = ("scatter", "noise", "patch", "cones", "simplex")
    REFERENCE = "Plexus (this work); cone seeding after Okuda, S. et al. (2018). Sci. Rep. 8:2386."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.vat = params.get("vertex_set", "vertex")
        self.seed = int(params.get("seed", 0))
        # WHICH SPECIES THIS SEEDER FILLS: 0 (columns 0,1) by default; 2 for a second RD system.
        # HOW MANY COLUMNS THIS SEEDER FILLS. Two unless told otherwise, so nothing archived moves.
        self.n_species = int(params.get("n_species", self.N_SPECIES))
        # (`simplex`) the total the species are normalised to. 1.0 is the May-Leonard simplex.
        self.p0 = float(params.get("p0", 1.0))
        self.chan = _chan(params, type(self).__name__, self.n_species)
        # How a pattern nucleates is on the `model:` axis, not a `mode:` setting. Five ways of
        # writing x_0 are five claims about HOW PATTERNING NUCLEATES -- from random fluctuation,
        # from a placed patch, from N fixed foci -- and seeding it in cones puts part of the answer
        # in by hand. That is a hypothesis, and it belongs where the registry can see it.
        #
        # It also matters because the variants READ DIFFERENT THINGS: `scatter` and `noise` use no
        # geometry at all, while `patch` and `cones` read the cell centroids. On the `model:` axis
        # each carries its own typed signature, so "may this seed be scheduled ahead of
        # `cell_geometry`?" is answerable from the registry rather than from a hard-coded list.
        if "mode" in params:
            raise ValueError(
                "seed_cell_chem: `mode` is gone -- write `model:`. How a pattern nucleates is a "
                "hypothesis, not a setting. "
                + ("`tip` is gone: it re-seeded every frame, which makes it a moving boundary "
                   "condition and annihilates every operator that writes to `chem`. Use "
                   "`model: scatter`." if params.get("mode") == "tip" else ""))
        self.mode = getattr(type(self), "NUCLEATION", "scatter")
        self.seed_frac = float(params.get("seed_frac", 0.06))  # (scatter) fraction of strong activator seeds
        self.A = float(params.get("A", 1.0)); self.B = float(params.get("B", 3.0))   # (noise) steady state (A, B/A)
        self.noise = float(params.get("noise", 0.04))
        self.patch_z = float(params.get("patch_z", 0.6))  # (patch) activate cells with cen_z > patch_z x z_max
        self.n_spots = int(params.get("n_spots", 5))            # (cones) number of fixed radial activation foci
        self.cone_deg = float(params.get("cone_deg", 18.0))     # (cones) half-angle of each activation cone
        self.seed_dir = params.get("seed_dir", None)            # (cones, n_spots=1) override the cone axis to a fixed
        #   direction -> aim the tube where we want it (e.g. FRONT of the render camera at elev18/azim30 ~ (.82,.48,.31))

    def _cone_dirs(self):
        """`n_spots` spread unit directions on the sphere (Fibonacci) -> fixed radial tube axes (Fig 5). A given
        `seed_dir` overrides the axis (used for n_spots=1 to point the single tube at the camera)."""
        if self.seed_dir is not None and self.n_spots == 1:
            v = np.asarray(self.seed_dir, float); return (v / (np.linalg.norm(v) + 1e-12))[None, :]
        i = np.arange(self.n_spots) + 0.5
        phi = np.arccos(1 - 2 * i / self.n_spots); theta = np.pi * (1 + 5 ** 0.5) * i
        return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)

    def forward(self, H, mask=None):
        clvl = H.level(self.at)
        # A MESH IS NOT REQUIRED, AND ASKING FOR ONE BY NAME WAS FATAL. This read
        # `H.level(self.vat)` unconditionally, so a spec with no `vertex` set died on
        # `KeyError: 'vertex'` -- and `SUPPORTED_DIMS` says [2, 3], so a flat 2D run is supposed to
        # be legal. All this operator wants from the mesh is nF, THE NUMBER OF CELLS; `scatter` and
        # `noise` use no geometry whatever, and `patch`/`cones` already guard on `"centroid" in
        # state_schema` and fall back to a uniform 0.02 without it. On a mesh-free set the cell
        # level IS the population, so its own occupancy answers the only question being asked.
        # `in` rather than `.get`: `H.levels` is an `nn.ModuleDict`, which has no `.get` -- the
        # exact trap that silently disabled `renumber_set`.
        vlvl = H.levels[self.vat] if self.vat in H.levels else None
        m = getattr(vlvl, "_mesh", None) if vlvl is not None else None
        if "chem" not in clvl.state_schema:
            return {}
        if m is not None:
            nF = m["nF"]
        else:
            occ = getattr(clvl, "occ", None)
            nF = int(occ.sum().item()) if occ is not None else int(clvl.state.shape[0])
            if nF <= 0:
                return {}
        dev = clvl.state.device
        g = torch.Generator(device="cpu"); g.manual_seed(self.seed)
        if self.mode == "patch":                                # localized activation source (a bud/tube driver)
            a = torch.full((nF,), 0.02, device=dev)
            if "centroid" in clvl.state_schema:
                ci0, ci1 = clvl.state_schema["centroid"]; zc = clvl.state[:nF, ci0 + 2]
                a = torch.where(zc > self.patch_z * float(zc.max()), torch.ones(nF, device=dev), a)
            u = torch.ones(nF, device=dev)
        elif self.mode == "cones":                              # N FIXED radial activation cones (Fig 5 multi-tube):
            a = torch.full((nF,), 0.02, device=dev)             # each cone's tip stays activated as it extends ->
            if "centroid" in clvl.state_schema:                      # N radial tubes. Re-seeded every frame (tracks tips).
                ci0, ci1 = clvl.state_schema["centroid"]; centroid = clvl.state[:nF, ci0:ci0 + 3]
                d = centroid / (centroid.norm(dim=1, keepdim=True) + 1e-9)
                dirs = torch.as_tensor(self._cone_dirs(), dtype=centroid.dtype, device=dev)
                cosmax = (d @ dirs.T).max(dim=1).values
                a = torch.where(cosmax > float(np.cos(np.radians(self.cone_deg))), torch.ones(nF, device=dev), a)
            u = torch.ones(nF, device=dev)
        elif self.mode == "simplex":
            # A SYMMETRIC START FOR A COMPETITION MODEL. Every other mode here is a Gray-Scott
            # initial condition: activator in column 0, SUBSTRATE = 1 in column 1. Feed that to
            # May-Leonard, where column 1 is just the second competitor, and the run begins with
            # v at 0.9 against u and w at 0.07 -- not a competition, a landslide. Measured: v
            # dominates immediately, w cyclically beats v, and the field collapses to w = 1
            # everywhere with zero spatial variance. No motif can emerge from that and none did.
            #
            # `simplex` gives all n species the same random field and then NORMALISES SO THEY SUM
            # TO `p0`, which is what ParticleGraph's RD_RPS does (`init_mesh`, case 'RD_Mesh':
            # `node_value = rand(n,3)` then `node_value[:,k] /= sum`). p0 = 1 is not a detail: the
            # logistic factor is (1 - p - a*rival), so starting at p = 0.175 leaves (1-p) = 0.83 of
            # net growth for EVERY species and the run spends itself climbing to the simplex
            # instead of competing on it. The heteroclinic cycle that produces spirals lives AT
            # p = 1, so an initial condition below it postpones the whole phenomenon.
            a = None
            u = None
        elif self.mode == "noise":                              # Brusselator: homogeneous steady state + noise
            a = (self.A + self.noise * torch.randn(nF, generator=g)).to(dev)
            u = (self.B / self.A + self.noise * torch.randn(nF, generator=g)).to(dev)
        else:                                                   # "scatter" -- and ONLY scatter. The mode is validated
            # in __init__, so this branch can no longer be reached by a typo or by a mode that was
            # deleted out from under an archived spec.
            # (a central spot is 2D-disk logic -- on a sphere every cell is equidistant, so scatter/noise)
            a = (0.04 * torch.rand(nF, generator=g)).to(dev)
            u = torch.ones(nF, device=dev)
            nucl = (torch.rand(nF, generator=g) < self.seed_frac).to(dev)
            a = torch.where(nucl, torch.full_like(a, 0.5), a)
            u = torch.where(nucl, torch.full_like(u, 0.25), u)
        # SEEDED INTO THIS SPECIES' OWN COLUMNS. `chan` offsets from the schema's base, so a
        # second seeder writes the second pair and leaves the first alone.
        h0, h1 = clvl.state_schema["chem"]
        base = h0 + self.chan
        st = clvl.state.clone()
        # SIMPLEX WRITES EVERY SPECIES AND RETURNS -- it has no activator/substrate pair to write,
        # which is the whole point of it, so it must run BEFORE the `a`/`u` write rather than after.
        if self.mode == "simplex":
            _cols = [k for k in range(self.n_species) if h1 - base > k]
            _r = torch.rand(nF, len(_cols), generator=g).to(dev)
            _r = self.p0 * _r / _r.sum(dim=1, keepdim=True).clamp(min=1e-9)
            for _i, _k in enumerate(_cols):
                st[:nF, base + _k:base + _k + 1] = _r[:, _i:_i + 1]
            clvl.state = st
            return {}
        st[:nF, base:base + 1] = a[:, None]
        if h1 - base > 1:
            st[:nF, base + 1:base + 2] = u[:, None]
        # A SPAN WIDER THAN TWO gets the remaining species seeded the same way the first was, from
        # the SAME generator, so a three-species start is three independent random fields rather
        # than one field and two zeros -- two zeros is extinction, not a neutral start.
        for _k in range(2, self.n_species):
            if h1 - base > _k:
                _v = (0.04 * torch.rand(nF, generator=g)).to(dev)
                _nu = (torch.rand(nF, generator=g) < self.seed_frac).to(dev)
                st[:nF, base + _k:base + _k + 1] = torch.where(
                    _nu, torch.full_like(_v, 0.5), _v)[:, None]
        clvl.state = st
        return {}


@register_operator("seed_cell_chem", "cell_chem_seed", set="cell", kind="seed", family="seed",
                   model="noise", title="set the initial morphogen with noise", equation=r"""$$c_{jk}=c^*_k\,(1+\epsilon\,z_{jk}),\qquad z_{jk}\sim\mathcal N(0,1)$$""")
class CellRDSeedNoise(CellRDSeed):
    r"""the `noise` model of seed_cell_chem: the homogeneous steady state plus noise, patterning from fluctuation alone, the strictest test that the pattern is emergent (reads no geometry).
    cell -> cell: writes the `chem` block, once.

        c_{jk}=c^*_k\,(1+\epsilon\,z_{jk}),\qquad z_{jk}\sim\mathcal N(0,1)
        c_{jk}         species k of cell j
        c^*_k          the reaction's homogeneous steady state
        \epsilon       `noise`, the relative perturbation
        z_{jk}         a standard normal draw from `seed`

    Reference: Turing, A. M. (1952). Phil. Trans. R. Soc. B 237:37-72 (patterning from a perturbed uniform state)."""
    PARAM_UNITS = {}
    SYMBOLS = {'c_{jk}': 'species k of cell j', 'c^*_k': "the reaction's homogeneous steady state", '\\epsilon': '`noise`, the relative perturbation', 'z_{jk}': 'a standard normal draw from `seed`'}
    NUCLEATION = "noise"


@register_operator("seed_cell_chem", "cell_chem_seed", set="cell", kind="seed", family="seed",
                   model="cones", title="seed morphogen in radial cones", equation=r"""$$a_j=1\ \ \text{if}\ \ \angle(\mathbf c_j-\mathbf o,\ \hat{\mathbf d}_k)<\alpha\ \text{for some cone }k,\quad\text{else }0$$""")
class CellRDSeedCones(CellRDSeed):
    r"""the `cones` model of seed_cell_chem: N fixed radial activation cones (Okuda Fig. 5's multi-tube), the strongest hand in the answer and the honest place to declare it (reads the cell centroids).
    cell -> cell: writes the `chem` block, once, after cell_geometry.

        a_j=1\ \ \text{if}\ \ \angle(\mathbf c_j-\mathbf o,\ \hat{\mathbf d}_k)<\alpha\ \text{for some cone }k,\quad\text{else }0
        a_j            the activator of cell j
        c_j            its centroid
        o              the tissue centre
        \hat d_k       the axis of cone k (`n_spots` cones)
        \alpha         `cone_deg`, the half-angle

    Reference: Okuda, S. et al. (2018). Combining Turing and 3D vertex models reproduces autonomous multicellular morphogenesis of the tissue. Sci. Rep. 8:2386, Fig. 5."""
    PARAM_UNITS = {}
    SYMBOLS = {'a_j': 'the activator of cell j', 'c_j': 'its centroid', 'o': 'the tissue centre', '\\hat d_k': 'the axis of cone k (`n_spots` cones)', '\\alpha': '`cone_deg`, the half-angle'}
    NUCLEATION = "cones"


@register_operator("seed_cell_chem", "cell_chem_seed", set="cell", kind="seed", family="seed",
                   model="simplex", title="seed three competing species", equation=r"""$$(u_j,v_j,w_j)\sim\text{uniform on the simplex } u+v+w=1$$""")
class CellRDSeedSimplex(CellRDSeed):
    r"""the `simplex` model of seed_cell_chem: three species normalised to a simplex, the May-Leonard initial condition for cyclic competition (reads no geometry).
    cell -> cell: writes the `chem` block, once.

        (u_j,v_j,w_j)\sim\text{uniform on the simplex } u+v+w=1
        u_j, v_j, w_j  the three species of cell j, positive and summing to 1

    Reference: May, R. M. & Leonard, W. J. (1975). SIAM J. Appl. Math. 29:243-253."""
    PARAM_UNITS = {}
    SYMBOLS = {'u_j, v_j, w_j': 'the three species of cell j, positive and summing to 1'}
    NUCLEATION = "simplex"


@register_operator("seed_cell_chem", "cell_chem_seed", set="cell", kind="seed", family="seed",
                   model="uniform", title="set the same initial state everywhere", equation=r"""$$c_{jk}=v_k\quad\text{for every cell } j$$""")
class CellRDSeedUniform(CellRDSeed):
    r"""the `uniform` model of seed_cell_chem: every cell the same, `values` written into the columns from `chan` on, the no-signal steady state a gene-regulatory circuit starts from.
    cell -> cell: writes the `chem` block, once.

        c_{jk}=v_k\quad\text{for every cell } j
        c_{jk}         species k of cell j
        v_k            `values`, one per column

    Reference: Plexus (this work); Balaskas, N. et al. (2012). Cell 148:273-284 (P starts high)."""
    PARAM_UNITS = {}
    SYMBOLS = {'c_{jk}': 'species k of cell j', 'v_k': '`values`, one per column'}
    PARAM_ROLES = {'values': 'the_initial_level_per_column'}
    NUCLEATION = "uniform"

    def __init__(self, params, device="cpu"):
        vals = params.get("values")
        if not vals:
            raise ValueError("seed_cell_chem[uniform]: needs `values: [..]`, one per column from `chan`")
        params = dict(params, n_species=len(vals))
        super().__init__(params, device)
        self.values = [float(v) for v in vals]

    def forward(self, H, mask=None):
        """Its own body, so the base class's is untouched: the cell count as `CellRDSeed` reads it
        (the mesh's nF, else the set's live count), then the declared values into the columns."""
        clvl = H.level(self.at)
        if "chem" not in clvl.state_schema:
            return {}
        vlvl = H.levels[self.vat] if self.vat in H.levels else None
        m = getattr(vlvl, "_mesh", None) if vlvl is not None else None
        if m is not None:
            nF = int(m["nF"])
        else:
            occ = getattr(clvl, "occ", None)
            nF = int(occ.sum().item()) if occ is not None else int(clvl.state.shape[0])
        h0, h1 = clvl.state_schema["chem"]
        _span(clvl.state[:, h0:h1], self.chan, len(self.values), type(self).__name__)
        st = clvl.state.clone()
        for _k, _v in enumerate(self.values):
            st[:nF, h0 + self.chan + _k] = float(_v)
        clvl.state = st
        return {}


@register_operator("seed_cell_chem", "cell_chem_seed", set="cell", kind="seed", family="seed",
                   model="lognormal", title="set initial state with cell variability", equation=r"""$$c_{jk}=v_k\,\exp\!\big(s_k z_{jk}-\tfrac12 s_k^2\big),\qquad s_k=\sqrt{\ln(1+\mathrm{cv}_k^2)}$$""")
class CellRDSeedLognormal(CellRDSeedUniform):
    r"""the `lognormal` model of seed_cell_chem: uniform's declared state with cell-to-cell variability, log-normal because a concentration is positive and its spread multiplicative.
    cell -> cell: writes the `chem` block, once.

        c_{jk}=v_k\,\exp\!\big(s_k z_{jk}-\tfrac12 s_k^2\big),\qquad s_k=\sqrt{\ln(1+\mathrm{cv}_k^2)}
        c_{jk}         species k of cell j (mean v_k)
        v_k            `values`
        s_k            the log-normal sigma from the coefficient of variation `cv`
        z_{jk}         a standard normal draw from `seed`

    Reference: Serra, D. et al. (2019). Nature 569:66-72 (the variability); Plexus (this work)."""
    PARAM_UNITS = {'cv': 'fraction'}
    SYMBOLS = {'c_{jk}': 'species k of cell j (mean v_k)', 'v_k': '`values`', 's_k': 'the log-normal sigma from the coefficient of variation `cv`', 'z_{jk}': 'a standard normal draw from `seed`'}
    PARAM_ROLES = {'cv': 'coefficient_of_variation_per_column'}
    NUCLEATION = "lognormal"

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        cv = params.get("cv", 0.0)
        cv = [float(cv)] * len(self.values) if not isinstance(cv, (list, tuple)) else [float(c) for c in cv]
        if len(cv) != len(self.values):
            raise ValueError(f"seed_cell_chem[lognormal]: `cv` has {len(cv)} entries, `values` {len(self.values)}")
        self.cv = cv

    def forward(self, H, mask=None):
        super().forward(H, mask)
        clvl = H.level(self.at)
        if "chem" not in clvl.state_schema or not any(self.cv):
            return {}
        vlvl = H.levels[self.vat] if self.vat in H.levels else None
        m = getattr(vlvl, "_mesh", None) if vlvl is not None else None
        if m is not None:
            nF = int(m["nF"])
        else:
            occ = getattr(clvl, "occ", None)
            nF = int(occ.sum().item()) if occ is not None else int(clvl.state.shape[0])
        g = torch.Generator(device="cpu"); g.manual_seed(self.seed)
        h0, _ = clvl.state_schema["chem"]
        st = clvl.state.clone()
        for _k, _c in enumerate(self.cv):
            if _c <= 0:
                continue
            sg = float(np.sqrt(np.log1p(_c * _c)))
            z = torch.randn(nF, generator=g, dtype=torch.float64)
            fac = torch.exp(sg * z - 0.5 * sg * sg).to(device=st.device, dtype=st.dtype)
            st[:nF, h0 + self.chan + _k] = st[:nF, h0 + self.chan + _k] * fac
        clvl.state = st
        return {}


@register_operator("cell_chem_diffuse", set="cell", kind="lateral", family="fields", implementation="graph_laplacian", title="diffuse morphogen between neighbours",
                   equation=r"""$$\frac{dc_i}{dt}=D_s\sum_{j\sim i}\frac{c_j-c_i}{\deg(i)}$$""")
class CellDiffuse(Lateral):
    r"""morphogen exchange between neighbouring cells as a purely combinatorial graph diffusion: every neighbour counts the same, whatever the geometry between them.
    cell -[cell adjacency]-> cell: reads chem and edge_index, emits d(chem)/dt.

        \frac{dc_i}{dt}=D_s\sum_{j\sim i}\frac{c_j-c_i}{\deg(i)}
        c_i            the concentration of one species in cell i, dimensionless
        D_s            that species' diffusion coefficient (`d`, or `d_a` / `d_h`), per time
        j \sim i       the cells sharing a mesh edge with i
        \deg(i)        their number (`norm: true`; without it the sum is unnormalised)

    Reference: Fick, A. (1855). Ann. Phys. 170:59-86; Turing, A. M. (1952). Phil. Trans. R. Soc. B 237:37-72."""
    PARAM_UNITS = {'chi': 'dimensionless', 'd': 'rate', 'd_a': 'rate', 'd_h': 'rate'}
    SYMBOLS = {'c_i': 'the concentration of one species in cell i, dimensionless', 'D_s': "that species' diffusion coefficient (`d`, or `d_a` / `d_h`), per time", 'j \\sim i': 'the cells sharing a mesh edge with i', '\\deg(i)': 'their number (`norm: true`; without it the sum is unnormalised)'}

    N_SPECIES = 2
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["chi"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]
    MECHANISM_TAGS = ["diffusion", "graph_laplacian", "turing"]
    REFERENCE = "Fick, A. (1855). Ueber Diffusion. Ann. Phys. 170:59-86; Turing, A. M. (1952). Phil. Trans. R. Soc. B 237:37-72."
    PARAM_ROLES = {"d_a": "activator_diffusivity", "d_h": "substrate_diffusivity", "chi": "spatial_scale", 'd': 'diffusion_coefficient_per_species', 'norm': 'divide_by_the_neighbour_count'}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        # A DIFFUSIVITY PER SPECIES, because two is not the only width. `d_a`/`d_h` name the roles
        # of the two Gray-Scott species and cannot express three (May-Leonard) or four (a coupled
        # pair), and FitzHugh-Nagumo needs one of them to be exactly ZERO -- only `u` diffuses
        # there, `v` has no Laplacian at all. `d: [..]` is the general spelling and its LENGTH
        # declares the span; `d_a`/`d_h` remain the two-species one and every archived spec keeps
        # working unchanged.
        _d = params.get("d")
        if _d is not None:
            self.d = [float(x) for x in _d]
        elif "d_a" in params and "d_h" in params:
            self.d = [float(params["d_a"]), float(params["d_h"])]
        else:
            raise ValueError(f"{type(self).__name__}: needs either `d: [..]` (one diffusivity per "
                             f"species) or both `d_a` and `d_h` (the two-species spelling).")
        self.d_a, self.d_h = self.d[0], (self.d[1] if len(self.d) > 1 else self.d[0])
        self.N_SPECIES = len(self.d)          # instance attribute: shadows the class default of 2
        self.chi = self.tunable(params["chi"], None, tensor=True)
        self.norm = bool(params.get("norm", True))
        # WHICH SPECIES THIS INSTANCE OWNS: 0 is the first pair (chem columns 0,1) and is the
        # default, so every existing spec is unchanged. A second RD system lives at chan 2.
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)


    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        ei = getattr(lvl, "edge_index", None)
        if ei is None or ei.numel() == 0:
            return {self.at: torch.zeros_like(chem)}
        i, j = ei[0], ei[1]; N = chem.shape[0]
        agg = torch.zeros_like(chem).index_add_(0, i, chem[j])
        deg = torch.zeros(N, device=chem.device, dtype=chem.dtype).index_add_(0, i, torch.ones_like(i, dtype=chem.dtype))
        lap = (agg / deg.clamp(min=1)[:, None] - chem) if self.norm else (agg - deg[:, None] * chem)
        # A CELL WITH NO NEIGHBOUR EXCHANGES NOTHING (exp06, 2026-11-14). With `norm` the line above gave it 0 / 1 - c = -c: every species DECAYED at its
        # diffusivity into nothing, so a lone MPM cell lost its fast-diffusing inhibitor and its Gierer-Meinhardt activator ran away (it21: every lone cell
        # took the activator-high fate). It was also inconsistent with the no-edge case above, which returns zero. Cells with neighbours are unchanged.
        lap = torch.where(deg[:, None] > 0, lap, torch.zeros_like(lap))
        # A PER-COLUMN COEFFICIENT, so a second SPECIES can live in the same `chem` buffer at its
        # own columns. This was `tensor([d_a, d_h])` -- exactly two entries -- which both assumed a
        # width-2 chem and diffused whatever happened to be in those two columns. With `chan` the
        # operator says WHICH pair it owns, and writes zeros everywhere else, so two instances at
        # chan 0 and chan 2 are two independent reaction-diffusion systems that cannot leak into
        # one another through the diffusion step.
        _span(chem, self.chan, len(self.d), type(self).__name__)      # bounds, loudly
        coef = torch.zeros(chem.shape[1], device=chem.device, dtype=chem.dtype)
        for _k, _dk in enumerate(self.d):
            coef[self.chan + _k] = _dk * self.chi
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: (coef[None, :] * lap) * occ}


# ============================================================================================
#  individuals on lattice sites: the shared machinery of the lattice variants
# ============================================================================================
# A SITE is a cell of the set; it holds one individual (a one-hot row of `chem` over the species'
# columns) or none (a row of zeros). The lattice variants below (`cell_chem_diffuse[lattice_exchange]`,
# `cell_chem_react[rps_lattice]`, `cell_chem_react[kerr_csr]`) each draw discrete events for one tick.
#
# HOW THEY COMBINE WITH THE REST OF THE SCHEDULE -- read before writing another. The engine SUMS the
# deltas of every operator writing `chem` and integrates once per tick (`Hierarchy.add_delta`,
# `engine._integrate`), so two lattice operators that each flipped sites of the SAME state could
# both empty one site, or swap an individual another had just killed, and the sum would no longer be
# one-hot. Each variant therefore reads the PROVISIONAL state -- the state plus the `chem` delta the
# operators before it added this tick -- draws its events on it, and returns exactly the increment
# that takes the provisional state to its result. The sum is then the operators applied one after
# the other in schedule order (a first-order operator splitting of the paper's Gillespie scheme, exact
# as dt -> 0), and every site stays empty or one individual.
def _lattice_provisional(H, at, chem):
    # WHERE THE ENGINE KEEPS THE `chem` DELTA DEPENDS ON THE SET: when `chem` is the set's coordinate
    # block (a lattice of sites whose `pos` is not integrated) it accumulates in `H._delta[at]`, else in
    # `H._delta_blocks[at]["chem"]` (`Hierarchy.add_delta`). Reading only the second made two lattice
    # operators blind to each other on exactly the sets they run on -- sites holding two individuals.
    co = H.level(at).state_schema.coordinate
    if co is not None and co.name == "chem":
        d = (getattr(H, "_delta", {}) or {}).get(at)
    else:
        d = (getattr(H, "_delta_blocks", {}) or {}).get(at, {}).get("chem")
    return chem if d is None else chem + float(getattr(H, "dt", 1.0)) * d


def _lattice_sites(x, chan, ns):
    """(occupied [N] bool, species [N] long) of the provisional state's species columns."""
    cols = x[:, chan:chan + ns]
    return cols.max(1).values > 0.5, cols.argmax(1)


def _lattice_onehot(occ, s, ns):
    t = torch.zeros(occ.shape[0], ns, device=occ.device, dtype=torch.float32)
    t[occ, s[occ]] = 1.0
    return t


def _lattice_neighbour(op, lvl, gen):
    """One uniformly random neighbour per site, from the set's relation (`edge_index`, receiver first).
    The CSR table is rebuilt only when the relation object changes."""
    ei = getattr(lvl, "edge_index", None)
    if ei is None or ei.numel() == 0:
        raise ValueError(f"{type(op).__name__}: the set has no neighbour relation -- schedule a "
                         f"`radius_graph` (e.g. `model: periodic_tiles`) before it")
    if getattr(op, "_csr_of", None) is not ei:
        n = lvl.state.shape[0]
        o = torch.argsort(ei[0], stable=True)
        op._col = ei[1][o]
        op._deg = torch.bincount(ei[0], minlength=n)
        op._off = torch.cumsum(op._deg, 0) - op._deg
        op._csr_of = ei
    u = torch.rand(op._deg.shape[0], generator=gen, device=op._deg.device)
    k = torch.clamp((u * op._deg).long(), max=(op._deg - 1).clamp(min=0))
    return op._col[(op._off + k).clamp(max=op._col.numel() - 1)]


def _lattice_one_per_target(tgt, gen):
    """Indices into `tgt` keeping ONE event per target site, chosen at random -- two individuals
    reaching for one site in one tick: one succeeds, the other waits a tick."""
    if tgt.numel() == 0:
        return tgt
    key = torch.rand(tgt.shape, generator=gen, device=tgt.device, dtype=torch.float64)
    o = torch.argsort(tgt.double() * 2.0 + key)
    ts = tgt[o]
    last = torch.ones_like(ts, dtype=torch.bool)
    last[:-1] = ts[1:] != ts[:-1]
    return o[last]


def _lattice_gen(op, dev):
    if getattr(op, "_gen", None) is None or op._gen.device != torch.device(dev):
        op._gen = torch.Generator(device=dev)
        op._gen.manual_seed(int(getattr(op, "seed", 0)))
    return op._gen


@register_operator("cell_chem_diffuse", set="cell", kind="lateral", family="fields", implementation="closed_junctions",
                   title="diffuse morphogen, some junctions closed",
                   equation=r"""$$\frac{dc_i}{dt}=D_s\sum_{j\sim i}w_{ij}\frac{c_j-c_i}{\deg(i)},\qquad w_{ij}=0\ \text{across the closed line, else }1$$""")
class CellDiffuseClosedJunctions(CellDiffuse):
    r"""the `closed_junctions` implementation of cell_chem_diffuse: the graph Laplacian with every junction that crosses a declared line (or segment) shut, the conduction-block experiment.
    cell -[cell adjacency]-> cell: as the default; a shut junction still counts in deg(i).

        \frac{dc_i}{dt}=D_s\sum_{j\sim i}w_{ij}\frac{c_j-c_i}{\deg(i)},\qquad w_{ij}=0\ \text{across the closed line, else }1
        c_i            the concentration in cell i
        D_s            the diffusion coefficient
        w_{ij}         0 when the centroids of i and j lie on opposite sides of the plane `closed`, else 1
        \deg(i)        the neighbour count

    Reference: Kleber, A. G. & Rudy, Y. (2004). Physiol. Rev. 84:431-488 (conduction block)."""
    PARAM_UNITS = {}
    SYMBOLS = {'c_i': 'the concentration in cell i', 'D_s': 'the diffusion coefficient', 'w_{ij}': '0 when the centroids of i and j lie on opposite sides of the plane `closed`, else 1', '\\deg(i)': 'the neighbour count'}
    READS = ["chem", "centroid"]
    PARAM_ROLES = {**CellDiffuse.PARAM_ROLES, "closed": "closed_junction_line"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        c = params.get("closed")
        if not c or "point" not in c or "normal" not in c:
            raise ValueError("cell_chem_diffuse[closed_junctions]: needs `closed: {point: [..], normal: [..]}` "
                             "-- without a line this is `graph_laplacian`, and should say so.")
        self.closed = c

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        ei = getattr(lvl, "edge_index", None)
        if ei is None or ei.numel() == 0:
            return {self.at: torch.zeros_like(chem)}
        i, j = ei[0], ei[1]; N = chem.shape[0]
        deg = torch.zeros(N, device=chem.device, dtype=chem.dtype).index_add_(0, i, torch.ones_like(i, dtype=chem.dtype))
        cen = lvl.get("centroid" if "centroid" in lvl.state_schema else "pos")
        p = torch.as_tensor([float(x) for x in self.closed["point"]], dtype=cen.dtype, device=cen.device)
        nrm = torch.as_tensor([float(x) for x in self.closed["normal"]], dtype=cen.dtype, device=cen.device)
        side = ((cen[:, :len(p)] - p) @ nrm) > 0
        shut = side[i] != side[j]
        if self.closed.get("half_length") is not None:
            tan = torch.stack([-nrm[1], nrm[0]])
            mid = 0.5 * (cen[i, :2] + cen[j, :2]) - p[:2]
            shut = shut & ((mid @ tan).abs() <= float(self.closed["half_length"]))
        w = (~shut).to(chem.dtype)[:, None]
        flux = torch.zeros_like(chem).index_add_(0, i, w * (chem[j] - chem[i]))
        lap = flux / deg.clamp(min=1)[:, None] if self.norm else flux
        _span(chem, self.chan, len(self.d), type(self).__name__)
        coef = torch.zeros(chem.shape[1], device=chem.device, dtype=chem.dtype)
        for _k, _dk in enumerate(self.d):
            coef[self.chan + _k] = _dk * self.chi
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: (coef[None, :] * lap) * occ}


@register_operator("cell_chem_diffuse", set="cell", kind="lateral", family="fields", model="contact_weighted",
                   title="diffuse morphogen by contact area",
                   equation=r"""$$\frac{dc_i}{dt}=D_s\sum_{j\sim i}\frac{\ell_{ij}}{\bar\ell}\,\frac{c_j-c_i}{\deg(i)}$$""")
class CellDiffuseContactWeighted(CellDiffuse):
    r"""the `contact_weighted` model of cell_chem_diffuse: a junction's conductance proportional to the length of membrane the two cells share, as gap-junction plaques are distributed along it.
    cell -[cell adjacency]-> cell: as the default, weighted by the set's `edge_weight` (the contact length cell_neighbours[label_image] writes) over its mean.

        \frac{dc_i}{dt}=D_s\sum_{j\sim i}\frac{\ell_{ij}}{\bar\ell}\,\frac{c_j-c_i}{\deg(i)}
        c_i            the concentration in cell i
        D_s            the diffusion coefficient
        \ell_{ij}      the contact length of the pair
        \bar\ell       its mean over the sheet's junctions
        \deg(i)        the neighbour count

    Reference: Kleber, A. G. & Rudy, Y. (2004). Physiol. Rev. 84:431-488, section IV (gap junctions scale with contact area); Plexus (this work)."""
    PARAM_UNITS = {}
    SYMBOLS = {'c_i': 'the concentration in cell i', 'D_s': 'the diffusion coefficient', '\\ell_{ij}': 'the contact length of the pair', '\\bar\\ell': "its mean over the sheet's junctions", '\\deg(i)': 'the neighbour count'}
    READS = ["chem"]

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        ei = getattr(lvl, "edge_index", None)
        w = getattr(lvl, "edge_weight", None)
        if ei is None or ei.numel() == 0:
            return {self.at: torch.zeros_like(chem)}
        if w is None or w.numel() != ei.shape[1]:
            raise ValueError("cell_chem_diffuse[contact_weighted]: the cell set has no `edge_weight` "
                             "aligned with its edge_index -- use cell_neighbours[model: label_image].")
        i, j = ei[0], ei[1]; N = chem.shape[0]
        deg = torch.zeros(N, device=chem.device, dtype=chem.dtype).index_add_(0, i, torch.ones_like(i, dtype=chem.dtype))
        wn = (w / w.mean()).to(chem.dtype)[:, None]
        flux = torch.zeros_like(chem).index_add_(0, i, wn * (chem[j] - chem[i]))
        lap = flux / deg.clamp(min=1)[:, None] if self.norm else flux
        _span(chem, self.chan, len(self.d), type(self).__name__)
        coef = torch.zeros(chem.shape[1], device=chem.device, dtype=chem.dtype)
        for _k, _dk in enumerate(self.d):
            coef[self.chan + _k] = _dk * self.chi
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: (coef[None, :] * lap) * occ}


@register_operator("cell_chem_diffuse", set="cell", kind="lateral", family="fields", model="fibre_anisotropic",
                   title="diffuse faster along fibres",
                   equation=r"""$$\frac{dc_i}{dt}=D_s\sum_{j\sim i}\frac{a_{ij}}{\bar a}\,\frac{c_j-c_i}{\deg(i)},\quad a_{ij}=\tfrac12\sum_{k\in\{i,j\}}\big(\kappa\cos^2(\theta_{ij}-\phi_k)+\sin^2(\theta_{ij}-\phi_k)\big)$$""")
class CellDiffuseFibreAnisotropic(CellDiffuse):
    r"""the `fibre_anisotropic` model of cell_chem_diffuse: cardiac conduction is faster along the fibre than across it because the junctions sit at the cells' ends, each junction weighted by its angle to the two cells' fibres.
    cell -[cell adjacency]-> cell: as the default, each junction weighted by a_ij over the mean.

        \frac{dc_i}{dt}=D_s\sum_{j\sim i}\frac{a_{ij}}{\bar a}\,\frac{c_j-c_i}{\deg(i)},\quad a_{ij}=\tfrac12\sum_{k\in\{i,j\}}\big(\kappa\cos^2(\theta_{ij}-\phi_k)+\sin^2(\theta_{ij}-\phi_k)\big)
        c_i            the concentration in cell i
        D_s            the diffusion coefficient
        a_{ij}         the junction's anisotropy weight
        \bar a         its mean
        \kappa         `ratio`, the along-over-across conductance
        \theta_{ij}    the direction from cell i's centroid to cell j's
        \phi_k         cell k's fibre angle (block `phi`)

    Reference: Kleber, A. G. & Rudy, Y. (2004). Physiol. Rev. 84:431-488, section IV.A."""
    PARAM_UNITS = {'ratio': 'dimensionless'}
    SYMBOLS = {'c_i': 'the concentration in cell i', 'D_s': 'the diffusion coefficient', 'a_{ij}': "the junction's anisotropy weight", '\\bar a': 'its mean', '\\kappa': '`ratio`, the along-over-across conductance', '\\theta_{ij}': "the direction from cell i's centroid to cell j's", '\\phi_k': "cell k's fibre angle (block `phi`)"}
    READS = ["chem", "phi"]
    PARAM_ROLES = {**CellDiffuse.PARAM_ROLES, "ratio": "along_across_conductance_ratio"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.ratio = self.tunable(params.get("ratio"), 4.41, tensor=True)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        ei = getattr(lvl, "edge_index", None)
        if ei is None or ei.numel() == 0:
            return {self.at: torch.zeros_like(chem)}
        if "phi" not in lvl.state_schema:
            raise ValueError("cell_chem_diffuse[fibre_anisotropic]: the cell set has no `phi` block (the "
                             "fibre angle); seed it, e.g. seed_state_from_file from a cardio fit.")
        i, j = ei[0], ei[1]; N = chem.shape[0]
        cen = lvl.get("centroid" if "centroid" in lvl.state_schema else "pos")
        dx = cen[j, :2] - cen[i, :2]
        th = torch.atan2(dx[:, 1], dx[:, 0])
        phi = lvl.get("phi")[:, 0]
        k = self.ratio
        a = 0.5 * sum(k * torch.cos(th - phi[q]) ** 2 + torch.sin(th - phi[q]) ** 2 for q in (i, j))
        wn = (a / a.mean()).to(chem.dtype)[:, None]
        deg = torch.zeros(N, device=chem.device, dtype=chem.dtype).index_add_(0, i, torch.ones_like(i, dtype=chem.dtype))
        flux = torch.zeros_like(chem).index_add_(0, i, wn * (chem[j] - chem[i]))
        lap = flux / deg.clamp(min=1)[:, None] if self.norm else flux
        _span(chem, self.chan, len(self.d), type(self).__name__)
        coef = torch.zeros(chem.shape[1], device=chem.device, dtype=chem.dtype)
        for _k, _dk in enumerate(self.d):
            coef[self.chan + _k] = _dk * self.chi
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: (coef[None, :] * lap) * occ}


# `cell_chem_diffuse[steady_uptake]` -- a nutrient supplied at the tissue's free edge and consumed by the
# living cells, at steady state every frame (exp12_tumor_invasion). Moved here from `nutrient_ops.py`
# on 2026-09-27 (experiments/INSTRUCTION.md: a variant lives beside its base operator).
def steady_uptake(pos, es, et, ef, nF, consume, c_init, uptake, K_m=0.01, supply=1.0,
                  newton_iters=40, tol=1e-6):
    """The nutrient at steady state on a cell mesh: the numerical core of
    `cell_chem_diffuse[steady_uptake]`, a pure numpy function so it can be tested on a hand-built mesh.

        0 = sum_j T_ij (c_j - c_i) + sum_b T_b (supply - c_i) - uptake * A_i * consume_i * c_i / (K_m + c_i)

    A two-point finite-volume balance on each cell i: T_ij = l_ij / |x_i - x_j| is the transmissibility
    of the wall cell i shares with j (l_ij the shared edge length, x the face centroids), T_b = l_b /
    |x_i - m_b| that of a FREE edge b (no twin) to the medium at its midpoint m_b, where the nutrient is
    held at `supply`. A_i is the cell's area (the norm of its vector area), `uptake` = a / D, the
    zero-order consumption over the diffusivity, in concentration / length^2, and K_m the Michaelis
    constant; both concentrations are in the units of `supply`, so lowering the supply at fixed
    uptake thins the fed rim as sqrt(supply). Solved by Newton on the uptake term from `c_init`; every
    iterate is clipped at 0. Returns (c [nF], newton iterations used, final max |dc|).
    """
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    pos = np.asarray(pos, float); es = np.asarray(es, np.int64); et = np.asarray(et, np.int64)
    ef = np.asarray(ef, np.int64)
    Nv = int(max(es.max(initial=0), et.max(initial=0))) + 1
    cnt = np.bincount(ef, minlength=nF).astype(float)
    cen = np.zeros((nF, 3)); np.add.at(cen, ef, pos[es]); cen /= np.maximum(cnt, 1)[:, None]
    va = np.zeros((nF, 3)); np.add.at(va, ef, 0.5 * np.cross(pos[es] - cen[ef], pos[et] - cen[ef]))
    A = np.linalg.norm(va, axis=1)
    key, tkey = es * Nv + et, et * Nv + es
    o = np.argsort(key); ks = key[o]
    j = np.clip(np.searchsorted(ks, tkey), 0, len(ks) - 1)
    has = ks[j] == tkey
    twin = np.where(has, ef[o[j]], -1)
    L = np.linalg.norm(pos[et] - pos[es], axis=1)
    inn = twin >= 0
    fi, fj = ef[inn], twin[inn]
    T = L[inn] / np.maximum(np.linalg.norm(cen[fi] - cen[fj], axis=1), 1e-12)
    bnd = ~inn
    mid = 0.5 * (pos[es[bnd]] + pos[et[bnd]])
    Tb = L[bnd] / np.maximum(np.linalg.norm(cen[ef[bnd]] - mid, axis=1), 1e-12)
    diag = np.bincount(fi, T, minlength=nF) + np.bincount(ef[bnd], Tb, minlength=nF)
    rhs0 = supply * np.bincount(ef[bnd], Tb, minlength=nF)
    M0 = coo_matrix((np.concatenate([diag + 1e-12, -T]), (np.concatenate([np.arange(nF), fi]),
                     np.concatenate([np.arange(nF), fj]))), shape=(nF, nF)).tocsr()
    q = float(uptake) * A * np.asarray(consume, float)
    K = float(K_m)
    c = np.clip(np.asarray(c_init, float), 0.0, None)
    dc, it = np.inf, 0
    for it in range(1, int(newton_iters) + 1):
        f = q * c / (K + c)
        fp = q * K / (K + c) ** 2
        J = M0 + coo_matrix((fp, (np.arange(nF), np.arange(nF))), shape=(nF, nF)).tocsr()
        cn = np.clip(spsolve(J.tocsc(), rhs0 - f + fp * c), 0.0, None)
        dc = float(np.max(np.abs(cn - c))) if nF else 0.0
        c = cn
        if dc < tol:
            break
    return c, it, dc


@register_operator("cell_chem_diffuse", set="cell", kind="lateral", family="fields", model="steady_uptake",
                   title="supply nutrient at the edge, consume inside",
                   equation=r"""$$0=D\nabla^{2}c-a\,\frac{c}{K_m+c}\,[\text{cell consumes}],\qquad c=c_{\text{supply}}\ \text{at the free edge}$$""")
class CellDiffuseSteadyUptake(Lateral):
    r"""the `steady_uptake` model of cell_chem_diffuse: a nutrient supplied by the medium at the tissue's free edge, diffusing through the cells and consumed by the living ones, at its steady state, solved every call.
    vertex -> cell: reads the mesh (walls, areas, the free edge), `alive` and `apop_flag`, writes chem[:, chan] in place with the solution (a two-point finite-volume balance, Newton on the Michaelis-Menten term).

        0=D\nabla^{2}c-a\,\frac{c}{K_m+c}\,[\text{cell consumes}],\qquad c=c_{\text{supply}}\ \text{at the free edge}
        D              the diffusion coefficient
        c              the nutrient concentration
        a              `uptake`, the consumption rate
        K_m            `K_m`, the Michaelis constant
        c_{supply}     `supply`, the medium's level at the free edge

    Reference: Grimes, D. R. et al. (2014). A method for estimating the oxygen consumption rate in multicellular tumour spheroids. J. R. Soc. Interface 11:20131124; Mueller-Klieser, W. (1984). Biophys. J. 46:343-348."""
    PARAM_UNITS = {'K_m': 'dimensionless', 'newton_iters': 'count', 'supply': 'dimensionless', 'tol': 'dimensionless', 'uptake': 'rate'}
    SYMBOLS = {'D': 'the diffusion coefficient', 'c': 'the nutrient concentration', 'a': '`uptake`, the consumption rate', 'K_m': '`K_m`, the Michaelis constant', 'c_{supply}': "`supply`, the medium's level at the free edge"}
    N_SPECIES = 1
    SUPPORTED_DIMS = [3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["uptake"]
    INPUTS = ["cell", "vertex"]; OUTPUTS = ["cell"]; READS = ["chem", "pos"]; WRITES = ["chem"]
    MECHANISM_TAGS = ["diffusion", "finite_volume", "nutrient", "consumption", "quasi_steady"]
    REFERENCE = ("Grimes, D. R. et al. (2014). J. R. Soc. Interface 11:20131124; Mueller-Klieser, W. "
                 "(1984). Biophys. J. 46:343-348.")
    PARAM_ROLES = {"uptake": "consumption_over_diffusivity", "supply": "edge_concentration",
                   "K_m": "michaelis_constant", "newton_iters": "solver_iterations", "tol": "solver_tolerance", 'vertex_set': 'the_vertex_set'}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.vat = params.get("vertex_set", "vertex")
        self.uptake = float(params["uptake"])
        self.supply = float(params.get("supply", 1.0))
        self.K_m = float(params.get("K_m", 0.01))
        self.newton_iters = int(params.get("newton_iters", 40))
        self.tol = float(params.get("tol", 1e-6))
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)
        self._warned = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at); vlvl = H.level(self.vat)
        chem = lvl.get("chem")
        m = getattr(vlvl, "_mesh", None)
        if m is None or int(m["nF"]) == 0:
            return {self.at: torch.zeros_like(chem)}
        _span(chem, self.chan, 1, type(self).__name__)
        nF = int(m["nF"]); Nv = int(m["Nv"])
        npy = lambda a: a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
        pos = npy(vlvl.get("pos")[:Nv]).astype(np.float64)
        consume = np.ones(nF)
        if m.get("alive") is not None:
            consume *= (npy(m["alive"])[:nF] > 0)
        flag = m.get("apop_flag")
        if flag is not None and len(flag) >= nF:
            consume *= (npy(flag)[:nF] <= 0)
        c0 = npy(chem[:nF, self.chan]).astype(np.float64)
        if not np.any(c0 > 0):                      # first call: start from the supply, not from zero
            c0 = np.full(nF, self.supply)
        c, it, dc = steady_uptake(pos, npy(m["E_srce"]), npy(m["E_trgt"]), npy(m["E_face"]), nF, consume,
                                  c0, self.uptake, self.K_m, self.supply, self.newton_iters, self.tol)
        if dc >= self.tol and not self._warned:
            print(f"[cell_chem_diffuse[steady_uptake]] Newton stopped at {it} iterations with max |dc| "
                  f"{dc:.2e} > tol {self.tol:.0e}; the field is not at its steady state.", flush=True)
            self._warned = True
        chem[:nF, self.chan] = torch.as_tensor(c, dtype=chem.dtype, device=chem.device)
        return {self.at: torch.zeros_like(chem)}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="gray_scott", title="run autocatalytic reactions",
                   species=(("a", "activator"), ("u", "substrate")),
                   equation=r"""$$\frac{da}{dt}=r\big(u a^{2}-(F+k)a\big),\qquad \frac{du}{dt}=r\big(-u a^{2}+F(1-u)\big)$$""")
class CellReactGrayScott(Lateral):
    r"""Gray-Scott autocatalysis: an activator that makes more of itself by consuming a substrate the system slowly replenishes, pattern by substrate depletion rather than inhibition.
    cell -> cell: reads chem = [a, u], emits d(chem)/dt; no relation traversed, the reaction is local.

        \frac{da}{dt}=r\big(u a^{2}-(F+k)a\big),\qquad \frac{du}{dt}=r\big(-u a^{2}+F(1-u)\big)
        a              the activator, dimensionless
        u              the substrate, fed toward 1
        r              `rate`, the time scale
        F              `F`, the feed rate, per time
        k              `kk`, the extra kill rate on the activator, per time

    Reference: Gray, P. & Scott, S. K. (1984). Chem. Eng. Sci. 39:1087-1097; Pearson, J. E. (1993). Complex patterns in a simple system. Science 261:189-192."""
    PARAM_UNITS = {'F': 'rate', 'kk': 'rate', 'rate': 'rate'}
    SYMBOLS = {'a': 'the activator, dimensionless', 'u': 'the substrate, fed toward 1', 'r': '`rate`, the time scale', 'F': '`F`, the feed rate, per time', 'k': '`kk`, the extra kill rate on the activator, per time'}

    N_SPECIES = 2
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["F", "kk"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]
    MECHANISM_TAGS = ["reaction", "autocatalysis", "turing", "gray_scott"]
    PARAM_ROLES = {"F": "feed_rate", "kk": "kill_rate", "rate": "reaction_time_scale"}
    REFERENCE = "Gray, P. & Scott, S. K. (1984). Chem. Eng. Sci. 39:1087-1097; Pearson, J. E. (1993). Science 261:189-192."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.F = float(params["F"]); self.kk = float(params["kk"]); self.rate = float(params.get("rate", 1.0))
        # WHICH SPECIES THIS INSTANCE OWNS: 0 is the first pair (chem columns 0,1) and is the
        # default, so every existing spec is unchanged. A second RD system lives at chan 2.
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        # `chan` NAMES THE PAIR THIS INSTANCE OWNS. It was hard-wired to columns 0 and 1, so a
        # second species in the same buffer was unreachable: two `cell_chem_react` operators would both
        # have driven the same two columns and the second would simply have overwritten the first.
        c = self.chan
        a = chem[:, c]; u = chem[:, c + 1]
        uaa = u * a * a
        da = uaa - (self.F + self.kk) * a
        du = -uaa + self.F * (1.0 - u)
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        # zero in every column but this species', so the delta is additive with the other species'
        out = torch.zeros_like(chem)
        out[:, c] = self.rate * da
        out[:, c + 1] = self.rate * du
        return {self.at: out * occ}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields",
                   model="rock_paper_scissor", title="run cyclic competition",
                   species=(("u", "competitor 1"), ("v", "competitor 2"), ("w", "competitor 3")),
                   equation=r"""$$\begin{aligned}\frac{du}{dt}&=r\,u\,(1-p-a\,v)\\ \frac{dv}{dt}&=r\,v\,(1-p-a\,w)\\ \frac{dw}{dt}&=r\,w\,(1-p-a\,u)\end{aligned}\qquad p=u+v+w$$""")
class CellReactRPS(Lateral):
    r"""May-Leonard cyclic competition: three species each suppressing the next, all limited by the total population; the asymmetry a is the whole model.
    cell -> cell: reads chem = [u, v, w], emits d(chem)/dt; local.

        \begin{aligned}\frac{du}{dt}&=r\,u\,(1-p-a\,v)\\ \frac{dv}{dt}&=r\,v\,(1-p-a\,w)\\ \frac{dw}{dt}&=r\,w\,(1-p-a\,u)\end{aligned}\qquad p=u+v+w
        u, v, w        the three species
        p              the total population u + v + w
        a              `a`, the cyclic suppression (0: neutral coexistence)
        r              `rate`

    Reference: May, R. M. & Leonard, W. J. (1975). Nonlinear aspects of competition between three species. SIAM J. Appl. Math. 29:243-253."""
    PARAM_UNITS = {'a': 'dimensionless', 'rate': 'rate'}
    SYMBOLS = {'u, v, w': 'the three species', 'p': 'the total population u + v + w', 'a': '`a`, the cyclic suppression (0: neutral coexistence)', 'r': '`rate`'}
    N_SPECIES = 3
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]
    MECHANISM_TAGS = ["reaction", "competition", "cyclic_dominance", "non_transitive",
                      "may_leonard", "rock_paper_scissor"]
    PARAM_ROLES = {"a": "cyclic_suppression", "rate": "reaction_time_scale"}
    REFERENCE = ("May, R. M. & Leonard, W. J. (1975). SIAM J. Appl. Math. 29:243-253; "
                 "Reichenbach, T., Mobilia, M. & Frey, E. (2007). Nature 448:1046-1049.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        # 0.6 is the value the ParticleGraph `RD_RPS` generator ran, kept so the two agree.
        self.a = self.tunable(params.get("a"), 0.6, tensor=True)
        self.rate = float(params.get("rate", 1.0))
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        u, v, w = _span(chem, self.chan, 3, type(self).__name__)
        p = u + v + w
        terms = (u * (1.0 - p - self.a * v),
                 v * (1.0 - p - self.a * w),
                 w * (1.0 - p - self.a * u))
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: _emit(chem, self.chan, terms, self.rate, occ)}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields",
                   model="gray_scott_coupled", title="run two coupled autocatalytic reactions",
                   species=(("a1", "activator 1"), ("u1", "substrate 1"), ("a2", "activator 2"), ("u2", "substrate 2")),
                   equation=r"""$$\begin{aligned}\frac{da_1}{dt}&=r\big(u_1a_1^{2}-(F_1+k_1)a_1-g\,a_1a_2\big), & \frac{du_1}{dt}&=r\big(-u_1a_1^{2}+F_1(1-u_1)\big)\\ \frac{da_2}{dt}&=r\big(u_2a_2^{2}-(F_2+k_2)a_2-g\,a_2a_1\big), & \frac{du_2}{dt}&=r\big(-u_2a_2^{2}+F_2(1-u_2)\big)\end{aligned}$$""")
class CellReactGrayScottCoupled(Lateral):
    r"""two Gray-Scott systems that compete for each other's activator, an activator-activator loss symmetric to both; at g = 0 exactly two independent systems, bit for bit.
    cell -> cell: reads chem = [a1, u1, a2, u2], emits d(chem)/dt; local.

        \begin{aligned}\frac{da_1}{dt}&=r\big(u_1a_1^{2}-(F_1+k_1)a_1-g\,a_1a_2\big), & \frac{du_1}{dt}&=r\big(-u_1a_1^{2}+F_1(1-u_1)\big)\\ \frac{da_2}{dt}&=r\big(u_2a_2^{2}-(F_2+k_2)a_2-g\,a_2a_1\big), & \frac{du_2}{dt}&=r\big(-u_2a_2^{2}+F_2(1-u_2)\big)\end{aligned}
        a_1, u_1       the first system's activator and substrate
        a_2, u_2       the second's
        F_1, F_2       `F`, `F2`, the feed rates
        k_1, k_2       `kk`, `kk2`, the kill rates
        g              `gamma`, the coupling
        r              `rate`

    Reference: Gray, P. & Scott, S. K. (1984). Chem. Eng. Sci. 39:1087-1097; the coupling: Plexus (this work)."""
    PARAM_UNITS = {'F': 'rate', 'F2': 'rate', 'gamma': 'dimensionless', 'kk': 'rate', 'kk2': 'rate', 'rate': 'rate'}
    SYMBOLS = {'a_1, u_1': "the first system's activator and substrate", 'a_2, u_2': "the second's", 'F_1, F_2': '`F`, `F2`, the feed rates', 'k_1, k_2': '`kk`, `kk2`, the kill rates', 'g': '`gamma`, the coupling', 'r': '`rate`'}
    N_SPECIES = 4
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["F", "kk"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]
    MECHANISM_TAGS = ["reaction", "autocatalysis", "turing", "gray_scott", "competition", "coupled"]
    PARAM_ROLES = {"F": "feed_rate", "kk": "kill_rate", "F2": "feed_rate_2", "kk2": "kill_rate_2",
                   "gamma": "cross_suppression", "rate": "reaction_time_scale"}
    REFERENCE = "Gray, P. & Scott, S. K. (1984). Chem. Eng. Sci. 39:1087-1097 (coupling: this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.F = float(params["F"]); self.kk = float(params["kk"])
        # SYSTEM B FALLS BACK TO SYSTEM A's PARAMETERS. Two identical systems that differ only by
        # their coupling is the control this model exists to be compared against.
        self.F2 = float(params.get("F2", self.F)); self.kk2 = float(params.get("kk2", self.kk))
        self.gamma = self.tunable(params.get("gamma"), 0.0, tensor=True)
        self.rate = float(params.get("rate", 1.0))
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        a1, u1, a2, u2 = _span(chem, self.chan, 4, type(self).__name__)
        x = self.gamma * a1 * a2
        terms = (u1 * a1 * a1 - (self.F + self.kk) * a1 - x,
                 -u1 * a1 * a1 + self.F * (1.0 - u1),
                 u2 * a2 * a2 - (self.F2 + self.kk2) * a2 - x,
                 -u2 * a2 * a2 + self.F2 * (1.0 - u2))
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: _emit(chem, self.chan, terms, self.rate, occ)}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="gierer_meinhardt", title="run activator-inhibitor reactions",
                   species=(("a", "activator"), ("h", "inhibitor")),
                   equation=r"""$$\frac{da}{dt}=r\Big(\rho\,\frac{a^{2}}{h}-\mu_a a+a_0\Big),\qquad \frac{dh}{dt}=r\big(\rho\,a^{2}-\mu_h h\big)$$""")
class CellReactGiererMeinhardt(Lateral):
    r"""Gierer-Meinhardt activator-inhibitor, the reaction-diffusion Okuda uses: a self-enhancing activator whose autocatalysis a^2/h the inhibitor damps.
    cell -> cell: reads chem = [a, h], emits d(chem)/dt; local.

        \frac{da}{dt}=r\Big(\rho\,\frac{a^{2}}{h}-\mu_a a+a_0\Big),\qquad \frac{dh}{dt}=r\big(\rho\,a^{2}-\mu_h h\big)
        a              the activator
        h              the inhibitor
        r              `rate`
        \rho           `gm_rho`, the production
        \mu_a          `mu_a`, the activator's decay, per time
        a_0            `a0`, the basal production
        \mu_h          `mu_h`, the inhibitor's decay (`sat` saturates the autocatalysis)

    Reference: Gierer, A. & Meinhardt, H. (1972). A theory of biological pattern formation. Kybernetik 12:30-39; Okuda, S. et al. (2018). Combining Turing and 3D vertex models reproduces autonomous multicellular morphogenesis of the tissue. Sci. Rep. 8:2386 (ref. 37)."""
    PARAM_UNITS = {'a0': 'dimensionless', 'gm_rho': 'dimensionless', 'mu_a': 'rate', 'mu_h': 'rate', 'rate': 'rate', 'sat': 'dimensionless'}
    SYMBOLS = {'a': 'the activator', 'h': 'the inhibitor', 'r': '`rate`', '\\rho': '`gm_rho`, the production', '\\mu_a': "`mu_a`, the activator's decay, per time", 'a_0': '`a0`, the basal production', '\\mu_h': "`mu_h`, the inhibitor's decay (`sat` saturates the autocatalysis)"}
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]
    MECHANISM_TAGS = ["reaction", "autocatalysis", "self_enhancing", "turing", "gierer_meinhardt"]
    PARAM_ROLES = {"gm_rho": "production", "mu_a": "activator_decay", "mu_h": "inhibitor_decay", "a0": "basal_source", 'rate': 'time_scale', 'sat': 'saturation_of_the_autocatalysis'}
    REFERENCE = "Gierer, A. & Meinhardt, H. (1972). A theory of biological pattern formation. Kybernetik 12:30-39."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.gm_rho = float(params.get("gm_rho", 1.0)); self.mu_a = float(params.get("mu_a", 1.0))
        self.mu_h = float(params.get("mu_h", 1.0)); self.a0 = float(params.get("a0", 0.01))
        self.rate = float(params.get("rate", 1.0))
        # Meinhardt SATURATION kappa: a^2/(h(1+kappa a^2)) bounds the activator peak so self-enhancement can't
        # run away under growth (keeps the red spot CONFINED -> red_over_tip ~1 instead of flooding). 0 = off.
        self.sat = self.tunable(params.get("sat"), 0.0, tensor=True)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        a = chem[:, 0].clamp(min=0.0); h = chem[:, 1].clamp(min=1e-3)   # h>0: the a^2/h autocatalysis stays finite
        auto = a * a / (h * (1.0 + self.sat * a * a))                   # SATURATED autocatalysis (bounded peak)
        da = self.gm_rho * auto - self.mu_a * a + self.a0
        dh = self.gm_rho * a * a - self.mu_h * h
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        # FULL WIDTH, columns 0 and 1 (exp05, 2026-11-14): the delta was [n, 2] whatever the width of `chem`, so a chem with a third
        # species (a crowding signal at chan 2) failed to add; for a width-2 chem the values are the same (rate x term x occ)
        return {self.at: _emit(chem, 0, (da, dh), self.rate, occ)}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="source_decay",
                   title="make and decay a morphogen",
                   species=(("c", "morphogen"),),
                   equation=r"""$$\frac{dc_i}{dt}=r\big(p\,\mathbb{1}[i\in S]-k\,c_i\big),\qquad S=\{i:\ x_i-x_{\min}<f\,(x_{\max}-x_{\min})\}$$""")
class CellReactSourceDecay(Lateral):
    r"""a morphogen made by a source population every frame and lost everywhere at a first-order rate: the floor plate secreting Shh into the neural tube, the source a fraction of the sheet's current extent.
    cell -> cell: reads chem = [c] and the centroids, emits dc/dt.

        \frac{dc_i}{dt}=r\big(p\,\mathbb{1}[i\in S]-k\,c_i\big),\qquad S=\{i:\ x_i-x_{\min}<f\,(x_{\max}-x_{\min})\}
        c_i            the morphogen in cell i
        r              `rate`
        p              `production`, the source cells' synthesis rate
        S              the source: the cells within the fraction f of the sheet's extent along `source.axis` from its edge
        k              `decay`, the first-order loss rate
        x_i            the centroid's coordinate
        f              `source.frac`

    Reference: Crick, F. (1970). Diffusion in embryogenesis. Nature 225:420-422; Kicheva, A. et al. (2014). Science 345:1254927 (the floor plate's share of the length)."""
    PARAM_UNITS = {'decay': 'rate', 'production': 'rate', 'rate': 'rate'}
    SYMBOLS = {'c_i': 'the morphogen in cell i', 'r': '`rate`', 'p': "`production`, the source cells' synthesis rate", 'S': "the source: the cells within the fraction f of the sheet's extent along `source.axis` from its edge", 'k': '`decay`, the first-order loss rate', 'x_i': "the centroid's coordinate", 'f': '`source.frac`'}
    N_SPECIES = 1
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["production", "decay"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem", "centroid"]; WRITES = ["chem"]
    MECHANISM_TAGS = ["reaction", "morphogen", "source", "decay", "gradient"]
    PARAM_ROLES = {"production": "source_synthesis_rate", "decay": "first_order_decay_rate",
                   "source": "source_region", "rate": "reaction_time_scale"}
    REFERENCE = "Crick, F. (1970). Nature 225:420-422; Kicheva, A. et al. (2014). Science 345:1254927."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.p = float(params["production"]); self.k = float(params["decay"])
        self.rate = float(params.get("rate", 1.0))
        src = params.get("source") or {}
        self.axis = int(src.get("axis", 0)); self.side = str(src.get("side", "low"))
        self.frac = float(src.get("frac", 0.07))
        if self.side not in ("low", "high"):
            raise ValueError(f"source_decay: source.side must be low or high, got {self.side!r}")
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)

    def source_mask(self, lvl):
        cen = lvl.get("centroid")
        x = cen[:, self.axis]
        occ = getattr(lvl, "occ", None)
        live = (occ > 0) if occ is not None else torch.ones_like(x, dtype=torch.bool)
        big = torch.finfo(x.dtype).max
        lo = torch.where(live, x, torch.full_like(x, big)).min()
        hi = torch.where(live, x, torch.full_like(x, -big)).max()
        L = hi - lo
        s = (x - lo) if self.side == "low" else (hi - x)
        return (s < self.frac * L) & live

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        (c,) = _span(chem, self.chan, 1, type(self).__name__)
        dc = self.p * self.source_mask(lvl).to(chem.dtype) - self.k * c
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: _emit(chem, self.chan, (dc,), self.rate, occ)}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="source_ptch",
                   title="make morphogen, trapped by its receptor",
                   species=(("c", "morphogen"), ("R", "receptor")),
                   equation=r"""$$\frac{dc_i}{dt}=r\big(p\,\mathbb{1}[i\in S]-k\,c_i-s\,R_i\,c_i\big),\qquad \frac{dR_i}{dt}=r\Big(b_0+b_1\frac{c_i^{\,n}}{K^n+c_i^{\,n}}-k_R\,R_i\Big)$$""")
class CellReactSourcePtch(CellReactSourceDecay):
    r"""source_decay with the morphogen also removed by a receptor the morphogen induces: Shh bound and sequestered by Ptch, whose own expression Hedgehog signalling raises.
    cell -> cell: reads chem = [c, R], emits d(chem)/dt (the source region as source_decay).

        \frac{dc_i}{dt}=r\big(p\,\mathbb{1}[i\in S]-k\,c_i-s\,R_i\,c_i\big),\qquad \frac{dR_i}{dt}=r\Big(b_0+b_1\frac{c_i^{\,n}}{K^n+c_i^{\,n}}-k_R\,R_i\Big)
        c_i            the morphogen
        R_i            the receptor level of cell i
        r, p, S, k     as source_decay
        s              `sequester`, the removal per unit receptor per unit time
        b_0            `r_basal`, the receptor's basal production
        b_1            `r_induced`, its induced production
        n, K           `r_hill`, `r_half`, the induction's Hill exponent and half point
        k_R            `r_decay`, the receptor's decay

    Reference: Li, P. et al. (2018). Morphogen gradient reconstitution reveals Hedgehog pathway design principles. Science 360:543-548."""
    PARAM_UNITS = {'r_basal': 'rate', 'r_decay': 'rate', 'r_half': 'dimensionless', 'r_hill': 'dimensionless', 'r_induced': 'rate', 'sequester': 'rate'}
    SYMBOLS = {'c_i': 'the morphogen', 'R_i': 'the receptor level of cell i', 'r, p, S, k': 'as source_decay', 's': '`sequester`, the removal per unit receptor per unit time', 'b_0': "`r_basal`, the receptor's basal production", 'b_1': '`r_induced`, its induced production', 'n, K': "`r_hill`, `r_half`, the induction's Hill exponent and half point", 'k_R': "`r_decay`, the receptor's decay"}
    N_SPECIES = 2
    MECHANISM_TAGS = CellReactSourceDecay.MECHANISM_TAGS + ["receptor", "sequestration", "feedback", "ptch"]
    PARAM_ROLES = dict(CellReactSourceDecay.PARAM_ROLES, sequester="receptor_removal_rate",
                       r_basal="receptor_basal_synthesis", r_induced="receptor_induced_synthesis",
                       r_half="receptor_induction_half_point", r_hill="receptor_induction_hill",
                       r_decay="receptor_turnover_rate")
    REFERENCE = "Li, P. et al. (2018). Science 360:543-548."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.s = self.tunable(params.get("sequester"), 0.0, tensor=True)
        self.b0 = float(params.get("r_basal", 0.0)); self.b1 = float(params.get("r_induced", 0.0))
        self.K = float(params.get("r_half", 1.0)); self.n = float(params.get("r_hill", 1.0))
        self.kR = self.tunable(params.get("r_decay"), 1.0, tensor=True)

    def rates(self, c, R, src):
        cp = c.clamp(min=0.0)
        dc = self.p * src - self.k * c - self.s * R * c
        dR = self.b0 + self.b1 * cp ** self.n / (self.K ** self.n + cp ** self.n) - self.kR * R
        return dc, dR

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        c, R = _span(chem, self.chan, 2, type(self).__name__)
        src = self.source_mask(lvl).to(chem.dtype)
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: _emit(chem, self.chan, self.rates(c, R, src), self.rate, occ)}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="stripe_decay",
                   title="make morphogen in a central stripe",
                   equation=r"""$$\frac{dc_i}{dt}=r\big(p\,\mathbf 1[\,|x_i-\bar x|<\tfrac f2 L\,]-k\,c_i\big)$$""")
class CellReactStripeDecay(CellReactSourceDecay):
    r"""source_decay with the source a stripe through the middle of the tissue instead of a band at one edge: Dpp made along the anterior-posterior compartment boundary of the wing-disc pouch.
    cell -> cell: reads chem = [c] and the centroids, emits dc/dt.

        \frac{dc_i}{dt}=r\big(p\,\mathbf 1[\,|x_i-\bar x|<\tfrac f2 L\,]-k\,c_i\big)
        c_i            the morphogen
        r, p, k        as source_decay
        x_i            the centroid's coordinate along `source.axis`
        \bar x         the live cells' mean on it
        f              `source.frac` (or `source.width`, a fixed width)
        L              the tissue's current extent

    Reference: Aegerter-Wilmsen, T. et al. (2012). Development 139:3221, Fig. 1B; Crick, F. (1970). Nature 225:420-422."""
    PARAM_UNITS = {}
    SYMBOLS = {'c_i': 'the morphogen', 'r, p, k': 'as source_decay', 'x_i': "the centroid's coordinate along `source.axis`", '\\bar x': "the live cells' mean on it", 'f': '`source.frac` (or `source.width`, a fixed width)', 'L': "the tissue's current extent"}
    PARAM_ROLES = {'source': 'axis_frac_or_width_of_the_stripe'}
    MECHANISM_TAGS = CellReactSourceDecay.MECHANISM_TAGS + ["stripe", "dpp"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        w = (params.get("source") or {}).get("width")
        self.width = None if w is None else float(w)

    def source_mask(self, lvl):
        cen = lvl.get("centroid")
        x = cen[:, self.axis]
        occ = getattr(lvl, "occ", None)
        live = (occ > 0) if occ is not None else torch.ones_like(x, dtype=torch.bool)
        big = torch.finfo(x.dtype).max
        lo = torch.where(live, x, torch.full_like(x, big)).min()
        hi = torch.where(live, x, torch.full_like(x, -big)).max()
        xbar = (x * live.to(x.dtype)).sum() / live.sum().clamp(min=1)
        half = 0.5 * self.width if self.width is not None else 0.5 * self.frac * (hi - lo)
        return ((x - xbar).abs() < half) & live


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="notch_delta",
                   title="pick fates by Notch-Delta inhibition",
                   species=(("N", "Notch activity"), ("D", "Delta (Dll1)"), ("Y", "nuclear YAP"), ("W", "Wnt")),
                   equation=r"""$$\begin{aligned}\frac{dN_i}{dt}&=r\Big(\frac{\bar D_i^{k}}{a+\bar D_i^{k}}-N_i\Big),\quad \bar D_i=\langle D_j\rangle_{j\sim i}\\ \frac{dD_i}{dt}&=r\,v\Big(\frac{Y_i}{1+bN_i^{h}}-D_i\Big)\\ \frac{dY_i}{dt}&=r\,k_y\Big(\frac{y_{\max}W_i^{n}}{K_w^{n}+W_i^{n}}-Y_i\Big)\ \ \text{after } t_{\rm Wnt}\\ \frac{dW_i}{dt}&=r\Big(p\,\frac{D_i^{q}}{\theta^{q}+D_i^{q}}-k_wW_i\Big)\end{aligned}$$""")
class CellReactNotchDelta(Lateral):
    r"""Serra et al. 2019's symmetry breaking as a reaction: YAP variability biases Notch-Delta lateral inhibition, the winners (Dll1-high, Paneth) secrete Wnt, and Wnt is what the crypt becomes.
    cell -[cell adjacency]-> cell: reads chem = [N, D, Y, W] and edge_index (juxtacrine: the mean Delta of the neighbours), emits d(chem)/dt.

        \begin{aligned}\frac{dN_i}{dt}&=r\Big(\frac{\bar D_i^{k}}{a+\bar D_i^{k}}-N_i\Big),\quad \bar D_i=\langle D_j\rangle_{j\sim i}\\ \frac{dD_i}{dt}&=r\,v\Big(\frac{Y_i}{1+bN_i^{h}}-D_i\Big)\\ \frac{dY_i}{dt}&=r\,k_y\Big(\frac{y_{\max}W_i^{n}}{K_w^{n}+W_i^{n}}-Y_i\Big)\ \ \text{after } t_{\rm Wnt}\\ \frac{dW_i}{dt}&=r\Big(p\,\frac{D_i^{q}}{\theta^{q}+D_i^{q}}-k_wW_i\Big)\end{aligned}
        N_i            cell i's Notch activity
        \bar D_i       the mean Delta of its neighbours
        k, a           `k`, `a`, the Notch activation's Hill exponent and constant
        D_i            its Delta (Dll1)
        v              `v`, Delta's rate
        Y_i            its YAP
        b, h           `b`, `h`, the inhibition's constant and exponent
        k_y            `k_y`, YAP's rate
        y_{max}        `y_max`
        W_i            its Wnt
        K_w, n         `K_w`, `n`, the Wnt induction of YAP
        p, q, \theta   `p`, `q`, `theta`, the Wnt secretion by Delta-high cells
        k_w            `k_w`, Wnt's decay

    Reference: Collier, J. R., Monk, N. A. M., Maini, P. K. & Lewis, J. H. (1996). Pattern formation by lateral inhibition with feedback. J. Theor. Biol. 183:429-446; Serra, D. et al. (2019). Nature 569:66-72."""
    PARAM_UNITS = {'K_w': 'dimensionless', 'a': 'dimensionless', 'b': 'dimensionless', 'h': 'dimensionless', 'k': 'dimensionless', 'k_w': 'rate', 'k_y': 'rate', 'm': 'dimensionless', 'n': 'dimensionless', 'p': 'rate', 'q': 'dimensionless', 'rate': 'rate', 'seed': 'count', 'sigma_d': 'fraction', 'theta': 'dimensionless', 'v': 'rate', 'wnt_off': 'frames', 'y_max': 'dimensionless', 'y_th': 'dimensionless'}
    SYMBOLS = {'N_i': "cell i's Notch activity", '\\bar D_i': 'the mean Delta of its neighbours', 'k, a': "`k`, `a`, the Notch activation's Hill exponent and constant", 'D_i': 'its Delta (Dll1)', 'v': "`v`, Delta's rate", 'Y_i': 'its YAP', 'b, h': "`b`, `h`, the inhibition's constant and exponent", 'k_y': "`k_y`, YAP's rate", 'y_{max}': '`y_max`', 'W_i': 'its Wnt', 'K_w, n': '`K_w`, `n`, the Wnt induction of YAP', 'p, q, \\theta': '`p`, `q`, `theta`, the Wnt secretion by Delta-high cells', 'k_w': "`k_w`, Wnt's decay"}
    N_SPECIES = 4
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem", "edge_index"]; WRITES = ["chem"]
    MECHANISM_TAGS = ["reaction", "lateral_inhibition", "notch_delta", "juxtacrine", "symmetry_breaking",
                      "wnt_source"]
    PARAM_ROLES = {"a": "notch_activation_threshold", "b": "delta_repression_strength", "k": "notch_hill",
                   "h": "delta_hill", "v": "delta_rate_over_notch_rate", "p": "wnt_production",
                   "theta": "delta_level_for_wnt", "q": "wnt_hill", "k_w": "wnt_decay",
                   "k_y": "yap_relaxation_after_wnt_withdrawal", "K_w": "wnt_that_holds_yap_half", "n": "yap_wnt_hill",
                   "y_th": "yap_threshold_for_delta", "m": "yap_delta_hill", "y_max": "yap_level_wnt_holds",
                   "sigma_d": "delta_expression_noise", "wnt_off": "frame_exogenous_wnt_ends",
                   "rate": "reaction_time_scale", 'seed': 'seed_of_the_expression_noise'}
    REFERENCE = ("Collier, J. R. et al. (1996). J. Theor. Biol. 183:429-446; "
                 "Serra, D. et al. (2019). Nature 569:66-72.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.a = float(params.get("a", 0.01)); self.b = float(params.get("b", 100.0))
        self.k = float(params.get("k", 2.0)); self.h = float(params.get("h", 2.0))
        self.v = self.tunable(params.get("v"), 1.0, tensor=True)
        self.p = float(params.get("p", 0.0)); self.theta = float(params.get("theta", 0.5))
        self.q = float(params.get("q", 4.0)); self.k_w = float(params.get("k_w", 1.0))
        self.k_y = float(params.get("k_y", 0.0)); self.K_w = float(params.get("K_w", 0.1))
        self.n = float(params.get("n", 2.0))
        _yt = params.get("y_th", None)
        self.y_th = None if _yt is None else float(_yt)
        self.m = float(params.get("m", 8.0))
        self.y_max = self.tunable(params.get("y_max"), 1.0, tensor=True)
        self.sigma_d = float(params.get("sigma_d", 0.0))
        self._gen = torch.Generator(device="cpu"); self._gen.manual_seed(int(params.get("seed", 0)))
        _wo = params.get("wnt_off", None)
        self.wnt_off = None if _wo is None else float(_wo)
        self.rate = float(params.get("rate", 1.0))
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        N, D, Y, W = _span(chem, self.chan, 4, type(self).__name__)
        ei = getattr(lvl, "edge_index", None)
        if ei is None or ei.numel() == 0:
            Dbar = torch.zeros_like(D)
        else:
            i, j = ei[0], ei[1]
            agg = torch.zeros_like(D).index_add_(0, i, D[j])
            deg = torch.zeros_like(D).index_add_(0, i, torch.ones_like(D[j]))
            Dbar = agg / deg.clamp(min=1)
        Dp = Dbar.clamp(min=0) ** self.k
        dN = Dp / (self.a + Dp) - N
        if self.y_th is None:
            prod = Y
        else:
            Ym = Y.clamp(min=0) ** self.m
            prod = Ym / (self.y_th ** self.m + Ym)
        if self.sigma_d > 0:
            xi = torch.randn(prod.shape[0], generator=self._gen, dtype=torch.float64)
            prod = prod * (1.0 + self.sigma_d * xi).clamp(min=0).to(device=prod.device, dtype=prod.dtype)
        dD = self.v * (prod / (1.0 + self.b * N.clamp(min=0) ** self.h) - D)
        fr = getattr(H, "frame", None)
        if self.wnt_off is not None and self.k_y > 0 and fr is not None and float(fr) >= self.wnt_off:
            Wn = W.clamp(min=0) ** self.n
            dY = self.k_y * (self.y_max * Wn / (self.K_w ** self.n + Wn) - Y)
        else:
            dY = torch.zeros_like(Y)
        Dq = D.clamp(min=0) ** self.q
        dW = self.p * Dq / (self.theta ** self.q + Dq) - self.k_w * W
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: _emit(chem, self.chan, (dN, dD, dY, dW), self.rate, occ)}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="balaskas",
                   title="read the Shh gradient into fates",
                   species=(("P", "Pax6"), ("O", "Olig2"), ("N", "Nkx2.2")),
                   equation=r"""$$\begin{aligned}\frac{dP}{dt}&=r\Big(\frac{\alpha}{1+(N/N_{cP})^{h_1}+(O/O_{cP})^{h_2}}-k_1P\Big)\\ \frac{dO}{dt}&=r\Big(\frac{\beta\,G^{n}}{1+G^{n}}\,\frac{1}{1+(N/N_{cO})^{h_3}}-k_2O\Big)\\ \frac{dN}{dt}&=r\Big(\frac{\gamma\,G^{m}}{1+G^{m}}\,\frac{1}{1+(O/O_{cN})^{h_4}+(P/P_{cN})^{h_5}}-k_3N\Big)\end{aligned}$$""")
class CellReactBalaskas(Lateral):
    r"""the three-gene cross-repressive circuit that reads the Shh gradient in the neural tube into three domains: Pax6, Olig2, Nkx2.2 (Balaskas et al. 2012, eqs. 1-3, every default Table S2).
    cell -> cell: reads chem = [P, O, N] and the morphogen column `g_col`, emits d(P, O, N)/dt; local.

        \begin{aligned}\frac{dP}{dt}&=r\Big(\frac{\alpha}{1+(N/N_{cP})^{h_1}+(O/O_{cP})^{h_2}}-k_1P\Big)\\ \frac{dO}{dt}&=r\Big(\frac{\beta\,G^{n}}{1+G^{n}}\,\frac{1}{1+(N/N_{cO})^{h_3}}-k_2O\Big)\\ \frac{dN}{dt}&=r\Big(\frac{\gamma\,G^{m}}{1+G^{m}}\,\frac{1}{1+(O/O_{cN})^{h_4}+(P/P_{cN})^{h_5}}-k_3N\Big)\end{aligned}
        P, O, N        the levels of Pax6, Olig2 and Nkx2.2 (high above 1)
        r              `rate`
        G              the Shh-Gli input, `g_gain` x chem[`g_col`]
        \alpha, \beta, \gamma the three productions (alpha 0: Pax6-/-, beta 0: Olig2-/-)
        h_1 \ldots h_5 the repressions' Hill exponents
        N_{cP}, O_{cP}, N_{cO}, O_{cN}, P_{cN} the critical levels
        n, m           the input's exponents
        k_1, k_2, k_3  the decays

    Reference: Balaskas, N. et al. (2012). Gene regulatory logic for reading the Sonic Hedgehog signaling gradient in the vertebrate neural tube. Cell 148:273-284."""
    PARAM_UNITS = {'g_col': 'count', 'g_gain': 'dimensionless', 'rate': 'rate'}
    SYMBOLS = {'P, O, N': 'the levels of Pax6, Olig2 and Nkx2.2 (high above 1)', 'r': '`rate`', 'G': 'the Shh-Gli input, `g_gain` x chem[`g_col`]', '\\alpha, \\beta, \\gamma': 'the three productions (alpha 0: Pax6-/-, beta 0: Olig2-/-)', 'h_1 \\ldots h_5': "the repressions' Hill exponents", 'N_{cP}, O_{cP}, N_{cO}, O_{cN}, P_{cN}': 'the critical levels', 'n, m': "the input's exponents", 'k_1, k_2, k_3': 'the decays'}
    N_SPECIES = 3
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["g_col"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]
    MECHANISM_TAGS = ["gene_regulatory_network", "cross_repression", "morphogen_readout", "balaskas"]
    PARAM_ROLES = {"alpha": "pax6_max_rate", "beta": "olig2_max_rate", "gamma": "nkx22_max_rate",
                   "h1": "N_represses_P", "h2": "O_represses_P", "h3": "N_represses_O",
                   "h4": "O_represses_N", "h5": "P_represses_N", "g_col": "signal_column",
                   "g_gain": "signal_scale", "rate": "reaction_time_scale"}
    REFERENCE = "Balaskas, N. et al. (2012). Cell 148:273-284 (eqs. 1-3, Table S2)."
    DEFAULTS = dict(alpha=3.0, beta=5.0, gamma=5.0, h1=6.0, h2=2.0, h3=5.0, h4=1.0, h5=1.0,
                    k1=1.0, k2=1.0, k3=1.0, NcritP=1.0, OcritP=1.0, NcritO=1.0, OcritN=1.0, PcritN=1.0,
                    n=1.0, m=1.0)

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        for k, v in self.DEFAULTS.items():
            setattr(self, k, float(params.get(k, v)))
        self.g_col = int(params["g_col"]); self.g_gain = float(params.get("g_gain", 1.0))
        self.rate = float(params.get("rate", 1.0))
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)
        if self.chan <= self.g_col < self.chan + 3:
            raise ValueError(f"balaskas: g_col={self.g_col} lies inside this model's own span "
                             f"{self.chan}..{self.chan + 2} -- the signal is read from another column")

    def rates(self, P, O, N, G):
        """(dP, dO, dN) before `rate`, on tensors; the levels are clipped at 0 before any power."""
        P, O, N, G = P.clamp(min=0.0), O.clamp(min=0.0), N.clamp(min=0.0), G.clamp(min=0.0)
        Gn = G ** self.n / (1.0 + G ** self.n)
        Gm = G ** self.m / (1.0 + G ** self.m)
        dP = self.alpha / (1.0 + (N / self.NcritP) ** self.h1 + (O / self.OcritP) ** self.h2) - self.k1 * P
        dO = self.beta * Gn / (1.0 + (N / self.NcritO) ** self.h3) - self.k2 * O
        dN = self.gamma * Gm / (1.0 + (O / self.OcritN) ** self.h4 + (P / self.PcritN) ** self.h5) - self.k3 * N
        return dP, dO, dN

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        P, O, N = _span(chem, self.chan, 3, type(self).__name__)
        if self.g_col >= chem.shape[1]:
            raise ValueError(f"balaskas: g_col={self.g_col} but `chem` is {chem.shape[1]} wide")
        G = self.g_gain * chem[:, self.g_col]
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: _emit(chem, self.chan, self.rates(P, O, N, G), self.rate, occ)}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="aliev_panfilov",
                   title="fire and recover excitable cells",
                   species=(("u", "excitation"), ("v", "recovery")),
                   equation=r"""$$\frac{du}{dt}=r\big(-k\,u(u-a)(u-1)-u\,v+I_{\mathrm{stim}}\big),\qquad \frac{dv}{dt}=r\,\varepsilon(u,v)\big(-v-k\,u(u-a-1)\big),\qquad \varepsilon=\varepsilon_0+\frac{\mu_1 v}{u+\mu_2}$$""")
class CellReactAlievPanfilov(Lateral):
    r"""Aliev-Panfilov excitable kinetics: the smallest model with a threshold, a full-size action potential and a pacing-dependent refractory tail; the stimulus is a membrane current of the model, not an operator.
    cell -> cell: reads chem = [u, v] and the centroids (for the stimulus region), emits d(chem)/dt; the coupling is cell_chem_diffuse with `d: [D, 0]`.

        \frac{du}{dt}=r\big(-k\,u(u-a)(u-1)-u\,v+I_{\mathrm{stim}}\big),\qquad \frac{dv}{dt}=r\,\varepsilon(u,v)\big(-v-k\,u(u-a-1)\big),\qquad \varepsilon=\varepsilon_0+\frac{\mu_1 v}{u+\mu_2}
        u              the excitation (E = 100 u - 80 mV)
        v              the recovery variable
        r              `rate` (one model time unit is 12.9 ms)
        k, a           `k`, `a`, the cubic's parameters
        I_{stim}       `stim`, the injected current on the cells in the stimulus region at the stimulus times
        \varepsilon(u, v) the recovery's rate
        \varepsilon_0, \mu_1, \mu_2 `eps0`, `mu1`, `mu2`

    Reference: Aliev, R. R. & Panfilov, A. V. (1996). A simple two-variable model of cardiac excitation. Chaos Solitons Fractals 7:293-301."""
    PARAM_UNITS = {'a': 'dimensionless', 'eps0': 'dimensionless', 'k': 'dimensionless', 'mu1': 'dimensionless', 'mu2': 'dimensionless', 'rate': 'rate'}
    SYMBOLS = {'u': 'the excitation (E = 100 u - 80 mV)', 'v': 'the recovery variable', 'r': '`rate` (one model time unit is 12.9 ms)', 'k, a': "`k`, `a`, the cubic's parameters", 'I_{stim}': '`stim`, the injected current on the cells in the stimulus region at the stimulus times', '\\varepsilon(u, v)': "the recovery's rate", '\\varepsilon_0, \\mu_1, \\mu_2': '`eps0`, `mu1`, `mu2`'}

    N_SPECIES = 2
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem", "centroid"]; WRITES = ["chem"]
    MECHANISM_TAGS = ["reaction", "excitable", "refractory", "action_potential", "aliev_panfilov"]
    PARAM_ROLES = {"k": "upstroke_rate", "a": "threshold", "eps0": "recovery_rate",
                   "mu1": "restitution_1", "mu2": "restitution_2", "rate": "reaction_time_scale",
                   "stim": "stimulus_current"}
    REFERENCE = "Aliev, R. R. & Panfilov, A. V. (1996). Chaos Solitons Fractals 7:293-301."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.k = float(params.get("k", 8.0)); self.a = float(params.get("a", 0.15))
        self.eps0 = self.tunable(params.get("eps0"), 0.002, tensor=True)
        self.mu1 = float(params.get("mu1", 0.2)); self.mu2 = float(params.get("mu2", 0.3))
        self.rate = float(params.get("rate", 1.0))
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)
        s = params.get("stim") or {}
        self.stims = [self._site(x) for x in (s if isinstance(s, (list, tuple)) else [s])]
        first = self.stims[0]
        self.stim_times, self.stim_dur, self.stim_amp = first["times"], first["dur"], first["amp"]

    @staticmethod
    def _site(s):
        st = dict(times=[float(t) for t in (s.get("times") or [])], dur=float(s.get("duration", 0.5)),
                  amp=float(s.get("amp", 0.0)), center=s.get("center"), radius=float(s.get("radius", 0.0)),
                  below=s.get("below"), box=s.get("box"))
        if (st["times"] and st["amp"] and st["center"] is None and st["below"] is None
                and st["box"] is None):
            raise ValueError("aliev_panfilov: `stim` needs a region -- `center` + `radius`, "
                             "`below: {axis, value}` or `box`. A stimulus on every cell is not a stimulus site.")
        return st

    def _stim(self, H, lvl, like):
        out = torch.zeros_like(like)
        if not any(s["times"] and s["amp"] for s in self.stims):
            return out
        cen = lvl.get("centroid" if "centroid" in lvl.state_schema else "pos")
        ft = getattr(H, "frame_t", None)
        t = (ft if ft is not None else torch.tensor(float(getattr(H, "frame", 0)))).to(like) * float(H.dt)
        for s in self.stims:
            if not (s["times"] and s["amp"]):
                continue
            if s["below"] is not None:
                where = cen[:, int(s["below"].get("axis", 0))] < float(s["below"]["value"])
            elif s["box"] is not None:
                lo = torch.as_tensor([float(x) for x in s["box"][0]], dtype=cen.dtype, device=cen.device)
                hi = torch.as_tensor([float(x) for x in s["box"][1]], dtype=cen.dtype, device=cen.device)
                k = len(lo)
                where = ((cen[:, :k] >= lo) & (cen[:, :k] <= hi)).all(1)
            else:
                c = torch.as_tensor([float(x) for x in s["center"]], dtype=cen.dtype, device=cen.device)
                where = (cen[:, :len(c)] - c).norm(dim=1) <= s["radius"]
            t0 = torch.as_tensor(s["times"], dtype=like.dtype, device=like.device)
            on = ((t >= t0) & (t < t0 + s["dur"])).any().to(like.dtype)
            out = out + s["amp"] * on * where.to(like.dtype)
        return out

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        u, v = _span(chem, self.chan, 2, type(self).__name__)
        eps = self.eps0 + self.mu1 * v / (u + self.mu2)
        du = -self.k * u * (u - self.a) * (u - 1.0) - u * v + self._stim(H, lvl, u)
        dv = eps * (-v - self.k * u * (u - self.a - 1.0))
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: _emit(chem, self.chan, (du, dv), self.rate, occ)}


# `cell_grow`, AND THE OLD NAME IS GONE RATHER THAN ALIASED. An alias makes two names for one
# thing and leaves a reader unable to tell which a specification meant; prior specifications are
# migrated instead.
#
# It is growth. It does not produce a morphogen, it READS one and uses it as a per-cell rate: the
# morphogen is a GATE on this operator, and the composition space already declares that gate as an
# optional slot. With the gate open (`a_sw = 0`) the same operator is plain uniform growth. Naming
# the gate in the operator made the optional half look mandatory, and made the sibling pair
# unreadable -- `cell_grow` / `cell_divide` says what the schedule actually does.
@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="balaskas_schedule",
                   title="read the Shh gradient, input changed midway",
                   species=(("P", "Pax6"), ("O", "Olig2"), ("N", "Nkx2.2")),
                   equation=r"""$$\begin{aligned}\frac{dP}{dt}&=r\Big(\frac{\alpha}{1+(N/N_{cP})^{h_1}+(O/O_{cP})^{h_2}}-k_1P\Big)\\ \frac{dO}{dt}&=r\Big(\frac{\beta\,G^{n}}{1+G^{n}}\,\frac{1}{1+(N/N_{cO})^{h_3}}-k_2O\Big)\\ \frac{dN}{dt}&=r\Big(\frac{\gamma\,G^{m}}{1+G^{m}}\,\frac{1}{1+(O/O_{cN})^{h_4}+(P/P_{cN})^{h_5}}-k_3N\Big)\end{aligned}$$""")
class CellReactBalaskasSchedule(CellReactBalaskas):
    r"""balaskas whose Gli input is multiplied by a piecewise-constant factor in circuit time: a drug added to the dish at a declared time (Dessaud's cyclopamine).
    cell -> cell: as balaskas, G = g c s(t), `schedule: [[t_1, s_1], ...]`, s = 1 before t_1.

        \begin{aligned}\frac{dP}{dt}&=r\Big(\frac{\alpha}{1+(N/N_{cP})^{h_1}+(O/O_{cP})^{h_2}}-k_1P\Big)\\ \frac{dO}{dt}&=r\Big(\frac{\beta\,G^{n}}{1+G^{n}}\,\frac{1}{1+(N/N_{cO})^{h_3}}-k_2O\Big)\\ \frac{dN}{dt}&=r\Big(\frac{\gamma\,G^{m}}{1+G^{m}}\,\frac{1}{1+(O/O_{cN})^{h_4}+(P/P_{cN})^{h_5}}-k_3N\Big)\end{aligned}
        P, O, N        the levels of Pax6, Olig2 and Nkx2.2 (high above 1)
        r              `rate`
        G              the Shh-Gli input, `g_gain` x chem[`g_col`]
        \alpha, \beta, \gamma the three productions (alpha 0: Pax6-/-, beta 0: Olig2-/-)
        h_1 \ldots h_5 the repressions' Hill exponents
        N_{cP}, O_{cP}, N_{cO}, O_{cN}, P_{cN} the critical levels
        n, m           the input's exponents
        k_1, k_2, k_3  the decays

    Reference: Dessaud, E. et al. (2007). Nature 450:717-720, Fig. 2c-e; Balaskas, N. et al. (2012). Cell 148:273-284."""
    PARAM_UNITS = {}
    SYMBOLS = {'P, O, N': 'the levels of Pax6, Olig2 and Nkx2.2 (high above 1)', 'r': '`rate`', 'G': 'the Shh-Gli input, `g_gain` x chem[`g_col`]', '\\alpha, \\beta, \\gamma': 'the three productions (alpha 0: Pax6-/-, beta 0: Olig2-/-)', 'h_1 \\ldots h_5': "the repressions' Hill exponents", 'N_{cP}, O_{cP}, N_{cO}, O_{cN}, P_{cN}': 'the critical levels', 'n, m': "the input's exponents", 'k_1, k_2, k_3': 'the decays'}
    PARAM_ROLES = dict(CellReactBalaskas.PARAM_ROLES, schedule="input_gain_schedule")
    REFERENCE = ("Dessaud, E. et al. (2007). Nature 450:717-720; Balaskas, N. et al. (2012). "
                 "Cell 148:273-284.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        sch = sorted([(float(t), float(f)) for t, f in (params.get("schedule") or [])])
        self.sched_t = [t for t, _ in sch]
        self.sched_f = [f for _, f in sch]

    def factor(self, t):
        """s(t) on a 0-d tensor: the factor of the last step at or before t, 1 before the first."""
        s = torch.ones_like(t)
        for tk, fk in zip(self.sched_t, self.sched_f):
            s = torch.where(t >= tk, torch.full_like(t, fk), s)
        return s

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        P, O, N = _span(chem, self.chan, 3, type(self).__name__)
        if self.g_col >= chem.shape[1]:
            raise ValueError(f"balaskas_schedule: g_col={self.g_col} but `chem` is {chem.shape[1]} wide")
        ft = getattr(H, "frame_t", None)
        frame = (ft if ft is not None else torch.tensor(float(getattr(H, "frame", 0)))).to(chem)
        G = self.g_gain * chem[:, self.g_col] * self.factor(self.rate * frame * float(H.dt))
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: _emit(chem, self.chan, self.rates(P, O, N, G), self.rate, occ)}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="balaskas_adapt",
                   title="read an adapting Shh input",
                   species=(("P", "Pax6"), ("O", "Olig2"), ("N", "Nkx2.2")),
                   equation=r"""$$\begin{aligned}\frac{dP}{dt}&=r\Big(\frac{\alpha}{1+(N/N_{cP})^{h_1}+(O/O_{cP})^{h_2}}-k_1P\Big)\\ \frac{dO}{dt}&=r\Big(\frac{\beta\,G^{n}}{1+G^{n}}\,\frac{1}{1+(N/N_{cO})^{h_3}}-k_2O\Big)\\ \frac{dN}{dt}&=r\Big(\frac{\gamma\,G^{m}}{1+G^{m}}\,\frac{1}{1+(O/O_{cN})^{h_4}+(P/P_{cN})^{h_5}}-k_3N\Big)\end{aligned}$$""")
class CellReactBalaskasAdapt(CellReactBalaskas):
    r"""balaskas with the paper's own time course of Gli activity: the intracellular input rises, peaks at `t_peak` and retracts while the morphogen itself does not.
    cell -> cell: as balaskas, G = g c (t/t_p)^2 e^{2(1 - t/t_p)}.

        \begin{aligned}\frac{dP}{dt}&=r\Big(\frac{\alpha}{1+(N/N_{cP})^{h_1}+(O/O_{cP})^{h_2}}-k_1P\Big)\\ \frac{dO}{dt}&=r\Big(\frac{\beta\,G^{n}}{1+G^{n}}\,\frac{1}{1+(N/N_{cO})^{h_3}}-k_2O\Big)\\ \frac{dN}{dt}&=r\Big(\frac{\gamma\,G^{m}}{1+G^{m}}\,\frac{1}{1+(O/O_{cN})^{h_4}+(P/P_{cN})^{h_5}}-k_3N\Big)\end{aligned}
        P, O, N        the levels of Pax6, Olig2 and Nkx2.2 (high above 1)
        r              `rate`
        G              the Shh-Gli input, `g_gain` x chem[`g_col`]
        \alpha, \beta, \gamma the three productions (alpha 0: Pax6-/-, beta 0: Olig2-/-)
        h_1 \ldots h_5 the repressions' Hill exponents
        N_{cP}, O_{cP}, N_{cO}, O_{cN}, P_{cN} the critical levels
        n, m           the input's exponents
        k_1, k_2, k_3  the decays

    Reference: Balaskas, N. et al. (2012). Cell 148:273-284, Figs. 7A and S7A; Dessaud, E. et al. (2007). Nature 450:717-720."""
    PARAM_UNITS = {'t_peak': 'time'}
    SYMBOLS = {'P, O, N': 'the levels of Pax6, Olig2 and Nkx2.2 (high above 1)', 'r': '`rate`', 'G': 'the Shh-Gli input, `g_gain` x chem[`g_col`]', '\\alpha, \\beta, \\gamma': 'the three productions (alpha 0: Pax6-/-, beta 0: Olig2-/-)', 'h_1 \\ldots h_5': "the repressions' Hill exponents", 'N_{cP}, O_{cP}, N_{cO}, O_{cN}, P_{cN}': 'the critical levels', 'n, m': "the input's exponents", 'k_1, k_2, k_3': 'the decays'}
    PARAM_ROLES = dict(CellReactBalaskas.PARAM_ROLES, t_peak="signal_peak_time")
    REFERENCE = ("Balaskas, N. et al. (2012). Cell 148:273-284 (eqs. 1-3, Table S2, Fig. S7A); "
                 "Dessaud, E. et al. (2007). Nature 450:717-720.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.t_peak = float(params.get("t_peak", 12.5))
        if self.t_peak <= 0:
            raise ValueError("balaskas_adapt: t_peak must be > 0 (circuit time units)")

    def profile(self, t):
        """f(t) = (t / t_p)^2 e^(2 (1 - t / t_p)): 1 at t = t_p, 0 at t = 0; a tensor or a float."""
        x = t / self.t_peak
        return x * x * torch.exp(2.0 * (1.0 - x)) if torch.is_tensor(x) else x * x * float(np.exp(2.0 * (1.0 - x)))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        P, O, N = _span(chem, self.chan, 3, type(self).__name__)
        if self.g_col >= chem.shape[1]:
            raise ValueError(f"balaskas_adapt: g_col={self.g_col} but `chem` is {chem.shape[1]} wide")
        # THE DEVICE CLOCK, as aliev_panfilov reads it, so a captured CUDA graph does not replay frame 1.
        ft = getattr(H, "frame_t", None)
        frame = (ft if ft is not None else torch.tensor(float(getattr(H, "frame", 0)))).to(chem)
        G = self.g_gain * chem[:, self.g_col] * self.profile(self.rate * frame * float(H.dt))
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: _emit(chem, self.chan, self.rates(P, O, N, G), self.rate, occ)}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="balaskas_commit",
                   title="read an adapting Shh input, then commit",
                   species=(("P", "Pax6"), ("O", "Olig2"), ("N", "Nkx2.2")),
                   equation=r"""$$\begin{aligned}\frac{dP}{dt}&=r\Big(\frac{\alpha}{1+(N/N_{cP})^{h_1}+(O/O_{cP})^{h_2}}-k_1P\Big)\\ \frac{dO}{dt}&=r\Big(\frac{\beta\,G^{n}}{1+G^{n}}\,\frac{1}{1+(N/N_{cO})^{h_3}}-k_2O\Big)\\ \frac{dN}{dt}&=r\Big(\frac{\gamma\,G^{m}}{1+G^{m}}\,\frac{1}{1+(O/O_{cN})^{h_4}+(P/P_{cN})^{h_5}}-k_3N\Big)\end{aligned}$$""")
class CellReactBalaskasCommit(CellReactBalaskasAdapt):
    r"""balaskas_adapt until the circuit time `t_commit`, then nothing: every cell keeps its P, O, N (inherited on division), the second phase of neural-tube patterning.
    cell -> cell: as balaskas_adapt while t < t_commit; zero rates after.

        \begin{aligned}\frac{dP}{dt}&=r\Big(\frac{\alpha}{1+(N/N_{cP})^{h_1}+(O/O_{cP})^{h_2}}-k_1P\Big)\\ \frac{dO}{dt}&=r\Big(\frac{\beta\,G^{n}}{1+G^{n}}\,\frac{1}{1+(N/N_{cO})^{h_3}}-k_2O\Big)\\ \frac{dN}{dt}&=r\Big(\frac{\gamma\,G^{m}}{1+G^{m}}\,\frac{1}{1+(O/O_{cN})^{h_4}+(P/P_{cN})^{h_5}}-k_3N\Big)\end{aligned}
        P, O, N        the levels of Pax6, Olig2 and Nkx2.2 (high above 1)
        r              `rate`
        G              the Shh-Gli input, `g_gain` x chem[`g_col`]
        \alpha, \beta, \gamma the three productions (alpha 0: Pax6-/-, beta 0: Olig2-/-)
        h_1 \ldots h_5 the repressions' Hill exponents
        N_{cP}, O_{cP}, N_{cO}, O_{cN}, P_{cN} the critical levels
        n, m           the input's exponents
        k_1, k_2, k_3  the decays

    Reference: Balaskas, N. et al. (2012). Cell 148:273-284; Kicheva, A. et al. (2014). Science 345:1254927 (the two-phase model)."""
    PARAM_UNITS = {'t_commit': 'time'}
    SYMBOLS = {'P, O, N': 'the levels of Pax6, Olig2 and Nkx2.2 (high above 1)', 'r': '`rate`', 'G': 'the Shh-Gli input, `g_gain` x chem[`g_col`]', '\\alpha, \\beta, \\gamma': 'the three productions (alpha 0: Pax6-/-, beta 0: Olig2-/-)', 'h_1 \\ldots h_5': "the repressions' Hill exponents", 'N_{cP}, O_{cP}, N_{cO}, O_{cN}, P_{cN}': 'the critical levels', 'n, m': "the input's exponents", 'k_1, k_2, k_3': 'the decays'}
    PARAM_ROLES = dict(CellReactBalaskasAdapt.PARAM_ROLES, t_commit="commitment_time")
    REFERENCE = ("Balaskas, N. et al. (2012). Cell 148:273-284; Kicheva, A. et al. (2014). "
                 "Science 345:1254927 (identities fixed after ~40 hph).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.t_commit = float(params.get("t_commit", 20.0))

    def forward(self, H, mask=None):
        out = super().forward(H, mask)
        ft = getattr(H, "frame_t", None)
        frame = ft if ft is not None else torch.tensor(float(getattr(H, "frame", 0)))
        live = (self.rate * frame * float(H.dt) < self.t_commit)
        return {k: v * live.to(v) for k, v in out.items()}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="balaskas_commit_schedule",
                   title="read Shh, commit, input changed midway",
                   species=(("P", "Pax6"), ("O", "Olig2"), ("N", "Nkx2.2")),
                   equation=r"""$$\begin{aligned}\frac{dP}{dt}&=r\Big(\frac{\alpha}{1+(N/N_{cP})^{h_1}+(O/O_{cP})^{h_2}}-k_1P\Big)\\ \frac{dO}{dt}&=r\Big(\frac{\beta\,G^{n}}{1+G^{n}}\,\frac{1}{1+(N/N_{cO})^{h_3}}-k_2O\Big)\\ \frac{dN}{dt}&=r\Big(\frac{\gamma\,G^{m}}{1+G^{m}}\,\frac{1}{1+(O/O_{cN})^{h_4}+(P/P_{cN})^{h_5}}-k_3N\Big)\end{aligned}$$""")
class CellReactBalaskasCommitSchedule(CellReactBalaskasCommit):
    r"""balaskas_commit (the adapting input and the commitment) with the drug of balaskas_schedule applied on top.
    cell -> cell: as balaskas_commit, G = g c f(t) s(t).

        \begin{aligned}\frac{dP}{dt}&=r\Big(\frac{\alpha}{1+(N/N_{cP})^{h_1}+(O/O_{cP})^{h_2}}-k_1P\Big)\\ \frac{dO}{dt}&=r\Big(\frac{\beta\,G^{n}}{1+G^{n}}\,\frac{1}{1+(N/N_{cO})^{h_3}}-k_2O\Big)\\ \frac{dN}{dt}&=r\Big(\frac{\gamma\,G^{m}}{1+G^{m}}\,\frac{1}{1+(O/O_{cN})^{h_4}+(P/P_{cN})^{h_5}}-k_3N\Big)\end{aligned}
        P, O, N        the levels of Pax6, Olig2 and Nkx2.2 (high above 1)
        r              `rate`
        G              the Shh-Gli input, `g_gain` x chem[`g_col`]
        \alpha, \beta, \gamma the three productions (alpha 0: Pax6-/-, beta 0: Olig2-/-)
        h_1 \ldots h_5 the repressions' Hill exponents
        N_{cP}, O_{cP}, N_{cO}, O_{cN}, P_{cN} the critical levels
        n, m           the input's exponents
        k_1, k_2, k_3  the decays

    Reference: Dessaud, E. et al. (2007). Nature 450:717-720; Balaskas, N. et al. (2012). Cell 148:273-284."""
    PARAM_UNITS = {}
    SYMBOLS = {'P, O, N': 'the levels of Pax6, Olig2 and Nkx2.2 (high above 1)', 'r': '`rate`', 'G': 'the Shh-Gli input, `g_gain` x chem[`g_col`]', '\\alpha, \\beta, \\gamma': 'the three productions (alpha 0: Pax6-/-, beta 0: Olig2-/-)', 'h_1 \\ldots h_5': "the repressions' Hill exponents", 'N_{cP}, O_{cP}, N_{cO}, O_{cN}, P_{cN}': 'the critical levels', 'n, m': "the input's exponents", 'k_1, k_2, k_3': 'the decays'}
    PARAM_ROLES = dict(CellReactBalaskasCommit.PARAM_ROLES, schedule="input_gain_schedule")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        sch = sorted([(float(t), float(f)) for t, f in (params.get("schedule") or [])])
        self.sched_t = [t for t, _ in sch]
        self.sched_f = [f for _, f in sch]

    factor = CellReactBalaskasSchedule.factor

    def profile(self, t):
        base = CellReactBalaskasAdapt.profile(self, t)
        return base * self.factor(t) if torch.is_tensor(t) else base * float(self.factor(torch.tensor(float(t))))


@register_operator("cell_grow", set="vertex", kind="lateral", family="population", title="grow where morphogen is high",
                   equation=r"""$$\frac{ds_j}{dt}=s_j\,\text{rate}\big(\rho+\mathrm{Hill}(a_j)\big),\qquad \mathrm{Hill}(a)=\frac{a^{n}}{a^{n}+a_{sw}^{n}}$$""")
class Grow3D(Lateral):
    r"""chemistry to shape: the morphogen decides where the tissue grows; each cell's mechanical targets are raised and the mechanics then inflates the cell by force balance.
    cell -> vertex: reads the cell set's chem, writes the mesh's target area A0, perimeter P0, volume V0f and radius R0 (it moves no vertex; cell_mechanics does).

        \frac{ds_j}{dt}=s_j\,\text{rate}\big(\rho+\mathrm{Hill}(a_j)\big),\qquad \mathrm{Hill}(a)=\frac{a^{n}}{a^{n}+a_{sw}^{n}}
        s_j            the cumulative per-cell growth scale, integrated by the engine
        rate           `rate`, the growth rate, per time
        \rho           `rho`, the basal growth (1 with a_sw 0: uniform)
        a_j            the cell's activator
        n              `hill`
        a_{sw}         `a_sw`, the switch level (absolute, or a fraction of the field's running maximum with `a_sw_rel`)

    Reference: Okuda, S. et al. (2018). Combining Turing and 3D vertex models reproduces autonomous multicellular morphogenesis of the tissue. Sci. Rep. 8:2386."""
    PARAM_UNITS = {'a_sw': 'dimensionless', 'cap': 'fraction', 'hill': 'dimensionless', 'rate': 'rate', 'rho': 'dimensionless', 'vth_frac': 'fraction', 'a_live': 'dimensionless', 'body_cap': 'fraction', 'inhib_hill': 'dimensionless', 'inhib_sw': 'dimensionless', 'inhib_chan': 'count'}
    SYMBOLS = {'s_j': 'the cumulative per-cell growth scale, integrated by the engine', 'rate': '`rate`, the growth rate, per time', '\\rho': '`rho`, the basal growth (1 with a_sw 0: uniform)', 'a_j': "the cell's activator", 'n': '`hill`', 'a_{sw}': "`a_sw`, the switch level (absolute, or a fraction of the field's running maximum with `a_sw_rel`)"}
    PARAM_ROLES = {'a_live': 'activator_below_which_a_cell_does_not_grow', 'a_sw': 'switch_level_of_the_activator', 'a_sw_rel': 'a_sw_as_a_fraction_of_the_running_maximum', 'body_cap': 'cap_on_the_body_targets', 'cap': 'cap_on_the_growth_scale', 'ceiling_with_divider': 'apply_the_ceiling_when_a_divider_is_scheduled', 'cell_set': 'the_cell_set', 'conserve_amount': 'conserve_the_morphogen_amount_on_growth', 'hill': 'hill_exponent', 'hold_in_m': 'hold_growth_in_m_phase', 'inhib_chan': 'chem_column_of_an_inhibitor', 'inhib_hill': 'its_hill_exponent', 'inhib_sw': 'its_switch_level', 'rate': 'growth_rate', 'rho': 'basal_growth', 'vth_frac': 'division_threshold_as_a_fraction_of_the_reference_volume', 'rate_decay_T': 'time_constant_of_the_rate_decay_zero_means_none', 'target_cap': 'cap_on_the_target_volume_as_a_multiple_of_the_reference_zero_means_none'}

    # Reads one species' activator; the span it points into is two wide because a Gray-Scott
    # system is. It never writes chem, so this only has to name the right column.
    N_SPECIES = 2
    # MAY_MUTATE_INTEGRATED_STATE IS STILL TRUE, AND GROWTH IS NO LONGER WHY. The four targets --
    # `mg_scale`, `A0`, `P0`, `V0f` -- are returned as deltas now and the engine integrates them,
    # so the dynamics this operator exists for writes nothing. Three things it does BESIDE the
    # dynamics still land in place on the cell set, and each is a different kind of write:
    #
    #   the baseline re-take   `A0_init` / `P0_init` / `V0f_init` are re-seeded whenever `nF` moves.
    #                          That is a RESET, not a rate, and it has no delta form. (The policy of
    #                          re-taking the whole population's baseline because some cell divided is
    #                          wrong and is its own parked rung -- see the note at the test below.)
    #   `conserve_amount`      c_j <- c_j * (v_old/v_new) when the cell's target volume grows: a
    #                          change of VARIABLE forced by a volume change, not a dynamics delta.
    #                          Returning it as an integrated delta would change the physics.
    #   `inhib_frac`           a readout of the inhibitor gate, recorded so the renderer can show it.
    #
    # None of the three is reached on every composition -- the rescale needs chemistry present, the
    # inhibitor needs `inhib_chan` set -- so a declaration of False would look correct on most runs
    # and fail on the ones that matter.
    # THE EXPONENT OF VOLUME IN THE LINEAR SCALE s: 3 = isotropic growth, every model here; the
    # `timer_planar` model sets 2 (see it). A class constant because it names the variant.
    GROWTH_DIMS = 3
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    # `rate` IS PER UNIT TIME. Applied once per call it would mean "fraction of itself a cell adds
    # per FRAME" and the spec's own `dt` would never enter; the engine integrates `s += dt * ds`
    # instead. Declaring `1/T` is therefore a statement the code
    # honours, and it is the one this whole vocabulary exists to be able to make.
    #
    # `a_sw` IS DELIBERATELY ABSENT. It is an absolute activator concentration when `a_sw_rel` is
    # false and a fraction of the field's own running maximum when it is true -- one parameter with
    # two dimensions, selected by another parameter. Either declaration would be wrong half the
    # time, and UNDECLARED is silent, which is the honest state until the two are separated.
    # `V0f` IS UNDECLARED FOR THE SAME REASON `cell_die`'s IS: growth SCALES the seed's target and
    # never chooses its convention. `vth_frac` is a multiple of whichever `v_ref` matches it, which
    # `cell_size` now supplies. The pair the checker should compare is the SEED against the ENERGY,
    # and those two both declare.
    BLOCK_UNITS = {"A0": "area[midsurface]", "P0": "length"}
    MECHANISM_TAGS = ["growth", "morphogen_driven", "budding", "cross_scale"]
    REFERENCE = "Okuda, S. et al. (2018). Combining Turing and 3D vertex models reproduces autonomous multicellular morphogenesis of the tissue. Sci. Rep. 8:2386."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex"); self._cat = params.get("cell_set")
        self.rate = float(params.get("rate", 0.01)); self.a_sw = float(params.get("a_sw", 0.20))
        # `rate_decay_T` (exp 11, 2026-09-27; default 0 = none, unchanged): the growth rate falls as rate / (1 + t / T),
        # t the simulated time. Exponential growth -- the law below at a fixed rate -- multiplied a gland's area x20 in
        # 53 h and ran it into the world's walls; Wang et al. 2021 Fig 1H measured the surface outline growing LINEARLY in
        # time in all 13 glands (r^2 0.80-0.99, 10-34 h), i.e. area A0 (1 + rho t)^2 with a relative rate 2 rho / (1 + rho t):
        # T = 1 / rho. Every model of cell_grow that calls `_law` or reads `self._decay` inherits it.
        self.rate_decay_T = float(params.get("rate_decay_T", 0.0))
        self._decay = 1.0
        self.hold_in_m = bool(params.get("hold_in_m", False))
        self.ceiling_with_divider = bool(params.get("ceiling_with_divider", False))
        # WHICH SPECIES GATES GROWTH: 0 (chem columns 0,1) by default, so existing specs are
        # unchanged; 2 reads a second RD system living in the same buffer.
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)
        # IS `a_sw` A VALUE OF THE ACTIVATOR OR A FRACTION OF ITS MAXIMUM? Default False = the
        # absolute reading every run in this project's history used, so no archived spec changes
        # meaning. See the gate itself in `forward` for why both are real mechanisms. `a_live` is
        # the floor below which a field counts as dead and the relative gate refuses to open --
        # 1e-3 against activators that reach 0.6-1.5 when alive and 1e-9 when they have collapsed.
        self.a_sw_rel = bool(params.get("a_sw_rel", False))
        self.a_live = float(params.get("a_live", 1e-3))
        # THE INHIBITOR: None = off, so every existing spec is unchanged. `inhib_sw` is a fraction
        # of the inhibitor's OWN maximum and `inhib_hill` its sharpness, mirroring a_sw/hill.
        _ic = params.get("inhib_chan", None)
        self.inhib_chan = None if _ic is None else int(_ic)
        self.inhib_sw = float(params.get("inhib_sw", 0.35))
        self.inhib_hill = float(params.get("inhib_hill", 4.0))
        self._inhib_applied = 0.0
        self.hill = float(params.get("hill", 3.0)); self.cap = float(params.get("cap", 2.5))
        from plexus.operators.vertex_ops import _engine_owns_clock
        self.every = _engine_owns_clock(params); self._k = 0
        # OKUDA uniform-cell mode: growth rate lambda = rate*(rho + Hill(a)); rho = baseline so ALL cells
        # cycle (the activator sets the RATE, not the size), and v_eq is capped at vth_frac*v_ref so every
        # cell oscillates in [~2/3, vth_frac]*v_ref -> uniform. rho=0 (default) = legacy activator-only bulge.
        self.rho = float(params.get("rho", 0.0)); self.vth_frac = float(params.get("vth_frac", 1.35))
        # OKUDA Appendix A: the morphogen is the AMOUNT m_j (conserved within the cell); the concentration
        # c_j=m_j/v_j is only READ by the kinetics. We store c_j, so growing v_j must DILUTE c_j to conserve
        # amount (else we silently CREATE mass each step -> spuriously feeds the tip). On (default) = correct.
        self.conserve_amount = bool(params.get("conserve_amount", True))
        self._said_no_cap = False       # the "ceiling not applied" note is printed once per run
        # `target_cap: c` (exp 11, 2026-09-28; default 0 = off): the TARGET `V0f` stops growing at c x the seed's median
        # target (`V0f_init` -- target against target, one convention), whether or not a `cell_divide` is scheduled.
        # The growth writes the target while `cell_cycle`'s sizer reads the MEASURED volume, so a squeezed cell that
        # never reached the checkpoint kept growing its target: 218 of the healthy gland's 1,006 cells above 2x the
        # median target at row 160, the largest 8.4x (exp 11 Finding 119). With `cell_cycle` `size_from: target` the
        # cap sits just above the checkpoint (`g1_size`) and a cell that cannot divide waits there.
        self.target_cap = float(params.get("target_cap", 0.0) or 0.0)
        # `body_cap: b` (exp 11, 2026-09-28; default 0 = off): the target may run at most b x ahead of the cell's own
        # MEASURED volume -- V0f <= b x k x v, k the seed's median V0f / median measured volume (the two conventions'
        # ratio, taken once on the first call). A squeezed cell stops growing its target instead of banking it (exp 11
        # Finding 119: targets up to 12.7x the seed median while the body sat at a tenth), its push on its neighbours
        # is bounded, and a cell with room is not held under a fixed ceiling (Finding 125: `target_cap` halved growth).
        self.body_cap = float(params.get("body_cap", 0.0) or 0.0)

    def _rate(self, s_prev, hillv, m, v_ref):
        """The RATE LAW, and the only thing a `model=` variant of cell_grow changes.

        Returns ds/dt: the per-UNIT-TIME rate of change of the per-cell linear growth scale `s`,
        which the engine integrates as `s <- s + dt * ds`. Volume is V0f_init * s**3, so a constant
        proportional rate on `s` is exponential growth in volume -- which is the default and is
        deliberate: it is what Okuda's growth term does. Ginzberg, Kafri & Kirschner (Science 2015)
        name the consequence exactly: "with exponential growth, larger cells grow faster than do
        smaller cells, amplifying any existing size disparities". Measured on this campaign's own
        basis, the coefficient of variation of cell volume climbs 0.160 at seed to 0.33-0.53 by
        frame 900 in every run. The variants below are the mechanisms that review says real cells
        use to stop that, written so the search can put them side by side.

        A RATE, NOT A PER-FRAME FACTOR. The law `s <- s (1 + rate (rho + Hill(a)))` applied once
        per CALL would make `rate` mean "fraction of itself a cell adds per frame", with the
        simulation's own `dt` nowhere in it. That is the same number only at `dt = 1`: every
        `config/tissue` spec is `dt: 1.0` and reads identically, while a spec at `dt: 0.0032` would
        run its growth 312x faster per
        unit of simulated time than the same `rate` now means, so those specs carry `rate / dt` to
        preserve what they did.
        """
        return s_prev * self.rate * self._decay * (self.rho + hillv)

    def forward(self, H, mask=None):
        if self.rate_decay_T > 0:
            # the clock of the RECORDED run: the engine's unrecorded warm-up (`H.warmup` ticks) is not growth time
            _t = max(float(getattr(H, "frame", 0) or 0) - float(getattr(H, "warmup", 0) or 0), 0.0) \
                * float(getattr(H, "dt", 1.0) or 1.0)
            self._decay = 1.0 / (1.0 + _t / self.rate_decay_T)
        # THE PAIRING IS READ FROM THE SET, ONCE PER CALL -- see `resolve_cell_set`.
        self.cat = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
        vlvl = H.level(self.at); m = getattr(vlvl, "_mesh", None)
        if m is None:
            return {}
        self._k += 1                    # monotonic tick only -- D1: the engine owns the period
        clvl = H.level(self.cat)
        nF = m["nF"]
        dev = m["V0f"].device
        # NO CHEMISTRY IS UNIFORM GROWTH, NOT NO GROWTH. This used to `return {}` when the cell set
        # carried no `chem` state, which made a growth operator silently inert on any composition
        # without an RD pair -- and hid a dependency that is not real: the activator is a GATE, and
        # with the gate open (a_sw = 0, rho = 1) the rate does not consult it at all. That guard is
        # also why `cell_grow` had to exist as a separate operator. a = 0 is the honest reading
        # of "there is no activator here"; the Hill term evaluates to 0 and the rho baseline stands.
        if "chem" in clvl.state_schema:
            # `chan` PICKS THE SPECIES THAT GATES GROWTH. With two RD systems in one buffer this is
            # what lets them do different jobs -- species 0 driving growth while species 1 drives
            # death -- instead of both operators reading the same activator, which would be one
            # mechanism wearing two names.
            h0, _h1 = clvl.state_schema["chem"]
            a = clvl.state[:nF, h0 + self.chan].detach().to(dev)  # per-cell activator
        else:
            a = torch.zeros(nF, device=dev, dtype=m["V0f"].dtype)
        # THE TEST WAS DOING TWO JOBS AND THE MOVE SEPARATES THEM. `"mg_scale" not in m or
        # m["mg_scale"].shape[0] != nF` meant BOTH "this operator has not run yet" AND "the cell
        # count changed since it last did" -- and on the cell set the name is always present with
        # length exactly `nF`, so the test can never fire, `s` stays 0 and every target collapses.
        #
        # `mg_scale_nF` IS THE SECOND MEANING, WRITTEN DOWN. It is a scalar on the mesh table, like
        # `n_div` and `div_blocked`, because it is one number per run and not one per cell. The
        # behaviour is exactly what it was: re-baseline whenever the face count moves, which with
        # `cell_divide` every four frames is most frames.
        #
        # THAT POLICY IS WRONG AND IS NOT CHANGED HERE. Re-taking the whole population's baseline
        # because SOME cell divided is why `mg_scale` stops meaning "how much this cell has grown"
        # -- measured, it climbs 1.0035 -> 1.0459 before the first division and then never exceeds
        # 1.0139 -- and `contact_ops.ecm_gate_growth` read it as cumulative and spent 400 frames
        # correcting nothing. Fixing it moves numbers, so it is its own rung with its own evidence;
        # this one only moves the storage, and is byte-identical because of that.
        if int(m.get("mg_scale_nF", -1)) != int(nF):
            m["mg_scale_nF"] = int(nF)
            m["mg_scale"] = torch.ones(nF, device=dev, dtype=m["V0f"].dtype)
            m["A0_init"] = m["A0"].clone(); m["P0_init"] = m["P0"].clone(); m["V0f_init"] = m["V0f"].clone()
        # THE GATE'S HALF-POINT, AND WHAT IT IS A FRACTION OF.
        #
        # `a_sw` is ABSOLUTE by default: the Hill half-point sits at a fixed value of the activator
        # and does not move when the field does. Every other threshold on a chemistry field in this
        # substrate is relative to that field's own maximum -- `interface_tension.a_sw`,
        # `interface_push.a_sw`, `cell_die` chem_low, `cell_divide.orient_asw` -- and two
        # comments in this project (crew/basis.yaml on `a_sw_gated`, and the inhibitor branch
        # twenty lines below) already describe THIS one as a fraction of the maximum. They were
        # describing an intention, not the code.
        #
        # Both are defensible and they are not the same mechanism, so this is a switch and not a
        # correction:
        #   absolute  the gate stops opening when the chemistry dies. A field that decays to 1e-9
        #             drives no growth, and the tissue goes static -- which is what the 16 runs
        #             with act_max < a_sw did, honestly.
        #   relative  the gate always selects the same TOP FRACTION of cells, so `a_sw` means the
        #             same thing in a run peaking at 0.6 and one peaking at 1.5. That is what a
        #             sweepable lever has to do; measured over 154 runs the absolute gate sat
        #             anywhere from 0.24 of the field to above all of it.
        # The floor is what keeps `relative` honest: without it, a field decayed to noise still has
        # cells "above 35% of the maximum" and the tissue would grow on numerical dust forever.
        thr = self.a_sw
        if self.a_sw_rel:
            amax = float(a.max()) if a.numel() else 0.0
            thr = self.a_sw * amax if amax > self.a_live else float("inf")   # inf -> Hill term 0
        hillv = a ** self.hill / (thr ** self.hill + a ** self.hill + 1e-12)   # Hill activation in [0,1]
        # A SECOND MORPHOGEN THAT STOPS GROWTH, so that only the activator-high spots grow.
        #
        # The growth law above is purely ACTIVATING: `rate * (rho + hill(a))`, so
        # the only thing a morphogen can do is make a cell grow FASTER, and the slowest a cell can
        # grow is the rho baseline -- which is why six rounds produced broad lobes and never a
        # finger. A bulge sharpens into a finger when the tissue grows at the tip AND STOPS at the
        # flanks, and no single activating field can say "stop": it has no zero to reach.
        #
        # `inhib_chan` names a species whose HIGH values switch growth off, multiplicatively:
        #
        #     growth  <-  rate * (rho + hill(a_act)) * (1 - hill(a_inhib))
        #
        # so where the inhibitor is saturated the cell does not grow at all, baseline included.
        # This is lateral inhibition, and it is a different mechanism from a sharper gate: `hill`
        # narrows the shoulder of the activating curve, this puts a floor of ZERO under it.
        if self.inhib_chan is not None and "chem" in clvl.state_schema:
            _h0, _ = clvl.state_schema["chem"]
            b = clvl.state[:nF, _h0 + self.inhib_chan].detach().to(dev).clamp(min=0.0)
            # RELATIVE TO THE INHIBITOR'S OWN MAXIMUM, for the same reason a_sw is: an absolute
            # threshold against a field whose scale the chemistry sets is either always on or
            # always off, and this project has paid for that twice (rd_interface_tension, chem_low).
            bmax = float(b.max()) if b.numel() else 0.0
            if bmax > 1e-9:
                bn = b / bmax
                inh = bn ** self.inhib_hill / (self.inhib_sw ** self.inhib_hill
                                               + bn ** self.inhib_hill + 1e-12)
                m["inhib_frac"] = inh                     # recorded, so the renderer can show it
                hillv = hillv * (1.0 - inh)
                self._inhib_applied = float(inh.mean())
        s_prev = m["mg_scale"]                                    # per-cell scale BEFORE this tick (for the dilution rate)
        # THE REFERENCE IN THE CONVENTION THE CELL IS ACTUALLY MEASURED IN -- polyhedron where the
        # run carries a separation, wedge where it does not. See `vertex_ops.cell_size`. `vth_frac`
        # is a multiple of this; measured in the wedge convention on an apicobasal run, the growth
        # ceiling would be stated in a volume the cell does not have.
        # ONE READER: the cell's ACTUAL volume and the reference, both from `cell_size`, so the
        # size-dependent growth rule compares what the cell has with what a typical cell had --
        # it used to compare the TARGET `V0f` with the reference, two different quantities.
        from plexus.operators.vertex_ops import cell_size
        _v_now, v_ref = cell_size(vlvl, m, nF, H=H, at=self.at)
        self._v_now = torch.as_tensor(np.asarray(_v_now, np.float64), device=dev,
                                      dtype=m["V0f"].dtype)
        dt = float(getattr(H, "dt", 1.0))
        ds = self._rate(s_prev, hillv, m, v_ref)                  # <-- the rate law; models override THIS only
        # `hold_in_m: true` (default false = unchanged): a cell in M (`phase` >= 3 on the cell set) does not grow.
        # exp 11's Type I loop: a division-ready cell WAITS in M for its turn to leave the layer (the dive cap), and
        # growing there it inflated its targets 5x and stretched the layer 4x over them (Finding 102). A mitotic cell
        # stops growing (Mitchison 1971); every cell that is not waiting is untouched.
        if self.hold_in_m:
            from plexus.operators.vertex_ops import cell_block
            _ph = cell_block(H, self.cat, "phase", nF) if H is not None else None
            if _ph is not None and len(_ph) == nF:
                ds = ds * torch.as_tensor((np.asarray(_ph, np.float64) < 3.0).astype(np.float64),
                                          device=ds.device, dtype=ds.dtype)
        # THE WARM-UP HOLDS GROWTH (2026-09-28). While the engine's unrecorded warm-up runs (H.frame < H.warmup, the
        # seed's settle window) the cell cycle and division idle; growing the targets there handed every cell about
        # twice its age's size at frame 0 -- the large, young cells and the first division wave of exp 11 (Finding 130).
        # `general.warmup: 0` turns the warm-up, and with it this hold, off.
        _wu = int(getattr(H, "warmup", 0) or 0)
        if _wu > 0 and int(getattr(H, "frame", 0) or 0) < _wu:
            ds = ds * 0.0
        # THE CEILING IS FOR A TISSUE WITH NO DIVIDER, AND ONLY FOR ONE.
        #
        # `vth_frac` is Okuda's uniform-cell mode: cap `v_eq` under `vth_frac * v_ref` so every cell
        # oscillates in a band and the population stays uniform WITHOUT anything resetting it. Once
        # `cell_divide` is in the schedule, division is what resets size -- a cell doubles, splits,
        # and each daughter starts at half -- so the ceiling is no longer size control. It is a lid,
        # and if it sits below the division threshold the tissue can never reach that threshold.
        #
        # IT DID. `divide_growing_ball` caps at `vth_frac 2.5 * v_ref 2.5433` = 6.36 in WEDGE units,
        # which plateaus the polyhedron volume -- the one `cell_divide`'s trigger reads -- at 1.987
        # against a reference of 1.229. That is 1.62x, and `factor: 2.0` needs 2x. Not one division
        # fired in 401 frames and the cell count sat flat at 200. The two conventions are the deeper
        # problem (AB_R7R8_TODO section 0a) but they are not what makes this spec dead: two
        # mechanisms were doing one job.
        #
        # A CEILING ON A RATE IS A GATE, NOT A CLAMP. When the operator wrote `s` itself the lid was
        # `s <- min(s, s_cap)`; an operator that only emits ds/dt cannot clamp a value it does not
        # own, so the lid is applied to the RATE -- a cell at or above its ceiling grows at zero.
        # Same fixed point (`s` settles on `s_cap`), and it no longer overshoots-then-truncates
        # within a step, so the band is entered rather than snapped to.
        _has_divider = "cell_divide" in getattr(H, "scheduled_ops", frozenset())
        # `ceiling_with_divider: true` (default false = unchanged): keep the ceiling although a `cell_divide` is
        # scheduled -- for a layer whose `cell_divide`s do NOT reset its cells' size in place (exp 11's Type I loop:
        # `reinsert` and `cell_divide[model: mpm]`; every division of a surface cell leaves the layer), where the
        # uncapped targets grew x5 and blew the shell up past its own membrane (Finding 103).
        if self.ceiling_with_divider:
            _has_divider = False
        if self.rho > 0 and not _has_divider:                        # OKUDA uniform-cell mode
            s_cap = ((self.vth_frac * v_ref
                      / m["V0f_init"].clamp(min=1e-9)) ** (1.0 / 3.0)).clamp(min=1.0)
            ds = torch.where(s_prev + dt * ds > s_cap, (s_cap - s_prev) / max(dt, 1e-12), ds)
        elif self.rho > 0:
            if not self._said_no_cap:
                self._said_no_cap = True
                print(f"[cell_grow] `cell_divide` is scheduled, so the `vth_frac` ceiling is not "
                      f"applied: division is what resets cell size here, and a ceiling below the "
                      f"division threshold would stop the tissue reaching it.", flush=True)
        else:
            # legacy: activator-only bulge, gated to a stop at `cap` for the same reason as above
            cap = torch.as_tensor(self.cap, device=ds.device, dtype=ds.dtype)
            ds = torch.where(s_prev + dt * ds > cap, (cap - s_prev) / max(dt, 1e-12), ds)
        # THE FOUR TARGETS ARE ONE INTEGRAND SEEN FOUR WAYS, and the chain rule is what keeps them
        # one. `s` is what grows; A0 = A0_init s^2, P0 = P0_init s, V0f = V0f_init s^3 are algebraic
        # functions of it, so their rates are its rate times the derivative of each map:
        #
        #     dA0/dt = 2 A0_init s (ds/dt)      dP0/dt = P0_init (ds/dt)      dV0f/dt = 3 V0f_init s^2 (ds/dt)
        #
        # WHAT THIS COSTS, said plainly, because it is a real difference and not a reassociation.
        # Writing `A0 = A0_init s_new^2` was exact in `s`; integrating its derivative is exact only
        # to first order, and leaves A0 short of A0_init s^2 by A0_init (dt ds)^2 per step. On this
        # campaign's own growth rate (5.78e-04 per unit time) that is 3.3e-07 of A0 per frame. The
        # alternative -- emitting the exact chord (s_new^2 - s^2)/dt -- would reproduce today's
        # numbers bit for bit but requires the operator to integrate `s` itself, which is the engine's
        # job and the thing this rung exists to give back to it.
        if self.body_cap > 0:
            _n = ds.shape[0]
            _vb = self._v_now[:_n].to(ds.dtype)
            if getattr(self, "_k_body", None) is None:
                self._k_body = float(torch.median(m["V0f"][:_n])) / max(float(torch.median(_vb)), 1e-12)
            ds = cap_target_rate(ds, s_prev, m["V0f"][:_n], m["V0f_init"][:_n],
                                 self.body_cap * self._k_body * _vb.clamp(min=0.0), dt, self.GROWTH_DIMS)
        if self.target_cap > 0:
            # THE REFERENCE IS TAKEN ONCE, on the first call (the seed's targets). `V0f_init` cannot serve: this
            # operator re-baselines it to the current `V0f` whenever the face count moves, so a cap on its median
            # floated with the population and held nothing (exp 11 mc_e_tc_ctl, first build).
            if getattr(self, "_v0_seed", None) is None:
                self._v0_seed = float(torch.median(m["V0f"][:ds.shape[0]]))
            _v0 = m["V0f"][:ds.shape[0]]
            _cap = self.target_cap * self._v0_seed
            ds = cap_target_rate(ds, s_prev, _v0, m["V0f_init"][:ds.shape[0]], _cap, dt, self.GROWTH_DIMS)
        s_next = s_prev + dt * ds                                # what `s` will be once the engine integrates
        deltas = {(self.cat, "mg_scale"): ds,
                  (self.cat, "A0"): 2.0 * m["A0_init"] * s_prev * ds,
                  (self.cat, "P0"): m["P0_init"] * ds,
                  (self.cat, "V0f"): (3.0 * m["V0f_init"] * s_prev * s_prev * ds if self.GROWTH_DIMS == 3
                                      else 2.0 * m["V0f_init"] * s_prev * ds)}   # the isotropic form verbatim: bit-identical
        s = s_next                                               # the scalars below summarise the POST-tick tissue
        m["V0"] = float((m["V0f_init"] * s ** self.GROWTH_DIMS).sum())
        # THE SHELL RADIUS MUST GROW WITH THE CELLS. cell_mechanics carries a radial spring,
        #     E += K_R * sum_i (|x_i| - R0)^2
        # and R0 is set once at seeding. An operator that grows the cells without rescaling R0
        # leaves the mechanics pinning the shell at the SEED radius while the
        # cells' target volumes grew sixteenfold, and the sheet had nowhere to put the extra area
        # but through itself. Measured on mini_grow_divide_bigger: rays cast from the tissue
        # centroid cross the surface exactly once at frame 384 (100% of them) and 13 times at
        # frame 423. The genus check reported "sphere (as built)" at every one of those frames --
        # Euler characteristic is combinatorial and cannot see a shell folded through its own
        # centre, which is why this survived until premise 11 was written.
        #
        # The radius of the sphere enclosing the current target volume, NOT the measured mean
        # radius: R0 must express what the cells are ASKING for. Setting it from |x| would make
        # the spring chase the shell's own excursions and quietly penalise a growing bud, which is
        # the one shape this campaign exists to produce.
        if "R0" in m:
            m["R0"] = float((3.0 * max(m["V0"], 1e-12) / (4.0 * np.pi)) ** (1.0 / 3.0))
        if self.conserve_amount and "chem" in clvl.state_schema:
            # conserve molecule AMOUNT: c_j <- c_j * (v_old/v_new) = c_j * (s_prev/s)^3 as v_eq grows ~ s^3.
            # Makes dilution STRUCTURAL (no continuum -c(div.v) term); it is LOAD-BEARING (Okuda's intra-domain
            # gradients come from it) -> keep it, don't cancel. The flood is a gamma (rate) problem, fixed elsewhere.
            g_vol = (s / s_prev.clamp(min=1e-9)) ** self.GROWTH_DIMS
            cst = clvl.state.clone()
            # THE ACTIVATOR COLUMN ONLY. Diluting BOTH columns extinguishes Gray-Scott outright:
            # measured, 1% loss per step kills the pattern within 250 steps, while the undiluted
            # one reaches 53% coverage by step 250 and holds indefinitely. Diluting either column
            # alone survives (a_max at t=60: 0.704 / 0.686); both together does not (0.047),
            # because the substrate's own feed term F(1-u) is what pulls u back up and diluting u
            # fights it directly. The activator has no source at all except its own autocatalysis,
            # which is QUADRATIC, so it is the one that genuinely loses material when a cell grows.
            # Correct physics on a fragile mechanism is still a broken model; this keeps the
            # physics where it belongs and stops it destroying the pattern it is meant to shape.
            cst[:nF, h0:h0 + 1] = cst[:nF, h0:h0 + 1] / g_vol.clamp(min=1e-9)[:, None]
            clvl.state = cst
        # THE LIVE CELLS ARE A PREFIX OF THE BUFFER, and a block delta has to be the whole buffer:
        # `_integrate` adds it to `lvl.state[:, cx0:cx1]` with no mask, so a short tensor would not
        # broadcast and a full one must be zero past `nF` or the dead tail of the buffer would grow.
        N = int(clvl.state.shape[0])
        out = {}
        for key, d in deltas.items():
            z = torch.zeros(N, 1, device=clvl.state.device, dtype=clvl.state.dtype)
            z[:nF, 0] = d.to(device=clvl.state.device, dtype=clvl.state.dtype)
            out[key] = z
        return out


# =========================================================== cell_grow: the size-control models
# Three mechanisms from Ginzberg, Kafri & Kirschner, "On being the right (cell) size", Science 348
# (2015) and Ginzberg et al., eLife 7:e26957 (2018). They are `model=` variants, not replacements,
# because each is a different biological hypothesis at the same slot and the search should put
# them side by side. The default `cell_grow` above has NO size control at all -- its rate reads the
# morphogen and never the cell's own size -- which by the review's Eq. 2 means cell-size variance
# can only ever increase, and in this campaign's basis it does: vol_cv 0.160 -> 0.53.
#
# WHY THIS MATTERS BEYOND TIDINESS. The review's Fig 2 sets a healthy mammary epithelium, uniform
# in cell size, beside a pleomorphic tumour that is not, and states that "pleomorphism ... is a
# histological characteristic of many malignant lesions". This project is trying to grow an
# epithelial TUBE -- a coherent structure -- and its tissue drifts toward the second picture.


@register_operator("cell_grow", model="planar", set="vertex", kind="lateral", family="population",
                   title="grow in area at constant height",
                   equation=r"""$$\frac{ds_j}{dt}=s_j\,\text{rate}\big(\rho+\mathrm{Hill}(a_j)\big),\qquad \mathrm{Hill}(a)=\frac{a^{n}}{a^{n}+a_{sw}^{n}}$$""")
class Grow3DPlanar(Grow3D):
    r"""cell_grow for a monolayer of fixed density: the target volume scales as the target area, V0f ~ A0 ~ s^2 (constant height), where the default scales it as s^3.
    cell -> vertex: as cell_grow.

        \frac{ds_j}{dt}=s_j\,\text{rate}\big(\rho+\mathrm{Hill}(a_j)\big),\qquad \mathrm{Hill}(a)=\frac{a^{n}}{a^{n}+a_{sw}^{n}}
        s_j            the cumulative per-cell growth scale, integrated by the engine
        rate           `rate`, the growth rate, per time
        \rho           `rho`, the basal growth (1 with a_sw 0: uniform)
        a_j            the cell's activator
        n              `hill`
        a_{sw}         `a_sw`, the switch level (absolute, or a fraction of the field's running maximum with `a_sw_rel`)

    Reference: as cell_grow; Plexus exp33 (2026-10-25)."""
    PARAM_UNITS = {}
    SYMBOLS = {'s_j': 'the cumulative per-cell growth scale, integrated by the engine', 'rate': '`rate`, the growth rate, per time', '\\rho': '`rho`, the basal growth (1 with a_sw 0: uniform)', 'a_j': "the cell's activator", 'n': '`hill`', 'a_{sw}': "`a_sw`, the switch level (absolute, or a fraction of the field's running maximum with `a_sw_rel`)"}
    GROWTH_DIMS = 2

    def forward(self, H, mask=None):
        # THE SHELL RADIUS FOLLOWS THE AREA. The default resets the radial spring's rest radius to the radius of the SOLID sphere holding the
        # total target volume, which for a thin sheet is far inside the shell (seed R 4.5, solid sphere 3.0 for 100 cells of height 0.45): at the
        # first growth tick the sheet collapsed by 25 % in radius and 43 % in cell area, in every run, whatever the onset frame (exp33 audit,
        # 2026-10-25). A monolayer of fixed height grows its surface, so R0 = R0_seed sqrt(total volume target / total at seeding).
        m = getattr(H.level(self.at), "_mesh", None)
        if m is not None and "R0" in m and not hasattr(self, "_seed_r0"):
            self._seed_r0, self._seed_v0 = float(m["R0"]), float(m["V0f"].sum())      # V0f_init is still empty on the first call
        out = super().forward(H, mask)
        if m is not None and "R0" in m and "V0" in m and hasattr(self, "_seed_r0") and self._seed_v0 > 0:
            m["R0"] = self._seed_r0 * float(np.sqrt(max(m["V0"], 1e-12) / self._seed_v0))
        return out


@register_operator("cell_grow", model="sizer", set="vertex", kind="lateral", family="population", title="grow slower as cells get bigger", equation=r"""$$\frac{ds_j}{dt}=s_j\,\text{rate}\,\big(\rho+\mathrm{Hill}(a_j)\big)\,f_j,\qquad f_j=\mathrm{clip}\Big(\big(\tfrac{v_{ref}}{v_j}\big)^{g},\ \tfrac{1}{f_{max}},\ f_{max}\Big)$$""")
class Grow3DSizer(Grow3D):
    r"""the growth rate falls with the cell's own size: small cells grow faster, large ones slower (a sizer); size_gain 0 recovers the default exactly.
    cell -> vertex: as cell_grow, the rate multiplied by f.

        \frac{ds_j}{dt}=s_j\,\text{rate}\,\big(\rho+\mathrm{Hill}(a_j)\big)\,f_j,\qquad f_j=\mathrm{clip}\Big(\big(\tfrac{v_{ref}}{v_j}\big)^{g},\ \tfrac{1}{f_{max}},\ f_{max}\Big)
        s_j, rate, \rho, a_j as cell_grow
        f_j            the size factor
        v_{ref}, v_j   the reference and the cell's volume
        g              `size_gain`
        f_{max}        `f_max`, the clip

    Reference: Ginzberg, M. B., Kafri, R. & Kirschner, M. W. (2015). On being the right (cell) size. Science 348:1245075, Fig. 3B."""
    PARAM_UNITS = {'f_max': 'dimensionless', 'size_gain': 'dimensionless'}
    SYMBOLS = {'s_j, rate, \\rho, a_j': 'as cell_grow', 'f_j': 'the size factor', 'v_{ref}, v_j': "the reference and the cell's volume", 'g': '`size_gain`', 'f_{max}': '`f_max`, the clip'}
    PARAM_ROLES = {'f_max': 'clip_on_the_size_factor', 'size_gain': 'exponent_of_the_size_feedback'}
    MECHANISM_TAGS = ["growth", "size_control", "sizer", "negative_feedback"]
    REFERENCE = ("Ginzberg, M.B., Kafri, R. & Kirschner, M.W. (2015). On being the right (cell) "
                 "size. Science 348:1245075; Ginzberg et al. (2018) eLife 7:e26957.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.size_gain = float(params.get("size_gain", 1.0))
        self.f_max = float(params.get("f_max", 4.0))

    def _rate(self, s_prev, hillv, m, v_ref):
        v_now = self._v_now.clamp(min=1e-9)                     # actual volume, from `cell_size`
        f = (v_ref / v_now) ** self.size_gain
        f = torch.clamp(f, 1.0 / max(self.f_max, 1e-9), self.f_max)
        return s_prev * self.rate * (self.rho + hillv) * f


@register_operator("cell_grow", model="balance", set="vertex", kind="lateral", family="population", title="grow by synthesis-degradation balance", equation=r"""$$\frac{dV_j}{dt}=k_{syn}\big(\rho+\mathrm{Hill}(a_j)\big)-k_{deg}V_j$$""")
class Grow3DBalance(Grow3D):
    r"""size emerges from a synthesis / degradation balance with no size sensor anywhere: the morphogen sets the target size, not the rate.
    cell -> vertex: as cell_grow, on the target volume.

        \frac{dV_j}{dt}=k_{syn}\big(\rho+\mathrm{Hill}(a_j)\big)-k_{deg}V_j
        V_j            the cell's target volume
        k_{syn}        `k_syn`, the synthesis rate
        k_{deg}        `k_deg`, the degradation rate (steady state k_syn (rho + Hill) / k_deg)
        \rho, a_j      as cell_grow

    Reference: Ginzberg, M. B., Kafri, R. & Kirschner, M. W. (2015). On being the right (cell) size. Science 348:1245075."""
    PARAM_UNITS = {'k_deg': 'rate', 'k_syn': 'rate'}
    SYMBOLS = {'V_j': "the cell's target volume", 'k_{syn}': '`k_syn`, the synthesis rate', 'k_{deg}': '`k_deg`, the degradation rate (steady state k_syn (rho + Hill) / k_deg)', '\\rho, a_j': 'as cell_grow'}
    PARAM_ROLES = {'k_deg': 'degradation_rate', 'k_syn': 'synthesis_rate'}
    MECHANISM_TAGS = ["growth", "size_control", "synthesis_degradation_balance", "homeostasis"]
    REFERENCE = ("Ginzberg, M.B., Kafri, R. & Kirschner, M.W. (2015). On being the right (cell) "
                 "size. Science 348:1245075.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        # k_syn is expressed as a multiple of v_ref per unit time, so the balance point lands near
        # v_ref at k_deg = rate and the parameter has the same meaning at any mesh scale.
        self.k_syn = self.tunable(params.get("k_syn"), 1.0, tensor=True)
        self.k_deg = self.tunable(params.get("k_deg"), 1.0, tensor=True)

    def _rate(self, s_prev, hillv, m, v_ref):
        # `dv` IS ALREADY dV/dt -- the docstring's own equation is written that way -- so the only
        # conversion needed is from a volume rate to a rate on the linear scale. With V = V0f_init s^3,
        # dV/dt = 3 V0f_init s^2 (ds/dt), hence ds/dt = (dV/dt) * s / (3 V).
        v_now = (m["V0f_init"] * s_prev ** 3).clamp(min=1e-9)
        dv = self.rate * (self.k_syn * v_ref * (self.rho + hillv) - self.k_deg * v_now)
        return dv * s_prev / (3.0 * v_now)


@register_operator("cell_grow", model="timer", set="vertex", kind="lateral", family="population", title="grow to a target size",
                   equation=r"""$$\frac{d\ln V_j}{dt}=\frac{1}{T}\ln\frac{V_{\mathrm{target}}}{V_j}$$""")
class Grow3DTimer(Grow3D):
    r"""grows at whatever rate lands the cell on its target size after `cycle_frames` frames, the partner of cell_divide[timer]: a proportional controller on log volume.
    cell -> vertex: as cell_grow.

        \frac{d\ln V_j}{dt}=\frac{1}{T}\ln\frac{V_{\mathrm{target}}}{V_j}
        V_j            the cell's target volume
        T              `cycle_frames`, the time constant
        V_{target}     the target size, scaled by (rho + Hill(a)) / (rho + 1)

    Reference: Ginzberg, M. B., Kafri, R. & Kirschner, M. W. (2015). On being the right (cell) size. Science 348:1245075, Fig. 3B."""
    PARAM_UNITS = {'cycle_frames': 'frames'}
    SYMBOLS = {'V_j': "the cell's target volume", 'T': '`cycle_frames`, the time constant', 'V_{target}': 'the target size, scaled by (rho + Hill(a)) / (rho + 1)'}
    PARAM_ROLES = {'cycle_frames': 'frames_to_reach_the_target'}
    MECHANISM_TAGS = ["growth", "size_control", "timer", "target_size", "proportional_control"]
    REFERENCE = ("Ginzberg, M.B., Kafri, R. & Kirschner, M.W. (2015). On being the right (cell) "
                 "size. Science 348:1245075.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.cycle_frames = float(params.get("cycle_frames", 100.0))

    def _rate(self, s_prev, hillv, m, v_ref):
        # THE PROPORTIONAL CONTROLLER, WRITTEN AS THE RATE IT ALWAYS WAS. The per-frame factor
        # (v_tgt/v_now)^(1/cycle_frames) is exp(ln(v_tgt/v_now)/cycle_frames), so the underlying law is
        # d(ln V)/dt = ln(v_tgt/v_now) / cycle_frames -- first-order relaxation of log-volume with
        # time constant `cycle_frames`. On the linear scale that is a third of it, since s = V^(1/3).
        # `cycle_frames` is now a DURATION IN SIMULATION TIME rather than a count of calls; at the
        # `dt: 1.0` every spec using this model runs at, the two are the same number.
        v_now = (m["V0f_init"] * s_prev ** self.GROWTH_DIMS).clamp(min=1e-9)
        share = (self.rho + hillv) / max(self.rho + 1.0, 1e-9)   # the morphogen sets WHERE, as a target
        v_tgt = (self.vth_frac * v_ref * share).clamp(min=1e-9)
        return s_prev * torch.log(v_tgt / v_now) / (self.GROWTH_DIMS * max(self.cycle_frames, 1.0))


# =========================================================== cell_grow[stretch]: growth read off the cell's stretch
# `model: stretch`, written for experiment 13 (`experiments/exp13_mechanical_size.md`), whose Stage 0 audit
# found that no `cell_grow` model reads the cell's mechanical state. `model=` and not `implementation=`, by
# the axis test: a different growth law, not the same one computed differently. Its three helpers
# (`face_neighbour_mean`, `polar_gf`, `clone_cap`) sit with it. Folded here from its own module
# `growth_feedback.py` on 2026-09-27 (INSTRUCTION.md: every variant lives beside its base operator).


def face_neighbour_mean(values, es, et, ef, nF, rings=1):
    """Each face's value averaged with its edge-neighbours', `rings` times: (v_j + sum_nb v_k) / (1 + deg_j).
    Two faces touch when half-edge (a, b) of one has its twin (b, a) in the other."""
    es, et, ef = (torch.as_tensor(a).long().to(values.device) for a in (es, et, ef))
    live = (ef >= 0) & (ef < nF)
    es, et, ef = es[live], et[live], ef[live]
    if len(es) == 0 or rings <= 0:
        return values
    big = int(torch.maximum(es.max(), et.max())) + 1
    key, twin = es * big + et, et * big + es
    ks, order = torch.sort(key)
    j = torch.searchsorted(ks, twin).clamp(max=len(ks) - 1)
    ok = ks[j] == twin
    fa, fb = ef[ok], ef[order[j[ok]]]
    keep = fa != fb
    fa, fb = fa[keep], fb[keep]
    deg = torch.zeros(nF, dtype=values.dtype, device=values.device).index_add_(
        0, fa, torch.ones_like(fa, dtype=values.dtype))
    v = values
    for _ in range(int(rings)):
        v = (v + torch.zeros_like(v).index_add_(0, fa, v[fb])) / (1.0 + deg)
    return v


def polar_gf(centroids, axis, angle_deg, width_deg):
    """A growth factor in [0, 1] highest around `axis`: 1 / (1 + exp((theta - angle) / width)), theta the
    angle between `axis` and the cell's centroid seen from the tissue's centroid, in degrees. Set by the
    ANGLE, so it scales with the tissue as it grows -- Aegerter-Wilmsen 2007's third assumption, "the
    Dpp activity gradient and that of the other growth factor are scaled"."""
    c0 = torch.as_tensor(centroids)
    c = c0.to(torch.float64)
    # THE AXIS ON THE CENTROIDS' DEVICE: live, they are CUDA tensors, and a CPU axis made batch 6's four
    # jobs die in their first tick (exp 13, 2026-09-26) -- a CPU smoke run cannot see it.
    a = torch.as_tensor(axis, dtype=torch.float64, device=c.device)
    a = a / a.norm().clamp(min=1e-12)
    x = c - c.mean(0)
    cos = (x @ a) / x.norm(dim=1).clamp(min=1e-12)
    theta = torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))
    return torch.sigmoid((float(angle_deg) - theta) / max(float(width_deg), 1e-9)).to(c0.dtype)


def isotropic_stress(pos, es, et, ef, nF, A, A0, P0, mech, myo=None):
    """Each face's isotropic (Batchelor) stress on a vertex sheet, over 2 K_A A0 -- dimensionless:

        s_f = [ 2 K_A (A_f - A0_f) + (1 / (4 A_f)) sum_{h in f} T_h |l_h| ] / (2 K_A A0_f)
        T_h = tau_h + tau_twin,   tau_h = Lambda m_h + 2 K_P (P_f - P0_f) + Gamma P_f

    half the trace of sigma_f = -Pi_f I + (1 / A_f) sum_h (T_h / 2) l_h l_h / |l_h| for the default
    `cell_mechanics` energy (K_A, K_P, Gamma, Lambda read from the mesh's `mech`). Tensile positive; 0 for a
    cell at force balance of ANY size, which is what a size-free stretch must be. m_h is the half-edge's
    myosin multiplier on Lambda (`junction_myosin` writes it as m["myo"], `cell_mechanics` uses it); 1 when
    `myo` is None or does not match the half-edge count, so a run without myosin reads as before."""
    es, et, ef = (torch.as_tensor(a).long().to(pos.device) for a in (es, et, ef))
    l = pos[et] - pos[es]
    L = l.norm(dim=1)
    dt = pos.dtype
    P = torch.zeros(nF, dtype=dt, device=pos.device).index_add_(0, ef, L)
    KA, KP, G, Lam = (float(mech.get(k, 0.0)) for k in ("K_A", "K_P", "Gamma", "Lambda"))
    mh = torch.ones_like(L) if myo is None or myo.shape[0] != L.shape[0] else myo.to(pos.device, dt).reshape(-1)
    tau = Lam * mh + (2 * KP * (P - P0.to(dt)) + G * P)[ef]            # per half-edge
    big = int(torch.maximum(es.max(), et.max())) + 1
    key, twin = es * big + et, et * big + es
    ks, order = torch.sort(key)
    j = torch.searchsorted(ks, twin).clamp(max=len(ks) - 1)
    has = ks[j] == twin
    T = tau + torch.where(has, tau[order[j]], torch.zeros_like(L))
    iso_t = torch.zeros(nF, dtype=dt, device=pos.device).index_add_(0, ef, T * L) / (4 * A.to(dt).clamp(min=1e-12))
    return (2 * KA * (A.to(dt) - A0.to(dt)) + iso_t) / (2 * max(KA, 1e-12) * A0.to(dt).clamp(min=1e-12))


def clone_disk(centroids, point, frac):
    """Indices of the `frac` of cells nearest the point centroid + `point` x R, R the largest distance of
    a cell centroid from the tissue's centroid -- a compact clone INSIDE a tissue (on a flat disc the
    polar cap of `clone_cap` is the free rim). At least one cell when frac > 0."""
    c = torch.as_tensor(centroids).to(torch.float64)
    g = c.mean(0)
    R = (c - g).norm(dim=1).max()
    target = g + torch.as_tensor(point, dtype=torch.float64, device=c.device) * R
    n = max(1, int(round(float(frac) * len(c)))) if frac > 0 else 0
    return torch.argsort((c - target).norm(dim=1))[:n] if n else torch.zeros(0, dtype=torch.long)


def hinge_ring(centroids, frac):
    """Indices of the `frac` of cells whose centroids lie FARTHEST from the tissue's centroid -- the outer
    annulus of a disc, the hinge that rings the wing pouch. At least one cell when frac > 0."""
    c = torch.as_tensor(centroids, dtype=torch.float64)
    n = c.shape[0]
    k = max(1, int(round(float(frac) * n))) if frac > 0 else 0
    if k == 0:
        return torch.zeros(0, dtype=torch.long)
    r = (c - c.mean(0)).norm(dim=1)
    return torch.argsort(r, descending=True)[:k]


def clone_cap(centroids, axis, frac):
    """Indices of the `frac` of cells whose centroid lies farthest along `axis` from the tissue's
    centroid -- a polar cap, the compact patch a clone is (Shraiman 2005 Fig. 1). At least one cell
    when frac > 0."""
    c = torch.as_tensor(centroids).to(torch.float64)
    a = torch.as_tensor(axis, dtype=torch.float64, device=c.device)
    a = a / a.norm().clamp(min=1e-12)
    proj = (c - c.mean(0)) @ a
    n = max(1, int(round(float(frac) * len(c)))) if frac > 0 else 0
    return torch.argsort(proj)[len(c) - n:] if n else torch.zeros(0, dtype=torch.long)


@register_operator("cell_grow", model="stretch", set="vertex", kind="lateral", family="population",
                   title="grow where the cell is stretched",
                   equation=r"""$$\frac{ds_j}{dt}=s_j\,\text{rate}\big(\rho+\mathrm{Hill}(a_j)\big)\,m_j\,\mathrm{clip}\big(1+g(\sigma_j-1),0,f_{max}\big),\qquad \sigma_j=\frac{V_j/V^0_j}{\mathrm{median}_k\,V_k/V^0_k}$$""")
class Grow3DStretch(Grow3D):
    r"""the growth rate rises when the cell is stretched and falls when it is compressed: mechanical feedback on growth (Hippo / YAP), with a clone, an AND gate, a hinge and Aegerter-Wilmsen's arrest as options.
    cell -> vertex: as cell_grow, the rate multiplied by the clone factor and the stretch factor.

        \frac{ds_j}{dt}=s_j\,\text{rate}\big(\rho+\mathrm{Hill}(a_j)\big)\,m_j\,\mathrm{clip}\big(1+g(\sigma_j-1),0,f_{max}\big),\qquad \sigma_j=\frac{V_j/V^0_j}{\mathrm{median}_k\,V_k/V^0_k}
        s_j, rate, \rho, a_j as cell_grow
        m_j            `clone_mult` for a clone cell, else 1
        g              `gain`, the fractional change of the rate per unit relative stretch (0: the default, bit for bit)
        \sigma_j       the cell's relative stretch (`readout`: volume, area, thickness, a0 or stress) over the tissue median
        f_{max}        `f_max`
        V_j, V^0_j     the cell's volume and its target

    Reference: Shraiman, B. I. (2005). Mechanical feedback as a possible regulator of tissue growth. PNAS 102:3318-3323; Aegerter-Wilmsen, T. et al. (2007). Mech. Dev. 124:318-326."""
    PARAM_UNITS = {'gain': 'dimensionless', 'f_max': 'dimensionless', 'and_hill': 'dimensionless', 'and_sw': 'dimensionless', 'and_chan': 'count', 'clone_frac': 'fraction', 'clone_mult': 'dimensionless', 'clone_point': 'length', 'gf_angle_deg': 'angle', 'gf_width_deg': 'angle', 'hinge_frac': 'fraction', 'hinge_mult': 'dimensionless', 'smooth': 'count', 'arrest_sw': 'dimensionless', 'clone_axis': 'count', 'gf_axis': 'count'}
    SYMBOLS = {'s_j, rate, \\rho, a_j': 'as cell_grow', 'm_j': '`clone_mult` for a clone cell, else 1', 'g': '`gain`, the fractional change of the rate per unit relative stretch (0: the default, bit for bit)', '\\sigma_j': "the cell's relative stretch (`readout`: volume, area, thickness, a0 or stress) over the tissue median", 'f_{max}': '`f_max`', 'V_j, V^0_j': "the cell's volume and its target"}
    PARAM_ROLES = {'and_chan': 'second_morphogen_column_of_the_and_gate', 'and_hill': 'its_hill_exponent', 'and_sw': 'its_switch_level', 'clone_axis': 'axis_of_the_clone', 'clone_block': 'cell_block_that_marks_the_clone', 'clone_frac': 'fraction_of_cells_in_the_clone', 'clone_mult': 'growth_multiplier_of_the_clone', 'clone_point': 'point_about_which_the_clone_is_chosen', 'drive': 'what_drives_the_growth', 'f_max': 'clip_on_the_stretch_factor', 'gain': 'stretch_feedback_gain', 'gf_angle_deg': 'angle_of_the_growth_factor_sector', 'gf_axis': 'axis_of_that_sector', 'gf_width_deg': 'its_width', 'hinge_block': 'cell_block_of_the_hinge', 'hinge_frac': 'fraction_of_cells_in_the_hinge', 'hinge_mult': 'growth_multiplier_of_the_hinge', 'hinge_ring': 'how_the_hinge_is_chosen', 'readout': 'which_stretch_is_read', 'smooth': 'neighbour_rings_the_stretch_is_averaged_over', 'arrest_sw': 'arrest_threshold_of_the_aw2007_law', 'ref_frame': 'frame_at_which_the_stretch_reference_is_taken', 'stretch_growth': 'growth_rate_added_per_unit_stretch', 'stretch_ref': 'what_the_stretch_is_measured_against', 'stretch_threshold': 'stretch_below_which_no_stretch_growth_is_added'}
    MECHANISM_TAGS = ["growth", "size_control", "mechanical_feedback", "stretch", "clone"]
    REFERENCE = ("Shraiman, B.I. (2005). PNAS 102:3318; Aegerter-Wilmsen, T., Aegerter, C.M., Hafen, E. "
                 "& Basler, K. (2007). Mech. Dev. 124:318.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.gain = float(params.get("gain", 0.0))
        self.f_max = float(params.get("f_max", 4.0))
        self.clone_mult = float(params.get("clone_mult", 1.0))
        self.clone_frac = float(params.get("clone_frac", 0.0))
        self.clone_axis = [float(v) for v in (params.get("clone_axis") or [0.0, 0.0, 1.0])]
        self.clone_block = str(params.get("clone_block", "clone"))
        self.hinge_frac = float(params.get("hinge_frac", 0.0))
        self.hinge_mult = float(params.get("hinge_mult", 0.0))
        self.hinge_block = str(params.get("hinge_block", "hinge"))
        self._hinge_marked = False
        _ac = params.get("and_chan", None)
        self.and_chan = None if _ac is None else int(_ac)
        self.and_sw = self.tunable(params.get("and_sw"), 0.3, tensor=True)
        self.and_hill = float(params.get("and_hill", 2.0))
        self._and = None
        _cp = params.get("clone_point", None)
        self.clone_point = None if _cp is None else [float(v) for v in _cp]
        self.readout = str(params.get("readout", "volume")).lower()
        self.smooth = int(params.get("smooth", 0))
        # AEGERTER-WILMSEN 2007's LAW, every option off by default (see the class docstring's last part)
        self.drive_kind = str(params.get("drive", "uniform")).lower()
        if self.drive_kind not in ("uniform", "polar"):
            raise ValueError(f"cell_grow[stretch]: drive is uniform or polar, got {self.drive_kind!r}")
        self.gf_axis = [float(v) for v in (params.get("gf_axis") or [0.0, 0.0, 1.0])]
        self.gf_angle = float(params.get("gf_angle_deg", 60.0))
        self.gf_width = float(params.get("gf_width_deg", 10.0))
        self.stretch_ref = str(params.get("stretch_ref", "tissue")).lower()
        if self.stretch_ref not in ("tissue", "settle", "settle_median"):
            raise ValueError(f"cell_grow[stretch]: stretch_ref is tissue, settle or settle_median, got {self.stretch_ref!r}")
        self._off = None
        self.ref_frame = int(params.get("ref_frame", 60))
        self.stretch_growth = float(params.get("stretch_growth", 0.0))
        self.stretch_threshold = self.tunable(params.get("stretch_threshold"), 0.0, tensor=True)
        self._ref = None
        self._gf = None
        if self.readout not in ("volume", "area", "thickness", "a0", "stress"):
            raise ValueError(f"cell_grow[stretch]: readout is volume, area, thickness, a0 or stress, got {self.readout!r}")
        self._marked = False
        self._drive = None

    def _clone_drive(self, H):
        """m_j per live cell, or None when there is no clone (so the default path multiplies nothing)."""
        if self.clone_frac <= 0 and self.clone_mult == 1.0:
            return None
        from plexus.operators.vertex_ops import cell_block_t, set_cell_block_t
        cat = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
        m = getattr(H.level(self.at), "_mesh", None)
        if m is None:
            return None
        nF = int(m["nF"])
        k = cell_block_t(H, cat, self.clone_block, nF)
        if k is None:
            raise ValueError(f"cell_grow[stretch]: a clone needs `sets.{cat}.state.{self.clone_block}: "
                             f"{{width: 1}}` declared on the cell set")
        if not self._marked and self.clone_frac > 0:
            clvl = H.level(cat)
            if "centroid" not in clvl.state_schema:
                raise ValueError(f"cell_grow[stretch]: marking a clone needs the `centroid` block on "
                                 f"`{cat}` (written by `cell_geometry`, scheduled before this operator)")
            vals = torch.zeros(nF, dtype=clvl.state.dtype, device=clvl.state.device)
            _c = clvl.get("centroid")[:nF].detach().cpu()
            vals[clone_disk(_c, self.clone_point, self.clone_frac) if self.clone_point is not None
                 else clone_cap(_c, self.clone_axis, self.clone_frac)] = 1.0
            set_cell_block_t(H, cat, self.clone_block, vals, nF)
            self._marked = True
            k = cell_block_t(H, cat, self.clone_block, nF)
        one = torch.ones((), dtype=k.dtype, device=k.device)
        return torch.where(k > 0.5, one * self.clone_mult, one)

    def _hinge_drive(self, H):
        """`hinge_mult` for hinge cells, 1 elsewhere; None without a hinge (the default path)."""
        if self.hinge_frac <= 0:
            return None
        from plexus.operators.vertex_ops import cell_block_t, set_cell_block_t
        cat = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
        m = getattr(H.level(self.at), "_mesh", None)
        if m is None:
            return None
        nF = int(m["nF"])
        k = cell_block_t(H, cat, self.hinge_block, nF)
        if k is None:
            raise ValueError(f"cell_grow[stretch]: a hinge needs `sets.{cat}.state.{self.hinge_block}: "
                             f"{{width: 1}}` declared on the cell set")
        if not self._hinge_marked:
            clvl = H.level(cat)
            if "centroid" not in clvl.state_schema:
                raise ValueError(f"cell_grow[stretch]: marking a hinge needs the `centroid` block on `{cat}`")
            vals = torch.zeros(nF, dtype=clvl.state.dtype, device=clvl.state.device)
            vals[hinge_ring(clvl.get("centroid")[:nF].detach().cpu(), self.hinge_frac)] = 1.0
            set_cell_block_t(H, cat, self.hinge_block, vals, nF)
            self._hinge_marked = True
            k = cell_block_t(H, cat, self.hinge_block, nF)
        one = torch.ones((), dtype=k.dtype, device=k.device)
        return torch.where(k > 0.5, one * self.hinge_mult, one)

    def forward(self, H, mask=None):
        self._drive = self._clone_drive(H)
        hd = self._hinge_drive(H)
        if hd is not None:
            self._drive = hd if self._drive is None else self._drive * hd.to(self._drive.device)
        self._H = H
        self._gf = self._polar_gf(H) if self.drive_kind == "polar" else None
        self._and = self._and_gate(H) if self.and_chan is not None else None
        return super().forward(H, mask)

    def _polar_gf(self, H):
        """The growth factor of each live cell, a logistic cap around `gf_axis` (`polar_gf`); written into
        a declared `gf` block for the movie and the rulers."""
        cat = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
        m = getattr(H.level(self.at), "_mesh", None)
        clvl = H.level(cat)
        if m is None or "centroid" not in clvl.state_schema:
            raise ValueError("cell_grow[stretch] drive: polar needs the `centroid` block (cell_geometry)")
        nF = int(m["nF"])
        gf = polar_gf(clvl.get("centroid")[:nF].detach(), self.gf_axis, self.gf_angle, self.gf_width)
        if "gf" in clvl.state_schema:
            from plexus.operators.vertex_ops import set_cell_block_t
            set_cell_block_t(H, cat, "gf", gf, nF)
        return gf

    def _and_gate(self, H):
        """Hill(c) of the `and_chan` morphogen, relative to its own maximum (`and_sw` of it, sharpness
        `and_hill`) -- the second growth factor of an AND gate (the class docstring)."""
        cat = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
        m = getattr(H.level(self.at), "_mesh", None)
        clvl = H.level(cat)
        if m is None or "chem" not in clvl.state_schema:
            return None
        nF = int(m["nF"])
        h0, _ = clvl.state_schema["chem"]
        c = clvl.state[:nF, h0 + self.and_chan].detach().clamp(min=0.0)
        cmax = float(c.max()) if c.numel() else 0.0
        if cmax <= 1e-9:
            return torch.zeros_like(c)
        thr = self.and_sw * cmax
        return c ** self.and_hill / (thr ** self.and_hill + c ** self.and_hill)

    def _rate(self, s_prev, hillv, m, v_ref):
        if self._and is not None:
            hillv = hillv * self._and[:int(s_prev.shape[0])].to(hillv.dtype).to(hillv.device)
        base = super()._rate(s_prev, hillv, m, v_ref)
        nF = int(s_prev.shape[0])
        r = base
        if self._gf is not None:
            r = r * self._gf[:nF].to(r.dtype).to(r.device)
        if self._drive is not None:
            r = r * self._drive[:nF].to(r.dtype).to(r.device)
        need_sigma = self.gain != 0.0 or self._stretch_declared() or self.stretch_growth > 0
        if not need_sigma:
            return r
        if self.readout == "stress":
            mech = m.get("mech")
            vlvl = self._H.level(self.at)
            if mech is None or not all(k in m for k in ("E_srce", "E_trgt", "E_face")):
                return r                                       # the mechanics has not run yet: no signal
            from plexus.operators.vertex_ops import cell_block_t
            a = cell_block_t(self._H, self.cat, "area", nF)
            pos = vlvl.get("pos")[:int(m["Nv"])].detach().to(r.dtype)
            sig = 1.0 + isotropic_stress(pos, m["E_srce"], m["E_trgt"], m["E_face"], nF, a.to(r.device),
                                         m["A0"][:nF], m["P0"][:nF], mech, myo=m.get("myo")).to(r.device)
            if self.stretch_ref == "tissue":
                sig = sig - sig.median() + 1.0
            elif self.stretch_ref == "settle_median":
                # the tissue's median stress at `ref_frame`, subtracted from then on as a FIXED offset: a
                # settled pouch reads 1 (its line tension's standing compression removed), and a later
                # global compression or stretch is still seen -- unlike `tissue`, which removes it every frame
                # -- and NO signal before it (sigma = 1): an unsettled mesh's stress spans -0.7..1.8, which a
                # stretch-growth term turns into x14 area in 30 frames (the batch-9 CPU smoke)
                if self._off is None and getattr(self, "_k", 0) >= self.ref_frame:
                    self._off = (sig.median() - 1.0).detach()
                sig = sig - self._off if self._off is not None else torch.ones_like(sig)
            if self.smooth > 0:
                sig = face_neighbour_mean(sig, m["E_srce"], m["E_trgt"], m["E_face"], nF, self.smooth)
            if self._stretch_declared():
                from plexus.operators.vertex_ops import set_cell_block_t
                set_cell_block_t(self._H, self.cat, "stretch", sig.detach(), nF)
            if self.gain != 0.0:
                r = r * (1.0 + self.gain * (sig - 1.0)).clamp(0.0, self.f_max)
            if self.stretch_growth > 0:
                r = r + s_prev.to(r.dtype) * self.rate * self.stretch_growth * (sig - 1.0 - self.stretch_threshold).clamp(min=0.0)
            return r
        v = self._v_now[:nF].to(r.dtype).clamp(min=1e-12)
        if self.readout == "a0":
            from plexus.operators.vertex_ops import cell_block_t
            a = cell_block_t(self._H, self.cat, "area", nF)
            if a is None:
                raise ValueError("cell_grow[stretch] readout: a0 needs the `area` block (cell_geometry)")
            ratio = a.to(r.dtype).to(r.device).clamp(min=1e-12) / m["A0"][:nF].to(r.dtype).clamp(min=1e-12)
        elif self.readout in ("area", "thickness"):
            from plexus.operators.vertex_ops import cell_block_t
            a = cell_block_t(self._H, self.cat, "area", nF)
            if a is None:
                raise ValueError("cell_grow[stretch] readout: area needs the `area` block on the cell set "
                                 "(written by `cell_geometry`)")
            a = a.to(r.dtype).to(r.device).clamp(min=1e-12)
            ratio = a / v ** (2.0 / 3.0) if self.readout == "area" else a / v
        else:
            ratio = v / m["V0f"][:nF].to(r.dtype).clamp(min=1e-12)
        ref = ratio.median().clamp(min=1e-12)
        if self.stretch_ref in ("settle", "settle_median"):
            if self._ref is None and getattr(self, "_k", 0) >= self.ref_frame:
                self._ref = ref.detach().clone()
            ref = self._ref if self._ref is not None else ref
        sigma = ratio / ref
        if self.smooth > 0:
            sigma = face_neighbour_mean(sigma, m["E_srce"], m["E_trgt"], m["E_face"], nF, self.smooth)
        if self._stretch_declared():
            from plexus.operators.vertex_ops import set_cell_block_t
            set_cell_block_t(self._H, self.cat, "stretch", sigma.detach(), nF)
        if self.gain != 0.0:
            r = r * (1.0 + self.gain * (sigma - 1.0)).clamp(0.0, self.f_max)
        if self.stretch_growth > 0:
            # growth INDUCED by stretch above a threshold, wherever the cell is (AW 2007: the periphery,
            # with no growth factor, grows only when stretched), so it is s * rate, not the drive.
            r = r + s_prev.to(r.dtype) * self.rate * self.stretch_growth * (
                sigma - 1.0 - self.stretch_threshold).clamp(min=0.0)
        return r

    def _stretch_declared(self):
        H = getattr(self, "_H", None)
        if H is None or getattr(self, "cat", None) is None:
            return False
        return "stretch" in getattr(H.level(self.cat), "state_schema", {})



@register_operator("cell_grow", model="timer_planar", set="vertex", kind="lateral", family="population",
                   title="grow to a target area",
                   equation=r"""$$\frac{d\ln V_j}{dt}=\frac{1}{T}\ln\frac{V_{\mathrm{target}}}{V_j}$$""")
class Grow3DTimerPlanar(Grow3DTimer):
    r"""timer for a monolayer: a cell grows in area at constant height, so its volume target scales as s^2 like its area; `hold_block` and `size_block` mark held or bigger cells.
    cell -> vertex: as timer.

        \frac{d\ln V_j}{dt}=\frac{1}{T}\ln\frac{V_{\mathrm{target}}}{V_j}
        V_j            the cell's target volume
        T              `cycle_frames`, the time constant
        V_{target}     the target size, scaled by (rho + Hill(a)) / (rho + 1)

    Reference: as timer; Plexus exp14 (this work)."""
    PARAM_UNITS = {'size_factor': 'dimensionless'}
    SYMBOLS = {'V_j': "the cell's target volume", 'T': '`cycle_frames`, the time constant', 'V_{target}': 'the target size, scaled by (rho + Hill(a)) / (rho + 1)'}
    PARAM_ROLES = {'hold_block': 'cell_block_of_the_held_cells', 'size_block': 'cell_block_of_the_bigger_cells', 'size_factor': 'their_target_factor'}
    GROWTH_DIMS = 2
    MECHANISM_TAGS = ["growth", "size_control", "timer", "target_size", "monolayer"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.hold_block = params.get("hold_block")
        self.size_block = params.get("size_block")
        self.size_factor = float(params.get("size_factor", 1.0))
        self._hold = None
        self._dying = None
        self._big = None

    def forward(self, H, mask=None):
        self._hold = self._dying = self._big = None
        if self.size_block and self.size_factor != 1.0:
            from plexus.operators.vertex_ops import cell_block, resolve_cell_set
            m = getattr(H.level(self.at), "_mesh", None)
            if m is not None:
                b = cell_block(H, resolve_cell_set(H, self.at, getattr(self, "_cat", None)),
                               self.size_block, int(m["nF"]))
                if b is not None:
                    self._big = torch.as_tensor(b > 0.5)
                    d = cell_block(H, resolve_cell_set(H, self.at, getattr(self, "_cat", None)),
                                   "apop_flag", int(m["nF"]))
                    self._dying = torch.as_tensor(d > 0) if d is not None else None
        if self.hold_block:
            from plexus.operators.vertex_ops import cell_block, resolve_cell_set
            m = getattr(H.level(self.at), "_mesh", None)
            if m is not None:
                nF = int(m["nF"])
                cat = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
                b = cell_block(H, cat, self.hold_block, nF)
                if b is not None:
                    self._hold = torch.as_tensor(b > 0.5)
                    d = cell_block(H, cat, "apop_flag", nF)
                    self._dying = torch.as_tensor(d > 0) if d is not None else None
        return super().forward(H, mask)

    def _rate(self, s_prev, hillv, m, v_ref):
        ds = super()._rate(s_prev, hillv, m, v_ref)
        if self._hold is not None and self._hold.shape[0] == ds.shape[0]:
            hold = self._hold.to(ds.device)
            v_now = (m["V0f_init"] * s_prev ** self.GROWTH_DIMS).clamp(min=1e-9)
            if bool((~hold).any()):
                # the same law with v_tgt = the cycling cells' mean volume
                v_tgt = v_now[~hold].mean()
                back = s_prev * torch.log(v_tgt / v_now) / (self.GROWTH_DIMS * max(self.cycle_frames, 1.0))
                ds = torch.where(hold, back, ds)
            if self._dying is not None and self._dying.shape[0] == ds.shape[0]:
                ds = torch.where(hold & self._dying.to(ds.device), torch.zeros_like(ds), ds)
        if self._big is not None and self._big.shape[0] == ds.shape[0]:
            big = self._big.to(ds.device)
            if self._dying is not None and self._dying.shape[0] == ds.shape[0]:
                big = big & ~self._dying.to(ds.device)
            extra = s_prev * float(np.log(self.size_factor)) / (self.GROWTH_DIMS * max(self.cycle_frames, 1.0))
            ds = torch.where(big, ds + extra, ds)
        return ds


@register_operator("interface_tension", set="vertex", kind="lateral", family="mechanics", title="tighten the activator boundary",
                   equation=r"""$$E=K_{\mathrm{purse}}\!\!\sum_{e\in\mathrm{interface}}\!\!\ell_e,\qquad \mathbf f_v=-\frac{\partial E}{\partial\mathbf x_v}=-K_{\mathrm{purse}}\sum_{e\ni v}\frac{\mathbf x_v-\mathbf x_{\mathrm{other}}}{\ell_e}$$""")
class InterfaceLineTension3D(Lateral):
    r"""a purse-string line tension on the activator interface, and nothing else: the mesh edges separating an activator-high cell from a low one pull their ends together.
    vertex -> vertex: reads pos and the cell set's chem, emits a velocity on the interface vertices, which the engine integrates beside the shape energy.

        E=K_{\mathrm{purse}}\!\!\sum_{e\in\mathrm{interface}}\!\!\ell_e,\qquad \mathbf f_v=-\frac{\partial E}{\partial\mathbf x_v}=-K_{\mathrm{purse}}\sum_{e\ni v}\frac{\mathbf x_v-\mathbf x_{\mathrm{other}}}{\ell_e}
        E              the purse-string energy
        K_{purse}      `K_purse`, the line tension, force
        \ell_e         the length of interface edge e
        f_v            the force on vertex v
        x_v, x_{other} the edge's two vertices (high means a > a_sw x the field's running maximum)

    Reference: Plexus (this work); purse-string apical constriction after Okuda, S. et al. (2018). Combining Turing and 3D vertex models reproduces autonomous multicellular morphogenesis of the tissue. Sci. Rep. 8:2386."""
    PARAM_UNITS = {'K_purse': 'force', 'a_sw': 'fraction', 'cap_frac': 'fraction', 'eta': 'fraction', 'iters': 'count'}
    SYMBOLS = {'E': 'the purse-string energy', 'K_{purse}': '`K_purse`, the line tension, force', '\\ell_e': 'the length of interface edge e', 'f_v': 'the force on vertex v', 'x_v, x_{other}': "the edge's two vertices (high means a > a_sw x the field's running maximum)"}
    SUPPORTED_DIMS = [3]; EMIT = "velocity"; DIFFERENTIABLE = True
    INPUTS = ["vertex", "cell"]; OUTPUTS = ["vertex"]; READS = ["pos", "chem"]; WRITES = ["pos"]
    MECHANISM_TAGS = ["interface_tension", "purse_string", "tube", "oriented", "cross_scale"]
    PARAM_ROLES = {"K_purse": "interface_line_tension", "a_sw": "red_threshold", 'cap_frac': 'displacement_cap_as_a_fraction_of_the_mean_edge', 'cell_set': 'the_cell_set', 'eta': 'mobility_factor', 'iters': 'relaxation_iterations'}
    REFERENCE = "Plexus (this work); purse-string / apical-constriction tubulation after Okuda, S. et al. (2018). Sci. Rep. 8:2386."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex"); self._cat = params.get("cell_set")
        self.K_purse = self.tunable(params.get("K_purse"), 1.0, tensor=True)
        # 0.6, AND IT WAS 1.0 -- A DEFAULT THAT CANNOT FIRE. The gate below is
        # `red = a > a_sw * amax`, so a_sw = 1.0 asks for cells STRICTLY ABOVE the maximum: the
        # empty set, by construction, at every operating point and for every value of K_purse.
        #
        # This is the SECOND time this operator has been written off as inert without ever having
        # run. The first was an absolute threshold against a field whose median maximum is 0.000,
        # fixed by making a_sw a fraction -- and the fix left a default that is a fraction of one.
        # Route A then swept K_purse [0, 0.25, 3, 6] on b_gs_gated_shaping, whose spec omits a_sw,
        # and got four runs identical to four significant figures with `acted = 0` on all of them.
        # Reported as "K_purse is inert"; nothing was measured. 0.6 is the composition space's own
        # declared default and means "the top 40% of the field is red".
        self.a_sw = float(params.get("a_sw", 0.6)); self.eta = float(params.get("eta", 0.05))
        self.cap_frac = float(params.get("cap_frac", 0.10)); self.iters = int(params.get("iters", 4))

    def forward(self, H, mask=None):
        # THE PAIRING IS READ FROM THE SET, ONCE PER CALL -- see `resolve_cell_set`.
        self.cat = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
        from plexus.operators.vertex_ops import ShapeEnergy3D
        vlvl = H.level(self.at); m = getattr(vlvl, "_mesh", None); clvl = H.level(self.cat)
        if m is None or "chem" not in clvl.state_schema:
            return {}
        nF = int(m["nF"]); Nv = int(m["Nv"]); dev = vlvl.state.device; dt = vlvl.state.dtype
        es = torch.as_tensor(m["E_srce"], device=dev, dtype=torch.long)      # robust to numpy/tensor after division
        et = torch.as_tensor(m["E_trgt"], device=dev, dtype=torch.long)
        ef = torch.as_tensor(m["E_face"], device=dev, dtype=torch.long)
        h0, _ = clvl.state_schema["chem"]
        a = clvl.state[:nF, h0].detach().to(dev)
        # `a_sw` IS A FRACTION OF THE FIELD'S OWN MAXIMUM, NOT AN ABSOLUTE VALUE.
        #
        # An ABSOLUTE threshold cannot work here, because the activator's own ceiling is whatever
        # the chemistry produces and that varies by orders of magnitude between parameter sets. A
        # threshold declared above the field selects nothing, the operator returns an empty delta
        # on every scheduled frame, and the ledger records
        # and the Analyst reported it "inert" for two rounds without being able to say why. Nine
        # edits, 10% of a campaign, on an operator that could not fire at any legal setting.
        #
        # Fixing the RANGE would have been a patch: the ceiling moves with the chemistry, so the
        # next parent could put it out of reach again. `cell_chem_from_shape` in this same repo
        # standardises its feature for exactly this reason -- so `beta` means one thing whatever
        # the units -- and its docstring names the alternative as finding F009. A threshold
        # relative to the field cannot be outside the field.
        amax = float(a.max()) if a.numel() else 0.0
        red = (a > self.a_sw * amax).to(dt) if amax > 0 else torch.zeros_like(a)
        twin = ShapeEnergy3D._twin_faces(es, et, ef, Nv)        # neighbour cell across each half-edge
        iface = (red[ef] != red[twin]).to(dt)                   # 1 on the red/white interface half-edges (the ring)
        if float(red.sum()) < 1.0 or float(iface.sum()) < 1.0:  # no red spot / no interface yet -> nothing to do
            return {}
        px0, px1 = vlvl.state_schema["pos"]
        x0 = vlvl.state[:, px0:px1].detach()
        x = x0[:Nv].clone()
        cap = self.cap_frac * float((x0[et] - x0[es]).norm(dim=-1).mean().clamp(min=1e-3))
        for _ in range(self.iters):              # DIRECT forces (no autograd): the purse-string alone
            force = torch.zeros(Nv, 3, device=dev, dtype=dt)
            d = x[et] - x[es]; u = d / (d.norm(dim=-1, keepdim=True) + 1e-9)   # interface edge shortens
            f = (self.K_purse * iface)[:, None] * u
            force.index_add_(0, es, f); force.index_add_(0, et, -f)
            x = x + (self.eta * force).clamp(-cap, cap)
        vel = torch.zeros_like(x0)
        vel[:Nv] = (x - x0[:Nv]) / max(float(getattr(H.config, "dt", 1.0)), 1e-6)
        occ = vlvl.occ[:, None] if getattr(vlvl, "occ", None) is not None else 1.0
        return {self.at: vel * occ}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="brusselator", title="run textbook activator-inhibitor reactions",
                   species=(("a", "activator"), ("h", "inhibitor")),
                   equation=r"""$$\frac{da}{dt}=\gamma\big(A-(B+1)a+a^{2}h\big),\qquad \frac{dh}{dt}=\gamma\big(B\,a-a^{2}h\big)$$""")
class CellReactBrusselator(Lateral):
    r"""the Brusselator: the textbook activator-inhibitor system whose Turing condition is exactly solvable, so whether a pattern is possible can be checked before running.
    cell -> cell: reads chem = [a, h], emits d(chem)/dt; local.

        \frac{da}{dt}=\gamma\big(A-(B+1)a+a^{2}h\big),\qquad \frac{dh}{dt}=\gamma\big(B\,a-a^{2}h\big)
        a              the activator
        h              the inhibitor
        \gamma         `gamma`, the time rescaling
        A              `A`, the constant feed, per time
        B              `B`, the conversion of activator into inhibitor, per time

    Reference: Prigogine, I. & Lefever, R. (1968). Symmetry breaking instabilities in dissipative systems II. J. Chem. Phys. 48:1695-1700."""
    PARAM_UNITS = {'A': 'rate', 'B': 'rate', 'gamma': 'dimensionless'}
    SYMBOLS = {'a': 'the activator', 'h': 'the inhibitor', '\\gamma': '`gamma`, the time rescaling', 'A': '`A`, the constant feed, per time', 'B': '`B`, the conversion of activator into inhibitor, per time'}
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["gamma", "A", "B"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]
    MECHANISM_TAGS = ["reaction", "activator_inhibitor", "turing", "brusselator"]
    REFERENCE = "Prigogine, I. & Lefever, R. (1968). Symmetry breaking instabilities in dissipative systems. J. Chem. Phys. 48:1695-1700."
    PARAM_ROLES = {"gamma": "reaction_rate", "A": "feed", "B": "conversion"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.gamma = float(params["gamma"]); self.A = float(params["A"]); self.B = float(params["B"])

    def forward(self, H, mask=None):
        lvl = H.level(self.at); chem = lvl.get("chem")
        a = chem[:, 0]; h = chem[:, 1]; a2h = a * a * h
        da = self.gamma * (self.A - (self.B + 1.0) * a + a2h)
        dh = self.gamma * (self.B * a - a2h)
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: torch.stack([da, dh], dim=1) * occ}


F_CEIL = 0.11


# --------------------------------------------------------------------------- shared machinery
def _cell_adjacency(es, et, ef, nF):
    """(src, dst) cell pairs: two cells are neighbours iff they share a mesh edge."""
    key = np.minimum(es, et).astype(np.int64) * (int(max(es.max(), et.max())) + 1) \
        + np.maximum(es, et)
    o = np.argsort(key, kind="stable")
    k, f = key[o], ef[o]
    src, dst = [], []
    i = 0
    while i < len(k):
        j = i
        while j + 1 < len(k) and k[j + 1] == k[i]:
            j += 1
        if j > i:
            for a in range(i, j + 1):
                for b in range(a + 1, j + 1):
                    if f[a] != f[b]:
                        src += [f[a], f[b]]; dst += [f[b], f[a]]
        i = j + 1
    return np.asarray(src, np.int64), np.asarray(dst, np.int64)


def _np(x):
    """Mesh arrays are torch tensors ON THE GPU in a real run and numpy in the self-test. Assuming
    numpy crashed the first end-to-end launch on cuda -- `can't convert cuda:0 device type tensor
    to numpy` -- after the CPU tests had all passed."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _standardise(phi, alive):
    """Median-centred, MAD-scaled, clipped. See the module docstring: without this, `beta` means a
    different physical quantity in each implementation and the sweep axis is meaningless."""
    ok = np.isfinite(phi) & (alive > 0)
    if ok.sum() < 8:
        return np.zeros_like(phi)
    x = phi[ok]
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    if mad < 1e-12:
        return np.zeros_like(phi)                      # a uniform field carries no signal
    out = np.zeros_like(phi)
    out[ok] = np.clip((x - med) / mad, -4.0, 4.0)      # clip: one spike must not drive the feed
    return out


class _ShapeToChemBase(Lateral):
    """The contract. Subclasses supply `_feature(...) -> per-cell scalar` and nothing else."""
    SUPPORTED_DIMS = [3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = False
    INPUTS = ["cell", "vertex"]; OUTPUTS = ["cell"]; READS = ["chem", "pos"]; WRITES = ["chem"]
    REQUIRES_PARAMS = ["beta"]
    MECHANISM_TAGS = ["shape_to_chemistry", "mechanochemical_feedback", "cross_scale", "closes_the_loop"]
    REFERENCE = ("Okuda, S. et al. (2018). Sci. Rep. 8:2386 (the shape-chemistry loop this closes); "
                 "Dupont, S. et al. (2011). Nature 474:179-183 (YAP/TAZ mechanotransduction); "
                 "Pearson, J. E. (1993). Science 261:189-192 (F selects the Gray-Scott morphology).")
    PARAM_ROLES = {"beta": "shape_feedback_strength", "F0": "baseline_feed_rate"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.vat = params.get("vertex_set", "vertex")
        self.beta = float(params["beta"])
        self.F0 = float(params.get("F0", 0.055))  # match cell_chem_react's feed, or it fights it
        self.rate = float(params.get("rate", 1.0))  # same time-scaling as cell_chem_react

    def _feature(self, pt, m, es, et, ef, nF):
        raise NotImplementedError

    def forward(self, H, mask=None):
        clvl = H.level(self.at); vlvl = H.level(self.vat)
        m = getattr(vlvl, "_mesh", None)
        if m is None or "chem" not in clvl.state_schema:
            return {}
        chem = clvl.get("chem")
        if self.beta == 0.0:
            return {self.at: torch.zeros_like(chem)}   # the NULL, and it must remain runnable
        nF = int(m["nF"])
        es = _np(m["E_srce"]); et = _np(m["E_trgt"]); ef = _np(m["E_face"])
        live = ef < nF
        es, et, ef = es[live], et[live], ef[live]
        pt = vlvl.get("pos")[:int(m["Nv"])].detach().cpu().numpy().astype(np.float64)
        alive = _np(m["alive"])[:nF] if "alive" in m else np.ones(nF)
        phi = self._feature(pt, m, es, et, ef, nF)
        if phi is None:                                # precondition absent: no-op, never a guess
            return {self.at: torch.zeros_like(chem)}
        return self._feed(clvl, chem, phi, alive, nF)

    def _feed(self, clvl, chem, phi, alive, nF):
        """The shape feature `phi` of every cell, standardised, as a change of the Gray-Scott feed: shared by every body (a mesh, a body of points)."""
        w = _standardise(np.asarray(phi, float), alive)
        dev, dt = chem.device, chem.dtype
        wt = torch.zeros(chem.shape[0], device=dev, dtype=dt)
        wt[:nF] = torch.as_tensor(w, device=dev, dtype=dt)
        # F_j = F0 (1 + beta phihat_j). The Gray-Scott feed acts on the SUBSTRATE: du/dt += F(1-u).
        # We contribute only the DIFFERENCE from the baseline feed cell_chem_react already applies, so
        # the two operators compose instead of double-counting.
        u = chem[:, 1]
        # A FEED RATE CANNOT BE NEGATIVE, and letting it go negative is not merely unphysical --
        # it is unstable. The substrate obeys du/dt = F (1 - u); with F < 0 and u < 1 the term is
        # negative, u falls, (1 - u) grows, and the whole thing diverges exponentially. Measured
        # before the clamp: `tension` at beta = 1.5 reached act_max 1.4e16 in forty frames, and
        # `apical_area` overflowed to NaN. The multiplier is clamped at zero, which caps the
        # feedback at "this cell is not fed at all" rather than "this cell is drained".
        # THE MODULATED FEED MUST STAY INSIDE THE GRAY-SCOTT REGIME, not merely stay positive.
        # Clamping only at zero was not enough: with phihat clipped at +/-4 and beta = 1.5 the
        # multiplier reached 7, so F rose to 0.385 -- far outside Pearson's diagram, which is
        # explored for F <~ 0.11. Measured consequence, in order: the activator climbed past 1.6
        # (Gray-Scott lives near 0.4), then u a^2 drained the substrate NEGATIVE at frame 15, and
        # the explicit step diverged to +/-inf by frame 25. A feedback strong enough to leave the
        # model's own parameter region is not a mechanism, it is a blow-up.
        F = torch.clamp(self.F0 * (1.0 + self.beta * wt), min=0.0, max=F_CEIL)
        dF = F - self.F0
        out = torch.zeros_like(chem)
        out[:, 1] = self.rate * dF * (1.0 - u)
        occ = clvl.occ[:, None] if getattr(clvl, "occ", None) is not None else 1.0
        return {self.at: out * occ}


# --------------------------------------------------------------------------- implementations
@register_operator("cell_chem_from_shape", set="cell", kind="lateral", family="fields",
                   model="curvature", title="make chemistry from curvature",
                   equation=r"""$$H_j=\frac{2\big(\overline{\mathbf x}_k-\mathbf x_j\big)\cdot\mathbf n_j}{d_j^{2}}$$""")
class ShapeToChemCurvature(_ShapeToChemBase):
    r"""the chemistry listens to curvature: the tissue's shape telling its cells where they are, a proxy for the mean curvature from the neighbours' centroids, certified against spheres of known radius.
    cell -> cell: reads the centroids, normals and edge_index, writes the curvature feedback into chem.

        H_j=\frac{2\big(\overline{\mathbf x}_k-\mathbf x_j\big)\cdot\mathbf n_j}{d_j^{2}}
        H_j            the curvature at cell j, per world length (positive where the surface bulges out)
        \bar x_k       the mean centroid of the neighbours k
        x_j            the cell's centroid
        n_j            its outward normal
        d_j            the mean centroid spacing

    Reference: Okuda, S. et al. (2018). Combining Turing and 3D vertex models reproduces autonomous multicellular morphogenesis of the tissue. Sci. Rep. 8:2386 (the shape-chemistry loop this closes)."""
    PARAM_UNITS = {}
    SYMBOLS = {'H_j': 'the curvature at cell j, per world length (positive where the surface bulges out)', '\\bar x_k': 'the mean centroid of the neighbours k', 'x_j': "the cell's centroid", 'n_j': 'its outward normal', 'd_j': 'the mean centroid spacing'}
    MECHANISM_TAGS = _ShapeToChemBase.MECHANISM_TAGS + ["curvature_sensing"]

    def _feature(self, pt, m, es, et, ef, nF):
        area, _, centroid, _ = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(es),
                                           torch.as_tensor(et), torch.as_tensor(ef), nF)
        centroid = centroid.numpy()
        nrm = np.zeros((nF, 3))                        # Newell normal per cell, outward
        for a, b, f in zip(es, et, ef):
            nrm[f] += np.cross(pt[a] - centroid[f], pt[b] - centroid[f])
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = nrm / np.maximum(ln, 1e-12)
        src, dst = _cell_adjacency(es, et, ef, nF)
        if not len(src):
            return None
        deg = np.bincount(src, minlength=nF).astype(float)
        nb = np.zeros((nF, 3))
        for d in range(3):
            nb[:, d] = np.bincount(src, weights=centroid[dst][:, d], minlength=nF)
        nb /= np.maximum(deg, 1)[:, None]
        delta = nb - centroid                                # umbrella vector
        # Divide by the NEIGHBOUR SPACING squared, not by |delta|^2. On a sphere the tangential
        # parts of the umbrella cancel, so |delta| is itself only ~L^2/2R -- dividing by it gives
        # 2R/L^2, which GROWS with radius. That reads as 1/R only if you hold the cell count fixed
        # so that L scales with R, which is exactly how the first version of this passed its own
        # test. With the spacing: delta.n = -L^2/2R, so H = 2 (delta.n) / L^2 = 1/R. Correct, and
        # now independent of how finely the sphere is meshed.
        sp = np.zeros(nF)
        np.add.at(sp, src, np.linalg.norm(centroid[dst] - centroid[src], axis=1))
        L = sp / np.maximum(deg, 1)
        return -2.0 * (delta * nrm).sum(1) / np.maximum(L ** 2, 1e-12)


@register_operator("cell_chem_from_shape", set="cell", kind="lateral", family="fields",
                   model="curvature_cells", title="make chemistry from the curvature of a sheet of cells",
                   equation=r"""$$H_j=-\frac{2\big(\overline{\mathbf x}_k-\mathbf x_j\big)\cdot\mathbf n_j}{d_j^{2}},\quad F_j=F_0\,(1+\beta\,\hat H_j)$$""")
class ShapeToChemCurvatureCells(ShapeToChemCurvature):
    r"""curvature for cells that are bodies of material points, not mesh faces: the same law on the cells' centroids and the point-contact graph, the normal from `centre`.
    cell -> cell: reads `pos` (aggregate_centroid), `edge_index` (cell_neighbours[point_contact]), writes the feedback F_j into the cell's drive.

        H_j=-\frac{2\big(\overline{\mathbf x}_k-\mathbf x_j\big)\cdot\mathbf n_j}{d_j^{2}},\quad F_j=F_0\,(1+\beta\,\hat H_j)
        H_j            the curvature at cell j
        \bar x_k       the neighbours' mean centroid
        x_j            the cell's centroid
        n_j            the outward normal from `centre`
        d_j            the mean centroid spacing
        F_j            the cell's drive
        F_0            its baseline
        \beta          the feedback gain
        \hat H_j       the median-centred curvature

    Reference: Okuda, S. et al. (2018). Combining Turing and 3D vertex models reproduces autonomous multicellular morphogenesis of the tissue. Sci. Rep. 8:2386; Plexus exp29 (2026-10-21)."""
    PARAM_UNITS = {'centre': 'length'}
    SYMBOLS = {'H_j': 'the curvature at cell j', '\\bar x_k': "the neighbours' mean centroid", 'x_j': "the cell's centroid", 'n_j': 'the outward normal from `centre`', 'd_j': 'the mean centroid spacing', 'F_j': "the cell's drive", 'F_0': 'its baseline', '\\beta': 'the feedback gain', '\\hat H_j': 'the median-centred curvature'}
    INPUTS = ["cell"]; READS = ["chem", "pos"]
    PARAM_ROLES = dict(_ShapeToChemBase.PARAM_ROLES, centre="point_the_normal_points_away_from")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        c = params.get("centre")
        self.centre = None if c is None else np.asarray([float(v) for v in c], dtype=np.float64)

    @staticmethod
    def curvature(X, edge_index, alive, centre=None):
        """H_j of cells at `X` [n, 3] over the directed neighbour pairs `edge_index` [2, E] (receiver first), `alive` [n] bool; None when
        no cell has a neighbour."""
        n = X.shape[0]
        src, dst = np.asarray(edge_index[0], dtype=np.int64), np.asarray(edge_index[1], dtype=np.int64)
        keep = alive[src] & alive[dst]
        src, dst = src[keep], dst[keep]
        if not len(src):
            return None
        deg = np.bincount(src, minlength=n).astype(np.float64)
        nb = np.zeros((n, 3))
        for d in range(3):
            nb[:, d] = np.bincount(src, weights=X[dst][:, d], minlength=n)
        nb /= np.maximum(deg, 1.0)[:, None]
        delta = nb - X
        sp = np.zeros(n)
        np.add.at(sp, src, np.linalg.norm(X[dst] - X[src], axis=1))
        L = sp / np.maximum(deg, 1.0)
        c0 = X[alive].mean(0) if centre is None else centre
        nrm = np.zeros((n, 3))
        for j in np.flatnonzero(deg > 0):
            pts = np.vstack([X[j][None], X[dst[src == j]]])
            u, s, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
            v = vt[-1] if vt.shape[0] >= 3 else np.cross(vt[0], np.array([0.0, 0.0, 1.0]))
            if float(v @ (X[j] - c0)) < 0.0:
                v = -v
            nrm[j] = v / max(float(np.linalg.norm(v)), 1e-12)
        H = -2.0 * (delta * nrm).sum(1) / np.maximum(L ** 2, 1e-12)
        return np.where(deg > 0, H, np.nan)

    def forward(self, H, mask=None):
        clvl = H.level(self.at)
        if "chem" not in clvl.state_schema:
            return {}
        chem = clvl.get("chem")
        if self.beta == 0.0:
            return {self.at: torch.zeros_like(chem)}
        ei = getattr(clvl, "edge_index", None)
        if ei is None or ei.numel() == 0:
            return {self.at: torch.zeros_like(chem)}
        X = clvl.get("pos").detach().cpu().numpy().astype(np.float64)
        alive = (clvl.occ.detach().cpu().numpy() > 0.5) if getattr(clvl, "occ", None) is not None else np.ones(len(X), bool)
        phi = self.curvature(X, ei.detach().cpu().numpy(), alive, self.centre)
        if phi is None:
            return {self.at: torch.zeros_like(chem)}
        return self._feed(clvl, chem, phi, alive.astype(np.float64), int(X.shape[0]))


@register_operator("cell_chem_from_shape", set="cell", kind="lateral", family="fields",
                   model="tension", title="make chemistry from cortical tension", equation=r"""$$T_j=2k_P\,(P_j-P^0_j)+\Gamma\,P_j+\Lambda$$""")
class ShapeToChemTension(_ShapeToChemBase):
    r"""the chemistry listens to cortical tension (mechanotransduction): the same quantity cell_mechanics charges for, needing the mechanical target P0.
    cell -> cell: reads the perimeters and targets, writes the tension into chem.

        T_j=2k_P\,(P_j-P^0_j)+\Gamma\,P_j+\Lambda
        T_j            cell j's cortical tension
        k_P            the perimeter elasticity
        P_j, P^0_j     the perimeter and its target
        \Gamma         the contractility
        \Lambda        the line tension

    Reference: Dupont, S. et al. (2011). Role of YAP/TAZ in mechanotransduction. Nature 474:179-183."""
    PARAM_UNITS = {}
    SYMBOLS = {'T_j': "cell j's cortical tension", 'k_P': 'the perimeter elasticity', 'P_j, P^0_j': 'the perimeter and its target', '\\Gamma': 'the contractility', '\\Lambda': 'the line tension'}
    MECHANISM_TAGS = _ShapeToChemBase.MECHANISM_TAGS + ["mechanotransduction", "tension_sensing"]

    def _feature(self, pt, m, es, et, ef, nF):
        if "P0" not in m:
            return None                                 # precondition absent -> no-op, not a guess
        _, perim, _, _ = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(es),
                                          torch.as_tensor(et), torch.as_tensor(ef), nF)
        P = perim.numpy()
        P0 = np.asarray(_np(m["P0"])[:nF], float)
        mech = m.get("mech", {}) or {}
        kP = float(mech.get("K_P", 1.0)); Gam = float(mech.get("Gam", mech.get("Gamma", 0.0)))
        Lam = float(mech.get("Lam", mech.get("Lambda", 0.0)))
        return 2.0 * kP * (P - P0) + Gam * P + Lam


@register_operator("cell_chem_from_shape", set="cell", kind="lateral", family="fields",
                   model="apical_area", title="make chemistry from apical area", equation=r"""$$\phi_j=\frac{A_j}{A^0_j}\ \ (\text{a target exists}),\qquad \phi_j=A_j\ \ \text{otherwise}$$""")
class ShapeToChemApicalArea(_ShapeToChemBase):
    r"""the chemistry listens to apical area, crowding and density sensing: the area over its target where a target exists, so a uniformly grown tissue reads as unstretched.
    cell -> cell: reads the areas (and targets), writes the feature into chem.

        \phi_j=\frac{A_j}{A^0_j}\ \ (\text{a target exists}),\qquad \phi_j=A_j\ \ \text{otherwise}
        \phi_j         the feature of cell j
        A_j            its apical area
        A^0_j          its target area

    Reference: Okuda, S. et al. (2018). Combining Turing and 3D vertex models reproduces autonomous multicellular morphogenesis of the tissue. Sci. Rep. 8:2386 (the shape-chemistry loop this closes)."""
    PARAM_UNITS = {}
    SYMBOLS = {'\\phi_j': 'the feature of cell j', 'A_j': 'its apical area', 'A^0_j': 'its target area'}
    MECHANISM_TAGS = _ShapeToChemBase.MECHANISM_TAGS + ["crowding_sensing", "density_sensing"]

    def _feature(self, pt, m, es, et, ef, nF):
        area, _, _, _ = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(es),
                                         torch.as_tensor(et), torch.as_tensor(ef), nF)
        a = area.numpy()
        if "A0" in m:                                   # strain, not size: a uniformly scaled
            A0 = np.asarray(_np(m["A0"])[:nF], float)      # tissue is not stretched
            return a / np.maximum(A0, 1e-12) - 1.0
        return a


@register_operator("cell_chem_from_shape", set="cell", kind="lateral", family="fields",
                   model="pressure", title="make chemistry from cell pressure", equation=r"""$$p_j=2k_V\,(V^0_j-v_j)$$""")
class ShapeToChemPressure(_ShapeToChemBase):
    r"""the chemistry listens to volume-elastic pressure, positive when a cell is below its target volume (squeezed); needs the mechanical target V0f.
    cell -> cell: reads the volumes and targets, writes the pressure into chem.

        p_j=2k_V\,(V^0_j-v_j)
        p_j            the pressure of cell j
        k_V            the volume elasticity
        V^0_j, v_j     the target and the current volume

    Reference: Okuda, S. et al. (2018). Combining Turing and 3D vertex models reproduces autonomous multicellular morphogenesis of the tissue. Sci. Rep. 8:2386 (the shape-chemistry loop this closes)."""
    PARAM_UNITS = {}
    SYMBOLS = {'p_j': 'the pressure of cell j', 'k_V': 'the volume elasticity', 'V^0_j, v_j': 'the target and the current volume'}
    MECHANISM_TAGS = _ShapeToChemBase.MECHANISM_TAGS + ["pressure_sensing", "compression_sensing"]

    def _feature(self, pt, m, es, et, ef, nF):
        if "V0f" not in m:
            return None
        _, _, _, vf = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(es),
                                       torch.as_tensor(et), torch.as_tensor(ef), nF,
                                       apex=wedge_apex(m, torch.as_tensor(pt)))
        V0 = np.asarray(_np(m["V0f"])[:nF], float)
        mech = m.get("mech", {}) or {}
        kV = float(mech.get("K_V", 1.0))
        return 2.0 * kV * (V0 - vf.numpy())


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    import os
    import sys
    from plexus.operators.vertex_ops import build_sphere_mesh
    fails = []

    def chk(c, what, extra=""):
        print(f"  [{'ok ' if c else 'FAIL'}] {what}{('  ' + extra) if extra else ''}")
        if not c:
            fails.append(what)

    print("CERTIFYING the shape features against shapes whose answer is known\n")

    # --- curvature must read ~1/R on a sphere, and must HALVE when the radius doubles
    op = ShapeToChemCurvature({"beta": 0.3})
    for R in (2.5, 5.0, 10.0):
        v, es, et, ef, nF = build_sphere_mesh(500, R, 0.0, 0)
        m = dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=v.shape[0])
        h = op._feature(v, m, es, et, ef, nF)
        print(f"        sphere R={R:<5} mean curvature {np.median(h):7.4f}   1/R = {1.0/R:.4f}")
        chk(abs(float(np.median(h)) - 1.0 / R) < 0.25 / R,
            f"sphere R={R} reads curvature ~1/R", f"{np.median(h):.4f} vs {1.0/R:.4f}")
        if R == 5.0:
            h5 = float(np.median(h))
    v, es, et, ef, nF = build_sphere_mesh(500, 10.0, 0.0, 0)
    m = dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=v.shape[0])
    h10 = float(np.median(op._feature(v, m, es, et, ef, nF)))
    chk(0.35 < h10 / max(h5, 1e-9) < 0.65, "curvature halves when the radius doubles",
        f"ratio {h10/max(h5,1e-9):.3f}")

    # --- curvature must be POSITIVE on a bump and NEGATIVE in a dimple
    v, es, et, ef, nF = build_sphere_mesh(600, 5.0, 0.0, 0)
    m = dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=v.shape[0])
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    cap = u[:, 2] > 0.86
    # A LOCALIZED GAUSSIAN DOME, not a scaled cap. Scaling a spherical cap outward leaves it on a
    # sphere of LARGER radius, i.e. genuinely flatter -- the first version of this test demanded a
    # positive curvature from a shape that is objectively less curved, and the operator was right
    # to disagree with it.
    for tag, amp in (("bump", +1.2), ("dimple", -1.2)):
        g = np.exp(-((u[:, 2] - 1.0) ** 2) / (2 * 0.05 ** 2))
        w = v + amp * g[:, None] * u
        h = op._feature(w, m, es, et, ef, nF)
        _, _, centroid, _ = face_geometry_3d(torch.as_tensor(w), torch.as_tensor(es),
                                        torch.as_tensor(et), torch.as_tensor(ef), nF)
        top = centroid.numpy()[:, 2] > 0.90 * np.linalg.norm(centroid.numpy(), axis=1)
        d = float(np.median(h[top]) - np.median(h[~top]))
        print(f"        {tag:7} curvature at the feature minus elsewhere: {d:+.4f}")
        chk((d > 0) if amp > 0 else (d < 0), f"a {tag} reads the right SIGN")

    # --- standardisation must make beta mean the same thing whatever the units
    for scale in (1.0, 1000.0):
        w = _standardise(np.arange(200.0) * scale, np.ones(200))
        print(f"        feature scaled x{scale:<8g} -> standardised spread {w.std():.4f}")
    a = _standardise(np.arange(200.0), np.ones(200))
    b = _standardise(np.arange(200.0) * 1000.0, np.ones(200))
    chk(np.allclose(a, b, atol=1e-9), "standardisation is invariant to the feature's units")
    chk(np.allclose(_standardise(np.full(50, 7.0), np.ones(50)), 0.0),
        "a UNIFORM feature carries no signal (all zeros, not noise)")

    # --- a single spike must not drive the feed
    x = np.r_[np.random.default_rng(0).normal(0, 1, 199), [1e6]]
    chk(abs(_standardise(x, np.ones(200))).max() <= 4.0 + 1e-9,
        "one extreme cell is clipped, not allowed to set the scale")

    # ----------------------------------------------------------------- everything above is CHECK 0
    # It certifies the ARITHMETIC INSIDE the operator, and every line of it passes -- which is
    # exactly why it is not enough on its own. An operator whose curvature reads 1/R on spheres of
    # known radius, whose bump is positive and whose dimple negative, can still never reach the
    # state at all, and every check above would still be green. The three below test `forward`.
    import torch

    print("\n  CHECK 0b -- the null, actually executed rather than asserted:")
    v, es, et, ef, nF = build_sphere_mesh(500, 5.0, 0.0, 0)
    m = dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=v.shape[0])

    class _Lvl:                                    # the smallest thing `forward` will accept
        def __init__(self, st, sch, mesh=None):
            self.state, self.state_schema, self._mesh, self.occ = st, sch, mesh, None

        def get(self, k):
            a, b = self.state_schema[k]
            return self.state[:, a:b]

    class _H:
        def __init__(self, d):
            self.levels = d

        def level(self, n):
            return self.levels[n]

    chem = torch.zeros(nF, 2)
    chem[:, 0] = 0.3
    chem[:, 1] = 0.6                                # u != 1, or (1 - u) zeroes the emission anyway
    H = _H({"cell": _Lvl(chem, {"chem": (0, 2)}),
            "vertex": _Lvl(torch.as_tensor(v, dtype=torch.float32), {"pos": (0, 3)}, m)})
    out0 = ShapeToChemCurvature({"beta": 0.0}).forward(H)["cell"]
    chk(float(out0.abs().max()) == 0.0, "beta = 0 emits exactly zero (executed)",
        f"max |emission| {float(out0.abs().max()):.3e}")

    print("\n  CHECK 1 -- is the emission set by BETA, or by the clamp?")
    # I expected zero here, on the reasoning that a sphere has uniform curvature so `_standardise`
    # returns zero. The operator disagreed and it was right: a FIBONACCI sphere is discrete, its
    # per-cell curvature carries mesh noise, and the MAD-scaling turns that noise into a full-range
    # standardised field. So the operator fires on a sphere -- driven by discretisation, not shape.
    #
    # The number that matters is not that it fires but WHAT SETS ITS SIZE. dF is clamped into
    # [0 - F0, F_CEIL - F0], so the largest correction the operator can ever emit is
    # (F_CEIL - F0) * (1 - u), independent of beta and of geometry. If the measured maximum equals
    # that bound, beta is not a strength -- it only chooses WHICH cells sit at the ceiling.
    outs = {b: ShapeToChemCurvature({"beta": b}).forward(H)["cell"] for b in (-2.0, -4.0)}
    bound = (F_CEIL - 0.055) * (1.0 - 0.6)
    for b, o in outs.items():
        print(f"        beta={b:<6} max |emission| {float(o.abs().max()):.4e}"
              f"   clamp bound {bound:.4e}")
    chk(all(abs(float(o.abs().max()) - bound) < 1e-6 for o in outs.values()),
        "the emission is PINNED TO THE CLAMP CEILING, so beta sets no magnitude",
        f"{float(outs[-2.0].abs().max()):.4e} vs bound {bound:.4e}")

    print("\n  CHECK 1b -- and on a shape that HAS curvature variation?")
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    g = np.exp(-((u[:, 2] - 1.0) ** 2) / (2 * 0.05 ** 2))
    w = (v + 1.2 * g[:, None] * u).astype(np.float32)
    H.levels["vertex"] = _Lvl(torch.as_tensor(w), {"pos": (0, 3)}, m)
    e2 = {b: float(ShapeToChemCurvature({"beta": b}).forward(H)["cell"].abs().max())
          for b in (-2.0, -4.0)}
    for b, mx in e2.items():
        print(f"        beta={b:<6} max |emission| {mx:.3e}")
    chk(all(x > 0 for x in e2.values()), "a bumped sphere emits a nonzero feed correction")
    # THE SATURATION, measured rather than reasoned about. F = clamp(F0 (1 + beta phihat), 0,
    # F_CEIL) with phihat clipped at +/-4: at beta = -2 the bracket already leaves [0, F_CEIL] on
    # most cells, so DOUBLING beta must not double the emission. If it does, the clamp is not
    # binding and this comment is wrong.
    ratio = e2[-4.0] / max(e2[-2.0], 1e-30)
    print(f"        doubling beta multiplies the emission by {ratio:.3f} (2.000 = unsaturated)")
    chk(ratio < 1.6, "the F clamp SATURATES -- beta is not proportional",
        f"ratio {ratio:.3f}")

    print("\n  " + ("ALL SHAPE FEATURES CERTIFIED" if not fails else f"{len(fails)} FAILURES"))
    raise SystemExit(1 if fails else 0)


# `_np` is defined once, above, and shared by both the shape-to-chemistry and the shape-probe
# operators.


class _ShapeProbeBase(Lateral):
    """Compute one scalar per cell and publish it on the mesh under `field`. No state is touched."""
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False
    INPUTS = ["cell", "vertex"]; OUTPUTS = []; READS = ["pos"]; WRITES = []
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["measurement", "cell_shape", "publishes_field"]
    REFERENCE = ("Bi, D. et al. (2015). Nat. Phys. 11:1074-1079 (the shape index as the tissue's "
                 "own order parameter, rigid below 3.81 and fluid above).")
    PARAM_ROLES = {"field": "published_field_name"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.vat = params.get("vertex_set", "vertex")
        # THE NAME IS THE WIRING. Whatever this is called is what a Die operator asks for.
        self.field = str(params.get("field", "elong"))

    def _measure(self, pos, m, es, et, ef, nF):
        raise NotImplementedError

    def forward(self, H, mask=None):
        vlvl = H.level(self.vat)
        m = getattr(vlvl, "_mesh", None)
        if m is None:
            return {}
        nF = int(m["nF"])
        es = _np(m["E_srce"]); et = _np(m["E_trgt"]); ef = _np(m["E_face"])
        live = ef < nF
        pos = _np(vlvl.get("pos"))[:int(m["Nv"])].astype(np.float64)
        val = self._measure(pos, m, es[live], et[live], ef[live], nF)
        if val is None:
            # A PRECONDITION IS ABSENT, so nothing is published -- rather than publishing zeros,
            # which a Die reading `field_high` would score as "no cell is elongated" and a Die
            # reading `field_low` would score as "every cell is". An absent field is undefined;
            # zero is a measurement. This substrate has paid for that distinction twice.
            m.pop(self.field, None)
            return {}
        v = np.asarray(val, float)
        v[~np.isfinite(v)] = np.nan          # a degenerate cell is UNMEASURED, not zero
        m[self.field] = v
        return {}


@register_operator("cell_shape_probe", set="cell", kind="lateral", family="hierarchy",
                   model="shape_index", title="measure the cell shape index", probe=True,
                   equation=r"""$$q_j=\frac{P_j}{\sqrt{A_j}}$$""")
class ShapeIndexProbe(_ShapeProbeBase):
    r"""the dimensionless shape index, what the vertex model minimises towards p0 and the tissue's own order parameter for rigid against fluid (about 3.72 for a regular hexagon, 3.81 the transition).
    cell -> cell: reads perimeter and area, writes the probe.

        q_j=\frac{P_j}{\sqrt{A_j}}
        q_j            the shape index of cell j
        P_j            its perimeter
        A_j            its area

    Reference: Bi, D., Lopez, J. H., Schwarz, J. M. & Manning, M. L. (2015). A density-independent rigidity transition in biological tissues. Nat. Phys. 11:1074-1079."""
    PARAM_UNITS = {}
    SYMBOLS = {'q_j': 'the shape index of cell j', 'P_j': 'its perimeter', 'A_j': 'its area'}

    def _measure(self, pos, m, es, et, ef, nF):
        pt = torch.as_tensor(pos)
        area, perim, _cen, _vf = face_geometry_3d(
            pt, torch.as_tensor(es), torch.as_tensor(et), torch.as_tensor(ef), nF, apex=wedge_apex(m, pt))
        a = _np(area)[:nF]; p = _np(perim)[:nF]
        out = np.full(nF, np.nan)
        ok = a > 1e-12
        out[ok] = p[ok] / np.sqrt(a[ok])
        return out


@register_operator("cell_shape_probe", set="cell", kind="lateral", family="hierarchy",
                   model="aspect", title="measure cell elongation", probe=True, equation=r"""$$\mathrm{aspect}_j=\sqrt{\lambda_1/\lambda_2},\qquad \lambda_1\ge\lambda_2\ge\lambda_3$$""")
class AspectProbe(_ShapeProbeBase):
    r"""thin and elongated as a number: the aspect ratio of the cell's own vertex ring from the first two eigenvalues of its covariance (the third is the sheet's thickness).
    cell -> cell: reads the ring's vertex positions, writes the probe.

        \mathrm{aspect}_j=\sqrt{\lambda_1/\lambda_2},\qquad \lambda_1\ge\lambda_2\ge\lambda_3
        aspect_j       the aspect ratio of cell j, 1 for a circular cell
        \lambda_1, \lambda_2, \lambda_3 the eigenvalues of the covariance of the ring's vertex positions

    Reference: none: a shape descriptor, not a mechanism. Plexus (this work)."""
    PARAM_UNITS = {}
    SYMBOLS = {'aspect_j': 'the aspect ratio of cell j, 1 for a circular cell', '\\lambda_1, \\lambda_2, \\lambda_3': "the eigenvalues of the covariance of the ring's vertex positions"}

    def _measure(self, pos, m, es, et, ef, nF):
        rings = rings_from_flat_3d(es, et, ef, nF)
        out = np.full(nF, np.nan)
        for f, r in enumerate(rings):
            if r is None or len(r) < 3:
                continue
            p = pos[np.asarray(r, int)]
            c = p.mean(0)
            w = np.linalg.eigvalsh(np.cov((p - c).T) + 1e-15 * np.eye(3))[::-1]
            s0, s1 = np.sqrt(max(w[0], 0.0)), np.sqrt(max(w[1], 0.0))
            if s1 > 1e-9:
                out[f] = s0 / s1
        return out


@register_operator("cell_grow", model="mpm", set="cell", kind="lateral", family="population",
                   title="grow the mass of cells made of material points",
                   equation=r"""$$\frac{dm_c}{dt}=d\,r_c\,\rho\,m_c,\qquad m_c(0)=N_c(0)$$""")
class CellGrowMPM(Lateral):
    r"""the MPM twin of cell_grow: the same law on the cell's mass, held as a number (in points) in a cell block, with no points added; cell_populate[mpm] adds them.
    cell -> cell: reads the point set once to seed `mass_block`, then returns dm/dt on the block.

        \frac{dm_c}{dt}=d\,r_c\,\rho\,m_c,\qquad m_c(0)=N_c(0)
        m_c            the mass of cell c, in points
        d              `dims`, the exponent of volume in the linear scale
        r_c            `rate_block` of the cell when above 0, else `rate`
        \rho           `rho`, the basal growth
        N_c(0)         the live point count at the seed

    Reference: Plexus exp33 (this work, 2026-10-25); the law: Okuda, S. et al. (2018). Combining Turing and 3D vertex models reproduces autonomous multicellular morphogenesis of the tissue. Sci. Rep. 8:2386."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"Grows the mass of each cell, held as a number (in points) in a cell block: $\dfrac{dm_c}{dt}=d\,r_c\,\rho\,m_c$ with "
        r"$m_c(0)=N_c(0)$, the point count at the start. Here $r_c$ is the cell's own `rate_block` value when above 0, else "
        r"`rate`, $d$ = `dims` the exponent of volume in the linear scale and $\rho$ the operator's `rho` factor. No points are "
        r"added here: cell_populate adds the points that carry the mass."
    )
    PARAM_UNITS = {'after_frame': 'frames', 'dims': 'dimensionless', 'rate': 'rate', 'rho': 'dimensionless'}
    SYMBOLS = {'m_c': 'the mass of cell c, in points', 'd': '`dims`, the exponent of volume in the linear scale', 'r_c': '`rate_block` of the cell when above 0, else `rate`', '\\rho': '`rho`, the basal growth', 'N_c(0)': 'the live point count at the seed'}
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["points", "rate"]
    PARAM_ROLES = {"points": "the_material_point_set", "rate": "linear_growth_rate_per_time_unit", "dims": "exponent_of_mass_in_the_linear_scale",
                   "rho": "baseline_growth_factor", "mass_block": "cell_block_holding_the_mass_in_points",
                   "rate_block": "cell_block_with_the_linear_rate_of_each_cell", "after_frame": "no_growth_before_this_frame"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.points = str(params["points"]); self.rate = float(params["rate"])
        self.dims = float(params.get("dims", 2.0)); self.rho = float(params.get("rho", 1.0))
        self.mass_block = str(params.get("mass_block", "mass")); self.rate_block = params.get("rate_block")
        self.after_frame = int(params.get("after_frame", 0))               # no growth before this frame (the onset the vertex twin's `cell_grow` also has)

    def forward(self, H, mask=None):
        C, P = H.level(self.at), H.level(self.points)
        sch = C.state_schema
        if self.mass_block not in sch:
            raise ValueError(f"cell_grow[mpm]: {self.at!r} needs a width-1 state block {self.mass_block!r}")
        mc = sch[self.mass_block][0]
        live = C.occ > 0.5
        m0 = C.state[:, mc]
        m, jump = m0, torch.zeros_like(m0)
        if bool(((m0 <= 0) & live).any()):                         # the seed: a cell's mass is its point count, once -- returned as a jump in the delta
            par = P.parent.reshape(-1).long()
            cnt = torch.zeros(C.state.shape[0], dtype=C.state.dtype, device=C.state.device).index_add_(0, par.clamp(0, C.state.shape[0] - 1), (P.occ > 0.5).to(C.state.dtype))
            m = torch.where((m0 <= 0) & live, cnt, m0)
            jump = (m - m0) / max(float(getattr(H, "dt", 1.0) or 1.0), 1e-12)
        r = torch.full_like(m, self.rate)
        if self.rate_block is not None and self.rate_block in sch:
            rc = C.state[:, sch[self.rate_block][0]]
            r = torch.where(rc > 0, rc, r)
        on = int(getattr(H, "frame", 0) or 0) >= self.after_frame
        d = torch.where(live, self.dims * r * self.rho * m, torch.zeros_like(m)) * (1.0 if on else 0.0) + jump
        return {(self.at, self.mass_block): d[:, None]}


@register_operator("cell_grow", model="mpm_adhesion", set="cell", kind="lateral", family="population",
                   title="grow cells of material points only while they adhere to other cells",
                   equation=r"""$$\frac{dA_c}{dt}=\frac{s_c-A_c}{\tau},\qquad s_c=\min\Big(1,\frac{1}{S_0}\sum_{j\ni c}S_j\Big),\qquad \frac{dm_c}{dt}=d\,r_c\,\rho\,m_c\;\Theta(A_c-A^{*})\,\Theta(A^{\max}-A_c)$$""")
class CellGrowMPMAdhesion(CellGrowMPM):
    r"""cell_grow[mpm] gated by adhesion: a cell's mass grows by the same law only while its ADHESION LEVEL is at or above a threshold; the level follows, with a time constant, the share of the cell's surface in contact with other cells.
    cell -> cell: reads the junction table of contact_finder (`contact_gate`: the two cells `ca`, `cb` and the contact `area` of every row); writes dm/dt on `mass_block` and dA/dt on channel 0 of `adh_block`.

        \frac{dA_c}{dt}=\frac{s_c-A_c}{\tau},\qquad s_c=\min\Big(1,\frac{1}{S_0}\sum_{j\ni c}S_j\Big),\qquad \frac{dm_c}{dt}=d\,r_c\,\rho\,m_c\;\Theta(A_c-A^{*})
        A_c            the adhesion level of cell c, 0 (touching nothing) to 1 (its whole surface in contact), channel 0 of `adh_block`
        s_c            the contact share now: the contact areas S_j of the junction rows j that name c, over S_0
        S_0            `gate_area`, the contact area of a cell whose whole surface touches others (4.84 length^2, the surface of a ball of volume 1.0)
        \tau           `tau`, the time constant of the adhesion level (the maturation of a junction), time units
        A^{*}          `threshold`, the adhesion level above which the cell grows
        A^{max}        `ceiling`, the adhesion level at or above which it stops again (contact inhibition of proliferation; none = no ceiling)
        \Theta         the step function, 1 at or above 0, else 0 (the ceiling's: 1 strictly above 0)
        m_c, d, r_c, \rho  as cell_grow[mpm]

    Not a `rate_block` of cell_grow[mpm]: that block falls back to `rate` where it is 0, so it cannot switch growth off. Not a new operator: the growth law and its mass block are
    cell_grow[mpm]'s, only the gate is added. The share is mpm_ops.contact_share, the same reading of the junction table as active_force[persistent_walk]'s `contact_gate`.

    Reference: Plexus exp06 (this work, 2026-11-14); contact-dependent proliferation of aggregating cells, e.g. Tzanakakis, E. S. et al. (2001). Cell Transplant. 10:329-342 (hepatocyte spheroids)."""
    EXPLAIN = (
        r"Grows each cell's mass by cell_grow[mpm]'s law, $\dfrac{dm_c}{dt}=d\,r_c\,\rho\,m_c$, but only while the cell's adhesion "
        r"level $A_c$ is at or above the `threshold` $A^*$. The level relaxes with time constant $\tau$ toward $s_c$, the share of "
        r"the cell's surface touching other cells (contact areas of its junction rows over `gate_area`, clipped to 1): "
        r"$\dfrac{dA_c}{dt}=\dfrac{s_c-A_c}{\tau}$. A cell alone keeps $A_c$ near 0 and does not grow; a cell inside a clump grows."
    )
    PARAM_UNITS = dict(CellGrowMPM.PARAM_UNITS, tau='time', threshold='fraction', gate_area='area')
    SYMBOLS = {'A_c': 'the adhesion level of cell c, 0 (touching nothing) to 1 (whole surface in contact)', 's_c': "the cell's contact share now",
               'S_0': '`gate_area`, the contact area of a fully touching cell', '\\tau': '`tau`, the time constant of the adhesion level',
               'A^{*}': '`threshold`, the adhesion level above which the cell grows', '\\Theta': 'the step function',
               'm_c, d, r_c, \\rho': 'as cell_grow[mpm]'}
    REQUIRES_PARAMS = ["points", "rate", "contact_gate", "threshold"]
    PARAM_ROLES = dict(CellGrowMPM.PARAM_ROLES, contact_gate="junction_set_whose_contact_area_is_the_adhesion", gate_area="contact_area_of_a_fully_touching_cell",
                       tau="time_constant_of_the_adhesion_level", threshold="adhesion_level_above_which_the_cell_grows", adh_block="cell_block_channel_0_holds_the_adhesion_level",
                       type_rate="growth_rate_of_the_cells_of_a_named_type", ceiling="adhesion_level_at_or_above_which_growth_stops_contact_inhibition")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.contact_gate = str(params["contact_gate"])
        self.gate_area = float(params.get("gate_area", 4.84))
        self.tau = float(params.get("tau", 10.0))
        self.threshold = float(params["threshold"])
        self.adh_block = str(params.get("adh_block", "chem"))
        # `type_rate: {B: 0.0046}` (default none, exp06 phase C 2026-11-14): the rate r of the cells of a named TYPE (a fate `cell_fate` gives from a signal), the other types keep `rate`
        self.type_rate = {str(k): float(v) for k, v in (params.get("type_rate") or {}).items()}
        # `ceiling` (default none, exp06 2026-11-14, Cedric chose it for the 1,000-min runs): CONTACT INHIBITION OF PROLIFERATION -- a cell whose adhesion level is at or above it
        # (surrounded on most of its surface: a spheroid's core sits at 0.71-0.77, its outer shell at 0.57-0.65, it05) stops growing, so only the outer layer proliferates
        self.ceiling = None if params.get("ceiling") is None else float(params["ceiling"])
        # `dying_block` (default "dying", used when the set declares it): a cell cell_die marks (block > 0.5) does not grow, its mass FOLLOWS its point count down, so that
        # cell_populate does not refill the points the death sheds (exp06 2026-11-14, cell death as a sculptor)
        self.dying_block = str(params.get("dying_block", "dying"))
        # `stop_when_full` (default true, exp06 2026-11-14): no growth once the cell set has no free slot -- cell_divide finds none and the cells would only swell (TRAPS T-40)
        self.stop_when_full = bool(params.get("stop_when_full", True))

    def forward(self, H, mask=None):
        from plexus.operators.mpm_ops import contact_share
        out = super().forward(H, mask)
        C = H.level(self.at)
        sch = C.state_schema
        if self.adh_block not in sch:
            raise ValueError(f"cell_grow[mpm_adhesion]: {self.at!r} needs a state block {self.adh_block!r} (channel 0 = the adhesion level)")
        nc = int(C.state.shape[0])
        live = C.occ > 0.5
        m0 = C.state[:, sch[self.mass_block][0]]
        key = (self.at, self.mass_block)
        if bool(((m0 <= 0) & live).any()):                         # the seed call of cell_grow[mpm]: keep only its jump (the mass set to the point count), no growth yet
            dt_h = max(float(getattr(H, "dt", 1.0) or 1.0), 1e-12)
            P = H.level(self.points)
            par = P.parent.reshape(-1).long()
            cnt = torch.zeros(nc, dtype=C.state.dtype, device=C.state.device).index_add_(0, par.clamp(0, nc - 1), (P.occ > 0.5).to(C.state.dtype))
            d = torch.where((m0 <= 0) & live, (cnt - m0) / dt_h, torch.zeros_like(m0))
        else:
            a = C.state[:, sch[self.adh_block][0]]
            gate = a >= self.threshold
            if self.ceiling is not None:
                gate = gate & (a < self.ceiling)
            d = out[key][:, 0] * gate.to(m0.dtype)
            if self.stop_when_full and bool(live.all()):            # every slot of the set holds a live cell: no daughter can be placed (T-40)
                d = torch.zeros_like(d)
            if self.type_rate:                                     # the growth law is linear in r: scale each typed cell's rate
                names = list(getattr(C, "type_names", []) or [])
                f = torch.ones_like(m0)
                nt = C.node_type.to(m0.device).long()
                for nm, r_t in self.type_rate.items():
                    if nm in names:
                        f = torch.where(nt == names.index(nm), torch.full_like(f, r_t / max(self.rate, 1e-30)), f)
                d = d * f
        if self.dying_block in sch and not bool(((m0 <= 0) & live).any()):
            dy = (C.state[:, sch[self.dying_block][0]] > 0.5) & live
            if bool(dy.any()):
                P = H.level(self.points)
                par = P.parent.reshape(-1).long()
                cnt = torch.zeros(nc, dtype=C.state.dtype, device=C.state.device).index_add_(0, par.clamp(0, nc - 1), (P.occ > 0.5).to(C.state.dtype))
                d = torch.where(dy, (cnt - m0) / max(float(getattr(H, "dt", 1.0) or 1.0), 1e-12), d)
        J = H.level(self.contact_gate)
        if J.state.shape[0] > 0:
            ca = J.get("ca")[:, 0].round().long().clamp(0, nc - 1)
            cb = J.get("cb")[:, 0].round().long().clamp(0, nc - 1)
            s = contact_share(ca, cb, J.get("area")[:, 0].to(C.state.dtype), J.occ > 0.5, nc, self.gate_area)
        else:
            s = torch.zeros(nc, dtype=C.state.dtype, device=C.state.device)
        a = C.state[:, sch[self.adh_block][0]]
        da = torch.where(live, (s - a) / max(self.tau, 1e-12), torch.zeros_like(a))
        w = sch[self.adh_block][1] - sch[self.adh_block][0]
        dA = torch.zeros(nc, w, dtype=C.state.dtype, device=C.state.device)
        dA[:, 0] = da
        return {key: d[:, None], (self.at, self.adh_block): dA}


@register_operator("cell_grow", model="mpm_activator", set="cell", kind="lateral", family="population",
                   title="grow cells of material points where their activator is high",
                   equation=r"""$$\frac{dm_c}{dt}=d\,r\,\rho\,m_c\,\Big(\beta+g\,\frac{a_c^{\,n}}{a_{sw}^{\,n}+a_c^{\,n}}\Big)$$""")
class CellGrowMPMActivator(CellGrowMPM):
    r"""cell_grow[mpm] with Okuda's mitogen law: a cell's growth rate is a Hill function of its own activator, so the cells of an activator spot proliferate and the rest barely.
    cell -> cell: reads channel `channel` of the cell block `chem_block` (the activator of a Turing system); writes dm/dt on `mass_block`.

        \frac{dm_c}{dt}=d\,r\,\rho\,m_c\,\Big(\beta+g\,\frac{a_c^{\,n}}{a_{sw}^{\,n}+a_c^{\,n}}\Big)
        m_c            the mass of cell c, in points (cell_grow[mpm])
        a_c            the cell's activator, channel `channel` of `chem_block` (clipped at 0)
        a_{sw}         `a_sw`, the activator level of half-maximal growth, in the activator's own units
        n              `hill`, the steepness of the switch (Okuda's alpha; 10 is near a step)
        \beta          `rho_lam`, the growth factor of a cell with no activator (0: only active cells grow)
        g              `lam_gain`, the extra growth factor of a fully active cell (a fully active cell grows at beta + g times r)
        d, r, \rho     as cell_grow[mpm]

    Not a `rate_block` of cell_grow[mpm]: that block is a stored per-cell rate that falls back to `rate` where it is 0, so it can neither follow the activator every frame nor switch growth
    off. Not cell_grow[mpm_adhesion]'s `type_rate` through cell_fate: that is a hard threshold, a pale cell just over it grows at the full rate (the Turing_vertex prototype's wave-4
    flaw); the Hill switch is graded. The parameter names are the prototype's `growth` operator's (prototype/Turing_vertex/coupled_ops.py), the law Okuda's Eq. 8. Like
    cell_grow[mpm_adhesion], no growth once every slot of the cell set is live (`stop_when_full`, TRAPS T-40).

    Reference: Okuda, S., Miura, T., Inoue, Y., Adachi, T. & Eiraku, M. (2018). Combining Turing and 3D vertex models reproduces autonomous multicellular morphogenesis of the tissue.
    Sci. Rep. 8:2386 (Eq. 8, the activator as a mitogen); Plexus exp08 (this work, 2026-11-14)."""
    EXPLAIN = (
        r"Grows each cell's mass by cell_grow[mpm]'s law with a rate set by the cell's own activator $a_c$: "
        r"$\dfrac{dm_c}{dt}=d\,r\,\rho\,m_c\,\big(\beta+g\,a_c^n/(a_{sw}^n+a_c^n)\big)$. A cell with no activator grows at $\beta$ times "
        r"the base rate $r$, a fully active one at $\beta+g$ times it; $a_{sw}$ (`a_sw`) is the activator of half the extra growth and "
        r"$n$ (`hill`) how sharp the switch is. Okuda et al. 2018's mitogen: the cells of a Turing spot proliferate."
    )
    PARAM_UNITS = dict(CellGrowMPM.PARAM_UNITS, a_sw='concentration', hill='dimensionless', rho_lam='dimensionless', lam_gain='dimensionless', channel='index')
    SYMBOLS = {'m_c': 'the mass of cell c, in points', 'a_c': "the cell's activator", 'a_{sw}': '`a_sw`, the activator of half-maximal extra growth',
               'n': '`hill`, the steepness of the switch', '\\beta': '`rho_lam`, the growth factor with no activator', 'g': '`lam_gain`, the extra factor of a fully active cell',
               'd, r, \\rho': 'as cell_grow[mpm]'}
    REQUIRES_PARAMS = ["points", "rate", "a_sw"]
    PARAM_ROLES = dict(CellGrowMPM.PARAM_ROLES, chem_block="cell_block_holding_the_activator", channel="channel_of_the_activator", a_sw="activator_of_half_maximal_extra_growth",
                       hill="steepness_of_the_switch", rho_lam="growth_factor_without_activator", lam_gain="extra_growth_factor_of_a_fully_active_cell",
                       stop_when_full="no_growth_once_every_cell_slot_is_live")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.chem_block = str(params.get("chem_block", "chem")); self.channel = int(params.get("channel", 0))
        self.a_sw = float(params["a_sw"]); self.hill = float(params.get("hill", 4.0))
        self.rho_lam = float(params.get("rho_lam", 0.0)); self.lam_gain = float(params.get("lam_gain", 1.0))
        self.stop_when_full = bool(params.get("stop_when_full", True))

    def forward(self, H, mask=None):
        out = super().forward(H, mask)
        C = H.level(self.at)
        sch = C.state_schema
        if self.chem_block not in sch:
            raise ValueError(f"cell_grow[mpm_activator]: {self.at!r} needs a state block {self.chem_block!r} (channel {self.channel} = the activator)")
        key = (self.at, self.mass_block)
        m0 = C.state[:, sch[self.mass_block][0]]
        live = C.occ > 0.5
        if bool(((m0 <= 0) & live).any()):                         # the seed call of cell_grow[mpm]: only its jump (the mass set to the point count), no growth yet (as cell_grow[mpm_adhesion])
            nc = int(C.state.shape[0])
            P = H.level(self.points)
            par = P.parent.reshape(-1).long()
            cnt = torch.zeros(nc, dtype=C.state.dtype, device=C.state.device).index_add_(0, par.clamp(0, nc - 1), (P.occ > 0.5).to(C.state.dtype))
            d = torch.where((m0 <= 0) & live, (cnt - m0) / max(float(getattr(H, "dt", 1.0) or 1.0), 1e-12), torch.zeros_like(m0))
            return {key: d[:, None]}
        # the Hill factor as 1 / (1 + (a_sw / a)^n), not a^n / (a_sw^n + a^n): at steepness 10 a^n overflows float32 above a = 7,000 and inf / inf = NaN went into the
        # mass (exp08 r09: a cell shed from the shell lost its neighbours' inhibitor, its activator ran away, and cell_populate stopped the run at frame ~670 on
        # "cannot convert float NaN to integer"). A non-finite activator reads as saturated
        a = torch.nan_to_num(C.state[:, sch[self.chem_block][0] + self.channel], nan=float("inf"), posinf=float("inf"), neginf=0.0).clamp(min=0.0)
        f = self.rho_lam + self.lam_gain / (1.0 + (self.a_sw / a.clamp(min=1e-30)) ** self.hill)
        d = out[key][:, 0] * f
        if self.stop_when_full and bool(live.all()):                # every slot of the set holds a live cell: no daughter can be placed (T-40)
            d = torch.zeros_like(d)
        return {key: d[:, None]}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="survival_factor",
                   title="a survival factor secreted by cells with free surface, spread through contacts",
                   equation=r"""$$\frac{ds_i}{dt}=r\big(p\,\Theta(A^{s}-A_i)-s_i\big)+D\sum_{j\sim i}\frac{s_j-s_i}{\deg(i)}$$""")
class CellReactSurvivalFactor(Lateral):
    r"""a SURVIVAL FACTOR: a cell with free surface (adhesion level below `a_src`) secretes it, every cell loses it at rate r, and it is exchanged with the touching cells; a cell deep in a
    tissue receives it only through its neighbours, so beyond a few cells it starves (cell_die[mpm_chem_low] with `block: surv` removes it): lumen formation by the death of the inner cells.
    cell -[cell adjacency]-> cell: reads its own block `block` (the factor), the adhesion block `adh_block` and edge_index; writes d(block)/dt.

        \frac{ds_i}{dt}=r\big(p\,\Theta(A^{s}-A_i)-s_i\big)+D\sum_{j\sim i}\frac{s_j-s_i}{\deg(i)}
        s_i            the factor in cell i (block `block`, channel 0)
        r              `rate`, its turnover, per time
        p              `p`, the level a secreting cell holds alone
        A_i            the cell's adhesion level (cell_grow[mpm_adhesion]'s, block `adh_block`)
        A^{s}          `a_src`, the adhesion level below which a cell secretes (a cell with free surface)
        D              `d`, the exchange rate with the touching cells, per time (none for a cell with no neighbour)
        \Theta        the step function

    Not cell_chem_react[gierer_meinhardt] or cell_chem_diffuse: those act on the `chem` block, whose width the Gierer-Meinhardt pair fixes at 2; the factor lives in its own block and
    carries its own exchange so that it can run beside them. Penetration depth into a tissue sqrt(D / r) cells.

    Reference: Coucouvanis, E. & Martin, G. R. (1995). Signals for death and survival: a two-step mechanism for cavitation in the vertebrate embryo. Cell 83:279-287; Debnath, J. et al.
    (2002). Cell 111:29-40 (luminal apoptosis in mammary acini); the rule is Plexus's (exp06, 2026-11-14)."""
    PARAM_UNITS = {'rate': 'rate', 'p': 'dimensionless', 'a_src': 'fraction', 'd': 'rate'}
    SYMBOLS = {'s_i': 'the factor in cell i', 'r': '`rate`, its turnover', 'p': '`p`, the level a secreting cell holds alone', 'A_i': "the cell's adhesion level",
               'A^{s}': '`a_src`, the adhesion level below which a cell secretes', 'D': '`d`, the exchange rate with the touching cells', '\\Theta': 'the step function'}
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; DIFFERENTIABLE = False
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["adh", "edge_index"]; WRITES = ["surv"]
    MECHANISM_TAGS = ["survival_factor", "cavitation", "lumen", "juxtacrine"]
    PARAM_ROLES = {"rate": "turnover_rate", "p": "secreted_level", "a_src": "adhesion_level_below_which_a_cell_secretes", "d": "exchange_rate_with_touching_cells",
                   "block": "cell_block_of_the_factor", "adh_block": "cell_block_of_the_adhesion_level"}
    REFERENCE = "Coucouvanis, E. & Martin, G. R. (1995). Cell 83:279-287; Debnath, J. et al. (2002). Cell 111:29-40."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.rate = float(params.get("rate", 0.05)); self.p = float(params.get("p", 1.0))
        self.a_src = float(params.get("a_src", 0.65)); self.d = float(params.get("d", 0.2))
        self.block = str(params.get("block", "surv")); self.adh_block = str(params.get("adh_block", "adh"))
        self._started = False                    # the first call sets every live cell's factor to p (a jump): at 0, a death rule on it would kill every cell at the start

    def forward(self, H, mask=None):
        C = H.level(self.at)
        sch = C.state_schema
        s = C.state[:, sch[self.block][0]]
        if not self._started:
            self._started = True
            w = sch[self.block][1] - sch[self.block][0]
            out = torch.zeros(s.shape[0], w, dtype=s.dtype, device=s.device)
            out[:, 0] = torch.where(C.occ > 0.5, (self.p - s) / max(float(getattr(H, "dt", 1.0) or 1.0), 1e-12), torch.zeros_like(s))
            return {(self.at, self.block): out}
        A = C.state[:, sch[self.adh_block][0]]
        live = C.occ > 0.5
        ds = self.rate * (self.p * (A < self.a_src).to(s.dtype) - s)
        ei = getattr(C, "edge_index", None)
        if ei is not None and ei.numel():
            i, j = ei[0], ei[1]
            agg = torch.zeros_like(s).index_add_(0, i, s[j])
            deg = torch.zeros_like(s).index_add_(0, i, torch.ones_like(s[j]))
            ds = ds + self.d * torch.where(deg > 0, agg / deg.clamp(min=1.0) - s, torch.zeros_like(s))
        ds = torch.where(live, ds, torch.zeros_like(ds))
        w = sch[self.block][1] - sch[self.block][0]
        out = torch.zeros(s.shape[0], w, dtype=s.dtype, device=s.device); out[:, 0] = ds
        return {(self.at, self.block): out}
