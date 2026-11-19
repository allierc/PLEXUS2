"""exp 25 (frog gastrulation): a stage-9 blastula of material cells, seeded around its blastocoel.

One registered variant (experiments/exp25_frog_gastrulation.md, `## The model`):

    seed_positions[model: blastula]    ~700 cells placed by zone -- an animal cap over the blastocoel, a marginal
                                       band, a vegetal mass of big yolk cells -- and ONE liquid body filling the
                                       blastocoel; every lattice point of the embryo handed to its nearest cell
                                       centre (Voronoi), each point's volume from its cell's share of the embryo

The cells, their contact, the envelope and the per-cell tension are exp23's (`cleavage_ops`); the bottle cells of
v2 use `mpm_grid_update[body_csf]`'s `cell_surface_tension_block` (a per-cell factor on the cortical tension) added for this
experiment.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from plexus.models.base import Lateral, Seed, Structural
from plexus.models.registry import register_operator
from plexus.operators.cleavage_ops import _set_block
from plexus.operators.mpm_ops import _mpm_reset_points


def _lattice(spacing, lo, hi, rng, jitter=0.25):
    """A jittered cubic lattice of `spacing` filling the box [lo, hi]."""
    ax = [np.arange(lo[k] + 0.5 * spacing, hi[k], spacing) for k in range(3)]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    return g + rng.uniform(-jitter, jitter, g.shape) * spacing


@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="blastula",
                   title="lay out a blastula",
                   equation=r"""$$\pi(i)=\arg\min_c\lvert\mathbf x_i-\mathbf c_c\rvert,\qquad v_i=\frac{n_c\,h^3}{N_c},\qquad y_i=\tfrac12\Big(1-\frac{(\mathbf x_i-\mathbf o)\cdot\hat{\mathbf a}}{R}\Big)$$""")
class SeedBlastula(Seed):
    r"""a Xenopus blastula at stage 9 laid out from its geometry: a ball of cells in zones about a blastocoel that is itself a liquid body, each cell a Voronoi cell of points carrying its yolk.
    particle -> particle (+ its cell set): per zone, cell centres on a jittered lattice of the zone's spacing inside the zone, as many as the cell set declares of that type; points assigned by nearest centre with the zone's volume per point and the yolk gradient.

        \pi(i)=\arg\min_c\lvert\mathbf x_i-\mathbf c_c\rvert,\qquad v_i=\frac{n_c\,h^3}{N_c},\qquad y_i=\tfrac12\Big(1-\frac{(\mathbf x_i-\mathbf o)\cdot\hat{\mathbf a}}{R}\Big)
        \pi(i)         the cell of point i, the nearest centre
        x_i, c_c       the point and the centre
        v_i            the point's volume, the zone's cell count times the lattice volume over the cell's points
        n_c, N_c, h    the zone's cell count, the cell's point count, the lattice spacing
        y_i            the point's yolk, 0 at the animal pole, 1 at the vegetal pole
        o, R, \hat a   the embryo's centre, radius and animal axis

    Reference: Nieuwkoop, P. D. & Faber, J. (1994). Normal Table of Xenopus laevis (Daudin), stage 9; Plexus exp25 (this work)."""
    PARAM_UNITS = {'blastocoel_spacing': 'length', 'centre': 'length', 'density': 'density', 'point_spacing': 'length', 'radius': 'length', 'seed': 'count', 'spacing': 'length'}
    SYMBOLS = {'\\pi(i)': 'the cell of point i, the nearest centre', 'x_i, c_c': 'the point and the centre', 'v_i': "the point's volume, the zone's cell count times the lattice volume over the cell's points", 'n_c, N_c, h': "the zone's cell count, the cell's point count, the lattice spacing", 'y_i': "the point's yolk, 0 at the animal pole, 1 at the vegetal pole", 'o, R, \\hat a': "the embryo's centre, radius and animal axis"}
    REQUIRES_PARAMS = ["cell_set", "radius", "centre"]
    SUPPORTED_DIMS = [3]
    PARAM_ROLES = {"cell_set": "the_cell_set", "radius": "embryo_radius_world", "centre": "embryo_centre_world",
                   "blastocoel": "height_and_radius_of_the_cavity", "spacing": "cell_spacing_per_zone",
                   "bands": "zone_heights_animal_above_vegetal_below", "point_spacing": "lattice_spacing_of_points",
                   "blastocoel_spacing": "lattice_spacing_in_the_cavity", "seed": "rng_seed",
                   "bottle": "latitude_band_azimuth_and_tension_factor_of_the_bottle_cells", 'density': 'density_of_the_material', 'type_names': 'names_of_the_zone_types'}
    REFERENCE = "Nieuwkoop, P. D. & Faber, J. (1994). Normal Table of Xenopus laevis (Daudin)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.cell_set = str(params["cell_set"])
        self.R = float(params["radius"])
        self.centre = [float(v) for v in params["centre"]]
        b = params.get("blastocoel") or {}
        self.b_h, self.b_r = float(b.get("height", 0.35 * self.R)), float(b.get("radius", 0.45 * self.R))
        sp = params.get("spacing") or {}
        self.spacing = {"animal": float(sp.get("animal", 1.1)), "marginal": float(sp.get("marginal", 1.6)),
                        "vegetal": float(sp.get("vegetal", 2.2))}
        bd = params.get("bands") or {}
        self.above, self.below = float(bd.get("animal_above", 1.0)), float(bd.get("vegetal_below", -1.5))
        self.h = float(params.get("point_spacing", 0.2))
        self.hb = float(params.get("blastocoel_spacing", 0.24))
        self.seed = int(params.get("seed", 0))
        self.density = float(params.get("density", 1.0))
        # `bottle: {lat: [lo, hi], azim: half_width, tension: f}` (v2) -- the dorsal marginal zone's BOTTLE CELLS:
        # tissue cells whose centre lies between `lat` degrees from the vegetal pole and within `azim` degrees of
        # the dorsal side (+x) get `tension` f instead of 1 -- a cortex contracting f times harder, read by
        # `body_csf`'s `cell_surface_tension_block`. Hardin & Keller 1988: the first lip forms where the bottle cells constrict.
        bt = params.get("bottle")
        self.bottle = None
        if bt:
            # `surface: d` (v3): only cells whose centre lies within d of the embryo's surface -- bottle cells are
            # surface cells; marked by angle alone (v2), half the marked cells were interior ones
            self.bottle = (float(bt["lat"][0]), float(bt["lat"][1]), float(bt.get("azim", 30.0)), float(bt["tension"]),
                           float(bt.get("surface", 1e9)))
            # `apical: a` (v5, 2026-10-13): the marked cells also get the cell block `apical` = a (0 elsewhere), read by
            # `apical_constriction` -- a contraction of the FREE face only (Finding 1), with `tension: 1` leaving the
            # whole cortex as the neighbours'
            self.apical = float(bt.get("apical", 0.0))
        self.names = {"animal": "animal", "marginal": "marginal", "vegetal": "vegetal", "blastocoel": "blastocoel"}
        self.names.update({k: str(v) for k, v in (params.get("type_names") or {}).items()})

    def _zone_mask(self, X, zone):
        o = np.array(self.centre); d = X - o
        r = np.linalg.norm(d, axis=1)
        cav = np.linalg.norm(d - np.array([0.0, 0.0, self.b_h]), axis=1)
        z = d[:, 2]
        inside = (r < self.R) & (cav > self.b_r)
        if zone == "animal":
            return inside & (z > self.above)
        if zone == "vegetal":
            return inside & (z < self.below)
        if zone == "marginal":
            return inside & (z <= self.above) & (z >= self.below)
        return cav < self.b_r                                               # the blastocoel

    def forward(self, H, mask=None):
        from scipy.spatial import cKDTree
        P, C = H.level(self.at), H.level(self.cell_set)
        rng = np.random.default_rng(self.seed)
        tn = list(getattr(C, "type_names", []) or [])
        nt = C.node_type.detach().cpu().numpy()
        o = np.array(self.centre)
        lo, hi = o - self.R, o + self.R
        # 1. cell centres per zone, as many as the cell set declares of that type
        centres = np.zeros((C.state.shape[0], 3)); zone_of = np.full(C.state.shape[0], -1)
        zid = {"animal": 0, "marginal": 1, "vegetal": 2, "blastocoel": 3}
        for zone in ("animal", "marginal", "vegetal"):
            tname = self.names[zone]
            if tname not in tn:
                raise ValueError(f"seed_positions[blastula]: {self.cell_set!r} declares no type {tname!r}")
            slots = np.flatnonzero(nt == tn.index(tname))
            margin = 0.4 * self.spacing[zone]                                # a centre sits a bit inside its region
            cand = _lattice(self.spacing[zone], lo, hi, rng)
            d = cand - o
            ok = self._zone_mask(cand, zone) & (np.linalg.norm(d, axis=1) < self.R - margin) \
                & (np.linalg.norm(d - np.array([0.0, 0.0, self.b_h]), axis=1) > self.b_r + margin)
            cand = cand[ok]
            if len(cand) < len(slots):
                raise ValueError(f"seed_positions[blastula]: the {zone} zone holds {len(cand)} centres at spacing "
                                 f"{self.spacing[zone]} but the cell set declares {len(slots)} {tname!r} cells")
            pick = rng.choice(len(cand), len(slots), replace=False)
            centres[slots] = cand[pick]; zone_of[slots] = zid[zone]
        bname = self.names["blastocoel"]
        if bname not in tn or int((nt == tn.index(bname)).sum()) != 1:
            raise ValueError(f"seed_positions[blastula]: the cell set must declare exactly one {bname!r} cell")
        b_slot = int(np.flatnonzero(nt == tn.index(bname))[0])
        centres[b_slot] = o + np.array([0.0, 0.0, self.b_h]); zone_of[b_slot] = 3
        tissue = np.flatnonzero(zone_of < 3)
        # 2. lattice points -> nearest centre; the cavity's points -> the blastocoel
        pts = _lattice(self.h, lo, hi, rng, jitter=0.15)
        pts = pts[self._zone_mask(pts, "animal") | self._zone_mask(pts, "marginal") | self._zone_mask(pts, "vegetal")]
        owner = tissue[cKDTree(centres[tissue]).query(pts, k=1)[1]]
        cav = _lattice(self.hb, lo, hi, rng, jitter=0.15)
        cav = cav[self._zone_mask(cav, "blastocoel")]
        parent = P.parent.detach().cpu().numpy()
        dev, dt_ = P.state.device, P.state.dtype
        P.occ = torch.ones_like(P.occ)
        p0, p1 = P.state_schema["pos"]
        counts = {}
        for c in range(C.state.shape[0]):
            blk = np.flatnonzero(parent == c)
            own = cav if c == b_slot else pts[owner == c]
            vol_region = len(own) * (self.hb if c == b_slot else self.h) ** 3
            if len(own) == 0:                                                 # a centre with no lattice point: a tiny ball
                own = centres[c][None] + rng.normal(0, 0.1, (8, 3)); vol_region = 8 * self.h ** 3
            if len(own) >= len(blk):
                take = own[rng.choice(len(own), len(blk), replace=False)]
            else:
                extra = own[rng.integers(0, len(own), len(blk) - len(own))] + rng.normal(0, 0.3 * self.h, (len(blk) - len(own), 3))
                take = np.concatenate([own, extra])
            counts[c] = (len(own), len(blk))
            idx = torch.as_tensor(blk, device=dev)
            _mpm_reset_points(P, idx, torch.as_tensor(take, device=dev, dtype=dt_),
                              torch.full((len(blk),), vol_region / len(blk), device=dev, dtype=dt_))
            if getattr(P, "mass", None) is not None:
                P.mass[idx] = self.density * P.p_vol[idx]
            _set_block(P, "pid", idx, float(c)); _set_block(P, "cellc", idx, float(c % 20))
            _set_block(P, "zone", idx, float(zone_of[c]))
        X = P.state[:, p0:p1].detach().cpu().numpy()
        y = np.clip(0.5 * (1.0 - (X[:, 2] - o[2]) / self.R), 0.0, 1.0)
        allp = torch.arange(X.shape[0], device=dev)
        _set_block(P, "yolk", allp, torch.as_tensor(y, dtype=dt_, device=dev))
        # 3. the cells: position, zone, tension, latitude from the vegetal pole, azimuth from +x (dorsal)
        ca, cb = C.state_schema["pos"]
        st = C.state.clone()
        st[:, ca:cb] = torch.as_tensor(centres, dtype=st.dtype, device=st.device)
        C.state = st
        allc = torch.arange(C.state.shape[0], device=C.state.device)
        d = centres - o
        lat = np.degrees(np.arccos(np.clip(-d[:, 2] / np.maximum(np.linalg.norm(d, axis=1), 1e-9), -1, 1)))
        azim = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        yc = np.array([float(y[parent == c].mean()) if (parent == c).any() else 0.5 for c in range(C.state.shape[0])])
        tension = (zone_of < 3).astype(float)
        n_bottle = 0
        if self.bottle is not None:
            lo_, hi_, half, f, surf_d = self.bottle
            r_c = np.linalg.norm(d, axis=1)
            ring = (zone_of < 3) & (lat >= lo_) & (lat <= hi_) & (np.abs(azim) <= half) & (r_c >= self.R - surf_d)
            tension[ring] = f; n_bottle = int(ring.sum())
            _set_block(C, "apical", allc, torch.as_tensor(ring.astype(float) * self.apical, dtype=C.state.dtype,
                                                          device=C.state.device))
        for name, val in (("zone", zone_of.astype(float)), ("tension", tension),
                          ("lat", lat), ("azim", azim), ("yolk", yc)):
            _set_block(C, name, allc, torch.as_tensor(val, dtype=C.state.dtype, device=C.state.device))
        n_cells = int((zone_of < 3).sum())
        short = sum(1 for c, (m, k) in counts.items() if m < k)
        print(f"[seed_positions:blastula] {self.at}: {X.shape[0]:,} points in {n_cells} cells + the blastocoel "
              f"({counts[b_slot][0]:,} lattice points in the cavity, radius {self.b_r:g} at height {self.b_h:g}); "
              f"zones animal/marginal/vegetal = {[(zone_of == k).sum() for k in range(3)]} cells; "
              f"{short} cells padded with copies; embryo radius {self.R:g}"
              + (f"; {n_bottle} bottle cells (tension x{self.bottle[3]:g}, lat {self.bottle[0]:g}-{self.bottle[1]:g} deg, "
                 f"|azim| <= {self.bottle[2]:g} deg)" if self.bottle else ""), flush=True)
        return {}


@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="collide",
                   title="launch cell pairs at each other",
                   equation=r"""$$\mathbf x_i\in h\,\mathbb Z^3,\ \lvert\mathbf x_i-\mathbf c_{\pi(i)}\rvert\le r,\qquad \mathbf v_i=\pm v\,\hat{\mathbf e}_x$$""")
class SeedCollide(Seed):
    r"""a collision probe: the cell set's slots taken in pairs, each pair on its own row, the two cells of a pair a gap apart and thrown at each other at `speed`.
    particle -> particle (+ its cell set): writes the points' pos and vel (a lattice ball per cell) and the cells' pos.

        \mathbf x_i\in h\,\mathbb Z^3,\ \lvert\mathbf x_i-\mathbf c_{\pi(i)}\rvert\le r,\qquad \mathbf v_i=\pm v\,\hat{\mathbf e}_x
        x_i            a point of the lattice ball of its cell
        h              the lattice spacing
        c_{\pi(i)}     the cell's centre
        r              `radius`
        v_i            the point's velocity, toward the partner along x
        v              `speed`

    Reference: Plexus exp23 (this work); Cedric, 2026-10-12."""
    PARAM_UNITS = {'centre': 'length', 'column_spacing': 'length', 'gap': 'length', 'pairs': 'count', 'radius': 'length', 'row_spacing': 'length', 'speed': 'velocity'}
    SYMBOLS = {'x_i': 'a point of the lattice ball of its cell', 'h': 'the lattice spacing', 'c_{\\pi(i)}': "the cell's centre", 'r': '`radius`', 'v_i': "the point's velocity, toward the partner along x", 'v': '`speed`'}
    REQUIRES_PARAMS = ["cell_set", "radius", "centre"]
    SUPPORTED_DIMS = [3]
    PARAM_ROLES = {"cell_set": "the_cell_set", "radius": "cell_radius_world", "centre": "centre_of_the_array",
                   "gap": "surface_gap_before_impact", "speed": "approach_speed_of_each_cell",
                   "row_spacing": "distance_between_pairs", 'column_spacing': 'spacing_of_the_columns', 'mat_block': 'cell_block_of_the_material_index', 'pairs': 'number_of_pairs'}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.cell_set = str(params["cell_set"])
        self.r = float(params["radius"]); self.c = np.array([float(v) for v in params["centre"]])
        self.gap = float(params.get("gap", 1.0)); self.v = float(params.get("speed", 1.0))
        self.dy = float(params.get("row_spacing", 3.0))
        # `pairs: [[column, row, left_type, right_type], ...]` -- an explicit layout (v2 of the probe): pairs of ANY
        # two materials, in columns `column_spacing` apart along x; each cell takes the next free slot of its type.
        # Absent, slots (2k, 2k+1) form pair k in one column, as before.
        self.pairs = params.get("pairs")
        self.dx_col = float(params.get("column_spacing", 12.0))
        self.mat_block = params.get("mat_block", "mat")

    def forward(self, H, mask=None):
        from plexus.operators.seed_ops import SeedPositionsLatticeBody
        P, C = H.level(self.at), H.level(self.cell_set)
        nC = C.state.shape[0]
        tn = list(getattr(C, "type_names", []) or [])
        nt = C.node_type.detach().cpu().numpy()
        if self.pairs is not None:
            free = {t: list(np.flatnonzero(nt == i)) for i, t in enumerate(tn)}
            ncol = 1 + max(int(q[0]) for q in self.pairs)
            nrow = 1 + max(int(q[1]) for q in self.pairs)
            layout = []
            for col, row, ta, tb in self.pairs:
                if not free.get(ta) or not free.get(tb):
                    raise ValueError(f"seed_positions[collide]: no free cell of type {ta!r} or {tb!r} left")
                xc = self.c[0] + (int(col) - 0.5 * (ncol - 1)) * self.dx_col
                y = self.c[1] + (int(row) - 0.5 * (nrow - 1)) * self.dy
                layout.append((free[ta].pop(0), xc, y, -1.0)); layout.append((free[tb].pop(0), xc, y, 1.0))
            if len(layout) != nC:
                raise ValueError(f"seed_positions[collide]: {len(layout)} cells placed, the cell set holds {nC}")
        else:
            if nC % 2:
                raise ValueError("seed_positions[collide]: the cell set must hold an even number of cells (pairs)")
            K0 = nC // 2
            layout = []
            for k in range(K0):
                y = self.c[1] + (k - 0.5 * (K0 - 1)) * self.dy
                layout.append((2 * k, self.c[0], y, -1.0)); layout.append((2 * k + 1, self.c[0], y, 1.0))
        K = nC // 2
        parent = P.parent.detach().cpu().numpy()
        dev, dt_ = P.state.device, P.state.dtype
        P.occ = torch.ones_like(P.occ)
        vel = P.state_schema.get("vel")
        cen = np.zeros((nC, 3))
        for j, xc, y, side in layout:
            if True:
                cen[j] = [xc + side * (0.5 * self.gap + self.r), y, self.c[2]]
                blk = np.flatnonzero(parent == j)
                off, _ = SeedPositionsLatticeBody._ball(len(blk), self.r, "cpu")
                idx = torch.as_tensor(blk, device=dev)
                vol = 4.0 / 3.0 * math.pi * self.r ** 3 / len(blk)
                _mpm_reset_points(P, idx, torch.as_tensor(cen[j] + off.numpy(), device=dev, dtype=dt_),
                                  torch.full((len(blk),), vol, device=dev, dtype=dt_))
                if vel is not None:
                    st = P.state.clone()
                    st[idx, vel[0]] = -side * self.v
                    P.state = st
                _set_block(P, "pid", idx, float(j)); _set_block(P, "cellc", idx, float(j % 20))
        ca, cb = C.state_schema["pos"]
        st = C.state.clone(); st[:, ca:cb] = torch.as_tensor(cen, dtype=st.dtype, device=st.device); C.state = st
        # `density`: each cell's density (mass over rest volume of its points), when the cell set declares the block
        if "density" in C.state_schema and getattr(P, "mass", None) is not None:
            m = torch.zeros(nC, dtype=dt_, device=dev).index_add_(0, P.parent, P.mass.to(dt_))
            v = torch.zeros(nC, dtype=dt_, device=dev).index_add_(0, P.parent, P.p_vol.to(dt_))
            _set_block(C, "density", torch.arange(nC, device=C.state.device), (m / v.clamp_min(1e-30)).to(C.state.dtype))
        # `mat`: each cell's material as its type's rank, 0 (first type) .. 1 (last) -- the colour key
        if self.mat_block and self.mat_block in C.state_schema and len(tn) > 1:
            _set_block(C, self.mat_block, torch.arange(nC, device=C.state.device),
                       torch.as_tensor(nt / (len(tn) - 1), dtype=C.state.dtype, device=C.state.device))
        print(f"[seed_positions:collide] {K} pairs of cells (radius {self.r:g}, {len(blk)} points each) "
              f"{self.gap:g} apart, each at {self.v:g} toward its partner", flush=True)
        return {}


@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="cell_row",
                   title="lay out one cell per material",
                   equation=r"""$$\mathbf x_i\in h\,\mathbb Z^3,\ \lvert\mathbf x_i-\mathbf c_{\pi(i)}\rvert\le r$$""")
class SeedCellRow(Seed):
    r"""a division probe: every live cell of the cell set on a grid of columns, in the order of its type (one material per cell), each a lattice ball.
    particle -> particle (+ its cell set): writes the points' pos (zero vel) and the cells' pos.

        \mathbf x_i\in h\,\mathbb Z^3,\ \lvert\mathbf x_i-\mathbf c_{\pi(i)}\rvert\le r
        x_i            a point of the lattice ball of its cell
        h              the lattice spacing
        c_{\pi(i)}     the cell's centre on the grid
        r              `radius`

    Reference: Plexus exp23 (this work); Cedric, 2026-10-13."""
    PARAM_UNITS = {'centre': 'length', 'columns': 'count', 'radius': 'length', 'spacing': 'length'}
    SYMBOLS = {'x_i': 'a point of the lattice ball of its cell', 'h': 'the lattice spacing', 'c_{\\pi(i)}': "the cell's centre on the grid", 'r': '`radius`'}
    REQUIRES_PARAMS = ["cell_set", "radius", "centre"]
    SUPPORTED_DIMS = [3]
    PARAM_ROLES = {"cell_set": "the_cell_set", "radius": "cell_radius_world", "centre": "centre_of_the_grid",
                   "spacing": "distance_between_cells", "columns": "cells_per_row"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.cell_set = str(params["cell_set"])
        self.r = float(params["radius"]); self.c = np.array([float(v) for v in params["centre"]])
        self.sp = float(params.get("spacing", 6.0)); self.ncol = int(params.get("columns", 5))

    def forward(self, H, mask=None):
        from plexus.operators.seed_ops import SeedPositionsLatticeBody
        P, C = H.level(self.at), H.level(self.cell_set)
        live = np.flatnonzero(C.occ.detach().cpu().numpy() > 0.5)
        tn = list(getattr(C, "type_names", []) or [])
        nt = C.node_type.detach().cpu().numpy()
        live = live[np.argsort(nt[live], kind="stable")]
        parent = P.parent.detach().cpu().numpy()
        dev, dt_ = P.state.device, P.state.dtype
        occ = torch.zeros_like(P.occ)
        nrow = int(math.ceil(len(live) / self.ncol))
        cen = C.state[:, C.state_schema["pos"][0]:C.state_schema["pos"][1]].detach().cpu().numpy().copy()
        for n, j in enumerate(live):
            col, row = n % self.ncol, n // self.ncol
            cen[j] = self.c + np.array([(col - 0.5 * (self.ncol - 1)) * self.sp, (row - 0.5 * (nrow - 1)) * self.sp, 0.0])
            blk = np.flatnonzero(parent == j)
            off, _ = SeedPositionsLatticeBody._ball(len(blk), self.r, "cpu")
            idx = torch.as_tensor(blk, device=dev)
            _mpm_reset_points(P, idx, torch.as_tensor(cen[j] + off.numpy(), device=dev, dtype=dt_),
                              torch.full((len(blk),), 4.0 / 3.0 * math.pi * self.r ** 3 / len(blk), device=dev, dtype=dt_))
            _set_block(P, "pid", idx, float(j)); _set_block(P, "cellc", idx, float(j % 20))
            occ[idx] = 1.0
        P.occ = occ
        ca, cb = C.state_schema["pos"]
        st = C.state.clone(); st[:, ca:cb] = torch.as_tensor(cen, dtype=st.dtype, device=st.device); C.state = st
        if "mat" in C.state_schema and len(tn) > 1:
            _set_block(C, "mat", torch.as_tensor(live, device=C.state.device),
                       torch.as_tensor(nt[live] / (len(tn) - 1), dtype=C.state.dtype, device=C.state.device))
        print(f"[seed_positions:cell_row] {len(live)} cells (radius {self.r:g}) in {nrow} rows of {self.ncol}, "
              f"{int(occ.sum()):,} live points; {C.state.shape[0] - len(live)} free slots for daughters", flush=True)
        return {}


@register_operator("cell_populate", model="mpm_points", set="cell", kind="divide", family="population",
                   title="grow cells by adding points",
                   equation=r"""$$\frac{dN_c}{dt}=g\,N_c,\qquad \text{mark}_c=\big[N_c\ge N^{\ast}\big]$$""")
class CellPopulateMPMPoints(Structural):
    r"""growth of cells made of material points by adding points from the set's dormant pool, at an exponential rate, and a division mark when the mother's point count is reached.
    cell -> cell and its point set: wakes dormant points seated beside the cell's live ones (`seat`), sets the cell's `mark` at the threshold.

        \frac{dN_c}{dt}=g\,N_c,\qquad \text{mark}_c=\big[N_c\ge N^{\ast}\big]
        N_c            the point count of cell c
        g              `rate`, the growth rate, per time (the fraction carried to the next call)
        mark_c         the division mark
        N^*            `target`, the point count at division

    Reference: Plexus exp23 (this work); Cedric, 2026-10-13."""
    PARAM_UNITS = {'jitter': 'fraction', 'rate': 'rate', 'seed': 'count', 'swell': 'dimensionless', 'target': 'count', 'tries': 'count'}
    SYMBOLS = {'N_c': 'the point count of cell c', 'g': '`rate`, the growth rate, per time (the fraction carried to the next call)', 'mark_c': 'the division mark', 'N^*': '`target`, the point count at division'}
    REQUIRES_PARAMS = ["points", "rate", "target"]
    SUPPORTED_DIMS = [3]
    MAY_MUTATE_INTEGRATED_STATE = True
    PARAM_ROLES = {"points": "the_point_set", "rate": "exponential_growth_rate_per_time_unit",
                   "target": "point_count_that_marks_a_cell_for_division", "mark": "cell_block_set_at_target",
                   "npts_block": "cell_block_with_each_cells_point_count", "jitter": "offset_of_a_new_point_in_spacings",
                   "seat": "random_or_inside", 'seed': 'random_seed', 'swell': 'rest_volume_compression_of_the_new_point', 'tries': 'seating_tries'}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.points = str(params["points"])
        self.g = float(params["rate"]); self.target = int(params["target"])
        self.mark = params.get("mark", "ready"); self.npts = params.get("npts_block", "npts")
        self.jitter = float(params.get("jitter", 0.3))
        # `seat: inside` (2026-10-13, exp25 growth probe 2): each new point is also drawn a random 5-25 % of the way from
        # its template toward the cell's centroid, so material is added INSIDE the cell. With `random` (the first
        # probe) a copy of a surface point could land outside the cell -- in a neighbour, where the contact fired
        # on it -- which scattered the colony and left the cells ragged (Finding 4).
        self.seat = str(params.get("seat", "random"))
        # `seat: fill` (2026-10-13, exp25_growdiv): a copy placed `jitter` (0.3) spacings from its template stays
        # glued to it -- MPM has no force between points -- so a cell that doubled 5 times was a cloud of tight pairs:
        # point spacing 0.19 at seeding -> 0.059 inside the cells (10th percentile 0.027), and the surface contoured
        # from it came out spiky (Cedric: "not a very nice growth, not uniform, too spiky"). `fill` draws `tries`
        # (default 8) candidates per new point, each a random live point of the cell moved one spacing in a random
        # direction, keeps those within the cell (distance to its centroid below the 90th percentile of its points'),
        # and seats the point at the candidate farthest from every point already there -- new material goes where
        # the cell is sparsest.
        if self.seat not in ("random", "inside", "fill"):
            raise ValueError("cell_populate[mpm_points]: seat is random, inside or fill")
        self.tries = int(params.get("tries", 8))
        # `swell: true` (2026-10-13, Finding 5): an added point is born at rest (F = I, J = 1, no pressure), so a cell
        # that gains points PACKS instead of growing -- one probe cell went 2,034 -> 8,022 points at a fixed edge, its
        # points' summed rest volume 4.4x the ball they filled, which the contact read as a 4x overlap. With `swell`,
        # a cell going from N to N + n points has the deformation gradient of EVERY one of its points (the new ones
        # included) scaled by (N / (N + n))^(1/D): the same material now sits in the old volume, compressed by
        # J = N / (N + n), so its own bulk pressure -la (J - 1) > 0 pushes it out to the volume its points ask for.
        self.swell = bool(params.get("swell", False))
        self._rng = np.random.default_rng(int(params.get("seed", 0)))
        self._acc = None
        self._warned = False

    def _wanted(self, H, C, c, counts, g_c, dt):
        """How many points cell `c` gains this call (a real number: its integer part is added, the rest carried): its rate times its count
        times the step, plus the carry. `cell_populate[mpm]` overrides this one method to follow a mass the cell holds instead."""
        return g_c[c] * counts[c] * dt + self._acc[c]

    def _rates(self, H, C, nC):
        """The growth rate of every cell slot, per time unit: `rate` for all (this model); a variant that reads a rate per cell from the
        cell's own state overrides this one method (`cell_populate[mpm_points_chem]`)."""
        return np.full(nC, self.g)

    def _fill(self, P, st, new, own, occ, par, p0, p1, h, dev):
        """`seat: fill`: each new point at the emptiest of `tries` candidates one spacing off random points of its
        cell (see __init__). Cells are processed one after another so later points see the earlier ones."""
        from scipy.spatial import cKDTree
        X = P.state[:, p0:p1].detach().cpu().numpy().astype(np.float64)
        new_np, own_np = new.cpu().numpy(), own.cpu().numpy()
        out = st[:, p0:p1].detach().cpu().numpy().astype(np.float64)
        for c in np.unique(own_np):
            mem = np.flatnonzero(occ & (par == c))
            Pc = X[mem]
            cen = Pc.mean(0)
            r90 = np.percentile(np.linalg.norm(Pc - cen, axis=1), 90)
            slots = new_np[own_np == c]
            m = len(slots)
            base = Pc[self._rng.integers(0, len(Pc), (m, self.tries))]                       # [m, tries, 3]
            u = self._rng.normal(size=(m, self.tries, 3))
            u /= np.linalg.norm(u, axis=2, keepdims=True)
            cand = base + h * u
            inside = np.linalg.norm(cand - cen, axis=2) < r90
            tree = cKDTree(Pc)
            dist = tree.query(cand.reshape(-1, 3), k=1)[0].reshape(m, self.tries)
            dist = np.where(inside, dist, -1.0)
            pick = cand[np.arange(m), dist.argmax(1)]
            # two new points of one call must not land on the same gap: keep a pick only if it is >= h/2 from the
            # picks already kept, else take its candidate's template plus a fresh half-spacing offset
            kept = []
            for i in range(m):
                q = pick[i]
                if kept and np.min(np.linalg.norm(np.asarray(kept) - q, axis=1)) < 0.5 * h:
                    q = base[i, 0] + 0.5 * h * u[i, 0]
                kept.append(q)
            out[slots] = np.asarray(kept)
        st = st.clone()
        st[:, p0:p1] = torch.as_tensor(out, dtype=st.dtype, device=dev)
        return st

    def forward(self, H, mask=None):
        C, P = H.level(self.at), H.level(self.points)
        nC, Np = C.state.shape[0], P.state.shape[0]
        if self._acc is None or len(self._acc) != nC:
            self._acc = np.zeros(nC)
        dev = P.state.device
        occ = (P.occ > 0.5).detach().cpu().numpy()
        par = P.parent.detach().cpu().numpy()
        live_c = np.flatnonzero(C.occ.detach().cpu().numpy() > 0.5)
        counts = np.bincount(par[occ], minlength=nC)
        dt = float(getattr(H, "dt", 1.0))
        pool = list(np.flatnonzero(~occ))
        news, temps, owners = [], [], []
        g_c = self._rates(H, C, nC)                      # the rate of each cell: one number for the base model
        for c in live_c:
            want = self._wanted(H, C, c, counts, g_c, dt)
            n = int(want); self._acc[c] = want - n
            n = min(n, len(pool))
            if n <= 0:
                continue
            own = np.flatnonzero(occ & (par == c))
            temps.append(self._rng.choice(own, n)); news.append(np.array(pool[:n])); owners.append(np.full(n, c))
            del pool[:n]
            counts[c] += n
        if news:
            new = torch.as_tensor(np.concatenate(news), device=dev, dtype=torch.long)
            tmp = torch.as_tensor(np.concatenate(temps), device=dev, dtype=torch.long)
            own = torch.as_tensor(np.concatenate(owners), device=dev, dtype=torch.long)
            for nm, buf in list(P._buffers.items()):      # every per-point buffer, copied from the template point -- FUNCTIONALLY
                if buf is None or nm in ("occ", "parent", "state", "F", "C") or buf.dim() == 0 or buf.shape[0] != Np:
                    continue
                nb = buf.clone()                          # (clone, write, rebind: a tape through the substep saved these; exp30)
                nb[new] = nb[tmp]
                setattr(P, nm, nb)
            st = P.state.clone()
            st[new] = st[tmp]
            p0, p1 = P.state_schema["pos"]
            h = float(P.p_vol[tmp].mean().item()) ** (1.0 / 3.0) if getattr(P, "p_vol", None) is not None else 0.1
            st[new, p0:p1] = st[tmp, p0:p1] + self.jitter * h * torch.randn(len(new), p1 - p0, device=dev, dtype=st.dtype)
            if self.seat == "inside":
                X_live = P.state[:, p0:p1].detach()
                cen = torch.zeros(nC, p1 - p0, device=dev, dtype=st.dtype).index_add_(0, P.parent[torch.as_tensor(occ, device=dev)],
                                                                                     X_live[torch.as_tensor(occ, device=dev)])
                cen = cen / torch.as_tensor(np.maximum(np.bincount(par[occ], minlength=nC), 1), device=dev,
                                            dtype=st.dtype)[:, None]
                f = torch.as_tensor(self._rng.uniform(0.05, 0.25, len(new)), device=dev, dtype=st.dtype)[:, None]
                st[new, p0:p1] = st[new, p0:p1] + f * (cen[own] - st[new, p0:p1])
            if self.seat == "fill":
                st = self._fill(P, st, new, own, occ, par, p0, p1, h, dev)
            P.state = st
            D = p1 - p0
            if getattr(P, "F", None) is not None:
                Fn = P.F.clone(); Fn[new] = torch.eye(D, device=dev, dtype=P.F.dtype); P.F = Fn
            if getattr(P, "C", None) is not None:
                Cn = P.C.clone(); Cn[new] = 0.0; P.C = Cn
            if self.swell and getattr(P, "F", None) is not None:
                n_add = np.bincount(np.concatenate(owners), minlength=nC).astype(np.float64)
                n_old = counts - n_add                                   # counts already include the additions
                s = np.ones(nC)
                g = n_add > 0
                s[g] = (n_old[g] / counts[g]) ** (1.0 / D)               # per-cell stretch: J = N / (N + n)
                par_all = P.parent.detach().clone()
                par_all[new] = own
                live_now = (P.occ > 0.5).clone()
                live_now[new] = True
                fac = torch.as_tensor(s, device=dev, dtype=P.F.dtype)[par_all.clamp(0, nC - 1)]
                fac = torch.where(live_now, fac, torch.ones_like(fac))
                P.F = P.F * fac[:, None, None]             # functional, not in place: an autograd tape (exp30) survives the swell
            pn = P.parent.clone(); pn[new] = own; P.parent = pn     # a NEW tensor: lift_index's cache keys on its identity and rebuilds
            _set_block(P, "pid", new, own.to(P.state.dtype)); _set_block(P, "cellc", new, (own % 20).to(P.state.dtype))
            oc = P.occ.clone(); oc[new] = 1.0; P.occ = oc
        if not pool and not self._warned:
            print(f"[cell_populate:mpm_points] the dormant point pool is EMPTY at tick {getattr(H, 'tick', '?')}: "
                  f"growth stops", flush=True)
            self._warned = True
        allc = torch.arange(nC, device=C.state.device)
        _set_block(C, self.npts, allc, torch.as_tensor(counts, dtype=C.state.dtype, device=C.state.device))
        if self.mark in C.state_schema:
            ready = torch.as_tensor((counts >= self.target) & (C.occ.detach().cpu().numpy() > 0.5),
                                    dtype=C.state.dtype, device=C.state.device)
            mc = C.state_schema[self.mark][0]
            marked = torch.maximum(C.state[:, mc], ready)         # from the ORIGINAL state: `maximum` saves its input for the backward,
            st = C.state.clone(); st[:, mc] = marked              # and writing into the same view afterwards broke the tape (exp30 probe 7)
            C.state = st
        return {}


@register_operator("cell_populate", model="mpm_points_chem", set="cell", kind="divide", family="population",
                   title="grow cells by adding points, a rate per cell from its chemistry",
                   equation=r"""$$\frac{dN_c}{dt}=d\,r\,\big(\rho+\mathrm{Hill}(a_c)\big)\,N_c,\qquad \mathrm{Hill}(a)=\frac{a^{n}}{a^{n}+a_{sw}^{n}}$$""")
class CellPopulateMPMPointsChem(CellPopulateMPMPoints):
    r"""cell_populate[mpm_points] with the rate of each cell read from its own chemistry, the growth law of the vertex operator cell_grow (the activator a mitogen) on a cell made of points.
    cell -> cell and its point set: as mpm_points, the per-cell rate d r (rho + Hill(a_c)) (d = `dims`, 3: the count grows as the volume).

        \frac{dN_c}{dt}=d\,r\,\big(\rho+\mathrm{Hill}(a_c)\big)\,N_c,\qquad \mathrm{Hill}(a)=\frac{a^{n}}{a^{n}+a_{sw}^{n}}
        N_c            the point count of cell c
        d              `dims`, the exponent of volume in the linear scale (3)
        r              `rate`, the published linear rate of cell_grow
        \rho           `rho`, the basal rate
        a_c            the cell's activator (chem column `chan`)
        n              `hill`, the Hill exponent
        a_{sw}         `a_sw`, the activator at half rate

    Reference: Okuda, S. et al. (2018). Sci. Rep. 8:2386, Eqs. 7-8 (the growth law); Plexus exp29 (this work)."""
    PARAM_UNITS = {'a_sw': 'dimensionless', 'chan': 'count', 'dims': 'dimensionless', 'hill': 'dimensionless', 'rho': 'dimensionless', 'target': 'count', 'target_jitter': 'fraction'}
    SYMBOLS = {'N_c': 'the point count of cell c', 'd': '`dims`, the exponent of volume in the linear scale (3)', 'r': '`rate`, the published linear rate of cell_grow', '\\rho': '`rho`, the basal rate', 'a_c': "the cell's activator (chem column `chan`)", 'n': '`hill`, the Hill exponent', 'a_{sw}': '`a_sw`, the activator at half rate'}
    REQUIRES_PARAMS = ["points", "rate", "target"]
    PARAM_ROLES = dict(CellPopulateMPMPoints.PARAM_ROLES, rate="published_linear_growth_rate_of_cell_grow_per_time_unit",
                       rho="growth_with_no_activator_as_a_fraction_of_the_full_rate", a_sw="absolute_activator_at_half_switch",
                       hill="hill_exponent", chan="chem_column_of_the_activator", dims="exponent_of_volume_in_the_linear_scale",
                       target_jitter="relative_spread_of_the_division_threshold")

    def __init__(self, params, device="cpu"):
        own = ("rho", "a_sw", "hill", "chan", "dims", "target_jitter")
        super().__init__({k: v for k, v in params.items() if k not in own}, device)
        self.rho = float(params.get("rho", 0.0)); self.a_sw = float(params.get("a_sw", 0.0)); self.hill = float(params.get("hill", 3.0))
        self.chan = int(params.get("chan", 0)); self.dims = float(params.get("dims", 3.0))
        self.jit = float(params.get("target_jitter", 0.0))
        self.target0 = int(params["target"])
        self._tg = None
        self._last = None
        self.last = {}

    def hill_of(self, a):
        """Hill(a) of the activator concentrations `a` (numpy), 0 where a = 0 and a_sw = 0."""
        an = np.power(np.maximum(a, 0.0), self.hill)
        den = an + self.a_sw ** self.hill
        return np.where(den > 0.0, an / np.where(den > 0.0, den, 1.0), 0.0)

    def _rates(self, H, C, nC):
        if "chem" in C.state_schema:
            c0 = C.state_schema["chem"][0] + self.chan
            a = C.state[:, c0].detach().cpu().numpy().astype(np.float64)
        else:
            a = np.zeros(nC)
        g = self.dims * self.g * (self.rho + self.hill_of(a))
        self.last = {"rate_min": float(g.min()), "rate_max": float(g.max()), "rate_mean": float(g.mean())}
        return g

    def forward(self, H, mask=None):
        C, P = H.level(self.at), H.level(self.points)
        nC = int(C.state.shape[0])
        if self.jit > 0.0:
            occ = (P.occ > 0.5).detach().cpu().numpy()
            counts = np.bincount(P.parent.detach().cpu().numpy()[occ], minlength=nC).astype(np.float64)
            if self._tg is None or len(self._tg) != nC:
                self._tg = self.target0 * (1.0 + self.jit * self._rng.uniform(-1.0, 1.0, nC))
                self._last = counts.copy()
            dropped = counts < 0.8 * self._last
            if dropped.any():
                self._tg[dropped] = self.target0 * (1.0 + self.jit * self._rng.uniform(-1.0, 1.0, int(dropped.sum())))
            self._last = counts.copy()
            self.target = self._tg
        return super().forward(H, mask)




@register_operator("cell_populate", model="mpm_points_block", set="cell", kind="divide", family="population",
                   title="grow cells by adding points, the linear rate of each cell from its own block",
                   equation=r"""$$\frac{dN_c}{dt}=d\,r\,\big(\rho+\mathrm{Hill}(a_c)\big)\,N_c,\qquad \mathrm{Hill}(a)=\frac{a^{n}}{a^{n}+a_{sw}^{n}}$$""")
class CellPopulateMPMPointsBlock(CellPopulateMPMPointsChem):
    r"""cell_populate[mpm_points_chem] with the published linear rate read per cell from a cell block in place of the one `rate`: the growth as a knob.
    cell -> cell and its point set: as mpm_points_chem plus `rate_block`.

        \frac{dN_c}{dt}=d\,r\,\big(\rho+\mathrm{Hill}(a_c)\big)\,N_c,\qquad \mathrm{Hill}(a)=\frac{a^{n}}{a^{n}+a_{sw}^{n}}
        N_c            the point count of cell c
        d              `dims`, the exponent of volume in the linear scale (3)
        r              `rate`, the published linear rate of cell_grow
        \rho           `rho`, the basal rate
        a_c            the cell's activator (chem column `chan`)
        n              `hill`, the Hill exponent
        a_{sw}         `a_sw`, the activator at half rate

    Reference: as mpm_points_chem; Plexus exp32 (2026-10-25)."""
    PARAM_UNITS = {}
    SYMBOLS = {'N_c': 'the point count of cell c', 'd': '`dims`, the exponent of volume in the linear scale (3)', 'r': '`rate`, the published linear rate of cell_grow', '\\rho': '`rho`, the basal rate', 'a_c': "the cell's activator (chem column `chan`)", 'n': '`hill`, the Hill exponent', 'a_{sw}': '`a_sw`, the activator at half rate'}
    PARAM_ROLES = dict(CellPopulateMPMPointsChem.PARAM_ROLES, rate_block="cell_block_with_the_linear_growth_rate_per_time_unit")

    def __init__(self, params, device="cpu"):
        super().__init__({k: v for k, v in params.items() if k != "rate_block"}, device)
        self.rate_block = str(params.get("rate_block", "grow"))

    def _rates(self, H, C, nC):
        g = super()._rates(H, C, nC)                                      # d r (rho + Hill(a)) with the op's rate r
        if self.rate_block in C.state_schema:
            col = C.state_schema[self.rate_block][0]
            rc = C.state[:, col].detach().cpu().numpy().astype(np.float64)
            rc = np.where(rc > 0.0, rc, self.g)                            # an unseeded block reads the op's rate
            g = g * (rc / max(self.g, 1e-12))
            self.last = {"rate_min": float(g.min()), "rate_max": float(g.max()), "rate_mean": float(g.mean())}
        return g
@register_operator("cell_populate", model="mpm", set="cell", kind="divide", family="population",
                   title="add points so each cell holds the mass it has",
                   equation=r"""$$N_c\leftarrow\lfloor m_c\rfloor\quad(\text{points added to a cell whose mass block exceeds its count})$$""")
class CellPopulateMPM(CellPopulateMPMPoints):
    r"""the point-adding half of the MPM growth, split from the law: cell_grow[mpm] grows each cell's mass (a number in `mass_block`, in points), this operator adds the points that carry it.
    cell -> cell and its point set: wakes dormant points for a cell whose mass block exceeds its point count.

        N_c\leftarrow\lfloor m_c\rfloor\quad(\text{points added to a cell whose mass block exceeds its count})
        N_c            the point count of cell c
        m_c            the cell's mass, in points (block `mass_block`)

    Reference: Plexus exp33 (this work, 2026-10-25)."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"Adds the points that carry the growth. When a cell's mass block $m_c$ exceeds its point count $N_c$, dormant points "
        r"(the reserve parked at the start) are woken for that cell: $N_c\leftarrow\lfloor m_c\rfloor$. cell_grow only changes "
        r"the number $m_c$; this operator makes the cell physically bigger."
    )
    PARAM_UNITS = {}
    SYMBOLS = {'N_c': 'the point count of cell c', 'm_c': "the cell's mass, in points (block `mass_block`)"}
    REQUIRES_PARAMS = ["points"]
    PARAM_ROLES = dict(CellPopulateMPMPoints.PARAM_ROLES, mass_block="cell_block_holding_the_mass_in_points")

    def __init__(self, params, device="cpu"):
        p = {k: v for k, v in params.items() if k != "mass_block"}
        p.setdefault("rate", 0.0); p["target"] = 0; p["mark"] = ""
        super().__init__(p, device)
        self.mass_block = str(params.get("mass_block", "mass"))
        self._mass = None

    def _rates(self, H, C, nC):
        if self.mass_block not in C.state_schema:
            raise ValueError(f"cell_populate[mpm]: the cell set needs a width-1 state block {self.mass_block!r}")
        self._mass = C.state[:, C.state_schema[self.mass_block][0]].detach().cpu().numpy().astype(np.float64)
        return np.zeros(nC)

    def _wanted(self, H, C, c, counts, g_c, dt):
        return max(self._mass[c] - counts[c], 0.0)


@register_operator("cell_populate", model="mpm_interior", set="cell", kind="divide", family="population",
                   title="add points inside each cell so it holds the mass it has",
                   equation=r"""$$\mathbf x^{*}=\arg\max_{\mathbf q\in Q_c}\ \min_{i\in c}\lVert\mathbf q-\mathbf x_i\rVert,\qquad Q_c=\big\{\mathbf q=\mathbf x_j+h\,\mathbf u\ :\ \#\{i\in c:\lVert\mathbf q-\mathbf x_i\rVert<\rho h\}\ge k\big\}$$""")
class CellPopulateMPMInterior(CellPopulateMPM):
    r"""cell_populate[mpm] with `seat: fill`, a new point seated only where the cell's own material surrounds it: growth by packing the interior and swelling, never by accretion on the outside, so a cell of any shape keeps its density.
    cell -> cell and its point set: as cell_populate[mpm].

        \mathbf x^{*}=\arg\max_{\mathbf q\in Q_c}\ \min_{i\in c}\lVert\mathbf q-\mathbf x_i\rVert,\qquad Q_c=\big\{\mathbf q=\mathbf x_j+h\,\mathbf u\ :\ \#\{i\in c:\lVert\mathbf q-\mathbf x_i\rVert<\rho h\}\ge k\big\}
        x^*            where a new point of cell c is seated
        q              a candidate: a random live point x_j of the cell moved one spacing h in a random direction u (`tries` of them per new point)
        Q_c            the candidates that lie inside the cell
        \rho           `near_reach`, the radius of the neighbourhood test, in spacings (1.5)
        k              `min_near`, the points of the cell a candidate needs within that radius (9: an interior site of a lattice has 14 on average, a site on a flat face 7, a site half a spacing outside it 4)
        h              the point spacing, the cube root of the points' rest volume

    The inside test counts the points the cell had before the call. When none of 4 x `tries` candidates is inside, the one with the most points of the cell around it is taken. cell_populate[mpm]'s `fill` calls a candidate inside when it is closer to the centroid than 90 % of the cell's points -- a ball,
    which for a hemisphere or a wedge includes the empty side of a flat face -- and takes the candidate farthest from every point, so on a cell that is not round it seats new points OUTSIDE, on the flat faces; with `swell`
    the cell then grows twice, by that accretion and by the swelling.

    Reference: Plexus exp09 (this work, 2026-11-17): step 20 (it24), replicators cleaved in two and regrown lost 16 % of their density per generation (0.46 of it after four) and drew as fragments; cell_populate[mpm] (exp33)."""
    PARAM_UNITS = dict(CellPopulateMPM.PARAM_UNITS, near_reach='fraction', min_near='count')
    SYMBOLS = {'x^*': 'where a new point of cell c is seated', 'q': 'a candidate: a random live point of the cell moved one spacing in a random direction', 'Q_c': 'the candidates that lie inside the cell',
               '\\rho': '`near_reach`, the radius of the neighbourhood test, in spacings', 'k': '`min_near`, the points of the cell a candidate needs within that radius', 'h': 'the point spacing'}
    PARAM_ROLES = dict(CellPopulateMPM.PARAM_ROLES, near_reach="radius_of_the_inside_test_in_spacings", min_near="own_points_a_candidate_needs_within_that_radius")

    def __init__(self, params, device="cpu"):
        super().__init__(dict(params, seat="fill"), device)
        self.near_reach = float(params.get("near_reach", 1.5))
        self.min_near = int(params.get("min_near", 9))

    def _fill(self, P, st, new, own, occ, par, p0, p1, h, dev):
        """As cell_populate[mpm]'s `_fill`, the inside test replaced: a candidate is inside when `min_near` points of its cell lie within `near_reach` spacings of it."""
        from scipy.spatial import cKDTree
        X = P.state[:, p0:p1].detach().cpu().numpy().astype(np.float64)
        new_np, own_np = new.cpu().numpy(), own.cpu().numpy()
        out = st[:, p0:p1].detach().cpu().numpy().astype(np.float64)
        for c in np.unique(own_np):
            mem = np.flatnonzero(occ & (par == c))
            Pc = X[mem]
            slots = new_np[own_np == c]
            kept = []
            old = cKDTree(Pc)                                            # the inside test counts the points the cell HAD: points seated in this call must not make an outside site pass
            for _ in range(len(slots)):                                  # one new point at a time: the next keeps its distance from it (the base seats a call's points against the old ones only)
                Q = Pc if not kept else np.concatenate([Pc, np.asarray(kept)])
                best, best_near = None, -1
                for _round in range(4):                                  # up to 4 x `tries` candidates before giving up on an inside one
                    base = Pc[self._rng.integers(0, len(Pc), self.tries)]
                    u = self._rng.normal(size=(self.tries, 3))
                    u /= np.linalg.norm(u, axis=1, keepdims=True)
                    cand = base + h * u
                    near = np.array([len(v) for v in old.query_ball_point(cand, self.near_reach * h)])
                    ok = near >= self.min_near
                    if ok.any():
                        dist = cKDTree(Q).query(cand, k=1)[0]
                        best = cand[np.where(ok, dist, -1.0).argmax()]
                        break
                    if near.max() > best_near:
                        best, best_near = cand[near.argmax()], int(near.max())
                kept.append(best)
            out[slots] = np.asarray(kept)
        st = st.clone()
        st[:, p0:p1] = torch.as_tensor(out, dtype=st.dtype, device=dev)
        return st


@register_operator("apical_constriction", family="mechanics", set="particle", kind="lateral",
                   title="constrict the apical face",
                   equation=r"""$$\mathbf a_i=-k\,a_c\,(\mathbf I-\mathbf n_c\mathbf n_c^{\top})(\mathbf x_i-\bar{\mathbf x}^{ap}_c),\quad i\in\mathrm{apical}(c)=\{i\in c:\ \phi(\mathbf x_i)<\phi_s,\ c^{o}_i<c_s\}$$""")
class ApicalConstriction(Lateral):
    r"""a bottle cell: a cell that contracts its apical (free) face while its contacts with the neighbours hold, the dorsal lip of the Xenopus blastopore.
    particle -> particle: reads every live point's position, rest volume and cell, and the cell block `block` (the contraction factor a_c, 0 = not a bottle cell); emits an acceleration on the apical points, summing to zero per cell.

        \mathbf a_i=-k\,a_c\,(\mathbf I-\mathbf n_c\mathbf n_c^{\top})(\mathbf x_i-\bar{\mathbf x}^{ap}_c),\quad i\in\mathrm{apical}(c)=\{i\in c:\ \phi(\mathbf x_i)<\phi_s,\ c^{o}_i<c_s\}
        a_i            the acceleration of apical point i of cell c
        k              `k`, the constriction stiffness, per time squared
        a_c            the cell's contraction factor
        I - n_c n_c^T  the projector onto the apical plane
        x_i            the point's position
        \bar x^{ap}_c  the centroid of the cell's apical points
        apical(c)      the cell's points in the outer shell (`shell`) where the total volume fraction phi is below phi_s (`free`) and the other cells' fraction below c_s (`core`)

    Reference: Hardin, J. & Keller, R. (1988). The behaviour and function of bottle cells during gastrulation of Xenopus laevis. Development 103:211-230."""
    PARAM_UNITS = {'cap': 'acceleration', 'core': 'fraction', 'dx': 'length', 'free': 'fraction', 'k': 'acceleration', 'share': 'fraction', 'share_k': 'acceleration', 'share_shell': 'length', 'shell': 'length'}
    SYMBOLS = {'a_i': 'the acceleration of apical point i of cell c', 'k': '`k`, the constriction stiffness, per time squared', 'a_c': "the cell's contraction factor", 'I - n_c n_c^T': 'the projector onto the apical plane', 'x_i': "the point's position", '\\bar x^{ap}_c': "the centroid of the cell's apical points", 'apical(c)': "the cell's points in the outer shell (`shell`) where the total volume fraction phi is below phi_s (`free`) and the other cells' fraction below c_s (`core`)"}
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["cells", "k"]
    MECHANISM_TAGS = ["apical_constriction", "bottle_cell", "active_force"]
    PARAM_ROLES = {"cells": "the_body_set_containing_the_points", "k": "apical_contraction_per_length",
                   "block": "cell_block_with_the_contraction_factor", "shell": "total_fraction_below_which_a_point_is_outer",
                   "free": "other_cells_fraction_below_which_a_point_is_free", "dx": "lattice_spacing_world",
                   "law": "spring_or_constant", "core": "radius_below_which_the_constant_pull_tapers",
                   "select": "shell_or_cap", "cap": "fraction_of_a_cells_points_that_is_its_apex",
                   "share": "reach_of_the_junctional_belt_into_the_neighbours", "share_k": "belt_pull_on_neighbours_over_k",
                   "share_shell": "tissue_fraction_below_which_a_neighbour_point_is_on_the_surface", 'grid': 'the_grid_field'}
    REFERENCE = "Hardin, J. & Keller, R. (1988). Development 103:211-230."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.cells = str(params["cells"])
        self.k = float(params["k"])
        self.block = str(params.get("block", "apical"))
        self.shell = float(params.get("shell", 0.6))
        self.free = float(params.get("free", 0.05))
        self.dx = params.get("dx")
        self.grid = params.get("grid", "mpm_grid")
        # `law: spring` (v5-v6): a = -k a_c d_t, a spring to the apex's centre -- its pull FADES as the apex shrinks:
        # v6 (k 200) constricted in the first minute (apex fraction 0.26 -> 0.15, elongation 1.11) and relaxed back
        # round within 30-60 min under the cell's own surface tension. `law: constant` (v7): a = -k a_c d_t / |d_t|,
        # the same pull whatever the apex's size -- a contractile ring at constant tension, as myosin holds it (k is
        # then an acceleration, length / time^2). Points within `core` (default 0.05) of the axis get d_t / core.
        self.law = str(params.get("law", "spring"))
        if self.law not in ("spring", "constant"):
            raise ValueError("apical_constriction: law is spring or constant")
        self.core = float(params.get("core", 0.05))
        # `select: shell` (v5-v7): apical = points where the total fraction phi < `shell` -- but a constricting apex
        # CROWDS its points, phi there rises past the threshold and they leave the set: v6 (k 200) and v7 (constant
        # k 100) both constricted in the first minute (apex fraction ~0.15-0.19) and switched themselves off, back to
        # ~0.29 by 60-120 min. `select: cap` (v8): the apical normal n_c of a marked cell is the mean of -grad phi over
        # its points (toward the empty side), and its apex is its `cap` fraction (default 0.2) of points farthest
        # along n_c -- a geometric set that crowding cannot empty.
        self.select = str(params.get("select", "shell"))
        if self.select not in ("shell", "cap"):
            raise ValueError("apical_constriction: select is shell or cap")
        self.cap = float(params.get("cap", 0.2))
        # `share: d` (v9, cap mode): the ring is a JUNCTIONAL belt shared with the neighbours -- every point of an
        # UNMARKED cell within d of the bottle cell's apical points, and no deeper than d below its apex plane, is
        # pulled toward the bottle apex's centre too (in the apex plane, the same law, factor `share_k`, default 1).
        # Without it (v8) a constricting cell slips out from under its neighbours, which close over it: section
        # montage of 0024 -- single cells ingress, the surface never notches (Wen & Winklbauer 2017 Fig. 1A shows a
        # notch at 0:30-1:00 with the bottle cells still in the surface).
        self.share = float(params.get("share", 0.0))
        self.share_k = float(params.get("share_k", 1.0))
        # `share_shell: f` (v10): the belt reaches only neighbour points on the OUTER surface (the tissue fraction phi
        # there below f, default 1 = no limit). v9 (no limit) pulled whatever lay near the sinking bottle cells --
        # the blastocoel body and deep cells -- and scrambled the embryo: the blastocoel broke up within 41-121 min.
        self.share_shell = float(params.get("share_shell", 1.0))
        self._off = None

    def forward(self, H, mask=None):
        P = H.level(self.at)
        X = P.get("pos")
        dev, dt_ = X.device, X.dtype
        C = H.level(self.cells)
        nC = int(C.state.shape[0])
        fac = C.state[:, C.state_schema[self.block][0]].to(dt_) * (C.occ > 0.5).to(dt_)
        acc = torch.zeros_like(X)
        if not bool((fac > 0).any()):
            return {self.at: acc}
        live = (P.occ > 0).to(dt_)
        cid = H.lift_index(self.at, self.cells)
        dx = float(self.dx) if self.dx is not None else float(H.field(self.grid).dx)
        shape = [int(math.ceil(float(w) / dx)) + 3 for w in H.world_size]
        Nw = shape[0] * shape[1] * shape[2]
        if self._off is None or self._off.device != dev:
            o = torch.arange(3, device=dev)
            self._off = torch.stack(torch.meshgrid(o, o, o, indexing="ij"), -1).reshape(-1, 3)
        off = self._off
        xg = X / dx
        base = torch.floor(xg - 0.5)
        fx = xg - base
        w = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], 1)
        idx = base.long()[:, None, :] + off[None]
        for d in range(3):
            idx[..., d] = idx[..., d].clamp(0, shape[d] - 1)
        flat = (idx[..., 0] * shape[1] + idx[..., 1]) * shape[2] + idx[..., 2]
        W = w[:, off[:, 0], 0] * w[:, off[:, 1], 1] * w[:, off[:, 2], 2]
        v = (P.p_vol.to(dt_) if getattr(P, "p_vol", None) is not None else torch.ones_like(live)) * live
        dep = (W * (v / dx ** 3)[:, None]).reshape(-1)
        if self.select == "cap":
            return {self.at: self._cap(X, W, w, fx, off, flat, dep, Nw, cid, fac, nC, live, dx, mask)}
        # only the MARKED cells need their own fraction; everyone's total is one lattice
        phi = torch.zeros(Nw, device=dev, dtype=dt_).index_add_(0, flat.reshape(-1), dep)
        marked_pt = (fac[cid.clamp(0, nC - 1)] > 0) & (live > 0)
        dep_own = (W * (v * marked_pt.to(dt_) / dx ** 3)[:, None]).reshape(-1)
        mid = torch.where(marked_pt, cid, torch.zeros_like(cid))
        mc = torch.unique(mid[marked_pt])
        remap = torch.full((nC,), -1, device=dev, dtype=torch.long); remap[mc] = torch.arange(len(mc), device=dev)
        r = remap[mid]
        own = torch.zeros(len(mc) * Nw, device=dev, dtype=dt_).index_add_(
            0, (r.clamp_min(0)[:, None] * Nw + flat).reshape(-1), dep_own)
        phi_i = (W * phi[flat]).sum(1)
        own_i = (W * own[r.clamp_min(0)[:, None] * Nw + flat]).sum(1)
        c_o = phi_i - own_i
        ap = marked_pt & (phi_i < self.shell) & (c_o < self.free)
        if not bool(ap.any()):
            return {self.at: acc}
        # per marked cell: its centroid, its apical centroid, the apical normal
        nM = len(mc)
        cnt = torch.zeros(nM, device=dev, dtype=dt_).index_add_(0, r[marked_pt], torch.ones_like(r[marked_pt], dtype=dt_))
        cen = torch.zeros(nM, 3, device=dev, dtype=dt_).index_add_(0, r[marked_pt], X[marked_pt]) / cnt.clamp_min(1)[:, None]
        cnt_a = torch.zeros(nM, device=dev, dtype=dt_).index_add_(0, r[ap], torch.ones_like(r[ap], dtype=dt_))
        cen_a = torch.zeros(nM, 3, device=dev, dtype=dt_).index_add_(0, r[ap], X[ap]) / cnt_a.clamp_min(1)[:, None]
        n = cen_a - cen
        n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-9)
        d = X[ap] - cen_a[r[ap]]
        nn = n[r[ap]]
        d_t = d - (d * nn).sum(1, keepdim=True) * nn
        if self.law == "constant":
            d_t = d_t / d_t.norm(dim=1, keepdim=True).clamp_min(self.core)
        acc[ap] = -(self.k * fac[cid[ap]])[:, None] * d_t
        if mask is not None:
            acc = acc * mask[:, None].to(dt_)
        H.apical_points = int(ap.sum())
        return {self.at: acc}

    def _cap(self, X, W, w, fx, off, flat, dep, Nw, cid, fac, nC, live, dx, mask):
        """`select: cap`: the apex of each marked cell is its `cap` fraction of points farthest along the cell's
        outward normal n_c = mean over its points of -grad phi (phi = all cells' fraction)."""
        dev, dt_ = X.device, X.dtype
        acc = torch.zeros_like(X)
        dw = torch.stack([fx - 1.5, -2.0 * (fx - 1.0), fx - 0.5], 1) / dx
        G = torch.stack([dw[:, off[:, 0], 0] * w[:, off[:, 1], 1] * w[:, off[:, 2], 2],
                         w[:, off[:, 0], 0] * dw[:, off[:, 1], 1] * w[:, off[:, 2], 2],
                         w[:, off[:, 0], 0] * w[:, off[:, 1], 1] * dw[:, off[:, 2], 2]], -1)
        phi = torch.zeros(Nw, device=dev, dtype=dt_).index_add_(0, flat.reshape(-1), dep)
        marked = (fac[cid.clamp(0, nC - 1)] > 0) & (live > 0)
        mi = torch.nonzero(marked).reshape(-1)
        gphi = (G[mi] * phi[flat[mi]][..., None]).sum(1)                   # grad phi at the marked points
        c = cid[mi]
        cells = torch.unique(c)
        n_ap = 0
        for cc in cells.tolist():                                           # a handful of bottle cells
            s_ = c == cc
            pi = mi[s_]
            Xc = X[pi]
            n = -gphi[s_].mean(0)
            if float(n.norm()) < 1e-9:
                continue
            n = n / n.norm()
            h = (Xc - Xc.mean(0)) @ n
            thr = torch.quantile(h, 1.0 - self.cap)
            ap = h >= thr
            Xa = Xc[ap]
            ca = Xa.mean(0)
            d = Xa - ca
            d_t = d - (d @ n)[:, None] * n[None]
            if self.law == "constant":
                d_t = d_t / d_t.norm(dim=1, keepdim=True).clamp_min(self.core)
            acc[pi[ap]] = -(self.k * fac[cc]) * d_t
            n_ap += int(ap.sum())
            if self.share > 0.0:
                lo_ = Xa.min(0).values - self.share
                hi_ = Xa.max(0).values + self.share
                box = (~marked) & (live > 0) & (X >= lo_).all(1) & (X <= hi_).all(1)
                bj = torch.nonzero(box).reshape(-1)
                if len(bj):
                    Xb = X[bj]
                    dmin = torch.cdist(Xb, Xa).min(1).values
                    keep = (dmin < self.share) & (((Xb - ca) @ n) > -self.share)
                    if self.share_shell < 1.0:
                        keep = keep & ((W[bj] * phi[flat[bj]]).sum(1) < self.share_shell)
                    bj, Xb = bj[keep], Xb[keep]
                    if len(bj):
                        db = Xb - ca
                        db_t = db - (db @ n)[:, None] * n[None]
                        if self.law == "constant":
                            db_t = db_t / db_t.norm(dim=1, keepdim=True).clamp_min(self.core)
                        acc[bj] = acc[bj] - (self.k * self.share_k * fac[cc]) * db_t
        if mask is not None:
            acc = acc * mask[:, None].to(dt_)
        self.last_apical_points = n_ap
        return acc


@register_operator("apical_junction", family="mechanics", set="particle", kind="lateral",
                   title="bond surface cells into a sheet",
                   equation=r"""$$\mathbf F_{ij}=k_j\,(\lVert\mathbf x_j-\mathbf x_i\rVert-L^0_{ij})\,\frac{\mathbf x_j-\mathbf x_i}{\lVert\mathbf x_j-\mathbf x_i\rVert},\quad \mathbf a_i=\sum_j\mathbf F_{ij}/m_i$$""")
class ApicalJunction(Lateral):
    r"""the superficial epithelium as a sheet: at the first call every pair of outer-surface points of two different cells closer than `reach` is bonded by a spring at its then distance (the apical adherens junctions), so a bottle cell cannot slip out from under its neighbours.
    particle -> particle: emits an acceleration F / m on both ends of every bond, equal and opposite, so the sheet conserves momentum.

        \mathbf F_{ij}=k_j\,(\lVert\mathbf x_j-\mathbf x_i\rVert-L^0_{ij})\,\frac{\mathbf x_j-\mathbf x_i}{\lVert\mathbf x_j-\mathbf x_i\rVert},\quad \mathbf a_i=\sum_j\mathbf F_{ij}/m_i
        F_{ij}         the force on point i from its bond to j
        k_j            `k`, the junction stiffness
        x_i, x_j       the two positions
        L^0_{ij}       the bond's rest length, the distance at the first call
        a_i            the acceleration
        m_i            the point's mass

    Reference: Hardin, J. & Keller, R. (1988). Development 103:211-230; Wen, J. W. H. & Winklbauer, R. (2017). eLife 6:e27190."""
    PARAM_UNITS = {'dx': 'length', 'k': 'acceleration', 'reach': 'length', 'shell': 'length'}
    SYMBOLS = {'F_{ij}': 'the force on point i from its bond to j', 'k_j': '`k`, the junction stiffness', 'x_i, x_j': 'the two positions', 'L^0_{ij}': "the bond's rest length, the distance at the first call", 'a_i': 'the acceleration', 'm_i': "the point's mass"}
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["cells", "k"]
    MECHANISM_TAGS = ["adherens_junction", "epithelial_sheet", "bond_network"]
    PARAM_ROLES = {"cells": "the_body_set_containing_the_points", "k": "junction_spring_force_per_length",
                   "reach": "bonding_distance_world", "shell": "total_fraction_below_which_a_point_is_on_the_surface",
                   "skip_block": "cell_block_zero_for_cells_without_junctions", "dx": "lattice_spacing_world", 'grid': 'the_grid_field'}
    REFERENCE = "Hardin, J. & Keller, R. (1988). Development 103:211-230."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.cells = str(params["cells"])
        self.k = float(params["k"])
        self.reach = float(params.get("reach", 0.25))
        self.shell = float(params.get("shell", 0.8))
        self.skip_block = params.get("skip_block", "tension")
        self.dx = params.get("dx")
        self.grid = params.get("grid", "mpm_grid")
        self.bonds = None

    def _build(self, H, P, X):
        from scipy.spatial import cKDTree
        dev, dt_ = X.device, X.dtype
        live = (P.occ > 0)
        cid = H.lift_index(self.at, self.cells)
        C = H.level(self.cells)
        nC = int(C.state.shape[0])
        dx = float(self.dx) if self.dx is not None else float(H.field(self.grid).dx)
        shape = [int(math.ceil(float(w) / dx)) + 3 for w in H.world_size]
        Nw = shape[0] * shape[1] * shape[2]
        o = torch.arange(3, device=dev)
        off = torch.stack(torch.meshgrid(o, o, o, indexing="ij"), -1).reshape(-1, 3)
        xg = X / dx
        base = torch.floor(xg - 0.5)
        fx = xg - base
        w = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], 1)
        idx = base.long()[:, None, :] + off[None]
        for d in range(3):
            idx[..., d] = idx[..., d].clamp(0, shape[d] - 1)
        flat = (idx[..., 0] * shape[1] + idx[..., 1]) * shape[2] + idx[..., 2]
        W = w[:, off[:, 0], 0] * w[:, off[:, 1], 1] * w[:, off[:, 2], 2]
        v = (P.p_vol.to(dt_) if getattr(P, "p_vol", None) is not None else torch.ones(X.shape[0], device=dev, dtype=dt_))
        v = v * live.to(dt_)
        phi = torch.zeros(Nw, device=dev, dtype=dt_).index_add_(0, flat.reshape(-1), (W * (v / dx ** 3)[:, None]).reshape(-1))
        phi_i = (W * phi[flat]).sum(1)
        ok = live & (phi_i < self.shell)
        if self.skip_block and self.skip_block in C.state_schema:
            blk = C.state[:, C.state_schema[self.skip_block][0]]
            ok = ok & (blk[cid.clamp(0, nC - 1)] != 0)
        si = torch.nonzero(ok).reshape(-1).cpu().numpy()
        Xs = X[ok].detach().cpu().numpy().astype(np.float64)
        cs = cid[ok].cpu().numpy()
        pairs = cKDTree(Xs).query_pairs(self.reach, output_type="ndarray")
        pairs = pairs[cs[pairs[:, 0]] != cs[pairs[:, 1]]]                 # only between DIFFERENT cells
        i = torch.as_tensor(si[pairs[:, 0]], device=dev, dtype=torch.long)
        j = torch.as_tensor(si[pairs[:, 1]], device=dev, dtype=torch.long)
        L0 = (X[j] - X[i]).norm(dim=1).detach()
        self.bonds = (i, j, L0)
        print(f"[apical_junction] {len(i):,} bonds between {len(np.unique(cs[pairs.ravel()])) if len(pairs) else 0} "
              f"surface cells ({int(ok.sum()):,} surface points, reach {self.reach:g})", flush=True)

    def forward(self, H, mask=None):
        P = H.level(self.at)
        X = P.get("pos")
        if self.bonds is None:
            self._build(H, P, X)
        i, j, L0 = self.bonds
        acc = torch.zeros_like(X)
        if len(i) == 0:
            return {self.at: acc}
        d = X[j] - X[i]
        L = d.norm(dim=1).clamp_min(1e-9)
        F = (self.k * (L - L0) / L)[:, None] * d                           # pulls i toward j when stretched
        m = (P.mass.to(X.dtype) if getattr(P, "mass", None) is not None else torch.ones(X.shape[0], device=X.device, dtype=X.dtype))
        acc.index_add_(0, i, F / m[i, None])
        acc.index_add_(0, j, -F / m[j, None])
        if mask is not None:
            acc = acc * mask[:, None].to(X.dtype)
        return {self.at: acc}


@register_operator("active_rotation", family="mechanics", set="particle", kind="lateral",
                   title="rotate the vegetal mass",
                   equation=r"""$$\mathbf a_i=k\,\frac{\hat{\mathbf a}\times(\mathbf x_i-\mathbf c)}{\lVert\hat{\mathbf a}\times(\mathbf x_i-\mathbf c)\rVert},\qquad \hat{\mathbf a}=\hat{\mathbf d}\times\hat{\mathbf z}$$""")
class ActiveRotation(Lateral):
    r"""vegetal rotation: at the onset of gastrulation the vegetal endoderm cells migrate actively so that the whole vegetal mass rotates about the dorsal-ventral axis, a force per point tangent to that rotation.
    particle -> particle: reads the points' positions and the cell block `block`; points of cells whose block equals `value` (vegetal) get the acceleration (`mpm_acceleration`).

        \mathbf a_i=k\,\frac{\hat{\mathbf a}\times(\mathbf x_i-\mathbf c)}{\lVert\hat{\mathbf a}\times(\mathbf x_i-\mathbf c)\rVert},\qquad \hat{\mathbf a}=\hat{\mathbf d}\times\hat{\mathbf z}
        a_i            the acceleration of point i
        k              `k`, its magnitude, per time squared times length
        \hat a         the rotation axis, the dorsal direction crossed with the animal axis
        x_i            the point's position
        c              `centre`, the embryo's centre
        \hat d         `dorsal`, the dorsal direction
        \hat z         the animal axis

    Reference: Winklbauer, R. & Schurfeld, M. (1999). Vegetal rotation, a new gastrulation movement involved in the internalization of the mesoderm and endoderm in Xenopus. Development 126:3703-3713; Wen, J. W. H. & Winklbauer, R. (2017). eLife 6:e27190."""
    PARAM_UNITS = {'centre': 'length', 'k': 'acceleration', 'value': 'count'}
    SYMBOLS = {'a_i': 'the acceleration of point i', 'k': '`k`, its magnitude, per time squared times length', '\\hat a': 'the rotation axis, the dorsal direction crossed with the animal axis', 'x_i': "the point's position", 'c': "`centre`, the embryo's centre", '\\hat d': '`dorsal`, the dorsal direction', '\\hat z': 'the animal axis'}
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["cells", "k", "centre"]
    MECHANISM_TAGS = ["vegetal_rotation", "active_migration", "gastrulation"]
    PARAM_ROLES = {"cells": "the_body_set_containing_the_points", "k": "tangential_acceleration",
                   "centre": "rotation_centre_world", "dorsal": "dorsal_direction", "block": "cell_block_selecting_cells",
                   "value": "block_value_of_the_rotating_cells"}
    REFERENCE = "Winklbauer, R. & Schurfeld, M. (1999). Development 126:3703-3713."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.cells = str(params["cells"])
        self.k = float(params["k"])
        self.c = [float(v) for v in params["centre"]]
        d = np.array([float(v) for v in params.get("dorsal", [1.0, 0.0, 0.0])])
        d[2] = 0.0
        d = d / np.linalg.norm(d)
        self.axis = np.cross(d, np.array([0.0, 0.0, 1.0]))
        self.block = str(params.get("block", "zone"))
        self.value = self.tunable(params.get("value"), 2.0, tensor=True)

    def forward(self, H, mask=None):
        P = H.level(self.at)
        X = P.get("pos")
        dev, dt_ = X.device, X.dtype
        C = H.level(self.cells)
        nC = int(C.state.shape[0])
        cid = H.lift_index(self.at, self.cells)
        blk = C.state[:, C.state_schema[self.block][0]]
        sel = (torch.abs(blk[cid.clamp(0, nC - 1)] - self.value) < 0.5) & (P.occ > 0)
        a = torch.as_tensor(self.axis, device=dev, dtype=dt_)
        r = X - torch.as_tensor(self.c, device=dev, dtype=dt_)
        t = torch.cross(a.expand_as(r), r, dim=1)
        t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-9)
        acc = self.k * t * sel[:, None].to(dt_)
        if mask is not None:
            acc = acc * mask[:, None].to(dt_)
        return {self.at: acc}
