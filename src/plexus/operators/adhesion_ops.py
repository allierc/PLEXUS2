"""exp27 (cell adhesion testbench): face-specific adhesion between cells made of MPM points, with the adhesion
held as a state SHARED by the two cells -- a `junction` set, one row per touching cell pair and face pair.

Cells are point clouds on their own grid blocks (`per_parent`, exp21 / exp23 / exp25): no grid block sees another
cell, so everything between two cells is an operator. `pair_potential[volume]` keeps them out of each other (with
`adhesion: 0` here, it only repels); this module adds what holds them together, and where.

    seed_positions[cell_grid]   arrays of cube-shaped cells; per point its FACE (lateral / apical / basal, from the
                                cell's polarity axis), per cell its adhesion coefficients and whether it is pulled
    contact_finder            structural, every frame: which cell pairs touch through which faces -- wakes a
                                `junction` row for a new contact, retires it once the contact is gone; writes its
                                contact area and the load it carried; builds the point-pair list the forces use
    bond_binding           lateral, every frame: d rho / dt, the bound-adhesion density of each junction --
                                binding toward the face pair's maximum while in contact, load-accelerated unbinding
                                (Bell 1978)
    bond_springs               exchange, every MPM substep: the forces between points of two touching cells,
                                weighted by their junction's rho -- a tether beyond the rest distance and a friction
                                against sliding, equal and opposite on the two points
    cell_pull                   lateral, every MPM substep: drags a marked cell after a target moving at constant speed

WHY THE JUNCTION IS A SET, NOT A STORE INSIDE AN OPERATOR. `junction_myosin` keys its state by vertex pair in a
private store on the mesh, so it survives T1s and divisions -- but nothing records it. A declared set is recorded
row by row in the trajectory (`junction__rho`, `junction__area`, ...), so the shared state can be plotted, measured
and checked like any other; its rows carry the two cells' indices as blocks (`ca`, `cb`), the way a half-edge
carries its face, because a Plexus set has ONE parent and a junction belongs to two cells.

WHY PAIRWISE FORCES WITH THE MEMORY ON THE JUNCTION. Springs pinned to material points (`apical_junction`, exp25)
break as soon as the material flows past them; here the point pairs are found again every frame and the history --
how much adhesion two cells have built, and lost under load -- lives on the cell pair.

Face-pair classes (unordered): LL 0, AA 1, BB 2, AB 3, LA 4, LB 5.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

from plexus.models.base import Exchange, Lateral, Seed, Structural
from plexus.models.registry import register_operator
from plexus.operators.cleavage_ops import _set_block
from plexus.operators.mpm_ops import _mpm_label_points, _mpm_reset_points

FACE_L, FACE_A, FACE_B = 0, 1, 2
FG_NAMES = ["LL", "AA", "BB", "AB", "LA", "LB"]
_FG = np.array([[0, 4, 5], [4, 1, 3], [5, 3, 2]])          # (face_i, face_j) -> unordered class


def _fg_class(fi, fj):
    return torch.as_tensor(_FG, device=fi.device)[fi.long(), fj.long()]


def ball_lattice_points(off, n, stretch=None):
    """Indices of the `n` points of the lattice `off` [m, 3] (offsets from the cell centre) nearest the centre, in lattice order: a ball of n points, the shape of `seed_positions[cell_grid]` with
    `ball` (exp39). With `stretch` [3] the distance is measured in units of the stretch along each axis: an ELLIPSOID of the same n points with semi-axes in the ratio of the stretch. Ties at the surface
    are broken by lattice order, so the choice is deterministic."""
    if n > len(off):
        raise ValueError(f"ball_lattice_points: a ball of {n} points needs a lattice of at least {n} points, got {len(off)}")
    d = np.linalg.norm(off if stretch is None else off / np.asarray(stretch, float), axis=1)
    return np.sort(np.argsort(d, kind="stable")[:n])


@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="cell_grid",
                   title="lay out arrays of cube cells", equation=r"""$$\mathbf x_{c,(a,b,d)}=\mathbf o_c+s\,(a,b,d),\quad a,b,d=0\ldots\tfrac{L}{s}-1,\qquad \mathbf o_c=\mathbf o+\big(i_x(L+g)+a_x\,\Delta,\ 0,\ i_z(L+g)+a_z\,\Delta\big)$$""")
class SeedCellGrid(Seed):
    r"""the testbench layout: arrays of cubes of material points in the x-z plane, one cell deep in y, cells touching, arrays apart, each cell's points on a cubic lattice with its face labels and adhesion classes.
    pt -> pt, cell: each cell's point block filled with a cubic lattice of `point_spacing` (rest volume spacing^3 per point), per point `pid`, `cellc`, `face`; per cell `pos`, `pol`, `adh` (6 face-pair classes), `pull`. With `live_per_cell` the cube fills only the head of a larger point block and the tail stays dormant for cell_populate; `cell_values` writes one value per cell into a cell block. With `cloud` each cell's points are instead a random subset of `live_per_cell` sites of a cubic lattice of spacing `cloud` jittered by `cloud_jitter` spacings (0.15 is the draw `seed_positions[vesicle]` makes for its cells), not a regular lattice; `cloud_clear` keeps them that far from the faces that touch another cell of the layout, for a skin laid there; with `box` ([L_x, L_y, L_z]) the cloud fills that box instead of the cube, a mother of twice the length that divides into two cubes.

        \mathbf x_{c,(a,b,d)}=\mathbf o_c+s\,(a,b,d),\quad a,b,d=0\ldots\tfrac{L}{s}-1,\qquad \mathbf o_c=\mathbf o+\big(i_x(L+g)+a_x\,\Delta,\ 0,\ i_z(L+g)+a_z\,\Delta\big)
        x_{c,(a,b,d)}  the lattice point (a, b, d) of cell c
        o_c            the cell's corner
        s              `point_spacing`
        L              `side`, the cube's side
        g              `gap` between cells (default 0, touching)
        (i_x, i_z)     the cell's column and row in its array (`cells` = [m_x, m_z])
        (a_x, a_z)     the array's index (`arrays` = [n_x, n_z])
        \Delta         `array_spacing`
        o              `centre`, the layout's origin

    Reference: Plexus exp27, the cell adhesion testbench (this work, 2026-10); Cedric, 2026-10-14."""
    PARAM_UNITS = {'stretch': 'dimensionless', 'array_values': 'dimensionless', 'ball': 'count', 'ball_step': 'length', 'array_spacing': 'length', 'gap': 'length', 'point_spacing': 'length', 'side': 'length', 'centre': 'length', 'layer_adhesion': 'dimensionless', 'layer_tension': 'dimensionless', 'vacant': 'count', 'cloud': 'length', 'cloud_clear': 'length', 'cloud_jitter': 'fraction', 'seed': 'count', 'box': 'length'}
    SYMBOLS = {'x_{c,(a,b,d)}': 'the lattice point (a, b, d) of cell c', 'o_c': "the cell's corner", 's': '`point_spacing`', 'L': "`side`, the cube's side", 'g': '`gap` between cells (default 0, touching)', '(i_x, i_z)': "the cell's column and row in its array (`cells` = [m_x, m_z])", '(a_x, a_z)': "the array's index (`arrays` = [n_x, n_z])", '\\Delta': '`array_spacing`', 'o': "`centre`, the layout's origin"}
    REQUIRES_PARAMS = ["cell_set", "centre"]
    SUPPORTED_DIMS = [3]
    PARAM_ROLES = {"stretch": "axis_ratios_of_an_ellipsoidal_start", "array_values": "cell_blocks_each_with_one_value_per_array", "ball": "points_per_cell_of_a_round_cell_the_nearest_to_its_centre", "ball_step": "centre_to_centre_distance_of_round_cells",
                   "cell_set": "the_cell_set", "centre": "centre_of_the_whole_layout", "side": "cube_edge_world",
                   "point_spacing": "lattice_spacing_world", "cells": "cells_per_array_x_z",
                   "arrays": "arrays_x_z", "array_spacing": "array_centre_spacing_x_z", "gap": "gap_between_cells",
                   "adhesion": "per_array_face_pair_adhesion", "pull": "per_array_pulled_cell_and_direction",
                   "layer_adhesion": "per_layer_face_pair_adhesion", "vacant": "grid_slots_left_empty",
                   "layer_tension": "per_layer_cortical_tension_factor", "live_per_cell": "points_laid_live_in_a_larger_block",
                   "cell_values": "block_and_one_value_per_cell_in_layout_order", "cloud": "fine_lattice_spacing_the_random_subset_is_drawn_from",
                   "cloud_clear": "depth_free_of_cloud_points_at_faces_touching_a_neighbour", "cloud_jitter": "jitter_of_the_fine_lattice_in_spacings", "seed": "rng_seed_of_the_cloud", "box": "edges_of_the_box_the_cloud_fills"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.cell_set = str(params["cell_set"])
        self.c = np.array([float(v) for v in params["centre"]])
        # `cloud` (exp38, 2026-11-07, Cedric: "that would end up on a regular lattice again"): each cell's points are a random subset of `live_per_cell` sites of a cubic lattice of spacing `cloud` jittered by 0.15 spacings, the
        # way `seed_positions[vesicle]` draws the points of its cells, so the body is a cloud and not a lattice; the cube is kept (flat faces), `cloud_clear` leaves the faces that touch a neighbour free for a skin
        self.cloud = None if params.get("cloud") is None else float(params["cloud"])
        self.cloud_clear = float(params.get("cloud_clear", 0.0))
        self.cloud_jitter = float(params.get("cloud_jitter", 0.15))                  # 0.15 as the vesicle draws; 0.5 leaves no layers of the lattice to line up with a view along an axis
        self.rng_seed = int(params.get("seed", 0))
        # `box` (exp38 division rig, 2026-11-07): the cloud of each cell fills this box [L_x, L_y, L_z] instead of the cube of `side`, so a MOTHER of twice the length can divide into two cubes
        self.box = None if params.get("box") is None else np.array([float(v) for v in params["box"]])
        if self.box is not None and not self.cloud:
            raise ValueError("seed_positions[cell_grid]: `box` is a box of the cloud, set `cloud` too")
        # `live_per_cell` (2026-10-28, the minimal growth-and-division rig): any value turns it on -- the cube is laid in the HEAD of each cell's point block and the tail
        # (`grow_reserve` slots) stays dormant for `cell_populate` to wake, as `seed_positions[vesicle]` does; absent, the block must be exactly the cube
        self.live_per_cell = params.get("live_per_cell")
        # `cell_values: {block: <cell block>, values: [v_0, v_1, ...]}` (2026-10-28, the overlap sweep): a width-1 cell block written with one value per cell in the layout's
        # order (layers bottom first, then rows in y, then columns in x: the order of the cells' slots), so a SWEEP of a per-cell knob is laid on the array at the seed
        self.cell_values = params.get("cell_values")
        # `ball: N` (exp39, 2026-11-07): each cell is a BALL, the N lattice points of the cube of `side` nearest its centre (a rounded cell at the start, so a doublet's equilibrium shape is
        # reached by flattening one contact and not by rounding two cubes), and neighbouring cells' centres are `ball_step` apart (default `side`); absent, the cube
        # `array_values: {block: [v_0 ... v_{n-1}]}` (exp39, 2026-11-07, as seed_positions[rosette]): a width-1 cell block written with ONE value per ARRAY on every cell of that array (a sweep laid on the arrays);
        # an entry `{values: [...], first_only: true}` writes it on the array's first cell only and NaN on the others (a label printed once over a doublet)
        self.array_values = params.get("array_values") or {}
        self.ball = None if params.get("ball") is None else int(params["ball"])
        self.ball_step = None if params.get("ball_step") is None else float(params["ball_step"])
        self.stretch = None if params.get("stretch") is None else [float(v) for v in params["stretch"]]      # exp39: the cell starts as an ellipsoid (axes in this ratio, volume kept by the point count)
        self.side = float(params.get("side", 1.0))
        self.h = float(params.get("point_spacing", 0.125))
        cl = [int(v) for v in params.get("cells", [3, 3])]
        # `cells: [m_x, m_z]` (one cell deep in y, the testbench) or `[m_x, m_y, m_z]` (a SHEET, m_z layers of m_x x m_y
        # cells -- the two-layer epithelium, 2026-10-14). Cells are laid bottom layer first, then y, then x, so with
        # `type_layout: ordered` the first m_x m_y cells are the bottom layer's type.
        self.cells = [cl[0], 1, cl[1]] if len(cl) == 2 else cl
        # `layer_adhesion`: one mapping per layer (z index), overriding the array's `adhesion` for that layer's cells
        la = params.get("layer_adhesion")
        self.layer_adh = None if la is None else [[float((a or {}).get(k, 0.0)) for k in FG_NAMES] for a in la]
        # `layer_tension`: one factor per layer (z index) written to the cell block `tension`, the per-cell factor on the
        # surface tension of `mpm_grid_update` (`cell_surface_tension_block: tension`) -- outer cells rounder or less round than inner
        lt = params.get("layer_tension")
        self.layer_tension = None if lt is None else [float(v) for v in lt]
        self.arrays = [int(v) for v in params.get("arrays", [4, 2])]
        self.asp = [float(v) for v in params.get("array_spacing", [8.0, 8.0])]
        self.gap = float(params.get("gap", 0.0))
        na = self.arrays[0] * self.arrays[1]
        adh = params.get("adhesion") or {}
        adh = [adh] * na if isinstance(adh, dict) else list(adh)
        if len(adh) != na:
            raise ValueError(f"seed_positions[cell_grid]: {len(adh)} adhesion entries for {na} arrays")
        self.adh = [[float((a or {}).get(k, 0.0)) for k in FG_NAMES] for a in adh]
        pull = params.get("pull") or [None] * na
        if len(pull) != na:
            raise ValueError(f"seed_positions[cell_grid]: {len(pull)} pull entries for {na} arrays")
        self.pull = pull
        # `vacant: [[i, j, k], ...]` (sheets, 2026-10-15): grid slots (column, row, layer) left without a cell -- holes
        # in a layer for its cells to fill; the cell set then has that many fewer live cells than the grid has slots
        self.vacant = {tuple(int(v) for v in t) for t in (params.get("vacant") or [])}

    def forward(self, H, mask=None):
        P, C = H.level(self.at), H.level(self.cell_set)
        m = int(round(self.side / self.h))
        npts = int(self.live_per_cell) if self.cloud else (self.ball if self.ball else m ** 3)
        mx, my, mz = self.cells
        ax_n, az_n = self.arrays
        ncell = mx * my * mz * ax_n * az_n - len(self.vacant) * ax_n * az_n
        live = np.flatnonzero(C.occ.detach().cpu().numpy() > 0.5)
        if len(live) != ncell:
            raise ValueError(f"seed_positions[cell_grid]: {len(live)} live cells, the layout needs {ncell}")
        parent = P.parent.detach().cpu().numpy()
        dev, dt_ = P.state.device, P.state.dtype
        g = (np.arange(m) + 0.5) * self.h - 0.5 * self.side                      # lattice offsets along one axis
        off = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        u = np.abs(off) / (0.5 * self.side)
        face = np.full(len(off), FACE_L, np.float64)
        zdom = (u[:, 2] > u[:, 0]) & (u[:, 2] > u[:, 1])
        face[zdom & (off[:, 2] > 0)] = FACE_A
        face[zdom & (off[:, 2] < 0)] = FACE_B
        if self.ball:
            keep = ball_lattice_points(off, self.ball, self.stretch)
            off, face = off[keep], face[keep]
        occ = torch.zeros_like(P.occ)
        cen = C.state[:, C.state_schema["pos"][0]:C.state_schema["pos"][1]].detach().cpu().numpy().copy()
        step = self.ball_step if (self.ball and self.ball_step) else self.side + self.gap
        rows = []
        n = 0
        for az in range(az_n):
            for ax in range(ax_n):
                a = az * ax_n + ax
                acen = self.c + np.array([(ax - 0.5 * (ax_n - 1)) * self.asp[0], 0.0, (az - 0.5 * (az_n - 1)) * self.asp[1]])
                pl = self.pull[a]
                for k in range(mz):
                  for jy in range(my):
                    for i in range(mx):
                        if (i, jy, k) in self.vacant:
                            continue
                        j = live[n]; n += 1
                        cen[j] = acen + np.array([(i - 0.5 * (mx - 1)) * step, (jy - 0.5 * (my - 1)) * step,
                                                  (k - 0.5 * (mz - 1)) * step])
                        blk = np.flatnonzero(parent == j)
                        if self.live_per_cell is not None and len(blk) >= npts:
                            blk = blk[:npts]
                        if len(blk) != npts:
                            raise ValueError(f"seed_positions[cell_grid]: cell {j} has {len(blk)} point slots, a cube "
                                             f"of {m}^3 needs {npts} -- set the point set's per_parent to {npts}")
                        idx = torch.as_tensor(blk, device=dev)
                        off_j, face_j, vol_j = off, face, self.h ** 3
                        if self.cloud:                                          # a random subset of a jittered fine lattice over the cube, minus a slab at each face that touches a neighbour of the layout
                            from plexus.operators.gastrulation_ops import _lattice
                            half = 0.5 * self.box if self.box is not None else np.full(3, 0.5 * self.side)
                            lo = -half.copy(); hi = half.copy()
                            for ax_, (ii, nn) in enumerate(((i, mx), (jy, my), (k, mz))):
                                if ii > 0:
                                    lo[ax_] += self.cloud_clear
                                if ii < nn - 1:
                                    hi[ax_] -= self.cloud_clear
                            sites = _lattice(self.cloud, lo, hi, np.random.default_rng((self.rng_seed, int(j))), jitter=self.cloud_jitter)
                            if len(sites) < npts:
                                raise ValueError(f"seed_positions[cell_grid]: the cloud lattice of spacing {self.cloud:g} has {len(sites)} sites for cell {j}, {npts} points are asked")
                            off_j = sites[np.random.default_rng((self.rng_seed, int(j), 1)).choice(len(sites), npts, replace=False)]
                            vol_j = float(np.prod(hi - lo)) / npts                 # the points share the volume of the region they fill
                            u_j = np.abs(off_j) / half; face_j = np.full(npts, FACE_L, np.float64); zd = (u_j[:, 2] > u_j[:, 0]) & (u_j[:, 2] > u_j[:, 1])
                            face_j[zd & (off_j[:, 2] > 0)] = FACE_A; face_j[zd & (off_j[:, 2] < 0)] = FACE_B
                        _mpm_reset_points(P, idx, torch.as_tensor(cen[j] + off_j, device=dev, dtype=dt_),
                                          torch.full((npts,), vol_j, device=dev, dtype=dt_))
                        _set_block(P, "pid", idx, float(j)); _set_block(P, "cellc", idx, float(j % 20))
                        _set_block(P, "face", idx, torch.as_tensor(face_j, device=dev, dtype=dt_))
                        occ[idx] = 1.0
                        pulled = pl is not None and [int(v) for v in pl["cell"]] == [i, k]
                        rows.append((j, a, i, k, pulled, pl))
        P.occ = occ
        ca, cb = C.state_schema["pos"]
        st = C.state.clone(); st[:, ca:cb] = torch.as_tensor(cen, dtype=st.dtype, device=st.device); C.state = st
        for j, a, i, k, pulled, pl in rows:
            jj = torch.as_tensor([j], device=C.state.device)
            if "adh" in C.state_schema:
                a0, a1 = C.state_schema["adh"]
                row_adh = self.layer_adh[k] if self.layer_adh is not None else self.adh[a]
                st = C.state.clone(); st[j, a0:a1] = torch.as_tensor(row_adh, dtype=st.dtype); C.state = st
            if "pull" in C.state_schema and pulled:
                d = np.array([float(v) for v in pl["dir"]]); d = d / max(np.linalg.norm(d), 1e-12)
                p0, p1 = C.state_schema["pull"]
                st = C.state.clone(); st[j, p0:p1] = torch.as_tensor(d, dtype=st.dtype); C.state = st
            if "pol" in C.state_schema:
                q0, q1 = C.state_schema["pol"]
                st = C.state.clone(); st[j, q0:q1] = torch.as_tensor([0.0, 0.0, 1.0], dtype=st.dtype); C.state = st
            if self.layer_tension is not None and "tension" in C.state_schema:
                _set_block(C, "tension", jj, self.layer_tension[k])
            _set_block(C, "arr", jj, float(a))
            for blk, av in self.array_values.items():
                vals_a, first_only = (av["values"], bool(av.get("first_only"))) if isinstance(av, dict) else (av, False)
                _set_block(C, str(blk), jj, float("nan") if (first_only and not (i == 0 and k == 0)) else float(vals_a[a]))
            if self.cell_values is not None:
                vals = [float(v) for v in self.cell_values["values"]]
                if len(vals) != ncell:
                    raise ValueError(f"seed_positions[cell_grid]: cell_values has {len(vals)} values for {ncell} cells")
                _set_block(C, str(self.cell_values["block"]), jj, vals[int(np.flatnonzero(live == j)[0])])
            if "ij" in C.state_schema:
                i0, _ = C.state_schema["ij"]
                st = C.state.clone(); st[j, i0] = float(i); st[j, i0 + 1] = float(k); C.state = st
            _set_block(C, "tone", jj, 1.0 if pulled else (0.15 if (i + k) % 2 == 0 else 0.3))
        print(f"[seed_positions:cell_grid] {ax_n} x {az_n} arrays of {mx} x {my} x {mz} cubes (side {self.side:g}, "
              f"{npts} points each, spacing {self.h:g}): {ncell} cells, {int(occ.sum()):,} points; "
              f"{sum(1 for r in rows if r[4])} pulled cells", flush=True)
        return {}


@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="face_layers",
                   title="add dense layers of points on the facing face of two cells",
                   equation=r"""$$\mathbf x_{c,l,(a,b)}=\mathbf o_c+s_c\,(\tfrac L2-\delta-l\,p)\,\hat{\mathbf e}_k+p\,(a+\tfrac12-\tfrac m2)\,\hat{\mathbf e}_\perp+p\,(b+\tfrac12-\tfrac m2)\,\hat{\mathbf e}_{\perp'},\qquad m=\tfrac Lp$$""")
class SeedFaceLayers(Seed):
    r"""adds a dense skin of points to the face of each of two cells that looks at the other: `layers` square lattices of pitch `pitch` covering the face, woken from the dormant tail of each cell's point block (laid by `cell_grid` with `live_per_cell`).
    pt -> pt: wakes `layers` x (side / pitch)^2 points per cell, each with volume pitch^3 and mass density x volume x `mass_scale`, `pid` of its cell.

        x_{c,l,(a,b)}=o_c+s_c (L/2 - delta - l p) e_k + p (a + 1/2 - m/2) e_perp + p (b + 1/2 - m/2) e_perp',   m = L / p
        o_c            the centre of cell c
        s_c            +1 for the cell on the low side along the axis (its high face looks at the other cell), -1 for the other
        L              `side`, the cell's edge;  p `pitch`, the lattice pitch of the skin;  delta `depth`, the first layer's distance inside the face (default p / 2);  l the layer, 0 ... `layers` - 1
        e_k            the unit vector of `axis` (the line of the two cells' centres)

    Reference: Plexus exp38 (this work, 2026-11)."""
    PARAM_UNITS = {'side': 'length', 'pitch': 'length', 'depth': 'length', 'layers': 'count', 'axis': 'count', 'mass_scale': 'dimensionless'}
    SYMBOLS = {'o_c': 'the centre of cell c', 's_c': 'the side of cell c facing the other cell', 'L': '`side`, the cell edge', 'p': '`pitch`, the skin lattice pitch', 'delta': '`depth`, the first layer inside the face', 'e_k': 'the unit vector of the axis between the cells'}
    REQUIRES_PARAMS = ["cell_set", "side", "pitch"]
    SUPPORTED_DIMS = [3]
    PARAM_ROLES = {"cell_set": "the_cell_set", "side": "cell_edge_world", "pitch": "skin_lattice_pitch_world", "layers": "number_of_dense_layers", "depth": "first_layer_depth_inside_the_face", "axis": "axis_between_the_cells_0_1_2", "mass_scale": "mass_factor_of_the_skin_points"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt"); self.cell_set = str(params["cell_set"]); self.side = float(params["side"]); self.pitch = float(params["pitch"])
        self.layers = int(params.get("layers", 3)); self.depth = float(params.get("depth", 0.5 * self.pitch)); self.axis = int(params.get("axis", 0)); self.mass_scale = float(params.get("mass_scale", 1.0))

    def forward(self, H, mask=None):
        P, C = H.level(self.at), H.level(self.cell_set)
        live = torch.nonzero(C.occ > 0.5).reshape(-1)
        if live.numel() != 2:
            raise ValueError(f"seed_positions[face_layers]: {live.numel()} live cells, this skin is for two")
        a0, a1 = C.state_schema["pos"]; cen = C.state[:, a0:a1].detach().cpu().numpy()
        m = int(round(self.side / self.pitch)); g = (np.arange(m) + 0.5) * self.pitch - 0.5 * self.side
        k = self.axis; o1, o2 = [d for d in range(3) if d != k]
        parent = P.parent.detach().cpu().numpy(); occ = P.occ.clone(); dev, dt_ = P.state.device, P.state.dtype
        order = sorted(live.tolist(), key=lambda j: cen[j, k])
        for j, sgn in ((order[0], 1.0), (order[1], -1.0)):
            tail = np.flatnonzero((parent == j) & (P.occ.detach().cpu().numpy() < 0.5)); n_new = self.layers * m * m
            if len(tail) < n_new:
                raise ValueError(f"seed_positions[face_layers]: cell {j} has {len(tail)} dormant slots, the skin needs {n_new} -- set per_parent to the cube's points plus {n_new}")
            idx = torch.as_tensor(tail[:n_new], device=dev)
            pos = np.zeros((self.layers, m, m, 3))
            for l in range(self.layers):
                pos[l, :, :, k] = sgn * (0.5 * self.side - self.depth - l * self.pitch)
                pos[l, :, :, o1] = g[:, None]; pos[l, :, :, o2] = g[None, :]
            _mpm_reset_points(P, idx, torch.as_tensor(cen[j] + pos.reshape(-1, 3), device=dev, dtype=dt_), torch.full((n_new,), self.pitch ** 3, device=dev, dtype=dt_))
            if self.mass_scale != 1.0 and getattr(P, "mass", None) is not None:
                P.mass[idx] = P.mass[idx] * self.mass_scale
            _set_block(P, "pid", idx, float(j)); occ[idx] = 1.0
        P.occ = occ
        print(f"[seed_positions:face_layers] {self.layers} layers of {m} x {m} points (pitch {self.pitch:g}) on the facing face of each cell: {self.layers * m * m:,} points per cell", flush=True)
        return {}


@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="rosette",
                   title="lay out arrays of one rounded inner cell inside a ring of squarish outer cells",
                   equation=r"""$$\mathbf c_{a,j}=\mathbf o_a+R\,(\cos\theta_j,\,0,\,\sin\theta_j),\quad \theta_j=\tfrac{2\pi j}{n},\qquad R=\min\{R:\ d(\mathrm{cell}_j,\mathrm{cell}_{j+1})\ge s\}$$""")
class SeedRosette(Seed):
    r"""the adhesion rosette testbench (exp29, 2026-10-31): in each array one rounded INNER cell (the 125 lattice points nearest its centre) sits at the middle of a ring of n squarish OUTER cells (cubes of points, each turned so its faces follow the radius), all the cells of an array in one plane (x, z), one cell deep in y.
    pt -> pt, cell: each cell's point block filled with its lattice, per point `pid`, `cellc`, `face` (all lateral); per cell `pos`, `pol` (the out-of-plane axis y, so every in-plane contact is lateral), `adh`, `arr`, `ij` (ring position, 0 = inner), `tone` and the `array_values` blocks.

        \mathbf c_{a,j}=\mathbf o_a+R\,(\cos\theta_j,\,0,\,\sin\theta_j),\quad \theta_j=\tfrac{2\pi j}{n},\qquad R=\min\{R:\ d(\mathrm{cell}_j,\mathrm{cell}_{j+1})\ge s\}
        c_{a,j}        the centre of outer cell j of array a
        o_a            the centre of array a: `centre` plus its place in the `arrays` grid, `array_spacing` apart
        R              the ring radius: the smallest at which two neighbouring cells' nearest points are one point spacing apart (plus `ring_gap`)
        \theta_j       the angle of outer cell j, n = `n_outer` cells round the ring
        d              the smallest distance between the points of two cells
        s              `point_spacing`

    Reference: Plexus exp29, the adhesion rosette testbench (this work, 2026-10-31)."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"Lays out the rosette testbench once, at the start. In each array one rounded inner cell (the 125 lattice points "
        r"nearest its centre) sits in the middle of a ring of $n$ squarish outer cells (cubes of points, each turned so its "
        r"faces follow the radius). The arrays stand on a grid, `array_spacing` apart, and each array carries its own value of "
        r"the swept parameter (`labels`). The ring radius $R$ is the smallest at which two neighbouring outer cells' nearest "
        r"points are one point spacing apart: $\mathbf c_{a,j}=\mathbf o_a+R\,(\cos\theta_j,\,0,\,\sin\theta_j),\ "
        r"\theta_j=\tfrac{2\pi j}{n}$, where $\mathbf o_a$ is the centre of array $a$. It writes each cell's points and blocks "
        r"(position, adhesion maxima, array index, ring position, 0 being the inner cell); after frame 0 it does nothing."
    )
    PARAM_UNITS = {'array_spacing': 'length', 'centre': 'length', 'inner_spacing': 'length', 'point_spacing': 'length', 'ring_gap': 'length', 'side': 'length', 'n_outer': 'count', 'live_per_cell': 'count', 'snapshot': 'path', 'labels': 'dimensionless', 'inner_values': 'dimensionless', 'outer_values': 'dimensionless'}
    SYMBOLS = {'c_{a,j}': 'the centre of outer cell j of array a', 'o_a': 'the centre of array a', 'R': 'the ring radius', '\\theta_j': 'the angle of outer cell j', 'd': 'the smallest distance between the points of two cells', 's': '`point_spacing`'}
    REQUIRES_PARAMS = ["cell_set", "centre"]
    SUPPORTED_DIMS = [3]
    PARAM_ROLES = {"snapshot": "npz_of_a_recorded_array_geometry_used_as_the_seed_of_every_array", "labels": "one_label_value_per_array_on_the_inner_cell", "inner_values": "block_values_for_the_inner_cells_only", "outer_values": "block_values_for_the_outer_cells_only", "cell_set": "the_cell_set", "centre": "centre_of_the_whole_layout", "arrays": "arrays_x_z", "array_spacing": "array_centre_spacing_x_z", "side": "outer_cube_edge_world",
                   "point_spacing": "outer_lattice_spacing_world", "inner_spacing": "inner_lattice_spacing_world", "n_outer": "outer_cells_per_ring", "ring_gap": "extra_ring_radius",
                   "live_per_cell": "points_laid_live_in_a_larger_block", "adhesion": "per_array_face_pair_adhesion", "array_values": "cell_block_to_one_value_per_array", "plane": "sheet_plane_xz_or_xy"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.cell_set = str(params["cell_set"])
        self.c = np.array([float(v) for v in params["centre"]])
        self.arrays = [int(v) for v in params.get("arrays", [3, 3])]
        self.asp = [float(v) for v in params.get("array_spacing", [8.0, 8.0])]
        self.side = float(params.get("side", 1.0))
        self.h = float(params.get("point_spacing", 0.2))
        self.hi = float(params.get("inner_spacing", 0.16))
        self.n_out = int(params.get("n_outer", 7))
        self.ring_gap = float(params.get("ring_gap", 0.0))
        self.live_per_cell = params.get("live_per_cell")
        self.plane = str(params.get("plane", "xz"))                 # the sheet's plane: xz (normal y, the default) or xy (normal z, as the exp29 cell-cycle movies lay theirs)
        if self.plane not in ("xz", "xy"):
            raise ValueError("seed_positions[rosette]: plane is xz or xy")
        na = self.arrays[0] * self.arrays[1]
        adh = params.get("adhesion") or {}
        adh = [adh] * na if isinstance(adh, dict) else list(adh)
        if len(adh) != na:
            raise ValueError(f"seed_positions[rosette]: {len(adh)} adhesion entries for {na} arrays")
        self.adh = [[float((a or {}).get(k, 0.0)) for k in FG_NAMES] for a in adh]
        # `labels`: one number per array written to the block `lab` of its INNER cell only (NaN on the outer cells: the studio labels a family unless its value is NaN, so one label per array);
        # `inner_values` / `outer_values`: {block: value} written to the inner cells only / the outer cells only (e.g. the growth rate: the inner cell grows, the ring does not)
        # `snapshot`: a .npz written by tools/exp29_rosette.py (`rel_pos` [n, 3] points relative to the inner cell's centroid, `cell_of` [n] 0 = inner, 1..n_outer = the ring): every array starts as that
        # recorded geometry instead of the lattice cubes and the blob (each cell keeps its own point count; the ring radius, `side`, `inner_spacing` are then unused)
        self.snapshot = params.get("snapshot")
        self.labels = None if params.get("labels") is None else [float(v) for v in params["labels"]]
        self.inner_values = {str(k): float(v) for k, v in (params.get("inner_values") or {}).items()}
        self.outer_values = {str(k): float(v) for k, v in (params.get("outer_values") or {}).items()}
        self.array_values = params.get("array_values") or {}
        for b_, v_ in self.array_values.items():
            if len(v_) != na:
                raise ValueError(f"seed_positions[rosette]: array_values[{b_}] has {len(v_)} values for {na} arrays")

    @staticmethod
    def _rot(off, theta):
        """Offsets turned about y so the local x axis follows the direction (cos theta, 0, sin theta)."""
        ct, st = np.cos(theta), np.sin(theta)
        return np.stack([off[:, 0] * ct - off[:, 2] * st, off[:, 1], off[:, 0] * st + off[:, 2] * ct], 1)

    def _ring_radius(self, off):
        from scipy.spatial import cKDTree
        th = 2.0 * np.pi / self.n_out
        for R in np.arange(0.5, 6.0, 0.005):
            a = self._rot(off, 0.0) + np.array([R, 0.0, 0.0])
            b = self._rot(off, th) + R * np.array([np.cos(th), 0.0, np.sin(th)])
            if cKDTree(a).query(b)[0].min() >= 0.98 * self.h:
                return float(R) + self.ring_gap
        raise ValueError("seed_positions[rosette]: no ring radius separates the neighbouring cells")

    def forward(self, H, mask=None):
        P, C = H.level(self.at), H.level(self.cell_set)
        m = int(round(self.side / self.h)); npts = m ** 3
        perm = [0, 2, 1] if self.plane == "xy" else [0, 1, 2]       # everything is laid in the xz frame (up = y), then the last two axes are swapped for plane xy
        ax_n, az_n = self.arrays; A = ax_n * az_n; n_out = self.n_out
        live = np.flatnonzero(C.occ.detach().cpu().numpy() > 0.5)
        if len(live) != A * (1 + n_out):
            raise ValueError(f"seed_positions[rosette]: {len(live)} live cells, {A} arrays of 1 + {n_out} need {A * (1 + n_out)}")
        parent = P.parent.detach().cpu().numpy()
        dev, dt_ = P.state.device, P.state.dtype
        g = (np.arange(m) + 0.5) * self.h - 0.5 * self.side
        off_out = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)             # one squarish cell: a cube of points
        gi = (np.arange(7) - 3) * self.hi
        cand = np.stack(np.meshgrid(gi, gi, gi, indexing="ij"), -1).reshape(-1, 3)
        off_in = cand[np.argsort(np.linalg.norm(cand, axis=1), kind="stable")[:npts]]          # the rounded inner cell: the npts lattice points nearest the middle
        snap = None
        if self.snapshot:
            zs = np.load(self.snapshot, allow_pickle=True); rel, co = zs["rel_pos"].astype(np.float64), zs["cell_of"]
            snap = [rel[co == k] for k in range(1 + n_out)]
        R = self._ring_radius(off_out) if snap is None else float("nan")
        occ = torch.zeros_like(P.occ)
        cen = C.state[:, C.state_schema["pos"][0]:C.state_schema["pos"][1]].detach().cpu().numpy().copy()
        inner, outer = live[:A], live[A:].reshape(A, n_out)                                      # `type_layout: ordered`: the inner cells first, then the rings, array by array
        cells_of = {}
        for az in range(az_n):
            for ax in range(ax_n):
                a = az * ax_n + ax
                acen = self.c + np.array([(ax - 0.5 * (ax_n - 1)) * self.asp[0], 0.0, (az - 0.5 * (az_n - 1)) * self.asp[1]])
                if snap is not None:
                    members = [(inner[a], acen, snap[0], self.h, 0.0, 0)] + [(outer[a, j], acen, snap[j + 1], self.h, 0.0, j + 1) for j in range(n_out)]
                else:
                    members = [(inner[a], acen, off_in, self.hi, 0.0, 0)]
                    for j in range(n_out):
                        th = 2.0 * np.pi * j / n_out
                        members.append((outer[a, j], acen + R * np.array([np.cos(th), 0.0, np.sin(th)]), self._rot(off_out, th), self.h, th, j + 1))
                for cj, cc, offs, hh, th, jj_ in members:
                    nc = len(offs)
                    blk = np.flatnonzero(parent == cj)
                    if self.live_per_cell is not None and len(blk) >= nc:
                        blk = blk[:nc]
                    if len(blk) != nc:
                        raise ValueError(f"seed_positions[rosette]: cell {cj} has {len(blk)} point slots, {nc} needed")
                    cen[cj] = (cc + offs.mean(0))[perm]
                    idx = torch.as_tensor(blk, device=dev)
                    _mpm_reset_points(P, idx, torch.as_tensor((cc + offs)[:, perm], device=dev, dtype=dt_), torch.full((nc,), hh ** 3, device=dev, dtype=dt_))
                    _set_block(P, "pid", idx, float(cj)); _set_block(P, "cellc", idx, float(cj % 20)); _set_block(P, "face", idx, float(FACE_L))
                    occ[idx] = 1.0
                    cells_of.setdefault(a, []).append((cj, jj_))
        P.occ = occ
        ca, cb = C.state_schema["pos"]
        st = C.state.clone(); st[:, ca:cb] = torch.as_tensor(cen, dtype=st.dtype, device=st.device); C.state = st
        for a, cl in cells_of.items():
            for cj, jj_ in cl:
                jj = torch.as_tensor([cj], device=C.state.device)
                if "adh" in C.state_schema:
                    a0, a1 = C.state_schema["adh"]; st = C.state.clone(); st[cj, a0:a1] = torch.as_tensor(self.adh[a], dtype=st.dtype); C.state = st
                if "pol" in C.state_schema:
                    q0, q1 = C.state_schema["pol"]; st = C.state.clone(); st[cj, q0:q1] = torch.as_tensor([0.0, 0.0, 1.0] if self.plane == "xy" else [0.0, 1.0, 0.0], dtype=st.dtype); C.state = st
                _set_block(C, "arr", jj, float(a))
                if "ij" in C.state_schema:
                    i0, _ = C.state_schema["ij"]; st = C.state.clone(); st[cj, i0] = float(jj_); st[cj, i0 + 1] = 0.0; C.state = st
                _set_block(C, "tone", jj, 0.15 if jj_ == 0 else 0.3)
                for b_, v_ in self.array_values.items():
                    _set_block(C, str(b_), jj, float(v_[a]))
                for b_, v_ in (self.inner_values if jj_ == 0 else self.outer_values).items():
                    _set_block(C, b_, jj, v_)
                if self.labels is not None and "lab" in C.state_schema:
                    _set_block(C, "lab", jj, self.labels[a] if jj_ == 0 else float("nan"))
        print(f"[seed_positions:rosette] {ax_n} x {az_n} arrays of 1 rounded inner cell + {n_out} squarish outer cells (ring radius {R:.3f}{", from the snapshot " + str(self.snapshot) if snap is not None else ""}, {npts} points each, outer spacing {self.h:g}, inner {self.hi:g}): "
              f"{len(live)} cells, {int(occ.sum()):,} points", flush=True)
        return {}


def _random_rotation(rng):
    """A uniformly random 3 x 3 rotation matrix from a random unit quaternion."""
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def spheroid_sites(n, spacing, lattice="fcc"):
    """The `n` cell centres of one spheroid: the sites of an FCC (or simple cubic) lattice of nearest-neighbour distance `spacing` nearest the origin, ties broken by coordinates (a compact cluster:
    FCC gives 1, 13, 19, 43, 55, 79, 87 sites at the shell boundaries, simple cubic 1, 7, 19, 27)."""
    r = 6
    g = np.arange(-r, r + 1)
    I, J, K = np.meshgrid(g, g, g, indexing="ij")
    P = np.stack([I.ravel(), J.ravel(), K.ravel()], 1)
    if lattice == "fcc":
        P = P[(P.sum(1) % 2) == 0] * (spacing / np.sqrt(2.0))
    elif lattice == "sc":
        P = P * spacing
    else:
        raise ValueError(f"spheroid_sites: lattice is fcc or sc, got {lattice!r}")
    d = np.linalg.norm(P, axis=1)
    order = np.lexsort((P[:, 2], P[:, 1], P[:, 0], np.round(d, 6)))
    if n > len(order):
        raise ValueError(f"spheroid_sites: {n} sites asked, the lattice box holds {len(order)}")
    return P[order[:n]]


def touching_offset(A, B, touch, axis=0):
    """The smallest shift s of the cluster B (centres [m, 3]) along `axis`, relative to A [n, 3] centred at -s/2 and B at +s/2, at which no centre of A is nearer than `touch` to a centre of B."""
    e = np.zeros(3)
    e[axis] = 1.0

    def dmin(sv):
        a, b = A - 0.5 * sv * e, B + 0.5 * sv * e
        return float(np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)))
    lo, hi = 0.0, 4.0 * (np.abs(A).max() + np.abs(B).max()) + 2.0 * touch
    if dmin(lo) >= touch:
        return lo
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if dmin(mid) >= touch:
            hi = mid
        else:
            lo = mid
    return hi


def spheroid_offsets(clusters, touch, layout="chain"):
    """The offsets [3] of each spheroid of an array so that neighbouring spheroids touch and the array is centred on the mean of all cell centres: two spheroids lie on the x axis, a `chain` of 3 or 4
    continues along x (each touching the one before), a `triangle` of 3 puts the third in the x-z plane touching the other two."""
    n = len(clusters)
    offs = [np.zeros(3)]
    ex = np.array([1.0, 0.0, 0.0])
    for k in range(1, n):
        prev = clusters[k - 1] + offs[k - 1]
        if layout == "triangle" and k == 2:
            xm = 0.5 * (offs[0][0] + offs[1][0])
            def dmin(z):
                c = clusters[2] + np.array([xm, 0.0, z])
                return min(float(np.min(np.linalg.norm(c[:, None, :] - (clusters[j] + offs[j])[None, :, :], axis=2))) for j in (0, 1))
            lo, hi = 0.0, 3.0 * (np.abs(clusters[0]).max() + touch) + 2.0 * touch
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if dmin(mid) >= touch:
                    hi = mid
                else:
                    lo = mid
            offs.append(np.array([xm, 0.0, hi]))
        else:
            sep = touching_offset(prev, clusters[k] + offs[k - 1], touch, axis=0)             # the new spheroid starts on the previous one's centre and is shifted along x until the two just touch
            offs.append(offs[k - 1] + sep * ex)
    centre = np.mean([(clusters[k] + offs[k]).mean(0) for k in range(n)], axis=0)
    return [o - centre for o in offs]


@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="spheroid_pair",
                   title="lay out arrays of two touching spheroids of ball cells",
                   equation=r"""$$\mathbf c_{a,s,i}=\mathbf o_a+(-1)^{s+1}\tfrac{\Delta}{2}\,\hat{\mathbf e}_x+\mathbf R_{a,s}\,\mathbf q_i,\qquad \min_{i,j}\lvert\mathbf c_{a,0,i}-\mathbf c_{a,1,j}
vert=d_0$$""")
class SeedSpheroidPair(Seed):
    r"""the fusion testbench layout (exp39): in each array of a 3 x 3 grid two spheroids of `n_cells` ball cells touch along x, each spheroid a compact FCC (or simple cubic) cluster turned at random, every cell a ball of `ball` material points.
    pt -> pt, cell: each cell's point block filled with the `ball` lattice points nearest its centre, per point `pid`, `cellc`, `face`; per cell `pos`, `pol`, `adh` (the array's adhesion in all six face-pair classes), `arr` (the array), `ij` (the spheroid 0 or 1, the cell's index in it), `tone` and the `array_values` blocks.

        \mathbf c_{a,s,i}=\mathbf o_a+(-1)^{s+1}\tfrac{\Delta}{2}\,\hat{\mathbf e}_x+\mathbf R_{a,s}\,\mathbf q_i,\qquad \min_{i,j}\lvert\mathbf c_{a,0,i}-\mathbf c_{a,1,j}\rvert=d_0
        c_{a,s,i}      the centre of cell i of spheroid s (0 left, 1 right) of array a
        o_a            the centre of array a: `centre` plus its place in the `arrays` grid, `array_spacing` apart
        \Delta         the smallest separation at which the two spheroids' nearest cell centres are `spacing` (d_0) apart
        R_{a,s}        the random rotation of spheroid s of array a (`seed`, the array and the spheroid index fix it)
        q_i            the i-th site of the lattice (nearest-neighbour distance `spacing`) nearest the origin
        d_0            `spacing`, the centre-to-centre distance of two touching cells (a little under two radii)

    Reference: Ongenae, S. et al. (2025). Nat. Commun. 16:10467 (two spheroids of about 100 cells brought into contact with random orientations); Plexus exp39 (this work, 2026-11-07)."""
    PARAM_UNITS = {'array_values': 'dimensionless', 'ball': 'count', 'n_cells': 'count', 'n_spheroids': 'count', 'spacing': 'length', 'side': 'length', 'point_spacing': 'length', 'array_spacing': 'length', 'centre': 'length',
                   'jitter': 'fraction', 'seed': 'count'}
    SYMBOLS = {'c_{a,s,i}': 'the centre of cell i of spheroid s (0 left, 1 right) of array a', 'o_a': 'the centre of array a', '\\Delta': "the smallest separation at which the spheroids' nearest cell centres are d_0 apart",
               'R_{a,s}': 'the random rotation of spheroid s of array a', 'q_i': 'the i-th lattice site nearest the origin', 'd_0': 'spacing, the centre-to-centre distance of two touching cells'}
    REQUIRES_PARAMS = ["cell_set", "centre", "n_cells"]
    SUPPORTED_DIMS = [3]
    PARAM_ROLES = {"cell_set": "the_cell_set", "centre": "centre_of_the_whole_layout", "n_cells": "cells_per_spheroid", "ball": "points_per_cell", "side": "lattice_side_the_ball_is_cut_from",
                   "point_spacing": "material_point_lattice_spacing", "spacing": "centre_to_centre_distance_of_touching_cells", "lattice": "fcc_or_sc_cell_centre_lattice",
                   "arrays": "arrays_x_z", "array_spacing": "array_centre_spacing_x_z", "adhesion": "per_array_adhesion_maximum_all_face_pair_classes", "array_values": "cell_blocks_each_with_one_value_per_array",
                   "jitter": "random_displacement_of_each_cell_centre_in_spacings", "seed": "rng_seed", "rotate": "turn_each_spheroid_at_random",
                   "same_orientation": "every_array_gets_the_same_random_turns", "n_spheroids": "spheroids_per_array_one_to_four", "layout": "chain_or_triangle_of_spheroids"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.cell_set = str(params["cell_set"])
        self.c = np.array([float(v) for v in params["centre"]])
        self.n = int(params["n_cells"])
        self.ball = int(params.get("ball", 1000))
        self.side = float(params.get("side", 1.4))
        self.h = float(params.get("point_spacing", 0.1))
        self.d0 = float(params.get("spacing", 1.2))
        self.lattice = str(params.get("lattice", "fcc"))
        self.arrays = [int(v) for v in params.get("arrays", [3, 3])]
        self.asp = [float(v) for v in params.get("array_spacing", [24.0, 24.0])]
        na = self.arrays[0] * self.arrays[1]
        adh = params.get("adhesion") or [1.0] * na
        adh = [adh] * na if isinstance(adh, (int, float)) else list(adh)
        if len(adh) != na:
            raise ValueError(f"seed_positions[spheroid_pair]: {len(adh)} adhesion entries for {na} arrays")
        self.adh = [float(a if not isinstance(a, dict) else a.get("LL", 0.0)) for a in adh]
        self.array_values = params.get("array_values") or {}
        self.jitter = float(params.get("jitter", 0.0))
        self.rotate = bool(params.get("rotate", True))
        self.same_orientation = bool(params.get("same_orientation", True))
        self.n_sph = int(params.get("n_spheroids", 2))                       # 2 (a fusion pair) or 1 (a single spheroid: compaction and granularity)
        if self.n_sph not in (1, 2, 3, 4):
            raise ValueError("seed_positions[spheroid_pair]: n_spheroids is 1 to 4")
        self.layout = str(params.get("layout", "chain"))                     # 3 or 4 spheroids: a chain along x, or (3 only) a triangle in the x-z plane
        self.rng_seed = int(params.get("seed", 0))

    def forward(self, H, mask=None):
        P, C = H.level(self.at), H.level(self.cell_set)
        m = int(round(self.side / self.h))
        g = (np.arange(m) + 0.5) * self.h - 0.5 * self.side
        off = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        off = off[ball_lattice_points(off, self.ball)]
        ax_n, az_n = self.arrays
        ncell = ax_n * az_n * self.n_sph * self.n
        live = np.flatnonzero(C.occ.detach().cpu().numpy() > 0.5)
        if len(live) != ncell:
            raise ValueError(f"seed_positions[spheroid_pair]: {len(live)} live cells, the layout needs {ncell}")
        parent = P.parent.detach().cpu().numpy()
        dev, dt_ = P.state.device, P.state.dtype
        occ = torch.zeros_like(P.occ)
        cen = C.state[:, C.state_schema["pos"][0]:C.state_schema["pos"][1]].detach().cpu().numpy().copy()
        base = spheroid_sites(self.n, self.d0, self.lattice)
        rows, n, seps = [], 0, []
        for az in range(az_n):
            for ax in range(ax_n):
                a = az * ax_n + ax
                rng = np.random.default_rng((self.rng_seed, 0 if self.same_orientation else a))            # the same turn in every array: arrays then differ by the swept value only
                acen = self.c + np.array([(ax - 0.5 * (ax_n - 1)) * self.asp[0], 0.0, (az - 0.5 * (az_n - 1)) * self.asp[1]])
                sp = []
                for sidx in range(self.n_sph):
                    Rm = _random_rotation(rng) if self.rotate else np.eye(3)
                    q = base @ Rm.T
                    if self.jitter > 0.0:
                        q = q + rng.uniform(-1.0, 1.0, q.shape) * self.jitter * self.d0
                    sp.append(q)
                offs = spheroid_offsets(sp, self.d0, self.layout)
                seps.append(float(np.linalg.norm(offs[1] - offs[0])) if len(offs) > 1 else 0.0)
                for sidx in range(self.n_sph):
                    q = sp[sidx] + offs[sidx]
                    for i in range(self.n):
                        j = live[n]; n += 1
                        cen[j] = acen + q[i]
                        blk = np.flatnonzero(parent == j)
                        if len(blk) != self.ball:
                            raise ValueError(f"seed_positions[spheroid_pair]: cell {j} has {len(blk)} point slots, a ball of {self.ball} points needs the point set's per_parent to be {self.ball}")
                        idx = torch.as_tensor(blk, device=dev)
                        _mpm_reset_points(P, idx, torch.as_tensor(cen[j] + off, device=dev, dtype=dt_), torch.full((self.ball,), self.h ** 3, device=dev, dtype=dt_))
                        _set_block(P, "pid", idx, float(j)); _set_block(P, "cellc", idx, float(j % 20)); _set_block(P, "face", idx, float(FACE_L))
                        occ[idx] = 1.0
                        rows.append((j, a, sidx, i))
        P.occ = occ
        ca, cb = C.state_schema["pos"]
        st = C.state.clone(); st[:, ca:cb] = torch.as_tensor(cen, dtype=st.dtype, device=st.device); C.state = st
        for j, a, sidx, i in rows:
            jj = torch.as_tensor([j], device=C.state.device)
            if "adh" in C.state_schema:
                a0, a1 = C.state_schema["adh"]
                st = C.state.clone(); st[j, a0:a1] = float(self.adh[a]); C.state = st
            if "pol" in C.state_schema:
                q0, q1 = C.state_schema["pol"]
                st = C.state.clone(); st[j, q0:q1] = torch.as_tensor([0.0, 0.0, 1.0], dtype=st.dtype); C.state = st
            _set_block(C, "arr", jj, float(a))
            for blk_, av in self.array_values.items():
                vals_a, first_only = (av["values"], bool(av.get("first_only"))) if isinstance(av, dict) else (av, False)
                _set_block(C, str(blk_), jj, float("nan") if (first_only and not (sidx == 0 and i == 0)) else float(vals_a[a]))
            if "ij" in C.state_schema:
                i0, _ = C.state_schema["ij"]
                st = C.state.clone(); st[j, i0] = float(sidx); st[j, i0 + 1] = float(i); C.state = st
            _set_block(C, "tone", jj, 0.15 if sidx == 0 else 0.3)
        print(f"[seed_positions:spheroid_pair] {ax_n} x {az_n} arrays of {self.n_sph} spheroid(s) of {self.n} ball cells ({self.ball} points each, {self.lattice} lattice, spacing {self.d0:g}, "
              f"separation {min(seps):.2f}-{max(seps):.2f}): {ncell} cells, {int(occ.sum()):,} points", flush=True)
        return {}


@register_operator("contact_finder", family="topology", set="junction", kind="rewire",
                   title="detect cell contacts", equation=r"""$$A_{ab,fg}=s^{2}\,\big\lvert\{\,i\in a:\ \exists j\in b,\ \lvert\mathbf x_i-\mathbf x_j\rvert<r_b,\ (f_i,f_j)=fg\}\big\rvert$$""")
class JunctionContact(Structural):
    r"""the shared state's membership: every frame the live points of different cells closer than `bond_range` are counted per (cell a, cell b, face-pair class), a touching pair without a junction row gets one, a pair apart for `retire_after` frames loses it.
    junction -> junction: writes `ca`, `cb` (the two cells, ca < cb), `fg` (the face-pair class), `area` (contact points x spacing^2), `load` (the mean tension carried over the frame) and the pair list the substep reads.

        A_{ab,fg}=s^{2}\,\big\lvert\{\,i\in a:\ \exists j\in b,\ \lvert\mathbf x_i-\mathbf x_j\rvert<r_b,\ (f_i,f_j)=fg\}\big\rvert
        A_{ab,fg}      the contact area of cells a and b through face-pair class fg, world length squared
        s              `spacing`, the point spacing
        i, j           live points of the two cells
        x_i, x_j       their positions
        r_b            `bond_range`, the contact distance (named `r_c`, then `bond_reach` until 2026-10-31)
        (f_i, f_j)     the faces of the two points (lateral, apical, basal), their class fg

    Reference: Plexus exp27, the cell adhesion testbench (this work, 2026-10); the shared junction set of two MPM cells."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"Every frame it finds the cell pairs whose points come closer than `bond_range` and gives each touching pair a junction "
        r"with a contact area $A_{ab,fg}=s^{2}\,\lvert\{\,i\in a:\ \exists j\in b,\ \lvert\mathbf x_i-\mathbf x_j\rvert<r_b,\ "
        r"(f_i,f_j)=fg\}\rvert$: the number of points $i$ of cell $a$ that have a partner $j$ in cell $b$ closer than $r_b$ "
        r"(`bond_range`), counted per face-pair class $fg$ (lateral, apical, basal), times $s^2$, the point spacing squared. "
        r"This is where the gap between cells enters: no contact means no binding, and a junction with no contact for "
        r"`retire_after` frames is retired. The implementations differ only in the partner search (a k-d tree on the CPU, a hash "
        r"grid on the GPU)."
    )
    PARAM_UNITS = {'bond_range': 'length', 'k_nn': 'count', 'period': 'length', 'retire_after': 'frames', 'skin': 'length', 'spacing': 'length', 'workers': 'count'}
    SYMBOLS = {'A_{ab,fg}': 'the contact area of cells a and b through face-pair class fg, world length squared', 's': '`spacing`, the point spacing', 'i, j': 'live points of the two cells', 'x_i, x_j': 'their positions', 'r_b': '`bond_range`, the contact distance (named `r_c`, then `bond_reach` until 2026-10-31)', '(f_i, f_j)': 'the faces of the two points (lateral, apical, basal), their class fg'}
    REQUIRES_PARAMS = ["points", "cells"]
    SUPPORTED_DIMS = [3]
    MAY_MUTATE_INTEGRATED_STATE = True
    PARAM_ROLES = {"points": "the_point_set", "cells": "the_cell_set", "bond_range": "contact_distance_world",
                   "skin": "pair_list_margin_world", "spacing": "point_spacing_world",
                   "retire_after": "frames_without_contact_before_a_row_is_retired", "period": "period_along_x_world",
                   "workers": "kd_tree_query_threads", 'k_nn': 'nearest_neighbours_queried_per_point'}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "junction")
        self.points = str(params["points"])
        self.cells = str(params["cells"])
        self.h = float(params.get("spacing", 0.125))
        if "r_c" in params:
            raise ValueError("contact_finder: the parameter `r_c` was renamed `bond_reach` (2026-10-26) and then `bond_range` (2026-10-31); no alias -- write `bond_range` in the spec")
        self.bond_reach = float(params.get("bond_range", 2.0 * self.h))
        self.skin = float(params.get("skin", 0.5 * self.h))
        self.retire_after = int(params.get("retire_after", 5))
        self.k_nn = int(params.get("k_nn", 16))                  # neighbours searched for a partner in another cell
        # `workers` (default 1, the scipy default): threads of the kd-tree query; the answer of each query does not depend on it
        self.workers = int(params.get("workers", 1))
        # `period` (world, default 0 = none): the layout repeats along x with this period -- a RIBBON JOINED INTO A RING
        # (periodic boundary condition). Positions stay UNWRAPPED (a cell that walks through the seam keeps its
        # coordinates), and every cross-cell distance along x is taken as the minimum image, x - L round(x / L).
        self.period = float(params.get("period", 0.0))
        self._idle = None

    def _cell_index(self, H):
        """The cell each point of `points` is counted in: its own (the containment map). contact_finder[sites] counts it in its site's first body instead."""
        return H.lift_index(self.points, self.cells)

    def _periods(self, H):
        """The period of each axis, world length (0 = not periodic): `period` along x alone. contact_finder[sites] with `periodic` takes the world box on every axis."""
        return np.array([float(self.period), 0.0, 0.0])

    def forward(self, H, mask=None):
        from scipy.spatial import cKDTree
        P, C, J = H.level(self.points), H.level(self.cells), H.level(self.at)
        dev, dt_ = P.state.device, P.state.dtype
        nC = int(C.state.shape[0])
        live = torch.nonzero(P.occ > 0.5).reshape(-1)
        X = P.get("pos")[live].detach().cpu().numpy().astype(np.float64)
        cid = self._cell_index(H)[live].long().cpu().numpy()      # int64: the parent map is int32, and the junction key a * nC + b overflows 2^31 above 46,340 cells (exp09's 554,500-body scene, 2026-11-19: 43 % of its bonds named a negative cell)
        # ONE PARTNER PER POINT: its nearest point in ANOTHER cell, within bond_range + skin. All pairs within the reach
        # would include lattice diagonals (1.41 and 1.73 spacings apart at rest), loaded beyond r_eq while the
        # cells sit still; the nearest partner of a surface point of two touching cubes is one spacing away.
        Lv = self._periods(H)                            # the period of each axis, 0 = not periodic (`period`: x alone)
        L = float(Lv[0])
        if (Lv > 0.0).any():
            Xw = X.copy()
            per = Lv > 0.0
            Xw[:, per] = np.mod(X[:, per], Lv[per])
            if (~per).any():
                lo = X[:, ~per].min(0)
                if bool((lo < -0.5).any()):              # a point far outside the world: say so, once per call
                    print(f"[contact_finder] frame {getattr(H, 'frame', None)}: lowest non-periodic coordinates of any point {np.round(lo, 2).tolist()}",
                          flush=True)
                Xw[:, ~per] -= lo                        # the periodic kd-tree wants every coordinate in [0, boxsize)
            tree = cKDTree(Xw, boxsize=np.where(per, Lv, 1.0e5)); Xq = Xw
        else:
            tree = cKDTree(X); Xq = X
        H._adh_period = L
        H._adh_box = Lv                                  # every axis's period, for bond_springs' nearest image
        dd, nn = tree.query(Xq, k=self.k_nn, distance_upper_bound=self.bond_reach + self.skin, workers=self.workers)
        ok = np.isfinite(dd)
        nn_c = np.where(ok, nn, 0)
        other = ok & (cid[nn_c] != cid[:, None])
        has = other.any(1)
        first = np.argmax(other, axis=1)
        i = np.flatnonzero(has)
        j = nn_c[i, first[i]]
        a = np.minimum(cid[i], cid[j]); b = np.maximum(cid[i], cid[j])
        # THE FACE OF A CONTACT IS THE SIDE THE CELL TOUCHES WITH, decided ONCE PER CELL PAIR: the mean direction of
        # all its point pairs (oriented from the lower cell to the higher), against each cell's polarity axis (cell
        # block `pol`, +z when absent) -- along it (|cos| > 1/sqrt 2) apical or basal by the sign, across it lateral.
        # Two finer rules failed in exp27 v1's smoke tests: the points' own `face` labels made the EDGE rows of two
        # stacked cubes' lateral faces a lateral-lateral contact, and the direction of EACH pair made the oblique
        # corner pairs of a sliding stack lateral-lateral once the row had slid two spacings -- either way the rows
        # above and below a pulled row followed it with no apical-basal adhesion.
        pol = (C.get("pol").detach().cpu().numpy().astype(np.float64) if "pol" in C.state_schema
               else np.tile([0.0, 0.0, 1.0], (nC, 1)))
        pol = pol / np.maximum(np.linalg.norm(pol, axis=1, keepdims=True), 1e-12)
        dvec = X[j] - X[i]
        for ax_ in np.flatnonzero(Lv > 0.0):
            dvec[:, ax_] -= Lv[ax_] * np.round(dvec[:, ax_] / Lv[ax_])
        dvec = dvec / np.maximum(np.linalg.norm(dvec, axis=1, keepdims=True), 1e-12)
        dvec = np.where((cid[i] == a)[:, None], dvec, -dvec)                     # from cell a toward cell b
        k0 = a * nC + b
        u0, inv0 = np.unique(k0, return_inverse=True)
        nsum = np.zeros((len(u0), 3)); np.add.at(nsum, inv0, dvec)
        nmean = nsum / np.maximum(np.linalg.norm(nsum, axis=1, keepdims=True), 1e-12)
        ua, ub = u0 // nC, u0 % nC
        cut = 1.0 / math.sqrt(2.0)
        ca_ = (nmean * pol[ua]).sum(1); cb_ = -(nmean * pol[ub]).sum(1)
        fa = np.where(ca_ > cut, FACE_A, np.where(ca_ < -cut, FACE_B, FACE_L))
        fb = np.where(cb_ > cut, FACE_A, np.where(cb_ < -cut, FACE_B, FACE_L))
        fg = _FG[fa, fb][inv0]
        key = k0 * 6 + fg
        dij = X[i] - X[j]
        for ax_ in np.flatnonzero(Lv > 0.0):
            dij[:, ax_] -= Lv[ax_] * np.round(dij[:, ax_] / Lv[ax_])
        r = np.linalg.norm(dij, axis=1)
        touch = r < self.bond_reach
        # contact per key: the distinct points of the LOWER cell within bond_range, x spacing^2
        ia = np.where(cid[i] == a, i, j)
        tk, tp = key[touch], ia[touch]
        uk = np.unique(tk)
        area = {}
        if len(uk):
            kp = np.unique(np.stack([tk, tp], 1), axis=0)
            cnt = np.bincount(np.searchsorted(uk, kp[:, 0]), minlength=len(uk))
            area = dict(zip(uk.tolist(), (cnt * self.h ** 2).tolist()))
        # the junction table as it is
        occJ = (J.occ > 0.5).cpu().numpy()
        sch = J.state_schema
        st = J.state.detach().cpu().numpy().copy()
        # the key in int64 (see adhesion_warp_ops): float32 a * nC + b is exact only below 2^24
        rowkey = ((np.rint(st[:, sch["ca"][0]]).astype(np.int64) * nC + np.rint(st[:, sch["cb"][0]]).astype(np.int64)) * 6
                  + np.rint(st[:, sch["fg"][0]]).astype(np.int64))
        rowkey[~occJ] = -1
        if self._idle is None or len(self._idle) != J.n:
            self._idle = np.zeros(J.n, np.int64)
        known = dict(zip(rowkey[occJ].tolist(), np.flatnonzero(occJ).tolist()))
        new = [k for k in area if k not in known]
        if new:
            free = torch.nonzero(J.occ <= 0.5).reshape(-1)[: len(new)].cpu().numpy()
            if len(free) < len(new):
                print(f"[contact_finder] the junction buffer is FULL: {len(new) - len(free)} contacts get no row",
                      flush=True)
            for k, row in zip(new, free):
                cab, f_ = divmod(int(k), 6)
                ca_, cb_ = divmod(cab, nC)
                st[row, sch["ca"][0]] = ca_; st[row, sch["cb"][0]] = cb_; st[row, sch["fg"][0]] = f_
                st[row, sch["rho"][0]] = 0.0
                known[int(k)] = int(row); occJ[row] = True; self._idle[row] = 0
                rowkey[row] = int(k)
        # area, load, idle frames, retirement
        st[:, sch["area"][0]] = 0.0
        for k, rr in known.items():
            st[rr, sch["area"][0]] = area.get(k, 0.0)
        acc = getattr(H, "_adh_load_acc", None)
        nsub = max(int(getattr(H, "_adh_load_n", 0)), 1)
        if acc is not None and acc.numel() == J.n:
            ld = acc.detach().cpu().numpy() / nsub
            ar = np.maximum(st[:, sch["area"][0]], 1e-12)
            st[:, sch["load"][0]] = np.where(st[:, sch["area"][0]] > 0, ld / ar, 0.0)
        H._adh_load_acc = torch.zeros(J.n, device=dev, dtype=dt_)
        H._adh_load_n = 0
        has = st[:, sch["area"][0]] > 0
        self._idle = np.where(has, 0, self._idle + 1)
        gone = occJ & (self._idle >= self.retire_after) & (st[:, sch["rho"][0]] < 1e-3)
        occJ[gone] = False
        J.state = torch.as_tensor(st, device=J.state.device, dtype=J.state.dtype)
        J.occ = torch.as_tensor(occJ.astype(np.float64), device=J.occ.device, dtype=J.occ.dtype)
        # the Verlet list for the substeps: global point indices and the junction row of each pair
        lk = np.array([known.get(int(k), -1) for k in key], np.int64) if len(key) else np.zeros(0, np.int64)
        gi = live.cpu().numpy()
        H._adh_pairs = (torch.as_tensor(gi[i], device=dev), torch.as_tensor(gi[j], device=dev),
                        torch.as_tensor(lk, device=dev))
        H.junction_rows = int(occJ.sum())
        return {}


@register_operator("bond_binding", family="mechanics", set="junction", kind="lateral",
                   title="bind and unbind adhesion",
                   equation=r"""$$\dot\rho=k_{on}\,(\rho^*_{fg}-\rho)\,[A>0]-k_{off}\,e^{\sigma/\sigma_\beta}\,\rho,\qquad \rho^*_{fg}=\min(\mathrm{adh}_a[fg],\mathrm{adh}_b[fg])$$""")
class JunctionKinetics(Lateral):
    r"""the shared state's dynamics: each junction's bound-adhesion density binds toward the face pair's maximum (the smaller of the two cells' `adh` entries) while the cells touch, and unbinds faster under tension (Bell).
    junction -> junction: returns d rho / dt on `rho` (first order, integrated by the engine at the frame dt).

        \dot\rho=k_{on}\,(\rho^*_{fg}-\rho)\,[A>0]-k_{off}\,e^{\sigma/\sigma_\beta}\,\rho,\qquad \rho^*_{fg}=\min(\mathrm{adh}_a[fg],\mathrm{adh}_b[fg])
        \dot\rho       the rate of change of the junction's bound density rho
        k_{on}         `binding_rate`, the binding rate, per time
        \rho^*_{fg}    the face pair's maximum density, min of the two cells' `adh` for that class (scaled by `type_matrix`)
        A              the contact area (above `min_area`)
        k_{off}        `unbinding_rate`, the unbinding rate, per time
        \sigma         the tension the junction carries (its `load`)
        \sigma_\beta   `bond_strength`, the Bell force scale, force per area

    Reference: Bell, G. I. (1978). Models for the specific adhesion of cells to cells. Science 200:618-627; Plexus exp27, the cell adhesion testbench (this work, 2026-10)."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"How strongly each junction is bound: $\rho$ is its bound-adhesion density, from 0 (unbound) up to its maximum. "
        r"$\dot\rho=k_{on}\,(\rho^*-\rho)\,[A>0]-k_{off}\,e^{\sigma/\sigma_\beta}\,\rho$. Here $k_{on}$ is the binding rate, "
        r"$\rho^*$ the smaller of the two cells' adhesion maxima for that face-pair class, $A$ the contact area (nothing binds "
        r"below `min_area`), $k_{off}$ the unbinding rate, $\sigma$ the tension the junction carries and $\sigma_\beta$ the Bell "
        r"force scale (`bond_strength`; Bell 1978). So $\rho$ rises toward $\rho^*$ while the cells touch and unbinds faster the "
        r"more tension the junction carries. With `bell: bond` the tension is read per bond rather than per area, so the "
        r"surviving bonds unbind faster as others are lost."
    )
    PARAM_UNITS = {'binding_rate': 'rate', 'unbinding_rate': 'rate', 'min_area': 'area', 'rho_floor': 'dimensionless', 'bond_strength': 'stress'}
    SYMBOLS = {'\\dot\\rho': "the rate of change of the junction's bound density rho", 'k_{on}': '`binding_rate`, the binding rate, per time', '\\rho^*_{fg}': "the face pair's maximum density, min of the two cells' `adh` for that class (scaled by `type_matrix`)", 'A': 'the contact area (above `min_area`)', 'k_{off}': '`unbinding_rate`, the unbinding rate, per time', '\\sigma': 'the tension the junction carries (its `load`)', '\\sigma_\\beta': '`bond_strength`, the Bell force scale, force per area'}
    REQUIRES_PARAMS = ["cells"]
    SUPPORTED_DIMS = [3]
    EMIT = "velocity"                                       # a first-order block: d rho / dt, as `chem` (diffusion_reaction)
    INTEGRAND = "rho"
    PARAM_ROLES = {"binding_rate_block": "cell_block_with_each_cells_binding_rate", "unbinding_rate_block": "cell_block_with_each_cells_unbinding_rate", "bond_strength_block": "cell_block_with_each_cells_bell_force_scale", "cells": "the_cell_set", "binding_rate": "binding_rate", "unbinding_rate": "unbinding_rate_at_zero_load",
                   "bond_strength": "load_scale_of_bell_unbinding", "min_area": "contact_area_below_which_nothing_binds",
                   "bell": "area_or_bond_load_in_the_exponent", "rho_floor": "rho_floor_of_the_per_bond_load",
                   "type_matrix": "homotypic_heterotypic_adhesion_over_cell_types"}
    REFERENCE = "Bell, G. I. (1978). Science 200:618-627."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "junction")
        self.cells = str(params["cells"])
        self.k_on = self.tunable(params.get("binding_rate"), 1.0, tensor=True)
        self.k_off = self.tunable(params.get("unbinding_rate"), 0.01, tensor=True)
        self.sigma_beta = self.tunable(params.get("bond_strength"), 1.0, tensor=True)
        # `binding_rate_block`, `unbinding_rate_block`, `bond_strength_block` (exp29 rosette arrays, 2026-10-31: one parameter of the junction kinetics swept per ARRAY of cells): a width-1 cell block read
        # instead of the scalar, the junction's value being the mean of its two cells'; absent: the scalar, as before
        self.k_on_block = params.get("binding_rate_block")
        self.k_off_block = params.get("unbinding_rate_block")
        self.sigma_beta_block = params.get("bond_strength_block")
        # `min_area` (default 0.25, world units^2): a contact binds only above it. An edge or a corner brushing past
        # (exp27 v1: a row slid one full cell, its corners met the stacked cells' corners) is a line of points with
        # no face to speak of, and its direction is ambiguous between apical and lateral.
        self.min_area = float(params.get("min_area", 0.25))
        # `bell: area` (default, v1-v2): the exponent reads the junction's load per AREA, sigma / sigma_beta. A load
        # per area does not grow as bonds are lost, so a junction under a fixed pull settles at a lower rho and never
        # runs away. `bell: bond` (v3 on): the load per BOND, sigma / (max(rho, rho_floor) sigma_beta) -- Bell's
        # own reading: as bonds unbind the survivors carry more, unbind faster, and past a critical load the
        # junction RUPTURES (Bell 1978; Erdmann & Schwarz 2004, PRL 92:108102).
        self.bell = str(params.get("bell", "area"))
        if self.bell not in ("area", "bond"):
            raise ValueError("bond_binding: bell is area or bond")
        self.rho_floor = self.tunable(params.get("rho_floor"), 0.02, tensor=True)
        # `type_matrix` (sorting, v4 on): a [T, T] matrix over the cell set's TYPES multiplying rho*_fg -- homotypic
        # and heterotypic adhesion, the differential adhesion of Steinberg's sorting experiments. Absent = all 1.
        tm = params.get("type_matrix")
        self.type_matrix = None if tm is None else [[float(v) for v in row] for row in tm]

    def _rho_next(self, H):
        """The exact one-frame step of every junction's rho: (J, rho now, rho at t + dt, the live-row mask)."""
        J, C = H.level(self.at), H.level(self.cells)
        rho = J.get("rho")[:, 0]
        ca = J.get("ca")[:, 0].round().long().clamp(0, C.state.shape[0] - 1)
        cb = J.get("cb")[:, 0].round().long().clamp(0, C.state.shape[0] - 1)
        fg = J.get("fg")[:, 0].round().long().clamp(0, 5)
        adh = C.get("adh")
        rmax = torch.minimum(adh[ca, fg], adh[cb, fg])
        if self.type_matrix is not None:
            tm = torch.as_tensor(self.type_matrix, device=rho.device, dtype=rho.dtype)
            nt = C.node_type.long().clamp(0, tm.shape[0] - 1)
            rmax = rmax * tm[nt[ca], nt[cb]]
        touch = (J.get("area")[:, 0] > self.min_area).to(rho.dtype)
        sig = J.get("load")[:, 0].clamp(min=0.0)
        if self.bell == "bond":
            sig = sig / rho.clamp(min=self.rho_floor)
        # THE EXACT STEP over one frame: with a = k_on [A > A_min] and b = k_off e^{sigma/sigma_b} the equation is linear in rho,
        # d rho/dt = a rho* - (a + b) rho, so rho(t + dt) = rho_eq + (rho - rho_eq) e^{-(a + b) dt}, rho_eq = a rho* / (a + b).
        # The forward-Euler rate it replaces blew up (exp27 s1: rho 10^24 then NaN within 10 frames) once a per-bond load made b ~ 10^9.
        def _pj(scalar, block):
            if not block:
                return scalar
            v = C.state[:, C.state_schema[block][0]].to(rho.dtype)
            return 0.5 * (v[ca] + v[cb])
        k_on, k_off, sigma_beta = _pj(self.k_on, self.k_on_block), _pj(self.k_off, self.k_off_block), _pj(self.sigma_beta, self.sigma_beta_block)
        a = k_on * touch
        b = k_off * torch.exp((sig / sigma_beta).clamp(max=30.0))
        dt = float(getattr(H, "dt", 1.0))
        rate = (a + b).clamp(min=1e-12)
        r_eq = a * rmax / rate
        rho_next = r_eq + (rho - r_eq) * torch.exp(-rate * dt)
        return J, rho, rho_next, (J.occ > 0.5)

    def forward(self, H, mask=None):
        # returned as a RATE: the engine's rho += dt x d reproduces the exact step with d = (rho(t + dt) - rho) / dt -- at the END of the
        # tick, after the substeps, which read the previous frame's rho (see `bond_binding[immediate]`)
        J, rho, rho_next, live = self._rho_next(H)
        dt = float(getattr(H, "dt", 1.0))
        d = (rho_next - rho) / dt * live.to(rho.dtype)
        return {(self.at, "rho"): d[:, None]}


@register_operator("bond_binding", implementation="immediate", family="mechanics", set="junction", kind="lateral",
                   title="bind and unbind adhesion, written before the substeps",
                   equation=r"""$$\rho\leftarrow\rho_{eq}+(\rho-\rho_{eq})\,e^{-(k_{on}[A>0]+k_{off}e^{\sigma/\sigma_\beta})\,\Delta t}$$""")
class JunctionKineticsImmediate(JunctionKinetics):
    r"""bond_binding whose exact step is written into rho inside the tick instead of returned as a rate, so the substeps of the same frame read the new density.
    junction -> junction: writes `rho`; parameters as bond_binding.

        \rho\leftarrow\rho_{eq}+(\rho-\rho_{eq})\,e^{-(k_{on}[A>0]+k_{off}e^{\sigma/\sigma_\beta})\,\Delta t}
        \rho           the junction's bound density
        \rho_{eq}      the fixed point of the default's rate law
        k_{on}, k_{off} the binding and unbinding rates
        A              the contact area
        \sigma, \sigma_\beta the load and the Bell force scale
        \Delta t       the frame step

    Reference: as bond_binding."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"The same binding and unbinding as bond_binding, but the exact step is written straight into $\rho$ inside the frame, "
        r"so the substeps that follow read the new density: "
        r"$\rho\leftarrow\rho_{eq}+(\rho-\rho_{eq})\,e^{-(k_{on}[A>0]+k_{off}e^{\sigma/\sigma_\beta})\,\Delta t}$, where "
        r"$\rho_{eq}$ is the fixed point of the rate law $\dot\rho=k_{on}(\rho^*-\rho)[A>0]-k_{off}e^{\sigma/\sigma_\beta}\rho$ "
        r"($\rho^*$ the smaller of the two cells' adhesion maxima, $\sigma$ the tension the junction carries, $\sigma_\beta$ the "
        r"Bell force scale, $k_{on}$ and $k_{off}$ the binding and unbinding rates)."
    )
    PARAM_UNITS = {}
    SYMBOLS = {'\\rho': "the junction's bound density", '\\rho_{eq}': "the fixed point of the default's rate law", 'k_{on}, k_{off}': 'the binding and unbinding rates', 'A': 'the contact area', '\\sigma, \\sigma_\\beta': 'the load and the Bell force scale', '\\Delta t': 'the frame step'}
    MAY_MUTATE_INTEGRATED_STATE = True

    def forward(self, H, mask=None):
        J, rho, rho_next, live = self._rho_next(H)
        a_, b_ = J.state_schema["rho"]
        st = J.state.clone()
        st[:, a_] = torch.where(live, rho_next, rho)
        J.state = st
        return {}


@register_operator("bond_binding", implementation="valence", family="mechanics", set="junction", kind="lateral",
                   title="bind adhesion, each cell up to its own number of bonds",
                   equation=r"""$$\dot\rho_J=k_{on}\,(\rho^*_{fg}-\rho_J)\,[A_J>A_{min}]\,g_J-k_{off}\,e^{\sigma/\sigma_\beta}\,\rho_J,\qquad g_J=1\ \text{if}\ \rho_J>\rho_b\ \text{or}\ J\ \text{is admitted},\qquad \#\{J\ni c:\ g_J=1\}\le v_c$$""")
class JunctionKineticsValence(JunctionKinetics):
    r"""bond_binding with a VALENCE: a cell holds at most its own number of bonds. A junction that holds a bond keeps it; among the touching junctions that hold none, the largest contacts are admitted first, each while both of its cells have a bond to spare; a junction that is not admitted does not bind.
    junction -> junction: as bond_binding; reads also the width-1 cell block `valence_block` (the number of bonds a cell may hold; 0 = none).

        \dot\rho_J=k_{on}\,(\rho^*_{fg}-\rho_J)\,[A_J>A_{min}]\,g_J-k_{off}\,e^{\sigma/\sigma_\beta}\,\rho_J,\qquad \#\{J\ni c:\ g_J=1\}\le v_c
        \rho_J         the bound density of junction J
        g_J            1 for a junction that holds a bond (rho_J above `bound_at`) or is admitted this frame, else 0
        v_c            the valence of cell c: block `valence_block`, rounded
        \rho_b         `bound_at`, the bound density above which a junction counts as one bond of each of its two cells (0.5)
        k_{on}, k_{off}, \rho^*_{fg}, A_J, A_{min}, \sigma, \sigma_\beta  as bond_binding (`type_matrix` still says which pairs of types may bind at all)

    Admission, every frame: the junctions that hold a bond are counted per cell; the live touching junctions that hold none and whose pair of types may bind are taken by decreasing contact area, and one is admitted when both its
    cells still have fewer bonds (held or admitted before it this frame) than their valence.

    Reference: Plexus exp09 (this work, 2026-11-17; Cedric: "adhesion only between given pairs of species and a number of adhesions per protein, protein dependent: 2, 3, 4"); Li, P. et al. (2012). Phase transitions in the
    assembly of multivalent signalling proteins. Nature 483:336-340 (the number of binding sites per molecule sets what assembles: pairs, chains, networks)."""
    EXPLAIN = (
        r"bond_binding with a limit on how many bonds a cell holds. A junction binds as in bond_binding, "
        r"$\dot\rho=k_{on}(\rho^*-\rho)[A>A_{min}]\,g-k_{off}e^{\sigma/\sigma_\beta}\rho$, but only when $g=1$: when it already holds a bond "
        r"($\rho$ above `bound_at`) or is admitted this frame. Junctions are admitted largest contact first, each while both of its cells "
        r"hold fewer bonds than their valence $v_c$ (the cell block `valence_block`). A cell of valence 2 ends in chains and rings, 3 in branched trees, 4 in networks."
    )
    PARAM_UNITS = dict(JunctionKinetics.PARAM_UNITS, bound_at='fraction') if hasattr(JunctionKinetics, "PARAM_UNITS") else {'bound_at': 'fraction'}
    SYMBOLS = {'\\rho_J': 'the bound density of junction J', 'g_J': '1 for a junction that holds a bond or is admitted this frame, else 0', 'v_c': 'the valence of cell c: block `valence_block`, rounded',
               '\\rho_b': '`bound_at`, the bound density above which a junction counts as one bond of each of its two cells',
               'k_{on}, k_{off}, \\rho^*_{fg}, A_J, A_{min}, \\sigma, \\sigma_\\beta': 'as bond_binding'}
    PARAM_ROLES = dict(JunctionKinetics.PARAM_ROLES, valence_block="cell_block_with_the_number_of_bonds_a_cell_may_hold", bound_at="bound_density_that_counts_as_a_bond") if hasattr(JunctionKinetics, "PARAM_ROLES") else {}
    REFERENCE = "Plexus exp09 (this work, 2026-11-17); Li, P. et al. (2012). Nature 483:336-340."

    def __init__(self, params, device="cpu"):
        if not params.get("valence_block"):
            raise ValueError("bond_binding[valence]: `valence_block`, the width-1 cell block with each cell's number of bonds, is required")
        super().__init__({k: v for k, v in params.items() if k not in ("valence_block", "bound_at")}, device)
        self.valence_block = str(params["valence_block"])
        self.bound_at = float(params.get("bound_at", 0.5))

    def _rho_next(self, H):
        """bond_binding's exact one-frame step with the binding term of every junction multiplied by its gate g_J (see the class)."""
        J, C = H.level(self.at), H.level(self.cells)
        nC = int(C.state.shape[0])
        rho = J.get("rho")[:, 0]
        ca = J.get("ca")[:, 0].round().long().clamp(0, nC - 1)
        cb = J.get("cb")[:, 0].round().long().clamp(0, nC - 1)
        fg = J.get("fg")[:, 0].round().long().clamp(0, 5)
        adh = C.get("adh")
        rmax = torch.minimum(adh[ca, fg], adh[cb, fg])
        if self.type_matrix is not None:
            tm = torch.as_tensor(self.type_matrix, device=rho.device, dtype=rho.dtype)
            nt = C.node_type.long().clamp(0, tm.shape[0] - 1)
            rmax = rmax * tm[nt[ca], nt[cb]]
        live = J.occ > 0.5
        area = J.get("area")[:, 0]
        touch = area > self.min_area
        # ---- the gate, on the CPU: a few hundred junction rows, a greedy pass that a vectorised form would not order the same way
        val = C.get(self.valence_block)[:, 0].detach().round().long().cpu().numpy()
        ca_n, cb_n = ca.detach().cpu().numpy(), cb.detach().cpu().numpy()
        held = (live & (rho > self.bound_at)).detach().cpu().numpy()
        n = np.bincount(ca_n[held], minlength=nC) + np.bincount(cb_n[held], minlength=nC)          # bonds held per cell
        cand = np.flatnonzero((live & ~(rho > self.bound_at) & touch & (rmax > 0)).detach().cpu().numpy())
        ar = area.detach().cpu().numpy()
        gate = held.copy()
        for j in cand[np.argsort(-ar[cand], kind="stable")]:                                       # largest contact first, the row order breaking ties
            a_, b_ = ca_n[j], cb_n[j]
            if a_ != b_ and n[a_] < val[a_] and n[b_] < val[b_]:
                gate[j] = True
                n[a_] += 1
                n[b_] += 1
        g = torch.as_tensor(gate, device=rho.device).to(rho.dtype)
        sig = J.get("load")[:, 0].clamp(min=0.0)
        if self.bell == "bond":
            sig = sig / rho.clamp(min=self.rho_floor)

        def _pj(scalar, block):
            if not block:
                return scalar
            v = C.state[:, C.state_schema[block][0]].to(rho.dtype)
            return 0.5 * (v[ca] + v[cb])
        k_on, k_off, sigma_beta = _pj(self.k_on, self.k_on_block), _pj(self.k_off, self.k_off_block), _pj(self.sigma_beta, self.sigma_beta_block)
        a = k_on * touch.to(rho.dtype) * g
        b = k_off * torch.exp((sig / sigma_beta).clamp(max=30.0))
        dt = float(getattr(H, "dt", 1.0))
        rate = (a + b).clamp(min=1e-12)
        r_eq = a * rmax / rate
        rho_next = r_eq + (rho - r_eq) * torch.exp(-rate * dt)
        return J, rho, rho_next, live


@register_operator("bond_springs", family="interaction", set="particle", kind="exchange",
                   title="pull adhered cells together",
                   equation=r"""$$\mathbf F_{ij}=\kappa\,\rho_J\,(r-r_{eq})^+\,\hat{\mathbf r}_{ij}\,[r<r_c]+\zeta\,\rho_J\,(1-r/r_c)^2\,(\mathbf v_j-\mathbf v_i)\,[r<r_c],\qquad \mathbf F_{ji}=-\mathbf F_{ij}$$""")
class CellAdhesion(Exchange):
    r"""the forces the shared state makes, every MPM substep: for each cross-cell point pair of a live junction, a tether that pulls them together beyond bond_rest_length and a viscous drag on their relative velocity, both scaled by the junction's bound density.
    pt -> pt (reads junction): emits the tether accelerations into the MPM body force and adds each junction's tether force into its `load`.

        \mathbf F_{ij}=\kappa\,\rho_J\,(r-r_{eq})^+\,\hat{\mathbf r}_{ij}\,[r<r_c]+\zeta\,\rho_J\,(1-r/r_c)^2\,(\mathbf v_j-\mathbf v_i)\,[r<r_c],\qquad \mathbf F_{ji}=-\mathbf F_{ij}
        F_{ij}         the force on point i from point j (and -F_ij on j)
        \kappa         `bond_stiffness`, the tether stiffness per pair (force per length)
        \rho_J         the bound density of their junction J
        r              the pair distance
        r_{eq}         `bond_rest_length`, the rest distance (default the point spacing)
        \hat r_{ij}    the unit vector from i to j
        r_c            `bond_range`, beyond which a pair is not bonded (default twice the spacing)
        \zeta          `bond_damping`, the drag per pair (force per velocity)
        v_j - v_i      the relative velocity

    Reference: Plexus exp27, the cell adhesion testbench (this work, 2026-10)."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"The force of adhesion, every MPM substep. For each cross-cell point pair of a live junction there is a tether that "
        r"pulls the two points together beyond `bond_rest_length` and a viscous drag on their relative velocity: $\mathbf "
        r"F_{ij}=\kappa\,\rho_J\,(r-r_{eq})^+\,\hat{\mathbf r}_{ij}\,[r<r_c]+\zeta\,\rho_J\,(1-r/r_c)^2\,(\mathbf v_j-\mathbf "
        r"v_i)\,[r<r_c]$, with $\mathbf F_{ji}=-\mathbf F_{ij}$. $\kappa$ is `bond_stiffness`, $\rho_J$ the bound density of "
        r"their junction, $r$ the pair distance, $r_{eq}$ the rest length, $r_c$ = `bond_range` and $\zeta$ = `bond_damping`. "
        r"The tether forces go into the MPM body force, and their sum is written into the junction's load: the tension $\sigma$ "
        r"that bond_binding reads to unbind faster."
    )
    PARAM_UNITS = {'bond_range': 'length', 'bond_stiffness': 'acceleration', 'bond_rest_length': 'length', 'spacing': 'length', 'bond_damping': 'rate'}
    SYMBOLS = {'F_{ij}': 'the force on point i from point j (and -F_ij on j)', '\\kappa': '`bond_stiffness`, the tether stiffness per pair (force per length)', '\\rho_J': 'the bound density of their junction J', 'r': 'the pair distance', 'r_{eq}': '`bond_rest_length`, the rest distance (default the point spacing)', '\\hat r_{ij}': 'the unit vector from i to j', 'r_c': '`bond_range`, beyond which a pair is not bonded (default twice the spacing)', '\\zeta': '`bond_damping`, the drag per pair (force per velocity)', 'v_j - v_i': 'the relative velocity'}
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["junctions"]
    MECHANISM_TAGS = ["cadherin_adhesion", "cell_cell_junction", "interfacial_friction"]
    PARAM_ROLES = {"bond_rest_length_block": "cell_block_with_each_cells_tether_rest_length", "bond_range_block": "cell_block_with_each_cells_tether_range", "bond_damping_block": "cell_block_with_each_cells_tether_damping", "junctions": "the_junction_set", "bond_stiffness": "tether_stiffness_per_pair",
                   "bond_damping": "friction_per_pair", "bond_rest_length": "rest_distance_world", "bond_range": "reach_world",
                   "spacing": "point_spacing_world", "bond_stiffness_block": "cell_block_with_each_cells_tether_stiffness", "bond_stiffness_cells": "the_cell_set_of_that_block"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.junctions = str(params["junctions"])
        h = float(params.get("spacing", 0.125))
        self.kappa = self.tunable(params.get("bond_stiffness"), 1.0, tensor=True)
        # `bond_stiffness_block` (exp33, 2026-10-26: the adhesion strength as a knob the optimisation holds per cell): a width-1 CELL block kappa_c read instead of the scalar, the tether stiffness of a pair of
        # points being the mean of their two cells', (kappa_a + kappa_b) / 2; `bond_stiffness_cells` the cell set (default "cell"). Absent: the scalar, as before
        self.kappa_block = params.get("bond_stiffness_block")
        self.kappa_cells = str(params.get("bond_stiffness_cells", "cell"))
        self.zeta = float(params.get("bond_damping", 0.0))
        self.r_eq = self.tunable(params.get("bond_rest_length"), h, tensor=True)
        # `bond_rest_length_block`, `bond_range_block`, `bond_damping_block` (exp29 rosette arrays, 2026-10-31): per-cell blocks read instead of the scalars, a pair of points taking the mean of its two cells';
        # `bond_range` stays the scalar upper bound the contact search uses, so set it to the largest value of the sweep
        self.r_eq_block = params.get("bond_rest_length_block")
        self.bond_reach_block = params.get("bond_range_block")
        self.zeta_block = params.get("bond_damping_block")
        if "r_c" in params:
            raise ValueError("bond_springs: the parameter `r_c` was renamed `bond_reach` (2026-10-26) and then `bond_range` (2026-10-31); no alias -- write `bond_range` in the spec")
        self.bond_reach = float(params.get("bond_range", 2.0 * h))

    def forward(self, H, mask=None):
        P = H.level(self.at)
        X = P.get("pos")
        acc = torch.zeros_like(X)
        pairs = getattr(H, "_adh_pairs", None)
        if pairs is None or pairs[0].numel() == 0:
            return {self.at: acc}
        i, j, row = pairs
        J = H.level(self.junctions)
        ok = row >= 0
        i, j, row = i[ok], j[ok], row[ok]
        if i.numel() == 0:
            return {self.at: acc}
        rho = J.get("rho")[row, 0].clamp(min=0.0) * (J.occ[row] > 0.5).to(X.dtype)
        d = X[j] - X[i]
        Lb = getattr(H, "_adh_box", None)                # every axis's period as the contact finder searched them (its `period` along x alone otherwise)
        if Lb is None:
            Lb = [float(getattr(H, "_adh_period", 0.0)), 0.0, 0.0]
        for ax_, Lp in enumerate(Lb):
            if float(Lp) > 0.0:
                d = torch.cat([d[:, :ax_], d[:, ax_:ax_ + 1] - float(Lp) * torch.round(d[:, ax_:ax_ + 1] / float(Lp)), d[:, ax_ + 1:]], dim=1)
        r = d.norm(dim=1).clamp_min(1e-9)
        par_ = P.parent.long()

        def _pp(scalar, block):
            if not block:
                return scalar
            Cl_ = H.level(self.kappa_cells)
            v = Cl_.state[:, Cl_.state_schema[block][0]].to(X.dtype)
            pc = par_.clamp(0, Cl_.state.shape[0] - 1)
            return 0.5 * (v[pc[i]] + v[pc[j]])
        reach = _pp(self.bond_reach, self.bond_reach_block)
        r_eq = _pp(self.r_eq, self.r_eq_block)
        zeta = _pp(self.zeta, self.zeta_block)
        inr = (r < reach).to(X.dtype)
        kap = self.kappa
        if self.kappa_block:
            Cl = H.level(self.kappa_cells)
            kc = Cl.state[:, Cl.state_schema[self.kappa_block][0]].to(X.dtype)
            par = P.parent.long().clamp(0, Cl.state.shape[0] - 1)
            kap = 0.5 * (kc[par[i]] + kc[par[j]])
        ten = kap * rho * (r - r_eq).clamp(min=0.0) * inr                       # tether tension, >= 0
        F = (ten / r)[:, None] * d
        if (self.zeta_block or self.zeta > 0.0) and "vel" in P.state_schema:
            V = P.get("vel")
            w = (1.0 - r / reach).clamp(min=0.0) ** 2
            F = F + (zeta * rho * w)[:, None] * (V[j] - V[i])
        m = P.mass.to(X.dtype) if getattr(P, "mass", None) is not None else torch.ones(X.shape[0], device=X.device,
                                                                                       dtype=X.dtype)
        acc.index_add_(0, i, F / m[i, None].clamp_min(1e-30))
        acc.index_add_(0, j, -F / m[j, None].clamp_min(1e-30))
        accL = getattr(H, "_adh_load_acc", None)
        if accL is not None and accL.numel() == J.n:
            accL.index_add_(0, row, ten)
            H._adh_load_n = int(getattr(H, "_adh_load_n", 0)) + 1
        if mask is not None:
            acc = acc * mask[:, None].to(X.dtype)
        return {self.at: acc}


@register_operator("cell_pull", family="mechanics", set="particle", kind="lateral",
                   title="pull cells with a probe", equation=r"""$$\mathbf F_c=k\,(\mathbf c_0+\min(v\,(t-t_d),\,d_{\max})\,\hat{\mathbf d}-\bar{\mathbf x}_c)$$""")
class CellPull(Lateral):
    r"""the probe, a micropipette or optical-trap pull: every cell whose block `pull` is non-zero is dragged by a spring from its centroid to a target that starts at the centroid at frame `delay` and moves along the pull direction at `speed`.
    pt -> pt (reads cell): emits the spring acceleration on the cell's points; the force is recoverable from the trajectory as k (target - centroid).

        \mathbf F_c=k\,(\mathbf c_0+\min(v\,(t-t_d),\,d_{\max})\,\hat{\mathbf d}-\bar{\mathbf x}_c)
        F_c            the force on cell c, spread over its points
        k              `k`, the spring stiffness, force per length
        c_0            the centroid at frame t_d
        v              `speed`
        t_d            `delay`, the frame the pull starts
        d_{\max}       `max_disp`, the target's largest excursion
        \hat d         the pull direction (block `pull`)
        \bar x_c       the cell's centroid

    Reference: Plexus exp27, the cell adhesion testbench (this work, 2026-10); the micropipette assay of Chu, Y.-S. et al. (2004). J. Cell Biol. 167:1183-1194."""
    PARAM_UNITS = {'delay': 'frames', 'k': 'acceleration', 'max_disp': 'length', 'speed': 'velocity'}
    SYMBOLS = {'F_c': 'the force on cell c, spread over its points', 'k': '`k`, the spring stiffness, force per length', 'c_0': 'the centroid at frame t_d', 'v': '`speed`', 't_d': '`delay`, the frame the pull starts', 'd_{\\max}': "`max_disp`, the target's largest excursion", '\\hat d': 'the pull direction (block `pull`)', '\\bar x_c': "the cell's centroid"}
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["cells"]
    PARAM_ROLES = {"cells": "the_cell_set", "k": "trap_stiffness", "speed": "target_speed",
                   "delay": "frames_before_the_pull", "max_disp": "maximum_target_displacement"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.cells = str(params["cells"])
        self.k = float(params.get("k", 50.0))
        self.speed = float(params.get("speed", 0.1))
        self.delay = int(params.get("delay", 10))
        self.max_disp = float(params.get("max_disp", 3.0))
        self._c0 = None

    def forward(self, H, mask=None):
        P, C = H.level(self.at), H.level(self.cells)
        X = P.get("pos")
        acc = torch.zeros_like(X)
        pdir = C.get("pull")
        on = (pdir.norm(dim=1) > 0) & (C.occ > 0.5)
        if not bool(on.any()):
            return {self.at: acc}
        cid = H.lift_index(self.at, self.cells)
        nC = C.state.shape[0]
        live = (P.occ > 0.5).to(X.dtype)
        m = (P.mass.to(X.dtype) if getattr(P, "mass", None) is not None else torch.ones_like(live)) * live
        M = torch.zeros(nC, device=X.device, dtype=X.dtype).index_add_(0, cid, m)
        cen = torch.zeros(nC, 3, device=X.device, dtype=X.dtype).index_add_(0, cid, X * m[:, None]) / M.clamp_min(1e-30)[:, None]
        f = int(getattr(H, "frame", 0) or 0)
        if self._c0 is None and f >= self.delay:
            self._c0 = cen.detach().clone()
        if self._c0 is None:
            return {self.at: acc}
        t = (f - self.delay) * float(getattr(H, "dt", 1.0))
        s = min(self.speed * max(t, 0.0), self.max_disp)
        target = self._c0 + s * pdir
        F = self.k * (target - cen) * on[:, None].to(X.dtype)
        acc = (F / M.clamp_min(1e-30)[:, None])[cid] * live[:, None]
        if mask is not None:
            acc = acc * mask[:, None].to(X.dtype)
        return {self.at: acc}


@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="clump",
                   title="seed a cell clump", equation=r"""$$\mathbf c_k=\mathbf o+a\,(\mathbf k+\boldsymbol\epsilon_k),\ \ \lvert\mathbf c_k-\mathbf o\rvert<R,\qquad \pi(i)=\arg\min_k\lvert\mathbf x_i-\mathbf c_k\rvert$$""")
class SeedClump(Seed):
    r"""a sorting aggregate: the live cells' centres on a jittered cubic lattice nearest the centre within a ball, the types mixed, each cell a Voronoi cell of points with its own adhesion.
    pt -> pt, cell: `pid`, `cellc`, `tcol` on the points; `pos`, `pol` (+z), `adh` (6 classes at `adh`) on the cells.

        \mathbf c_k=\mathbf o+a\,(\mathbf k+\boldsymbol\epsilon_k),\ \ \lvert\mathbf c_k-\mathbf o\rvert<R,\qquad \pi(i)=\arg\min_k\lvert\mathbf x_i-\mathbf c_k\rvert
        c_k            the centre of cell k
        o              `centre`
        a              `spacing`, the lattice pitch
        k              an integer lattice site
        \epsilon_k     the jitter
        R              `radius`, the clump's radius
        \pi(i)         the cell point i belongs to (its Voronoi cell)
        x_i            a point of the cubic point lattice of `point_spacing`

    Reference: Steinberg, M. S. (1963). Science 141:401-408 (differential adhesion sorting); Plexus exp27, the cell adhesion testbench (this work, 2026-10)."""
    PARAM_UNITS = {'centre': 'length', 'point_spacing': 'length', 'radius': 'length', 'seed': 'count', 'spacing': 'length', 'adh': 'dimensionless'}
    SYMBOLS = {'c_k': 'the centre of cell k', 'o': '`centre`', 'a': '`spacing`, the lattice pitch', 'k': 'an integer lattice site', '\\epsilon_k': 'the jitter', 'R': "`radius`, the clump's radius", '\\pi(i)': 'the cell point i belongs to (its Voronoi cell)', 'x_i': 'a point of the cubic point lattice of `point_spacing`'}
    REQUIRES_PARAMS = ["cell_set", "centre", "radius"]
    SUPPORTED_DIMS = [3]
    PARAM_ROLES = {"cell_set": "the_cell_set", "centre": "clump_centre", "radius": "clump_radius",
                   "spacing": "cell_centre_lattice_spacing", "point_spacing": "material_point_lattice_spacing",
                   "adh": "isotropic_adhesion_maximum", "seed": "rng_seed"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.cell_set = str(params["cell_set"])
        self.c = np.array([float(v) for v in params["centre"]])
        self.R = float(params["radius"])
        self.sp = float(params.get("spacing", 1.0))
        self.h = float(params.get("point_spacing", 0.2))
        self.adh = float(params.get("adh", 1.0))
        self.seed = int(params.get("seed", 0))

    def forward(self, H, mask=None):
        from scipy.spatial import cKDTree
        from plexus.operators.gastrulation_ops import _lattice
        P, C = H.level(self.at), H.level(self.cell_set)
        rng = np.random.default_rng(self.seed)
        live = np.flatnonzero(C.occ.detach().cpu().numpy() > 0.5)
        nt = C.node_type.detach().cpu().numpy()
        lo, hi = self.c - self.R - self.sp, self.c + self.R + self.sp
        cand = _lattice(self.sp, lo, hi, rng, jitter=0.15)
        cand = cand[np.argsort(np.linalg.norm(cand - self.c, axis=1))[: len(live)]]
        cand = cand[rng.permutation(len(cand))]                       # types shuffled over the positions
        centres = np.zeros((C.state.shape[0], 3)); centres[live] = cand
        pts = _lattice(self.h, self.c - self.R, self.c + self.R, rng, jitter=0.15)
        Rc = max(self.R, float(np.linalg.norm(cand - self.c, axis=1).max()) + 0.5 * self.sp)
        pts = pts[np.linalg.norm(pts - self.c, axis=1) < Rc]
        owner = live[cKDTree(cand).query(pts, k=1)[1]]
        parent = P.parent.detach().cpu().numpy()
        dev, dt_ = P.state.device, P.state.dtype
        occ = torch.zeros_like(P.occ)
        short = 0
        for c in live:
            blk = np.flatnonzero(parent == c)
            own = pts[owner == c]
            vol = max(len(own), 1) * self.h ** 3
            if len(own) == 0:
                own = centres[c][None] + rng.normal(0, 0.1 * self.sp, (8, 3)); vol = 8 * self.h ** 3
            if len(own) >= len(blk):
                take = own[rng.choice(len(own), len(blk), replace=False)]
            else:
                short += 1
                extra = own[rng.integers(0, len(own), len(blk) - len(own))] + rng.normal(0, 0.3 * self.h, (len(blk) - len(own), 3))
                take = np.concatenate([own, extra])
            idx = torch.as_tensor(blk, device=dev)
            _mpm_reset_points(P, idx, torch.as_tensor(take, device=dev, dtype=dt_),
                              torch.full((len(blk),), vol / len(blk), device=dev, dtype=dt_))
            _set_block(P, "pid", idx, float(c)); _set_block(P, "cellc", idx, float(c % 20))
            _set_block(P, "tcol", idx, float(nt[c]))
            occ[idx] = 1.0
        P.occ = occ
        ca, cb = C.state_schema["pos"]
        st = C.state.clone(); st[:, ca:cb] = torch.as_tensor(centres, dtype=st.dtype, device=st.device)
        if "pol" in C.state_schema:
            q0, q1 = C.state_schema["pol"]; st[:, q0:q1] = torch.as_tensor([0.0, 0.0, 1.0], dtype=st.dtype)
        if "adh" in C.state_schema:
            a0, a1 = C.state_schema["adh"]; st[:, a0:a1] = self.adh
        C.state = st
        tn = list(getattr(C, "type_names", []) or [])
        print(f"[seed_positions:clump] {len(live)} cells ({', '.join(f'{int((nt[live] == k).sum())} {n}' for k, n in enumerate(tn))}) "
              f"in a ball of radius {Rc:.2f}, {int(occ.sum()):,} points; {short} cells padded with copies", flush=True)
        return {}


from plexus.operators.cleavage_ops import CellContactVolume  # noqa: E402


@register_operator("pair_potential", model="volume_energy", family="interaction", set="particle", kind="exchange",
                   title="keep cells from overlapping, energy form",
                   equation=r"""$$E=\Delta x^3\sum_n\tfrac{k}{2}\,\big(P_n-s_0\big)_+^2,\quad P_n=\sum_{A<B}\phi_A(n)\,\phi_B(n),\qquad \mathbf F_i=-V_i\sum_n k\,(P_n-s_0)_+\,\big(\phi(n)-\phi_{\pi(i)}(n)\big)\,\nabla w_{in}$$""")
class CellContactVolumeEnergy(CellContactVolume):
    r"""the contact of pair_potential[volume] derived from an energy: the overlap penalty on the shared lattice's pairwise volume-fraction products, so the forces are its gradient and equal and opposite.
    particle -> particle: reads every live point's position, rest volume and body; emits an acceleration F_i / m_i (`mpm_acceleration`).

        E=\Delta x^3\sum_n\tfrac{k}{2}\,\big(P_n-s_0\big)_+^2,\quad P_n=\sum_{A<B}\phi_A(n)\,\phi_B(n),\qquad \mathbf F_i=-V_i\sum_n k\,(P_n-s_0)_+\,\big(\phi(n)-\phi_{\pi(i)}(n)\big)\,\nabla w_{in}
        E              the overlap energy
        \Delta x       the shared lattice spacing
        k              `repulsion_strength`, the penalty stiffness
        P_n            the sum of pairwise overlaps phi_A phi_B at node n
        s_0            `contact_slack`, the tolerated overlap
        \phi_A(n)      body A's volume fraction at node n
        F_i            the force on point i
        V_i            its rest volume
        \phi(n) - \phi_{\pi(i)}(n) the other bodies' fraction at the node
        \nabla w_{in}  the gradient of the B-spline weight

    Reference: Plexus exp27, the cell adhesion testbench (this work, 2026-10) (L1, 2026-10-14); the volume-fraction contact: Bardenhagen, S. G. et al. (2000). Comput. Methods Appl. Mech. Eng. 187:529-541."""
    PARAM_UNITS = {'period': 'length'}
    SYMBOLS = {'E': 'the overlap energy', '\\Delta x': 'the shared lattice spacing', 'k': '`repulsion_strength`, the penalty stiffness', 'P_n': 'the sum of pairwise overlaps phi_A phi_B at node n', 's_0': '`contact_slack`, the tolerated overlap', '\\phi_A(n)': "body A's volume fraction at node n", 'F_i': 'the force on point i', 'V_i': 'its rest volume', '\\phi(n) - \\phi_{\\pi(i)}(n)': "the other bodies' fraction at the node", '\\nabla w_{in}': 'the gradient of the B-spline weight'}
    PARAM_ROLES = {'period': 'periodic_length_along_x_0_none'}

    def __init__(self, params, device="cpu"):
        super().__init__({k: v for k, v in params.items() if k != "period"}, device)
        self.period = float(params.get("period", 0.0))

    def forward(self, H, mask=None):
        P = H.level(self.at)
        X = P.get("pos")
        dev, dt_ = X.device, X.dtype
        live = (P.occ > 0).to(dt_)
        cid = H.lift_index(self.at, self.cells)
        nC = int(H.level(self.cells).state.shape[0])
        dx = float(self.dx) if self.dx is not None else float(H.field(self.grid).dx)
        world = H.world_size
        shape = [int(math.ceil(float(w) / dx)) + 3 for w in world]
        if self.period > 0.0:
            nx = int(round(self.period / dx))
            if abs(self.period / dx - nx) > 1e-6:
                raise ValueError(f"pair_potential[volume_energy]: period {self.period} is not a whole number of lattice "
                                 f"spacings dx {dx}")
            shape[0] = nx
        Nw = shape[0] * shape[1] * shape[2]
        if CellContactVolume._OFF is None or CellContactVolume._OFF.device != dev:
            o = torch.arange(3, device=dev)
            CellContactVolume._OFF = torch.stack(torch.meshgrid(o, o, o, indexing="ij"), -1).reshape(-1, 3)
        off = CellContactVolume._OFF
        xg = X / dx
        base = torch.floor(xg - 0.5)
        fx = xg - base
        w = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], 1)
        dw = torch.stack([fx - 1.5, -2.0 * (fx - 1.0), fx - 0.5], 1) / dx
        idx = base.long()[:, None, :] + off[None]
        for d in range(3):
            if d == 0 and self.period > 0.0:
                idx[..., 0] = torch.remainder(idx[..., 0], shape[0])
            else:
                idx[..., d] = idx[..., d].clamp(0, shape[d] - 1)
        flat = (idx[..., 0] * shape[1] + idx[..., 1]) * shape[2] + idx[..., 2]
        wx, wy, wz = w[:, :, 0], w[:, :, 1], w[:, :, 2]
        W = (wx[:, off[:, 0]] * wy[:, off[:, 1]] * wz[:, off[:, 2]])
        G = torch.stack([dw[:, off[:, 0], 0] * wy[:, off[:, 1]] * wz[:, off[:, 2]],
                         wx[:, off[:, 0]] * dw[:, off[:, 1], 1] * wz[:, off[:, 2]],
                         wx[:, off[:, 0]] * wy[:, off[:, 1]] * dw[:, off[:, 2], 2]], -1)
        V = (P.p_vol.to(dt_) if getattr(P, "p_vol", None) is not None else torch.ones_like(live)) * live
        dep = (W * (V / dx ** 3)[:, None]).reshape(-1)
        phi_b = torch.zeros(nC * Nw, device=dev, dtype=dt_).index_add_(0, (cid[:, None] * Nw + flat).reshape(-1), dep)
        phi_b = phi_b.view(nC, Nw)
        phi = phi_b.sum(0)
        Pn = 0.5 * (phi * phi - (phi_b * phi_b).sum(0))                          # sum over pairs A<B of phi_A phi_B
        dU = self.k * (Pn - self.c0).clamp(min=0.0)                               # dU/dP at each node
        other = (phi[flat] - phi_b.view(-1)[cid[:, None] * Nw + flat])            # phi - phi_A at the point's nodes
        F = -V[:, None] * (G * (dU[flat] * other)[..., None]).sum(1)
        extra = self._extra_force(H, V, G, flat, idx, cid, phi_b, nC, Nw, shape, dx, dev, dt_)
        if extra is not None:
            F = F + extra
        m = P.mass.to(dt_) if getattr(P, "mass", None) is not None else torch.ones_like(live)
        acc = F / m.clamp_min(1e-30)[:, None]
        acc = acc * live[:, None]
        if mask is not None:
            acc = acc * mask[:, None].to(dt_)
        H.cell_contact_max = float(Pn.max()) if Pn.numel() else 0.0
        return {self.at: acc}

    def _extra_force(self, H, V, G, flat, idx, cid, phi_b, nC, Nw, shape, dx, dev, dt_):
        """a hook for variants: an extra force on every point from the same lattice fields (None here)"""
        return None


@register_operator("pair_potential", model="cell_contact_dense", family="interaction", set="particle", kind="exchange",
                   title="keep cells apart and set the tension of each contact",
                   equation=r"""$$E=E_{\rm overlap}-w\sum_{A<B}\gamma_{\tau_A\tau_B}\int\nabla\phi_A\cdot\nabla\phi_B\,dV\;\approx\;E_{\rm overlap}+\sum_{A<B}\gamma_{\tau_A\tau_B}\,S_{AB},\qquad \frac{\partial E}{\partial\phi_A(n)}=w\,\Delta x\sum_{B\neq A}\gamma_{\tau_A\tau_B}\,\nabla^2\phi_B(n)$$""")
class CellContactVolumeInterface(CellContactVolumeEnergy):
    r"""pair_potential[volume_energy] plus an interface tension of each contact that depends on the types of the two cells in contact: the differential interfacial tension of Brodland and of the zebrafish germ-layer sorting.
    particle -> particle: as volume_energy, with the contact energy sum gamma_AB S_AB added; emits an acceleration.

        E=E_{\rm overlap}-w\sum_{A<B}\gamma_{\tau_A\tau_B}\int\nabla\phi_A\cdot\nabla\phi_B\,dV\;\approx\;E_{\rm overlap}+\sum_{A<B}\gamma_{\tau_A\tau_B}\,S_{AB},\qquad \frac{\partial E}{\partial\phi_A(n)}=w\,\Delta x\sum_{B\neq A}\gamma_{\tau_A\tau_B}\,\nabla^2\phi_B(n)
        E              the total contact energy
        E_{overlap}    the overlap penalty of volume_energy
        w              `contact_edge_width`, the interface width factor
        \gamma_{\tau_A\tau_B} `contact_surface_tension`, the tension of a contact between types tau_A and tau_B (a symmetric matrix, in the units of cell_surface_tension)
        \phi_A, \phi_B the two cells' volume fractions
        S_{AB}         the contact area of A and B
        \Delta x       the lattice spacing

    Reference: Brodland, G. W. (2002). J. Biomech. Eng. 124:188-197 (the differential interfacial tension hypothesis); Krieg, M. et al. (2008). Nat. Cell Biol. 10:429-436 (zebrafish germ-layer sorting); Plexus exp27, the cell adhesion testbench (this work, 2026-10)."""
    PARAM_UNITS = {'contact_surface_tension': 'energy_per_area', 'contact_edge_width': 'dimensionless'}
    SYMBOLS = {'E': 'the total contact energy', 'E_{overlap}': 'the overlap penalty of volume_energy', 'w': '`contact_edge_width`, the interface width factor', '\\gamma_{\\tau_A\\tau_B}': '`contact_surface_tension`, the tension of a contact between types tau_A and tau_B (a symmetric matrix, in the units of cell_surface_tension)', '\\phi_A, \\phi_B': "the two cells' volume fractions", 'S_{AB}': 'the contact area of A and B', '\\Delta x': 'the lattice spacing'}
    PARAM_ROLES = dict(CellContactVolumeEnergy.PARAM_ROLES, contact_surface_tension="contact_energy_per_area_by_type_pair",
                       contact_edge_width="diffuse_boundary_width_world", contact_surface_tension_block="cell_block_of_per_cell_tensions",
                       repulsion_strength_block="cell_block_of_the_overlap_stiffness", contact_slack_block="cell_block_of_per_cell_tolerated_overlap",
                       contact_edge_width_block="cell_block_of_per_cell_boundary_width") if hasattr(CellContactVolumeEnergy, "PARAM_ROLES") else {}

    def __init__(self, params, device="cpu"):
        super().__init__({k: v for k, v in params.items() if k not in ("contact_surface_tension", "contact_edge_width")}, device)
        ct = params.get("contact_surface_tension")
        if ct is None:
            raise ValueError("pair_potential[cell_contact_dense]: contact_surface_tension, a [T, T] matrix over the cell types, is required")
        self.gamma = [[float(v) for v in row] for row in ct]
        self.width = self.tunable(params.get("contact_edge_width"), 0.2, tensor=True)
        # `contact_surface_tension_block` (exp30, 2026-10-23): a width-1 CELL block gamma_c read instead of the type matrix, the tension of a
        # contact being the mean of its two cells', gamma_ab = (gamma_a + gamma_b) / 2 -- a knob the refiner can hold per cell. Honoured
        # by the sparse implementation (the dense one keeps the matrix).
        self.ct_block = params.get("contact_surface_tension_block")
        # `repulsion_strength_block` (exp33, 2026-10-26): the overlap stiffness k of the SPARSE implementation as a knob -- ONE value for the whole vesicle, the mean over the live cells of a width-1 cell block (the
        # overlap energy is a function of the nodes' summed densities, not of one cell). Absent: `repulsion_strength`, as before
        self.stiffness_block = params.get("repulsion_strength_block")
        # `contact_slack_block` (2026-10-28, the overlap sweep): the tolerated overlap s_0 of the SPARSE implementation read per cell from a width-1 cell block, so cells of one spec can differ. At a
        # lattice node it is the volume-fraction-weighted mean of the values of the cells that deposit there, s_0(n) = sum_A phi_A(n) s_A / sum_A phi_A(n): between two cells it is
        # their mean, inside one cell it is that cell's own. Absent: `contact_slack`, as before
        self.overlap_block = params.get("contact_slack_block")
        # `contact_edge_width_block` (2026-10-28, the width sweep): the interface width w of the SPARSE implementation read per cell from a width-1 cell block (a length, as `contact_edge_width`); the interface-tension force on a (cell, node)
        # key uses its own cell's w. Absent: `contact_edge_width`, as before
        self.width_block = params.get("contact_edge_width_block")
        if any(abs(self.gamma[i][j] - self.gamma[j][i]) > 1e-12 for i in range(len(self.gamma)) for j in range(len(self.gamma))):
            raise ValueError("pair_potential[cell_contact_dense]: contact_surface_tension must be symmetric")

    @staticmethod
    def _shift(A, axis, s, periodic):
        """A[..., n + s e_axis] on a dense lattice (axis 0..2 of dims 1..3): wrapped when `periodic`, else the edge repeated"""
        dim = 1 + axis
        if periodic:
            return torch.roll(A, -s, dim)
        n = A.shape[dim]
        if s > 0:
            return torch.cat([A.narrow(dim, 1, n - 1), A.narrow(dim, n - 1, 1)], dim)
        return torch.cat([A.narrow(dim, 0, 1), A.narrow(dim, 0, n - 1)], dim)

    def _extra_force(self, H, V, G, flat, idx, cid, phi_b, nC, Nw, shape, dx, dev, dt_):
        C = H.level(self.cells)
        nt = C.node_type.long().to(dev)
        T = int(nt.max()) + 1
        g = torch.as_tensor(self.gamma, device=dev, dtype=dt_)
        if g.shape[0] < T:
            raise ValueError(f"pair_potential[cell_contact_dense]: contact_surface_tension is {tuple(g.shape)}, the cell set has {T} types")
        g = g[:T, :T]
        per = self.period > 0.0
        Phi = torch.zeros(T, Nw, device=dev, dtype=dt_).index_add_(0, nt, phi_b)            # the phi of each type, summed
        A3 = Phi.view(T, *shape)
        L = -6.0 * A3
        for d in range(3):
            L = L + self._shift(A3, d, 1, per and d == 0) + self._shift(A3, d, -1, per and d == 0)
        Lflat = L.reshape(T, Nw)
        tp = nt[cid]                                                                         # the type of each point's cell
        S = torch.zeros(flat.shape, device=dev, dtype=dt_)
        for U in range(T):
            S = S + g[tp, U][:, None] * Lflat[U][flat]
        base = cid[:, None] * Nw
        phiA = phi_b.reshape(-1)
        Lown = -6.0 * phiA[base + flat]
        Ny, Nz = shape[1], shape[2]
        for d in range(3):
            for sgn in (1, -1):
                idn = idx.clone()
                idn[..., d] = idn[..., d] + sgn
                if d == 0 and per:
                    idn[..., 0] = torch.remainder(idn[..., 0], shape[0])
                else:
                    idn[..., d] = idn[..., d].clamp(0, shape[d] - 1)
                fnb = (idn[..., 0] * Ny + idn[..., 1]) * Nz + idn[..., 2]
                Lown = Lown + phiA[base + fnb]
        S = S - g[tp, tp][:, None] * Lown                                                    # the other cells only
        dEdphi = self.width * dx * S
        return -(V / dx ** 3)[:, None] * (G * dEdphi[..., None]).sum(1)


from plexus.operators.cleavage_ops import MPMGridUpdateBodyCSF  # noqa: E402
from plexus.operators.mpm_ops import sub_dt  # noqa: E402


@register_operator("mpm_grid_update", implementation="body_csf_balanced", family="mpm", set="field", kind="field",
                   title="solve grid velocities, tension without net push", equation=r"""$$\mathbf v_g=\frac{\mathbf p_g}{m_g},\qquad \mathbf v_g\mathrel{+}=\Delta t\,\frac{\sigma\,\kappa_b\,\nabla c_b}{m_g/\Delta x^3},\qquad \kappa_b=-\nabla\cdot\frac{\nabla c_b}{\lvert\nabla c_b\rvert}\ \text{ on body } b\text{'s block}$$""")
class MPMGridUpdateBodyCSFBalanced(MPMGridUpdateBodyCSF):
    r"""mpm_grid_update[body_csf] with each block's net surface-tension force removed: the discrete CSF on a deformed droplet does not integrate to zero, and a packed cluster was propelled by it.
    field -> field: as body_csf; `tension_on: liquid` takes the colour from the liquid-mass channel, `all` from the block's whole mass, `face` from the face-colour channel.

        \mathbf v_g=\frac{\mathbf p_g}{m_g},\qquad \mathbf v_g\mathrel{+}=\Delta t\,\frac{\sigma\,\kappa_b\,\nabla c_b}{m_g/\Delta x^3},\qquad \kappa_b=-\nabla\cdot\frac{\nabla c_b}{\lvert\nabla c_b\rvert}\ \text{ on body } b\text{'s block}
        v_g            the velocity of grid node g, in world length per time
        p_g            its momentum
        m_g            its mass
        \Delta t       the substep
        \sigma         the surface tension (`cell_surface_tension`, scaled per cell by `cell_surface_tension_block`), force per length
        \kappa_b       the curvature of body b's colour interface at the node, per world length
        c_b            body b's liquid colour on its own block, normalised to a volume fraction by csf_rho dx^3
        \Delta x       the grid spacing

    Reference: as body_csf; Plexus exp27, the cell adhesion testbench (this work, 2026-10) (e-series, 2026-10-14)."""
    PARAM_UNITS = {}
    SYMBOLS = {'v_g': 'the velocity of grid node g, in world length per time', 'p_g': 'its momentum', 'm_g': 'its mass', '\\Delta t': 'the substep', '\\sigma': 'the surface tension (`cell_surface_tension`, scaled per cell by `cell_surface_tension_block`), force per length', '\\kappa_b': "the curvature of body b's colour interface at the node, per world length", 'c_b': "body b's liquid colour on its own block, normalised to a volume fraction by csf_rho dx^3", '\\Delta x': 'the grid spacing'}
    PARAM_ROLES = {'tension_on': 'liquid_all_or_face'}

    def __init__(self, params, device="cpu"):
        super().__init__({k: v for k, v in params.items() if k != "tension_on"}, device)
        self.tension_on = str(params.get("tension_on", "liquid"))
        # `tension_on: face` (exp29, 2026-10-23): the interface (normal, curvature, band) is read from the whole mass as `all`,
        # and the force at a node is scaled by c / m, the face-weighted colour `mpm_scatter` deposited under
        # `colour_from_face` -- a tension per face (lateral, apical, basal) with one geometry, no internal interfaces
        if self.tension_on not in ("liquid", "all", "face"):
            raise ValueError("mpm_grid_update[body_csf_balanced]: tension_on is liquid, all or face")

    def _gmag(self, grad, eps):
        """|grad c| per node. The `body_csf_balanced_diff` implementation adds eps^2 under the root: sqrt has no gradient at 0."""
        return torch.sqrt(sum(q * q for q in grad))

    def forward(self, H, mask=None):
        g = H.field(self.at)
        if not getattr(g, "per_parent", None):
            raise ValueError("mpm_grid_update[body_csf_balanced]: needs a `per_parent` grid (one block per cell)")
        dt = sub_dt(H, self.dt_sub)
        floor = self._const("mass_floor", g)
        gm = g.m
        gv = g.mv / torch.where(gm > floor, gm, torch.ones_like(gm))[:, None]     # NOT gm.clamp(min=floor): the backward of the clamp's division is -mv / floor^2, which overflows to inf when the floor is tiny, and inf x a zero gradient is NaN (exp35 grad smoke, 2026-11-03); nodes under the floor are zeroed below anyway
        bf = getattr(H, "_mpm_body_accel", None)
        if bf is not None:
            gv = gv + dt * bf
        surf = self.surface_tension
        if surf > 0.0:
            D, dx, inv_dx = g.dim, float(g.dx), float(g.inv_dx)
            S = tuple(g.shape)
            nb = int(g.n_blocks)
            c = ((g.c if self.tension_on == "liquid" else g.m) / (self.csf_rho * dx ** D)).view(nb, *S)
            for _ in range(self.csf_smooth):
                for k in range(1, D + 1):
                    c = 0.25 * torch.roll(c, 1, k) + 0.5 * c + 0.25 * torch.roll(c, -1, k)
            grad = [(torch.roll(c, -1, k) - torch.roll(c, 1, k)) * (0.5 * inv_dx) for k in range(1, D + 1)]
            eps = self._const("csf_eps", g)
            gmag = self._gmag(grad, eps)
            nrm = [q / (gmag + eps) for q in grad]
            kappa = -sum((torch.roll(nrm[k - 1], -1, k) - torch.roll(nrm[k - 1], 1, k)) * (0.5 * inv_dx)
                         for k in range(1, D + 1))
            if self.csf_band > 0.0:
                gain = 1.0 / max(1.0 - 2.0 * self.csf_band, 1e-6)
                mfull = self.csf_rho * dx ** D
                fmask = ((c > self.csf_band) & (c < 1.0 - self.csf_band)
                         & (gm.view(nb, *S) > self.csf_band * mfull)).to(c.dtype) * gain
            else:
                fmask = (gmag > 0.02 * gmag.amax(dim=tuple(range(1, D + 1)), keepdim=True)).to(c.dtype)
            _cf = self._const("csf_mass_floor", g)
            inv_m = (dx ** D) / torch.where(gm > _cf, gm, torch.ones_like(gm))       # as above: a node under the floor is outside the tension band (fmask 0)
            if self.tension_block:
                Cl = H.level(g.per_parent)
                fac = Cl.state[:, Cl.state_schema[self.tension_block][0]].to(c.dtype)
                surf = surf * fac.view(nb, *([1] * D))
            if self.tension_on == "face":
                if float(g.c.abs().max()) == 0.0:
                    raise ValueError("mpm_grid_update[body_csf_balanced]: tension_on face needs `mpm_scatter` with `colour_from_face`")
                surf = surf * (g.c / torch.where(gm > _cf, gm, torch.ones_like(gm))).view(nb, *S)
            dv = dt * torch.stack([(surf * kappa * grad[k] * fmask).reshape(-1) * inv_m for k in range(D)], 1)
            # the block's net momentum change from the tension, removed as a uniform shift over its massive nodes
            has = (gm > floor).to(dv.dtype)
            mb = (gm * has).view(nb, -1)
            P = (mb[..., None] * dv.view(nb, -1, D)).sum(1)                           # [nb, D]
            Mb = mb.sum(1).clamp_min(1e-18)[:, None]                                  # NOT 1e-30: an empty block (a dead cell slot) has P = 0, and P / Mb differentiates to -P / Mb^2 = 0 / 1e-60 -> 0 / 0 = NaN in float32 (exp35 grad smoke, 2026-11-03); 1e-36 stays a normal number
            dv = (dv.view(nb, -1, D) - (P / Mb)[:, None, :]).reshape(-1, D) * has[:, None]
            gv = gv + dv
        g.v = torch.where((gm > floor)[:, None], gv, torch.zeros_like(gv))
        return {}



@register_operator("mpm_grid_update", implementation="body_csf_balanced_diff", family="mpm", set="field", kind="field",
                   title="solve grid velocities, tension without net push, differentiable", equation=r"""$$\mathbf v_g=\frac{\mathbf p_g}{m_g},\qquad \mathbf v_g\mathrel{+}=\Delta t\,\frac{\sigma\,\kappa_b\,\nabla c_b}{m_g/\Delta x^3},\qquad \kappa_b=-\nabla\cdot\frac{\nabla c_b}{\lvert\nabla c_b\rvert}\ \text{ on body } b\text{'s block}$$""")
class MPMGridUpdateBodyCSFBalancedDiff(MPMGridUpdateBodyCSFBalanced):
    r"""body_csf_balanced with |grad c| = sqrt(sum (d_k c)^2 + eps^2): the plain square root has an infinite derivative at 0, where the colour gradient is 0 on every interior node, so a tape through the CSF term returned NaN.
    field -> field: as body_csf_balanced.

        \mathbf v_g=\frac{\mathbf p_g}{m_g},\qquad \mathbf v_g\mathrel{+}=\Delta t\,\frac{\sigma\,\kappa_b\,\nabla c_b}{m_g/\Delta x^3},\qquad \kappa_b=-\nabla\cdot\frac{\nabla c_b}{\lvert\nabla c_b\rvert}\ \text{ on body } b\text{'s block}
        v_g            the velocity of grid node g, in world length per time
        p_g            its momentum
        m_g            its mass
        \Delta t       the substep
        \sigma         the surface tension (`cell_surface_tension`, scaled per cell by `cell_surface_tension_block`), force per length
        \kappa_b       the curvature of body b's colour interface at the node, per world length
        c_b            body b's liquid colour on its own block, normalised to a volume fraction by csf_rho dx^3
        \Delta x       the grid spacing

    Reference: as body_csf_balanced."""
    PARAM_UNITS = {}
    SYMBOLS = {'v_g': 'the velocity of grid node g, in world length per time', 'p_g': 'its momentum', 'm_g': 'its mass', '\\Delta t': 'the substep', '\\sigma': 'the surface tension (`cell_surface_tension`, scaled per cell by `cell_surface_tension_block`), force per length', '\\kappa_b': "the curvature of body b's colour interface at the node, per world length", 'c_b': "body b's liquid colour on its own block, normalised to a volume fraction by csf_rho dx^3", '\\Delta x': 'the grid spacing'}
    DIFFERENTIABLE = True

    def _gmag(self, grad, eps):
        return torch.sqrt(sum(q * q for q in grad) + float(eps) ** 2)


from plexus.operators.mpm_ops import ActiveForceSubstrateWalk  # noqa: E402


@register_operator("active_force", "pulse_to_contraction", family="mechanics", set="particle", kind="exchange",
                   model="gap_crawl", title="crawl and reach into gaps",
                   equation=r"""$$\mathbf a_i=f_c\,\chi_i\,\mathbf p_c+g\,e_i\,u_i\,\hat{\mathbf d}_i-\frac{g}{N_c}\sum_{j\in c}e_j\,u_j\,\hat{\mathbf d}_j,\qquad e_i=\mathrm{clip}\Big(\frac{1-n_i/n_{full}-e_0}{e_1-e_0},0,1\Big),\qquad d\theta_c=-k\,s_c\sin(\theta_c-\theta^{gap}_c)\,dt+\sqrt{2D_c}\,dW_c$$""")
class ActiveForceGapCrawl(ActiveForceSubstrateWalk):
    r"""a cell of an epithelium that crawls on its substrate and reaches into the gaps around it: substrate_walk's in-plane polarity and basal drive, plus a protrusion that extends the cell's base wherever it borders free space and a polarity steered toward it.
    particle -[containment]-> cell, once per frame (outside the substep block): emits a frame-level acceleration every substep sees; the per-cell sum of the reach is removed so the reach is force-free.

        \mathbf a_i=f_c\,\chi_i\,\mathbf p_c+g\,e_i\,u_i\,\hat{\mathbf d}_i-\frac{g}{N_c}\sum_{j\in c}e_j\,u_j\,\hat{\mathbf d}_j,\qquad e_i=\mathrm{clip}\Big(\frac{1-n_i/n_{full}-e_0}{e_1-e_0},0,1\Big),\qquad d\theta_c=-k\,s_c\sin(\theta_c-\theta^{gap}_c)\,dt+\sqrt{2D_c}\,dW_c
        a_i            the acceleration of point i of cell c
        f_c            the cell's drive
        \chi_i         1 in the basal band
        p_c            the in-plane polarity
        g              `reach`, the protrusion's strength
        e_i            point i's free-space measure, from the live points of any cell within `probe` of it (n_i against a full ball n_full, between `free_floor` e_0 and `free_ref` e_1)
        u_i            the point's outwardness
        \hat d_i       the direction into the free space
        N_c            the cell's point count
        \theta_c       the polarity angle
        k              `steer`, the turning rate toward the gaps
        s_c            the steering's strength from the cell's free-space sum
        \theta^{gap}_c the direction of the gaps
        D_c            `Dr`
        dW_c           a Wiener increment

    Reference: Farooqui, R. & Fenteany, G. (2005). J. Cell Sci. 118:51-63; Abercrombie, M. (1979). Nature 281:259-262; Carmona-Fontaine, C. et al. (2008). Nature 456:957-961; Plexus exp27, the cell adhesion testbench (this work, 2026-10)."""
    PARAM_UNITS = {'ahead': 'length', 'free_floor': 'fraction', 'free_ref': 'fraction', 'period': 'length', 'probe': 'length', 'reach': 'acceleration', 'reach_band': 'length', 'side': 'length', 'spacing': 'length', 'steer': 'rate', 'traction': 'fraction', 'workers': 'count'}
    SYMBOLS = {'a_i': 'the acceleration of point i of cell c', 'f_c': "the cell's drive", '\\chi_i': '1 in the basal band', 'p_c': 'the in-plane polarity', 'g': "`reach`, the protrusion's strength", 'e_i': "point i's free-space measure, from the live points of any cell within `probe` of it (n_i against a full ball n_full, between `free_floor` e_0 and `free_ref` e_1)", 'u_i': "the point's outwardness", '\\hat d_i': 'the direction into the free space', 'N_c': "the cell's point count", '\\theta_c': 'the polarity angle', 'k': '`steer`, the turning rate toward the gaps', 's_c': "the steering's strength from the cell's free-space sum", '\\theta^{gap}_c': 'the direction of the gaps', 'D_c': '`Dr`', 'dW_c': 'a Wiener increment'}
    PARAM_ROLES = dict(ActiveForceSubstrateWalk.PARAM_ROLES, reach="force_free_push_of_a_free_basal_edge",
                       probe="radius_of_the_free_space_count", steer="rate_of_turning_toward_free_space",
                       free_ref="free_fraction_of_an_open_basal_edge", free_floor="free_fraction_read_as_packed",
                       bounds="walls_counted_as_filled_lo_hi_per_plane_axis", spacing="point_spacing_world",
                       reach_band="height_above_the_lowest_point_that_may_reach", ahead="free_space_read_this_far_outward", period="period_along_x_world", traction="contact_area_of_full_drive", traction_fg="grip_face_pair_classes",
                       traction_on="cell_type_that_is_the_substrate", junctions="the_junction_set",
                       side="cell_edge_world", workers="kd_tree_query_threads")
    MECHANISM_TAGS = ActiveForceSubstrateWalk.MECHANISM_TAGS + ["contact_inhibition_of_locomotion", "gap_filling"]
    REFERENCE = ("Farooqui, R. & Fenteany, G. (2005). J. Cell Sci. 118:51-63; Abercrombie, M. (1979). Nature "
                 "281:259-262; Carmona-Fontaine, C. et al. (2008). Nature 456:957-961.")

    def __init__(self, params, device="cpu"):
        p = dict(params)
        p.setdefault("basal_band", 0.5)
        super().__init__(p, device)
        self.reach = float(params.get("reach", 0.0))
        self.workers = int(params.get("workers", 1))                      # kd-tree query threads (scipy default 1)
        self.probe = float(params.get("probe", 0.375))
        self.steer = float(params.get("steer", 0.0))
        # `reach_band` (default `basal_band`): the height above the cell's lowest point within which a point may reach
        # into free space; the walk's push stays on the basal band, so a tall reach band (the cell's whole height) is a
        # reach that does not also change where the cell pushes
        self.reach_band = float(params.get("reach_band", self.basal_band))
        # `ahead` (default 0, world): read the free space at the point shifted this far along its in-plane outward direction
        # instead of at the point -- a point inside the cell reads its own cell (filled), a point at the side reads what
        # is beside it, and the open air ABOVE a cell's top face no longer counts as free space to reach into
        self.ahead = float(params.get("ahead", 0.0))
        self.period = float(params.get("period", 0.0))                     # along x, as `contact_finder` (a ring)
        # `traction` (default 0 = off, world^2): the walk needs something to pull on. Each cell's drive is scaled by
        # min(1, sum of area x bound adhesion density over its apical-basal junctions with a cell of type `traction_on`
        # / traction): a cell that has left the substrate (extruded, or climbed on its neighbours) stops walking
        self.traction = float(params.get("traction", 0.0))
        self.traction_on = int(params.get("traction_on", 0))
        # `traction_fg` (default [3], apical-basal only): the face-pair classes of the junctions that count as a grip
        self.traction_fg = [int(v) for v in params.get("traction_fg", [3])]
        self.junctions = str(params.get("junctions", "junction"))
        self.free_ref = float(params.get("free_ref", 0.25))
        self.free_floor = float(params.get("free_floor", 0.08))
        self.h = float(params.get("spacing", 0.125))
        self.side = float(params.get("side", 1.0))
        b = params.get("bounds")
        self.bounds = None if b is None else [float(v) for v in b]
        if self.bounds is not None and len(self.bounds) != 4:
            raise ValueError("active_force[gap_crawl]: bounds is [lo, hi] of the plane's first axis then its second")
        self._gap = None
        self.last = {}

    def _step(self, h):
        if self.steer > 0.0 and self._gap is not None and self._theta is not None:
            tg, s = self._gap
            n = max(1, int(math.ceil(self.steer * h / 0.1)))
            for _ in range(n):
                self._theta = torch.remainder(self._theta - self.steer * s * torch.sin(self._theta - tg) * (h / n),
                                              2.0 * math.pi)
        super()._step(h)

    def _free_space(self, H, mask):
        """per point: e_i (0 off the band), d_i (in-plane unit from the centroid); per cell: the steering target."""
        from scipy.spatial import cKDTree
        lvl = H.level(self.at)
        X = lvl.get("pos").detach().cpu().numpy().astype(np.float64)
        alive = (lvl.occ > 0.5).cpu().numpy()
        nc = int(H.level(self.cell_set).state.shape[0])
        idx = H.lift_index(self.at, self.cell_set).cpu().numpy()
        sel = alive.copy()
        if mask is not None:
            sel &= (mask.detach().cpu().numpy() > 0)
        ax = self.plane_axis
        pa = [a for a in range(3) if a != ax]
        z = X[:, ax]
        zmin = np.full(nc, np.inf); np.minimum.at(zmin, idx[sel], z[sel])
        band = sel & (z <= zmin[idx] + float(self.reach_band))
        cnt = np.zeros(nc); np.add.at(cnt, idx[sel], 1.0)
        xb = np.zeros((nc, 3)); np.add.at(xb, idx[sel], X[sel]); xb /= np.maximum(cnt, 1.0)[:, None]
        e = np.zeros(len(X)); d = np.zeros((len(X), 3))
        bi = np.flatnonzero(band)
        if len(bi):
            r = self.probe
            v = X[bi] - xb[idx[bi]]; v[:, ax] = 0.0
            dl = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            Q = X[bi] + self.ahead * dl                                              # where the free space is read
            if self.period > 0.0:
                Xa = X[alive].copy(); Xa[:, 0] = np.mod(Xa[:, 0], self.period)
                Qw = Q.copy(); Qw[:, 0] = np.mod(Qw[:, 0], self.period)
                lo = Xa[:, 1:].min(0) - 1.0                    # every coordinate into [0, boxsize) for the periodic tree
                Xa[:, 1:] -= lo; Qw[:, 1:] -= lo
                n = cKDTree(Xa, boxsize=[self.period, 1.0e5, 1.0e5]).query_ball_point(Qw, r=r, return_length=True, workers=self.workers).astype(np.float64)
            else:
                n = cKDTree(X[alive]).query_ball_point(Q, r=r, return_length=True, workers=self.workers).astype(np.float64)
            filled = n / ((4.0 / 3.0) * math.pi * r ** 3 / self.h ** 3)
            if self.bounds is not None:
                for k, a in enumerate(pa):
                    for dist in (Q[:, a] - self.bounds[2 * k], self.bounds[2 * k + 1] - Q[:, a]):
                        dd = np.clip(dist, -r, r)
                        filled += np.where(dist < r, (r - dd) ** 2 * (2.0 * r + dd) / (4.0 * r ** 3), 0.0)
            free = np.clip(1.0 - filled, 0.0, 1.0)
            e[bi] = np.clip((free - self.free_floor) / max(self.free_ref - self.free_floor, 1e-9), 0.0, 1.0)
            d[bi] = dl
        G = np.zeros((nc, 3)); np.add.at(G, idx[bi], e[bi, None] * d[bi])
        n_face = (self.side / self.h) * (float(self.reach_band) / self.h)
        Gm = np.linalg.norm(G, axis=1)
        tg = np.arctan2(G[:, pa[1]], G[:, pa[0]])
        self._gap = (torch.as_tensor(tg, dtype=torch.float64), torch.as_tensor(np.minimum(1.0, Gm / n_face)))
        walkers = cnt > 0
        self.last = {"edge_points": int((e > 0).sum()),
                     "steer_mean": float(np.minimum(1.0, Gm / n_face)[walkers].mean()) if walkers.any() else 0.0}
        return e, d, idx, sel

    def _traction_gate(self, H, nc):
        J, C = H.level(self.junctions), H.level(self.cell_set)
        occ = (J.occ > 0.5).cpu().numpy()
        ca = np.rint(J.get("ca")[:, 0].cpu().numpy()).astype(int)[occ]
        cb = np.rint(J.get("cb")[:, 0].cpu().numpy()).astype(int)[occ]
        fg = np.rint(J.get("fg")[:, 0].cpu().numpy()).astype(int)[occ]
        w = (J.get("area")[:, 0].cpu().numpy() * np.clip(J.get("rho")[:, 0].cpu().numpy(), 0.0, None))[occ]
        nt = C.node_type.cpu().numpy()
        sa, sb = nt[ca] == self.traction_on, nt[cb] == self.traction_on
        m = np.isin(fg, self.traction_fg) & (sa != sb)                      # a grip class, one side the substrate
        crawler = np.where(sa[m], cb[m], ca[m])
        tot = np.zeros(nc); np.add.at(tot, crawler, w[m])
        return np.clip(tot / self.traction, 0.0, 1.0)

    def forward(self, H, mask=None):
        e, d, idx, sel = self._free_space(H, mask)
        out = super().forward(H, mask)
        a = out[self.at]
        if self.traction > 0.0:
            nc_ = int(H.level(self.cell_set).state.shape[0])
            gate = torch.as_tensor(self._traction_gate(H, nc_), device=a.device, dtype=a.dtype)
            a = a * gate[torch.as_tensor(idx, device=a.device)][:, None]
            out = {self.at: a}
            self.last["gate_mean"] = float(gate[torch.as_tensor(np.unique(idx[sel]), device=a.device)].mean())
        if self.reach == 0.0 or self._p is None:
            return out
        dev, dt_ = a.device, a.dtype
        nc = int(H.level(self.cell_set).state.shape[0])
        p = self._p.detach().cpu().numpy()
        u = np.maximum(0.0, (d * p[idx]).sum(1))
        push = self.reach * (e * u)[:, None] * d
        cnt = np.zeros(nc); np.add.at(cnt, idx[sel], 1.0)
        tot = np.zeros((nc, 3)); np.add.at(tot, idx[sel], push[sel])
        push[sel] -= (tot / np.maximum(cnt, 1.0)[:, None])[idx[sel]]               # force-free per cell
        push[~sel] = 0.0
        return {self.at: a + torch.as_tensor(push, device=dev, dtype=dt_)}


from plexus.operators.mpm_ops import ActiveForcePersistentWalk  # noqa: E402

@register_operator("active_force", "pulse_to_contraction", family="mechanics", set="particle", kind="exchange",
                   model="footprint_spreading", title="spread each cell to its target footprint",
                   equation=r"""$$A_c=2\pi\,\big\langle|\mathbf r_{t,i}|^{2}\big\rangle_{i\in c},\qquad A^{0}_c=a_0\,N_c^{\,2/3},\qquad
\mathbf a_i=k\,\frac{A^{0}_c-A_c}{A^{0}_c}\,\hat{\mathbf r}_{t,i}-\frac{1}{N_c}\sum_{j\in c}\mathbf a_j$$""")
class ActiveForceApicalArea(Exchange):
    r"""lateral growth of a polarised epithelial cell: each cell is given a target footprint that grows with its volume, and a force that spreads its points in the plane of the sheet until the footprint is reached.
    particle -[containment]-> cell, once per frame (outside the substep block): a frame-level acceleration every substep sees as the MPM body force; the per-cell mean is removed.

        A_c=2\pi\,\big\langle|\mathbf r_{t,i}|^{2}\big\rangle_{i\in c},\qquad A^{0}_c=a_0\,N_c^{\,2/3},\qquad \mathbf a_i=k\,\frac{A^{0}_c-A_c}{A^{0}_c}\,\hat{\mathbf r}_{t,i}-\frac{1}{N_c}\sum_{j\in c}\mathbf a_j
        A_c            the cell's footprint, 2 pi times the mean squared tangential offset of its points
        r_{t,i}        the tangential offset of point i from its cell's centroid (the normal n_c from `centre` through the centroid)
        A^0_c          the target footprint, a_0 N_c^{2/3} (`target_scale`, `growth_law_exponent`)
        N_c            the cell's point count
        a_i            the acceleration of point i
        k              `spreading_strength`, the spreading stiffness
        a_j            the other points' accelerations (the mean removed)

    Reference: Okuda, S. et al. (2018). Sci. Rep. 8:2386 (the apical area target); Guillot, C. & Lecuit, T. (2013). Science 340:1185-1189; Chan, C. J. et al. (2019). Nature 571:112-116; Plexus exp29 (this work, 2026-10-22)."""
    PARAM_UNITS = {'spreading_strength': 'acceleration', 'growth_law_exponent': 'dimensionless', 'max_deficit': 'fraction', 'target_scale': 'fraction', 'centre': 'length'}
    SYMBOLS = {'A_c': "the cell's footprint, 2 pi times the mean squared tangential offset of its points", 'r_{t,i}': "the tangential offset of point i from its cell's centroid (the normal n_c from `centre` through the centroid)", 'A^0_c': 'the target footprint, a_0 N_c^{2/3} (`target_scale`, `growth_law_exponent`)', 'N_c': "the cell's point count", 'a_i': 'the acceleration of point i', 'k': '`spreading_strength`, the spreading stiffness', 'a_j': "the other points' accelerations (the mean removed)"}
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["cell_set", "centre"]
    PARAM_ROLES = {"cell_set": "the_cell_set", "centre": "vesicle_centre", "spreading_strength": "spreading_acceleration_at_unit_deficit",
                   "growth_law_exponent": "power_of_the_point_count_in_the_target_footprint", "growth_law_exponent_block": "cell_block_with_each_cells_target_exponent",
                   "target_scale": "factor_on_the_seed_footprint_per_volume_two_thirds", "max_deficit": "clamp_of_the_relative_deficit",
                   "centre_block": "cell_block_with_each_cells_vesicle_centre", "spreading_strength_block": "cell_block_with_each_cells_k"}
    MECHANISM_TAGS = ["active_contraction", "apical_area_target", "lateral_growth", "force_free"]
    REFERENCE = ("Okuda, S. et al. (2018). Sci. Rep. 8:2386; Guillot, C. & Lecuit, T. (2013). Science 340:1185-1189; "
                 "Chan, C. J. et al. (2019). Nature 571:112-116.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.cell_set = str(params["cell_set"])
        self.c = [float(v) for v in params["centre"]]
        self.centre_block = params.get("centre_block")            # a cell block holding each cell's own vesicle centre (exp29 rig)
        kk = params.get("spreading_strength", 1.0)
        self.k_by_type = [float(v) for v in kk] if isinstance(kk, (list, tuple)) else None   # a list: the spreading acceleration of each cell TYPE
        self.k = float(kk) if self.k_by_type is None else 0.0
        self.k_block = params.get("spreading_strength_block")       # a cell block (width 1) holding each cell's own k: exp30's differentiable knob
        self.exponent = float(params.get("growth_law_exponent", 2.0 / 3.0))
        # `growth_law_exponent_block` (exp33, 2026-10-25, Cedric: "footprint drive reaches a ceiling, maybe this is the reason the cells do not grow
        # laterally"): a cell block (width 1) holding each cell's own exponent e_c, THE TARGET LAW AS A KNOB. The target becomes
        # a_0 N_c^exponent (N_c / N_ref)^(e_c - exponent), N_ref the seed's median count: at the seed it is the same target whatever e_c, and
        # e_c says how the footprint scales with the mass from there (1 a sheet of constant height, above 1 cells that thin as they grow --
        # the vertex twin's area grows as its mass^1.2-1.4 over the record). An unseeded block (0) reads `growth_law_exponent`. Differentiable in e_c.
        self.exponent_block = params.get("growth_law_exponent_block")
        self.target_scale = float(params.get("target_scale", 1.0))
        self.max_deficit = float(params.get("max_deficit", 1.0))
        self._a0 = None                                            # the sheet's footprint per N^exponent, read once on the seed
        self._nref = None                                          # the seed's median point count per cell (the exponent block's pivot)
        self.last = {}

    def forward(self, H, mask=None):
        P = H.level(self.at)
        X = P.get("pos")
        dev, dt_ = X.device, X.dtype
        C = H.level(self.cell_set)
        nC = int(C.state.shape[0])
        w = (P.occ > 0.5).to(dt_)
        par = P.parent.long().clamp(0, nC - 1)
        cnt = torch.zeros(nC, device=dev, dtype=dt_).index_add_(0, par, w)
        live_c = cnt > 0
        cen = torch.zeros(nC, 3, device=dev, dtype=dt_).index_add_(0, par, X * w[:, None]) / cnt.clamp(min=1.0)[:, None]
        if self.centre_block:
            b0, b1 = C.state_schema[self.centre_block]
            n = cen - C.state[:, b0:b1].to(dt_)
        else:
            n = cen - torch.tensor(self.c, device=dev, dtype=dt_)
        n = n / n.norm(dim=1, keepdim=True).clamp(min=1e-9)
        r = X - cen[par]
        rt = r - (r * n[par]).sum(1, keepdim=True) * n[par]
        A = 2.0 * math.pi * torch.zeros(nC, device=dev, dtype=dt_).index_add_(0, par, (rt * rt).sum(1) * w) / cnt.clamp(min=1.0)
        if self._a0 is None:                                       # once, on the seed: the sheet's aspect ratio
            self._a0 = float((A[live_c] / cnt[live_c].pow(self.exponent)).median()) * self.target_scale
            self._nref = float(cnt[live_c].median())
        A0 = self._a0 * cnt.pow(self.exponent)
        if self.exponent_block and self.exponent_block in C.state_schema:
            ec = C.state[:, C.state_schema[self.exponent_block][0]].to(dt_)
            ec = torch.where(ec > 0, ec, torch.full_like(ec, self.exponent))
            A0 = A0 * (cnt.clamp(min=1.0) / max(self._nref or 1.0, 1.0)).pow(ec - self.exponent)
        d = ((A0 - A) / A0.clamp(min=1e-9)).clamp(-self.max_deficit, self.max_deficit)
        d = torch.where(live_c, d, torch.zeros_like(d))
        rhat = rt / rt.norm(dim=1, keepdim=True).clamp(min=1e-6)
        if self.k_block:
            kc = C.state[:, C.state_schema[self.k_block][0]].to(dt_)
        elif self.k_by_type is not None:
            kt = torch.tensor(self.k_by_type, device=dev, dtype=dt_)
            kc = kt[C.node_type.long().to(dev).clamp(0, len(self.k_by_type) - 1)]
        else:
            kc = torch.full((nC,), self.k, device=dev, dtype=dt_)
        acc = kc[par][:, None] * d[par][:, None] * rhat * w[:, None]
        mean_c = torch.zeros(nC, 3, device=dev, dtype=dt_).index_add_(0, par, acc) / cnt.clamp(min=1.0)[:, None]
        acc = (acc - mean_c[par]) * w[:, None]                     # force-free per cell
        if mask is not None:
            acc = acc * mask[:, None].to(dt_)
        with torch.no_grad():                                      # diagnostics only: never a tape through `self.last`
            self.last = {"a0": self._a0, "footprint_mean": float(A[live_c].mean()), "target_mean": float(A0[live_c].mean()),
                         "deficit_mean": float(d[live_c].mean())}
        return {self.at: acc}




@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="spheroid",
                   title="seed a spheroid of inner cells under an outer layer", equation=r"""$$\mathbf c^{in}_k=\mathbf o+a(\mathbf k+\boldsymbol\epsilon_k),\quad \mathbf c^{out}_m=\mathbf o+R_s\,\hat{\mathbf f}_m,\ \ R_s=\max_k\lvert\mathbf c^{in}_k-\mathbf o\rvert+h,\qquad \pi(i)=\arg\min\lvert\mathbf x_i-\mathbf c\rvert$$""")
class SeedSpheroid(Seed):
    r"""a spheroid of two identities: the live cells of type 0 are the inner cells, a ball on a jittered lattice; the live cells of type 1 the outer layer, one cell thick on a Fibonacci sphere over it; each cell a Voronoi cell of points.
    pt -> pt, cell: `pid`, `cellc`, `tcol` on the points; `pos`, `pol` (zero), `adh` (6 classes at `adh`) on the cells.

        \mathbf c^{in}_k=\mathbf o+a(\mathbf k+\boldsymbol\epsilon_k),\quad \mathbf c^{out}_m=\mathbf o+R_s\,\hat{\mathbf f}_m,\ \ R_s=\max_k\lvert\mathbf c^{in}_k-\mathbf o\rvert+h,\qquad \pi(i)=\arg\min\lvert\mathbf x_i-\mathbf c\rvert
        c^{in}_k       the centre of inner cell k, the n_in nearest lattice sites to the centre
        o              `centre`
        a              `spacing`
        \epsilon_k     the jitter
        c^{out}_m      the centre of outer cell m, on a Fibonacci sphere
        R_s            the outer radius: the largest inner-centre radius plus `shell`
        h              `shell`
        \pi(i)         the cell of point i (Voronoi, within `inner_extent` / `outer_extent`)

    Reference: Plexus exp27, the cell adhesion testbench (this work, 2026-10) (2026-10-18; Shaohe Wang: a 3D spheroid before the basement membrane)."""
    PARAM_UNITS = {'centre': 'length', 'point_spacing': 'length', 'r_inner': 'length', 'seed': 'count', 'shell': 'length', 'spacing': 'length', 'inner_extent': 'length', 'outer_extent': 'length', 'adh': 'dimensionless', 'live_per_cell': 'count'}
    SYMBOLS = {'c^{in}_k': 'the centre of inner cell k, the n_in nearest lattice sites to the centre', 'o': '`centre`', 'a': '`spacing`', '\\epsilon_k': 'the jitter', 'c^{out}_m': 'the centre of outer cell m, on a Fibonacci sphere', 'R_s': 'the outer radius: the largest inner-centre radius plus `shell`', 'h': '`shell`', '\\pi(i)': 'the cell of point i (Voronoi, within `inner_extent` / `outer_extent`)'}
    REQUIRES_PARAMS = ["cell_set", "centre"]
    SUPPORTED_DIMS = [3]
    PARAM_ROLES = {"live_per_cell": "points_seeded_per_cell_the_rest_of_the_block_dormant", "cell_set": "the_cell_set", "centre": "spheroid_centre", "spacing": "inner_cell_centre_lattice_spacing",
                   "hollow": "inner_cells_a_layer_over_an_empty_centre", "r_inner": "radius_of_the_inner_layer_centres",
                   "inner_extent": "inner_layer_extent_inward_of_its_centres",
                   "shell": "inner_to_outer_centre_distance", "outer_extent": "outer_cells_extent_beyond_their_centres",
                   "point_spacing": "material_point_lattice_spacing", "adh": "isotropic_adhesion_maximum", "seed": "rng_seed"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.cell_set = str(params["cell_set"])
        self.c = np.array([float(v) for v in params["centre"]])
        self.sp = float(params.get("spacing", 1.05))
        self.shell = float(params.get("shell", 1.0))
        self.ext = float(params.get("outer_extent", 0.6))
        self.h = float(params.get("point_spacing", 0.125))
        self.adh = float(params.get("adh", 1.0))
        self.seed = int(params.get("seed", 0))
        # `hollow` (default false): the type-0 cells are a LAYER too, on a Fibonacci sphere of radius `r_inner` (the centres), and
        # the points keep only the shell from `r_inner - inner_extent` outward -- two layers over an empty centre
        self.hollow = bool(params.get("hollow", False))
        self.r_inner = float(params.get("r_inner", 0.0))
        self.inner_ext = float(params.get("inner_extent", 0.5))
        self.live_per_cell = params.get("live_per_cell")                   # (exp36, 2026-10-31) as seed_positions[vesicle]: the live head of the block; None = the whole block
        if self.hollow and self.r_inner <= 0.0:
            raise ValueError("seed_positions[spheroid]: hollow needs r_inner, the radius of the inner layer's centres")

    def forward(self, H, mask=None):
        from scipy.spatial import cKDTree
        from plexus.operators.gastrulation_ops import _lattice
        P, C = H.level(self.at), H.level(self.cell_set)
        rng = np.random.default_rng(self.seed)
        live = np.flatnonzero(C.occ.detach().cpu().numpy() > 0.5)
        nt = C.node_type.detach().cpu().numpy()
        in_ids, out_ids = live[nt[live] == 0], live[nt[live] == 1]
        n_in, n_out = len(in_ids), len(out_ids)
        if self.hollow:
            kk = np.arange(n_in) + 0.5
            ph = np.arccos(1.0 - 2.0 * kk / n_in); tt = math.pi * (1.0 + math.sqrt(5.0)) * kk
            cand = self.c + self.r_inner * np.stack([np.sin(ph) * np.cos(tt), np.sin(ph) * np.sin(tt), np.cos(ph)], 1)
            r_edge = self.r_inner
        else:
            R0 = 0.6 * (3.0 * n_in * self.sp ** 3 / (4.0 * math.pi)) ** (1.0 / 3.0) + 2 * self.sp
            cand = _lattice(self.sp, self.c - R0, self.c + R0, rng, jitter=0.15)
            cand = cand[np.argsort(np.linalg.norm(cand - self.c, axis=1))[:n_in]]
            r_edge = float(np.linalg.norm(cand - self.c, axis=1).max())
        R_s = r_edge + self.shell
        k = np.arange(n_out) + 0.5
        phi = np.arccos(1.0 - 2.0 * k / n_out); th = math.pi * (1.0 + math.sqrt(5.0)) * k
        outer = self.c + R_s * np.stack([np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)], 1)
        centres = np.zeros((C.state.shape[0], 3))
        centres[in_ids] = cand[rng.permutation(n_in)]
        centres[out_ids] = outer
        R_ball = R_s + self.ext
        pts = _lattice(self.h, self.c - R_ball, self.c + R_ball, rng, jitter=0.15)
        rp_ = np.linalg.norm(pts - self.c, axis=1)
        pts = pts[(rp_ < R_ball) & (rp_ > (r_edge - self.inner_ext if self.hollow else 0.0))]
        owner = live[cKDTree(centres[live]).query(pts, k=1)[1]]
        parent = P.parent.detach().cpu().numpy()
        dev, dt_ = P.state.device, P.state.dtype
        occ = torch.zeros_like(P.occ)
        short = 0
        for c in live:
            blk = np.flatnonzero(parent == c)
            if self.live_per_cell:
                blk = blk[: int(self.live_per_cell)]
            own = pts[owner == c]
            vol = max(len(own), 1) * self.h ** 3
            if len(own) == 0:
                own = centres[c][None] + rng.normal(0, 0.1 * self.sp, (8, 3)); vol = 8 * self.h ** 3
            if len(own) >= len(blk):
                take = own[rng.choice(len(own), len(blk), replace=False)]
            else:
                short += 1
                extra = own[rng.integers(0, len(own), len(blk) - len(own))] + rng.normal(0, 0.3 * self.h, (len(blk) - len(own), 3))
                take = np.concatenate([own, extra])
            idx = torch.as_tensor(blk, device=dev)
            _mpm_reset_points(P, idx, torch.as_tensor(take, device=dev, dtype=dt_),
                              torch.full((len(blk),), vol / len(blk), device=dev, dtype=dt_))
            _set_block(P, "pid", idx, float(c)); _set_block(P, "cellc", idx, float(c % 20))
            if "tcol" in P.state_schema:
                _set_block(P, "tcol", idx, float(nt[c]))
            occ[idx] = 1.0
        P.occ = occ
        ca, cb = C.state_schema["pos"]
        st = C.state.clone(); st[:, ca:cb] = torch.as_tensor(centres, dtype=st.dtype, device=st.device)
        if "adh" in C.state_schema:
            a0, a1 = C.state_schema["adh"]; st[:, a0:a1] = self.adh
        C.state = st
        sizes = np.array([(owner == c).sum() for c in live]) * self.h ** 3
        print(f"[seed_positions:spheroid] {n_in} inner + {n_out} outer cells, inner centres to radius {r_edge:.2f}, outer centres at "
              f"{R_s:.2f}, ball {R_ball:.2f}, {int(occ.sum()):,} points; region volume per cell median {np.median(sizes):.2f} "
              f"(inner {np.median(sizes[nt[live] == 0]):.2f}, outer {np.median(sizes[nt[live] == 1]):.2f}); {short} cells padded", flush=True)
        return {}


@register_operator("active_force", "pulse_to_contraction", family="mechanics", set="particle", kind="exchange",
                   model="sphere_crawl", title="crawl cells over a sphere and reach into gaps",
                   equation=r"""$$\mathbf a_i=f_c\,\chi_i\,\mathbf p_c+g\,e_i\,u_i\,\hat{\mathbf d}_i-\overline{(\cdot)}_c,\qquad \mathbf p_c\perp\hat{\mathbf n}_c=\frac{\mathbf x_c-\mathbf o}{\lvert\mathbf x_c-\mathbf o\rvert}$$""")
class ActiveForceSphereCrawl(ActiveForceGapCrawl):
    r"""gap_crawl on a sphere: the cell's surface normal is the radial direction from `centre`, its polarity lies in the tangent plane and turns by rotational diffusion about the normal, the push is on the cell's innermost band.
    particle -[containment]-> cell, once per frame as gap_crawl: p_c projected on the new tangent plane, rotated about the normal, turned toward the gaps; the per-cell mean removed (`zero_net`).

        \mathbf a_i=f_c\,\chi_i\,\mathbf p_c+g\,e_i\,u_i\,\hat{\mathbf d}_i-\overline{(\cdot)}_c,\qquad \mathbf p_c\perp\hat{\mathbf n}_c=\frac{\mathbf x_c-\mathbf o}{\lvert\mathbf x_c-\mathbf o\rvert}
        a_i            the acceleration of point i
        f_c            the drive
        \chi_i         1 in the innermost band
        p_c            the tangent-plane polarity
        g              `reach`
        e_i            the free-space measure
        u_i            the outwardness
        \hat d_i       the direction into free space
        \overline{(\cdot)}_c the per-cell mean, removed
        \hat n_c       the radial unit vector from the centre o to the cell's centroid x_c

    Reference: Plexus exp27, the cell adhesion testbench (this work, 2026-10); as gap_crawl."""
    PARAM_UNITS = {'centre': 'length'}
    SYMBOLS = {'a_i': 'the acceleration of point i', 'f_c': 'the drive', '\\chi_i': '1 in the innermost band', 'p_c': 'the tangent-plane polarity', 'g': '`reach`', 'e_i': 'the free-space measure', 'u_i': 'the outwardness', '\\hat d_i': 'the direction into free space', '\\overline{(\\cdot)}_c': 'the per-cell mean, removed', '\\hat n_c': "the radial unit vector from the centre o to the cell's centroid x_c"}
    PARAM_ROLES = dict(ActiveForceGapCrawl.PARAM_ROLES, centre="centre_of_the_sphere")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        if params.get("centre") is None:
            raise ValueError("active_force[sphere_crawl]: centre, the sphere's centre, is required")
        self.c0 = np.array([float(v) for v in params["centre"]])
        # `zero_net` (default true): the push of the outer cells is internal to the spheroid -- each cell grips its neighbours and
        # they take the reaction -- so the mass-weighted mean acceleration of ALL the live points is taken off every point; a
        # free spheroid has no floor or wall to carry the reaction and would otherwise be pushed along by the sum of its cells'
        # random drives (exp27 sph1, 2026-10-18: drifted to the box wall within the first hour, cells flung off)
        self.zero_net = bool(params.get("zero_net", True))
        self._p3 = None
        self._G = None

    def _ball_counts(self, lvl, X, alive, Q, rad):
        """per query point Q [m, 3]: the number of live points (`X[alive]`, [n, 3] float64) within `rad` of it. A method so that
        `active_force[sphere_crawl_hashgrid]` can replace the kd-tree and nothing else."""
        from scipy.spatial import cKDTree
        return cKDTree(X[alive]).query_ball_point(Q, r=rad, return_length=True, workers=self.workers).astype(np.float64)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        Xt = lvl.get("pos")
        dev, dt_ = Xt.device, Xt.dtype
        X = Xt.detach().cpu().numpy().astype(np.float64)
        alive = (lvl.occ > 0.5).cpu().numpy()
        nc = int(H.level(self.cell_set).state.shape[0])
        idx = H.lift_index(self.at, self.cell_set).cpu().numpy()
        sel = alive.copy()
        if mask is not None:
            sel &= (mask.detach().cpu().numpy() > 0)
        h = float(getattr(H, "dt", 1.0))
        cnt = np.zeros(nc); np.add.at(cnt, idx[alive], 1.0)
        xb = np.zeros((nc, 3)); np.add.at(xb, idx[alive], X[alive]); xb /= np.maximum(cnt, 1.0)[:, None]
        nrm = xb - self.c0
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        if self._p3 is None or self._p3.shape[0] != nc:
            ActiveForcePersistentWalk._init_cells(self, nc)
            self._p3 = self._p.detach().cpu().numpy().copy()
            fresh = True
        else:
            fresh = False
        p = self._p3
        p = p - (p * nrm).sum(1, keepdims=True) * nrm                                  # onto the tangent plane
        pn = np.linalg.norm(p, axis=1, keepdims=True)
        bad = pn[:, 0] < 1e-6
        if bad.any():
            alt = np.cross(nrm[bad], np.array([1.0, 0.3, 0.1])); p[bad] = alt; pn[bad] = np.linalg.norm(alt, axis=1, keepdims=True)
        p = p / pn
        if not fresh:
            Dc = self._Dc.detach().cpu().numpy()
            n_sub = max(1, int(math.ceil(float(Dc.max()) * h / 0.01)))
            hs = h / n_sub
            for _ in range(n_sub):
                z = torch.randn(nc, generator=self._gen, dtype=torch.float64).numpy()
                dth = np.sqrt(2.0 * Dc * hs) * z
                if self.steer > 0.0 and self._G is not None:
                    Gv, sg = self._G
                    Gh = Gv - (Gv * nrm).sum(1, keepdims=True) * nrm
                    Gh = Gh / np.maximum(np.linalg.norm(Gh, axis=1, keepdims=True), 1e-9)
                    cross = np.cross(p, Gh)
                    ang = np.arctan2((cross * nrm).sum(1), (p * Gh).sum(1))          # signed angle from p toward the gap
                    dth = dth + self.steer * sg * np.sin(ang) * hs
                p = p * np.cos(dth)[:, None] + np.cross(nrm, p) * np.sin(dth)[:, None]
            p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
        self._p3 = p
        self._p = torch.as_tensor(p, dtype=torch.float64)
        # the drive on the innermost band, the cell's total unchanged
        rr = np.linalg.norm(X - self.c0, axis=1)
        rmin = np.full(nc, np.inf); np.minimum.at(rmin, idx[sel], rr[sel])
        chi = (sel & (rr <= rmin[idx] + float(self.basal_band))).astype(np.float64)
        n_all = np.zeros(nc); np.add.at(n_all, idx[sel], 1.0)
        n_bas = np.zeros(nc); np.add.at(n_bas, idx[sel], chi[sel])
        chi = chi * (n_all / np.maximum(n_bas, 1.0))[idx]
        amp = (self._fc * self._s).detach().cpu().numpy()
        a = (amp[idx] * chi)[:, None] * p[idx]
        a[~sel] = 0.0
        gate = np.ones(nc)
        if self.traction > 0.0:
            gate = self._traction_gate(H, nc)
            a = a * gate[idx][:, None]
            self.last["gate_mean"] = float(gate[np.unique(idx[sel])].mean()) if sel.any() else 0.0
        # free space to the side, in the tangent plane
        band = sel & (rr <= rmin[idx] + float(self.reach_band))
        bi = np.flatnonzero(band)
        e = np.zeros(len(X)); d = np.zeros((len(X), 3))
        G = np.zeros((nc, 3))
        if len(bi):
            rad = self.probe
            v = X[bi] - xb[idx[bi]]
            v = v - (v * nrm[idx[bi]]).sum(1, keepdims=True) * nrm[idx[bi]]
            dl = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            Q = X[bi] + self.ahead * dl
            n = self._ball_counts(lvl, X, alive, Q, rad)
            filled = n / ((4.0 / 3.0) * math.pi * rad ** 3 / self.h ** 3)
            free = np.clip(1.0 - filled, 0.0, 1.0)
            e[bi] = np.clip((free - self.free_floor) / max(self.free_ref - self.free_floor, 1e-9), 0.0, 1.0)
            d[bi] = dl
            np.add.at(G, idx[bi], e[bi, None] * d[bi])
        n_face = (self.side / self.h) * (float(self.reach_band) / self.h)
        sg = np.minimum(1.0, np.linalg.norm(G, axis=1) / n_face)
        self._G = (G, sg)
        self.last = dict(self.last, edge_points=int((e > 0).sum()), steer_mean=float(sg[cnt > 0].mean()) if (cnt > 0).any() else 0.0)
        if self.reach > 0.0:
            u = np.maximum(0.0, (d * p[idx]).sum(1))
            push = self.reach * (e * u)[:, None] * d
            tot = np.zeros((nc, 3)); np.add.at(tot, idx[sel], push[sel])
            push[sel] -= (tot / np.maximum(n_all, 1.0)[:, None])[idx[sel]]
            push[~sel] = 0.0
            # the reach is gated like the walk: a cell with no grip has free space on every side, so every point of it would
            # reach outward -- a force-free but strong stretching (reach 8 against a cortex of 0.15) that tore the cells
            # extruded from the sphere into clouds filling their whole grid block (exp27 sph_dz, 2026-10-18)
            push *= gate[idx][:, None]
            a = a + push
        if self.zero_net:
            mp = getattr(lvl, "mass", None)
            m = mp.detach().cpu().numpy().astype(np.float64) if mp is not None else np.ones(len(X))
            m = m * alive
            net = (m[:, None] * a).sum(0) / max(float(m.sum()), 1e-30)
            self.last["net_acc_before"] = float(np.linalg.norm(net))
            a = a - net[None, :] * alive[:, None]
        return {self.at: torch.as_tensor(a, device=dev, dtype=dt_)}


@register_operator("state_dump", family="io", set="particle", kind="lateral",
                   title="save the current state for the live view", equation=r"""$$\text{file}\leftarrow\{\mathrm{pt\_pos},\ \mathrm{pt\_occ},\ \mathrm{pt\_pid},\ \mathrm{cell\_pos},\ \mathrm{cell\_node\_type},\ t,\ T\}\quad\text{every } E \text{ frames}$$""")
class StateDump(Lateral):
    r"""every `every` frames, writes the positions of the live points and of the cells to a file in the keys of a one-row trajectory, so a watcher can read the live state of a running job.
    pt -> file: reads `pos`, `pid`, `occ` (and the cells' `pos`, `node_type`), writes `<dir>/live_state.npz` with `frame` and `n_frames`.

        \text{file}\leftarrow\{\mathrm{pt\_pos},\ \mathrm{pt\_occ},\ \mathrm{pt\_pid},\ \mathrm{cell\_pos},\ \mathrm{cell\_node\_type},\ t,\ T\}\quad\text{every } E \text{ frames}
        t              the frame
        T              `n_frames`
        E              `every`, the dump interval, frames

    Reference: none: a watcher's probe, not a mechanism. Plexus (this work)."""
    PARAM_UNITS = {'every': 'frames'}
    SYMBOLS = {'t': 'the frame', 'T': '`n_frames`', 'E': '`every`, the dump interval, frames'}
    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["dir"]
    PARAM_ROLES = {"dir": "the_run_data_folder", "every": "frames_between_dumps", "file": "file_name", "cells": "the_cell_set", 'nodes': 'a_node_set_also_dumped'}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.dir = str(params["dir"])
        self.every = int(params.get("every", 60))
        self.file = str(params.get("file", "live_state.npz"))
        self.cells = str(params.get("cells", "cell"))
        self.nodes = params.get("nodes")                        # a basement membrane node set to write as well (`bm__pos`, the live nodes)

    def forward(self, H, mask=None):
        fr = int(getattr(H, "frame", 0) or 0)
        if fr % max(self.every, 1) != 0:
            return {}
        P, C = H.level(self.at), H.level(self.cells)
        occ = (P.occ > 0.5).detach().cpu().numpy()
        pos = P.get("pos").detach().cpu().numpy().astype(np.float32)
        pid = P.get("pid").detach().cpu().numpy().astype(np.float32) if "pid" in P.state_schema else np.zeros((len(pos), 1), np.float32)
        cpos = C.get("pos").detach().cpu().numpy().astype(np.float32)
        os.makedirs(self.dir, exist_ok=True)
        tmp = os.path.join(self.dir, "." + self.file + ".tmp.npz")
        extra = {}
        if self.nodes:
            Nd = H.level(self.nodes)
            extra["bm__pos"] = Nd.get("pos").detach()[Nd.alive].cpu().numpy().astype(np.float32)[None]
        np.savez(tmp, pt__pos=pos[None], pt__occ=occ[None], pt__pid=pid[None], cell__pos=cpos[None],
                 cell__node_type=C.node_type.detach().cpu().numpy().astype(np.int16), frame=np.int64(fr),
                 n_frames=np.int64(getattr(H, "n_frames", 0) or 0), **extra)
        os.replace(tmp, os.path.join(self.dir, self.file))
        return {}


# ---------------------------------------------------------------------------------------------------------------------
# GROWTH AND DIVISION OF THE LAYERS OF A SPHEROID, AT DIFFERENT RATES (exp27, 2026-10-19: Cedric, "cell division and cell growth
# on the top layer, faster than the inner layer -- may division and growth favour shape change, hence cell motility")
# ---------------------------------------------------------------------------------------------------------------------
from plexus.operators.cleavage_ops import CellDivideCleave                    # noqa: E402
from plexus.operators.gastrulation_ops import CellPopulateMPMPoints             # noqa: E402


@register_operator("cell_populate", model="mpm_points_typed", set="cell", kind="divide", family="population",
                   title="grow cells by adding points, a rate per cell type",
                   equation=r"""$$\frac{dN_c}{dt}=g_{\tau(c)}\,N_c,\qquad \text{mark}_c=\big[N_c\ge N^{\ast}_c\big],\quad N^{\ast}_c=N^{\ast}(1+j\,u_c),\ u_c\sim U(-1,1)$$""")
class CellPopulateMPMPointsTyped(CellPopulateMPMPoints):
    r"""cell_populate[mpm_points] with a growth rate per cell type and a division threshold that differs from cell to cell.
    cell -> cell and its point set: each named type grows exponentially at its own rate (as `rate`), a cell is marked for division at a jittered point count; the mother class's forward runs once per type.

        \frac{dN_c}{dt}=g_{\tau(c)}\,N_c,\qquad \text{mark}_c=\big[N_c\ge N^{\ast}_c\big],\quad N^{\ast}_c=N^{\ast}(1+j\,u_c),\ u_c\sim U(-1,1)
        N_c            the point count of cell c
        g_{\tau(c)}    `rates`, the growth rate of the cell's type, per time (a type not named does not grow)
        mark_c         the division mark
        N^*_c          the cell's threshold
        N^*            `target`
        j              `target_jitter`
        u_c            uniform in (-1, 1), redrawn after a division

    Reference: Plexus exp27, the cell adhesion testbench (this work, 2026-10); the exponential growth of cell_populate[mpm_points]."""
    PARAM_UNITS = {'target': 'count', 'target_jitter': 'fraction'}
    SYMBOLS = {'N_c': 'the point count of cell c', 'g_{\\tau(c)}': "`rates`, the growth rate of the cell's type, per time (a type not named does not grow)", 'mark_c': 'the division mark', 'N^*_c': "the cell's threshold", 'N^*': '`target`', 'j': '`target_jitter`', 'u_c': 'uniform in (-1, 1), redrawn after a division'}
    REQUIRES_PARAMS = ["points", "rates", "target"]
    PARAM_ROLES = dict(CellPopulateMPMPoints.PARAM_ROLES, rates="per_type_exponential_growth_rate_per_time_unit",
                       target_jitter="relative_spread_of_the_division_threshold")

    def __init__(self, params, device="cpu"):
        own = ("rates", "target_jitter")
        super().__init__({**{k: v for k, v in params.items() if k not in own}, "rate": 0.0}, device)
        self.rates = {str(k): float(v) for k, v in dict(params["rates"]).items()}
        self.jit = float(params.get("target_jitter", 0.0))
        self.target0 = int(params["target"])
        self._tg = None
        self._last = None

    def forward(self, H, mask=None):
        C, P = H.level(self.at), H.level(self.points)
        nC = int(C.state.shape[0])
        live = (C.occ > 0.5)
        nt = C.node_type.detach().cpu().numpy()
        names = list(getattr(C, "type_names", []) or [])
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
        try:
            for tname, g in self.rates.items():
                if tname not in names:
                    raise ValueError(f"cell_populate[mpm_points_typed]: no cell type {tname!r} (the set has {names})")
                C.occ = (live & torch.as_tensor(nt == names.index(tname), device=live.device)).to(C.occ.dtype)
                self.g = g
                super().forward(H, mask)
        finally:
            C.occ = live.to(C.occ.dtype)
        return {}


@register_operator("cell_divide", model="cleave_typed", set="particle", kind="divide", family="population",
                   title="cleave cells, the daughter of the same type", equation=r"""$$w_i=1-a\,y_i,\quad \mathbf n_c=\frac{\sum_i w_i v_i\mathbf x_i}{\sum_i w_i v_i},\quad \mathbf u_c=\arg\max_{\lvert\mathbf u\rvert=1}\mathbf u^{\!\top}\Big(\sum_i w_i v_i(\mathbf x_i-\mathbf n_c)(\mathbf x_i-\mathbf n_c)^{\!\top}\Big)\mathbf u,\quad i\to\begin{cases}c&(\mathbf x_i-\mathbf n_c)\cdot\mathbf u_c\ge0\\ d&\text{else}\end{cases}$$""")
class CellDivideCleaveTyped(CellDivideCleave):
    r"""cell_divide[cleave] whose daughter takes its mother's cell type (`node_type`) and, with `inherit`, cell blocks of any width as they are (a width-2 `chem`, say).
    particle cell -> two particle cells: as cleave, plus the type and the inherited blocks.

        w_i=1-a\,y_i,\quad \mathbf n_c=\frac{\sum_i w_i v_i\mathbf x_i}{\sum_i w_i v_i},\quad \mathbf u_c=\arg\max_{\lvert\mathbf u\rvert=1}\mathbf u^{\!\top}\Big(\sum_i w_i v_i(\mathbf x_i-\mathbf n_c)(\mathbf x_i-\mathbf n_c)^{\!\top}\Big)\mathbf u,\quad i\to\begin{cases}c&(\mathbf x_i-\mathbf n_c)\cdot\mathbf u_c\ge0\\ d&\text{else}\end{cases}
        w_i            the share of point i that is yolk-free cytoplasm, 1 - a y_i
        a              `yolk_weight`, how much yolk excludes cytoplasm
        y_i            the point's yolk (block `yolk`)
        n_c            the nucleus: the volume-weighted yolk-free centre of cell c
        v_i            the rest volume of point i
        x_i            the position of point i
        u_c            the spindle: the longest axis of the yolk-free cytoplasm about the nucleus (the top eigenvector)
        c, c'          the mother and the daughter: a point goes to the daughter when it lies behind the plane through n_c normal to u_c

    Reference: as cleave; Plexus exp29 (2026-10-21)."""
    PARAM_UNITS = {}
    SYMBOLS = {'w_i': 'the share of point i that is yolk-free cytoplasm, 1 - a y_i', 'a': '`yolk_weight`, how much yolk excludes cytoplasm', 'y_i': "the point's yolk (block `yolk`)", 'n_c': 'the nucleus: the volume-weighted yolk-free centre of cell c', 'v_i': 'the rest volume of point i', 'x_i': 'the position of point i', 'u_c': 'the spindle: the longest axis of the yolk-free cytoplasm about the nucleus (the top eigenvector)', "c, c'": 'the mother and the daughter: a point goes to the daughter when it lies behind the plane through n_c normal to u_c'}
    PARAM_ROLES = {'inherit': 'cell_blocks_the_daughter_copies_from_the_mother'}

    def __init__(self, params, device="cpu"):
        super().__init__({k: v for k, v in params.items() if k != "inherit"}, device)
        self.inherit = [str(b) for b in (params.get("inherit") or [])]

    def forward(self, H, mask=None):
        log = getattr(H, "cleave_log", None)
        n0 = len(log) if log is not None else 0
        out = super().forward(H, mask)
        log = getattr(H, "cleave_log", None)
        if log is not None and len(log) > n0:
            C = H.level(self.at)
            nt, st = C.node_type.clone(), C.state.clone()          # FUNCTIONALLY (clone, write, rebind): a tape through the frame holds the old tensors (exp30)
            for row in log[n0:]:
                nt[int(row[2])] = nt[int(row[1])]
                for b_ in self.inherit:
                    a_, e_ = C.state_schema[b_]
                    st[int(row[2]), a_:e_] = st[int(row[1]), a_:e_]
            C.node_type, C.state = nt, st
        return out


def _relax_points(P, idx):
    """Born relaxed: the shape the points `idx` have is their rest shape (F = I, C = 0, Jp = 1), as the fresh material of cell_divide[mpm]. A body cut with the strain of its frames of surface tension kept releases it at the new free
    faces at once: exp38 step 23, the two faces of a cut cell 390 nm apart within the frame, whatever binds them (div2 and div4 against div6, 2026-11-07)."""
    D = P.state_schema["pos"][1] - P.state_schema["pos"][0]
    if getattr(P, "F", None) is not None:
        P.F[idx] = torch.eye(D, device=P.F.device, dtype=P.F.dtype)
    if getattr(P, "C", None) is not None:
        P.C[idx] = 0.0
    if getattr(P, "Jp", None) is not None:
        P.Jp[idx] = 1.0


@register_operator("cell_divide", model="cleave_faces", set="particle", kind="divide", family="population",
                   title="cleave a cell and lay a dense skin on its two new faces",
                   equation=r"""$$\mathbf x_{s,l,(a,b)}=\mathbf n_c+s\,\big(\tfrac g2+l\,p\big)\,\mathbf u_c+\mathbf o_s+p\,\big(a+\tfrac12-\tfrac{m}{2}\big)\,\mathbf e_1+p\,\big(b+\tfrac12-\tfrac{m}{2}\big)\,\mathbf e_2$$""")
class CellDivideCleaveFaces(CellDivideCleaveTyped):
    r"""a cell of material points that divides once, at `after_frame`, and whose two NEW faces get the dense skin of the two-cell rig: the cloud is cut by the plane through its centroid normal to its long axis (cell_divide[cleave_typed]), the cloud points within one pitch of the skin are put to sleep, and `layers` square lattices of pitch `pitch` are woken on each new face, the two front layers `face_gap` apart.
    particle cell -> two particle cells, the daughter in a free slot: as cleave_typed, then per daughter the points near the cut sleep, the points that stay are reset to rest (F = I, C = 0, Jp = 1) and `layers` x m x m dormant points of the point set wake as its skin (volume pitch^3, mass density x volume, at rest), a square lattice of `side` centred on the middle of the points nearest the cut.

        \mathbf x_{s,l,(a,b)}=\mathbf n_c+s\,\big(\tfrac g2+l\,p\big)\,\mathbf u_c+\mathbf o_s+p\,\big(a+\tfrac12-\tfrac{m}{2}\big)\,\mathbf e_1+p\,\big(b+\tfrac12-\tfrac{m}{2}\big)\,\mathbf e_2
        x_{s,l,(a,b)}  the skin point (a, b) of layer l on the face of side s
        n_c            the cut: the volume-weighted centroid of the mother's points
        s              +1 for the mother, which keeps the points ahead of the plane along u_c, -1 for the daughter
        g              `face_gap`, the distance between the two daughters' front layers
        l, p           the layer, 0 ... `layers` - 1, and `pitch`, the lattice pitch of the skin
        u_c            the spindle: the long axis of the mother's points
        o_s            the centre of the face in its plane: the middle of the range of the daughter's points nearest the cut
        m              the lattice count across the face, `side` over `pitch`
        e_1, e_2       the unit vectors in the plane of the cut

    Reference: Plexus exp38 (this work, 2026-11)."""
    PARAM_UNITS = {'layers': 'count', 'pitch': 'length', 'side': 'length', 'face_gap': 'length', 'after_frame': 'frames', 'skin_block': 'dimensionless'}
    SYMBOLS = {'x_{s,l,(a,b)}': 'the skin point (a, b) of layer l on the face of side s', 'n_c': "the cut: the volume-weighted centroid of the mother's points", 's': '+1 for the mother, -1 for the daughter', 'g': "`face_gap`, the distance between the two daughters' front layers", 'l, p': 'the layer and `pitch`, the lattice pitch of the skin', 'u_c': "the spindle: the long axis of the mother's points", 'o_s': "the centre of the face's rectangle in its plane", 'm': 'the lattice count across the face, `side` over `pitch`', 'e_1, e_2': 'the unit vectors in the plane of the cut'}
    PARAM_ROLES = dict(CellDivideCleaveTyped.PARAM_ROLES, layers="number_of_dense_layers", pitch="skin_lattice_pitch_world", side="edge_of_the_square_face_world", face_gap="distance_between_the_two_front_layers_at_birth", after_frame="no_division_before_this_frame", skin_block="point_block_set_to_1_on_the_skin_points_0_on_the_points_put_to_sleep")

    def __init__(self, params, device="cpu"):
        own = ("layers", "pitch", "side", "face_gap", "after_frame", "skin_block")
        no_skin = int(params.get("layers", 3)) == 0
        super().__init__(dict({k: v for k, v in params.items() if k not in own}, gap=float(params.get("gap", 0.0)) if no_skin else 0.0), device)          # with a skin the halves are not drawn apart: its front layers sit `face_gap` apart on the cut itself; without one the furrow `gap` (default 0) draws them apart as cleave does
        self.layers = int(params.get("layers", 3)); self.pitch = float(params["pitch"]); self.side = float(params["side"]); self.face_gap = float(params.get("face_gap", 0.004)); self.after_frame = int(params.get("after_frame", 0)); self.skin_block = params.get("skin_block")

    def forward(self, H, mask=None):
        if int(getattr(H, "frame", 0) or 0) < self.after_frame:
            return {}
        log = getattr(H, "cleave_log", None); n0 = len(log) if log is not None else 0
        out = super().forward(H, mask)
        log = getattr(H, "cleave_log", None)
        for row in (log[n0:] if log is not None else []):
            self._skin(H, int(row[1]), int(row[2]), np.asarray(row[3], np.float64), np.asarray(row[4], np.float64))
        return out

    def _skin(self, H, c, d, ax, nuc):
        P = H.level(self.points); p0, p1 = P.state_schema["pos"]
        X = P.state[:, p0:p1].detach().double().cpu().numpy(); par = P.parent.detach().cpu().numpy(); occ = (P.occ > 0.5).detach().cpu().numpy()
        if self.layers == 0:                                               # a cell too coarse for a skin (125-1000 points): the cut and the relaxed birth only
            _relax_points(P, torch.as_tensor(np.flatnonzero(occ & ((par == c) | (par == d))), device=P.state.device))
            print(f"[cell_divide:cleave_faces] cell {c} -> {c}, {d}: no skin (layers 0), the points of both daughters born relaxed", flush=True)
            return
        ax = ax / np.linalg.norm(ax); e1 = np.array([0.0, 1.0, 0.0]) - ax[1] * ax
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.array([0.0, 0.0, 1.0]) - ax[2] * ax
        e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
        m = int(round(self.side / self.pitch)); xi = (X - nuc) @ ax; g2 = 0.5 * self.face_gap; reach = g2 + self.layers * self.pitch           # the cloud keeps one pitch beyond the back layer
        mine = occ & ((par == c) | (par == d))
        asleep = mine & (np.abs(xi) < reach); occ = occ & ~asleep
        faces = []
        for slot, s in ((c, 1.0), (d, -1.0)):
            depth = s * xi
            ext = depth[mine & (par == slot)].max() - reach
            near = np.flatnonzero(occ & (par == slot) & (depth >= reach) & (depth < reach + 0.15 * ext))       # the daughter's points in the first 15% of its length from the cut: their range gives the middle of the face
            if near.size < 20:
                raise ValueError(f"cell_divide[cleave_faces]: {near.size} points of cell {slot} near its cut, too few for a face")
            u1, u2 = (X[near] - nuc) @ e1, (X[near] - nuc) @ e2
            a1, a2 = 0.5 * (u1.max() + u1.min()) + (np.arange(m) + 0.5 - 0.5 * m) * self.pitch, 0.5 * (u2.max() + u2.min()) + (np.arange(m) + 0.5 - 0.5 * m) * self.pitch      # the lattice is centred on the middle of the points' range
            pos = np.zeros((self.layers, m, m, 3))
            for l in range(self.layers):
                pos[l] = nuc + s * (g2 + l * self.pitch) * ax + a1[:, None, None] * e1 + a2[None, :, None] * e2
            faces.append((slot, pos.reshape(-1, 3)))
        pool = np.flatnonzero(~occ); need = sum(len(p) for _, p in faces)
        if pool.size < need:
            raise ValueError(f"cell_divide[cleave_faces]: {pool.size} dormant points, the two skins need {need} -- raise per_parent of the point set")
        oc = P.occ.clone(); oc[torch.as_tensor(np.flatnonzero(asleep), device=oc.device)] = 0.0; P.occ = oc
        _relax_points(P, torch.as_tensor(np.flatnonzero(occ & ((par == c) | (par == d))), device=P.state.device))          # the daughters are born relaxed
        if self.skin_block:                                                # a point block that tells the skin from the cloud in the record
            _set_block(P, self.skin_block, torch.as_tensor(np.flatnonzero(asleep), device=P.state.device), 0.0)
        k = 0
        for slot, pos in faces:
            idx = torch.as_tensor(pool[k:k + len(pos)], device=P.state.device); k += len(pos)
            _mpm_reset_points(P, idx, torch.as_tensor(pos, device=P.state.device, dtype=P.state.dtype), torch.full((len(pos),), self.pitch ** 3, device=P.state.device, dtype=P.state.dtype))
            P.parent[idx] = slot                                           # IN PLACE, as cleave: lift_index caches this very tensor
            _mpm_label_points(P, idx, slot, self.pid, self.label)
            if self.skin_block:
                _set_block(P, self.skin_block, idx, 1.0)
            oc = P.occ.clone(); oc[idx] = 1.0; P.occ = oc
        print(f"[cell_divide:cleave_faces] cell {c} -> {c}, {d}: {int(asleep.sum()):,} cloud points asleep at the cut, a skin of {self.layers} layers ({[len(p) // self.layers for _, p in faces]} points per layer) on each new face, front layers {self.face_gap:g} apart", flush=True)


@register_operator("cell_divide", model="cleave_tangent", set="particle", kind="divide", family="population",
                   title="cleave cells across the layer, the spindle along the surface",
                   equation=r"""$$\mathbf n_c=\frac{\bar{\mathbf x}_c-\mathbf o}{\lvert\bar{\mathbf x}_c-\mathbf o\rvert},\quad \mathbf P=\mathbf I-\mathbf n_c\mathbf n_c^{\!\top},\quad \mathbf u_c=\arg\max_{\lvert\mathbf u\rvert=1}\mathbf u^{\!\top}\mathbf P\Big(\tfrac{\sum_i v_i\mathbf y_i\mathbf y_i^{\!\top}}{\sum_i v_i}\Big)\mathbf P\,\mathbf u$$""")
class CellDivideCleaveTangent(CellDivideCleaveTyped):
    r"""cell_divide[cleave_typed] with the spindle in the layer: the cell's long axis is taken in the tangent plane of the sphere, so the cut is perpendicular to the epithelium and daughters stay in the layer instead of piling up.
    particle cell -> two particle cells: as cleave_typed; `split: centroid` cuts through the centroid, `median` at the median along the spindle (equal halves).

        \mathbf n_c=\frac{\bar{\mathbf x}_c-\mathbf o}{\lvert\bar{\mathbf x}_c-\mathbf o\rvert},\quad \mathbf P=\mathbf I-\mathbf n_c\mathbf n_c^{\!\top},\quad \mathbf u_c=\arg\max_{\lvert\mathbf u\rvert=1}\mathbf u^{\!\top}\mathbf P\Big(\tfrac{\sum_i v_i\mathbf y_i\mathbf y_i^{\!\top}}{\sum_i v_i}\Big)\mathbf P\,\mathbf u
        n_c            the outward normal, from the sphere's centre o (`centre`) through the cell's centroid
        P              the projector onto the tangent plane
        u_c            the spindle: the longest axis of the projected volume-weighted covariance of the cell's points
        v_i            the point volumes
        y_i            the point's yolk (0 here: a nucleus at the centroid)
        x_i            the point positions

    Reference: Plexus exp27, the cell adhesion testbench (this work, 2026-10) (2026-10-19; Cedric: the division should be perpendicular to the epithelium)."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"A cell that cell_cycle has marked ready is cut in two. The cut is perpendicular to the layer: the spindle $\mathbf "
        r"u_c$ is the longest axis of the volume-weighted covariance of the cell's points projected with $\mathbf P=\mathbf "
        r"I-\mathbf n_c\mathbf n_c^{\!\top}$, where $\mathbf n_c=(\bar{\mathbf x}_c-\mathbf o)/\lvert\bar{\mathbf x}_c-\mathbf "
        r"o\rvert$ is the outward normal from the sphere's centre $\mathbf o$ through the cell's centroid $\bar{\mathbf x}_c$. "
        r"`split: median` cuts at the median along $\mathbf u_c$ (equal halves), `centroid` through the centroid. The two "
        r"daughters are marked `born`, and they stay in the layer instead of piling up."
    )
    PARAM_UNITS = {'centre': 'length'}
    SYMBOLS = {'n_c': "the outward normal, from the sphere's centre o (`centre`) through the cell's centroid", 'P': 'the projector onto the tangent plane', 'u_c': "the spindle: the longest axis of the projected volume-weighted covariance of the cell's points", 'v_i': 'the point volumes', 'y_i': "the point's yolk (0 here: a nucleus at the centroid)", 'x_i': 'the point positions'}
    PARAM_ROLES = dict(CellDivideCleave.PARAM_ROLES, centre="sphere_centre_world", split="cut_at_centroid_or_median")
    REQUIRES_PARAMS = ["points", "centre"]

    def __init__(self, params, device="cpu"):
        super().__init__({k: v for k, v in params.items() if k not in ("centre", "split", "centre_block")}, device)
        self.centre = torch.as_tensor([float(v) for v in params["centre"]], dtype=torch.float64)
        # `centre_block` (exp29 rig, 2026-10-23): a cell block (width 3) holding EACH CELL's own vesicle centre, read in place of `centre`,
        # so several vesicles in one box each divide in their own tangent plane
        self.centre_block = params.get("centre_block")
        self.split = str(params.get("split", "centroid"))
        if self.split not in ("centroid", "median"):
            raise ValueError("cell_divide[cleave_tangent]: split is centroid or median")

    def forward(self, H, mask=None):
        C, P = H.level(self.at), H.level(self.points)
        live = (C.occ > 0.5).detach().cpu().numpy()
        ok = live.copy()
        if self.mark:
            if self.mark not in C.state_schema:
                raise ValueError(f"cell_divide[cleave_tangent]: mark {self.mark!r} is not a block of {self.at!r}")
            ok &= C.state[:, C.state_schema[self.mark][0]].detach().cpu().numpy() > 0.5
        cand = np.flatnonzero(ok)
        if cand.size == 0:
            return {}
        u = torch.rand(cand.size, generator=self._gen, dtype=torch.float64).numpy()
        go = cand[u < self.p_div][: self.max_per_call]
        free = np.flatnonzero(~live)
        if go.size > free.size:
            H.mpm_div_blocked = int(getattr(H, "mpm_div_blocked", 0)) + int(go.size - free.size)
            go = go[: free.size]
        if go.size == 0:
            return {}
        p0, p1 = P.state_schema["pos"]
        par = P.parent
        pocc = P.occ > 0.5
        X_all = P.state[:, p0:p1].detach().double()
        v_all = (P.p_vol.detach().double() if getattr(P, "p_vol", None) is not None else torch.ones(X_all.shape[0], dtype=torch.float64, device=X_all.device))
        o = self.centre.to(X_all.device)
        st = P.state.clone()
        par_new = par.clone()
        cst, coc = C.state.clone(), C.occ.clone()
        ntype = C.node_type.clone()                              # the daughters' types, written FUNCTIONALLY with the rest (a tape holds the old tensor; exp30)
        ca, cb = C.state_schema["pos"]
        log = getattr(H, "cleave_log", None)
        if log is None:
            log = H.cleave_log = []
        tick = int(getattr(H, "tick", -1))
        n_done = 0
        for j, c in enumerate(go.tolist()):
            d = int(free[j])
            idx = torch.nonzero((par == c) & pocc).flatten()
            if idx.numel() < 8:
                continue
            X, v = X_all[idx], v_all[idx]
            nuc = (v[:, None] * X).sum(0) / v.sum().clamp_min(1e-30)
            Y = X - nuc
            M = (v[:, None, None] * Y[:, :, None] * Y[:, None, :]).sum(0) / v.sum().clamp_min(1e-30)
            if self.centre_block:
                b0, b1 = C.state_schema[self.centre_block]
                o_c = C.state[c, b0:b1].detach().double().to(X_all.device)
            else:
                o_c = o
            nrm = nuc - o_c
            nrm = nrm / nrm.norm().clamp_min(1e-12)
            Pm = torch.eye(3, dtype=M.dtype, device=M.device) - nrm[:, None] * nrm[None, :]
            _, V = torch.linalg.eigh(Pm @ M @ Pm)
            ax = V[:, -1]
            xi = Y @ ax
            if self.split == "median":                             # the lower half of the points along the spindle goes to the daughter
                to_d = torch.zeros_like(xi, dtype=torch.bool)
                to_d[torch.argsort(xi)[: xi.numel() // 2]] = True
            else:
                to_d = xi < 0.0
            if bool(to_d.all()) or not bool(to_d.any()):
                continue
            side = torch.where(to_d, -1.0, 1.0).to(X.dtype)
            st[idx, p0:p1] = (X + (0.5 * self.gap) * side[:, None] * ax[None, :]).to(st.dtype)
            moved = idx[to_d]
            par_new[moved] = d
            if self.pid in P.state_schema:
                st[moved, P.state_schema[self.pid][0]] = float(d)
            if self.label in P.state_schema:
                st[moved, P.state_schema[self.label][0]] = float(d % 20)
            for k_, (a_, b_) in C.state_schema.items():
                if b_ - a_ == 1:
                    cst[d, a_:b_] = cst[c, a_:b_]
            for k_ in self.inherit:                                # blocks wider than 1 the daughter takes from its mother (`inherit`)
                a_, b_ = C.state_schema[k_]
                cst[d, a_:b_] = cst[c, a_:b_]
            keep = idx[~to_d]
            cst[c, ca:cb] = st[keep, p0:p1].mean(0).to(cst.dtype)
            cst[d, ca:cb] = st[moved, p0:p1].mean(0).to(cst.dtype)
            if self.mark and self.clear_mark:
                cst[[c, d], C.state_schema[self.mark][0]] = 0.0
            if self.daughter_mark:
                cst[[c, d], C.state_schema[self.daughter_mark][0]] = 1.0
            coc[d] = 1.0
            ntype[d] = ntype[c]                                    # the daughter is of its mother's type
            log.append((tick, int(c), d, ax.cpu().numpy().tolist(), nuc.cpu().numpy().tolist(),
                        float(v[~to_d].sum()), float(v[to_d].sum())))
            n_done += 1
        if n_done:
            P.state = st
            P.parent = par_new                   # a NEW tensor, not copy_: lift_index's cache keys on its identity and rebuilds; a tape survives
            C.state, C.occ, C.node_type = cst, coc, ntype
            H.mpm_divisions = int(getattr(H, "mpm_divisions", 0)) + n_done
        return {}


@register_operator("cell_divide", model="cleave_tangent_relaxed", set="particle", kind="divide", family="population",
                   title="cleave cells across the layer, the daughters born relaxed",
                   equation=r"""$$F_p\leftarrow\mathbf I,\qquad C_p\leftarrow 0,\qquad J_p\leftarrow 1,\qquad p\in\text{the points of the two daughters}$$""")
class CellDivideCleaveTangentRelaxed(CellDivideCleaveTangent):
    r"""cell_divide[cleave_tangent] whose daughters are born RELAXED: at the cut the points of the mother and of the daughter are reset to rest, the shape they have becoming their rest shape.
    particle cell -> two particle cells: as cleave_tangent, then per point of the two daughters the deformation gradient, the affine velocity and the plastic volume ratio are reset.

        F_p\leftarrow\mathbf I,\qquad C_p\leftarrow 0,\qquad J_p\leftarrow 1,\qquad p\in\text{the points of the two daughters}
        F_p            the deformation gradient of point p: the strain its material remembers
        C_p            the affine velocity of point p (the velocity gradient it carries between grid and point)
        J_p            the plastic volume ratio of point p
        \mathbf I      the identity

    Reference: Plexus exp38 (this work, 2026-11)."""
    PARAM_UNITS = dict(CellDivideCleaveTangent.PARAM_UNITS)
    SYMBOLS = {'F_p': 'the deformation gradient of point p: the strain its material remembers', 'C_p': 'the affine velocity of point p', 'J_p': 'the plastic volume ratio of point p', '\\mathbf I': 'the identity'}
    PARAM_ROLES = dict(CellDivideCleaveTangent.PARAM_ROLES)

    def forward(self, H, mask=None):
        log = getattr(H, "cleave_log", None); n0 = len(log) if log is not None else 0
        out = super().forward(H, mask)
        log = getattr(H, "cleave_log", None)
        if log is not None and len(log) > n0:
            P = H.level(self.points); live = P.occ > 0.5
            for row in log[n0:]:
                _relax_points(P, torch.nonzero(live & ((P.parent == int(row[1])) | (P.parent == int(row[2])))).flatten())
        return out


@register_operator("cell_divide", model="cleave_tangent_skin", set="particle", kind="divide", family="population",
                   title="cleave cells across the layer and lay a dense skin on the two new faces",
                   equation=r"""$$\mathbf x_{s,l,(a,b)}=\mathbf n_c+\big(\xi_{\rm cut}+s\,(\tfrac g2+l\,p)\big)\,\mathbf u_c+a\,p\,\mathbf e_1+b\,p\,\mathbf e_2,\qquad (a,b)\ \text{kept if}\ \min_i\lVert(a p,b p)-\mathbf y_i\rVert\le r$$""")
class CellDivideCleaveTangentSkin(CellDivideCleaveTangentRelaxed):
    r"""cell_divide[cleave_tangent_relaxed] that also wakes a skin of `layers` square-lattice layers of dormant points on each of the two new faces, as cell_divide[cleave_faces] does for the cube of the two-cell rig (exp38 steps 23, 27), on a vesicle cell of a few hundred points:
    the halves are not drawn apart (no furrow `gap`): the front layers of the two skins sit `face_gap` apart on the cut itself, and the lattice of each face covers the cross-section of its daughter at the cut.
    particle cell -> two particle cells: as cleave_tangent_relaxed, then per daughter the lattice nodes (spacing `pitch`) within `trim` of the in-plane position of the `face_points` of its points nearest the cut wake as skin points (volume pitch^3, at rest), taken from the dormant slots of the two blocks.

        \mathbf x_{s,l,(a,b)}=\mathbf n_c+\big(\xi_{\rm cut}+s\,(\tfrac g2+l\,p)\big)\,\mathbf u_c+a\,p\,\mathbf e_1+b\,p\,\mathbf e_2,\qquad (a,b)\ \text{kept if}\ \min_i\lVert(a p,b p)-\mathbf y_i\rVert\le r
        x_{s,l,(a,b)}  the skin point (a, b) of layer l on the face of side s
        n_c            the cut's reference point: the volume-weighted centroid of the mother's points
        \xi_{cut}      the position of the cut along the spindle: halfway between the daughter's last point and the mother's first (the cut is at the median, not at the centroid)
        s              +1 for the mother's face, -1 for the daughter's
        g              `face_gap`, the distance between the two front layers
        l, p           the layer, 0 ... `layers` - 1, and `pitch`, the lattice pitch of the skin
        u_c            the spindle, the longest axis of the cell's points in the tangent plane
        e_1, e_2       the unit vectors in the plane of the cut
        a, b           the lattice indices, whole numbers
        y_i            the in-plane position of one of the `face_points` nearest points of the daughter
        r              `trim`, how far a node may lie from those points and still belong to the face

    Reference: Plexus exp38 (this work, 2026-11); cell_divide[cleave_faces]."""
    PARAM_UNITS = dict(CellDivideCleaveTangent.PARAM_UNITS, layers="count", pitch="length", face_gap="length", trim="length", face_points="count", skin_block="dimensionless", seed_skin="flag", seed_reach="length", seed_min_pairs="count")
    SYMBOLS = {'x_{s,l,(a,b)}': 'the skin point (a, b) of layer l on the face of side s', 'n_c': "the cut's reference point: the volume-weighted centroid of the mother's points", '\\xi_{cut}': "the position of the cut along the spindle", 's': '+1 for the mother face, -1 for the daughter face', 'g': '`face_gap`, the distance between the two front layers', 'l, p': 'the layer and `pitch`, the lattice pitch of the skin', 'u_c': 'the spindle', 'e_1, e_2': 'the unit vectors in the plane of the cut', 'a, b': 'the lattice indices', 'y_i': 'the in-plane position of one of the `face_points` nearest points of the daughter', 'r': '`trim`, how far a node may lie from those points'}
    PARAM_ROLES = dict(CellDivideCleaveTangent.PARAM_ROLES, layers="number_of_dense_layers", pitch="skin_lattice_pitch_world", face_gap="distance_between_the_two_front_layers_at_birth", trim="largest_distance_of_a_node_from_the_cross_section_points",
                       face_points="points_nearest_the_cut_whose_in_plane_positions_give_the_face", skin_block="point_block_set_to_1_on_the_skin_points_0_on_the_points_of_the_cloud",
                       seed_skin="whether_to_wake_a_skin_on_every_face_where_two_cells_already_touch_at_the_first_tick", seed_reach="largest_distance_between_two_points_of_different_cells_that_count_as_touching", seed_min_pairs="number_of_touching_point_pairs_that_make_two_cells_neighbours")

    def __init__(self, params, device="cpu"):
        own = ("layers", "pitch", "face_gap", "trim", "face_points", "skin_block", "gap", "seed_skin", "seed_reach", "seed_min_pairs")
        super().__init__({k: v for k, v in params.items() if k not in own}, device)        # gap 0: the skins' front layers sit on the cut, the halves are not drawn apart
        self.layers = int(params.get("layers", 1)); self.pitch = float(params["pitch"]); self.face_gap = float(params.get("face_gap", 0.004)); self.trim = float(params.get("trim", 0.15)); self.face_points = int(params.get("face_points", 16)); self.skin_block = params.get("skin_block"); self.seed_skin = bool(params.get("seed_skin", False)); self.seed_reach = float(params.get("seed_reach", 0.3)); self.seed_min_pairs = int(params.get("seed_min_pairs", 3))

    def forward(self, H, mask=None):
        if self.seed_skin and not getattr(H, "_contact_skin_done", False):                # once, on the first tick: a skin on every face where two cells already touch
            H._contact_skin_done = True; self._seed_contact_skins(H)
        log = getattr(H, "cleave_log", None); n0 = len(log) if log is not None else 0
        out = super().forward(H, mask)
        log = getattr(H, "cleave_log", None)
        for row in (log[n0:] if log is not None else []):
            self._skin(H, int(row[1]), int(row[2]), np.asarray(row[3], np.float64), np.asarray(row[4], np.float64))
        return out

    def _seed_contact_skins(self, H):
        """The skin of `_skin` on both sides of every pair of cells that already touch (two cells touch when `seed_min_pairs` point pairs, one point from each, lie within `seed_reach`): for the pair (a, b) the spindle is the centroid of a minus the centroid of b, the cut halfway between b's last and a's first point along it.
        The cut and the face come from the cloud points alone (not the skin points of earlier faces), so the faces of one cell do not move each other."""
        from scipy.spatial import cKDTree
        P = H.level(self.points); p0, p1 = P.state_schema["pos"]
        X = P.state[:, p0:p1].detach().double().cpu().numpy(); par = P.parent.detach().cpu().numpy(); live = np.flatnonzero((P.occ > 0.5).detach().cpu().numpy())
        pr = cKDTree(X[live]).query_pairs(self.seed_reach, output_type="ndarray"); cp = par[live][pr]; cp = np.sort(cp[cp[:, 0] != cp[:, 1]], axis=1)
        pairs, cnt = np.unique(cp, axis=0, return_counts=True); pairs = pairs[cnt >= self.seed_min_pairs]
        cells = np.unique(par[live]); cen = {int(k): X[live][par[live] == k].mean(axis=0) for k in cells}
        for a, b in pairs:
            ax = cen[int(a)] - cen[int(b)]
            self._skin(H, int(a), int(b), ax, 0.5 * (cen[int(a)] + cen[int(b)]), cloud_only=True)
        print(f"[cell_divide:cleave_tangent_skin] contact skins on {len(pairs)} cell pairs of {len(cells)} cells", flush=True)

    def _skin(self, H, c, d, ax, nuc, cloud_only=False):
        from scipy.spatial import cKDTree
        P = H.level(self.points); p0, p1 = P.state_schema["pos"]
        X = P.state[:, p0:p1].detach().double().cpu().numpy(); par = P.parent.detach().cpu().numpy(); occ = (P.occ > 0.5).detach().cpu().numpy()
        cl = occ & ~(P.state[:, P.state_schema[self.skin_block][0]].detach().cpu().numpy() > 0.5) if (cloud_only and self.skin_block in P.state_schema) else occ
        ax = ax / np.linalg.norm(ax); e1 = np.array([0.0, 1.0, 0.0]) - ax[1] * ax
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.array([0.0, 0.0, 1.0]) - ax[2] * ax
        e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
        xi = (X - nuc) @ ax; live_c, live_d = cl & (par == c), cl & (par == d)
        cut = 0.5 * (xi[live_d].max() + xi[live_c].min()); g2 = 0.5 * self.face_gap; reach = g2 + self.layers * self.pitch
        asleep = (live_c | live_d) & (np.abs(xi - cut) < reach); occ = occ & ~asleep
        faces = []
        for slot, s in ((c, 1.0), (d, -1.0)):
            mine = np.flatnonzero(cl & ~asleep & (par == slot)); depth = s * (xi[mine] - cut)
            near = mine[np.argsort(depth)[:max(self.face_points, 3)]]
            u = np.column_stack([(X[near] - nuc) @ e1, (X[near] - nuc) @ e2])
            ia = np.arange(int(np.floor((u[:, 0].min() - self.trim) / self.pitch)), int(np.ceil((u[:, 0].max() + self.trim) / self.pitch)) + 1)
            ib = np.arange(int(np.floor((u[:, 1].min() - self.trim) / self.pitch)), int(np.ceil((u[:, 1].max() + self.trim) / self.pitch)) + 1)
            A, B = np.meshgrid(ia, ib, indexing="ij"); node = np.column_stack([A.ravel() * self.pitch, B.ravel() * self.pitch])
            node = node[cKDTree(u).query(node)[0] <= self.trim]
            pos = np.concatenate([nuc + (cut + s * (g2 + l * self.pitch)) * ax + node[:, :1] * e1 + node[:, 1:] * e2 for l in range(self.layers)])
            faces.append((slot, pos))
        pool = np.flatnonzero(~occ & ((par == c) | (par == d))); need = sum(len(p) for _, p in faces)
        if pool.size < need:
            raise ValueError(f"cell_divide[cleave_tangent_skin]: {pool.size} dormant points in the blocks of cells {c} and {d}, the two skins need {need} -- raise grow_reserve of the point set")
        oc = P.occ.clone(); oc[torch.as_tensor(np.flatnonzero(asleep), device=oc.device)] = 0.0; P.occ = oc
        if self.skin_block:
            _set_block(P, self.skin_block, torch.as_tensor(np.flatnonzero(asleep), device=P.state.device), 0.0)
        k = 0
        for slot, pos in faces:
            idx = torch.as_tensor(pool[k:k + len(pos)], device=P.state.device); k += len(pos)
            _mpm_reset_points(P, idx, torch.as_tensor(pos, device=P.state.device, dtype=P.state.dtype), torch.full((len(pos),), self.pitch ** 3, device=P.state.device, dtype=P.state.dtype))
            P.parent[idx] = slot                                           # IN PLACE, as cleave: lift_index caches this very tensor
            _mpm_label_points(P, idx, slot, self.pid, self.label)
            if self.skin_block:
                _set_block(P, self.skin_block, idx, 1.0)
            oc = P.occ.clone(); oc[idx] = 1.0; P.occ = oc
        print(f"[cell_divide:cleave_tangent_skin] cell {c} -> {c}, {d}: a skin of {self.layers} layers ({[len(p) // self.layers for _, p in faces]} points per layer) on each new face, front layers {self.face_gap:g} apart", flush=True)


@register_operator("pair_potential", model="cell_contact", family="interaction", set="particle", kind="exchange",
                   title="keep cells apart and set the tension of each contact, on the lattice nodes each cell touches", equation=r"""$$E=E_{\rm overlap}-w\sum_{A<B}\gamma_{\tau_A\tau_B}\int\nabla\phi_A\cdot\nabla\phi_B\,dV\;\approx\;E_{\rm overlap}+\sum_{A<B}\gamma_{\tau_A\tau_B}\,S_{AB},\qquad E_{\rm overlap}=\Delta x^3\sum_n\tfrac{k}{2}\,(P_n-s_0)_+^2,\quad P_n=\sum_{A<B}\phi_A(n)\,\phi_B(n),\qquad \mathbf F_i=-\frac{\partial E}{\partial\mathbf x_i}$$""")
class CellContactVolumeInterfaceSparse(CellContactVolumeInterface):
    r"""pair_potential[cell_contact_dense] with each cell's volume fraction stored only on the lattice nodes its points deposit on, not on every node of the world: the dense [cells, nodes] array was 10^9 values for 634 cells.
    particle -> particle: as cell_contact_dense; with `contact_slack_block` (a width-1 cell block) the tolerated overlap is read per cell, the volume-fraction-weighted mean of the cells at each lattice node, instead of the one `contact_slack`.

        E=E_{\rm overlap}-w\sum_{A<B}\gamma_{\tau_A\tau_B}\int\nabla\phi_A\cdot\nabla\phi_B\,dV\;\approx\;E_{\rm overlap}+\sum_{A<B}\gamma_{\tau_A\tau_B}\,S_{AB},\qquad E_{\rm overlap}=\Delta x^3\sum_n\tfrac{k}{2}\,(P_n-s_0)_+^2,\quad P_n=\sum_{A<B}\phi_A(n)\,\phi_B(n),\qquad \mathbf F_i=-\frac{\partial E}{\partial\mathbf x_i}
        E              the total contact energy
        E_{overlap}    the overlap penalty: zero until two cells overlap by more than s_0 at a lattice node, then quadratic in the excess
        P_n            the overlap at lattice node n, the sum of the pairwise products of the cells' volume fractions there
        k              `repulsion_strength`, the overlap penalty's stiffness
        s_0            `contact_slack`, the overlap tolerated at a node (per cell with `contact_slack_block`)
        w              `contact_edge_width`, the interface width factor
        \gamma_{\tau_A\tau_B} `contact_surface_tension`, the tension of a contact between types tau_A and tau_B (a symmetric matrix, in the units of cell_surface_tension; negative = adhesion)
        \phi_A, \phi_B the two cells' volume fractions (each cell's points smeared onto the lattice by quadratic B-spline weights)
        S_{AB}         the contact area of A and B
        \Delta x       the lattice spacing
        F_i, x_i       the force on material point i and its position; F_i / m_i is the acceleration handed to the MPM substep

    Reference: as cell_contact_dense; Plexus exp27, the cell adhesion testbench (this work, 2026-10) (2026-10-20)."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"Keeps cells from overlapping and gives their contact a surface tension. Each cell's points are smeared onto a "
        r"lattice as a volume fraction $\phi$ (1 inside the cell, 0 outside); the energy is $E=E_{\rm "
        r"overlap}-w\sum_{A<B}\gamma_{\tau_A\tau_B}\int\nabla\phi_A\cdot\nabla\phi_B\,dV\;\approx\;E_{\rm "
        r"overlap}+\sum_{A<B}\gamma_{\tau_A\tau_B}\,S_{AB}$. The first term is the overlap penalty "
        r"$E_{\rm overlap}=\Delta x^3\sum_n\tfrac{k}{2}\,(P_n-s_0)_+^2$, with $P_n=\sum_{A<B}\phi_A(n)\,\phi_B(n)$ the overlap of "
        r"the cells at lattice node $n$: zero until the overlap passes $s_0$ (`contact_slack`), then quadratic in the excess, "
        r"with stiffness $k$ (`repulsion_strength`); $\Delta x$ is the lattice spacing. The second gives every pair of cells "
        r"$A$ and $B$ the tension $\gamma$ between their types times the area $S_{AB}$ of their interface ($w$ is "
        r"`contact_edge_width`); a negative $\gamma$ is adhesion. The force on each point is minus the gradient of $E$ with "
        r"respect to its position, $\mathbf F_i=-\partial E/\partial\mathbf x_i$, taken through $\phi$ in one pass (no "
        r"minimisation); divided by the point's mass it is an acceleration the MPM substep adds to the point's velocity. "
        r"This sparse variant stores $\phi$ only on the lattice nodes a cell's points deposit on, not on every node of the "
        r"world (the dense array was $10^9$ values for 634 cells); with `contact_slack_block` the tolerated overlap is read per cell."
    )
    PARAM_UNITS = {}
    SYMBOLS = {'E': 'the total contact energy', 'E_{overlap}': 'the overlap penalty: zero until two cells overlap by more than s_0 at a lattice node, then quadratic in the excess', 'P_n': "the overlap at lattice node n, the sum of the pairwise products of the cells' volume fractions there", 'k': '`repulsion_strength`, the overlap penalty\'s stiffness', 's_0': '`contact_slack`, the overlap tolerated at a node (per cell with `contact_slack_block`)', 'w': '`contact_edge_width`, the interface width factor', '\\gamma_{\\tau_A\\tau_B}': '`contact_surface_tension`, the tension of a contact between types tau_A and tau_B (a symmetric matrix, in the units of cell_surface_tension; negative = adhesion)', '\\phi_A, \\phi_B': "the two cells' volume fractions", 'S_{AB}': 'the contact area of A and B', '\\Delta x': 'the lattice spacing', '\\mathbf F_i, \\mathbf x_i': 'the force on material point i and its position; F_i / m_i is the acceleration handed to the MPM substep'}

    def _lattice_laplacian(self, A, per):
        """The lattice Laplacian of the dense fields A [k, nx, ny, nz], times the lattice spacing squared: the seven-point stencil (the six face neighbours less six times the node)."""
        L = -6.0 * A
        for d in range(3):
            L = L + self._shift(A, d, 1, per and d == 0) + self._shift(A, d, -1, per and d == 0)
        return L

    def _own_laplacian(self, phi_u, uk, ck, ix, iy, iz, shape, per, Nw):
        """The same stencil of each cell's own fraction at its own keys (a neighbour node the cell does not touch holds 0): `phi_u` at the keys `uk` = cell * Nw + node, whose cell is `ck` and lattice indices `ix, iy, iz`."""
        Ny, Nz = shape[1], shape[2]
        Lown = -6.0 * phi_u
        for d, (a_, n_) in enumerate(((ix, shape[0]), (iy, Ny), (iz, Nz))):
            for sgn in (1, -1):
                nb = a_ + sgn
                nb = torch.remainder(nb, n_) if (d == 0 and per) else nb.clamp(0, n_ - 1)
                if d == 0:
                    fnb = (nb * Ny + iy) * Nz + iz
                elif d == 1:
                    fnb = (ix * Ny + nb) * Nz + iz
                else:
                    fnb = (ix * Ny + iy) * Nz + nb
                kq = ck * Nw + fnb
                pos = torch.searchsorted(uk, kq).clamp_(max=uk.numel() - 1)
                Lown = Lown + torch.where(uk[pos] == kq, phi_u[pos], torch.zeros_like(phi_u))
        return Lown

    def forward(self, H, mask=None):
        P = H.level(self.at)
        X = P.get("pos")
        dev, dt_ = X.device, X.dtype
        live = (P.occ > 0).to(dt_)
        cid = H.lift_index(self.at, self.cells).long()
        C = H.level(self.cells)
        nC = int(C.state.shape[0])
        dx = float(self.dx) if self.dx is not None else float(H.field(self.grid).dx)
        world = H.world_size
        shape = [int(math.ceil(float(w) / dx)) + 3 for w in world]
        per = self.period > 0.0
        if per:
            nx = int(round(self.period / dx))
            if abs(self.period / dx - nx) > 1e-6:
                raise ValueError(f"pair_potential[cell_contact]: period {self.period} is not a whole number of lattice spacings dx {dx}")
            shape[0] = nx
        Ny, Nz = shape[1], shape[2]
        Nw = shape[0] * Ny * Nz
        if CellContactVolume._OFF is None or CellContactVolume._OFF.device != dev:
            o = torch.arange(3, device=dev)
            CellContactVolume._OFF = torch.stack(torch.meshgrid(o, o, o, indexing="ij"), -1).reshape(-1, 3)
        off = CellContactVolume._OFF
        xg = X / dx
        base = torch.floor(xg - 0.5)
        fx = xg - base
        w = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], 1)
        dw = torch.stack([fx - 1.5, -2.0 * (fx - 1.0), fx - 0.5], 1) / dx
        idx = base.long()[:, None, :] + off[None]
        for d in range(3):
            if d == 0 and per:
                idx[..., 0] = torch.remainder(idx[..., 0], shape[0])
            else:
                idx[..., d] = idx[..., d].clamp(0, shape[d] - 1)
        flat = (idx[..., 0] * Ny + idx[..., 1]) * Nz + idx[..., 2]
        wx, wy, wz = w[:, :, 0], w[:, :, 1], w[:, :, 2]
        W = (wx[:, off[:, 0]] * wy[:, off[:, 1]] * wz[:, off[:, 2]])
        G = torch.stack([dw[:, off[:, 0], 0] * wy[:, off[:, 1]] * wz[:, off[:, 2]],
                         wx[:, off[:, 0]] * dw[:, off[:, 1], 1] * wz[:, off[:, 2]],
                         wx[:, off[:, 0]] * wy[:, off[:, 1]] * dw[:, off[:, 2], 2]], -1)
        V = (P.p_vol.to(dt_) if getattr(P, "p_vol", None) is not None else torch.ones_like(live)) * live
        dep = (W * (V / dx ** 3)[:, None]).reshape(-1)
        # the keys: (cell, node) pairs, sorted; phi_u the cell's fraction at each
        key = (cid[:, None] * Nw + flat).reshape(-1)
        uk, inv = torch.unique(key, return_inverse=True)
        phi_u = torch.zeros(uk.numel(), device=dev, dtype=dt_).index_add_(0, inv, dep)
        node = uk % Nw
        ck = uk // Nw
        phi = torch.zeros(Nw, device=dev, dtype=dt_).index_add_(0, node, phi_u)
        sq = torch.zeros(Nw, device=dev, dtype=dt_).index_add_(0, node, phi_u * phi_u)
        Pn = 0.5 * (phi * phi - sq)
        kk = self.k
        if self.stiffness_block:
            kk = C.state[:, C.state_schema[self.stiffness_block][0]][C.occ > 0.5].mean().to(dt_)
        s0 = self.c0
        if self.overlap_block:
            oc = C.state[:, C.state_schema[self.overlap_block][0]].to(dt_)
            s0 = torch.zeros(Nw, device=dev, dtype=dt_).index_add_(0, node, phi_u * oc[ck]) / phi.clamp(min=1e-12)
        dU = kk * (Pn - s0).clamp(min=0.0)
        # per key: dE/dphi_A(n) of the overlap energy, (dU)(phi - phi_A) at the node
        q = dU[node] * (phi[node] - phi_u)
        # the interface tension of each contact
        nt = C.node_type.long().to(dev)
        T = int(nt.max()) + 1
        g = torch.as_tensor(self.gamma, device=dev, dtype=dt_)
        if g.shape[0] < T:
            raise ValueError(f"pair_potential[cell_contact]: contact_surface_tension is {tuple(g.shape)}, the cell set has {T} types")
        g = g[:T, :T]
        tk = nt[ck]
        if self.ct_block:
            gam = C.state[:, C.state_schema[self.ct_block][0]].to(dt_)
            gk = gam[ck]                                              # the key's cell's contact tension (with its graph)
            # gamma_ab = (gamma_a + gamma_b) / 2:  S_a = sum_{b != a} gamma_ab L_b = gamma_a (L_all - L_a) / 2 + (L_G - gamma_a L_a) / 2,
            # L_all the Laplacian of the total fraction, L_G that of the gamma-weighted total; uniform gamma gives gamma (L_all - L_a)
            Phi2 = torch.zeros(2 * Nw, device=dev, dtype=dt_).index_add_(0, node, phi_u).index_add_(0, Nw + node, gk * phi_u).view(2, *shape)
            L2 = self._lattice_laplacian(Phi2, per)
            L2 = L2.reshape(2, Nw)
            S = 0.5 * gk * L2[0][node] + 0.5 * L2[1][node]
        else:
            Phi = torch.zeros(T * Nw, device=dev, dtype=dt_).index_add_(0, tk * Nw + node, phi_u).view(T, *shape)
            L = self._lattice_laplacian(Phi, per)
            Lflat = L.reshape(T, Nw)
            S = torch.zeros_like(phi_u)
            for U in range(T):
                S = S + g[tk, U] * Lflat[U][node]
        # the seven-point sum of the cell's own fraction: look the six neighbouring keys up
        ix = node // (Ny * Nz)
        iy = (node // Nz) % Ny
        iz = node % Nz
        Lown = self._own_laplacian(phi_u, uk, ck, ix, iy, iz, shape, per, Nw)
        S = S - (gk if self.ct_block else g[tk, tk]) * Lown
        wid = self.width
        if self.width_block:
            wid = C.state[:, C.state_schema[self.width_block][0]].to(dt_)[ck]                    # each key's own cell's boundary width
        e_u = q + (wid * dx / dx ** 3) * S                      # the force's node factor, overlap and tension: (V grad w) . e_u[key]
        # per point: F_i = - V_i sum_stencil G_in (dU other) - (V_i / dx^3) sum G_in (w dx S_n)   [same as the dense]
        eu = e_u[inv].view(-1, 27)
        F = -V[:, None] * (G * eu[..., None]).sum(1)
        m = P.mass.to(dt_) if getattr(P, "mass", None) is not None else torch.ones_like(live)
        acc = F / m.clamp_min(1e-30)[:, None]
        acc = acc * live[:, None]
        if mask is not None:
            acc = acc * mask[:, None].to(dt_)
        H.cell_contact_max = float(Pn.max()) if Pn.numel() else 0.0
        return {self.at: acc}


@register_operator("pair_potential", implementation="cell_contact_touched_nodes", family="interaction", set="particle", kind="exchange",
                   title="keep cells apart and set the tension of each contact, with every field stored on the lattice nodes some cell touches", equation=r"""$$E=E_{\rm overlap}-w\sum_{A<B}\gamma_{\tau_A\tau_B}\int\nabla\phi_A\cdot\nabla\phi_B\,dV\;\approx\;E_{\rm overlap}+\sum_{A<B}\gamma_{\tau_A\tau_B}\,S_{AB},\qquad \frac{\partial E}{\partial\phi_A(n)}=w\,\Delta x\sum_{B\neq A}\gamma_{\tau_A\tau_B}\,\nabla^2\phi_B(n)$$""")
class CellContactVolumeInterfaceTouched(CellContactVolumeInterfaceSparse):
    r"""pair_potential[cell_contact] with the TOTAL volume fraction and its Laplacian also stored only on the lattice nodes some cell touches: memory and time follow the number of points, not the volume of the world.
    particle -> particle: as cell_contact (same parameters, same equation, same arithmetic, results equal to float round-off); emits an acceleration.

        E=E_{\rm overlap}-w\sum_{A<B}\gamma_{\tau_A\tau_B}\int\nabla\phi_A\cdot\nabla\phi_B\,dV\;\approx\;E_{\rm overlap}+\sum_{A<B}\gamma_{\tau_A\tau_B}\,S_{AB},\qquad \frac{\partial E}{\partial\phi_A(n)}=w\,\Delta x\sum_{B\neq A}\gamma_{\tau_A\tau_B}\,\nabla^2\phi_B(n)
        E              the total contact energy
        E_{overlap}    the overlap penalty of volume_energy
        w              `contact_edge_width`, the interface width factor
        \gamma_{\tau_A\tau_B} `contact_surface_tension`, the tension of a contact between types tau_A and tau_B (a symmetric matrix, in the units of cell_surface_tension)
        \phi_A, \phi_B the two cells' volume fractions
        S_{AB}         the contact area of A and B
        \Delta x       the lattice spacing

    Reference: as cell_contact; Plexus exp39, the active foam fusion testbench (this work, 2026-11-08: nine pairs of 100-cell spheroids in one world needed 227 million lattice nodes of the dense total fraction)."""
    EXPLAIN = (
        r"The same contact energy as `cell_contact`: cells are kept from overlapping and every contact has the tension $\gamma$ of the two cell types times the area $S_{AB}$ of its interface. "
        r"`cell_contact` stores each cell's own volume fraction only where the cell is, but the sum of all cells' fractions and its Laplacian over the whole world; this variant stores "
        r"those too only on the lattice nodes some cell touches and reads a neighbour node by a sorted search, so the cost follows the number of points and not the volume of the box "
        r"(nine pairs of 100 cells in one box of 122 length units: 227 million nodes dense). The result is the same to float round-off."
    )
    PARAM_UNITS = {}
    SYMBOLS = dict(CellContactVolumeInterfaceSparse.SYMBOLS)
    PARAM_ROLES = dict(CellContactVolumeInterfaceSparse.PARAM_ROLES)

    @staticmethod
    def _neighbours(un, shape, per):
        """The six face neighbours of the sorted unique node ids `un` as (position in `un`, found) pairs, in the order of the dense stencil (axis 0 up and down, then 1, then 2); a neighbour no cell touches holds 0, an edge repeats itself (as `_shift`), the x axis wraps when periodic"""
        Ny, Nz = shape[1], shape[2]
        ix, iy, iz = un // (Ny * Nz), (un // Nz) % Ny, un % Nz
        out = []
        for d, (a_, n_) in enumerate(((ix, shape[0]), (iy, Ny), (iz, Nz))):
            for sgn in (1, -1):
                nb = a_ + sgn
                nb = torch.remainder(nb, n_) if (d == 0 and per) else nb.clamp(0, n_ - 1)
                if d == 0:
                    fnb = (nb * Ny + iy) * Nz + iz
                elif d == 1:
                    fnb = (ix * Ny + nb) * Nz + iz
                else:
                    fnb = (ix * Ny + iy) * Nz + nb
                pos = torch.searchsorted(un, fnb).clamp_(max=un.numel() - 1)
                out.append((pos, un[pos] == fnb))
        return out

    @staticmethod
    def _touched_laplacian(F, nbr):
        """The seven-point Laplacian (times the lattice spacing squared) of the fields F [k, nodes] on the touched nodes, summed in the order of `_lattice_laplacian`"""
        L = -6.0 * F
        for i in range(0, 6, 2):
            (p1, h1), (p2, h2) = nbr[i], nbr[i + 1]
            L = L + torch.where(h1, F[:, p1], torch.zeros_like(F)) + torch.where(h2, F[:, p2], torch.zeros_like(F))
        return L

    def forward(self, H, mask=None):
        P = H.level(self.at)
        X = P.get("pos")
        dev, dt_ = X.device, X.dtype
        live = (P.occ > 0).to(dt_)
        cid = H.lift_index(self.at, self.cells).long()
        C = H.level(self.cells)
        dx = float(self.dx) if self.dx is not None else float(H.field(self.grid).dx)
        world = H.world_size
        shape = [int(math.ceil(float(w) / dx)) + 3 for w in world]
        per = self.period > 0.0
        if per:
            nx = int(round(self.period / dx))
            if abs(self.period / dx - nx) > 1e-6:
                raise ValueError(f"pair_potential[cell_contact_touched_nodes]: period {self.period} is not a whole number of lattice spacings dx {dx}")
            shape[0] = nx
        Ny, Nz = shape[1], shape[2]
        Nw = shape[0] * Ny * Nz                                   # the key radix only: nothing of this size is allocated
        if CellContactVolume._OFF is None or CellContactVolume._OFF.device != dev:
            o = torch.arange(3, device=dev)
            CellContactVolume._OFF = torch.stack(torch.meshgrid(o, o, o, indexing="ij"), -1).reshape(-1, 3)
        off = CellContactVolume._OFF
        xg = X / dx
        base = torch.floor(xg - 0.5)
        fx = xg - base
        w = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], 1)
        dw = torch.stack([fx - 1.5, -2.0 * (fx - 1.0), fx - 0.5], 1) / dx
        idx = base.long()[:, None, :] + off[None]
        for d in range(3):
            if d == 0 and per:
                idx[..., 0] = torch.remainder(idx[..., 0], shape[0])
            else:
                idx[..., d] = idx[..., d].clamp(0, shape[d] - 1)
        flat = (idx[..., 0] * Ny + idx[..., 1]) * Nz + idx[..., 2]
        wx, wy, wz = w[:, :, 0], w[:, :, 1], w[:, :, 2]
        W = (wx[:, off[:, 0]] * wy[:, off[:, 1]] * wz[:, off[:, 2]])
        G = torch.stack([dw[:, off[:, 0], 0] * wy[:, off[:, 1]] * wz[:, off[:, 2]],
                         wx[:, off[:, 0]] * dw[:, off[:, 1], 1] * wz[:, off[:, 2]],
                         wx[:, off[:, 0]] * wy[:, off[:, 1]] * dw[:, off[:, 2], 2]], -1)
        V = (P.p_vol.to(dt_) if getattr(P, "p_vol", None) is not None else torch.ones_like(live)) * live
        dep = (W * (V / dx ** 3)[:, None]).reshape(-1)
        # the keys: (cell, node) pairs, sorted; phi_u the cell's fraction at each; the nodes: the sorted unique node ids, `ninv` the node of each key
        key = (cid[:, None] * Nw + flat).reshape(-1)
        uk, inv = torch.unique(key, return_inverse=True)
        phi_u = torch.zeros(uk.numel(), device=dev, dtype=dt_).index_add_(0, inv, dep)
        node = uk % Nw
        ck = uk // Nw
        un, ninv = torch.unique(node, return_inverse=True)
        Nn = un.numel()
        phi = torch.zeros(Nn, device=dev, dtype=dt_).index_add_(0, ninv, phi_u)
        sq = torch.zeros(Nn, device=dev, dtype=dt_).index_add_(0, ninv, phi_u * phi_u)
        Pn = 0.5 * (phi * phi - sq)
        kk = self.k
        if self.stiffness_block:
            kk = C.state[:, C.state_schema[self.stiffness_block][0]][C.occ > 0.5].mean().to(dt_)
        s0 = self.c0
        if self.overlap_block:
            oc = C.state[:, C.state_schema[self.overlap_block][0]].to(dt_)
            s0 = torch.zeros(Nn, device=dev, dtype=dt_).index_add_(0, ninv, phi_u * oc[ck]) / phi.clamp(min=1e-12)
        dU = kk * (Pn - s0).clamp(min=0.0)
        # per key: dE/dphi_A(n) of the overlap energy, (dU)(phi - phi_A) at the node
        q = dU[ninv] * (phi[ninv] - phi_u)
        # the interface tension of each contact
        nt = C.node_type.long().to(dev)
        T = int(nt.max()) + 1
        g = torch.as_tensor(self.gamma, device=dev, dtype=dt_)
        if g.shape[0] < T:
            raise ValueError(f"pair_potential[cell_contact_touched_nodes]: contact_surface_tension is {tuple(g.shape)}, the cell set has {T} types")
        g = g[:T, :T]
        tk = nt[ck]
        nbr = self._neighbours(un, shape, per)
        if self.ct_block:
            gam = C.state[:, C.state_schema[self.ct_block][0]].to(dt_)
            gk = gam[ck]                                              # the key's cell's contact tension (with its graph)
            # gamma_ab = (gamma_a + gamma_b) / 2:  S_a = sum_{b != a} gamma_ab L_b = gamma_a (L_all - L_a) / 2 + (L_G - gamma_a L_a) / 2,
            # L_all the Laplacian of the total fraction, L_G that of the gamma-weighted total; uniform gamma gives gamma (L_all - L_a)
            F2 = torch.zeros(2 * Nn, device=dev, dtype=dt_).index_add_(0, ninv, phi_u).index_add_(0, Nn + ninv, gk * phi_u).view(2, Nn)
            L2 = self._touched_laplacian(F2, nbr)
            S = 0.5 * gk * L2[0][ninv] + 0.5 * L2[1][ninv]
        else:
            F = torch.zeros(T * Nn, device=dev, dtype=dt_).index_add_(0, tk * Nn + ninv, phi_u).view(T, Nn)
            L = self._touched_laplacian(F, nbr)
            S = torch.zeros_like(phi_u)
            for U in range(T):
                S = S + g[tk, U] * L[U][ninv]
        # the seven-point sum of the cell's own fraction: look the six neighbouring keys up
        ix = node // (Ny * Nz)
        iy = (node // Nz) % Ny
        iz = node % Nz
        Lown = self._own_laplacian(phi_u, uk, ck, ix, iy, iz, shape, per, Nw)
        S = S - (gk if self.ct_block else g[tk, tk]) * Lown
        wid = self.width
        if self.width_block:
            wid = C.state[:, C.state_schema[self.width_block][0]].to(dt_)[ck]                    # each key's own cell's boundary width
        e_u = q + (wid * dx / dx ** 3) * S                      # the force's node factor, overlap and tension: (V grad w) . e_u[key]
        eu = e_u[inv].view(-1, 27)
        F_ = -V[:, None] * (G * eu[..., None]).sum(1)
        m = P.mass.to(dt_) if getattr(P, "mass", None) is not None else torch.ones_like(live)
        acc = F_ / m.clamp_min(1e-30)[:, None]
        acc = acc * live[:, None]
        if mask is not None:
            acc = acc * mask[:, None].to(dt_)
        H.cell_contact_max = float(Pn.max()) if Pn.numel() else 0.0
        return {self.at: acc}


@register_operator("active_force", "pulse_to_contraction", family="mechanics", set="particle", kind="exchange",
                   model="sphere_crawl_seek", title="crawl cells over a sphere and look for the layer below",
                   equation=r"""$$\mathbf a_i=\mathbf a_i^{\rm crawl}-s\,(1-\gamma_c)\,\chi_i\,\hat{\mathbf n}_c-\overline{(\cdot)},\qquad \gamma_c=\min\Big(1,\ \tfrac{1}{A^\ast}\sum_{J\ni c}A_J\,\rho_J\Big)$$""")
class ActiveForceSphereCrawlSeek(ActiveForceSphereCrawl):
    r"""active_force[sphere_crawl] with a search for adhesion to the layer below: the walk is switched off in proportion to the cell's grip on the substrate, and a cell without grip is pushed inward on its innermost band.
    particle -[containment]-> cell, once per frame as sphere_crawl; needs `traction`.

        \mathbf a_i=\mathbf a_i^{\rm crawl}-s\,(1-\gamma_c)\,\chi_i\,\hat{\mathbf n}_c-\overline{(\cdot)},\qquad \gamma_c=\min\Big(1,\ \tfrac{1}{A^\ast}\sum_{J\ni c}A_J\,\rho_J\Big)
        a_i            the acceleration of point i
        a_i^{crawl}    sphere_crawl's acceleration
        s              `seek`, the inward push per unit mass of a cell with no grip
        \gamma_c       the cell's grip: its bound junction area A_J rho_J summed over its junctions, over the full-grip area A^*, capped at 1
        \chi_i         1 on the innermost band
        \hat n_c       the radial direction
        \overline{(\cdot)} the per-cell mean, removed

    Reference: Plexus exp27, the cell adhesion testbench (this work, 2026-10) (2026-10-20; Cedric: the cells looking for adhesion to the inner layer)."""
    PARAM_UNITS = {'seek': 'acceleration'}
    SYMBOLS = {'a_i': 'the acceleration of point i', 'a_i^{crawl}': "sphere_crawl's acceleration", 's': '`seek`, the inward push per unit mass of a cell with no grip', '\\gamma_c': "the cell's grip: its bound junction area A_J rho_J summed over its junctions, over the full-grip area A^*, capped at 1", '\\chi_i': '1 on the innermost band', '\\hat n_c': 'the radial direction', '\\overline{(\\cdot)}': 'the per-cell mean, removed'}
    PARAM_ROLES = dict(ActiveForceSphereCrawl.PARAM_ROLES, seek="inward_acceleration_of_a_cell_with_no_grip")

    def __init__(self, params, device="cpu"):
        super().__init__({k: v for k, v in params.items() if k != "seek"}, device)
        self.seek = float(params.get("seek", 0.0))
        if self.seek > 0.0 and self.traction <= 0.0:
            raise ValueError("active_force[sphere_crawl_seek]: seek needs `traction` > 0 (the grip it answers to)")

    def forward(self, H, mask=None):
        out = super().forward(H, mask)
        if self.seek <= 0.0:
            return out
        lvl = H.level(self.at)
        a0 = out[self.at]
        dev, dt_ = a0.device, a0.dtype
        X = lvl.get("pos").detach().cpu().numpy().astype(np.float64)
        alive = (lvl.occ > 0.5).cpu().numpy()
        nc = int(H.level(self.cell_set).state.shape[0])
        idx = H.lift_index(self.at, self.cell_set).cpu().numpy()
        sel = alive.copy()
        if mask is not None:
            sel &= (mask.detach().cpu().numpy() > 0)
        cnt = np.zeros(nc); np.add.at(cnt, idx[alive], 1.0)
        xb = np.zeros((nc, 3)); np.add.at(xb, idx[alive], X[alive]); xb /= np.maximum(cnt, 1.0)[:, None]
        nrm = xb - self.c0
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        gate = self._traction_gate(H, nc)
        rr = np.linalg.norm(X - self.c0, axis=1)
        rmin = np.full(nc, np.inf); np.minimum.at(rmin, idx[sel], rr[sel])
        chi = (sel & (rr <= rmin[idx] + float(self.basal_band))).astype(np.float64)
        n_all = np.zeros(nc); np.add.at(n_all, idx[sel], 1.0)
        n_bas = np.zeros(nc); np.add.at(n_bas, idx[sel], chi[sel])
        chi = chi * (n_all / np.maximum(n_bas, 1.0))[idx]
        s = self.seek * (1.0 - gate)
        a = -(s[idx] * chi)[:, None] * nrm[idx]
        a[~sel] = 0.0
        mp = getattr(lvl, "mass", None)
        m = (mp.detach().cpu().numpy().astype(np.float64) if mp is not None else np.ones(len(X))) * alive
        a = a - ((m[:, None] * a).sum(0) / max(float(m.sum()), 1e-30))[None, :] * alive[:, None]
        self.last["seek_mean"] = float(s[np.unique(idx[sel])].mean()) if sel.any() else 0.0
        return {self.at: a0 + torch.as_tensor(a, device=dev, dtype=dt_)}


@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="spheroid_coat",
                   title="lay out a spheroid with a basement membrane or a mesenchyme outside it", equation=r"""$$\mathbf c^{coat}_m=\mathbf o+(R_{out}+g+\tfrac{h}{2})\,\hat{\mathbf f}_m\ \ (\text{tiles}),\qquad \mathbf c^{coat}_k=\mathbf o+a_c(\mathbf k+\boldsymbol\epsilon_k),\ R_{out}+g<\lvert\mathbf c-\mathbf o\rvert<R_{out}+g+T\ \ (\text{cells})$$""")
class SeedSpheroidCoat(SeedSpheroid):
    r"""seed_positions[spheroid] and a coat outside the outer layer, the live cells of type 2: flat tiles on a Fibonacci sphere (a basement membrane) or ordinary cells in a shell (a mesenchyme).
    pt -> pt, cell: as spheroid, plus the type-2 cells' points and `tension` (the per-cell factor of body_csf_balanced's cell_surface_tension_block).

        \mathbf c^{coat}_m=\mathbf o+(R_{out}+g+\tfrac{h}{2})\,\hat{\mathbf f}_m\ \ (\text{tiles}),\qquad \mathbf c^{coat}_k=\mathbf o+a_c(\mathbf k+\boldsymbol\epsilon_k),\ R_{out}+g<\lvert\mathbf c-\mathbf o\rvert<R_{out}+g+T\ \ (\text{cells})
        c^{coat}_m     the centre of coat tile m on a Fibonacci sphere
        R_{out}        the outer cells' outer surface
        g              `coat_gap`
        h              the tile's thickness, `layers` x point_spacing
        c^{coat}_k     the centre of a coat cell on a jittered lattice
        a_c            `coat_spacing`
        T              `coat_thickness`

    Reference: Plexus exp27, the cell adhesion testbench (this work, 2026-10) (2026-10-20; Shaohe Wang: a viscoelastic basement membrane outside, or viscous mesenchyme)."""
    PARAM_UNITS = {'coat_gap': 'length', 'coat_spacing': 'length', 'coat_tension': 'dimensionless', 'coat_thickness': 'length', 'layers': 'count'}
    SYMBOLS = {'c^{coat}_m': 'the centre of coat tile m on a Fibonacci sphere', 'R_{out}': "the outer cells' outer surface", 'g': '`coat_gap`', 'h': "the tile's thickness, `layers` x point_spacing", 'c^{coat}_k': 'the centre of a coat cell on a jittered lattice', 'a_c': '`coat_spacing`', 'T': '`coat_thickness`'}
    PARAM_ROLES = dict(SeedSpheroid.PARAM_ROLES, coat="tiles_or_cells", tile="tile_side_world", layers="tile_point_layers",
                       coat_gap="gap_between_outer_cells_and_coat", coat_thickness="mesenchyme_shell_thickness",
                       coat_spacing="mesenchyme_cell_centre_spacing", coat_tension="tension_factor_of_the_coat_cells")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.coat = str(params.get("coat", "tiles"))
        if self.coat not in ("tiles", "cells"):
            raise ValueError("seed_positions[spheroid_coat]: coat is tiles or cells")
        self.layers = int(params.get("layers", 2))
        self.tile = 16.0 * self.h
        self.coat_gap = float(params.get("coat_gap", 0.05))
        self.coat_thick = float(params.get("coat_thickness", 1.5))
        self.coat_sp = float(params.get("coat_spacing", 1.05))
        self.coat_tension = float(params.get("coat_tension", 0.0 if self.coat == "tiles" else 1.0))

    def forward(self, H, mask=None):
        from scipy.spatial import cKDTree
        from plexus.operators.gastrulation_ops import _lattice
        P, C = H.level(self.at), H.level(self.cell_set)
        rng = np.random.default_rng(self.seed)
        live = np.flatnonzero(C.occ.detach().cpu().numpy() > 0.5)
        nt = C.node_type.detach().cpu().numpy()
        in_ids, out_ids, coat_ids = live[nt[live] == 0], live[nt[live] == 1], live[nt[live] == 2]
        n_in, n_out, n_coat = len(in_ids), len(out_ids), len(coat_ids)
        if n_coat == 0:
            raise ValueError("seed_positions[spheroid_coat]: the cell set has no type-2 cells (declare the coat type after `outer`)")
        if self.hollow:
            kk = np.arange(n_in) + 0.5
            ph = np.arccos(1.0 - 2.0 * kk / n_in); tt = math.pi * (1.0 + math.sqrt(5.0)) * kk
            cand = self.c + self.r_inner * np.stack([np.sin(ph) * np.cos(tt), np.sin(ph) * np.sin(tt), np.cos(ph)], 1)
            r_edge = self.r_inner
        else:
            R0 = 0.6 * (3.0 * n_in * self.sp ** 3 / (4.0 * math.pi)) ** (1.0 / 3.0) + 2 * self.sp
            cand = _lattice(self.sp, self.c - R0, self.c + R0, rng, jitter=0.15)
            cand = cand[np.argsort(np.linalg.norm(cand - self.c, axis=1))[:n_in]]
            r_edge = float(np.linalg.norm(cand - self.c, axis=1).max())
        R_s = r_edge + self.shell
        k = np.arange(n_out) + 0.5
        phi = np.arccos(1.0 - 2.0 * k / n_out); th = math.pi * (1.0 + math.sqrt(5.0)) * k
        outer = self.c + R_s * np.stack([np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)], 1)
        centres = np.zeros((C.state.shape[0], 3))
        centres[in_ids] = cand[rng.permutation(n_in)]
        centres[out_ids] = outer
        R_ball = R_s + self.ext
        parent = P.parent.detach().cpu().numpy()
        blocks = {int(c): np.flatnonzero(parent == c) for c in live}
        pos_of, vol_of = {}, {}
        # inner and outer cells: the points of the ball, each to the nearest inner / outer centre
        core = np.concatenate([in_ids, out_ids])
        pts = _lattice(self.h, self.c - R_ball, self.c + R_ball, rng, jitter=0.15)
        rp_ = np.linalg.norm(pts - self.c, axis=1)
        pts = pts[(rp_ < R_ball) & (rp_ > (r_edge - self.inner_ext if self.hollow else 0.0))]
        owner = core[cKDTree(centres[core]).query(pts, k=1)[1]]
        if self.coat == "tiles":
            z = self.layers * self.h
            R_c = R_ball + self.coat_gap + 0.5 * z
            kc = np.arange(n_coat) + 0.5
            pc = np.arccos(1.0 - 2.0 * kc / n_coat); tc = math.pi * (1.0 + math.sqrt(5.0)) * kc
            nrm = np.stack([np.sin(pc) * np.cos(tc), np.sin(pc) * np.sin(tc), np.cos(pc)], 1)
            centres[coat_ids] = self.c + R_c * nrm
            ii, jj, kk3 = np.meshgrid(np.arange(16), np.arange(16), np.arange(self.layers), indexing="ij")
            loc = np.stack([(ii.ravel() - 7.5) * self.h, (jj.ravel() - 7.5) * self.h, (kk3.ravel() + 0.5 - 0.5 * self.layers) * self.h], 1)
            for q, c in enumerate(coat_ids):
                n_ = nrm[q]
                e1 = np.cross(np.array([0.0, 0.0, 1.0]), n_)
                if np.linalg.norm(e1) < 1e-6:
                    e1 = np.array([1.0, 0.0, 0.0])
                e1 = e1 / np.linalg.norm(e1)
                e2 = np.cross(n_, e1)
                pos_of[int(c)] = centres[c] + loc[:, 0:1] * e1 + loc[:, 1:2] * e2 + loc[:, 2:3] * n_ + rng.normal(0, 0.02 * self.h, (len(loc), 3))
                vol_of[int(c)] = (self.tile * self.tile * z)
            R_outer = R_c + 0.5 * z
        else:
            R_in2, R_out2 = R_ball + self.coat_gap, R_ball + self.coat_gap + self.coat_thick
            cc = _lattice(self.coat_sp, self.c - R_out2, self.c + R_out2, rng, jitter=0.15)
            rc = np.linalg.norm(cc - self.c, axis=1)
            cc = cc[(rc > R_in2 + 0.4) & (rc < R_out2 - 0.2)]
            cc = cc[np.argsort(np.abs(np.linalg.norm(cc - self.c, axis=1) - 0.5 * (R_in2 + R_out2)))[:n_coat]]
            if len(cc) < n_coat:
                raise ValueError(f"seed_positions[spheroid_coat]: room for {len(cc)} mesenchyme cells in the shell, {n_coat} declared (thicken `coat_thickness`)")
            centres[coat_ids] = cc[rng.permutation(len(cc))][:n_coat]
            pts2 = _lattice(self.h, self.c - R_out2, self.c + R_out2, rng, jitter=0.15)
            r2 = np.linalg.norm(pts2 - self.c, axis=1)
            pts2 = pts2[(r2 >= R_in2) & (r2 < R_out2)]
            own2 = coat_ids[cKDTree(centres[coat_ids]).query(pts2, k=1)[1]]
            for c in coat_ids:
                pos_of[int(c)] = pts2[own2 == c]
                vol_of[int(c)] = max(len(pos_of[int(c)]), 1) * self.h ** 3
            R_outer = R_out2
        for c in core:
            pos_of[int(c)] = pts[owner == c]
            vol_of[int(c)] = max(len(pos_of[int(c)]), 1) * self.h ** 3
        dev, dt_ = P.state.device, P.state.dtype
        occ = torch.zeros_like(P.occ)
        short = 0
        for c in live:
            blk = blocks[int(c)]
            own = pos_of[int(c)]
            vol = vol_of[int(c)]
            if len(own) == 0:
                own = centres[c][None] + rng.normal(0, 0.1 * self.sp, (8, 3)); vol = 8 * self.h ** 3
            if len(own) >= len(blk):
                take = own[rng.choice(len(own), len(blk), replace=False)]
            else:
                short += 1
                extra = own[rng.integers(0, len(own), len(blk) - len(own))] + rng.normal(0, 0.3 * self.h, (len(blk) - len(own), 3))
                take = np.concatenate([own, extra])
            idx = torch.as_tensor(blk, device=dev)
            _mpm_reset_points(P, idx, torch.as_tensor(take, device=dev, dtype=dt_),
                              torch.full((len(blk),), vol / len(blk), device=dev, dtype=dt_))
            _set_block(P, "pid", idx, float(c)); _set_block(P, "cellc", idx, float(c % 20))
            if "tcol" in P.state_schema:
                _set_block(P, "tcol", idx, float(nt[c]))
            occ[idx] = 1.0
        P.occ = occ
        ca, cb = C.state_schema["pos"]
        st = C.state.clone()
        cen = np.zeros((C.state.shape[0], 3))
        for c in live:                                        # a cell's position is the mean of its points
            cen[c] = pos_of[int(c)].mean(0) if len(pos_of[int(c)]) else centres[c]
        st[:, ca:cb] = torch.as_tensor(cen, dtype=st.dtype, device=st.device)
        if "adh" in C.state_schema:
            a0, a1 = C.state_schema["adh"]; st[:, a0:a1] = self.adh
        if "tension" in C.state_schema:
            t0 = C.state_schema["tension"][0]
            st[:, t0] = 1.0
            st[torch.as_tensor(coat_ids, device=st.device), t0] = self.coat_tension
        C.state = st
        print(f"[seed_positions:spheroid_coat] {n_in} inner + {n_out} outer cells + {n_coat} coat ({self.coat}) cells; inner centres to radius "
              f"{r_edge:.2f}, outer centres at {R_s:.2f}, coat out to {R_outer:.2f}; {int(occ.sum()):,} points; {short} cells padded", flush=True)
        return {}


@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="vesicle",
                   title="seed a closed one-cell-thick shell of cells round an empty lumen",
                   equation=r"""$$\mathbf c_k=\mathbf o+R\,\hat{\mathbf n}_k,\ \ \hat{\mathbf n}_k=\text{Fibonacci}(k,n),\quad R=\ell\sqrt{n/4\pi},\qquad \pi(i)=\arg\min_k\lvert\mathbf x_i-\mathbf c_k\rvert$$""")
class SeedVesicle(Seed):
    r"""a vesicle of MPM cells: every live cell is one cell of one layer on a sphere, nothing inside it, the MPM twin of the vertex clips' seed_mesh vesicle.
    pt -> pt, cell: `pid`, `cellc`, `tcol` on the points; `pos`, `adh` (all six at `adh`), `tension` (1, when the block exists) on the cells; `chem` untouched.

        \mathbf c_k=\mathbf o+R\,\hat{\mathbf n}_k,\ \ \hat{\mathbf n}_k=\text{Fibonacci}(k,n),\quad R=\ell\sqrt{n/4\pi},\qquad \pi(i)=\arg\min_k\lvert\mathbf x_i-\mathbf c_k\rvert
        c_k            the centre of cell k
        o              `centre`
        R              `radius`, or auto: spacing sqrt(n / 4 pi) so each cell has about spacing^2 of the sphere
        \hat n_k       the k-th direction of the Fibonacci sphere
        n              the live cells
        \ell           `spacing`
        \pi(i)         the cell of point i (Voronoi, between `inner_extent` and `outer_extent`)

    Reference: Plexus exp29 (this work, 2026-10-21); the Fibonacci sphere: Gonzalez, A. (2010). Math. Geosci. 42:49-64."""
    PARAM_UNITS = {'centre': 'length', 'jitter': 'fraction', 'live_per_cell': 'count', 'point_spacing': 'length', 'radius': 'length', 'radius_shift': 'length', 'seed': 'count', 'spacing': 'length', 'inner_extent': 'length', 'outer_extent': 'length', 'adh': 'dimensionless', 'vesicles': 'count'}
    SYMBOLS = {'c_k': 'the centre of cell k', 'o': '`centre`', 'R': '`radius`, or auto: spacing sqrt(n / 4 pi) so each cell has about spacing^2 of the sphere', '\\hat n_k': 'the k-th direction of the Fibonacci sphere', 'n': 'the live cells', '\\ell': '`spacing`', '\\pi(i)': 'the cell of point i (Voronoi, between `inner_extent` and `outer_extent`)'}
    REQUIRES_PARAMS = ["cell_set", "centre"]
    SUPPORTED_DIMS = [3]
    PARAM_ROLES = {"cell_set": "the_cell_set", "centre": "vesicle_centre", "radius": "radius_of_the_cell_centres_or_auto",
                   "spacing": "cell_centre_spacing_on_the_sphere", "inner_extent": "cell_extent_inward_of_its_centre",
                   "outer_extent": "cell_extent_outward_of_its_centre", "point_spacing": "material_point_lattice_spacing",
                   "adh": "isotropic_adhesion_maximum", "adh_by_class": "adhesion_maximum_per_face_pair_class",
                   "vesicles": "one_vesicle_per_cell_type_with_its_own_centre_tension_polarity_adhesion",
                   "seed": "rng_seed", "jitter": "centre_jitter_in_spacings", "live_per_cell": "points_seeded_per_cell_the_rest_of_the_block_dormant", "rotate": "euler_angles_degrees_of_the_whole_vesicle", 'radius_shift': 'shift_of_the_radius'}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.cell_set = str(params["cell_set"])
        self.c = np.array([float(v) for v in params["centre"]])
        self.radius = params.get("radius", "auto")
        # `live_per_cell` (exp30/31, 2026-10-24): the points seeded per cell = the LIVE HEAD of its block; the rest of the block (the set's
        # `grow_reserve`, dormant slots `cell_populate` wakes) stays dormant. Without it the seed fills the WHOLE block -- with a reserve of
        # 78 on a 125-point cell that made 154-point cells at the seed (23 % more points in the same volume) and ate 1,160 of the pool.
        # None (the default) keeps the old behaviour: the block is the cell.
        self.live_per_cell = params.get("live_per_cell")
        self.sp = float(params.get("spacing", 1.05))
        self.ext_in = float(params.get("inner_extent", 0.5))
        self.ext_out = float(params.get("outer_extent", 0.5))
        self.h = float(params.get("point_spacing", 0.125))
        self.adh = float(params.get("adh", 1.0))
        # `adh_by_class: {LL: .., AA: .., BB: .., AB: .., LA: .., LB: ..}` (exp29, 2026-10-23): the cell's six adhesion maxima by
        # face-pair class in place of one value for all (a polarised epithelium: lateral-lateral high, the rest low); a class
        # left out takes `adh`
        ac = params.get("adh_by_class")
        self.adh6 = [float(ac.get(k, self.adh)) for k in FG_NAMES] if ac else None
        # `vesicles: [{centre: [x, y, z], tension: 1.0, polar: true, adh_by_class: {..}}, ...]` (exp29 rig, 2026-10-23): one vesicle per cell type
        # (vesicle g = the cells of type g) about its own centre, each with its own per-cell surface-tension factor, polarity and adhesion
        self.vesicles = params.get("vesicles")
        self.seed = int(params.get("seed", 0))
        self.jit = float(params.get("jitter", 0.0))
        # `radius_shift` (exp33, 2026-10-25, Cedric: "reduce the radius, keep the cells as they are and close the holes"): after the tiling,
        # every cell is moved as a RIGID body along its radius by this length (negative = inward), its points' relative positions and the
        # cell's size untouched; the cells then overlap in-plane by about the shift x (cell width / R). 0 (the default) changes nothing.
        self.radius_shift = float(params.get("radius_shift", 0.0))
        # `rotate: [ax, ay, az]` (exp33, 2026-10-25): the whole seeded vesicle turned about its centre by these angles in degrees (x, then y, then z), points and cell
        # centres together; a test of whether a shape follows the MPM grid's axes (it stays on them) or the seed (it turns). None (the default) changes nothing.
        self.rotate = None if params.get("rotate") is None else [float(v) for v in params["rotate"]]

    def forward(self, H, mask=None):
        from scipy.spatial import cKDTree
        from plexus.operators.gastrulation_ops import _lattice
        P, C = H.level(self.at), H.level(self.cell_set)
        rng = np.random.default_rng(self.seed)
        live_all = np.flatnonzero(C.occ.detach().cpu().numpy() > 0.5)
        nt = C.node_type.detach().cpu().numpy()
        # ONE vesicle over every live cell (`centre`), or, with `vesicles`, one per entry: the cells of type g form vesicle g about its own
        # `centre`, with its own `tension` factor (the per-cell surface-tension factor), `polar` flag (face labels by radial thirds, else
        # every point lateral) and `adh_by_class` (exp29 rig, 2026-10-23: several vesicles with different parameters in one box)
        if self.vesicles is None:
            groups = [(live_all, self.c, {})]
        else:
            groups = [(live_all[nt[live_all] == g], np.array([float(x) for x in v["centre"]]), v) for g, v in enumerate(self.vesicles)]
        parent = P.parent.detach().cpu().numpy()
        dev, dt_ = P.state.device, P.state.dtype
        nC = C.state.shape[0]
        occ = torch.zeros_like(P.occ)
        short = 0
        cen = np.zeros((nC, 3))
        vcen = np.tile(groups[0][1], (nC, 1))
        tens = np.ones(nC)
        adh_rows = np.tile(np.asarray(self.adh6 if self.adh6 else [self.adh] * 6, dtype=np.float64), (nC, 1))
        msgs = []
        for gi, (live, cg, opt) in enumerate(groups):
            n = len(live)
            core = n < 4 and bool(opt.get("core", False))     # `core: true` (exp29 sphere twin, 2026-11-01): 1-3 cells at the vesicle's centre (a ball of `core_radius`), not a shell, so a cell can sit inside the lumen
            if n < 4 and not core:
                raise ValueError(f"seed_positions[vesicle]: {n} live cells in vesicle {gi}; a shell needs at least 4")
            R = 0.0 if core else self.sp * math.sqrt(n / (4.0 * math.pi)) if self.radius == "auto" else float(self.radius)
            k = np.arange(n) + 0.5
            phi = np.arccos(1.0 - 2.0 * k / n); th = math.pi * (1.0 + math.sqrt(5.0)) * k
            nrm = np.stack([np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)], 1)
            if self.jit > 0.0:                                    # a tangential wobble, so no cell sits on a lattice site
                nrm = nrm + self.jit * self.sp / R * rng.normal(size=nrm.shape)
                nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
            centres = np.zeros((nC, 3))
            centres[live] = (cg + R * nrm)[rng.permutation(n)]
            R_out = float(opt.get("core_radius", 0.62)) if core else R + self.ext_out
            pts = _lattice(self.h, cg - R_out, cg + R_out, rng, jitter=0.15)
            rp_ = np.linalg.norm(pts - cg, axis=1)
            pts = pts[(rp_ < R_out) & (rp_ > R - self.ext_in)]
            owner = live[cKDTree(centres[live]).query(pts, k=1)[1]]
            polar = bool(opt.get("polar", True))
            ac = opt.get("adh_by_class")
            row = np.asarray([float(ac.get(kk, self.adh)) for kk in FG_NAMES] if ac else (self.adh6 if self.adh6 else [self.adh] * 6), dtype=np.float64)
            for c in live:
                blk = np.flatnonzero(parent == c)
                if self.live_per_cell:
                    blk = blk[: int(self.live_per_cell)]                 # the live head; the reserve tail stays dormant (occ 0)
                own = pts[owner == c]
                vol = max(len(own), 1) * self.h ** 3
                if len(own) == 0:
                    own = centres[c][None] + rng.normal(0, 0.1 * self.sp, (8, 3)); vol = 8 * self.h ** 3
                cen[c] = own.mean(0)
                if len(own) >= len(blk):
                    take = own[rng.choice(len(own), len(blk), replace=False)]
                else:
                    short += 1
                    extra = own[rng.integers(0, len(own), len(blk) - len(own))] + rng.normal(0, 0.3 * self.h, (len(blk) - len(own), 3))
                    take = np.concatenate([own, extra])
                idx = torch.as_tensor(blk, device=dev)
                _mpm_reset_points(P, idx, torch.as_tensor(take, device=dev, dtype=dt_),
                                  torch.full((len(blk),), vol / len(blk), device=dev, dtype=dt_))
                _set_block(P, "pid", idx, float(c)); _set_block(P, "cellc", idx, float(c % 20))
                if "tcol" in P.state_schema:
                    _set_block(P, "tcol", idx, float(nt[c]))
                if "face" in P.state_schema:                  # material face labels by radial thirds of the cell's points: the inner third
                    face = np.full(len(blk), float(FACE_L))    # apical (toward the lumen), the outer third basal, the rest lateral;
                    if polar:                                  # `cell_populate` copies the label to the points it adds; a non-polar vesicle
                        rr = np.linalg.norm(take - cg, axis=1)  # is all lateral (colour weight 1: the isotropic cortex)
                        order = np.argsort(rr); third = len(blk) // 3
                        face[order[:third]] = FACE_A; face[order[len(blk) - third:]] = FACE_B
                    _set_block(P, "face", idx, torch.as_tensor(face, device=dev, dtype=dt_))
                occ[idx] = 1.0
            vcen[live] = cg
            tens[live] = float(opt.get("tension", 1.0))
            for b_, v_ in (opt.get("values") or {}).items():       # `values: {grow: 0.0029}`: a cell block set to one value for this group's cells (the vesicle seed writes no growth rate)
                _set_block(C, str(b_), torch.as_tensor(live, device=C.state.device), float(v_))
            adh_rows[live] = row
            sizes = np.array([(owner == c).sum() for c in live]) * self.h ** 3
            msgs.append(f"vesicle {gi}: {n} cells, radius {R:.2f}, centre {np.round(cg, 2).tolist()}, region volume per cell median {np.median(sizes):.2f}")
        P.occ = occ
        live = np.concatenate([g[0] for g in groups])
        if self.rotate is not None:                                     # the vesicle turned about its centre (see __init__)
            from scipy.spatial.transform import Rotation
            Rm = Rotation.from_euler("xyz", self.rotate, degrees=True).as_matrix()
            pa, pb = P.state_schema["pos"]
            X = P.state[:, pa:pb].detach().cpu().numpy()
            pid_np = P.parent.detach().cpu().numpy().astype(int)
            occ_np = occ.detach().cpu().numpy() > 0.5
            for c in live:
                m = occ_np & (pid_np == c)
                X[m] = (X[m] - vcen[c]) @ Rm.T + vcen[c]
                cen[c] = (cen[c] - vcen[c]) @ Rm.T + vcen[c]
            stp = P.state.clone(); stp[:, pa:pb] = torch.as_tensor(X, dtype=stp.dtype, device=stp.device); P.state = stp
        if self.radius_shift != 0.0:                                    # each live cell translated along its radius, rigidly (see __init__)
            pa, pb = P.state_schema["pos"]
            X = P.state[:, pa:pb].detach().cpu().numpy()
            pid_np = P.parent.detach().cpu().numpy().astype(int)
            occ_np = occ.detach().cpu().numpy() > 0.5
            for c in live:
                u = cen[c] - vcen[c]; nrm = np.linalg.norm(u)
                if nrm < 1e-9:
                    continue
                d = (u / nrm) * self.radius_shift
                m = occ_np & (pid_np == c)
                X[m] += d
                cen[c] = cen[c] + d
            stp = P.state.clone(); stp[:, pa:pb] = torch.as_tensor(X, dtype=stp.dtype, device=stp.device); P.state = stp
        ca, cb = C.state_schema["pos"]
        st = C.state.clone(); st[:, ca:cb] = torch.as_tensor(cen, dtype=st.dtype, device=st.device)
        if "pol" in C.state_schema:                       # the polarity axis: outward along the radius (contact_finder classes a
            q0, q1 = C.state_schema["pol"]                  # contact across it lateral, along it apical / basal)
            u = cen[live] - vcen[live]
            st[live, q0:q1] = torch.as_tensor(u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12), dtype=st.dtype, device=st.device)
        if "adh" in C.state_schema:
            a0, a1 = C.state_schema["adh"]
            st[:, a0:a1] = torch.as_tensor(adh_rows, dtype=st.dtype, device=st.device)
        if "tension" in C.state_schema:
            st[:, C.state_schema["tension"][0]] = torch.as_tensor(tens, dtype=st.dtype, device=st.device)
        if "vcentre" in C.state_schema:                   # each cell's own vesicle centre, for the operators that read the radial direction
            v0, v1 = C.state_schema["vcentre"]
            st[:, v0:v1] = torch.as_tensor(vcen, dtype=st.dtype, device=st.device)
        C.state = st
        print(f"[seed_positions:vesicle] {len(live)} cells in {len(groups)} vesicle(s), {int(occ.sum()):,} points; " + "; ".join(msgs)
              + f"; {short} cells padded", flush=True)
        return {}


# ==========================================================================================================
# `seed_positions[from_labels]` -- MPM CELLS SEEDED INSIDE SEGMENTED NUCLEI OF A RECORDED STACK (exp35, 2026-10-30).
# Cedric: "seed MPM in the nuclei, do not disperse yet" -- the first step of seeding a Plexus tissue from the Wang lab's
# salivary-gland movies. One live cell per label of a 3D label image (Cellpose on the denoised nuclear channel); its
# `live_per_cell` points sit on randomly drawn voxels of the label, one voxel's worth of jitter, so the cell IS the nucleus
# at the seed, dense; the later dispersion (growth bounded by the membrane image and the neighbours) is another operator.
# ==========================================================================================================
@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="from_labels",
                   title="seed one MPM cell per segmented nucleus of a recorded image stack",
                   equation=r"""$$\mathbf x_i=\mathbf o+\frac{\mathbf v_i\odot\mathbf s-\mathbf m}{u}+\boldsymbol\epsilon_i,\ \ \mathbf v_i\sim\{\mathbf v:\ L(\mathbf v)=k\},\quad \mathbf c_k=\overline{\mathbf x}_{i\in k},\quad V_i=\frac{|L=k|\,\prod\mathbf s}{u^3 N_k}$$""")
class SeedFromLabels(Seed):
    r"""every live cell is one labelled object of a 3D image (a segmented nucleus): its points are drawn from the object's voxels, so the cell starts dense inside the nucleus, its volume the object's.
    pt -> pt, cell: `pid`, `cellc`, `face` (all lateral), `tcol` (when present) on the points; `pos` on the cells.

        \mathbf x_i=\mathbf o+\frac{\mathbf v_i\odot\mathbf s-\mathbf m}{u}+\boldsymbol\epsilon_i,\ \ \mathbf v_i\sim\{\mathbf v:\ L(\mathbf v)=k\},\quad \mathbf c_k=\overline{\mathbf x}_{i\in k},\quad V_i=\frac{|L=k|\,\prod\mathbf s}{u^3 N_k}
        x_i            the position of point i, world length
        o              `centre`: the world point the stack's centre maps to (default the box centre)
        v_i            a voxel index (x, y, z) drawn uniformly, with replacement, from the voxels of label k
        s              `voxel_um`: the voxel size in micrometres (x, y, z)
        m              the stack's centre in micrometres
        u              `um_per_unit`: micrometres per world length unit
        \epsilon_i     a uniform jitter inside the voxel
        L              the label image `labels` (a tif, axes z y x; 0 = background)
        c_k            the centre of cell k, the mean of its points
        V_i            the material volume per point: the object's volume shared by its N_k = `live_per_cell` points

    Reference: Plexus exp35 (this work, 2026-10-30); the labels: Stringer, C. & Pachitariu, M. (2025). Cellpose3. Nat. Methods 22:592-599."""
    PARAM_UNITS = {'labels': 'path', 'voxel_um': 'length', 'um_per_unit': 'length', 'centre': 'length', 'live_per_cell': 'count', 'points_per_median': 'count', 'min_points': 'count', 'seed': 'count', 'min_volume_um3': 'volume', 'max_cells': 'count', 'crop_um': 'length', 'dilate_um': 'length', 'layers_csv': 'path', 'adh': 'dimensionless', 'spread_iters': 'count', 'min_obj_um3': 'volume', 'type_values': 'dimensionless'}
    SYMBOLS = {'x_i': 'the position of point i, world length', 'o': "`centre`: the world point the stack's centre maps to (default the box centre)",
               'v_i': 'a voxel index (x, y, z) drawn uniformly, with replacement, from the voxels of label k', 's': '`voxel_um`: the voxel size in micrometres (x, y, z)',
               'm': "the stack's centre in micrometres", 'u': '`um_per_unit`: micrometres per world length unit', '\\epsilon_i': 'a uniform jitter inside the voxel',
               'L': 'the label image `labels` (a tif, axes z y x; 0 = background)', 'c_k': 'the centre of cell k, the mean of its points',
               'V_i': "the material volume per point: the object's volume shared by its N_k = `live_per_cell` points"}
    REQUIRES_PARAMS = ["cell_set", "labels"]
    SUPPORTED_DIMS = [3]
    PARAM_ROLES = {"cell_set": "the_cell_set", "labels": "label_image_path_tif_zyx", "voxel_um": "voxel_size_um_xyz", "um_per_unit": "micrometres_per_world_unit",
                   "centre": "world_point_of_the_stack_centre", "live_per_cell": "points_seeded_per_cell_the_rest_of_the_block_dormant_or_the_cap_with_points_per_median",
                   "points_per_median": "points_of_a_median_volume_cell_one_point_volume_for_all", "min_points": "fewest_points_of_a_cell",
                   "min_volume_um3": "objects_smaller_than_this_are_skipped", "crop_um": "stack_box_um_x0_x1_y0_y1_z0_z1_of_the_centroids_kept", "dilate_um": "radius_um_the_object_is_dilated_by_before_the_points_are_drawn", "layers_csv": "table_label_layer_ordering_the_cells_by_the_sets_type_order", "adh": "adhesion_maximum_in_every_face_pair_class", "spread_iters": "lloyd_iterations_spreading_each_cells_points_evenly", "min_obj_um3": "objects_smaller_than_this_are_dilated_up_to_it", "type_values": "cell_block_to_one_value_per_cell_type", "max_cells": "at_most_this_many_objects_the_largest_first", "seed": "rng_seed"}
    REFERENCE = "Plexus exp35 (this work, 2026-10-30); Stringer, C. & Pachitariu, M. (2025). Nat. Methods 22:592-599."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.cell_set = str(params["cell_set"])
        self.labels = str(params["labels"])
        self.voxel_um = np.array([float(v) for v in params.get("voxel_um", [0.7598, 0.7598, 1.0])])   # x, y, z
        self.u = float(params.get("um_per_unit", 10.0))
        self.centre = None if params.get("centre") is None else np.array([float(v) for v in params["centre"]])
        self.live_per_cell = int(params.get("live_per_cell", 125))
        # `points_per_median` (exp35, 2026-11-04): ONE POINT VOLUME FOR EVERY CELL. A cell of volume V gets N = clip(round(V / v0), min_points, its block's capacity) points, v0 = (the median object volume) /
        # `points_per_median`; with it, `live_per_cell` is no longer the count but the CAP (default: the block's own size, `pt.per_parent`). 0 = the old rule, `live_per_cell` points in every cell.
        self.points_per_median = int(params.get("points_per_median", 0))
        self._given = set(params)
        self.min_points = int(params.get("min_points", 20))
        self.min_vol = float(params.get("min_volume_um3", 0.0))
        self.max_cells = params.get("max_cells")
        # `crop_um: [x0, x1, y0, y1, z0, z1]` (exp35, 2026-10-31): only the objects whose centroid lies in this box of the stack (um) are seeded,
        # so a search can run on one lobe (the tape of the differentiable rig grows with the points). None = the whole stack.
        self.crop_um = None if params.get("crop_um") is None else [float(v) for v in params["crop_um"]]
        # `dilate_um` (exp35, 2026-10-31): the points are drawn from the object DILATED by this radius (um), so a cell starts as a ball round its
        # nucleus rather than the nucleus itself; the volume per point follows. 0 = the nucleus (125 points in 330 um^3 = 1.4 um apart, which the
        # differentiable MPM twin does not survive: F -> inf in the first window; the warp rig only compacts).
        self.dilate_um = float(params.get("dilate_um", 0.0))
        # `layers_csv` (exp35, 2026-10-31): a table `label, layer` (tools/exp35_layers.py: `outer` = a nucleus touching the tissue surface, `inner` otherwise).
        # The labels are then ordered by the cell set's type order (inner cells first when the spec declares `types: {inner: .., outer: ..}`), so the
        # cells of each type ARE the nuclei of that layer, and the per-type counts must match the spec's. None = no layers: one type, largest first.
        self.layers_csv = params.get("layers_csv")
        self.min_obj = float(params.get("min_obj_um3", 0.0))        # objects smaller than this are dilated up to it (0: as segmented)
        self.spread_iters = int(params.get("spread_iters", 0))      # Lloyd iterations spreading each cell's points evenly over its object (0: random draws, as before)
        # `type_values: {<cell block>: {<type>: value}}` (exp35, 2026-11-03): a width-1 cell block written with one value per cell TYPE, e.g. grow: {inner: 0.0029, outer: 0.0}
        # (only the inner nuclei grow; cell_grow[mpm] reads it with `rate_block: grow`)
        self.type_values = params.get("type_values") or {}
        self.adh = float(params.get("adh", 1.0))                    # the cells' adhesion maximum in every face-pair class when the set carries an `adh` block (all lateral: no polarity here)
        self.seed = int(params.get("seed", 0))

    def forward(self, H, mask=None):
        import tifffile
        from scipy import ndimage as ndi
        from scipy.spatial import cKDTree
        P, C = H.level(self.at), H.level(self.cell_set)
        rng = np.random.default_rng(self.seed)
        lab = tifffile.imread(self.labels)
        if lab.ndim != 3:
            raise ValueError(f"seed_positions[from_labels]: {self.labels} is {lab.ndim}D, a z y x label stack is needed")
        ids = np.unique(lab); ids = ids[ids > 0]
        vox_um3 = float(np.prod(self.voxel_um))
        counts = ndi.sum_labels(np.ones(lab.shape, np.float32), lab, ids)
        keep = counts * vox_um3 >= self.min_vol
        ids, counts = ids[keep], counts[keep]
        if self.crop_um is not None:
            com = np.array(ndi.center_of_mass(np.ones(lab.shape, bool), lab, ids))[:, ::-1] * self.voxel_um     # x y z, um
            b = self.crop_um
            inside = (com[:, 0] >= b[0]) & (com[:, 0] < b[1]) & (com[:, 1] >= b[2]) & (com[:, 1] < b[3]) & (com[:, 2] >= b[4]) & (com[:, 2] < b[5])
            ids, counts = ids[inside], counts[inside]
        if self.layers_csv:
            import csv as _csv
            lay = {int(r["label"]): str(r["layer"]) for r in _csv.DictReader(open(self.layers_csv))}
            names = list(getattr(C, "type_names", []) or [])
            tix = np.array([names.index(lay.get(int(k), names[0])) if lay.get(int(k), names[0]) in names else 0 for k in ids])
            order = np.lexsort((-counts, tix))                                      # by type order, then the largest first
        else:
            tix = None
            order = np.argsort(-counts)
        if self.max_cells is not None:
            order = order[: int(self.max_cells)]
        ids, counts = ids[order], counts[order]
        live_all = np.flatnonzero(C.occ.detach().cpu().numpy() > 0.5)
        if tix is not None:
            nt_live = C.node_type.detach().cpu().numpy()[live_all]
            want = np.bincount(tix[order], minlength=len(names)); have = np.bincount(nt_live, minlength=len(names))
            if not np.array_equal(want[: len(names)], have[: len(names)]):
                raise ValueError(f"seed_positions[from_labels]: the layers need cells per type {dict(zip(names, want.tolist()))} but the set declares {dict(zip(names, have.tolist()))}; "
                                 f"set `sets.{self.cell_set}.types` counts to the layer counts")
        if len(live_all) < len(ids):
            raise ValueError(f"seed_positions[from_labels]: the cell set has {len(live_all)} live cells but {self.labels} holds {len(ids)} objects "
                             f"(after min_volume_um3 {self.min_vol:g} and the crop); set `sets.{self.cell_set}.n` to at least {len(ids)}")
        live = live_all[: len(ids)]
        if len(live_all) > len(ids):                                              # the spare cells are switched off: a spec may declare more cells than the image holds
            cocc = C.occ.clone(); cocc[torch.as_tensor(live_all[len(ids):], device=cocc.device)] = 0.0; C.occ = cocc
        shape_zyx = np.array(lab.shape, dtype=np.float64)
        m_um = 0.5 * shape_zyx[::-1] * self.voxel_um                               # the stack's centre, x y z, um
        D = 3
        o = (0.5 * np.asarray(H.world_size, dtype=np.float64)[:D]) if self.centre is None else self.centre
        objs = ndi.find_objects(lab)
        parent = P.parent.detach().cpu().numpy()
        dev, dt_ = P.state.device, P.state.dtype
        nC = C.state.shape[0]
        occ = torch.zeros_like(P.occ)
        cen = np.zeros((nC, 3))
        nt = C.node_type.detach().cpu().numpy()
        def _object(k, sl0, r_um):
            """The voxels of object k dilated by r_um (um): (z, y, x indices, box slice, voxel count)."""
            if r_um <= 0:
                zz_, yy_, xx_ = np.nonzero(lab[sl0] == k)
                return zz_, yy_, xx_, sl0, float(len(zz_))
            rv = np.round(r_um / self.voxel_um[::-1]).astype(int)                  # z y x
            sl_ = tuple(slice(max(0, s_.start - r_), min(n_, s_.stop + r_)) for s_, r_, n_ in zip(sl0, rv, lab.shape))
            zo, yo, xo = np.ogrid[-rv[0]:rv[0] + 1, -rv[1]:rv[1] + 1, -rv[2]:rv[2] + 1]
            ball = ((zo * self.voxel_um[2]) ** 2 + (yo * self.voxel_um[1]) ** 2 + (xo * self.voxel_um[0]) ** 2) <= r_um ** 2
            m_ = ndi.binary_dilation(lab[sl_] == k, structure=ball)
            zz_, yy_, xx_ = np.nonzero(m_)
            return zz_, yy_, xx_, sl_, float(m_.sum())
        n_grown = 0; n_capped = 0; n_pts = []
        v0 = float(np.median(counts) * vox_um3) / max(self.points_per_median, 1)       # the volume of one point, um^3 (points_per_median > 0)
        for c, k, nv in zip(live, ids, counts):
            sl0 = objs[int(k) - 1]
            r_um = self.dilate_um
            zz, yy, xx, sl, nv = _object(k, sl0, r_um)
            # `min_obj_um3` (exp35, 2026-10-31): the rig seeds the same number of points in every nucleus, so a small segmented object (a quarter of the crop is under 212 um^3, a tenth under
            # 104) gets points two to three times closer than a normal one (the median 327 um^3), the MPM sees a density peak and the strain update inverts them; a small object is dilated
            # in 0.75 um steps (at most 5 um) until it reaches this volume
            if self.min_obj > 0:
                grown = False
                while nv * vox_um3 < self.min_obj and r_um < 5.0:
                    r_um = max(r_um, 0.0) + 0.75; grown = True
                    zz, yy, xx, sl, nv = _object(k, sl0, r_um)
                n_grown += int(grown)
            vox = np.stack([xx + sl[2].start, yy + sl[1].start, zz + sl[0].start], 1).astype(np.float64)   # x y z voxel indices
            blk_all = np.flatnonzero(parent == c)
            if self.points_per_median > 0:
                n_c = int(np.clip(round(nv * vox_um3 / v0), self.min_points, min(len(blk_all), self.live_per_cell if "live_per_cell" in self._given else len(blk_all))))
                n_capped += int(nv * vox_um3 / v0 > n_c + 0.5)
            else:
                n_c = self.live_per_cell
            blk = blk_all[:n_c]
            n_pts.append(len(blk))
            if self.spread_iters > 0 and len(vox) > len(blk):
                # EVENLY SPREAD POINTS (exp35, 2026-10-31): points drawn at random from a nucleus's voxels clump, the MPM sees density peaks and already at the first substep some points
                # have det F 0.14 (a lattice-seeded ball: >= 0.9); `spread_iters` Lloyd iterations of k-means (k = points of the cell) on the object's voxels give a blue-noise set
                Xv = (vox + 0.5) * self.voxel_um
                cen_ = Xv[rng.choice(len(Xv), len(blk), replace=False)].copy()
                for _ in range(self.spread_iters):
                    asg = cKDTree(cen_).query(Xv)[1]
                    cnt = np.bincount(asg, minlength=len(blk)).astype(float)
                    new = np.stack([np.bincount(asg, weights=Xv[:, d], minlength=len(blk)) for d in range(3)], 1)
                    okc = cnt > 0
                    cen_[okc] = new[okc] / cnt[okc, None]
                pos = o + (cen_ - m_um) / self.u
            else:
                pick = vox[rng.integers(0, len(vox), len(blk))] + rng.uniform(0.0, 1.0, (len(blk), 3))
                pos = o + (pick * self.voxel_um - m_um) / self.u
            vol = float(nv) * vox_um3 / self.u ** 3
            idx = torch.as_tensor(blk, device=dev)
            _mpm_reset_points(P, idx, torch.as_tensor(pos, device=dev, dtype=dt_), torch.full((len(blk),), vol / len(blk), device=dev, dtype=dt_))
            _set_block(P, "pid", idx, float(c)); _set_block(P, "cellc", idx, float(c % 20))
            if "tcol" in P.state_schema:
                _set_block(P, "tcol", idx, float(nt[c]))
            if "face" in P.state_schema:
                _set_block(P, "face", idx, float(FACE_L))
            occ[idx] = 1.0
            cen[c] = pos.mean(0)
        P.occ = occ
        ca, cb = C.state_schema["pos"]
        st = C.state.clone(); st[:, ca:cb] = torch.as_tensor(cen, dtype=st.dtype, device=st.device)
        if "adh" in C.state_schema:
            a0, a1 = C.state_schema["adh"]; st[:, a0:a1] = self.adh
        if self.type_values:
            tnames = list(getattr(C, "type_names", []) or [])
            ntc = C.node_type.detach().cpu().numpy()
            for blk_, per_type in self.type_values.items():
                b0, b1 = C.state_schema[str(blk_)]
                for tn, val in per_type.items():
                    if tn not in tnames:
                        raise ValueError(f"seed_positions[from_labels]: type_values names the type {tn!r}, the set has {tnames}")
                    rows = live[ntc[live] == tnames.index(tn)]
                    st[torch.as_tensor(rows, device=st.device), b0:b1] = float(val)
        C.state = st
        ext = (np.array(lab.shape)[::-1] * self.voxel_um / self.u)
        print(f"[seed_positions:from_labels] {len(live)} cells from {os.path.basename(self.labels)} ({lab.shape[0]}x{lab.shape[1]}x{lab.shape[2]} voxels = "
              f"{ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} world units), {int(occ.sum()):,} points; object volume median {np.median(counts) * vox_um3:.0f} um^3, "
              f"{(str(self.live_per_cell) + ' points per cell') if self.points_per_median <= 0 else ('points per cell ' + str(int(min(n_pts))) + ' / ' + str(int(np.median(n_pts))) + ' / ' + str(int(max(n_pts))) + ' (min / median / max) at ' + format(v0, '.2f') + ' um^3 per point, ' + str(n_capped) + ' cells capped')}; {n_grown} small objects dilated to >= {self.min_obj:g} um^3", flush=True)
        return {}


# ==========================================================================================================
# `seed_positions[from_points]` -- MPM CELLS SEEDED AT GIVEN POINTS (exp35, 2026-11-05): the cells of a time point of the recording, each with its points
# already drawn (points250/tNNN.npz), optionally moved (the exploded view).
# ==========================================================================================================
@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="from_points",
                   title="seed MPM cells at the points of a table, one value per cell on a cell block",
                   equation=r"""$$\mathbf x_{k,j}=\mathbf o+\frac{\mathbf p_{k,j}-\mathbf m}{u},\quad V_{k,j}=\frac{V_k}{u^3 N},\quad q_k=\mathrm{value}_k$$""")
class SeedFromPoints(Seed):
    r"""cell k of the table `points` (an npz: `pos` (cells, points, 3) in micrometres, axes x y z; `volume_um3` (cells,); optional `value` (cells,)) is the k-th live cell, its N points
    at p_{k,j}; the optional value goes into the width-1 cell block `value_block`.

        \mathbf x_{k,j}=\mathbf o+\frac{\mathbf p_{k,j}-\mathbf m}{u},\quad V_{k,j}=\frac{V_k}{u^3 N},\quad q_k=\mathrm{value}_k
        x_{k,j}        the world position of point j of cell k, world length
        o              `centre`: the world point the micrometre point m maps to (default the box centre)
        p_{k,j}        the table's position, micrometres, x y z order
        m              `centre_um`: the micrometre point (x y z) placed at o
        u              `um_per_unit`: micrometres per world length unit
        V_k            the cell's volume `volume_um3`, micrometres cubed, shared by its N points
        q_k            the cell's value, written to the cell block `value_block`

    Reference: Plexus exp35 (this work, 2026-11-05)."""
    PARAM_UNITS = {'points': 'path', 'um_per_unit': 'length', 'centre': 'length', 'centre_um': 'length'}
    SYMBOLS = {'x_{k,j}': 'the world position of point j of cell k, world length', 'o': "`centre`: the world point the micrometre point m maps to (default the box centre)",
               'p_{k,j}': "the table's position, micrometres, x y z order", 'm': '`centre_um`: the micrometre point (x y z) placed at o',
               'u': '`um_per_unit`: micrometres per world length unit', 'V_k': "the cell's volume `volume_um3`, micrometres cubed, shared by its N points",
               'q_k': "the cell's value, written to the cell block `value_block`"}
    REQUIRES_PARAMS = ["cell_set", "points"]
    SUPPORTED_DIMS = [3]
    PARAM_ROLES = {"cell_set": "the_cell_set", "points": "npz_pos_volume_um3_value", "um_per_unit": "micrometres_per_world_unit", "centre": "world_point_of_centre_um",
                   "centre_um": "micrometre_point_xyz_placed_at_centre", "value_block": "width_1_cell_block_that_gets_the_value"}
    REFERENCE = "Plexus exp35 (this work, 2026-11-05)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.cell_set = str(params["cell_set"])
        self.points = str(params["points"])
        self.u = float(params.get("um_per_unit", 10.0))
        self.centre = None if params.get("centre") is None else np.array([float(v) for v in params["centre"]])
        self.centre_um = np.array([float(v) for v in params.get("centre_um", [0.0, 0.0, 0.0])])
        self.value_block = params.get("value_block")

    def forward(self, H, mask=None):
        P, C = H.level(self.at), H.level(self.cell_set)
        z = np.load(self.points); pos = z["pos"].astype(np.float64); vol = z["volume_um3"].astype(np.float64)       # (cells, N, 3) x y z, um
        nk, npt = pos.shape[:2]
        live_all = np.flatnonzero(C.occ.detach().cpu().numpy() > 0.5)
        if len(live_all) < nk:
            raise ValueError(f"seed_positions[from_points]: the cell set has {len(live_all)} live cells but {self.points} holds {nk}; set `sets.{self.cell_set}.n` to at least {nk}")
        live = live_all[:nk]
        if len(live_all) > nk:
            cocc = C.occ.clone(); cocc[torch.as_tensor(live_all[nk:], device=cocc.device)] = 0.0; C.occ = cocc
        o = (0.5 * np.asarray(H.world_size, dtype=np.float64)[:3]) if self.centre is None else self.centre
        parent = P.parent.detach().cpu().numpy(); dev, dt_ = P.state.device, P.state.dtype
        occ = torch.zeros_like(P.occ); cen = np.zeros((C.state.shape[0], 3)); nt = C.node_type.detach().cpu().numpy()
        order = np.argsort(parent, kind="stable"); first = np.searchsorted(parent[order], np.arange(C.state.shape[0] + 1))      # the points of cell c: order[first[c]:first[c + 1]]
        for k, c in enumerate(live):
            blk = order[first[c]:first[c + 1]][:npt]
            x = o + (pos[k, :len(blk)] - self.centre_um) / self.u
            idx = torch.as_tensor(blk, device=dev)
            _mpm_reset_points(P, idx, torch.as_tensor(x, device=dev, dtype=dt_), torch.full((len(blk),), vol[k] / self.u ** 3 / len(blk), device=dev, dtype=dt_))
            _set_block(P, "pid", idx, float(c)); _set_block(P, "cellc", idx, float(c % 20))
            if "tcol" in P.state_schema:
                _set_block(P, "tcol", idx, float(nt[c]))
            if "face" in P.state_schema:
                _set_block(P, "face", idx, float(FACE_L))
            occ[idx] = 1.0; cen[c] = x.mean(0)
        P.occ = occ
        ca, cb = C.state_schema["pos"]
        st = C.state.clone(); st[:, ca:cb] = torch.as_tensor(cen, dtype=st.dtype, device=st.device)
        if self.value_block and "value" in z.files:
            b0, b1 = C.state_schema[str(self.value_block)]
            st[torch.as_tensor(live, device=st.device), b0:b1] = torch.as_tensor(z["value"].astype(np.float64), dtype=st.dtype, device=st.device)[:, None]
        C.state = st
        print(f"[seed_positions:from_points] {nk} cells x {npt} points from {os.path.basename(self.points)}", flush=True)
        return {}


# ==========================================================================================================
# `seed_positions[from_points_ragged]` -- BODIES OF DIFFERENT POINT COUNTS IN ONE SET (exp09, 2026-11-14): a molecular landscape holds bodies of 27 to 343
# points (a 7 nm enzyme, a 19 nm ribosome, at one point per 2 nm), and `from_points` gives every cell the table's one point count. Here the table is padded to
# the largest body with NaN rows; each cell takes its finite rows only and the rest of its slots stay empty (occ 0), exactly as the reserve slots of a growing
# cell are. Registered beside from_points and not as an edit of it: the same table and contract, one more freedom; a table with no NaN row seeds what
# from_points seeds (tests/test_exp09_landscape_ops.py).
# ==========================================================================================================
@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="from_points_ragged",
                   title="seed MPM bodies of different point counts at the points of a table",
                   equation=r"""$$\mathbf x_{k,j}=\mathbf o+\frac{\mathbf p_{k,j}-\mathbf m}{u},\quad V_{k,j}=\frac{V_k}{u^3 N_k},\qquad j<N_k$$""")
class SeedFromPointsRagged(SeedFromPoints):
    r"""seed_positions[from_points] for bodies that do not all have the same number of points: cell k takes the N_k rows of its table entry that are finite (the table is padded with NaN rows to the largest body) and leaves its other point slots empty.
    pt -> pt, cell: as from_points; the point set's `per_parent` must be at least the largest N_k.

        \mathbf x_{k,j}=\mathbf o+\frac{\mathbf p_{k,j}-\mathbf m}{u},\quad V_{k,j}=\frac{V_k}{u^3 N_k},\qquad j<N_k
        x_{k,j}        the world position of point j of cell k, world length
        N_k            the number of finite rows of cell k in the table: its point count
        o              `centre`: the world point the micrometre point m maps to (default the box centre)
        p_{k,j}        the table's position, micrometres, x y z order (NaN in the rows a body does not use)
        m              `centre_um`: the micrometre point (x y z) placed at o
        u              `um_per_unit`: micrometres per world length unit
        V_k            the cell's volume `volume_um3`, micrometres cubed, shared by its N_k points

    Reference: Plexus exp09 (this work, 2026-11-14); seed_positions[from_points] (exp35)."""
    SYMBOLS = dict(SeedFromPoints.SYMBOLS, **{'N_k': 'the number of finite rows of cell k in the table: its point count',
                                              'V_k': "the cell's volume `volume_um3`, micrometres cubed, shared by its N_k points"})
    REFERENCE = "Plexus exp09 (this work, 2026-11-14)."

    def forward(self, H, mask=None):
        P, C = H.level(self.at), H.level(self.cell_set)
        z = np.load(self.points); pos = z["pos"].astype(np.float64); vol = z["volume_um3"].astype(np.float64)       # (cells, N_max, 3) x y z, um; NaN rows = unused
        nk = pos.shape[0]
        live_all = np.flatnonzero(C.occ.detach().cpu().numpy() > 0.5)
        if len(live_all) < nk:
            raise ValueError(f"seed_positions[from_points_ragged]: the cell set has {len(live_all)} live cells but {self.points} holds {nk}; set `sets.{self.cell_set}.n` to at least {nk}")
        live = live_all[:nk]
        if len(live_all) > nk:
            cocc = C.occ.clone(); cocc[torch.as_tensor(live_all[nk:], device=cocc.device)] = 0.0; C.occ = cocc
        o = (0.5 * np.asarray(H.world_size, dtype=np.float64)[:3]) if self.centre is None else self.centre
        parent = P.parent.detach().cpu().numpy(); dev, dt_ = P.state.device, P.state.dtype
        occ = torch.zeros_like(P.occ); cen = np.zeros((C.state.shape[0], 3)); nt = C.node_type.detach().cpu().numpy()
        order = np.argsort(parent, kind="stable"); first = np.searchsorted(parent[order], np.arange(C.state.shape[0] + 1))      # the points of cell c: order[first[c]:first[c + 1]]
        counts = []
        for k, c in enumerate(live):
            pk = pos[k][np.isfinite(pos[k]).all(1)]                                                                          # the body's own points, in table order
            slots = order[first[c]:first[c + 1]]
            if len(pk) > len(slots):
                raise ValueError(f"seed_positions[from_points_ragged]: body {k} has {len(pk)} points but a cell of `{self.at}` has {len(slots)} slots; raise `sets.{self.at}.per_parent`")
            if len(pk) == 0:
                raise ValueError(f"seed_positions[from_points_ragged]: body {k} of {self.points} has no finite point")
            blk = slots[:len(pk)]
            x = o + (pk - self.centre_um) / self.u
            idx = torch.as_tensor(blk, device=dev)
            _mpm_reset_points(P, idx, torch.as_tensor(x, device=dev, dtype=dt_), torch.full((len(blk),), vol[k] / self.u ** 3 / len(blk), device=dev, dtype=dt_))
            _set_block(P, "pid", idx, float(c)); _set_block(P, "cellc", idx, float(c % 20))
            if "tcol" in P.state_schema:
                _set_block(P, "tcol", idx, float(nt[c]))
            if "face" in P.state_schema:
                _set_block(P, "face", idx, float(FACE_L))
            occ[idx] = 1.0; cen[c] = x.mean(0); counts.append(len(blk))
        P.occ = occ
        ca, cb = C.state_schema["pos"]
        st = C.state.clone(); st[:, ca:cb] = torch.as_tensor(cen, dtype=st.dtype, device=st.device)
        if self.value_block and "value" in z.files:
            b0, b1 = C.state_schema[str(self.value_block)]
            st[torch.as_tensor(live, device=st.device), b0:b1] = torch.as_tensor(z["value"].astype(np.float64), dtype=st.dtype, device=st.device)[:, None]
        # `cell__<block>` in the table, [cells] or [cells, width]: written into the cell block of that name when the cell set declares it -- the per-body state
        # an operator reads at the first frame (adhesion maxima `adh`, a chemical's starting level, a drive); a [cells] array fills every column of the block
        for key in z.files:
            if key.startswith("cell__") and key[6:] in C.state_schema:
                b0, b1 = C.state_schema[key[6:]]
                v = torch.as_tensor(z[key].astype(np.float64), dtype=st.dtype, device=st.device).reshape(nk, -1)
                if v.shape[1] not in (1, b1 - b0):
                    raise ValueError(f"seed_positions[from_points_ragged]: `{key}` has {v.shape[1]} columns, the cell block `{key[6:]}` has {b1 - b0}")
                st[torch.as_tensor(live, device=st.device), b0:b1] = v.expand(nk, b1 - b0)
        C.state = st
        print(f"[seed_positions:from_points_ragged] {nk} bodies of {min(counts)} to {max(counts)} points ({int(sum(counts)):,} points in all) from {os.path.basename(self.points)}", flush=True)
        return {}


@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="from_points_ragged_batched",
                   title="seed MPM bodies of different point counts at the points of a table, all at once",
                   equation=r"""$$\mathbf x_{k,j}=\mathbf o+\frac{\mathbf p_{k,j}-\mathbf m}{u},\quad V_{k,j}=\frac{V_k}{u^3 N_k},\qquad j<N_k$$""")
class SeedFromPointsRaggedBatched(SeedFromPointsRagged):
    r"""seed_positions[from_points_ragged_batched]: seed_positions[from_points_ragged] writing every body in one call per block instead of one body at a time: the same slots, positions, volumes, masses and
    labels (tests/test_exp09_landscape_ops.py: the state equal to the default's), in seconds instead of hours at 10 million bodies.
    pt -> pt, cell: as from_points_ragged.

        \mathbf x_{k,j}=\mathbf o+\frac{\mathbf p_{k,j}-\mathbf m}{u},\quad V_{k,j}=\frac{V_k}{u^3 N_k},\qquad j<N_k
        x_{k,j}        the world position of point j of cell k, world length
        N_k            the number of finite rows of cell k in the table: its point count
        o              `centre`: the world point the micrometre point m maps to (default the box centre)
        p_{k,j}        the table's position, micrometres, x y z order (NaN in the rows a body does not use)
        m              `centre_um`: the micrometre point (x y z) placed at o
        u              `um_per_unit`: micrometres per world length unit
        V_k            the cell's volume `volume_um3`, micrometres cubed, shared by its N_k points

    Reference: as from_points_ragged; Plexus exp09 (this work, 2026-11-17)."""

    def forward(self, H, mask=None):
        P, C = H.level(self.at), H.level(self.cell_set)
        z = np.load(self.points); pos = z["pos"].astype(np.float64); vol = z["volume_um3"].astype(np.float64)       # (cells, N_max, 3) x y z, um; NaN rows = unused
        nk = pos.shape[0]
        live_all = np.flatnonzero(C.occ.detach().cpu().numpy() > 0.5)
        if len(live_all) < nk:
            raise ValueError(f"seed_positions[from_points_ragged]: the cell set has {len(live_all)} live cells but {self.points} holds {nk}; set `sets.{self.cell_set}.n` to at least {nk}")
        live = live_all[:nk]
        if len(live_all) > nk:
            cocc = C.occ.clone(); cocc[torch.as_tensor(live_all[nk:], device=cocc.device)] = 0.0; C.occ = cocc
        o = (0.5 * np.asarray(H.world_size, dtype=np.float64)[:3]) if self.centre is None else self.centre
        parent = P.parent.detach().cpu().numpy(); dev, dt_ = P.state.device, P.state.dtype
        occ = torch.zeros_like(P.occ); cen = np.zeros((C.state.shape[0], 3)); nt = C.node_type.detach().cpu().numpy()
        order = np.argsort(parent, kind="stable"); first = np.searchsorted(parent[order], np.arange(C.state.shape[0] + 1))      # the points of cell c: order[first[c]:first[c + 1]]
        # ALL BODIES AT ONCE: the slots, positions, volumes and labels of every body gathered in numpy and written in one call each -- the loop above writes the
        # same values body by body, but every call clones the whole point state (_mpm_reset_points, _set_block), 2.8 GB at 70 million points: about 14 hours for
        # 10 million bodies (exp09 b10m, 2026-11-17)
        fin = np.isfinite(pos).all(2)                                                     # [cells, N_max]: the rows a body uses, in table order
        npk = fin.sum(1).astype(np.int64)
        if (npk == 0).any():
            raise ValueError(f"seed_positions[from_points_ragged]: body {int(np.flatnonzero(npk == 0)[0])} of {self.points} has no finite point")
        have = first[live + 1] - first[live]
        if (npk > have).any():
            k = int(np.flatnonzero(npk > have)[0])
            raise ValueError(f"seed_positions[from_points_ragged]: body {k} has {int(npk[k])} points but a cell of `{self.at}` has {int(have[k])} slots; raise `sets.{self.at}.per_parent`")
        kk = np.repeat(np.arange(nk), npk)                                               # the body of every seeded point, bodies in table order
        jj = np.arange(len(kk)) - np.repeat(np.cumsum(npk) - npk, npk)                   # its rank among its body's points
        blk_all = order[first[live][kk] + jj]                                            # its slot: the j-th slot of its cell, as the loop takes them
        x_all = o + (pos[fin] - self.centre_um) / self.u
        idx = torch.as_tensor(blk_all, device=dev)
        _mpm_reset_points(P, idx, torch.as_tensor(x_all, device=dev, dtype=dt_), torch.as_tensor(vol[kk] / self.u ** 3 / npk[kk], device=dev, dtype=dt_))
        cells = live[kk].astype(np.float64)
        _set_block(P, "pid", idx, torch.as_tensor(cells, device=dev, dtype=dt_)); _set_block(P, "cellc", idx, torch.as_tensor(cells % 20, device=dev, dtype=dt_))
        if "tcol" in P.state_schema:
            _set_block(P, "tcol", idx, torch.as_tensor(nt[live[kk]].astype(np.float64), device=dev, dtype=dt_))
        if "face" in P.state_schema:
            _set_block(P, "face", idx, float(FACE_L))
        occ[idx] = 1.0
        sums = np.zeros((nk, 3))
        np.add.at(sums, kk, x_all)
        cen[live] = sums / npk[:, None]
        counts = npk.tolist()
        P.occ = occ
        ca, cb = C.state_schema["pos"]
        st = C.state.clone(); st[:, ca:cb] = torch.as_tensor(cen, dtype=st.dtype, device=st.device)
        if self.value_block and "value" in z.files:
            b0, b1 = C.state_schema[str(self.value_block)]
            st[torch.as_tensor(live, device=st.device), b0:b1] = torch.as_tensor(z["value"].astype(np.float64), dtype=st.dtype, device=st.device)[:, None]
        # `cell__<block>` in the table, [cells] or [cells, width]: written into the cell block of that name when the cell set declares it -- the per-body state
        # an operator reads at the first frame (adhesion maxima `adh`, a chemical's starting level, a drive); a [cells] array fills every column of the block
        for key in z.files:
            if key.startswith("cell__") and key[6:] in C.state_schema:
                b0, b1 = C.state_schema[key[6:]]
                v = torch.as_tensor(z[key].astype(np.float64), dtype=st.dtype, device=st.device).reshape(nk, -1)
                if v.shape[1] not in (1, b1 - b0):
                    raise ValueError(f"seed_positions[from_points_ragged]: `{key}` has {v.shape[1]} columns, the cell block `{key[6:]}` has {b1 - b0}")
                st[torch.as_tensor(live, device=st.device), b0:b1] = v.expand(nk, b1 - b0)
        C.state = st
        print(f"[seed_positions:from_points_ragged] {nk} bodies of {min(counts)} to {max(counts)} points ({int(sum(counts)):,} points in all) from {os.path.basename(self.points)}", flush=True)
        return {}


# ==========================================================================================================
# `mpm_scatter` and `mpm_gather` WITH THE TINY MATRIX PRODUCTS AS BROADCASTS (exp27 speed audit, 2026-10-21).
# The default bodies write two products as batched matmuls over every (point, stencil node) pair: 3x3 @ 3x1 in the scatter
# (P x 27 = 8.6M batches at 320k points) and 3x1 @ 1x3 in the gather. A batch that small is launch-bound in the matmul library
# (10.6 of 20.7 ms per scatter call and 13.7 of 19.1 ms per gather call, RTX A6000, 320k points). Same equations, same
# grids and per-cell blocks; only the two product lines differ, behind the hooks `_affine_dpos` and `_outer_sum` of `MPMWrites`.
# ==========================================================================================================
from plexus.operators.mpm_ops import MPMGather, MPMScatter  # noqa: E402


# ==========================================================================================================
# `cell_populate[mpm_points_typed_own]` -- NEW POINTS SEATED INSIDE THEIR OWN CELL (exp27 course correction, 2026-10-21).
# `seat: fill` puts each new point at the emptiest of `tries` candidates one spacing off random points of its cell, inside the 90th-percentile radius
# about the centroid. The emptiest candidate is the farthest from the cell's own points, so it sits at the cell's edge, and in a confluent tissue the
# edge of a cell is inside a neighbour: in bm3c the share of cells with under 95 percent of their points in one piece rose 0 -> 0.2 -> 0.5 over the first
# 70 min (growth only, no division yet) and 94 percent of the stray points were nearer to another cell's centroid than to their own (Finding 16).
# Here a candidate counts only if the nearest live point of ANY cell is one of its own cell's; the emptiest of those is taken (a cell with none gets a
# point half a spacing off a random point of its own, as `fill` does when two picks coincide).
# ==========================================================================================================
@register_operator("cell_populate", model="mpm_points_typed_own", set="cell", kind="divide", family="population",
                   title="grow cells by adding points inside the cell, a rate per cell type", equation=r"""$$\frac{dN_c}{dt}=g_{\tau(c)}\,N_c,\qquad \text{mark}_c=\big[N_c\ge N^{\ast}_c\big],\quad N^{\ast}_c=N^{\ast}(1+j\,u_c),\ u_c\sim U(-1,1)$$""")
class CellPopulateMPMPointsTypedOwn(CellPopulateMPMPointsTyped):
    r"""cell_populate[mpm_points_typed] with `seat: fill` restricted to candidates whose nearest live point belongs to the growing cell.
    cell -> cell and its point set: as mpm_points_typed.

        \frac{dN_c}{dt}=g_{\tau(c)}\,N_c,\qquad \text{mark}_c=\big[N_c\ge N^{\ast}_c\big],\quad N^{\ast}_c=N^{\ast}(1+j\,u_c),\ u_c\sim U(-1,1)
        N_c            the point count of cell c
        g_{\tau(c)}    `rates`, the growth rate of the cell's type, per time (a type not named does not grow)
        mark_c         the division mark
        N^*_c          the cell's threshold
        N^*            `target`
        j              `target_jitter`
        u_c            uniform in (-1, 1), redrawn after a division

    Reference: as mpm_points_typed."""
    PARAM_UNITS = {}
    SYMBOLS = {'N_c': 'the point count of cell c', 'g_{\\tau(c)}': "`rates`, the growth rate of the cell's type, per time (a type not named does not grow)", 'mark_c': 'the division mark', 'N^*_c': "the cell's threshold", 'N^*': '`target`', 'j': '`target_jitter`', 'u_c': 'uniform in (-1, 1), redrawn after a division'}

    def _fill(self, P, st, new, own, occ, par, p0, p1, h, dev):
        from scipy.spatial import cKDTree
        X = P.state[:, p0:p1].detach().cpu().numpy().astype(np.float64)
        live = np.flatnonzero(occ)
        tree_all, owner_all = cKDTree(X[live]), par[live]
        new_np, own_np = new.cpu().numpy(), own.cpu().numpy()
        out = st[:, p0:p1].detach().cpu().numpy().astype(np.float64)
        for c in np.unique(own_np):
            mem = np.flatnonzero(occ & (par == c))
            Pc = X[mem]
            cen = Pc.mean(0)
            r90 = np.percentile(np.linalg.norm(Pc - cen, axis=1), 90)
            slots = new_np[own_np == c]
            m = len(slots)
            base = Pc[self._rng.integers(0, len(Pc), (m, self.tries))]
            u = self._rng.normal(size=(m, self.tries, 3))
            u /= np.linalg.norm(u, axis=2, keepdims=True)
            cand = base + h * u
            flat = cand.reshape(-1, 3)
            mine = (owner_all[tree_all.query(flat, k=1)[1]] == c).reshape(m, self.tries)
            d_own = cKDTree(Pc).query(flat, k=1)[0].reshape(m, self.tries)
            score = np.where(mine & (np.linalg.norm(cand - cen, axis=2) < r90), d_own, -1.0)
            pick = cand[np.arange(m), score.argmax(1)]
            none = score.max(1) < 0.0
            pick[none] = (base[:, 0] + 0.5 * h * u[:, 0])[none]
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


@register_operator("lumen_pressure", family="mechanics", set="particle", kind="exchange",
                   title="push each cell's inner face out to its share of the lumen volume",
                   equation=r"""$$v_c=\tfrac13\,r_c\,A_c,\qquad V^{0}_c=v^{\rm ref}_c\Big(\frac{N_c}{N^{\rm ref}_c}\Big)^{3/2},\qquad
\mathbf a_i=k_c\,\frac{V^{0}_c-v_c}{V^{0}_c}\,\hat{\mathbf r}_i\quad[i\ \text{apical}]$$""")
class LumenPressure(Exchange):
    r"""the sheet's volume law on a cell of points: each cell is pushed outward on its apical points by k times the relative deficit of its wedge volume, the cone from the vesicle's centre to its footprint, against a target that grows with its point count.
    particle -[containment]-> cell, once per frame beside active_force[footprint_spreading]: a frame-level acceleration every substep sees; differentiable in the positions and in k_c.

        v_c=\tfrac13\,r_c\,A_c,\qquad V^{0}_c=v^{\rm ref}_c\Big(\frac{N_c}{N^{\rm ref}_c}\Big)^{3/2},\qquad \mathbf a_i=k_c\,\frac{V^{0}_c-v_c}{V^{0}_c}\,\hat{\mathbf r}_i\quad[i\ \text{apical}]
        v_c            the cell's wedge volume r_c A_c / 3
        r_c            the cell's inner radius: the soft minimum (`softmin_beta`) of the apical points' distance from the centre
        A_c            its footprint
        V^0_c          the target wedge volume
        v^{ref}_c      the reference volume (block `vref_block`, captured after `capture_after` frames)
        N_c, N^{ref}_c the point count and its reference (`nref_block`)
        a_i            the acceleration of apical point i
        k_c            `k` (or per cell `k_block`), the pressure stiffness
        \hat r_i       the outward radial direction

    Reference: Okuda, S. et al. (2018). Sci. Rep. 8:2386 (the volume term); Chan, C. J. et al. (2019). Nature 571:112-116 (lumen pressure sets the size of a closed epithelium); Plexus exp32 (this work, 2026-10-25)."""
    PARAM_UNITS = {'capture_after': 'frames', 'k': 'acceleration', 'exponent': 'dimensionless', 'max_deficit': 'fraction', 'softmin_beta': 'dimensionless', 'target_scale': 'fraction', 'centre': 'length'}
    SYMBOLS = {'v_c': "the cell's wedge volume r_c A_c / 3", 'r_c': "the cell's inner radius: the soft minimum (`softmin_beta`) of the apical points' distance from the centre", 'A_c': 'its footprint', 'V^0_c': 'the target wedge volume', 'v^{ref}_c': 'the reference volume (block `vref_block`, captured after `capture_after` frames)', 'N_c, N^{ref}_c': 'the point count and its reference (`nref_block`)', 'a_i': 'the acceleration of apical point i', 'k_c': '`k` (or per cell `k_block`), the pressure stiffness', '\\hat r_i': 'the outward radial direction'}
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["cell_set", "centre"]
    MAY_MUTATE_INTEGRATED_STATE = True                         # writes the reference blocks vref / nref on the cell set
    PARAM_ROLES = {"cell_set": "the_cell_set", "centre": "vesicle_centre", "centre_block": "cell_block_with_each_cells_vesicle_centre",
                   "k": "outward_acceleration_at_unit_deficit", "k_block": "cell_block_with_each_cells_k",
                   "exponent": "power_of_the_point_count_in_the_target_wedge_volume", "max_deficit": "clamp_of_the_relative_deficit",
                   "face_block": "point_block_with_the_face_labels", "apical": "the_apical_label", "softmin_beta": "sharpness_of_the_inner_radius",
                   "target_scale": "factor_on_the_reference_wedge_volume",
                   "vref_block": "cell_block_with_the_reference_wedge_volume", "nref_block": "cell_block_with_the_reference_point_count", 'capture_after': 'frame_at_which_the_references_are_captured'}
    MECHANISM_TAGS = ["lumen_pressure", "volume_target", "sheet_constraint"]
    REFERENCE = "Okuda, S. et al. (2018). Sci. Rep. 8:2386; Chan, C. J. et al. (2019). Nature 571:112-116."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.cell_set = str(params["cell_set"])
        self.c = [float(v) for v in params["centre"]]
        self.centre_block = params.get("centre_block")
        self.k = float(params.get("k", 0.0))
        self.k_block = params.get("k_block")
        self.exponent = float(params.get("exponent", 1.5))
        self.target_scale = self.tunable(params.get("target_scale"), 1.0, tensor=True)  # 1 for a twin; a planted test's lever (x2: a deficit of 0.5 at once)
        self.max_deficit = float(params.get("max_deficit", 1.0))
        self.face_block = params.get("face_block", "face")
        self.apical = float(params.get("apical", FACE_A))
        self.beta = self.tunable(params.get("softmin_beta"), 20.0, tensor=True)
        self.vref_block = str(params.get("vref_block", "vref"))
        self.nref_block = str(params.get("nref_block", "nref"))
        self.capture_after = int(params.get("capture_after", 0))        # the frame from which the references are taken (the growth onset = the
        self.last = {}                                                  # warm-up's end on the exp32 rig), so the term is defined at k = 0 too

    def _clamp_deficit(self, d):
        """The relative deficit as the force reads it: two-sided, clamped to +-max_deficit (a cell beyond its target is pulled in)."""
        return d.clamp(-self.max_deficit, self.max_deficit)

    def forward(self, H, mask=None):
        P = H.level(self.at)
        X = P.get("pos")
        dev, dt_ = X.device, X.dtype
        C = H.level(self.cell_set)
        nC = int(C.state.shape[0])
        w = (P.occ > 0.5).to(dt_)
        par = P.parent.long().clamp(0, nC - 1)
        cnt = torch.zeros(nC, device=dev, dtype=dt_).index_add_(0, par, w)
        live_c = cnt > 0
        cen = torch.zeros(nC, 3, device=dev, dtype=dt_).index_add_(0, par, X * w[:, None]) / cnt.clamp(min=1.0)[:, None]
        if self.centre_block:
            b0, b1 = C.state_schema[self.centre_block]
            o = C.state[:, b0:b1].to(dt_)[par]
            n = cen - C.state[:, b0:b1].to(dt_)
        else:
            o = torch.tensor(self.c, device=dev, dtype=dt_)
            n = cen - o
        n = n / n.norm(dim=1, keepdim=True).clamp(min=1e-9)
        r = X - o
        rn = r.norm(dim=1).clamp(min=1e-9)
        if self.face_block and self.face_block in P.state_schema:
            inner = (P.get(self.face_block)[:, 0].round() == self.apical).to(dt_) * w
        else:
            inner = w
        # the inner radius per cell: a soft minimum over the apical points, exp(-beta (rn - m_c)) summed per cell about the cell's hard
        # minimum m_c (a constant shift: the log-sum-exp is exact and the exponent never overflows)
        m_c = torch.full((nC,), float("inf"), device=dev, dtype=dt_).scatter_reduce(0, par, torch.where(inner > 0, rn, torch.full_like(rn, float("inf"))), "amin", include_self=True)
        m_c = torch.where(torch.isfinite(m_c), m_c, torch.zeros_like(m_c))
        # THE EXPONENT IS CLAMPED TO <= 0 (exp33 R1, 2026-10-26: the first non-finite gradient of the long-horizon fits was HERE). A point of the cell that lies more than
        # 88.7 / beta (4.4 at beta 20) INSIDE the cell's apical minimum (a stray point in the lumen) has exp(+beta (m_c - rn)) = inf in float32, and inf x inner (0) = NaN:
        # the cell's inner radius was NaN in the forward, the cell dropped out of `ok` (so the loss stayed finite), and the backward met that NaN with a zero gradient
        # (0 x NaN) in `r_c * A` and sent it through every substep to every knob. For an APICAL point rn >= m_c by definition, so the clamp changes nothing there;
        # a non-apical point is multiplied by 0 whatever its exponent
        e = torch.exp(-self.beta * (rn - m_c[par]).clamp(min=0.0)) * inner
        se = torch.zeros(nC, device=dev, dtype=dt_).index_add_(0, par, e)
        has_in = se > 0
        r_c = m_c - torch.log(se.clamp(min=1e-30)) / self.beta
        r_c = torch.where(has_in, r_c, torch.zeros_like(r_c))              # a cell with no apical point has no inner radius (it is outside `ok`): a finite 0, not 3.45
        rr = X - cen[par]
        rt = rr - (rr * n[par]).sum(1, keepdim=True) * n[par]
        A = 2.0 * math.pi * torch.zeros(nC, device=dev, dtype=dt_).index_add_(0, par, (rt * rt).sum(1) * w) / cnt.clamp(min=1.0)
        v = r_c * A / 3.0
        if self.k_block:
            ka, _ = C.state_schema[self.k_block]
            k = C.state[:, ka].to(dt_)
        else:
            k = torch.full((nC,), self.k, device=dev, dtype=dt_)
        # the references: taken at the first call at or after `capture_after` (the growth onset: on the exp32 rig the END OF THE WARM-UP,
        # when the seed has relaxed -- a reference taken at frame 0 read every cell as over its target and the term PULLED the shell in,
        # the first GPU test of 2026-10-25) WHATEVER k is, so that d/dk of the force exists at k = 0 (the search starts there; a capture
        # gated on k > 0 left the knob with a zero gradient and the descent's dead-knob guard refused the pass) -- halved at a division
        # (a count drop below 0.75 of the reference)
        if self.vref_block in C.state_schema and self.nref_block in C.state_schema:
            va, _ = C.state_schema[self.vref_block]
            na, _ = C.state_schema[self.nref_block]
            vref = C.state[:, va].to(dt_)
            nref = C.state[:, na].to(dt_)
            fresh = live_c & has_in & (nref <= 0) & (int(getattr(H, "frame", 0) or 0) >= self.capture_after)   # H.frame: the engine's tick
            dropped = live_c & (nref > 0) & (cnt < 0.75 * nref)
            if bool(fresh.any()) or bool(dropped.any()):
                st = C.state.clone()
                v_new = torch.where(fresh, v.detach(), torch.where(dropped, vref * 0.5, vref))
                n_new = torch.where(fresh | dropped, cnt.detach(), nref)
                st[:, va] = v_new.to(st.dtype)
                st[:, na] = n_new.to(st.dtype)
                C.state = st
                vref, nref = v_new, n_new
        else:
            raise ValueError(f"lumen_pressure: the cell set needs the blocks {self.vref_block!r} and {self.nref_block!r} (width 1) for its references")
        ok = live_c & has_in & (nref > 0) & (vref > 0)
        V0 = self.target_scale * vref.detach() * (cnt / nref.detach().clamp(min=1.0)).pow(self.exponent)
        d = self._clamp_deficit((V0 - v) / V0.clamp(min=1e-12))
        d = torch.where(ok, d, torch.zeros_like(d))
        a = (k * d)[par, None] * (r / rn[:, None]) * inner[:, None]
        if bool(ok.any()):
            self.last = {"deficit_mean": float(d[ok].mean().detach()), "deficit_max": float(d[ok].abs().max().detach()),
                         "r_in_mean": float(r_c[ok].mean().detach()), "wedge_over_target": float((v[ok] / V0[ok]).mean().detach())}
        return {self.at: a}


@register_operator("lumen_pressure", model="outward", family="mechanics", set="particle", kind="exchange",
                   title="push each cell's inner face out to its share of the lumen volume -- outward only",
                   equation=r"""$$\mathbf a_i=k_c\,\max\!\Big(0,\frac{V^{0}_c-v_c}{V^{0}_c}\Big)\,\hat{\mathbf r}_i\quad[i\ \text{apical}]$$""")
class LumenPressureOutward(LumenPressure):
    r"""lumen_pressure with the deficit one-sided: a cell below its wedge target is pushed out, a cell above it is left alone (the default pulls it in).
    particle -[containment]-> cell: as lumen_pressure.

        \mathbf a_i=k_c\,\max\!\Big(0,\frac{V^{0}_c-v_c}{V^{0}_c}\Big)\,\hat{\mathbf r}_i\quad[i\ \text{apical}]
        a_i            the acceleration of apical point i
        k_c            the pressure stiffness
        V^0_c          the target wedge volume
        v_c            the wedge volume
        \hat r_i       the outward radial direction

    Reference: as lumen_pressure; Plexus exp33 (2026-10-25)."""
    PARAM_UNITS = {}
    SYMBOLS = {'a_i': 'the acceleration of apical point i', 'k_c': 'the pressure stiffness', 'V^0_c': 'the target wedge volume', 'v_c': 'the wedge volume', '\\hat r_i': 'the outward radial direction'}

    def _clamp_deficit(self, d):
        return d.clamp(0.0, self.max_deficit)
