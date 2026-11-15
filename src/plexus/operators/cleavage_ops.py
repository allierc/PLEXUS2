"""exp 23 (frog cleavage): an egg of material points that cleaves into blastomeres without growing.

Three VARIANTS of existing contracts, registered here so no shared class changes
(experiments/exp23_frog_cleavage.md, `## The model`):

    seed_positions[model: egg]             one ball of material points owned by ONE cell, with a yolk value per
                                           point graded from the animal pole (0) to the vegetal pole (1)
    cell_divide[model: cleave]             the cell is cut through its nucleus -- the yolk-free centre of its
                                           cytoplasm -- across the long axis of that yolk-free cytoplasm; the
                                           points on one side are HANDED to the daughter (nothing resampled,
                                           nothing padded), so volume is conserved point by point and the two
                                           daughters are as unequal as the cut makes them
    mpm_grid_update[implementation: body_csf]
                                           the continuum surface force on EVERY cell's own grid block: each
                                           blastomere is a liquid drop with a cortical tension, so it rounds
    pair_potential[model: volume]          cells that cannot pass into one another: each point is pushed out of
                                           the OTHER cells' volume fraction on a shared lattice, held at a slight
                                           overlap that also makes them stick

WHY POINTS ARE HANDED OVER AND NOT COPIED. `cell_divide[mpm]` / `[mpm_planar]` give each daughter the block's full
point count (a resampled ball, or a half padded with copies), so the point count doubles every round and the
1-cell egg would have as few points as one 128-cell blastomere. Cleavage is the opposite of growth: the egg's
material is partitioned, never added to. Here the containment map `parent` of the moved points is rewritten in
place -- the per-cell grid (`_pp_frame`), `aggregate_centroid` and the renderer all read it through
`lift_index`, which caches the SAME tensor -- and the total point count is the egg's for the whole run.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator
from plexus.operators.mpm_ops import MPMGridUpdate, sub_dt
from plexus.operators.seed_ops import SeedPositionsLatticeBody
from plexus.operators.vertex_ops import CellDivideMPM


def _set_block(lvl, name, idx, values):
    """Write a width-1 block on rows `idx` (no-op when the set does not declare it)."""
    if not name or name not in lvl.state_schema:
        return
    st = lvl.state.clone()
    st[idx, lvl.state_schema[name][0]] = torch.as_tensor(values, dtype=st.dtype, device=st.device)
    lvl.state = st


@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="egg",
                   title="lay out the egg with graded yolk",
                   equation=r"""$$\mathbf x_i\in h\,\mathbb Z^3,\ \lvert\mathbf x_i-\mathbf o\rvert\le R,\qquad \pi(i)=c_0,\qquad y_i=\tfrac12\Big(1-\frac{(\mathbf x_i-\mathbf o)\cdot\hat{\mathbf a}}{R}\Big)$$""")
class SeedEgg(SeedPositionsLatticeBody):
    r"""the whole point set laid out as one solid ball owned by the cell set's single live cell, each point carrying its yolk, densest at the vegetal pole.
    particle -> particle: wakes every point of `at`, sets its containment map to the live cell c_0 of `cell_set`, lays the ball (pos, zero vel, p_vol, mass), writes `yolk`, `pid` and `label` on the points and the cell's `pos` and `yolk`.

        \mathbf x_i\in h\,\mathbb Z^3,\ \lvert\mathbf x_i-\mathbf o\rvert\le R,\qquad \pi(i)=c_0,\qquad y_i=\tfrac12\Big(1-\frac{(\mathbf x_i-\mathbf o)\cdot\hat{\mathbf a}}{R}\Big)
        x_i            the position of point i, on a lattice of spacing h inside the ball
        h              the lattice spacing, in world length
        o              the ball's centre (the cell's `start`)
        R              the ball's radius, in world length
        \pi(i)         the containment map: the cell point i belongs to, c_0 for all
        y_i            the point's yolk, 0 at the animal pole and 1 at the vegetal pole, linear in the height along the animal axis
        \hat a         the unit animal axis (`animal_axis`, default +z)

    Reference: Nieuwkoop, P. D. & Faber, J. (1994). Normal Table of Xenopus laevis (the yolk gradient along the animal-vegetal axis); the layout: Plexus (this work)."""
    PARAM_UNITS = {'animal_axis': 'count'}
    SYMBOLS = {'x_i': 'the position of point i, on a lattice of spacing h inside the ball', 'h': 'the lattice spacing, in world length', 'o': "the ball's centre (the cell's `start`)", 'R': "the ball's radius, in world length", '\\pi(i)': 'the containment map: the cell point i belongs to, c_0 for all', 'y_i': "the point's yolk, 0 at the animal pole and 1 at the vegetal pole, linear in the height along the animal axis", '\\hat a': 'the unit animal axis (`animal_axis`, default +z)'}
    REQUIRES_PARAMS = ["radius"]                 # the shape is always a ball
    PARAM_ROLES = {"cell_set": "the_cell_set_whose_single_live_cell_is_the_egg", "radius": "egg_radius_world",
                   "centre": "egg_centre_world", "yolk": "point_block_written_with_the_yolk",
                   "animal_axis": "axis_pointing_to_the_animal_pole", 'pid': 'block_that_receives_the_point_id', 'label': 'block_that_receives_the_cell_label'}

    def __init__(self, params, device="cpu"):
        own = ("cell_set", "yolk", "animal_axis", "pid", "label")
        p = {k: v for k, v in params.items() if k not in own}
        p.setdefault("shape", "ball")
        super().__init__(p, device)
        if self.shape != "ball":
            raise ValueError("seed_positions[egg]: the egg is a solid ball (shape: ball)")
        self.cell_set = str(params.get("cell_set", "cell"))
        self.yolk = params.get("yolk", "yolk")
        self.axis = int(params.get("animal_axis", 2))
        self.pid = params.get("pid", "pid")
        self.label = params.get("label", "cellc")

    def forward(self, H, mask=None):
        P, C = H.level(self.at), H.level(self.cell_set)
        live_c = torch.nonzero(C.occ > 0.5).flatten()
        if live_c.numel() != 1:
            raise ValueError(f"seed_positions[egg]: {self.cell_set!r} must start with ONE live cell (n: 1), "
                             f"has {int(live_c.numel())}")
        c0 = int(live_c[0])
        P.occ = torch.ones_like(P.occ)
        P.parent.fill_(c0)                       # IN PLACE: lift_index caches this very tensor
        super().forward(H, mask)
        a, b = P.state_schema["pos"]
        X = P.state[:, a:b].double()
        o = torch.as_tensor(self.centre, dtype=X.dtype, device=X.device)
        y = (0.5 * (1.0 - (X[:, self.axis] - o[self.axis]) / float(self.radius))).clamp(0.0, 1.0)
        allp = torch.arange(X.shape[0], device=X.device)
        _set_block(P, self.yolk, allp, y)
        _set_block(P, self.pid, allp, float(c0))
        _set_block(P, self.label, allp, float(c0 % 20))
        ca, cb = C.state_schema["pos"]
        st = C.state.clone()
        st[c0, ca:cb] = o.to(st.dtype)
        C.state = st
        _set_block(C, self.yolk, torch.tensor([c0]), float(y.mean()))
        print(f"[seed_positions:egg] {self.at}: {X.shape[0]:,} points in one egg (cell {c0}) of radius "
              f"{float(self.radius):g}; yolk 0 (animal pole) .. 1 (vegetal pole) along axis {self.axis}", flush=True)
        return {}


@register_operator("cell_divide", model="cleave", set="particle", kind="divide", family="population",
                   title="cleave cells through the nucleus",
                   equation=r"""$$w_i=1-a\,y_i,\quad \mathbf n_c=\frac{\sum_i w_i v_i\mathbf x_i}{\sum_i w_i v_i},\quad \mathbf u_c=\arg\max_{\lvert\mathbf u\rvert=1}\mathbf u^{\!\top}\Big(\sum_i w_i v_i(\mathbf x_i-\mathbf n_c)(\mathbf x_i-\mathbf n_c)^{\!\top}\Big)\mathbf u,\quad i\to\begin{cases}c&(\mathbf x_i-\mathbf n_c)\cdot\mathbf u_c\ge0\\ d&\text{else}\end{cases}$$""")
class CellDivideCleave(CellDivideMPM):
    r"""a cleaving blastomere made of material points: the nucleus at the centre of its yolk-free cytoplasm, the spindle along that cytoplasm's long axis, the cut through the nucleus normal to the spindle.
    particle cell -> two particle cells, the daughter in a free slot of `at` (the trigger `mark` / `p_div` / `max_per_call`, `daughter_mark` and the copied width-1 cell blocks as `cell_divide[model: mpm]`): the points behind the plane change cell in the containment map.

        w_i=1-a\,y_i,\quad \mathbf n_c=\frac{\sum_i w_i v_i\mathbf x_i}{\sum_i w_i v_i},\quad \mathbf u_c=\arg\max_{\lvert\mathbf u\rvert=1}\mathbf u^{\!\top}\Big(\sum_i w_i v_i(\mathbf x_i-\mathbf n_c)(\mathbf x_i-\mathbf n_c)^{\!\top}\Big)\mathbf u,\quad i\to\begin{cases}c&(\mathbf x_i-\mathbf n_c)\cdot\mathbf u_c\ge0\\ d&\text{else}\end{cases}
        w_i            the share of point i that is yolk-free cytoplasm, 1 - a y_i
        a              `yolk_weight`, how much yolk excludes cytoplasm
        y_i            the point's yolk (block `yolk`)
        n_c            the nucleus: the volume-weighted yolk-free centre of cell c
        v_i            the rest volume of point i
        x_i            the position of point i
        u_c            the spindle: the longest axis of the yolk-free cytoplasm about the nucleus (the top eigenvector)
        c, c'          the mother and the daughter: a point goes to the daughter when it lies behind the plane through n_c normal to u_c

    Reference: Wuhr, M., Tan, E. S., Parker, S. K., Detrich, H. W. & Mitchison, T. J. (2010). A model for cleavage plane determination in early amphibian and fish embryos. Curr. Biol. 20:2040-2045."""
    PARAM_UNITS = {'yolk_weight': 'fraction'}
    SYMBOLS = {'w_i': 'the share of point i that is yolk-free cytoplasm, 1 - a y_i', 'a': '`yolk_weight`, how much yolk excludes cytoplasm', 'y_i': "the point's yolk (block `yolk`)", 'n_c': 'the nucleus: the volume-weighted yolk-free centre of cell c', 'v_i': 'the rest volume of point i', 'x_i': 'the position of point i', 'u_c': 'the spindle: the longest axis of the yolk-free cytoplasm about the nucleus (the top eigenvector)', "c, c'": 'the mother and the daughter: a point goes to the daughter when it lies behind the plane through n_c normal to u_c'}
    PARAM_ROLES = dict(CellDivideMPM.PARAM_ROLES, yolk="point_block_with_each_points_yolk_fraction",
                       yolk_weight="how_much_yolk_excludes_the_aster", gap="furrow_width_world",
                       yolk_law="linear_1_minus_a_y_or_exp_minus_a_y",
                       pid="point_block_with_the_cell_id", label="point_block_with_the_colour_label")
    REFERENCE = "Wuhr, M. et al. (2010). Curr. Biol. 20:2040-2045."

    def __init__(self, params, device="cpu"):
        own = ("yolk", "yolk_weight", "yolk_law", "pid", "label")
        super().__init__({k: v for k, v in params.items() if k not in own}, device)
        self.yolk = params.get("yolk", "yolk")
        self.a = float(params.get("yolk_weight", 1.0))
        self.pid = params.get("pid", "pid")
        self.label = params.get("label", "cellc")
        # `yolk_law: exp` -- w = exp(-a y) instead of 1 - a y (exp23 v9, 2026-10-11). The linear weight halves
        # across one small vegetal cell (y 0.8 -> 0.9: w 0.2 -> 0.1), so every later nucleus is pulled far off
        # the cell's centre and the unequal cuts compound: in v8 the cells' sizes within a half spread to a
        # coefficient of variation of 0.34-0.40 at 128 cells (smallest 0.18-0.21 of the largest). The exponential
        # changes by the same factor exp(-a dy) everywhere, so a cell's own offset depends only on how much yolk
        # gradient it spans, not on where it sits.
        self.law = str(params.get("yolk_law", "linear"))
        if self.law not in ("linear", "exp"):
            raise ValueError("cell_divide[cleave]: yolk_law is linear or exp")
        if self.law == "linear" and not 0.0 <= self.a <= 1.0:
            raise ValueError("cell_divide[cleave]: with yolk_law linear, yolk_weight is a fraction in [0, 1]")

    def forward(self, H, mask=None):
        C, P = H.level(self.at), H.level(self.points)
        live = (C.occ > 0.5).detach().cpu().numpy()
        ok = live.copy()
        if self.mark:
            if self.mark not in C.state_schema:
                raise ValueError(f"cell_divide[cleave]: mark {self.mark!r} is not a block of {self.at!r}")
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
        y_all = (P.state[:, P.state_schema[self.yolk][0]].detach().double() if self.yolk in P.state_schema
                 else torch.zeros(X_all.shape[0], dtype=torch.float64, device=X_all.device))
        v_all = (P.p_vol.detach().double() if getattr(P, "p_vol", None) is not None
                 else torch.ones_like(y_all))
        st = P.state.clone()
        par_new = par.clone()
        cst, coc = C.state.clone(), C.occ.clone()
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
            w = (torch.exp(-self.a * y_all[idx]) if self.law == "exp"
                 else (1.0 - self.a * y_all[idx]).clamp_min(0.0)) * v
            nuc = (w[:, None] * X).sum(0) / w.sum().clamp_min(1e-30)
            Y = X - nuc
            M = (w[:, None, None] * Y[:, :, None] * Y[:, None, :]).sum(0) / w.sum().clamp_min(1e-30)
            _, V = torch.linalg.eigh(M)
            ax = V[:, -1]
            xi = Y @ ax
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
            keep = idx[~to_d]
            cst[c, ca:cb] = st[keep, p0:p1].mean(0).to(cst.dtype)
            cst[d, ca:cb] = st[moved, p0:p1].mean(0).to(cst.dtype)
            if self.yolk in C.state_schema:
                cst[c, C.state_schema[self.yolk][0]] = float(y_all[keep].mean())
                cst[d, C.state_schema[self.yolk][0]] = float(y_all[moved].mean())
            if self.mark and self.clear_mark:
                cst[[c, d], C.state_schema[self.mark][0]] = 0.0
            if self.daughter_mark:
                cst[[c, d], C.state_schema[self.daughter_mark][0]] = 1.0
            coc[d] = 1.0
            log.append((tick, int(c), d, ax.cpu().numpy().tolist(), nuc.cpu().numpy().tolist(),
                        float(v[~to_d].sum()), float(v[to_d].sum())))
            n_done += 1
        if n_done:
            P.state = st
            P.parent.copy_(par_new)              # IN PLACE: lift_index caches this very tensor
            C.state, C.occ = cst, coc
            H.mpm_divisions = int(getattr(H, "mpm_divisions", 0)) + n_done
        return {}


@register_operator("mpm_grid_update", implementation="body_csf", family="mpm", set="field", kind="field",
                   title="solve grid velocities with cell surface tension",
                   equation=r"""$$\mathbf v_g=\frac{\mathbf p_g}{m_g},\qquad \mathbf v_g\mathrel{+}=\Delta t\,\frac{\sigma\,\kappa_b\,\nabla c_b}{m_g/\Delta x^3},\qquad \kappa_b=-\nabla\cdot\frac{\nabla c_b}{\lvert\nabla c_b\rvert}\ \text{ on body } b\text{'s block}$$""")
class MPMGridUpdateBodyCSF(MPMGridUpdate):
    r"""the grid solve of a per-parent grid (one block per cell) with the continuum surface force computed on each cell's block from that cell's own liquid colour: every cell a drop with its own cortical tension.
    field -> field: reads the blocks' mass, momentum and liquid colour (deposited by `mpm_scatter` from points of a `material: liquid` type), writes the blocks' velocity.

        \mathbf v_g=\frac{\mathbf p_g}{m_g},\qquad \mathbf v_g\mathrel{+}=\Delta t\,\frac{\sigma\,\kappa_b\,\nabla c_b}{m_g/\Delta x^3},\qquad \kappa_b=-\nabla\cdot\frac{\nabla c_b}{\lvert\nabla c_b\rvert}\ \text{ on body } b\text{'s block}
        v_g            the velocity of grid node g, in world length per time
        p_g            its momentum
        m_g            its mass
        \Delta t       the substep
        \sigma         the surface tension (`cell_surface_tension`, scaled per cell by `cell_surface_tension_block`), force per length
        \kappa_b       the curvature of body b's colour interface at the node, per world length
        c_b            body b's liquid colour on its own block, normalised to a volume fraction by csf_rho dx^3
        \Delta x       the grid spacing

    Reference: Brackbill, J. U., Kothe, D. B. & Zemach, C. (1992). A continuum method for modeling surface tension. J. Comput. Phys. 100:335-354."""
    PARAM_UNITS = {}
    SYMBOLS = {'v_g': 'the velocity of grid node g, in world length per time', 'p_g': 'its momentum', 'm_g': 'its mass', '\\Delta t': 'the substep', '\\sigma': 'the surface tension (`cell_surface_tension`, scaled per cell by `cell_surface_tension_block`), force per length', '\\kappa_b': "the curvature of body b's colour interface at the node, per world length", 'c_b': "body b's liquid colour on its own block, normalised to a volume fraction by csf_rho dx^3", '\\Delta x': 'the grid spacing'}
    PARAM_ROLES = {'cell_surface_tension_block': 'cell_block_that_scales_the_tension_per_cell'}

    def __init__(self, params, device="cpu"):
        super().__init__({k: v for k, v in params.items() if k != "cell_surface_tension_block"}, device)
        # `cell_surface_tension_block: <cell block>` (exp25, 2026-10-12): a per-cell FACTOR on `cell_surface_tension`, read from the
        # cell set every call -- 0 for a fluid body with no cortex (the blastocoel), > 1 for a cell whose cortex
        # contracts (a bottle cell). None keeps one tension for every block, exactly as before.
        self.tension_block = params.get("cell_surface_tension_block")

    def forward(self, H, mask=None):
        g = H.field(self.at)
        if not getattr(g, "per_parent", None):
            raise ValueError("mpm_grid_update[body_csf]: needs a `per_parent` grid (one block per cell)")
        if self.csf_rho <= 0.0:
            raise ValueError("mpm_grid_update[body_csf]: set `csf_rho` (the liquid's density) -- the colour must "
                             "be a volume fraction for the tension to be a tension")
        dt = sub_dt(H, self.dt_sub)
        floor = self._const("mass_floor", g)
        gm = g.m
        gv = g.mv / gm.clamp(min=floor)[:, None]
        bf = getattr(H, "_mpm_body_accel", None)
        if bf is not None:
            gv = gv + dt * bf
        surf = self.surface_tension
        if surf > 0.0:
            D, dx, inv_dx = g.dim, float(g.dx), float(g.inv_dx)
            S = tuple(g.shape)
            nb = int(g.n_blocks)
            c = (g.c / (self.csf_rho * dx ** D)).view(nb, *S)
            for _ in range(self.csf_smooth):
                for k in range(1, D + 1):
                    c = 0.25 * torch.roll(c, 1, k) + 0.5 * c + 0.25 * torch.roll(c, -1, k)
            grad = [(torch.roll(c, -1, k) - torch.roll(c, 1, k)) * (0.5 * inv_dx) for k in range(1, D + 1)]
            gmag = torch.sqrt(sum(q * q for q in grad))
            eps = self._const("csf_eps", g)
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
            inv_m = (dx ** D) / gm.clamp(min=self._const("csf_mass_floor", g))
            if self.tension_block:
                Cl = H.level(g.per_parent)
                fac = Cl.state[:, Cl.state_schema[self.tension_block][0]].to(c.dtype)
                if fac.shape[0] != nb:
                    raise ValueError(f"mpm_grid_update[body_csf]: {nb} blocks but {fac.shape[0]} cells")
                surf = surf * fac.view(nb, *([1] * D))
            gv = gv + dt * torch.stack([(surf * kappa * grad[k] * fmask).reshape(-1) * inv_m for k in range(D)], 1)
        g.v = torch.where((gm > floor)[:, None], gv, torch.zeros_like(gv))
        return {}


@register_operator("pair_potential", model="volume", family="interaction", set="particle", kind="exchange",
                   title="keep cells from overlapping",
                   equation=r"""$$c^{o}_i=\sum_{n}w_{in}\big(\phi_n-\phi^{\pi(i)}_n\big),\qquad \phi^{b}_n=\sum_{j\in b}\frac{v_j\,w_{jn}}{\Delta x^3},\qquad \mathbf a_i=-\frac{g(c^{o}_i)}{\rho}\,\nabla c^{o}_i,\quad g(c)=k\,(c-c_0)\ \ (c\ge c_0),\ \ \alpha\,k\,(c-c_0)\ \ (0<c<c_0)$$""")
class CellContactVolume(Exchange):
    r"""bodies made of material points, each on its own grid, that cannot pass into one another: every point feels the other bodies' volume fraction where it sits, pushed down its gradient when overlapped by more than c_0, pulled up it alpha times as hard when nearer.
    particle -> particle: reads every live point's position, rest volume and body (`cells`, the containment map); emits an acceleration on each (`mpm_acceleration`, the MPM substep's a_ext).

        c^{o}_i=\sum_{n}w_{in}\big(\phi_n-\phi^{\pi(i)}_n\big),\qquad \phi^{b}_n=\sum_{j\in b}\frac{v_j\,w_{jn}}{\Delta x^3},\qquad \mathbf a_i=-\frac{g(c^{o}_i)}{\rho}\,\nabla c^{o}_i,\quad g(c)=k\,(c-c_0)\ \ (c\ge c_0),\ \ \alpha\,k\,(c-c_0)\ \ (0<c<c_0)
        c^o_i          the other bodies' volume fraction at point i (the total minus its own body's), dimensionless
        w_{in}         the quadratic B-spline weight of point i at lattice node n
        \phi_n         the total volume fraction at node n
        \phi^{\pi(i)}_n the fraction of point i's own body there
        v_j            the rest volume of point j
        \Delta x       the shared lattice spacing (`dx`, default the per-cell grid's)
        a_i            the acceleration of point i
        g(c)           the penalty law: k (c - c_0) in overlap, alpha k (c - c_0) in the adhesive band below c_0
        \rho           the material density (`density`)
        k              `repulsion_strength`, the contact stiffness, in acceleration per unit fraction
        c_0            `contact_slack`, the fraction tolerated
        \alpha         `adhesion`, the pull's share of the push

    Reference: Bardenhagen, S. G., Brackbill, J. U. & Sulsky, D. (2000). The material-point method for granular materials. Comput. Methods Appl. Mech. Eng. 187:529-541 (multi-body MPM contact); the penalty form: Plexus (this work)."""
    PARAM_UNITS = {'repulsion_strength': 'acceleration', 'contact_slack': 'fraction', 'adhesion': 'fraction', 'density': 'density', 'dx': 'length'}
    SYMBOLS = {'c^o_i': "the other bodies' volume fraction at point i (the total minus its own body's), dimensionless", 'w_{in}': 'the quadratic B-spline weight of point i at lattice node n', '\\phi_n': 'the total volume fraction at node n', '\\phi^{\\pi(i)}_n': "the fraction of point i's own body there", 'v_j': 'the rest volume of point j', '\\Delta x': "the shared lattice spacing (`dx`, default the per-cell grid's)", 'a_i': 'the acceleration of point i', 'g(c)': 'the penalty law: k (c - c_0) in overlap, alpha k (c - c_0) in the adhesive band below c_0', '\\rho': 'the material density (`density`)', 'k': '`repulsion_strength`, the contact stiffness, in acceleration per unit fraction', 'c_0': '`contact_slack`, the fraction tolerated', '\\alpha': "`adhesion`, the pull's share of the push"}
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["cells", "repulsion_strength"]
    MECHANISM_TAGS = ["excluded_volume", "adhesion", "multi_body_contact"]
    PARAM_ROLES = {"cells": "the_body_set_containing_the_points", "repulsion_strength": "overlap_pressure_per_unit_fraction",
                   "contact_slack": "rest_overlap_fraction", "adhesion": "attractive_side_over_repulsive_side",
                   "dx": "lattice_spacing_world", "density": "material_density_when_density_from_param",
                   "density_from": "points_or_param",
                   "envelope": "radius_centre_width_of_the_vitelline_envelope", 'grid': 'the_per_cell_grid_field'}
    REFERENCE = "Bardenhagen, S. G. et al. (2000). Comput. Methods Appl. Mech. Eng. 187:529-541."

    _OFF = None

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.cells = str(params["cells"])
        self.k = self.tunable(params["repulsion_strength"], None, tensor=True)
        self.c0 = float(params.get("contact_slack", 0.25))
        self.alpha = float(params.get("adhesion", 0.0))
        self.dx = params.get("dx")
        self.rho = self.tunable(params.get("density"), 1.0, tensor=True)
        # `density_from: points` (default, 2026-10-13): the pressure k (c - c0) is divided by EACH POINT'S OWN density
        # (its mass over its rest volume), so a heavy cell decelerates less than a light one under the same contact
        # pressure and the pair's momentum is conserved. `param` keeps the first law -- one `density` for every
        # point -- which is the same whenever all cells share one density (every exp23 / exp25 run before this) and
        # wrong otherwise (Cedric asked: the push ignored whether the neighbour is heavy or light).
        self.density_from = str(params.get("density_from", "points"))
        if self.density_from not in ("points", "param"):
            raise ValueError("pair_potential[volume]: density_from is points or param")
        self.grid = params.get("grid", "mpm_grid")
        env = params.get("envelope")
        self.env = None
        if env:
            self.env = (float(env["radius"]), [float(v) for v in env["centre"]], float(env.get("width", 0.5)))
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("pair_potential[volume]: adhesion is a fraction in [0, 1]")

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
        Nw = shape[0] * shape[1] * shape[2]
        if CellContactVolume._OFF is None or CellContactVolume._OFF.device != dev:
            o = torch.arange(3, device=dev)
            CellContactVolume._OFF = torch.stack(torch.meshgrid(o, o, o, indexing="ij"), -1).reshape(-1, 3)
        off = CellContactVolume._OFF
        xg = X / dx
        base = torch.floor(xg - 0.5)
        fx = xg - base                                                     # in [0.5, 1.5)
        w = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], 1)   # [N,3,D]
        dw = torch.stack([fx - 1.5, -2.0 * (fx - 1.0), fx - 0.5], 1) / dx                             # d/dx
        idx = base.long()[:, None, :] + off[None]                          # [N,27,3]
        for d in range(3):
            idx[..., d] = idx[..., d].clamp(0, shape[d] - 1)
        flat = (idx[..., 0] * shape[1] + idx[..., 1]) * shape[2] + idx[..., 2]          # [N,27]
        wx, wy, wz = w[:, :, 0], w[:, :, 1], w[:, :, 2]
        W = (wx[:, off[:, 0]] * wy[:, off[:, 1]] * wz[:, off[:, 2]])                    # [N,27]
        G = torch.stack([dw[:, off[:, 0], 0] * wy[:, off[:, 1]] * wz[:, off[:, 2]],
                         wx[:, off[:, 0]] * dw[:, off[:, 1], 1] * wz[:, off[:, 2]],
                         wx[:, off[:, 0]] * wy[:, off[:, 1]] * dw[:, off[:, 2], 2]], -1)  # [N,27,3]
        v = (P.p_vol.to(dt_) if getattr(P, "p_vol", None) is not None else torch.ones_like(live)) * live
        dep = (W * (v / dx ** 3)[:, None]).reshape(-1)
        phi_b = torch.zeros(nC * Nw, device=dev, dtype=dt_).index_add_(0, (cid[:, None] * Nw + flat).reshape(-1), dep)
        phi = phi_b.view(nC, Nw).sum(0)
        other = phi[flat] - phi_b[cid[:, None] * Nw + flat]                 # [N,27]: the others' fraction per node
        c_o = (W * other).sum(1)
        grad = (G * other[..., None]).sum(1)
        g = self.k * (c_o - self.c0)
        g = torch.where(c_o < self.c0, self.alpha * g, g)
        g = torch.where(c_o > 1e-6, g, torch.zeros_like(g))
        rho = self.rho
        if self.density_from == "points" and getattr(P, "mass", None) is not None and getattr(P, "p_vol", None) is not None:
            rho = (P.mass.to(dt_) / P.p_vol.to(dt_).clamp_min(1e-30)).clamp_min(1e-12)
        acc = -(g / rho)[:, None] * grad
        if self.env is not None:
            Re, oe, we = self.env
            d = X - torch.as_tensor(oe, device=dev, dtype=dt_)
            r = d.norm(dim=1).clamp_min(1e-9)
            ph = ((r - Re + we) / we).clamp_min(0.0)
            acc = acc - ((self.k / rho) * ph / we / r)[:, None] * d
            H.cell_envelope_max = float(ph.max())
        acc = acc * live[:, None]
        if mask is not None:
            acc = acc * mask[:, None].to(dt_)
        H.cell_contact_max = float(c_o.max()) if c_o.numel() else 0.0
        return {self.at: acc}
