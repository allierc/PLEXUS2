"""Point cells that grow and divide only while they adhere: the spheroid of experiments/exp05_spheroid_from_words (Cedric, 2026-11-14).

One variant, `cell_divide[point_adhesion]`, of the point-cell division `cell_divide[point]` (vertex_ops.py): a cell that touches at least
`min_contacts` other cells grows; a cell that touches fewer does not; a grown cell (size twice its birth size) divides in two.
"""
import math

import torch

from plexus.models.registry import register_operator
from plexus.operators.vertex_ops import CellDividePoint, POINT_DIVIDE_TRACE
from plexus.operators.channel_ops import _block, _write

BLOCKWISE_N = 4000           # live cells above which the contact counts and neighbour lists use the blockwise radius search (geometry.edges_radius_blockwise)
                             # instead of the all-pairs table: 10,000 cells (Cedric, 2026-11-14: "increase the cell reservoir to 10000") make a 10^8-entry table


def _touch_pairs(Xl, r):
    """[2, E] local index pairs (both directions) of the cells closer than `r`, the cell itself excluded."""
    if Xl.shape[0] > BLOCKWISE_N:
        from plexus.geometry import edges_radius_blockwise
        return edges_radius_blockwise(Xl.detach(), torch.ones(Xl.shape[0], device=Xl.device), 0.0, float(r))
    D = torch.cdist(Xl, Xl)
    D.fill_diagonal_(float("inf"))
    return (D < r).nonzero().T


def _touch_counts(Xl, r):
    """[n] how many cells each cell is closer than `r` to (itself excluded)."""
    if Xl.shape[0] > BLOCKWISE_N:
        e = _touch_pairs(Xl, r)
        return torch.bincount(e[0], minlength=Xl.shape[0])
    return (torch.cdist(Xl, Xl) < r).sum(1) - 1


@register_operator("cell_divide", implementation="point_adhesion", set="particle", kind="divide", family="population",
                   title="grow and divide point cells that adhere",
                   equation=r"""$$n_i=\#\{j:\ |\mathbf x_j-\mathbf x_i|<d\},\qquad \frac{ds_i}{dt}=g\,s_i\,[n_i\ge n^*],\qquad s_i\ge 2\ \Rightarrow\ s_{m,d}=s_i/2,\ \mathbf x_{m,d}=\mathbf x\mp\tfrac{\delta}{2}\hat{\mathbf u}$$""")
class CellDividePointAdhesion(CellDividePoint):
    r"""`point_adhesion` implementation of cell_divide: a point cell grows only while it touches enough other cells, and divides when it has doubled.
    particle -> particle: reads pos, writes the blocks `contacts` (cells touched) and `size` (1 at birth); wakes a dormant slot per division (every per-slot buffer copied from the mother), places the two daughters about the mother and zeroes their `vel`.

        n_i=\#\{j:\ |x_j-x_i|<d\},\quad ds_i/dt=g\,s_i\,[n_i\ge n^*],\quad s_i\ge 2 \Rightarrow s_{m,d}=s_i/2,\ x_{m,d}=x\mp\tfrac{\delta}{2}\hat u
        n_i            the number of cells cell i touches (its adhesion)
        d              `touch`, the centre distance under which two cells touch, world length
        s_i            the cell's size, 1 at birth (a volume in birth volumes)
        g              `rate`, the growth rate of an adhering cell, per unit time (it doubles in ln 2 / g)
        n^*            `min_contacts`, the adhesion threshold: the fewest touched cells that let a cell grow
        x_{m,d}        the two daughters' positions
        \delta         `offset`, the distance between the daughters at birth, world length
        \hat u         a uniform random unit vector

    The founders start at sizes drawn uniformly in [1, 2) so they do not divide in step; each daughter's size is s_i/2 times
    1 +- `size_cv` (a uniform draw), so lineages lose step too.
    Reference: Drasdo, D. & Hoehme, S. (2005). Phys. Biol. 2:133-147 (the centre-based cell); Shraiman, B. I. (2005). PNAS
    102:3318-3323 and Puliafito, A. et al. (2012). PNAS 109:739-744 (growth gated by contact)."""
    PARAM_UNITS = {'rate': 'rate', 'touch': 'length', 'min_contacts': 'count', 'offset': 'length', 'size_cv': 'fraction',
                   'max_cells': 'count', 'seed': 'count'}
    SYMBOLS = {'n_i': 'the number of cells cell i touches (its adhesion)', 'd': '`touch`, the centre distance under which two cells touch',
               's_i': "the cell's size, 1 at birth", 'g': '`rate`, the growth rate of an adhering cell, per unit time',
               'n^*': '`min_contacts`, the adhesion threshold', 'x_{m,d}': "the two daughters' positions",
               '\\delta': '`offset`, the distance between the daughters at birth', '\\hat u': 'a uniform random unit vector'}
    REQUIRES_PARAMS = ["rate", "touch"]
    MECHANISM_TAGS = ["division", "cell_division", "proliferation", "centre_based_cells", "adhesion_gated_growth"]
    PARAM_ROLES = {"rate": "growth_rate_of_an_adhering_cell_per_time", "touch": "centre_distance_under_which_two_cells_touch",
                   "min_contacts": "fewest_touched_cells_that_let_a_cell_grow", "offset": "daughter_separation_at_birth",
                   "size_cv": "spread_of_the_daughters_birth_size", "max_cells": "population_ceiling", "seed": "rng_seed",
                   "size_block": "block_holding_the_size", "contacts_block": "block_holding_the_number_of_touched_cells",
                   "founder_block": "block_holding_the_founder_index_inherited_by_every_descendant",
                   "signal_block": "optional_block_a_cell_must_also_read_high_to_grow", "signal_channel": "channel_of_that_block",
                   "signal_min": "level_of_the_signal_at_or_above_which_an_adhering_cell_grows",
                   "rate_active": "growth_rate_of_a_cell_whose_rate_block_is_high", "rate_block": "block_read_for_rate_active", "rate_channel": "channel_of_that_block",
                   "rate_min": "level_at_or_above_which_a_cell_takes_rate_active", "rate_types": "growth_rate_per_cell_type",
                   "rate_chem": "growth_rate_as_a_weighted_sum_of_a_block's_channels"}
    REFERENCE = ("Drasdo, D. & Hoehme, S. (2005). Phys. Biol. 2:133-147; Puliafito, A. et al. (2012). PNAS 109:739-744.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.touch = float(params["touch"])
        self.min_contacts = int(params.get("min_contacts", 1))
        self.size_cv = float(params.get("size_cv", 0.1))
        self.size_block = str(params.get("size_block", "size"))
        self.contacts_block = str(params.get("contacts_block", "contacts"))
        self.founder_block = params.get("founder_block", "founder")
        # `signal_block` (default none): A SECOND GATE, the cell's own signal -- an adhering cell grows only where channel
        # `signal_channel` of that block is at least `signal_min` (exp05 iterations 21+, growth where a morphogen is high).
        self.signal_block = params.get("signal_block")
        self.signal_channel = int(params.get("signal_channel", 0))
        self.signal_min = float(params.get("signal_min", 0.0))
        # `rate_active` (default none): with `rate_block`, a cell whose `rate_block` channel `rate_channel` is at least `rate_min` grows at this rate
        # instead of `rate` (differential growth: an active patch outgrows the wall round it); it does not gate growth as signal_block does
        ra = params.get("rate_active")
        self.rate_active = None if ra is None else float(ra)
        self.rate_block = params.get("rate_block"); self.rate_channel = int(params.get("rate_channel", 0)); self.rate_min = float(params.get("rate_min", 0.0))
        # `rate_types: {tip: 0.0173}` (exp05 route A, 2026-11-15): the growth rate of the cells of a type, over `rate` and `rate_active` (tip cells that divide faster)
        self.rate_types = {str(k): float(v) for k, v in (params.get("rate_types") or {}).items()}
        # `rate_chem: {block: chem, weights: [w0, w1, w2], after: 540}` (exp05 route A with rock-paper-scissors, 2026-11-15): from world time `after` on, a
        # cell's growth rate is the weighted sum of its block's first channels, sum_k w_k c_k (never under 0), instead of `rate`: growth follows the
        # three chemicals continuously, with no mark. `rate_types` still overrides it for the cells of a listed type.
        rc = params.get("rate_chem")
        self.rate_chem = None if not rc else dict(block=str(rc.get("block", "chem")), weights=[float(v) for v in rc["weights"]], after=float(rc.get("after", 0.0)))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dt = float(getattr(H, "dt", 1.0) or 1.0)
        fr = getattr(H, "frame", None)
        fr = -1 if fr is None else int(fr)
        s = _block(lvl, self.size_block)
        if s is None:
            raise ValueError(f"cell_divide[point_adhesion]: set {self.at!r} declares no `{self.size_block}` block (width 1)")
        live = (lvl.occ > 0.5)
        X = lvl.get("pos")
        dev, dt_ = X.device, X.dtype
        size = s[:, 0].clone()
        if not bool((size[live] > 0).any()):                       # first call: founders desynchronised, founders named
            size = torch.where(live, 1.0 + torch.rand(size.shape[0], generator=self._gen).to(dev, dt_), size)
            fb = _block(lvl, self.founder_block) if self.founder_block else None
            if fb is not None:
                _write(lvl, self.founder_block, torch.arange(size.shape[0], device=dev, dtype=dt_))
        idx = live.nonzero(as_tuple=True)[0]
        Xl = X[idx]
        n = _touch_counts(Xl, self.touch).to(dt_)
        contacts = torch.zeros(size.shape[0], device=dev, dtype=dt_)
        contacts[idx] = n
        ok = n >= self.min_contacts
        if self.signal_block:
            sb = _block(lvl, self.signal_block)
            if sb is None:
                raise ValueError(f"cell_divide[point_adhesion]: `signal_block` {self.signal_block!r} is not declared on {self.at!r}")
            ok = ok & (sb[idx, self.signal_channel] >= self.signal_min)
        grow = ok.to(dt_)
        rate = self.rate
        if self.rate_active is not None and self.rate_block:
            rb = _block(lvl, self.rate_block)
            rate = torch.where(rb[idx, self.rate_channel] >= self.rate_min, torch.full_like(grow, self.rate_active), torch.full_like(grow, self.rate))
        if self.rate_chem is not None and fr * dt >= self.rate_chem["after"]:
            cb = _block(lvl, self.rate_chem["block"])
            if cb is None:
                raise ValueError(f"cell_divide[point_adhesion]: `rate_chem.block` {self.rate_chem['block']!r} is not declared on {self.at!r}")
            wk = torch.as_tensor(self.rate_chem["weights"], device=dev, dtype=dt_)
            rate = (cb[idx, : wk.numel()].to(dt_) * wk[None]).sum(1).clamp_min(0.0)
        if self.rate_types:
            tnames = list(getattr(lvl, "type_names", []) or [])
            ntl = lvl.node_type.to(dev).long()[idx]
            rate = rate if torch.is_tensor(rate) else torch.full_like(grow, float(rate))
            for tn, v in self.rate_types.items():
                rate = torch.where(ntl == tnames.index(tn), torch.full_like(grow, v), rate)
        size[idx] = size[idx] * torch.exp(rate * dt * grow)
        mother = idx[size[idx] >= 2.0]
        if self.max_cells > 0:
            mother = mother[: max(0, self.max_cells - int(idx.numel()))]
        new = torch.zeros(0, dtype=torch.long, device=dev)
        if mother.numel():
            self._now = max(fr, 0) * dt                          # world time, for a variant whose direction rule starts at a time (point_oriented `orient_after`)
            d = self._directions(lvl, X, idx, mother).to(dev, lvl.state.dtype)
            new, src = lvl.spawn(mother)
            if new.numel() < mother.numel() and not self._said_full:
                self._said_full = True
                print(f"[cell_divide/point_adhesion] `{self.at}` has no free slot for {int(mother.numel() - new.numel())} "
                      f"division(s) at frame {fr} ({int(idx.numel())} live of {lvl.n}); raise the set's `buffer`", flush=True)
            if new.numel():
                half = 0.5 * self.offset * d[: new.numel()]
                p0, p1 = lvl.state_schema["pos"]
                x = lvl.state[src, p0:p1].clone()
                st = lvl.state.clone()
                st[src, p0:p1] = x - half
                st[new, p0:p1] = x + half
                if "vel" in lvl.state_schema:
                    v0, v1 = lvl.state_schema["vel"]
                    st[src, v0:v1] = 0.0
                    st[new, v0:v1] = 0.0
                lvl.state = st
                jit = 1.0 + self.size_cv * (2.0 * torch.rand(2, int(new.numel()), generator=self._gen) - 1.0)
                jit = jit.to(dev, dt_)
                half_size = 0.5 * size[src]
                size[src] = half_size * jit[0]
                size[new] = half_size * jit[1]
                contacts[new] = contacts[src]
        _write(lvl, self.size_block, size)
        if _block(lvl, self.contacts_block) is not None:
            _write(lvl, self.contacts_block, contacts)
        POINT_DIVIDE_TRACE.append((fr, int(new.numel()), int(idx.numel()) + int(new.numel())))
        return {}

    def _directions(self, lvl, X, idx, mother):
        """[mothers, dim] unit vectors along which each mother's two daughters are placed: uniform random here."""
        d = torch.randn(int(mother.numel()), X.shape[-1], generator=self._gen)
        return d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)


from plexus.models.base import Structural  # noqa: E402


@register_operator("cell_fate", set="particle", kind="structural", family="population",
                   title="give each cell a fate by its signal level",
                   equation=r"""$$\mathrm{type}_i=T_k,\qquad k=\#\{m:\ c_{i}\ge\theta_m\}$$""")
class CellFate(Structural):
    r"""a cell's fate (its type) read from the level of one of its signals: below the first threshold the first type, above the last the last; `commit` makes a fate other than the first irreversible.
    particle -> particle: reads channel `channel` of block `block`; rewrites `node_type` of the live cells (the renderer colours by type, `pair_potential[typed]` adheres by type).

        \mathrm{type}_i=T_k,\qquad k=\#\{m:\ c_{i}\ge\theta_m\}
        c_i            the signal of cell i, channel `channel` of block `block`
        \theta_m       `thresholds`, increasing
        T_k            `types`, one more name than thresholds, in the set's `types:` order or any order

    Reference: Wolpert, L. (1969). J. Theor. Biol. 25:1-47 (positional information: a fate read from a signal by thresholds)."""
    PARAM_UNITS = {'channel': 'count'}
    SYMBOLS = {'c_i': 'the signal of cell i', '\\theta_m': '`thresholds`, increasing', 'T_k': '`types`, one more than thresholds'}
    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["block", "thresholds", "types"]
    MECHANISM_TAGS = ["cell_fate", "positional_information", "threshold"]
    PARAM_ROLES = {"block": "the_signal_block", "channel": "channel_of_that_block", "thresholds": "increasing_signal_levels",
                   "types": "the_fate_names_one_more_than_thresholds", "commit": "a_fate_other_than_the_first_is_irreversible",
                   "only_types": "current_types_of_the_cells_the_fate_is_read_for_default_all"}
    REFERENCE = "Wolpert, L. (1969). J. Theor. Biol. 25:1-47."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.block = str(params["block"]); self.channel = int(params.get("channel", 0))
        self.thresholds = [float(v) for v in params["thresholds"]]
        self.types = [str(v) for v in params["types"]]
        if len(self.types) != len(self.thresholds) + 1:
            raise ValueError("cell_fate: `types` has one more name than `thresholds`")
        self.commit = bool(params.get("commit", False))
        # `only_types` (default none = every live cell): the fate is read only for the cells whose CURRENT type is one of these, the others keep
        # theirs (exp05 route B: a `pacer` lineage that a wave-borne mark must not overwrite)
        ot = params.get("only_types")
        self.only_types = None if ot is None else [str(v) for v in ot]
        self._assigned = False                                   # the first call assigns from the signal alone: the seeded types are not fates

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        b = _block(lvl, self.block)
        if b is None:
            raise ValueError(f"cell_fate: block {self.block!r} is not declared on {self.at!r}")
        names = list(getattr(lvl, "type_names", []) or [])
        ids = torch.as_tensor([names.index(t) for t in self.types], device=lvl.node_type.device, dtype=lvl.node_type.dtype)
        c = b[:, self.channel]
        k = torch.zeros_like(c, dtype=torch.long)
        for th in self.thresholds:
            k = k + (c >= th).long()
        if self.commit and self._assigned:                       # once past the first fate, a cell keeps its highest fate
            cur = torch.zeros_like(k)
            for kk in range(len(self.types)):
                cur = torch.where(lvl.node_type.to(k.device).long() == int(ids[kk]), torch.full_like(k, kk), cur)
            k = torch.maximum(k, cur)
        live = lvl.occ > 0.5
        if self.only_types is not None:
            cur_t = lvl.node_type.to(live.device).long()
            sel = torch.zeros_like(live)
            for t in self.only_types:
                sel = sel | (cur_t == names.index(t))
            live = live & sel
        nt = lvl.node_type.clone()
        nt[live] = ids[k[live].to(ids.device)]
        lvl.node_type.copy_(nt)
        self._assigned = True
        return {}


from plexus.operators.channel_ops import PairPotentialSoftAdhesive  # noqa: E402


@register_operator("pair_potential", implementation="soft_adhesive_typed", family="interaction", set="particle", kind="exchange",
                   title="apply soft sticky contact, the stickiness set by the two cells' types",
                   equation=r"""$$U_{ij}(r)=\tfrac{\epsilon}{2}\big(1-\tfrac{r}{\sigma}\big)^2\ (r<\sigma)\;-\;\epsilon_a^{T_iT_j}\cos^2\!\Big(\frac{\pi(r-\sigma)}{2w}\Big)\ (\sigma\le r\le\sigma+w)$$""")
class PairPotentialSoftAdhesiveTyped(PairPotentialSoftAdhesive):
    r"""pair_potential[soft_adhesive] with the adhesive well's depth read from a table by the types of the two cells (differential adhesion); the soft core is shared.
    A -> A, one set: reads pos and node_type, emits a velocity (v = mu F) on every live element.

        U_{ij}(r)=\tfrac{\epsilon}{2}(1-r/\sigma)^2\ (r<\sigma)\;-\;\epsilon_a^{T_iT_j}\cos^2(\pi(r-\sigma)/(2w))\ (\sigma\le r\le\sigma+w)
        U_{ij}(r)      the pair energy of cells i and j at distance r
        \epsilon       `epsilon`, the core's stiffness, energy
        \sigma         `sigma`, the contact distance, world length
        \epsilon_a^{T_iT_j}  `adhesion_table[T_i][T_j]`, the well depth for the pair of types (symmetric; `adhesion` for a pair not listed)
        w              `tail`, the shoulder's width, world length

    Reference: Steinberg, M. S. (1963). Science 141:401-408 (differential adhesion sorts cells); Graner, F. & Glazier, J. A. (1992). PRL 69:2013-2016."""
    PARAM_ROLES = dict(PairPotentialSoftAdhesive.PARAM_ROLES, adhesion_table="well_depth_per_pair_of_types_by_name")
    MECHANISM_TAGS = ["soft_sphere", "adhesion", "differential_adhesion", "cell_sorting"]
    REFERENCE = ("Steinberg, M. S. (1963). Science 141:401-408; Graner, F. & Glazier, J. A. (1992). Phys. Rev. Lett. 69:2013-2016.")

    def __init__(self, params, device="cpu"):
        p = dict(params)
        self.table = {str(a): {str(b): float(v) for b, v in row.items()} for a, row in (p.pop("adhesion_table", None) or {}).items()}
        p.setdefault("adhesion", max([v for row in self.table.values() for v in row.values()] or [0.0]))
        super().__init__(p, device)
        self._W = None

    def _matrix(self, lvl, dev, dt):
        names = list(getattr(lvl, "type_names", []) or [])
        W = torch.full((len(names), len(names)), self.eps_a, device=dev, dtype=dt)
        for a, row in self.table.items():
            for b, v in row.items():
                i, j = names.index(a), names.index(b)
                W[i, j] = W[j, i] = v
        return W

    def forward(self, H, mask=None):
        if len(self.A) != 1 or not self.same:
            raise ValueError("pair_potential[soft_adhesive_typed]: one set, with itself")
        lvl = H.level(self.A[0])
        X = lvl.get("pos"); occ = lvl.occ > 0
        if self._W is None:
            self._W = self._matrix(lvl, X.device, X.dtype)
        nt = lvl.node_type.to(X.device).long()
        F = torch.zeros_like(X)
        s, w = self.sigma, self.w
        g = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
        for gi, jj in self._pairs(X, X, occ, occ, g, g, None, None):
            d = X[jj] - X[gi]
            r = d.norm(dim=1).clamp_min(1e-9)
            m = torch.where(r < s, self.eps * (1.0 - r / s) / s, torch.zeros_like(r))
            u = (r - s) / w
            ea = self._W[nt[gi], nt[jj]]
            m = m + torch.where((u >= 0.0) & (u <= 1.0), -ea * (math.pi / (2.0 * w)) * torch.sin(math.pi * u), torch.zeros_like(r))
            f = -(m / r)[:, None] * d
            F.index_add_(0, gi, f)
            F.index_add_(0, jj, -f)
        mu = self.mu_a if not isinstance(self.mu_a, dict) else float(self.mu_a[self.A[0]])
        return {self.A[0]: mu * F * occ[:, None].to(X.dtype)}


# ----------------------------------------------------------------------------------------------------------------- tissue topology (exp05 iterations 31-50)
# Cedric, 2026-11-14: "continue developing cell signalling modifying cell tissue topology ... from torus to tube to budding and tubing; you can
# add cell death as a sculptor of the cell tissue". Three operators: a death rule, an oriented division, a signal made by crowded cells.


@register_operator("cell_divide", implementation="point_oriented", set="particle", kind="divide", family="population",
                   title="grow and divide point cells that adhere, along a chosen direction",
                   equation=r"""$$\hat{\mathbf u}_i=\frac{(1-\eta)\,\mathbf o_i+\eta\,\mathbf z_i}{\lVert(1-\eta)\,\mathbf o_i+\eta\,\mathbf z_i\rVert},\qquad \hat{\mathbf n}_i=\frac{\mathbf x_i-\bar{\mathbf x}_{N(i)}}{\lVert\mathbf x_i-\bar{\mathbf x}_{N(i)}\rVert}$$""")
class CellDividePointOriented(CellDividePointAdhesion):
    r"""cell_divide[point_adhesion] whose daughters are placed along a chosen direction instead of a random one: a fixed axis (planar cell polarity: the
    tissue elongates along it), the cell's outward normal (a sheet thickens, a tip elongates) or a direction in the sheet's plane (a sheet stays one cell thick and spreads).
    particle -> particle: as point_adhesion.

        \hat u_i = ((1-\eta) o_i + \eta z_i) / |(1-\eta) o_i + \eta z_i|,   \hat n_i = (x_i - mean x_{N(i)}) / |x_i - mean x_{N(i)}|
        \hat u_i       the division direction of cell i (its daughters at x_i -+ (\delta/2) \hat u_i)
        o_i            the chosen direction: `axis` (orient: axis), \hat n_i (orient: normal), or a uniform random direction projected off \hat n_i (orient: tangential)
        \hat n_i       the cell's outward normal: from the mean position of the live cells within `reach` of it to the cell
        N(i)           the live cells closer than `reach` to cell i, itself excluded
        z_i            a uniform random unit vector
        \eta           `orient_noise`, from 0 (exactly along o_i) to 1 (random)

    A cell with no neighbour within `reach` (or neighbours all round it, |x_i - mean x_N| under 0.05 `reach`) has no normal: it divides randomly.
    Reference: Gong, Y. et al. (2004). Nature 430:689-693 (oriented divisions elongate the zebrafish body axis); Gillies, T. E. & Cabernard, C. (2011). Curr. Biol. 21:R806-R816 (spindle orientation and tissue shape)."""
    PARAM_UNITS = dict(CellDividePointAdhesion.PARAM_UNITS, reach='length', orient_noise='fraction', orient_after='time')
    SYMBOLS = dict(CellDividePointAdhesion.SYMBOLS, **{'\\hat u_i': 'the division direction of cell i', 'o_i': 'the chosen direction', '\\hat n_i': "the cell's outward normal",
                                                       'N(i)': 'the live cells closer than `reach`', 'z_i': 'a uniform random unit vector', '\\eta': '`orient_noise`'})
    PARAM_ROLES = dict(CellDividePointAdhesion.PARAM_ROLES, orient="axis_normal_or_tangential", axis="the_fixed_division_axis_for_orient_axis",
                       reach="radius_of_the_neighbourhood_whose_centre_defines_the_normal", orient_noise="mix_of_a_random_direction_0_to_1",
                       orient_after="world_time_from_which_the_direction_applies")
    MECHANISM_TAGS = CellDividePointAdhesion.MECHANISM_TAGS + ["oriented_division", "planar_cell_polarity"]
    REFERENCE = ("Gong, Y. et al. (2004). Nature 430:689-693; Gillies, T. E. & Cabernard, C. (2011). Curr. Biol. 21:R806-R816.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.orient = str(params.get("orient", "axis"))
        if self.orient not in ("axis", "normal", "tangential"):
            raise ValueError(f"cell_divide[point_oriented]: orient must be axis, normal or tangential, got {self.orient!r}")
        ax = torch.as_tensor([float(v) for v in params.get("axis", [0.0, 0.0, 1.0])])
        self.axis = ax / ax.norm().clamp_min(1e-12)
        self.reach = float(params.get("reach", 2.0 * self.touch))
        self.noise = float(params.get("orient_noise", 0.1))
        # `orient_after` (world time, default 0; exp05 route B, 2026-11-15): before it the daughters are placed at random as in point_adhesion, from it on
        # along the chosen direction: a cyst first grows round, then lengthens along the axis (a tube)
        self.orient_after = float(params.get("orient_after", 0.0))

    def _directions(self, lvl, X, idx, mother):
        m = int(mother.numel())
        z = torch.randn(m, X.shape[-1], generator=self._gen)
        z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
        if float(getattr(self, "_now", 0.0)) < self.orient_after:
            return z
        if self.orient == "axis":
            o = self.axis[None].expand(m, -1).clone()
        else:
            Xm, Xl = X[mother].detach().cpu().float(), X[idx].detach().cpu().float()
            W = (torch.cdist(Xm, Xl) < self.reach).float()
            W = W * (idx.cpu()[None] != mother.cpu()[:, None]).float()                 # the cell itself excluded
            cnt = W.sum(1, keepdim=True)
            nrm = Xm - (W @ Xl) / cnt.clamp_min(1.0)
            ln = nrm.norm(dim=1, keepdim=True)
            has = (cnt[:, 0] > 0) & (ln[:, 0] > 0.05 * self.reach)
            nh = nrm / ln.clamp_min(1e-12)
            if self.orient == "normal":
                o = torch.where(has[:, None], nh, z)
            else:
                r = torch.randn(m, X.shape[-1], generator=self._gen)
                t = r - (r * nh).sum(1, keepdim=True) * nh
                t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-12)
                o = torch.where(has[:, None], t, z)
        d = (1.0 - self.noise) * o + self.noise * z
        return d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)


CELL_DIE_TRACE = []          # (frame, deaths this call, live cells after) per call of cell_die[point_signal]


@register_operator("cell_die", implementation="point_signal", set="particle", kind="die", family="population",
                   title="remove point cells where a signal or crowding tells them to die",
                   equation=r"""$$P(\text{cell } i \text{ dies in } \Delta t)=\big(1-e^{-k\,\Delta t}\big)\,[\,n_i\ge n_c\,]\,[\,n_i\le n_l\,]\,[\,c_i\ge\theta_+\,]\,[\,c_i<\theta_-\,]$$""")
class CellDiePointSignal(Structural):
    r"""a point cell dies, at a rate, while every condition given holds: it is crowded (touches at least `crowd` cells), alone (touches at most
    `lonely`), its signal is high (at least `signal_above`) or low (under `signal_below`). A condition not given is always true. Death removes the cell
    (its slot goes dormant and is reused by a later division).
    particle -> particle: reads pos and the signal block; retires the dying slots (`occ = 0`).

        P(i dies in \Delta t) = (1 - e^{-k \Delta t}) [n_i >= n_c] [n_i <= n_l] [c_i >= \theta_+] [c_i < \theta_-]
        k              `rate`, the death rate of a cell that meets the conditions, per unit time
        \Delta t       the frame length
        n_i            the number of cells cell i touches (centre distance under `touch`)
        n_c            `crowd`, the fewest touched cells that make a cell crowded
        n_l            `lonely`, the most touched cells of a cell that dies alone (anoikis)
        c_i            channel `signal_channel` of block `signal_block` of cell i
        \theta_+       `signal_above`: a cell dies only at or above it
        \theta_-       `signal_below`: a cell dies only under it

    No death before time `after` (world time, the frame count times the frame length), and never below `min_cells` live cells.
    Reference: Eisenhoffer, G. T. et al. (2012). Nature 484:546-549 (crowding extrudes and kills epithelial cells); Coucouvanis, E. & Martin, G. R.
    (1995). Cell 83:279-287 (a signal kills the inner cells of the embryoid body and hollows it); Debnath, J. et al. (2002). Cell 111:29-40 (lumen by death of inner cells)."""
    PARAM_UNITS = {'rate': 'rate', 'touch': 'length', 'crowd': 'count', 'lonely': 'count', 'signal_channel': 'count', 'min_cells': 'count', 'after': 'time', 'seed': 'count'}
    SYMBOLS = {'k': '`rate`, the death rate, per unit time', '\\Delta t': 'the frame length', 'n_i': 'the number of cells cell i touches', 'n_c': '`crowd`', 'n_l': '`lonely`',
               'c_i': 'the signal of cell i', '\\theta_+': '`signal_above`', '\\theta_-': '`signal_below`'}
    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["rate"]
    MECHANISM_TAGS = ["cell_death", "apoptosis", "extrusion", "lumen", "anoikis"]
    PARAM_ROLES = {"rate": "death_rate_of_a_cell_meeting_the_conditions", "touch": "centre_distance_under_which_two_cells_touch",
                   "crowd": "fewest_touched_cells_that_make_a_cell_crowded", "lonely": "most_touched_cells_of_a_cell_that_dies_alone",
                   "signal_block": "the_signal_block", "signal_channel": "channel_of_that_block", "signal_above": "a_cell_dies_only_at_or_above_this_signal",
                   "signal_below": "a_cell_dies_only_under_this_signal", "min_cells": "never_fewer_live_cells", "after": "no_death_before_this_time", "seed": "rng_seed",
                   "types": "cell_types_that_can_die"}
    REFERENCE = ("Eisenhoffer, G. T. et al. (2012). Nature 484:546-549; Coucouvanis, E. & Martin, G. R. (1995). Cell 83:279-287; "
                 "Debnath, J. et al. (2002). Cell 111:29-40.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.rate = float(params["rate"])
        self.touch = float(params.get("touch", 1.1))
        self.crowd = params.get("crowd"); self.lonely = params.get("lonely")
        self.signal_block = params.get("signal_block"); self.signal_channel = int(params.get("signal_channel", 0))
        self.above = params.get("signal_above"); self.below = params.get("signal_below")
        if (self.above is not None or self.below is not None) and not self.signal_block:
            raise ValueError("cell_die[point_signal]: signal_above / signal_below need a signal_block")
        self.min_cells = int(params.get("min_cells", 20))
        self.after = float(params.get("after", 0.0))
        # `types: [ring]` (exp05 route B, 2026-11-15: the cells that kept a wave's mark die, the mark is a cut line): only cells of these types can die
        self.types = [str(t) for t in (params.get("types") or [])]
        self._gen = torch.Generator().manual_seed(int(params.get("seed", 0)))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dt = float(getattr(H, "dt", 1.0) or 1.0)
        fr = getattr(H, "frame", None)
        fr = 0 if fr is None else int(fr)
        if fr * dt < self.after:
            return {}
        live = lvl.occ > 0.5
        idx = live.nonzero(as_tuple=True)[0]
        if idx.numel() <= self.min_cells:
            return {}
        X = lvl.get("pos")[idx]
        ok = torch.ones(idx.numel(), dtype=torch.bool, device=X.device)
        if self.crowd is not None or self.lonely is not None:
            n = _touch_counts(X, self.touch)
            if self.crowd is not None:
                ok = ok & (n >= int(self.crowd))
            if self.lonely is not None:
                ok = ok & (n <= int(self.lonely))
        if self.signal_block:
            b = _block(lvl, self.signal_block)
            if b is None:
                raise ValueError(f"cell_die[point_signal]: `signal_block` {self.signal_block!r} is not declared on {self.at!r}")
            c = b[idx, self.signal_channel]
            if self.above is not None:
                ok = ok & (c >= float(self.above))
            if self.below is not None:
                ok = ok & (c < float(self.below))
        if self.types:
            names = list(getattr(lvl, "type_names", []) or [])
            nt = lvl.node_type.to(X.device).long()[idx]
            of_type = torch.zeros_like(ok)
            for tn in self.types:
                of_type = of_type | (nt == names.index(tn))
            ok = ok & of_type
        p = 1.0 - math.exp(-self.rate * dt)
        die = ok & (torch.rand(idx.numel(), generator=self._gen).to(X.device) < p)
        dead = idx[die]
        room = int(idx.numel()) - self.min_cells
        if dead.numel() > room:
            dead = dead[:max(room, 0)]
        if dead.numel():
            lvl.kill(dead)
        CELL_DIE_TRACE.append((fr, int(dead.numel()), int(idx.numel()) - int(dead.numel())))
        return {}


from plexus.models.base import Lateral  # noqa: E402
from plexus.operators.diffusion_reaction import _chan, _span, _emit  # noqa: E402


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="crowding",
                   title="make a signal in proportion to the cells a cell touches",
                   species=(("c", "crowding signal"),),
                   equation=r"""$$\frac{dc_i}{dt}=r\,\big(p\,n_i-k\,c_i\big)$$""")
class CellReactCrowding(Lateral):
    r"""a signal each cell makes in proportion to how many cells it touches, lost at a first-order rate: crowded cells (inside an aggregate) make
    more than cells at its surface, so with exchange between neighbours the signal is highest deep inside the tissue (a stand-in for the
    mechanical and contact signals that tell inner cells of an aggregate they are buried).
    cell -> cell: reads chem = [c] and the block `contacts_block` (written by cell_divide[point_adhesion] / [point_oriented]), emits dc/dt.

        dc_i/dt = r (p n_i - k c_i)
        c_i            the signal in cell i (channel `chan`)
        r              `rate`
        p              `production`, made per touched cell, per unit time
        n_i            the number of cells cell i touches
        k              `decay`, the first-order loss rate

    Alone, a cell settles at c_i = p n_i / k: production 0.1 and decay 1 give 0.6 at 6 touched cells (a sheet) and 1.0 at 10 (buried in a ball).
    Reference: Eisenhoffer, G. T. et al. (2012). Nature 484:546-549 (crowding sensed and acted on); Coucouvanis, E. & Martin, G. R. (1995). Cell 83:279-287."""
    PARAM_UNITS = {'production': 'rate', 'decay': 'rate', 'rate': 'rate'}
    SYMBOLS = {'c_i': 'the signal in cell i', 'r': '`rate`', 'p': '`production`, per touched cell', 'n_i': 'the number of cells cell i touches', 'k': '`decay`, the first-order loss rate'}
    N_SPECIES = 1
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["production", "decay"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem", "contacts"]; WRITES = ["chem"]
    MECHANISM_TAGS = ["reaction", "morphogen", "crowding", "decay"]
    PARAM_ROLES = {"production": "signal_made_per_touched_cell_per_time", "decay": "first_order_decay_rate", "rate": "reaction_time_scale",
                   "contacts_block": "block_holding_the_number_of_touched_cells"}
    REFERENCE = "Eisenhoffer, G. T. et al. (2012). Nature 484:546-549; Coucouvanis, E. & Martin, G. R. (1995). Cell 83:279-287."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.p = float(params["production"]); self.k = float(params["decay"])
        self.rate = float(params.get("rate", 1.0))
        self.contacts_block = str(params.get("contacts_block", "contacts"))
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        (c,) = _span(chem, self.chan, 1, type(self).__name__)
        nb = _block(lvl, self.contacts_block)
        if nb is None:
            raise ValueError(f"cell_chem_react[crowding]: set {self.at!r} declares no `{self.contacts_block}` block")
        dc = self.p * nb[:, 0].to(chem.dtype) - self.k * c
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: _emit(chem, self.chan, (dc,), self.rate, occ)}


from plexus.models.base import Exchange  # noqa: E402


@register_operator("lumen_pressure", implementation="point_cluster", family="mechanics", set="particle", kind="exchange",
                   title="push the point cells of each aggregate away from its centre, as a lumen's pressure would",
                   equation=r"""$$\mathbf v_i=p\,g_i\,\frac{\mathbf r_i}{\lVert\mathbf r_i\rVert},\qquad \mathbf r_i=(\mathbf I-\hat{\mathbf a}\hat{\mathbf a}^{\!\top})(\mathbf x_i-\bar{\mathbf x}_{K(i)}),\qquad g_i=[\theta_-\le c_i<\theta_+]$$""")
class LumenPressurePointCluster(Exchange):
    r"""a lumen's pressure on point cells: every cell of an aggregate (cells linked by touching, at least `min_cells` of them) moves outward from
    the aggregate's centre at speed `pressure` -- from its centre point (a cyst), or from the line through it along `axis` (a tube; with axis z in
    a flat slab, a ring) -- optionally only the cells whose signal lies in a band (the epithelial cells that pump the lumen).
    particle -> particle, one set: reads pos (and the signal block), emits a velocity on the live cells.

        v_i = p g_i r_i / |r_i|,   r_i = (I - a a^T)(x_i - mean x_{K(i)}),   g_i = [\theta_- <= c_i < \theta_+]
        v_i            the velocity added to cell i
        p              `pressure`, world length per unit time
        K(i)           the aggregate of cell i (the connected cells of the touch graph, centre distance under `touch`)
        a              `axis`, a unit vector; none = a point centre (I - a a^T becomes I)
        c_i            channel `signal_channel` of block `signal_block`; g_i = 1 when no signal is given
        \theta_-, \theta_+   `signal_min`, `signal_max`

    The aggregates are recomputed every `every` calls (the touch graph changes slowly); a cell within 0.05 of the centre is not pushed.
    Reference: Chan, C. J. et al. (2019). Nature 571:112-116 (hydraulic pressure of the lumen sizes the blastocyst); Navis, A. & Nelson, C. M. (2016).
    Semin. Cell Dev. Biol. 55:139-147 (pressure from the lumen in tube formation)."""
    PARAM_UNITS = {'pressure': 'velocity', 'touch': 'length', 'min_cells': 'count', 'every': 'count', 'signal_channel': 'count'}
    SYMBOLS = {'v_i': 'the velocity added to cell i', 'p': '`pressure`', 'K(i)': 'the aggregate of cell i', 'a': '`axis`', 'c_i': 'the signal of cell i',
               '\\theta_-, \\theta_+': '`signal_min`, `signal_max`'}
    EMIT = "velocity"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["pressure"]
    MECHANISM_TAGS = ["lumen_pressure", "lumen", "cyst", "tube"]
    PARAM_ROLES = {"pressure": "outward_speed_of_a_cell", "touch": "centre_distance_under_which_two_cells_touch", "min_cells": "smallest_aggregate_pushed",
                   "axis": "the_line_the_push_is_radial_to_none_is_a_point", "every": "calls_between_two_recomputations_of_the_aggregates",
                   "signal_block": "the_signal_block", "signal_channel": "channel_of_that_block", "signal_min": "a_cell_pushes_only_at_or_above_this_signal",
                   "signal_max": "a_cell_pushes_only_under_this_signal"}
    REFERENCE = "Chan, C. J. et al. (2019). Nature 571:112-116; Navis, A. & Nelson, C. M. (2016). Semin. Cell Dev. Biol. 55:139-147."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.p = float(params["pressure"])
        self.touch = float(params.get("touch", 1.1))
        self.min_cells = int(params.get("min_cells", 8))
        ax = params.get("axis")
        self.axis = None if ax is None else torch.as_tensor([float(v) for v in ax]) / max(float(torch.as_tensor([float(v) for v in ax]).norm()), 1e-12)
        self.every = int(params.get("every", 25))
        self.signal_block = params.get("signal_block"); self.signal_channel = int(params.get("signal_channel", 0))
        self.smin = params.get("signal_min"); self.smax = params.get("signal_max")
        self._calls = 0
        self._lab = None

    def _labels(self, X, live):
        """[n] aggregate label of every slot (-1: dormant or in an aggregate under min_cells), from the touch graph's connected components."""
        import numpy as np
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        idx = live.nonzero(as_tuple=True)[0]
        Xl = X[idx].detach()
        A = _touch_pairs(Xl, self.touch).T.cpu().numpy()
        m = len(idx)
        _, lab = connected_components(coo_matrix((np.ones(len(A)), (A[:, 0], A[:, 1])), shape=(m, m)), directed=False)
        sizes = np.bincount(lab)
        lab = np.where(sizes[lab] >= self.min_cells, lab, -1)
        out = torch.full((X.shape[0],), -1, dtype=torch.long)
        out[idx.cpu()] = torch.as_tensor(lab, dtype=torch.long)
        return out.to(X.device)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        X = lvl.get("pos")
        live = lvl.occ > 0.5
        if self._lab is None or self._calls % self.every == 0 or self._lab.shape[0] != X.shape[0]:
            self._lab = self._labels(X, live)
        self._calls += 1
        lab = torch.where(live, self._lab, torch.full_like(self._lab, -1))
        ok = lab >= 0
        V = torch.zeros_like(X)
        if not bool(ok.any()):
            return {self.at: V}
        k = int(lab.max()) + 1
        li = lab.clamp_min(0)
        w = ok.to(X.dtype)
        S = torch.zeros(k, X.shape[1], device=X.device, dtype=X.dtype).index_add_(0, li, X * w[:, None])
        N = torch.zeros(k, device=X.device, dtype=X.dtype).index_add_(0, li, w)
        C = S / N.clamp_min(1.0)[:, None]
        r = X - C[li]
        if self.axis is not None:
            a = self.axis.to(X.device, X.dtype)
            r = r - (r @ a)[:, None] * a[None]
        ln = r.norm(dim=1, keepdim=True)
        g = ok & (ln[:, 0] > 0.05)
        if self.signal_block:
            b = _block(lvl, self.signal_block)
            if b is None:
                raise ValueError(f"lumen_pressure[point_cluster]: `signal_block` {self.signal_block!r} is not declared on {self.at!r}")
            c = b[:, self.signal_channel]
            if self.smin is not None:
                g = g & (c >= float(self.smin))
            if self.smax is not None:
                g = g & (c < float(self.smax))
        V = self.p * r / ln.clamp_min(1e-9) * g.to(X.dtype)[:, None]
        if mask is not None:
            V = V * mask[:, None].to(X.dtype)
        return {self.at: V}


@register_operator("pair_potential", implementation="polar_nissen", family="interaction", set="particle", kind="exchange",
                   title="attract point cells by how their apical-basal and planar polarities line up (Nissen 2018)",
                   equation=r"""$$V_{ij}=e^{-r_{ij}/\ell}-S_{ij}\,e^{-r_{ij}/(\beta\ell)},\quad S_{ij}=\lambda_1(\mathbf p_i\times\hat{\mathbf r})\cdot(\mathbf p_j\times\hat{\mathbf r})+\lambda_2(\mathbf p_i\times\mathbf q_i)\cdot(\mathbf p_j\times\mathbf q_j)+\lambda_3(\mathbf q_i\times\hat{\mathbf r})\cdot(\mathbf q_j\times\hat{\mathbf r})$$""")
class PairPotentialPolarNissen(Exchange):
    r"""polarised point cells (Nissen, Ronhild, Trusina & Sneppen 2018): two neighbouring cells repel at short range and attract at long range, the attraction
    weighted by S, how their polarities line up. Apical-basal polarity p alone (lambda_1 = 1) makes a cell attract the neighbours BESIDE it (p
    perpendicular to the line between them) and not those above or below: cells form one-cell-thick sheets that close into hollow cysts. Planar
    polarity q (lambda_2, lambda_3) adds an in-sheet direction: sheets roll into tubes. Positions and both polarities follow the gradient of the energy.
    particle -> particle, one set: reads pos and the blocks `p_block`, `q_block` (width 3, integration first_order); emits the velocity of the
    positions and the rates of p and q (keys (set, p_block), (set, q_block)).

        V_ij = exp(-r/l) - S_ij exp(-r/(beta l)),   S_ij = l1 (p_i x r)(p_j x r) + l2 (p_i x q_i)(p_j x q_j) + l3 (q_i x r)(q_j x r)
        dx_i/dt = -m dV_i/dx_i,   dp_i/dt = -k dV_i/dp_i (lambda_2 off for p, as in the paper),   dq_i/dt = -k dV_i/dq_i
        r              the distance between the cells, r its unit vector
        l              the length unit: `spacing` / (ln(beta) / (1 - 1/beta)), so two cells with S = 1 rest `spacing` apart
        beta           `beta`, the attraction's range over the repulsion's (5)
        p_i, q_i       the cell's apical-basal and planar polarities, unit vectors (normalised where read)
        l1, l2, l3     `lambda`, per cell; a cell whose signal (`signal_block` channel `signal_channel`) is at least `signal_min` takes `lambda_active`
                       and its rest spacing `spacing_active` (a contraction) and its wedging `alpha_active`; a pair takes the mean of its two cells'
        alpha          wedging (apical constriction): in the first term p_i is replaced by (p_i - alpha r)/|.| and p_j by (p_j + alpha r)/|.|, so two
                       cells prefer polarities tilted toward each other and the sheet a curvature; the sign sets which way it bends
        m, k           `mobility`, `polarity_rate`

    Neighbours (Nissen's Voronoi rule): i and j interact only when they are closer than `cutoff` x `spacing` and no third cell is nearer the
    midpoint of i and j than they are; recomputed every `every` calls and whenever the live cells change. A cell with no polarity yet (|p| under
    1e-6: a seeded set) gets on the first call a unit p, random or (`p_init: radial`) pointing from the live cells' centre to it, and a unit q
    perpendicular to p, random, (`q_init: [x, y, z]`) that direction, or (`q_init: azimuthal`) round the z axis through the live cells' centre,
    its part along p removed.
    Reference: Nissen, S. B. et al. (2018). eLife 7:e38407 (polarities bridge cells to sheets, cysts and tubes)."""
    PARAM_UNITS = {'spacing': 'length', 'beta': 'dimensionless', 'mobility': 'rate', 'polarity_rate': 'rate', 'cutoff': 'dimensionless', 'every': 'count',
                   'signal_channel': 'count', 'seed': 'count'}
    SYMBOLS = {'V_ij': 'the pair energy', 'S_ij': 'how the polarities line up', 'l': 'the length unit', 'beta': '`beta`', 'p_i, q_i': 'the polarities',
               'l1, l2, l3': '`lambda`', 'm, k': '`mobility`, `polarity_rate`'}
    EMIT = "velocity"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True                      # writes the first polarities of a seeded set
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["polarity", "apical_basal_polarity", "planar_cell_polarity", "epithelium", "cyst", "tube"]
    PARAM_ROLES = {"lambda": "weights_of_the_three_polarity_terms", "lambda_active": "the_weights_of_a_cell_whose_signal_is_high",
                   "signal_block": "the_signal_block", "signal_channel": "channel_of_that_block", "signal_min": "signal_at_or_above_which_a_cell_takes_lambda_active",
                   "spacing": "rest_distance_of_two_aligned_cells", "beta": "range_of_the_attraction_over_the_repulsion", "mobility": "speed_per_unit_force",
                   "polarity_rate": "turning_rate_of_the_polarities_per_unit_torque", "cutoff": "largest_neighbour_distance_in_spacings",
                   "spacing_active": "rest_spacing_of_a_cell_whose_signal_is_high", "alpha_active": "wedging_of_a_cell_whose_signal_is_high", "alpha": "wedging_of_every_other_cell",
                   "alpha_types": "wedging_per_cell_type", "alpha_q_types": "wedging_along_the_planar_polarity_only_per_cell_type", "alpha_chem": "wedging_as_a_weighted_sum_of_a_block's_channels", "lambda_types": "polarity_weights_per_cell_type", "spacing_types": "rest_spacing_per_cell_type", "every": "calls_between_two_neighbour_searches", "p_block": "block_of_the_apical_basal_polarity", "q_block": "block_of_the_planar_polarity", "seed": "rng_seed",
                   "p_init": "random_or_radial_first_apical_basal_polarity", "q_init": "random_or_a_direction_for_the_first_planar_polarity"}
    REFERENCE = "Nissen, S. B. et al. (2018). eLife 7:e38407."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.lam = [float(v) for v in params.get("lambda", [1.0, 0.0, 0.0])]
        la = params.get("lambda_active")
        self.lam_act = None if la is None else [float(v) for v in la]
        self.signal_block = params.get("signal_block"); self.signal_channel = int(params.get("signal_channel", 0))
        self.signal_min = float(params.get("signal_min", 0.0))
        self.beta = float(params.get("beta", 5.0))
        self.spacing = float(params.get("spacing", 1.0))
        self.dstar = math.log(self.beta) / (1.0 - 1.0 / self.beta)      # the rest distance of two aligned cells in units of l
        self.ell = self.spacing / self.dstar
        sa = params.get("spacing_active")                                  # the rest spacing of a cell whose signal is high (an excited cell contracts)
        self.spacing_act = None if sa is None else float(sa)
        aa = params.get("alpha_active")                                    # the wedging of a cell whose signal is high (apical constriction: the sheet curves there)
        self.alpha_act = None if aa is None else float(aa)
        self.alpha0 = float(params.get("alpha", 0.0))                     # the wedging of every other cell (0: none)
        # BY TYPE (exp05 routes A and B, 2026-11-15: "cells keep a mark of the signal that passed through them and wedge by that mark"; the mark is the
        # cell's type, written by cell_fate): `alpha_types: {tip: 0.5}`, `lambda_types: {stalk: [0.41, 0.5, 0.09]}`, `spacing_types: {ring: 0.6}` give the
        # cells of a type their own wedging, polarity weights and rest spacing, over the signal's gate and the set's own values
        self.alpha_types = {str(k): float(v) for k, v in (params.get("alpha_types") or {}).items()}
        self.lambda_types = {str(k): [float(x) for x in v] for k, v in (params.get("lambda_types") or {}).items()}
        self.spacing_types = {str(k): float(v) for k, v in (params.get("spacing_types") or {}).items()}
        # WEDGING ALONG THE PLANAR POLARITY ONLY (exp05 route A, 2026-11-15): `alpha_q_types: {stalk: -0.5}` tilts a pair of that type by alpha times
        # (r . q)^2, i.e. fully when the pair lies along the cells' planar polarity q and not at all across it: the sheet bends round ONE axis, a tube
        # wall (isotropic wedging, `alpha_types`, bends it round both, a dome). With polarity_guide[signal_slope] q runs round a spot, so the wall
        # round a spot curls into a cylinder whose axis points away from the spot.
        self.alpha_q_types = {str(k): float(v) for k, v in (params.get("alpha_q_types") or {}).items()}
        # `alpha_chem: {block: chem, weights: [w0, w1, w2], after: 540}` (exp05 route A with rock-paper-scissors, 2026-11-15): from world time `after` on, a
        # cell's wedging is `alpha` plus the weighted sum of its block's first channels, sum_k w_k c_k: the sheet's curvature follows the three
        # chemicals continuously, with no mark. `alpha_types` still overrides it for the cells of a listed type.
        ac = params.get("alpha_chem")
        self.alpha_chem = None if not ac else dict(block=str(ac.get("block", "chem")), weights=[float(v) for v in ac["weights"]], after=float(ac.get("after", 0.0)))
        self.mob = float(params.get("mobility", 10.0))
        self.krot = float(params.get("polarity_rate", 2.0))
        self.cutoff = float(params.get("cutoff", 2.0))
        self.every = int(params.get("every", 5))
        self.p_block = str(params.get("p_block", "pol_p")); self.q_block = str(params.get("q_block", "pol_q"))
        self.p_init = str(params.get("p_init", "random"))
        if self.p_init not in ("random", "radial"):
            raise ValueError(f"pair_potential[polar_nissen]: p_init must be random or radial, got {self.p_init!r}")
        qi = params.get("q_init")
        self.q_init = None if qi in (None, "random") else ("azimuthal" if qi == "azimuthal" else [float(v) for v in qi])
        self._gen = torch.Generator().manual_seed(int(params.get("seed", 0)))
        self._pairs, self._key, self._calls = None, None, 0

    def _init_polarities(self, lvl, idx):
        p, q = _block(lvl, self.p_block), _block(lvl, self.q_block)
        if p is None or q is None:
            raise ValueError(f"pair_potential[polar_nissen]: set {self.at!r} needs blocks {self.p_block!r} and {self.q_block!r} (width 3)")
        bad = idx[p[idx].norm(dim=1) < 1e-6]
        if bad.numel() == 0:
            return
        P = p.clone(); Q = q.clone()
        a = torch.randn(int(bad.numel()), 3, generator=self._gen).to(P.device, P.dtype)
        if self.p_init == "radial":                                   # p from the live cells' centre to the cell: an outward-facing (apical-out) cyst
            X = lvl.get("pos")
            a = X[bad] - X[idx].mean(0, keepdim=True) + 1e-3 * a
        a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-12)
        b = torch.randn(int(bad.numel()), 3, generator=self._gen).to(P.device, P.dtype)
        if self.q_init == "azimuthal":                                # q round the z axis through the live cells' centre (a ring of planar polarity)
            X = lvl.get("pos")
            rr = X[bad] - X[idx].mean(0, keepdim=True)
            b = torch.cross(torch.tensor([0.0, 0.0, 1.0], device=P.device, dtype=P.dtype)[None].expand_as(rr), rr, dim=1) + 1e-3 * b
        elif self.q_init is not None:                                 # every q along one direction (a body axis), as far as it lies off p
            b = torch.as_tensor(self.q_init, device=P.device, dtype=P.dtype)[None].expand_as(a) + 1e-3 * b
        b = b - (b * a).sum(1, keepdim=True) * a
        b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-12)
        P[bad], Q[bad] = a, b
        _write(lvl, self.p_block, P); _write(lvl, self.q_block, Q)

    def _neighbours(self, Xl):
        """(a, b) local index pairs, a < b: within `cutoff` spacings and no third cell nearer their midpoint than they are. A third cell nearer the
        midpoint than d/2 lies within d of a, so only a's own neighbours (within the cutoff) are tested."""
        n = Xl.shape[0]
        e = _touch_pairs(Xl, self.cutoff * self.spacing)
        if e.numel() == 0:
            return e[0], e[1]
        o = torch.argsort(e[0] * n + e[1])
        r, c = e[0][o], e[1][o]
        deg = torch.bincount(r, minlength=n)
        start = torch.cumsum(deg, 0) - deg
        K = int(deg.max())
        nb = torch.full((n, K), -1, dtype=torch.long, device=Xl.device)
        nb[r, torch.arange(r.numel(), device=Xl.device) - start[r]] = c
        sel = r < c
        a, b = r[sel], c[sel]
        keep = torch.ones(a.numel(), dtype=torch.bool, device=Xl.device)
        for s0 in range(0, a.numel(), 65536):
            aa, bb = a[s0:s0 + 65536], b[s0:s0 + 65536]
            M = 0.5 * (Xl[aa] + Xl[bb])
            h2 = 0.25 * ((Xl[bb] - Xl[aa]) ** 2).sum(1)
            kk = nb[aa]                                              # [B, K] a's neighbours
            dk = ((Xl[kk.clamp_min(0)] - M[:, None]) ** 2).sum(2)
            dk = torch.where((kk < 0) | (kk == bb[:, None]), torch.full_like(dk, float("inf")), dk)
            keep[s0:s0 + 65536] = dk.min(1).values >= h2 * (1.0 - 1e-6)
        return a[keep], b[keep]

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        live = lvl.occ > 0.5
        idx = live.nonzero(as_tuple=True)[0]
        X = lvl.get("pos")
        out = torch.zeros_like(X)
        if idx.numel() < 2:
            return {self.at: out}
        self._init_polarities(lvl, idx)
        Xl = X[idx].detach()
        key = (int(idx.numel()), int(idx.sum()))
        if self._pairs is None or key != self._key or self._calls % self.every == 0:
            self._pairs, self._key = self._neighbours(Xl), key
        self._calls += 1
        a, b = self._pairs
        P0, Q0 = _block(lvl, self.p_block)[idx].detach(), _block(lvl, self.q_block)[idx].detach()
        lam = torch.as_tensor(self.lam, device=X.device, dtype=X.dtype)[None].expand(idx.numel(), 3).clone()
        if self.lam_act is not None and self.signal_block:
            sb = _block(lvl, self.signal_block)
            act = sb[idx, self.signal_channel] >= self.signal_min
            lam[act] = torch.as_tensor(self.lam_act, device=X.device, dtype=X.dtype)
        by_type = bool(self.alpha_types or self.lambda_types or self.spacing_types or self.alpha_q_types)
        if by_type:
            names = list(getattr(lvl, "type_names", []) or [])
            nt = lvl.node_type.to(X.device).long()[idx]
            for tn, v in self.lambda_types.items():
                lam[nt == names.index(tn)] = torch.as_tensor(v, device=X.device, dtype=X.dtype)
        lp = 0.5 * (lam[a] + lam[b])
        alpha = None
        chem_on = False
        if self.alpha_chem is not None:
            fr_ = getattr(H, "frame", None)
            chem_on = (0 if fr_ is None else int(fr_)) * float(getattr(H, "dt", 1.0) or 1.0) >= self.alpha_chem["after"]
        if (self.alpha_act is not None and self.signal_block) or self.alpha0 != 0.0 or self.alpha_types or chem_on:
            al = torch.full((idx.numel(),), self.alpha0, device=X.device, dtype=X.dtype)
            if self.alpha_act is not None and self.signal_block:
                al[_block(lvl, self.signal_block)[idx, self.signal_channel] >= self.signal_min] = self.alpha_act
            if chem_on:
                cb = _block(lvl, self.alpha_chem["block"])
                if cb is None:
                    raise ValueError(f"pair_potential[polar_nissen]: `alpha_chem.block` {self.alpha_chem['block']!r} is not declared on {self.at!r}")
                wk = torch.as_tensor(self.alpha_chem["weights"], device=X.device, dtype=X.dtype)
                al = al + (cb[idx, : wk.numel()].detach().to(X.dtype) * wk[None]).sum(1)
            for tn, v in self.alpha_types.items():
                al[nt == names.index(tn)] = v
            alpha = 0.5 * (al[a] + al[b])
        alq = None
        if self.alpha_q_types:
            aq = torch.zeros(idx.numel(), device=X.device, dtype=X.dtype)
            for tn, v in self.alpha_q_types.items():
                aq[nt == names.index(tn)] = v
            alq = 0.5 * (aq[a] + aq[b])
        ell = self.ell
        if (self.spacing_act is not None and self.signal_block) or self.spacing_types:
            spc = torch.full((idx.numel(),), self.spacing, device=X.device, dtype=X.dtype)
            if self.spacing_act is not None and self.signal_block:
                spc[_block(lvl, self.signal_block)[idx, self.signal_channel] >= self.signal_min] = self.spacing_act
            for tn, v in self.spacing_types.items():
                spc[nt == names.index(tn)] = v
            ell = 0.5 * (spc[a] + spc[b]) / self.dstar                     # a pair rests at the mean of its two cells' spacings
        with torch.enable_grad():
            Xg = Xl.clone().requires_grad_(True); Pg = P0.clone().requires_grad_(True); Qg = Q0.clone().requires_grad_(True)
            r = Xg[b] - Xg[a]
            d = r.norm(dim=1).clamp_min(1e-9)
            rh = r / d[:, None]
            ph = Pg / Pg.norm(dim=1, keepdim=True).clamp_min(1e-9)
            qh = Qg / Qg.norm(dim=1, keepdim=True).clamp_min(1e-9)
            phd = ph.detach()
            pa, pb = ph[a], ph[b]
            if alq is not None:                                          # wedging along q only: alpha x (r . q)^2, q held (the wedge does not turn q)
                qd = qh.detach()
                along = alq * 0.5 * ((rh * qd[a]).sum(1) ** 2 + (rh * qd[b]).sum(1) ** 2)
                alpha = along if alpha is None else alpha + along
            if alpha is not None:                                        # WEDGING: each polarity tilted toward the other cell by alpha, so the pair prefers a bent sheet
                pa = pa - alpha[:, None] * rh; pa = pa / pa.norm(dim=1, keepdim=True).clamp_min(1e-9)
                pb = pb + alpha[:, None] * rh; pb = pb / pb.norm(dim=1, keepdim=True).clamp_min(1e-9)
            S1 = (torch.cross(pa, rh, dim=1) * torch.cross(pb, rh, dim=1)).sum(1)
            S2 = (torch.cross(phd[a], qh[a], dim=1) * torch.cross(phd[b], qh[b], dim=1)).sum(1)    # lambda_2 does not turn p (the paper's rule)
            S3 = (torch.cross(qh[a], rh, dim=1) * torch.cross(qh[b], rh, dim=1)).sum(1)
            S = lp[:, 0] * S1 + lp[:, 1] * S2 + lp[:, 2] * S3
            u = d / ell
            E = (torch.exp(-u) - S * torch.exp(-u / self.beta)).sum()
            gX, gP, gQ = torch.autograd.grad(E, [Xg, Pg, Qg], allow_unused=True)
        gP = torch.zeros_like(P0) if gP is None else gP
        gQ = torch.zeros_like(Q0) if gQ is None else gQ
        out[idx] = -self.mob * gX
        dp = torch.zeros_like(X); dq = torch.zeros_like(X)
        dp[idx] = -self.krot * gP
        dq[idx] = -self.krot * gQ
        if mask is not None:
            out = out * mask[:, None].to(X.dtype)
        return {self.at: out, (self.at, self.p_block): dp, (self.at, self.q_block): dq}


@register_operator("polarity_guide", implementation="signal_slope", family="interaction", set="particle", kind="exchange",
                   title="turn each cell's planar polarity along the level lines of a signal (round a spot of it)",
                   equation=r"""$$\frac{d\mathbf q_i}{dt}=k\,(\mathbf t_i-\hat{\mathbf q}_i),\quad \mathbf t_i=\frac{\hat{\mathbf p}_i\times\mathbf g_i}{|\hat{\mathbf p}_i\times\mathbf g_i|},\quad \mathbf g_i=\sum_{j\sim i}(c_j-c_i)\frac{\hat{\mathbf r}_{ij}}{r_{ij}}$$""")
class PolarityGuideSignalSlope(Exchange):
    r"""a signal's slope orients planar polarity (exp05 route A, 2026-11-15): each cell turns its planar polarity q toward t, the direction in its own
    sheet plane that runs ALONG the signal's level lines, i.e. round a spot of the signal. With `pair_potential[polar_nissen]`'s lambda_3 the cells then
    converge along q: the wall round a spot keeps a narrow girth and lengthens away from the spot (a tube), where an unguided wall widens into a dome.
    Only cells of `types` (all when absent) whose slope is at least `min_slope` are turned; the others keep following their neighbours.
    particle -> particle, one set: reads pos, `p_block`, `q_block` and the signal; emits d(q)/dt only.

        dq_i/dt = k (t_i - q_i/|q_i|),  t_i = p_i x g_i / |p_i x g_i|,  g_i = sum_j (c_j - c_i) r_ij / |r_ij|^2
        k              `rate`, per unit time
        c_i            the signal of cell i (`signal_block`, `signal_channel`)
        g_i            its slope over the cells i touches (centre distance under `touch`), signal per length
        p_i            the apical-basal polarity, unit vector
        r_ij           from cell i to cell j

    Reference: Nissen, S. B. et al. (2018). eLife 7:e38407 (the polarities); Gao, B. et al. (2011). Dev. Cell 20:163-176 (a signal's gradient orients planar polarity)."""
    PARAM_UNITS = {'rate': 'rate', 'touch': 'length', 'min_slope': 'per_length', 'signal_channel': 'count'}
    SYMBOLS = {'q_i': 'the planar polarity of cell i', 't_i': "the direction along the signal's level lines in the cell's sheet plane", 'g_i': "the signal's slope at cell i",
               'p_i': 'the apical-basal polarity', 'c_i': 'the signal of cell i', 'k': '`rate`'}
    EMIT = "velocity"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["signal_block"]
    MECHANISM_TAGS = ["polarity", "planar_cell_polarity", "gradient_sensing", "tube"]
    PARAM_ROLES = {"rate": "turning_rate_of_the_planar_polarity_toward_the_level_lines", "signal_block": "block_of_the_signal", "signal_channel": "channel_of_that_block",
                   "touch": "centre_distance_under_which_two_cells_touch", "min_slope": "smallest_in_plane_slope_of_a_guided_cell", "types": "cell_types_that_are_guided",
                   "p_block": "block_of_the_apical_basal_polarity", "q_block": "block_of_the_planar_polarity"}
    REFERENCE = "Nissen, S. B. et al. (2018). eLife 7:e38407; Gao, B. et al. (2011). Dev. Cell 20:163-176."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.rate = float(params.get("rate", 1.0))
        self.signal_block = str(params["signal_block"]); self.signal_channel = int(params.get("signal_channel", 0))
        self.touch = float(params.get("touch", 1.1))
        self.min_slope = float(params.get("min_slope", 0.02))
        self.types = [str(t) for t in (params.get("types") or [])]
        self.p_block = str(params.get("p_block", "pol_p")); self.q_block = str(params.get("q_block", "pol_q"))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        X = lvl.get("pos")
        p, q, sb = _block(lvl, self.p_block), _block(lvl, self.q_block), _block(lvl, self.signal_block)
        if p is None or q is None or sb is None:
            raise ValueError(f"polarity_guide[signal_slope]: set {self.at!r} needs blocks {self.p_block!r}, {self.q_block!r} and {self.signal_block!r}")
        out = torch.zeros_like(X); dq = torch.zeros_like(X)
        idx = (lvl.occ > 0.5).nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            return {self.at: out, (self.at, self.q_block): dq}
        Xl = X[idx].detach()
        c = sb[idx, self.signal_channel].detach()
        e = _touch_pairs(Xl, self.touch)                                   # both directions: (i, j) and (j, i)
        r = Xl[e[1]] - Xl[e[0]]
        d2 = (r * r).sum(1).clamp_min(1e-12)
        g = torch.zeros_like(Xl).index_add_(0, e[0], (c[e[1]] - c[e[0]])[:, None] * r / d2[:, None])
        ph = p[idx] / p[idx].norm(dim=1, keepdim=True).clamp_min(1e-9)
        t = torch.cross(ph, g, dim=1)                                      # in the sheet plane, perpendicular to the slope: along the level line
        tn = t.norm(dim=1)
        on = tn >= self.min_slope
        if self.types:
            names = list(getattr(lvl, "type_names", []) or [])
            nt = lvl.node_type.to(X.device).long()[idx]
            ok = torch.zeros_like(on)
            for tname in self.types:
                ok = ok | (nt == names.index(tname))
            on = on & ok
        qh = q[idx] / q[idx].norm(dim=1, keepdim=True).clamp_min(1e-9)
        dq[idx] = self.rate * on.to(X.dtype)[:, None] * (t / tn.clamp_min(1e-12)[:, None] - qh)
        if mask is not None:
            dq = dq * mask[:, None].to(X.dtype)
        return {self.at: out, (self.at, self.q_block): dq}


@register_operator("lumen_pressure", implementation="polarity", family="mechanics", set="particle", kind="exchange",
                   title="push each polarised point cell along its apical-basal polarity, as the lumen it faces would",
                   equation=r"""$$\mathbf v_i=-P\,\hat{\mathbf p}_i\,[n_i\ge n_{\min}]$$""")
class LumenPressurePolarity(Exchange):
    r"""a lumen's pressure on a polarised sheet of point cells: each cell that touches at least `min_contacts` others moves at speed `pressure`
    AWAY from its apical side (p points to the apical side; a sheet whose apical side faces the lumen is pushed outward, the lumen inflates).
    Local: no aggregate centre, so a cyst, a tube or a branched network is inflated alike. A negative `pressure` pushes toward the apical side
    (a sheet whose apical side faces out, `p_init: radial`, inflates with pressure < 0).
    particle -> particle, one set: reads pos and `p_block`; emits a velocity.

        v_i = -P p_i [n_i >= n_min]
        P              `pressure`, world length per unit time
        p_i            the cell's apical-basal polarity, unit vector (block `p_block`)
        n_i            the cells it touches (centre distance under `touch`)

    Reference: Chan, C. J. et al. (2019). Nature 571:112-116; Nissen, S. B. et al. (2018). eLife 7:e38407 (the polarity)."""
    PARAM_UNITS = {'pressure': 'velocity', 'touch': 'length', 'min_contacts': 'count'}
    SYMBOLS = {'v_i': 'the velocity added to cell i', 'P': '`pressure`', 'p_i': "the cell's apical-basal polarity", 'n_i': 'the cells it touches'}
    EMIT = "velocity"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["pressure"]
    MECHANISM_TAGS = ["lumen_pressure", "lumen", "polarity", "cyst", "tube"]
    PARAM_ROLES = {"pressure": "speed_of_a_cell_away_from_its_apical_side", "p_block": "block_of_the_apical_basal_polarity",
                   "touch": "centre_distance_under_which_two_cells_touch", "min_contacts": "fewest_touched_cells_of_a_pushed_cell"}
    REFERENCE = "Chan, C. J. et al. (2019). Nature 571:112-116; Nissen, S. B. et al. (2018). eLife 7:e38407."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.P = float(params["pressure"])
        self.p_block = str(params.get("p_block", "pol_p"))
        self.touch = float(params.get("touch", 1.1))
        self.min_contacts = int(params.get("min_contacts", 3))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        X = lvl.get("pos")
        p = _block(lvl, self.p_block)
        if p is None:
            raise ValueError(f"lumen_pressure[polarity]: set {self.at!r} has no block {self.p_block!r}")
        live = lvl.occ > 0.5
        idx = live.nonzero(as_tuple=True)[0]
        V = torch.zeros_like(X)
        if idx.numel() < 2:
            return {self.at: V}
        Xl = X[idx]
        n = _touch_counts(Xl, self.touch)
        ph = p[idx] / p[idx].norm(dim=1, keepdim=True).clamp_min(1e-9)
        V[idx] = -self.P * ph * (n >= self.min_contacts).to(X.dtype)[:, None]
        if mask is not None:
            V = V * mask[:, None].to(X.dtype)
        return {self.at: V}


from plexus.operators.seed_ops import SeedPositions  # noqa: E402


@register_operator("seed_positions", family="seed", set="cell", kind="seed", model="shell",
                   title="place cell centres evenly on a sphere's surface",
                   equation=r"""$$\mathbf x_i=\mathbf c+R\,\big(\sqrt{1-z_i^2}\cos\phi_i,\ \sqrt{1-z_i^2}\sin\phi_i,\ z_i\big),\quad z_i=1-\tfrac{2i+1}{n},\quad \phi_i=i\,\pi(3-\sqrt5),\quad R=a\sqrt{\tfrac{\sqrt3\,n}{8\pi}}$$""")
class SeedPositionsShell(SeedPositions):
    r"""the centres of a closed one-cell-thick shell: n cells on a sphere's surface, evenly spread along the golden-angle (Fibonacci) spiral, the
    radius set so that neighbours sit `spacing` apart (the area of a hexagonal packing), or given as `radius`.
    set -> set: writes `pos` for the set's n live elements.

        x_i = c + R (sqrt(1 - z_i^2) cos phi_i, sqrt(1 - z_i^2) sin phi_i, z_i),  z_i = 1 - (2i + 1)/n,  phi_i = i pi (3 - sqrt 5),  R = a sqrt(sqrt3 n / (8 pi))
        x_i            the position of cell i (i = 0 .. n-1)
        c              `centre` (a point, or a set whose live centroid is read; default the world box's centre)
        R              the radius: `radius` when given, else from `spacing` a
        a              `spacing`, the distance between neighbours on the shell, world length

    Reference: none: an initial condition (the closed shell Nissen et al. 2018 start their cysts from). Plexus (this work)."""
    PARAM_UNITS = {'spacing': 'length', 'radius': 'length'}
    SYMBOLS = {'x_i': 'the position of cell i', 'c': '`centre`', 'R': 'the radius', 'a': '`spacing`'}
    REQUIRES_PARAMS = []
    MAY_MUTATE_INTEGRATED_STATE = True
    PARAM_ROLES = {"spacing": "neighbour_distance_on_the_shell", "radius": "shell_radius_overrides_spacing", "centre": "placement_centre_point_or_set_centroid", "seed": "rng_seed"}

    def __init__(self, params, device="cpu"):
        p2 = dict(params); p2.setdefault("mode", "ball")
        super().__init__(p2, device)
        self.spacing = float(params.get("spacing", 1.0))
        self.R = params.get("radius")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        p0, p1 = lvl.state_schema["pos"]
        n = int(lvl.occ.sum().item())
        if p1 - p0 != 3:
            raise ValueError("seed_positions[shell] is a 3D placement")
        i = torch.arange(n, dtype=torch.float64)
        z = 1.0 - (2.0 * i + 1.0) / n
        phi = i * math.pi * (3.0 - math.sqrt(5.0))
        rho = torch.sqrt((1.0 - z * z).clamp_min(0.0))
        R = float(self.R) if self.R is not None else self.spacing * math.sqrt(math.sqrt(3.0) * n / (8.0 * math.pi))
        c = (self._centre(H, 3, "cpu").double().cpu() if self.centre is not None
             else 0.5 * torch.as_tensor(H.world_size, dtype=torch.float64).cpu()[:3])      # world_size may sit on the GPU (packed_ball's trap)
        pos = c + R * torch.stack([rho * torch.cos(phi), rho * torch.sin(phi), z], 1)
        st = lvl.state.clone()
        st[:n, p0:p1] = pos.to(device=lvl.state.device, dtype=st.dtype)
        lvl.state = st
        return {}


@register_operator("lumen_pressure", implementation="volume", family="mechanics", set="particle", kind="exchange",
                   title="inflate each aggregate of point cells toward the radius of a closed monolayer of its cells",
                   equation=r"""$$\mathbf v_i=k\,\max\!\Big(0,\frac{R^0_K-R_K}{R^0_K}\Big)\frac{\mathbf x_i-\bar{\mathbf x}_K}{\lVert\mathbf x_i-\bar{\mathbf x}_K\rVert},\qquad R^0_K=a\sqrt{\frac{\sqrt3\,N_K}{8\pi}},\qquad R_K=\langle\lVert\mathbf x_j-\bar{\mathbf x}_K\rVert\rangle_{j\in K}$$""")
class LumenPressureVolume(LumenPressurePointCluster):
    r"""a lumen that regulates itself: each aggregate K (cells linked by touching, at least `min_cells`) is pushed outward from its centre only while
    its mean radius R_K is below R^0_K, the radius of a closed one-cell-thick shell of its N_K cells at spacing a (the shell seed's law), and not at all
    once it reaches it: a cyst that grows is kept round and hollow without the fixed push that burst 48 or the weak one that let 53 collapse.
    particle -> particle, one set: reads pos; emits a velocity.

        v_i = k max(0, (R0 - R)/R0) (x_i - c_K)/|x_i - c_K|,   R0 = a sqrt(sqrt3 N_K / (8 pi)),   R = mean |x_j - c_K|
        k              `pressure`, the speed at a full deficit, world length per unit time
        R_K            with `measure: min_axis`, sqrt(3 lambda_min), lambda_min the smallest variance of the aggregate's cells about c_K (a round shell: R; a flat lens: small)
        a              `spacing`, the neighbour distance in the shell
        N_K, c_K       the aggregate's cell count and centre
        R_K            its mean radius

    Reference: Chan, C. J. et al. (2019). Nature 571:112-116 (the lumen's pressure is regulated: the blastocyst inflates, leaks and reseals)."""
    PARAM_UNITS = dict(LumenPressurePointCluster.PARAM_UNITS, spacing='length')
    PARAM_ROLES = dict(LumenPressurePointCluster.PARAM_ROLES, pressure="outward_speed_at_a_full_volume_deficit", spacing="neighbour_distance_of_the_target_shell")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.spacing = float(params.get("spacing", 1.0))
        self.measure = str(params.get("measure", "mean"))          # `min_axis`: R_K from the aggregate's flattest axis, sqrt(3 x its smallest variance) (a sphere's shell: R)
        if self.measure not in ("mean", "min_axis"):
            raise ValueError(f"lumen_pressure[volume]: measure must be mean or min_axis, got {self.measure!r}")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        X = lvl.get("pos")
        live = lvl.occ > 0.5
        if self._lab is None or self._calls % self.every == 0 or self._lab.shape[0] != X.shape[0]:
            self._lab = self._labels(X, live)
        self._calls += 1
        lab = torch.where(live, self._lab, torch.full_like(self._lab, -1))
        ok = lab >= 0
        V = torch.zeros_like(X)
        if not bool(ok.any()):
            return {self.at: V}
        k = int(lab.max()) + 1
        li = lab.clamp_min(0)
        w = ok.to(X.dtype)
        N = torch.zeros(k, device=X.device, dtype=X.dtype).index_add_(0, li, w)
        C = torch.zeros(k, X.shape[1], device=X.device, dtype=X.dtype).index_add_(0, li, X * w[:, None]) / N.clamp_min(1.0)[:, None]
        r = X - C[li]
        ln = r.norm(dim=1)
        Rm = torch.zeros(k, device=X.device, dtype=X.dtype).index_add_(0, li, ln * w) / N.clamp_min(1.0)
        if self.measure == "min_axis":                               # a flattened lens has a mean radius near the shell's but a small spread across its thin axis
            Rm = Rm.clone()
            for kk in range(k):
                m = ok & (lab == kk)
                if int(m.sum()) >= 4:
                    Y = r[m]
                    Rm[kk] = torch.sqrt(3.0 * torch.linalg.eigvalsh(Y.T @ Y / Y.shape[0])[0].clamp_min(0.0))
        R0 = self.spacing * torch.sqrt(math.sqrt(3.0) * N / (8.0 * math.pi))
        deficit = ((R0 - Rm) / R0.clamp_min(1e-9)).clamp_min(0.0)
        g = ok & (ln > 0.05)
        V = (self.p * deficit[li] * g.to(X.dtype))[:, None] * r / ln.clamp_min(1e-9)[:, None]
        if mask is not None:
            V = V * mask[:, None].to(X.dtype)
        return {self.at: V}


@register_operator("lumen_pressure", implementation="tube", family="mechanics", set="particle", kind="exchange",
                   title="inflate each aggregate of point cells toward a fixed radius round an axis: a tube that keeps its girth as it lengthens",
                   equation=r"""$$\mathbf v_i=k\,\max\!\Big(0,\frac{R^0-R_K}{R^0}\Big)\frac{\boldsymbol\rho_i}{\lVert\boldsymbol\rho_i\rVert},\qquad \boldsymbol\rho_i=(\mathbf I-\hat{\mathbf a}\hat{\mathbf a}^{\!\top})(\mathbf x_i-\bar{\mathbf x}_K),\qquad R_K=Q_{0.75}\big(\lVert\boldsymbol\rho_j\rVert\big)_{j\in K}$$""")
class LumenPressureTube(LumenPressurePointCluster):
    r"""a tube's lumen that regulates itself (exp05 route B, 2026-11-15): each aggregate is pushed outward from the LINE through its centre along `axis`
    only while its radius about that line, R_K, is below a fixed target R0, so a body that lengthens along the axis keeps its girth and stays hollow.
    lumen_pressure[volume] pushes from a centre POINT toward the radius of a sphere of the aggregate's cells: it rounds a tube up; the fixed pushes of
    point_cluster and polarity opened a 20-cell cyst into a flat sheet (step 112).
    particle -> particle, one set: reads pos; emits a velocity.

        v_i = k max(0, (R0 - R_K)/R0) rho_i/|rho_i|,   rho_i = (I - a a^T)(x_i - c_K),   R_K = the 0.75 quantile of |rho_j| over the aggregate
        k              `pressure`, the speed at a full deficit, world length per unit time
        a              `axis`, unit vector
        R0             `radius`; when absent, the radius of a closed one-cell-thick shell of the largest aggregate's cells at the operator's FIRST call,
                       a sqrt(sqrt3 N / (8 pi)) with a = `spacing` (the radius the cyst has when the rule starts)
        R_K            the aggregate's radius about the line: the 0.75 quantile, which is 0.97 R for a sphere of radius R and the tube's radius for a long
                       tube, and which a furrow of cells folded inward does not lower as a mean would
        c_K            the aggregate's centre

    Reference: Chan, C. J. et al. (2019). Nature 571:112-116; Navis, A. & Nelson, C. M. (2016). Semin. Cell Dev. Biol. 55:139-147 (lumen pressure in tubes)."""
    PARAM_UNITS = dict(LumenPressurePointCluster.PARAM_UNITS, spacing='length', radius='length')
    PARAM_ROLES = dict(LumenPressurePointCluster.PARAM_ROLES, pressure="outward_speed_at_a_full_radius_deficit", spacing="neighbour_distance_of_the_shell_that_sets_the_first_radius",
                       radius="target_radius_about_the_axis")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        if self.axis is None:
            self.axis = torch.tensor([0.0, 0.0, 1.0])
        self.spacing = float(params.get("spacing", 1.0))
        r0 = params.get("radius")
        self.R0 = None if r0 is None else float(r0)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        X = lvl.get("pos")
        live = lvl.occ > 0.5
        if self._lab is None or self._calls % self.every == 0 or self._lab.shape[0] != X.shape[0]:
            self._lab = self._labels(X, live)
        self._calls += 1
        lab = torch.where(live, self._lab, torch.full_like(self._lab, -1))
        ok = lab >= 0
        V = torch.zeros_like(X)
        if not bool(ok.any()):
            return {self.at: V}
        k = int(lab.max()) + 1
        li = lab.clamp_min(0)
        w = ok.to(X.dtype)
        N = torch.zeros(k, device=X.device, dtype=X.dtype).index_add_(0, li, w)
        C = torch.zeros(k, X.shape[1], device=X.device, dtype=X.dtype).index_add_(0, li, X * w[:, None]) / N.clamp_min(1.0)[:, None]
        if self.R0 is None:                                          # the first call fixes the target: the shell radius of the largest aggregate's cells
            self.R0 = self.spacing * math.sqrt(math.sqrt(3.0) * float(N.max()) / (8.0 * math.pi))
        ax = self.axis.to(X.device, X.dtype)
        r = X - C[li]
        rho = r - (r @ ax)[:, None] * ax[None]
        ln = rho.norm(dim=1)
        deficit = torch.zeros(k, device=X.device, dtype=X.dtype)
        for kk in range(k):
            m = ok & (lab == kk)
            if int(m.sum()) >= 4:
                deficit[kk] = max(0.0, (self.R0 - float(torch.quantile(ln[m], 0.75))) / self.R0)
        g = ok & (ln > 0.05)
        V = (self.p * deficit[li] * g.to(X.dtype))[:, None] * rho / ln.clamp_min(1e-9)[:, None]
        if mask is not None:
            V = V * mask[:, None].to(X.dtype)
        return {self.at: V}


from plexus.operators.diffusion_reaction import CellReactAlievPanfilov  # noqa: E402


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", implementation="aliev_panfilov_paced",
                   title="fire and recover excitable cells, paced by the cells of one type",
                   species=(("u", "excitation"), ("v", "recovery")),
                   equation=r"""$$\frac{du}{dt}=r\big(-k\,u(u-a)(u-1)-u\,v+I\,[T_i\in P]\,[t\in\text{pulse}]\big),\qquad \frac{dv}{dt}=r\,\varepsilon(u,v)\big(-v-k\,u(u-a-1)\big)$$""")
class CellReactAlievPanfilovPaced(CellReactAlievPanfilov):
    r"""cell_chem_react[aliev_panfilov] whose stimulus is injected into the cells of given TYPES instead of the cells of a region of space: the
    pacemaker is a cell lineage and moves and grows with the tissue (exp05 steps 64-72: a ring grew out of its fixed pacemaker box after 75 h).
    cell -> cell: as aliev_panfilov; reads node_type.

        du/dt = r (-k u (u - a)(u - 1) - u v + I [T_i in P] [t in pulse]),   dv/dt = r eps(u, v) (-v - k u (u - a - 1))
        I              `stim.amp`, the injected current
        P              `stim.types`, the pacemaker's cell types
        pulse          the windows [t0, t0 + `stim.duration`) for every t0 of `stim.times`

    Reference: Aliev, R. R. & Panfilov, A. V. (1996). Chaos Solitons Fractals 7:293-301."""
    PARAM_ROLES = dict(CellReactAlievPanfilov.PARAM_ROLES, stim="stimulus_current_times_and_pacemaker_types")

    @staticmethod
    def _site(s):
        st = dict(times=[float(t) for t in (s.get("times") or [])], dur=float(s.get("duration", 0.5)), amp=float(s.get("amp", 0.0)),
                  types=[str(t) for t in (s.get("types") or [])], center=None, radius=0.0, below=None, box=None)
        if st["times"] and st["amp"] and not st["types"]:
            raise ValueError("aliev_panfilov_paced: `stim` needs `types`, the pacemaker's cell types")
        return st

    def _stim(self, H, lvl, like):
        out = torch.zeros_like(like)
        names = list(getattr(lvl, "type_names", []) or [])
        nt = lvl.node_type.to(like.device).long()
        ft = getattr(H, "frame_t", None)
        t = (ft if ft is not None else torch.tensor(float(getattr(H, "frame", 0)))).to(like) * float(H.dt)
        for s in self.stims:
            if not (s["times"] and s["amp"]):
                continue
            where = torch.zeros_like(nt, dtype=torch.bool)
            for tn in s["types"]:
                where = where | (nt == names.index(tn))
            t0 = torch.as_tensor(s["times"], dtype=like.dtype, device=like.device)
            on = ((t >= t0) & (t < t0 + s["dur"])).any().to(like.dtype)
            out = out + s["amp"] * on * where.to(like.dtype)
        return out
