"""Lateral laws on a particle set, and the Rewire that builds the relation they read.

Every contract here acts on one `particle` set and reads that set's `edge_index`, the neighbour
relation; `radius_graph` is the Rewire that writes it, and it must be scheduled before any law
below it.

In the order they appear below:

    attraction_repulsion  lateral   two competing Gaussians: dispersed, clustered or crystalline
    squared_law           lateral   inverse square: gravity between masses, Coulomb between charges
    cohesion              lateral   steer toward the mean neighbour position        (boids)
    separation            lateral   steer away from the closest neighbours          (boids)
    velocity_align        lateral   steer toward the mean neighbour velocity        (boids, Vicsek)
    stillinger_weber      lateral   a two-body well plus a three-body angular penalty
    radius_graph          rewire    the relation: which particles count as neighbours

then the alternative implementations, which change only the numerics:

    squared_law[warp]     one Warp kernel: O(N^2) arithmetic, O(N) memory
    squared_law[mesh]     particle-mesh: deposit, FFT Poisson, gather -- O(N + M log M)
"""
from __future__ import annotations
import torch
from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.geometry import minimum_image
from plexus.geometry import neighbour_mean
from plexus.models.base import Rewire
from plexus.geometry import edges_radius_blockwise

import math

# `warp` is optional. Two of `squared_law`'s implementations are Warp kernels; `HAVE_WARP` gates
# every kernel definition below so the module still imports where warp is absent. One guard and
# one `wp.init()` for the whole module.
try:
    import warp as wp
    wp.init()
    HAVE_WARP = True
except Exception:
    HAVE_WARP = False


@register_operator("attraction_repulsion", family="interaction", set="particle", kind="lateral", title="attract and repel particles",
                   equation=r"""$$f(r) \;=\; p_1\,e^{-r^{2p_2}/2\sigma^2} \;-\; p_3\,e^{-r^{2p_4}/2\sigma^2},
\qquad
\dot{\mathbf{x}}_i \;=\; \sum_{j\in\mathcal{N}(i)} f(r_{ij})\,(\mathbf{x}_j-\mathbf{x}_i)$$""")
class AttractionRepulsion(Lateral):
    r"""soft-core attraction-repulsion: a smooth pairwise law whose two competing Gaussians, a long-range pull and a short-range push, set the phase the set settles into.
    particle -[neighbour relation]-> particle: reads pos (and the edge_index radius_graph wrote), emits a velocity (the law is overdamped: the force is the velocity).

        f(r) \;=\; p_1\,e^{-r^{2p_2}/2\sigma^2} \;-\; p_3\,e^{-r^{2p_4}/2\sigma^2}, \qquad \dot{\mathbf{x}}_i \;=\; \sum_{j\in\mathcal{N}(i)} f(r_{ij})\,(\mathbf{x}_j-\mathbf{x}_i)
        f(r)           the pair law at distance r, positive for attraction
        p_1            the strength of the long-range pull, dimensionless (per receiver type)
        p_2            its range exponent
        p_3            the strength of the short-range push
        p_4            its range exponent
        \sigma         `sigma`, the interaction length, in world length
        \dot x_i       the velocity of particle i (`aggr: mean` divides the sum by the live-neighbour count)
        N(i)           the neighbours of i within the cutoff
        r_{ij}         the distance |x_j - x_i|

    Reference: D'Orsogna, M. R., Chuang, Y. L., Bertozzi, A. L. & Chayes, L. S. (2006). Self-propelled particles with soft-core interactions: patterns, stability and collapse. Phys. Rev. Lett. 96:104302."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"A pair law with a long-range pull and a short-range push. Particle $i$ moves with the velocity $\dot{\mathbf "
        r"x}_i=\sum_{j\in\mathcal N(i)} f(r_{ij})\,(\mathbf x_j-\mathbf x_i)$, summed over its neighbours $\mathcal N(i)$ "
        r"within the cutoff, with $f(r)=p_1e^{-r^{2p_2}/2\sigma^2}-p_3e^{-r^{2p_4}/2\sigma^2}$ at distance $r_{ij}$ "
        r"($p_1$, $p_3$ the pull and push strengths of the receiving particle's type, $p_2$, $p_4$ their range exponents, "
        r"$\sigma$ the interaction length `sigma`). It is overdamped: the force is the velocity. `aggr: mean` divides the "
        r"sum by the neighbour count."
    )
    PARAM_UNITS = {'sigma': 'length', 'noise': 'velocity'}
    SYMBOLS = {'f(r)': 'the pair law at distance r, positive for attraction', 'p_1': 'the strength of the long-range pull, dimensionless (per receiver type)', 'p_2': 'its range exponent', 'p_3': 'the strength of the short-range push', 'p_4': 'its range exponent', '\\sigma': '`sigma`, the interaction length, in world length', '\\dot x_i': 'the velocity of particle i (`aggr: mean` divides the sum by the live-neighbour count)', 'N(i)': 'the neighbours of i within the cutoff', 'r_{ij}': 'the distance |x_j - x_i|'}

    EMIT = "velocity"             # overdamped: the force IS the velocity, no inertia
    SUPPORTED_DIMS = [2, 3]                      # dimension-generic (reads D = pos.shape[-1])
    REQUIRES_PARAMS = ["sigma"]                 # the cutoff lives on the radius_graph rewire op
    REQUIRES_TYPE_PROPS = ["p"]                 # per-type force-law params [p1,p2,p3,p4]
    MECHANISM_TAGS = ["long_range_attraction", "short_range_repulsion", "coarsening", "lattice_forming"]
    PARAM_ROLES = {"sigma": "interaction_length", "noise": "exploration_noise",
                   "p": "[pull_strength, pull_range, push_strength, push_range] per type", 'aggr': 'mean_or_sum_over_the_neighbours'}
    REFERENCE = ("D'Orsogna, M. R., Chuang, Y. L., Bertozzi, A. L. & Chayes, L. S. (2006). "
                 "Self-propelled particles with soft-core interactions: patterns, stability "
                 "and collapse. Phys. Rev. Lett. 96:104302.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.sigma = float(params["sigma"])
        self.aggr = params.get("aggr", "mean")               # mean (default, matches the reference) or sum
        self.noise = float(params.get("noise", 0.0))  # isotropic velocity noise (off by default)
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        occ = lvl.occ
        N, D = pos.shape[0], pos.shape[-1]
        ei = lvl.edge_index                                  # [2, E]: row0 = receiver i, row1 = neighbour j
        if ei.numel() == 0:
            return {self.at: torch.zeros(N, D, device=pos.device)}
        i, j = ei[0], ei[1]

        d = minimum_image(pos[j] - pos[i], getattr(H, "periodic", False),
                          getattr(H, "world_size", getattr(H, "world_width", 1.0)))   # j - i  [E, D]
        r2 = (d * d).sum(-1)                                  # [E]
        p = lvl.type_params[lvl.node_type[i]]                # receiver-type params [E, 4]
        s2 = 2.0 * self.sigma ** 2
        f = (p[:, 0] * torch.exp(-(r2 ** p[:, 1]) / s2)
             - p[:, 2] * torch.exp(-(r2 ** p[:, 3]) / s2))   # [E]
        f = f * occ[j]                                       # ignore dormant neighbours

        dpos = torch.zeros(N, D, device=pos.device)
        dpos.index_add_(0, i, f[:, None] * d)                # aggregate at the receiver
        if self.aggr == "mean":                              # average over neighbours (density-independent)
            deg = torch.zeros(N, device=pos.device).index_add_(0, i, occ[j])
            dpos = dpos / deg.clamp(min=1.0)[:, None]
        dpos = dpos * occ[:, None]
        if self.noise > 0.0:                                 # exploratory noise on the overdamped velocity
            dpos = dpos + self.noise * torch.randn(N, D, generator=getattr(H, "rng", None),
                                                   device=pos.device) * occ[:, None]
        if mask is not None:
            dpos = dpos * mask[:, None].float()
        return {self.at: dpos}


@register_operator("attraction_repulsion", implementation="force_curve", family="interaction", set="particle", kind="lateral", title="attract and repel by a force curve of the distance",
                   equation=r"""$$\dot{\mathbf{x}}_i \;=\; \sum_{j\in\mathcal{N}(i)} g_{ab}(r_{ij})\,\frac{\mathbf{x}_j-\mathbf{x}_i}{r_{ij}},\qquad g_{ab}(r)=\text{linear interpolation of the table } c_{ab,k} \text{ at the knots } r_k$$""")
class AttractionRepulsionForceCurve(Lateral):
    r"""the pair force written as a free curve of the distance, one curve per pair of types, in 1D, 2D or 3D: a table of values at fixed knots, linear between them, so the curve can take any shape (a push inside a gap, a pull out to a reach) and its values can be trained.
    particle -[neighbour relation]-> particle: reads pos and node_type (and the edge_index radius_graph wrote), emits a velocity (the law is overdamped: the force is the velocity).

        \dot{\mathbf x}_i=\sum_{j\in\mathcal N(i)} g_{ab}(r_{ij})\,\hat{\mathbf r}_{ij},\qquad g_{ab}(r)=c_{ab,k}+\frac{r-r_k}{r_{k+1}-r_k}\,(c_{ab,k+1}-c_{ab,k})\ \text{for}\ r_k\le r<r_{k+1}
        \dot x_i        the velocity of particle i
        g_ab(r)         the force of a particle of type b on one of type a at distance r, positive for attraction (toward the neighbour); zero beyond the last knot, constant below the first
        a, b            the types of the receiver i and the neighbour j
        c_{ab,k}        `curves`[a][b][k], the curve's value at knot k, a velocity
        r_k             `knots`[k], the knots, increasing, in world length
        \hat r_{ij}     the unit vector from i to j, (x_j - x_i) / r_ij
        N(i)            the neighbours of i within the cutoff of the radius_graph

    Reference: Plexus exp37 (this work, 2026-11); the Gaussian pair of D'Orsogna et al. (2006), Phys. Rev. Lett. 96:104302, is one shape this curve can take."""
    EXPLAIN = (r"A pair law that is a free curve. The force between two particles is read from a table of values at fixed distances (`knots`), one table per pair of types (`curves`), linear between the knots, "
               r"positive meaning a pull toward the neighbour and negative a push away, zero beyond the last knot. It is overdamped: the force is the velocity. The same operator serves a line, a plane or a volume.")
    PARAM_UNITS = {'knots': 'length', 'curves': 'velocity', 'curve': 'velocity', 'noise': 'velocity', 'between': 'count'}
    SYMBOLS = {'\\dot x_i': 'the velocity of particle i', 'g_ab(r)': 'the force of a type-b neighbour on a type-a particle at distance r, positive for attraction', 'a, b': 'the types of the receiver and of the neighbour',
               'c_{ab,k}': "`curves`[a][b][k], the curve's value at knot k", 'r_k': '`knots`[k], the distances at which the curve is given', '\\hat r_{ij}': 'the unit vector from i to j', 'N(i)': 'the neighbours of i within the cutoff'}
    EMIT = "velocity"
    SUPPORTED_DIMS = [1, 2, 3]
    REQUIRES_PARAMS = ["knots"]                       # and one of `curves` (a table per pair of types) or `curve` (ONE table for every pair of different types, none within a type)
    PARAM_ROLES = {"knots": "distances_at_which_the_curve_is_given", "curves": "force_value_at_each_knot_per_pair_of_types", "curve": "force_value_at_each_knot_for_every_pair_of_different_types", "between": "types_or_parents_which_pairs_count_as_different", "noise": "exploration_noise"}
    MECHANISM_TAGS = ["long_range_attraction", "short_range_repulsion", "learnable_curve"]
    REFERENCE = "Plexus exp37 (this work, 2026-11)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle"); self.noise = float(params.get("noise", 0.0))
        self.knots = torch.as_tensor(params["knots"], dtype=torch.float32, device=device)                 # [K]
        self.curves = torch.as_tensor(params["curves"], dtype=torch.float32, device=device) if "curves" in params else None   # [T, T, K]; trained by setting this attribute to a Parameter
        self.curve = torch.as_tensor(params["curve"], dtype=torch.float32, device=device) if "curve" in params else None       # [K], the table of every pair of different types; trained the same way
        self.between = str(params.get("between", "types"))                                               # `curve` acts between points of different "types" (node_type) or different "parents" (the cell a point belongs to)
        if (self.curves is None) == (self.curve is None):
            raise ValueError("attraction_repulsion[force_curve]: give exactly one of `curves` (a table per pair of types) or `curve` (one table for every pair of different types)")

    def forward(self, H, mask=None):
        lvl = H.level(self.at); pos = lvl.get("pos"); occ = lvl.occ; N, D = pos.shape[0], pos.shape[-1]; ei = lvl.edge_index
        if ei.numel() == 0:
            return {self.at: torch.zeros(N, D, device=pos.device)}
        i, j = ei[0], ei[1]
        d = minimum_image(pos[j] - pos[i], getattr(H, "periodic", False), getattr(H, "world_size", getattr(H, "world_width", 1.0)))     # [E, D] from i to j
        r = d.norm(dim=-1).clamp(min=1e-12)
        kn, K = self.knots.to(pos.dtype), self.knots.numel()
        k = torch.searchsorted(kn, r.clamp(kn[0], kn[-1]).contiguous(), right=True).clamp(1, K - 1)                                   # the segment [r_{k-1}, r_k] each distance falls in
        if self.curve is not None:
            who = lvl.parent if self.between == "parents" else lvl.node_type
            cv = self.curve.to(pos.dtype)[None, :] * (who[i] != who[j]).to(pos.dtype)[:, None]                                       # [E, K]: the one table between different types (or parents), zero within one
        else:
            cv = self.curves.to(pos.dtype)[lvl.node_type[i], lvl.node_type[j]]                                                       # [E, K]
        w = (r.clamp(kn[0], kn[-1]) - kn[k - 1]) / (kn[k] - kn[k - 1])
        g = cv.gather(1, (k - 1)[:, None])[:, 0] * (1 - w) + cv.gather(1, k[:, None])[:, 0] * w
        g = g * (r <= kn[-1]).to(pos.dtype) * occ[j]                                                                                  # nothing beyond the last knot, nothing from a dormant neighbour
        dpos = torch.zeros(N, D, device=pos.device, dtype=pos.dtype).index_add_(0, i, g[:, None] * d / r[:, None]) * occ[:, None]
        if self.noise > 0.0:
            dpos = dpos + self.noise * torch.randn(N, D, generator=getattr(H, "rng", None), device=pos.device) * occ[:, None]
        if mask is not None:
            dpos = dpos * mask[:, None].float()
        return {self.at: dpos}


@register_operator("attraction_repulsion", implementation="force_curve_mpm", family="interaction", set="particle", kind="lateral", title="attract and repel material points by a force curve of the distance",
                   equation=r"""$$\mathbf a_i \;=\; \sum_{j\in\mathcal{N}(i)} g_{ab}(r_{ij})\,\frac{\mathbf{x}_j-\mathbf{x}_i}{r_{ij}}$$""")
class AttractionRepulsionForceCurveMPM(AttractionRepulsionForceCurve):
    r"""`force_curve` for material points: the table gives an ACCELERATION, which the grid's Stokes drag turns into a velocity, so bodies of material points feel it through their own elasticity.
    pt -[neighbour relation]-> pt: reads pos and node_type, emits `mpm_acceleration`.

        \mathbf a_i=\sum_{j\in\mathcal N(i)} g_{ab}(r_{ij})\,\hat{\mathbf r}_{ij}
        a_i            the body acceleration of point i
        g_ab(r)        the table of the pair of types (a, b) at distance r, an acceleration, positive toward the neighbour; see `force_curve`
        \hat r_{ij}    the unit vector from i to j
        N(i)           the neighbours of i within the cutoff of the radius_graph

    Reference: Plexus exp37 (this work, 2026-11)."""
    EXPLAIN = (r"The force-curve pair law for bodies of material points: the table is read as an acceleration added to each point, positive toward the neighbour, zero beyond the last knot, one table per pair of types.")
    PARAM_UNITS = {'knots': 'length', 'curves': 'acceleration', 'curve': 'acceleration', 'noise': 'acceleration', 'between': 'count'}
    SYMBOLS = {'a_i': 'the body acceleration of point i', 'g_ab(r)': 'the table of the pair of types at distance r, an acceleration, positive toward the neighbour', '\\hat r_{ij}': 'the unit vector from i to j', 'N(i)': 'the neighbours of i within the cutoff'}
    EMIT = "mpm_acceleration"


@register_operator("attraction_repulsion", implementation="force_curve_local_gap", family="interaction", set="particle", kind="lateral", title="attract and repel two bodies by a force curve of the local gap between their faces",
                   equation=r"""$$\mathbf a_i \;=\; g(\delta_i)\,\mathbf n_i,\qquad \delta_i=\min_j\ (\mathbf x_j-\mathbf x_i)\cdot\mathbf n_i\ \ \text{over the points of the other body in the cone around}\ \mathbf n_i$$""")
class AttractionRepulsionForceCurveLocalGap(AttractionRepulsionForceCurveMPM):
    r"""the force curve of the LOCAL GAP between the faces of two bodies of material points, not of the distance between two points: each front surface point feels an acceleration along its own outward normal, whatever the point spacing.
    pt -> pt: reads pos and parent, emits `mpm_acceleration`; the two bodies are the first two parent cells.

        \mathbf a_i=g(\delta_i)\,\mathbf n_i
        a_i            the body acceleration of front surface point i
        g(d)           the table `curve` at the gap d, linear between the `knots`, zero beyond the last; positive = toward the other body
        \delta_i       the local gap at point i: the soft minimum, along its outward normal, of the distance to the other body's points inside the cone (`cone`, `reach` in point spacings; `plexus.geometry.facing_gaps`)
        n_i            the outward unit normal at point i, away from the mean of its neighbours within `ring` point spacings in its own body

    Reference: Plexus exp38 (this work, 2026-11)."""
    EXPLAIN = (r"A pair law between two bodies that reads the gap between their FACES. Each point on the front surface of a body looks along its outward normal for the other body, and is accelerated along that normal by the table `curve` at that gap, "
               r"positive toward the other body and zero beyond the last knot, so a face is held at a set gap whatever the spacing of the points.")
    PARAM_UNITS = {'knots': 'length', 'curve': 'acceleration', 'ring': 'dimensionless', 'cone': 'dimensionless', 'reach': 'dimensionless', 'tau': 'length', 'edge': 'dimensionless'}
    SYMBOLS = {'a_i': 'the body acceleration of front surface point i', 'g(d)': 'the table at the gap d, positive toward the other body', '\\delta_i': 'the local gap at point i along its outward normal', 'n_i': 'the outward unit normal at point i'}
    REQUIRES_PARAMS = ["knots", "curve"]
    PARAM_ROLES = {"knots": "gaps_at_which_the_curve_is_given", "curve": "acceleration_at_each_knot", "ring": "neighbourhood_radius_for_the_normal_in_point_spacings", "cone": "cone_half_width_in_point_spacings", "reach": "longest_gap_in_point_spacings", "tau": "soft_minimum_length", "edge": "surface_threshold_in_point_spacings"}

    def __init__(self, params, device="cpu"):
        params = dict(params); params.pop("curves", None)
        super().__init__(params, device)
        self.ring = float(params.get("ring", 2.0)); self.cone = float(params.get("cone", 3.0)); self.reach = float(params.get("reach", 12.0)); self.tau = float(params.get("tau", 0.00005)); self.edge = float(params.get("edge", 0.25))

    def forward(self, H, mask=None):
        from plexus.geometry import facing_gaps, facing_pairs
        P = H.level(self.at); X = P.get("pos"); live = P.occ > 0.5; par = P.parent
        ids = torch.unique(par[live])[:2]
        if ids.numel() < 2:                                                                                # one body only (a mother before its division): no pair, no force
            return {self.at: torch.zeros_like(X)}
        kn, K = self.knots.to(X.dtype), self.knots.numel(); cv = self.curve.to(X.dtype)
        acc = torch.zeros_like(X); frame = getattr(H, "frame", None)
        fresh = frame is None or frame != getattr(self, "_frame", None)                                    # the front points, normals and candidate pairs are searched once per frame, not per substep
        if fresh:
            self._frame, self._pairs = frame, {}
        for ia_, ib_ in ((ids[0], ids[1]), (ids[1], ids[0])):
            ia = torch.nonzero(live & (par == ia_)).reshape(-1); ib = torch.nonzero(live & (par == ib_)).reshape(-1)
            if fresh:
                with torch.no_grad():
                    self._pairs[int(ia_)] = facing_pairs(X[ia], X[ib], self.ring, self.cone, self.reach, self.edge, max_gap=float(kn[-1]))
            idx, g, has, n = facing_gaps(X[ia], X[ib], self._pairs[int(ia_)], tau=self.tau)
            gc = g.clamp(kn[0], kn[-1]); seg = torch.searchsorted(kn, gc.detach().contiguous(), right=True).clamp(1, K - 1)
            w = (gc - kn[seg - 1]) / (kn[seg] - kn[seg - 1]); f = cv[seg - 1] * (1 - w) + cv[seg] * w
            f = f * (has & (g <= kn[-1])).to(X.dtype)                                                  # nothing beyond the last knot, nothing where the other body is not in the cone
            acc = acc.index_add(0, ia[idx], f[:, None] * n)
        if mask is not None:
            acc = acc * mask.float()[:, None]
        return {self.at: acc}


@register_operator("attraction_repulsion", implementation="force_curve_local_gap_all_bodies", family="interaction", set="particle", kind="lateral", title="attract and repel every pair of nearby bodies by a force curve of the local gap between their faces",
                   equation=r"""$$\mathbf a_i \;=\; \sum_{B\neq A(i)} g(\delta_{iB})\,\mathbf n_i,\qquad \delta_{iB}=\min_j\ (\mathbf x_j-\mathbf x_i)\cdot\mathbf n_i\ \ \text{over the points of body}\ B\ \text{in the cone around}\ \mathbf n_i$$""")
class AttractionRepulsionForceCurveLocalGapAllBodies(AttractionRepulsionForceCurveLocalGap):
    r"""force_curve_local_gap for every pair of bodies that can touch, not only the first two parent cells; on two bodies it is that law, number for number.
    pt -> pt: reads pos and parent, emits `mpm_acceleration`; the pairs are the ordered pairs of parent cells whose bounding spheres come within two last-knot gaps of each other (searched once per frame).

        \mathbf a_i=\sum_{B\neq A(i)} g(\delta_{iB})\,\mathbf n_i
        a_i            the body acceleration of front surface point i
        g(d)           the table `curve` at the gap d, linear between the `knots`, zero beyond the last; positive = toward the other body
        \delta_{iB}    the local gap at point i to body B, as in force_curve_local_gap
        n_i            the outward unit normal at point i, away from the mean of its neighbours within `ring` point spacings in its own body
        A(i), B        the body of point i, and any other body near it

    A pair is searched when the distance of the two bodies' centroids is at most the sum of their radii (the largest distance of a point from its centroid) plus two last-knot gaps: no point of one is within the last knot of a point of the other beyond that, so the pairs left out would have had no force.

    Reference: Plexus exp38 (this work, 2026-11)."""
    EXPLAIN = (r"The two-body face law over a whole tissue. Every ordered pair of bodies whose bounding spheres are close enough is treated as force_curve_local_gap treats its one pair: the front points of one body look along their outward normals for the other, and are accelerated by the table `curve` at that gap. "
               r"A point in contact with two bodies feels both. With exactly two bodies the accelerations are those of force_curve_local_gap.")
    PARAM_UNITS = AttractionRepulsionForceCurveLocalGap.PARAM_UNITS
    SYMBOLS = {'a_i': 'the body acceleration of front surface point i', 'g(d)': 'the table at the gap d, positive toward the other body', '\\delta_{iB}': 'the local gap at point i to body B along its outward normal', 'n_i': 'the outward unit normal at point i', 'A(i), B': 'the body of point i, and any other body near it'}
    PARAM_ROLES = AttractionRepulsionForceCurveLocalGap.PARAM_ROLES

    @staticmethod
    def _near_bodies(X, live, par, ids, gap):
        """The ordered pairs (a, b) of parent ids whose bounding spheres lie within 2 `gap` of each other, a before b in id order (for two bodies: (a, b), (b, a), the order of force_curve_local_gap)."""
        with torch.no_grad():
            xl, pl = X[live], par[live]; inv = torch.searchsorted(ids, pl); n = ids.numel()
            cnt = torch.zeros(n, device=X.device, dtype=X.dtype).index_add_(0, inv, torch.ones_like(xl[:, 0]))
            c = torch.zeros(n, X.shape[1], device=X.device, dtype=X.dtype).index_add_(0, inv, xl) / cnt[:, None]
            rho = torch.zeros(n, device=X.device, dtype=X.dtype).scatter_reduce_(0, inv, (xl - c[inv]).norm(dim=1), "amax", include_self=False)
            D = torch.cdist(c, c, compute_mode="donot_use_mm_for_euclid_dist"); near = (D <= rho[:, None] + rho[None, :] + 2.0 * gap) & ~torch.eye(n, dtype=torch.bool, device=X.device)
        il = ids.tolist()
        return [(il[i], il[j]) for i, j in near.nonzero().tolist()]

    def forward(self, H, mask=None):
        from plexus.geometry import facing_gaps, facing_pairs_bodies
        P = H.level(self.at); X = P.get("pos"); live = P.occ > 0.5; par = P.parent
        ids = torch.unique(par[live])
        if ids.numel() < 2:                                                                                # one body only: no pair, no force
            return {self.at: torch.zeros_like(X)}
        kn, K = self.knots.to(X.dtype), self.knots.numel(); cv = self.curve.to(X.dtype)
        acc = torch.zeros_like(X); frame = getattr(H, "frame", None)
        if frame is None or frame != getattr(self, "_frame", None):                                        # the pairs of bodies, front points, normals and candidate pairs are searched once per frame, for all pairs at once, not per substep
            self._frame, self._near = frame, self._near_bodies(X, live, par, ids, float(kn[-1]))           # (points change body only between frames)
            self._pairs = facing_pairs_bodies(X, live, par, ids, self._near, self.ring, self.cone, self.reach, self.edge, max_gap=float(kn[-1])) if self._near else None
        if self._pairs is not None:
            idx, g, has, n = facing_gaps(X, X, self._pairs, tau=self.tau)                                  # one pass over every (front point, other body) entry of the frame; idx indexes X
            gc = g.clamp(kn[0], kn[-1]); seg = torch.searchsorted(kn, gc.detach().contiguous(), right=True).clamp(1, K - 1)
            w = (gc - kn[seg - 1]) / (kn[seg] - kn[seg - 1]); f = cv[seg - 1] * (1 - w) + cv[seg] * w
            f = f * (has & (g <= kn[-1])).to(X.dtype)                                                  # nothing beyond the last knot, nothing where the other body is not in the cone
            acc = acc.index_add(0, idx, f[:, None] * n)
        if mask is not None:
            acc = acc * mask.float()[:, None]
        return {self.at: acc}


def _inv_square_sum(pos, src, soft2):
    """The unsigned all-pairs sum of `SquaredLaw`, before the law's own sign and coupling:

        pull_i = sum_j src_j (x_j - x_i) / (|x_j - x_i|^2 + soft2)^(3/2)

    A per-particle vector, not a Plexus `Field` -- no grid is involved. The caller scales it by
    the signed strength and the receiver's coupling to get the acceleration. `src` folds in
    occupancy, so a dormant particle contributes nothing. Written per dimension, so only [N, N]
    arrays appear and never an [N, N, D] one, and kept a free function so a fusing backend can
    collapse the reduction and materialise neither."""
    N, D = pos.shape
    r2 = torch.full((N, N), soft2, device=pos.device, dtype=pos.dtype)
    for k in range(D):
        dk = pos[:, k].unsqueeze(0) - pos[:, k].unsqueeze(1)       # dk[i,j] = pos[j]-pos[i]
        r2 = r2 + dk * dk
    inv_r3 = r2.clamp(min=1e-12).pow(-1.5)                          # diagonal dk=0 -> 0 contribution
    pull = torch.empty(N, D, device=pos.device, dtype=pos.dtype)
    for k in range(D):
        dk = pos[:, k].unsqueeze(0) - pos[:, k].unsqueeze(1)
        pull[:, k] = (dk * inv_r3) @ src
    return pull


_A, _B, _P = 7.049556277, 0.6022245584, 4.0


@register_operator("radius_graph", family="topology", set="particle", kind="rewire", title="connect neighbours within a radius",
                   equation=r"""$$\mathrm{edges} \;=\; \big\{\,(i,j)\;:\; r_{\min} \le \lVert\mathbf x_i-\mathbf x_j\rVert < r\,\big\}$$""")
class RadiusGraph(Rewire):
    r"""the neighbour relation: two particles interact when they are close enough; every lateral law of this module reads what it writes, so it is scheduled before them.
    particle -> particle: reads pos, writes the set's `edge_index` in place (receiver, neighbour), live pairs only.

        \mathrm{edges} \;=\; \big\{\,(i,j)\;:\; r_{\min} \le \lVert\mathbf x_i-\mathbf x_j\rVert < r\,\big\}
        \mathrm{edges}  the edge set
        (i, j)         a receiver and one of its neighbours
        r_{min}        `min_radius`, pairs closer than it are excluded (keeps a 1/|d| law finite)
        x_i, x_j       the two positions
        r              `radius`, the interaction cutoff, in world length (a claim that the law is negligible beyond it)

    Reference: none: a cutoff neighbour list is standard practice, not a result. Plexus (this work)."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"The neighbour list. Two particles $i$ and $j$ are neighbours when their distance is at least `min_radius` and "
        r"below `radius`, the interaction cutoff in world length. It rebuilds the list from the current positions each "
        r"time it runs, so the pair laws scheduled after it read who is close now."
    )
    PARAM_UNITS = {'radius': 'length', 'min_radius': 'length', 'block': 'count'}
    SYMBOLS = {'\\mathrm{edges}': 'the edge set', '(i, j)': 'a receiver and one of its neighbours', 'r_{min}': '`min_radius`, pairs closer than it are excluded (keeps a 1/|d| law finite)', 'x_i, x_j': 'the two positions', 'r': '`radius`, the interaction cutoff, in world length (a claim that the law is negligible beyond it)'}

    EMIT = None                                 # rewire: rebuilds edge_index in place, returns no delta
    SUPPORTED_DIMS = [1, 2, 3]                   # pairwise distances are dimension-generic
    REQUIRES_PARAMS = ["radius"]
    MECHANISM_TAGS = ["radius_graph", "neighbor_search", "rewire"]
    PARAM_ROLES = {"radius": "interaction_cutoff_in_world_length", "min_radius": "inner_cutoff_radius", "block": "block_size"}
    REFERENCE = "Plexus (this work); a cutoff neighbour list is standard practice."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.r_max = float(params["radius"])
        self.r_min = float(params.get("min_radius", 0.0))
        self.block = int(params.get("block", 2048))
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        lvl.edge_index = edges_radius_blockwise(
            lvl.get("pos"), lvl.occ, self.r_min, self.r_max,
            periodic=getattr(H, "periodic", False),
            world_width=getattr(H, "world_size", getattr(H, "world_width", 1.0)),
            block=self.block,
        )
        return {}


# ==========================================================================================================
#  implementations -- same biology, different numerics
# ==========================================================================================================

@register_operator("radius_graph", family="topology", set="particle", kind="rewire", model="periodic_tiles",
                   title="connect neighbours on periodic tiles",
                   equation=r"""$$\mathrm{edges}=\{(s,s'):\ \mathrm{tile}(s)=\mathrm{tile}(s'),\ 0<\lVert (i,j)-(i',j')\rVert_{\mathrm{mod}\,L}\le r\}$$""")
class RadiusGraphPeriodicTiles(Rewire):
    r"""the same relation as radius_graph computed on the sites of square lattices that are periodic each on its own, the geometry the individual-based lattice models are defined on.
    particle -> particle: writes the set's `edge_index` (receiver, neighbour) and `tile_index`; the distance is the minimum image on each L x L tile.

        \mathrm{edges}=\{(s,s'):\ \mathrm{tile}(s)=\mathrm{tile}(s'),\ 0<\lVert (i,j)-(i',j')\rVert_{\mathrm{mod}\,L}\le r\}
        \mathrm{edges}  the edge set
        s, s'          two sites of the same tile
        (i, j)         a site's lattice coordinates
        L              `tile_side`, the tile's side in sites
        r              `radius` in site spacings: 1 the 4 nearest neighbours (von Neumann), 1.5 the 8 surrounding sites (Moore)

    Reference: Reichenbach, T., Mobilia, M. & Frey, E. (2007). Nature 448:1046; Kerr, B. et al. (2002). Nature 418:171, Box 1."""
    PARAM_UNITS = {'radius': 'dimensionless', 'tile_side': 'count'}
    SYMBOLS = {'\\mathrm{edges}': 'the edge set', "s, s'": 'two sites of the same tile', '(i, j)': "a site's lattice coordinates", 'L': "`tile_side`, the tile's side in sites", 'r': '`radius` in site spacings: 1 the 4 nearest neighbours (von Neumann), 1.5 the 8 surrounding sites (Moore)'}

    EMIT = None
    SUPPORTED_DIMS = [2]
    REQUIRES_PARAMS = ["radius", "tile_side"]
    MECHANISM_TAGS = ["radius_graph", "neighbor_search", "rewire", "lattice", "periodic"]
    PARAM_ROLES = {"tile_side": "lattice_side_sites", 'radius': 'neighbourhood_radius_in_site_spacings'}
    REFERENCE = "Reichenbach, T., Mobilia, M. & Frey, E. (2007). Nature 448:1046; Kerr, B. et al. (2002). Nature 418:171."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.r = float(params["radius"])
        self.L = int(params["tile_side"])
        self.at = params.get("_at", "particle")
        self._ei = None

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        if self._ei is None or self._ei.device != lvl.state.device:
            n, L = lvl.state.shape[0], self.L
            if n % (L * L):
                raise ValueError(f"radius_graph[periodic_tiles]: {n} slots is not a whole number of "
                                 f"{L} x {L} tiles")
            dev = lvl.state.device
            pos = lvl.get("pos")                             # THE LAYOUT, CHECKED: a row's sites are evenly spaced
            if L > 2:
                d = torch.linalg.norm(pos[1:L] - pos[:L - 1], dim=1)
                if float((d - d.mean()).abs().max()) > 1e-4 * float(d.mean()) + 1e-9:
                    raise ValueError("radius_graph[periodic_tiles]: the positions are not laid out as "
                                     "seed_positions[tiled_lattice] lays them (slot r L^2 + i L + j)")
            R = int(math.floor(self.r))
            offs = [(di, dj) for di in range(-R, R + 1) for dj in range(-R, R + 1)
                    if 0 < di * di + dj * dj <= self.r * self.r + 1e-9]
            s = torch.arange(n, device=dev)
            tile, loc = s // (L * L), s % (L * L)
            i, j = loc // L, loc % L
            recv, nbr = [], []
            for di, dj in offs:
                recv.append(s)
                nbr.append(tile * L * L + ((i + di) % L) * L + (j + dj) % L)
            self._ei = torch.stack([torch.cat(recv), torch.cat(nbr)])
            self._tile = tile
        lvl.edge_index = self._ei
        lvl.tile_index = self._tile
        return {}


# `squared_law[implementation: warp]` -- the all-pairs inverse square without the [N, N] matrices.
if HAVE_WARP:

    @wp.kernel
    def _pull3(P: wp.array(dtype=wp.vec3), S: wp.array(dtype=float), soft2: float, n: int,
               OUT: wp.array(dtype=wp.vec3)):
        i = wp.tid()
        pi = P[i]
        a = wp.vec3(0.0, 0.0, 0.0)
        for j in range(n):
            d = P[j] - pi
            r2 = wp.dot(d, d) + soft2
            # The same 1e-12 floor on r^2 the torch path uses, and what makes the j == i term
            # safe: unsoftened, the self term has r^2 = 0, so the floor caps the reciprocal at a
            # finite 1e18 and it is then multiplied by d = 0. A particle pulls on itself by 0.
            if r2 < 1.0e-12:
                r2 = 1.0e-12
            a = a + d * (S[j] / (r2 * wp.sqrt(r2)))
        OUT[i] = a

    @wp.kernel
    def _pull2(P: wp.array(dtype=wp.vec2), S: wp.array(dtype=float), soft2: float, n: int,
               OUT: wp.array(dtype=wp.vec2)):
        i = wp.tid()
        pi = P[i]
        a = wp.vec2(0.0, 0.0)
        for j in range(n):
            d = P[j] - pi
            r2 = wp.dot(d, d) + soft2
            if r2 < 1.0e-12:
                r2 = 1.0e-12
            a = a + d * (S[j] / (r2 * wp.sqrt(r2)))
        OUT[i] = a


def _launch(kernel, n, dev, inputs):
    """`wp.launch` on pytorch's current stream -- see mpm_warp._wp_launch for why this matters."""
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(dev)),
                         sync_enter=False, sync_exit=False):
        wp.launch(kernel, dim=n, device=f"cuda:{dev.index or 0}", inputs=inputs)


# `squared_law[implementation: mesh]` -- particle-mesh gravity: O(N), not O(N^2).
if HAVE_WARP:

    @wp.kernel
    def _cic_deposit(P: wp.array(dtype=wp.vec3), M: wp.array(dtype=float),
                     lo: wp.vec3, inv_h: float, n: int, RHO: wp.array(dtype=float)):
        """Cloud-in-cell: split each mass over the 8 cells of its containing cube, by volume."""
        i = wp.tid()
        x = (P[i] - lo) * inv_h
        ix = int(wp.floor(x[0]))
        iy = int(wp.floor(x[1]))
        iz = int(wp.floor(x[2]))
        fx = x[0] - float(ix)
        fy = x[1] - float(iy)
        fz = x[2] - float(iz)
        m = M[i]
        for a in range(2):
            wx = 1.0 - fx
            if a == 1:
                wx = fx
            gx = ix + a
            if gx >= 0 and gx < n:
                for b in range(2):
                    wy = 1.0 - fy
                    if b == 1:
                        wy = fy
                    gy = iy + b
                    if gy >= 0 and gy < n:
                        for c in range(2):
                            wz = 1.0 - fz
                            if c == 1:
                                wz = fz
                            gz = iz + c
                            if gz >= 0 and gz < n:
                                wp.atomic_add(RHO, (gx * n + gy) * n + gz, m * wx * wy * wz)

    @wp.kernel
    def _cic_gather(P: wp.array(dtype=wp.vec3), lo: wp.vec3, inv_h: float, n: int,
                    AX: wp.array(dtype=float), AY: wp.array(dtype=float), AZ: wp.array(dtype=float),
                    OUT: wp.array(dtype=wp.vec3)):
        """The same weights, read back: the transfer is its own transpose, which is what keeps
        momentum conserved to round-off rather than leaking it into the grid."""
        i = wp.tid()
        x = (P[i] - lo) * inv_h
        ix = int(wp.floor(x[0]))
        iy = int(wp.floor(x[1]))
        iz = int(wp.floor(x[2]))
        fx = x[0] - float(ix)
        fy = x[1] - float(iy)
        fz = x[2] - float(iz)
        a = wp.vec3(0.0, 0.0, 0.0)
        for p in range(2):
            wx = 1.0 - fx
            if p == 1:
                wx = fx
            gx = ix + p
            if gx >= 0 and gx < n:
                for q in range(2):
                    wy = 1.0 - fy
                    if q == 1:
                        wy = fy
                    gy = iy + q
                    if gy >= 0 and gy < n:
                        for r in range(2):
                            wz = 1.0 - fz
                            if r == 1:
                                wz = fz
                            gz = iz + r
                            if gz >= 0 and gz < n:
                                w = wx * wy * wz
                                k = (gx * n + gy) * n + gz
                                a = a + wp.vec3(AX[k] * w, AY[k] * w, AZ[k] * w)
        OUT[i] = a


def _launch(kernel, n, dev, inputs):
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(dev)),
                         sync_enter=False, sync_exit=False):
        wp.launch(kernel, dim=n, device=f"cuda:{dev.index or 0}", inputs=inputs)


# ==========================================================================================================
# `radius_graph[implementation: warp]` -- the same neighbour relation, found by a hash grid.
#
# WHY A SECOND IMPLEMENTATION AND NOT A FASTER LOOP. The default body is `edges_radius_blockwise`,
# which never materialises the [N, N] distance matrix but still SCORES every pair: the work is
# O(N^2) whatever the radius, so it is bounded by the population and not by how many neighbours a
# cell actually has. Measured on a 2D boids disc: 8 ms a frame at 4,000 cells and 87.6 SECONDS a
# frame at 500,000 -- a factor 11,000 for a factor 125 in the count, which is the quadratic, and it
# puts a 2,000-frame run at two days. A hash grid visits only the cells within the cutoff, so the
# work is O(N) times the neighbours per point, which is the number the physics already fixed.
#
# The biology is untouched: the relation this writes is the same set of pairs the default writes,
# to the last edge, and the two are checked against each other in `tests/`. Selecting it is a
# numerical decision, which is exactly what `implementation:` means (plexus2.tex sec. 5).
if HAVE_WARP:

    @wp.kernel
    def _rg_scan(P: wp.array(dtype=wp.vec3), grid: wp.uint64, r_min: float, r_max: float,
                 ORIG: wp.array(dtype=wp.int32), LIVE: wp.array(dtype=wp.int32),
                 OFF: wp.array(dtype=wp.int32), COUNT_ONLY: int,
                 CNT: wp.array(dtype=wp.int32),
                 EI: wp.array(dtype=wp.int32), EJ: wp.array(dtype=wp.int32)):
        """One kernel, run twice: once to COUNT each receiver's neighbours and once to WRITE them.

        Counting first and writing at an exact offset is what makes the edge list exact rather than
        capped. The alternative -- one pass into a preallocated buffer with a per-point maximum --
        silently drops a neighbour the moment a cluster exceeds the cap, and a flocking run is made
        of clusters, so the cap would be exceeded precisely where the physics is happening.
        """
        i = wp.tid()
        if LIVE[i] == 0:
            if COUNT_ONLY == 1:
                CNT[i] = 0
            return
        pi = P[i]
        c = int(0)
        q = wp.hash_grid_query(grid, pi, r_max)
        j = int(0)
        while wp.hash_grid_query_next(q, j):
            oj = ORIG[j]                      # a ghost answers for the real point it copies
            if oj != i and LIVE[oj] != 0:
                d = P[j] - pi
                r = wp.length(d)
                if r > r_min and r <= r_max:
                    if COUNT_ONLY == 1:
                        c = c + 1
                    else:
                        k = OFF[i] + c
                        EI[k] = i
                        EJ[k] = oj
                        c = c + 1
        if COUNT_ONLY == 1:
            CNT[i] = c


@register_operator("radius_graph", implementation="warp", family="interaction",
                   set="particle", kind="rewire", title="connect neighbours with a hash grid", equation=r"""$$\mathrm{edges}=\big\{(i,j):\ r_{\min}\le\lVert\mathbf x_i-\mathbf x_j\rVert<r\big\}$$""")
class RadiusGraphWarp(RadiusGraph):
    r"""the same relation as radius_graph, found with a Warp hash grid in O(N x neighbours) instead of O(N^2); periodic boxes are handled by ghost copies across the faces.
    particle -> particle: reads pos, writes the set's `edge_index` in place, as the default.

        \mathrm{edges}=\big\{(i,j):\ r_{\min}\le\lVert\mathbf x_i-\mathbf x_j\rVert<r\big\}
        \mathrm{edges}  the edge set
        (i, j)         a receiver and one of its neighbours (a ghost carries the index of the point it copies)
        r_{min}        `min_radius`
        x_i, x_j       the two positions (a 2D set is embedded at z = 0)
        r              `radius`, the cutoff in world length

    Reference: NVIDIA Warp, `wp.HashGrid` (Macklin, M. (2022). Warp: a high-performance Python framework for GPU simulation and graphics); the relation: radius_graph."""
    PARAM_UNITS = {}
    SYMBOLS = {'\\mathrm{edges}': 'the edge set', '(i, j)': 'a receiver and one of its neighbours (a ghost carries the index of the point it copies)', 'r_{min}': '`min_radius`', 'x_i, x_j': 'the two positions (a 2D set is embedded at z = 0)', 'r': '`radius`, the cutoff in world length'}

    def forward(self, H, mask=None):
        import numpy as _np
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev = pos.device
        if not str(dev).startswith("cuda") or pos.shape[0] < 4096:
            # SMALL SETS AND CPU RUNS STAY ON THE DEFAULT BODY. Building a grid costs more than
            # scanning a few thousand points, and warp's grid is CUDA-only here; falling back keeps
            # one operator rather than making a spec choose an implementation by population.
            return super().forward(H, mask)
        n = int(pos.shape[0])
        D = int(pos.shape[-1])
        box = getattr(H, "world_size", None)
        L = (box.to(dev).float() if torch.is_tensor(box)
             else torch.as_tensor([float(getattr(H, "world_width", 1.0))] * D, device=dev))
        p3 = torch.zeros(n, 3, device=dev, dtype=torch.float32)
        p3[:, :D] = pos[:, :D].float()
        orig = torch.arange(n, device=dev, dtype=torch.int32)

        if getattr(H, "periodic", False):
            p3[:, :D] = torch.remainder(p3[:, :D], L[:D])
            shifts, r = [], float(self.r_max)
            rng = (-1, 0, 1)
            for sx in rng:
                for sy in rng:
                    for sz in (rng if D == 3 else (0,)):
                        if sx == 0 and sy == 0 and sz == 0:
                            continue
                        s = torch.zeros(3, device=dev)
                        for a, sv in enumerate((sx, sy, sz)[:D]):
                            s[a] = sv * float(L[a])
                        q = p3 + s
                        keep = torch.ones(n, dtype=torch.bool, device=dev)
                        for a in range(D):
                            keep &= (q[:, a] > -r) & (q[:, a] < float(L[a]) + r)
                        if bool(keep.any()):
                            shifts.append((q[keep], orig[keep]))
            if shifts:
                p3 = torch.cat([p3] + [q for q, _ in shifts])
                orig = torch.cat([orig] + [o for _, o in shifts])

        live = (lvl.occ > 0).to(torch.int32)
        m = int(p3.shape[0])
        cell = max(float(self.r_max), 1e-6)
        dim = [max(1, min(256, int(float(L[a]) / cell) + 1)) for a in range(3)] if D == 3 else \
              [max(1, min(256, int(float(L[a]) / cell) + 1)) for a in range(2)] + [1]
        gr = getattr(self, "_grid", None)
        if gr is None or getattr(self, "_grid_dim", None) != tuple(dim):
            gr = self._grid = wp.HashGrid(dim[0], dim[1], dim[2], device=wp.device_from_torch(dev))
            self._grid_dim = tuple(dim)
        wp_pts = wp.from_torch(p3.contiguous(), dtype=wp.vec3)
        gr.build(points=wp_pts, radius=float(self.r_max))

        cnt = torch.zeros(n, device=dev, dtype=torch.int32)
        args = [wp_pts, gr.id, float(self.r_min), float(self.r_max),
                wp.from_torch(orig.contiguous()), wp.from_torch(live.contiguous())]
        zero = torch.zeros(1, device=dev, dtype=torch.int32)
        # THE LAUNCH DEVICE IS NAMED, NOT INFERRED. `wp.launch` without it uses warp's CURRENT
        # device, which is cuda:0 whatever the tensors say -- so a run on cuda:1 compiled the module
        # for one card and read arrays living on the other, and the first launch died with
        # `cudaErrorIllegalAddress`. The failure is not local to the launch: warp then cannot free
        # any of its buffers and the process prints the same error for the rest of its life.
        _wdev = wp.device_from_torch(dev)
        wp.launch(_rg_scan, dim=n, device=_wdev,
                  inputs=args + [wp.from_torch(zero), 1, wp.from_torch(cnt),
                                 wp.from_torch(zero), wp.from_torch(zero)])
        off = torch.cat([torch.zeros(1, device=dev, dtype=torch.int32),
                         torch.cumsum(cnt, 0, dtype=torch.int32)[:-1]])
        E = int(cnt.sum().item())
        if E == 0:
            lvl.edge_index = torch.empty(2, 0, dtype=torch.long, device=dev)
            return {}
        ei = torch.empty(E, device=dev, dtype=torch.int32)
        ej = torch.empty(E, device=dev, dtype=torch.int32)
        wp.launch(_rg_scan, dim=n, device=_wdev,
                  inputs=args + [wp.from_torch(off.contiguous()), 0, wp.from_torch(cnt),
                                 wp.from_torch(ei), wp.from_torch(ej)])
        lvl.edge_index = torch.stack([ei.long(), ej.long()])
        return {}
