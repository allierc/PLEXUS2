"""exp37: a chain is made of domains, a domain of four points (Cedric, 2026-11-02: "an entity is a chain, being made of cadherin entities, cadherin made of 2x2 MPM points").

The cadherin ectodomain as a CONTAINMENT TREE of three sets:  chain (one molecule)  >  domain (one EC domain, five per chain)  >  pt (four points, a 2 x 2 square, per domain). Four points are too few to be an MPM body (no
grid can carry a stress through them), so this level is a bead model: points joined by springs, a thermal bath, no grid. The four points give each domain an orientation, so a chain's bending and twisting come out of the
springs between neighbouring domains' points with no angle term.

    bonded_network    springs inside every domain (rigid square) and between neighbouring domains of one chain (the EC-EC linker); the linker's stiffness is a per-chain block: the calcium switch
    thermal_bath      the overdamped bath: v = mu F + sqrt(2 mu kT / dt) xi
    membrane_anchor   the first domain of a chain held to its membrane, every point kept on its side of it
    tip_well          the trans interaction: a Gaussian well between the EC1 points of chains from opposite membranes

EVERY FORCE IS OVERDAMPED (`EMIT = "velocity"`, v = mobility x force); lengths are nm and energies kT, so the time is Brownian-dynamics time, not calibrated to seconds.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from plexus.models.base import Lateral, Seed
from plexus.models.registry import register_operator


def _maps(H, at, domain, chain):
    """(domain of each point, chain of each point, chain of each domain, rank of each domain in its chain)."""
    P, D = H.level(at), H.level(domain)
    dom = P.parent.long()
    cd = D.parent.long()
    rank = torch.zeros(D.state.shape[0], dtype=torch.long, device=dom.device)
    for k in range(1, 64):                                        # domains of a chain are contiguous blocks: rank = position in the block
        prev = torch.arange(D.state.shape[0], device=dom.device) - k
        m = (prev >= 0) & (cd[prev.clamp(min=0)] == cd)
        if not bool(m.any()):
            break
        rank = rank + m.long()
    return dom, cd[dom], cd, rank


@register_operator("bonded_network", family="interaction", set="particle", kind="lateral",
                   title="join the points of a chain by springs",
                   equation=r"""$$\mathbf F_{ij}=k_{ij}\,(r_{ij}-r^0_{ij})\,\hat{\mathbf r}_{ij},\qquad k_{ij}=\begin{cases}k_{in}&i,j\in\text{one domain}\\ k_{link}(c)&i,j\in\text{neighbouring domains of chain }c\end{cases}$$""")
class BondedNetwork(Lateral):
    r"""a chain's fold as springs between its points at the seeded geometry: every pair inside a domain with stiffness `k_in` (a rigid square) and every pair across a linker (neighbouring domains of one chain) with the chain's stiffness `k_link_block` (the calcium switch).
    pt -> pt (reads its domain and chain): emits the overdamped velocity mobility x spring force.

        \mathbf F_{ij}=k_{ij}\,(r_{ij}-r^0_{ij})\,\hat{\mathbf r}_{ij},\qquad k_{ij}=\begin{cases}k_{in}&i,j\in\text{one domain}\\ k_{link}(c)&i,j\in\text{neighbouring domains of chain }c\end{cases}
        F_{ij}         the spring force on point i from point j
        k_{ij}         the stiffness of the pair: `k_in` inside a domain, the chain's `k_link_block` across a linker
        r_{ij}, r^0_{ij} the pair distance and its value at the seeded geometry
        \hat r_{ij}    the unit vector from i to j

    Reference: Tirion, M. M. (1996). Large amplitude elastic motions in proteins from a single-parameter, atomic analysis. Phys. Rev. Lett. 77:1905 (springs at the deposited geometry)."""
    EXPLAIN = (r"A chain's fold as springs at its seeded geometry: a stiff spring on every pair of points inside one domain, so each domain stays a square, and a spring of the chain's own stiffness on every pair across a linker between "
               r"neighbouring domains. A chain whose linker stiffness is high stands straight; one whose stiffness is low bends freely: that is the calcium switch.")
    PARAM_UNITS = {'k_in': 'energy_per_length2', 'mobility': 'dimensionless'}
    SYMBOLS = {'F_{ij}': 'the spring force on point i from point j', 'k_{ij}': "the stiffness of the pair: `k_in` inside a domain, the chain's `k_link_block` across a linker", 'r_{ij}, r^0_{ij}': 'the pair distance and its value at the seeded geometry',
               '\\hat r_{ij}': 'the unit vector from i to j'}
    EMIT = "velocity"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["domain", "chain"]
    PARAM_ROLES = {"domain": "the_domain_set", "chain": "the_chain_set", "k_in": "stiffness_inside_a_domain", "k_link_block": "chain_block_with_the_linker_stiffness", "mobility": "mobility_of_a_point"}
    MECHANISM_TAGS = ["elastic_network", "protein_fold", "calcium_switch"]
    REFERENCE = "Tirion, M. M. (1996). Phys. Rev. Lett. 77:1905."
    MAY_MUTATE_INTEGRATED_STATE = False

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.domain, self.chain = str(params["domain"]), str(params["chain"])
        self.k_in = float(params.get("k_in", 60.0))
        self.k_link_block = str(params.get("k_link_block", "klink"))
        self.mu = float(params.get("mobility", 1.0))
        self._edges = None

    def _build(self, H):
        P, C = H.level(self.at), H.level(self.chain)
        dom, ch, cd, rank = _maps(H, self.at, self.domain, self.chain)
        X = P.get("pos").detach()
        n = X.shape[0]
        live = (P.occ > 0.5)
        idx = torch.arange(n, device=X.device)
        I, J = torch.meshgrid(idx, idx, indexing="ij")
        m = (I < J) & live[I] & live[J] & (ch[I] == ch[J])
        same = dom[I] == dom[J]
        nb = (rank[dom[I]] - rank[dom[J]]).abs() == 1
        sel = m & (same | nb)
        i, j = I[sel], J[sel]
        k0 = torch.where(dom[i] == dom[j], torch.full((i.numel(),), self.k_in, device=X.device, dtype=X.dtype),
                         C.state[:, C.state_schema[self.k_link_block][0]].to(X.dtype)[ch[i]])
        r0 = (X[j] - X[i]).norm(dim=1)
        self._edges = (i, j, k0, r0)

    def forward(self, H, mask=None):
        if self._edges is None:
            self._build(H)
        P = H.level(self.at)
        X = P.get("pos")
        i, j, k0, r0 = self._edges
        d = X[j] - X[i]
        r = d.norm(dim=1).clamp_min(1e-9)
        F = (k0 * (r - r0) / r)[:, None] * d
        f = torch.zeros_like(X).index_add(0, i, F).index_add(0, j, -F)
        return {self.at: self.mu * f}


@register_operator("thermal_bath", family="mechanics", set="particle", kind="lateral",
                   title="the overdamped thermal bath",
                   equation=r"""$$\mathbf v_i=\sqrt{2\mu\,k_BT/\Delta t}\;\boldsymbol\xi_i,\qquad \xi_i\sim\mathcal N(0,1)$$""")
class ThermalBath(Lateral):
    r"""the thermal kicks of an overdamped bead: a Gaussian velocity of variance 2 mu kT / dt on every live point, so that over a frame the point diffuses by sqrt(2 mu kT dt).
    pt -> pt: emits the random velocity.

        \mathbf v_i=\sqrt{2\mu\,k_BT/\Delta t}\;\boldsymbol\xi_i,\qquad \xi_i\sim\mathcal N(0,1)
        v_i            the random velocity of point i
        \mu            `mobility`, the point's mobility
        k_BT           `kT`, the thermal energy
        \Delta t       the frame step
        \xi_i          a standard normal vector

    Reference: Ermak, D. L. & McCammon, J. A. (1978). Brownian dynamics with hydrodynamic interactions. J. Chem. Phys. 69:1352."""
    EXPLAIN = r"Adds the thermal kick of an overdamped bead: over one frame a point diffuses by $\sqrt{2\mu k_BT\,\Delta t}$ in each direction."
    PARAM_UNITS = {'kT': 'energy', 'mobility': 'dimensionless'}
    SYMBOLS = {'v_i': 'the random velocity of point i', '\\mu': '`mobility`', 'k_BT': '`kT`, the thermal energy', '\\Delta t': 'the frame step', '\\xi_i': 'a standard normal vector'}
    EMIT = "velocity"
    SUPPORTED_DIMS = [2, 3]
    PARAM_ROLES = {"kT": "thermal_energy", "mobility": "mobility_of_a_point", "seed": "rng_seed"}
    MECHANISM_TAGS = ["brownian", "thermal_noise"]
    REFERENCE = "Ermak, D. L. & McCammon, J. A. (1978). J. Chem. Phys. 69:1352."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.kT = float(params.get("kT", 1.0))
        self.mu = float(params.get("mobility", 1.0))
        self.seed = int(params.get("seed", 0))
        self._g = None

    def forward(self, H, mask=None):
        P = H.level(self.at)
        X = P.get("pos")
        if self._g is None:
            self._g = torch.Generator(device=X.device).manual_seed(self.seed)
        dt = float(getattr(H, "dt", 1.0))
        z = torch.randn(X.shape, device=X.device, dtype=X.dtype, generator=self._g)
        return {self.at: math.sqrt(2.0 * self.mu * self.kT / dt) * z * (P.occ > 0.5).to(X.dtype)[:, None]}


@register_operator("membrane_anchor", family="interaction", set="particle", kind="lateral",
                   title="hold a chain's first domain to its membrane",
                   equation=r"""$$\mathbf F_i=-k_a\,(\mathbf x_i-\mathbf s_i)_{\perp n}\ -\ k_y\,(y_i-y_m)\ \ (\text{first domain}),\qquad F_{y,i}=k_w\,\max(0,\,d_w-\lvert y_i-y_m\rvert)\,\sigma\ \ (\text{every point})$$""")
class MembraneAnchor(Lateral):
    r"""a chain stands on its membrane: the points of its first domain are held at the membrane's height `k_y` and to their site in the plane weakly (`k_a`, so the anchor diffuses a little, as a lipid-anchored protein does), and every point is pushed back if it nears or crosses the membrane.
    pt -> pt (reads its domain, chain and the chain's block `side`): emits the overdamped velocity.

        \mathbf F_i=-k_a\,(\mathbf x_i-\mathbf s_i)_{\perp n}\ -\ k_y\,(y_i-y_m)\ \ (\text{first domain}),\qquad F_{y,i}=k_w\,\max(0,\,d_w-\lvert y_i-y_m\rvert)\,\sigma\ \ (\text{every point})
        F_i            the force on point i
        k_a, k_y       `k_site` (in the plane) and `k_height` (across it)
        s_i            the point's site, its seeded position
        y_m            the height of the chain's membrane (0 for side +1, `gap` for side -1)
        k_w, d_w       `k_wall` and `wall_distance`: the wall's stiffness and reach
        \sigma         the chain's side, +1 above the lower membrane, -1 below the upper one

    Reference: Plexus exp37, the cadherin chain (this work, 2026-11)."""
    EXPLAIN = r"Holds each chain to its membrane: the first domain's points keep the membrane's height and stay near their site in the plane, and every point is pushed back from the membrane."
    PARAM_UNITS = {'gap': 'length', 'k_site': 'energy_per_length2', 'k_height': 'energy_per_length2', 'k_wall': 'energy_per_length2', 'wall_distance': 'length'}
    SYMBOLS = {'F_i': 'the force on point i', 'k_a, k_y': '`k_site` (in the plane) and `k_height` (across it)', 's_i': "the point's site, its seeded position", 'y_m': "the height of the chain's membrane",
               'k_w, d_w': "`k_wall` and `wall_distance`", '\\sigma': "the chain's side"}
    EMIT = "velocity"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["domain", "chain", "gap"]
    PARAM_ROLES = {"domain": "the_domain_set", "chain": "the_chain_set", "gap": "distance_between_the_two_membranes", "k_site": "anchor_stiffness_in_the_plane", "k_height": "anchor_stiffness_across",
                   "k_wall": "wall_stiffness", "wall_distance": "wall_reach", "side_block": "chain_block_with_the_side"}
    MECHANISM_TAGS = ["membrane", "anchor"]
    REFERENCE = "Plexus exp37 (this work, 2026-11)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.domain, self.chain = str(params["domain"]), str(params["chain"])
        self.gap = float(params["gap"])
        self.k_a, self.k_y = float(params.get("k_site", 0.03)), float(params.get("k_height", 20.0))
        self.k_w, self.d_w = float(params.get("k_wall", 50.0)), float(params.get("wall_distance", 1.6))
        self.side_block = str(params.get("side_block", "side"))
        self.mu = float(params.get("mobility", 1.0))
        self._site = None

    def forward(self, H, mask=None):
        P, C = H.level(self.at), H.level(self.chain)
        X = P.get("pos")
        dom, ch, cd, rank = _maps(H, self.at, self.domain, self.chain)
        if self._site is None:
            self._site = X.detach().clone()
        side = C.state[:, C.state_schema[self.side_block][0]].to(X.dtype)[ch]
        ym = torch.where(side > 0, torch.zeros_like(side), torch.full_like(side, self.gap))
        first = (rank[dom] == 0).to(X.dtype)
        F = torch.zeros_like(X)
        F[:, [0, 2]] = -self.k_a * first[:, None] * (X[:, [0, 2]] - self._site[:, [0, 2]])
        F[:, 1] = -self.k_y * first * (X[:, 1] - ym)
        yy = (X[:, 1] - ym) * side
        F[:, 1] += side * self.k_w * (self.d_w - yy).clamp(min=0.0) * (1.0 - first)
        return {self.at: self.mu * F * (P.occ > 0.5).to(X.dtype)[:, None]}


@register_operator("tip_well", family="interaction", set="particle", kind="lateral",
                   title="the trans interaction between two EC1 tips",
                   equation=r"""$$U=-\varepsilon\,e^{-r^2/w^2}\ \ \text{between EC1 points of chains from opposite membranes},\qquad \mathbf F=-\nabla U$$""")
class TipWell(Lateral):
    r"""a Gaussian well between the EC1 points of chains that stand on opposite membranes: the strand-swapped or X trans-dimer reduced to its range and depth. Both numbers are ASSUMED in this rig and are what the structure-based work has to set.
    pt -> pt (reads domain, chain and `side`): emits the overdamped velocity.

        U=-\varepsilon\,e^{-r^2/w^2}\ \ \text{between EC1 points of chains from opposite membranes},\qquad \mathbf F=-\nabla U
        U              the pair energy
        \varepsilon    `depth`, the well depth in kT
        w              `width`, the well's Gaussian width
        r              the distance between two EC1 points

    Reference: Plexus exp37 (this work, 2026-11); the trans-dimer of E-cadherin, Harrison, O. J. et al. (2011). Structure 19:244."""
    EXPLAIN = r"Draws the EC1 points of chains from opposite membranes together with a Gaussian well of depth $\varepsilon$ and width $w$: the trans bond of two cadherins."
    PARAM_UNITS = {'depth': 'energy', 'width': 'length'}
    SYMBOLS = {'U': 'the pair energy', '\\varepsilon': '`depth`, the well depth in kT', 'w': '`width`, the Gaussian width', 'r': 'the distance between two EC1 points'}
    EMIT = "velocity"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["domain", "chain"]
    PARAM_ROLES = {"domain": "the_domain_set", "chain": "the_chain_set", "depth": "well_depth_kT", "width": "well_gaussian_width", "side_block": "chain_block_with_the_side", "tip_rank": "rank_of_the_EC1_domain_in_the_chain"}
    MECHANISM_TAGS = ["trans_dimer", "cadherin"]
    REFERENCE = "Harrison, O. J. et al. (2011). Structure 19:244."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.domain, self.chain = str(params["domain"]), str(params["chain"])
        self.eps, self.w = float(params.get("depth", 8.0)), float(params.get("width", 1.2))
        self.side_block = str(params.get("side_block", "side"))
        self.tip_rank = int(params.get("tip_rank", 4))
        self.mu = float(params.get("mobility", 1.0))

    def forward(self, H, mask=None):
        P, C = H.level(self.at), H.level(self.chain)
        X = P.get("pos")
        dom, ch, cd, rank = _maps(H, self.at, self.domain, self.chain)
        side = C.state[:, C.state_schema[self.side_block][0]].to(X.dtype)[ch]
        tips = torch.nonzero((rank[dom] == self.tip_rank) & (P.occ > 0.5)).reshape(-1)
        F = torch.zeros_like(X)
        if tips.numel() > 1:
            a, b = torch.meshgrid(tips, tips, indexing="ij")
            m = (side[a] > 0) & (side[b] < 0)
            a, b = a[m], b[m]
            d = X[b] - X[a]
            r2 = (d * d).sum(1)
            near = r2 < (4.0 * self.w) ** 2
            a, b, d, r2 = a[near], b[near], d[near], r2[near]
            f = (self.eps * 2.0 / self.w ** 2 * torch.exp(-r2 / self.w ** 2))[:, None] * d
            F = F.index_add(0, a, f).index_add(0, b, -f)
        return {self.at: self.mu * F}


@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="chains",
                   title="stand cadherin chains on two membranes",
                   equation=r"""$$\mathbf x_{c,d,(a,b)}=\mathbf s_c+\sigma_c\,\big(h_0+\ell\,d\big)\,\hat{\mathbf y}+\big(0,\ \pm\tfrac{h}{2}\,\text{along }y,\ \pm\tfrac{w}{2}\,\text{along }z\big)$$""")
class SeedChains(Seed):
    r"""chains of EC domains standing on two membranes: `n_per_side` chains on the lower membrane (side +1, up) and as many on the upper one (side -1, down), each of `domains` domains `link` apart whose four points form a `h` x `w` square; every chain gets its side and the stiffness of its linkers (`klink` for calcium bound, `klink_free` for none, by `ca`).
    pt -> pt, domain, chain: writes the points' positions and the chain blocks `side` and `klink`.

        \mathbf x_{c,d,(a,b)}=\mathbf s_c+\sigma_c\,\big(h_0+\ell\,d\big)\,\hat{\mathbf y}+\big(0,\ \pm\tfrac{h}{2}\,\text{along }y,\ \pm\tfrac{w}{2}\,\text{along }z\big)
        x_{c,d,(a,b)}  the point (a, b) of domain d of chain c
        s_c            the chain's site on its membrane
        \sigma_c      the chain's side, +1 or -1
        h_0, \ell     `first` (the first domain's height above the membrane) and `link` (the spacing of the domains)
        h, w           `height`, `width` of a domain's square

    Reference: Plexus exp37, the cadherin chain (this work, 2026-11)."""
    PARAM_UNITS = {'gap': 'length', 'box': 'length', 'link': 'length', 'first': 'length', 'height': 'length', 'width': 'length'}
    SYMBOLS = {'x_{c,d,(a,b)}': 'the point (a, b) of domain d of chain c', 's_c': "the chain's site on its membrane", '\\sigma_c': "the chain's side", 'h_0, \\ell': '`first` and `link`', 'h, w': '`height`, `width` of a domain'}
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["chain", "domain", "gap", "n_per_side"]
    PARAM_ROLES = {"chain": "the_chain_set", "domain": "the_domain_set", "gap": "membrane_to_membrane_distance", "n_per_side": "chains_per_membrane", "box": "side_of_the_square_patch", "ca": "calcium_bound_or_not",
                   "klink": "linker_stiffness_with_calcium", "klink_free": "linker_stiffness_without"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.chain, self.domain = str(params["chain"]), str(params["domain"])
        self.gap, self.nps, self.box = float(params["gap"]), int(params["n_per_side"]), float(params.get("box", 36.0))
        self.link, self.first = float(params.get("link", 4.6)), float(params.get("first", 3.0))
        self.h, self.w = float(params.get("height", 2.2)), float(params.get("width", 1.6))
        self.ca = bool(params.get("ca", True))
        self.klink, self.klink_free = float(params.get("klink", 40.0)), float(params.get("klink_free", 1.5))
        self.seed = int(params.get("seed", 0))

    def forward(self, H, mask=None):
        P, D, C = H.level(self.at), H.level(self.domain), H.level(self.chain)
        rng = np.random.default_rng(self.seed)
        nC = int(C.state.shape[0])
        g = int(math.ceil(math.sqrt(self.nps)))
        cells = [(i, j) for i in range(g) for j in range(g)]
        rng.shuffle(cells)
        side = np.array([1.0] * self.nps + [-1.0] * (nC - self.nps))
        Xs = P.state.detach().cpu().numpy().copy()
        pa, pb = P.state_schema["pos"]
        par = P.parent.detach().cpu().numpy(); dpar = D.parent.detach().cpu().numpy()
        for c in range(nC):
            i, j = cells[c % self.nps]
            off = 0.0 if side[c] > 0 else 0.3 * self.box / g
            sx, sz = (i + 0.5) * self.box / g + off + rng.normal(0, 0.8), (j + 0.5) * self.box / g + off + rng.normal(0, 0.8)
            y0 = 0.0 if side[c] > 0 else self.gap
            doms = np.flatnonzero(dpar == c)
            for d, dm in enumerate(doms):
                pts = np.flatnonzero(par == dm)
                yc = y0 + side[c] * (self.first + self.link * d)
                corners = [(-self.h / 2, -self.w / 2), (-self.h / 2, self.w / 2), (self.h / 2, -self.w / 2), (self.h / 2, self.w / 2)]
                for q, pt in enumerate(pts):
                    Xs[pt, pa:pb] = [sx, yc + corners[q][0], sz + corners[q][1]]
        P.state = torch.as_tensor(Xs, device=P.state.device, dtype=P.state.dtype)
        P.occ = torch.ones_like(P.occ)
        st = C.state.clone()
        st[:, C.state_schema["side"][0]] = torch.as_tensor(side, dtype=st.dtype, device=st.device)
        st[:, C.state_schema["klink"][0]] = self.klink if self.ca else self.klink_free
        C.state = st
        C.occ = torch.ones_like(C.occ); D.occ = torch.ones_like(D.occ)
        return {}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# THE CHAIN AS ONE MPM BODY (Cedric, 2026-11-02: "model the cadherin as a set of points without partitioning the EC"): the molecule is a rod of material points on its own grid; the EC domains are a LABEL along it
# (the point block `ec`), and the linkers between them are softer or stiffer by the calcium switch: the elasticity is the Young's modulus, not a spring constant.
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from plexus.operators.adhesion_ops import _mpm_reset_points, _set_block                      # noqa: E402


@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="rods",
                   title="stand cadherin rods of material points on two membranes",
                   equation=r"""$$\mathbf x_{c,(a,j,b)}=\mathbf s_c+\sigma_c\,\big(h_0+s\,j\big)\,\hat{\mathbf y}+s\,(a-\tfrac{n-1}{2})\,\hat{\mathbf x}+s\,(b-\tfrac{n-1}{2})\,\hat{\mathbf z},\qquad E_i=E\,r_c^{[i\in\text{linker}]}$$""")
class SeedRods(Seed):
    r"""cadherin molecules as rods of material points on two membranes: `n_per_side` rods standing up from the lower membrane (side +1) and as many hanging from the upper one (side -1), each `length` long and `n` x `n` points across at spacing `spacing`; the point block `ec` labels the five EC domains along it (`domain_length` each), `lk` the linker, the last `linker_length` of a domain; the linker's Young's modulus is the domain's times `ratio` (per chain, from the cell block `ratio`: the calcium switch).
    pt -> pt, cell: writes the points' positions, volumes and blocks `ec`, `lk`, rescales `mu` and `la` of the linker points, writes the chain blocks `side`, `ratio`.

        \mathbf x_{c,(a,j,b)}=\mathbf s_c+\sigma_c\,\big(h_0+s\,j\big)\,\hat{\mathbf y}+s\,(a-\tfrac{n-1}{2})\,\hat{\mathbf x}+s\,(b-\tfrac{n-1}{2})\,\hat{\mathbf z},\qquad E_i=E\,r_c^{[i\in\text{linker}]}
        x_{c,(a,j,b)}  the point at lattice site (a, j, b) of rod c
        s_c            the rod's site on its membrane
        \sigma_c       the rod's side, +1 or -1
        h_0, s         `first` (height of the first point above the membrane) and `spacing`
        n              `across`, points across the rod
        E_i, E, r_c    the Young's modulus of point i, the domain's, and the linker's ratio of rod c

    Reference: Plexus exp37, the cadherin chain (this work, 2026-11)."""
    PARAM_UNITS = {'gap': 'length', 'box': 'length', 'length': 'length', 'spacing': 'length', 'pitch': 'length', 'shift': 'length', 'origin': 'length', 'first': 'length', 'domain_length': 'length', 'linker_length': 'length'}
    SYMBOLS = {'x_{c,(a,j,b)}': 'the point at lattice site (a, j, b) of rod c', 's_c': "the rod's site on its membrane", '\\sigma_c': "the rod's side", 'h_0, s': '`first` and `spacing`', 'n': '`across`',
               'E_i, E, r_c': "the Young's modulus of point i, the domain's, and the linker's ratio of rod c"}
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["cell_set", "gap", "n_per_side"]
    PARAM_ROLES = {"cell_set": "the_rod_set", "gap": "membrane_to_membrane_distance", "n_per_side": "rods_per_membrane", "box": "side_of_the_square_patch", "length": "rod_length", "across": "points_across",
                   "spacing": "point_lattice_spacing", "pitch": "distance_between_rod_sites_on_a_membrane", "shift": "upper_lattice_displacement", "origin": "lattice_start", "domain_length": "length_of_one_EC_domain_with_its_linker", "linker_length": "length_of_the_linker", "ratio_block": "cell_block_with_the_linker_ratio"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.cell_set = str(params["cell_set"])
        self.gap, self.nps, self.box = float(params["gap"]), int(params["n_per_side"]), float(params.get("box", 36.0))
        self.length, self.n = float(params.get("length", 23.0)), int(params.get("across", 3))
        self.s, self.first = float(params.get("spacing", 1.0)), float(params.get("first", 1.5))
        self.dl, self.ll = float(params.get("domain_length", 4.6)), float(params.get("linker_length", 1.0))
        self.ratio = float(params.get("ratio", 1.0))
        self.seed = int(params.get("seed", 0))
        self.pitch = float(params["pitch"]) if params.get("pitch") else None
        self.origin = float(params.get("origin", 0.0))                                         # where the lattice starts along x and z, to leave room for the rods to slide
        self.shift = float(params.get("shift", 0.0))                                           # `pitch`: a regular square lattice of rod sites, the upper membrane's the mirror image of the lower's (Cedric, 2026-11-02: 12 x 12 on each plane); the upper lattice displaced by `shift` along x and along z, so a rod meets its partner side by side, not through it

    def forward(self, H, mask=None):
        P, C = H.level(self.at), H.level(self.cell_set)
        rng = np.random.default_rng(self.seed)
        nC = int(C.state.shape[0]); ny = int(round(self.length / self.s)); npts = self.n * self.n * ny
        g = int(math.ceil(math.sqrt(self.nps))); cells = [(i, j) for i in range(g) for j in range(g)]
        if self.pitch is None:
            rng.shuffle(cells)
        side = np.array([1.0] * self.nps + [-1.0] * (nC - self.nps))
        live = np.flatnonzero(C.occ.detach().cpu().numpy() > 0.5)
        parent = P.parent.detach().cpu().numpy(); dev, dt_ = P.state.device, P.state.dtype
        occ = torch.zeros_like(P.occ); cen = C.state[:, C.state_schema["pos"][0]:C.state_schema["pos"][1]].detach().cpu().numpy().copy()
        a = (np.arange(self.n) - 0.5 * (self.n - 1)) * self.s
        jj = np.arange(ny) * self.s
        lat = np.stack(np.meshgrid(a, jj, a, indexing="ij"), -1).reshape(-1, 3)                       # [n * ny * n, 3]: x across, y along, z across
        along = lat[:, 1]
        ec = np.minimum(np.floor(along / self.dl), 4.0); lk = ((along % self.dl) >= (self.dl - self.ll)) & (ec < 4)
        for c, cj in enumerate(live):
            i, j = cells[c % self.nps]
            if self.pitch is not None:
                sx, sz = self.origin + (i + 0.5) * self.pitch + (self.shift if side[c] < 0 else 0.0), self.origin + (j + 0.5) * self.pitch + (self.shift if side[c] < 0 else 0.0)
            else:
                off = 0.0 if side[c] > 0 else 0.3 * self.box / g
                sx, sz = (i + 0.5) * self.box / g + off + rng.normal(0, 0.8), (j + 0.5) * self.box / g + off + rng.normal(0, 0.8)
            y0 = 0.0 if side[c] > 0 else self.gap
            pos = np.stack([sx + lat[:, 0], y0 + side[c] * (self.first + lat[:, 1]), sz + lat[:, 2]], 1)
            cen[cj] = pos.mean(0)
            blk = np.flatnonzero(parent == cj)[:npts]
            if len(blk) != npts:
                raise ValueError(f"seed_positions[rods]: rod {cj} has {len(blk)} point slots, {npts} needed: set per_parent to {npts}")
            idx = torch.as_tensor(blk, device=dev)
            _mpm_reset_points(P, idx, torch.as_tensor(pos, device=dev, dtype=dt_), torch.full((npts,), self.s ** 3, device=dev, dtype=dt_))
            _set_block(P, "ec", idx, torch.as_tensor(ec, device=dev, dtype=dt_)); _set_block(P, "lk", idx, torch.as_tensor(lk.astype(float), device=dev, dtype=dt_))
            _set_block(P, "pid", idx, float(cj))
            lkt = idx[torch.as_tensor(lk, device=dev)]
            r_c = float(C.state[cj, C.state_schema[self.ratio_block][0]]) if getattr(self, "ratio_block", None) else self.ratio
            P.mu[lkt] = P.mu[lkt] * r_c; P.la[lkt] = P.la[lkt] * r_c
            occ[idx] = 1.0
        P.occ = occ
        ca_, cb_ = C.state_schema["pos"]
        st = C.state.clone(); st[:, ca_:cb_] = torch.as_tensor(cen, dtype=st.dtype, device=st.device)
        st[:, C.state_schema["side"][0]] = torch.as_tensor(side, dtype=st.dtype, device=st.device); C.state = st
        print(f"[seed_positions:rods] {len(live)} rods of {npts} points ({self.n} x {ny}, spacing {self.s:g}), linker modulus x{self.ratio:g}, gap {self.gap:g}", flush=True)
        return {}


@register_operator("thermal_bath", implementation="langevin", family="mechanics", set="particle", kind="lateral",
                   title="the thermal bath of an MPM body: friction and kicks",
                   equation=r"""$$\mathbf a_i=-\gamma\,\mathbf v_i+\sqrt{2\gamma\,k_BT/(m_i\,\Delta t_s)}\;\boldsymbol\xi_i$$""")
class ThermalBathLangevin(Lateral):
    r"""a Langevin bath for material points: the acceleration -gamma v plus a random one of variance 2 gamma kT / (m dt_s) per MPM substep, so the body fluctuates at kT.
    pt -> pt: emits `mpm_acceleration`.

        \mathbf a_i=-\gamma\,\mathbf v_i+\sqrt{2\gamma\,k_BT/(m_i\,\Delta t_s)}\;\boldsymbol\xi_i
        a_i            the acceleration of point i
        \gamma         `friction`, the drag rate per time
        k_BT           `kT`, the thermal energy
        m_i            the point's mass
        \Delta t_s     `substep_dt`, the MPM substep
        \xi_i          a standard normal vector

    Reference: Langevin, P. (1908). Sur la theorie du mouvement brownien. C. R. Acad. Sci. 146:530."""
    EXPLAIN = r"The bath of a body of material points: a drag $-\gamma\mathbf v$ and a random kick of the size that makes the body fluctuate with energy $k_BT$."
    PARAM_UNITS = {'friction': 'rate', 'kT': 'energy', 'substep_dt': 'time'}
    SYMBOLS = {'a_i': 'the acceleration of point i', '\\gamma': '`friction`, the drag rate per time', 'k_BT': '`kT`, the thermal energy', 'm_i': "the point's mass", '\\Delta t_s': '`substep_dt`, the MPM substep', '\\xi_i': 'a standard normal vector'}
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    PARAM_ROLES = {"friction": "drag_rate", "kT": "thermal_energy", "substep_dt": "mpm_substep", "seed": "rng_seed"}
    MECHANISM_TAGS = ["langevin", "thermal_noise"]
    REFERENCE = "Langevin, P. (1908). C. R. Acad. Sci. 146:530."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.gamma, self.kT = float(params.get("friction", 5.0)), float(params.get("kT", 1.0))
        self.dts, self.seed = float(params.get("substep_dt", 0.01)), int(params.get("seed", 0))
        self.drift = [float(v) for v in params.get("drift", [0.0, 0.0, 0.0])]            # a constant acceleration on every point (a test load: the cantilever)
        self._g = None

    def forward(self, H, mask=None):
        P = H.level(self.at)
        X = P.get("pos"); V = P.get("vel")
        if self._g is None:
            self._g = torch.Generator(device=X.device).manual_seed(self.seed)
        m = (P.mass.to(X.dtype) if getattr(P, "mass", None) is not None else torch.ones(X.shape[0], device=X.device, dtype=X.dtype)).clamp_min(1e-30)
        z = torch.randn(X.shape, device=X.device, dtype=X.dtype, generator=self._g)
        a = -self.gamma * V + torch.sqrt(2.0 * self.gamma * self.kT / (m * self.dts))[:, None] * z + torch.as_tensor(self.drift, device=X.device, dtype=X.dtype)[None]
        return {self.at: a * (P.occ > 0.5).to(X.dtype)[:, None]}


@register_operator("membrane_anchor", implementation="mpm", family="interaction", set="particle", kind="lateral",
                   title="hold a rod's base to its membrane",
                   equation=r"""$$\mathbf a_i=\big[-k_a\,(\mathbf x_i-\mathbf s_i)_{\perp y}-k_y\,(y_i-y_m)\,\hat{\mathbf y}\big]_{ec_i=0}\,/\,m_i+\sigma\,k_w\max(0,\,d_w-\sigma(y_i-y_m))\,\hat{\mathbf y}/m_i$$""")
class MembraneAnchorMPM(Lateral):
    r"""a rod of material points standing on its membrane: the points of its foot (seeded within `clamp_height` of the membrane) are held to their sites, and every point is pushed back from the membrane.
    pt -> pt (reads `ec` and the cell's `side`): emits `mpm_acceleration`.

        \mathbf a_i=\big[-k_a\,(\mathbf x_i-\mathbf s_i)_{\perp y}-k_y\,(y_i-y_m)\,\hat{\mathbf y}\big]_{ec_i=0}\,/\,m_i+\sigma\,k_w\max(0,\,d_w-\sigma(y_i-y_m))\,\hat{\mathbf y}/m_i
        a_i            the acceleration of point i
        k_a, k_y       `k_site` (in the plane) and `k_height` (across it)
        s_i            the point's site, its seeded position
        y_m            the height of the rod's membrane (0 for side +1, `gap` for side -1)
        k_w, d_w       `k_wall` and `wall_distance`
        \sigma         the rod's side

    Reference: Plexus exp37 (this work, 2026-11)."""
    EXPLAIN = r"Holds each rod to its membrane: the first domain's points keep the membrane's height and stay near their site, and every point is pushed back from the membrane."
    PARAM_UNITS = {'gap': 'length', 'diffusion': 'length_squared_per_time', 'exclusion': 'length', 'k_exclusion': 'energy_per_length2', 'kT': 'energy', 'substep_dt': 'time', 'k_site': 'stress', 'k_height': 'stress', 'k_wall': 'stress', 'wall_distance': 'length'}
    SYMBOLS = {'a_i': 'the acceleration of point i', 'k_a, k_y': '`k_site` and `k_height`', 's_i': "the point's seeded position", 'y_m': "the height of the rod's membrane", 'k_w, d_w': '`k_wall`, `wall_distance`', '\\sigma': "the rod's side"}
    MAY_MUTATE_INTEGRATED_STATE = True                       # with `diffusion` > 0 it translates the whole rod and its site by the sliding step
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["cells", "gap"]
    PARAM_ROLES = {"cells": "the_rod_set", "gap": "membrane_to_membrane_distance", "k_site": "anchor_in_plane", "diffusion": "foot_lateral_diffusion_coefficient", "exclusion": "foot_to_foot_minimum_distance", "k_exclusion": "foot_exclusion_stiffness", "kT": "thermal_energy", "substep_dt": "mpm_substep", "k_height": "anchor_across", "k_wall": "wall_stiffness", "wall_distance": "wall_reach", "clamp_height": "height_of_the_rod_foot_held_to_its_site"}
    MECHANISM_TAGS = ["membrane", "anchor"]
    REFERENCE = "Plexus exp37 (this work, 2026-11)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt"); self.cells = str(params["cells"]); self.gap = float(params["gap"])
        self.k_a, self.k_y = float(params.get("k_site", 0.5)), float(params.get("k_height", 40.0))
        self.k_w, self.d_w = float(params.get("k_wall", 60.0)), float(params.get("wall_distance", 1.6))
        self.clamp_h = float(params.get("clamp_height", 2.3))
        self.d_ex, self.k_ex = float(params.get("exclusion", 0.0)), float(params.get("k_exclusion", 20.0))      # two rods of one membrane keep their feet at least `exclusion` apart (a soft wall of stiffness `k_exclusion`)
        self.D, self.kT, self.dts = float(params.get("diffusion", 0.0)), float(params.get("kT", 1.0)), float(params.get("substep_dt", 0.005))       # the foot's lateral diffusion in its membrane (Cedric, 2026-11-02): D in length^2 per time, kT as the thermal bath's
        self._site, self._S, self._g = None, None, None

    def forward(self, H, mask=None):
        P, C = H.level(self.at), H.level(self.cells)
        X = P.get("pos")
        if self._site is None:
            self._site = X.detach().clone()
        cid = H.lift_index(self.at, self.cells).long()
        side = C.state[:, C.state_schema["side"][0]].to(X.dtype)[cid]
        ym = torch.where(side > 0, torch.zeros_like(side), torch.full_like(side, self.gap))
        first = ((self._site[:, 1] - ym).abs() < self.clamp_h).to(X.dtype)                     # the points seeded within `clamp_height` of the membrane: the rod's foot
        site = self._site[:, [0, 2]]
        nC = int(C.state.shape[0])
        if self.D > 0.0:
            # the rod's site slides in its membrane as a Brownian particle of diffusion D and mobility D/kT, driven by the net trans force on its rod (`tip_well`, one substep late); the WHOLE rod is carried with it
            if self._S is None:
                self._S = torch.zeros(nC, 2, device=X.device, dtype=X.dtype); self._g = torch.Generator(device=X.device).manual_seed(11)
            site = site + self._S[cid]
            ft = getattr(H, "_tip_force", None)
            drive = ft[:, [0, 2]].to(X.dtype) if ft is not None and ft.shape[0] == nC else torch.zeros(nC, 2, device=X.device, dtype=X.dtype)
            if self.d_ex > 0.0:
                occf0 = (P.occ > 0.5).to(X.dtype); ft_ = (first * occf0)[:, None]
                n0 = torch.zeros(nC, device=X.device, dtype=X.dtype).index_add_(0, cid, ft_[:, 0]).clamp_min(1.0)
                cen = torch.zeros(nC, 2, device=X.device, dtype=X.dtype).index_add_(0, cid, X[:, [0, 2]] * ft_) / n0[:, None]
                sd = C.state[:, C.state_schema["side"][0]].to(X.dtype)
                dv = cen[:, None, :] - cen[None, :, :]; r = dv.norm(dim=2).clamp_min(1e-6)
                same = (sd[:, None] == sd[None, :]) & ~torch.eye(nC, dtype=torch.bool, device=X.device)
                push = (self.k_ex * (self.d_ex - r).clamp(min=0.0) * same.to(X.dtype) / r)[..., None] * dv
                drive = drive + push.sum(1)                                                          # the push of the other rods of its membrane on each rod's site
            dS = (self.D / self.kT) * drive * self.dts + (2.0 * self.D * self.dts) ** 0.5 * torch.randn(nC, 2, device=X.device, dtype=X.dtype, generator=self._g)
            self._S = self._S + dS
            occf = (P.occ > 0.5).to(X.dtype)
            with torch.no_grad():                                                                   # translate the whole rod and its site by the step: a rigid slide, no strain, nothing for the bath to drag
                X[:, 0] += dS[cid, 0] * occf; X[:, 2] += dS[cid, 1] * occf
        F = torch.zeros_like(X)
        F[:, [0, 2]] = -self.k_a * first[:, None] * (X[:, [0, 2]] - site)
        F[:, 1] = -self.k_y * first * (X[:, 1] - self._site[:, 1])
        F[:, 1] += side * self.k_w * (self.d_w - (X[:, 1] - ym) * side).clamp(min=0.0) * (1.0 - first)
        m = (P.mass.to(X.dtype) if getattr(P, "mass", None) is not None else torch.ones(X.shape[0], device=X.device, dtype=X.dtype)).clamp_min(1e-30)
        return {self.at: F / m[:, None] * (P.occ > 0.5).to(X.dtype)[:, None]}


@register_operator("tip_well", implementation="mpm", family="interaction", set="particle", kind="lateral",
                   title="the trans interaction between two EC1 domains of MPM rods",
                   equation=r"""$$U=-\varepsilon\,e^{-r^2/w^2},\quad r=\lvert\bar{\mathbf x}_b-\bar{\mathbf x}_a\rvert,\qquad \mathbf a_i=\mathbf F_a/M_a\ \ \text{for each point }i\text{ of the EC1 domain of rod }a$$""")
class TipWellMPM(Lateral):
    r"""a Gaussian well between the centres of the EC1 domains (point block `ec` = 4) of two rods that stand on opposite membranes; the pair force is shared equally by the points of each EC1 domain.
    pt -> pt (reads `ec` and the cell's `side`): emits `mpm_acceleration`.

        U=-\varepsilon\,e^{-r^2/w^2},\quad r=\lvert\bar{\mathbf x}_b-\bar{\mathbf x}_a\rvert,\qquad \mathbf a_i=\mathbf F_a/M_a\ \ \text{for each point }i\text{ of the EC1 domain of rod }a
        U              the pair energy
        \varepsilon, w `depth` (kT) and `width` of the well, ASSUMED
        \bar x_a       the centre of the EC1 points of rod a
        F_a, M_a       the force on rod a's EC1 domain and its mass
        a_i            the acceleration of point i

    Reference: Harrison, O. J. et al. (2011). Structure 19:244 (the E-cadherin trans dimer)."""
    EXPLAIN = r"The trans bond between two cadherin rods from opposite membranes: a Gaussian well of depth $\varepsilon$ and width $w$ between the centres of their EC1 domains, the force shared by the points of each."
    PARAM_UNITS = {'depth': 'energy', 'width': 'length'}
    SYMBOLS = {'U': 'the pair energy', '\\varepsilon, w': '`depth` (kT) and `width` of the well', '\\bar x_a': 'the centre of the EC1 points of rod a', 'F_a, M_a': "the force on rod a's EC1 domain and its mass", 'a_i': 'the acceleration of point i'}
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["cells"]
    PARAM_ROLES = {"cells": "the_rod_set", "depth": "well_depth_kT", "width": "well_gaussian_width"}
    MECHANISM_TAGS = ["trans_dimer", "cadherin"]
    REFERENCE = "Harrison, O. J. et al. (2011). Structure 19:244."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt"); self.cells = str(params["cells"])
        self.eps, self.w = float(params.get("depth", 8.0)), float(params.get("width", 1.2))
        self.rcut = 4.0 * self.w

    def forward(self, H, mask=None):
        P, C = H.level(self.at), H.level(self.cells)
        X = P.get("pos"); nC = int(C.state.shape[0])
        cid = H.lift_index(self.at, self.cells).long()
        tip = ((P.get("ec")[:, 0] - 4.0).abs() < 0.5) & (P.occ > 0.5)
        w_ = tip.to(X.dtype)
        n = torch.zeros(nC, device=X.device, dtype=X.dtype).index_add_(0, cid, w_)
        cen = torch.zeros(nC, 3, device=X.device, dtype=X.dtype).index_add_(0, cid, X * w_[:, None]) / n.clamp_min(1.0)[:, None]
        side = C.state[:, C.state_schema["side"][0]].to(X.dtype)
        a = torch.nonzero((side > 0) & (n > 0)).reshape(-1); b = torch.nonzero((side < 0) & (n > 0)).reshape(-1)
        F = torch.zeros_like(X)
        if a.numel() and b.numel():
            ia, ib = torch.meshgrid(a, b, indexing="ij"); ia, ib = ia.reshape(-1), ib.reshape(-1)
            d = cen[ib] - cen[ia]; r2 = (d * d).sum(1); m = r2 < self.rcut ** 2
            ia, ib, d, r2 = ia[m], ib[m], d[m], r2[m]
            f = (self.eps * 2.0 / self.w ** 2 * torch.exp(-r2 / self.w ** 2))[:, None] * d            # on rod a, toward b
            fc = torch.zeros(nC, 3, device=X.device, dtype=X.dtype).index_add(0, ia, f).index_add(0, ib, -f)
            mass = (P.mass.to(X.dtype) if getattr(P, "mass", None) is not None else torch.ones(X.shape[0], device=X.device, dtype=X.dtype))
            M = torch.zeros(nC, device=X.device, dtype=X.dtype).index_add_(0, cid, mass * w_)
            F = (fc / M.clamp_min(1e-30)[:, None])[cid] * w_[:, None]
            H._tip_force = fc                                                                       # the net trans force on each rod, read by membrane_anchor's sliding foot on the next substep
        return {self.at: F}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CADHERIN RODS ON THE FACING SURFACES OF TWO CELLS (Cedric, 2026-11-02: "replace the yellow tube with cadherins on each facing cell surface that would ultimately bind; not a fake rendering").
# The rod of exp37 (one MPM body, EC domains as a label, a soft linker) enlarged so that two of them, tip to tip, span the gap between the cells' surfaces; each rod stands on ONE point of its cell by a spring at
# its foot; the rods and the cells live in two MPM sets with a grid each, and the rods are run in their own block of small substeps (they equilibrate far faster than a cell moves), their reaction on the cell
# points averaged over the block and applied in the cell block.
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="rods_on_cells",
                   title="stand cadherin rods on the facing surfaces of two cells",
                   equation=r"""$$\mathbf x_{r,(a,j,b)}=\mathbf q_r+\sigma_r\,(s\,j)\,\hat{\mathbf x}+s\,(a-\tfrac{n-1}{2})\,\hat{\mathbf y}+s\,(b-\tfrac{n-1}{2})\,\hat{\mathbf z},\qquad \mathbf q_r=\mathbf x_{\pi(r)}$$""")
class SeedRodsOnCells(Seed):
    r"""cadherin rods on the facing surfaces of two cells: `n_rods` per face on a square grid of sites over the contact face, each standing from the cell point nearest its site (its anchor, stored in the rod's block `anchor`) toward the other cell, `length` long, `n` x `n` points across at spacing `spacing`; the block `ec` labels the five EC domains, `lk` the linker (softer by `ratio`), `side` is +1 for the rods of the first cell, -1 for the second.
    pt_rod, pt_cell -> pt_rod: writes the rod points' positions, volumes and blocks and each rod's anchor point and side.

        \mathbf x_{r,(a,j,b)}=\mathbf q_r+\sigma_r\,(s\,j)\,\hat{\mathbf x}+s\,(a-\tfrac{n-1}{2})\,\hat{\mathbf y}+s\,(b-\tfrac{n-1}{2})\,\hat{\mathbf z},\qquad \mathbf q_r=\mathbf x_{\pi(r)}
        x_{r,(a,j,b)}  the point (a, j, b) of rod r
        q_r            the rod's anchor, the position of the cell point pi(r) it stands on
        \sigma_r       the rod's side, +1 for the first cell's rods, -1 for the second's
        s, n           `spacing` and `across`: the lattice of the rod's points

    Reference: Plexus exp37, the cadherin rod (this work, 2026-11)."""
    PARAM_UNITS = {'length': 'length', 'spacing': 'length', 'domain_length': 'length', 'linker_length': 'length'}
    SYMBOLS = {'x_{r,(a,j,b)}': 'the point (a, j, b) of rod r', 'q_r': "the rod's anchor", '\\sigma_r': "the rod's side", 's, n': '`spacing` and `across`'}
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["rods", "cells_points", "n_rods"]
    PARAM_ROLES = {"rods": "the_rod_set", "cells_points": "the_cell_point_set", "n_rods": "rods_per_face", "length": "rod_length", "across": "points_across", "spacing": "point_lattice_spacing",
                   "ratio": "linker_modulus_ratio", "domain_length": "length_of_one_EC_domain_with_its_linker", "linker_length": "length_of_the_linker"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt_rod"); self.rods = str(params["rods"]); self.cpts = str(params["cells_points"]); self.n_rods = int(params["n_rods"])
        self.length, self.n, self.s = float(params.get("length", 0.015)), int(params.get("across", 6)), float(params.get("spacing", 0.00036))
        self.dl, self.ll = float(params.get("domain_length", 0.0030)), float(params.get("linker_length", 0.00145)); self.ratio = float(params.get("ratio", 0.001))

    def forward(self, H, mask=None):
        P, R, CP = H.level(self.at), H.level(self.rods), H.level(self.cpts)
        nR = int(R.state.shape[0]); per = self.n_rods; g = int(math.ceil(math.sqrt(per))); ny = int(round(self.length / self.s)); npts = self.n * self.n * ny
        Xc = CP.get("pos").detach().cpu().numpy(); occ_c = (CP.occ > 0.5).detach().cpu().numpy(); pid = np.rint(CP.get("pid")[:, 0].detach().cpu().numpy()).astype(int)
        c0, c1 = np.unique(pid[occ_c])[:2]; A, B = Xc[occ_c & (pid == c0)], Xc[occ_c & (pid == c1)]
        face = 0.5 * (A[:, 0].max() + B[:, 0].min()); ylo, yhi = max(A[:, 1].min(), B[:, 1].min()), min(A[:, 1].max(), B[:, 1].max()); zlo, zhi = max(A[:, 2].min(), B[:, 2].min()), min(A[:, 2].max(), B[:, 2].max())
        sites = [(ylo + (i + 0.5) * (yhi - ylo) / g, zlo + (j + 0.5) * (zhi - zlo) / g) for i in range(g) for j in range(g)][:per]
        a = (np.arange(self.n) - 0.5 * (self.n - 1)) * self.s; jj = np.arange(ny) * self.s
        lat = np.stack(np.meshgrid(a, jj, a, indexing="ij"), -1).reshape(-1, 3)                      # x across, y along, z across: re-ordered below to (along x)
        along, ca_, cb_ = lat[:, 1], lat[:, 0], lat[:, 2]
        ec = np.minimum(np.floor(along / self.dl), 4.0); lk = ((along % self.dl) >= (self.dl - self.ll)) & (ec < 4)
        parent = P.parent.detach().cpu().numpy(); dev, dt_ = P.state.device, P.state.dtype
        live = np.flatnonzero(R.occ.detach().cpu().numpy() > 0.5); occ = torch.zeros_like(P.occ)
        anchor = np.zeros(nR); side = np.zeros(nR); cen = R.state[:, R.state_schema["pos"][0]:R.state_schema["pos"][1]].detach().cpu().numpy().copy()
        from scipy.spatial import cKDTree
        iA, iB = np.flatnonzero(occ_c & (pid == c0)), np.flatnonzero(occ_c & (pid == c1))
        iA, iB = iA[Xc[iA, 0] > A[:, 0].max() - 0.25 * float(np.median(np.diff(np.sort(np.unique(np.round(A[:, 0], 6))))))], iB[Xc[iB, 0] < B[:, 0].min() + 0.25 * float(np.median(np.diff(np.sort(np.unique(np.round(B[:, 0], 6))))))]
        trees = {+1: cKDTree(Xc[iA][:, [1, 2]]), -1: cKDTree(Xc[iB][:, [1, 2]])}; idxs = {+1: iA, -1: iB}                  # only the points of each cell's FACING layer can carry a rod
        for r, rj in enumerate(live):
            sg = 1.0 if r < per else -1.0; y, z = sites[r % per]
            k = int(trees[int(sg)].query([y, z])[1]); ap = idxs[int(sg)][k]; q = Xc[ap]
            pos = np.stack([q[0] + sg * along, q[1] + ca_, q[2] + cb_], 1); cen[rj] = pos.mean(0)
            blk = np.flatnonzero(parent == rj)[:npts]
            if len(blk) != npts:
                raise ValueError(f"seed_positions[rods_on_cells]: rod {rj} has {len(blk)} point slots, {npts} needed: set per_parent to {npts}")
            idx = torch.as_tensor(blk, device=dev)
            _mpm_reset_points(P, idx, torch.as_tensor(pos, device=dev, dtype=dt_), torch.full((npts,), self.s ** 3, device=dev, dtype=dt_))
            _set_block(P, "ec", idx, torch.as_tensor(ec, device=dev, dtype=dt_)); _set_block(P, "lk", idx, torch.as_tensor(lk.astype(float), device=dev, dtype=dt_)); _set_block(P, "pid", idx, float(rj))
            lkt = idx[torch.as_tensor(lk, device=dev)]; P.mu[lkt] = P.mu[lkt] * self.ratio; P.la[lkt] = P.la[lkt] * self.ratio
            anchor[rj] = ap; side[rj] = sg; occ[idx] = 1.0
        P.occ = occ
        st = R.state.clone(); ca2, cb2 = R.state_schema["pos"]; st[:, ca2:cb2] = torch.as_tensor(cen, dtype=st.dtype, device=st.device)
        st[:, R.state_schema["anchor"][0]] = torch.as_tensor(anchor, dtype=st.dtype, device=st.device); st[:, R.state_schema["side"][0]] = torch.as_tensor(side, dtype=st.dtype, device=st.device); R.state = st
        print(f"[seed_positions:rods_on_cells] {len(live)} rods of {npts} points on the facing surfaces ({per} per face), length {self.length:g}, spacing {self.s:g}", flush=True)
        return {}


@register_operator("rod_foot", family="interaction", set="particle", kind="lateral", title="stand a rod on its cell's surface point",
                   equation=r"""$$\mathbf a_i=-\frac{k_f}{m_i}\,\big(\mathbf x_i-\mathbf q_{\pi(r)}-\mathbf o_i\big)\ \ (i\in\text{foot of rod }r),\qquad \mathbf f_{\pi(r)}\mathrel{+}=\sum_i k_f\,(\mathbf x_i-\mathbf q_{\pi(r)}-\mathbf o_i)$$""")
class RodFoot(Lateral):
    r"""the foot of every rod (its points seeded within `foot_length` of its anchor) is held by a spring `k_foot` to the anchor cell point's current position plus the point's seeded offset, and the equal and opposite force is added to that cell point's running sum (`H._rod_force`, divided by the substeps of the block when `rod_reaction` reads it).
    pt_rod -> pt_rod and pt_cell: emits `mpm_acceleration` on the rod points; emits the reaction as an acceleration on the cell points.

        \mathbf a_i=-\frac{k_f}{m_i}\,\big(\mathbf x_i-\mathbf q_{\pi(r)}-\mathbf o_i\big)\ \ (i\in\text{foot of rod }r),\qquad \mathbf f_{\pi(r)}\mathrel{+}=\sum_i k_f\,(\mathbf x_i-\mathbf q_{\pi(r)}-\mathbf o_i)
        a_i            the acceleration of foot point i of rod r
        k_f            `k_foot`, the foot spring
        q_{pi(r)}      the current position of the cell point the rod stands on
        o_i            the point's seeded offset from that cell point
        f_{pi(r)}      the running sum of the reactions on that cell point

    Reference: Plexus exp37 (this work, 2026-11)."""
    EXPLAIN = r"Stands each rod on one point of its cell: the rod's foot is held by a spring to that point, and the same force pulls the cell point back, so the rods carry load between the cells."
    PARAM_UNITS = {'k_foot': 'stress', 'foot_length': 'length'}
    SYMBOLS = {'a_i': 'the acceleration of foot point i of rod r', 'k_f': '`k_foot`, the foot spring', 'q_{pi(r)}': 'the current position of the cell point the rod stands on', 'o_i': "the point's seeded offset from that cell point",
               'f_{pi(r)}': 'the running sum of the reactions on that cell point'}
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["rods", "cells_points"]
    PARAM_ROLES = {"rods": "the_rod_set", "cells_points": "the_cell_point_set", "k_foot": "foot_spring", "foot_length": "length_of_the_clamped_foot", "carry": "carry_rod_rigidly_with_its_cell_point"}
    MECHANISM_TAGS = ["cadherin", "anchor", "two_scale"]
    REFERENCE = "Plexus exp37 (this work, 2026-11)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt_rod"); self.rods, self.cpts = str(params["rods"]), str(params["cells_points"])
        self.kf, self.foot = float(params.get("k_foot", 5.0)), float(params.get("foot_length", 0.001))
        self.carry = bool(params.get("carry", False))                      # carry the whole rod rigidly with its cell point (Cedric, 2026-11-03): the spring alone cannot drag a heavy rod body, which tore away from its foot
        self._off, self._q = None, None

    MAY_MUTATE_INTEGRATED_STATE = True                                      # with `carry` it translates the rod's points by the step of the cell point it stands on

    def forward(self, H, mask=None):
        P, R, CP = H.level(self.at), H.level(self.rods), H.level(self.cpts)
        X, Xc = P.get("pos"), CP.get("pos")
        rid = H.lift_index(self.at, self.rods).long()
        anchor = R.state[:, R.state_schema["anchor"][0]].round().long()[rid]
        if self._off is None:
            q0 = R.state[:, R.state_schema["pos"][0]:R.state_schema["pos"][1]]
            side = R.state[:, R.state_schema["side"][0]][rid]
            self._off = (X.detach() - Xc.detach()[anchor]).clone()
            self._foot = ((self._off[:, 0] * side) < self.foot) & (P.occ > 0.5)
        if self.carry:
            q = Xc.detach()[anchor]
            if self._q is not None:
                with torch.no_grad():
                    X += (q - self._q) * (P.occ > 0.5).to(X.dtype)[:, None]                     # the rod moves rigidly with the cell point
            self._q = q.clone()
        F = -self.kf * (X - Xc[anchor] - self._off) * self._foot.to(X.dtype)[:, None]
        m = (P.mass.to(X.dtype) if getattr(P, "mass", None) is not None else torch.ones(X.shape[0], device=X.device, dtype=X.dtype)).clamp_min(1e-30)
        acc = F / m[:, None]
        mc = (CP.mass.to(X.dtype) if getattr(CP, "mass", None) is not None else torch.ones(Xc.shape[0], device=X.device, dtype=X.dtype)).clamp_min(1e-30)
        a_cell = torch.zeros_like(Xc).index_add(0, anchor, -F) / mc[:, None]          # the reaction: minus the force on the rod's foot point, an acceleration on the cell point it stands on
        return {self.at: acc * (P.occ > 0.5).to(X.dtype)[:, None], self.cpts: a_cell * (CP.occ > 0.5).to(X.dtype)[:, None]}


@register_operator("rod_reaction", family="interaction", set="particle", kind="lateral", title="the rods' reaction on their cell points",
                   equation=r"""$$\mathbf a_j=\frac{\bar{\mathbf f}_j}{m_j},\qquad \bar{\mathbf f}_j=\frac{1}{N}\sum_{n=1}^{N}\mathbf f_j^{(n)}$$""")
class RodReaction(Lateral):
    r"""the force the rods of the last rod block put on each cell point, averaged over that block's substeps, as an acceleration on the cell point; the sum is cleared for the next block.
    pt_cell -> pt_cell: emits `mpm_acceleration` from the running sum of `rod_foot`.

        \mathbf a_j=\frac{\bar{\mathbf f}_j}{m_j},\qquad \bar{\mathbf f}_j=\frac{1}{N}\sum_{n=1}^{N}\mathbf f_j^{(n)}
        a_j            the acceleration of cell point j
        f_j^{(n)}      the reaction on j at substep n of the rod block
        N              the number of substeps of the rod block
        m_j            the mass of cell point j

    Reference: Plexus exp37 (this work, 2026-11)."""
    EXPLAIN = r"Hands the cells the force their cadherin rods carried: the average over the rod block, as an acceleration on the cell points they stand on."
    PARAM_UNITS = {}
    SYMBOLS = {'a_j': 'the acceleration of cell point j', 'f_j^{(n)}': 'the reaction on j at substep n of the rod block', 'N': 'the number of substeps of the rod block', 'm_j': 'the mass of cell point j'}
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    PARAM_ROLES = {}
    MECHANISM_TAGS = ["cadherin", "two_scale"]
    REFERENCE = "Plexus exp37 (this work, 2026-11)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")

    def forward(self, H, mask=None):
        P = H.level(self.at)
        X = P.get("pos")
        rf = getattr(H, "_rod_force", None)
        if rf is None or rf.shape[0] != X.shape[0] or int(getattr(H, "_rod_n", 0)) == 0:
            return {self.at: torch.zeros_like(X)}
        f = rf / max(int(H._rod_n), 1)
        m = (P.mass.to(X.dtype) if getattr(P, "mass", None) is not None else torch.ones(X.shape[0], device=X.device, dtype=X.dtype)).clamp_min(1e-30)
        return {self.at: f / m[:, None] * (P.occ > 0.5).to(X.dtype)[:, None]}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SURFACE SPACER (Cedric, 2026-11-04: "an operator that brings the face points within 20 nm, like a van der Waals force"): holds the facing layers of two cells at a set gap so the cadherin rods meet tip to tip.
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@register_operator("surface_spacer", family="interaction", set="particle", kind="lateral", title="hold the facing layers of two cells at a set gap",
                   equation=r"""$$\mathbf a_i=\frac{k}{m_i\,n_i}\sum_{j}\big(g_{ij}-g_0\big)\,\hat{\mathbf n}\,[\,g_{ij}<g_{\max},\ \lvert\mathbf r^{\perp}_{ij}\rvert<\ell\,]$$""")
class SurfaceSpacer(Lateral):
    r"""a normal-only spring between the points of the two facing layers of two cells: each point is pulled to rest gap `gap` from the points of the other cell's layer that lie within lateral distance `lateral`; it neither breaks nor binds, and the cells slide freely.
    pt -> pt: emits `mpm_acceleration`.

        a_i = k / (m_i n_i) * sum_j (g_ij - g_0) n [g_ij < g_max, |r_ij perp| < l]
        a_i      the acceleration of layer point i
        k        `k`, the spring per point and per length, a force per length
        m_i, n_i the point's mass and its number of partners
        g_ij     the gap along the cell-to-cell normal n, from point i to partner j
        g_0      `gap`, the rest gap between the two layers' point centres
        g_max    `reach`, the largest gap that interacts
        l        `lateral`, the largest sideways distance of a partner

    Reference: Plexus exp37 (this work, 2026-11)."""
    EXPLAIN = r"Holds the two cells' facing point layers at a set gap along their normal, so cadherin rods standing on them can meet; sideways the cells are free."
    PARAM_UNITS = {'gap': 'length', 'reach': 'length', 'lateral': 'length', 'layer': 'length', 'k': 'energy_per_length2'}
    SYMBOLS = {'a_i': 'the acceleration of layer point i', 'k': '`k`, the spring', 'm_i, n_i': "the point's mass and its number of partners", 'g_ij': 'the gap along the normal from point i to partner j', 'g_0': '`gap`, the rest gap', 'g_max': '`reach`, the largest gap that interacts', 'l': '`lateral`, the largest sideways distance of a partner'}
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["cells", "gap"]
    PARAM_ROLES = {"cells": "the_cell_set", "gap": "rest_gap_between_layers", "k": "spring_per_length", "reach": "largest_interacting_gap", "lateral": "largest_sideways_distance", "layer": "thickness_of_the_facing_layer"}
    MECHANISM_TAGS = ["spacer", "two_scale"]
    REFERENCE = "Plexus exp37 (this work, 2026-11)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt"); self.cells = str(params["cells"]); self.g0 = float(params["gap"])
        self.k = float(params.get("k", 0.3)); self.reach = float(params.get("reach", 4.0 * self.g0)); self.lat = float(params.get("lateral", 0.03)); self.layer = float(params.get("layer", 0.02))

    def forward(self, H, mask=None):
        P = H.level(self.at); X = P.get("pos")
        if getattr(self, "_ia", None) is None:                                                       # first call: the candidate points of each cell's facing side and the normal (a sync here only)
            cid = H.lift_index(self.at, self.cells).long(); live = P.occ > 0.5
            c = torch.unique(cid[live])[:2]
            ia, ib = torch.nonzero(live & (cid == c[0])).reshape(-1), torch.nonzero(live & (cid == c[1])).reshape(-1)
            n = X[ib].mean(0) - X[ia].mean(0); self._n = (n / n.norm()).detach()
            sa, sb = X[ia] @ self._n, X[ib] @ self._n
            self._ia, self._ib = ia[sa > sa.max() - 3.0 * self.layer], ib[sb < sb.min() + 3.0 * self.layer]
        ia, ib, n = self._ia, self._ib, self._n
        XA, XB = X[ia], X[ib]; sA, sB = XA @ n, XB @ n
        wa = (sA > sA.max() - self.layer).to(X.dtype); wb = (sB < sB.min() + self.layer).to(X.dtype)           # the facing layers now, as 0/1 weights
        lat = torch.cdist(XA - sA[:, None] * n, XB - sB[:, None] * n)                                  # sideways distance of every pair
        g = sB[None, :] - sA[:, None]
        w = ((lat < self.lat) & (g < self.reach)).to(X.dtype) * wa[:, None] * wb[None, :]
        f = self.k * (g - self.g0) * w                                                                  # spring along the normal, positive pulls a toward b
        npa, npb = w.sum(1).clamp_min(1.0), w.sum(0).clamp_min(1.0)
        Fa = (f.sum(1) / npa)[:, None] * n; Fb = -(f.sum(0) / npb)[:, None] * n
        m = (P.mass.to(X.dtype) if getattr(P, "mass", None) is not None else torch.ones(X.shape[0], device=X.device, dtype=X.dtype)).clamp_min(1e-30)
        acc = torch.zeros_like(X)
        acc = acc.index_add(0, ia, Fa / m[ia, None]).index_add(0, ib, Fb / m[ib, None])
        return {self.at: acc}
