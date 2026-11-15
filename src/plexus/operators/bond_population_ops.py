"""The junction one level down (exp29, 2026-11-02): a POPULATION OF BONDS in place of the junction's bound density.

`bond_binding` evolves ONE number per junction, the bound density rho, by an ODE, and `bond_springs` weights every tether of the junction by it. Here the junction keeps its bonds one by one: a bond joins a
point of one cell to a point of the other (a nano-cluster of E-cadherin, finer than a point), is bound or not, binds with the rate k_on while the cells touch and unbinds with the rate k_off exp(F / sigma_beta), F the
tension of THAT bond (Bell 1978). The junction's rho is what is left of the old ODE: the bound share of its contact sites, times rho*.

    contact_finder -> bond_binding[population] (once per frame) -> bond_springs[population] (every substep)

A point holds at most `max_per_point` bonds (default 1: one nano-cluster per point of contact), so the bound density cannot exceed its sites and the expected force equals the mean-field one when the bound
share equals rho / rho*. The bonds are written, each frame, into the set named by `bond_set` (state `ia`, `ib` the two points, `row` the junction, `pos` the middle of the bond) so the watcher draws them.
"""
import torch

from plexus.models.registry import register_operator
from plexus.operators.adhesion_ops import CellAdhesion, JunctionKinetics


def _bond_state(H):
    return getattr(H, "_bond", None)


@register_operator("bond_binding", implementation="population", family="mechanics", set="junction", kind="lateral",
                   title="bind and unbind adhesion bond by bond",
                   equation=r"""$$P(\text{bind})=1-e^{-k_{on}[A>A_{min}]\Delta t},\qquad P(\text{unbind})=1-e^{-k_{off}\,e^{F_b/\sigma_\beta}\Delta t},\qquad \rho_J=\rho^*_J\,\frac{n^{bound}_J}{n^{sites}_J}$$""")
class BondPopulation(JunctionKinetics):
    r"""the junction's bonds, one by one: each frame an unbound pair of facing points of two touching cells binds with probability 1 - exp(-k_on dt), a bound bond unbinds with probability 1 - exp(-k_off exp(F / sigma_beta) dt), F its own tether tension, and breaks when stretched past `break_factor` times `bond_range`.
    junction -> junction, point: reads the pairs `contact_finder` found; writes the bound bonds (for `bond_springs[population]` and the `bond_set` record) and each junction's `rho`.

        P(\text{bind})=1-e^{-k_{on}[A>A_{min}]\Delta t},\qquad P(\text{unbind})=1-e^{-k_{off}\,e^{F_b/\sigma_\beta}\Delta t},\qquad \rho_J=\rho^*_J\,\frac{n^{bound}_J}{n^{sites}_J}
        k_{on}, k_{off} `binding_rate`, `unbinding_rate`, per time
        A, A_{min}     the contact area and `min_area`
        F_b            the tension of this bond, its stiffness x rho* x its stretch beyond the rest length
        \sigma_\beta   `bond_strength`, the Bell force scale
        \rho^*_J       the junction's maximum density, as bond_binding
        n^{bound}_J, n^{sites}_J the junction's bound bonds and its contact sites (its area over spacing squared)

    Reference: Bell, G. I. (1978). Models for the specific adhesion of cells to cells. Science 200:618-627."""
    EXPLAIN = (r"The same binding and unbinding as bond_binding, kept bond by bond instead of as one density per junction: every bond is bound or free, binds with probability $1-e^{-k_{on}\Delta t}$ while its cells touch, "
               r"and a bound one unbinds with probability $1-e^{-k_{off}e^{F_b/\sigma_\beta}\Delta t}$, where $F_b$ is the tension of that bond alone. A junction's $\rho$ is its bound bonds over its contact sites, times $\rho^*$.")
    PARAM_UNITS = {'break_factor': 'dimensionless', 'max_per_point': 'count', 'spacing': 'length', 'r_open': 'length'}
    SYMBOLS = {'k_{on}, k_{off}': '`binding_rate`, `unbinding_rate`, per time', 'A, A_{min}': 'the contact area and `min_area`', 'F_b': 'the tension of this bond', '\\sigma_\\beta': '`bond_strength`, the Bell force scale',
               '\\rho^*_J': "the junction's maximum density", 'n^{bound}_J, n^{sites}_J': "the junction's bound bonds and its contact sites"}
    MAY_MUTATE_INTEGRATED_STATE = True
    PARAM_ROLES = dict(getattr(JunctionKinetics, "PARAM_ROLES", {}), points="the_point_set_the_bonds_join", bond_set="set_that_records_the_bound_bonds", break_factor="stretch_over_bond_range_that_breaks_a_bond",
                       max_per_point="bonds_a_point_can_hold", spacing="point_spacing_world", gap_table="json_of_the_molecular_rig_bound_fraction_against_the_cleft", r_open="pair_distance_at_the_best_cleft")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.points = str(params.get("points", "pt"))
        self.bond_set = params.get("bond_set")
        self.h = float(params.get("spacing", 0.2))
        self.break_factor = float(params.get("break_factor", 1.5))
        self.max_per_point = int(params.get("max_per_point", 1))
        self.bond_reach = float(params.get("bond_range", 0.38))
        self.seed = int(params.get("seed", 0))
        # `gap_table` (exp37, 2026-11-02, "the cadherin junction"): a json written by the molecular rig, {gaps: nm, bound: the fraction of EC1 tips in a trans-dimer at that gap}. A pair binds with its probability times
        # bound(g) / max(bound) where g = cleft_nm * r / r_open maps the pair's distance r to a cleft width: r_open (world length) is the distance at which the cleft is the table's best, `cleft_nm`. The shape comes from the rig,
        # the scale (rates, stiffness) stays the spec's.
        self.gap_w = None
        gt = params.get("gap_table")
        if gt:
            import json
            import os
            from plexus.paths import get_repo_root
            tab = json.load(open(gt if os.path.isabs(gt) else os.path.join(get_repo_root(), gt)))     # a relative path is read from the repository root
            gaps = [g for g, b in zip(tab["gaps"], tab["bound"]) if b is not None]; bnd = [b for b in tab["bound"] if b is not None]
            self.gap_w = (torch.as_tensor(gaps, dtype=torch.float64), torch.as_tensor(bnd, dtype=torch.float64) / max(max(bnd), 1e-12))
            self.cleft_nm = float(gaps[max(range(len(bnd)), key=lambda i: bnd[i])])
            self.r_open = float(params.get("r_open", 0.19))
        self._g = None

    def _gap_weight(self, r):
        """the table's relative bound fraction at the pair distances r (world length): 1 at the best cleft, less at the others, 0 where no tip pair met."""
        g = (self.cleft_nm * r / self.r_open).double()
        xs, ys = self.gap_w[0].to(g.device), self.gap_w[1].to(g.device)
        i = torch.bucketize(g, xs).clamp(1, xs.numel() - 1)
        t = ((g - xs[i - 1]) / (xs[i] - xs[i - 1]).clamp_min(1e-9)).clamp(0.0, 1.0)
        return (ys[i - 1] * (1 - t) + ys[i] * t).to(r.dtype)

    def _rand(self, n, dev):
        if self._g is None:
            self._g = torch.Generator(device=dev).manual_seed(self.seed)
        return torch.rand(n, device=dev, generator=self._g)

    def forward(self, H, mask=None):
        J, C, P = H.level(self.at), H.level(self.cells), H.level(self.points)
        dev = J.state.device
        X = P.get("pos").detach()
        Np = int(X.shape[0])
        dt = float(getattr(H, "dt", 1.0))
        live_p = P.occ > 0.5
        nj = J.n
        # per-junction parameters, as bond_binding's exact step reads them
        ca = J.get("ca")[:, 0].round().long().clamp(0, C.state.shape[0] - 1)
        cb = J.get("cb")[:, 0].round().long().clamp(0, C.state.shape[0] - 1)
        fg = J.get("fg")[:, 0].round().long().clamp(0, 5)
        adh = C.get("adh")
        rstar = torch.minimum(adh[ca, fg], adh[cb, fg])
        if self.type_matrix is not None:
            tm = torch.as_tensor(self.type_matrix, device=dev, dtype=rstar.dtype)
            nt = C.node_type.long().clamp(0, tm.shape[0] - 1)
            rstar = rstar * tm[nt[ca], nt[cb]]

        def _pj(scalar, block):
            if not block:
                return scalar * torch.ones(nj, device=dev, dtype=rstar.dtype)
            v = C.state[:, C.state_schema[block][0]].to(rstar.dtype)
            return 0.5 * (v[ca] + v[cb])
        k_on, k_off, sig_b = _pj(self.k_on, self.k_on_block), _pj(self.k_off, self.k_off_block), _pj(self.sigma_beta, self.sigma_beta_block)
        touch = J.get("area")[:, 0] > self.min_area
        live_j = J.occ > 0.5
        # the bound bonds of the last frame, thinned and kept
        B = _bond_state(H)
        if B is not None and B["i"].numel():
            bi, bj, br, bw = B["i"], B["j"], B["row"], B["w"]
            r = (X[bj] - X[bi]).norm(dim=1)
            n_sub = max(int(getattr(H, "_bond_n", 0)), 1)
            ten = B["ten"] / n_sub                                               # the mean tension of each bond over the substeps of the frame
            ok = live_p[bi] & live_p[bj] & live_j[br] & (r < self.break_factor * self.bond_reach)
            p_off = 1.0 - torch.exp(-k_off[br] * torch.exp((ten / sig_b[br]).clamp(max=30.0)) * dt)
            keep = ok & (self._rand(bi.numel(), dev) >= p_off)
            bi, bj, br, bw = bi[keep], bj[keep], br[keep], bw[keep]
        else:
            bi = bj = br = torch.zeros(0, dtype=torch.long, device=dev); bw = torch.zeros(0, device=dev, dtype=X.dtype)
        # the pairs facing each other now: each can bind if both its points are free
        pairs = getattr(H, "_adh_pairs", None)
        if pairs is not None and pairs[0].numel():
            ci, cj, crow = pairs[0].long(), pairs[1].long(), pairs[2].long()
            valid = crow >= 0
            ci, cj, crow = ci[valid], cj[valid], crow[valid]
            cnt = torch.zeros(Np, device=dev, dtype=torch.long)
            cnt.index_add_(0, bi, torch.ones_like(bi)); cnt.index_add_(0, bj, torch.ones_like(bj))
            free = (cnt[ci] < self.max_per_point) & (cnt[cj] < self.max_per_point) & touch[crow] & live_j[crow]
            p_on = 1.0 - torch.exp(-k_on[crow] * dt)
            if self.gap_w is not None:
                p_on = p_on * self._gap_weight((X[cj] - X[ci]).norm(dim=1))
            go = free & (self._rand(ci.numel(), dev) < p_on)
            ni, nj_, nr = ci[go], cj[go], crow[go]
            if ni.numel():                                                       # a point binds once per frame: a pair is kept only if both its points appear first in this list
                e = torch.cat([ni, nj_])
                u, inv = torch.unique(e, return_inverse=True)
                first = torch.full((u.numel(),), e.numel(), device=dev, dtype=torch.long).scatter_reduce_(0, inv, torch.arange(e.numel(), device=dev), "amin")
                m = torch.zeros(e.numel(), dtype=torch.bool, device=dev); m[first] = True
                keep_new = m[: ni.numel()] & m[ni.numel():]
                ni, nj_, nr = ni[keep_new], nj_[keep_new], nr[keep_new]
                bi, bj, br = torch.cat([bi, ni]), torch.cat([bj, nj_]), torch.cat([br, nr])
                bw = torch.cat([bw, rstar[nr].to(bw.dtype)])
        H._bond = {"i": bi, "j": bj, "row": br, "w": bw if bw.numel() else torch.zeros(0, device=dev, dtype=X.dtype), "ten": torch.zeros(bi.numel(), device=dev, dtype=X.dtype)}
        H._bond_n = 0
        # the junction's rho: its bound bonds over its contact sites, times rho*
        nb = torch.zeros(nj, device=dev, dtype=rstar.dtype).index_add_(0, br, torch.ones(br.numel(), device=dev, dtype=rstar.dtype)) if br.numel() else torch.zeros(nj, device=dev, dtype=rstar.dtype)
        sites = (J.get("area")[:, 0] / self.h ** 2).clamp(min=1.0)
        a_, b_ = J.state_schema["rho"]
        st = J.state.clone()
        st[:, a_] = torch.where(live_j, rstar * (nb / sites).clamp(max=1.0), st[:, a_])
        J.state = st
        # the record: the bound bonds as a set the watcher can draw
        if self.bond_set:
            S = H.level(self.bond_set)
            sch = S.state_schema
            n = int(min(bi.numel(), S.n))
            ss = torch.zeros_like(S.state)
            if n:
                ss[:n, sch["ia"][0]] = bi[:n].to(ss.dtype); ss[:n, sch["ib"][0]] = bj[:n].to(ss.dtype); ss[:n, sch["row"][0]] = br[:n].to(ss.dtype)
                ss[:n, sch["pos"][0]:sch["pos"][1]] = (0.5 * (X[bi[:n]] + X[bj[:n]])).to(ss.dtype)
            S.state = ss
            oc = torch.zeros_like(S.occ); oc[:n] = 1.0; S.occ = oc
        return {}


@register_operator("bond_springs", implementation="population", family="interaction", set="particle", kind="exchange",
                   title="pull adhered cells together through their bound bonds",
                   equation=r"""$$\mathbf F_{ij}=\kappa\,\rho^*\,(r-r_{eq})^+\,\hat{\mathbf r}_{ij}\,[r<r_c]+\zeta\,\rho^*\,(1-r/r_c)^2\,(\mathbf v_j-\mathbf v_i)\,[r<r_c]\qquad\text{for each BOUND bond }(i,j)$$""")
class BondSpringsPopulation(CellAdhesion):
    r"""bond_springs on the bound bonds of `bond_binding[population]` instead of every facing pair weighted by the junction's density: each bound bond is a tether of stiffness kappa rho* (the junction's maximum), so the sum over a junction equals the mean-field force when its bound share is rho / rho*.
    pt -> pt (reads the bonds): emits the tether accelerations into the MPM body force and stores each bond's tension for its unbinding.

        \mathbf F_{ij}=\kappa\,\rho^*\,(r-r_{eq})^+\,\hat{\mathbf r}_{ij}\,[r<r_c]+\zeta\,\rho^*\,(1-r/r_c)^2\,(\mathbf v_j-\mathbf v_i)\,[r<r_c]
        F_{ij}         the force on point i from point j (and -F_ij on j), for each bound bond
        \kappa         `bond_stiffness`, force per length
        \rho^*         the junction's maximum bound density
        r, r_{eq}, r_c the pair distance, `bond_rest_length`, `bond_range`
        \zeta          `bond_damping`
        v_j - v_i      the relative velocity

    Reference: as bond_springs; Bell, G. I. (1978). Science 200:618-627."""
    EXPLAIN = (r"The force of adhesion, every MPM substep, on the bound bonds only: $\mathbf F_{ij}=\kappa\rho^*(r-r_{eq})^+\hat{\mathbf r}_{ij}[r<r_c]$ plus the drag on the relative velocity, for each bound bond $(i,j)$ of bond_binding[population]; "
               r"the tension of every bond is kept so that bond unbinds faster the harder it is pulled.")
    PARAM_UNITS = {'bond_range': 'length', 'bond_stiffness': 'acceleration', 'bond_rest_length': 'length', 'spacing': 'length', 'bond_damping': 'rate'}
    SYMBOLS = {'F_{ij}': 'the force on point i from point j (and -F_ij on j), for each bound bond', '\\kappa': '`bond_stiffness`, force per length', '\\rho^*': "the junction's maximum bound density",
               'r, r_{eq}, r_c': 'the pair distance, `bond_rest_length`, `bond_range`', '\\zeta': '`bond_damping`', 'v_j - v_i': 'the relative velocity'}

    def forward(self, H, mask=None):
        P = H.level(self.at)
        X = P.get("pos")
        acc = torch.zeros_like(X)
        B = _bond_state(H)
        if B is None or not B["i"].numel():
            return {self.at: acc}
        i, j, row, w = B["i"], B["j"], B["row"], B["w"].to(X.dtype)
        d = X[j] - X[i]
        r = d.norm(dim=1).clamp_min(1e-9)
        par_ = P.parent.long()

        def _pp(scalar, block):
            if not block:
                return scalar
            Cl_ = H.level(self.kappa_cells)
            v = Cl_.state[:, Cl_.state_schema[block][0]].to(X.dtype)
            pc = par_.clamp(0, Cl_.state.shape[0] - 1)
            return 0.5 * (v[pc[i]] + v[pc[j]])
        reach, r_eq, zeta = _pp(self.bond_reach, self.bond_reach_block), _pp(self.r_eq, self.r_eq_block), _pp(self.zeta, self.zeta_block)
        kap = self.kappa
        if self.kappa_block:
            Cl = H.level(self.kappa_cells)
            kc = Cl.state[:, Cl.state_schema[self.kappa_block][0]].to(X.dtype)
            pc = par_.clamp(0, Cl.state.shape[0] - 1)
            kap = 0.5 * (kc[pc[i]] + kc[pc[j]])
        ten = kap * w * (r - r_eq).clamp(min=0.0) * (r < reach).to(X.dtype)
        F = (ten / r)[:, None] * d
        if (self.zeta_block or self.zeta > 0.0) and "vel" in P.state_schema:
            V = P.get("vel")
            F = F + (zeta * w * (1.0 - r / reach).clamp(min=0.0) ** 2)[:, None] * (V[j] - V[i])
        m = P.mass.to(X.dtype) if getattr(P, "mass", None) is not None else torch.ones(X.shape[0], device=X.device, dtype=X.dtype)
        acc.index_add_(0, i, F / m[i, None].clamp_min(1e-30))
        acc.index_add_(0, j, -F / m[j, None].clamp_min(1e-30))
        B["ten"] = B["ten"] + ten.detach()                                          # the bond's tension, summed over the frame's substeps (bond_binding divides by their number)
        H._bond_n = int(getattr(H, "_bond_n", 0)) + 1
        J = H.level(self.junctions)
        accL = getattr(H, "_adh_load_acc", None)
        if accL is not None and accL.numel() == J.n:
            accL.index_add_(0, row, ten)
            H._adh_load_n = int(getattr(H, "_adh_load_n", 0)) + 1
        if mask is not None:
            acc = acc * mask[:, None].to(X.dtype)
        return {self.at: acc}
