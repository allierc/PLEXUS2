"""exp09 (2026-11-19): `bond_binding[delta_g]` -- the binding and unbinding of a junction set by a BINDING FREE ENERGY per pair of types.

Cedric: "implement Delta G in the adhesion operator". bond_binding and its valence variant take a rate of binding, a rate of unbinding and a type matrix that
scales how much may bind; none of the three is a quantity anyone measures for two molecules. What is measured is the free energy of binding Delta G (or the
dissociation constant K_d = c0 exp(Delta G0 / kT)), and it fixes one thing: the ratio of the two rates of a pair that is in contact,

    k_off / k_on = exp(Delta G / kT)                    Delta G < 0: binding is favoured; each -1.42 kcal/mol at 37 C is a factor 10

so a junction left alone settles at the Boltzmann share  rho_eq = 1 / (1 + exp(Delta G / kT)).

UNITS. Delta G, the temperature and the binding volume are PHYSICAL quantities (kcal/mol or kJ/mol, kelvin, nm^3), not simulation units: Plexus's unit
contract (plexus/units.py) has three scales -- length, time, force -- and neither a temperature nor a molar energy, and exp09's specs declare no force
scale, so no energy of those runs can be quoted in joules. This operator therefore uses Delta G only as the ratio Delta G / (R T), a pure number that needs
the temperature and nothing of the simulation's scales; `delta_g_unit: kT` takes that number directly. They are not in PARAM_UNITS for that reason.

`delta_g_reference`: "contact" (default) -- Delta G is that of a pair already in contact, as the equation above reads it. "standard" -- Delta G0 as
tabulated, referred to the standard concentration c0 = 1 mol/L; a pair held in contact within the volume v_b (`binding_volume_nm3`) is at the
concentration 1 / (N_A v_b), so  Delta G = Delta G0 + kT ln(c0 N_A v_b),  c0 N_A = 0.6022 per nm^3  (a shell of 4.5 nm: +3.35 kcal/mol at 37 C).

`stochastic: true`: a junction is ONE bond that is bound or not (rho 1 or 0), not a share -- each frame its next state is drawn from the two-state process
with the rates a = k_on (when admitted and touching) and b = k_off e^{sigma / sigma_beta}: an unbound one is bound a frame later with probability
a / (a + b) x (1 - e^{-(a + b) dt}), a bound one unbound with b / (a + b) x (1 - e^{-(a + b) dt}) (flips within the frame included, so the mean over many
junctions is the differential equation's exact step and the long-run bound fraction is rho_eq). Default false: the mean share.
"""
import math

import numpy as np
import torch

from plexus.models.registry import register_operator
from plexus.operators.adhesion_ops import JunctionKinetics, JunctionKineticsValence

R_KCAL = 1.987204259e-3                # the gas constant, kcal / (mol K)
R_KJ = 8.314462618e-3                  # the gas constant, kJ / (mol K)
C0_PER_NM3 = 0.602214076               # molecules per nm^3 at the standard concentration 1 mol/L (N_A x 1 mol/L)


def delta_g_over_kt(value, unit, temperature_K):
    """A free energy in `unit` (kcal/mol, kJ/mol or kT) as a number of kT at `temperature_K`."""
    u = str(unit).strip().lower().replace(" ", "")
    if u in ("kt", "kbt", "rt"):
        return float(value)
    if u == "kcal/mol":
        return float(value) / (R_KCAL * float(temperature_K))
    if u == "kj/mol":
        return float(value) / (R_KJ * float(temperature_K))
    raise ValueError(f"bond_binding[delta_g]: delta_g_unit is kcal/mol, kJ/mol or kT, not {unit!r}")


@register_operator("bond_binding", implementation="delta_g", family="mechanics", set="junction", kind="lateral",
                   title="bind adhesion by each pair's binding free energy",
                   equation=r"""$$\dot\rho_J=k_{on}\,(\rho^*-\rho_J)\,[A_J>A_{min}]\,g_J-k_{on}\,e^{\Delta G_{\tau_a\tau_b}/kT}\,e^{\sigma/\sigma_\beta}\,\rho_J,\qquad \rho_{eq}=\frac{1}{1+e^{\Delta G/kT}}$$""")
class JunctionKineticsDeltaG(JunctionKineticsValence):
    r"""bond_binding[valence] with the unbinding rate of each junction set by the BINDING FREE ENERGY of its two cells' types: k_off = k_on exp(Delta G / kT), so a pair in contact settles at the Boltzmann share 1 / (1 + exp(Delta G / kT)). A pair of types with no entry never binds.
    junction -> junction: as bond_binding[valence] (`valence_block` optional here: without it no cell is limited); `delta_g` replaces `unbinding_rate` and `type_matrix`, which are refused.

        \dot\rho_J=k_{on}\,(\rho^*-\rho_J)\,[A_J>A_{min}]\,g_J-k_{on}\,e^{\Delta G_{\tau_a\tau_b}/kT}\,e^{\sigma/\sigma_\beta}\,\rho_J,\qquad \rho_{eq}=\frac{1}{1+e^{\Delta G/kT}}
        \rho_J         the bound share of junction J (with `stochastic`: 1 bound, 0 not)
        \Delta G_{\tau_a\tau_b}  `delta_g`, the binding free energy of the types of the junction's two cells, a [types, types] matrix in `delta_g_unit` (kcal/mol, kJ/mol or kT); null = that pair never binds
        kT             the thermal energy at `temperature_K` (R T per mole: 0.616 kcal/mol at 310.15 K)
        k_{on}         `binding_rate`, the binding rate of a pair in contact, per time; the unbinding rate is k_on e^{Delta G / kT}
        \rho^*         the ceiling: the lower of the two cells' `adh` (1 when both are 1)
        \rho_{eq}      the share an unloaded junction in contact settles at
        g_J, A_J, A_{min}, \sigma, \sigma_\beta  as bond_binding[valence]: the valence gate, the contact area and its threshold, the junction's load and Bell's scale `bond_strength`

    `delta_g_reference: standard` shifts a tabulated Delta G0 (referred to 1 mol/L) by kT ln(0.6022 x `binding_volume_nm3`) to the pair in contact; `stochastic: true` makes each junction one bond, bound or not, its next state drawn every frame from the two-state process of the two rates.

    Reference: Bell, G. I. (1978). Models for the specific adhesion of cells to cells. Science 200:618-627 (rates, their ratio and the load); Plexus exp09 (this work, 2026-11-19)."""
    EXPLAIN = (
        r"bond_binding with what is measured for two molecules, their binding free energy $\Delta G$, in place of an unbinding rate. For a pair in contact the two rates "
        r"obey $k_{off}/k_{on}=e^{\Delta G/kT}$ ($kT$ the thermal energy, 0.616 kcal/mol at 37 °C), so the junction's bound share follows "
        r"$\dot\rho=k_{on}(\rho^*-\rho)[A>A_{min}]\,g-k_{on}e^{\Delta G/kT}e^{\sigma/\sigma_\beta}\rho$ and, unloaded, settles at $\rho_{eq}=1/(1+e^{\Delta G/kT})$: "
        r"0.5 at $\Delta G=0$, 0.88 at $-2\,kT$, 0.99995 at $-10\,kT$. Each pair of types has its own $\Delta G$ (a matrix); a pair with none never binds. "
        r"A load $\sigma$ speeds the unbinding as in bond_binding (Bell), and the valence gate $g$ still limits how many bonds a cell holds."
    )
    PARAM_UNITS = {k: v for k, v in dict(JunctionKinetics.PARAM_UNITS, bound_at='fraction').items() if k != "unbinding_rate"}
    SYMBOLS = {'\\rho_J': 'the bound share of junction J', '\\Delta G_{\\tau_a\\tau_b}': "`delta_g`, the binding free energy of the two cells' types", 'kT': 'the thermal energy at `temperature_K`',
               'k_{on}': '`binding_rate`, per time; the unbinding rate is k_on e^{Delta G / kT}', '\\rho^*': "the ceiling: the lower of the two cells' `adh`",
               '\\rho_{eq}': 'the share an unloaded junction in contact settles at', 'g_J, A_J, A_{min}, \\sigma, \\sigma_\\beta': 'as bond_binding[valence]'}
    PARAM_ROLES = {k: v for k, v in dict(JunctionKinetics.PARAM_ROLES, valence_block="cell_block_with_the_number_of_bonds_a_cell_may_hold", bound_at="bound_density_that_counts_as_a_bond",
                                         delta_g="binding_free_energy_over_cell_types", delta_g_unit="kcal_per_mol_or_kJ_per_mol_or_kT", temperature_K="temperature_in_kelvin",
                                         delta_g_reference="contact_or_standard_state", binding_volume_nm3="volume_a_pair_in_contact_is_held_in",
                                         stochastic="one_bond_bound_or_not_instead_of_a_share").items() if k not in ("type_matrix",)}
    REFERENCE = "Bell, G. I. (1978). Science 200:618-627; Plexus exp09 (this work, 2026-11-19)."

    OWN = ("delta_g", "delta_g_unit", "temperature_K", "delta_g_reference", "binding_volume_nm3", "stochastic", "seed", "valence_block", "bound_at")

    def __init__(self, params, device="cpu"):
        if params.get("delta_g") is None:
            raise ValueError("bond_binding[delta_g]: `delta_g`, the [types, types] matrix of binding free energies (null = the pair never binds), is required")
        for k in ("unbinding_rate", "unbinding_rate_block", "type_matrix"):
            if params.get(k) is not None:
                raise ValueError(f"bond_binding[delta_g]: `{k}` is not taken -- the unbinding rate is binding_rate x exp(delta_g / kT) and `delta_g` says which pairs bind")
        JunctionKinetics.__init__(self, {k: v for k, v in params.items() if k not in self.OWN}, device)
        self.valence_block = str(params["valence_block"]) if params.get("valence_block") else None
        self.bound_at = float(params.get("bound_at", 0.5))
        self.temperature_K = float(params.get("temperature_K", 310.15))
        self.unit = str(params.get("delta_g_unit", "kcal/mol"))
        ref = str(params.get("delta_g_reference", "contact"))
        if ref not in ("contact", "standard"):
            raise ValueError("bond_binding[delta_g]: delta_g_reference is contact or standard")
        shift = 0.0
        if ref == "standard":
            if params.get("binding_volume_nm3") is None:
                raise ValueError("bond_binding[delta_g]: delta_g_reference standard needs `binding_volume_nm3`, the volume a pair in contact is held in")
            shift = math.log(C0_PER_NM3 * float(params["binding_volume_nm3"]))
        rows = [[(float("nan") if v is None else delta_g_over_kt(v, self.unit, self.temperature_K) + shift) for v in row] for row in params["delta_g"]]
        g = np.asarray(rows, np.float64)
        if g.ndim != 2 or g.shape[0] != g.shape[1]:
            raise ValueError(f"bond_binding[delta_g]: delta_g is a square [types, types] matrix, got {g.shape}")
        both = np.isfinite(g) & np.isfinite(g.T)
        if (np.isfinite(g) != np.isfinite(g.T)).any() or (np.abs(g - g.T)[both] > 1e-9).any():
            raise ValueError("bond_binding[delta_g]: delta_g must be symmetric (the pair a-b is the pair b-a)")
        self.g_kT = g                                                      # [types, types], Delta G / kT of a pair in contact; NaN = never binds
        self.stochastic = bool(params.get("stochastic", False))
        self.seed = int(params.get("seed", 0))
        self._gen = None

    def _may_bind(self, C, ca, cb, g_ab):
        """[junctions] bool: may this junction bind at all. Here: its two cells' types have a free energy. (bond_binding[delta_g_molecule] adds its own condition.)"""
        return torch.isfinite(g_ab)

    def _rho_next(self, H):
        """The one-frame step of every junction's rho: bond_binding[valence]'s, the unbinding rate of each junction k_on exp(Delta G / kT) of its two cells' types;
        with `stochastic` the junction's next state drawn, 1 or 0."""
        J, C = H.level(self.at), H.level(self.cells)
        nC = int(C.state.shape[0])
        rho = J.get("rho")[:, 0]
        ca = J.get("ca")[:, 0].round().long().clamp(0, nC - 1)
        cb = J.get("cb")[:, 0].round().long().clamp(0, nC - 1)
        fg = J.get("fg")[:, 0].round().long().clamp(0, 5)
        adh = C.get("adh")
        G = torch.as_tensor(self.g_kT, device=rho.device, dtype=rho.dtype)
        nt = C.node_type.long().clamp(0, G.shape[0] - 1)
        g_ab = G[nt[ca], nt[cb]]
        binds = self._may_bind(C, ca, cb, g_ab)
        rmax = torch.minimum(adh[ca, fg], adh[cb, fg]) * binds.to(rho.dtype)
        live = J.occ > 0.5
        area = J.get("area")[:, 0]
        touch = area > self.min_area
        bound = live & (rho > self.bound_at)
        if self.valence_block:                                             # the valence gate of bond_binding[valence]: held bonds kept, the largest contacts admitted first
            val = C.get(self.valence_block)[:, 0].detach().round().long().cpu().numpy()
            ca_n, cb_n = ca.detach().cpu().numpy(), cb.detach().cpu().numpy()
            held = bound.detach().cpu().numpy()
            n = np.bincount(ca_n[held], minlength=nC) + np.bincount(cb_n[held], minlength=nC)
            cand = np.flatnonzero((live & ~(rho > self.bound_at) & touch & (rmax > 0)).detach().cpu().numpy())
            ar = area.detach().cpu().numpy()
            gate = held.copy()
            for j in cand[np.argsort(-ar[cand], kind="stable")]:
                a_, b_ = ca_n[j], cb_n[j]
                if a_ != b_ and n[a_] < val[a_] and n[b_] < val[b_]:
                    gate[j] = True
                    n[a_] += 1
                    n[b_] += 1
            g = torch.as_tensor(gate, device=rho.device).to(rho.dtype)
        else:
            g = torch.ones_like(rho)
        sig = J.get("load")[:, 0].clamp(min=0.0)
        if self.bell == "bond":
            sig = sig / rho.clamp(min=self.rho_floor)

        def _pj(scalar, block):
            if not block:
                return scalar
            v = C.state[:, C.state_schema[block][0]].to(rho.dtype)
            return 0.5 * (v[ca] + v[cb])
        k_on, sigma_beta = _pj(self.k_on, self.k_on_block), _pj(self.sigma_beta, self.sigma_beta_block)
        k_off = k_on * torch.exp(torch.where(binds, g_ab, torch.zeros_like(g_ab)).clamp(max=30.0))        # k_on e^{Delta G / kT}
        a = k_on * touch.to(rho.dtype) * g * binds.to(rho.dtype)
        b = k_off * torch.exp((sig / sigma_beta).clamp(max=30.0))
        dt = float(getattr(H, "dt", 1.0))
        if self.stochastic:                                                # one bond, bound or not: the exact probabilities of the two rates over the frame
            if self._gen is None or self._gen.device != rho.device:
                self._gen = torch.Generator(device=rho.device).manual_seed(self.seed)
            u = torch.rand(rho.shape, generator=self._gen, device=rho.device, dtype=rho.dtype)
            # the two-state process over one frame, flips within the frame included: from unbound, bound a frame later with probability a / (a + b) x (1 - e^{-(a + b) dt});
            # from bound, unbound with b / (a + b) x (1 - e^{-(a + b) dt}). (1 - e^{-a dt} and 1 - e^{-b dt}, one event at most per frame, settle at the wrong share when a
            # rate is not small against 1 / dt: 0.672 instead of 0.731 at Delta G = -1 kT, k_on = 1 per frame -- the unit test of 2026-11-19)
            tot = (a + b).clamp(min=1e-12)
            relax = 1.0 - torch.exp(-tot * dt)
            p_on = a / tot * relax
            p_off = b / tot * relax
            stays = (rho > self.bound_at) & (u >= p_off)
            joins = ~(rho > self.bound_at) & (u < p_on)
            rho_next = (stays | joins).to(rho.dtype) * rmax.clamp(max=1.0).gt(0).to(rho.dtype)
            return J, rho, rho_next, live
        rate = (a + b).clamp(min=1e-12)
        r_eq = a * rmax / rate
        rho_next = r_eq + (rho - r_eq) * torch.exp(-rate * dt)
        return J, rho, rho_next, live


@register_operator("bond_binding", implementation="delta_g_molecule", family="mechanics", set="junction", kind="lateral",
                   title="bind adhesion by each pair's binding free energy, never inside one molecule",
                   equation=r"""$$\dot\rho_J=k_{on}\,(\rho^*-\rho_J)\,[A_J>A_{min}]\,[m_a\neq m_b]\,g_J-k_{on}\,e^{\Delta G_{\tau_a\tau_b}/kT}\,e^{\sigma/\sigma_\beta}\,\rho_J$$""")
class JunctionKineticsDeltaGMolecule(JunctionKineticsDeltaG):
    r"""bond_binding[delta_g] for molecules made of several cells (an mRNA's subunits, a protein's domains): a junction whose two cells belong to ONE molecule never binds, whatever the free energy of their types. Without it a strand whose kind binds its own kind binds itself: two neighbouring sites of one strand are always in contact.
    junction -> junction: as bond_binding[delta_g], with the cell block `molecule_block`: per cell, the molecule it belongs to (for a cell of a chain, the chain's first cell; for a cell on its own, itself).

        \dot\rho_J=k_{on}\,(\rho^*-\rho_J)\,[A_J>A_{min}]\,[m_a\neq m_b]\,g_J-k_{on}\,e^{\Delta G_{\tau_a\tau_b}/kT}\,e^{\sigma/\sigma_\beta}\,\rho_J
        m_a, m_b       the molecules of the junction's two cells, read from `molecule_block`
        [m_a\neq m_b]  1 when the two cells belong to different molecules, 0 when to the same one: that junction never binds
        \rho_J, \Delta G_{\tau_a\tau_b}, kT, k_{on}, \rho^*, g_J, A_J, A_{min}, \sigma, \sigma_\beta  as bond_binding[delta_g]

    Reference: as bond_binding[delta_g]; Plexus exp10 (this work, 2026-11-19: mRNAs that bind their own kind, oskar with oskar)."""
    EXPLAIN = (
        r"bond_binding[delta_g] for molecules built from several cells, such as an mRNA drawn as a chain of subunits. Each cell carries the number of its molecule, $m$; "
        r"a junction between two cells of the same molecule, $m_a=m_b$, never binds, so a strand whose kind binds its own kind sticks to another strand and not to itself. "
        r"Every other junction follows $\dot\rho=k_{on}(\rho^*-\rho)[A>A_{min}]\,g-k_{on}e^{\Delta G/kT}e^{\sigma/\sigma_\beta}\rho$ as before."
    )
    PARAM_UNITS = dict(JunctionKineticsDeltaG.PARAM_UNITS)
    SYMBOLS = dict(JunctionKineticsDeltaG.SYMBOLS, **{'m_a, m_b': "the molecules of the junction's two cells, read from `molecule_block`",
                                                    '[m_a\\neq m_b]': '1 when the two cells belong to different molecules, 0 when to the same one'})
    PARAM_ROLES = dict(JunctionKineticsDeltaG.PARAM_ROLES, molecule_block="cell_block_with_the_molecule_a_cell_belongs_to")
    REFERENCE = "Bell, G. I. (1978). Science 200:618-627; Plexus exp10 (this work, 2026-11-19)."

    OWN = JunctionKineticsDeltaG.OWN + ("molecule_block",)

    def __init__(self, params, device="cpu"):
        if not params.get("molecule_block"):
            raise ValueError("bond_binding[delta_g_molecule]: `molecule_block`, the cell block holding each cell's molecule (a chain's first cell; a lone cell, itself), is required")
        super().__init__(params, device)
        self.molecule_block = str(params["molecule_block"])

    def _may_bind(self, C, ca, cb, g_ab):
        if self.molecule_block not in C.state_schema:
            raise ValueError(f"bond_binding[delta_g_molecule]: the cell set `{self.cells}` has no block `{self.molecule_block}`")
        mol = torch.round(C.state[:, C.state_schema[self.molecule_block][0]]).long()
        return torch.isfinite(g_ab) & (mol[ca] != mol[cb])
