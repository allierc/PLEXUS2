"""Membrane pores and channels: proteins as elastic bead networks in a bead membrane, the
electrolyte as a conductor, and the measurements that read a channel's state off them.

THE CHANNEL IS NOT AN OPERATOR THAT SAYS "OPEN". builder/instruction.md section 0b, and
experiments/exp04_membrane_channels.md: a mechanosensitive channel opens because the stretched
membrane pulls on its helices, a potassium channel conducts because ions repel each other through
its filter, a toxin's pore exists because its monomers stick together in a ring. So nothing here
knows what a channel is. There are beads (the alpha carbons of a deposited structure, lipid
patches, ions), the bonds that hold a protein's own fold, the non-bonded laws between beads, a
thermal bath, a frame that stretches the membrane the way a patch pipette does, and an
electrolyte that carries current wherever there is water. Opening, conduction and assembly are to
come out of those; the operators at the end only MEASURE them.

In the order they appear below:

    elastic_network        lateral   a protein's fold: springs between its own beads within a cutoff
                                     (Tirion 1996), at the deposited geometry
    pair_potential         exchange  non-bonded law between two sets, or within one: Lennard-Jones,
                                     its repulsive core (WCA), or screened Coulomb
    brownian               lateral   the thermal bath of an overdamped bead: sqrt(2 mu kT / dt) xi
    tether                 lateral   a harmonic hold to the seeded place, on chosen axes: the
                                     bilayer's confinement of a lipid to its plane, an anchor
    radial_drive           lateral   the patch pipette: a ring of frame beads whose radius follows
                                     an applied tension (or strain) protocol; the tension the
                                     membrane actually carries is READ from the frame's reaction
    electrolyte_conduction exchange  the bath as a conductor: div(sigma grad phi) = 0 on a grid,
                                     protein and lipid beads insulating, the membrane potential
                                     across the box; the channel current is the flux through the
                                     membrane plane -- the conductance is the pore's own geometry
    pore_probe             lateral   a measurement: the radius of the narrowest free lumen along
                                     the channel axis, from a protein set's beads

EVERY FORCE IS OVERDAMPED (`EMIT = "velocity"`): v = mobility x force. A protein bead in water, a
lipid in a bilayer, an ion -- all are at Reynolds numbers of 1e-9 and forget their momentum in
picoseconds, so inertia is never carried. The engine integrates a set ONCE per frame (substep
blocks are the MPM's), so a run here is many short frames. The time of such a run is the
relaxation clock of the bead model -- set by one bead's Stokes drag and the stiffest spring -- and
it is stated in each spec's units, not claimed as the channel's physiological clock.
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Exchange, Lateral
from plexus.models.registry import register_operator


def _slab_step(u, a, w):
    """1 below a - w/2, 0 above a + w/2, a cosine between; and its derivative (the slab edge of pair_potential's membrane band; was SlabBarrier._step)."""
    if w <= 0:
        return (u < a).to(u.dtype), torch.zeros_like(u)
    t = ((u - (a - 0.5 * w)) / w).clamp(0.0, 1.0)
    f = 0.5 * (1.0 + torch.cos(math.pi * t))
    df = torch.where((t > 0) & (t < 1), -0.5 * math.pi * torch.sin(math.pi * t) / w, torch.zeros_like(u))
    return f, df


def _block(lvl, name):
    """A state block's [n, w] tensor, or None when the set does not declare it."""
    return lvl.get(name) if name and name in lvl.state_schema._slices else None


def _write(lvl, name, value):
    """Write `value` ([n] or [n, w]) into block `name` in place (a readout, not a delta)."""
    a, b = lvl.state_schema[name]
    st = lvl.state.clone()
    v = value if value.dim() == 2 else value[:, None]
    st[:, a:b] = v[:, : b - a].to(st.dtype)
    lvl.state = st


def _frame(H):
    return int(getattr(H, "frame", 0))


def _cell_pairs(Xa, Xb, cut, max_cells=3e7, max_pairs=6e7):
    """Batches of (i, j), one per neighbouring-cell offset, of every pair with |Xb_j - Xa_i| < cut.

    Both sets are binned on one grid of cubic cells at least `cut` wide, so a pair in range sits in the
    same or an adjacent cell, and each A point is compared only with the B points of its 3^D cells:
    O(N x the points per cell) where the all-pairs matrix is O(N^2). The cell grows past `cut` (never
    shrinks below it) when the box would otherwise need more than `max_cells` cells -- a short WCA range
    across a wide bath -- which costs candidates and never loses a pair."""
    import itertools
    if Xa.shape[0] == 0 or Xb.shape[0] == 0:
        return
    D = Xa.shape[1]
    lo = torch.minimum(Xa.min(0).values, Xb.min(0).values)
    ext = (torch.maximum(Xa.max(0).values, Xb.max(0).values) - lo).tolist()
    c = float(cut)
    while math.prod(int(e / c) + 1 for e in ext) > max_cells:
        c *= 1.5
    dims = torch.tensor([int(e / c) + 1 for e in ext], device=Xa.device)
    stride = torch.ones(D, dtype=torch.long, device=Xa.device)
    for a in range(D - 2, -1, -1):
        stride[a] = stride[a + 1] * dims[a + 1]
    ca = torch.minimum(((Xa - lo) / c).floor().long().clamp_min(0), dims - 1)
    cb = torch.minimum(((Xb - lo) / c).floor().long().clamp_min(0), dims - 1)
    kb = (cb * stride).sum(1)
    order = torch.argsort(kb)
    counts = torch.bincount(kb, minlength=int(dims.prod()))
    start = torch.cumsum(counts, 0) - counts
    # ALL 3^D NEIGHBOUR CELLS AT ONCE, in as few batches as `max_pairs` candidates allow: one loop pass per
    # offset was 27 launches and host syncs per call, and on an H100 node the call was host-bound (55 ms
    # against 37 on an A6000 for the same 300,000 ions).
    offs = torch.tensor(list(itertools.product((-1, 0, 1), repeat=D)), device=Xa.device)
    cn = ca[:, None, :] + offs[None]
    ok = ((cn >= 0) & (cn < dims)).all(-1)
    k = (torch.minimum(cn.clamp_min(0), dims - 1) * stride).sum(-1)
    cnt = torch.where(ok, counts[k], torch.zeros_like(k))
    row = torch.cumsum(cnt.sum(1), 0)
    tot_all = int(row[-1])
    if tot_all == 0:
        return
    nb_ = max(1, math.ceil(tot_all / max_pairs))
    cuts = [0] + torch.searchsorted(row, torch.linspace(0, tot_all, nb_ + 1, device=Xa.device)[1:-1].to(row.dtype)).tolist() + [Xa.shape[0]]
    for a0, a1 in zip(cuts[:-1], cuts[1:]):
        if a1 <= a0:
            continue
        c_ = cnt[a0:a1].reshape(-1)
        tot = int(c_.sum())
        if tot == 0:
            continue
        i = torch.repeat_interleave(torch.arange(a0, a1, device=Xa.device).repeat_interleave(offs.shape[0]), c_,
                                    output_size=tot)
        first = torch.cumsum(c_, 0) - c_
        j = order[torch.repeat_interleave(start[k[a0:a1].reshape(-1)] - first, c_, output_size=tot)
                  + torch.arange(tot, device=Xa.device)]
        m = ((Xb[j] - Xa[i]) ** 2).sum(1) < cut * cut
        yield i[m], j[m]


# the frames whose readouts go to the log: a decade ladder, then every 25,000 -- a 400,000-frame run
# was blind between 80,000 and its end (exp04, 2026-09-25)
_REPORT = frozenset((1, 10, 100, 1000, 5000, 10000, 20000, 40000, 80000) + tuple(range(100000, 4000001, 25000)))


# =============================================================================================
# THE PROTEIN'S OWN FOLD
# =============================================================================================
# =============================================================================================
# NON-BONDED LAWS BETWEEN BEADS
# =============================================================================================
@register_operator("pair_potential", family="interaction", set="particle", kind="exchange",
                   title="apply pair forces between sets",
                   equation=r"""$$U_{\mathrm{LJ}}=4\epsilon\Big[\big(\tfrac{\sigma}{r}\big)^{12}-\big(\tfrac{\sigma}{r}\big)^{6}\Big],\qquad U_{\mathrm C}=\frac{k_C\,q_i q_j\,e^{-r/\lambda_D}}{r}$$""")
class PairPotential(Exchange):
    r"""a non-bonded pair law between the beads of two groups of sets (or within one), both ways: Lennard-Jones, WCA, screened Coulomb, harmonic soft sphere or Cooke's cos^2 tail, summed pair by pair.
    A -> A and B -> B: reads both sides' pos; emits a velocity on every set of A (v = mu F) and, with `react`, the equal and opposite on every set of B.

        U_{\mathrm{LJ}}=4\epsilon\Big[\big(\tfrac{\sigma}{r}\big)^{12}-\big(\tfrac{\sigma}{r}\big)^{6}\Big],\qquad U_{\mathrm C}=\frac{k_C\,q_i q_j\,e^{-r/\lambda_D}}{r}
        U_{LJ}         the Lennard-Jones pair energy, in the spec's energy unit
        \epsilon       `epsilon`, the well depth (the core's stiffness), energy
        \sigma         `sigma`, the contact distance, world length
        r              the pair distance
        U_C            the screened Coulomb (Debye-Hueckel) pair energy
        k_C            `k_c`, the Coulomb constant in the spec's units
        q_i, q_j       the two beads' charges (`charge`, `charge_with`)
        \lambda_D      `debye`, the screening length, world length

    Reference: Lennard-Jones, J. E. (1924). Proc. R. Soc. A 106:463; Weeks, J. D., Chandler, D. & Andersen, H. C. (1971). J. Chem. Phys. 54:5237 (the repulsive core); Debye, P. & Hueckel, E. (1923). Phys. Z. 24:185; O'Hern, C. S. et al. (2003). Phys. Rev. E 68:011306 (the harmonic soft sphere); Cooke, I. R., Kremer, K. & Deserno, M. (2005). Phys. Rev. E 72:011506 (the cos^2 tail)."""
    PARAM_UNITS = {'sigma': 'length', 'epsilon': 'energy', 'debye': 'length', 'cutoff': 'dimensionless', 'f_max': 'force', 'step_max': 'length', 'tail': 'length'}
    SYMBOLS = {'U_{LJ}': "the Lennard-Jones pair energy, in the spec's energy unit", '\\epsilon': "`epsilon`, the well depth (the core's stiffness), energy", '\\sigma': '`sigma`, the contact distance, world length', 'r': 'the pair distance', 'U_C': 'the screened Coulomb (Debye-Hueckel) pair energy', 'k_C': "`k_c`, the Coulomb constant in the spec's units", 'q_i, q_j': "the two beads' charges (`charge`, `charge_with`)", '\\lambda_D': '`debye`, the screening length, world length'}

    EMIT = "velocity"
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    INPUTS = ["particle", "particle"]
    OUTPUTS = ["particle", "particle"]
    READS = ["pos"]
    REQUIRES_PARAMS = ["law"]
    MECHANISM_TAGS = ["lennard_jones", "excluded_volume", "adhesion", "electrostatics", "screened_coulomb",
                      "soft_sphere"]
    PARAM_ROLES = {"sets": "side_A_default_the_at_set", "with": "side_B_one_set", "with_sets": "side_B_several_sets",
                   "law": "lj_wca_coulomb_cooke_or_harmonic_or_a_list_summed", "epsilon": "well_depth_sim_energy",
                   "sigma": "contact_distance_world_or_per_set_map_lorentz_mixed",
                   "tail": "cooke_cosine_tail_width_world",
                   "cutoff": "range_world", "k_c": "kT_times_bjerrum_length",
                   "charge": "charge_of_A_in_e_or_a_block", "charge_with": "charge_of_B_in_e_or_a_block",
                   "debye": "screening_length_world", "exclude": "integer_block_whose_equal_pairs_are_skipped",
                   "exclude_same_set": "skip_pairs_inside_one_set", "mobility": "mobility_of_A",
                   "mobility_with": "mobility_of_B", "react": "return_the_reaction_on_B",
                   "f_max": "cap_on_one_pair_force_a_guard",
                   "step_max": "cap_on_the_step_one_pair_force_makes_per_frame_a_guard", "slab": "z_window_and_eps_ratio_of_a_low_permittivity_region"}
    MAY_MUTATE_INTEGRATED_STATE = True             # writes B's `force` readout block in place
    REFERENCE = ("Lennard-Jones, J.E. (1924). Proc. R. Soc. A 106:463; Weeks, Chandler & Andersen "
                 "(1971). J. Chem. Phys. 54:5237; Debye & Hueckel (1923). Phys. Z. 24:185.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.A = [str(s) for s in (params.get("sets") or [self.at])]
        if params.get("with_sets"):
            self.B = [str(s) for s in params["with_sets"]]
        elif params.get("with"):
            self.B = [str(params["with"])]
        else:
            self.B = list(self.A)
        self.same = self.B == self.A
        # `law`: ONE LAW OR A LIST, summed pair by pair. exp04's hole spec took EIGHT instances for its
        # ions -- Coulomb and WCA for each of K-K, Cl-Cl, K-Cl, and WCA for each ion with the lipids --
        # because a law, a charge, a size and a mobility were one scalar per instance. With a list of
        # laws and per-set values they are two instances (ions with ions, ions with the lipids) giving
        # the same forces: tests/test_channel_ops.py::test_merged_pair_potential_equals_the_split_one.
        _law = params["law"]
        self.laws = [str(l).lower() for l in (_law if isinstance(_law, (list, tuple)) else [_law])]
        for l in self.laws:
            if l not in ("lj", "wca", "coulomb", "cooke", "harmonic"):
                raise ValueError(f"pair_potential: law is lj, wca, coulomb, cooke or harmonic (or a list of them), "
                                 f"got {l!r}")
        if len(set(self.laws)) != len(self.laws):
            raise ValueError(f"pair_potential: a law listed twice in {self.laws}")
        self.law = "+".join(self.laws)
        self.eps = float(params.get("epsilon", 1.0))
        # `sigma`: a number, or `{set: sigma_i}` -- then a pair's sigma_ij = (sigma_i + sigma_j) / 2
        # (Lorentz's rule; for ions sized by 2^(1/6) sigma_i = 2 r_i it is the contact r_i + r_j).
        _sg = params.get("sigma", 0.01)
        self.sigma_of = {str(k): float(v) for k, v in _sg.items()} if isinstance(_sg, dict) else None
        self.sigma = max(self.sigma_of.values()) if self.sigma_of else float(_sg)
        # `law: cooke` -- THE COSINE TAIL (Cooke, Kremer & Deserno 2005, PRE 72:011506): the WCA core,
        # a flat well -eps out to r_c = 2^(1/6) sigma, and a smooth cos^2 tail of width `tail` (world)
        # beyond it. A BROAD attraction keeps coarse-grained lipids a cohesive FLUID at eps ~ kT --
        # the Lennard-Jones layer of exp04 round 1, a liquid near its critical point, yielded under
        # tension (10 mN/m applied, 1.5 carried) -- which is the property the model was built for.
        self.tail_given = float(params["tail"]) if "tail" in params else None
        self.tail = self.tail_given if self.tail_given is not None else 1.6 * self.sigma
        self.r_c = 2.0 ** (1.0 / 6.0) * self.sigma
        # `cutoff`: a number for every law, or `{law: range}`; a law not given takes its own default
        # (lj 2.5 sigma, coulomb 0.25, cooke r_c + tail), and WCA never reaches past its minimum
        # 2^(1/6) sigma. The neighbour search runs to the largest of them, `self.cutoff`.
        _cut = params.get("cutoff")
        self.cut_given = {l: (float(_cut[l]) if isinstance(_cut, dict) and l in _cut else
                              None if isinstance(_cut, dict) or _cut is None else float(_cut)) for l in self.laws}
        reach = {"lj": 2.5 * self.sigma, "wca": 2.0 ** (1.0 / 6.0) * self.sigma, "coulomb": 0.25,
                 "cooke": self.r_c + self.tail, "harmonic": self.sigma}
        self.cutoff = max((self.cut_given[l] if self.cut_given[l] is not None else reach[l])
                          if l not in ("wca", "harmonic")
                          else min(self.cut_given[l] if self.cut_given[l] is not None else reach[l], reach[l])
                          for l in self.laws)
        self.k_c = float(params.get("k_c", 0.0))
        # `charge` / `charge_with`: the beads' charges in e -- a number, the name of a block, or
        # `{set: number | block}`. `charge_with` defaults to `charge` (a map is looked up on B's sets).
        self.q_a = params.get("charge", 1.0)
        self.q_b = params.get("charge_with", self.q_a)
        self.debye = float(params.get("debye", 0.0))
        # `slab: {z: [z0, z1], eps_ratio: r}` -- A LOW-PERMITTIVITY REGION. When BOTH charges of a pair
        # are inside the membrane's slab (a channel's filter and cavity: protein and lipid around, little
        # water), their interaction is unscreened and multiplied by r = eps_water / eps_slab; outside it
        # is the bath's (k_c at eps_water, Debye-screened). The one physical lever on how hard ions in a
        # filter push on each other (knock-on) against how hard its carbonyls hold them -- the balance
        # Berneche & Roux (2001) computed, and a continuum dielectric boundary would set.
        _sl = params.get("slab")
        self.slab_z = [float(v) for v in _sl["z"]] if _sl else None
        self.slab_r = float(_sl.get("eps_ratio", 1.0)) if _sl else 1.0
        # `slab.edge` (world) -- THE REGION'S BOUNDARY SMOOTH, AND THE FORCE CONSERVATIVE. With edge 0 the region is a
        # switch: a pair's Coulomb jumps by eps_ratio when both charges are in, with no force for the jump, so an ion
        # carried through the region and out gains or loses the jump for free -- a pump. Measured along Kv1.2's
        # filter at ratio 8: the charged residues' term ended 14 kT below where it began once the ion was out
        # (tools/channel_axis_energy.py, 2026-09-26), against the 8 kT a 200 mV clamp supplies. With edge > 0 the
        # pair energy is U = k_c qa qb f / r, f = 1 + (eps_ratio - 1) w(za) w(zb), w a cosine window, and the force
        # is its whole gradient, the z-derivative of f included (unscreened: `debye` must be 0).
        # `slab.born: {set: kT}` -- THE ION'S OWN COST OF BEING THERE: +born w(z) on each bead of that set on side A,
        # the Born energy 56 / (2 a) (1 / eps_region - 1 / eps_water) kT of an ion of radius a (nm) moved from water
        # into the region. The carbonyls' stronger pull at the region's permittivity is what pays it back; one
        # without the other is a trap (the pull alone) or a wall (the cost alone).
        self.slab_e = float(_sl.get("edge", 0.0)) if _sl else 0.0
        self.slab_born = {str(k): float(v) for k, v in (_sl.get("born") or {}).items()} if _sl else {}
        if self.slab_e > 0 and self.debye > 0:
            raise ValueError("pair_potential: a smooth `slab` (edge > 0) is unscreened; set `debye: 0`")
        self.exclude = params.get("exclude")
        self.excl_set = bool(params.get("exclude_same_set", False))
        # `mobility` / `mobility_with`: a number, or `{set: mu}` (v = mu F on each bead of that set).
        _mu = params.get("mobility", 1.0)
        self.mu_a = {str(k): float(v) for k, v in _mu.items()} if isinstance(_mu, dict) else float(_mu)
        _mw = params.get("mobility_with", _mu)
        self.mu_b = {str(k): float(v) for k, v in _mw.items()} if isinstance(_mw, dict) else float(_mw)
        self.react = bool(params.get("react", not self.same))
        # TWO GUARDS AGAINST SEEDING OVERLAPS, printed when they act, never a law. `f_max` caps one
        # pair's force (sim force) whatever the law. `step_max` caps the STEP that force makes the
        # A bead take in one frame (world; a number, or `{law: step}`): |F| <= step / (mu_i dt). The
        # step is what the guard is about, and it holds per set with one number -- the f_max it
        # replaces had to be written once per set, each divided by that set's mobility and dt.
        self.f_max = float(params.get("f_max", 0.0))
        _st = params.get("step_max")
        self.step_max = ({l: (float(_st[l]) if isinstance(_st, dict) and l in _st else
                              None if isinstance(_st, dict) or _st is None else float(_st)) for l in self.laws})

    @staticmethod
    def _per_bead(lvls, names, v, what, X):
        """One value per bead of a side: `v` a number, a block's name, or `{set: number | block}`."""
        parts = []
        for nm, lv in zip(names, lvls):
            u = v.get(nm) if isinstance(v, dict) else v
            if u is None:
                raise ValueError(f"pair_potential: no {what} for set {nm!r} in {v}")
            if isinstance(u, str):
                b = _block(lv, u)
                if b is None:
                    raise ValueError(f"pair_potential: {what} block {u!r} not declared on {nm!r}")
                parts.append(b[:, 0].to(X.dtype))
            else:
                parts.append(torch.full((lv.get("pos").shape[0],), float(u), device=X.device, dtype=X.dtype))
        return torch.cat(parts)

    def _side(self, H, names, q, mu, moved=True):
        lvls = [H.level(n) for n in names]
        X = torch.cat([lv.get("pos") for lv in lvls])
        occ = torch.cat([lv.occ for lv in lvls]) > 0
        sizes = [lv.get("pos").shape[0] for lv in lvls]
        grp = torch.cat([torch.full((s,), k, device=X.device, dtype=torch.long) for k, s in enumerate(sizes)])
        blk = None
        if self.exclude:
            bs = [_block(lv, self.exclude) for lv in lvls]
            if all(b is not None for b in bs):
                blk = torch.cat([b[:, 0] for b in bs]).round().long()
        qq = self._per_bead(lvls, names, q, "charge", X) if "coulomb" in self.laws else None
        mm = self._per_bead(lvls, names, mu, "mobility", X) if moved and isinstance(mu, dict) else None
        sg = self._per_bead(lvls, names, self.sigma_of, "sigma", X) if self.sigma_of else None
        return lvls, X, occ, sizes, grp, blk, qq, mm, sg

    def _force(self, d, r, qa=None, qb=None, inslab=None, sig=None, mu_i=None, h=1.0):
        """Force on the A bead of each pair (vector; d = x_b - x_a points A -> B), summed over the
        laws, and the number of pair-law terms a guard capped. `sig` is the pair's sigma (a number
        or one per pair), `mu_i` the A bead's mobility (for `step_max`)."""
        s = self.sigma if sig is None else sig
        per_pair = torch.is_tensor(s)
        mag, n_capped = None, 0
        ez = None
        for l in self.laws:
            if l in ("lj", "wca"):
                sr6 = (s / r) ** 6
                m = 24.0 * self.eps * (2.0 * sr6 * sr6 - sr6) / r        # -dU/dr: positive = repulsion
            elif l == "cooke":
                r_c = 2.0 ** (1.0 / 6.0) * s if per_pair else self.r_c
                tl = self.tail if (self.tail_given is not None or not per_pair) else 1.6 * s
                sr6 = (s / r) ** 6
                core = torch.where(r < r_c, 24.0 * self.eps * (2.0 * sr6 * sr6 - sr6) / r, torch.zeros_like(r))
                x = (math.pi / (2.0 * tl)) * (r - r_c)
                m = core + torch.where((r >= r_c) & (r <= r_c + tl),
                                       -self.eps * (math.pi / (2.0 * tl)) * torch.sin(2.0 * x), torch.zeros_like(r))
            elif l == "harmonic":                                       # -dU/dr = eps (1 - r/s) / s, r < s
                m = self.eps * (1.0 - r / s) / s
            else:
                scr = torch.exp(-r / self.debye) if self.debye > 0 else torch.ones_like(r)
                m = self.k_c * qa * qb * scr * (1.0 / (r * r) + ((1.0 / (self.debye * r)) if self.debye > 0 else 0.0))
                if isinstance(inslab, tuple):                              # the smooth region: f and its z-gradient
                    wa_, wb_, dwa_, dwb_ = inslab
                    g_ = self.k_c * qa * qb * (self.slab_r - 1.0) / r
                    m = self.k_c * qa * qb * (1.0 + (self.slab_r - 1.0) * wa_ * wb_) / (r * r)
                    ez = (-g_ * dwa_ * wb_, -g_ * wa_ * dwb_)
                elif inslab is not None:                                   # inside the slab: unscreened, eps lowered
                    m = torch.where(inslab, self.k_c * self.slab_r * qa * qb / (r * r), m)
            # THIS LAW'S OWN RANGE, where it is shorter than the search's: WCA's minimum per pair,
            # or a law listed beside a longer-ranged one. One law alone is searched at its own range.
            if l in ("wca", "harmonic"):
                cut = 2.0 ** (1.0 / 6.0) * s if l == "wca" else s
                if self.cut_given[l] is not None:
                    cut = torch.clamp(cut, max=self.cut_given[l]) if per_pair else min(cut, self.cut_given[l])
            elif l == "cooke" and self.cut_given[l] is None:
                cut = (2.0 ** (1.0 / 6.0) * s + (self.tail if self.tail_given is not None else 1.6 * s)) \
                    if per_pair else self.r_c + self.tail
            elif l == "lj" and self.cut_given[l] is None:
                cut = 2.5 * s
            else:
                cut = self.cut_given[l] if self.cut_given[l] is not None else 0.25
            if per_pair or cut < self.cutoff:
                m = torch.where(r < cut, m, torch.zeros_like(m))
            if self.f_max > 0:
                m = m.clamp(-self.f_max, self.f_max)
                n_capped += int((m.abs() >= 0.999 * self.f_max).sum())
            if self.step_max[l] is not None:
                cap = self.step_max[l] / (mu_i * h)                 # a number, or one per pair
                n_capped += int((m.abs() >= 0.999 * cap).sum())
                m = torch.maximum(torch.minimum(m, cap), -cap) if torch.is_tensor(cap) else m.clamp(-cap, cap)
            mag = m if mag is None else mag + m
        f = -(mag / r)[:, None] * d
        if ez is None:
            return f, n_capped, None
        if self.step_max.get("coulomb") is not None:                  # the z-term under the Coulomb guard as well
            cap = self.step_max["coulomb"] / (mu_i * h)
            ez = tuple(torch.maximum(torch.minimum(e_, cap), -cap) if torch.is_tensor(cap) else e_.clamp(-cap, cap)
                       for e_ in ez)
        return f, n_capped, ez

    # THE NEIGHBOUR SEARCH CHANGES METHOD ABOVE THIS MANY CANDIDATE PAIRS (na x nb), not the pairs it
    # finds. All-pairs in chunks of 4096 is exact and fast up to ~1e8 -- every exp04 rig -- and at a bath
    # holding 100 channels (3e5 ions, 1e11 pairs) it is a day per frame; the cell grid finds the same pairs
    # in O(N). The threshold sits far above any single-channel rig, so those run the path they always ran.
    CELL_LIST_ABOVE = 2e9

    def _pairs(self, Xa, Xb, oa, ob, ga, gb, ba, bb):
        """Batches of (i, j) -- i into A, j into B -- of every live pair nearer than `cutoff`, each unordered
        pair once when A is B, less the pairs `exclude` / `exclude_same_set` remove."""
        na, nb = Xa.shape[0], Xb.shape[0]
        if float(na) * float(nb) <= self.CELL_LIST_ABOVE:
            step = 4096
            colb = torch.arange(nb, device=Xa.device)
            for s in range(0, na, step):
                xa = Xa[s:s + step]
                near = (torch.cdist(xa, Xb) < self.cutoff) & oa[s:s + step, None] & ob[None, :]
                if self.same:
                    near &= torch.arange(s, s + xa.shape[0], device=Xa.device)[:, None] < colb[None, :]
                    if self.excl_set:
                        near &= ga[s:s + step, None] != gb[None, :]
                if ba is not None and bb is not None:
                    near &= ba[s:s + step, None] != bb[None, :]
                ii, jj = torch.nonzero(near, as_tuple=True)
                if ii.numel():
                    yield ii + s, jj
            return
        la, lb = torch.nonzero(oa).squeeze(1), torch.nonzero(ob).squeeze(1)
        for i, j in _cell_pairs(Xa[la], Xb[lb], self.cutoff):
            i, j = la[i], lb[j]
            keep = torch.ones_like(i, dtype=torch.bool)
            if self.same:
                keep &= i < j
                if self.excl_set:
                    keep &= ga[i] != gb[j]
            if ba is not None and bb is not None:
                keep &= ba[i] != bb[j]
            yield i[keep], j[keep]

    def forward(self, H, mask=None):
        LA, Xa, oa, sa, ga, ba, qa, ma, sga = self._side(H, self.A, self.q_a, self.mu_a)
        if self.same:
            LB, Xb, ob, sb, gb, bb, qb, mb, sgb = LA, Xa, oa, sa, ga, ba, qa, ma, sga
        else:
            LB, Xb, ob, sb, gb, bb, qb, mb, sgb = self._side(H, self.B, self.q_b, self.mu_b, moved=self.react)
        h = float(getattr(H, "dt", 1.0))
        na, nb = Xa.shape[0], Xb.shape[0]
        Fa = torch.zeros_like(Xa)
        Fb = torch.zeros_like(Xb)
        n_pairs, n_capped = 0, 0
        if self.slab_z is not None and self.slab_e > 0:
            zm_, hz_ = 0.5 * (self.slab_z[0] + self.slab_z[1]), 0.5 * (self.slab_z[1] - self.slab_z[0])

            def _w(X_):
                u_ = X_[:, 2] - zm_
                w_, dw_ = _slab_step(u_.abs(), hz_, self.slab_e)
                return w_, dw_ * torch.sign(u_)
            wA, dwA = _w(Xa)
            wB, dwB = (wA, dwA) if self.same else _w(Xb)
            if self.slab_born:                                          # the ion's own cost of being in the region
                s0 = 0
                for nm_, n_ in zip(self.A, sa):
                    if self.slab_born.get(nm_, 0.0):
                        Fa[s0:s0 + n_, 2] += -self.slab_born[nm_] * dwA[s0:s0 + n_]
                    s0 += n_
        for gi, jj in self._pairs(Xa, Xb, oa, ob, ga, gb, ba, bb):
            d = Xb[jj] - Xa[gi]
            r = d.norm(dim=1).clamp_min(1e-9)
            inslab = None
            if self.slab_z is not None and "coulomb" in self.laws and self.slab_e > 0:
                inslab = (wA[gi], wB[jj], dwA[gi], dwB[jj])
            elif self.slab_z is not None and "coulomb" in self.laws:
                za, zb = Xa[gi, 2], Xb[jj, 2]
                inslab = ((za > self.slab_z[0]) & (za < self.slab_z[1]) & (zb > self.slab_z[0]) & (zb < self.slab_z[1]))
            sig = None if sga is None else 0.5 * (sga[gi] + sgb[jj])
            mu_i = self.mu_a if ma is None else ma[gi]
            f, nc, ez = self._force(d, r, None if qa is None else qa[gi], None if qb is None else qb[jj], inslab,
                                    sig, mu_i, h)
            n_capped += nc
            Fa.index_add_(0, gi, f)
            Fb.index_add_(0, jj, -f)
            if ez is not None:
                Fa[:, 2].index_add_(0, gi, ez[0])
                Fb[:, 2].index_add_(0, jj, ez[1])
            n_pairs += int(gi.numel())
        fr = _frame(H)
        if fr in _REPORT:
            name_a = self.A[0] + (f"..{self.A[-1]}" if len(self.A) > 1 else "")
            name_b = self.B[0] + (f"..{self.B[-1]}" if len(self.B) > 1 else "")
            print(f"[pair_potential f{fr}] {self.law} {name_a}<->{name_b}: {n_pairs:,} pairs in range"
                  + (f"; {n_capped} CAPPED by a guard" if n_capped else ""), flush=True)
        out = {}

        def _put(names, sizes, V):
            s0 = 0
            for nm, n in zip(names, sizes):
                out[nm] = out.get(nm, 0) + V[s0:s0 + n]
                s0 += n

        def _mob(m, scalar, F):
            return scalar * F if m is None else m[:, None] * F
        if self.same:
            _put(self.A, sa, _mob(ma, self.mu_a, Fa + Fb) * oa[:, None].to(Xa.dtype))
            return out
        if len(LB) == 1 and "force" in LB[0].state_schema._slices:
            _write(LB[0], "force", Fb.detach())
        _put(self.A, sa, _mob(ma, self.mu_a, Fa) * oa[:, None].to(Xa.dtype))
        if self.react:
            _put(self.B, sb, _mob(mb, self.mu_b, Fb) * ob[:, None].to(Xb.dtype))
        return out


# pair_potential[typed] -- the well depth per pair of cell TYPES inside one set (experiment 9: Steinberg
# 1963 / Graner & Glazier 1992 adhesion tables; a cell retyped by `contact_retype` adheres by its new type).
# Moved here from sorting_ops.py on 2026-09-27, unchanged.
def _type_index(lvl, name, where):
    names = list(getattr(lvl, "type_names", None) or [])
    if name not in names:
        raise ValueError(f"{where}: type {name!r} is not one of the set's types {names}")
    return names.index(name)

# ---- pair_potential[mpm]: the same pair law, delivered to MATERIAL POINTS (exp04 MscS morph, 2026-09-29) ---------
@register_operator("pair_potential", implementation="mpm", family="interaction", set="particle", kind="exchange",
                   title="apply pair forces on material points",
                   equation=r"""$$\mathbf a_i=\frac{1}{m_i}\sum_j -\nabla_i U(r_{ij})$$""")
class PairPotentialMPM(PairPotential):
    r"""pair_potential for a set the MPM substep integrates: the same laws, sigma, epsilon, charges and cutoffs, each bead's force divided by its mass and emitted as an acceleration.
    A -> A (and B with `react`): reads pos and mass, emits `mpm_acceleration` (the substep's a_ext); `mobility` is 1 inside and `step_max` is refused, `f_max` caps the force as in the default.

        \mathbf a_i=\frac{1}{m_i}\sum_j -\nabla_i U(r_{ij})
        a_i            the acceleration of bead i
        m_i            its mass (unit mass for a set without one)
        U(r_{ij})      the pair energy of the default's law(s)
        r_{ij}         the pair distance

    Reference: as pair_potential; the acceleration routing of stator_push and mpm_anchor (Plexus, this work)."""
    PARAM_UNITS = {'exclude_span': 'count'}
    SYMBOLS = {'a_i': 'the acceleration of bead i', 'm_i': 'its mass (unit mass for a set without one)', 'U(r_{ij})': "the pair energy of the default's law(s)", 'r_{ij}': 'the pair distance'}
    PARAM_ROLES = {'exclude_span': 'how_many_neighbouring_values_of_the_exclude_block_also_exclude_a_pair', 'exclude': 'block_whose_equal_values_exclude_a_pair', 'mobility_with': 'velocity_per_force_on_side_B', 'react': 'whether_the_reaction_is_returned_on_B', 'step_max': 'refused_here'}
    EMIT = "mpm_acceleration"

    # `exclude_span: s` (with `exclude: <block>`): skip a pair when its two block values differ by AT MOST s -- with
    # the block a residue number (chain x 10000 + residue), s = 1 keeps a residue's own atoms and the peptide bond to
    # its neighbour (0.13 nm) out of the repulsion, as a clash count does; the default's `exclude` skips equals only.
    def __init__(self, params, device="cpu"):
        if params.get("step_max") is not None:
            raise ValueError("pair_potential[mpm]: `step_max` caps a step of an overdamped bead; use `f_max`")
        self.span = params.get("exclude_span")
        if self.span is not None and not params.get("exclude"):
            raise ValueError("pair_potential[mpm]: `exclude_span` needs `exclude: <block>` to read")
        params = {k: v for k, v in params.items() if k != "exclude_span"}
        params = {**params, "mobility": 1.0}
        if params.get("react"):
            params["mobility_with"] = 1.0
        super().__init__(params, device)

    def _pairs(self, Xa, Xb, oa, ob, ga, gb, ba, bb):
        if self.span is None or ba is None or bb is None:
            yield from super()._pairs(Xa, Xb, oa, ob, ga, gb, ba, bb)
            return
        for gi, jj in super()._pairs(Xa, Xb, oa, ob, ga, gb, None, None):
            keep = (ba[gi] - bb[jj]).abs() > float(self.span)
            yield gi[keep], jj[keep]

    def forward(self, H, mask=None):
        out = super().forward(H, mask)
        for nm, F in list(out.items()):
            m = getattr(H.level(nm), "mass", None)
            if m is not None:
                out[nm] = F / m.to(F.dtype).reshape(-1, 1).clamp_min(1e-30)
        return out



# ---- pair_potential[table]: a LAW, a size and a depth per pair of types, within one set or across two -------------
# (exp04 Phase G, 2026-09-30: Cooke, Kremer & Deserno's three-bead lipid needs heads repulsive at 0.95 sigma against
# every bead and tails attractive among themselves, inside the lipid set AND against the frozen lipids of the rim.)
@register_operator("pair_potential", model="soft_adhesive", family="interaction", set="particle", kind="exchange",
                   title="apply soft sticky contact",
                   equation=r"""$$U(r)=\tfrac{\epsilon}{2}\big(1-\tfrac{r}{\sigma}\big)^2\ (r<\sigma)\;-\;\epsilon_a\cos^2\!\Big(\frac{\pi(r-\sigma)}{2w}\Big)\ (\sigma\le r\le\sigma+w),\quad -\epsilon_a\ (r<\sigma)$$""")
class PairPotentialSoftAdhesive(PairPotential):
    r"""pair_potential whose contact both resists overlap softly and holds: the harmonic soft sphere with Cooke's cos^2 adhesive shoulder grafted onto its contact distance, de-adhesion when pulled harder than eps_a pi / 2w.
    A -> A (and B with `react`): reads pos, emits a velocity (or an acceleration under MPM), as the default; the neighbour search switches to a cell list above 5e7 candidate pairs.

        U(r)=\tfrac{\epsilon}{2}\big(1-\tfrac{r}{\sigma}\big)^2\ (r<\sigma)\;-\;\epsilon_a\cos^2\!\Big(\frac{\pi(r-\sigma)}{2w}\Big)\ (\sigma\le r\le\sigma+w),\quad -\epsilon_a\ (r<\sigma)
        U(r)           the pair energy at distance r
        \epsilon       `epsilon`, the core's stiffness, energy
        \sigma         `sigma`, the contact distance, world length
        \epsilon_a     `adhesion`, the well's depth, energy
        w              `tail`, the shoulder's width, world length

    Reference: Cooke, I. R., Kremer, K. & Deserno, M. (2005). Phys. Rev. E 72:011506 (the cos^2 tail); O'Hern, C. S. et al. (2003). Phys. Rev. E 68:011306 (the harmonic core); Szabo, B. et al. (2006). Phys. Rev. E 74:061908."""
    PARAM_UNITS = {'sigma': 'length'}
    SYMBOLS = {'U(r)': 'the pair energy at distance r', '\\epsilon': "`epsilon`, the core's stiffness, energy", '\\sigma': '`sigma`, the contact distance, world length', '\\epsilon_a': "`adhesion`, the well's depth, energy", 'w': "`tail`, the shoulder's width, world length"}
    CELL_LIST_ABOVE = 5e7
    REQUIRES_PARAMS = []                                  # `law` is the harmonic core, implied
    PARAM_ROLES = dict(PairPotential.PARAM_ROLES, adhesion="well_depth_of_the_adhesive_shoulder_sim_energy",
                       tail="width_of_the_adhesive_shoulder_world")
    MECHANISM_TAGS = ["soft_sphere", "adhesion", "de_adhesion", "excluded_volume"]
    REFERENCE = ("Cooke, I.R., Kremer, K. & Deserno, M. (2005). Phys. Rev. E 72:011506; O'Hern, C.S. et al. (2003). "
                 "Phys. Rev. E 68:011306; Szabo, B. et al. (2006). Phys. Rev. E 74:061908.")

    def __init__(self, params, device="cpu"):
        law = params.get("law", "harmonic")
        if (law if isinstance(law, str) else "+".join(law)) != "harmonic":
            raise ValueError("pair_potential[soft_adhesive]: the core is the harmonic soft sphere; `law` is harmonic "
                             f"(or omitted), got {law!r}")
        if isinstance(params.get("sigma"), dict):
            raise ValueError("pair_potential[soft_adhesive]: one contact distance `sigma` (a number)")
        p = dict(params); p["law"] = "harmonic"
        self.eps_a = float(p.pop("adhesion", 0.0))
        tail = p.pop("tail", None)
        super().__init__(p, device)
        self.w = float(tail) if tail is not None else 0.5 * self.sigma
        if self.w <= 0.0:
            raise ValueError("pair_potential[soft_adhesive]: `tail` must be > 0")
        if self.eps_a < 0.0:
            raise ValueError("pair_potential[soft_adhesive]: `adhesion` is a well depth, >= 0")
        self.cutoff = self.sigma + (self.w if self.eps_a > 0.0 else 0.0)

    def _force(self, d, r, qa=None, qb=None, inslab=None, sig=None, mu_i=None, h=1.0):
        s = self.sigma
        m = torch.where(r < s, self.eps * (1.0 - r / s) / s, torch.zeros_like(r))
        if self.eps_a > 0.0:
            u = (r - s) / self.w
            m = m + torch.where((u >= 0.0) & (u <= 1.0),
                                -self.eps_a * (math.pi / (2.0 * self.w)) * torch.sin(math.pi * u), torch.zeros_like(r))
        n_capped = 0
        if self.f_max > 0:
            m = m.clamp(-self.f_max, self.f_max)
            n_capped = int((m.abs() >= 0.999 * self.f_max).sum())
        return -(m / r)[:, None] * d, n_capped, None


# =============================================================================================
# THE BATH, THE PLANE, THE PIPETTE
# =============================================================================================
def _path_from_beads(H, cfg, r_ion, z0):
    """THE ION PATH OF THE PROTEIN AS IT IS NOW, from its beads (one per residue): per height z (bins `dz` from
    `z` lo to hi, world), R = the nearest bead to the axis less `reach` (a residue's side chain beyond its alpha
    carbon, fitted per channel so the deposited states' all-atom paths are reproduced), ENCLOSED when the beads
    within R + reach + `ring` cover `sectors` of 8 sectors round the axis, and HYDROPHOBIC when fewer than
    `hydrophobic_below` of the beads lining that height (within `lining` of the nearest) are polar (`hyd` block
    0). Returns the barrier's table [z, rp, R, dry, cx, cy] (world) for an ion of radius `r_ion`, the axis the
    xy centroid of the beads within `axis_half` of the mid-plane. The same rules as channel_spec._ion_paths."""
    sets = [str(s_) for s_ in cfg["sets"]]
    A = torch.cat([H.level(s_).get("pos").detach() for s_ in sets]).to(torch.float64)
    hb = cfg.get("hyd")
    hyd = (torch.cat([_block(H.level(s_), hb)[:, 0] for s_ in sets]).to(torch.float64) if hb
           else torch.ones(A.shape[0], dtype=torch.float64, device=A.device))
    ah = float(cfg.get("axis_half", 0.1))
    core = (A[:, 2] - z0).abs() < ah
    c = A[core].mean(0) if bool(core.any()) else A.mean(0)
    zlo, zhi = [float(v) for v in cfg["z"]]
    dz = float(cfg["dz"]); win = float(cfg.get("window", dz * 3))
    reach = float(cfg["reach"]); ring = float(cfg.get("ring", reach * 2)); lin = float(cfg.get("lining", reach * 0.6))
    zb = torch.arange(zlo, zhi + 1e-12, dz, dtype=torch.float64, device=A.device)
    dx = A[:, 0] - c[0]; dy = A[:, 1] - c[1]
    r = torch.sqrt(dx * dx + dy * dy)
    sec = (((torch.atan2(dy, dx) + math.pi) / (2 * math.pi) * 8).long() % 8)
    m = (A[None, :, 2] - zb[:, None]).abs() < win                                   # [bins, beads]
    big = torch.full_like(r, 1e9)
    rmin = torch.where(m, r[None, :], big[None, :]).min(1).values                   # [bins]
    has = m.any(1)
    ringm = m & (r[None, :] < (rmin[:, None] + ring))
    onehot = torch.nn.functional.one_hot(sec, 8).to(torch.float64)                 # [beads, 8]
    cover = (ringm.to(torch.float64) @ onehot > 0).sum(1)                           # sectors covered
    enclosed = has & (cover >= int(cfg.get("sectors", 6)))
    linm = m & (r[None, :] < (rmin[:, None] + lin))
    npol = (linm.to(torch.float64) * (1.0 - hyd)[None, :]).sum(1)
    nlin = linm.to(torch.float64).sum(1).clamp_min(1.0)
    polar = (npol / nlin) >= float(cfg.get("hydrophobic_below", 0.2))
    R = torch.where(has, rmin - reach, torch.zeros_like(rmin)).clamp_min(0.0)
    rp = torch.where(enclosed, (R - r_ion).clamp_min(0.0), torch.zeros_like(R))
    sf = float(cfg.get("single_file", 0.0))
    rp = torch.where(enclosed & polar & (rp < sf), torch.full_like(rp, sf), rp)
    d0, d1 = [float(v) for v in cfg["dry"]]
    t = ((R - d0) / max(d1 - d0, 1e-12)).clamp(0.0, 1.0)
    dry = torch.where(enclosed & ~polar, 0.5 * (1.0 + torch.cos(math.pi * t)), torch.zeros_like(R))
    w = max(int(round(float(cfg.get("dry_smooth", 2 * dz)) / dz)), 1)
    pad = torch.nn.functional.pad(dry[None, None, :], (w, w))
    dmax = torch.nn.functional.max_pool1d(pad, 2 * w + 1, stride=1)[0, 0]
    ker = torch.cat([torch.arange(1, w + 1), torch.arange(w, 0, -1)]).to(torch.float64).to(A.device)
    ker = ker / ker.sum()
    dry_s = torch.nn.functional.conv1d(torch.nn.functional.pad(dmax[None, None, :], (w - 1, w)), ker[None, None, :])[0, 0]
    dry_s = dry_s[:dry.numel()].clamp(0.0, 1.0)
    cx = torch.full_like(R, float(c[0])); cy = torch.full_like(R, float(c[1]))
    return torch.stack([zb, rp, R, dry_s, cx, cy], 1)


# ---- morph_gate[waypoints]: the gate along a RECORDED path (exp04 Phase G, 2026-09-29) --------------------------
# ---- morph_gate[clamp]: the gate HELD at one lambda, the force along the path recorded (exp04 Phase G1) ---------
# =============================================================================================
# THE ELECTROLYTE, AND THE CURRENT THROUGH WHAT THE PROTEIN LEAVES OPEN
# =============================================================================================
# =============================================================================================
# IONS: THE FIELD THAT DRIVES THEM, AND THE CURRENT THEY CARRY
# =============================================================================================
# =============================================================================================
# RIGID DOMAINS
# =============================================================================================
