"""Myosin, and the two places an epithelial cell can put it: on its junctions, or across its apex.

Myosin is what makes a vertex model contractile beyond a constant line tension: where the vertex
model writes one Lambda for every edge, these operators give each junction its own multiplier
and a rule for how it changes. All of them keep that multiplier keyed by VERTEX PAIR rather than
by half-edge index, which is what lets it survive a T1, a division or a death without any
topology operator knowing it exists.

In the order they appear below:

    junction_myosin     structural   per-junction myosin, recruited by tension
    junction_sync       rewire       re-key the store onto half-edge arrays topology has changed
    medioapical_myosin  lateral      the apical pool: an areal density on the face, not its edges
    cytokinetic_ring    structural   the ring a dividing cell leaves on the junction it just built
    junction_pcp        lateral      planar cell polarity: two complexes on each side of each junction

then the second model of `junction_myosin`, a different hypothesis in the same slot:

    junction_myosin[two_pool]  the belt fed by the medioapical pool, as a conserved amount
    junction_myosin[rest_length]  each junction an elastic element whose rest length remodels (exp 13)

The medioapical operators live in this file because the two-pool model is not a separate
mechanism: it is the same junction bookkeeping with a second reservoir on the face, and it calls
the same helpers -- `_live_edges`, `_lookup`, `_scatter_full`, `edge_tension`. Split across two
files, the shared half of one model would be private to the other.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from plexus.models.base import Rewire, Structural
from plexus.models.registry import register_operator
from plexus.models.base import Lateral, Structural
from plexus.operators.vertex_ops import face_geometry_3d


MYOSIN_TRACE: list = []


def _edge_key(vi, vj, stride):
    """Unordered vertex pair -> one integer. `stride` must exceed the vertex buffer size."""
    lo = torch.minimum(vi, vj).long()
    hi = torch.maximum(vi, vj).long()
    return lo * stride + hi


def _lookup(m, key, length, vi, vj, stride, myo_new, inherit, dev, dt_, new_val=None):
    """The keyed store on the mesh, mapped onto the half-edge arrays as they are NOW.

    One function with two callers -- `junction_myosin` and `junction_sync` -- so the myosin a
    frame is recorded with is by construction the myosin that frame's mechanics used. They
    cannot drift apart because only one piece of code decides it.

    Returns `(myo, n_new)` and touches nothing: the store is read, never written.

    `new_val` is what a junction with no history gets, per half-edge, overriding the scalar
    `myo_new`. A scalar is right only when the stored quantity has a fixed scale, and in the
    two-pool model it does not -- the tissue's mean line density drifts by a factor of about two
    over a run, so a newborn junction pinned at an absolute 1.0 would be set to an arbitrary
    fraction of what its neighbours happen to hold. Passing new_val = myo_new * n*_f, where
    n*_f = tau_jun * k_ex * rho_f is the density the supply into that cell's belt sustains, makes
    `myo_new` mean what its name says: a newborn junction as a FRACTION of what a mature one
    there would carry.
    """
    keys = m.get("myo_keys")
    vals = m.get("myo_vals")
    if keys is None or vals is None:
        keys = torch.empty(0, dtype=torch.long, device=dev)
        vals = torch.empty(0, dtype=dt_, device=dev)
    order = torch.argsort(keys)
    ks, vs = keys[order], vals[order]
    idx = torch.searchsorted(ks, key)
    idx_c = idx.clamp(max=max(ks.numel() - 1, 0))
    found = (ks.numel() > 0) & (idx_c < ks.numel())
    hit = found & (ks[idx_c] == key) if ks.numel() else torch.zeros_like(key, dtype=torch.bool)
    base = vs[idx_c] if vs.numel() else torch.zeros_like(length)
    fresh = (new_val if new_val is not None else torch.full_like(length, myo_new))
    myo = torch.where(hit, base, fresh)
    n_new = int((~hit).sum())

    # ---- A JUNCTION WITH A PARENT INHERITS FROM IT ------------------------------------------
    # Not every edge that misses the lookup is a new junction. `cell_divide` inserts a new vertex on
    # each of two edges of the dividing cell and then joins them, so a division produces two KINDS
    # of edge, and only one of them is new:
    #
    #   one endpoint new, one old   -- a SPLIT HALF of the cut junction (a,b). The same physical
    #                                  contact, now in two pieces. It has a myosin history and
    #                                  giving it `myo_new` throws that history away.
    #   both endpoints new          -- the interface between the two daughters. This one really is
    #                                  new, and `myo_new` is the honest answer for it.
    #
    # The parent is recoverable without any help from `cell_divide`, which is the point of keying by
    # vertex pair: a new vertex `n` has exactly two OLD neighbours, and they are the endpoints of the
    # edge it was inserted into. So the parent key is (min, max) over `n`'s old neighbours, and both
    # halves look it up. Falls back to `myo_new` where there is no parent to find.
    vseen = m.get("myo_vseen")
    if inherit and n_new and vseen is not None:
        oi = vseen[vi.clamp(max=vseen.numel() - 1)] & (vi < vseen.numel())
        oj = vseen[vj.clamp(max=vseen.numel() - 1)] & (vj < vseen.numel())
        half = (~hit) & (oi ^ oj)                       # exactly one endpoint is new
        if bool(half.any()):
            newv = torch.where(oi[half], vj[half], vi[half])     # the inserted vertex
            oldv = torch.where(oi[half], vi[half], vj[half])     # its old neighbour
            uq, inv = torch.unique(newv, return_inverse=True)
            lo = torch.full((uq.numel(),), stride, dtype=torch.long, device=dev)
            hi_ = torch.zeros(uq.numel(), dtype=torch.long, device=dev)
            lo = lo.scatter_reduce(0, inv, oldv, reduce="amin", include_self=True)
            hi_ = hi_.scatter_reduce(0, inv, oldv, reduce="amax", include_self=True)
            pkey = lo * stride + hi_                             # the edge the vertex was cut into
            pidx = torch.searchsorted(ks, pkey).clamp(max=max(ks.numel() - 1, 0))
            phit = (ks.numel() > 0) & (ks[pidx] == pkey) & (lo < hi_)
            # A HALF THAT CANNOT FIND ITS PARENT falls back to the same value a junction with no
            # history gets, taken per-edge so it follows `new_val` when one is supplied.
            fb = fresh[half]
            fb_u = torch.zeros(uq.numel(), device=dev, dtype=dt_).scatter_reduce(
                0, inv, fb, reduce="amax", include_self=False)
            inh = torch.where(phit, vs[pidx] if vs.numel() else torch.zeros_like(pkey, dtype=dt_),
                              fb_u)
            myo = myo.masked_scatter(half, inh[inv])
            n_inherited = int(phit[inv].sum())
            INHERIT_TRACE.append((n_new, int(half.sum()), n_inherited))
    return myo, n_new


def edge_tension(m, length, myo, ef, lam, k_perim, gam, dev, dt_):
    """Per-edge tension, as dE/dl_e of the energy `cell_mechanics` actually minimises.

    E = K_A(A-A0)^2 + K_P(P-P0)^2 + 0.5 Gam P^2 + K_V(...)^2 + Lambda*sum(m_e l_e), and only the
    perimeter and line terms depend on an individual edge length, so

        T_e = Lambda*m_e + 2 K_P (P_f - P0_f) + Gam * P_f      (f = the edge's own face)

    The CURRENT perimeter is not stored on the mesh -- `face_geometry_3d` computes and discards it --
    so it is recomputed here by the same index_add. P0 is on the mesh; K_P, Gam and Lambda come from
    the spec and MUST match the values `cell_mechanics` was given, or this is the tension of a
    different tissue.

    Raises rather than falling back. Falling back to `length` would make a tension-keyed run a
    LENGTH experiment wearing a tension label, and length is the exact variable such a run exists
    to replace -- a silent fallback to the hypothesis under test is the worst available failure
    mode.

    A module function rather than a method, because the medioapical operators need the same
    expression: the tension a junction is under is a property of the mesh and the energy, not of
    whichever operator happens to be asking. Two copies of it would be two tensions.
    """
    nF = int(m["nF"])
    P0 = m.get("P0")
    if P0 is None:
        raise RuntimeError(
            "edge_tension: this needs the target perimeter P0 on the mesh and it is absent. Refusing "
            "to fall back to length -- that is the variable a tension-keyed run exists to replace, and "
            "the fallback would have produced a length-keyed run labelled as tension.")
    perim = torch.zeros(nF, device=dev, dtype=dt_).index_add(0, ef, length)
    f = ef.long()
    T = lam * myo + 2.0 * k_perim * (perim - P0.to(dt_))[f] + gam * perim[f]
    return T.clamp_min(0.0)


def _scatter_full(es, live, myo, dev, dt_, fill=1.0):
    """Live values -> a per-half-edge array sized to the CURRENT buffer, `fill` on the dead slots.

    `fill` is 1.0 for a MULTIPLIER (a dead half-edge then contributes its usual tension if anything
    ever reads it unmasked) and 0.0 for an AMOUNT or a flux (a dead half-edge holds no myosin, and
    filling it with 1.0 would put the reservoir into the conservation ledger).
    """
    full = torch.full((es.shape[0],), float(fill), device=dev, dtype=dt_)
    full[live] = myo
    return full


def _live_edges(m, pos):
    """The live half-edges and their identities, for whoever is asking this frame.

    `live` is `E_face < nF`, the same mask `cell_mechanics` uses, NOT `E_srce < Nv`. The two differ
    after a division and using the wrong one is how a half-edge ends up with a myosin belonging to a
    face that no longer exists.
    """
    es, et, ef = m["E_srce"], m["E_trgt"], m["E_face"]
    nF, Nv = int(m["nF"]), int(m["Nv"])
    live = ef < nF
    vi, vj = es[live].long(), et[live].long()
    stride = max(pos.shape[0], Nv) + 1
    length = (pos[vj] - pos[vi]).norm(dim=-1).to(pos.dtype)
    return es, live, vi, vj, stride, _edge_key(vi, vj, stride), length


@register_operator("junction_myosin", family="mechanics", set="vertex", kind="structural",
                   equation=r"""$$m^{\mathrm{ss}}=a\,\frac{\ell_e}{\ell_{\mathrm{ref}}},\qquad \frac{dm}{dt}=\frac{m^{\mathrm{ss}}-m}{\tau}$$""", title="recruit myosin to stretched junctions")
class JunctionMyosin(Structural):
    r"""per-junction myosin recruited by tension: a stretched junction recruits more, pulls harder and so shrinks, the positive mechanical feedback a constant line tension cannot express; keyed by vertex pair so it survives T1, division and death.
    vertex -> vertex: reads pos and the keyed myosin store on the mesh, writes m["myo"] in place (the multiplier cell_mechanics puts on Lambda; myo = 1 the uninhibited baseline).

        m^{\mathrm{ss}}=a\,\frac{\ell_e}{\ell_{\mathrm{ref}}},\qquad \frac{dm}{dt}=\frac{m^{\mathrm{ss}}-m}{\tau}
        m^{ss}         the myosin set point of a junction
        a              `activity`, the global myosin level, dimensionless (the in-silico blebbistatin, acting over tau)
        \ell_e         the junction's current length
        \ell_{ref}     the current mean live edge length (not a constant: a growing tissue would otherwise read every junction as stretched)
        m              the junction's myosin multiplier
        \tau           `tau`, the relaxation time, frames

    Reference: Rauzi, M. et al. (2008). Nat. Cell Biol. 10:1401-1410 (myosin on shrinking junctions); Fernandez-Gonzalez, R. et al. (2009). Dev. Cell 17:736-743 (myosin II dynamics regulated by tension)."""
    PARAM_UNITS = {'activity': 'dimensionless', 'tau': 'frames', 'myo_new': 'dimensionless', 'dt': 'frames'}
    SYMBOLS = {'m^{ss}': 'the myosin set point of a junction', 'a': '`activity`, the global myosin level, dimensionless (the in-silico blebbistatin, acting over tau)', '\\ell_e': "the junction's current length", '\\ell_{ref}': 'the current mean live edge length (not a constant: a growing tissue would otherwise read every junction as stretched)', 'm': "the junction's myosin multiplier", '\\tau': '`tau`, the relaxation time, frames'}

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = False        # writes m["myo"], not positions
    MECHANISM_TAGS = ["actomyosin_contraction", "mechanosensitive_recruitment",
                      "junction_state", "topology_persistent"]
    PARAM_ROLES = {"activity": "global_myosin_activity", "tau": "recruitment_timescale",
                   "myo_new": "myosin_of_a_newborn_junction", "beta": "tension_sensitivity",
                   "activity_from": "per-type cell property giving each junction its drive", 'cell_set': 'the_cell_set', 'destabilising': 'whether_the_feedback_destabilises', 'dt': 'frames_per_call', 'gam': 'perimeter_term_scale', 'inherit': 'whether_a_split_junction_inherits_its_mothers_myosin', 'k_perim': 'perimeter_stiffness_scale', 'keyed_on': 'what_the_store_is_keyed_on', 'lam': 'line_tension_scale'}
    REFERENCE = ("Rauzi, M. et al. (2008). Nature and function of lateral tension during tissue "
                 "elongation. Nat. Cell Biol. 10:1401-1410; Fernandez-Gonzalez, R. et al. (2009). "
                 "Myosin II dynamics are regulated by tension in intercalating cells. Dev. Cell "
                 "17:736-743.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.activity = float(params.get("activity", 1.0))
        self.tau = float(params.get("tau", 20.0))
        self.beta = float(params.get("beta", 1.0))
        self.myo_new = float(params.get("myo_new", 1.0))
        # "length" reproduces every run up to 83 bit-for-bit; "tension" and "strain_rate" are the two
        # the literature supports. `destabilising` flips the sign so high drive -> more myosin -> higher
        # drive, which is the positive feedback that produces T1s rather than suppressing them.
        self.keyed_on = str(params.get("keyed_on", "length")).lower()
        # HETEROGENEITY, THROUGH THE TYPES. `activity` is one scalar over the whole tissue, so nothing
        # in a spec could say "these cells contract harder" -- and two operator instances cannot say it
        # either, because each rewrites the WHOLE keyed store every frame (`m["myo_keys"] = key`), so
        # the second would erase the first. `mask` does not help: the selector is per NODE of `at:`,
        # i.e. per vertex, while myosin lives per junction.
        #
        # `activity_from: <prop>` names a PER-TYPE property of the cell set, which is how every other
        # heterogeneous spec in this corpus says the same thing (`types: {t0: {fraction, p}, ...}`).
        # Each half-edge takes the value of the cell that owns it, and the junction takes the MEAN of
        # its two -- a junctional belt is fed from the cortices on both sides, and averaging is the
        # only choice symmetric in them, which matters because the store keeps ONE value per
        # undirected edge and would otherwise depend on which half-edge was written last.
        self.act_prop = params.get("activity_from")
        self.cell_set = params.get("cell_set", "cell")
        self.destabilising = bool(params.get("destabilising", True))
        self._prev_len = None
        self.lam = float(params.get("lam", 1.0))          # Lambda, for the tension expression
        self.k_perim = float(params.get("k_perim", 1.0))
        self.gam = float(params.get("gam", 0.0))
        self.inherit = bool(params.get("inherit", True))
        self.dt = float(params.get("dt", 1.0))
        # THE STORE IS ON THE MESH (`myo_keys` / `myo_vals` / `myo_vseen`), not here. See the module
        # docstring: state of an edge-set has to be reachable by any operator scheduled after a
        # topology change, and an operator attribute is reachable by exactly one.
        self._said = False

    def _edge_tension(self, m, length, myo, live, ef, dev, dt_):
        return edge_tension(m, length, myo, ef, self.lam, self.k_perim, self.gam, dev, dt_)

    def _edge_strain_rate(self, key, length):
        """d ln(l_e)/dt, matched to the PREVIOUS frame by the same vertex-pair key the myosin uses.

        Cheaper than tension and closer to Gustafson et al. 2022, whose recruitment variable is strain
        rate. Keyed rather than positional because cell_divide and edge_flip permute the half-edge
        arrays every frame -- comparing arrays by index would silently difference unrelated junctions.
        """
        cur = torch.stack([key, length])
        if self._prev_len is None:
            self._prev_len = cur
            return torch.zeros_like(length)
        pk, pl = self._prev_len
        order = torch.argsort(pk)
        ks, vs = pk[order], pl[order]
        idx = torch.searchsorted(ks, key).clamp(max=max(ks.numel() - 1, 0))
        hit = (ks.numel() > 0) & (ks[idx] == key)
        prev = torch.where(hit, vs[idx], length)
        self._prev_len = cur
        return torch.log(length.clamp_min(1e-9) / prev.clamp_min(1e-9))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        pos = lvl.get("pos")
        ef = m["E_face"]
        nF = int(m["nF"])
        dev, dt_ = pos.device, pos.dtype
        es, live, vi, vj, stride, key, length = _live_edges(m, pos)
        if not bool(live.any()):
            return {}
        l_ref = length.mean().clamp_min(1e-9)

        # ---- look each live junction up by its identity ------------------------------------------
        myo, n_new = _lookup(m, key, length, vi, vj, stride, self.myo_new, self.inherit, dev, dt_)

        # ---- recruitment ------------------------------------------------------------------------
        # ---- WHAT THE FEEDBACK IS KEYED TO ------------------------------------------------------
        # `length` was the original choice and it is the wrong variable, in two separate ways.
        #
        #   WRONG VARIABLE. Edge tension in this energy is Lambda*m_e + sum_f 2 K_P (P_f - P_f^0); length
        #   does not appear in it, so a long edge is not necessarily a taut one. The docstring used to
        #   assert "setpoint rises with junction length, i.e. with tension", which is an identification,
        #   not a derivation, and it is false here.
        #
        #   WRONG SIGN. Keyed to length the feedback is longer -> more myosin -> shorter: a STABILISER
        #   that homogenises junction lengths. The measured relationship runs the other way -- Bertet
        #   et al. 2004 find myosin enriched in DISASSEMBLING junctions, Fernandez-Gonzalez et al. 2009
        #   find more myosin on linked edges "regardless of edge length" -- and the resulting feedback is
        #   POSITIVE, a mechanical instability that generates T1s. The modelling literature keys on
        #   tension or strain rate and never on length; Sknepnek et al. 2023 say so verbatim.
        #
        # And keyed to length the operator is not even independent evidence for itself: in the tau -> 0
        # limit (the biologically correct one, myosin FRAP t1/2 ~ 6 s) substituting the setpoint gives
        # Lambda*a*sum[(1-beta) l_e + beta l_e^2/<l>], i.e. a line tension plus a HARMONIC EDGE SPRING of
        # zero rest length. Lowering the coefficient of variation of edge lengths is what such a spring
        # does by definition, so "beta lowers CV at constant radius" is a tautology and cannot
        # discriminate a right law from a wrong one. The T1 rate can, and the two signs move it oppositely.
        if self.keyed_on == "tension":
            drive = self._edge_tension(m, length, myo, live, ef[live].long(), dev, dt_)
        elif self.keyed_on == "strain_rate":
            drive = self._edge_strain_rate(key, length)
        else:
            drive = length
        d_ref = drive.mean().clamp_min(1e-9) if self.keyed_on != "strain_rate" else torch.ones((), device=dev, dtype=dt_)
        # SIGN. Getting this backwards silently converts the experiment into its own control, so it is
        # written out rather than inferred:
        #   length-keyed, +   longer -> more myosin -> shorter.  Negative feedback ON LENGTH: homogenises.
        #   tension-keyed, +  tauter -> more myosin -> tauter.   POSITIVE feedback ON TENSION: this is the
        #                     germband-extension instability, and it should RAISE the T1 rate.
        #   tension-keyed, -  tension homeostasis, the stabilising variant, kept as the contrast.
        # So the destabilising choice is +1, not -1. I had it as -1 on the first pass, which would have
        # made every "tension" run a stabiliser and the whole 84-91 comparison vacuous.
        sgn = 1.0 if self.destabilising else -1.0
        act = self.activity
        if self.act_prop:
            cl = H.level(self.cell_set)
            a_cell = getattr(cl, self.act_prop, None)
            if a_cell is None:
                raise ValueError(
                    f"junction_myosin activity_from={self.act_prop!r}: set {self.cell_set!r} has no "
                    f"per-node buffer of that name -- declare it as a per-type scalar under "
                    f"`sets.{self.cell_set}.types`.")
            a_he = a_cell[ef[live].long()].to(dt_)          # per half-edge, from the cell that owns it
            uk, inv = torch.unique(key, return_inverse=True)
            tot = torch.zeros(int(uk.numel()), device=dev, dtype=dt_).index_add_(0, inv, a_he)
            cnt = torch.zeros(int(uk.numel()), device=dev, dtype=dt_).index_add_(
                0, inv, torch.ones_like(a_he))
            act = self.activity * (tot / cnt.clamp_min(1.0))[inv]
        ss = act * (1.0 + sgn * self.beta * (drive / d_ref - 1.0)).clamp_min(0.0)
        myo = myo + (ss - myo) * (self.dt / max(self.tau, 1e-9))
        myo = myo.clamp(0.0, 5.0)

        # ---- store back, keyed. Junctions that no longer exist are simply not written, so a T1 or a
        # death drops them without anyone having to notice.
        m["myo_keys"], m["myo_vals"] = key.detach().clone(), myo.detach().clone()
        # which vertices existed this frame, so next frame can tell an inserted vertex from an old one
        vs_seen = torch.zeros(stride, dtype=torch.bool, device=dev)
        vs_seen[vi] = True; vs_seen[vj] = True
        m["myo_vseen"] = vs_seen
        # the per-half-edge array the mechanics reads: full length, 1.0 on dead slots so a masked-out
        # half-edge contributes its usual tension if anything ever reads it unmasked
        m["myo"] = _scatter_full(es, live, myo, dev, dt_)
        MYOSIN_TRACE.append((int(live.sum()), float(myo.mean()), float(myo.min()),
                             float(myo.max()), n_new))
        if not self._said:
            if self.act_prop:
                _u = torch.unique(act)
                print(f"[junction_myosin] activity_from={self.act_prop!r} on set {self.cell_set!r}: "
                      f"{int(_u.numel())} distinct drives across {int(live.sum()):,} junctions, "
                      f"{[round(float(v), 3) for v in _u[:6]]}"
                      f"{' ...' if _u.numel() > 6 else ''} (a boundary junction gets the mean of its "
                      f"two cells)", flush=True)
            print(f"[junction_myosin] {int(live.sum())} live junctions, activity={self.activity}, "
                  f"tau={self.tau}, myo_new={self.myo_new}; keyed by vertex pair so T1 / division / "
                  f"death need no edits", flush=True)
            self._said = True
        return {}


# Per frame, only when inheritance fires: (new edges, split halves, halves that found a parent).
INHERIT_TRACE: list = []

# Per frame: (half-edges now, half-edges when myosin was written, live junctions re-keyed). A run
# whose second column never differs from the first is a run in which no topology operator changed
# the edge buffer at all, which in a growing tissue is itself a defect worth noticing.
SYNC_TRACE: list = []


@register_operator("junction_sync", family="mechanics", set="vertex", kind="rewire",
                   equation=r"""$$m_e=\text{gain}\cdot\mathrm{store}\big[\mathrm{key}(v_i,v_j)\big],\qquad N_e=\mathrm{store}\big[\mathrm{key}(v_i,v_j)\big]\,\ell_e$$""", title="re-index junction myosin after rewiring")
class JunctionMyosinSync(Rewire):
    r"""re-keys the per-junction myosin onto the half-edge arrays a topology operator has just changed, so what is recorded for a frame is what that frame's mechanics actually used; bookkeeping that cannot change a trajectory.
    vertex -> vertex: reads pos and the keyed store, rewrites m["myo"] and m["myo_amount"]; scheduled after every operator that rewires the half-edges and before topo_record.

        m_e=\text{gain}\cdot\mathrm{store}\big[\mathrm{key}(v_i,v_j)\big],\qquad N_e=\mathrm{store}\big[\mathrm{key}(v_i,v_j)\big]\,\ell_e
        m_e            the myosin multiplier of half-edge e, re-read at the current indices
        gain           the global myosin gain
        store[key(v_i, v_j)] the keyed store, by unordered vertex pair
        N_e            the myosin amount of the junction, for the conservation ledger
        \ell_e         the junction's length

    Reference: none: bookkeeping that keeps a mechanism correct, not a mechanism. Plexus (this work)."""
    PARAM_UNITS = {'myo_new': 'dimensionless'}
    SYMBOLS = {'m_e': 'the myosin multiplier of half-edge e, re-read at the current indices', 'gain': 'the global myosin gain', 'store[key(v_i, v_j)]': 'the keyed store, by unordered vertex pair', 'N_e': 'the myosin amount of the junction, for the conservation ledger', '\\ell_e': "the junction's length"}

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = False
    MECHANISM_TAGS = ["junction_state", "topology_persistent", "bookkeeping"]
    PARAM_ROLES = {"myo_new": "myosin_of_a_newborn_junction", 'inherit': 'whether_a_split_junction_inherits_its_mothers_myosin'}
    REFERENCE = "Plexus (this work); bookkeeping that keeps a mechanism correct, not a mechanism."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.myo_new = float(params.get("myo_new", 1.0))
        self.inherit = bool(params.get("inherit", True))
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        if m is None or m.get("myo_keys") is None:
            return {}                      # no junction myosin in this specification: nothing to carry
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        es, live, vi, vj, stride, key, length = _live_edges(m, pos)
        if not bool(live.any()):
            return {}
        was = int(m["myo"].shape[0]) if m.get("myo") is not None else 0
        val, _ = _lookup(m, key, length, vi, vj, stride, self.myo_new, self.inherit, dev, dt_)
        # The store holds what the model chose to store; `myo` is what the mechanics reads. For
        # the default model those are the same number and `myo_gain` is 1. For `two_pool` the
        # store is a line density and the multiplier is that density scaled to a tissue mean of
        # `activity`, so the gain is activity / <n>. The owning operator leaves the gain on the
        # mesh and this operator applies it, which is what keeps this one model-agnostic. The
        # gain used is the one computed BEFORE the topology operators ran -- a mean over some
        # thousands of junctions, of which a handful were just cut, so it does not move at that
        # scale.
        gain = float(m.get("myo_gain", 1.0))
        m["myo"] = _scatter_full(es, live, (gain * val).clamp(0.0, 5.0), dev, dt_)
        # And the amount, on the same footing. `myo` is a density -- normalised, for `two_pool` --
        # so no sum of it says how much myosin there is; N_e = (stored density) * l_e does, and it
        # is the quantity the conservation ledger is written in. Zero on dead slots: an amount.
        m["myo_amount"] = _scatter_full(es, live, val * length, dev, dt_, fill=0.0)
        SYNC_TRACE.append((int(es.shape[0]), was, int(live.sum())))
        if not self._said:
            print(f"[junction_sync] re-keying myosin after the topology operators "
                  f"({was} half-edges written -> {int(es.shape[0])} now)", flush=True)
            self._said = True
        return {}


POOL_TRACE: list = []


def _face_carry(m, name):
    """Ask the topology operators to carry a per-face array across a rebuild.

    `cell_divide` and `cell_die` reindex every per-face array through the `keep` map (new face ->
    old face), so nothing is left pointing at a face that has moved. The list of names they carry
    is open rather than literal: an operator declares its own array here once, and the topology
    operators still know nothing about what is in it. A closed list would silently drop the
    per-face state of any operator added later -- the same class of defect as the per-half-edge
    myosin that `junction_sync` exists to fix, one level up.
    """
    m.setdefault("face_carry", set()).add(name)


# Per frame: (newborn daughter--daughter interfaces seeded, myosin moved onto them). The second column
# is what the medioapical pool paid, so a ring that creates myosin instead of moving it shows up as a
# ledger that no longer closes.
RING_TRACE: list = []


# junction_myosin[type_pair] -- the line tension per pair of cell TYPES (experiment 9, Graner & Glazier 1992's
# J table on the vertex model). Moved here from sorting_ops.py on 2026-09-27, unchanged.
def type_pair_multiplier(es, et, ef, face_type, table, medium=None):
    """Per half-edge: table[type of its face, type of the face across], or table[type, medium] on a
    free edge (no twin). `table` [K+1, K+1] when `medium` is its last index, else [K, K]."""
    from plexus.operators.junction_ops import pcp_twins
    stride = int(max(int(es.max()), int(et.max()))) + 1 if es.numel() else 1
    tw = pcp_twins(es, et, stride)
    t_in = face_type[ef]
    has = tw >= 0
    t_out = torch.where(has, face_type[ef[tw.clamp(min=0)]], torch.full_like(t_in, -1))
    if medium is None and not bool(has.all()):
        raise ValueError("junction_myosin[type_pair]: the mesh has free edges and the table no `medium` entry")
    t_out = torch.where(has, t_out, torch.full_like(t_in, medium if medium is not None else 0))
    return table[t_in, t_out]


@register_operator("junction_myosin", model="type_pair", family="mechanics", set="vertex", kind="structural",
                   title="set junction tension by cell types",
                   equation=r"""$$m_e=\gamma_{\tau(f_e)\,\tau(f'_e)},\qquad m_e^{\mathrm{free}}=\gamma_{\tau(f_e)\,M}$$""")
class JunctionMyosinTypePair(Structural):
    r"""the vertex model's differential adhesion: each junction's line tension set by the types of the two cells it separates, and by the cell's type and the medium on a free edge; recomputed every call, no store.
    vertex -> vertex: reads the cell set's node_type, writes m["myo"] (one multiplier per half-edge) for this frame's topology.

        m_e=\gamma_{\tau(f_e)\,\tau(f'_e)},\qquad m_e^{\mathrm{free}}=\gamma_{\tau(f_e)\,M}
        m_e            the multiplier on Lambda for half-edge e
        \gamma         the symmetric table `tensions` keyed by type pair (and `medium`)
        \tau(f_e)      the type of the cell owning half-edge e
        \tau(f'_e)     the type of the cell across
        m_e^{free}     the multiplier of an edge with no cell across
        M              the medium

    Reference: Graner, F. & Glazier, J. A. (1992). Phys. Rev. Lett. 69:2013-2016; Steinberg, M. S. (1963). Science 141:401-408; Farhadifar, R. et al. (2007). Curr. Biol. 17:2095-2104 (the line tension of the vertex model)."""
    PARAM_UNITS = {'frames': 'frames', 'noise': 'fraction', 'noise_end': 'frames', 'seed': 'count', 'tau': 'frames'}
    SYMBOLS = {'m_e': 'the multiplier on Lambda for half-edge e', '\\gamma': 'the symmetric table `tensions` keyed by type pair (and `medium`)', '\\tau(f_e)': 'the type of the cell owning half-edge e', "\\tau(f'_e)": 'the type of the cell across', 'm_e^{free}': 'the multiplier of an edge with no cell across', 'M': 'the medium'}
    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = False             # writes m["myo"], not positions
    REQUIRES_PARAMS = ["tensions"]
    MECHANISM_TAGS = ["differential_adhesion", "cell_sorting", "type_pair_tension"]
    PARAM_ROLES = {"tensions": "line_tension_multiplier_per_pair_of_types_and_medium",
                   "noise": "active_cortical_fluctuation_amplitude_in_W", "tau": "fluctuation_correlation_frames",
                   "noise_end": "amplitude_at_the_end_of_the_ramp", "frames": "ramp_length_calls",
                   "seed": "rng_seed", 'cell_set': 'the_cell_set'}
    REFERENCE = ("Graner, F. & Glazier, J.A. (1992). Phys. Rev. Lett. 69:2013-2016; Steinberg, M.S. "
                 "(1963). Science 141:401-408.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self._cat = params.get("cell_set")
        self.spec = params["tensions"]
        self._tab = None
        self._medium = None
        # ACTIVE CORTICAL FLUCTUATIONS (experiment 9, Phase 2, Finding 20). A deterministic vertex
        # aggregate only descends to the nearest local minimum: two halves never engulf and a labyrinth
        # never rounds. Cells in an aggregate fluctuate their cortex; here each CELL f carries an
        # Ornstein-Uhlenbeck variable eta_f (unit variance, correlation time `tau` frames) and a junction
        # between f and g adds `noise` x (eta_f + eta_g) / sqrt(2) to its multiplier (a rim junction,
        # `noise` x eta_f). `noise` is in the table's own units (a W); 0, the default, is the table alone.
        self.noise = float(params.get("noise", 0.0))
        # `noise_end` and `frames`: the amplitude cooled linearly from `noise` to `noise_end` over
        # `frames` calls, then held (an annealing schedule, Finding 23); absent, the amplitude is constant.
        self.noise0 = self.noise
        self.noise_end = float(params.get("noise_end", self.noise))
        self.frames = max(1, int(params.get("frames", 1)))
        self._calls = 0
        self.tau = float(params.get("tau", 10.0))
        self.seed = int(params.get("seed", 0))
        self._eta = None
        self._gen = None

    def _table(self, clvl, device, dtype):
        names = list(getattr(clvl, "type_names", None) or [])
        want_m = any("medium" in (row or {}) for row in self.spec.values())
        K = len(names) + (1 if want_m else 0)
        idx = {n: i for i, n in enumerate(names)}
        if want_m:
            idx["medium"] = len(names)
        tab = torch.full((K, K), float("nan"), device=device, dtype=dtype)
        for a, row in self.spec.items():
            for b, g in (row or {}).items():
                if a not in idx or b not in idx:
                    raise ValueError(f"junction_myosin[type_pair]: {a!r}/{b!r} is not one of {list(idx)}")
                tab[idx[a], idx[b]] = tab[idx[b], idx[a]] = float(g)
        miss = [(n, m) for n in names for m in list(idx) if torch.isnan(tab[idx[n], idx[m]])]
        if miss:
            raise ValueError(f"junction_myosin[type_pair]: no tension for {miss}")
        return tab, (idx["medium"] if want_m else None)

    def forward(self, H, mask=None):
        from plexus.operators.vertex_ops import resolve_cell_set
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        clvl = H.level(resolve_cell_set(H, self.at, self._cat))
        es, et, ef = m["E_srce"].long(), m["E_trgt"].long(), m["E_face"].long()
        pos = lvl.get("pos")
        if self._tab is None or self._tab.device != pos.device:
            self._tab, self._medium = self._table(clvl, pos.device, pos.dtype)
        nF = int(m["nF"])
        ft = clvl.node_type[:nF].to(pos.device).long()
        myo = type_pair_multiplier(es, et, ef, ft, self._tab, self._medium)
        self.noise = self.noise0 + (self.noise_end - self.noise0) * min(1.0, self._calls / self.frames)
        self._calls += 1
        if self.noise0 > 0.0 or self.noise_end > 0.0:
            from plexus.operators.junction_ops import pcp_twins
            n_buf = clvl.node_type.shape[0]
            if self._eta is None:
                self._gen = torch.Generator(device="cpu").manual_seed(self.seed)
                self._eta = torch.randn(n_buf, generator=self._gen, dtype=torch.float64)
            h = float(getattr(H, "dt", 1.0))
            a = float(np.exp(-h / max(self.tau, 1e-9)))
            self._eta = a * self._eta + float(np.sqrt(1.0 - a * a)) * torch.randn(n_buf, generator=self._gen, dtype=torch.float64)
            eta = self._eta.to(device=pos.device, dtype=myo.dtype)
            stride = int(max(int(es.max()), int(et.max()))) + 1 if es.numel() else 1
            tw = pcp_twins(es, et, stride)
            has = tw >= 0
            other = torch.where(has, eta[ef[tw.clamp(min=0)]], torch.zeros_like(eta[ef]))
            myo = myo + self.noise * torch.where(has, (eta[ef] + other) / float(np.sqrt(2.0)), eta[ef])
        m["myo"] = myo
        return {}


def rest_length_step(L, L0, dt, tau, kappa, m_max):
    """One step of a junction's viscoelastic rest length, and the line-tension multiplier it gives.

        L0'  = L0 + (L - L0) dt / tau               the rest length remodels toward the length
        m    = clip(1 + kappa (L - L0') / L0', 0, m_max)

    A module function so the model and its test share one expression."""
    L0n = L0 + (L - L0) * (dt / max(tau, 1e-9))
    m = (1.0 + kappa * (L - L0n) / L0n.clamp_min(1e-12)).clamp(0.0, m_max)
    return L0n, m


@register_operator("junction_myosin", model="rest_length", family="mechanics", set="vertex", kind="structural",
                   title="remodel junction rest lengths",
                   equation=r"""$$T_e=\Lambda\Big(1+\kappa\,\frac{\ell_e-\ell^0_e}{\ell^0_e}\Big),\qquad \frac{d\ell^0_e}{dt}=\frac{\ell_e-\ell^0_e}{\tau}$$""")
class JunctionRestLength(Structural):
    r"""each junction an elastic element with a rest length that remodels toward its length over tau: a junction stretched faster than it remodels pulls harder, one compressed faster pulls less.
    vertex -> vertex: reads pos, keeps each junction's rest length in its own keyed store (`rl_keys` / `rl_vals`), writes m["myo"], the multiplier cell_mechanics puts on Lambda (clipped to [0, m_max]); scheduled before cell_mechanics.

        T_e=\Lambda\Big(1+\kappa\,\frac{\ell_e-\ell^0_e}{\ell^0_e}\Big),\qquad \frac{d\ell^0_e}{dt}=\frac{\ell_e-\ell^0_e}{\tau}
        T_e            the junction's tension
        \Lambda        the vertex model's line tension
        \kappa         `kappa`, the junction's stiffness over Lambda, dimensionless
        \ell_e         the junction's length
        \ell^0_e       its rest length (a junction with no history starts at rest)
        \tau           `tau`, the remodelling time, frames

    Reference: Staddon, M. F. et al. (2019). Mechanosensitive junction remodeling promotes robust epithelial morphogenesis. Biophys. J. 117:1739-1750; Noll, N. et al. (2017). Active tension network model suggests an exotic mechanical state realized in epithelial tissues. Nat. Phys. 13:1221-1226."""
    PARAM_UNITS = {'kappa': 'dimensionless', 'tau': 'frames', 'tau_m': 'frames', 'start': 'frames', 'm_max': 'dimensionless', 'dt': 'frames'}
    SYMBOLS = {'T_e': "the junction's tension", '\\Lambda': "the vertex model's line tension", '\\kappa': "`kappa`, the junction's stiffness over Lambda, dimensionless", '\\ell_e': "the junction's length", '\\ell^0_e': 'its rest length (a junction with no history starts at rest)', '\\tau': '`tau`, the remodelling time, frames'}

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = False
    MECHANISM_TAGS = ["junction_state", "viscoelastic", "rest_length_remodeling", "topology_persistent"]
    PARAM_ROLES = {"kappa": "junction_stiffness_over_line_tension", "tau": "rest_length_remodeling_time",
                   "m_max": "multiplier_ceiling", "start": "first_frame_of_remodelling",
                   "tau_m": "multiplier_relaxation_time", "advance": "false = recompute m only, for the record", 'dt': 'frames_per_call'}
    REFERENCE = ("Staddon, M.F. et al. (2019). Biophys. J. 117:1739-1750; Noll, N. et al. (2017). "
                 "Nat. Phys. 13:1221-1226.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.kappa = float(params.get("kappa", 4.0))
        self.tau = float(params.get("tau", 150.0))
        self.m_max = float(params.get("m_max", 5.0))
        self.dt = float(params.get("dt", 1.0))
        self.start = int(params.get("start", 0))
        self.tau_m = float(params.get("tau_m", 0.0))
        self.advance = bool(params.get("advance", True))
        self._k = 0
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        es, live, vi, vj, stride, key, length = _live_edges(m, pos)
        if not bool(live.any()):
            return {}
        keys, vals = m.get("rl_keys"), m.get("rl_vals")
        if not self.advance:
            # READ-ONLY: m from the stored l0 (and the stored damped m) on the half-edges as they are now
            mult = torch.ones_like(length)
            if keys is not None and keys.numel():
                order = torch.argsort(keys)
                ks = keys[order]
                idx = torch.searchsorted(ks, key).clamp(max=ks.numel() - 1)
                hit = ks[idx] == key
                mv = m.get("rl_mvals")
                if mv is not None and mv.numel() == keys.numel():
                    mult = torch.where(hit, mv[order].to(dt_)[idx], mult)
            m["myo"] = _scatter_full(es, live, mult, dev, dt_)
            return {}
        self._k += 1
        L0 = length.clone()
        m_prev = torch.ones_like(length)
        if keys is not None and keys.numel() and self._k > self.start:
            order = torch.argsort(keys)
            ks, vs = keys[order], vals[order].to(dt_)
            idx = torch.searchsorted(ks, key).clamp(max=ks.numel() - 1)
            hit = ks[idx] == key
            L0 = torch.where(hit, vs[idx], length)
            mv = m.get("rl_mvals")
            if mv is not None and mv.numel() == keys.numel():
                m_prev = torch.where(hit, mv[order].to(dt_)[idx], m_prev)
        L0n, mult = rest_length_step(length, L0, self.dt, self.tau, self.kappa, self.m_max)
        if self.tau_m > 0:
            mult = m_prev + (mult - m_prev) * min(self.dt / self.tau_m, 1.0)
        m["rl_keys"], m["rl_vals"], m["rl_mvals"] = key.detach().clone(), L0n.detach().clone(), mult.detach().clone()
        # for cell_mechanics[junction_spring]: NO spring before `start`. The store is written from the first
        # frame (l0 = l), and a spring at a rest length reset every frame to the current length is a drag on
        # every junction's change within the relaxation; batch 18 ran with it (all four arms wrecked).
        m["rl_kappa"], m["rl_stride"] = (self.kappa if self._k > self.start else 0.0), int(stride)
        m["myo"] = _scatter_full(es, live, mult, dev, dt_)
        MYOSIN_TRACE.append((int(live.sum()), float(mult.mean()), float(mult.min()), float(mult.max()), 0))
        if not self._said:
            print(f"[junction_myosin/rest_length] {int(live.sum())} live half-edges, kappa={self.kappa}, "
                  f"tau={self.tau}, m_max={self.m_max}; rest lengths keyed by vertex pair", flush=True)
            self._said = True
        return {}


# ================================================================================================ #
#  junction_pcp -- planar cell polarity on the junction sides
# ================================================================================================ #
PCP_TRACE: list = []


def pcp_twins(es, et, stride):
    """Index of each half-edge's twin (t, s), or -1 on the free boundary."""
    key = es.long() * stride + et.long()
    tw = et.long() * stride + es.long()
    order = torch.argsort(key)
    ks = key[order]
    j = torch.searchsorted(ks, tw).clamp(max=max(ks.numel() - 1, 0))
    hit = ks[j] == tw
    return torch.where(hit, order[j], torch.full_like(j, -1))


def pcp_side_pairs(ef, ang, perim, nF, spread):
    """The within-cell kernel of `junction_pcp`: pairs (a, b) of sides of one cell and their weights.

    w_ab = exp(-d_ab / (spread * s_i)), d_ab the distance between the two sides' midpoints ALONG the
    perimeter, taken as (P_i / 2 pi) times their angular separation about the centroid (exact for a
    circle, close for a convex cell), s_i = P_i / n_i the cell's mean side length. Returns
    (a, b, w) with the self pairs (w = 1) included; `spread` = 0 returns only the self pairs.
    """
    E = ef.shape[0]
    idx = torch.arange(E, device=ef.device)
    if spread <= 0.0:
        return idx, idx, torch.ones(E, dtype=ang.dtype, device=ef.device)
    order = torch.argsort(ef, stable=True)
    fs = ef[order]
    deg = torch.bincount(ef, minlength=nF)
    start = torch.cumsum(deg, 0) - deg
    kmax = int(deg.max())
    a_l, b_l = [], []
    for k in range(kmax):
        ok = deg[fs] > k
        j = start[fs] + k
        a_l.append(order[ok]); b_l.append(order[j[ok]])
    a, b = torch.cat(a_l), torch.cat(b_l)
    dphi = torch.remainder(ang[a] - ang[b] + torch.pi, 2 * torch.pi) - torch.pi
    f = ef[a]
    d = perim[f] / (2 * torch.pi) * dphi.abs()
    side = perim[f] / deg[f].clamp_min(1).to(perim.dtype)
    return a, b, torch.exp(-d / (spread * side).clamp_min(1e-12))


def pcp_rates(F, V, twin, ef, L, nF, Ftot, Vtot, A, k_on, k_off, g, beta, Kb, Kp, kern=None, AV=None,
              hill=None):
    """dF/dt and dV/dt per half-edge for `junction_pcp` (its docstring has the equations).

    F, V, L, A: [E] line densities, lengths and the cue factor of the live half-edges; twin: [E]
    index into the same arrays or -1; ef: [E] face; Ftot, Vtot: [nF] each cell's conserved amounts;
    kern: (a, b, w) from `pcp_side_pairs`, or None for the side-local inhibition; AV: [E] a factor on
    V's unbinding (the elongation coupling), None for 1; hill: (n, K) makes the recruitment by the
    partner across the junction cooperative, x -> (1 + K^n) x^n / (K^n + x^n) (equal to x at x = 1),
    None for linear (`junction_pcp[cooperative]`); (n, K, p) with p [E] in [0, 1] blends the two per
    side, (1 - p) x + p T(x) (`junction_pcp[primed]`).
    A module function so the two-cell test calls exactly what the operator integrates.
    """
    perim = torch.zeros(nF, dtype=F.dtype, device=F.device).index_add(0, ef, L).clamp_min(1e-12)
    Fb = torch.zeros_like(perim).index_add(0, ef, F * L)
    Vb = torch.zeros_like(perim).index_add(0, ef, V * L)
    cF = (Ftot - Fb).clamp_min(0.0) / perim                     # the free pool, per unit perimeter
    cV = (Vtot - Vb).clamp_min(0.0) / perim
    has = twin >= 0
    tw = twin.clamp_min(0)
    Vn = torch.where(has, V[tw], torch.zeros_like(V))           # the neighbour's side of the junction
    Fn = torch.where(has, F[tw], torch.zeros_like(F))
    if kern is None:
        Vs = V
    else:                                                       # Vang nearby along the membrane
        a, b, w = kern
        num = torch.zeros_like(V).index_add(0, a, w * V[b] * L[b])
        den = torch.zeros_like(V).index_add(0, a, w * L[b]).clamp_min(1e-12)
        Vs = num / den
    B = 1.0 + Kb * Vs.clamp_min(0.0) ** Kp                      # Vang on and near this side blocks Fz binding
    if hill is not None:                                        # cooperative recruitment across the junction
        n_, K_ = hill[0], hill[1]
        Tv = (1 + K_ ** n_) * Vn.clamp_min(0.0) ** n_ / (K_ ** n_ + Vn.clamp_min(0.0) ** n_)
        Tf = (1 + K_ ** n_) * Fn.clamp_min(0.0) ** n_ / (K_ ** n_ + Fn.clamp_min(0.0) ** n_)
        if len(hill) > 2:                                       # a per-side weight p: (1 - p) x + p T(x)
            p_ = hill[2]
            Tv, Tf = (1 - p_) * Vn + p_ * Tv, (1 - p_) * Fn + p_ * Tf
        Vn, Fn = Tv, Tf
    dF = k_on * cF[ef] * (beta + g * Vn) - k_off * A * B * F
    dV = k_on * cV[ef] * (beta + g * Fn) - k_off * (V if AV is None else AV * V)
    return dF, dV


def pcp_celsr_q(P, vi, vj, ef, cen, I, nF):
    """Each cell's Celsr1-like nematic q_i (complex), Aigouy et al. 2010's angular integral of the border
    intensity I round the cell: q_i = sum_h I_h int_{phi_a}^{phi_b} exp(2 i phi) dphi / sum_h I_h (phi_b - phi_a),
    phi_a, phi_b the angles of side h's ends about the centroid -- the same readout as the exp08.celsr
    ruler (a uniform I reads 0 on any cell shape). Returns (Re q, Im q), each [nF]; the axis of the
    enriched borders' POSITIONS is half the angle of q.
    """
    ra, rb = P[vi] - cen[ef], P[vj] - cen[ef]
    pa, pb = torch.atan2(ra[:, 1], ra[:, 0]), torch.atan2(rb[:, 1], rb[:, 0])
    dphi = torch.remainder(pb - pa + math.pi, 2 * math.pi) - math.pi
    pb = pa + dphi
    re = torch.zeros(nF, dtype=P.dtype, device=P.device).index_add(0, ef, I * (torch.sin(2 * pb) - torch.sin(2 * pa)) / 2)
    im = torch.zeros(nF, dtype=P.dtype, device=P.device).index_add(0, ef, I * (torch.cos(2 * pa) - torch.cos(2 * pb)) / 2)
    den = torch.zeros(nF, dtype=P.dtype, device=P.device).index_add(0, ef, I * dphi)
    den = torch.where(den.abs() > 1e-30, den, torch.full_like(den, 1e-30))
    return re / den, im / den


def pcp_elongation(P, vi, vj, ef, nF):
    """Each cell's elongation (e1, e2) = sum over its sides of l^2 (cos 2 psi, sin 2 psi) / (2 A), psi
    the side's direction and A the cell's area (shoelace): Aigouy et al. 2010, Suppl. Eq. 25, written
    with one consistent factor. 0 for a regular polygon, positive e1 for a cell long along axis 0."""
    d = P[vj] - P[vi]
    l2 = (d ** 2).sum(1)
    psi = torch.atan2(d[:, 1], d[:, 0])
    A = 0.5 * torch.zeros(nF, dtype=P.dtype, device=P.device).index_add(
        0, ef, P[vi, 0] * P[vj, 1] - P[vj, 0] * P[vi, 1]).abs().clamp_min(1e-12)
    e1 = torch.zeros_like(A).index_add(0, ef, l2 * torch.cos(2 * psi)) / (2 * A)
    e2 = torch.zeros_like(A).index_add(0, ef, l2 * torch.sin(2 * psi)) / (2 * A)
    return e1, e2


@register_operator("junction_pcp", family="polarity", set="vertex", kind="lateral",
                   equation=r"""$$\frac{dF_h}{dt}=k_{\mathrm{on}}c^F_i(\beta+gV_{\bar h})-k_{\mathrm{off}}A_hB_hF_h,\quad \frac{dV_h}{dt}=k_{\mathrm{on}}c^V_i(\beta+gF_{\bar h})-k_{\mathrm{off}}V_h,\quad B_h=1+K_bV_h^{K_p}$$""", title="set up planar cell polarity")
class JunctionPCP(Lateral):
    r"""planar cell polarity on the half-edges: two junctional complexes (Fz, Vang) on each cell's side of each junction, exchanged across the junction with the neighbour's side, from a well-mixed free pool per cell.
    vertex -> vertex: reads pos and the half-edge table, writes m["fz"] and m["vang"] (line densities per half-edge, recorded as `e_fz` / `e_vang`) and, when the cell set declares them, `mutant`, `pcp_vec` per cell; explicit Euler, `substeps` steps of `dt` per call.

        \frac{dF_h}{dt}=k_{\mathrm{on}}c^F_i(\beta+gV_{\bar h})-k_{\mathrm{off}}A_hB_hF_h,\quad \frac{dV_h}{dt}=k_{\mathrm{on}}c^V_i(\beta+gF_{\bar h})-k_{\mathrm{off}}V_h,\quad B_h=1+K_bV_h^{K_p}
        F_h            the Fz(-Dsh) complex line density on half-edge h (cell i's side)
        V_h            the Vang(-Pk) complex line density there
        k_{on}         `k_on`, the binding rate from the free pool
        c^F_i, c^V_i   cell i's free pool of each complex per unit perimeter
        \beta          `beta`, the baseline recruitment
        g              `g`, the recruitment by the partner across the junction
        V_{\bar h}, F_{\bar h} the twin side's densities (cell j's side of the same junction)
        k_{off}        `k_off`, the unbinding rate
        A_h            the mutant / cue factor on side h
        B_h            the Vang-dependent destabilisation of Fz, 1 + K_b Vs_h^{K_p}, Vs_h the spread-weighted Vang on the cell's sides
        K_b, K_p       `K_b`, `K_p`, its strength and exponent

    Reference: Amonlirdviman, K., Khare, N. A., Tree, D. R. P., Chen, W.-S., Axelrod, J. D. & Tomlin, C. J. (2005). Mathematical modeling of planar cell polarity to understand domineering nonautonomy. Science 307:423-426, SOM; Burak, Y. & Shraiman, B. I. (2009). PLoS Comput. Biol. 5:e1000628 (the well-mixed pool)."""
    PARAM_UNITS = {'K_b': 'dimensionless', 'K_p': 'dimensionless', 'beta': 'dimensionless', 'g': 'dimensionless', 'k_on': 'rate', 'k_off': 'rate', 'dt': 'time', 'substeps': 'count', 'noise': 'fraction', 'init_bound': 'fraction', 'init_noise': 'fraction', 'seed': 'count', 'spread': 'dimensionless', 'elong': 'dimensionless', 'fz_total': 'dimensionless', 'vang_total': 'dimensionless', 'plane_axis': 'count', 'cue_cos': 'dimensionless', 'cue_m1': 'dimensionless', 'cue_until': 'frames'}
    SYMBOLS = {'F_h': "the Fz(-Dsh) complex line density on half-edge h (cell i's side)", 'V_h': 'the Vang(-Pk) complex line density there', 'k_{on}': '`k_on`, the binding rate from the free pool', 'c^F_i, c^V_i': "cell i's free pool of each complex per unit perimeter", '\\beta': '`beta`, the baseline recruitment', 'g': '`g`, the recruitment by the partner across the junction', 'V_{\\bar h}, F_{\\bar h}': "the twin side's densities (cell j's side of the same junction)", 'k_{off}': '`k_off`, the unbinding rate', 'A_h': 'the mutant / cue factor on side h', 'B_h': "the Vang-dependent destabilisation of Fz, 1 + K_b Vs_h^{K_p}, Vs_h the spread-weighted Vang on the cell's sides", 'K_b, K_p': '`K_b`, `K_p`, its strength and exponent'}

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    DIFFERENTIABLE = False
    # DERIVED READOUTS ON THE CELL SET: `mutant`, `pcp_vec` and the colour channels are written from
    # the half-edge state and integrated by nothing, which is what the flag is for (as for the
    # centroid aggregate). No position or velocity is touched; the dynamics live on the mesh table.
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["planar_cell_polarity", "junction_state", "contact_dependent_signalling",
                      "topology_persistent"]
    PARAM_ROLES = {"g": "exchange_gain_across_the_junction", "beta": "basal_recruitment",
                   "spread": "range_of_the_cis_inhibition_along_the_membrane_in_side_lengths",
                   "elong": "coupling_of_polarity_to_cell_elongation",
                   "elong_ref": "elongation_measured_from_the_seeded_shape_or_absolute",
                   "K_b": "cis_inhibition_strength", "K_p": "cis_inhibition_exponent",
                   "cue_m1": "distal_unbinding_factor_of_the_global_cue", "k_on": "binding_rate",
                   "k_off": "unbinding_rate", 'dt': 'model_time_per_substep', 'substeps': 'substeps_per_call', 'noise': 'relative_gaussian_kick_per_step', 'init_bound': 'starting_bound_fraction', 'init_noise': 'relative_noise_of_the_start', 'seed': 'seed_of_the_start', 'fz_total': 'total_fz_per_unit_perimeter', 'vang_total': 'total_vang_per_unit_perimeter', 'plane_axis': 'axis_normal_to_the_sheet', 'cue_axis': 'axis_of_the_global_cue', 'cue_cos': 'strength_of_the_cue', 'cue_until': 'frame_the_cue_stops', 'cell_at': 'the_cell_set', 'clone': 'which_cells_are_mutant', 'color_block': 'cell_block_the_polarity_colour_is_written_to', 'color_mode': 'what_the_colour_shows'}
    REFERENCE = ("Amonlirdviman, K. et al. (2005). Mathematical modeling of planar cell polarity to "
                 "understand domineering nonautonomy. Science 307:423-426 (SOM); Burak, Y. & Shraiman, "
                 "B. I. (2009). Order and stochastic dynamics in Drosophila planar cell polarity. "
                 "PLoS Comput Biol 5:e1000628.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.cell_at = params.get("cell_at", "cell")
        self.k_on = float(params.get("k_on", 1.0))
        self.k_off = float(params.get("k_off", 1.0))
        self.g = float(params.get("g", 10.0))
        self.beta = float(params.get("beta", 0.1))
        self.Kb = float(params.get("K_b", 5.0))
        self.Kp = float(params.get("K_p", 2.0))
        self.fz_total = float(params.get("fz_total", 1.0))
        self.vang_total = float(params.get("vang_total", 1.0))
        self.cue_m1 = float(params.get("cue_m1", 1.0))
        self.cue_until = params.get("cue_until", None)
        self.cue_axis = [float(x) for x in params.get("cue_axis", [1.0, 0.0, 0.0])]
        self.cue_cos = float(params.get("cue_cos", 0.5))
        self.plane_axis = int(params.get("plane_axis", 2))
        self.init_bound = float(params.get("init_bound", 0.5))
        self.init_noise = float(params.get("init_noise", 0.5))
        self.noise = float(params.get("noise", 0.0))
        self.seed = int(params.get("seed", 0))
        self.dt = float(params.get("dt", 0.05))
        self.substeps = int(params.get("substeps", 4))
        self.spread = float(params.get("spread", 1.0))
        self.elong = float(params.get("elong", 0.0))
        self.elong_ref = str(params.get("elong_ref", "none"))
        self._e0 = None
        self.clone = params.get("clone") or None
        self.color_block = params.get("color_block") or None
        self.color_mode = str(params.get("color_mode", "arrow"))
        if self.color_mode not in ("arrow", "celsr"):
            raise ValueError(f"junction_pcp: color_mode must be 'arrow' or 'celsr', not {self.color_mode!r}")
        self._tot = None
        self._gen = None
        self._said = False

    def _hill(self):
        """The cooperativity of the cross-junction recruitment: None (linear) for this model."""
        return None

    def _cell_block(self, H, name):
        try:
            lvl = H.level(self.cell_at)
        except Exception:                                                    # noqa: BLE001
            return None
        if name not in getattr(lvl, "state_schema", {}):
            return None
        return lvl.get(name)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        es, live, vi, vj, stride, _ukey, L = _live_edges(m, pos)
        if not bool(live.any()):
            return {}
        nF = int(m["nF"])
        ef = m["E_face"][live].long()
        ax = [i for i in range(pos.shape[1]) if i != self.plane_axis]
        P = pos[:, ax]
        cen = torch.zeros(nF, 2, device=dev, dtype=dt_).index_add(0, ef, P[vi])
        cen = cen / torch.bincount(ef, minlength=nF).clamp_min(1).to(dt_)[:, None]
        d = P[vj] - P[vi]
        n = torch.stack([d[:, 1], -d[:, 0]], 1) / L.clamp_min(1e-12)[:, None]
        n = n * torch.sign(((0.5 * (P[vi] + P[vj]) - cen[ef]) * n).sum(1))[:, None]
        perim = torch.zeros(nF, device=dev, dtype=dt_).index_add(0, ef, L)
        twin = pcp_twins(vi, vj, stride)
        okey = vi * stride + vj
        mid = 0.5 * (P[vi] + P[vj]) - cen[ef]
        kern = pcp_side_pairs(ef, torch.atan2(mid[:, 1], mid[:, 0]), perim, nF, self.spread)

        # ---- first call: totals, clone, the random start --------------------------------------
        if self._tot is None:
            Ft = self.fz_total * perim
            Vt = self.vang_total * perim
            mut = torch.zeros(nF, dtype=torch.bool, device=dev)
            if self.clone:
                c0 = torch.tensor([float(x) for x in self.clone.get("center", [0.0, 0.0])][:2], device=dev, dtype=dt_)
                mut = (cen - c0).norm(dim=1) <= float(self.clone.get("radius", 0.0))
                rem = str(self.clone.get("removes", "fz"))
                if rem not in ("fz", "vang"):
                    raise ValueError(f"junction_pcp: clone.removes must be 'fz' or 'vang', not {rem!r}")
                (Ft if rem == "fz" else Vt)[mut] = 0.0
            self._tot = (Ft, Vt)
            blk = self._cell_block(H, "mutant")
            if blk is not None:
                blk[..., :nF, 0] = mut.to(blk.dtype)
            self._gen = torch.Generator(device="cpu").manual_seed(self.seed)
            u = lambda: (2 * torch.rand(L.shape[0], generator=self._gen, dtype=torch.float64) - 1).to(dev, dt_)  # noqa: E731
            F = self.init_bound * self.fz_total * (1 + self.init_noise * u())
            V = self.init_bound * self.vang_total * (1 + self.init_noise * u())
            F = torch.where(Ft[ef] > 0, F, torch.zeros_like(F))            # a clone cell has none to place
            V = torch.where(Vt[ef] > 0, V, torch.zeros_like(V))
        else:
            keys, sF, sV = m["pcp_keys"], m["pcp_F"], m["pcp_V"]
            order = torch.argsort(keys)
            ks = keys[order]
            j = torch.searchsorted(ks, okey).clamp(max=max(ks.numel() - 1, 0))
            hit = ks[j] == okey
            mF = torch.zeros(nF, device=dev, dtype=dt_).index_add(0, ef, torch.where(hit, sF[order[j]], torch.zeros_like(L)))
            nh = torch.zeros(nF, device=dev, dtype=dt_).index_add(0, ef, hit.to(dt_)).clamp_min(1)
            mV = torch.zeros(nF, device=dev, dtype=dt_).index_add(0, ef, torch.where(hit, sV[order[j]], torch.zeros_like(L)))
            F = torch.where(hit, sF[order[j]], (mF / nh)[ef])
            V = torch.where(hit, sV[order[j]], (mV / nh)[ef])
        Ft, Vt = self._tot
        if Ft.shape[0] != nF:
            raise RuntimeError(f"junction_pcp: the sheet has {nF} cells, the totals were set for "
                               f"{Ft.shape[0]} -- a division or death needs the totals carried")

        cue = torch.tensor(self.cue_axis, device=dev, dtype=dt_)[ax]
        cue = cue / cue.norm().clamp_min(1e-12)
        m1 = self.cue_m1
        if self.cue_until is not None and int(getattr(H, "frame", 0) or 0) >= int(self.cue_until):
            m1 = 1.0
        A = torch.where((n @ cue) >= self.cue_cos, torch.full_like(L, m1), torch.ones_like(L))
        AV = None
        if self.elong != 0.0:
            e1, e2 = pcp_elongation(P, vi, vj, ef, nF)
            if self.elong_ref == "seed":
                if self._e0 is None:
                    self._e0 = (e1.clone(), e2.clone())
                if self._e0[0].shape[0] != nF:
                    raise RuntimeError("junction_pcp: elong_ref seed needs a fixed set of cells")
                e1, e2 = e1 - self._e0[0], e2 - self._e0[1]
            self._eps = (e1, e2, ef)                                         # read by `junction_pcp[primed]`
            phi = torch.atan2(n[:, 1], n[:, 0])
            AV = torch.exp(-self.elong * (e1[ef] * torch.cos(2 * phi) + e2[ef] * torch.sin(2 * phi)))
            A = A * AV
        for _ in range(self.substeps):
            dF, dV = pcp_rates(F, V, twin, ef, L, nF, Ft, Vt, A, self.k_on, self.k_off, self.g,
                               self.beta, self.Kb, self.Kp, kern, AV, self._hill())
            F = (F + self.dt * dF).clamp_min(0.0)
            V = (V + self.dt * dV).clamp_min(0.0)
            if self.noise > 0.0:
                kF = torch.randn(L.shape[0], generator=self._gen, dtype=torch.float64).to(dev, dt_)
                kV = torch.randn(L.shape[0], generator=self._gen, dtype=torch.float64).to(dev, dt_)
                s = self.noise * (self.dt ** 0.5)
                F = (F * (1 + s * kF)).clamp_min(0.0)
                V = (V * (1 + s * kV)).clamp_min(0.0)

        m["pcp_keys"], m["pcp_F"], m["pcp_V"] = okey.detach(), F.detach(), V.detach()
        m["fz"] = _scatter_full(es, live, F.detach(), dev, dt_, fill=0.0)
        m["vang"] = _scatter_full(es, live, V.detach(), dev, dt_, fill=0.0)
        vec = self._cell_block(H, "pcp_vec")
        col = self._cell_block(H, self.color_block) if self.color_block else None
        if vec is not None or col is not None:
            p = torch.zeros(nF, 2, device=dev, dtype=dt_).index_add(0, ef, ((F - V) * L)[:, None] * n)
            s_ = torch.zeros(nF, device=dev, dtype=dt_).index_add(0, ef, (F + V) * L).clamp_min(1e-12)
            asym = p.norm(dim=1) / s_
            if vec is not None:
                vec[..., :nF, 0:2] = p.to(vec.dtype)
                if vec.shape[-1] > 2:
                    vec[..., :nF, 2] = asym.to(vec.dtype)
            if col is not None and col.shape[-1] >= 3 and self.color_mode == "celsr":
                qr, qi = pcp_celsr_q(P, vi, vj, ef, cen, F + V, nF)
                two = torch.atan2(qi, qr) - 2.0 * math.atan2(float(cue[1]), float(cue[0]))   # 2 x the axis, from the cue's
                w = (2.0 * torch.sqrt(qr * qr + qi * qi)).clamp(max=1.0)
                for k in range(3):                                     # a colour wheel of the axis
                    col[..., :nF, k] = (torch.cos(two - 2.0 * math.pi * k / 3).clamp_min(0.0) * w).to(col.dtype)
            elif col is not None and col.shape[-1] >= 3:
                u = p / p.norm(dim=1, keepdim=True).clamp_min(1e-12)
                along = u @ cue
                w = (2.0 * asym).clamp(max=1.0)
                col[..., :nF, 0] = (along.clamp_min(0.0) * w).to(col.dtype)
                col[..., :nF, 1] = ((-along).clamp_min(0.0) * w).to(col.dtype)
                col[..., :nF, 2] = ((1 - along ** 2).clamp_min(0.0).sqrt() * w).to(col.dtype)
        PCP_TRACE.append((nF, float((F * L).sum()), float((V * L).sum())))
        if not self._said:
            print(f"[junction_pcp] {nF} cells, {int(live.sum())} junction sides, g={self.g}, "
                  f"K_b={self.Kb}, K_p={self.Kp}, cue_m1={self.cue_m1}"
                  + (f", clone removes {self.clone.get('removes')}" if self.clone else ""), flush=True)
            self._said = True
        return {}


@register_operator("junction_pcp", model="cooperative", family="polarity", set="vertex", kind="lateral",
                   equation=r"""$$\frac{dF_h}{dt}=k_{\mathrm{on}}c^F_i\big(\beta+g\,T(V_{\bar h})\big)-k_{\mathrm{off}}A_hB_hF_h,\quad T(x)=\frac{(1+K^n)\,x^n}{K^n+x^n}$$""", title="set up planar polarity, cooperative binding")
class JunctionPCPCooperative(JunctionPCP):
    r"""junction_pcp with a cooperative exchange: the recruitment of a complex by its partner across the junction is a Hill function of the partner's density instead of linear in it (T(1) = 1, so g keeps its scale).
    vertex -> vertex: as junction_pcp (reads pos and the half-edge table, writes m["fz"], m["vang"] and the per-cell blocks).

        \frac{dF_h}{dt}=k_{\mathrm{on}}c^F_i\big(\beta+g\,T(V_{\bar h})\big)-k_{\mathrm{off}}A_hB_hF_h,\quad T(x)=\frac{(1+K^n)\,x^n}{K^n+x^n}
        F_h            the Fz complex line density on side h
        k_{on}, k_{off} the binding and unbinding rates
        c^F_i          cell i's free Fz per unit perimeter
        \beta          the baseline recruitment
        g              the recruitment by the partner
        T(x)           the Hill function (1 + K^n) x^n / (K^n + x^n)
        V_{\bar h}     the twin side's Vang
        A_h B_h        the cue and destabilisation factors of the default
        K              `hill_k`, the partner density at half recruitment (must sit at the operating density ~1)
        n              `hill_n`, the cooperativity

    Reference: Amonlirdviman, K. et al. (2005). Science 307:423-426, SOM reactions S1-S10."""
    PARAM_UNITS = {'hill_k': 'dimensionless', 'hill_n': 'dimensionless'}
    SYMBOLS = {'F_h': 'the Fz complex line density on side h', 'k_{on}, k_{off}': 'the binding and unbinding rates', 'c^F_i': "cell i's free Fz per unit perimeter", '\\beta': 'the baseline recruitment', 'g': 'the recruitment by the partner', 'T(x)': 'the Hill function (1 + K^n) x^n / (K^n + x^n)', 'V_{\\bar h}': "the twin side's Vang", 'A_h B_h': 'the cue and destabilisation factors of the default', 'K': '`hill_k`, the partner density at half recruitment (must sit at the operating density ~1)', 'n': '`hill_n`, the cooperativity'}

    PARAM_ROLES = dict(JunctionPCP.PARAM_ROLES, hill_n="cooperativity_of_the_exchange",
                       hill_k="partner_density_at_half_saturation")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.hill_n = float(params.get("hill_n", 2.0))
        self.hill_k = float(params.get("hill_k", 1.0))

    def _hill(self):
        return (self.hill_n, self.hill_k)


@register_operator("junction_pcp", model="primed", family="polarity", set="vertex", kind="lateral",
                   equation=r"""$$\frac{dF_h}{dt}=k_{\mathrm{on}}c^F_i\big(\beta+g\,[(1-p_i)V_{\bar h}+p_i T(V_{\bar h})]\big)-k_{\mathrm{off}}A_hB_hF_h,\quad p_i=\min\big(1,\max_{t'\le t}|\epsilon_i(t')-\epsilon_i(0)|/\epsilon_*\big)$$""", title="set up planar polarity primed by stretch")
class JunctionPCPPrimed(JunctionPCPCooperative):
    r"""junction_pcp[cooperative] whose cooperativity a cell acquires by being stretched: linear exchange until its shape has changed by `prime_strain`, cooperative once it has, for good.
    vertex -> vertex: as junction_pcp; needs `elong_ref: seed` and a non-zero `elong` (the coupling that measures the elongation).

        \frac{dF_h}{dt}=k_{\mathrm{on}}c^F_i\big(\beta+g\,[(1-p_i)V_{\bar h}+p_i T(V_{\bar h})]\big)-k_{\mathrm{off}}A_hB_hF_h,\quad p_i=\min\big(1,\max_{t'\le t}|\epsilon_i(t')-\epsilon_i(0)|/\epsilon_*\big)
        F_h            the Fz complex line density on side h
        k_{on}, k_{off} the binding and unbinding rates
        c^F_i          cell i's free Fz per unit perimeter
        \beta          the baseline recruitment
        g              the recruitment by the partner
        p_i            cell i's priming, 0 to 1
        V_{\bar h}     the twin side's Vang
        T              the cooperative model's Hill function
        A_h B_h        the cue and destabilisation factors
        \epsilon_i(t)  cell i's elongation tensor (Aigouy et al. 2010) at frame t, from its seeded shape
        \epsilon_*     `prime_strain`, the change of shape that primes a cell

    Reference: Amonlirdviman, K. et al. (2005). Science 307:423-426; Aigouy, B. et al. (2010). Cell 142:773-786 (the elongation tensor); the priming rule: Plexus exp08 (this work)."""
    PARAM_UNITS = {'prime_strain': 'dimensionless'}
    SYMBOLS = {'F_h': 'the Fz complex line density on side h', 'k_{on}, k_{off}': 'the binding and unbinding rates', 'c^F_i': "cell i's free Fz per unit perimeter", '\\beta': 'the baseline recruitment', 'g': 'the recruitment by the partner', 'p_i': "cell i's priming, 0 to 1", 'V_{\\bar h}': "the twin side's Vang", 'T': "the cooperative model's Hill function", 'A_h B_h': 'the cue and destabilisation factors', '\\epsilon_i(t)': "cell i's elongation tensor (Aigouy et al. 2010) at frame t, from its seeded shape", '\\epsilon_*': '`prime_strain`, the change of shape that primes a cell'}

    PARAM_ROLES = dict(JunctionPCPCooperative.PARAM_ROLES,
                       prime_strain="cell_shape_change_at_which_the_exchange_is_fully_cooperative")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.prime_strain = float(params.get("prime_strain", 0.1))
        if self.elong == 0.0 or self.elong_ref != "seed":
            raise ValueError("junction_pcp[primed] reads each cell's strain from the elongation "
                             "coupling: set elong != 0 and elong_ref: seed")
        self._eps = None
        self._smax = None

    def _hill(self):
        e1, e2, ef = self._eps
        s = torch.sqrt(e1 * e1 + e2 * e2)
        self._smax = s if self._smax is None or self._smax.shape != s.shape else torch.maximum(self._smax, s)
        p = (self._smax / self.prime_strain).clamp(max=1.0)
        return (self.hill_n, self.hill_k, p[ef])
