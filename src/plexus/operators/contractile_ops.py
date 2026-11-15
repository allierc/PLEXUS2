"""Active strain: a contractile material that shortens along its own fibre.

`active_strain` is a muscle's mechanism, written the way a sarcomere works: not as a force added
to the material, but as a change of the length the material considers UNSTRESSED. Once a frame,
every material point of cell j has its deformation gradient multiplied on the left,

    F  <-  ( I + c_j(t) f_j f_j^T + c2_j(t) f_perp f_perp^T )  F

so that the accumulated active part after a whole beat is exactly

    I + gamma_j(t) ( g_j f_j f_j^T + g2_j f_perp f_perp^T )

The engine computes stress from F, so multiplying F this way tells the elastic law "you are
stretched beyond your rest length along f", and the material pulls itself shorter along f until
the stress balances against its neighbours. The mass never changes; only the length the material
considers unstressed.

    g       THE ACTIVE STRAIN along the fibre: the fractional shortening the cell would reach if
            nothing resisted it. Dimensionless. `g = 0.04` means "this cell wants to be 4%
            shorter than its rest length". It is NOT the shortening seen on a film -- a cell that
            pulls hard while its neighbours pull the other way barely moves -- and the gap between
            the two is decided by the mechanics. That gap is the whole reason to fit g rather than
            measure it.
    g2      the active strain ACROSS the fibre, negative for a cell that thickens as it shortens.
            Rank-1 contraction (g2 = 0) cannot conserve area, and a cardiomyocyte sheet very
            nearly does: adding this term took the reference fit from explaining 69% of the
            per-cell motion to 87%, and dissolved a 2.6x stiffness difference between two sheets
            that had been the rank-1 model faking the thickening it could not express.
    f       the FIBRE AXIS, f_j = (cos phi_j, sin phi_j), modulo 180 degrees because an axis has
            no head or tail.
    gamma   the activation, 0 to 1, shared by the sheet and shifted per cell by its own delay.

THE SIGN IS THE WHOLE CONTENT. `I + c f f^T` contracts and `I - c f f^T` grows; the growth
convention is what `plexus.morph` uses to inflate a ball into a cow, and the reference model had
it backwards at first -- measured consequence, the fitted axes came out PERPENDICULAR to the
tissue's (axis agreement -0.85 instead of +0.85) and every number downstream was a mirror.

WHY THE INCREMENT IS NOT `gamma * g`. The operator multiplies F every frame, so the per-frame
factor must be the RATIO of consecutive accumulated states, not the accumulated state itself:

    c_j(t) = ( gamma(t) - gamma(t-1) ) g_j  /  ( 1 + gamma(t-1) g_j )

which telescopes to `1 + gamma(t) g_j` exactly. Using `gamma(t) g_j` directly would compound and
the cell would contract geometrically.

WHERE THE PER-CELL NUMBERS COME FROM, and why this operator does not say. They are state blocks
of the PARENT set -- one row per cell -- and the material points reach them through
`H.lift_index(child, ancestor)`, the language's Broadcast along the containment map. So the same
operator serves a sheet whose parameters were fitted (`seed_state_from_file`), drawn
(`seed_state_random`) or written by hand, and nothing about the fit leaks in here. On the
reference segmentation `lift_index` agrees with the per-particle `cell_id` label on 100% of
2,832 particles, and it is the safer of the two: the label happens to line up, the containment
map is maintained.

Reference: prototype/cardio_mpm/strain (Plexus, this work) -- human iPSC-derived cardiomyocyte
sheets fitted from motion alone; gradient certified against central differences in float64.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


def _clock(t, t0, tr, dur, td):
    """The activation pulse: a rise multiplied by a fall, both logistic.

        s(t) = sigmoid( (t - t0) / tr ) * sigmoid( (t0 + dur - t) / td )

    t0 is the onset in frames, tr the rise time, dur the plateau and td the decay -- four numbers
    for the whole sheet. Every argument may carry a per-cell axis, which is how a cell gets its
    own rise and decay around the shared one.
    """
    return torch.sigmoid((t - t0) / tr) * torch.sigmoid((t0 + dur - t) / td)


@register_operator("active_strain", family="mechanics", set="particle", kind="lateral",
                   equation=r"""$$\mathbf F\leftarrow\big(\mathbf I+\delta\gamma_j\,\mathbf f_j\mathbf f_j^{\mathsf T}+\delta\gamma^{(2)}_j\,\mathbf p_j\mathbf p_j^{\mathsf T}\big)\mathbf F,\qquad \gamma_j(t)=g_j\,s\!\left(\phi_j(t)\right)$$""", title="contract cells by shortening rest length")
class ActiveStrain(Lateral):
    r"""the contraction as a rest-length change on the material points of each cell: the fibre shortens and the material thickens across it, by the cell's own amplitude and phase.
    cell -> mpm_particle: reads the parent's `phi`, `g`, `g2` (and its timing blocks if declared), writes the particle set's deformation gradient F and the parent's `gam_prev`; emits nothing (`mpm_strain` turns the new rest configuration into stress on the next substep).

        \mathbf F\leftarrow\big(\mathbf I+\delta\gamma_j\,\mathbf f_j\mathbf f_j^{\mathsf T}+\delta\gamma^{(2)}_j\,\mathbf p_j\mathbf p_j^{\mathsf T}\big)\mathbf F,\qquad \gamma_j(t)=g_j\,s\!\left(\phi_j(t)\right)
        F              the deformation gradient of a material point (rewritten in place)
        I              the identity
        \delta\gamma_j the increment gamma_j(t) - gamma_j(t-1) of cell j's fibre contraction, dimensionless strain
        f_j            cell j's unit fibre direction
        \delta\gamma^{(2)}_j the increment of the strain across the fibre (amplitude g2_j)
        p_j            the unit direction across the fibre
        \gamma_j(t)    cell j's contraction at time t
        g_j            cell j's contraction amplitude along the fibre (block `g`), dimensionless strain
        s(\phi_j(t))   the beat's shape as a function of the cell's own phase phi_j

    Reference: Nardinocchi, P. & Teresi, L. (2007). On the active response of soft living tissues. J. Elasticity 88:27-39; Ambrosi, D. & Pezzuto, S. (2012). Active stress vs. active strain in mechanobiology. J. Elasticity 107:199-212 (the multiplicative active-strain decomposition); the per-cell fit: prototype/cardio_mpm/strain (Plexus, this work)."""
    PARAM_UNITS = {'period': 'frames'}
    SYMBOLS = {'F': 'the deformation gradient of a material point (rewritten in place)', 'I': 'the identity', '\\delta\\gamma_j': "the increment gamma_j(t) - gamma_j(t-1) of cell j's fibre contraction, dimensionless strain", 'f_j': "cell j's unit fibre direction", '\\delta\\gamma^{(2)}_j': 'the increment of the strain across the fibre (amplitude g2_j)', 'p_j': 'the unit direction across the fibre', '\\gamma_j(t)': "cell j's contraction at time t", 'g_j': "cell j's contraction amplitude along the fibre (block `g`), dimensionless strain", 's(\\phi_j(t))': "the beat's shape as a function of the cell's own phase phi_j"}

    EMIT = None
    INTEGRAND = None
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    MAY_MUTATE_INTEGRATED_STATE = True     # writes F (an attribute) and the parent's `gam_prev`
    REQUIRES_PARAMS = ["parent"]
    READS = ["phi", "g", "g2"]
    WRITES = ["gam_prev"]
    MECHANISM_TAGS = ["active_strain", "contraction", "muscle", "rest_length", "excitation"]
    PARAM_ROLES = {
        "parent": "the_set_carrying_the_per_cell_parameters",
        "fit": "npz_holding_the_shared_clock_and_temporal_modes",
        "clock": "four_numbers_t0_logrise_logdur_logdecay",
        "period": "frames_between_beats_for_an_evenly_spaced_clock",
        "segments": "frame_offsets_at_which_the_clock_fires_again",
    }
    REFERENCE = ("prototype/cardio_mpm/strain (Plexus, this work): per-cell active strain fitted "
                 "from motion alone on cardiomyocyte sheets.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.parent = str(params["parent"])
        self.period = params.get("period")
        # SEGMENTS: the frame offsets at which the clock fires AGAIN, so one continuous rollout
        # spans several beats without ever being reset. `period:` is the same idea when the beats
        # are evenly spaced; the recording's are not (50 then 51 frames), and a uniform period
        # drifts by a frame per beat. They also keep the clock's local time inside the range the
        # temporal modes were fitted over -- psi has one column per frame OF A BEAT, so indexing
        # it with an absolute frame would run off the end and clamp.
        self.segments = list(params.get("segments") or [])
        # THE SHARED CLOCK IS A PARAMETER OF THE MAP, NOT STATE OF A SET, and it is read the same
        # way `muscle_pose_map` reads its 27 quadratic coefficients: from a `fit:` file, or inline.
        # Four numbers and a K x T mode table belong to no element of any set, so there is no
        # block that could hold them honestly.
        self.psi = None
        fit = params.get("fit")
        if fit is not None:
            z = np.load(self._resolve(fit), allow_pickle=False)
            if "clock" not in z.files:
                raise ValueError(
                    f"active_strain: {fit} holds no 'clock' array (has: {sorted(z.files)}). "
                    f"The shared clock is four numbers: onset, log rise, log plateau, log decay.")
            clk = np.asarray(z["clock"], np.float32).ravel()
            if clk.size != 4:
                raise ValueError(f"active_strain: 'clock' is {clk.size} numbers, expected 4 "
                                 f"(onset, log rise, log plateau, log decay).")
            self.clock = torch.as_tensor(clk).to(torch.get_default_dtype())
            if "psi" in z.files:
                self.psi = torch.as_tensor(np.asarray(z["psi"], np.float32)).to(
                    torch.get_default_dtype())                                  # [K, T]
        elif "clock" in params:
            self.clock = torch.as_tensor(
                np.asarray(params["clock"], np.float32).ravel()).to(torch.get_default_dtype())
        else:
            raise ValueError(
                "active_strain needs the shared clock: `fit:` naming an npz with a 'clock' array, "
                "or `clock: [t0, log_rise, log_dur, log_decay]` inline.")
        self._idx = None

    @staticmethod
    def _resolve(path) -> str:
        from plexus.paths import graphs_data_path
        if os.path.isabs(path) and os.path.isfile(path):
            return path
        p = graphs_data_path(path)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"active_strain: fit {path!r} -> {p} does not exist.")
        return p

    def _block(self, lvl, name, n, dev, dt):
        """A parent block if the spec declares one, else zeros -- so a sheet with no per-cell
        timing is the shared-clock model exactly, with nothing to switch on."""
        if name not in lvl.state_schema:
            return torch.zeros(n, 1, device=dev, dtype=dt)
        return lvl.get(name)

    def gamma(self, cell, frame, dev, dt):
        """[C, 1] activation of every cell at `frame`, its own delay and time-course applied."""
        n = cell.n
        t0, tr, dur, td = (self.clock.to(device=dev, dtype=dt)[i] for i in range(4))
        tr, dur, td = tr.exp(), dur.exp(), td.exp()
        delay = self._block(cell, "delay", n, dev, dt)
        tr = tr * self._block(cell, "logtr", n, dev, dt).exp()
        dur = dur * self._block(cell, "logdur", n, dev, dt).exp()
        td = td * self._block(cell, "logtau", n, dev, dt).exp()
        # A BEAT IS ONE FIRING OF THE CLOCK, and a rollout spanning several is ONE run rather
        # than several stitched together: the state carries over, so the drift a real tissue
        # accumulates across beat boundaries is in the result rather than reset away.
        t = float(frame)
        if self.segments:
            t = t - max([s for s in self.segments if s <= t], default=self.segments[0])
        elif self.period:
            t = t % float(self.period)
        t = torch.as_tensor(t, device=dev, dtype=dt)
        # gamma(0) = 0 EXACTLY, even for a delayed cell, by normalising against the cell's own
        # value at t = 0. Without it a delayed cell starts already part-way contracted and the
        # first frame is not the rest state the fit was defined against.
        s0 = _clock(-delay, t0, tr, dur, td)
        gam = ((_clock(t - delay, t0, tr, dur, td) - s0) / (1.0 - s0)).clamp(min=0.0)
        if self.psi is not None and "amode" in cell.state_schema:
            # THE MODES ARE OFF AT FRAME 0, and that is not a detail. gamma(0) = 0 is what makes
            # the first frame the REST STATE the fit was defined against: the per-cell clock is
            # normalised to give exactly zero there, and a temporal mode added on top would undo
            # that. Measured cost of getting it wrong: gamma(0) = 0.0014 instead of 0, the sheet
            # starts already slightly contracted, and every later increment is computed from the
            # wrong baseline -- 23% relative error on the per-cell strain maps.
            i = int(min(max(int(t.item()), 0), self.psi.shape[1] - 1))
            if i > 0:
                psi = self.psi.to(device=dev, dtype=dt)
                gam = (gam + cell.get("amode") @ psi[:, i:i + 1]).clamp(min=0.0, max=1.5)
        return gam

    def forward(self, H, mask=None):
        lvl, cell = H.level(self.at), H.level(self.parent)
        if self.parent not in H.ancestors(lvl.name):
            raise ValueError(
                f"active_strain: {self.at!r} is not contained in {self.parent!r} (its chain is "
                f"{' -> '.join(H.ancestors(lvl.name)) or '(none)'}). The per-cell parameters are "
                f"broadcast DOWN the containment map; without it there is no fibre to read.")
        for b in ("phi", "g"):
            if b not in cell.state_schema:
                raise ValueError(
                    f"active_strain: {self.parent!r} has no state block {b!r} "
                    f"(has: {', '.join(x.name for x in cell.state_schema.blocks)}). Declare "
                    f"`sets.{self.parent}.state: {{phi: 1, g: 1, g2: 1, gam_prev: 1}}` and seed "
                    f"them -- `seed_state_from_file` loads a fit, `seed_state_random` draws one.")
        F = getattr(lvl, "F", None)
        if F is None:
            raise ValueError(
                f"active_strain: {self.at!r} carries no deformation gradient F. This operator "
                f"changes the material's REST LENGTH, so it needs a material-point set whose "
                f"strain is tracked -- put it in a spec with `mpm_strain`.")
        dev, dt = F.device, F.dtype
        if self._idx is None:
            self._idx = H.lift_index(lvl.name, self.parent)        # cached; read inside a substep
        D = F.shape[-1]
        phi = cell.get("phi")[:, 0]
        f = torch.stack([phi.cos(), phi.sin()] + ([torch.zeros_like(phi)] if D == 3 else []), -1)
        fp = torch.stack([-phi.sin(), phi.cos()] + ([torch.zeros_like(phi)] if D == 3 else []), -1)
        gam = self.gamma(cell, int(getattr(H, "frame", 0)), dev, dt)               # [C, 1]
        prev = (cell.get("gam_prev") if "gam_prev" in cell.state_schema
                else torch.zeros_like(gam))
        g = cell.get("g")
        g2 = cell.get("g2") if "g2" in cell.state_schema else torch.zeros_like(g)
        c = (gam - prev) * g / (1.0 + prev * g)
        c2 = (gam - prev) * g2 / (1.0 + prev * g2)
        idx = self._idx
        eye = torch.eye(D, device=dev, dtype=dt)
        G = (eye
             + c[idx][:, :, None] * (f[:, :, None] * f[:, None, :])[idx]
             + c2[idx][:, :, None] * (fp[:, :, None] * fp[:, None, :])[idx])
        if mask is not None:
            G = torch.where(mask[:, None, None].to(torch.bool), G, eye.expand_as(G))
        lvl.F = torch.bmm(G.to(F.dtype), F)
        if "gam_prev" in cell.state_schema:
            # CLONE-AND-REASSIGN, so the tape keeps the previous value alive: `gam_prev` is read
            # on the next frame and an in-place write would make the graph reference a tensor it
            # had already overwritten.
            b0, b1 = cell.state_schema["gam_prev"]
            st = cell.state.clone()
            st[..., b0:b1] = gam
            cell.state = st
        return {}


@register_operator("active_strain", family="mechanics", set="particle", kind="lateral", model="excitation",
                   equation=r"""$$\gamma_j(t)=\mathbb 1[\,j\text{ excited}\,]\;\gamma^{\mathrm{fit}}_j\big(t-(t^{\mathrm{exc}}_j-t_{\mathrm{ref}})\big)$$""", title="contract cells when excited")
class ActiveStrainExcitation(ActiveStrain):
    r"""the `excitation` model of active_strain: the contraction is triggered by the cell's own excitation instead of the shared clock alone, the fitted time course kept and started at the moment the cell was excited.
    cell -> mpm_particle: as the default (reads the parent's `phi`, `g`, `g2` and its excitation `chem` column `chan`, writes the particle set's F and the parent's `gam_prev`); a cell never excited never contracts.

        \gamma_j(t)=\mathbb 1[\,j\text{ excited}\,]\;\gamma^{\mathrm{fit}}_j\big(t-(t^{\mathrm{exc}}_j-t_{\mathrm{ref}})\big)
        \gamma_j(t)    cell j's contraction at frame t, dimensionless strain
        \mathbb 1[j excited] 1 once cell j's excitation variable has crossed `thr`, else 0
        \gamma^{fit}_j the default's fitted activation of cell j (shared clock, delay, time course)
        t^{exc}_j      the frame at which cell j's excitation reached `thr` (re-armed below thr/2, so a paced sheet triggers every beat)
        t_{ref}        `t_ref`, the frame of the stimulus the fit is counted from

    Reference: Bers, D. M. (2002). Cardiac excitation-contraction coupling. Nature 415:198-205 (excitation as the trigger); the fitted time course: prototype/cardio_mpm/strain (Plexus, this work)."""
    PARAM_UNITS = {'t_ref': 'frames', 'thr': 'fraction', 'chan': 'count', 'frames_per_clock_frame': 'count'}
    SYMBOLS = {'\\gamma_j(t)': "cell j's contraction at frame t, dimensionless strain", '\\mathbb 1[j excited]': "1 once cell j's excitation variable has crossed `thr`, else 0", '\\gamma^{fit}_j': "the default's fitted activation of cell j (shared clock, delay, time course)", 't^{exc}_j': "the frame at which cell j's excitation reached `thr` (re-armed below thr/2, so a paced sheet triggers every beat)", 't_{ref}': '`t_ref`, the frame of the stimulus the fit is counted from'}
    READS = ["phi", "g", "g2", "chem"]
    PARAM_ROLES = {**ActiveStrain.PARAM_ROLES, "t_ref": "stimulus_frame", "thr": "excitation_threshold",
                   "chan": "excitation_column", "frames_per_clock_frame": "engine_frames_per_recording_frame"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.t_ref = self.tunable(params.get("t_ref"), 0.0, tensor=True)
        self.thr = float(params.get("thr", 0.5))
        self.chan = int(params.get("chan", 0))
        self.fpf = float(params.get("frames_per_clock_frame", 1))
        self._t_exc = None
        self._shift = None

    def _block(self, lvl, name, n, dev, dt):
        b = super()._block(lvl, name, n, dev, dt)
        if name == "delay" and self._shift is not None:
            return b + self._shift
        return b

    def gamma(self, cell, frame, dev, dt):
        frame = frame / self.fpf if self.fpf != 1 else frame
        if "chem" not in cell.state_schema:
            raise ValueError(f"active_strain[excitation]: {self.parent!r} has no `chem` block -- the "
                             f"excitation this model reads. Put an excitable chemistry on the cell set.")
        u = cell.get("chem")[:, self.chan].detach().to(dt)
        if self._t_exc is None or self._t_exc.shape[0] != u.shape[0]:
            self._t_exc = torch.full((u.shape[0],), float("nan"), device=dev, dtype=dt)
            self._armed = torch.ones(u.shape[0], dtype=torch.bool, device=dev)
        # ONE ONSET PER BEAT: a cell's onset is its latest upstroke through `thr`; it re-arms once its
        # excitation has fallen below thr / 2, so a paced sheet is triggered again at every beat
        new = self._armed & (u >= self.thr)
        self._t_exc = torch.where(new, torch.full_like(self._t_exc, float(frame)), self._t_exc)
        self._armed = (self._armed & ~new) | (u < 0.5 * self.thr)
        # the beat this frame belongs to starts at the last segment offset (0 without segments), and
        # only an onset inside it triggers this beat's contraction
        seg = max([float(x) for x in self.segments if float(x) <= float(frame)], default=0.0)
        excited = ~torch.isnan(self._t_exc) & (self._t_exc >= seg)
        self._shift = torch.nan_to_num(self._t_exc - seg - self.t_ref, nan=0.0)[:, None]
        try:
            gam = super().gamma(cell, frame, dev, dt)
        finally:
            self._shift = None
        return gam * excited.to(gam.dtype)[:, None]


@register_operator("material_from_cell", family="mechanics", set="particle", kind="lateral",
                   equation=r"""$$E_j=e^{\,\log E_j}$$""", title="copy cell stiffness to its points")
class MaterialFromCell(Lateral):
    r"""per-cell stiffness onto the material points that belong to the cell, once at frame 0, stored as log E because a stiffness is positive.
    cell -> mpm_particle: reads the parent's `logE`, writes the particle set's Lame parameters mu and lambda (mu = E/(2(1+nu)), lambda = E nu/((1+nu)(1-2nu))).

        E_j=e^{\,\log E_j}
        E_j            cell j's Young's modulus, in the spec's stress unit
        \log E_j       the parent's `logE` block, the fitted (or loaded) log stiffness of cell j

    Reference: prototype/cardio_mpm/strain (Plexus, this work): per-cell stiffness, reported and never claimed (it is not identifiable per cell from motion alone); the Lame conversion is the standard isotropic one."""
    PARAM_UNITS = {'nu': 'dimensionless'}
    SYMBOLS = {'E_j': "cell j's Young's modulus, in the spec's stress unit", '\\log E_j': "the parent's `logE` block, the fitted (or loaded) log stiffness of cell j"}

    EMIT = None
    INTEGRAND = None
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["parent"]
    READS = ["logE"]
    WRITES = []
    MECHANISM_TAGS = ["heterogeneous_material", "stiffness", "lame"]
    PARAM_ROLES = {"parent": "the_set_carrying_the_per_cell_stiffness",
                   "nu": "poisson_ratio", "block": "parent_block_holding_log_youngs"}
    REFERENCE = ("prototype/cardio_mpm/strain (Plexus, this work): per-cell stiffness, reported "
                 "and never claimed -- not identifiable per cell.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.parent = str(params["parent"])
        self.nu = self.tunable(params.get("nu"), 0.3, tensor=True)
        self.block = str(params.get("block", "logE"))
        self._idx = None

    def forward(self, H, mask=None):
        if int(getattr(H, "frame", 0)) != 0:
            return {}
        lvl, cell = H.level(self.at), H.level(self.parent)
        if self.block not in cell.state_schema:
            raise ValueError(
                f"material_from_cell: {self.parent!r} has no state block {self.block!r} "
                f"(has: {', '.join(x.name for x in cell.state_schema.blocks)}).")
        if self._idx is None:
            self._idx = H.lift_index(lvl.name, self.parent)
        E = cell.get(self.block)[:, 0].exp()[self._idx]
        nu = self.nu
        lvl.mu = E / (2 * (1 + nu))
        lvl.la = E * nu / ((1 + nu) * (1 - 2 * nu))
        return {}


from plexus.operators.mpm_ops import MPMAnchor                          # noqa: E402


@register_operator("mpm_anchor", model="per_cell", family="mechanics", set="particle", kind="lateral",
                   equation=r"""$$\mathbf a_p=k\,e^{\log\kappa_{j(p)}}\,(\mathbf x^{\mathrm{rest}}_p-\mathbf x_p)$$""", title="anchor cells with per-cell springs")
class MPMAnchorPerCell(MPMAnchor):
    r"""the substrate spring with a stiffness per cell: how firmly each cell is held by its gel.
    cell -> mpm_particle: reads the parent's `logkappa` block through the containment map, emits an acceleration toward each point's rest position (`mpm_acceleration`).

        \mathbf a_p=k\,e^{\log\kappa_{j(p)}}\,(\mathbf x^{\mathrm{rest}}_p-\mathbf x_p)
        a_p            the acceleration of point p, in world length per time squared
        k              the sheet's anchor stiffness, per time squared
        \log\kappa_{j(p)} the dimensionless log factor of the cell j that holds point p (block `logkappa`)
        x^{rest}_p     the rest position of point p, in world length
        x_p            its current position

    Reference: prototype/cardio_mpm/strain `anchor_percell` (Plexus, this work), the per-cell adhesion of the cardiomyocyte reporting fits."""
    SYMBOLS = {'a_p': 'the acceleration of point p, in world length per time squared', 'k': "the sheet's anchor stiffness, per time squared", '\\log\\kappa_{j(p)}': 'the dimensionless log factor of the cell j that holds point p (block `logkappa`)', 'x^{rest}_p': 'the rest position of point p, in world length', 'x_p': 'its current position'}
    PARAM_UNITS = {}
    READS = ["logkappa"]
    PARAM_ROLES = {**MPMAnchor.PARAM_ROLES, "parent": "the_set_carrying_logkappa",
                   "block": "per_cell_log_stiffness_block"}

    def __init__(self, params, device="cpu"):
        params = {"applies_to": "substrate", **params}
        super().__init__(params, device)
        self.parent = str(params.get("parent", "cell"))
        self.block = str(params.get("block", "logkappa"))
        self._idx = None

    def forward(self, H, mask=None):
        lvl, cell = H.level(self.at), H.level(self.parent)
        if self._rest is None:
            self._init(lvl, H)
        if self._idx is None:
            self._idx = H.lift_index(lvl.name, self.parent)
        k = self.k
        if self.block in cell.state_schema:
            k = self.k * cell.get(self.block)[..., 0].exp()[..., self._idx]
            k = k[..., None]
        acc = k * (self._rest - lvl.get("pos")) * (self._sel * lvl.occ)[:, None].float()
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}


@register_operator("active_strain", family="mechanics", set="particle", kind="lateral", model="coupled",
                   equation=r"""$$a_j(t)=\mathrm{clip}\Big(\gamma_j(t)+\sum_{i\in N(j)}W_2\tanh\!\big(W_1[a_i(t-1),a_j(t-1)]\big),\,0,\,1.5\Big)$$""", title="contract cells with learned neighbour coupling")
class ActiveStrainCoupled(ActiveStrain):
    r"""the `coupled` model of active_strain: each cell's pulse is modulated by a learned message from the cells it touches, one message-passing step per frame (a graph neural network between cells).
    cell -> mpm_particle: as the default, with the activation a_j the contraction follows read from last frame's `gam_prev` of the neighbours (`edge_index` of cell_neighbours[label_image]); W1, W2 are tensors on the instance a training spec frees as `{param: W1, op: active_strain}`.

        a_j(t)=\mathrm{clip}\Big(\gamma_j(t)+\sum_{i\in N(j)}W_2\tanh\!\big(W_1[a_i(t-1),a_j(t-1)]\big),\,0,\,1.5\Big)
        a_j(t)         the activation cell j's contraction follows at frame t, clipped to [0, 1.5]
        \gamma_j(t)    the default's activation (shared clock, delay, time course)
        N(j)           the cells whose segmented outlines touch j's
        W_1            the message's first weights, [hidden, 2 + 2 d] with d the embedding width
        W_2            the message's output weights, [1, hidden]
        a_i(t-1)       neighbour i's activation last frame

    Reference: Kleber, A. G. & Rudy, Y. (2004). Basic mechanisms of cardiac impulse propagation and associated arrhythmias. Physiol. Rev. 84:431-488 (gap-junction coupling); Gilmer, J. et al. (2017). Neural message passing for quantum chemistry. ICML (the message-passing form)."""
    PARAM_UNITS = {'hidden': 'count', 'embedding_dim': 'count', 'seed': 'count'}
    SYMBOLS = {'a_j(t)': "the activation cell j's contraction follows at frame t, clipped to [0, 1.5]", '\\gamma_j(t)': "the default's activation (shared clock, delay, time course)", 'N(j)': "the cells whose segmented outlines touch j's", 'W_1': "the message's first weights, [hidden, 2 + 2 d] with d the embedding width", 'W_2': "the message's output weights, [1, hidden]", 'a_i(t-1)': "neighbour i's activation last frame"}
    PARAM_ROLES = {**ActiveStrain.PARAM_ROLES, "hidden": "message_network_width",
                   "seed": "message_network_init_seed",
                   "embedding": "latent_block_of_the_parent_read_by_the_message",
                   "embedding_dim": "width_of_that_block"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        h = int(params.get("hidden", 8))
        self.embedding = params.get("embedding")
        d = int(params.get("embedding_dim", 2)) if self.embedding else 0
        gen = torch.Generator().manual_seed(int(params.get("seed", 0)))
        self.W1 = torch.randn(h, 2 + 2 * d, generator=gen) / np.sqrt(2.0 + 2 * d)
        self.W2 = torch.zeros(1, h)

    def gamma(self, cell, frame, dev, dt):
        gam = super().gamma(cell, frame, dev, dt)                        # [C, 1]
        ei = getattr(cell, "edge_index", None)
        if frame == 0 or ei is None or ei.numel() == 0 or "gam_prev" not in cell.state_schema:
            return gam
        a = cell.get("gam_prev")                                         # a(t-1), [C, 1]
        src, dst = ei[0].to(a.device), ei[1].to(a.device)
        x = torch.cat([a[src], a[dst]], -1)                              # [E, 2]
        if self.embedding:
            e = cell.get(self.embedding)                                 # [C, d]
            if e.shape[-1] * 2 + 2 != self.W1.shape[1]:
                raise ValueError(f"active_strain[coupled]: embedding {self.embedding!r} has width "
                                 f"{e.shape[-1]}, but `embedding_dim` said {(self.W1.shape[1] - 2) // 2}")
            x = torch.cat([x, e[src], e[dst]], -1)                       # [E, 2 + 2d]
        m = torch.tanh(x @ self.W1.to(device=dev, dtype=dt).T) @ self.W2.to(device=dev, dtype=dt).T
        msg = torch.zeros_like(a).index_add(0, dst, m)                   # sum over each cell's contacts
        return (gam + msg).clamp(min=0.0, max=1.5)
