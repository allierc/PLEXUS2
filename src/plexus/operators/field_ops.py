"""A continuum bound to a set: what writes into it, what happens inside it, and what reads it.

Deposit, diffuse, decay and sense are one mechanism -- stigmergy, a trail laid and followed --
written as four operators so each can be swapped independently. Reading them apart is also how
a specification ends up depositing into a field that nothing senses.

In the order they appear below:

    grid              field     a C-channel scalar grid; pure state, no behaviour
    deposit           exchange  set -> field: each element adds to the voxel it stands on
    diffuse           field     field -> field: one step of dc/dt = D grad^2 c
    decay             field     field -> field: remove a constant amount everywhere
    sense             exchange  field -> set: read a sensor fan, turn toward the strongest
    chemotax          exchange  field -> set: move along the field's gradient
    prescribed        field     a field read from a video, not solved
    playback          field     advance a prescribed field to this tick's frame
    pacemaker         field     a periodic scalar clock signal p(t)
    activation_pulse  field     paint a clocked activation field, shared clock or travelling wave
    signal            lateral   set -> set along an edge set: connectome signalling

then the alternative implementation, which changes only the numerics:

    diffuse[spectral] the exact heat kernel in Fourier space, against the box-blur default
"""
from __future__ import annotations
import torch
from plexus.models.base import Field
from plexus.models.registry import register_field
from plexus.models.base import Exchange
from plexus.models.registry import register_operator
import math
import torch.fft as fft
import torch.nn.functional as Fnn
from plexus.models.base import FieldUpdate
import os
from plexus.models.base import Field, FieldUpdate
from plexus.models.registry import register_field, register_operator
from plexus.paths import graphs_data_path
import torch.nn.functional as F
from plexus.models.base import Lateral


@register_field("grid", frame="grid")
class ScalarField(Field):
    """A C-channel scalar field on a square-pixel grid. Pure state: it holds a continuum and
    the geometry to index it, and has no dynamics of its own -- operators supply those.

    The domain is the box [0, W] x [0, 1] in 2D, or [0, W] x [0, 1] x [0, 1] in 3D, where W is
    `width` in world units. `res` is R, the resolution in pixels per world unit, so a pixel is
    dx = 1/R world units on every axis and the grid is nx = round(W R) by R (by R). The state is
    one `grid` buffer of shape [C, nx, ny(, nz)], C being the number of channels: a channel per
    species is the usual reading, and `deposit` writes each element into the channel of its own
    type.

    `per_axis: true` makes EVERY axis follow the world box: axis k spans [0, world_size[k]] and
    holds round(world_size[k] R) pixels, still dx = 1/R on every axis. A recorded volume that is
    not square in its last two axes is then held as it is, not padded. Off by default: every existing spec,
    whatever its `world:`, keeps the nx = round(W R) by R (by R) grid above.

    Reference: none -- a regular scalar grid is a representation, not a result.
    """

    def __init__(self, name, couples_to=None, components=1, res=200, width=1.0, dim=2, device="cpu",
                 per_axis=False, world_size=None):
        super().__init__(name, couples_to)
        self.C = int(components)
        self.R = int(res)
        self.width = float(width)
        self.dim = int(dim)
        self.nx = int(round(self.width * self.R))      # square pixels, dx = 1/R
        self.ny = self.R
        if per_axis:
            if world_size is None or len(world_size) < self.dim:
                raise ValueError(f"field {name}: `per_axis: true` needs the world box on all {self.dim} axes")
            self.box = tuple(float(w) for w in world_size[:self.dim])
            self.shape = tuple(int(round(w * self.R)) for w in self.box)
            self.nx, self.ny = self.shape[0], self.shape[1]
            if self.dim == 3:
                self.nz = self.shape[2]
        elif self.dim == 2:
            self.shape = (self.nx, self.ny)
        else:                                          # 3D: axes 1,2 span [0,1]
            self.nz = self.R
            self.shape = (self.nx, self.ny, self.nz)
        if not per_axis:
            self.box = (self.width,) + (1.0,) * (self.dim - 1)
        self.periodic = False                          # set by the engine from the spec boundary
        self.register_buffer("grid", torch.zeros((self.C,) + self.shape, device=device))

    def pix(self, *coords):
        """The voxel a world position falls in: nearest-voxel, not interpolated.

        Takes D coordinate tensors (x, y[, z]) and returns a D-tuple of index tensors. Axis 0
        spans [0, W], every other axis spans [0, 1], and all of them map through the same
        pixels-per-world-unit R. Under `self.periodic` the index wraps modulo the grid rather
        than clamping to the edge, so a sensor reaching past one side reads the other -- the
        same torus the periodic particle wrap uses."""
        out = []
        for k, c in enumerate(coords):
            box = self.box[k]
            if getattr(self, "periodic", False):
                # floor (not trunc-toward-0) so a coord just below 0 wraps to the far edge
                out.append(torch.remainder(torch.floor(c * self.R).long(), self.shape[k]))   # torus wrap
            else:
                out.append((c.clamp(0, box - 1e-6) * self.R).long().clamp(0, self.shape[k] - 1))
        return tuple(out)


@register_operator("diffuse", family="fields", set="field", kind="field",
                   implementation="finite_difference",
                   equation=r"""$$c \;\leftarrow\; (1-w)\,c \;+\; w\,\overline{c}_{3\times3},
\qquad w=\operatorname{sat}(\text{rate}\cdot\Delta t)$$""", title="diffuse the field")
class Diffuse(FieldUpdate):
    """Diffusion: the field spreads down its own gradient. One step of the heat equation.

    field -> field: acts on the field named by `at:`, writing its grid in place. No set is
    involved, which is why the contract's set is `field`.

        dc/dt = D grad^2 c

    c is the field value, per channel and independently, and D the diffusion coefficient in
    world units squared per unit time. This implementation steps it by blending toward a
    3x3 (2D) or 3x3x3 (3D) box mean B(c):

        c <- (1 - w) c + w B(c),        w = saturate(rate * dt)

    Expanding B in Taylor series gives B(c) - c = (dx^2 / 3) grad^2 c in both 2D and 3D, so the
    coefficient this implementation actually realises is D = rate * dx^2 / 3 = rate / (3 R^2)
    world units squared per time, R being the field's pixels per world unit. `rate` is
    therefore a blend weight and not D itself. Saturating w at 1 is what keeps an explicit step
    stable at large rate * dt, at the cost of silently capping the diffusion once it binds.

    A periodic field wraps the blur across the seam; otherwise the edge value is replicated,
    which is a no-flux (reflecting) boundary.

    Reference: Fick, A. (1855). Ueber Diffusion. Ann. Phys. 170:59-86; Turing, A. M. (1952).
    The chemical basis of morphogenesis. Phil. Trans. R. Soc. B 237:37-72.
    """

    EMIT = None                                # field->field: writes the grid in place; returns {} — no integrable delta
    SUPPORTED_DIMS = [2, 3]                     # 3x3 (2D) / 3x3x3 (3D) box-blur step
    REQUIRES_PARAMS = []                        # no required params — target field comes from `at:` (engine-injected)
    MECHANISM_TAGS = ["diffusion", "field_smoothing", "laplacian"]
    PARAM_ROLES = {"rate": "diffusion_rate"}
    REFERENCE = ("Fick, A. (1855). Ueber Diffusion. Ann. Phys. 170:59-86; Turing, A. M. "
                 "(1952). The chemical basis of morphogenesis. Phil. Trans. R. Soc. B "
                 "237:37-72.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params["_at"]   # the field at `at:`
        self.rate = float(params.get("rate", 0.35))     # diffusion weight per unit time

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        g = fld.grid                                                    # [C, *shape]
        dt = float(getattr(H.config, "dt", 1.0))
        # periodic field -> wrap the blur across the seam (`circular`); else edge-clamp.
        pmode = "circular" if getattr(fld, "periodic", False) else "replicate"
        if g.dim() == 3:                                               # 2D field [C, nx, ny]
            gp = Fnn.pad(g.unsqueeze(0), (1, 1, 1, 1), mode=pmode)
            blur = Fnn.avg_pool2d(gp, 3, stride=1).squeeze(0)             # 3x3 mean, same size
        else:                                                         # 3D field [C, nx, ny, nz]
            gp = Fnn.pad(g.unsqueeze(0), (1, 1, 1, 1, 1, 1), mode=pmode)
            blur = Fnn.avg_pool3d(gp, 3, stride=1).squeeze(0)            # 3x3x3 mean, same size
        dw = min(max(self.rate * dt, 0.0), 1.0)                        # saturate(rate*dt)
        fld.grid = g * (1.0 - dw) + blur * dw
        return {}


@register_operator("diffuse", family="fields", set="field", kind="field",
                   implementation="spectral", title="diffuse the field exactly")
class DiffuseSpectral(FieldUpdate):
    """The same diffusion, stepped exactly instead of approximately: in Fourier space the
    heat equation is diagonal, so one step is a multiplication rather than a stencil.

        c_hat(k) <- c_hat(k) exp(-D k^2 dt)

    k is the wavenumber, and the step is exact for any dt -- there is no stability limit and no
    numerical broadening, which is the reason to choose it. Differentiable through torch.fft,
    so an inverse loop filtering `capabilities()` for differentiability keeps this one.

    2D only, and periodic by construction: an FFT cannot express any other boundary.

    THE OPERATING POINT IS NOT THE SAME AS THE DEFAULT IMPLEMENTATION'S. Here `rate` is D
    measured in pixels squared per unit time, because the wavenumbers are built per grid cell.
    In `Diffuse` the realised coefficient is rate / 3 in those same units. The same `rate` in a
    specification therefore diffuses three times faster through this implementation than
    through the box-blur one, so the two are not interchangeable at a fixed parameter value.

    Reference: Fick, A. (1855). Ueber Diffusion. Ann. Phys. 170:59-86; Turing, A. M. (1952).
    The chemical basis of morphogenesis. Phil. Trans. R. Soc. B 237:37-72.
    """

    EMIT = None
    SUPPORTED_DIMS = [2]                        # FFT step is 2D here (N-D is a follow-up)
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["diffusion", "field_smoothing", "spectral"]
    PARAM_ROLES = {"rate": "diffusion_coefficient"}
    REFERENCE = ("Fick, A. (1855). Ueber Diffusion. Ann. Phys. 170:59-86; Turing, A. M. "
                 "(1952). The chemical basis of morphogenesis. Phil. Trans. R. Soc. B "
                 "237:37-72.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params["_at"]
        self.rate = float(params.get("rate", 0.35))     # diffusion coefficient D

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        g = fld.grid                                                    # [C, nx, ny]
        if g.dim() != 3:
            raise NotImplementedError("diffuse:spectral is 2D-only (grid must be [C, nx, ny])")
        dt = float(getattr(H.config, "dt", 1.0))
        _, nx, ny = g.shape
        kx = fft.fftfreq(nx, device=g.device) * (2 * math.pi)          # radians / cell
        ky = fft.fftfreq(ny, device=g.device) * (2 * math.pi)
        k2 = kx[:, None] ** 2 + ky[None, :] ** 2                        # [nx, ny]
        decay = torch.exp(-self.rate * dt * k2)                         # exact heat kernel
        ghat = fft.fftn(g, dim=(-2, -1))
        fld.grid = fft.ifftn(ghat * decay, dim=(-2, -1)).real
        return {}


@register_operator("decay", family="fields", set="field", kind="field",
                   equation=r"""$$c \;\leftarrow\; \max\!\big(0,\; c - k\,\Delta t\big)$$""", title="evaporate the field")
class Decay(FieldUpdate):
    """Evaporation: the field loses a fixed amount everywhere, floored at zero. What stops a
    deposited trail from being permanent, and so what sets how long the past is remembered.

    field -> field: acts on the field named by `at:`, writing its grid in place.

        c <- max(c - r dt, 0)

    r is `rate`, in field units per unit time. Note the form: this is a CONSTANT amount removed
    per unit time, not exponential decay -- it is not dc/dt = -r c. A voxel at 0.2 and one at
    1.0 lose the same absolute amount each tick, so a faint trail vanishes proportionally much
    sooner than a strong one, and every voxel reaches exactly zero in finite time rather than
    approaching it. Combined with `deposit`'s ceiling of 1, the field's memory is at most
    1 / (r dt) ticks.

    Reference: none -- linear removal is a modelling choice here, not a published law. Plexus
    (this work).
    """

    EMIT = None                                # writes the grid in place, returns no delta
    SUPPORTED_DIMS = [2, 3]                     # elementwise evaporation, dimension-agnostic
    REQUIRES_PARAMS = []                        # no required params — field target from `at:`; `rate` optional
    MECHANISM_TAGS = ["evaporation", "field_decay", "stigmergy"]
    PARAM_ROLES = {"rate": "evaporation_rate"}
    REFERENCE = "Plexus (this work); linear removal, not exponential decay."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params["_at"]   # the field at `at:`
        self.rate = float(params.get("rate", 0.012))    # evaporation per unit time

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        dt = float(getattr(H.config, "dt", 1.0))
        fld.grid = (fld.grid - self.rate * dt).clamp(min=0.0)
        return {}



_RING = 6                                                  # 3D sensors around the heading axis


def _perp_basis(h):
    """D-1 orthonormal unit vectors spanning the plane perpendicular to each heading
    h [N, D]. In 2D the perpendicular is unique; in 3D we build two (robust to h ~
    +/-z by falling back to a different reference there)."""
    D = h.shape[1]
    if D == 2:
        return [torch.stack([-h[:, 1], h[:, 0]], dim=1)]   # the unique perpendicular
    ref = h.new_tensor([0.0, 0.0, 1.0]).expand_as(h)
    u = torch.cross(h, ref, dim=1)
    small = u.norm(dim=1) < 1e-4                            # h nearly parallel to z
    ref2 = h.new_tensor([0.0, 1.0, 0.0]).expand_as(h)
    u = torch.where(small[:, None], torch.cross(h, ref2, dim=1), u)
    u = u / u.norm(dim=1, keepdim=True).clamp(min=1e-9)
    v = torch.cross(h, u, dim=1)                            # (h, u, v) orthonormal
    return [u, v]


def _ring_dirs(h, ca, sa):
    """The tilted sensor directions around the heading: `cos(ang)*h + sin(ang)*r` for
    each unit r in the perpendicular plane. 2D -> {ahead-left, ahead-right}; 3D -> a
    ring of `_RING` directions. Returns a list of [N, D] unit vectors."""
    D = h.shape[1]
    basis = _perp_basis(h)
    if D == 2:
        rs = [basis[0], -basis[0]]                         # left / right
    else:
        u, v = basis
        rs = [math.cos(2.0 * math.pi * k / _RING) * u + math.sin(2.0 * math.pi * k / _RING) * v
              for k in range(_RING)]
    return [ca * h + sa * r for r in rs]


def _read(fld, centers, weights, ssz):
    """Windowed, species-weighted trail read at a BATCH of sensors (field -> [N, S]).

    Sums dot(weights, grid[:, *window]) over a (2k+1)^D voxel window around each of the S
    sensor centres [N, S, D]; the per-agent `ssz` masks out offsets falling outside that
    agent's own window half-width. Vectorised over both the S sensors and the (2k+1)^D window
    voxels, so the whole fan is one gather rather than a Python loop over sensors and voxels."""
    N, S, D = centers.shape
    dev = centers.device
    g = fld.grid                                           # [C, *shape]
    shape = fld.shape
    per = getattr(fld, "periodic", False)                  # torus field: wrap the window across the seam
    ssz = ssz if torch.is_tensor(ssz) else centers.new_full((N,), float(ssz))
    ks = int(ssz.max().item())

    flat = centers.reshape(N * S, D)
    gidx = torch.stack(fld.pix(*[flat[:, k] for k in range(D)]), dim=-1).reshape(N, S, D)   # [N, S, D]
    rng = torch.arange(-ks, ks + 1, device=dev)
    offs = torch.stack(torch.meshgrid(*([rng] * D), indexing="ij"), dim=-1).reshape(-1, D)  # [W, D] window
    W = offs.shape[0]

    # per-axis wrapped/clamped voxel index for the whole [N, S, W] window (D=2/3, not a hot loop)
    axes = []
    for k in range(D):
        col = gidx[:, :, None, k] + offs[None, None, :, k]                 # [N, S, W]
        axes.append(torch.remainder(col, shape[k]) if per else col.clamp(0, shape[k] - 1))
    vals = g[(slice(None),) + tuple(axes)].permute(1, 2, 3, 0)             # [N, S, W, C]

    inwin = (offs.abs()[None, :, :] <= ssz[:, None, None]).all(-1)         # [N, W] offset inside agent window
    contrib = (weights[:, None, None, :] * vals).sum(-1)                   # [N, S, W]
    return (contrib * inwin[:, None, :].float()).sum(-1)                   # [N, S]


@register_field("prescribed", frame="prescribed")
class PrescribedField(Field):
    """A field that is measured rather than solved: its values come from a video, one frame
    per tick, so the continuum is an input to the model instead of an output of it.

    Pure state: the whole `video` buffer [T, nx, ny] read from a TIFF, the current `grid`
    [1, nx, ny], and the world-to-pixel geometry. One channel only. No dynamics of its own --
    `playback` is the operator that advances it, and nothing writes back into it.

    The video is flipped vertically on load, because image rows run top to bottom while the
    domain's y axis runs bottom to top; without the flip every gradient read from it would
    point the wrong way.

    Reference: none -- a prescribed field is data, not a model.
    """

    def __init__(self, name, source=None, res=None, width=1.0, device="cpu"):
        super().__init__(name)                                 # a video binds to no set (no couples_to)
        import tifffile
        path = source if os.path.isabs(source) else graphs_data_path(source)
        vid = tifffile.imread(path).astype("float32")          # [T, ny, nx] (image rows top->bottom)
        vid = vid[:, ::-1, :].copy()                           # flip vertically: image-top -> domain-top
        v = torch.tensor(vid, device=device).permute(0, 2, 1).contiguous()  # -> [T, nx, ny]
        self.C = 1
        self.T = v.shape[0]
        self.nx, self.ny = v.shape[1], v.shape[2]
        self.width = float(width)
        self.R = self.nx / self.width                          # pixels per world unit (x)
        self.register_buffer("video", v)                       # [T, nx, ny]
        self.register_buffer("grid", v[0:1].clone())           # [1, nx, ny]

    def pix(self, x, y):
        gx = (x.clamp(0, self.width - 1e-6) / self.width * self.nx).long().clamp(0, self.nx - 1)
        gy = (y.clamp(0, 1 - 1e-6) * self.ny).long().clamp(0, self.ny - 1)
        return gx, gy


@register_operator("pacemaker", family="fields", set="field", kind="field",
                   equation=r"""$$p(t) \;=\;
\begin{cases}
\sin\!\big(\pi\, s/\tau_d\big) & s<\tau_d\\[2pt]
0 & \text{otherwise}
\end{cases},
\qquad s=(t+\varphi)\bmod T$$""", title="tick a shared clock")
class Pacemaker(FieldUpdate):
    """A clock: one periodic scalar p(t), shared by every operator that reads it. Not a field
    over space -- a single number per tick, published under `name` for others to consume.

    field -> signal: reads the engine's frame counter, writes H.signals[name].

        s(t) = (t + phi) mod P
        p(t) = sin(pi s / d)  if s < d,  else 0

    P is `period`, the interval between beats in ticks; d is `duration`, how many ticks each
    beat stays active; phi is `phase`, a tick offset that lets two pacemakers run out of step.
    The active part is a half sine, so p rises smoothly from 0 to 1 and back rather than
    switching -- a square pulse would inject a discontinuity into whatever integrates it. The
    duty cycle is d / P, and p = 0 for the rest of the period.

    Reference: none -- a periodic forcing term is a modelling choice, not a published law.
    Plexus (this work).
    """

    EMIT = None                 # writes a scalar into H.signals, returns no delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []        # no required params — all knobs optional (defaults in __init__)
    MECHANISM_TAGS = ["periodic_source", "clock", "pacemaker"]
    PARAM_ROLES = {"period": "beat_interval", "duration": "active_width", "phase": "beat_offset"}
    REFERENCE = "Plexus (this work); a half-sine periodic forcing term."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.signal = str(params.get("name", "pacemaker"))   # the H.signals key it writes
        self.period = float(params.get("period", 180.0))     # ticks between beats
        self.duration = float(params.get("duration", 20.0))  # active width (ticks)
        self.phase = float(params.get("phase", 0.0))         # tick offset

    def clock(self, frame: int) -> float:
        s = (frame + self.phase) % self.period
        if s < self.duration:
            return math.sin(math.pi * s / max(self.duration, 1e-9))   # smooth 0 -> 1 -> 0 bump
        return 0.0

    def forward(self, H, mask=None):
        if getattr(H, "signals", None) is None:
            H.signals = {}
        H.signals[self.signal] = float(self.clock(int(getattr(H, "frame", 0))))
        return {}


@register_operator("activation_pulse", family="fields", set="field", kind="field",
                   equation=r"""$$a(\mathbf x,t)=p(t)\,e^{-\lVert\mathbf x-\mathbf x_0\rVert^2/2\sigma^2}$$""", title="paint a timed activation wave")
class ActivationPulse(FieldUpdate):
    """Paint a clocked activation field: where a stimulus is, and when it arrives there. One
    operator with two timing modes, chosen by whether a delay map is given.

    field -> field: writes one channel of the field named by `at:`, in place.

    Without `delay_from`, every point shares one clock and differs only in how strongly it is
    stimulated:

        a(x, t) = p(t) exp(-|x - x0|^2 / 2 sigma^2)        profile: gaussian
        a(x, t) = p(t)                                     profile: uniform

    p(t) is the scalar published by a `pacemaker` under the key named in `clock`, x0 is
    `center` in world coordinates and sigma is `radius`, the stimulus width in world units.
    Every point beats at the same instant, which is a claim that conduction is instantaneous.

    With `delay_from` naming a normalised [0, 1] field m(x), each point instead runs the same
    beat shifted in time:

        tau(x) = T_max m(x)
        s(x, t) = (t - tau(x) + phi) mod P
        a(x, t) = sin(pi s / d)  if s < d,  else 0

    T_max is `max_delay`, the delay in ticks where the map reads 1, and P, d, phi are the
    period, duration and phase in ticks as in `pacemaker`. The activation is then a wave
    travelling outward along whatever gradient the delay map encodes, at a speed set by
    1 / grad tau -- which is how a conduction system is expressed without simulating one. A
    delay map at a different resolution from the field is resampled by interpolation.

    Reference: none -- a prescribed stimulus is a boundary condition, not a mechanism. Plexus
    (this work).
    """

    EMIT = None                       # writes a prescribed field in place; never engine-integrated
    SUPPORTED_DIMS = [2, 3]           # dimension-generic: N-D Gaussian/uniform profile on the [C,nx,ny(,nz)] field
    REQUIRES_PARAMS = []              # no required params — field target from `at:`; all timing knobs optional
    MECHANISM_TAGS = ["activation_field", "gaussian_source", "phase_delay", "travelling_wave", "spatial_clock"]
    PARAM_ROLES = {"radius": "stimulus_width", "center": "stimulus_site", "clock": "pacemaker_signal",
                   "period": "beat_interval", "duration": "active_width",
                   "max_delay": "phase_delay_gain", "delay_from": "delay_map"}
    REFERENCE = "Plexus (this work); a prescribed stimulus, i.e. a boundary condition."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params["_at"]   # activation field at `at:`
        self.channel = int(params.get("channel", 0))
        self.delay_from = params.get("delay_from")                 # None -> shared clock; set -> per-pixel wave
        # shared-clock mode: one clock, a spatial profile
        self.clock = str(params.get("clock", "pacemaker"))         # H.signals key to read p(t)
        self.profile = str(params.get("profile", "gaussian"))      # "gaussian" (localised) | "uniform" (global)
        c = params.get("center", [0.5, 0.5])
        self.center = [float(x) for x in c]                        # N-D site; missing axes default to 0.5 at forward
        self.sigma = float(params.get("radius", 0.12))
        # per-pixel wave mode: the same beat, delayed by a map
        self.period = float(params.get("period", 150.0))           # ticks between beats
        self.duration = float(params.get("duration", 30.0))        # active width (ticks)
        self.phase = float(params.get("phase", 0.0))               # global tick offset
        self.max_delay = float(params.get("max_delay", 10.0))      # ticks of delay at map==1

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        if self.delay_from is None:
            # --- shared clock x spatial profile: every point beats at once --------------- #
            dev = fld.grid.device
            shape, R, D = fld.shape, fld.R, len(fld.shape)         # (nx,ny) 2D or (nx,ny,nz) 3D
            # pixel-centre world coordinates per axis: axis 0 spans [0, width], the rest [0, 1]
            axes = [(torch.arange(shape[k], device=dev) + 0.5) / R for k in range(D)]
            grids = torch.meshgrid(*axes, indexing="ij")          # D tensors, each [*shape]
            if self.profile == "uniform":
                bump = torch.ones(shape, device=dev)               # global stimulus: a(x,t) = p(t)
            else:
                ctr = [self.center[k] if k < len(self.center) else 0.5 for k in range(D)]
                r2 = sum((grids[k] - ctr[k]) ** 2 for k in range(D))
                bump = torch.exp(-r2 / (2.0 * self.sigma * self.sigma))   # localised Gaussian site (N-D)
            p = float((getattr(H, "signals", None) or {}).get(self.clock, 0.0))   # this tick's clock value
            fld.grid[self.channel] = p * bump
        else:
            # --- per-pixel delayed wave: the beat arrives late where the map is high ----- #
            out = fld.grid[self.channel]                           # [nx, ny] activation channel to write
            delay = H.fields[self.delay_from].grid[0].to(out.device)   # [nx, ny] normalised 0..1
            if delay.shape != out.shape:                           # map at a different resolution: resample
                mode = "bilinear" if delay.dim() == 2 else "trilinear"   # 2D grid vs 3D volume
                delay = Fnn.interpolate(delay[None, None].float(), size=tuple(out.shape),
                                        mode=mode, align_corners=True)[0, 0]
            tau = self.max_delay * delay                           # per-pixel delay (ticks)
            t = float(getattr(H, "frame", 0))
            s = torch.remainder(t - tau + self.phase, self.period)   # local phase, handles t-tau < 0
            act = torch.where(s < self.duration,
                              torch.sin((math.pi / max(self.duration, 1e-9)) * s),
                              torch.zeros_like(s))                 # smooth bump while active, else 0
            fld.grid[self.channel] = act
        return {}


_ACT = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "softplus": F.softplus,
    "sigmoid": torch.sigmoid,
    "identity": lambda x: x,
}

