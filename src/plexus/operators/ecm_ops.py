"""The extracellular matrix as MPM (material point method) material, and the stiff blocks that
confine it.

The matrix is a continuum the growing tissue has to push its way into. Every operator here acts
on a set of MPM particles, and the tissue enters as a prescribed moving BOUNDARY rather than as
a second body -- so these describe how an epithelium loads a matrix, not how the two negotiate.

In the order they appear below:

    ecm_seed        seed       the box MINUS a cavity, as aligned fibres, once at frame 0
    ecm_from_cell   lateral    the tissue surface as a moving boundary the particles feel
    ecm_stress      lateral    |J - 1|, banded, so the stress front is visible in a movie
    cell_exclude    structural the hard backstop: no matrix particle inside the lumen
    mpm_block       entity     a material point of a solid block
    block_seed      seed       two slabs beyond a free gap, as a second and much stiffer set
    block_stress    lateral    the block's own strain, at its own full scale

then the two implementations of `ecm_from_cell`, which differ in where the surface comes from:

    ecm_from_cell[sphere]   a prescribed ball of radius r(t): a stand-in with a known answer
    ecm_from_cell[replay]   a recorded epithelium's own apical surface, frame by frame

`ecm_seed` and `block_seed` are not redundant: one fills the complement of a cavity with aligned
fibres, the other fills two slabs with a jittered lattice. Same family, same module, different
geometry. `ecm_stress` and `block_stress` ARE the same body, kept apart only by a module-level
history list per set; moving that onto the Level is what would let one operator serve both.
"""
from __future__ import annotations
import math
import torch
from plexus.models.base import Lateral, Structural
from plexus.models.registry import register_operator
from plexus.models.entities import MPMParticle
from plexus.models.registry import register_entity, register_operator
from plexus.models.state import spatial_schema


STRESS_HISTORY: list = []
BALL_RADIUS: list = []

# The raw scalar, not only the band. Banding at SIMULATION time would make the colour scale a
# property of the run: `stress_scale` baked into 8 levels, everything above it clipped to the top
# band, and changing the palette costing a full re-simulation. A scale that resolves the front
# early in a run leaves most of the matrix saturated late in it, and once the numbers are banded
# no re-render can recover the gradient. Kept as float16, about 0.1% of a particle's trajectory,
# so that the renderer bands it and a new palette or scale costs only a re-render.
STRESS_RAW: list = []

# The reaction the tissue never felt. `ecm_from_cell` computes the force the tissue puts on the
# matrix, and by Newton's third law that force has an equal and opposite partner on the tissue --
# which a REPLAY has nowhere to put, its tissue pass having finished before the matrix pass began.
# Recording it here is what makes the second half of the coupling possible: `ecm_load` reads this
# map in a later tissue pass and pushes back with it. One row per frame, as an equirectangular map
# of pressure by direction.
PRESSURE_HISTORY: list = []


# --------------------------------------------------------------------------- seeding
# Canonical `ecm_seed`, alias `seed_ecm`. Both spellings stay registered because archived
# specifications use each of them, and a rename that leaves the corpus behind is a rename that
# deletes the corpus.


# --------------------------------------------------------------------------- the coupling
# --------------------------------------------------------------------------- what you can see
@register_operator("ecm_stress", family="hierarchy", set="particle", kind="lateral", probe=True,
                   equation=r"""$$s_i=\lvert\det\mathbf F_i-1\rvert,\qquad \mathrm{band}_i=\left\lfloor K\min(s_i/S,1)\right\rfloor$$""", title="colour the matrix by compression")
class ECMStress(Lateral):
    r"""a measurement as an operator: colours the matrix by how hard it is being squeezed, so the stress front is the thing a movie shows rather than the positions (banded, because the renderer's colour is a palette index).
    particle -> particle: reads the deformation gradient F, writes a colour band on node_type (K types of identical material).

        s_i=\lvert\det\mathbf F_i-1\rvert,\qquad \mathrm{band}_i=\left\lfloor K\min(s_i/S,1)\right\rfloor
        s_i            the local volume change |det F_i - 1|, dimensionless, 0 for unstrained material
        F_i            the deformation gradient of point i
        band_i         the palette index, an integer in [0, K)
        K              `bands`, the number of colour bands
        S              `scale`, the strain at which the palette saturates

    Reference: Hu, Y. et al. (2018). A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. ACM Trans. Graph. 37(4):150 (the deformation gradient)."""
    PARAM_UNITS = {'bands': 'count', 'scale': 'fraction', 'channel': 'count'}
    SYMBOLS = {'s_i': 'the local volume change |det F_i - 1|, dimensionless, 0 for unstrained material', 'F_i': 'the deformation gradient of point i', 'band_i': 'the palette index, an integer in [0, K)', 'K': '`bands`, the number of colour bands', 'S': '`scale`, the strain at which the palette saturates'}
    EMIT = None                     # writes a colour channel in place; no integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["diagnostic", "strain_visualisation"]
    PARAM_ROLES = {"scale": "strain_full_scale", "bands": "colour_bands", 'channel': 'node_type_channel_written', 'measure': 'which_strain_measure'}
    REFERENCE = ("Hu, Y. et al. (2018). A moving least squares material point method with "
                 "displacement discontinuity and two-way rigid body coupling. ACM Trans. Graph. "
                 "37(4):150.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        # THE FULL-SCALE VALUE IS FIXED, NOT PER-FRAME. Auto-scaling each frame to its own
        # maximum would make every run look identically stressed -- the brightest fibre is always
        # full brightness -- and a sweep over stiffness would then show no difference at all,
        # which is the one thing it exists to show.
        self.scale = float(params.get("scale", 0.15))
        self.bands = int(params.get("bands", 8))
        self.channel = str(params.get("channel", "node_type"))
        # WHICH STRAIN. `vol` is |J-1|, the local VOLUME change, and it is the honest default because it
        # needs no material constants. But it is blind to the deformation this experiment actually
        # produces: a matrix pushed outward by a growing sphere is SHEARED and dragged far more than it
        # is compressed, and MLS-MPM's fixed-corotated material resists volume change stiffly, so
        # |J-1| stays near zero while the fibres are visibly splayed. That is why movies of a plainly
        # moving matrix read as unstressed -- the MEASURE, not the mechanics. `dev` is the
        # volume-normalised equivalent deviatoric strain, which is where the signal is.
        # `vol` |J-1| volume change | `dev` equivalent deviatoric STRAIN | `vonmises` the von Mises
        # invariant of the CAUCHY STRESS the solver itself computed (`mpm_scatter: store_stress: true`).
        # `vonmises` is the physical one: it is in stress units, it weights shear the way this material's
        # own mu and la do, and it is the same tensor that generated the grid forces -- not a proxy
        # re-derived from F with a constitutive law chosen by the diagnostic.
        self.measure = str(params.get("measure", "vol"))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        F = getattr(lvl, "F", None)
        if F is None:
            return {}
        if self.measure == "vonmises":
            sig = getattr(lvl, "sigma", None)
            if sig is None:
                if not getattr(self, "_warned_vm", False):
                    print("[ecm_stress] measure=vonmises but no `sigma` buffer -- the scatter was not "
                          "asked to keep it (`store_stress: true`). Falling back to |J-1|, which is a "
                          "DIFFERENT quantity: read the colours accordingly.", flush=True)
                    self._warned_vm = True
            else:
                tr = sig.diagonal(dim1=-2, dim2=-1).sum(-1)
                eye = torch.eye(sig.shape[-1], device=sig.device, dtype=sig.dtype)
                dv = sig - (tr / 3.0)[:, None, None] * eye
                vm = torch.sqrt((1.5 * (dv * dv).sum((-1, -2))).clamp_min(0.0))
                band = ((vm / max(self.scale, 1e-9)).clamp(0, 1)
                        * (self.bands - 1)).round().long()
                ch = getattr(lvl, self.channel, None)
                if ch is not None:
                    ch[:] = band.to(ch.dtype).reshape(ch.shape)
                STRESS_HISTORY.append(band.detach().to("cpu", torch.uint8).numpy())
                STRESS_RAW.append(vm.detach().to("cpu", torch.float16).numpy())
                return {}
        J = torch.linalg.det(F)
        if self.measure == "dev":
            # Volume-normalised left Cauchy-Green, then its deviator: shape change with the volume
            # change divided out, so `dev` and `vol` are independent readings of the same F rather than
            # two views of mostly the same number.
            B = F @ F.transpose(-1, -2)
            Bb = B / J.abs().clamp_min(1e-9).pow(2.0 / 3.0)[:, None, None]
            tr = Bb.diagonal(dim1=-2, dim2=-1).sum(-1)
            eye = torch.eye(Bb.shape[-1], device=Bb.device, dtype=Bb.dtype)
            dev = Bb - (tr / 3.0)[:, None, None] * eye
            s = torch.sqrt((1.5 * (dev * dev).sum((-1, -2))).clamp_min(0.0)) / max(self.scale, 1e-9)
        else:
            s = (J - 1.0).abs() / max(self.scale, 1e-9)
        band = (s.clamp(0, 1) * (self.bands - 1)).round().long()
        # `node_type` IS A BUFFER, NOT A STATE BLOCK. It is registered by the provision
        # (base.py:348) alongside mass/F/C, so `lvl.get()` -- which slices the state matrix --
        # raises KeyError on it. Written directly, in place, so the buffer the renderer reads and
        # the buffer the material was assigned from stay the same object.
        ch = getattr(lvl, self.channel, None)
        if ch is None:
            if not getattr(self, "_warned", False):
                print(f"[ecm_stress] no `{self.channel}` buffer -- the movie will not be "
                      f"stress-coloured", flush=True)
                self._warned = True
            return {}
        ch[:] = band.to(ch.dtype).reshape(ch.shape)
        STRESS_HISTORY.append(band.detach().to("cpu", torch.uint8).numpy())
        STRESS_RAW.append((s * max(self.scale, 1e-9)).detach().to("cpu", torch.float16).numpy())
        return {}


BLOCK_STRESS: list = []
BLOCK_RAW: list = []            # the un-banded scalar -- see `ecm_ops.STRESS_RAW`


# --------------------------------------------------------------------------- the entity
@register_entity(
    # `spatial_schema` is a `dim -> StateSchema` callable, not a fixed dict: this set runs in 3D
    # specifications, and a hard-coded 2D schema would silently truncate its state.
    "mpm_block", depth=0,
    state_schema=spatial_schema,
    render={"color_by": "node_type", "arrows": None},
)
class MPMBlock:
    """A material point of a solid block: identical continuum state to the matrix's
    `mpm_particle` -- the deformation gradient F, the affine velocity C, mass, the Lame parameters
    mu and lambda, and the rest volume p_vol. The stock provision allocates it.

    It exists as a separate registration because the entity is resolved BY SET NAME: the registry
    is looked up under the set's own name and falls back to a bare position/velocity schema for
    anything unregistered. A set called `mpm_block` with every MPM operator pointed at it would
    otherwise have no F, and the run would die in `mpm_strain` with a missing attribute -- which
    reads like a bug in the operator and is a missing registration.
    """
    provision = MPMParticle.provision

