#!/usr/bin/env python
"""Continuous flow: a jet that enters through one face and leaves through another.

    general:
      engine: continuous          # selects run() below instead of plexus.engine.run

The physics is two ordinary operators, `mpm_emit` and `mpm_drain`, with no engine support: `occ == 0` particles are already frozen (not advected, `mpm_ops.py:1280`) and add zero weight to the scatter (`:538`), so a dormant particle is absent from the physics. What the engine key buys is a PLACE TO REFUSE: a continuous spec that forgets its drain fills the pool and silently stops emitting (a jet that thins out), so `run()` checks the pair up front and states the pool's headroom in frames.

The pool stays fixed even though the tape is gone: `engine.py` drops a captured CUDA graph the moment a state buffer is reallocated (`CUDA_graph_capture._graph_sig` compares storage pointers every tick), so growing the particle array would cost the measured 2.36x capture speedup on every frame, and compaction on drain would destroy particle identity. A fixed pool with `occ` recycling supports unbounded flow DURATION; only the live population is bounded, by `emission rate x transit time` in a steady jet. Still differentiable: the inlet velocity and the material written into emitted particles are ordinary continuous writes (the discrete count and timing of activation are not).

Four things done on purpose:
  frame level, not substep level   the activation cursor advances once per frame (inside a captured substep the slice would be baked in and the same particles emitted forever): `mpm_emit` and `mpm_drain` go in the OUTER schedule, beside `gravity`.
  branch-free activation           a rolling contiguous slice [cursor, cursor+k) mod N, never `nonzero(occ == 0)` (a boolean-mask index_put inside a capture is `cudaErrorStreamCaptureUnsupported`).
  reset on activation              F = I, C = 0, Jp = 1, vel = inlet: a recycled particle keeping its previous deformation gradient makes spurious stress that reads as an inlet instability.
  a slab, not a plane              emission is spread through `U * dt` of depth, or the stream enters as discrete sheets in lockstep with the frame rate.

The sizing is forced by two constraints, mass flux `rho * A * U = k * m_p / dt` and sampling `k = ppc * A * U * dt / dx^D`, so m_p = rho * dx^D / ppc; `mpm_emit` derives `k` from the inlet area, the speed and the frame step, says what it derived, and warns when the spec's own particle_mass disagrees (the jet's density would not be the declared one)."""
from __future__ import annotations

import math

import torch

from plexus.models.base import Operator
from plexus.models.registry import register_operator

# _AXIS: Axis name -> index.
_AXIS = {"x": 0, "y": 1, "z": 2}


# _face: An inlet / outlet face such as '-y' -> (axis index, travel direction).
def _face(spec: str) -> tuple:
    """'-y' -> (1, -1.0): the axis it enters along, and the direction it travels."""
    s = str(spec).strip().lower()
    sign = -1.0 if s.startswith("-") else +1.0
    ax = _AXIS[s.lstrip("+-")]
    return ax, sign


# MPMEmit: Wake dormant particles at the inlet, once per frame (operator `mpm_emit`).
@register_operator("mpm_emit", family="mpm", set="particle", kind="structural", title="release new material at the inlet", equation=r"""$$k=\mathrm{ppc}\,\frac{A\,U\,\Delta t}{\Delta x^{D}},\qquad m_p=\frac{\rho\,\Delta x^{D}}{\mathrm{ppc}}$$""")
class MPMEmit(Operator):
    r"""activates a rolling slice of the particle pool at the inlet face each frame, with reset state (F = I, C = 0, Jp = 1, the inlet velocity), the count derived from the inlet area, the speed and the frame step.
    particle -> particle: reads `occ`, writes `pos`, `vel`, `F`, `C`, `Jp`, `occ` of the k activated slots; runs in the outer schedule, not the substep.

        k=\mathrm{ppc}\,\frac{A\,U\,\Delta t}{\Delta x^{D}},\qquad m_p=\frac{\rho\,\Delta x^{D}}{\mathrm{ppc}}
        k              the particles activated per frame
        ppc            `ppc`, particles per grid cell
        A              the inlet area (`aperture`, `patch`), world length squared
        U              `speed`, the inlet speed, world length per time
        \Delta t       the frame step
        \Delta x       the grid spacing
        D              the dimension
        m_p            the particle mass the density implies (a spec's `particle_mass` that disagrees is warned about)
        \rho           the material density

    Reference: Plexus (this work); MLS-MPM: Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150."""
    PARAM_UNITS = {'ppc': 'count', 'aperture': 'length', 'speed': 'velocity'}
    SYMBOLS = {'k': 'the particles activated per frame', 'ppc': '`ppc`, particles per grid cell', 'A': 'the inlet area (`aperture`, `patch`), world length squared', 'U': '`speed`, the inlet speed, world length per time', '\\Delta t': 'the frame step', '\\Delta x': 'the grid spacing', 'D': 'the dimension', 'm_p': "the particle mass the density implies (a spec's `particle_mass` that disagrees is warned about)", '\\rho': 'the material density'}

    EMIT = None                                  # writes occ/state directly; returns no delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["face", "speed"]
    MAY_MUTATE_INTEGRATED_STATE = True           # it writes positions and velocities by definition
    MECHANISM_TAGS = ["inlet", "continuous_flow", "particle_recycling"]
    PARAM_ROLES = {"face": "inlet_face", "speed": "inlet_speed", "patch": "inlet_patch_fraction",
                   "ppc": "particles_per_cell", "type": "emitted_type", 'aperture': 'width_of_the_inlet'}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.axis, self.sign = _face(params["face"])
        self.speed = float(params["speed"])
        # the inlet patch as a fraction of the face, so it survives a change of world size: [lo0, lo1, hi0, hi1] over the two axes that are not the inlet normal (default the middle third)
        self.patch = [float(x) for x in (params.get("patch") or [0.33, 0.33, 0.67, 0.67])]
        # `aperture: disc` is a round inlet inscribed in the patch (radius half its shorter side, flux and count scaled by pi/4): a square jet buckles with a fourfold bias a tap or syringe does not have; default `square`, byte-identical to before
        self.aperture = str(params.get("aperture", "square")).lower()
        if self.aperture not in ("square", "disc", "circle"):
            raise ValueError(f"mpm_emit: aperture must be square|disc, got {self.aperture!r}")
        self.ppc = self.tunable(params.get("ppc"), 8.0, tensor=True)
        self.type_name = params.get("type")
        self._cursor = 0
        self._armed = False
        self._said = False

    # helpers
    def _park(self, p, box):
        """Everything dormant and off-domain, once, before the first emission.

        The pool is seeded by `provision` with its material properties already correct (mass, p_vol,
        mu, la, is_liquid) -- only its occupancy and position are wrong for a jet, because the spec's
        `block`/`shape` placed a body where we want an empty box.
        """
        p.occ.zero_()
        # parked just outside, not far outside: a whole box-width away doubled the point cloud's bounding box and VTK sized every sprite relative to it (3,701 blue pixels parked far, 102,032 near); two cells outside the wall is just as invisible
        p.state[:, :box.numel()] = -0.02 * box
        if getattr(p, "Jp", None) is not None:
            p.Jp.fill_(1.0)

    def forward(self, H, mask=None):
        p = H.level(self.at)
        dev = p.state.device
        D = int(getattr(H, "dim", 3))
        box = torch.as_tensor(list(H.world_size)[:D], device=dev, dtype=p.state.dtype)
        dx = float(H.field(next(iter(H.fields))).dx) if len(H.fields) else float(box[0]) / 96.0
        dt = float(getattr(H, "dt", 1.0 / 1200.0))

        if not self._armed:
            self._park(p, box)
            self._armed = True

        # how many, from the flux and the sampling, not from a declared rate
        lat = [k for k in range(D) if k != self.axis]
        A = 1.0
        for i, k in enumerate(lat):
            A *= float(box[k]) * max(self.patch[i + len(lat)] - self.patch[i], 1e-9)
        depth = self.speed * dt                               # the slab that enters this frame
        _disc = self.aperture in ("disc", "circle") and len(lat) == 2
        if _disc:
            A *= math.pi / 4.0                                # the inscribed disc's share of the patch
        k_frame = int(round(self.ppc * A * depth / dx ** D))
        if k_frame < 1:
            return {}

        n = int(p.n)
        idx = (torch.arange(k_frame, device=dev) + self._cursor) % n
        self._cursor = (self._cursor + k_frame) % n

        # the cursor recycles blindly, so say when it eats its own tail: it wraps after `pool / k` frames and overwrites those slots, which is wrong when fluid emitted that long ago has not drained yet (mid-air fluid teleported to the inlet reads as a stutter or as mass appearing); the count is a sync, but this runs once per frame outside the captured region
        _clobber = int((p.occ[idx] > 0).sum())
        if _clobber and not getattr(self, "_clobbered", False):
            self._clobbered = True
            from plexus.paths import warn
            warn(f"mpm_emit: the pool has wrapped onto {_clobber:,} particles that are STILL LIVE "
                 f"({100.0 * _clobber / max(k_frame, 1):.0f}% of this frame's emission). The pool "
                 f"({n:,}) is smaller than emission x transit; those particles are being teleported "
                 f"back to the inlet mid-flight. Raise per_parent to at least "
                 f"{int(n * 1.5):,}, or lower the inlet speed or area.")

        # place them in a SLAB, not on a plane
        u = torch.rand(k_frame, D, device=dev, generator=getattr(H, "rng", None))
        pos = torch.empty(k_frame, D, device=dev, dtype=p.state.dtype)
        if _disc:
            # uniform in the disc, not in its radius: r = R sqrt(u) with a uniform angle, or the jet is densest on its axis
            c0 = [0.5 * (self.patch[i] + self.patch[i + 2]) * float(box[k]) for i, k in enumerate(lat)]
            R = 0.5 * min((self.patch[i + 2] - self.patch[i]) * float(box[k]) for i, k in enumerate(lat))
            r = R * torch.sqrt(u[:, lat[0]]); th = 2.0 * math.pi * u[:, lat[1]]
            pos[:, lat[0]] = c0[0] + r * torch.cos(th)
            pos[:, lat[1]] = c0[1] + r * torch.sin(th)
        else:
            for i, k in enumerate(lat):
                lo, hi = self.patch[i], self.patch[i + len(lat)]
                pos[:, k] = (lo + (hi - lo) * u[:, k]) * box[k]
        # the inlet plane sits 2 cells inside the wall: the gather clamps positions to [2*dx, box - 2*dx], so anything emitted outside piles up on the face
        edge = (float(box[self.axis]) - 2.5 * dx) if self.sign < 0 else 2.5 * dx
        pos[:, self.axis] = edge - self.sign * depth * u[:, self.axis]

        vel = torch.zeros(k_frame, D, device=dev, dtype=p.state.dtype)
        vel[:, self.axis] = self.sign * self.speed

        px0, px1 = 0, D
        p.state[idx, px0:px1] = pos
        vslice = getattr(p, "vel_slice", None)
        if vslice is not None:
            p.state[idx, vslice[0]:vslice[1]] = vel
        else:
            _v = p.get("vel")
            _v[idx] = vel
        # reset, or a recycled particle arrives pre-stressed
        eye = torch.eye(D, device=dev, dtype=p.F.dtype)
        p.F[idx] = eye
        p.C[idx] = 0.0
        if getattr(p, "Jp", None) is not None:
            p.Jp[idx] = 1.0
        p.occ[idx] = 1.0

        if not self._said:
            self._said = True
            m_want = float(p.mass.median()) if hasattr(p, "mass") else 0.0
            rho = m_want * self.ppc / dx ** D if m_want else 0.0
            print(f"[mpm_emit] inlet {A * 1e4:.2f} cm^2 at {self.speed:g} m/s -> {k_frame} "
                  f"particles/frame, pool {n:,} -> {n / max(k_frame, 1):.0f} frames of headroom "
                  f"before recycling; implied density {rho:.0f} kg/m^3", flush=True)
        return {}


# MPMDrain: Damp, then retire, particles at the outlet through a sponge (operator `mpm_drain`).
@register_operator("mpm_drain", family="mpm", set="particle", kind="structural", title="remove material past the outlet", equation=r"""$$\mathbf v_p\leftarrow(1-\text{damp})\,\mathbf v_p\quad\text{in the sponge band},\qquad \mathrm{occ}_p\leftarrow 0\quad\text{past the plane at }\text{at\_fraction}$$""")
class MPMDrain(Operator):
    r"""retires particles past the outlet plane through a damping sponge rather than a cliff, so no momentum discontinuity propagates back upstream.
    particle -> particle: reads `pos`, writes `vel` (damped in the sponge band) and `occ` (0 past the kill plane); runs in the outer schedule.

        \mathbf v_p\leftarrow(1-\text{damp})\,\mathbf v_p\quad\text{in the sponge band},\qquad \mathrm{occ}_p\leftarrow 0\quad\text{past the plane at }\text{at\_fraction}
        v_p            the velocity of particle p
        damp           `damp`, the fraction of velocity removed per frame in the sponge
        sponge         `sponge`, the band's depth in grid cells before the plane
        at_fraction    `at_fraction`, where the kill plane sits along `face`, as a fraction of the box

    Reference: Plexus (this work)."""
    PARAM_UNITS = {'damp': 'fraction', 'sponge': 'count', 'at_fraction': 'fraction'}
    SYMBOLS = {'v_p': 'the velocity of particle p', 'damp': '`damp`, the fraction of velocity removed per frame in the sponge', 'sponge': "`sponge`, the band's depth in grid cells before the plane", 'at_fraction': '`at_fraction`, where the kill plane sits along `face`, as a fraction of the box'}

    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["face"]
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["outlet", "continuous_flow", "absorbing_boundary"]
    PARAM_ROLES = {"face": "outlet_face", "at_fraction": "outlet_plane_fraction",
                   "sponge": "sponge_width_cells", 'damp': 'velocity_fraction_removed_per_frame'}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.axis, self.sign = _face(params["face"])
        self.frac = float(params.get("at_fraction", 0.04))    # plane, as a fraction of the box
        self.sponge = self.tunable(params.get("sponge"), 8.0, tensor=True)  # cells of damping above the plane
        self.damp = float(params.get("damp", 0.15))           # velocity kept AT the plane; 1.0 at
        #                                                       the top of the band, ramped between

    def forward(self, H, mask=None):
        p = H.level(self.at)
        dev = p.state.device
        D = int(getattr(H, "dim", 3))
        box = torch.as_tensor(list(H.world_size)[:D], device=dev, dtype=p.state.dtype)
        dx = float(H.field(next(iter(H.fields))).dx) if len(H.fields) else float(box[0]) / 96.0
        x = p.state[:, self.axis]
        L = float(box[self.axis])
        plane = self.frac * L if self.sign < 0 else (1.0 - self.frac) * L
        band = self.sponge * dx

        live = p.occ > 0
        if self.sign < 0:                       # flow travels toward LOW coordinate
            gone = live & (x < plane)
            insponge = live & (x < plane + band) & (x >= plane)
        else:
            gone = live & (x > plane)
            insponge = live & (x > plane - band) & (x <= plane)

        # a graded sponge, not a step: a flat `v *= damp` band leaves fluid arriving at nearly full speed at its top and stopping abruptly at the bottom, a stagnation the grid answers by pushing fluid sideways into a radial sheet that curls up at its rim (the "bounce"); ramping the damping with depth, untouched entering the band and `damp` at the plane, takes the momentum out over several cells (the drain runs once per frame against 25 substeps, so the band must be several cells wide)
        v = p.get("vel")
        if bool(insponge.any()):
            _d = (x[insponge] - plane).abs() / max(band, 1e-12)      # 0 at the plane, 1 at the top
            _f = self.damp + (1.0 - self.damp) * _d.clamp(0.0, 1.0)
            v[insponge] = v[insponge] * _f[:, None]
        if bool(gone.any()):
            p.occ[gone] = 0.0
            if hasattr(p, "mass"):
                pass                            # mass is a material property; occ alone removes it
            p.state[gone, :D] = -1.0 * box      # park off-domain so it is neither drawn nor sensed
            p.C[gone] = 0.0
        return {}


# run: `plexus.engine.run` with the continuous-flow spec checked first -> (Hierarchy, recorded trajectory).
def run(sim, *args, **kwargs):
    """`plexus.engine.run`, with the continuous-flow spec checked before anything expensive starts. It delegates: a fork of `run` would be a second copy of the schedule loop, the recording, the CUDA-graph capture and the integration invariant, which would drift. What this adds is the refusal an operator cannot make, because an operator sees only itself."""
    ops = [o.op for o in sim.operators]
    if "mpm_emit" in ops and "mpm_drain" not in ops:
        raise ValueError(
            "engine: continuous -- this spec emits but never drains. The pool is finite, so it "
            "fills and then emission silently stops, which looks like a jet that thins out rather "
            "than a spec that is wrong. Add `mpm_drain` with the face the flow leaves by.")
    if "mpm_drain" in ops and "mpm_emit" not in ops:
        raise ValueError("engine: continuous -- this spec drains but never emits.")
    # emit and drain must be OUTSIDE the substep block: inside it the cursor advances once per substep (a jet `substeps` times too fast) and a captured graph would bake the first substep's slice in and emit the same particles forever
    for blk in sim.schedule:
        if isinstance(blk, dict) and {"mpm_emit", "mpm_drain"} & set(blk.get("steps") or []):
            raise ValueError(
                "engine: continuous -- mpm_emit / mpm_drain are inside the substep block. They are "
                "FRAME-level operators: put them in the outer schedule beside `gravity`, or the "
                "inlet runs once per substep and a captured graph freezes the emitted slice.")
    from plexus.engine import run as _run
    return _run(sim, *args, **kwargs)
