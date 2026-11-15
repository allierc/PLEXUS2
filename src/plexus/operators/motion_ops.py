"""Single-body motion: how one element moves when nothing else is acting on it.

Every contract here reads one element's own state and writes one element's own delta, except the
last three, the boids rules, which read the neighbour relation of a `radius_graph` (restored here
on 2026-11-09, Cedric: "put them in motion_ops"; they were dropped from `interaction_ops` in
curation step 3 on 2026-10-27).

In the order they appear below:

    drag             lateral   velocity-proportional damping: the overdamped limit's other half
    glide            lateral   move along the heading at the type's own speed
    sediment         lateral   a constant settling drift
    attractor_flow   lateral   ride a prescribed chaotic vector field, one of ten systems
    velocity_cruise  lateral   relax the speed toward a target without turning
    bounce           lateral   the wall and obstacle response: reflect the heading
    gravity          lateral   a uniform body force
    cohesion         lateral   steer toward the neighbours' centre           (boids)
    separation       lateral   steer away from close neighbours              (boids)
    velocity_align   lateral   steer toward the mean neighbour velocity      (boids, Vicsek)
"""
from __future__ import annotations
import torch
from plexus.geometry import minimum_image, neighbour_mean
from plexus.models.base import Lateral
from plexus.models.registry import register_operator


def _rod_edges(lvl):
    """The chain's segment table, or the one its contiguous layout implies.

    `drag [along: chain]` is the one place outside `rod_ops` that needs to know which nodes are
    joined, and it used to answer that from row adjacency. It asks the table instead, so that a
    ring, a branch or a cross-linked axoneme gets the right tangent -- and so there is ONE
    statement of the rod's topology rather than two that can disagree. The fallback builds the
    chain table a bare `n_rod`/`per` set implies, so a spec that never declared a `mesh:` is
    unaffected.
    """
    m = getattr(lvl, "mesh", None)
    if m is not None and "E_srce" in m and m["E_srce"].numel():
        return m["E_srce"], m["E_trgt"], m.get("rest"), m.get("valence")
    n_rod = int(getattr(lvl, "n_rod", 1))
    per = lvl.n // max(n_rod, 1)
    idx = torch.arange(lvl.n, device=lvl.state.device, dtype=torch.long)
    srce = idx[(idx % per) < (per - 1)]
    return srce, srce + 1, None, None


@register_operator("drag", family="motion", set="particle", kind="lateral",
                   equation=r"""$$\mathbf a_i \;=\; -\,k\,\mathbf v_i$$""", title="slow motion by viscous drag")
class Drag(Lateral):
    r"""viscous drag, a force opposing a particle's velocity and proportional to it, against a still fluid or a grid's flow, isotropic or resolved about a chain's tangent.
    particle -> particle: reads vel (and the named grid's velocity with `field`), emits an acceleration (a velocity with `emit: velocity`); `react: true` puts the equal and opposite momentum back on the grid.

        \mathbf a_i \;=\; -\,k\,\mathbf v_i
        a_i            the acceleration of particle i, in world length per time squared
        k              the drag coefficient, per time (1/k is the time to lose 1 - 1/e of the speed)
        v_i            the velocity of particle i relative to the fluid (the grid's interpolated velocity u when `field` is set), in world length per time

    Reference: Stokes, G. G. (1851). On the effect of the internal friction of fluids on the motion of pendulums. Trans. Camb. Phil. Soc. 9:8-106. Anisotropy: Gray, J. & Hancock, G. J. (1955). J. Exp. Biol. 32:802 (resistive force theory)."""
    PARAM_UNITS = {'k': 'rate', 'noise': 'acceleration', 'ratio': 'dimensionless', 'zeta': 'mass_per_time', 'coupling': 'fraction'}
    SYMBOLS = {'a_i': 'the acceleration of particle i, in world length per time squared', 'k': 'the drag coefficient, per time (1/k is the time to lose 1 - 1/e of the speed)', 'v_i': "the velocity of particle i relative to the fluid (the grid's interpolated velocity u when `field` is set), in world length per time"}

    EMIT = "acceleration"            # second-order: a force on a body that has inertia
    SUPPORTED_DIMS = [2, 3]                      # acts on the D-vector velocity, dimension-generic
    REQUIRES_PARAMS = ["k"]                     # drag coefficient
    MECHANISM_TAGS = ["viscous_drag", "friction", "damping", "resistive_force_theory"]
    PARAM_ROLES = {"k": "drag_coefficient", "noise": "thermal_noise", "zeta": "fluid_drag_coefficient_of_the_reaction_force_on_the_grid",
                   "along": "chain_to_make_the_drag_anisotropic_about_the_local_tangent",
                   "ratio": "transverse_over_tangential_drag_about_2_for_a_slender_filament",
                   "field": "grid_whose_velocity_the_drag_is_measured_against",
                   "react": "put_the_equal_and_opposite_back_into_that_fluid",
                   "coupling": "dial_the_fluid_structure_exchange_from_0_to_1",
                   "react_set": "the_fluid_s_particle_set_that_takes_the_reaction"}
    REFERENCE = ("Stokes, G. G. (1851). On the effect of the internal friction of fluids on "
                 "the motion of pendulums. Trans. Camb. Phil. Soc. 9:8-106.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.k = self.tunable(params["k"], None, tensor=True)
        self.noise = float(params.get("noise", 0.0))     # isotropic Langevin noise (off by default)
        self.at = params.get("_at", "particle")
        self.along = str(params.get("along", "") or "").lower()
        if self.along not in ("", "chain"):
            raise ValueError(f"drag: `along` is `chain` or nothing, got {self.along!r}")
        self.ratio = float(params.get("ratio", 2.0))
        self.field = params.get("field")
        self.field = str(self.field) if self.field else None
        self.react = bool(params.get("react", False))
        self.react_set = params.get("react_set")
        self.react_set = str(self.react_set) if self.react_set else None
        # ---- THE OVERDAMPED BRANCH -----------------------------------------------------------
        # `emit: velocity` on this operator says the ROD has no inertia, which is what a cilium at
        # Reynolds number 1e-5 actually is (see builder step 0343: a real Platynereis cilium is
        # overdamped by a factor of 6.1 million, and carrying that explicitly would need a drag of
        # 6.1e7 per second and a 3.3e-08 s timestep). In that regime this operator's job changes
        # completely. It no longer applies a drag FORCE -- the rod's own mobility already is the
        # drag, since `rod_elastic` and `rod_motor` emit f/zeta directly -- so all that is left for
        # it to do on the rod is ADVECTION: the overdamped law in a moving fluid is
        #
        #     v_rod = u + f_internal / zeta
        #
        # and the operators supply the second term, so this one supplies `u`. The rod's stored
        # velocity is then u + f/zeta, which makes `v_rod - u` exactly f/zeta and gives the
        # reaction on the fluid for free: f_on_fluid = -zeta (v_rod - u).
        #
        # `zeta` IS A DRAG COEFFICIENT IN SIM MASS PER SECOND, not the per-unit-mass `k` above, and
        # it has to be because the reaction is a FORCE handed to a set whose masses are real. For
        # the 8-node Platynereis rod it is zeta_perp x h / (1 sim mass) = 2.0974e-03 Pa s x 2.857
        # um / 1.25e-13 kg = 4.794e+04. The check that it is the right number is that the rod's
        # own mobility already encodes the same zeta: k_bend = B/(zeta_perp h^4) = 4150 per second,
        # which is what run 0347 used, to a ratio of 1.000.
        self.zeta = float(params.get("zeta", 0.0))
        # `coupling` DIALS THE FLUID-STRUCTURE EXCHANGE FROM 0 TO 1, and it exists because going
        # from an uncoupled rod straight to a fully coupled one changes two things at once -- the
        # rod starts feeling the flow AND the flow starts feeling the rod -- so a run that breaks
        # cannot say which. At 0 this operator is inert in both directions and the spec is exactly
        # its own baseline; at 1 it is the physical coupling. Anything between is a weaker fluid,
        # not a different model: it scales the advection the rod feels and the reaction the fluid
        # receives by the SAME factor, so momentum still balances exactly and the residual stays at
        # machine precision whatever the dial is set to.
        self.coupling = float(params.get("coupling", 1.0))
        if not 0.0 <= self.coupling <= 1.0:
            raise ValueError(f"drag: coupling must be in [0, 1], got {self.coupling}")
        self.overdamped = str(params.get("emit", "") or "").lower() == "velocity"
        if self.overdamped and self.react and self.zeta <= 0.0:
            raise ValueError(
                "drag [emit: velocity] hands a FORCE back to the fluid, so it needs `zeta:`, the "
                "node drag coefficient in sim mass per second. The per-unit-mass `k` cannot serve: "
                "an overdamped rod has no mass for it to be per unit of.")
        if self.react and not self.field:
            raise ValueError("drag: `react` needs a `field` to react against -- there is nothing "
                             "to push back on otherwise.")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        occ = lvl.occ
        V = lvl.get("vel")
        # THE FLUID'S OWN VELOCITY AT THIS PARTICLE, by the same B-spline the MPM transfer uses,
        # so the rod feels the flow it is actually sitting in rather than a still medium.
        _u = None
        if self.field is not None:
            from plexus.operators.mpm_ops import bspline, stencil_offsets
            g = H.fields[self.field]
            D = V.shape[1]
            off = stencil_offsets(D, V.device)
            inv_dx = 1.0 / float(g.dx)
            _, w, flat = bspline(lvl.get("pos"), inv_dx, off, g.shape,
                                 bool(getattr(H, "periodic", False)))
            _u = (w[..., None] * g.v[flat].view(lvl.n, off.shape[0], D)).sum(1)
            _u = torch.nan_to_num(_u)
            V = V - _u
        if self.along == "chain":
            # THE LOCAL TANGENT, READ OFF THE SEGMENT TABLE rather than off adjacent rows of the
            # state array. Every segment (i, j) hands its vector e = x_j - x_i to BOTH its ends, so
            # an interior node of a chain accumulates e_prev + e_next = x_{i+1} - x_{i-1} and an end
            # node accumulates its one segment -- which is exactly the centred/one-sided difference
            # this used to write out by slicing, and is now also defined on a ring, a branch or a
            # cross-linked axoneme, none of which have a "previous row".
            X = lvl.get("pos")
            i, j, _, _ = _rod_edges(lvl)
            e = X[j] - X[i]
            t = torch.zeros_like(X)
            t.index_add_(0, i, e)
            t.index_add_(0, j, e)
            t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-12)
            self._tan = t                                  # reused by the overdamped anisotropy below
            v_par = (V * t).sum(1, keepdim=True) * t
            acc = -self.k * (v_par + self.ratio * (V - v_par)) * occ[:, None]
        else:
            self._tan = None
            acc = -self.k * V * occ[:, None]
        if self.noise > 0.0:                             # drag + noise = a Brownian/Langevin bath
            N, D = acc.shape
            acc = acc + self.noise * torch.randn(N, D, generator=getattr(H, "rng", None),
                                                 device=acc.device) * occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        # ---- OVERDAMPED: the rod is ADVECTED, and the force it hands the fluid is zeta * (v - u).
        # `acc` above computed a drag force per unit mass, which an inertialess rod has no use for:
        # its mobility already IS the drag. What it still needs is the fluid's own motion, so the
        # emitted quantity becomes `u` itself -- the engine reads it as dx/dt and the rod is carried
        # by the flow on top of the f/zeta its own operators supply. `dv` is then exactly f/zeta,
        # so the reaction is a force this operator can state without ever seeing f.
        if self.overdamped:
            V_rod = lvl.get("vel").to(acc.dtype)
            dv = V_rod - (_u if _u is not None else 0.0)
            acc = (_u if _u is not None else torch.zeros_like(V_rod)) * (self.coupling * occ[:, None])
            if mask is not None:
                acc = acc * mask[:, None].to(acc.dtype)
            # THE SIGN, AND IT IS THE ONE THING HERE THAT CANNOT BE GUESSED. `_od_force` has to be
            # the force ON THE ROD, because the reaction scatter below negates whatever it is given
            # -- in the inertial branch it receives m_p * acc, and `acc` is already -k (v - u), so
            # the negation hands the fluid +zeta (v - u), which is the drag the rod exerts on it.
            # Writing +zeta * dv here instead handed the fluid -zeta (v - u): the water was pushed
            # AWAY from the rod's motion, which is anti-drag, so the relative velocity grew, the
            # push grew with it, and the loop ran away. Measured on cil_s31_odwater2 -- the
            # coupling itself was exact (residual 4e-08 with 3,298 fluid particles touched at
            # frame 1) and the rod still left the domain, ending at |pos| 1.01e+04 with the box at
            # 0 to 1 while holding its shape at 19.95 um of 20. A sign, not a magnitude.
            # ANISOTROPIC DRAG, Gray-Hancock: a slender filament resists NORMAL motion about twice
            # as hard as TANGENTIAL (`ratio` = zeta_perp/zeta_par ~ 2). The overdamped branch was
            # isotropic -- every rate carried the one zeta -- and run 0473 measured the consequence:
            # a released cell whose cilia beat perfectly did not swim, because a periodic stroke
            # with isotropic drag pumps no net fluid (the mean force over a beat is zero). The thrust
            # of a cilium IS this anisotropy: broadside on the power stroke (full zeta), edge-on on
            # recovery (zeta/ratio), so the two half-strokes push unequal fluid and the net is
            # forward. Applied ONLY when the drag declares `along: chain` (the filament flag); every
            # existing overdamped spec leaves it unset and is unchanged, and `ratio: 1` with the flag
            # ON is the paddle control that must not swim. The fluid reaction is scaled, its equal
            # and opposite still lands on the same water, so the momentum residual stays at zero.
            dvf = dv
            if getattr(self, "_tan", None) is not None and self.ratio != 1.0:
                dv_par = (dv * self._tan).sum(1, keepdim=True) * self._tan
                dvf = dv - (1.0 - 1.0 / self.ratio) * dv_par     # tangential drag reduced by ratio
            self._od_force = (-self.zeta * self.coupling) * dvf * occ[:, None]
        if self.react and _u is not None and self.react_set:
            # THE REACTION GOES BACK AS A DELTA ON THE FLUID'S OWN SET, not as a write to the grid,
            # and the difference is the whole reason the first two takes returned NaN.
            #
            # The grid is REBUILT from scratch by `mpm_scatter` every substep, so anything written
            # into g.v is erased before the next one reads it. Worse, an engine-integrated operator
            # placed inside a substep block is EVALUATED every substep but INTEGRATED once per tick
            # (engine.py:2695, `_integrate(H, sim.dt)`), so the rod moved once per frame while its
            # reaction was applied to the grid sixty-one times.
            #
            # `mpm_scatter` reads its own set's delta as a body acceleration every substep
            # (`a_ext = a_ext + H.delta(p.name)`, mpm_ops.py:552) and the engine snapshots and
            # restores deltas around the substep loop, so a delta returned HERE is seen once per
            # substep at its own value -- which is what an acceleration means. One force, applied
            # once, through a route that already existed.
            #
            # Localised by the same B-spline both ways: the rod's force is scattered onto the grid
            # nodes it overlaps, then gathered onto whichever fluid particles overlap those nodes.
            b = H.level(self.react_set)
            m_p = getattr(lvl, "mass", None)
            m_p = torch.ones(lvl.n, device=acc.device) if m_p is None else m_p.to(acc.dtype)
            f_node = torch.zeros_like(g.v)
            # WHICH FORCE IS HANDED BACK. Inertial: minus the drag force the rod felt, m_p * acc.
            # Overdamped: minus zeta (v_rod - u), computed above -- `acc` no longer holds a force
            # in that branch, it holds the advection velocity, so reusing it here would hand the
            # fluid a quantity with the wrong units AND the wrong sign.
            Xb = b.get("pos")
            _, wb, fb = bspline(Xb, inv_dx, off, g.shape, bool(getattr(H, "periodic", False)))
            m_b = getattr(b, "mass", None)
            m_b = torch.ones(b.n, device=acc.device) if m_b is None else m_b.to(acc.dtype)
            # a node's force is shared among the particles that overlap it, weighted as they are
            n_node = torch.zeros(g.v.shape[0], device=acc.device)
            n_node.index_add_(0, fb, wb.reshape(-1))
            # ---- THE IMPLICIT EXCHANGE, which is what lets the full coupling run at all -------
            # The explicit form hands the fluid zeta (v - u) dt every step, and that is unstable
            # whenever the fluid a node pushes on is light enough to overshoot within the step:
            # the condition is zeta dt / m_local < 1. Here m_local is the water mass under a rod
            # node's 3x3x3 stencil, about 0.24 sim-mass (8 particles per grid cell), so the ratio
            # is 1.96 at coupling 1.0 and 0.20 at 0.1 -- measured, and exactly why 0.1 "worked"
            # while 1.0 sent the rod to |pos| 1.4e25 with the startup ramp already in place
            # (builder 0442). The remedy is not a smaller dt (5x cost) but the implicit-Euler
            # solution of the two-body exchange within the step,
            #
            #     u_new = (u + (zeta dt/m) v) / (1 + zeta dt/m)
            #     force to the fluid = zeta (v - u) / (1 + zeta dt / m_local)
            #
            # which is unconditionally stable, identical to the explicit form when the ratio is
            # small, and NOT a fudge: it is the same physics integrated implicitly. The factor is
            # applied to `_od_force` in place so the momentum residual below is computed against
            # the force the fluid actually received. The smallest factor is printed with the
            # residual, because a factor well below 1 means the grid is too coarse for this
            # zeta and the run is leaning on the integrator.
            _f_src = self._od_force if self.overdamped else (m_p[:, None] * acc)
            self._implicit_min = 1.0
            if self.overdamped and self.zeta > 0.0:
                mass_node = torch.zeros(g.v.shape[0], device=acc.device, dtype=acc.dtype)
                mass_node.index_add_(0, fb, (wb * m_b[:, None]).reshape(-1))
                m_local = (w * mass_node[flat].view(w.shape)).sum(1).clamp_min(1e-12)   # per rod node
                _dt = float(getattr(H, "dt", 0.0))
                fac = 1.0 / (1.0 + (self.coupling * self.zeta * _dt) / m_local)
                self._od_force = self._od_force * fac[:, None]
                _f_src = self._od_force
                self._implicit_min = float(fac.min())
            f_node.index_add_(0, flat,
                              ((-_f_src)[:, None, :] * w[..., None]
                               ).reshape(-1, _f_src.shape[1]))
            share = f_node[fb] / n_node[fb].clamp_min(1e-12)[:, None]
            a_b = (wb[..., None] * share.view(b.n, off.shape[0], -1)).sum(1) / m_b[:, None]
            a_b = torch.nan_to_num(a_b)

            # ---- THE GATE: is momentum actually exchanged? --------------------------------------
            # Copied in spirit from `mesh_contact._record`, which has measured its own exchange
            # this way all along: the force the rod takes from the fluid and the force handed back
            # to the fluid must cancel, so
            #
            #     residual = |sum f_rod + sum f_water| / sum |f_rod|
            #
            # is dimensionless, is 0 for an exact exchange, and is 1 when the reaction never landed.
            # THIS IS THE NUMBER THE COUPLING IS JUDGED ON, and it exists because the previous three
            # attempts were judged on a movie instead: the water was bit-identical across 46,612,200
            # velocity components while the printed accelerations looked healthy, which a residual
            # would have caught in one frame. `n_react` counts the water particles that received
            # anything at all -- a residual of 0 with n_react 0 means the rod felt no drag either,
            # which is a different failure wearing the same number.
            f_rod = self._od_force if self.overdamped else (m_p[:, None] * acc)
            f_wat = m_b[:, None] * a_b
            den = f_rod.abs().sum().clamp_min(1e-30)
            resid = float((f_rod.sum(0) + f_wat.sum(0)).abs().sum() / den)
            n_react = int((a_b.abs().sum(1) > 0).sum())
            self._resid = max(getattr(self, "_resid", 0.0), resid)
            if int(getattr(H, "frame", 0)) in (1, 5, 20, 50, 200, 500, 800):
                print(f"[drag f{int(getattr(H, 'frame', 0))}] react -> {self.react_set}: "
                      f"momentum residual {resid:.3e} (0 = exact, 1 = the reaction never landed); "
                      f"{n_react} of {b.n} fluid particles touched; "
                      f"|f_rod| sum {float(den):.3e}, |f_water| sum {float(f_wat.abs().sum()):.3e}; "
                      f"|a_water| max {float(a_b.abs().max()):.3e}; implicit factor min {self._implicit_min:.3f}; "
                      f"fluid u max {float(_u.abs().max()):.3e}", flush=True)
            out_extra = {self.react_set: a_b}
        else:
            out_extra = {}
        out_extra[self.at] = acc
        return out_extra


def _halvorsen(x, y, z, p):
    a = p.get("a", 1.4)
    return (-a * x - 4.0 * y - 4.0 * z - y * y,
            -a * y - 4.0 * z - 4.0 * x - z * z,
            -a * z - 4.0 * x - 4.0 * y - x * x)


def _lorenz(x, y, z, p):
    s = p.get("sigma", 10.0); r = p.get("rho", 28.0); b = p.get("beta", 8.0 / 3.0)
    return (s * (y - x), x * (r - z) - y, x * y - b * z)


def _aizawa(x, y, z, p):
    a = p.get("a", 0.95); b = p.get("b", 0.7); c = p.get("c", 0.6)
    d = p.get("d", 3.5); e = p.get("e", 0.25); f = p.get("f", 0.1)
    return ((z - b) * x - d * y,
            d * x + (z - b) * y,
            c + a * z - (z ** 3) / 3.0 - (x * x + y * y) * (1.0 + e * z) + f * z * (x ** 3))


def _sprott_b(x, y, z, p):
    a = p.get("a", 1.0)
    return (a * y * z, x - y, 1.0 - x * y)


def _thomas(x, y, z, p):
    b = p.get("b", 0.208)
    return (torch.sin(y) - b * x, torch.sin(z) - b * y, torch.sin(x) - b * z)


def _rossler(x, y, z, p):
    a = p.get("a", 0.2); b = p.get("b", 0.2); c = p.get("c", 5.7)
    return (-y - z, x + a * y, b + z * (x - c))


def _dadras(x, y, z, p):
    a = p.get("a", 3.0); b = p.get("b", 2.7); c = p.get("c", 1.7)
    d = p.get("d", 2.0); e = p.get("e", 9.0)
    return (y - a * x + b * y * z, c * y - x * z + z, d * x * y - e * z)


def _chen(x, y, z, p):
    a = p.get("a", 35.0); b = p.get("b", 3.0); c = p.get("c", 28.0)
    return (a * (y - x), (c - a) * x - x * z + c * y, x * y - b * z)


def _chua(x, y, z, p):
    a = p.get("alpha", 15.6); b = p.get("beta", 28.58)
    m0 = p.get("m0", -1.1428571); m1 = p.get("m1", -0.7142857)
    h = m1 * x + 0.5 * (m0 - m1) * (torch.abs(x + 1.0) - torch.abs(x - 1.0))
    return (a * (y - x - h), x - y + z, -b * y)


def _rabinovich_fabrikant(x, y, z, p):
    al = p.get("alpha", 1.1); g = p.get("gamma", 0.87)
    return (y * (z - 1.0 + x * x) + g * x,
            x * (3.0 * z + 1.0 - x * x) + g * y,
            -2.0 * z * (al + x * y))


_FIELDS = {
    "halvorsen": _halvorsen, "lorenz": _lorenz, "aizawa": _aizawa, "sprott_b": _sprott_b,
    "thomas": _thomas, "rossler": _rossler, "dadras": _dadras, "chen": _chen, "chua": _chua,
    "rabinovich_fabrikant": _rabinovich_fabrikant,
}
ATTRACTOR_SYSTEMS = tuple(_FIELDS)


def attractor_velocity(system: str, pos: torch.Tensor, params: dict | None = None) -> torch.Tensor:
    """The strange-attractor vector field f(x): dx/dt for every point of `pos` [N, 3].
    `system` is one of `ATTRACTOR_SYSTEMS`; `params` overrides that system's constants.
    Returns a velocity tensor [N, 3] on the same device/dtype as `pos`."""
    if system not in _FIELDS:
        raise ValueError(f"attractor_flow: unknown system {system!r}; "
                         f"choose one of {list(ATTRACTOR_SYSTEMS)}")
    p = params or {}
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    xd, yd, zd = _FIELDS[system](x, y, z, p)
    return torch.stack([xd, yd, zd], dim=-1)


def _in_obstacles(x, y, obstacles):
    """Bool mask: is (x, y) inside any obstacle? rect=[x0,y0,x1,y1], disc=[cx,cy,r]."""
    hit = torch.zeros_like(x, dtype=torch.bool)
    for o in (obstacles or []):
        if len(o) == 4:
            x0, y0, x1, y1 = o
            hit = hit | ((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1))
        elif len(o) == 3:
            cx, cy, r = o
            hit = hit | (((x - cx) ** 2 + (y - cy) ** 2) <= r * r)
    return hit


def _random_unit(n, D, rng, device):
    """A random unit vector per agent [n, D] (isotropic re-heading off obstacles)."""
    v = torch.randn(n, D, generator=rng, device=device)
    return v / v.norm(dim=1, keepdim=True).clamp(min=1e-9)


@register_operator("gravity", family="mechanics", set="cell", kind="lateral", title="apply uniform gravity",
                   equation=r"""$$\mathbf a_{\text{ext}} \;=\; \mathbf g \qquad (\text{default } \mathbf g=(0,-g_y))$$""")
class GravityOperator(Lateral):
    r"""a uniform body force, the same acceleration on every element whatever its state.
    cell -> cell: reads nothing, emits the acceleration `mpm_acceleration` the MPM substep consumes as its external body force (not engine-integrated: a cell here is a centroid read off its material points).

        \mathbf a_{\text{ext}} \;=\; \mathbf g \qquad (\text{default } \mathbf g=(0,-g_y))
        a_ext          the external acceleration handed to the MPM substep, in world length per time squared
        g              the gravity vector (gx, gy, gz), in world length per time squared
        g_y            its y component; `g` alone sets gy = -g, the fall of a 2D scene (a 3D scene with z up writes gy 0 and gz -g)

    Reference: Newton, I. (1687). Philosophiae Naturalis Principia Mathematica."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"A uniform body force: the same acceleration $\mathbf a_{\text{ext}}=\mathbf g$ on every point, whatever its "
        r"state, with $\mathbf g$ the gravity vector (`g` world lengths per time squared, default 10, pointing along $-y$ "
        r"unless `gx`, `gy`, `gz` say otherwise). It moves nothing itself: it hands $\mathbf a_{\text{ext}}$ to the MPM "
        r"substep, whose scatter adds it to each point's velocity."
    )
    PARAM_UNITS = {'g': 'acceleration', 'gx': 'acceleration', 'gy': 'acceleration', 'gz': 'acceleration'}
    SYMBOLS = {'a_ext': 'the external acceleration handed to the MPM substep, in world length per time squared', 'g': 'the gravity vector (gx, gy, gz), in world length per time squared', 'g_y': 'its y component; `g` alone sets gy = -g, the fall of a 2D scene (a 3D scene with z up writes gy 0 and gz -g)'}

    EMIT = "mpm_acceleration"                  # consumed by the MPM substep as a_ext, not engine-integrated
    SUPPORTED_DIMS = [2, 3]                           # uniform body force is dimension-generic
    REQUIRES_PARAMS = []                              # no required params — direction/magnitude optional (default -y down)
    PARAM_ROLES = {"g": "gravity_magnitude", "gx": "gravity_x", "gy": "gravity_y",
                   "gz": "gravity_z"}
    REFERENCE = "Newton, I. (1687). Philosophiae Naturalis Principia Mathematica."
    MECHANISM_TAGS = ["body_force", "uniform_acceleration"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")              # the set this acts on (engine-injected)
        self.g = float(params.get("g", 10.0))            # magnitude (world units / time^2)
        self.gx = self.tunable(params.get("gx"), 0.0, tensor=True)  # x-component (default 0)
        self.gy = self.tunable(params.get("gy"), -self.g, tensor=True)  # y-component (default -g: down)
        self.gz = self.tunable(params.get("gz"), 0.0, tensor=True)

    def forward(self, H, mask=None):
        cell = H.level(self.at)
        dev = cell.state.device
        D = int(getattr(H, "dim", 2))                    # gravity is a D-vector
        accel = torch.zeros(cell.n, D, device=dev)
        accel[:, 0] = self.gx
        accel[:, 1] = self.gy
        if D > 2:
            accel[:, 2] = self.gz
        if mask is not None:
            accel = accel * mask.float()[:, None]
        return {cell.name: accel}


@register_operator("gravity", model="edge_pull", family="mechanics", set="cell", kind="lateral",
                   title="pull the far edge of each body toward the other",
                   equation=r"""$$\mathbf a_i=a_{t(i)}\,\hat{\mathbf e}_k\ \text{if}\ i\ \text{is within}\ w\ \text{of the far end of its type along axis}\ k,\ \text{else}\ 0$$""")
class GravityEdgePull(Lateral):
    r"""a constant acceleration on the points of the far edge of each type, along one axis, so two bodies are squeezed together through their own elasticity.
    pt -> pt: emits `mpm_acceleration`.

        \mathbf a_i=a_{t(i)}\,\hat{\mathbf e}_k\quad\text{if}\quad s_{t(i)}\,(x_{i,k}-x^{\rm far}_{t(i)})<w,\qquad 0\ \text{otherwise}
        a_i            the acceleration of point i
        a_t            `accel`[t], the signed acceleration along the axis for points of type t, positive toward +x_k
        e_k            the unit vector of axis k (`axis`)
        w              `strip`, the thickness of the pulled edge, in world length
        x^far_t        the extreme of type t's points on the side the pull comes from: the lowest x_k when a_t > 0, the highest when a_t < 0
        s_t            +1 when a_t > 0, -1 when a_t < 0

    Reference: Plexus exp38 (this work, 2026-11)."""
    EXPLAIN = (r"A squeeze for two bodies of different types: points within `strip` of each body's far edge get a constant acceleration `accel` of its type along `axis`, so the pull reaches the interface through the bodies' own elasticity.")
    PARAM_UNITS = {'accel': 'acceleration', 'strip': 'length', 'axis': 'count', 'by': 'count'}
    SYMBOLS = {'a_i': 'the acceleration of point i', 'a_t': '`accel`[t], the signed acceleration along the axis for type t', 'e_k': 'the unit vector of the axis', 'w': '`strip`, the thickness of the pulled edge', 'x^far_t': "the extreme of type t's points on the side the pull comes from", 's_t': 'the sign of the pull of type t'}
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["accel", "strip"]
    PARAM_ROLES = {"accel": "signed_acceleration_per_type_along_the_axis", "strip": "thickness_of_the_pulled_edge", "axis": "axis_0_1_2", "by": "types_or_parents_which_bodies_are_pulled"}
    MECHANISM_TAGS = ["body_force", "squeeze"]
    REFERENCE = "Plexus exp38 (this work, 2026-11)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle"); self.axis = int(params.get("axis", 0)); self.strip = float(params["strip"])
        self.accel = torch.as_tensor(params["accel"], dtype=torch.float32, device=device)                 # [T]
        self.by = str(params.get("by", "types"))                                                          # the bodies are the node types, or the parent cells

    def forward(self, H, mask=None):
        P = H.level(self.at); X = P.get("pos"); D = X.shape[-1]; t = P.parent if self.by == "parents" else P.node_type; live = P.occ > 0.5
        x = X[:, self.axis].detach(); a = self.accel.to(X.dtype)[t]
        far = torch.zeros_like(x)
        for k in range(self.accel.numel()):
            m = live & (t == k)
            if m.any():
                far[t == k] = x[m].min() if float(self.accel[k]) > 0 else x[m].max()
        near = (torch.sign(a) * (x - far) < self.strip) & live
        acc = torch.zeros(X.shape[0], D, device=X.device, dtype=X.dtype); acc[:, self.axis] = a * near.to(X.dtype)
        if mask is not None:
            acc = acc * mask.float()[:, None]
        return {self.at: acc}


@register_operator("gravity", model="perlin", family="mechanics", set="cell", kind="lateral",
                   title="apply a frozen coherent-noise body force",
                   equation=r"""$$\mathbf a_{\text{ext},p}=A_c\,\big(\mathbf N(f_c\,\mathbf x^{\rm rest}_p)-\langle\mathbf N\rangle_c\big)$$""")
class GravityPerlin(Lateral):
    r"""`perlin` model of gravity: a body force that varies with where in the body an element sits, frozen in time.
    particle -> particle: reads the rest position once, emits an acceleration the MPM substep consumes (mpm_acceleration).

        \mathbf a_{\text{ext},p}=A_c\,\big(\mathbf N(f_c\,\mathbf x^{\rm rest}_p)-\langle\mathbf N\rangle_c\big)
        a_{ext,p}      the acceleration of particle p
        A_c            `amplitude` of its parent body c, an acceleration in length per time squared (one number, or one per parent)
        N              three independent Perlin noise fields
        f_c            `frequency` of body c, cycles per world length
        x^{rest}_p     the particle's rest position (frame 0)
        \langle N\rangle_c the body's mean of N, subtracted when `zero_mean` is true so nothing is pushed as a whole

    Reference: Perlin, K. (1985). An image synthesizer. ACM SIGGRAPH Computer Graphics 19(3):287-296; Perlin, K. (2002). Improving noise. ACM Trans. Graph. 21(3):681-682."""
    PARAM_UNITS = {'amplitude': 'acceleration', 'frequency': 'dimensionless', 'seed': 'count'}
    SYMBOLS = {'a_{ext,p}': 'the acceleration of particle p', 'A_c': '`amplitude` of its parent body c, an acceleration in length per time squared (one number, or one per parent)', 'N': 'three independent Perlin noise fields', 'f_c': '`frequency` of body c, cycles per world length', 'x^{rest}_p': "the particle's rest position (frame 0)", '\\langle N\\rangle_c': "the body's mean of N, subtracted when `zero_mean` is true so nothing is pushed as a whole"}

    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["amplitude", "frequency"]
    MECHANISM_TAGS = ["body_force", "coherent_noise", "material_frame"]
    PARAM_ROLES = {"amplitude": "acceleration_scale_per_body", "frequency": "noise_cycles_per_length_per_body",
                   "zero_mean": "no_net_force_or_torque_on_a_body", "seed": "lattice_offset"}
    REFERENCE = "Perlin, K. (1985). ACM SIGGRAPH Computer Graphics 19(3):287-296; Perlin, K. (2002). ACM Trans. Graph. 21(3):681-682."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        a, f = params["amplitude"], params["frequency"]
        self.amplitude = [float(v) for v in a] if isinstance(a, (list, tuple)) else float(a)
        self.frequency = [float(v) for v in f] if isinstance(f, (list, tuple)) else float(f)
        self.zero_mean = bool(params.get("zero_mean", True))
        self.seed = float(params.get("seed", 0.0))
        self._acc = None

    def _per_parent(self, value, n_par, par, dev):
        """A per-particle tensor from a scalar or a per-parent-cell list."""
        if isinstance(value, list):
            if par is None:
                raise ValueError("gravity[perlin]: a per-body list needs a set with a parent")
            if len(value) != n_par:
                raise ValueError(f"gravity[perlin]: {len(value)} values for {n_par} parent cells")
            return torch.as_tensor(value, dtype=torch.float32, device=dev)[par]
        return torch.full((par.shape[0] if par is not None else 0,), float(value), device=dev)

    def _init(self, lvl, H):
        import numpy as np
        from plexus.operators.mpm_ops import _PERLIN_CORNERS, _PERLIN_GRAD4, _PERLIN_PERM, perlin4
        x = lvl.get("pos").detach()
        dev = x.device
        par = getattr(lvl, "parent", None)
        par = par.to(torch.long) if par is not None else None
        pname = getattr(lvl, "parent_name", None) or "cell"
        n_par = int(H.level(pname).n) if (par is not None and pname in H.levels) else (int(par.max()) + 1 if par is not None else 0)
        N = x.shape[0]
        A = self._per_parent(self.amplitude, n_par, par, dev) if par is not None else torch.full((N,), float(self.amplitude), device=dev)
        f = self._per_parent(self.frequency, n_par, par, dev) if par is not None else torch.full((N,), float(self.frequency), device=dev)
        perm = torch.as_tensor(np.concatenate([_PERLIN_PERM, _PERLIN_PERM]), device=dev)
        grad = torch.as_tensor(_PERLIN_GRAD4, device=dev)
        corners = torch.as_tensor(_PERLIN_CORNERS, device=dev)
        P = x * f[:, None] + self.seed
        comps = [perlin4(torch.cat([P, torch.full((N, 1), 17.31 * (k + 1), device=dev)], 1), perm, grad, corners)
                 for k in range(3)]
        n3 = torch.stack(comps, 1)                               # [N, 3]
        occ = getattr(lvl, "occ", None)
        live = (occ > 0).float() if occ is not None else torch.ones(N, device=dev)
        # NOT BEFORE THE SEED. The GUI calls every operator once on the built, unseeded hierarchy; there every
        # position is 0, a lattice point, where Perlin noise is exactly 0 -- and a force of zeros was cached for the
        # whole run (exp01 step 0007, 2026-10-27: sixteen balls that never moved). A rest state with no live element
        # or no spread is not a rest state: nothing is cached, zeros are returned, the next call looks again.
        self._valid = bool(float(live.sum()) > 0 and float(x[live > 0].std()) > 0) if N else False
        if not self._valid:
            self._acc = torch.zeros(N, 3, device=dev)
            return
        if self.zero_mean and par is not None:
            # NO RIGID MODE LEFT IN THE FORCE. Subtracting the mean removes the push; what remains still carries a
            # net TORQUE, and a free elastic ball resists spin with nothing, so the whole field went into rigid
            # rotation (exp01 step 0007 re-run, 2026-10-27: every particle moved 5-7 mm, every ball stayed round).
            # Per body: a <- a - <a>, then a <- a - alpha x r with alpha = I^-1 sum r x a, I = sum(|r|^2 1 - r r^T)
            # about the body's centre -- the rigid rotation of the field, projected out. What is left can only deform.
            s = torch.zeros(n_par, 3, device=dev).index_add_(0, par, n3 * live[:, None])
            c = torch.zeros(n_par, device=dev).index_add_(0, par, live).clamp(min=1.0)
            n3 = n3 - (s / c[:, None])[par]
            xc = torch.zeros(n_par, 3, device=dev).index_add_(0, par, x * live[:, None]) / c[:, None]
            r = (x - xc[par]) * live[:, None]
            L = torch.zeros(n_par, 3, device=dev).index_add_(0, par, torch.cross(r, n3, dim=1))
            rr = (r * r).sum(1)
            I = torch.zeros(n_par, 3, 3, device=dev).index_add_(
                0, par, rr[:, None, None] * torch.eye(3, device=dev) - r[:, :, None] * r[:, None, :])
            I = I + 1e-12 * rr.sum() * torch.eye(3, device=dev)
            alpha = torch.linalg.solve(I, L)                      # [n_par, 3]
            n3 = n3 - torch.cross(alpha[par], r, dim=1)
        self._acc = (A[:, None] * n3 * live[:, None]).contiguous()
        print(f"[gravity.perlin] frozen force on {int(live.sum())} live of {N} points, {n_par} bodies: "
              f"|a| mean {float(self._acc.norm(dim=1).mean()):.3g}, max {float(self._acc.norm(dim=1).max()):.3g}", flush=True)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        if self._acc is None or not getattr(self, "_valid", False) or self._acc.shape[0] != lvl.get("pos").shape[0]:
            self._init(lvl, H)
        acc = self._acc
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}


# ==========================================================================================================
# The three boids rules (Reynolds 1987): the one place in this module that reads a neighbour relation.
# ==========================================================================================================
@register_operator("cohesion", family="interaction", set="particle", kind="lateral",
                   equation=r"""$$\mathbf a_i \;=\; a_1\,w^c_i\,\big\langle \mathbf x_j-\mathbf x_i\big\rangle_{j\in\mathcal N(i)}$$""", title="steer toward the neighbours' centre")
class Cohesion(Lateral):
    """Cohesion, the first boids steering rule: accelerate toward the mean position of the
    live neighbours, which is what holds a flock together against the other two rules.

    particle -[neighbour relation]-> particle: reads pos, emits an acceleration.

        d2x_i/dt2 = a w_i  mean_j (x_j - x_i)

    The neighbour mean of (x_j - x_i) is the offset from the particle to its neighbours'
    centroid, in world units. `scale` is a, the cohesion strength, in inverse time squared
    -- unbounded in r, so this is the rule that dominates at long range. w_i is the per-type
    `cohesion` weight, dimensionless, letting one type in a mixed set flock more tightly
    than another.

    Emits an acceleration, not a velocity: a boid has momentum, and steering rules add to it.

    Reference: Reynolds, C. W. (1987). Flocks, herds and schools: a distributed behavioral
    model. SIGGRAPH Comput. Graph. 21(4):25-34.
    """

    EMIT = "acceleration"                            # second-order: a boid has momentum
    SUPPORTED_DIMS = [2, 3]                          # neighbour_mean is N-D; the rule is dimension-generic
    REQUIRES_PARAMS = []                             # no required params — `scale` optional
    REQUIRES_TYPE_PROPS = ["cohesion"]
    MECHANISM_TAGS = ["cohesion", "collective_motion"]
    PARAM_ROLES = {"scale": "cohesion_strength"}
    REFERENCE = ("Reynolds, C. W. (1987). Flocks, herds and schools: a distributed "
                 "behavioral model. SIGGRAPH Comput. Graph. 21(4):25-34.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.a = float(params.get("scale", 0.5e-5))     # PDE_B cohesion scale a1
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        w = lvl.cohesion
        acc = neighbour_mean(lvl.get("pos"), lvl.occ, lvl.edge_index,
                             getattr(H, "periodic", False),
                             getattr(H, "world_size", getattr(H, "world_width", 1.0)),
                             lambda i, j, d: w[i, None] * self.a * d)
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}


@register_operator("separation", family="interaction", set="particle", kind="lateral",
                   equation=r"""$$\mathbf a_i \;=\; -\,a_3\,w^s_i\,
\Big\langle \frac{\mathbf x_j-\mathbf x_i}{\lVert\mathbf x_j-\mathbf x_i\rVert^{2}}\Big\rangle_{j\in\mathcal N(i)}$$""", title="steer away from close neighbours")
class Separation(Lateral):
    """Separation, the second boids steering rule: accelerate away from close neighbours, which
    is what stops `cohesion` from collapsing the flock to a point.

    particle -[neighbour relation]-> particle: reads pos, emits an acceleration.

        d2x_i/dt2 = -a w_i  mean_j (x_j - x_i) / |x_j - x_i|^2

    Dividing by the squared distance makes each contribution fall off as 1/r, so the rule is
    short-range where `cohesion` is long-range, and the two balance at a preferred spacing.
    `scale` is a, the separation strength, in world units squared over time squared (it carries
    the length the 1/|d| leaves behind). w_i is the per-type `separation` weight, dimensionless.
    Division by zero is prevented upstream: the rewire's `min_radius` keeps |d| off zero.

    Reference: Reynolds, C. W. (1987). Flocks, herds and schools: a distributed behavioral
    model. SIGGRAPH Comput. Graph. 21(4):25-34.
    """

    EMIT = "acceleration"                            # second-order: a boid has momentum
    SUPPORTED_DIMS = [2, 3]                          # neighbour_mean is N-D; the rule is dimension-generic
    REQUIRES_PARAMS = []                             # no required params — `scale` optional (separation is a type prop)
    REQUIRES_TYPE_PROPS = ["separation"]
    MECHANISM_TAGS = ["short_range_repulsion", "collision_avoidance"]
    PARAM_ROLES = {"scale": "separation_strength"}
    REFERENCE = ("Reynolds, C. W. (1987). Flocks, herds and schools: a distributed "
                 "behavioral model. SIGGRAPH Comput. Graph. 21(4):25-34.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.a = float(params.get("scale", 1e-8))       # PDE_B separation scale a3
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        w = lvl.separation

        def msg(i, j, d):
            d2 = (d * d).sum(-1, keepdim=True)          # |d|^2 (> 0 via the graph's min_radius)
            return -w[i, None] * self.a * d / d2

        acc = neighbour_mean(lvl.get("pos"), lvl.occ, lvl.edge_index,
                             getattr(H, "periodic", False),
                             getattr(H, "world_size", getattr(H, "world_width", 1.0)), msg)
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}


@register_operator("velocity_align", "alignment", family="interaction", set="particle", kind="lateral",
                   equation=r"""$$\mathbf a_i \;=\; a_2\,w^a_i\,\big\langle \mathbf v_j-\mathbf v_i\big\rangle_{j\in\mathcal N(i)}$$""", title="match the neighbours' velocity")
class VelocityAlign(Lateral):
    """Velocity alignment: accelerate toward the mean velocity of the neighbours. The third
    boids steering rule, and on its own the Vicsek model -- the one term that turns a set of
    independently moving particles into a collectively moving one.

    particle -[neighbour relation]-> particle: reads pos and vel, emits an acceleration.

        d2x_i/dt2 = a  ( sum_j g_ij (v_j - v_i) ) / ( sum_j g_ij )  +  eta xi_i

    a is the alignment strength, in inverse time, since (v_j - v_i) already carries a velocity.
    g_ij is the weight the receiver gives that neighbour: `gate: none` sets every g_ij to 1, so
    all neighbours inside the rewire's radius count equally (Vicsek and boids both do this);
    `gate: contact` instead applies a smoothstep falling from 1 to 0 between (1 - softness) r
    and r, in world units, so influence fades with distance rather than stopping at a cliff.
    eta is `noise`, isotropic Vicsek noise on the alignment, and xi_i a standard normal vector;
    it is the control parameter of the order-disorder transition, and 0 by default.

    With `per_type: true` the whole term is scaled by the receiver's own `alignment` weight,
    dimensionless -- the boids special case, where types align to different degrees.

    Reference: Vicsek, T., Czirok, A., Ben-Jacob, E., Cohen, I. & Shochet, O. (1995). Novel
    type of phase transition in a system of self-driven particles. Phys. Rev. Lett.
    75:1226-1229. The per-type variant follows Reynolds, C. W. (1987). SIGGRAPH Comput.
    Graph. 21(4):25-34.
    """

    EMIT = "acceleration"                       # second-order: a boid has momentum
    SUPPORTED_DIMS = [2, 3]                     # velocity neighbour-mean is dimension-generic
    REQUIRES_PARAMS = []                        # no required params — all knobs optional (defaults in __init__)
    OPTIONAL_TYPE_PROPS = ["alignment"]        # read per-receiver only when `per_type: true` (boids)
    MECHANISM_TAGS = ["velocity_alignment", "collective_motion", "vicsek"]
    PARAM_ROLES = {"a": "alignment_strength", "gate": "neighbour_weighting",
                   "r": "contact_radius", "noise": "orientation_noise"}
    REFERENCE = ("Vicsek, T., Czirok, A., Ben-Jacob, E., Cohen, I. & Shochet, O. (1995). "
                 "Novel type of phase transition in a system of self-driven particles. "
                 "Phys. Rev. Lett. 75:1226-1229; per-type variant: Reynolds, C. W. (1987). "
                 "SIGGRAPH Comput. Graph. 21(4):25-34.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.a = float(params.get("a", params.get("scale", 5e-4)))    # alignment scale (`scale`: legacy alias)
        self.gate = str(params.get("gate", "none"))                   # "none" (Vicsek/boids) | "contact"
        self.r = float(params.get("r", 0.05))                         # contact radius (gate="contact")
        self.softness = float(params.get("softness", 0.5))            # falloff band [0,1]; 0 = hard cutoff
        self.per_type = bool(params.get("per_type", False))           # boids special case: per-type weight
        self.weight_prop = str(params.get("weight", "alignment"))     # which type property holds the weight
        self.noise = float(params.get("noise", 0.0))                  # isotropic Vicsek orientation noise (off by default)
        self.at = params.get("_at", "particle")
        if self.gate not in ("none", "contact"):
            raise ValueError(f"alignment: gate must be 'none' or 'contact', got {self.gate!r}")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos, vel, occ = lvl.get("pos"), lvl.get("vel"), lvl.occ
        N, D = vel.shape[0], vel.shape[-1]
        ei = lvl.edge_index                                 # row0 = receiver i, row1 = neighbour j
        if ei.numel() == 0:
            return {self.at: torch.zeros(N, D, device=vel.device)}
        i, j = ei[0], ei[1]

        if self.gate == "contact":                          # smoothstep contact gate: 1 -> 0 over [r_in, r]
            d = minimum_image(pos[j] - pos[i], getattr(H, "periodic", False),
                              getattr(H, "world_size", getattr(H, "world_width", 1.0)))
            dist = d.norm(dim=-1)
            r_in = (1.0 - self.softness) * self.r
            t = ((self.r - dist) / max(self.r - r_in, 1e-12)).clamp(0.0, 1.0)
            gate = t * t * (3.0 - 2.0 * t)
        else:                                               # gate="none": every neighbour equal (Vicsek/boids)
            gate = torch.ones(ei.shape[1], device=vel.device)
        w = gate * occ[j]                                   # gate + mask dormant neighbours

        msg = self.a * (vel[j] - vel[i]) * w[:, None]
        if self.per_type:                                   # boids: per-receiver alignment weight
            pw = getattr(lvl, self.weight_prop)[i]
            msg = msg * pw[:, None]
        acc = torch.zeros(N, D, device=vel.device).index_add_(0, i, msg)
        deg = torch.zeros(N, device=vel.device).index_add_(0, i, w)
        acc = (acc / deg.clamp(min=1.0)[:, None]) * occ[:, None]       # (weighted) mean over neighbours
        if self.noise > 0.0:                                           # Vicsek angular noise: order vs disorder
            acc = acc + self.noise * torch.randn(N, D, generator=getattr(H, "rng", None),
                                                 device=vel.device) * occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}
