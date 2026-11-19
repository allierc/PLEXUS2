"""MLS-MPM: a continuum solved on particles that carry the history and a grid that carries the
coupling. The grid, the four-step cycle, and the forces that act on it.

The material point method holds all the state on PARTICLES -- position, velocity, and the
deformation gradient F recording everything a piece of material remembers about its history --
and uses a background grid only as scratch, rebuilt every substep. That is what lets it handle
large deformation and self-contact without ever remeshing: the grid is where bodies that never
touch can influence one another, and the particles are where the physics lives.

One substep is four operators, in this order and no other:

    mpm_scatter      particle -> grid      mass, momentum, and the internal stress impulse
    mpm_grid_update  grid -> grid          the solve, body forces, and the wall conditions
    mpm_gather       grid -> particle      velocity, the affine velocity gradient C, advection
    mpm_strain       particle -> particle  the F update and the material's response

The substep is bounded by the CFL condition dt < dx / sqrt(E/rho): the sound speed of the
material must not cross a grid cell in one step. That ceiling, the quadratic B-spline kernel and
the stencil are all defined in this file, because they are the discretisation every operator here
shares and none of them owns.

In the order they appear below:

    mpm_grid                field       the background grid and the B-spline kernel
    mpm_scatter             exchange    step 1 of the cycle
    mpm_grid_update         field       step 2
    mpm_gather              exchange    step 3
    mpm_strain              lateral     step 4
    mpm_turgor              lateral     an isotropic outward pressure: a cell against its cortex
    mpm_viscosity           lateral     the shear dissipation a mu = 0 liquid does not have
    mpm_anchor              lateral     a spring to a rest position, for a body that must not drift
    mpm_spin                lateral     a controller driving slow solid-body rotation
    image / vector_grid     field       a scalar map and a unit-vector map, read from TIFFs
    apply_material_map      exchange    a per-particle material assignment from a map
    mls_mpm_mechanics       lateral     the FENCED oracle: the whole cycle in one operator
    active_force            exchange    an activation field -> a per-particle body force
    active_stress           exchange    an activation field -> a per-particle active stress
    label_image             field       an integer instance map, indexed and never interpolated
    seed_from_segmentation  seed        a measured segmentation -> particles and their material

then the alternative implementations, which change only the numerics:

    mpm_scatter[warp] [triton] [triton_colour]     fused kernels; the colour variant is atomic-free
    mpm_gather[warp] [torch_loop27]                one batched reduction, or 27 sequential passes
    mpm_strain[warp]     mpm_grid_update[warp] [nosync]

The oracle is fenced and stays fenced. `mls_mpm_mechanics` does in one operator what the four
steps do in four, and it exists so the decomposition can be checked against something. It is not
the recommended path and not what a new specification should schedule.
"""
from __future__ import annotations
import functools
import itertools
import math
import numpy as np
import torch
from plexus.models.base import Field, Structural
from plexus.models.registry import register_field
from plexus.models.base import Exchange
from plexus.models.registry import register_operator
# (was `from plexus.operators.mpm_grid import stencil_offsets, bspline, sub_dt`) -- same module now
from plexus.models.base import FieldUpdate
# (was `from plexus.operators.mpm_grid import sub_dt`) -- same module now
from plexus.models.base import Lateral
import os
from plexus.models.base import Field, Exchange
from plexus.models.registry import register_field, register_operator
from plexus.paths import graphs_data_path
import json
from plexus.models.base import Seed


@functools.lru_cache(maxsize=None)
def _perm_index(device, perm) -> torch.Tensor:
    """A constant column permutation as a DEVICE tensor, built once. See `_polar_higham`."""
    return torch.tensor(list(perm), dtype=torch.long, device=device)


@functools.lru_cache(maxsize=None)
def _stencil_cached(dim: int, devstr: str) -> torch.Tensor:
    return torch.tensor(list(itertools.product(range(3), repeat=dim)),
                        dtype=torch.float32, device=devstr)


def _const_any(ob, attr, t) -> bool:
    """A RUN-CONSTANT tensor predicate, resolved ONCE per operator instance.

    `bool(t.any())` inside a python `if` is `.item()`: a GPU->CPU round trip that drains the
    launch queue and stalls the CPU that should be running ahead queueing kernels. There are
    seven of them across the four MPM operators -- "does this set contain liquid / snow /
    viscoelastic particles", "are there wall obstacles" -- and every one asks a question whose
    answer is fixed for the whole run: the material masks are written once at seeding
    (models/entities.py) and the obstacle set does not move.

    Twelve evaluations per substep on a two-set spec, measured at 19.7 ms of a 156 ms frame.
    They are also, on their own, the reason a substep cannot be stream-captured at all.

    FIRST-CALL CACHE rather than a constructor argument, because the operator is built before
    the level it acts on has been seeded. The first call still syncs; every later one does not,
    and the answer is identical either way -- so the eager path stays bit-for-bit unchanged.
    """
    v = getattr(ob, attr, None)
    if v is None:
        v = bool(t is not None and bool(t.any()))
        setattr(ob, attr, v)
    return v


def stencil_offsets(dim: int, device="cpu") -> torch.Tensor:
    """The 3^dim quadratic-B-spline stencil offsets, row-major (last axis fastest).
    2D -> [9,2] == `[[i,j] for i in 0..2 for j in 0..2]`; 3D -> [27,3] (matches the
    MPM_3D offset ordering: idx//9, (idx%9)//3, idx%3)."""
    # MEMOISED. Built from a python list, this is a PAGEABLE HOST->DEVICE COPY -- a host sync in
    # eager, and outright illegal inside a CUDA stream capture. It depends only on (dim, device)
    # and is called four times per substep, so unmemoised it rebuilds a constant 27x3 table on
    # every one of them -- worth a noticeable fraction of a frame. Removing host syncs is
    # all-or-nothing: any single survivor drains the launch queue and hides the cost of the rest.
    # The returned tensor is SHARED and must be treated as read-only; today every caller only
    # reads it (`offsets.long()`, arithmetic) and none mutates it in place.
    return _stencil_cached(int(dim), str(device))


# The 2D stencil as a module constant; the scatter and gather build theirs per dimension.
_OFFSETS = stencil_offsets(2)


@register_field("mpm_grid")
class MPMGrid(Field):
    """MLS-MPM background grid over the world box, with square cells dx = world_size[1] / n_grid.
    Channels: m (mass), mv (momentum [.,dim]), c (liquid colour for CSF), v (velocity
    [.,dim]). Pure scratch: p2g zeroes + scatters into it each substep, grid_update
    solves on it, g2p reads it back.

    THE CELL SIZE COMES FROM THE WORLD: dx = world_size[1] / n_grid, so `n_grid` is CELLS ACROSS
    AXIS 1 and not cells per unit length. The two readings agree only on a box one unit across.
    Deriving dx from a fixed 1.0 instead would put a unit-wide grid on whatever world the
    specification declared: on a box a tenth of a unit across the cell is ten times too large,
    a tenth of the cells per axis carry any mass, and `mpm_gather` -- which clamps positions into
    [2 dx, box - 2 dx] -- would be clamping with a correct box and a wrong dx, crushing the body
    into a slab on the first substep.

    `n_grid` means CELLS ACROSS AXIS 1, not cells per unit length. The two readings coincide when
    the world is one unit across on that axis, which is the default, so a specification that does
    not declare a world gets the same dx and the same node count either way. They diverge exactly
    where it matters: on a box that is not one unit across.

    inv_dx IS NOT 1.0/dx. `1.0 / (1.0 / n) != float(n)` for 640 of the first 4,096 integers -- the
    first offender is n = 49 -- so the reciprocal round-trip is exact only by luck at the n_grid
    values in use (48, 64, 96, 128, 192). Deriving it from the integer directly keeps byte-identity
    on the corpus, and keeps it for a spec that picks n_grid = 49 tomorrow.

    Reference: the grid, the quadratic B-spline kernel and the CFL ceiling dt < dx / sqrt(E/rho)
    are those of Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150; Sulsky, D. et al. (1994).
    Comput. Methods Appl. Mech. Eng. 118:179-196.
    """

    RECORD = False                                   # transient scratch -- not recorded/rendered

    def __init__(self, name, width=1.0, n_grid=128, dim=2, device="cpu", world_size=None,
                 box=None, origin=None, per_parent=None, cell_box=None, **kw):
        super().__init__(name)
        self.dim = int(dim)
        # `per_parent: <set>` + `cell_box: L` -- ONE SMALL GRID PER BODY, the cell's own (exp 11,
        # 2026-09-27). On one shared grid every body that touches another shares its nodes, so the
        # two move with one velocity there: MEASURED on 370 packed MPM cells (Finding 89), no cell
        # lost a neighbour in 150 frames and the walk tore cells instead of moving them. Here each
        # parent (a cell of `per_parent`) gets its own n_grid^3 block of side `cell_box` (world
        # units), re-centred on the cell's centroid every substep by `mpm_scatter`: a cell's
        # elasticity is solved on its own nodes only, and bodies meet through a contact force
        # between their points (`pair_potential ... exclude: <the parent-id block>`) that can
        # slide. The block must hold the cell with a 2-node margin; `mpm_gather` clamps a point
        # that reaches its block's edge. Default-off; only the torch bodies read it.
        self.per_parent = str(per_parent) if per_parent else None
        if self.per_parent:
            if cell_box is None:
                raise ValueError("MPMGrid: `per_parent` needs `cell_box`, the side of each body's grid (world units)")
            box = [float(cell_box)] * self.dim
            origin = None
        self.pp_origin = None
        _box_decl = box
        # `width` is the legacy axis-0 scalar and stays the fallback: a caller that does not pass
        # the per-axis box gets exactly the old geometry, [width] x [1] x [1].
        box = [float(w) for w in world_size] if world_size else \
              [float(width)] + [1.0] * (self.dim - 1)
        # `box` + `origin` -- A GRID THAT COVERS PART OF THE WORLD, in world units (exp 11, 2026-09-27).
        # The grid spans [origin, origin + box] instead of [0, world]. A tissue built about the
        # ORIGIN of a 200-unit free world (exp 11's gland, radius 5 -> 18) needs material points
        # only inside a ~48-unit box around it: over the whole world, a dx small enough for a
        # 50-point cell (~0.6) is ~350 nodes per axis, 43M nodes, and the torch gather clamps every
        # point into [2 dx, world - 2 dx] -- a body at negative coordinates is crushed onto the
        # first plane. Default-off: with neither key the grid is the world box and `origin` is None,
        # which is the branch every existing spec takes, byte for byte. Only the torch bodies of
        # `mpm_scatter` and `mpm_gather` read `origin`; every other body refuses it (`_no_origin`).
        if _box_decl is not None:
            box = [float(w) for w in _box_decl]
        self.origin = None
        if origin is not None:
            o = [float(v) for v in origin]
            if len(o) != self.dim:
                raise ValueError(f"MPMGrid: origin has {len(o)} entries but dim={self.dim}")
            self.origin = torch.tensor(o, dtype=torch.float32, device=device)
        # THE GATHER CLAMPS TO THIS GRID'S BOX, not the world's, once the spec has declared either key.
        self.box_own = (_box_decl is not None) or (origin is not None)
        if len(box) != self.dim:
            raise ValueError(f"MPMGrid: world_size has {len(box)} entries but dim={self.dim}")
        self.world_size = box
        self.width = box[0]
        self.n_grid = int(n_grid)                    # cells across axis 1
        self.inv_dx = float(n_grid) / box[1]         # NOT 1.0/dx -- see the note above
        self.dx = box[1] / float(n_grid)
        n_k = [max(1, int(round(box[k] * self.inv_dx))) for k in range(self.dim)]
        self.nx, self.ny = n_k[0], n_k[1]
        if self.dim == 2:
            self.shape = (self.nx, self.ny)
        else:
            self.nz = n_k[2]
            self.shape = (self.nx, self.ny, self.nz)
        n = 1
        for s in self.shape:
            n *= s
        self.n_cells = n
        self.register_buffer("m", torch.zeros(n, device=device))
        self.register_buffer("mv", torch.zeros(n, self.dim, device=device))
        self.register_buffer("c", torch.zeros(n, device=device))
        self.register_buffer("v", torch.zeros(n, self.dim, device=device))
        self.n_local = n                             # nodes of ONE block (the whole grid without `per_parent`)
        self.n_blocks = 1

    def ensure_blocks(self, n_blocks):
        """`per_parent`: size the buffers to `n_blocks` blocks of `n_local` nodes (once, or when the
        parent set's buffer grows)."""
        if int(n_blocks) == self.n_blocks:
            return
        n, dev = int(n_blocks) * self.n_local, self.m.device
        self.m = torch.zeros(n, device=dev); self.mv = torch.zeros(n, self.dim, device=dev)
        self.c = torch.zeros(n, device=dev); self.v = torch.zeros(n, self.dim, device=dev)
        self.n_blocks = int(n_blocks)

    @property
    def grid(self):                                  # [1,*shape] view for the recorder (mass density)
        return self.m[: self.n_local].view((1,) + self.shape)


def _pp_frame(H, g, p, X, occ=None, recentre=False):
    """`per_parent` grids: (the block index of every point, the block origins [n_blocks, D]).
    `recentre` (the substep's first scatter) puts every block's centre on its cell's live centroid; the
    gather reads the SAME origins, so a substep's scatter and gather agree on where each node is."""
    pidx = H.lift_index(p.name, g.per_parent)
    nb = int(H.level(g.per_parent).state.shape[0])
    g.ensure_blocks(nb)
    if recentre or g.pp_origin is None or g.pp_origin.shape[0] != nb:
        w = torch.ones(X.shape[0], device=X.device, dtype=X.dtype) if occ is None else (occ > 0).to(X.dtype)
        cnt = torch.zeros(nb, device=X.device, dtype=X.dtype).index_add_(0, pidx, w)
        cen = torch.zeros(nb, X.shape[1], device=X.device, dtype=X.dtype).index_add_(0, pidx, X * w[:, None])
        cen = cen / cnt.clamp_min(1.0)[:, None]
        half = 0.5 * torch.as_tensor(g.world_size, device=X.device, dtype=X.dtype)
        g.pp_origin = cen - half
    return pidx, g.pp_origin


def _pp_flat(flat, pidx, n_local, S):
    """The local stencil index of every point, offset into its own block."""
    return (flat.view(pidx.shape[0], S) + (pidx * n_local)[:, None]).reshape(-1)


def _no_origin(op, g):
    """Refuse a grid with an `origin` (see `MPMGrid`) in a body that does not implement it: that body
    would read world positions as grid positions and put every point `origin` away from its node."""
    if getattr(g, "origin", None) is not None:
        raise NotImplementedError(
            f"{type(op).__name__}: the grid {getattr(g, 'name', '?')!r} declares an `origin`, and only "
            f"the torch bodies of mpm_scatter / mpm_gather / mpm_grid_update / mpm_strain honour it. "
            f"Give these MPM operators `implementation: default`.")


def bspline(X, inv_dx, offsets, shape, periodic):
    """Quadratic B-spline weights of each particle over its 3^dim grid stencil.
    Returns (fx [N,D], weight [N,S], flat [N*S]) where S = 3^D. Dimension-generic; the
    2D call reduces bit-identically to the original `w[:,oi,0]*w[:,oj,1]`, `gx*ny+gy`."""
    D = X.shape[1]
    base = (X * inv_dx - 0.5).floor().long()                          # [N,D]
    fx = X * inv_dx - base.float()                                    # [N,D]
    w = torch.stack([0.5 * (1.5 - fx) ** 2,
                     0.75 - (fx - 1) ** 2,
                     0.5 * (fx - 0.5) ** 2], dim=1)                   # [N,3,D]
    oidx = offsets.long()                                            # [S,D]
    weight = torch.ones(X.shape[0], offsets.shape[0], device=X.device)
    for k in range(D):                                               # prod_k w[:, o_k, k]
        weight = weight * w[:, oidx[:, k], k]
    gpos = base[:, None, :] + oidx[None]                             # [N,S,D]
    comps = []
    for k in range(D):
        comps.append(gpos[..., k] % shape[k] if periodic
                     else gpos[..., k].clamp(0, shape[k] - 1))
    flat = comps[0]                                                  # row-major flatten over `shape`
    for k in range(1, D):
        flat = flat * shape[k] + comps[k]
    return fx, weight, flat.reshape(-1)


def sub_dt(H, fallback):
    """The current substep dt: the schedule's `{substep: N, dt}` loop sets `H.sub_dt`;
    fall back to the operator's own `dt_sub` param if run outside a substep loop."""
    sd = getattr(H, "sub_dt", None)
    return float(sd if sd is not None else fallback)


def _polar_higham(F, iters=6):
    """Orthogonal polar factor R of F = R S, by Newton's iteration  R <- (R + R^-T)/2.

    Quadratically convergent from R0 = F whenever F is non-singular, which a valid
    deformation gradient is. Six iterations reach float32 on deformations of the size
    an eye muscle produces; the caller can ask for fewer.

    Only the ROTATION is wanted here -- the singular values the SVD also returns are
    not used by the fixed-corotated stress below -- so this is a drop-in, not an
    approximation of a different quantity.

    det R follows sign(det F), so an INVERTED particle would give an improper R where
    the SVD path forces a proper rotation. That case is caught rather than hidden: an
    inverted deformation gradient means the simulation has already failed.
    """
    D = F.shape[-1]
    R = F.clone()
    if D == 3:
        for _ in range(iters):
            # inverse-transpose by the ADJUGATE, not by a solve. `torch.linalg.solve`
            # RAISES on a singular batch element, and one degenerate particle out of
            # 58,200 then kills the whole run -- which is exactly what happened on the
            # SR staircase, 65 minutes in. The cofactor form cannot raise: the only
            # division is by the determinant, and clamping that away from zero leaves a
            # degenerate particle with a finite (meaningless) rotation instead of taking
            # the simulation down with it. A collapsed deformation gradient is a failure
            # to report, not a reason to lose the other 58,199.
            # `R[:, :, [1, 2, 0]]` LOOKS free and is not: a python list index builds a CPU
            # int tensor and copies it to the device, six times per call. In eager that is a
            # pageable H2D per iteration; inside a stream capture it is fatal, and measured as
            # exactly that -- list index FAILS with cudaErrorStreamCaptureInvalidated where a
            # cached device index tensor captures cleanly. index_select on a memoised index is
            # the same gather with the constant hoisted out.
            _i1 = _perm_index(R.device, (1, 2, 0))
            _i2 = _perm_index(R.device, (2, 0, 1))
            c = torch.cross(R.index_select(2, _i1), R.index_select(2, _i2), dim=1)
            det = (R[:, :, 0] * c[:, :, 0]).sum(1)[:, None, None]
            det = torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
            R = 0.5 * (R + c / det)
    else:
        eyeT = torch.eye(D, device=F.device, dtype=F.dtype)
        for _ in range(iters):
            R = 0.5 * (R + torch.linalg.solve(R, eyeT).transpose(-2, -1))
    return R


# ==========================================================================================================
# DIMENSIONAL CONSTANTS, EXPRESSED RELATIVE TO THEIR OWN NATURAL SCALE
# ==========================================================================================================
_REF_DX, _REF_RHO = 1.0 / 96, 1.0          # the box every one of these numbers was chosen against

#   name -> (historical value, exponent of density, exponent of length)
#   a mass is rho * L^3;  a length is L;  a reciprocal length is L^-1.
_CONST_DIMS = {
    "mass_floor":     (1e-10, 1, 3),
    "csf_mass_floor": (1e-8, 1, 3),
    "ring":           (0.04, 0, 1),
    "csf_eps":        (1e-6, 0, -1),
}


def _scale_constant(name, dx, rho=1.0):
    """The value a dimensional constant takes at this (dx, rho).

    WHY THESE ARE NOT NUMBERS. Each was chosen against a UNIT BOX at n_grid 96 with density 1, and
    each means something else the moment any of the three changes. `wall_contact: 0.04` -- the first
    one converted -- was the plain case: in a 0.1 m box it selected everything but a 0.02 m sliver,
    so the whole fluid read as permanently in wall contact and was permanently damped.

    The others are quieter and one of them is already costing accuracy. The CSF regulariser `eps` in
    n = grad(c)/(|grad c| + eps) has units of 1/LENGTH, and it is why `csf_rho` is not a pure gain
    rescaling: at the parity tension sigma = 120/192^2, material_two_drops_st reproduces only to
    2.376% RMS in CSF force, 32.7% once `csf_band` is on, because an absolute epsilon added to
    |grad c| bites differently once the colour is divided by rho*dx^D.

    CALIBRATED TO REPRODUCE THE HISTORICAL NUMBER EXACTLY at the reference, so no existing spec
    moves: at rho = 1 and dx = 1/96 both ratios are exactly 1.0 and the result is the original float,
    not a value close to it.

    NOT EVERY CONSTANT BELONGS HERE, and the ones left out are left out for a reason:
      a_max 200   an acceleration. The natural scale is `g`, which DIFFERS BETWEEN SPECS (14, 16,
                  9.81), so no single multiple reproduces them all -- converting it would be a
                  behaviour change dressed as a refactor.
      vmax 1e9    a velocity, and inert: the binding cap is min(vmax, 0.4*dx/dt) in the gather, so
                  the absolute default never applies.
      spin_k 30   a rate, and there is no natural time scale in the operator to divide it by.
    """
    v0, a, b = _CONST_DIMS[name]
    return v0 * (float(rho) / _REF_RHO) ** a * (float(dx) / _REF_DX) ** b


def _ancestor_accel(H, p, a_max, D, dev):
    """The body acceleration reaching these material points from EVERY level above them.

    `[Np, D]`, the sum over the containment chain of each ancestor's accumulated delta, broadcast
    down the composed containment map:

        a_ext = sum_{L in ancestors(p)}  delta_L [ pi_L (p) ]

    with pi_L the composition of the parent maps from the particle set up to L (`H.lift_index`).

    WHY IT IS A SUM OVER THE CHAIN AND NOT ONE HOP. Reading `H.delta(p.parent_name)[p.parent]` in the
    scatter bodies is Broadcast across exactly one
    containment map. That is correct and complete for the two-level models the corpus is made of
    -- `cell -> mpm_particle` -- and silently wrong the moment a model has three:
    `cell -> compartment -> mpm_particle` with `gravity at: cell` builds, validates, runs, and
    the cell does not fall, because the delta sits on a set no reader ever reaches. Nothing
    raises, because a body force of zero is a legal body force.

    Summing over the whole chain is also the right SEMANTICS and not just the reachable one:
    gravity declared on the organism, a pressure declared on the tissue and a drag declared on
    the cell are three body forces on the same material point, and superposing them is what
    forces do. Each level's delta is clamped as before, once, at the level it was written.
    """
    a_ext = None
    for anc in H.ancestors(p.name):
        d = H.delta(anc)
        if d is None or d.numel() == 0:
            continue
        d = torch.nan_to_num(d, posinf=a_max, neginf=-a_max).clamp(-a_max, a_max)
        if d.shape[1] != D:
            continue                        # a non-spatial ancestor (voltage, chemistry): not a body force
        term = d[H.lift_index(p.name, anc)]
        a_ext = term if a_ext is None else a_ext + term
    return torch.zeros(p.n, D, device=dev) if a_ext is None else a_ext


def _hand_body_force_to_grid(op, H, a_ext, dev, D):
    """WHERE A BODY FORCE BELONGS. Canonical MLS-MPM applies gravity ON THE GRID, as an
    acceleration, AFTER the momentum has been divided by nodal mass -- Taichi's mpm88/mpm99 read

        if grid_m[I] > 0:  grid_v[I] = (1 / grid_m[I]) * grid_v[I];  grid_v[I][1] -= dt * gravity

    This operator instead folds it into the PARTICLE velocity before P2G, so it rides through the
    scatter as momentum and comes back out of a division it never needed to be inside.

    IN EXACT ARITHMETIC THE TWO ARE THE SAME THING: sum_p w m (v + a dt) / sum_p w m = v_bar + a dt,
    because the mass cancels. The placement only starts to matter when something breaks that
    cancellation, and exactly one thing does -- `gm.clamp(min=mass_floor)` in the grid solve. On a
    node lighter than the floor the whole momentum, gravity's share included, is scaled by
    gm/mass_floor < 1, and a particle there cannot accelerate.

    MEASURED, before assuming that is what ails anything: a lone block dropped in an empty box
    accelerates at 0.9996 OF g with drag 0, and 0.9902 with the spec's Stokes drag 0.1 -- the
    1% being the drag, exactly. So on this spec family the floor does NOT bind and gravity IS
    delivered. Turning `mass_floor` down from 1e-10 to 1e-14 changes the suspended-particle count
    from 224 to 224 on material_3d_water_st000. This path is therefore the CORRECT ARCHITECTURE,
    not a cure for the haze, and it is offered as `body_force: grid` so the two can be compared
    rather than argued about.

    WHAT MOVES AND WHAT CANNOT. Only the PARENT-LEVEL acceleration -- gravity, buoyancy, anything
    emitted `at:` the parent set -- is a field quantity with a grid representation, and only when
    every parent carries the SAME vector (checked once, cached). Per-particle body forces (turgor,
    active stress, per-particle drag) have no grid representation and stay on the particle, where
    the clamp still reaches them; that is an argument for fixing the clamp, not for moving them.
    """
    if getattr(op, "body_force", "particle") != "grid":
        return a_ext
    if getattr(op, "_c_bf", None) is None:
        # a device sync, so it happens ONCE and is cached: is the parent acceleration uniform?
        amax = (a_ext - a_ext[:1]).abs().max() if a_ext.numel() else torch.zeros((), device=dev)
        op._c_bf = bool(float(amax) < 1e-12)
        if not op._c_bf:
            import warnings
            warnings.warn(
                "mpm_scatter: body_force='grid' needs a body force that is the same for every "
                "particle, and this one is not (per-particle accelerations have no grid "
                "representation). Falling back to the particle path for all of it.",
                RuntimeWarning, stacklevel=2)
        op._bf_buf = torch.zeros(D, device=dev)
    if not op._c_bf:
        return a_ext
    op._bf_buf.copy_(a_ext[0])              # persistent buffer -> safe inside a captured graph
    H._mpm_body_accel = op._bf_buf          # consumed by MPMGridUpdate after normalisation
    return a_ext - op._bf_buf                # particle keeps only what the grid cannot carry




class MPMWrites:
    # ---- the write hooks -------------------------------------------------------------------
    # THE ONLY THING AN `implementation: differentiable` HAS TO CHANGE. Everything above computes
    # the same numbers either way; what autograd cannot take is the WRITE -- a grid zeroed and
    # re-accumulated in place, a slice assignment into a view, a copy_ into a buffer a later
    # substep reads. Each is one line, each is behind a method, and the alternative implementation
    # overrides the methods rather than restating the physics. `FUNCTIONAL` is a class flag and not
    # `torch.is_grad_enabled()`, so which body runs is a property of the SPECIFICATION -- a run says
    # `implementation: differentiable` and gets it -- rather than of an ambient context.
    FUNCTIONAL = False

    def _const_read(self, t):
        """A material constant, read. A seed may rewrite these buffers, and under a tape that write
        moves a version counter the backward is holding -- so the differentiable body clones."""
        return t

    def _affine_dpos(self, affine, dpos):
        """`affine[p] @ dpos[p, s]` for every point p and stencil node s: [P,3,3] x [P,S,3] -> [P,S,3]. One tiny matrix product
        per (point, node) here; `mpm_scatter[implementation: broadcast]` replaces it with a multiply and a sum."""
        return (affine[:, None] @ dpos[..., None]).squeeze(-1)

    def _outer_sum(self, weight, gvn, dpos):
        """`sum_s w[p,s] gvn[p,s] (x) dpos[p,s]` for every point p: [P,S], [P,S,3], [P,S,3] -> [P,3,3]. The outer product of a
        node's velocity and its offset, weighted and summed over the stencil; `mpm_gather[implementation: broadcast]` replaces
        the matrix product (inner size 1) by a broadcast multiply."""
        return (weight[..., None, None] * (gvn[..., :, None] @ dpos[..., None, :])).sum(1)

    def _write_v(self, g, gv):
        g.v.copy_(gv)

    def _write_F(self, p, F):
        p.F.copy_(F)

    def _write_Jp(self, p, Jp):
        p.Jp.copy_(Jp)

    def _write_particles(self, p, pa, pb, va, vb, Xn, new_V, new_C):
        p.state[:, pa:pb] = Xn
        p.state[:, va:vb] = new_V
        p.C.copy_(new_C)


@register_operator("mpm_scatter", "p2g", family="mpm", set="particle", kind="exchange", title="spread point mass onto the grid",
                   equation=r"""$$\boldsymbol\sigma = 2\mu(\mathbf F-\mathbf R)\mathbf F^{\!\top} + \lambda(J-1)J\,\mathbf I,
\qquad
\mathbf Q = -\frac{4\,\Delta t}{\Delta x^{2}}\,V\,\boldsymbol\sigma + m\,\mathbf C$$""")
class MPMScatter(MPMWrites, Exchange):                
    r"""particle to grid, the first step of the MLS-MPM cycle: every particle deposits its mass, its momentum and the impulse of its own internal stress onto the grid nodes around it.
    particle -> mpm_grid: reads pos, vel, F, C and the Lame parameters; writes the grid field in place (the grid is the output, nothing is emitted).

        \boldsymbol\sigma = 2\mu(\mathbf F-\mathbf R)\mathbf F^{\!\top} + \lambda(J-1)J\,\mathbf I, \qquad \mathbf Q = -\frac{4\,\Delta t}{\Delta x^{2}}\,V\,\boldsymbol\sigma + m\,\mathbf C
        \sigma         the Kirchhoff stress of the fixed-corotated law
        \mu, \lambda   the particle's Lame parameters, stress
        F              the deformation gradient
        R              its rotation (polar decomposition F = R S; `polar`, `polar_iters`)
        J              det F, the volume ratio
        I              the identity
        Q              the affine momentum matrix deposited with the particle: the stress impulse plus the APIC term
        \Delta t       `dt_sub`, the substep (stable below dx / sqrt(E/rho))
        \Delta x       the grid spacing
        V              the particle's rest volume
        m              its mass
        C              its affine velocity gradient (APIC)

    Reference: Hu, Y. et al. (2018). A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. ACM Trans. Graph. 37(4):150; Sulsky, D., Chen, Z. & Schreyer, H. L. (1994). A particle method for history-dependent materials. Comput. Methods Appl. Mech. Eng. 118:179-196."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"The particle-to-grid transfer. First each point's velocity takes the substep's body acceleration (gravity) and "
        r"the Stokes drag `drag`. Then each point computes its Kirchhoff stress $\boldsymbol\sigma=2\mu(\mathbf F-\mathbf "
        r"R)\mathbf F^{\!\top}+\lambda(J-1)J\,\mathbf I$ (the fixed-corotated law: $\mu$ and $\lambda$ its Lamé "
        r"parameters, $\mathbf R$ the rotation of $\mathbf F$, $J=\det\mathbf F$), adds any viscous stress `mpm_viscosity` "
        r"wrote, and spreads its mass $m$ and its momentum, which carries $\mathbf Q=-\tfrac{4\,\Delta t}{\Delta "
        r"x^{2}}\,V\,\boldsymbol\sigma+m\,\mathbf C$, to the grid nodes around it ($V$ the point's rest volume, $\mathbf "
        r"C$ its affine velocity matrix, $\Delta x$ the grid spacing, $\Delta t$ the substep `dt_sub`)."
    )
    PARAM_UNITS = {'a_max': 'acceleration', 'drag': 'rate', 'dt_sub': 'time', 'polar_iters': 'count', 'body_force': 'acceleration'}
    SYMBOLS = {'\\sigma': 'the Kirchhoff stress of the fixed-corotated law', '\\mu, \\lambda': "the particle's Lame parameters, stress", 'F': 'the deformation gradient', 'R': 'its rotation (polar decomposition F = R S; `polar`, `polar_iters`)', 'J': 'det F, the volume ratio', 'I': 'the identity', 'Q': 'the affine momentum matrix deposited with the particle: the stress impulse plus the APIC term', '\\Delta t': '`dt_sub`, the substep (stable below dx / sqrt(E/rho))', '\\Delta x': 'the grid spacing', 'V': "the particle's rest volume", 'm': 'its mass', 'C': 'its affine velocity gradient (APIC)'}

    EMIT = None                 # particle->grid: writes the mpm_grid field in place; returns {} — no integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []        # no required params — `to` defaults to mpm_grid, all knobs optional
    REQUIRES_TYPE_PROPS = ["youngs"]
    # a liquid has no Young's modulus (nu -> 1/2 makes E = 3K(1-2nu) -> 0); it has a bulk modulus,
    # and for mu = 0 that IS lambda. Either spelling satisfies the requirement.
    TYPE_PROP_ALTERNATIVES = {"youngs": ("bulk_modulus",)}
    MECHANISM_TAGS = ["particle_to_grid", "fixed_corotated_stress", "shared_grid_accumulate"]
    PARAM_ROLES = {"dt_sub": "MLS-MPM substep dt", "drag": "Stokes drag coefficient",
                   "a_max": "external-acceleration clamp",
                   "body_force": "body_force_application_site",
                   "store_stress": "cache Cauchy stress to a per-particle buffer", 'colour_from_face': 'face_weights_of_the_colour_channel_lateral_apical_basal', 'face_block': 'point_block_that_holds_the_face', 'face_cells': 'cell_set_of_the_face_colour', 'polar': 'polar_decomposition_method', 'polar_iters': 'its_iterations', 'to': 'the_grid_field'}
    REFERENCE = "Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150 (MLS-MPM P2G); Sulsky, D. et al. (1994)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("from", "mpm_particle")      # the particle set it reads (`from:`)
        self.grid = params.get("_at", "mpm_grid")         # the grid it writes (`at:`)
        self.dt_sub = float(params.get("dt_sub", 2e-4))
        self.drag = float(params.get("drag", 0.0))
        self.a_max = float(params.get("a_max", 200.0))
        # WHERE THE BODY FORCE IS APPLIED. "particle" (default, and every existing run) folds it
        # into the particle velocity before P2G; "grid" hands the uniform part to the grid solve,
        # which is where canonical MLS-MPM puts it. See _hand_body_force_to_grid for why the two
        # are algebraically identical until the mass clamp binds, and for the measurement showing
        # that on this spec family it does not.
        self.body_force = str(params.get("body_force", "particle"))
        if self.body_force not in ("particle", "grid"):
            raise ValueError(f"mpm_scatter: body_force must be 'particle' or 'grid', "
                             f"got {self.body_force!r}")
        # `colour_from_face: [w_lateral, w_apical, w_basal]` (exp29, 2026-10-23): the colour channel `c` is deposited as
        # weight x mass x w[face of the point] (its `face` block: 0 lateral, 1 apical, 2 basal) in place of x is_liquid, so
        # `mpm_grid_update[body_csf_balanced | warp_pp_balanced]` with `tension_on: face` can read a PER-NODE surface-tension
        # factor c / m: a cortex whose tension differs by face, as a polarised epithelial cell's (SimuCell3D, Runser et al.
        # 2024, gives each face type its own tension; lateral 20x apical by default). Unset, the colour is the liquid mass.
        self.colour_from_face = params.get("colour_from_face")
        if self.colour_from_face is not None:
            w = [float(v) for v in self.colour_from_face]
            if len(w) != 3 or min(w) < 0.0:
                raise ValueError("mpm_scatter: colour_from_face is [w_lateral, w_apical, w_basal], each >= 0")
            self._face_w = torch.tensor(w, dtype=torch.float32)
        # `face_block` (exp30, 2026-10-23): a width-1 block of the grid's `per_parent` set giving each CELL's apical-and-basal weight
        # in place of w_apical / w_basal (the lateral weight stays w_lateral): the polarity of the cortex as a per-cell knob.
        self.face_block = params.get("face_block")
        self.face_cells = params.get("face_cells", "cell")
        # HOW THE POLAR ROTATION IS FOUND, in 3D. The fixed-corotated stress needs the rotation R
        # from F = R S, and there are two numerical routes to the same rotation:
        #
        #   svd     `torch.linalg.svd` -> a cuSOLVER call. It is an EXTERN kernel: no compiler
        #           fuses it, and -- decisively -- it CANNOT BE CAPTURED into a CUDA graph, so a
        #           specification using it forfeits the substep capture, which is worth about 2x
        #           on its own. It also costs roughly a microsecond per 3x3 matrix, and this runs
        #           once per particle per substep.
        #   higham  the Newton polar iteration R <- (R + R^-T)/2, built from matmuls and a cofactor
        #           cross product. Converges quadratically from F, captures, fuses, and is several
        #           times faster on the same batch. It agrees with the SVD rotation to float32.
        #
        # The default is `higham`. The difference in a trajectory is bounded rather than
        # amplifying, and 2D specifications are unaffected entirely: `forward` takes the analytic
        # 2x2 rotation branch and never consults this parameter.
        #
        # `polar: svd` remains available for a specification that must reproduce a stored 3D result
        # byte for byte.
        # core operators -- both sides of the comparison move together.
        self.polar = str(params.get("polar", "higham")).lower()
        self.polar_iters = int(params.get("polar_iters", 6))
        # KEEP THE CAUCHY STRESS, OPTIONALLY. The fixed-corotated law below produces the Kirchhoff
        # stress tau = J.sigma, uses it to build the affine momentum matrix, and then overwrites the
        # variable with its dt-scaled form -- so the one tensor in the solver that says what the
        # material is actually carrying is computed 8,000 times a run and discarded every time. With
        # `store_stress: true` it is cached to a per-particle `sigma` buffer (Cauchy, i.e. tau/J) that
        # diagnostics and colourings can read instead of re-deriving a proxy from F.
        #
        # DEFAULT OFF, and the guard is what makes this safe to add to a shared operator: when off,
        # nothing is allocated and the only cost is one branch per substep. The cached value is read,
        # never written back, so the mechanics cannot be changed by asking for it.
        self.store_stress = bool(params.get("store_stress", False))

    def forward(self, H, mask=None):
        p = H.level(self.at); g = H.field(self.grid); dev = p.state.device
        dt = sub_dt(H, self.dt_sub)
        nx, ny, inv_dx, dx = g.nx, g.ny, g.inv_dx, g.dx
        D = p.F.shape[-1]
        periodic = bool(getattr(H, "periodic", False))
        offsets = stencil_offsets(D, dev)
        X, V = p.get("pos"), p.get("vel")
        # external per-cell acceleration from the parent set's accumulated delta (gravity)
        a_ext = _ancestor_accel(H, p, self.a_max, D, dev)
        part_accel = getattr(H, "part_accel", None)
        if part_accel is not None:
            a_ext = a_ext + part_accel
        # per-particle body force from particle-level force operators (e.g. pulse_to_contraction,
        # drag) -- the symmetric counterpart of the parent-delta path above (gravity).
        a_ext = a_ext + torch.nan_to_num(H.delta(p.name))
        a_ext = _hand_body_force_to_grid(self, H, a_ext, dev, D)
        V = V + dt * (a_ext - self.drag * V)                       # body force + Stokes drag (local; G2P resets V)

        # mass CLONED AT THE POINT IT IS FIRST READ, not later: the momentum line below holds
        # `mass[:, None, None]`, and a clone taken after that has already let the view into the tape.
        F, C, mass = p.F, p.C, self._const_read(p.mass)
        # RESIDUAL STRESS / PRESTRESS (optional, default OFF): compute the fixed-corotated stress
        # relative to a non-identity per-particle REST tensor F_res (multiplicative morphoelastic split
        # F = Fe . F_res, so Fe = F @ F_res_inv is the elastic part). At the mesh rest state F=I this
        # leaves Fe = F_res_inv != I -> a STANDING PRELOAD; an incompatible F_res(x,y) holds a
        # self-equilibrated residual-stress field. Absent buffer -> byte-identical; F_res=I (alpha=0) ->
        # F @ I = F exactly, so the operator truly ablates. Only the STRESS reference shifts; the
        # kinematic F (updated in mpm_strain) is untouched.
        Fres_inv = getattr(p, "F_res_inv", None)
        if Fres_inv is not None:
            F = F @ Fres_inv
        eye = torch.eye(D, device=dev).expand(p.n, D, D)
        if D == 2:
            a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
            J = a * d - b * c
        else:
            J = torch.linalg.det(F)
        # THE MATERIAL CONSTANTS THROUGH CLONES WHILE A TAPE IS KEPT. mu, la, p_vol and mass are
        # buffers a seed operator is allowed to rewrite (`set_material` does), and a version counter
        # that moves after a forward has read the buffer is what autograd reports as "[N, 1, 1] is
        # at version 2; expected version 1" -- from a line that did nothing wrong. One vector each,
        # per substep, and only under grad.
        mu, la = self._const_read(p.mu), self._const_read(p.la)
        snow = getattr(p, "is_snow", None)
        if _const_any(self, "_c_snow", snow):                      # snow hardening from the plastic ratio Jp
            h = torch.exp((10.0 * (1.0 - p.Jp)).clamp(-6.0, 6.0))
            mu = torch.where(snow, p.mu * h, p.mu)
            la = torch.where(snow, p.la * h, p.la)
        if D == 2:                                                 # analytic 2D polar rotation (bit-identical)
            cs, sn = (F[:, 0, 0] + F[:, 1, 1]), (F[:, 1, 0] - F[:, 0, 1])
            r = torch.sqrt(cs * cs + sn * sn) + 1e-9
            cs, sn = cs / r, sn / r
            R = torch.stack([torch.stack([cs, -sn], -1), torch.stack([sn, cs], -1)], -2)
        elif self.polar == "higham":                               # Newton polar iteration
            R = _polar_higham(F, self.polar_iters)
        else:                                                      # SVD polar rotation R = U Vh (proper rotation)
            # THE SIGN FLIP IS A MULTIPLY, NOT A BOOLEAN MASK, and the two are bit-for-bit the
            # same: `torch.equal` True, max|diff| 0.000e+00 over 1,500 matrices. What changes is
            # the SHAPE of the work. `U[det(U) < 0, :, -1] *= -1` is `aten.nonzero` -- the number
            # of rows it writes depends on how many matrices are improper rotations, which is
            # ~50% here and drifts as the body deforms. Under torch.compile that is a
            # data-dependent shape: dynamo cuts the graph in half at this line and then recompiles
            # on every new count until it hits the recompile limit and falls back to eager for the
            # rest of the run ("___stack2 size mismatch at index 0, expected 13, actual 15").
            # Multiplying the last column by +-1 touches every row unconditionally, so the shape
            # is static and the scatter can be one graph -- which is also the precondition for
            # capturing the substep as a CUDA graph.
            #
            # IN PLACE ON THE CLONE, not `torch.cat` of the columns. Reassembling with `cat`
            # changes the memory layout, the following matmul picks a different kernel, and the
            # result moves by 6e-8 -- small, but enough to break the promotion suite's
            # byte-identity gate for no gain. Writing the column back keeps U contiguous.
            U, sig, Vh = torch.linalg.svd(F)
            U = U.clone(); Vh = Vh.clone()
            sgnU = torch.where(torch.det(U) < 0, -1.0, 1.0)
            sgnV = torch.where(torch.det(Vh) < 0, -1.0, 1.0)
            U[:, :, -1] = U[:, :, -1] * sgnU[:, None]
            Vh[:, -1, :] = Vh[:, -1, :] * sgnV[:, None]
            R = U @ Vh
        stress = 2 * mu[:, None, None] * ((F - R) @ F.transpose(-2, -1)) \
            + eye * (la * J * (J - 1))[:, None, None]
        # optional MLS-MPM ACTIVE STRESS (-A n n^T from pulse_to_active_stress), added to the
        # fixed-corotated elastic stress before the affine scatter. Default off (absent -> None ->
        # pure elastic); same units / scaling / scatter as the elastic stress. Same H side-channel
        # idiom as part_accel; it feeds the tissue through stress divergence, not a pointwise force.
        # PER-LEVEL FIRST, THEN THE GLOBAL SIDE-CHANNEL. `H.active_stress` is one tensor for the
        # whole run, which is exactly right while a model has ONE particle cloud and unusable the
        # moment it has fifteen: the second scatter would read a stress sized for the first. A
        # buffer on the LEVEL is per-set by construction. Both are honoured and summed, so the
        # existing 2D field-driven operator is untouched.
        act = getattr(p, "act_stress", None)
        _gact = getattr(H, "active_stress", None)
        if _gact is not None and _gact.shape[0] != p.n:
            _gact = None                        # the global slot is another set's; see below
        if _gact is not None:
            act = _gact if act is None else act + _gact
        if act is not None:
            stress = stress + act
        # THE SAME CHANNEL, FOR A DIFFERENT MECHANISM. `extra_stress` is any additive Kirchhoff
        # stress written by an operator upstream in the substep -- today the Newtonian viscous
        # stress from `mpm_viscosity`. Kept separate from `active_stress` so the two compose
        # rather than clobber, and so a spec that reads `active_stress` back means what it says.
        xtr = getattr(p, "extra_stress", None)
        if xtr is None:
            _gx = getattr(H, "extra_stress", None)
            # A GLOBAL SLOT IS ONLY MEANINGFUL FOR THE SET WHOSE SHAPE IT HAS. With two particle
            # sets on one grid it belongs to whichever ran last, so it is taken only when it fits.
            xtr = _gx if (_gx is not None and _gx.shape[0] == p.n) else None
        if xtr is not None:
            stress = stress + xtr
        # TURGOR / OSMOTIC PRESSURE (optional, default OFF): an isotropic OUTWARD pressure carried
        # by this set, written as a per-particle `turgor` buffer by the `mpm_turgor` operator. The
        # sign is the one that makes a cell a cell: tau here is Kirchhoff (positive = tension), a
        # fluid at pressure P has sigma = -P.I, so a POSITIVE `turgor` SUBTRACTS from tau and pushes
        # the material outward -- the same sign the liquid's own `la*J*(J-1)` takes when J < 1
        # (compressed -> pushes out). Absent buffer -> byte-identical; the buffer lives on the LEVEL,
        # so `mpm_turgor at: cytosol` pressurises the cytosol and leaves nucleus/membrane untouched
        # even though all three scatter into the same grid.
        turg = getattr(p, "turgor", None)
        if turg is not None:
            stress = stress - eye * turg[:, None, None]
        if self.store_stress:
            # CAUCHY, NOT KIRCHHOFF. What the lines above build is tau = J.sigma (the fixed-corotated
            # first Piola P times F^T), which is the form MLS-MPM scatters; sigma = tau / J is the
            # stress per unit CURRENT area, which is what "Cauchy stress" means and what a von Mises
            # invariant is normally quoted from. Captured here, after any active stress has been added
            # and BEFORE the dt / p_vol rescale on the next line, so it is the material's stress and
            # not a momentum increment.
            sig = stress / J.abs().clamp_min(1e-9)[:, None, None]
            if getattr(p, "sigma", None) is None or p.sigma.shape != sig.shape:
                p.register_buffer("sigma", torch.zeros_like(sig))
            p.sigma.copy_(sig.detach())
        # p_vol THROUGH A CLONE UNDER GRAD, for the same reason as `mass` in the scatter: it is a
        # buffer a seed may rewrite, and its version counter moving after this line is read is what
        # a backward reports as "[4000, 1, 1] is at version 2; expected version 1".
        _pv = self._const_read(p.p_vol)
        stress = (-dt * 4 * inv_dx * inv_dx) * _pv[:, None, None] * stress
        affine = stress + mass[:, None, None] * C

        # `origin` -- the grid's lower corner (see `MPMGrid`); None on every spec that does not declare it.
        # `per_parent` -- each body on its own block, re-centred here on the body (see `MPMGrid`).
        if getattr(g, "per_parent", None):
            _pi, _po = _pp_frame(H, g, p, X, getattr(p, "occ", None), recentre=getattr(self, "_zeroes_grid", True))
            fx, weight, flat = bspline(X - _po[_pi], inv_dx, offsets, g.shape, periodic)
            flat = _pp_flat(flat, _pi, g.n_local, offsets.shape[0])
            gm, gmv, gc = g.m, g.mv, g.c
        else:
            fx, weight, flat = bspline(X if getattr(g, "origin", None) is None else X - g.origin,
                                       inv_dx, offsets, g.shape, periodic)
        # DORMANT particles (occ==0, e.g. a agent_grow reserve) contribute NOTHING to the grid:
        # mask the scatter weights by occupancy. Byte-identical when all particles are live.
        occ = getattr(p, "occ", None)
        if occ is not None:
            weight = weight * (occ > 0).to(weight.dtype)[:, None]
        dpos_phys = (offsets[None] - fx[:, None, :]) * dx
        mom = mass[:, None, None] * V[:, None, :] + self._affine_dpos(affine, dpos_phys)
        # THE GRID IS SHARED BY EVERY SET THAT SCATTERS INTO IT, so only the FIRST scatter of a
        # micro-step may zero it. Allocating fresh zeros unconditionally keeps only the LAST set:
        # the nucleus deposits its momentum, the cytosol's scatter throws it away, and the grid
        # solve runs on the cytosol alone.
        #
        # IT DOES NOT ANNOUNCE ITSELF. The nucleus then falls SLOWER than gravity and is crushed
        # in mid-air with nothing touching it, which reads as "the nucleus is too soft" and is
        # actually "the nucleus is gathering a velocity field it never contributed to": inside the
        # cavity the cytosol deposits no mass, so `gmv / gm.clamp(1e-10)` there is the B-spline
        # tails divided by nearly nothing. It only bites on multi-SET MPM -- multi-material within
        # one set uses `types:` and one scatter -- so single-set runs are unaffected.
        #
        # WHICH SCATTER ZEROES IS STATIC, so it is not asked at run time: a per-substep counter
        # would be a side effect no CUDA-graph replay can reproduce, and it would make dynamo
        # recompile every substep by specialising on an integer attribute. The engine binds the
        # i-th occurrence of a token to the i-th instance, so occurrence 0 for a given grid is
        # always the first in a substep, and it stamps `_zeroes_grid` when `inst` is built.
        _fresh = getattr(self, "_zeroes_grid", True)
        # WRITE INTO THE FIELD'S OWN BUFFERS, never a fresh allocation. Assigning
        # `g.m = torch.zeros(...)` rebinds the attribute to new storage every substep; a captured
        # graph holds the address it saw at capture time, so the replay silently writes somewhere
        # the rest of the run no longer reads. That failure is SILENT: capture succeeds, nothing
        # raises, and the checksum is simply wrong.
        # FUNCTIONAL UNDER GRAD, IN PLACE OTHERWISE, and the two are the same numbers. Autograd
        # cannot use the in-place form: the grid is zeroed and re-accumulated every substep, so the
        # tensor a later substep needs for its backward has already been overwritten, and an
        # inverse rollout dies on "a variable needed for gradient computation has been modified by
        # an inplace operation". The `differentiable` implementations at the end of this file are
        # where that trade-off, its cost and its citations live.
        gm, gmv, gc = g.m, g.mv, g.c
        liquid = getattr(p, "is_liquid", None)
        if self.colour_from_face is not None:
            lw = (weight * (mass * self._face_colour(p, H).to(mass.dtype))[:, None]).reshape(-1)
        else:
            lw = ((weight * (mass * liquid.to(mass.dtype))[:, None]).reshape(-1)
                  if _const_any(self, "_c_liquid", liquid) else None)
        # THE DEPOSIT IS A HOOK, so an alternative IMPLEMENTATION can make it functional without
        # this body changing at all -- see MPMScatterDiff. In place here, as it has always been.
        self._deposit(g, gm, gmv, gc, flat, weight, mass, mom, D, _fresh, lw)
        return {}

    def _face_colour(self, p, H=None):
        """[n] the colour weight of every point under `colour_from_face`: w[face], the point's `face` block read as 0 lateral,
        1 apical, 2 basal (clamped); with `face_block`, the apical and basal weight is the point's CELL's block value."""
        if "face" not in p.state_schema:
            raise ValueError("mpm_scatter: colour_from_face needs a `face` block on the point set (0 lateral, 1 apical, 2 basal)")
        f0, _ = p.state_schema["face"]
        face = p.state[:, f0].long().clamp(0, 2)
        w = self._face_w.to(p.state.device)[face]
        if self.face_block and H is not None:
            g = H.field(self.grid)
            Cl = H.level(getattr(g, "per_parent", None) or self.face_cells)
            fc = Cl.state[:, Cl.state_schema[self.face_block][0]].to(w.dtype)[p.parent.long().clamp(0, Cl.state.shape[0] - 1)]
            w = torch.where(face == 0, w, fc)
        return w

    def _deposit(self, g, gm, gmv, gc, flat, weight, mass, mom, D, fresh, lw):
        """Zero (if this is the substep's first scatter) and accumulate, IN PLACE."""
        if fresh:
            gm.zero_(); gmv.zero_(); gc.zero_()
        gm.index_add_(0, flat, (weight * mass[:, None]).reshape(-1))
        gmv.index_add_(0, flat, (weight[..., None] * mom).reshape(-1, D))
        if lw is not None:
            gc.index_add_(0, flat, lw)


@register_operator("mpm_grid_update", family="mpm", set="field", kind="field", title="solve grid velocities",
                   equation=r"""$$\mathbf v_g = \frac{\mathbf p_g}{m_g},
\qquad
\mathbf v_g \mathrel{+}= \Delta t\,\mathbf f^{\,\text{CSF}}_g,
\qquad
\mathbf v_g \leftarrow \mathrm{BC}(\mathbf v_g)$$""")
class MPMGridUpdate(MPMWrites, FieldUpdate):
    r"""the grid solve, the second step of the MLS-MPM cycle and the only place the whole system is coupled: momentum becomes velocity, body forces and the continuum surface force are added, the walls and plates are imposed.
    mpm_grid -> mpm_grid: reads and writes the grid field in place; no set is involved.

        \mathbf v_g = \frac{\mathbf p_g}{m_g}, \qquad \mathbf v_g \mathrel{+}= \Delta t\,\mathbf f^{\,\text{CSF}}_g, \qquad \mathbf v_g \leftarrow \mathrm{BC}(\mathbf v_g)
        v_g            the velocity of grid node g
        p_g            its momentum
        m_g            its mass (nodes below `mass_floor`, a mass that scales with the particle mass, get zero)
        \Delta t       `dt_sub`, the substep
        f^{CSF}_g      the continuum surface force sigma kappa n from the mollified colour field (`cell_surface_tension`, `csf_*`)
        BC             the walls (`wall_damp` the restitution, `wall_friction`), the closing plates (`plate_*`), the obstacles and buoyancy (`buoyancy`, `rho_ref`)

    Reference: Hu, Y. et al. (2018). A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. ACM Trans. Graph. 37(4):150; Sulsky, D., Chen, Z. & Schreyer, H. L. (1994). A particle method for history-dependent materials. Comput. Methods Appl. Mech. Eng. 118:179-196. The CSF: Brackbill, J. U., Kothe, D. B. & Zemach, C. (1992). J. Comput. Phys. 100:335-354."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"The grid solve. Each grid node turns its summed momentum $\mathbf p_g$ into a velocity, $\mathbf v_g=\mathbf "
        r"p_g/m_g$ ($m_g$ the node's summed mass). Where `cell_surface_tension` is set it adds the surface-tension force "
        r"$\mathbf f^{\,\text{CSF}}_g$ of the liquid interface over the substep $\Delta t$ (`dt_sub`). Then the boundary "
        r"conditions $\mathrm{BC}$ are imposed: the box walls (`wall_damp` the restitution, `wall_friction` the friction), "
        r"obstacles and closing plates stop any velocity into them. It is the only place the whole system is coupled."
    )
    PARAM_UNITS = {'buoyancy': 'acceleration', 'csf_band': 'fraction', 'csf_eps': 'dimensionless', 'csf_mass_floor': 'mass', 'csf_rho': 'density', 'csf_smooth': 'count', 'dt_sub': 'time', 'mass_floor': 'mass', 'plate_axis': 'count', 'plate_centre': 'length', 'plate_close_from': 'frames', 'plate_close_to': 'frames', 'plate_gap_half': 'length', 'plate_gap_half_end': 'length', 'rho_ref': 'density', 'cell_surface_tension': 'energy_per_area', 'wall_damp': 'fraction', 'wall_friction': 'fraction'}
    SYMBOLS = {'v_g': 'the velocity of grid node g', 'p_g': 'its momentum', 'm_g': 'its mass (nodes below `mass_floor`, a mass that scales with the particle mass, get zero)', '\\Delta t': '`dt_sub`, the substep', 'f^{CSF}_g': 'the continuum surface force sigma kappa n from the mollified colour field (`cell_surface_tension`, `csf_*`)', 'BC': 'the walls (`wall_damp` the restitution, `wall_friction`), the closing plates (`plate_*`), the obstacles and buoyancy (`buoyancy`, `rho_ref`)'}

    EMIT = None                                 # field->field grid solve: writes grid velocity in place; returns {} — no integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []                        # no required params — all optional (grid from `at:`, engine-injected)
    MECHANISM_TAGS = ["grid_solve", "surface_tension", "boundary_conditions"]
    PARAM_ROLES = {"dt_sub": "substep_timestep", "cell_surface_tension": "interface_cohesion",
                   "plate_gap_half": "free_half_gap", "plate_gap_half_end": "final_free_half_gap",
                   "plate_close_from": "frame_closing_starts", "plate_close_to": "frame_closing_ends",
                   "plate_axis": "confined_axis", "plate_centre": "gap_centre_on_axis",
                   "wall_damp": "wall_restitution", "wall_friction": "wall_coulomb_coefficient",
                   "csf_rho": "liquid_reference_density",
                   "csf_band": "interface_fraction_band", "csf_smooth": "colour_mollify_passes", 'buoyancy': 'buoyant_acceleration_per_unit_density_deficit', 'buoyancy_dir': 'its_direction', 'csf_eps': 'regularisation_of_the_normal', 'csf_mass_floor': 'mass_floor_of_the_csf_division', 'mass_floor': 'mass_below_which_a_node_is_empty', 'rho_ref': 'reference_density_of_the_buoyancy'}
    REFERENCE = "Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150; Sulsky, D. et al. (1994)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_grid")
        self.dt_sub = float(params.get("dt_sub", 2e-4))
        self.surface_tension = float(params.get("cell_surface_tension", 0.0))
        self.mass_floor = params.get("mass_floor", None)      # None -> derived from dx, rho
        # THE SAME DEFECT, AND HERE IT IS WORSE. The surface-tension term converts a nodal force
        # to an acceleration by dividing by nodal mass, and a FIXED floor on that division is a
        # mass -- so it must scale with the per-particle mass of the run, which falls as the
        # particle count rises. A floor left at a fixed epsilon binds on a large fraction of
        # occupied nodes in a fine run and attenuates the term there by an order of magnitude.
        # Surface tension acts AT THE INTERFACE, which is exactly where nodal mass is lowest, so
        # the floor weakens it precisely where it is supposed to work, and progressively worse the
        # more particles a specification uses. Setting `csf_rho` makes this floor inert.
        self.csf_mass_floor = params.get("csf_mass_floor", None)   # None -> derived
        # WHAT "COLOUR = 1" MEANS, and why the floor above is only treating a symptom.
        # `mpm_scatter` deposits `weight * mass * is_liquid`, so `gc` is a liquid MASS PER NODE in
        # absolute units -- on an all-liquid spec it is bitwise the mass field -- and not the
        # dimensionless volume fraction (0 in air, 1 in liquid) that Brackbill's
        # f = sigma * kappa * grad(c) is written for.
        #
        # Setting `csf_rho` to the liquid's density divides by the mass of one FULL liquid cell,
        # rho * dx^D, and that single division is the repair. Without it the delivered acceleration
        # is short by exactly that factor, so `cell_surface_tension` is not a tension: it is a tension
        # divided by a cell mass, and its PHYSICAL MEANING CHANGES WITH `n_grid`. At n_grid 192 a
        # declared 60 is a physical sigma of order 1e-5, a Bond number in the thousands -- gravity
        # beating capillarity by four orders of magnitude -- which is the whole content of "sigma
        # 0 -> 150 moves the centre of mass by 0.0001 of the box".
        #
        # DEFAULT 0.0 KEEPS THE OLD PATH BIT-IDENTICAL, the branch being a static Python `if` and
        # `range(0)` emitting no ops, so CUDA-graph capture is unchanged. A spec that sets it MUST
        # re-fit its sigma -- sigma_new = sigma_old * rho * dx^D for parity, or from a Bond number
        # for physics, rho*g*L^2 giving Bond 1.
        self.csf_rho = float(params.get("csf_rho", 0.0))
        # THE INTERFACE TEST NEEDS THE SAME REFERENCE, and for want of it selects the bulk.
        # `gmag > 0.02 * gmag.max()` is a percentile of a running maximum with no absolute scale,
        # and the P2G shot noise at 8 particles per cell gives the BULK a |grad c| of the same order
        # as the interface's, so a 2% cut admits nearly every occupied node. It is an occupancy mask
        # wearing an interface's name. An absolute band on the volume fraction cuts it to 21.7% and
        # 25.8% of occupied at `csf_band: 0.2`.
        #
        # THE UPPER BOUND IS THE LOAD-BEARING HALF. At sigma 0.64 (Bond 1) on the drop, the band
        # 0.2 < c < 0.8 gives spread r90 0.311 with gm_max 2.2e-6 against one full cell's 1.1e-6 --
        # clean -- while `c > 0.1` with NO upper bound, which is what the relative threshold
        # effectively does, implodes to r90 0.0006 on 422 surviving occupied nodes. The bulk noise
        # force is harmless today only because everything is 1e6 too small; it detonates the moment
        # the gain is right, so the two parameters have to move together.
        self.csf_band = float(params.get("csf_band", 0.0))
        # MOLLIFY BEFORE THE SECOND DERIVATIVE -- worth little at 96^3 and a lot at 192^3. Band-median
        # kappa on a B-spline-splatted sphere with real 8-ppc shot noise, as a fraction of the exact
        # 2/R: normalised colour reads 1.07 unsmoothed / 1.02 with two passes at 96^3, and 1.27 /
        # 1.06 at 192^3. The SHIPPED raw colour reads 0.70 at 96^3 and 0.08 at 192^3, because
        # noise-driven |div n| grows as alpha/dx while the true curvature stays at 2/R. It does not
        # change the trajectory (r90 0.302 unsmoothed vs 0.306 smoothed at Bond 1) and it leaks
        # colour into empty cells, which is why the band below carries an explicit mass clause.
        self.csf_smooth = int(params.get("csf_smooth", 0))
        self.csf_eps = params.get("csf_eps", None)                 # None -> derived (a 1/length)
        # A BAND WITHOUT A REFERENCE DENSITY IS A SILENT OFF-SWITCH: `c` would still be a mass, of
        # order rho*dx^D ~ 1e-6, so `c > 0.2` is false everywhere, the mask is empty and the term
        # vanishes without a word. Refuse the combination rather than run it.
        # SAY SO, LOUDLY, RATHER THAN LET A MEANINGLESS NUMBER THROUGH. A spec that asks for surface
        # tension without a reference density gets the legacy path, which is what keeps old runs
        # reproducible -- but nothing about the yaml says the number is a tension divided by a cell
        # mass, and nothing warns that it will mean something different if `n_grid` is touched.
        if self.surface_tension > 0.0 and self.csf_rho <= 0.0:
            import warnings
            warnings.warn(
                f"mpm_grid_update: cell_surface_tension={self.surface_tension:g} is being applied to an "
                f"UNNORMALISED colour field (csf_rho unset), so it is a tension divided by the mass "
                f"of one cell, rho*dx^D. It is ~1/(rho*dx^D) too small in physical units and its "
                f"meaning CHANGES WITH n_grid. Set csf_rho to the liquid's density and csf_band "
                f"(0.2) and re-fit sigma from a Bond number: sigma = rho*g*L^2 / Bond.",
                RuntimeWarning, stacklevel=2)
        if self.csf_band > 0.0 and self.csf_rho <= 0.0:
            raise ValueError(
                "mpm_grid_update: csf_band needs csf_rho -- the band is a test on the liquid "
                "VOLUME FRACTION, and without csf_rho the colour is still a mass (~rho*dx^D), so "
                "the band would select nothing and surface tension would be silently disabled. "
                "Set csf_rho to the liquid's density.")
        # BUOYANCY, IN THE SAME PLACE AND FOR THE SAME REASON AS SURFACE TENSION: it depends on the
        # LOCAL DENSITY, and only the grid knows what a particle is displacing.
        #
        # IT CANNOT BE A PER-PARTICLE FORCE HERE, and that was measured rather than assumed. The
        # `gravity` operator emits an ACCELERATION the substep consumes as `a_ext`, so every
        # particle receives the same `a` and mass cancels out of the grid velocity entirely. A
        # one-cell run with two liquid species at densities 0.6 and 1.8, mixed, gave a light-heavy
        # separation of -0.0020 over 1,200 frames -- zero, and faintly the wrong sign. Nothing
        # sorts under a uniform acceleration.
        #
        # THE FORM IS ARCHIMEDES ON THE NODE. With rho = m_node / dx^D, a node heavier than the
        # reference falls and a node lighter than it rises:
        #     dv = dt * g * (rho - rho_ref) / rho
        # so a node AT the reference density feels nothing, which is what makes this buoyancy and
        # not a second gravity. `rho_ref` is the fluid the body is displacing -- for a mixture, its
        # mean density -- and leaving it at 0 reduces the term to plain gravity, so the default is
        # off (`buoyancy: 0`) and no existing run changes.
        self.buoyancy = float(params.get("buoyancy", 0.0))
        self.rho_ref = float(params.get("rho_ref", 1.0))
        _bd = params.get("buoyancy_dir")
        self.buoy_dir = [float(x) for x in _bd] if _bd else None
        self.wall_damp = float(params.get("wall_damp", 1.0))
        # `wall_friction` -- COULOMB FRICTION AT THE DOMAIN WALLS, beside `wall_damp` because the two
        # are the two coefficients a rigid wall has: restitution for the normal direction and
        # friction for the tangential one. It is a parameter of THIS operator and not an operator of
        # its own for a reason that is physics, not taste: Coulomb's rule spends the NORMAL IMPULSE
        # the wall absorbs -- how hard the node hit -- and that number exists only here, in the
        # instant before the normal velocity is clamped away. A separate operator scheduled after
        # this one would have to guess it (gravity times dt for a body at rest on the floor, right
        # for resting and wrong for every impact).
        #
        # WHAT `wall_damp` IS NOT. It multiplies the tangential velocity by a fraction every
        # substep, which is viscous damping: it scales with speed, so a fast body loses more than a
        # slow one and a body at rest on a slope still creeps. Coulomb is the opposite -- a fixed
        # budget set by the normal load, independent of speed -- which is what stops thirty cows
        # sliding across the floor of si_obj_cow.
        #
        # THE RULE, per wall node with inward normal speed v_n > 0 and tangential velocity v_t:
        #
        #     |v_t| <= mu v_n   ->  v_t = 0                        (stick)
        #     |v_t| >  mu v_n   ->  v_t = v_t (1 - mu v_n / |v_t|) (slip, reduced by the budget)
        #
        # Stomakhin, A., Schroeder, C., Chai, L., Teran, J. & Selle, A. (2013). A material point
        # method for snow simulation. ACM Trans. Graph. 32(4):102, sec. 8 (boundary conditions).
        # Default 0 = today's frictionless wall, byte-identical: the branch is skipped entirely.
        self.wall_friction = float(params.get("wall_friction", 0.0))
        if self.wall_friction < 0:
            raise ValueError(f"wall_friction must be >= 0, got {self.wall_friction}")
        if self.wall_friction > 0 and not getattr(self, "HONOURS_WALL_FRICTION", True):
            print(f"[warn] {type(self).__name__}: wall_friction {self.wall_friction:g} is NOT applied "
                  f"by this implementation -- only the default mpm_grid_update imposes Coulomb "
                  f"friction at the walls. The run continues frictionless.", flush=True)
        # A MOVING PLATEN, AS A GRID VELOCITY BOUNDARY CONDITION: clamp the normal grid velocity
        # of every node beyond the plate to the PLATE'S OWN velocity, in the same place and the
        # same way as the reflective domain walls.
        #
        # IT CANNOT BE A POSITION PROJECTION. Stress comes from the deformation gradient F, and
        # `mpm_strain` updates F from the VELOCITY GRADIENT read off the grid, so teleporting a
        # particle -- which is what `plate_confine` does, after the substep block -- never enters
        # that path. The constitutive model is never told it has been compressed, so no stress
        # builds and nothing bulges sideways: the body's height falls, its equatorial radius does
        # not move, and its volume falls with the height. That is a CROP, not a squash.
        #
        # A rigid obstacle cannot do it either: `_walls3d` zeroes grid velocity inside a solid,
        # which stops material entering but cannot expel what is already there. A plate has to
        # CARRY a velocity, so the field near it is compressive, F contracts along the axis, and
        # the lateral bulge and the volume conservation come out of the physics.
        self.plate_axis = params.get("plate_axis", None)
        self.plate_axis = None if self.plate_axis is None else int(self.plate_axis)
        self.plate_centre = self.tunable(params.get("plate_centre"), 0.5, tensor=True)
        self.plate_gap_half = float(params.get("plate_gap_half", 0.0) or 0.0)
        _pge = params.get("plate_gap_half_end", None)
        self.plate_gap_half_end = None if _pge is None else float(_pge)
        self.plate_close_from = int(params.get("plate_close_from", 0))
        self.plate_close_to = int(params.get("plate_close_to", 0))
        self._wall_key = None; self._wall_cache = None
        self._wall3d_key = None; self._wall3d_cache = None

    def _plate_bc(self, H, g, gv, dev, dt):
        """Two rigid plates closing along `plate_axis`, as a velocity condition on the grid.

        Frictionless and one-sided, the standard MPM rigid-body collision: a node the plate has
        reached may not move further into it, and is carried at the plate's own speed. Everything
        else -- the compression, the lateral bulge, the volume -- follows from the solve rather
        than being imposed.
        """
        ax = self.plate_axis
        gap, v_plate = self.plate_state(H)
        shape = g.shape
        idx = (torch.arange(shape[ax], device=dev) + 0.5) * g.dx
        shp = [1] * g.dim
        shp[ax] = shape[ax]
        coord = (idx - self.plate_centre).view(shp)                  # signed distance from the centre

        gvv = gv.view(*shape, g.dim)
        c = gvv[..., ax]
        # Top plate moves at -|v|, bottom at +|v|; a covered node is clamped so it cannot outrun
        # the plate outward. Clamping rather than assigning leaves material free to move INWARD
        # faster than the plate, which is what being squeezed out from between them looks like.
        top = coord >= gap
        bot = coord <= -gap
        c = torch.where(top, c.clamp(max=-abs(v_plate)), c)
        c = torch.where(bot, c.clamp(min=abs(v_plate)), c)
        gvv[..., ax] = c
        return gvv.view(g.n_cells, g.dim)

    def plate_state(self, H):
        """(current half-gap, plate closing speed in world units per second)."""
        g0, g1 = self.plate_gap_half, self.plate_gap_half_end
        if g1 is None or self.plate_close_to <= self.plate_close_from:
            return g0, 0.0
        span = self.plate_close_to - self.plate_close_from
        # THE FRAME IS READ AS A TENSOR, and that is not a stylistic choice. A captured CUDA graph
        # bakes python constants in at capture time, so a gap computed from `int(H.frame)` freezes
        # at whatever it was on the capture tick: the plates would sit still for the whole run and
        # nothing would say so -- the same silent shape as the `H.micro` recompile defect.
        # `H.frame_t` is a device scalar the engine fills in place each tick, OUTSIDE the graph,
        # so the kernels read the current value on every replay. Everything downstream of `u` is
        # therefore a tensor op; `torch.where` rather than a python branch for the same reason.
        fr = getattr(H, "frame_t", None)
        if fr is None:                                  # engine too old / called outside run()
            fr = torch.as_tensor(float(getattr(H, "frame", 0)))
        u = ((fr.double() - self.plate_close_from) / span).clamp(0.0, 1.0)
        gap = g0 + u * (g1 - g0)
        # Per FRAME in the schedule, per SECOND on the grid: the grid solve advances by a substep
        # dt, so the closing rate has to be divided by the frame's own dt to become a velocity.
        # Zero once the plates have arrived, or they keep pushing on a gap that no longer changes.
        dt_frame = float(getattr(H, "dt", 0.0) or 0.0) or 1.0
        rate = ((g1 - g0) / span) / dt_frame
        v = torch.where(u < 1.0, torch.full_like(u, rate), torch.zeros_like(u))
        return gap, v

    def _walls3d(self, H, g, dev):
        """Rasterize 3D obstacles onto the grid (cached). Formats: a box
        [x0,y0,z0,x1,y1,z1] (6 values) or a sphere [cx,cy,cz,r] (4 values). Returns a
        flat [n_cells] bool mask of solid (obstacle-occupied) cells."""
        key = g.shape
        if self._wall3d_key == key and self._wall3d_cache is not None:
            return self._wall3d_cache
        nx, ny, nz = g.shape
        walls = torch.zeros(nx, ny, nz, dtype=torch.bool, device=dev)
        obs = list(getattr(H, "obstacles", []) or [])
        if obs:
            xs = (torch.arange(nx, device=dev) + 0.5) * g.dx
            ys = (torch.arange(ny, device=dev) + 0.5) * g.dx
            zs = (torch.arange(nz, device=dev) + 0.5) * g.dx
            gx = xs[:, None, None]; gy = ys[None, :, None]; gz = zs[None, None, :]
            for o in obs:
                v = [float(x) for x in o]
                if len(v) == 6:                                  # box [x0,y0,z0,x1,y1,z1]
                    walls = walls | ((gx >= v[0]) & (gx <= v[3]) & (gy >= v[1]) & (gy <= v[4])
                                     & (gz >= v[2]) & (gz <= v[5]))
                elif len(v) == 4:                                # sphere [cx,cy,cz,r]
                    walls = walls | (((gx - v[0]) ** 2 + (gy - v[1]) ** 2 + (gz - v[2]) ** 2) <= v[3] ** 2)
        walls = walls.reshape(-1)
        self._wall3d_key = key; self._wall3d_cache = walls
        return walls

    def _walls(self, H, g, dev):
        key = (g.nx, g.ny)
        if self._wall_key == key and self._wall_cache is not None:
            return self._wall_cache
        walls = torch.zeros(g.nx, g.ny, dtype=torch.bool, device=dev)
        obs = list(getattr(H, "obstacles", []) or [])
        if obs:
            xs = (torch.arange(g.nx, device=dev) + 0.5) * g.dx
            ys = (torch.arange(g.ny, device=dev) + 0.5) * g.dx
            gx = xs[:, None].expand(g.nx, g.ny); gy = ys[None, :].expand(g.nx, g.ny)
            for rect in obs:
                v = [float(x) for x in rect]
                if len(v) == 4:
                    walls = walls | ((gx >= v[0]) & (gx <= v[2]) & (gy >= v[1]) & (gy <= v[3]))
                elif len(v) == 3:
                    walls = walls | (((gx - v[0]) ** 2 + (gy - v[1]) ** 2) <= v[2] ** 2)
        walls = walls.reshape(-1)
        self._wall_key = key; self._wall_cache = walls
        return walls

    def _wall_bc_2d(self, gv, nx, ny, dev, wd):
        """Reflective domain walls, 2D. EXTRACTED VERBATIM so `default` is unchanged, and made a
        method so an implementation can replace THIS and nothing else.

        It is 8 of the 113 aten calls a 2D grid solve issues and 66% of a 2D frame, because
        `gv[lox, :, 0] = ...` is boolean-mask indexing: it lowers to `index_put_`, which goes
        through `nonzero`, which SYNCHRONISES. Eight of those per call at 18 substeps is ~144
        pipeline drains a frame. The 3D branch expresses the identical condition with `torch.where`
        on broadcast masks and has none -- see `mpm_grid_update[implementation: nosync]`.
        """
        gv = gv.view(nx, ny, 2)
        ix = torch.arange(nx, device=dev); iy = torch.arange(ny, device=dev); bnd = 3
        lox, hix = ix < bnd, ix > nx - bnd
        loy, hiy = iy < bnd, iy > ny - bnd
        gv[lox, :, 0] = gv[lox, :, 0].clamp(min=0); gv[hix, :, 0] = gv[hix, :, 0].clamp(max=0)
        gv[:, loy, 1] = gv[:, loy, 1].clamp(min=0); gv[:, hiy, 1] = gv[:, hiy, 1].clamp(max=0)
        if wd != 1.0:
            gl = gv[lox, :, 1]; gv[lox, :, 1] = torch.where(gl > 0, gl * wd, gl)
            gh = gv[hix, :, 1]; gv[hix, :, 1] = torch.where(gh > 0, gh * wd, gh)
            gv[:, loy, 0] = gv[:, loy, 0] * wd
            gv[:, hiy, 0] = gv[:, hiy, 0] * wd
        return gv.view(nx * ny, 2)

    def _const(self, name, g):
        """A declared value wins; otherwise the constant is derived from THIS grid's cell size and
        the liquid's own density. `csf_rho` is that density when it is given (it is, by definition,
        the divisor that turns deposited mass into a volume fraction); `rho_ref` otherwise, whose
        default 1.0 is exactly the reference these numbers were chosen against."""
        v = getattr(self, name)
        if v is not None:
            return float(v)
        rho = self.csf_rho if self.csf_rho > 0.0 else self.rho_ref
        return _scale_constant(name, float(g.dx), rho)

    def forward(self, H, mask=None):
        g = H.field(self.at); dev = g.m.device
        # A GRID WITH AN `origin` (see `MPMGrid`) KEEPS ITS WALLS AT ITS OWN FACES, which this solve
        # already does -- it works in node indices. Plates and obstacles are stated in WORLD units,
        # and this body places them in the grid's frame, so with an origin they would be misplaced.
        if getattr(g, "origin", None) is not None and (self.plate_axis is not None
                                                       or getattr(H, "obstacles", None)):
            raise NotImplementedError("mpm_grid_update: plates / obstacles on a grid with an `origin` "
                                      "are not implemented (they are placed in the grid's frame)")
        dt = sub_dt(H, self.dt_sub)
        _mass_floor = self._const("mass_floor", g)
        # `per_parent` BLOCKS ARE FREE: a body's block moves with it and holds it with a margin, so there
        # is no wall to impose -- momentum over mass (plus the uniform body force) is the whole solve.
        if getattr(g, "per_parent", None):
            gv = g.mv / g.m.clamp(min=_mass_floor)[:, None]
            _bf = getattr(H, "_mpm_body_accel", None)
            if _bf is not None:
                gv = gv + dt * _bf
            g.v = torch.where((g.m > _mass_floor)[:, None], gv, torch.zeros_like(gv))
            return {}
        _csf_floor = self._const("csf_mass_floor", g)
        _csf_eps = self._const("csf_eps", g)
        nx, ny, inv_dx, dx = g.nx, g.ny, g.inv_dx, g.dx
        D = g.dim
        periodic = bool(getattr(H, "periodic", False))
        wd = self.wall_damp
        gm, gmv, gc = g.m, g.mv, g.c
        # THE MASS FLOOR IS ABSOLUTE AND THE PARTICLE MASS IS NOT. `p_vol = body_volume /
        # per_parent`, so per-particle mass falls as 1/N: 1.5e-08 at 945k particles, 1.4e-09 at
        # 10M, 1.4e-10 at 100M. A node carrying one particle at a typical tail weight of the
        # B-spline holds 1.05e-09 at 945k -- comfortably above the floor -- and 9.9e-11 at 10M,
        # BELOW it. Once the floor binds, `gmv / 1e-10` is smaller than the true `gmv / gm`, so an
        # isolated particle's gathered velocity is scaled DOWN every substep and it can never
        # accelerate under gravity. It hangs in mid-air, which is what a 10M run showed: 21,826
        # particles suspended up to y = 0.60 above a bulk surface at y = 0.115, descending 0.13 in
        # 3.6 s where free fall at g = 16 would be 104.
        #
        # Kept as a PARAMETER at its historical value so no existing run changes; a spec whose
        # particle mass has outgrown it sets `mass_floor` smaller.
        gv = gmv / gm.clamp(min=_mass_floor)[:, None]
        # THE BODY FORCE, IF mpm_scatter HANDED IT OVER (`body_force: grid`). Applied here and not
        # in the scatter because this is after the division by nodal mass: as a pure addition to a
        # velocity it carries no mass factor, so `gm.clamp` cannot attenuate it and a node holding
        # one lone particle receives exactly the same `dt * a` as a node in the bulk. That is the
        # canonical MLS-MPM ordering (Hu et al. 2018; Taichi mpm88/mpm99 apply gravity immediately
        # after `grid_v = (1/grid_m) * grid_v` and before the wall conditions, which is the order
        # kept here). Absent the handover the attribute is None and this is not even a tensor op.
        _bf = getattr(H, "_mpm_body_accel", None)
        if _bf is not None:
            gv = gv + dt * _bf

        if D == 2:                                                  # --- 2D: verbatim (bit-identical) ---
            surf = self.surface_tension
            # CACHED, not dropped. Unlike the material masks this one is not run-constant in
            # PRINCIPLE -- gc is rebuilt every substep -- but it is settled by the first grid
            # solve, which always runs after every scatter of the substep. Deleting the guard
            # instead would turn gv into gv + 0.0 on a no-liquid spec and flip -0.0 to +0.0.
            if getattr(self, "_c_csf", None) is None:
                self._c_csf = bool(surf > 0.0 and bool((gc > 0).any()))
            if self._c_csf:                                        # CSF continuum surface force
                # SAME MISSING REFERENCE AS 3D, SAME REPAIR, and the 2D corpus is the reason it is
                # opt-in rather than applied: material_two_drops_st runs g = 0, so its Bond number
                # is ZERO and a tension 1/(rho*dx^2) = 36,864x too small is still the only force in
                # the scene -- measured Rg 0.13406 -> 0.10341 and r90 0.19205 -> 0.14136 over 4000
                # frames, against a sigma = 0 twin that does not move at all (Rg identical to five
                # decimals for 601 frames). That demo is correct AS WRITTEN and must stay so.
                # What the defect costs is scenes where gravity competes, and the resolution
                # dependence: the same yaml number is a different physical tension at every n_grid.
                if self.csf_rho > 0.0:
                    c = (gc / (self.csf_rho * dx * dx)).view(nx, ny)   # liquid VOLUME FRACTION
                else:
                    c = gc.view(nx, ny)                                # legacy: raw liquid MASS
                for _ in range(self.csf_smooth):        # separable [1/4,1/2,1/4]; no-op when 0
                    for k in range(2):
                        c = 0.25 * torch.roll(c, 1, k) + 0.5 * c + 0.25 * torch.roll(c, -1, k)
                cx = (torch.roll(c, -1, 0) - torch.roll(c, 1, 0)) * (0.5 * inv_dx)
                cy = (torch.roll(c, -1, 1) - torch.roll(c, 1, 1)) * (0.5 * inv_dx)
                gmag = torch.sqrt(cx * cx + cy * cy); eps = _csf_eps
                nxg, nyg = cx / (gmag + eps), cy / (gmag + eps)
                kappa = -((torch.roll(nxg, -1, 0) - torch.roll(nxg, 1, 0)) * (0.5 * inv_dx)
                          + (torch.roll(nyg, -1, 1) - torch.roll(nyg, 1, 1)) * (0.5 * inv_dx))
                _gain2 = 1.0 / max(1.0 - 2.0 * self.csf_band, 1e-6) if self.csf_band > 0 else 1.0
                if self.csf_band > 0.0:
                    _mfull = self.csf_rho * dx * dx
                    fmask = ((c > self.csf_band) & (c < 1.0 - self.csf_band)
                             & (gm.view(nx, ny) > self.csf_band * _mfull)).to(c.dtype) * _gain2
                else:
                    fmask = (gmag > 0.02 * gmag.max()).to(c.dtype)
                stfx = (surf * kappa * cx * fmask).view(-1); stfy = (surf * kappa * cy * fmask).view(-1)
                inv_m = (dx * dx) / gm.clamp(min=_csf_floor)
                gv = gv + dt * torch.stack([stfx * inv_m, stfy * inv_m], dim=1)

            if not periodic:
                gv = self._wall_bc_2d(gv, nx, ny, dev, wd)
            walls = self._walls(H, g, dev)
            if wd != 1.0 and _const_any(self, "_c_walls2d", walls):  # friction in cells touching obstacles
                w2 = walls.view(nx, ny)
                near = (torch.roll(w2, 1, 0) | torch.roll(w2, -1, 0)
                        | torch.roll(w2, 1, 1) | torch.roll(w2, -1, 1)) & ~w2
                gvv = gv.view(nx, ny, 2); gx_ = gvv[..., 0]; gy_ = gvv[..., 1]
                gvv[..., 0] = torch.where(near, gx_ * wd, gx_)
                gvv[..., 1] = torch.where(near & (gy_ > 0), gy_ * wd, gy_)
                gv = gvv.view(nx * ny, 2)
            gv = torch.where(walls[:, None], torch.zeros_like(gv), gv)  # interior wall BC
        else:                                                       # --- 3D: CSF + reflective box walls ---
            # SURFACE TENSION IN 3D. With this term confined to the `D == 2` branch above,
            # `cell_surface_tension` in a `dim: 3` spec is read, stored and never used -- and a liquid
            # cytosol dropped on the floor then spreads into a one-particle-thick puddle over the
            # whole domain, which reads as "the liquid is too soft" and is actually "there is no
            # cohesion term at all". A liquid
            # in MLS-MPM has mu = 0 by construction -- it resists volume change and nothing else --
            # so surface tension is not a refinement on top of the constitutive model, it is the
            # ONLY thing that makes a droplet a droplet.
            #
            # SAME CSF AS 2D, WRITTEN OVER D AXES. Brackbill's continuum surface force: the liquid
            # colour `gc` deposited by the scatter is smooth across the interface, its gradient is
            # the interface normal times the interface's sharpness, and the divergence of the unit
            # normal is the curvature. f = sigma * kappa * grad(c) then pulls a bulge in and pushes
            # a dimple out. The 2D code above is left untouched rather than generalised, because it
            # is on the promotion path and must stay bit-identical.
            surf = self.surface_tension
            if getattr(self, "_c_csf", None) is None:
                self._c_csf = bool(surf > 0.0 and bool((gc > 0).any()))
            if self._c_csf:
                if self.csf_rho > 0.0:
                    c = (gc / (self.csf_rho * dx ** D)).view(*g.shape)   # liquid VOLUME FRACTION
                else:
                    c = gc.view(*g.shape)                                # legacy: raw liquid MASS
                for _ in range(self.csf_smooth):        # separable [1/4,1/2,1/4]; no-op when 0
                    for k in range(D):
                        c = 0.25 * torch.roll(c, 1, k) + 0.5 * c + 0.25 * torch.roll(c, -1, k)
                grad = [(torch.roll(c, -1, k) - torch.roll(c, 1, k)) * (0.5 * inv_dx)
                        for k in range(D)]
                gmag = torch.sqrt(sum(gk * gk for gk in grad))
                eps = _csf_eps
                nrm = [gk / (gmag + eps) for gk in grad]
                kappa = -sum((torch.roll(nrm[k], -1, k) - torch.roll(nrm[k], 1, k)) * (0.5 * inv_dx)
                             for k in range(D))
                # ONLY WHERE THERE IS AN INTERFACE. A gradient-magnitude test admits nearly every
                # occupied node; the colour band admits a fifth of them. The mass clause is also
                # what makes `csf_mass_floor` inert rather than load-bearing -- the lightest node
                # the band admits holds csf_band * rho * dx^D, orders above the floor.
                #
                # THE BAND IS A GAIN, NOT ONLY A FILTER, SO COMPENSATE FOR IT. The force is
                # f = sigma*kappa*grad(c), and the total impulse across an interface telescopes:
                # int grad(c) dx = c_in - c_out. Restricting it to `band < c < 1-band` therefore
                # delivers exactly (1 - 2*band) of the tension, whatever the shape of the profile
                # -- the fundamental theorem, not an approximation -- so a band of 0.2 throws away
                # 40% of sigma by construction. A Young-Laplace ladder (mean(J) = 1 - 2 sigma/(R K),
                # nothing fitted) follows that prediction and goes to 1 as the band closes.
                #
                # Dividing by (1 - 2*band) restores the magnitude and leaves the band its real job,
                # which is to keep the force OFF the bulk: the test was never about how much
                # tension to apply, only about where.
                _gain = 1.0 / max(1.0 - 2.0 * self.csf_band, 1e-6) if self.csf_band > 0 else 1.0
                if self.csf_band > 0.0:
                    _mfull = self.csf_rho * dx ** D
                    fmask = ((c > self.csf_band) & (c < 1.0 - self.csf_band)
                             & (gm.view(*g.shape) > self.csf_band * _mfull)).to(c.dtype) * _gain
                else:
                    fmask = (gmag > 0.02 * gmag.max()).to(c.dtype)
                inv_m = (dx ** D) / gm.clamp(min=_csf_floor)   # force -> acceleration
                gv = gv + dt * torch.stack(
                    [(surf * kappa * grad[k] * fmask).view(-1) * inv_m for k in range(D)], dim=1)
            if not periodic:
                shape = g.shape; bnd = 3
                gv = gv.view(*shape, D)
                # THE COMPONENTS AS A LIST UNDER GRAD, and the same numbers either way. The values
                # are already computed with `torch.where`; it is writing them back with
                # `gv[..., k] = ck` that autograd cannot take -- an in-place index_put into a VIEW,
                # which is the AsStridedBackward an inverse rollout dies on. Keeping the three
                # components as separate tensors and stacking them at the end is the same
                # arithmetic in the same order: each axis clamps its own component, then damps the
                # others, reading whatever the previous axis left.
                _grad = self.FUNCTIONAL          # an implementation flag, not a mode
                comps = [gv[..., k] for k in range(D)] if _grad else None
                for k in range(D):
                    n_k = shape[k]
                    idx = torch.arange(n_k, device=dev)
                    shp = [1] * D; shp[k] = n_k
                    lo_m = (idx < bnd).view(shp); hi_m = (idx > n_k - bnd).view(shp)
                    ck = comps[k] if _grad else gv[..., k]
                    # THE NORMAL IMPULSE, READ BEFORE IT IS CLAMPED AWAY: the inward speed the wall
                    # is about to cancel. This is the one quantity Coulomb friction spends, and it
                    # is why friction lives here and not in an operator of its own.
                    mu = self.wall_friction
                    if mu > 0.0:
                        v_in = torch.where(lo_m, (-ck).clamp(min=0), torch.zeros_like(ck)) \
                             + torch.where(hi_m, ck.clamp(min=0), torch.zeros_like(ck))
                    ck = torch.where(lo_m, ck.clamp(min=0), ck)     # don't penetrate the wall
                    ck = torch.where(hi_m, ck.clamp(max=0), ck)
                    if _grad:
                        comps[k] = ck
                    else:
                        gv[..., k] = ck
                    if mu > 0.0:
                        # STICK OR SLIP, per node. The tangential speed is the norm over the other
                        # axes; the budget is mu times the impulse; whatever is left after paying it
                        # is the slip. `factor` is 0 where the budget covers the whole tangential
                        # speed (stick) and (1 - mu v_n / |v_t|) otherwise, applied to every
                        # tangential component so the direction is kept and only the speed drops.
                        slab = lo_m | hi_m
                        tang = [(comps[j] if _grad else gv[..., j]) for j in range(D) if j != k]
                        vt = torch.sqrt(sum(c * c for c in tang)).clamp(min=1e-12)
                        factor = (1.0 - mu * v_in / vt).clamp(min=0.0)
                        for j in range(D):
                            if j == k:
                                continue
                            cj = comps[j] if _grad else gv[..., j]
                            if _grad:
                                comps[j] = torch.where(slab, cj * factor, cj)
                            else:
                                gv[..., j] = torch.where(slab, cj * factor, cj)
                    if wd != 1.0:                                   # tangential friction on the wall slabs
                        slab = lo_m | hi_m
                        for j in range(D):
                            if j == k:
                                continue
                            cj = comps[j] if _grad else gv[..., j]
                            if _grad:
                                comps[j] = torch.where(slab, cj * wd, cj)
                            else:
                                gv[..., j] = torch.where(slab, cj * wd, cj)
                gv = (torch.stack(comps, -1) if _grad else gv).view(g.n_cells, D)
            if self.plate_axis is not None and self.plate_gap_half > 0.0:
                gv = self._plate_bc(H, g, gv, dev, dt)
            walls = self._walls3d(H, g, dev)                        # solid 3D obstacles (box / sphere)
            if _const_any(self, "_c_walls3d", walls):
                gv = torch.where(walls[:, None], torch.zeros_like(gv), gv)   # no-slip: zero grid velocity inside
        # BUOYANCY IS NOT A DIMENSION BRANCH, and putting it here made it look like one. Inserted
        # in 40e1d0c9 between `if D == 2:` and its `else:`, it STOLE THAT ELSE: the 3D wall code
        # then ran whenever buoyancy was zero -- for a 2D spec too, where `_walls3d` unpacks
        # `nx, ny, nz = g.shape` and dies -- and was SKIPPED whenever buoyancy was on, so every 3D
        # buoyant run had no wall boundary condition at all. Both halves of that were invisible
        # until a 2D spec was written without buoyancy. It now sits after the branch, in both
        # dimensions, which is what a body force on the grid is.
        if self.buoyancy != 0.0:
            # rho at each node from the mass the particles deposited there. `gm` is a node mass and
            # dx^D its volume, so this is a genuine density and the comparison with `rho_ref` is
            # dimensionally honest rather than a tuned ratio.
            _rho = gm / (dx ** D)
            # BRANCH-FREE, BECAUSE A BOOLEAN-MASK ASSIGNMENT CANNOT BE CAPTURED. `_f[_act] = ...`
            # compiles to index_put, which needs `nonzero`, which is a device->host sync -- and a
            # sync inside a CUDA graph capture is `cudaErrorStreamCaptureUnsupported`, not a
            # slowdown. Both 3D buoyancy scenes died on it at the first captured substep with
            # "operation not permitted when stream is capturing". `torch.where` is the same
            # arithmetic with no data-dependent control flow; the clamp keeps the empty-node
            # division finite before the `where` discards it.
            _safe = _rho.clamp(min=1e-9)
            _f = torch.where(_rho > 1e-9, (_rho - self.rho_ref) / _safe,
                             torch.zeros_like(_rho))            # empty nodes have no buoyancy
            # BUILT ONCE, NOT PER SUBSTEP. Writing a python float into a device tensor --
            # `_dir[1] = -1.0` -- is a host-to-device copy, and that is as uncapturable as the
            # boolean mask above was: the same run died again, four lines further down, on the same
            # `cudaErrorStreamCaptureUnsupported`. The direction is a run constant, so it is
            # assembled on the host and moved once; the captured region only ever reads it.
            _key = (D, str(dev), gv.dtype)
            if getattr(self, "_dir_key", None) != _key:
                _v = [0.0] * D
                if self.buoy_dir is not None:
                    for _i, _x in enumerate(self.buoy_dir[:D]):
                        _v[_i] = float(_x)
                else:
                    # DOWN IS WHEREVER GRAVITY SAYS IT IS, and `gravity` says -y: it defaults to
                    # `gy = -g, gz = 0` in 3D as well as in 2D. Buoyancy must use the SAME axis or
                    # it pushes at right angles to the weight it is meant to oppose, and a bubble
                    # drifts sideways instead of rising. `buoyancy_dir` overrides the axis for a
                    # specification that needs another.
                    _v[1] = -1.0
                self._dir_cache = torch.tensor(_v, device=dev, dtype=gv.dtype)
                self._dir_key = _key
            _dir = self._dir_cache
            gv = gv + dt * self.buoyancy * _f[:, None] * _dir[None, :]
        self._write_v(g, gv)            # in place: a captured graph holds this address
        return {}


@register_operator("mpm_grid_viscosity", family="mpm", set="field", kind="field",
                   equation=r"""$$\big(\mathbf I-\nu\,\Delta t\,\mathbf L\big)\mathbf u^{\,\mathrm{new}}=\mathbf u^{\,\mathrm{old}},\qquad \nu=\frac{\eta}{\rho}$$""", title="damp grid shear implicitly")
class MPMGridViscosity(FieldUpdate):
    r"""implicit (backward-Euler) shear diffusion of the grid velocity, unconditionally stable where the explicit particle viscosity needs dt below dx^2 / 2 nu; a matrix-free conjugate gradient with no host sync, differentiable through the solve.
    field -> field on mpm_grid: runs after mpm_grid_update, in addition to it, rewriting the grid velocity in place.

        \big(\mathbf I-\nu\,\Delta t\,\mathbf L\big)\mathbf u^{\,\mathrm{new}}=\mathbf u^{\,\mathrm{old}},\qquad \nu=\frac{\eta}{\rho}
        I              the identity
        \nu            the kinematic viscosity eta / rho
        \Delta t       `dt_sub`, the substep
        L              the fluid-cell 6-point Laplacian
        u^{new}, u^{old} the grid velocity after and before the step
        \eta           `eta`, the dynamic viscosity
        \rho           `rho`, the density

    Reference: Batty, C. & Bridson, R. (2008). ACM Trans. Graph. 27(3):219 (variational implicit viscosity); Shao, H. et al. (2022). ACM Trans. Graph. 41(4):49 (UAAMG accelerator)."""
    PARAM_UNITS = {'cg_iters': 'count', 'dt_sub': 'time', 'eta': 'viscosity', 'rho': 'density'}
    SYMBOLS = {'I': 'the identity', '\\nu': 'the kinematic viscosity eta / rho', '\\Delta t': '`dt_sub`, the substep', 'L': 'the fluid-cell 6-point Laplacian', 'u^{new}, u^{old}': 'the grid velocity after and before the step', '\\eta': '`eta`, the dynamic viscosity', '\\rho': '`rho`, the density'}

    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["eta"]
    MECHANISM_TAGS = ["viscous_stress", "momentum_diffusion", "implicit", "unconditionally_stable"]
    PARAM_ROLES = {"eta": "dynamic_viscosity", "rho": "reference_density",
                   "cg_iters": "conjugate_gradient_iterations", 'dt_sub': 'substep'}
    REFERENCE = "Batty & Bridson (2008). ACM TOG 27(3):219; Shao et al. (2022). ACM TOG 41(4):49."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_grid")
        self.eta = float(params["eta"])
        self.rho = float(params.get("rho", 1000.0))
        self.dt_sub = float(params.get("dt_sub", 2e-4))
        self.cg_iters = int(params.get("cg_iters", 40))
        if self.eta < 0:
            raise ValueError(f"mpm_grid_viscosity: eta must be >= 0, got {self.eta}")

    def forward(self, H, mask=None):
        g = H.field(self.at)
        dt = sub_dt(H, self.dt_sub)
        a = (self.eta / self.rho) * dt / (g.dx * g.dx)      # a = nu dt / dx^2
        if a <= 0.0:
            return {}
        D = g.dim; shp = g.shape
        fluid = (g.m.view(shp) > 0)                          # diffuse only across fluid cells
        wrap = bool(getattr(H, "periodic", False))
        fm = fluid[..., None]

        def lap(u):                                          # graph Laplacian, zero-flux at fluid edge
            out = torch.zeros_like(u)
            for d in range(D):
                for s in (1, -1):
                    un = torch.roll(u, s, dims=d)
                    c = (fluid & torch.roll(fluid, s, dims=d))[..., None]
                    if not wrap:                             # a rolled-in face from the far side is not a neighbour
                        sl = [slice(None)] * (D + 1); sl[d] = (0 if s == 1 else -1)
                        c = c.clone(); c[tuple(sl)] = False
                    out = out + torch.where(c, un - u, torch.zeros_like(u))
            return out

        def A(u):                                            # backward-Euler operator
            return u - a * lap(u)

        b = g.v.view(*shp, D) * fm
        x = b.clone()
        r = b - A(x); p = r.clone(); rs = (r * r).sum()
        for _ in range(self.cg_iters):                       # fixed count: no host-side break under capture
            Ap = A(p); al = rs / (p * Ap).sum().clamp_min(1e-30)
            x = x + al * p; r = r - al * Ap; rs2 = (r * r).sum()
            p = r + (rs2 / rs.clamp_min(1e-30)) * p; rs = rs2
        g.v.copy_((x * fm).reshape(-1, D))
        return {}


# ==========================================================================================================
# `mpm_grid_update[implementation: nosync]` -- the 2D wall BC without boolean-mask indexing.
#
# WHY THIS EXISTS, MEASURED. In 3D `mpm_grid_update` is 2.5% of a frame at 5M particles and 12% at
# 945k. In 2D it is 66% -- 163 ms of 246 on `material_dam_break`, 109 of 163 on
# `material_active_swirl` -- and 2D is 58 of the 78 specs in config/material. The cause is not
# arithmetic: one 2D grid solve issues 113 aten calls, and 8 of them are `index_put_` from
# `gv[lox, :, 0] = ...`. Boolean-mask indexing lowers through `nonzero`, which SYNCHRONISES, so
# every one drains the pipeline; at 18 substeps that is ~144 drains a frame.
#
# THE MASKS ARE RUN-CONSTANT. `lox = ix < bnd` depends only on the grid extent, so nothing here is
# data-dependent -- the sync buys nothing at all. The 3D branch already writes the identical
# condition as `torch.where` on broadcast masks; this is the 2D branch written the same way.
#
# IT SHOULD BE BIT-IDENTICAL TO `default`, which is a much stronger claim than the `warp`
# implementations can make and is gated as such: same operations, same order, same reads. The
# ordering matters and is preserved -- the friction block reads the velocity AFTER the clamps, so
# `vy` carries steps 3-4 before step 5 uses it, and a row that is both `lox` and `hix` sees the
# first update before the second, exactly as the in-place version does.
# ==========================================================================================================
@register_operator("mpm_gather", "g2p", family="mpm", set="particle", kind="exchange", title="move points with the grid",
                   equation=r"""$$\mathbf v_i = \sum_g w_{ig}\,\mathbf v_g,
\qquad
\mathbf C_i = \frac{4}{\Delta x^{2}}\sum_g w_{ig}\,\mathbf v_g\,(\mathbf x_g-\mathbf x_i)^{\!\top},
\qquad
\mathbf x_i \mathrel{+}= \Delta t\,\mathbf v_i$$""")
class MPMGather(MPMWrites, Exchange):                 
    r"""grid to particle, the third step of the MLS-MPM cycle: each particle reads back a velocity and a velocity gradient from the nodes around it and is advected.
    mpm_grid -> particle: reads the grid, writes pos and vel in place (MAY_MUTATE_INTEGRATED_STATE: it advects inside the substep rather than returning a delta).

        \mathbf v_i = \sum_g w_{ig}\,\mathbf v_g, \qquad \mathbf C_i = \frac{4}{\Delta x^{2}}\sum_g w_{ig}\,\mathbf v_g\,(\mathbf x_g-\mathbf x_i)^{\!\top}, \qquad \mathbf x_i \mathrel{+}= \Delta t\,\mathbf v_i
        v_i            the velocity of particle i
        w_{ig}         the quadratic B-spline weight of node g at particle i
        v_g            the velocity of node g
        C_i            the APIC affine velocity matrix (local velocity gradient) carried to the next scatter and to mpm_strain
        \Delta x       the grid spacing
        x_g, x_i       the node and particle positions
        \Delta t       `dt_sub`, the substep

    Reference: Hu, Y. et al. (2018). A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. ACM Trans. Graph. 37(4):150; Sulsky, D., Chen, Z. & Schreyer, H. L. (1994). A particle method for history-dependent materials. Comput. Methods Appl. Mech. Eng. 118:179-196."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"The grid-to-particle transfer. Each point takes the grid velocities around it, $\mathbf v_i=\sum_g "
        r"w_{ig}\,\mathbf v_g$ ($w_{ig}$ the quadratic B-spline weight of node $g$ at point $i$), renews its affine "
        r"velocity matrix $\mathbf C_i=\tfrac{4}{\Delta x^{2}}\sum_g w_{ig}\,\mathbf v_g(\mathbf x_g-\mathbf "
        r"x_i)^{\!\top}$ ($\Delta x$ the grid spacing, $\mathbf x_g$ and $\mathbf x_i$ the node and point positions) and "
        r"moves, $\mathbf x_i\mathrel{+}=\Delta t\,\mathbf v_i$ ($\Delta t$ the substep `dt_sub`). The point's new "
        r"$\mathbf C_i$ is what the next scatter and the strain update read."
    )
    PARAM_UNITS = {'dt_sub': 'time', 'vmax': 'velocity', 'wall_contact': 'length', 'wall_contact_cells': 'count', 'wall_damp': 'fraction'}
    SYMBOLS = {'v_i': 'the velocity of particle i', 'w_{ig}': 'the quadratic B-spline weight of node g at particle i', 'v_g': 'the velocity of node g', 'C_i': 'the APIC affine velocity matrix (local velocity gradient) carried to the next scatter and to mpm_strain', '\\Delta x': 'the grid spacing', 'x_g, x_i': 'the node and particle positions', '\\Delta t': '`dt_sub`, the substep'}

    EMIT = None                                    # advects pos/vel inside the MPM substep (MAY_MUTATE_INTEGRATED_STATE); returns {} — no integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []                           # no required params — all optional (source grid defaults to `mpm_grid`)
    MAY_MUTATE_INTEGRATED_STATE = True             # advects pos/vel inside the substep (like the oracle)
    MECHANISM_TAGS = ["grid_to_particle", "advection"]
    PARAM_ROLES = {"dt_sub": "substep_timestep", "wall_damp": "wall_restitution",
                   "wall_contact": "contact_layer_thickness", "vmax": "speed_cap",
                   "wall_damp_mode": "restitution_application_rate", 'from': 'the_grid_field', 'wall_contact_cells': 'the_same_in_grid_cells'}
    REFERENCE = "Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150 (MLS-MPM G2P); Sulsky, D. et al. (1994)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.frm = params.get("from", "mpm_grid")
        self.dt_sub = float(params.get("dt_sub", 2e-4))
        self.wall_damp = float(params.get("wall_damp", 1.0))
        # A LENGTH, AND THEREFORE A TRAP IN ANY BOX THAT IS NOT 1 UNIT WIDE. The contact test is
        # `(x < cb) | (x > box[k] - cb)`, so in a 0.1 m box the historical 0.04 selects everything
        # but a 0.02 m sliver -- the entire fluid reads as permanently in wall contact and is
        # permanently damped. `wall_contact_cells` states it in the only scale the grid has:
        # 0.04 / (1/96) = 3.84 cells, so the default reproduces 0.04 exactly at n_grid 96 and
        # follows the world everywhere else. An explicit `wall_contact` still wins, for the specs
        # that tuned it.
        self.wall_contact_cells = float(params.get("wall_contact_cells", 3.84))
        self.wall_contact = params.get("wall_contact", None)
        if self.wall_contact is not None:
            self.wall_contact = float(self.wall_contact)
        # HOW OFTEN `wall_damp` IS APPLIED, and it is the difference between a restitution
        # coefficient and a decay rate. See the note in `forward`. `per_substep` is the historical
        # behaviour and stays the default so no existing run changes.
        self.wall_damp_mode = str(params.get("wall_damp_mode", "per_substep")).lower()
        if self.wall_damp_mode not in ("per_substep", "per_impact"):
            raise ValueError(f"mpm_gather: wall_damp_mode must be 'per_substep' or 'per_impact', "
                             f"got {self.wall_damp_mode!r}")
        self.vmax = float(params.get("vmax", 1e9))

    def _contact_band(self, g):
        """The contact-layer thickness, in world length. An explicit `wall_contact` wins; otherwise
        it is `wall_contact_cells * dx`, which follows the box instead of assuming it is 1 wide."""
        return self.wall_contact if self.wall_contact is not None \
            else self.wall_contact_cells * float(g.dx)

    def forward(self, H, mask=None):
        p = H.level(self.at); g = H.field(self.frm); dev = p.state.device
        dt = sub_dt(H, self.dt_sub)
        inv_dx, dx = g.inv_dx, g.dx
        D = p.F.shape[-1]
        periodic = bool(getattr(H, "periodic", False))
        # CACHED, because it is constant for the whole run and `float()` on a tensor element is a
        # host sync. Read fresh every call it cost a GPU->CPU round trip per substep, and under
        # torch.compile it broke the graph in the middle of the gather on every single call.
        # Computed once, the branch is False on every later call and never enters the traced graph.
        if getattr(self, "_box", None) is None:
            self._box = ([float(b) for b in g.world_size][:D] if getattr(g, "box_own", False) else
                         [float(b) for b in
                          getattr(H, "world_size", torch.tensor([g.width, 1.0]))][:D])
        box = self._box
        offsets = stencil_offsets(D, dev); S = offsets.shape[0]
        X, V = p.get("pos"), p.get("vel")
        # `origin` -- the grid's lower corner (see `MPMGrid`). Every test below is in the GRID's frame
        # (`_o` subtracted), and the new positions are handed back in the world's (`_o` added).
        _o = getattr(g, "origin", None)
        if getattr(g, "per_parent", None):
            _pi, _po = _pp_frame(H, g, p, X, getattr(p, "occ", None))
            _o = _po[_pi]
            box = [float(b) for b in g.world_size][:D]
        if _o is not None:
            X = X - _o
        fx, weight, flat = bspline(X, inv_dx, offsets, g.shape, periodic)
        if getattr(g, "per_parent", None):
            flat = _pp_flat(flat, _pi, g.n_local, S)
        gvn = g.v[flat].view(p.n, S, D)
        new_V = (weight[..., None] * gvn).sum(1)
        dpos_grid = offsets[None] - fx[:, None, :]
        new_C = 4 * inv_dx * self._outer_sum(weight, gvn, dpos_grid)
        new_V = torch.nan_to_num(new_V)
        if self.wall_damp != 1.0 and not periodic:                 # inelastic wall contact (solids)
            cb = self._contact_band(g)
            near = torch.zeros(p.n, dtype=torch.bool, device=dev)
            for k in range(D):
                near = near | (X[:, k] < cb) | (X[:, k] > box[k] - cb)
            liquid = getattr(p, "is_liquid", None)
            if liquid is not None:
                near = near & ~liquid
            # ONCE PER IMPACT, NOT ONCE PER SUBSTEP. `wall_damp` reads as a restitution coefficient
            # -- "keep 60% of the velocity on a bounce" -- and under `per_substep` it is not one:
            # the multiplier lands on every substep the particle spends inside `wall_contact`, so
            # one impact removes wall_damp ** (substeps in the layer). That exponent grows with grid
            # resolution, because a finer grid resolves the contact more stiffly and the body
            # lingers. MEASURED on material_3d_ball_drop, energy removed by the wall:
            #
            #     n_grid  64   2.4%      n_grid  96  71.0%      n_grid 128  70.8%
            #
            # -- the same spec, the same wall_damp, 68.6 percentage points apart. It is not even
            # monotonic across scenes: genA_code_star_ball loses MORE at 64 (84.1%) than at 96
            # (67.5%). A parameter whose meaning depends on the discretisation cannot be calibrated
            # once, which is why retuning every spec was the wrong fix.
            #
            # `per_impact` applies it on the RISING EDGE of contact only -- the substep where the
            # particle enters the layer -- so one impact costs exactly one multiplication whatever
            # the substep count. The edge state is a persistent per-particle buffer written IN
            # PLACE, so a captured graph keeps reading the address it saw.
            if self.wall_damp_mode == "per_impact":
                prev = getattr(p, "_wall_near", None)
                if prev is None:
                    p.register_buffer("_wall_near",
                                      torch.zeros(p.n, dtype=torch.bool, device=dev))
                    prev = p._wall_near
                near, _keep = near & ~prev, near
                prev.copy_(_keep)
            new_V = torch.where(near[:, None], new_V * self.wall_damp, new_V)
        sp = new_V.norm(dim=1, keepdim=True).clamp(min=1e-9)
        vmax = min(self.vmax, 0.4 * dx / dt)                       # CFL velocity cap
        new_V = new_V * (sp.clamp(max=vmax) / sp)
        new_C = torch.nan_to_num(new_C)
        Xn = torch.nan_to_num(X + dt * new_V, nan=0.5)
        if periodic:
            Xn = torch.stack([torch.remainder(Xn[:, k], box[k]) for k in range(D)], dim=1)
        else:
            Xn = torch.stack([Xn[:, k].clamp(2 * dx, box[k] - 2 * dx) for k in range(D)], dim=1)
        # DORMANT particles (occ==0, a agent_grow reserve) are FROZEN -- not advected -- so they sit as a
        # compact reservoir until agent_grow activates + repositions them. Byte-identical when all are live.
        occ = getattr(p, "occ", None)
        if occ is not None:
            live = occ > 0
            Xn = torch.where(live[:, None], Xn, X)
            new_V = torch.where(live[:, None], new_V, V)
            new_C = torch.where(live[:, None, None], new_C, p.C)
        # IN PLACE. Every read of X / V above happens before this write, and this operator declares
        # MAY_MUTATE_INTEGRATED_STATE, so the engine's tick-0 integration-invariant guard does not
        # apply to it. The clone-and-rebind it replaces gave `p.state` a new address every substep.
        if _o is not None:
            Xn = Xn + _o
        pa, pb = p.state_schema["pos"]; va, vb = p.state_schema["vel"]
        self._write_particles(p, pa, pb, va, vb, Xn, new_V, new_C)
        return {}


@register_operator("mpm_strain", family="mpm", set="particle", kind="lateral", title="update material strain",
                   equation=r"""$$\mathbf F \;\leftarrow\; (\mathbf I + \Delta t\,\mathbf C)\,\mathbf F
\qquad\text{(liquid: } \mathbf F\leftarrow J^{1/d}\mathbf I,\ J=\det\mathbf F\text{)}$$""")
class MPMStrain(MPMWrites, Lateral):
    r"""the material update, the fourth step of the MLS-MPM cycle: each particle advances its deformation gradient from the velocity gradient it gathered, then applies its material law (elastic keeps F, a liquid keeps only J, a plastic clips its singular values).
    particle -> particle: reads F and C, writes F and the derived material state in place.

        \mathbf F \;\leftarrow\; (\mathbf I + \Delta t\,\mathbf C)\,\mathbf F \qquad\text{(liquid: } \mathbf F\leftarrow J^{1/d}\mathbf I,\ J=\det\mathbf F\text{)}
        F              the deformation gradient, the identity for unstrained material
        I              the identity
        \Delta t       `dt_sub`, the substep
        C              the affine velocity gradient
        J              det F, the ratio of current to rest volume (`liquid_volume`: carried through F, or integrated as dJ/dt = J tr C)
        d              the dimension

    Reference: Hu, Y. et al. (2018). A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. ACM Trans. Graph. 37(4):150; Sulsky, D., Chen, Z. & Schreyer, H. L. (1994). A particle method for history-dependent materials. Comput. Methods Appl. Mech. Eng. 118:179-196."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"The material update. Each point advances its deformation gradient $\mathbf F$ (how far the material around the "
        r"point is stretched and sheared away from rest; the identity when unstrained) by the velocity gradient it just "
        r"gathered, $\mathbf F\leftarrow(\mathbf I+\Delta t\,\mathbf C)\,\mathbf F$, with $\mathbf C$ the point's affine "
        r"velocity matrix, $\Delta t$ the substep `dt_sub` and $\mathbf I$ the identity. Then its material law applies: an "
        r"elastic point keeps $\mathbf F$; a liquid forgets its shape and keeps only its volume, $\mathbf F\leftarrow "
        r"J^{1/d}\mathbf I$ ($J=\det\mathbf F$ the ratio of current to rest volume, $d$ the dimension); a viscoelastic "
        r"point relaxes $\mathbf F$ part of the way to that same state with its own time constant; snow clips the "
        r"principal stretches of $\mathbf F$ to a narrow band."
    )
    PARAM_UNITS = {'dt_sub': 'time'}
    SYMBOLS = {'F': 'the deformation gradient, the identity for unstrained material', 'I': 'the identity', '\\Delta t': '`dt_sub`, the substep', 'C': 'the affine velocity gradient', 'J': 'det F, the ratio of current to rest volume (`liquid_volume`: carried through F, or integrated as dJ/dt = J tr C)', 'd': 'the dimension'}

    EMIT = None                 # particle->particle: updates F + material in place; returns {} — no delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []        # no required params — all knobs optional (defaults in __init__)
    MECHANISM_TAGS = ["elastic_strain", "plastic_flow", "incompressible_volume"]
    PARAM_ROLES = {"dt_sub": "MLS-MPM substep dt", 'liquid_volume': 'det_or_trace_how_a_liquids_volume_is_advanced', 'tau_block': 'cell_block_of_the_viscoelastic_relaxation_time', 'tau_cells': 'cell_set_that_block_belongs_to'}
    REFERENCE = "Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150 (MLS-MPM); Sulsky, D. et al. (1994). Comput. Methods Appl. Mech. Eng. 118:179-196."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.dt_sub = float(params.get("dt_sub", 2e-4))
        # HOW A LIQUID'S VOLUME IS ADVANCED. `det` (the default) carries J through F exactly as
        # every existing spec has; `trace` uses J *= 1 + dt*tr(C) instead. See the long note in
        # forward() -- `det` loses a column's pressure at ~1.1%/s at rest and gets worse as the mesh
        # is refined. It is OPT-IN because it changes the numbers of every existing MPM
        # specification, and the default stays where it is until a gate says what the new ones are.
        self.liquid_volume = str(params.get("liquid_volume", "det")).lower()
        # `tau_block` (exp30, 2026-10-23): a width-1 block of the set `tau_cells` (the points' parent set) holding each CELL's
        # shape-memory time, read in place of the per-point `visco_tau` buffer: the Maxwell time as a per-cell knob.
        self.tau_block = params.get("tau_block")
        self.tau_cells = params.get("tau_cells", "cell")
        if self.liquid_volume not in ("det", "trace"):
            raise ValueError(f"liquid_volume must be 'det' or 'trace', got "
                             f"{self.liquid_volume!r}")

    def _tau(self, p, H):
        """Each point's shape-memory time: its cell's `tau_block` when declared (with the block's graph), else the `visco_tau` buffer."""
        if self.tau_block:
            Cl = H.level(self.tau_cells)
            return Cl.state[:, Cl.state_schema[self.tau_block][0]].to(p.F.dtype)[p.parent.long().clamp(0, Cl.state.shape[0] - 1)]
        return p.visco_tau

    def _visco_relax(self, Fv, a, D):
        """The Maxwell partial reset of the stretches: sig <- Jl + (sig - Jl) a, Jl = J^(1/D) the isotropic volume-keeping stretch,
        by an SVD (sig = principal stretches). The `differentiable` implementation rewrites it without the SVD."""
        U, sig, Vh = torch.linalg.svd(Fv)
        Jl = sig.prod(-1).clamp(min=1e-6).pow(1.0 / D)
        sig = Jl[:, None] + (sig - Jl[:, None]) * a[:, None]
        return U @ torch.diag_embed(sig) @ Vh

    def forward(self, H, mask=None):
        p = H.level(self.at); dev = p.state.device
        dt = sub_dt(H, self.dt_sub)
        D = p.F.shape[-1]
        eye = torch.eye(D, device=dev).expand(p.n, D, D)
        F = (eye + dt * p.C) @ p.F
        if D == 2:
            a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
            J = a * d - b * c
        else:
            J = torch.linalg.det(F)
        liquid = getattr(p, "is_liquid", None)
        if _const_any(self, "_c_liquid", liquid):                  # LIQUID: drop shape memory
            if self.liquid_volume == "trace":
                # A LIQUID AT REST BLEEDS ITS PRESSURE AWAY, and this is why.
                #
                # A liquid drops SHAPE memory (F becomes isotropic below), so its whole state is
                # the volume J -- carried through F, i.e. multiplied by det(I + dt*C) every substep.
                # The law being discretised is dJ/dt = J*tr(C), whose exact step is J *= exp(dt*trC).
                # Those agree to first order and NOT to second:
                #
                #     det(I + dt C)  = 1 + dt*trC + dt^2*(trC^2 - tr(C^2))/2 + dt^3*det C
                #     exp(dt*trC)    = 1 + dt*trC + dt^2* trC^2            /2 + ...
                #
                # The difference is -dt^2*tr(C^2)/2, and tr(C^2) is a SUM OF SQUARES: positive
                # whatever the sign of the noise, so it does not average out. Every substep
                # multiplies J by slightly less than it should, and a column that is not moving at
                # all loses its stored compression -- on the order of a percent per second.
                #
                # AND THE SCALING IDENTIFIES IT. Grid noise gives C ~ v_noise/dx and CFL gives
                # dt ~ dx, so the accumulated bias over a fixed TIME goes as dt*tr(C^2) ~ v^2/dx:
                # it gets WORSE as the mesh is refined, very nearly linearly in 1/dx. It is also
                # why `drag` helps, since drag suppresses exactly the v_noise that drives it.
                #
                # THE FIX, which is what Taichi's mpm88/mpm99 do for water: advance the volume by
                # its OWN first-order law, J *= 1 + dt*tr(C). No more accurate in dt -- both are
                # first order -- but it does not couple to the deviatoric part of C at all, so
                # noise has no preferred direction to push J in.
                trC = p.C.diagonal(dim1=-2, dim2=-1).sum(-1)
                J = torch.where(liquid, torch.linalg.det(p.F) * (1.0 + dt * trC), J)
            Jc = J.clamp(min=1e-6)
            Jl = torch.sqrt(Jc) if D == 2 else Jc.pow(1.0 / D)     # volume-preserving isotropic reset
            F = torch.where(liquid[:, None, None], eye * Jl[:, None, None], F)
        visco = getattr(p, "is_visco", None)
        if _const_any(self, "_c_visco", visco):                    # VISCOELASTIC (Maxwell): PARTIAL shape reset
            vm = visco                                             # relax F toward isotropic with time-constant tau,
            tau_p = self._tau(p, H)
            a = torch.exp(-dt / tau_p[vm].clamp(min=1e-6))         # memory retained this substep: a->1 elastic, a->0 liquid
            F = F.clone()
            F[vm] = self._visco_relax(F[vm], a, D)                 # keeping VOLUME (J) -> stress builds then decays
        snow = getattr(p, "is_snow", None)
        if _const_any(self, "_c_snow", snow):                      # SNOW: clamp singular values, harden via Jp
            sm = snow; Fs = F[sm]
            if Fs.shape[0] > 0:
                U, sig, Vh = torch.linalg.svd(Fs)
                if D == 3:                                          # proper-rotation sign fix (MPM_3D)
                    U = U.clone(); sig = sig.clone(); Vh = Vh.clone()
                    negU = torch.det(U) < 0
                    U[negU, :, -1] *= -1; sig[negU, -1] *= -1
                    negV = torch.det(Vh) < 0
                    Vh[negV, -1, :] *= -1; sig[negV, -1] *= -1
                sig_c = sig.clamp(1.0 - 2.5e-2, 1.0 + 7.5e-3)
                F = F.clone(); F[sm] = U @ torch.diag_embed(sig_c) @ Vh
                ratio = sig.prod(-1) / sig_c.prod(-1).clamp(min=1e-6)
                Jp = p.Jp.clone(); Jp[sm] = (Jp[sm] * ratio).clamp(0.6, 20.0)
                self._write_Jp(p, Jp)
        # DORMANT PARTICLES DO NOT DEFORM. `mpm_scatter` masks its weights by occupancy and
        # `mpm_gather` freezes occ==0 rather than advecting it, but this operator integrated F for the
        # reserve regardless -- so a particle waiting to be spawned accumulated an arbitrary deformation
        # for as long as it waited, and was then promoted into real material carrying it. Byte-identical
        # when every particle is live, which is every composition that has no reserve.
        occ = getattr(p, "occ", None)
        if occ is not None:
            live = (occ > 0)[:, None, None]
            F = torch.where(live, F, p.F)
        self._write_F(p, F)             # in place; every read of p.F above precedes it
        return {}


# ==========================================================================================================
# mpm_turgor -- the isotropic outward pressure a cell's interior holds against its cortex.
# ==========================================================================================================
@register_operator("mpm_turgor", "osmotic_pressure", family="mpm", set="particle", kind="lateral",
                   equation=r"""$$P=\frac{2\gamma}{R},\qquad P\simeq\lambda\,\epsilon\ \ \text{for a volume strain }\epsilon$$""", title="inflate cells with turgor pressure")
class MPMTurgor(Lateral):
    r"""gives a set an isotropic outward pressure, turgor or excess osmotic pressure, by writing a per-particle `turgor` buffer that mpm_scatter subtracts from the Kirchhoff stress: a self-limiting shift of the liquid's rest volume.
    particle -> particle: writes the `turgor` buffer (mode `constant` holds P fixed).

        P=\frac{2\gamma}{R},\qquad P\simeq\lambda\,\epsilon\ \ \text{for a volume strain }\epsilon
        P              `pressure`, the turgor, in the spec's stress unit (the same as `youngs`)
        \gamma         the cortical tension it balances (Young-Laplace)
        R              the cell radius
        \lambda        the liquid's bulk modulus (its Lame lambda)
        \epsilon       the volume strain at which the liquid's own law balances P

    Reference: van 't Hoff, J. H. (1887). Z. Phys. Chem. 1:481-508 (the osmotic pressure a turgor is); the balance against the cortex is Young-Laplace, P = 2 gamma / R."""
    PARAM_UNITS = {'pressure': 'stress'}
    SYMBOLS = {'P': "`pressure`, the turgor, in the spec's stress unit (the same as `youngs`)", '\\gamma': 'the cortical tension it balances (Young-Laplace)', 'R': 'the cell radius', '\\lambda': "the liquid's bulk modulus (its Lame lambda)", '\\epsilon': "the volume strain at which the liquid's own law balances P"}

    EMIT = None                 # particle->particle: writes the `turgor` buffer in place; no delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["pressure"]
    MECHANISM_TAGS = ["osmotic_pressure", "turgor", "volume_regulation"]
    PARAM_ROLES = {"pressure": "isotropic_outward_pressure", "mode": "pressure_law"}
    REFERENCE = ("Stewart, M.P. et al. (2011). Nature 469:226-230 (hydrostatic pressure and the "
                 "actomyosin cortex in mitotic rounding); Jiang, H. & Sun, S.X. (2013). Biophys. J. "
                 "105:609-619 (cell volume and osmotic pressure).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.pressure = float(params["pressure"])           # excess of interior over exterior pressure
        self.at = params.get("_at", "mpm_particle")
        # There is no `mode:` here. A vocabulary with one member is not an axis, and a second
        # turgor law would go on the `model:` axis where the registry can see it.
        if "mode" in params:
            raise ValueError(
                "mpm_turgor: `mode` is gone -- it only ever accepted 'constant'. A second turgor "
                "law would be `model:`, not a mode.")

    def forward(self, H, mask=None):
        p = H.level(self.at)
        P = torch.full((p.n,), self.pressure, device=p.state.device)
        # DORMANT PARTICLES CARRY NO PRESSURE. `mpm_scatter` masks a reserve particle's weights to
        # zero so it deposits nothing, but `store_stress` and any diagnostic reading `turgor` would
        # otherwise report a pressurised particle that is not in the material yet.
        occ = getattr(p, "occ", None)
        if occ is not None:
            P = P * (occ > 0).to(P.dtype)
        if mask is not None:
            P = P * mask.to(P.dtype)
        if getattr(p, "turgor", None) is None or p.turgor.shape != P.shape:
            p.register_buffer("turgor", P)
        else:
            p.turgor.copy_(P)
        return {}


@register_operator("mpm_turgor", model="cycle", family="mpm", set="particle", kind="lateral",
                   equation=r"""$$P_i=\lambda_i\,J^*_c\,(J^*_c-1),\qquad J^*_c=s_c\,2^{p_c}$$""", title="grow cell rest volume through the cycle")
class MPMTurgorCycle(MPMTurgor):
    r"""growth through the cell cycle as a shift of each cell's rest volume: every point of a cell is given the turgor that moves the zero of its own equation of state to J* = size0 2^progress.
    particle -[containment]-> cell: reads the cell's `cycle_progress` and `size0` (written by cell_cycle[mpm]) and each point's Lame lambda; writes the points' `turgor` buffer.

        P_i=\lambda_i\,J^*_c\,(J^*_c-1),\qquad J^*_c=s_c\,2^{p_c}
        P_i            the turgor of point i, stress
        \lambda_i      its Lame lambda
        J^*_c          the rest-volume ratio cell c is driven to
        s_c            `size0`, the cell's size at birth relative to the seeded volume
        p_c            `progress`, the cell's cycle progress, 0 at birth to 1 at division (doubling the volume)

    Reference: Rodriguez, E. K., Hoger, A. & McCulloch, A. D. (1994). Stress-dependent finite growth in soft elastic tissues. J. Biomech. 27:455-467 (growth as a change of the rest configuration)."""
    PARAM_UNITS = {'max_ratio': 'dimensionless'}
    SYMBOLS = {'P_i': 'the turgor of point i, stress', '\\lambda_i': 'its Lame lambda', 'J^*_c': 'the rest-volume ratio cell c is driven to', 's_c': "`size0`, the cell's size at birth relative to the seeded volume", 'p_c': "`progress`, the cell's cycle progress, 0 at birth to 1 at division (doubling the volume)"}
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["growth", "turgor", "volume_regulation", "cell_cycle"]
    PARAM_ROLES = {"cell_set": "the_cells_owning_the_points", "progress": "cell_block_cycle_progress",
                   "size0": "cell_block_birth_size", "max_ratio": "cap_on_J_star"}
    REFERENCE = ("Rodriguez, E. K., Hoger, A. & McCulloch, A. D. (1994). J. Biomech. 27:455-467.")

    def __init__(self, params, device="cpu"):
        Lateral.__init__(self, params, device)
        self.at = params.get("_at", "mpm_particle")
        self.cell_set = str(params.get("cell_set", "cell"))
        self.progress = str(params.get("progress", "cycle_progress"))
        self.size0 = str(params.get("size0", "size0"))
        self.max_ratio = float(params.get("max_ratio", 2.5))

    def forward(self, H, mask=None):
        p = H.level(self.at)
        C = H.level(self.cell_set)
        par = p.parent.reshape(-1).long()
        prog = C.get(self.progress).reshape(-1)
        s0 = C.get(self.size0).reshape(-1)
        # size0 = 0 IS A CELL THE CYCLE HAS NOT SEEN YET (the first frame runs this before `cell_cycle`'s first
        # call): its rest volume stays where it was seeded, J* = 1, rather than collapsing toward the clamp
        Js = torch.where(s0 > 0, s0 * torch.pow(2.0, prog), torch.ones_like(s0))
        Js = Js.clamp(0.2, self.max_ratio)[par].to(p.la.dtype)
        P = p.la * Js * (Js - 1.0)
        occ = getattr(p, "occ", None)
        if occ is not None:
            P = P * (occ > 0).to(P.dtype)
        if getattr(p, "turgor", None) is None or p.turgor.shape != P.shape:
            p.register_buffer("turgor", P.detach().clone())
        else:
            p.turgor.copy_(P.detach())
        return {}


# ==========================================================================================================
# mpm_viscosity -- the Newtonian viscous stress a liquid in this MPM does not otherwise have.
# ==========================================================================================================
@register_operator("mpm_viscosity", family="mpm", set="particle", kind="lateral", title="damp shear inside liquid cells",
                   equation=r"""$$\boldsymbol\tau_p=\eta\big(\mathbf C_p+\mathbf C_p^{\mathsf T}\big)$$""")
class MPMViscosity(Lateral):
    r"""shear dissipation for a liquid, which the constitutive law does not otherwise provide (a liquid has mu = 0, so nothing else resists shear).
    particle -> particle: reads the affine velocity gradient C, emits a viscous stress the scatter adds (liquid particles only with `liquid_only`).

        \boldsymbol\tau_p=\eta\big(\mathbf C_p+\mathbf C_p^{\mathsf T}\big)
        \tau_p         the viscous Kirchhoff stress of particle p
        \eta           `eta`, the dynamic viscosity (tau = 2 eta D, D the strain rate)
        C_p            the particle's affine velocity gradient

    Reference: the Navier-Stokes viscous term; Hu, Y. et al. (2018). A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. ACM Trans. Graph. 37(4):150; Sulsky, D., Chen, Z. & Schreyer, H. L. (1994). A particle method for history-dependent materials. Comput. Methods Appl. Mech. Eng. 118:179-196."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"The viscous stress of a liquid. A liquid has shear modulus $\mu=0$, so nothing else resists shear; each point "
        r"therefore computes the Newtonian viscous stress $\boldsymbol\tau=\eta(\mathbf C+\mathbf C^{\mathsf T})$, with "
        r"$\eta$ the dynamic viscosity (`eta`) and $\mathbf C$ the point's affine velocity matrix, i.e. its local velocity "
        r"gradient. The scatter adds $\boldsymbol\tau$ to the point's Kirchhoff stress. By default it acts on liquid "
        r"points only (`liquid_only`)."
    )
    PARAM_UNITS = {'eta': 'viscosity'}
    SYMBOLS = {'\\tau_p': 'the viscous Kirchhoff stress of particle p', '\\eta': '`eta`, the dynamic viscosity (tau = 2 eta D, D the strain rate)', 'C_p': "the particle's affine velocity gradient"}

    EMIT = None                 # writes `p.extra_stress`, consumed by mpm_scatter in the substep
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["eta"]
    MECHANISM_TAGS = ["viscous_stress", "momentum_diffusion", "dissipation"]
    PARAM_ROLES = {"eta": "dynamic_viscosity", "liquid_only": "restrict_to_liquid_particles", "eta_block": "cell_block_with_each_cells_viscosity", "eta_cells": "the_cell_set_of_that_block"}
    REFERENCE = ("Batchelor, G.K. (1967). An Introduction to Fluid Dynamics, ch. 3 (Newtonian "
                 "stress); Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150 (MLS-MPM, C = grad v).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.eta = float(params["eta"])                    # DYNAMIC viscosity mu_dyn
        # ONLY THE LIQUID, BY DEFAULT. A solid already carries a deviatoric stress through mu > 0;
        # adding a viscous one on top makes it Kelvin-Voigt, which is a different material and
        # should be asked for rather than inherited.
        self.liquid_only = bool(params.get("liquid_only", True))
        if self.eta < 0:
            raise ValueError(f"mpm_viscosity: eta must be >= 0, got {self.eta}")
        # `eta_block` (exp33, 2026-10-26: the viscosity as a knob the optimisation holds per cell): a width-1 CELL block eta_c read for every particle of the cell instead of the scalar (`eta_cells`: the cell
        # set, default "cell"). Absent: the scalar, as before
        self.eta_block = params.get("eta_block")
        self.eta_cells = str(params.get("eta_cells", "cell"))

    def forward(self, H, mask=None):
        p = H.level(self.at)
        C = p.C
        # PER-PARTICLE eta WHERE THE SPEC DECLARED ONE, this operator's scalar everywhere else.
        # `entities.py` builds the buffer from the type table exactly as it builds mu and la, and
        # leaves NaN where a type said nothing -- so a spec that never mentions per-type eta has no
        # buffer at all and this is the same multiply it always was.
        _pe = getattr(p, "eta", None)
        if self.eta_block:
            Cl = H.level(self.eta_cells)
            _ec = Cl.state[:, Cl.state_schema[self.eta_block][0]].to(C.dtype)
            _e = _ec[p.parent.long().clamp(0, Cl.state.shape[0] - 1)]
            tau = _e[:, None, None] * (C + C.transpose(-2, -1))
        elif _pe is not None:
            _e = torch.where(torch.isnan(_pe), torch.full_like(_pe, self.eta), _pe)
            tau = _e[:, None, None] * (C + C.transpose(-2, -1))
        else:
            tau = self.eta * (C + C.transpose(-2, -1))
        if self.liquid_only:
            liq = getattr(p, "is_liquid", None)
            if _const_any(self, "_c_liquid", liq):
                tau = tau * liq[:, None, None].to(tau.dtype)
        occ = getattr(p, "occ", None)
        if occ is not None:
            tau = tau * (occ > 0).to(tau.dtype)[:, None, None]
        if mask is not None:
            tau = tau * mask.to(tau.dtype)[:, None, None]
        # ON THE SET, NOT ON THE HIERARCHY. `H.extra_stress` is one slot shared by every particle
        # set in the model, which is right while there is one cloud and wrong the moment there are
        # two: an animal of 98,808 points and a pool of 140,000 both scatter into one grid, and the
        # water's viscous stress was added to the ANIMAL's -- "the size of tensor a (98808) must
        # match the size of tensor b (140000)". The same flaw `active_stress` carries, and
        # `polar_active_stress` already avoids by writing `p.act_stress`.
        #
        # The global is still written when this is the only set using it, because specs and
        # prototypes read it back, and a reader of `H.extra_stress` on a one-cloud model must keep
        # meaning what it meant. `mpm_scatter` prefers the per-set buffer and falls back.
        self._write_stress(p, H, tau)
        return {}

    def _write_stress(self, p, H, tau):
        """The stress into the set's persistent buffer, IN PLACE (a captured graph keeps the same storage); the
        `differentiable` implementation rebinds instead."""
        prev = getattr(p, "extra_stress", None)
        if prev is None or prev.shape != tau.shape:
            p.register_buffer("extra_stress", tau.detach().clone())
        else:
            prev.copy_(tau)                     # persistent buffer -> safe inside a captured graph
        _g = getattr(H, "extra_stress", None)
        if _g is None or _g.shape == tau.shape:
            H.extra_stress = p.extra_stress


@register_operator("mpm_viscosity", implementation="differentiable", family="mpm", set="particle", kind="lateral",
                   title="damp shear inside liquid cells, differentiable", equation=r"""$$\boldsymbol\tau_p=\eta\big(\mathbf C_p+\mathbf C_p^{\mathsf T}\big)$$""")
class MPMViscosityDiff(MPMViscosity):
    r"""mpm_viscosity with the viscous stress rebound rather than copied into the buffer, so a tape through the substep survives the backward; same values, only the binding differs.
    particle -> particle: as mpm_viscosity.

        \boldsymbol\tau_p=\eta\big(\mathbf C_p+\mathbf C_p^{\mathsf T}\big)
        \tau_p         the viscous Kirchhoff stress of particle p
        \eta           `eta`, the dynamic viscosity (tau = 2 eta D, D the strain rate)
        C_p            the particle's affine velocity gradient

    Reference: as mpm_viscosity."""
    PARAM_UNITS = {'eta': 'viscosity'}
    SYMBOLS = {'\\tau_p': 'the viscous Kirchhoff stress of particle p', '\\eta': '`eta`, the dynamic viscosity (tau = 2 eta D, D the strain rate)', 'C_p': "the particle's affine velocity gradient"}
    DIFFERENTIABLE = True

    def _write_stress(self, p, H, tau):
        if getattr(p, "extra_stress", None) is None:
            p.register_buffer("extra_stress", tau)
        else:
            p.extra_stress = tau
        _g = getattr(H, "extra_stress", None)
        if _g is None or _g.shape == tau.shape:
            H.extra_stress = p.extra_stress


@register_operator("mpm_anchor", family="mechanics", set="particle", kind="lateral",
                   equation=r"""$$\mathbf a_p=k\,(\mathbf x^{\mathrm{rest}}_p-\mathbf x_p)$$""", title="anchor a body with springs")
class MPMAnchor(Lateral):
    r"""a spring to a rest position: what holds a body that must not drift, without pinning it rigidly; `where` selects the extent (a boundary ring or the whole substrate), not the law.
    particle -> particle: reads pos, emits an external acceleration the MPM substep consumes.

        \mathbf a_p=k\,(\mathbf x^{\mathrm{rest}}_p-\mathbf x_p)
        a_p            the acceleration of particle p
        k              `k`, the anchor stiffness, per time squared (a clamp when stiff, a compliant substrate when soft)
        x^{rest}_p     the position the particle was seeded at (frame 0)
        x_p            its position

    Reference: Hu, Y. et al. (2018). A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. ACM Trans. Graph. 37(4):150; Sulsky, D., Chen, Z. & Schreyer, H. L. (1994). A particle method for history-dependent materials. Comput. Methods Appl. Mech. Eng. 118:179-196."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"A spring to the rest position. Each selected point is pulled back toward its rest position $\mathbf "
        r"x^{\mathrm{rest}}_p$ (where it was when the operator first ran: frame 0 unless gated), $\mathbf a_p=k\,(\mathbf "
        r"x^{\mathrm{rest}}_p-\mathbf x_p)$, with $k$ the stiffness `k` (per time squared) and $\mathbf x_p$ the point's "
        r"position. `applies_to` chooses which points: a boundary ring, the whole substrate or a band. It holds a body in "
        r"place without pinning it rigidly: a large $k$ clamps, a small $k$ is a compliant support."
    )
    PARAM_UNITS = {'k': 'acceleration', 'ring': 'count'}
    SYMBOLS = {'a_p': 'the acceleration of particle p', 'k': '`k`, the anchor stiffness, per time squared (a clamp when stiff, a compliant substrate when soft)', 'x^{rest}_p': 'the position the particle was seeded at (frame 0)', 'x_p': 'its position'}

    EMIT = "mpm_acceleration"   # consumed by the MPM substep as a_ext, not engine-integrated
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["k"]
    MECHANISM_TAGS = ["substrate_anchor", "boundary_condition", "rest_restoring"]
    PARAM_ROLES = {"k": "anchor_stiffness", "ring": "boundary_width", "mode": "anchor_extent", "band": "rest_band_axis_lo_hi", 'applies_to': 'which_particles_are_anchored'}
    REFERENCE = "Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150 (MLS-MPM)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.k = self.tunable(params["k"], None, tensor=True)
        # WHERE THE ANCHOR ACTS -- a VALUE, not a mode, and renamed to say so. `boundary` holds a
        # ring of particles, `substrate` holds all of them; the spring, the rest state and the
        # hypothesis are identical either way -- the same claim about the attachment applied over
        # two different regions -- which is what makes this a VALUE and not a model.
        if "mode" in params:
            raise ValueError(
                "mpm_anchor: `mode` is now `applies_to` (boundary | substrate). It is a value, not "
                "a model -- the spring and the rest state are the same either way.")
        self.applies_to = str(params.get("applies_to", "boundary"))
        self.band = params.get("band")                        # applies_to: band -> [axis, lo, hi] of the rest position
        if self.applies_to == "band" and (not isinstance(self.band, (list, tuple)) or len(self.band) != 3):
            raise ValueError("mpm_anchor: `applies_to: band` needs `band: [axis, lo, hi]`")
        self.ring = params.get("ring", None)                  # None -> derived (a length)
        # The rest state is the positions when the operator first runs: frame 0 if ungated, but gate it with
        # `after_frame` and it becomes whatever the run had drifted to by then (a merge-then-separate
        # sequence would hold the merged drop, not pull back to the two it started as).
        if "anchor_frame" in params:
            raise ValueError(
                "mpm_anchor: `anchor_frame` was removed. The rest state is the positions at the operator's "
                "first call (frame 0 unless the operator is gated by `after_frame`).")
        self.at = params.get("_at", "particle")
        self._rest = None
        self._sel = None

    def _init(self, lvl, H=None):
        # THE BAND IS A LENGTH, AND ITS NATURAL SCALE IS THE CELL. An anchor ring narrower than a
        # couple of cells cannot be resolved by the grid that carries the force, so `dx` is the
        # right yardstick -- 3.84 cells, which is exactly 0.04 at n_grid 96 and therefore leaves
        # every existing spec where it was. Falls back to the historical constant when no MPM grid
        # is in the run (the operator does not require one).
        self._rest = lvl.get("pos").clone()                   # undeformed sheet (frame 0)
        if self.ring is None:
            # H.fields is a torch ModuleDict: it supports `in` and `[...]` but NOT `.get`.
            _fl = getattr(H, "fields", None) if H is not None else None
            _g = _fl["mpm_grid"] if (_fl is not None and "mpm_grid" in _fl) else None
            self._ring = (_scale_constant("ring", float(_g.dx)) if _g is not None
                          else _CONST_DIMS["ring"][0])
        else:
            self._ring = float(self.ring)
        if self.applies_to == "substrate":
            self._sel = torch.ones(self._rest.shape[0], dtype=torch.bool, device=self._rest.device)
        elif self.applies_to == "band":
            # A BAND OF THE REST EXTENT along one axis (`band: [axis, lo, hi]`, world units): the feet of a standing
            # body held to its pedestal while everything above is free -- the `boundary` ring anchors every extreme
            # of the extent at once, hands included (exp01 minisite, the armadillo pulled apart, 2026-10-27).
            ax, lo, hi = int(self.band[0]), float(self.band[1]), float(self.band[2])
            c = self._rest[:, ax]
            self._sel = (c >= lo) & (c <= hi)
            print(f"[mpm_anchor] band axis {ax} in [{lo:g}, {hi:g}]: {int(self._sel.sum()):,} of {self._sel.numel():,} "
                  f"points anchored", flush=True)
        else:                                                 # outer ring of the tissue's rest extent
            lo = self._rest.min(0).values
            hi = self._rest.max(0).values
            near = ((self._rest - lo) < self._ring) | ((hi - self._rest) < self._ring)  # [N,2]
            self._sel = near.any(dim=1)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        if self._rest is None:
            self._init(lvl, H)
        acc = self.k * (self._rest - lvl.get("pos")) * (self._sel * lvl.occ)[:, None].float()
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}


@register_field("image", frame="image")
class ImageField(Field):
    """A 1-channel scalar map read from a 2D image (TIFF/PNG), normalised to [0, 1].

    Pure state and no dynamics: it holds its grid `[1, nx, ny]` and the world-to-pixel geometry,
    and `apply_material_map` samples it. Normalising to [0, 1] is what makes the image carry the
    PATTERN and the operator reading it carry the units. Same orientation convention as
    `PrescribedField` -- flipped vertically, so image-top maps to domain-top and a gradient read
    from it does not point the wrong way.

    Reference: none -- a measured map is data, not a model.
    """

    def __init__(self, name, source=None, res=None, width=1.0, device="cpu",
                 normalize=True, **kw):
        super().__init__(name)                                  # binds to no set (no couples_to)
        if source is None:
            raise ValueError(f"image field {name!r} needs a `source:` (path to a .tif/.png)")
        import tifffile
        path = source if os.path.isabs(source) else graphs_data_path(source)
        img = tifffile.imread(path).astype("float32")          # [ny, nx] (image rows top->bottom)
        if img.ndim == 3:                                      # collapse any channels to grayscale
            img = img.mean(axis=-1)
        img = img[::-1, :].copy()                              # flip vertical: image-top -> domain-top
        if normalize:
            lo, hi = float(img.min()), float(img.max())
            img = (img - lo) / (hi - lo + 1e-9)                # -> [0,1]
        v = torch.tensor(img, device=device).permute(1, 0).contiguous()   # [ny,nx] -> [nx,ny]
        self.C = 1
        self.nx, self.ny = int(v.shape[0]), int(v.shape[1])
        self.width = float(width)
        self.R = self.nx / self.width                          # pixels per world unit (x)
        self.register_buffer("grid", v[None])                  # [1, nx, ny]

    def pix(self, x, y):
        gx = (x.clamp(0, self.width - 1e-6) / self.width * self.nx).long().clamp(0, self.nx - 1)
        gy = (y.clamp(0, 1 - 1e-6) * self.ny).long().clamp(0, self.ny - 1)
        return gx, gy


@register_field("vector_grid", frame="vector_grid")
class VectorGrid(Field):
    """A 2-channel UNIT-VECTOR map d(x) = (dx, dy) read from a TIFF: the contraction direction,
    or the active-stress orientation.

    A 2-channel TIFF `[ny, nx, 2]` is read as (dx, dy); a 1-channel one as an angle, theta in
    [0, 1] rescaled to [0, 2pi) and converted to (cos theta, sin theta). Every vector is
    normalised to unit length, so the map carries a DIRECTION only and the operator reading it
    carries the magnitude. Same vertical flip as `ImageField`.

    Reference: none -- a measured orientation map is data, not a model.
    """

    def __init__(self, name, source=None, res=None, width=1.0, device="cpu", **kw):
        super().__init__(name)
        if source is None:
            raise ValueError(f"vector_grid field {name!r} needs a `source:` (path to a .tif)")
        import tifffile
        import numpy as np
        path = source if os.path.isabs(source) else graphs_data_path(source)
        img = tifffile.imread(path).astype("float32")
        img = img[::-1].copy()                                 # flip vertical (image-top -> domain-top)
        if img.ndim == 2:                                      # angle map theta in [0,1] -> [0,2pi)
            th = img * (2 * np.pi)
            dx, dy = np.cos(th), np.sin(th)
        else:                                                  # [ny,nx,2] vector map (dx, dy), [-1,1]
            dx, dy = img[..., 0], img[..., 1]
        v = np.stack([dx, dy], axis=0)                         # [2, ny, nx]
        n = np.sqrt(v[0] ** 2 + v[1] ** 2); n[n < 1e-9] = 1.0
        v = (v / n).astype("float32")                          # unit vectors
        vt = torch.tensor(v, device=device).permute(0, 2, 1).contiguous()   # [2, nx, ny]
        self.C = 2
        self.nx, self.ny = int(vt.shape[1]), int(vt.shape[2])
        self.width = float(width)
        self.R = self.nx / self.width
        self.register_buffer("grid", vt)                       # [2, nx, ny]


_OFFSETS = torch.tensor([[i, j] for i in range(3) for j in range(3)], dtype=torch.float32)


# --------------------------------------------------------------------------- #
#  Backend kernel: one MLS-MPM substep (pure -> compilable). NOT a Plexus
#  primitive -- the fenced operator below is the only thing that calls it.
# --------------------------------------------------------------------------- #
def mls_mpm_substep(X, V, C, F, mass, mu, la, a_ext, offsets,
                    nx, ny, dx, inv_dx, dt, p_vol, drag, walls_flat, vmax_user, periodic, width,
                    wall_damp, wall_contact, liquid_mask, snow_mask, Jp, surf):
    """One MLS-MPM substep. All tensors batched over particles. Pure -> compilable.
    Grid is [nx, ny] of square cells (dx); the world is [0,width]x[0,1]."""
    N = X.shape[0]
    eye = torch.eye(2, device=X.device).expand(N, 2, 2)

    # external cell-level accel + Stokes drag (overdamped tissue): V saturates at a/drag
    V = V + dt * (a_ext - drag * V)

    # deformation gradient update
    F = (eye + dt * C) @ F
    a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
    J = a * d - b * c                                   # det(F)

    if liquid_mask is not None:                         # LIQUID: drop shape memory, keep only volume J.
        Jl = torch.sqrt(J.clamp(min=1e-6))              # F := sqrt(J)*I  -> isotropic, no shear/rotation.
        F = torch.where(liquid_mask[:, None, None], eye * Jl[:, None, None], F)
        a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]   # (mu=0 for liquid -> stress is pure pressure)

    mu_e, la_e = mu, la
    if snow_mask is not None:                           # SNOW: plastic flow -> clamp singular values of F,
        sm = snow_mask                                  # accumulate plastic volume Jp, harden mu/la with Jp.
        Fs = F[sm]
        if Fs.shape[0] > 0:
            U, sig, Vh = torch.linalg.svd(Fs)           # F = U diag(sig) Vh   (per snow particle)
            sig_c = sig.clamp(1.0 - 2.5e-2, 1.0 + 7.5e-3)   # snow yield: theta_c compress, theta_s stretch
            Fp = U @ torch.diag_embed(sig_c) @ Vh
            F = F.clone(); F[sm] = Fp
            ratio = sig.prod(-1) / sig_c.prod(-1).clamp(min=1e-6)   # volume pushed into plastic part
            Jp = Jp.clone(); Jp[sm] = (Jp[sm] * ratio).clamp(0.6, 20.0)
            a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
            J = a * d - b * c
        h = torch.exp((10.0 * (1.0 - Jp)).clamp(-6.0, 6.0))   # Jp<1 (packed) -> harder; Jp>1 -> softer
        mu_e = torch.where(sm, mu * h, mu)
        la_e = torch.where(sm, la * h, la)

    # analytic 2x2 polar rotation R (closest rotation to F)
    cs, sn = (a + d), (c - b)
    r = torch.sqrt(cs * cs + sn * sn) + 1e-9
    cs, sn = cs / r, sn / r
    R = torch.stack([torch.stack([cs, -sn], -1),
                     torch.stack([sn, cs], -1)], -2)    # [N,2,2]

    # fixed-corotated stress -> affine momentum matrix  (mu_e/la_e carry snow hardening)
    FmR = F - R
    stress = 2 * mu_e[:, None, None] * (FmR @ F.transpose(-2, -1)) \
        + eye * (la_e * J * (J - 1))[:, None, None]
    pv = p_vol[:, None, None] if torch.is_tensor(p_vol) else p_vol
    stress = (-dt * 4 * inv_dx * inv_dx) * pv * stress
    affine = stress + mass[:, None, None] * C

    # --- P2G ---
    base = (X * inv_dx - 0.5).floor().long()            # [N,2]
    fx = X * inv_dx - base.float()
    w = torch.stack([0.5 * (1.5 - fx) ** 2,
                     0.75 - (fx - 1) ** 2,
                     0.5 * (fx - 0.5) ** 2], dim=1)      # [N,3,2]
    oi = offsets[:, 0].long(); oj = offsets[:, 1].long()             # [9]
    weight = w[:, oi, 0] * w[:, oj, 1]                               # [N,9]
    gpos = base[:, None, :] + offsets.long()[None]                  # [N,9,2]
    if periodic:
        gpos = torch.stack([gpos[..., 0] % nx, gpos[..., 1] % ny], dim=-1)
    else:
        gpos = torch.stack([gpos[..., 0].clamp(0, nx - 1), gpos[..., 1].clamp(0, ny - 1)], dim=-1)
    dpos_phys = (offsets[None] - fx[:, None, :]) * dx               # [N,9,2]

    mom = mass[:, None, None] * V[:, None, :] \
        + (affine[:, None] @ dpos_phys[..., None]).squeeze(-1)      # [N,9,2]
    flat = (gpos[..., 0] * ny + gpos[..., 1]).reshape(-1)          # [N*9]  (row-major nx x ny)
    grid_m = torch.zeros(nx * ny, device=X.device)
    grid_mv = torch.zeros(nx * ny, 2, device=X.device)
    grid_m.index_add_(0, flat, (weight * mass[:, None]).reshape(-1))
    grid_mv.index_add_(0, flat, (weight[..., None] * mom).reshape(-1, 2))

    # grid velocity
    gv = grid_mv / grid_m.clamp(min=1e-10)[:, None]

    if surf > 0.0 and liquid_mask is not None:        # SURFACE TENSION as a CSF (continuum surface force):
        # 1) liquid colour field c on the grid (scatter liquid mass via the same P2G weights)
        lw = (weight * (mass * liquid_mask.to(mass.dtype))[:, None]).reshape(-1)
        c = torch.zeros(nx * ny, device=X.device).index_add_(0, flat, lw).view(nx, ny)
        # 2) normal n = grad(c)/|grad(c)|  (central differences in physical units)
        cx = (torch.roll(c, -1, 0) - torch.roll(c, 1, 0)) * (0.5 * inv_dx)
        cy = (torch.roll(c, -1, 1) - torch.roll(c, 1, 1)) * (0.5 * inv_dx)
        gmag = torch.sqrt(cx * cx + cy * cy)
        eps = 1e-6
        nxg, nyg = cx / (gmag + eps), cy / (gmag + eps)
        # 3) curvature kappa = -div(n)
        kappa = -((torch.roll(nxg, -1, 0) - torch.roll(nxg, 1, 0)) * (0.5 * inv_dx)
                  + (torch.roll(nyg, -1, 1) - torch.roll(nyg, 1, 1)) * (0.5 * inv_dx))
        # 4) surface force density f = surf * kappa * grad(c)  (acts only where |grad c|>0: the interface)
        fmask = (gmag > 0.02 * gmag.max()).to(c.dtype)
        stfx = (surf * kappa * cx * fmask).view(-1)        # surface-tension force, x (not the P2G fx!)
        stfy = (surf * kappa * cy * fmask).view(-1)
        # 5) apply as grid acceleration a = f * cell_area / grid_mass, carried to particles by G2P
        inv_m = (dx * dx) / grid_m.clamp(min=1e-8)
        gv = gv + dt * torch.stack([stfx * inv_m, stfy * inv_m], dim=1)

    if not periodic:                                  # reflective domain walls (toroidal otherwise)
        gv = gv.view(nx, ny, 2)
        ix = torch.arange(nx, device=X.device); iy = torch.arange(ny, device=X.device); bnd = 3
        lox, hix = ix < bnd, ix > nx - bnd
        loy, hiy = iy < bnd, iy > ny - bnd
        gv[lox, :, 0] = gv[lox, :, 0].clamp(min=0); gv[hix, :, 0] = gv[hix, :, 0].clamp(max=0)
        gv[:, loy, 1] = gv[:, loy, 1].clamp(min=0); gv[:, hiy, 1] = gv[:, hiy, 1].clamp(max=0)
        if wall_damp != 1.0:                          # tangential wall FRICTION (kills wall jets), but
            # at SIDE walls damp only UPWARD flow -> jets die yet gravity still drains stuck droplets down
            gl = gv[lox, :, 1]; gv[lox, :, 1] = torch.where(gl > 0, gl * wall_damp, gl)
            gh = gv[hix, :, 1]; gv[hix, :, 1] = torch.where(gh > 0, gh * wall_damp, gh)
            gv[:, loy, 0] = gv[:, loy, 0] * wall_damp   # floor/ceiling: horizontal tangential (symmetric ok)
            gv[:, hiy, 0] = gv[:, hiy, 0] * wall_damp
        gv = gv.view(nx * ny, 2)
    if wall_damp != 1.0 and walls_flat.any():     # friction in the fluid cells touching any INTERIOR
        w2 = walls_flat.view(nx, ny)              # obstacle wall (general: works for any obstacle shape)
        near = (torch.roll(w2, 1, 0) | torch.roll(w2, -1, 0)
                | torch.roll(w2, 1, 1) | torch.roll(w2, -1, 1)) & ~w2
        gvv = gv.view(nx, ny, 2); gx = gvv[..., 0]; gy = gvv[..., 1]
        gvv[..., 0] = torch.where(near, gx * wall_damp, gx)              # horizontal: full friction
        gvv[..., 1] = torch.where(near & (gy > 0), gy * wall_damp, gy)   # vertical: damp only upward -> gravity drains
        gv = gvv.view(nx * ny, 2)
    gv = torch.where(walls_flat[:, None], torch.zeros_like(gv), gv)   # interior wall BC

    # --- G2P ---
    gvn = gv[flat].view(N, 9, 2)                                    # [N,9,2]
    new_V = (weight[..., None] * gvn).sum(1)
    dpos_grid = offsets[None] - fx[:, None, :]                      # [N,9,2]
    new_C = 4 * inv_dx * (weight[..., None, None]
                          * (gvn[..., :, None] @ dpos_grid[..., None, :])).sum(1)
    # robustness: bound velocity (CFL) and sanitize NaN/inf so a bad design can't
    # poison the CUDA context -- it just produces a poor (low-food) trajectory.
    new_V = torch.nan_to_num(new_V)
    if wall_damp != 1.0 and not periodic:            # inelastic walls: bleed kinetic energy from the
        cb = wall_contact                            # SOLID layer in contact with a wall (bounce restitution)
        near = ((X[:, 0] < cb) | (X[:, 0] > width - cb)
                | (X[:, 1] < cb) | (X[:, 1] > 1.0 - cb))
        if liquid_mask is not None:                  # liquids are handled by the asymmetric grid wall
            near = near & ~liquid_mask               # friction -> don't pin them here (else they can't drain)
        new_V = torch.where(near[:, None], new_V * wall_damp, new_V)
    sp = new_V.norm(dim=1, keepdim=True).clamp(min=1e-9)
    vmax = min(vmax_user, 0.4 * dx / dt)             # user cap, never above CFL
    new_V = new_V * (sp.clamp(max=vmax) / sp)
    new_C = torch.nan_to_num(new_C)
    F = torch.nan_to_num(F, nan=1.0)
    X = torch.nan_to_num(X + dt * new_V, nan=0.5)
    if periodic:
        X = torch.stack([torch.remainder(X[:, 0], width),            # bc_pos: wrap onto the torus
                         torch.remainder(X[:, 1], 1.0)], dim=1)
    else:
        X = torch.stack([X[:, 0].clamp(2 * dx, width - 2 * dx),
                         X[:, 1].clamp(2 * dx, 1 - 2 * dx)], dim=1)
    return X, new_V, new_C, F, Jp


# --------------------------------------------------------------------------- #
#  The fenced transitional operator
# --------------------------------------------------------------------------- #
# ==========================================================================================================
# The four MPM operators in Warp: `mpm_scatter`, `mpm_gather`, `mpm_grid_update`, `mpm_strain`.
#
# THE DEFAULTS ARE REGISTERED ABOVE THIS LINE AND MUST STAY THERE, here and in every backend
# section below. A contract's `default` is whichever variant registers FIRST, so appending the
# backends after the torch bodies is not a style choice: a warp registration placed above
# `MPMScatter` makes `mpm_scatter[default]` silently become the warp kernel, and the contract's
# `.signature` is the default's too.
# ==========================================================================================================
try:
    import warp as wp
    wp.init()
    HAVE_WARP = True
except Exception:
    HAVE_WARP = False


def _wp_launch(kernel, dim, dev, inputs):
    """`wp.launch` ON PYTORCH'S CURRENT STREAM.

    THIS IS NOT A DETAIL. Warp launches on its OWN default stream for the device unless told
    otherwise, and PyTorch captures a CUDA graph on ITS current stream. Launched on a different
    stream, the warp kernels are simply NOT RECORDED into the graph: they run once, eagerly, while
    the capture is being taken, and never again on any replay. The symptom is a simulation that
    advances exactly one frame and then freezes, while the engine cheerfully reports the substep
    as captured. A gate that sets `capture: false` will never see it.
    """
    import torch as _t
    # `sync_enter=False`: ScopedStream's default is to make the new stream WAIT on the old one via
    # an event, and recording a cross-stream wait is itself illegal inside a capture -- it turned
    # the silent freeze into `cudaErrorStreamCaptureInvalidated`. There is nothing to synchronise
    # anyway: we are asking warp to launch on the very stream torch is already using.
    with wp.ScopedStream(wp.stream_from_torch(_t.cuda.current_stream(dev)),
                         sync_enter=False, sync_exit=False):
        wp.launch(kernel, dim=dim, device=f"cuda:{dev.index or 0}", inputs=inputs)


if HAVE_WARP:

    @wp.func
    def polar_R(F: wp.mat33, iters: int) -> wp.mat33:
        """Orthogonal polar factor by Newton: R <- (R + R^-T) / 2.

        The same iteration `mpm_ops._polar_higham` runs, and the same one the Triton kernel spells
        out as cofactor cross-products. Here `wp.inverse` and `wp.transpose` are builtins, so the
        loop body is the formula rather than a transcription of it.
        """
        R = F
        for _ in range(iters):
            d = wp.determinant(R)
            if wp.abs(d) < 1.0e-12:                 # a collapsed particle gets a finite, meaningless
                return R                            # rotation rather than taking the run down
            R = 0.5 * (R + wp.transpose(wp.inverse(R)))
        return R

    @wp.kernel
    def p2g_atomic(STATE: wp.array2d(dtype=float), pa: int, va: int,
                   C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
                   MASS: wp.array(dtype=float), MU: wp.array(dtype=float),
                   LA: wp.array(dtype=float), PVOL: wp.array(dtype=float),
                   AEXT: wp.array(dtype=wp.vec3), JP: wp.array(dtype=float),
                   SNW: wp.array(dtype=float), LIQ: wp.array(dtype=float),
                   OCC: wp.array(dtype=float), TURG: wp.array(dtype=float),
                   ACT: wp.array(dtype=wp.mat33),
                   GM: wp.array(dtype=float), GMV: wp.array(dtype=wp.vec3),
                   GC: wp.array(dtype=float),
                   ngx: int, ngy: int, ngz: int,
                   dx: float, dt: float, drag: float, iters: int,
                   has_snw: int, has_liq: int, has_turg: int, has_act: int):
        p = wp.tid()
        inv_dx = 1.0 / dx
        # DORMANT PARTICLES CONTRIBUTE NOTHING. The default masks the scatter weights by occupancy
        # (`weight * (occ > 0)`); this kernel had no `occ` at all, so a growth reservoir waiting to
        # be spawned was depositing mass and momentum into the grid the whole time it waited.
        if OCC[p] <= 0.0:
            return

        # READ STRAIGHT OUT OF `p.state`. `p.get("pos")` is a NON-CONTIGUOUS column slice, so
        # `.contiguous()` allocated a fresh temporary on every call -- see the note on
        # MPMScatterWarp.forward for why that was fatal under CUDA-graph capture. Indexing the
        # state here costs nothing and removes 2 x N x 3 x 4 B of copy traffic per substep.
        x = wp.vec3(STATE[p, pa + 0], STATE[p, pa + 1], STATE[p, pa + 2])
        v0 = wp.vec3(STATE[p, va + 0], STATE[p, va + 1], STATE[p, va + 2])
        v = v0 + dt * (AEXT[p] - drag * v0)         # body force + Stokes drag, as the torch op does
        mass = MASS[p]
        Fp = F[p]
        Cp = C[p]

        J = wp.determinant(Fp)
        R = polar_R(Fp, iters)
        # SNOW HARDENS AS IT PACKS, and this kernel did not know it. `mpm_strain` accumulates the
        # plastic volume ratio Jp, and the DEFAULT scatter scales both Lame parameters by
        # exp(10(1-Jp)) -- Jp<1 (packed) stiffens, Jp>1 softens (see `mpm_strain`). Omitting it left
        # snow with its virgin stiffness no matter how compacted it got, so a snow block compressed
        # without limit into a flat pancake instead of holding a packed shape.
        #
        # THE GATE COULD NOT SEE IT: over its 20 frames Jp barely leaves 1, so h ~ 1 and the two
        # implementations agreed to 2.1e-07 on the centre of mass. The divergence needs hundreds of
        # frames of sustained plastic flow to appear, which is the length a SCENE runs and not the
        # length a gate ran.
        mu_p = MU[p]
        la_p = LA[p]
        if has_snw == 1 and SNW[p] > 0.0:
            h = wp.exp(wp.clamp(10.0 * (1.0 - JP[p]), -6.0, 6.0))
            mu_p = mu_p * h
            la_p = la_p * h
        # fixed-corotated Kirchhoff stress: 2 mu (F - R) F^T + I la J (J - 1)
        S = 2.0 * mu_p * ((Fp - R) * wp.transpose(Fp)) + wp.identity(n=3, dtype=float) * (
            la_p * J * (J - 1.0))
        # THE TWO OPTIONAL STRESS TERMS the default adds and this kernel dropped. `active_stress`
        # is written by pulse_to_active_stress; `turgor` by mpm_turgor, and its sign is the one
        # that makes a cell a cell -- tau is Kirchhoff (positive = tension) and a fluid at pressure
        # P has sigma = -P.I, so a POSITIVE turgor SUBTRACTS and pushes the material outward.
        if has_act == 1:
            S = S + ACT[p]
        if has_turg == 1:
            S = S - wp.identity(n=3, dtype=float) * TURG[p]
        S = S * ((-dt * 4.0 * inv_dx * inv_dx) * PVOL[p])
        affine = S + Cp * mass

        base = wp.vec3(wp.floor(x[0] * inv_dx - 0.5),
                       wp.floor(x[1] * inv_dx - 0.5),
                       wp.floor(x[2] * inv_dx - 0.5))
        fx = wp.vec3(x[0] * inv_dx - base[0], x[1] * inv_dx - base[1], x[2] * inv_dx - base[2])

        for i in range(3):
            wi = float(0.0)
            if i == 0:
                wi = 0.5 * (1.5 - fx[0]) * (1.5 - fx[0])
            elif i == 1:
                wi = 0.75 - (fx[0] - 1.0) * (fx[0] - 1.0)
            else:
                wi = 0.5 * (fx[0] - 0.5) * (fx[0] - 0.5)
            for j in range(3):
                wj = float(0.0)
                if j == 0:
                    wj = 0.5 * (1.5 - fx[1]) * (1.5 - fx[1])
                elif j == 1:
                    wj = 0.75 - (fx[1] - 1.0) * (fx[1] - 1.0)
                else:
                    wj = 0.5 * (fx[1] - 0.5) * (fx[1] - 0.5)
                for k in range(3):
                    wk = float(0.0)
                    if k == 0:
                        wk = 0.5 * (1.5 - fx[2]) * (1.5 - fx[2])
                    elif k == 1:
                        wk = 0.75 - (fx[2] - 1.0) * (fx[2] - 1.0)
                    else:
                        wk = 0.5 * (fx[2] - 0.5) * (fx[2] - 0.5)
                    w = wi * wj * wk
                    # THREE COUNTS, AS THE GATHER ALREADY TAKES. This kernel clamped and flattened
                    # with ONE `ng` while `g2p` used ngx/ngy/ngz, so on any non-cubic box the
                    # scatter wrote to different nodes than the gather read -- silently, since a
                    # cubic box makes the two identical and every MPM spec until now was cubic.
                    # `MPMGrid` has derived per-axis counts since the world-box change (P0); this is
                    # the half of the pair that never caught up.
                    gi = wp.clamp(int(base[0]) + i, 0, ngx - 1)
                    gj = wp.clamp(int(base[1]) + j, 0, ngy - 1)
                    gk = wp.clamp(int(base[2]) + k, 0, ngz - 1)
                    idx = (gi * ngy + gj) * ngz + gk
                    dpos = wp.vec3((float(i) - fx[0]) * dx,
                                   (float(j) - fx[1]) * dx,
                                   (float(k) - fx[2]) * dx)
                    mom = mass * v + affine * dpos
                    wp.atomic_add(GM, idx, w * mass)
                    wp.atomic_add(GMV, idx, w * mom)
                    # LIQUID COLOUR, the field the CSF surface tension is computed from. Without
                    # it `mpm_grid_update` finds gc all zero, its `_c_csf` predicate is False, and
                    # the ENTIRE surface-tension branch is skipped -- so `cell_surface_tension: 60.0`
                    # in a spec did exactly nothing on any run using this implementation.
                    if has_liq == 1:
                        wp.atomic_add(GC, idx, w * mass * LIQ[p])


@register_operator("mpm_scatter", implementation="warp", family="mpm",
                   set="particle", kind="exchange", title="spread point mass, one atomic kernel", equation=r"""$$\boldsymbol\sigma = 2\mu(\mathbf F-\mathbf R)\mathbf F^{\!\top} + \lambda(J-1)J\,\mathbf I,
\qquad
\mathbf Q = -\frac{4\,\Delta t}{\Delta x^{2}}\,V\,\boldsymbol\sigma + m\,\mathbf C$$""")
class MPMScatterWarp(MPMScatter):
    r"""mpm_scatter as one Warp kernel with global atomics: the same physics, one launch instead of a batched gather-scatter over the stencil (CUDA only; float atomics add in thread order, so not bit-reproducible).
    particle -> mpm_grid: as mpm_scatter.

        \boldsymbol\sigma = 2\mu(\mathbf F-\mathbf R)\mathbf F^{\!\top} + \lambda(J-1)J\,\mathbf I, \qquad \mathbf Q = -\frac{4\,\Delta t}{\Delta x^{2}}\,V\,\boldsymbol\sigma + m\,\mathbf C
        \sigma         the Kirchhoff stress of the fixed-corotated law
        \mu, \lambda   the particle's Lame parameters, stress
        F              the deformation gradient
        R              its rotation (polar decomposition F = R S; `polar`, `polar_iters`)
        J              det F, the volume ratio
        I              the identity
        Q              the affine momentum matrix deposited with the particle: the stress impulse plus the APIC term
        \Delta t       `dt_sub`, the substep (stable below dx / sqrt(E/rho))
        \Delta x       the grid spacing
        V              the particle's rest volume
        m              its mass
        C              its affine velocity gradient (APIC)

    Reference: as mpm_scatter; NVIDIA Warp (Macklin, M. (2022))."""
    PARAM_UNITS = {'a_max': 'acceleration', 'drag': 'rate', 'dt_sub': 'time', 'polar_iters': 'count', 'body_force': 'acceleration'}
    SYMBOLS = {'\\sigma': 'the Kirchhoff stress of the fixed-corotated law', '\\mu, \\lambda': "the particle's Lame parameters, stress", 'F': 'the deformation gradient', 'R': 'its rotation (polar decomposition F = R S; `polar`, `polar_iters`)', 'J': 'det F, the volume ratio', 'I': 'the identity', 'Q': 'the affine momentum matrix deposited with the particle: the stress impulse plus the APIC term', '\\Delta t': '`dt_sub`, the substep (stable below dx / sqrt(E/rho))', '\\Delta x': 'the grid spacing', 'V': "the particle's rest volume", 'm': 'its mass', 'C': 'its affine velocity gradient (APIC)'}

    MECHANISM_TAGS = ["particle_to_grid", "fixed_corotated_stress", "shared_grid_accumulate",
                      "fused_kernel"]
    # 3D ONLY, DECLARED. Inherited unchanged this would say [2, 3], so `contract.capabilities()`
    # would report the fused kernel as able to run 2D -- it cannot, `forward` raises -- and any
    # capability-driven dispatch built on that table would route a 2D specification into a kernel
    # that refuses it. Most specifications are 2D, `general.dim` defaulting to 2.
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False

    def forward(self, H, mask=None):
        if not HAVE_WARP:
            raise RuntimeError("mpm_scatter[warp] needs warp-lang; none importable")
        # `store_stress` IS A TORCH-BODY FEATURE AND SAYING SO IS THE WHOLE POINT. The kernel below
        # accumulates momentum straight into the grid with atomics and never materialises the
        # per-particle Cauchy stress the default body caches to `sigma`. Asking for it here used to
        # do nothing at all, and the failure surfaced two operators away: `ecm_stress[measure:
        # vonmises]` found no buffer, said so, and fell back to |J-1| -- a DIFFERENT quantity, drawn
        # in the same colours, on a run whose subject is stress. A silently ignored declaration is
        # worse than an unsupported one.
        if getattr(self, "store_stress", False):
            raise ValueError(
                "mpm_scatter[warp]: `store_stress: true` is not implemented on the warp body, "
                "which never forms the per-particle stress it would cache. Drop "
                "`implementation: warp` on this operator to use the default body, which does, or "
                "drop `store_stress` and accept that anything reading `sigma` (such as "
                "`ecm_stress[measure: vonmises]`) will fall back to a different quantity.")
        from plexus.operators.mpm_ops import sub_dt
        p = H.level(self.at); g = H.field(self.grid); dev = p.state.device
        _no_origin(self, g)
        D = p.F.shape[-1]
        if D != 3 or str(dev) == "cpu":
            raise RuntimeError(f"mpm_scatter[warp] is 3D CUDA only (got dim={D}, dev={dev})")
        dt = sub_dt(H, self.dt_sub)

        # NOT `pa`/`va`: `pa` is rebound to H.part_accel eleven lines down, which silently turned
        # the pos column offset into a tensor.
        p_off, _ = p.state_schema["pos"]; v_off, _ = p.state_schema["vel"]
        from plexus.operators.mpm_ops import _ancestor_accel
        a_ext = _ancestor_accel(H, p, self.a_max, D, dev)
        pa = getattr(H, "part_accel", None)
        if pa is not None:
            a_ext = a_ext + pa
        from plexus.operators.mpm_ops import _hand_body_force_to_grid
        a_ext = a_ext + torch.nan_to_num(H.delta(p.name))
        # SAME SPLIT AS THE TORCH SCATTER. This is the path the material specs actually run
        # (`implementation: warp` on mpm_scatter), so leaving it out would mean `body_force: grid`
        # silently did nothing on every spec that asked for it.
        a_ext = _hand_body_force_to_grid(self, H, a_ext, dev, D).contiguous()

        # run-constant, and it must be cached: `bool(t.any())` is a sync and a sync inside a
        # CUDA-graph capture is illegal.
        from plexus.operators.mpm_ops import _const_any
        _has_snw = _const_any(self, "_c_snow", getattr(p, "is_snow", None))
        _has_liq = _const_any(self, "_c_liquid", getattr(p, "is_liquid", None))
        if getattr(self, "_sbuf", None) is None:
            z = torch.zeros(p.n, device=dev)
            def _mf(t):
                return t.float().contiguous() if t is not None else z
            self._sbuf = (_mf(getattr(p, "is_snow", None)), _mf(getattr(p, "is_liquid", None)),
                          torch.ones(p.n, device=dev) if getattr(p, "occ", None) is None else None,
                          z)
        _snw, _liq, _occ1, _z = self._sbuf
        _occ = _occ1 if _occ1 is not None else p.occ.contiguous()
        # OPTIONAL SIDE CHANNELS, re-read every call because they are written by OTHER operators
        # within the same tick (mpm_turgor, pulse_to_active_stress) and are not run-constant.
        _turg = getattr(p, "turgor", None)
        _has_turg = _turg is not None
        _turg = _turg.contiguous() if _has_turg else _z
        _act = getattr(H, "active_stress", None)
        # SAME GUARD AS `extra_stress` BELOW: the global slot belongs to whichever set wrote it
        # last, so it is taken only when its shape is this set's.
        if _act is not None and _act.shape[0] != p.n:
            _act = None
        _lact = getattr(p, "act_stress", None)          # per-set active stress; see the torch path
        if _lact is not None:
            _act = _lact if _act is None else (_act + _lact)
        # THE SAME ADDITIVE-STRESS SLOT CARRIES BOTH. The kernel takes one mat33 per particle, so
        # active stress and the viscous stress from `mpm_viscosity` are summed here rather than
        # given separate inputs -- they enter the momentum identically and the kernel cannot tell
        # them apart. Summing in torch also keeps the kernel signature (and its cached compile)
        # unchanged, which is why `mpm_viscosity` needs no warp kernel of its own.
        _xtr = getattr(p, "extra_stress", None)
        if _xtr is None:
            _gx = getattr(H, "extra_stress", None)
            _xtr = _gx if (_gx is not None and _gx.shape[0] == p.n) else None
        if _xtr is not None:
            _act = _xtr if _act is None else (_act + _xtr)
        _has_act = _act is not None
        _act = _act.contiguous() if _has_act else torch.zeros(p.n, 3, 3, device=dev)

        gm, gmv = g.m, g.mv
        if getattr(self, "_zeroes_grid", True):
            gm.zero_(); gmv.zero_(); g.c.zero_()

        # ZERO-COPY. `wp.from_torch` wraps the same device memory, so the grid the kernel writes IS
        # the field the rest of the substep reads -- no staging, and the in-place discipline the
        # capture guard depends on is preserved.
        wdev = f"cuda:{dev.index or 0}"
        n = int(p.n)
        _wp_launch(
            p2g_atomic, n, dev,
            [wp.from_torch(p.state), int(p_off), int(v_off),
                    wp.from_torch(p.C.contiguous(), dtype=wp.mat33),
                    wp.from_torch(p.F.contiguous(), dtype=wp.mat33),
                    wp.from_torch(p.mass.contiguous()), wp.from_torch(p.mu.contiguous()),
                    wp.from_torch(p.la.contiguous()), wp.from_torch(p.p_vol.contiguous()),
                    wp.from_torch(a_ext, dtype=wp.vec3),
                    wp.from_torch(p.Jp.contiguous()), wp.from_torch(_snw),
                    wp.from_torch(_liq), wp.from_torch(_occ), wp.from_torch(_turg),
                    wp.from_torch(_act, dtype=wp.mat33),
                    wp.from_torch(gm), wp.from_torch(gmv.view(-1, 3), dtype=wp.vec3),
                    wp.from_torch(g.c),
                    int(g.nx), int(g.ny), int(g.nz),
                    float(g.dx), float(dt), float(self.drag), int(self.polar_iters),
                    int(_has_snw), int(_has_liq), int(_has_turg), int(_has_act)])
        return {}


# ==========================================================================================================
# G2P -- `mpm_gather[implementation: warp]`
#
# THE SCATTER WAS 64% OF THE FRAME AND IS NOW ~21%; PROFILED AFTER THAT, THE GATHER IS 60.5%.
# It is also by far the easier kernel: grid -> particle is a pure READ. Each particle reads the
# velocity of its 27 neighbouring nodes and forms two weighted sums (the new velocity, and the
# affine matrix C). Nothing is shared, nothing collides, there are no atomics and no sort. The
# PyTorch version is slow for one reason only: it materialises [N, 27, 3] and [N, 27, 3, 3]
# intermediates through global memory, and never needs to.
# ==========================================================================================================
if HAVE_WARP:

    @wp.kernel
    def g2p(STATE: wp.array2d(dtype=float),
            C: wp.array(dtype=wp.mat33), GV: wp.array(dtype=wp.vec3),
            OCC: wp.array(dtype=float), LIQ: wp.array(dtype=float),
            NEAR: wp.array(dtype=float),
            pa: int, va: int, ngx: int, ngy: int, ngz: int, dx: float, dt: float,
            wall_damp: float, wall_contact: float, vmax: float,
            bx: float, by: float, bz: float, has_liq: int, per_impact: int):
        p = wp.tid()
        inv_dx = 1.0 / dx
        x = wp.vec3(STATE[p, pa + 0], STATE[p, pa + 1], STATE[p, pa + 2])   # no copy; see p2g
        base = wp.vec3(wp.floor(x[0] * inv_dx - 0.5),
                       wp.floor(x[1] * inv_dx - 0.5),
                       wp.floor(x[2] * inv_dx - 0.5))
        fx = wp.vec3(x[0] * inv_dx - base[0], x[1] * inv_dx - base[1], x[2] * inv_dx - base[2])

        newv = wp.vec3(0.0, 0.0, 0.0)
        newC = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for i in range(3):
            wi = float(0.0)
            if i == 0:
                wi = 0.5 * (1.5 - fx[0]) * (1.5 - fx[0])
            elif i == 1:
                wi = 0.75 - (fx[0] - 1.0) * (fx[0] - 1.0)
            else:
                wi = 0.5 * (fx[0] - 0.5) * (fx[0] - 0.5)
            for j in range(3):
                wj = float(0.0)
                if j == 0:
                    wj = 0.5 * (1.5 - fx[1]) * (1.5 - fx[1])
                elif j == 1:
                    wj = 0.75 - (fx[1] - 1.0) * (fx[1] - 1.0)
                else:
                    wj = 0.5 * (fx[1] - 0.5) * (fx[1] - 0.5)
                for k in range(3):
                    wk = float(0.0)
                    if k == 0:
                        wk = 0.5 * (1.5 - fx[2]) * (1.5 - fx[2])
                    elif k == 1:
                        wk = 0.75 - (fx[2] - 1.0) * (fx[2] - 1.0)
                    else:
                        wk = 0.5 * (fx[2] - 0.5) * (fx[2] - 0.5)
                    w = wi * wj * wk
                    # row-major over `g.shape`, EXACTLY as `bspline` flattens it: axis 0 spans the
                    # world width and carries `nx`, axes 1-2 span [0,1] and carry `ny`.
                    gi = wp.clamp(int(base[0]) + i, 0, ngx - 1)
                    gj = wp.clamp(int(base[1]) + j, 0, ngy - 1)
                    gk = wp.clamp(int(base[2]) + k, 0, ngz - 1)
                    gv = GV[(gi * ngy + gj) * ngz + gk]
                    dpos = wp.vec3(float(i) - fx[0], float(j) - fx[1], float(k) - fx[2])
                    newv = newv + w * gv
                    newC = newC + (4.0 * inv_dx * w) * wp.outer(gv, dpos)

        # inelastic wall contact for SOLIDS: a liquid is handled by the asymmetric grid wall
        # friction instead, so pinning it here would stop it draining.
        if wall_damp != 1.0:
            near = (x[0] < wall_contact or x[0] > bx - wall_contact or
                    x[1] < wall_contact or x[1] > by - wall_contact or
                    x[2] < wall_contact or x[2] > bz - wall_contact)
            if has_liq == 1 and LIQ[p] > 0.0:
                near = False
            hit = near
            if per_impact == 1:
                # RISING EDGE ONLY -- see the long note in mpm_ops.MPMGather.forward. One impact
                # must cost one multiplication, not one per substep spent in the contact layer,
                # or the coefficient's meaning moves with the grid resolution.
                hit = near and (NEAR[p] < 0.5)
                if near:
                    NEAR[p] = 1.0
                else:
                    NEAR[p] = 0.0
            if hit:
                newv = newv * wall_damp

        sp = wp.length(newv)
        if sp > vmax:
            newv = newv * (vmax / sp)
        xn = x + dt * newv
        xn = wp.vec3(wp.clamp(xn[0], 2.0 * dx, bx - 2.0 * dx),
                     wp.clamp(xn[1], 2.0 * dx, by - 2.0 * dx),
                     wp.clamp(xn[2], 2.0 * dx, bz - 2.0 * dx))

        if OCC[p] <= 0.0:                      # DORMANT particles are frozen, not advected
            return
        STATE[p, pa + 0] = xn[0]; STATE[p, pa + 1] = xn[1]; STATE[p, pa + 2] = xn[2]
        STATE[p, va + 0] = newv[0]; STATE[p, va + 1] = newv[1]; STATE[p, va + 2] = newv[2]
        C[p] = newC


@register_operator("mpm_gather", implementation="warp", family="mpm",
                   set="particle", kind="exchange", title="move points with the grid, one kernel", equation=r"""$$\mathbf v_i = \sum_g w_{ig}\,\mathbf v_g,
\qquad
\mathbf C_i = \frac{4}{\Delta x^{2}}\sum_g w_{ig}\,\mathbf v_g\,(\mathbf x_g-\mathbf x_i)^{\!\top},
\qquad
\mathbf x_i \mathrel{+}= \Delta t\,\mathbf v_i$$""")
class MPMGatherWarp(MPMGather):
    r"""mpm_gather as one Warp kernel: pure reads, no atomics (CUDA only).
    mpm_grid -> particle: as mpm_gather.

        \mathbf v_i = \sum_g w_{ig}\,\mathbf v_g, \qquad \mathbf C_i = \frac{4}{\Delta x^{2}}\sum_g w_{ig}\,\mathbf v_g\,(\mathbf x_g-\mathbf x_i)^{\!\top}, \qquad \mathbf x_i \mathrel{+}= \Delta t\,\mathbf v_i
        v_i            the velocity of particle i
        w_{ig}         the quadratic B-spline weight of node g at particle i
        v_g            the velocity of node g
        C_i            the APIC affine velocity matrix (local velocity gradient) carried to the next scatter and to mpm_strain
        \Delta x       the grid spacing
        x_g, x_i       the node and particle positions
        \Delta t       `dt_sub`, the substep

    Reference: as mpm_gather; NVIDIA Warp."""
    PARAM_UNITS = {'dt_sub': 'time', 'vmax': 'velocity', 'wall_contact': 'length', 'wall_contact_cells': 'count', 'wall_damp': 'fraction'}
    SYMBOLS = {'v_i': 'the velocity of particle i', 'w_{ig}': 'the quadratic B-spline weight of node g at particle i', 'v_g': 'the velocity of node g', 'C_i': 'the APIC affine velocity matrix (local velocity gradient) carried to the next scatter and to mpm_strain', '\\Delta x': 'the grid spacing', 'x_g, x_i': 'the node and particle positions', '\\Delta t': '`dt_sub`, the substep'}

    MECHANISM_TAGS = ["grid_to_particle", "advection", "fused_kernel"]
    SUPPORTED_DIMS = [3]                       # see MPMScatterWarp: inherited [2, 3] was a lie
    DIFFERENTIABLE = False

    def forward(self, H, mask=None):
        if not HAVE_WARP:
            raise RuntimeError("mpm_gather[warp] needs warp-lang")
        from plexus.operators.mpm_ops import sub_dt
        p = H.level(self.at); g = H.field(self.frm); dev = p.state.device
        _no_origin(self, g)
        D = p.F.shape[-1]
        if D != 3 or str(dev) == "cpu" or bool(getattr(H, "periodic", False)):
            raise RuntimeError("mpm_gather[warp] is 3D, non-periodic, CUDA only")
        dt = sub_dt(H, self.dt_sub)
        if getattr(self, "_box", None) is None:
            self._box = [float(b) for b in
                         getattr(H, "world_size", torch.tensor([g.width, 1.0]))][:D]
        bx, by, bz = self._box
        pa, _pb = p.state_schema["pos"]; va, _vb = p.state_schema["vel"]
        occ = getattr(p, "occ", None)
        if occ is None:
            occ = torch.ones(p.n, device=dev)
        liq = getattr(p, "is_liquid", None)
        has_liq = 1 if liq is not None else 0
        liqf = (liq.float().contiguous() if liq is not None
                else torch.zeros(p.n, device=dev))
        vmax = min(self.vmax, 0.4 * float(g.dx) / float(dt))
        # PERSISTENT contact-edge state for `wall_damp_mode: per_impact`. float, not bool, because
        # warp arrays of bool are awkward to alias from torch; written in place so a captured graph
        # keeps the address. Allocated once, on the first call, which is before capture installs.
        if getattr(self, "_near", None) is None:
            self._near = torch.zeros(p.n, device=dev)
        _near = self._near
        wdev = f"cuda:{dev.index or 0}"
        _wp_launch(g2p, int(p.n), dev,
                  [wp.from_torch(p.state),
                          wp.from_torch(p.C, dtype=wp.mat33),
                          wp.from_torch(g.v.view(-1, 3), dtype=wp.vec3),
                          wp.from_torch(occ), wp.from_torch(liqf), wp.from_torch(_near),
                          int(pa), int(va), int(g.shape[0]), int(g.shape[1]), int(g.shape[2]),
                          float(g.dx), float(dt),
                          float(self.wall_damp), float(self._contact_band(g)), float(vmax),
                          float(bx), float(by), float(bz), int(has_liq),
                          int(self.wall_damp_mode == "per_impact")])
        return {}


# ==========================================================================================================
# F UPDATE -- `mpm_strain[implementation: warp]`
#
# With the scatter and the gather fused, this and `mpm_grid_update` are the whole remaining frame.
# It is the easier of the two and the larger lever at scale: `mpm_strain` is O(particles) while
# `mpm_grid_update` is O(cells), so the grid solve's cost is FLAT as the particle count grows and
# this one's is not.
#
# ALL FOUR BRANCHES: elastic, liquid, viscoelastic, snow. The last two need a 3x3 SVD and were
# refused at first, on the grounds that torch returns singular values DESCENDING while `wp.svd3`
# made no such promise -- and the snow branch's proper-rotation fix indexes the LAST singular value
# specifically, so the order changes the answer. MEASURED over 20,000 deformation-gradient-like
# matrices rather than assumed:
#
#   sigma descending          100.0% of cases          -- matches torch
#   det(U), det(V)            +1.000 ALWAYS            -- torch gives +-1
#   |sigma| vs torch          1.9e-06 near identity, 1.0e-04 on pathological input
#
# The second row is the surprise and it makes the port SIMPLER than the default, not riskier:
# `wp.svd3` already returns proper rotations with a SIGNED sigma, which is exactly the state the
# default reaches by hand through `negU`/`negV`. Reconstruction error is larger than torch's
# (6.2e-03 worst case against 4.9e-06) but that lives in U and V -- a valid alternative
# factorisation where sigma are near-degenerate -- and the material branches use only sigma, which
# agrees to 1.9e-06 in the regime snow occupies (|F - I| < 0.05, since snow clamps sigma into a
# window 0.0325 wide).
#
# TWO KERNELS, NOT ONE WITH FLAGS. `strain_elastic` is the common path and carrying the SVD in it
# would cost register pressure on every spec to serve the two that need it.
# ==========================================================================================================
if HAVE_WARP:

    @wp.kernel
    def strain_elastic(C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
                       LIQ: wp.array(dtype=float), OCC: wp.array(dtype=float),
                       dt: float, has_liq: int):
        p = wp.tid()
        # DORMANT PARTICLES DO NOT DEFORM. The default writes `where(live, F_new, F_old)`; leaving
        # early leaves F[p] untouched, which is the same thing and skips the work.
        if OCC[p] <= 0.0:
            return
        Fp = (wp.identity(n=3, dtype=float) + dt * C[p]) * F[p]
        if has_liq == 1 and LIQ[p] > 0.0:
            # LIQUID: drop shape memory, keep volume. J is taken from the UPDATED F, as the default
            # does -- computing it before the (I + dt C) step would reset to last substep's volume.
            J = wp.determinant(Fp)
            Jl = wp.pow(wp.max(J, 1.0e-6), 1.0 / 3.0)
            Fp = wp.identity(n=3, dtype=float) * Jl
        F[p] = Fp


    @wp.kernel
    def strain_full(C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
                    LIQ: wp.array(dtype=float), VIS: wp.array(dtype=float),
                    SNW: wp.array(dtype=float), TAU: wp.array(dtype=float),
                    JP: wp.array(dtype=float), OCC: wp.array(dtype=float),
                    dt: float, has_liq: int, has_vis: int, has_snw: int):
        p = wp.tid()
        if OCC[p] <= 0.0:
            return
        I3 = wp.identity(n=3, dtype=float)
        Fp = (I3 + dt * C[p]) * F[p]

        if has_liq == 1 and LIQ[p] > 0.0:                     # LIQUID: drop shape memory
            Jl = wp.pow(wp.max(wp.determinant(Fp), 1.0e-6), 1.0 / 3.0)
            Fp = I3 * Jl

        if has_vis == 1 and VIS[p] > 0.0:                     # VISCOELASTIC: partial shape reset
            U = wp.mat33(); sg = wp.vec3(); V = wp.mat33()
            wp.svd3(Fp, U, sg, V)
            Jl = wp.pow(wp.max(sg[0] * sg[1] * sg[2], 1.0e-6), 1.0 / 3.0)
            a = wp.exp(-dt / wp.max(TAU[p], 1.0e-6))          # memory kept: a->1 elastic, a->0 liquid
            sg = wp.vec3(Jl + (sg[0] - Jl) * a, Jl + (sg[1] - Jl) * a, Jl + (sg[2] - Jl) * a)
            Fp = (U * wp.diag(sg)) * wp.transpose(V)

        if has_snw == 1 and SNW[p] > 0.0:                     # SNOW: clamp stretches, harden via Jp
            U = wp.mat33(); sg = wp.vec3(); V = wp.mat33()
            # NO SIGN FIX NEEDED. The default computes one because torch can return det(U) = -1;
            # `wp.svd3` returns proper rotations already, with the sign carried in sigma -- the same
            # state, reached by the library instead of by hand.
            wp.svd3(Fp, U, sg, V)
            lo = 1.0 - 2.5e-2
            hi = 1.0 + 7.5e-3
            sc = wp.vec3(wp.clamp(sg[0], lo, hi), wp.clamp(sg[1], lo, hi), wp.clamp(sg[2], lo, hi))
            Fp = (U * wp.diag(sc)) * wp.transpose(V)
            ratio = (sg[0] * sg[1] * sg[2]) / wp.max(sc[0] * sc[1] * sc[2], 1.0e-6)
            JP[p] = wp.clamp(JP[p] * ratio, 0.6, 20.0)

        F[p] = Fp


@register_operator("mpm_strain", implementation="warp", family="mpm",
                   set="particle", kind="lateral", title="update material strain, one kernel", equation=r"""$$\mathbf F \;\leftarrow\; (\mathbf I + \Delta t\,\mathbf C)\,\mathbf F
\qquad\text{(liquid: } \mathbf F\leftarrow J^{1/d}\mathbf I,\ J=\det\mathbf F\text{)}$$""")
class MPMStrainWarp(MPMStrain):
    r"""mpm_strain as one Warp kernel, elastic and liquid branches both (CUDA only).
    particle -> particle: as mpm_strain.

        \mathbf F \;\leftarrow\; (\mathbf I + \Delta t\,\mathbf C)\,\mathbf F \qquad\text{(liquid: } \mathbf F\leftarrow J^{1/d}\mathbf I,\ J=\det\mathbf F\text{)}
        F              the deformation gradient, the identity for unstrained material
        I              the identity
        \Delta t       `dt_sub`, the substep
        C              the affine velocity gradient
        J              det F, the ratio of current to rest volume (`liquid_volume`: carried through F, or integrated as dJ/dt = J tr C)
        d              the dimension

    Reference: as mpm_strain; NVIDIA Warp."""
    PARAM_UNITS = {'dt_sub': 'time'}
    SYMBOLS = {'F': 'the deformation gradient, the identity for unstrained material', 'I': 'the identity', '\\Delta t': '`dt_sub`, the substep', 'C': 'the affine velocity gradient', 'J': 'det F, the ratio of current to rest volume (`liquid_volume`: carried through F, or integrated as dJ/dt = J tr C)', 'd': 'the dimension'}

    MECHANISM_TAGS = ["elastic_strain", "plastic_flow", "incompressible_volume", "fused_kernel"]
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    # Tells the engine's capture-refusal list that this implementation's snow/viscoelastic branches
    # are NOT the uncapturable cuSOLVER-plus-boolean-mask pair the default's are.
    CAPTURABLE_MATERIAL_BRANCHES = True

    def forward(self, H, mask=None):
        if not HAVE_WARP:
            raise RuntimeError("mpm_strain[warp] needs warp-lang")
        from plexus.operators.mpm_ops import sub_dt
        p = H.level(self.at); dev = p.state.device
        D = p.F.shape[-1]
        if D != 3 or str(dev) == "cpu":
            raise RuntimeError(f"mpm_strain[warp] is 3D CUDA only (got dim={D}, dev={dev})")
        # CACHED, and it MUST be. `bool(m.any())` is a device->host sync, and a sync inside a
        # CUDA-graph capture is illegal -- `cudaErrorStreamCaptureUnsupported`, which took down
        # every 3D spec the moment this operator was used, because `capture` defaults to True
        # (engine.py:1586). The predicate is run-constant: which particles are snow or
        # viscoelastic is fixed at seeding. `_const_any` is the codebase's existing answer to
        # exactly this and is what the default bodies use.
        from plexus.operators.mpm_ops import _const_any
        has_vis = _const_any(self, "_c_is_visco", getattr(p, "is_visco", None))
        has_snw = _const_any(self, "_c_is_snow", getattr(p, "is_snow", None))
        dt = sub_dt(H, self.dt_sub)
        liq = getattr(p, "is_liquid", None)
        has_liq = 1 if liq is not None else 0
        if getattr(self, "_side", None) is None:      # ALL run-constant; built once, not per substep
            z = torch.zeros(p.n, device=dev)
            def _f(t):
                return t.float().contiguous() if t is not None else z
            self._side = (_f(liq), _f(getattr(p, "is_visco", None)), _f(getattr(p, "is_snow", None)),
                          (p.visco_tau.contiguous() if getattr(p, "visco_tau", None) is not None
                           else torch.ones(p.n, device=dev)),
                          torch.ones(p.n, device=dev) if getattr(p, "occ", None) is None else None)
        liqf, visf, snwf, tau, occ1 = self._side
        occ = occ1 if occ1 is not None else p.occ.contiguous()
        wdev = f"cuda:{dev.index or 0}"
        if not (has_vis or has_snw):                  # the common path, no SVD in the kernel at all
            _wp_launch(strain_elastic, int(p.n), dev,
                      [wp.from_torch(p.C, dtype=wp.mat33), wp.from_torch(p.F, dtype=wp.mat33),
                              wp.from_torch(liqf), wp.from_torch(occ), float(dt), int(has_liq)])
        else:
            _wp_launch(strain_full, int(p.n), dev,
                      [wp.from_torch(p.C, dtype=wp.mat33), wp.from_torch(p.F, dtype=wp.mat33),
                              wp.from_torch(liqf), wp.from_torch(visf), wp.from_torch(snwf),
                              wp.from_torch(tau), wp.from_torch(p.Jp.contiguous()),
                              wp.from_torch(occ), float(dt), int(has_liq),
                              int(bool(has_vis)), int(bool(has_snw))])
        return {}


# ==========================================================================================================
# GRID SOLVE -- `mpm_grid_update[implementation: warp]`
#
# WHY. With the aggregate's deterministic index_add removed, `mpm_grid_update` is the LARGEST
# remaining item in the warp frame -- 42.7% of si_waterfall (24.2 ms of 56.6), against the scatter's
# 40.2% and the gather's 4.8%. It is also the one whose cost has nothing to do with the physics it
# computes. Two separate inefficiencies stack:
#
#   IT VISITS EVERY CELL, AND ALMOST NONE OF THEM HOLD ANYTHING. Grid nodes carrying mass are
#   3.3-4.0% of si_waterfall's 884,736 and 2.5-2.6% of the 5M scene's 7,077,888 -- so ~96% of the
#   work is on empty space. (A sparse grid is the fix for that one; this is not it.)
#
#   IT VISITS EACH CELL ~40 TIMES. The torch form is a chain of whole-grid tensor ops: twelve
#   `torch.roll`s per substep for `grad` and `kappa` alone, each one a full 3.5 MB copy in and
#   3.5 MB out, plus the elementwise chain and the boundary slabs. Measured ~450 MB of traffic per
#   call for a grid whose entire state is 14 MB, at ~240 GB/s of an A6000's 768 -- so it is not even
#   bandwidth-bound, it is kernel-count-bound.
#
# This fixes the SECOND one only, and does it in the cheapest way available: the same arithmetic in
# two warp kernels instead of forty torch ones. No data structure changes, nothing to rebuild, and
# capture still works.
#
# TWO KERNELS, NOT ONE, AND THE REASON IS THE CURVATURE. `kappa = -div(n)` needs the unit normal at
# the six face neighbours, and each of those needs `grad(c)` at that neighbour, i.e. `c` two cells
# out. One pass would make every thread evaluate the gradient seven times over a 25-point stencil.
# Splitting after `grad` costs one [n_cells, 3] scratch buffer (10.6 MB at 96^3) and leaves each
# thread reading seven gradients it did not compute. `nrm` is NOT stored: it is `grad/(|grad|+eps)`,
# recovered in the second kernel from the gradients it is already reading.
#
# WHAT IT REFUSES, LOUDLY. A fused kernel that quietly skipped a term would be far worse than no
# fused kernel, so every feature outside the 3D non-periodic normalised-CSF path raises rather than
# being ignored: 2D, periodic, `csf_smooth > 0`, the legacy unnormalised colour (`csf_rho == 0`,
# whose interface test is a reduction over the whole grid), and the plate BC. Those specs keep the
# default implementation, which is why this is an `implementation:` and not a rewrite.
#
# NOT BIT-IDENTICAL to `default`: same operations in the same order per cell, but float addition is
# not associative across a different lowering. `tools/mpm_grid_probe.py` measures the difference
# against the torch operator on real state rather than asserting it.
# ==========================================================================================================
if HAVE_WARP:

    @wp.kernel
    def grid_colour_grad(gc: wp.array(dtype=wp.float32),
                         grad: wp.array(dtype=wp.vec3),
                         nx: int, ny: int, nz: int,
                         inv_cell_mass: float, half_inv_dx: float):
        """grad(c) by central differences, with `torch.roll`'s WRAPAROUND at the box faces.

        The torch form uses `torch.roll`, which is periodic even on a non-periodic run, so a cell on
        the -x face differences against the +x face. That is arguably wrong and it is what every
        existing run did; reproducing it is the point of an alternative implementation.
        """
        t = wp.tid()
        k = t % nz
        j = (t // nz) % ny
        i = t // (nz * ny)
        ip = (i + 1) % nx
        im = (i - 1 + nx) % nx
        jp = (j + 1) % ny
        jm = (j - 1 + ny) % ny
        kp = (k + 1) % nz
        km = (k - 1 + nz) % nz
        # `c` is formed BEFORE the difference, as the torch code forms the whole `c` field first.
        gx = (gc[(ip * ny + j) * nz + k] * inv_cell_mass
              - gc[(im * ny + j) * nz + k] * inv_cell_mass) * half_inv_dx
        gy = (gc[(i * ny + jp) * nz + k] * inv_cell_mass
              - gc[(i * ny + jm) * nz + k] * inv_cell_mass) * half_inv_dx
        gz = (gc[(i * ny + j) * nz + kp] * inv_cell_mass
              - gc[(i * ny + j) * nz + km] * inv_cell_mass) * half_inv_dx
        grad[t] = wp.vec3(gx, gy, gz)

    @wp.func
    def _unit(g: wp.vec3, eps: float) -> wp.vec3:
        """n = grad / (|grad| + eps) -- the CSF normal, with the same additive regularisation the
        torch path uses. NOT `wp.normalize`, which divides by |g| alone and blows up in the bulk."""
        m = wp.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) + eps
        return wp.vec3(g[0] / m, g[1] / m, g[2] / m)

    @wp.kernel
    def grid_solve(gm: wp.array(dtype=wp.float32),
                   gmv: wp.array(dtype=wp.vec3),
                   gc: wp.array(dtype=wp.float32),
                   grad: wp.array(dtype=wp.vec3),
                   walls: wp.array(dtype=wp.uint8),
                   bf: wp.array(dtype=wp.vec3),
                   gv: wp.array(dtype=wp.vec3),
                   nx: int, ny: int, nz: int,
                   dt: float, half_inv_dx: float, cell_vol: float,
                   mass_floor: float, csf_floor: float,
                   surf: float, inv_cell_mass: float, band: float, gain: float, eps: float,
                   full_mass: float, wd: float, mu: float,
                   buoy: float, rho_ref: float, bdir: wp.vec3,
                   has_bf: int, has_csf: int, has_walls: int, has_buoy: int):
        t = wp.tid()
        k = t % nz
        j = (t // nz) % ny
        i = t // (nz * ny)

        m = gm[t]
        mv = gmv[t]
        mc = wp.max(m, mass_floor)
        v = wp.vec3(mv[0] / mc, mv[1] / mc, mv[2] / mc)

        if has_bf == 1:                                  # `body_force: grid` handover from the scatter
            a = bf[t]
            v = wp.vec3(v[0] + dt * a[0], v[1] + dt * a[1], v[2] + dt * a[2])

        if has_csf == 1:
            ip = (i + 1) % nx
            im = (i - 1 + nx) % nx
            jp = (j + 1) % ny
            jm = (j - 1 + ny) % ny
            kp = (k + 1) % nz
            km = (k - 1 + nz) % nz
            nxp = _unit(grad[(ip * ny + j) * nz + k], eps)
            nxm = _unit(grad[(im * ny + j) * nz + k], eps)
            nyp = _unit(grad[(i * ny + jp) * nz + k], eps)
            nym = _unit(grad[(i * ny + jm) * nz + k], eps)
            nzp = _unit(grad[(i * ny + j) * nz + kp], eps)
            nzm = _unit(grad[(i * ny + j) * nz + km], eps)
            # kappa = -div(n), summed left to right exactly as the torch generator expression is.
            kap = -(((nxp[0] - nxm[0]) * half_inv_dx + (nyp[1] - nym[1]) * half_inv_dx)
                    + (nzp[2] - nzm[2]) * half_inv_dx)
            c = gc[t] * inv_cell_mass
            fm = float(0.0)
            if c > band and c < 1.0 - band and m > band * full_mass:
                fm = gain
            inv_m = cell_vol / wp.max(m, csf_floor)
            gr = grad[t]
            v = wp.vec3(v[0] + dt * (surf * kap * gr[0] * fm) * inv_m,
                        v[1] + dt * (surf * kap * gr[1] * fm) * inv_m,
                        v[2] + dt * (surf * kap * gr[2] * fm) * inv_m)

        # REFLECTIVE BOX WALLS, IN THE TORCH ORDER, WHICH IS NOT SYMMETRIC AND MATTERS.
        # The torch loop writes axis k's clamped component back before damping the OTHER components
        # on that axis's slabs, and axis 1 then reads a y-velocity axis 0 may already have damped.
        # Keeping `v` in registers and walking k in the same order reproduces that exactly. Note the
        # slabs are asymmetric -- `idx < 3` is three cells, `idx > n - 3` is two -- because the torch
        # test is strict at the top; this is behaviour, not a bug being fixed here.
        lo0 = i < 3
        hi0 = i > nx - 3
        lo1 = j < 3
        hi1 = j > ny - 3
        lo2 = k < 3
        hi2 = k > nz - 3
        vx = v[0]
        vy = v[1]
        vz = v[2]
        # COULOMB FRICTION, MIRRORING THE TORCH LOOP: the inward normal speed is read BEFORE the
        # clamp removes it -- that impulse is the friction budget -- and after the clamp the other
        # two components on that axis's slabs are reduced by mu times it, to zero if the budget
        # covers them (stick). `mu == 0` takes none of these branches, so a frictionless run is the
        # kernel it always was. See MPMGridUpdate.__init__ for the rule and the reference.
        vin0 = 0.0
        if lo0:
            vin0 = wp.max(-vx, 0.0)
            vx = wp.max(vx, 0.0)
        if hi0:
            vin0 = wp.max(vx, 0.0)
            vx = wp.min(vx, 0.0)
        if wd != 1.0 and (lo0 or hi0):
            vy = vy * wd
            vz = vz * wd
        if mu > 0.0 and (lo0 or hi0):
            vt0 = wp.max(wp.sqrt(vy * vy + vz * vz), 1.0e-12)
            f0 = wp.max(1.0 - mu * vin0 / vt0, 0.0)
            vy = vy * f0
            vz = vz * f0
        vin1 = 0.0
        if lo1:
            vin1 = wp.max(-vy, 0.0)
            vy = wp.max(vy, 0.0)
        if hi1:
            vin1 = wp.max(vy, 0.0)
            vy = wp.min(vy, 0.0)
        if wd != 1.0 and (lo1 or hi1):
            vx = vx * wd
            vz = vz * wd
        if mu > 0.0 and (lo1 or hi1):
            vt1 = wp.max(wp.sqrt(vx * vx + vz * vz), 1.0e-12)
            f1 = wp.max(1.0 - mu * vin1 / vt1, 0.0)
            vx = vx * f1
            vz = vz * f1
        vin2 = 0.0
        if lo2:
            vin2 = wp.max(-vz, 0.0)
            vz = wp.max(vz, 0.0)
        if hi2:
            vin2 = wp.max(vz, 0.0)
            vz = wp.min(vz, 0.0)
        if wd != 1.0 and (lo2 or hi2):
            vx = vx * wd
            vy = vy * wd
        if mu > 0.0 and (lo2 or hi2):
            vt2 = wp.max(wp.sqrt(vx * vx + vy * vy), 1.0e-12)
            f2 = wp.max(1.0 - mu * vin2 / vt2, 0.0)
            vx = vx * f2
            vy = vy * f2
        v = wp.vec3(vx, vy, vz)

        if has_walls == 1 and walls[t] != wp.uint8(0):   # no-slip inside a solid obstacle
            v = wp.vec3(0.0, 0.0, 0.0)

        if has_buoy == 1:
            rho = m / cell_vol
            f = float(0.0)
            if rho > 1.0e-9:
                f = (rho - rho_ref) / wp.max(rho, 1.0e-9)
            v = wp.vec3(v[0] + dt * buoy * f * bdir[0],
                        v[1] + dt * buoy * f * bdir[1],
                        v[2] + dt * buoy * f * bdir[2])

        gv[t] = v


@register_operator("mpm_grid_update", implementation="warp", family="mpm",
                   set="field", kind="field", title="solve grid velocities, one kernel", equation=r"""$$\mathbf v_g = \frac{\mathbf p_g}{m_g},
\qquad
\mathbf v_g \mathrel{+}= \Delta t\,\mathbf f^{\,\text{CSF}}_g,
\qquad
\mathbf v_g \leftarrow \mathrm{BC}(\mathbf v_g)$$""")
class MPMGridUpdateWarp(MPMGridUpdate):
    r"""the 3D grid solve (mass normalisation, the continuum surface force, box walls, obstacles and buoyancy) as two Warp kernels rather than several dozen whole-grid torch operations (CUDA, 3D only).
    mpm_grid -> mpm_grid: as mpm_grid_update.

        \mathbf v_g = \frac{\mathbf p_g}{m_g}, \qquad \mathbf v_g \mathrel{+}= \Delta t\,\mathbf f^{\,\text{CSF}}_g, \qquad \mathbf v_g \leftarrow \mathrm{BC}(\mathbf v_g)
        v_g            the velocity of grid node g
        p_g            its momentum
        m_g            its mass (nodes below `mass_floor`, a mass that scales with the particle mass, get zero)
        \Delta t       `dt_sub`, the substep
        f^{CSF}_g      the continuum surface force sigma kappa n from the mollified colour field (`cell_surface_tension`, `csf_*`)
        BC             the walls (`wall_damp` the restitution, `wall_friction`), the closing plates (`plate_*`), the obstacles and buoyancy (`buoyancy`, `rho_ref`)

    Reference: as mpm_grid_update; NVIDIA Warp."""
    PARAM_UNITS = {'buoyancy': 'acceleration', 'csf_band': 'fraction', 'csf_eps': 'dimensionless', 'csf_mass_floor': 'mass', 'csf_rho': 'density', 'csf_smooth': 'count', 'dt_sub': 'time', 'mass_floor': 'mass', 'plate_axis': 'count', 'plate_centre': 'length', 'plate_close_from': 'frames', 'plate_close_to': 'frames', 'plate_gap_half': 'length', 'plate_gap_half_end': 'length', 'rho_ref': 'density', 'cell_surface_tension': 'energy_per_area', 'wall_damp': 'fraction', 'wall_friction': 'fraction'}
    SYMBOLS = {'v_g': 'the velocity of grid node g', 'p_g': 'its momentum', 'm_g': 'its mass (nodes below `mass_floor`, a mass that scales with the particle mass, get zero)', '\\Delta t': '`dt_sub`, the substep', 'f^{CSF}_g': 'the continuum surface force sigma kappa n from the mollified colour field (`cell_surface_tension`, `csf_*`)', 'BC': 'the walls (`wall_damp` the restitution, `wall_friction`), the closing plates (`plate_*`), the obstacles and buoyancy (`buoyancy`, `rho_ref`)'}

    # `wall_friction` is imposed in `grid_solve`, mirroring the torch loop axis for axis. This is
    # the implementation the engine selects when a spec leaves it unset, so it had to be -- the
    # first friction measurement ran entirely on this path and the base __init__'s refusal was the
    # only thing that stopped a frictionless run being read as a null result.
    HONOURS_WALL_FRICTION = True

    MECHANISM_TAGS = ["grid_solve", "surface_tension", "boundary_conditions", "fused_kernel"]
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False

    def forward(self, H, mask=None):
        if not HAVE_WARP:
            raise RuntimeError("mpm_grid_update[warp] needs warp-lang")
        from plexus.operators.mpm_ops import sub_dt
        g = H.field(self.at); dev = g.m.device
        D = g.dim
        if D != 3 or str(dev) == "cpu" or bool(getattr(H, "periodic", False)):
            raise RuntimeError("mpm_grid_update[warp] is 3D, non-periodic, CUDA only")
        if self.csf_smooth:
            raise RuntimeError("mpm_grid_update[warp] does not implement csf_smooth "
                               "(it needs a third pass); drop `implementation: warp`")
        if self.plate_axis is not None:
            raise RuntimeError("mpm_grid_update[warp] does not implement the plate BC; "
                               "drop `implementation: warp`")
        if self.surface_tension > 0.0 and self.csf_rho <= 0.0:
            raise RuntimeError("mpm_grid_update[warp] does not implement the LEGACY unnormalised "
                               "colour (csf_rho unset): its interface test is a reduction over the "
                               "whole grid. Set csf_rho, or drop `implementation: warp`")
        dt = sub_dt(H, self.dt_sub)
        dx = float(g.dx); inv_dx = float(g.inv_dx)
        cell_vol = dx ** 3
        n = int(g.m.numel())
        nx, ny, nz = g.shape

        # CACHED ONCE, at a stable address so a captured graph keeps reading the buffer the run
        # keeps writing. `grad` is scratch between the two kernels and never leaves this operator.
        if getattr(self, "_grad", None) is None or self._grad.shape[0] != n:
            self._grad = torch.zeros(n, 3, device=dev)
            self._zero3 = torch.zeros(1, 3, device=dev)
        # RESOLVED ONCE. `bool(mask.any())` is a device->host sync: run per substep it drains the
        # launch queue 13 times a frame, and inside a CUDA-graph capture it is illegal outright.
        # The obstacle set is a run constant (`_walls3d` itself caches on the grid shape), so the
        # flag is settled on the first call and read from an int thereafter.
        if getattr(self, "_wall_u8", None) is None or self._wall_u8.numel() != n:
            walls = self._walls3d(H, g, dev)
            self._wall_u8 = walls.to(torch.uint8).contiguous()
            self._has_walls = 1 if bool(self._wall_u8.any()) else 0
        has_walls = self._has_walls

        if getattr(self, "_c_csf", None) is None:
            self._c_csf = bool(self.surface_tension > 0.0 and bool((g.c > 0).any()))
        has_csf = 1 if self._c_csf else 0
        inv_cell_mass = 1.0 / (self.csf_rho * cell_vol) if self.csf_rho > 0.0 else 1.0
        gain = 1.0 / max(1.0 - 2.0 * self.csf_band, 1e-6) if self.csf_band > 0 else 1.0
        if self.csf_band <= 0.0 and has_csf:
            raise RuntimeError("mpm_grid_update[warp] needs csf_band > 0 when surface tension is "
                               "on: the band-free interface test is a global reduction")

        _bf = getattr(H, "_mpm_body_accel", None)
        bf = _bf if _bf is not None else self._zero3
        buoy_dir = [0.0, -1.0, 0.0]
        if self.buoy_dir is not None:
            for _i, _x in enumerate(self.buoy_dir[:3]):
                buoy_dir[_i] = float(_x)

        if has_csf:
            _wp_launch(grid_colour_grad, n, dev,
                       [wp.from_torch(g.c.contiguous()),
                        wp.from_torch(self._grad, dtype=wp.vec3),
                        int(nx), int(ny), int(nz),
                        float(inv_cell_mass), float(0.5 * inv_dx)])
        _wp_launch(grid_solve, n, dev,
                   [wp.from_torch(g.m.contiguous()),
                    wp.from_torch(g.mv.view(-1, 3), dtype=wp.vec3),
                    wp.from_torch(g.c.contiguous()),
                    wp.from_torch(self._grad, dtype=wp.vec3),
                    wp.from_torch(self._wall_u8),
                    wp.from_torch(bf.view(-1, 3), dtype=wp.vec3),
                    wp.from_torch(g.v.view(-1, 3), dtype=wp.vec3),
                    int(nx), int(ny), int(nz),
                    float(dt), float(0.5 * inv_dx), float(cell_vol),
                    float(self._const("mass_floor", g)), float(self._const("csf_mass_floor", g)),
                    float(self.surface_tension), float(inv_cell_mass),
                    float(self.csf_band), float(gain), float(self._const("csf_eps", g)),
                    float(self.csf_rho * cell_vol), float(self.wall_damp),
                    float(self.wall_friction),
                    float(self.buoyancy), float(self.rho_ref),
                    wp.vec3(buoy_dir[0], buoy_dir[1], buoy_dir[2]),
                    int(_bf is not None), int(has_csf), int(has_walls),
                    int(self.buoyancy != 0.0)])
        return {}


# ==========================================================================================================
# `mpm_scatter` in Triton: one fused kernel, and an atomic-free colour-ordered variant.
#
# THE DEFAULTS ARE REGISTERED ABOVE THIS LINE AND MUST STAY THERE -- see the Warp section for why.
# ==========================================================================================================
try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:                                        # no triton -> the operator refuses at build
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _p2g(X, V, C, F, MASS, MU, LA, PVOL, AEXT, GM, GMV, GC, LIQ,
             N, NG: tl.constexpr, DX: tl.constexpr, DT: tl.constexpr,
             DRAG: tl.constexpr, ITERS: tl.constexpr, HAS_LIQ: tl.constexpr,
             BLOCK: tl.constexpr):
        """One program handles BLOCK particles: strain -> stress -> weights -> scatter."""
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        m = off < N
        inv_dx = 1.0 / DX

        x0 = tl.load(X + off * 3 + 0, mask=m, other=0.0)
        x1 = tl.load(X + off * 3 + 1, mask=m, other=0.0)
        x2 = tl.load(X + off * 3 + 2, mask=m, other=0.0)
        v0 = tl.load(V + off * 3 + 0, mask=m, other=0.0)
        v1 = tl.load(V + off * 3 + 1, mask=m, other=0.0)
        v2 = tl.load(V + off * 3 + 2, mask=m, other=0.0)
        a0 = tl.load(AEXT + off * 3 + 0, mask=m, other=0.0)
        a1 = tl.load(AEXT + off * 3 + 1, mask=m, other=0.0)
        a2 = tl.load(AEXT + off * 3 + 2, mask=m, other=0.0)
        mass = tl.load(MASS + off, mask=m, other=0.0)
        mu = tl.load(MU + off, mask=m, other=0.0)
        la = tl.load(LA + off, mask=m, other=0.0)
        pv = tl.load(PVOL + off, mask=m, other=0.0)
        # LIQUID COLOUR, the field the CSF surface tension is computed from -- `w * mass * liquid`,
        # the same deposit the torch and warp scatters make. Without it `mpm_grid_update` finds gc
        # all zero, its `_c_csf` predicate is False, and the ENTIRE surface-tension branch is
        # skipped: `cell_surface_tension: 0.64` measured spread_r90 0.17831 and level_p95 0.61445,
        # identical to SEVEN FIGURES to the same run at sigma = 0. HAS_LIQ is constexpr so a
        # non-liquid spec compiles the loads and the atomic away entirely.
        liq = tl.load(LIQ + off, mask=m, other=0.0) if HAS_LIQ else 0.0

        # body force + Stokes drag, as the torch operator does before the scatter
        v0 = v0 + DT * (a0 - DRAG * v0)
        v1 = v1 + DT * (a1 - DRAG * v1)
        v2 = v2 + DT * (a2 - DRAG * v2)

        c00 = tl.load(C + off * 9 + 0, mask=m, other=0.0); c01 = tl.load(C + off * 9 + 1, mask=m, other=0.0)
        c02 = tl.load(C + off * 9 + 2, mask=m, other=0.0); c10 = tl.load(C + off * 9 + 3, mask=m, other=0.0)
        c11 = tl.load(C + off * 9 + 4, mask=m, other=0.0); c12 = tl.load(C + off * 9 + 5, mask=m, other=0.0)
        c20 = tl.load(C + off * 9 + 6, mask=m, other=0.0); c21 = tl.load(C + off * 9 + 7, mask=m, other=0.0)
        c22 = tl.load(C + off * 9 + 8, mask=m, other=0.0)
        f00 = tl.load(F + off * 9 + 0, mask=m, other=1.0); f01 = tl.load(F + off * 9 + 1, mask=m, other=0.0)
        f02 = tl.load(F + off * 9 + 2, mask=m, other=0.0); f10 = tl.load(F + off * 9 + 3, mask=m, other=0.0)
        f11 = tl.load(F + off * 9 + 4, mask=m, other=1.0); f12 = tl.load(F + off * 9 + 5, mask=m, other=0.0)
        f20 = tl.load(F + off * 9 + 6, mask=m, other=0.0); f21 = tl.load(F + off * 9 + 7, mask=m, other=0.0)
        f22 = tl.load(F + off * 9 + 8, mask=m, other=1.0)

        # F IS READ, NEVER WRITTEN. An earlier draft folded `F <- (I + dt C) F` in here, on the
        # grounds that the updated F never leaves a register. It is wrong twice over. Mechanically:
        # `mpm_strain` is scheduled separately in every spec, so F was advanced TWICE per substep --
        # caught by comparing against the default, which agreed on grid mass to eleven digits and
        # disagreed on F by 3.3e-04. And contractually: `implementation` means the same biology
        # computed differently, and the deformation-gradient update is a DIFFERENT MECHANISM with
        # its own operator. Fusing across that boundary would make this a new operator wearing the
        # scatter's name. It costs one extra elementwise kernel per substep, which is 8% of the
        # frame against atomics at 99.8%.
        J = (f00 * (f11 * f22 - f12 * f21) - f01 * (f10 * f22 - f12 * f20)
             + f02 * (f10 * f21 - f11 * f20))

        # --- polar factor R by Newton on the cofactor form (the `higham` path) ---
        r00, r01, r02 = f00, f01, f02
        r10, r11, r12 = f10, f11, f12
        r20, r21, r22 = f20, f21, f22
        for _ in tl.static_range(ITERS):
            # cofactor columns: c_k = r_{k+1} x r_{k+2}  (column cross products)
            k00 = r11 * r22 - r12 * r21; k10 = r12 * r20 - r10 * r22; k20 = r10 * r21 - r11 * r20
            k01 = r21 * r02 - r22 * r01; k11 = r22 * r00 - r20 * r02; k21 = r20 * r01 - r21 * r00
            k02 = r01 * r12 - r02 * r11; k12 = r02 * r10 - r00 * r12; k22 = r00 * r11 - r01 * r10
            d = r00 * k00 + r01 * k10 + r02 * k20
            d = tl.where(tl.abs(d) < 1e-12, 1e-12, d)
            r00 = 0.5 * (r00 + k00 / d); r01 = 0.5 * (r01 + k01 / d); r02 = 0.5 * (r02 + k02 / d)
            r10 = 0.5 * (r10 + k10 / d); r11 = 0.5 * (r11 + k11 / d); r12 = 0.5 * (r12 + k12 / d)
            r20 = 0.5 * (r20 + k20 / d); r21 = 0.5 * (r21 + k21 / d); r22 = 0.5 * (r22 + k22 / d)

        # --- fixed-corotated Kirchhoff stress: 2 mu (F - R) F^T + I la J (J-1) ---
        d00 = f00 - r00; d01 = f01 - r01; d02 = f02 - r02
        d10 = f10 - r10; d11 = f11 - r11; d12 = f12 - r12
        d20 = f20 - r20; d21 = f21 - r21; d22 = f22 - r22
        two_mu = 2.0 * mu
        p = la * J * (J - 1.0)
        s00 = two_mu * (d00 * f00 + d01 * f01 + d02 * f02) + p
        s01 = two_mu * (d00 * f10 + d01 * f11 + d02 * f12)
        s02 = two_mu * (d00 * f20 + d01 * f21 + d02 * f22)
        s10 = two_mu * (d10 * f00 + d11 * f01 + d12 * f02)
        s11 = two_mu * (d10 * f10 + d11 * f11 + d12 * f12) + p
        s12 = two_mu * (d10 * f20 + d11 * f21 + d12 * f22)
        s20 = two_mu * (d20 * f00 + d21 * f01 + d22 * f02)
        s21 = two_mu * (d20 * f10 + d21 * f11 + d22 * f12)
        s22 = two_mu * (d20 * f20 + d21 * f21 + d22 * f22) + p
        k = (-DT * 4.0 * inv_dx * inv_dx) * pv
        # affine = stress_scaled + mass * C
        q00 = k * s00 + mass * c00; q01 = k * s01 + mass * c01; q02 = k * s02 + mass * c02
        q10 = k * s10 + mass * c10; q11 = k * s11 + mass * c11; q12 = k * s12 + mass * c12
        q20 = k * s20 + mass * c20; q21 = k * s21 + mass * c21; q22 = k * s22 + mass * c22

        # --- quadratic B-spline base cell and fractional offset ---
        b0 = tl.floor(x0 * inv_dx - 0.5); b1 = tl.floor(x1 * inv_dx - 0.5); b2 = tl.floor(x2 * inv_dx - 0.5)
        fx0 = x0 * inv_dx - b0; fx1 = x1 * inv_dx - b1; fx2 = x2 * inv_dx - b2

        for i in tl.static_range(3):
            wi = tl.where(i == 0, 0.5 * (1.5 - fx0) * (1.5 - fx0),
                 tl.where(i == 1, 0.75 - (fx0 - 1.0) * (fx0 - 1.0),
                          0.5 * (fx0 - 0.5) * (fx0 - 0.5)))
            for j in tl.static_range(3):
                wj = tl.where(j == 0, 0.5 * (1.5 - fx1) * (1.5 - fx1),
                     tl.where(j == 1, 0.75 - (fx1 - 1.0) * (fx1 - 1.0),
                              0.5 * (fx1 - 0.5) * (fx1 - 0.5)))
                for kk in tl.static_range(3):
                    wk = tl.where(kk == 0, 0.5 * (1.5 - fx2) * (1.5 - fx2),
                         tl.where(kk == 1, 0.75 - (fx2 - 1.0) * (fx2 - 1.0),
                                  0.5 * (fx2 - 0.5) * (fx2 - 0.5)))
                    w = wi * wj * wk
                    gi = tl.minimum(tl.maximum(b0 + i, 0.0), NG - 1.0)
                    gj = tl.minimum(tl.maximum(b1 + j, 0.0), NG - 1.0)
                    gk = tl.minimum(tl.maximum(b2 + kk, 0.0), NG - 1.0)
                    idx = ((gi * NG + gj) * NG + gk).to(tl.int32)
                    dp0 = (i - fx0) * DX; dp1 = (j - fx1) * DX; dp2 = (kk - fx2) * DX
                    mom0 = mass * v0 + (q00 * dp0 + q01 * dp1 + q02 * dp2)
                    mom1 = mass * v1 + (q10 * dp0 + q11 * dp1 + q12 * dp2)
                    mom2 = mass * v2 + (q20 * dp0 + q21 * dp1 + q22 * dp2)
                    tl.atomic_add(GM + idx, w * mass, mask=m)
                    tl.atomic_add(GMV + idx * 3 + 0, w * mom0, mask=m)
                    tl.atomic_add(GMV + idx * 3 + 1, w * mom1, mask=m)
                    tl.atomic_add(GMV + idx * 3 + 2, w * mom2, mask=m)
                    if HAS_LIQ:
                        tl.atomic_add(GC + idx, w * mass * liq, mask=m)


# ==========================================================================================================
# COLOURED, ATOMIC-FREE P2G -- `mpm_scatter[implementation: triton_colour]`
#
# THE ATOMICS ARE THE WALL, measured: for 1M particles the loads and index arithmetic cost 0.027 ms
# and the 108 global atomics cost 12.7 ms -- 99.8% of the kernel -- and it is SUPERLINEAR in the
# atomic count (27 atomics 0.46 ms, 108 atomics 12.7 ms), because the three momentum components hit
# consecutive addresses at the same node and warps serialise on them. No amount of further fusion
# touches that.
#
# THE FIX IS TO MAKE THE CONFLICTS IMPOSSIBLE RATHER THAN TO SERIALISE THEM. A particle in base cell
# c writes nodes c, c+1, c+2 on each axis. Two cells whose indices differ by >= 3 on ANY axis
# therefore have disjoint node stencils. Colour the cells by (cx%3, cy%3, cz%3) -- 27 colours -- and
# every cell within a colour is conflict-free with every other, so the scatter becomes a plain
# load-add-store. Twenty-seven launches per substep instead of one, each touching 1/27 of the cells.
#
# WHY NOT THE NODE-CENTRIC GATHER, which also removes atomics. A gather has every particle read by
# each of the 27 nodes it touches: 27x read amplification, and a ragged inner loop per node that
# vectorises badly. Colouring reads each particle ONCE. The cost is that the sort has to exist.
#
# THE SORT IS CHEAP AND CAPTURE-SAFE. Particles move at most 0.4 dx per substep (the CFL cap), so
# the binning is stable; `torch.sort` and a `bincount` with an explicit `minlength` both have static
# output shapes, so nothing here blocks a CUDA-graph capture.
# ==========================================================================================================
if HAVE_TRITON:

    @triton.jit
    def _p2g_colour(XS, VS, CS, FS, MASSS, MUS, LAS, PVOLS, AEXTS,
                    CELL_OFF, CELL_ID, GM, GMV, GC, LIQS, NCOL,
                    NG: tl.constexpr, DX: tl.constexpr, DT: tl.constexpr,
                    DRAG: tl.constexpr, ITERS: tl.constexpr, HAS_LIQ: tl.constexpr):
        """One program = one CELL. The 27-node stencil is the VECTOR dimension.

        THE FIRST VERSION PUT THE NODE LOOP OUTSIDE THE PARTICLE LOOP, which recomputed the polar
        decomposition and the stress once per node -- 27x the constitutive work per particle -- and
        came out 2x SLOWER than the PyTorch operator it was meant to beat (646 ms/frame against
        330). Here the particle loop is outer and its 27 weights are a `tl.arange` vector, so the
        stress is computed ONCE and the scatter is a 27-wide vector accumulate into registers,
        written to the grid once at the end.
        """
        pid = tl.program_id(0)
        if pid >= NCOL:
            return
        cell = tl.load(CELL_ID + pid)
        lo = tl.load(CELL_OFF + cell)
        hi = tl.load(CELL_OFF + cell + 1)
        inv_dx = 1.0 / DX
        ci = cell // (NG * NG); cj = (cell // NG) % NG; ck = cell % NG

        n = tl.arange(0, 32)                      # 27 stencil slots, padded to a power of two
        act = n < 27
        si = n // 9; sj = (n // 3) % 3; sk = n % 3
        gi = tl.minimum(tl.maximum(ci + si, 0), NG - 1)
        gj = tl.minimum(tl.maximum(cj + sj, 0), NG - 1)
        gk = tl.minimum(tl.maximum(ck + sk, 0), NG - 1)
        idx = (gi * NG + gj) * NG + gk
        fi = si.to(tl.float32); fj = sj.to(tl.float32); fk = sk.to(tl.float32)

        am = tl.zeros([32], dtype=tl.float32)
        a0 = tl.zeros([32], dtype=tl.float32)
        a1 = tl.zeros([32], dtype=tl.float32)
        a2 = tl.zeros([32], dtype=tl.float32)
        ac = tl.zeros([32], dtype=tl.float32)      # liquid colour, for the CSF surface tension

        for q in tl.range(lo, hi):
            x0 = tl.load(XS + q*3+0); x1 = tl.load(XS + q*3+1); x2 = tl.load(XS + q*3+2)
            b0 = tl.floor(x0*inv_dx-0.5); b1 = tl.floor(x1*inv_dx-0.5); b2 = tl.floor(x2*inv_dx-0.5)
            fx0 = x0*inv_dx-b0; fx1 = x1*inv_dx-b1; fx2 = x2*inv_dx-b2
            mass = tl.load(MASSS + q)
            v0 = tl.load(VS + q*3+0); v1 = tl.load(VS + q*3+1); v2 = tl.load(VS + q*3+2)
            e0 = tl.load(AEXTS + q*3+0); e1 = tl.load(AEXTS + q*3+1); e2 = tl.load(AEXTS + q*3+2)
            v0 = v0 + DT*(e0 - DRAG*v0); v1 = v1 + DT*(e1 - DRAG*v1); v2 = v2 + DT*(e2 - DRAG*v2)
            mu = tl.load(MUS + q); la = tl.load(LAS + q); pv = tl.load(PVOLS + q)
            f00 = tl.load(FS + q*9+0); f01 = tl.load(FS + q*9+1); f02 = tl.load(FS + q*9+2)
            f10 = tl.load(FS + q*9+3); f11 = tl.load(FS + q*9+4); f12 = tl.load(FS + q*9+5)
            f20 = tl.load(FS + q*9+6); f21 = tl.load(FS + q*9+7); f22 = tl.load(FS + q*9+8)
            c00 = tl.load(CS + q*9+0); c01 = tl.load(CS + q*9+1); c02 = tl.load(CS + q*9+2)
            c10 = tl.load(CS + q*9+3); c11 = tl.load(CS + q*9+4); c12 = tl.load(CS + q*9+5)
            c20 = tl.load(CS + q*9+6); c21 = tl.load(CS + q*9+7); c22 = tl.load(CS + q*9+8)
            J = f00*(f11*f22-f12*f21) - f01*(f10*f22-f12*f20) + f02*(f10*f21-f11*f20)
            r00 = f00; r01 = f01; r02 = f02
            r10 = f10; r11 = f11; r12 = f12
            r20 = f20; r21 = f21; r22 = f22
            for _ in tl.static_range(ITERS):
                k00 = r11*r22-r12*r21; k10 = r12*r20-r10*r22; k20 = r10*r21-r11*r20
                k01 = r21*r02-r22*r01; k11 = r22*r00-r20*r02; k21 = r20*r01-r21*r00
                k02 = r01*r12-r02*r11; k12 = r02*r10-r00*r12; k22 = r00*r11-r01*r10
                d = r00*k00 + r01*k10 + r02*k20
                d = tl.where(tl.abs(d) < 1e-12, 1e-12, d)
                r00 = 0.5*(r00+k00/d); r01 = 0.5*(r01+k01/d); r02 = 0.5*(r02+k02/d)
                r10 = 0.5*(r10+k10/d); r11 = 0.5*(r11+k11/d); r12 = 0.5*(r12+k12/d)
                r20 = 0.5*(r20+k20/d); r21 = 0.5*(r21+k21/d); r22 = 0.5*(r22+k22/d)
            two_mu = 2.0*mu; pp = la*J*(J-1.0); kk = (-DT*4.0*inv_dx*inv_dx)*pv
            d00 = f00-r00; d01 = f01-r01; d02 = f02-r02
            d10 = f10-r10; d11 = f11-r11; d12 = f12-r12
            d20 = f20-r20; d21 = f21-r21; d22 = f22-r22
            q00 = kk*(two_mu*(d00*f00+d01*f01+d02*f02)+pp) + mass*c00
            q01 = kk*(two_mu*(d00*f10+d01*f11+d02*f12))    + mass*c01
            q02 = kk*(two_mu*(d00*f20+d01*f21+d02*f22))    + mass*c02
            q10 = kk*(two_mu*(d10*f00+d11*f01+d12*f02))    + mass*c10
            q11 = kk*(two_mu*(d10*f10+d11*f11+d12*f12)+pp) + mass*c11
            q12 = kk*(two_mu*(d10*f20+d11*f21+d12*f22))    + mass*c12
            q20 = kk*(two_mu*(d20*f00+d21*f01+d22*f02))    + mass*c20
            q21 = kk*(two_mu*(d20*f10+d21*f11+d22*f12))    + mass*c21
            q22 = kk*(two_mu*(d20*f20+d21*f21+d22*f22)+pp) + mass*c22
            # the 27 weights, as vectors
            wi = tl.where(si == 0, 0.5*(1.5-fx0)*(1.5-fx0),
                 tl.where(si == 1, 0.75-(fx0-1.0)*(fx0-1.0), 0.5*(fx0-0.5)*(fx0-0.5)))
            wj = tl.where(sj == 0, 0.5*(1.5-fx1)*(1.5-fx1),
                 tl.where(sj == 1, 0.75-(fx1-1.0)*(fx1-1.0), 0.5*(fx1-0.5)*(fx1-0.5)))
            wk = tl.where(sk == 0, 0.5*(1.5-fx2)*(1.5-fx2),
                 tl.where(sk == 1, 0.75-(fx2-1.0)*(fx2-1.0), 0.5*(fx2-0.5)*(fx2-0.5)))
            w = tl.where(act, wi*wj*wk, 0.0)
            dp0 = (fi-fx0)*DX; dp1 = (fj-fx1)*DX; dp2 = (fk-fx2)*DX
            am += w*mass
            a0 += w*(mass*v0 + (q00*dp0+q01*dp1+q02*dp2))
            a1 += w*(mass*v1 + (q10*dp0+q11*dp1+q12*dp2))
            a2 += w*(mass*v2 + (q20*dp0+q21*dp1+q22*dp2))
            if HAS_LIQ:
                ac += w*mass*tl.load(LIQS + q)

        # CONFLICT-FREE: no other cell of this colour writes these nodes, so plain read-add-write.
        tl.store(GM + idx, tl.load(GM + idx, mask=act, other=0.0) + am, mask=act)
        tl.store(GMV + idx*3+0, tl.load(GMV + idx*3+0, mask=act, other=0.0) + a0, mask=act)
        tl.store(GMV + idx*3+1, tl.load(GMV + idx*3+1, mask=act, other=0.0) + a1, mask=act)
        tl.store(GMV + idx*3+2, tl.load(GMV + idx*3+2, mask=act, other=0.0) + a2, mask=act)
        if HAS_LIQ:
            tl.store(GC + idx, tl.load(GC + idx, mask=act, other=0.0) + ac, mask=act)


# ==========================================================================================================
# `mpm_gather[torch_loop27]`: the 27-stencil loop, kept as the readable reference and the
# low-memory path.
#
# THE DEFAULTS ARE REGISTERED ABOVE THIS LINE AND MUST STAY THERE -- see the Warp section for why.
# ==========================================================================================================
# ----------------------------------------------------------------------------------------------
# The two active constitutive laws. Both turn an activation field into something the MPM substep
# consumes -- a body acceleration (`active_force`) or an active stress (`active_stress`) -- and
# `mpm_scatter` is the only reader of what either emits.
# ----------------------------------------------------------------------------------------------
@register_operator("active_force", "pulse_to_contraction", family="mechanics", set="particle", kind="exchange",
                   equation=r"""$$\mathbf F_i \;=\;
\begin{cases}
\pm\,\text{amplitude}\,\nabla a(\mathbf x_i) & \text{gradient mode (inward / outward)}\\[4pt]
\text{amplitude}\,a(\mathbf x_i)\,\mathbf d(\mathbf x_i) & \text{directional mode}
\end{cases}$$""", title="contract by an active body force")
class ActiveForce(Exchange):                    
    r"""active contraction as a body force: an activation field becomes a per-particle acceleration pointing along the field's own gradient (the cruder claim; active_stress is the stress route).
    field -> particle: reads the activation field at each particle's position, emits an external acceleration the MPM substep consumes as a_ext.

        \mathbf F_i \;=\; \begin{cases} \pm\,\text{amplitude}\,\nabla a(\mathbf x_i) & \text{gradient mode (inward / outward)}\\[4pt] \text{amplitude}\,a(\mathbf x_i)\,\mathbf d(\mathbf x_i) & \text{directional mode} \end{cases}
        F_i            the acceleration on particle i
        amplitude      `amplitude`, converting a field gradient into an acceleration
        a(x_i)         the activation field at the particle, dimensionless (`from`, `channel`)
        d(x_i)         the field's direction in directional mode
        \pm            the sign `along` chooses: inward contracts toward high activation, outward pushes away

    Reference: Marchetti, M. C. et al. (2013). Hydrodynamics of soft active matter. Rev. Mod. Phys. 85:1143-1189."""
    PARAM_UNITS = {'amplitude': 'acceleration', 'channel': 'count'}
    SYMBOLS = {'F_i': 'the acceleration on particle i', 'amplitude': '`amplitude`, converting a field gradient into an acceleration', 'a(x_i)': 'the activation field at the particle, dimensionless (`from`, `channel`)', 'd(x_i)': "the field's direction in directional mode", '\\pm': 'the sign `along` chooses: inward contracts toward high activation, outward pushes away'}

    EMIT = "mpm_acceleration"           # a body accel the MPM substep consumes as a_ext, not engine-integrated
    SUPPORTED_DIMS = [2]                 # 2D — reads a 2-vector activation gradient / direction field
    REQUIRES_PARAMS = ["from"]                # the activation field to read
    MECHANISM_TAGS = ["active_contraction", "field_gradient_force", "directed_active_stress"]
    PARAM_ROLES = {"amplitude": "contraction_strength", "mode": "gradient_or_directional", 'along': 'inward_or_outward', 'channel': 'field_channel_read', 'from': 'the_activation_field'}
    REFERENCE = "Marchetti, M. C. et al. (2013). Hydrodynamics of soft active matter. Rev. Mod. Phys. 85:1143-1189."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("from")
        self.amplitude = float(params.get("amplitude", 50.0))
        self.channel = int(params.get("channel", 0))
        # `mode:` carried three words doing two different jobs: `inward`/`outward` chose a SIGN
        # on one rule, while `directional` chose a DIFFERENT RULE. One key over two axes, and
        # nothing in the registry could tell them apart. `along:` is the value; `model:
        # directional` is the hypothesis.
        if "mode" in params:
            raise ValueError(
                "active_force: `mode` is gone. `inward`/`outward` are now `along:` (a value on this "
                "model -- the sign of the gradient), and `directional` is now "
                "`model: directional` (a different rule: a prescribed direction field).")
        self.along = str(params.get("along", "inward"))
        if self.along not in ("inward", "outward"):
            raise ValueError(f"active_force: along must be inward|outward, got {self.along!r}")
        self.sign = 1.0 if self.along == "inward" else -1.0
        self.at = params.get("_at", "particle")

    def _accel(self, H, lvl, pos, fld):
        """THE HYPOTHESIS, and the only thing a `model=` variant of active_force changes.

        Default: the contraction follows the ACTIVATION GRADIENT -- direction = +/- grad(a), with
        `along:` choosing the sign. `inward` pulls up the gradient, toward higher activation.
        """
        grad = fld.grad_at(pos, self.channel, periodic=getattr(H, "periodic", False))   # [N, 2]
        return self.sign * self.amplitude * grad                          # inward for sign>0

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        fld = H.fields[self.field_name]
        acc = self._accel(H, lvl, pos, fld)
        acc = acc * lvl.occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        # return a per-particle force delta; the engine sums it (with drag's) into
        # H.delta(mpm_particle), which p2g consumes as the MPM body force. EMIT=None,
        # so the engine never integrates the particle set (g2p owns advection).
        return {self.at: acc}


def contact_share(ca, cb, area, live, n_cells, full_area):
    """The share of each cell's surface that touches another cell, from a junction table (exp39, 2026-11-07): the contact area of cell c, summed over the live junction rows that name it (`ca`, `cb`
    the two cells of a row, `area` the row's contact area), over `full_area`, the contact area at which the cell counts as fully in contact, clipped to [0, 1]. The paper's S_c / S of the active
    pressure (Ongenae et al. 2025, Methods), the gate that TRAPS T-84 asks for."""
    a = area * live.to(area.dtype)
    A = torch.zeros(n_cells, device=area.device, dtype=area.dtype)
    A = A.index_add(0, ca, a).index_add(0, cb, a)
    return (A / max(float(full_area), 1e-12)).clamp(0.0, 1.0)


def contact_push(ca, cb, area, live, drive, pol, xbar, full_area, vel=None, xi=None):
    """The acceleration of every cell from the tangential pushes at its contacts (exp39, 2026-11-07), one row of a junction table at a time: cell `ca` pushes cell `cb` along the part of its polarity `pol`
    that lies in the contact plane (normal = the line of the two centroids `xbar`), with the strength `drive` of the pusher times the row's contact area over `full_area`, and `cb` pushes back with the
    opposite force; the row then also carries the push of `cb` on `ca`, with its own reaction. With `vel` (the mean velocity of each cell) and `xi` (the cell-cell friction of each cell), the row also
    damps the sliding of the two cells over each other: the part of their relative velocity in the contact plane, times the mean friction of the pair and the row's share, taken from the faster cell and given
    to the slower. Every row is an equal and opposite pair, so the forces of all cells add to exactly zero."""
    d = xbar[cb] - xbar[ca]
    n = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
    tang = lambda v: v - (v * n).sum(1, keepdim=True) * n
    s = (area * live.to(area.dtype) / max(float(full_area), 1e-12)).clamp(0.0, 1.0)[:, None]
    delta = s * (drive[ca][:, None] * tang(pol[ca]) - drive[cb][:, None] * tang(pol[cb]))
    if vel is not None and xi is not None:
        delta = delta - s * (0.5 * (xi[ca] + xi[cb]))[:, None] * tang(vel[cb] - vel[ca])
    A = torch.zeros(xbar.shape[0], 3, device=area.device, dtype=area.dtype)
    return A.index_add(0, cb, delta).index_add(0, ca, -delta)


@register_operator("active_force", "pulse_to_contraction", family="mechanics", set="particle",
                   kind="exchange", model="persistent_walk",
                   equation=r"""$$\mathbf a_i=f_c\,s_c(t)\,\big[1+b\tanh(\xi_i/w)\big]\,\mathbf p_c,\qquad \xi_i=\frac{(\mathbf x_i-\bar{\mathbf x}_c)\cdot\mathbf p_c}{R_c},\qquad d\mathbf p_c=\sqrt{2D_c}\,(\mathbf I-\mathbf p_c\mathbf p_c^{\!\top})\,d\mathbf W_c$$""", title="drive cells on a random walk")
class ActiveForcePersistentWalk(ActiveForce):
    r"""the `persistent_walk` model of active_force: every material point of a cell pushed along its cell's polarity, the polarity wandering on the sphere, with the cell-to-cell and moment-to-moment variability real crawling cells show.
    particle -[containment]-> cell: reads which cell owns each point (`cell_set`) and the points' positions; keeps one polarity, drive and speed factor per cell as operator state; emits the acceleration the MPM substep consumes.

        \mathbf a_i=f_c\,s_c(t)\,\big[1+b\tanh(\xi_i/w)\big]\,\mathbf p_c,\qquad \xi_i=\frac{(\mathbf x_i-\bar{\mathbf x}_c)\cdot\mathbf p_c}{R_c},\qquad d\mathbf p_c=\sqrt{2D_c}\,(\mathbf I-\mathbf p_c\mathbf p_c^{\!\top})\,d\mathbf W_c
        a_i            the acceleration of point i of cell c
        f_c            the cell's drive per unit mass (`f` the median; log-normal spread `speed_cv`)
        s_c(t)         the stop-and-go speed factor (`speed_noise`, `speed_tau`)
        b              `front_bias`, the leading edge's extra push
        \xi_i          where along its cell point i sits, from -1 (rear) to 1 (front)
        w              `front_width`
        p_c            the cell's unit polarity
        \bar x_c       the cell's centroid
        R_c            its radius
        D_c            the rotational diffusion (`Dr`; persistence time 1/(2 D) on the sphere; `ucsp` couples it to speed)
        dW_c           a Wiener increment

    Reference: Romanczuk, P. et al. (2012). Active Brownian particles. Eur. Phys. J. Spec. Top. 202:1-162; Wu, P.-H. et al. (2014). PNAS 111:3949-3954 (3D cell migration is not a random walk); Maiuri, P. et al. (2015). Cell 161:374-386 (the speed-persistence coupling)."""
    PARAM_UNITS = {'gate_area': 'dimensionless', 'f_block': 'dimensionless', 'Dr_block': 'dimensionless', 'Dr': 'rate', 'f': 'acceleration', 'front_bias': 'dimensionless', 'front_width': 'fraction', 'seed': 'count', 'speed_cv': 'fraction', 'speed_noise': 'fraction', 'speed_tau': 'time', 'ucsp': 'dimensionless'}
    SYMBOLS = {'a_i': 'the acceleration of point i of cell c', 'f_c': "the cell's drive per unit mass (`f` the median; log-normal spread `speed_cv`)", 's_c(t)': 'the stop-and-go speed factor (`speed_noise`, `speed_tau`)', 'b': "`front_bias`, the leading edge's extra push", '\\xi_i': 'where along its cell point i sits, from -1 (rear) to 1 (front)', 'w': '`front_width`', 'p_c': "the cell's unit polarity", '\\bar x_c': "the cell's centroid", 'R_c': 'its radius', 'D_c': 'the rotational diffusion (`Dr`; persistence time 1/(2 D) on the sphere; `ucsp` couples it to speed)', 'dW_c': 'a Wiener increment'}
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["cell_set", "f"]
    MECHANISM_TAGS = ["motility", "self_propulsion", "persistent_random_walk", "active_brownian",
                      "speed_persistence_coupling", "cell_variability"]
    PARAM_ROLES = {"f": "median_drive_per_unit_mass", "Dr": "polarity_rotational_diffusion",
                   "cell_set": "the_cells_owning_the_points", "seed": "rng_seed",
                   "speed_cv": "log_normal_spread_of_drive_across_cells",
                   "ucsp": "speed_persistence_coupling_exponent",
                   "speed_noise": "ou_fluctuation_of_each_cells_speed", "speed_tau": "ou_correlation_time",
                   "front_bias": "leading_edge_excess_of_the_push", "front_width": "front_rear_softness", 'cil_surface': 'surface_set_for_contact_inhibition',
                   "f_block": "cell_block_with_each_cells_drive", "Dr_block": "cell_block_with_each_cells_rotational_diffusion",
                   "contact_gate": "junction_set_whose_contact_area_gates_the_drive", "gate_area": "contact_area_of_a_fully_touching_cell",
                   "type_f": "drive_of_the_cells_of_a_named_type"}
    REFERENCE = ("Romanczuk, P. et al. (2012). Eur. Phys. J. Spec. Top. 202:1-162; Wu, P.-H. et al. "
                 "(2014). PNAS 111:3949-3954; Maiuri, P. et al. (2015). Cell 161:374-386.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.cell_set = str(params["cell_set"])
        self.f = float(params["f"])
        self.Dr = float(params.get("Dr", 0.1))
        self.seed = int(params.get("seed", 0))
        self.speed_cv = float(params.get("speed_cv", 0.0))
        self.ucsp = float(params.get("ucsp", 0.0))
        self.speed_noise = float(params.get("speed_noise", 0.0))
        self.speed_tau = float(params.get("speed_tau", 10.0))
        self.front_bias = float(params.get("front_bias", 0.0))
        self.front_width = float(params.get("front_width", 0.5))
        # `cil_surface: <vertex set>` (default none = off): CONTACT INHIBITION OF LOCOMOTION at a tissue surface
        # (Abercrombie 1979; Carmona-Fontaine et al. 2008). Each call, a cell with any point in this frame's
        # `bm_contact[live]` list against that surface (its apical side, the lumen's wall) loses the part of its
        # polarity that points INTO the wall -- it turns back into the lumen instead of pushing on. Without it the
        # walking interior of exp 11 pushed the layer with the contact's full force while receiving a tenth of it
        # back, and inflated the gland like a gas (Finding 105).
        self.cil_surface = params.get("cil_surface")
        # exp39 (2026-11-07), all three off by default: `f_block` / `Dr_block` read the drive and the rotational diffusion of each cell from a cell block (a SWEEP of `f` or `Dr`, one value per array of a
        # bench, as the `_block` options of the adhesion operators do); `contact_gate: <junction set>` multiplies each cell's drive by the share of its surface in contact with other cells, the contact
        # area of its junction rows over `gate_area` (default 4.84 length^2, the surface of a ball of volume 1.0), clipped to 1: Ongenae et al. 2025's active pressure p_a (x.p) S_c / S, which a free
        # cell does not have (TRAPS T-84: an ungated drive lets the surface cells of a clump walk off while the interior jams)
        self.f_block = params.get("f_block")
        self.Dr_block = params.get("Dr_block")
        self.contact_gate = params.get("contact_gate")
        self.gate_area = float(params.get("gate_area", 4.84))
        # `type_f: {B: 0.08, A: 0.0}` (default none, exp06 2026-11-14): the drive of the cells of a named TYPE (a fate cell_fate gives from a signal), replacing `f` for them each
        # call, the per-cell spread `speed_cv` kept: active cells crawl, the others sit
        self.type_f = {str(k): float(v) for k, v in (params.get("type_f") or {}).items()}
        self._p = None

    def _init_cells(self, nc):
        # ONE GENERATOR, ON THE CPU, SEEDED ONCE: the draws do not depend on the device or on how
        # many other operators drew before this one (the reason `brownian[active_cell]` does it too).
        self._gen = torch.Generator(device="cpu").manual_seed(self.seed)
        p = torch.randn(nc, 3, generator=self._gen, dtype=torch.float64)
        self._p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-12)
        g = torch.randn(nc, generator=self._gen, dtype=torch.float64)
        sf = self.speed_cv
        # MEDIAN-PRESERVING, NOT MEAN-PRESERVING: exp(sigma g - sigma^2/2) has mean 1, so `f` stays
        # the population's mean drive whatever the spread.
        self._fc = self.f * torch.exp(sf * g - 0.5 * sf * sf)
        self._Dc = self.Dr * torch.exp(-self.ucsp * (self._fc / max(abs(self.f), 1e-12) - 1.0))
        self._s = torch.ones(nc, dtype=torch.float64)

    def _inhibit(self, H, nc):
        """`cil_surface`: remove, per cell touching the surface this frame, its polarity's component against the
        contacts' mean normal (the normal INTO the lumen), and renormalise."""
        m = getattr(H.level(str(self.cil_surface)), "_mesh", None)
        bc = m.get("basal_contacts") if m is not None else None
        C = bc.get(self.at) if isinstance(bc, dict) else None
        fr = getattr(H, "frame", None)
        if not C or C.get("frame") != (None if fr is None else int(fr)) or C["node"].numel() == 0:
            return
        par = H.lift_index(self.at, self.cell_set)[C["node"].long()].cpu()
        n = C["n"].detach().to(torch.float64).cpu()
        ns = torch.zeros(nc, 3, dtype=torch.float64).index_add_(0, par, n)
        hit = ns.norm(dim=1) > 1e-12
        if not bool(hit.any()):
            return
        nh = ns[hit] / ns[hit].norm(dim=1, keepdim=True)
        p = self._p[hit]
        into_wall = (p * nh).sum(1, keepdim=True).clamp(max=0.0)             # < 0: pointing out of the lumen
        p = p - into_wall * nh
        self._p[hit] = p / p.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def _step(self, h):
        """Advance every cell's polarity and speed factor by one frame of length h.

        ALONG THE SPHERE, IN SUB-STEPS: each sub-step draws a tangent Gaussian v (variance 2 D_c h_s
        per tangent axis) and moves along the great circle it points to, p <- p cos|v| + v/|v| sin|v|
        (the exponential map). A straight tangent step followed by a renormalisation is the obvious
        alternative and it UNDER-TURNS: its mean cosine is 1 - s^2 + 3 s^4 per step (s^2 = 2 D_c h_s)
        against the exact exp(-s^2) = 1 - s^2 + s^4 / 2, measured 0.632 against 0.619 per frame at
        Dr 0.24 with 24 sub-steps -- a 4% slower walk. The arc's mean cosine is 1 - s^2 + s^4 / 3,
        and with s^2 <= 0.02 per sub-step its error is ~0.3% of the decorrelation."""
        n_sub = max(1, int(math.ceil(float(self._Dc.max()) * h / 0.01)))
        hs = h / n_sub
        for _ in range(n_sub):
            z = torch.randn(self._p.shape[0], 3, generator=self._gen, dtype=torch.float64)
            z = z - (z * self._p).sum(1, keepdim=True) * self._p           # tangent to the sphere
            v = torch.sqrt(2.0 * self._Dc * hs)[:, None] * z
            th = v.norm(dim=1, keepdim=True)
            p = self._p * torch.cos(th) + v * (torch.sin(th) / th.clamp_min(1e-12))
            self._p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-12)
        if self.speed_noise > 0.0:
            tau = max(self.speed_tau, 1e-9)
            zs = torch.randn(self._s.shape[0], generator=self._gen, dtype=torch.float64)
            self._s = (self._s - (self._s - 1.0) * (h / tau)
                       + math.sqrt(2.0 * h / tau) * self.speed_noise * zs).clamp_min(0.0)

    def _advance(self, H, cl, nc):
        """This frame's polarity (and speed-factor) step of every cell; the first call draws the initial polarities."""
        h = float(getattr(H, "dt", 1.0))
        if self._p is None or self._p.shape[0] != nc:
            self._init_cells(nc)
            if self.Dr_block:
                self._Dc = cl.get(self.Dr_block)[:, 0].detach().to("cpu", torch.float64).clamp_min(0.0)
        else:
            if self.Dr_block:                                               # the diffusion of this frame's step, per cell, from its block
                self._Dc = cl.get(self.Dr_block)[:, 0].detach().to("cpu", torch.float64).clamp_min(0.0)
            self._step(h)
        if self.cil_surface:
            self._inhibit(H, nc)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        X = lvl.get("pos")
        cl = H.level(self.cell_set)
        nc = int(cl.state.shape[0])
        self._advance(H, cl, nc)
        if self.f == 0.0 and not self.f_block and not any(v != 0.0 for v in self.type_f.values()):
            return {self.at: torch.zeros_like(X)}
        idx = H.lift_index(self.at, self.cell_set)                          # [n] the cell of point i
        dev, dt_ = X.device, X.dtype
        pc = self._p.to(device=dev, dtype=dt_)[idx]
        drive = self._fc if not self.f_block else cl.get(self.f_block)[:, 0].detach().to("cpu", torch.float64)
        if self.type_f:
            names = list(getattr(cl, "type_names", []) or [])
            nt = cl.node_type.detach().to("cpu").long()
            rel = self._fc / max(abs(self.f), 1e-12) if self.f != 0.0 else torch.ones_like(self._fc)
            drive = drive.clone()
            for nm, fv in self.type_f.items():
                if nm in names:
                    m = nt[: drive.shape[0]] == names.index(nm)
                    drive[m] = fv * rel[m]
        amp = (drive * self._s).to(device=dev, dtype=dt_)
        if self.contact_gate:
            J = H.level(str(self.contact_gate))
            ca = J.get("ca")[:, 0].round().long().clamp(0, nc - 1)
            cb = J.get("cb")[:, 0].round().long().clamp(0, nc - 1)
            amp = amp * contact_share(ca, cb, J.get("area")[:, 0].to(dt_), J.occ > 0.5, nc, self.gate_area)
        amp = amp[idx]
        if self.front_bias != 0.0:
            live = lvl.occ.to(dt_)
            w = torch.zeros(nc, device=dev, dtype=dt_).index_add_(0, idx, live)
            xbar = torch.zeros(nc, 3, device=dev, dtype=dt_).index_add_(0, idx, X * live[:, None])
            xbar = xbar / w.clamp_min(1.0)[:, None]
            d = X - xbar[idx]
            r2 = torch.zeros(nc, device=dev, dtype=dt_).index_add_(0, idx, (d * d).sum(1) * live)
            R = (r2 / w.clamp_min(1.0)).clamp_min(1e-12).sqrt()
            xi = (d * pc).sum(1) / R[idx]
            amp = amp * (1.0 + self.front_bias * torch.tanh(xi / max(self.front_width, 1e-6)))
        a = amp[:, None] * pc * lvl.occ[:, None].to(dt_)
        if mask is not None:
            a = a * mask[:, None].to(dt_)
        return {self.at: a}


@register_operator("active_force", "pulse_to_contraction", family="mechanics", set="particle",
                   kind="exchange", model="contact_push",
                   equation=r"""$$\mathbf a_c=\sum_{c'\in\partial c}\frac{A_{cc'}}{A_0}\,\mathbf P_{cc'}\big(f_{c'}\mathbf p_{c'}-f_c\mathbf p_c+\xi_{cc'}(\mathbf v_{c'}-\mathbf v_c)\big),\qquad \mathbf P_{cc'}=\mathbf I-\mathbf n_{cc'}\mathbf n_{cc'}^{\!\top},\qquad d\mathbf p_c=\sqrt{2D_c}\,(\mathbf I-\mathbf p_c\mathbf p_c^{\!\top})\,d\mathbf W_c$$""", title="push neighbours along the polarity at the contacts")
class ActiveForceContactPush(ActiveForcePersistentWalk):
    r"""the `contact_push` model of active_force: the cell's protrusion pushes on its neighbours across the contact plane and the neighbours push back, so a cell that touches nothing is not driven and the forces of all cells add to zero.
    particle -[containment]-> cell: as persistent_walk for the polarity; reads the junction set `contact_gate` (cell pair and contact area of every row) and the centroids of the cells; emits a uniform acceleration on the points of each cell.

        \mathbf a_c=\sum_{c'\in\partial c}\frac{A_{cc'}}{A_0}\,\mathbf P_{cc'}\big(f_{c'}\mathbf p_{c'}-f_c\mathbf p_c+\xi_{cc'}(\mathbf v_{c'}-\mathbf v_c)\big),\qquad \mathbf P_{cc'}=\mathbf I-\mathbf n_{cc'}\mathbf n_{cc'}^{\!\top},\qquad d\mathbf p_c=\sqrt{2D_c}\,(\mathbf I-\mathbf p_c\mathbf p_c^{\!\top})\,d\mathbf W_c
        a_c            the acceleration of cell c (the same on each of its points, scaled by mean points per cell over its own)
        c'             a neighbour of cell c in the junction set (`contact_gate`)
        A_{cc'}        the contact area of the pair, clipped to A_0 per junction row
        A_0            `gate_area`, the contact area at which a pair counts as fully in contact
        P_{cc'}        the projector onto the contact plane
        n_{cc'}        the unit vector from the centroid of c to that of c'
        f_c            the cell's drive (`f` or `f_block`, times the stop-and-go factor)
        p_c            the cell's unit polarity
        \xi_{cc'}       the mean cell-cell friction of the pair (`friction`, `friction_block`)
        v_c            the mean velocity of the points of cell c
        D_c            the rotational diffusion (`Dr`)
        dW_c           a Wiener increment

    Reference: Ongenae, S. et al. (2025). Nat. Commun. 16:10467 (Methods, 3D active foam model: the reaction traction on the contacting surfaces); Romanczuk, P. et al. (2012). Eur. Phys. J. Spec. Top. 202:1-162."""
    PARAM_UNITS = dict(ActiveForcePersistentWalk.PARAM_UNITS, friction='rate', friction_block='dimensionless')
    SYMBOLS = {'a_c': 'the acceleration of cell c (the same on each of its points, scaled by mean points per cell over its own)', "c'": 'a neighbour of cell c in the junction set (`contact_gate`)',
               "A_{cc'}": 'the contact area of the pair, clipped to A_0 per junction row', 'A_0': '`gate_area`, the contact area at which a pair counts as fully in contact',
               "P_{cc'}": 'the projector onto the contact plane', "n_{cc'}": 'the unit vector from the centroid of c to that of c\'',
               'f_c': "the cell's drive (`f` or `f_block`, times the stop-and-go factor)", 'p_c': "the cell's unit polarity",
               "\\xi_{cc'}": 'the mean cell-cell friction of the pair (`friction`, `friction_block`)', 'v_c': 'the mean velocity of the points of cell c', 'D_c': 'the rotational diffusion (`Dr`)', 'dW_c': 'a Wiener increment'}
    PARAM_ROLES = dict(ActiveForcePersistentWalk.PARAM_ROLES, contact_gate="junction_set_whose_rows_carry_the_pushes", gate_area="contact_area_of_a_fully_touching_pair",
                       friction="cell_cell_friction_on_the_sliding_in_the_contact_plane", friction_block="cell_block_with_each_cells_friction")
    REQUIRES_PARAMS = ["cell_set", "f", "contact_gate"]
    MECHANISM_TAGS = ["motility", "contact_traction", "momentum_conserving", "persistent_random_walk", "active_foam"]
    REFERENCE = ("Ongenae, S. et al. (2025). Nat. Commun. 16:10467; Romanczuk, P. et al. (2012). Eur. Phys. J. Spec. Top. 202:1-162.")

    def __init__(self, params, device="cpu"):
        if not params.get("contact_gate"):
            raise ValueError("active_force[contact_push]: `contact_gate` (the junction set of contact_finder) is required, the push exists only at contacts")
        super().__init__(params, device)
        # `friction` (default 0 = off) or `friction_block`: the paper's cell-cell friction xi (Ongenae et al. 2025, Methods): the sliding of two touching cells in their contact plane is damped
        self.friction = float(params.get("friction", 0.0))
        self.friction_block = params.get("friction_block")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        X = lvl.get("pos")
        cl = H.level(self.cell_set)
        nc = int(cl.state.shape[0])
        self._advance(H, cl, nc)
        if self.f == 0.0 and not self.f_block:
            return {self.at: torch.zeros_like(X)}
        idx = H.lift_index(self.at, self.cell_set)
        dev, dt_ = X.device, X.dtype
        drive = self._fc if not self.f_block else cl.get(self.f_block)[:, 0].detach().to("cpu", torch.float64)
        amp = (drive * self._s).to(device=dev, dtype=dt_)
        live = lvl.occ.to(dt_)
        cnt = torch.zeros(nc, device=dev, dtype=dt_).index_add_(0, idx, live)
        xbar = torch.zeros(nc, 3, device=dev, dtype=dt_).index_add_(0, idx, X * live[:, None]) / cnt.clamp_min(1.0)[:, None]
        J = H.level(str(self.contact_gate))
        ca = J.get("ca")[:, 0].round().long().clamp(0, nc - 1)
        cb = J.get("cb")[:, 0].round().long().clamp(0, nc - 1)
        vel = xi = None
        if self.friction_block or self.friction != 0.0:
            V = lvl.get("vel")
            vel = torch.zeros(nc, 3, device=dev, dtype=dt_).index_add_(0, idx, V * live[:, None]) / cnt.clamp_min(1.0)[:, None]
            xi = cl.get(self.friction_block)[:, 0].detach().to(device=dev, dtype=dt_) if self.friction_block else torch.full((nc,), self.friction, device=dev, dtype=dt_)
        acc = contact_push(ca, cb, J.get("area")[:, 0].to(dt_), J.occ > 0.5, amp, self._p.to(device=dev, dtype=dt_), xbar, self.gate_area, vel, xi)
        acc = acc * (cnt[cnt > 0].mean() / cnt.clamp_min(1.0))[:, None]
        a = acc[idx] * live[:, None]
        if mask is not None:
            a = a * mask[:, None].to(dt_)
        return {self.at: a}


@register_operator("active_force", "pulse_to_contraction", family="mechanics", set="particle",
                   kind="exchange", model="track_push",
                   equation=r"""$$\mathbf p_c\leftarrow\sigma_c\,\frac{\sum_{k\in\partial c\cap T}\mathbf t_k}{\lVert\sum_{k\in\partial c\cap T}\mathbf t_k\rVert},\qquad \mathbf t_k\parallel\mathbf x_{k+1}-\mathbf x_{k-1};\qquad \mathbf a_c\ \text{as contact\_push}$$""", title="walk along the filament a motor holds")
class ActiveForceTrackPush(ActiveForceContactPush):
    r"""the `track_push` model of active_force: contact_push for a motor on a filament. The polarity of a cell that touches cells of the type `track` is set, at every call, to the axis of the track where it touches it, with the cell's own sign; the push is contact_push's, so the motor runs along the filament instead of across it.
    particle -[containment]-> cell: as contact_push; reads also the cell types and the centroids (the block `pos` of the cell set) for the track's axis. A cell that touches no track cell keeps the wandering polarity of persistent_walk.

        \mathbf p_c\leftarrow\sigma_c\,\frac{\sum_{k\in\partial c\cap T}\mathbf t_k}{\lVert\sum_{k\in\partial c\cap T}\mathbf t_k\rVert},\qquad \mathbf t_k\parallel\mathbf x_{k+1}-\mathbf x_{k-1}
        p_c            the unit polarity of cell c, used by contact_push's push
        \sigma_c       the cell's own sign, +1 or -1, drawn once: which end of a track it walks toward
        \partial c \cap T   the cells of type `track` that share a live junction row with c
        t_k            the unit axis of the track at its cell k: toward the cell of the next slot, k+1, from that of the slot before, k-1, when those are track cells closer than `track_reach` (one side alone at a track's end)
        x_k            the centroid of cell k
        a_c            the acceleration of cell c, contact_push's

    The track is a chain of cells seeded in slot order (seed_positions[from_points_ragged] keeps the order of its table): the slot order is what gives the axis one sense along the whole chain, which a nearest-neighbour direction would not.

    Reference: Plexus exp09 (this work, 2026-11-17): a motor whose polarity wanders pushes in the plane of its contact with a filament whichever way that polarity points, and leaves the filament (exp09 it23: 7 of 25 motors still on a filament after 1,000 frames); Howard, J. (2001). Mechanics of Motor Proteins and the Cytoskeleton. Sinauer (a motor's direction is set by the filament's polarity)."""
    PARAM_UNITS = dict(ActiveForceContactPush.PARAM_UNITS, track_reach='length')
    SYMBOLS = {'p_c': "the unit polarity of cell c, used by contact_push's push", '\\sigma_c': "the cell's own sign, +1 or -1, drawn once: which end of a track it walks toward",
               '\\partial c \\cap T': 'the cells of type `track` that share a live junction row with c',
               't_k': "the unit axis of the track at its cell k: toward the cell of the next slot from that of the slot before, when those are track cells closer than `track_reach`",
               'x_k': 'the centroid of cell k', 'a_c': "the acceleration of cell c, contact_push's"}
    PARAM_ROLES = dict(ActiveForceContactPush.PARAM_ROLES, track="cell_type_of_the_filament_subunits", track_reach="largest_distance_between_two_consecutive_subunits_of_a_track")
    REQUIRES_PARAMS = ["cell_set", "f", "contact_gate", "track"]
    MECHANISM_TAGS = ["motility", "contact_traction", "momentum_conserving", "filament_track", "processive_motor"]
    REFERENCE = ("Plexus exp09 (this work, 2026-11-17); Howard, J. (2001). Mechanics of Motor Proteins and the Cytoskeleton. Sinauer.")

    def __init__(self, params, device="cpu"):
        if not params.get("track"):
            raise ValueError("active_force[track_push]: `track`, the cell type of the filament subunits, is required")
        super().__init__(params, device)
        self.track = str(params["track"])
        self.track_reach = float(params.get("track_reach", 1.5))
        self._sign = None

    def _advance(self, H, cl, nc):
        super()._advance(H, cl, nc)                                        # the wandering polarity of every cell, as persistent_walk
        names = list(getattr(cl, "type_names", []) or [])
        if self.track not in names:
            return
        if self._sign is None or self._sign.shape[0] != nc:                # one sign per cell, drawn once from the operator's own generator
            self._sign = torch.where(torch.rand(nc, generator=self._gen, dtype=torch.float64) < 0.5, -1.0, 1.0).to(torch.float64)
        is_t = (cl.node_type.detach().to("cpu").long()[:nc] == names.index(self.track)) & (cl.occ.detach().to("cpu")[:nc] > 0.5)
        x = cl.get("pos").detach().to("cpu", torch.float64)[:nc]
        t = torch.zeros(nc, 3, dtype=torch.float64)                        # the track's axis at each of its cells, in the sense of increasing slot
        k = torch.arange(nc)
        for d in (1, -1):
            j = (k + d).clamp(0, nc - 1)
            v = (x[j] - x) * float(d)
            ok = is_t & is_t[j] & (j != k) & (v.norm(dim=1) < self.track_reach)
            t[ok] += v[ok]
        t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-12)
        J = H.level(str(self.contact_gate))
        ca = J.get("ca")[:, 0].detach().to("cpu").round().long().clamp(0, nc - 1)
        cb = J.get("cb")[:, 0].detach().to("cpu").round().long().clamp(0, nc - 1)
        live = J.occ.detach().to("cpu") > 0.5
        axis = torch.zeros(nc, 3, dtype=torch.float64)                     # per cell, the sum of the axes of the track cells it touches
        m = live & is_t[ca] & ~is_t[cb]
        axis.index_add_(0, cb[m], t[ca[m]])
        m = live & is_t[cb] & ~is_t[ca]
        axis.index_add_(0, ca[m], t[cb[m]])
        n = axis.norm(dim=1)
        on = n > 1e-9
        if bool(on.any()):
            p = self._p.clone()
            p[on] = self._sign[on, None] * axis[on] / n[on, None]
            self._p = p


@register_operator("active_force", "pulse_to_contraction", family="mechanics", set="particle",
                   kind="exchange", model="substrate_walk",
                   equation=r"""$$\mathbf a_i=f_c\,s_c(t)\,\chi_i\,\mathbf p_c,\qquad \mathbf p_c=(\cos\theta_c,\sin\theta_c,0),\qquad d\theta_c=-k\sin(\theta_c-\theta^{\rm cue}_c)\,dt+\sqrt{2D_c}\,dW_c$$""", title="crawl cells on their substrate")
class ActiveForceSubstrateWalk(ActiveForcePersistentWalk):
    r"""the `substrate_walk` model of active_force: a cell of an epithelium crawling on its substrate, the persistent walk with the polarity confined to the plane and the push applied at the cell's basal side (cryptic lamellipodia).
    particle -[containment]-> cell: as persistent_walk; the polarity is an angle in the plane normal to `plane_axis`, turned toward a cue (`cue`) and toward the cell's own velocity (`align`).

        \mathbf a_i=f_c\,s_c(t)\,\chi_i\,\mathbf p_c,\qquad \mathbf p_c=(\cos\theta_c,\sin\theta_c,0),\qquad d\theta_c=-k\sin(\theta_c-\theta^{\rm cue}_c)\,dt+\sqrt{2D_c}\,dW_c
        a_i            the acceleration of point i of cell c (points in the basal band `basal_band`)
        f_c            the cell's drive
        s_c(t)         its stop-and-go factor
        \chi_i         1 for a point in the basal band, 0 above it
        p_c            the in-plane unit polarity
        \theta_c       its angle
        k              `cue.strength`, the turning rate toward the cue direction, per time
        \theta^{cue}_c the cue's direction
        D_c            `Dr`, the in-plane rotational diffusion (persistence time 1/D)
        dW_c           a Wiener increment

    Reference: Farooqui, R. & Fenteany, G. (2005). J. Cell Sci. 118:51-63 (cryptic lamellipodia); Romanczuk, P. et al. (2012). Eur. Phys. J. Spec. Top. 202:1-162; Szabo, B. et al. (2006). Phys. Rev. E 74:061908 (alignment to one's own velocity)."""
    PARAM_UNITS = {'align': 'rate', 'basal_band': 'fraction', 'plane_axis': 'count'}
    SYMBOLS = {'a_i': 'the acceleration of point i of cell c (points in the basal band `basal_band`)', 'f_c': "the cell's drive", 's_c(t)': 'its stop-and-go factor', '\\chi_i': '1 for a point in the basal band, 0 above it', 'p_c': 'the in-plane unit polarity', '\\theta_c': 'its angle', 'k': '`cue.strength`, the turning rate toward the cue direction, per time', '\\theta^{cue}_c': "the cue's direction", 'D_c': '`Dr`, the in-plane rotational diffusion (persistence time 1/D)', 'dW_c': 'a Wiener increment'}
    PARAM_ROLES = dict(ActiveForcePersistentWalk.PARAM_ROLES, plane_axis="normal_of_the_substrate_plane",
                       basal_band="height_above_the_cells_lowest_point_that_is_pushed",
                       cue="dir_split_axis_strength_of_a_polarity_cue",
                       align="rate_of_polarity_turning_toward_own_velocity")
    MECHANISM_TAGS = ActiveForcePersistentWalk.MECHANISM_TAGS + ["substrate_crawling", "cryptic_lamellipodia"]
    REFERENCE = ("Farooqui, R. & Fenteany, G. (2005). J. Cell Sci. 118:51-63; Romanczuk, P. et al. (2012). "
                 "Eur. Phys. J. Spec. Top. 202:1-162.")

    def __init__(self, params, device="cpu"):
        if params.get("cil_surface"):
            raise ValueError("active_force[substrate_walk]: `cil_surface` is the gland interior's contact rule; "
                             "a cell on a substrate has no lumen wall to turn back from")
        super().__init__(params, device)
        self.plane_axis = int(params.get("plane_axis", 2))
        if self.plane_axis not in (0, 1, 2):
            raise ValueError("active_force[substrate_walk]: plane_axis is 0, 1 or 2")
        ax = [a for a in range(3) if a != self.plane_axis]
        E = torch.zeros(2, 3, dtype=torch.float64)
        E[0, ax[0]] = 1.0; E[1, ax[1]] = 1.0
        self._E = E                                                        # rows e1, e2 span the plane
        bb = params.get("basal_band")
        self.basal_band = None if bb is None else float(bb)
        cue = params.get("cue")
        self.cue_k, self.cue_dir, self.cue_split = 0.0, None, None
        if cue:
            d = torch.as_tensor([float(v) for v in cue["dir"]], dtype=torch.float64)
            d2 = E @ d
            if float(d2.norm()) < 1e-12:
                raise ValueError("active_force[substrate_walk]: cue.dir has no component in the plane")
            self.cue_dir = float(torch.atan2(d2[1], d2[0]))
            self.cue_k = float(cue.get("strength", 0.1))
            self.cue_split = None if cue.get("split_axis") is None else int(cue["split_axis"])
        self._theta = None
        self._cue_sign = None
        self.align = float(params.get("align", 0.0))
        if self.align < 0.0:
            raise ValueError("active_force[substrate_walk]: align is a rate, >= 0")
        self._prev_xb = None
        self._theta_v = None

    def _init_cells(self, nc):
        super()._init_cells(nc)                                            # drives, speed factors, generator
        self._theta = torch.rand(nc, generator=self._gen, dtype=torch.float64) * (2.0 * math.pi)
        self._sync_p()

    def _sync_p(self):
        c, s = torch.cos(self._theta), torch.sin(self._theta)
        self._p = c[:, None] * self._E[0][None, :] + s[:, None] * self._E[1][None, :]

    def _step(self, h):
        """theta by Euler-Maruyama sub-steps (variance 2 D_c h_s each, <= 0.02 rad^2), the cue's torque in each."""
        n_sub = max(1, int(math.ceil(float(self._Dc.max()) * h / 0.01)),
                    int(math.ceil(self.cue_k * h / 0.1)) if self.cue_k > 0 else 1,
                    int(math.ceil(self.align * h / 0.1)) if self.align > 0 else 1)
        hs = h / n_sub
        for _ in range(n_sub):
            z = torch.randn(self._theta.shape[0], generator=self._gen, dtype=torch.float64)
            dth = torch.sqrt(2.0 * self._Dc * hs) * z
            if self.cue_k > 0.0 and self._cue_sign is not None:
                tgt = self.cue_dir + torch.where(self._cue_sign > 0, 0.0, math.pi)
                dth = dth - self.cue_k * torch.sin(self._theta - tgt) * hs
            if self.align > 0.0 and self._theta_v is not None:
                tv, moving = self._theta_v
                dth = dth - self.align * torch.where(moving, torch.sin(self._theta - tv), torch.zeros_like(tv)) * hs
            self._theta = torch.remainder(self._theta + dth, 2.0 * math.pi)
        self._sync_p()
        if self.speed_noise > 0.0:
            tau = max(self.speed_tau, 1e-9)
            zs = torch.randn(self._s.shape[0], generator=self._gen, dtype=torch.float64)
            self._s = (self._s - (self._s - 1.0) * (h / tau)
                       + math.sqrt(2.0 * h / tau) * self.speed_noise * zs).clamp_min(0.0)

    def _cells_xbar(self, lvl, X, idx, nc):
        live = lvl.occ.to(X.dtype)
        w = torch.zeros(nc, device=X.device, dtype=X.dtype).index_add_(0, idx, live)
        xb = torch.zeros(nc, 3, device=X.device, dtype=X.dtype).index_add_(0, idx, X * live[:, None])
        return xb / w.clamp_min(1.0)[:, None], live, w

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        X = lvl.get("pos")
        nc = int(H.level(self.cell_set).state.shape[0])
        idx = H.lift_index(self.at, self.cell_set)
        if self._theta is None or self._theta.shape[0] != nc:
            self._init_cells(nc)
            fresh = True
        else:
            fresh = False
        if self.cue_k > 0.0 and (self._cue_sign is None or self._cue_sign.shape[0] != nc):
            xb, _live, _w = self._cells_xbar(lvl, X, idx, nc)
            if self.cue_split is None:
                self._cue_sign = torch.ones(nc, dtype=torch.float64)
            else:
                v = xb[:, self.cue_split].detach().double().cpu()
                alive = (_w > 0).detach().cpu()
                med = float(v[alive].median()) if bool(alive.any()) else 0.0
                self._cue_sign = torch.where(v > med, 1.0, -1.0).double()
        if self.align > 0.0:
            xb_now, _l, _w = self._cells_xbar(lvl, X, idx, nc)
            xb_now = xb_now.detach().double().cpu()
            if self._prev_xb is not None and self._prev_xb.shape == xb_now.shape:
                dv = (xb_now - self._prev_xb) @ self._E.T                        # [nc, 2] in-plane displacement
                self._theta_v = (torch.atan2(dv[:, 1], dv[:, 0]), dv.norm(dim=1) > 1e-9)
            self._prev_xb = xb_now
        if not fresh:
            self._step(float(getattr(H, "dt", 1.0)))
        if self.f == 0.0:
            return {self.at: torch.zeros_like(X)}
        dev, dt_ = X.device, X.dtype
        pc = self._p.to(device=dev, dtype=dt_)[idx]
        amp = (self._fc * self._s).to(device=dev, dtype=dt_)[idx]
        live = lvl.occ.to(dt_)
        chi = live
        if self.basal_band is not None:
            z = X[:, self.plane_axis]
            big = torch.full((nc,), float("inf"), device=dev, dtype=dt_)
            zmin = big.scatter_reduce(0, idx, torch.where(live > 0, z, torch.full_like(z, float("inf"))),
                                      reduce="amin", include_self=True)
            chi = live * (z <= zmin[idx] + self.basal_band).to(dt_)
            n_all = torch.zeros(nc, device=dev, dtype=dt_).index_add_(0, idx, live)
            n_bas = torch.zeros(nc, device=dev, dtype=dt_).index_add_(0, idx, chi)
            chi = chi * (n_all / n_bas.clamp_min(1.0))[idx]                  # the cell's total push unchanged
        if self.front_bias != 0.0:
            xbar, _l, w = self._cells_xbar(lvl, X, idx, nc)
            d = X - xbar[idx]
            r2 = torch.zeros(nc, device=dev, dtype=dt_).index_add_(0, idx, (d * d).sum(1) * live)
            R = (r2 / w.clamp_min(1.0)).clamp_min(1e-12).sqrt()
            xi = (d * pc).sum(1) / R[idx]
            amp = amp * (1.0 + self.front_bias * torch.tanh(xi / max(self.front_width, 1e-6)))
        a = (amp * chi)[:, None] * pc
        if mask is not None:
            a = a * mask[:, None].to(dt_)
        return {self.at: a}


# ----------------------------------------------------------------------------------------------
# Seeding MPM particles from a measured segmentation: their positions inside a measured mask and
# their per-cell Lame parameters. It lives here beside `ImageField`, whose label-image sibling it
# defines, and beside the material it writes -- there is no central seed module, and every other
# `kind="seed"` operator likewise sits with its own domain.
# ----------------------------------------------------------------------------------------------
@register_field("label_image", frame="label_image")
class LabelImageField(Field):
    """An integer instance map read from a TIFF: NOT normalised, and NEVER interpolated.

    The one thing it does that `image` cannot is return the id that is actually there. A bilinear
    weight between label 7 and label 12 is a number that means nothing and points at a cell that
    may not exist, so `sample_label` indexes rather than interpolates.

    Reference: none -- a measured segmentation is data, not a model.
    """

    def __init__(self, name, source=None, res=None, width=1.0, device="cpu", **kw):
        super().__init__(name)
        if source is None:
            raise ValueError(f"label_image field {name!r} needs a `source:` (path to a label .tif)")
        import tifffile
        path = source if os.path.isabs(source) else graphs_data_path(source)
        img = tifffile.imread(path)
        if img.ndim == 3:
            img = img[..., 0]
        img = img[::-1, :].copy()                       # image-top -> domain-top, as ImageField
        v = torch.tensor(img.astype("int64"), device=device).permute(1, 0).contiguous()
        self.C = 1
        self.nx, self.ny = int(v.shape[0]), int(v.shape[1])
        self.width = float(width)
        self.R = self.nx / self.width
        self.register_buffer("grid", v[None])           # [1, nx, ny] int64 labels
        self.n_labels = int(v.max())

    def sample_label(self, pos):
        """[N,2] world positions -> [N] integer label, nearest neighbour."""
        x = pos[:, 0].clamp(0, self.width - 1e-6) / self.width * self.nx
        y = pos[:, 1].clamp(0, self.width - 1e-6) / self.width * self.ny
        gx = x.long().clamp(0, self.nx - 1)
        gy = y.long().clamp(0, self.ny - 1)
        return self.grid[0][gx, gy]


@register_operator("seed_from_segmentation", family="seed", set="particle", kind="seed",
                   equation=r"""$$E_j=y_{\mathrm{lo}}+f_j\,(y_{\mathrm{hi}}-y_{\mathrm{lo}})$$""", title="build cells from a segmentation")
class SeedFromSegmentation(Seed):
    r"""builds a hierarchy from a measured instance segmentation: the cells are the ones in the image, not a lattice, and each carries its own material.
    label_image -> (cell, particle): reads an integer label map, writes the particle positions, their cell assignment and each cell's Lame parameters; runs once, at x_0.

        E_j=y_{\mathrm{lo}}+f_j\,(y_{\mathrm{hi}}-y_{\mathrm{lo}})
        E_j            cell j's Young's modulus, in the run's stress units
        y_{lo}, y_{hi} `youngs_min`, `youngs_max`
        f_j            the cell's stiffness fraction in [0, 1], from the `props` file (a cell that moved little in the recording is stiff) or a constant

    Reference: the segmentation is measured, not modelled; the material model whose mu and lambda this writes: Hu, Y. et al. (2018). A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. ACM Trans. Graph. 37(4):150."""
    PARAM_UNITS = {'jitter': 'fraction', 'youngs_min': 'stress', 'youngs_max': 'stress'}
    SYMBOLS = {'E_j': "cell j's Young's modulus, in the run's stress units", 'y_{lo}, y_{hi}': '`youngs_min`, `youngs_max`', 'f_j': "the cell's stiffness fraction in [0, 1], from the `props` file (a cell that moved little in the recording is stiff) or a constant"}

    EMIT = None
    # It establishes the configuration -- where the cells are and which particles belong to
    # them -- and writes the state buffer directly to do it. The engine's integration
    # invariant forbids that for a dynamics operator, correctly; a Seed is exempted because
    # establishing x_0 IS writing the state buffer (see base.Seed).
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["from"]
    SUPPORTED_DIMS = [2]
    MECHANISM_TAGS = ["instance_segmentation", "cell_identity", "heterogeneous_material"]
    PARAM_ROLES = {"youngs_min": "param_lo", "youngs_max": "param_hi",
                   "from": "label_field", "cell_set": "middle_level", 'jitter': 'sub_pixel_jitter_of_the_points', 'props': 'file_of_per_cell_properties'}
    REFERENCE = "instance segmentation measured from the beat; see prototype/cardio_cells"

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("from")
        self.at = params.get("_at", "mpm_particle")
        self.cell_set = params.get("cell_set", "cell")
        self.y_lo = float(params.get("youngs_min", 40.0))
        self.y_hi = float(params.get("youngs_max", 220.0))
        self.props = params.get("props")               # optional measured per-cell json
        self.jitter = float(params.get("jitter", 0.0))
        self._done = False

    def _cell_values(self, n_cells, device):
        """Per-cell Young's modulus: from the MEASURED beat when a props file is given, else a
        deterministic spread so the tissue is heterogeneous but reproducible.

        Measured is the interesting case. A cell that moved little in the recording is either stiff
        or weakly contractile; mapping amplitude to stiffness INVERSELY is one hypothesis about
        which, it is stated here rather than hidden, and the alternative (amplitude -> contraction
        gain) is the same one line the other way round.
        """
        if self.props:
            path = self.props if os.path.isabs(self.props) else graphs_data_path(self.props)
            if os.path.exists(path):
                d = json.load(open(path))
                amp = torch.tensor([d.get(str(k), {}).get("amp", float("nan"))
                                    for k in range(1, n_cells + 1)], device=device)
                good = torch.isfinite(amp)
                if good.any():
                    a = amp.clone()
                    a[~good] = a[good].median()
                    lo, hi = torch.quantile(a, 0.05), torch.quantile(a, 0.95)
                    u = ((a - lo) / (hi - lo + 1e-9)).clamp(0, 1)
                    return self.y_lo + (1.0 - u) * (self.y_hi - self.y_lo), "measured beat amplitude"
        g = torch.Generator(device="cpu").manual_seed(12345)
        u = torch.rand(n_cells, generator=g).to(device)
        return self.y_lo + u * (self.y_hi - self.y_lo), "deterministic spread (no props file)"

    def forward(self, H, mask=None):
        if self._done:
            return {}
        self._done = True
        lvl = H.level(self.at)
        fld = H.fields[self.field_name]
        dev = lvl.state.device
        px0, px1 = lvl.state_schema["pos"]
        n_cells = int(fld.n_labels)

        # ---- where each label lives, in world coordinates ---------------------------------
        gridl = fld.grid[0]                                     # [nx,ny] int64
        nx, ny = gridl.shape
        gx, gy = torch.meshgrid(torch.arange(nx, device=dev), torch.arange(ny, device=dev),
                                indexing="ij")
        flat = gridl.reshape(-1)
        wx = (gx.reshape(-1).double() + 0.5) / nx * fld.width
        wy = (gy.reshape(-1).double() + 0.5) / ny * fld.width
        inside = flat > 0
        lab_in, wx_in, wy_in = flat[inside], wx[inside], wy[inside]
        cnt = torch.bincount(lab_in, minlength=n_cells + 1).clamp(min=1)
        cx = torch.bincount(lab_in, weights=wx_in, minlength=n_cells + 1) / cnt
        cy = torch.bincount(lab_in, weights=wy_in, minlength=n_cells + 1) / cnt

        # ---- the CELL level moves onto its own segmented cell -----------------------------
        moved_cells = 0
        if self.cell_set in H.levels:
            cl = H.level(self.cell_set)
            cx0, cx1 = cl.state_schema["pos"]
            m = min(cl.n, n_cells)
            st = cl.state.clone()
            st[:m, cx0] = cx[1:m + 1].float()
            st[:m, cx0 + 1] = cy[1:m + 1].float()
            cl.state = st
            moved_cells = m
            if cl.n != n_cells:
                print(f"  [seed_from_segmentation] the {self.cell_set!r} set has {cl.n} entities "
                      f"and the map has {n_cells} cells -- seeding {m}. Declare "
                      f"per_parent: {n_cells} to use all of them.", flush=True)

        # ---- each particle is placed INSIDE its own cell's mask ---------------------------
        # ordering by label makes the members of one cell contiguous, so a particle can be given a
        # pixel of its OWN cell by index arithmetic instead of a python loop over 472 cells
        order = torch.argsort(lab_in)
        lab_s, wx_s, wy_s = lab_in[order], wx_in[order], wy_in[order]
        start = torch.cumsum(torch.bincount(lab_s, minlength=n_cells + 1), 0) - \
            torch.bincount(lab_s, minlength=n_cells + 1)

        pidx = lvl.parent if lvl.parent is not None else torch.zeros(lvl.n, dtype=torch.long,
                                                                     device=dev)
        pcell = (pidx % n_cells) + 1 if lvl.parent is not None else None
        if pcell is None or moved_cells == 0:
            # no declared cell level: assign each particle the label it already sits on
            pos = lvl.state[:, px0:px1]
            cid = fld.sample_label(pos)
        else:
            cid = pcell.clamp(1, n_cells)
            g = torch.Generator(device="cpu").manual_seed(777)
            u = torch.rand(lvl.n, generator=g).to(dev)
            k = (start[cid] + (u * cnt[cid].float()).long().clamp(max=0 + cnt[cid] - 1))
            k = k.clamp(0, lab_s.numel() - 1)
            newpos = torch.stack([wx_s[k].float(), wy_s[k].float()], 1)
            if self.jitter > 0:
                newpos = newpos + (torch.rand_like(newpos) - 0.5) * self.jitter
            st = lvl.state.clone(); st[:, px0:px1] = newpos; lvl.state = st

        # ---- one material per cell, shared exactly by its particles -----------------------
        yc, how = self._cell_values(n_cells, dev)
        y_all = torch.cat([yc[:1], yc])                          # index 0 = background, unused
        p_y = y_all[cid.clamp(0, n_cells)]
        from plexus.models.entities import _lame
        mu, la = _lame(p_y)
        liquid = getattr(lvl, "is_liquid", None)
        if liquid is not None:
            mu = torch.where(liquid, torch.zeros_like(mu), mu)
        lvl.mu, lvl.la = mu, la
        for nm, val in (("youngs", p_y), ("cell_id", cid.float())):
            if nm in getattr(lvl, "_buffers", {}):
                setattr(lvl, nm, val)
            else:
                lvl.register_buffer(nm, val)

        print(f"  [seed_from_segmentation] {n_cells} cells from {self.field_name!r}; "
              f"{lvl.n} particles ({lvl.n / max(n_cells,1):.0f} per cell); "
              f"youngs {float(yc.min()):.0f}-{float(yc.max()):.0f} from {how}; "
              f"cell centres seeded: {moved_cells}", flush=True)
        return {}


# --------------------------------------------------------------------------- material, re-stated
@register_operator("set_material", family="mpm", set="particle", kind="seed",
                   equation=r"""$$\mu=\frac{E}{2(1+\nu)},\qquad \lambda=\frac{E\nu}{(1+\nu)(1-2\nu)}$$""", title="set what the body is made of")
class SetMaterial(Seed):
    r"""re-states what a body is made of, once, at x_0, after a load and before the dynamics: the modulus, the material masks and, if given, the density and mass; volumes and the deformation history are untouched.
    particle -> particle: rewrites `mu`, `la`, the material masks and, with a density, `density` and `mass`, all in place; reads nothing.

        \mu=\frac{E}{2(1+\nu)},\qquad \lambda=\frac{E\nu}{(1+\nu)(1-2\nu)}
        \mu            the shear Lame parameter
        E              `youngs`, the modulus (or `bulk_modulus` K = lambda for a liquid), stress
        \nu            `nu`, the Poisson ratio
        \lambda        the first Lame parameter

    Reference: none: a restatement of material parameters, not a mechanism. Plexus (this work)."""
    PARAM_UNITS = {'bulk_modulus': 'stress', 'density': 'density', 'nu': 'dimensionless', 'tau': 'time', 'youngs': 'stress'}
    SYMBOLS = {'\\mu': 'the shear Lame parameter', 'E': '`youngs`, the modulus (or `bulk_modulus` K = lambda for a liquid), stress', '\\nu': '`nu`, the Poisson ratio', '\\lambda': 'the first Lame parameter'}

    EMIT = None                         # writes buffers at x_0; no integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["material", "initial_condition", "continuation"]
    PARAM_ROLES = {"youngs": "youngs_modulus", "bulk_modulus": "liquid_bulk_modulus",
                   "material": "constitutive_model", "tau": "maxwell_relaxation_time",
                   "density": "mass_density", "nu": "poisson_ratio"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.youngs = params.get("youngs")
        self.bulk = params.get("bulk_modulus")
        self.material = params.get("material")
        self.tau = params.get("tau")
        self.density = params.get("density")
        self.nu = float(params.get("nu", 0.2))
        if self.youngs is not None and self.bulk is not None:
            raise ValueError("set_material: give `youngs` OR `bulk_modulus`, not both -- for a "
                             "liquid they are two routes to the same lambda, and for a solid they "
                             "are different numbers.")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        mat = self.material
        if self.bulk is not None and mat not in (None, "liquid"):
            raise ValueError(
                f"set_material: `bulk_modulus` on material {mat!r}. It is defined for a liquid, "
                f"where mu = 0 makes K = lambda exactly; on a solid K = lambda + 2*mu/3 and "
                f"setting lambda from it would silently be the wrong number.")
        n = lvl.n
        said = []
        if self.youngs is not None or self.bulk is not None:
            if self.bulk is not None:
                mu = torch.zeros(n, device=lvl.state.device)
                la = torch.full((n,), float(self.bulk), device=lvl.state.device)
                said.append(f"K = {float(self.bulk):g} (liquid)")
            else:
                E = float(self.youngs)
                mu = torch.full((n,), E / (2 * (1 + self.nu)), device=lvl.state.device)
                la = torch.full((n,), E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu)),
                                device=lvl.state.device)
                said.append(f"youngs {E:g} -> mu {float(mu[0]):.4g}, la {float(la[0]):.4g}")
            if mat == "liquid" and self.bulk is None:
                mu = torch.zeros_like(mu)       # a fluid carries no shear
            lvl.mu.copy_(mu)
            lvl.la.copy_(la)
        if mat is not None:
            for nm, on in (("is_liquid", mat == "liquid"), ("is_snow", mat == "snow"),
                           ("is_visco", mat == "viscoelastic")):
                b = getattr(lvl, nm, None)
                if b is not None:
                    b.fill_(bool(on))
            if hasattr(lvl, "visco_tau"):
                # 1e9 is "no relaxation", the value `provision` uses for a purely elastic point
                lvl.visco_tau.fill_(float(self.tau) if (mat == "viscoelastic" and self.tau)
                                    else 1e9)
            said.append(f"material {mat}" + (f", tau {float(self.tau):g}"
                                             if mat == "viscoelastic" and self.tau else ""))
        if self.density is not None:
            rho = float(self.density)
            if hasattr(lvl, "density"):
                lvl.density.fill_(rho)
            if hasattr(lvl, "mass") and hasattr(lvl, "p_vol"):
                lvl.mass.copy_(lvl.p_vol * rho)     # the volumes are the geometry's; only rho moves
            said.append(f"density {rho:g}")
        print(f"[set_material] {lvl.name} ({n:,} points): " + "; ".join(said or ["nothing given"]),
              flush=True)
        return {}


@register_operator("set_material", family="mpm", set="particle", kind="seed", model="nucleus",
                   equation=r"""$$(\mu_i,\lambda_i)\leftarrow s\,(\mu_i,\lambda_i)\quad\text{for }\ |\mathbf x_i-\bar{\mathbf x}_{\pi(i)}-o\,\mathbf e_n|<r_n$$""", title="set the nucleus material")
class SetMaterialNucleus(SetMaterial):
    r"""re-states what a cell is made of inside its nucleus, once, at x_0: every material point within a radius of its cell's centre is made `stiffness` times stiffer than the cytoplasm around it, and marked.
    particle -[containment]-> cell: reads the points' positions and their parent cells; rewrites `mu` and `la` of the nuclear points in place and writes 1 / 0 into the width-1 `label` block of every point.

        (\mu_i,\lambda_i)\leftarrow s\,(\mu_i,\lambda_i)\quad\text{for }\ |\mathbf x_i-\bar{\mathbf x}_{\pi(i)}-o\,\mathbf e_n|<r_n
        \mu_i, \lambda_i the Lame parameters of point i
        s              `stiffness`, the nucleus-to-cytoplasm stiffness ratio
        x_i            the point's position
        \bar x_{\pi(i)} the centroid of its cell's points at x_0
        o              `offset`, along the axis
        e_n            the unit axis `axis` (default the sheet's normal)
        r_n            `radius`, or `radius_frac` times the cell's rms point radius

    Reference: Guilak, F., Tedrow, J. R. & Burgkart, R. (2000). Viscoelastic properties of the cell nucleus. Biochem. Biophys. Res. Commun. 269:781-786."""
    PARAM_UNITS = {'axis': 'count', 'offset': 'length', 'radius': 'length', 'radius_frac': 'fraction', 'stiffness': 'dimensionless'}
    SYMBOLS = {'\\mu_i, \\lambda_i': 'the Lame parameters of point i', 's': '`stiffness`, the nucleus-to-cytoplasm stiffness ratio', 'x_i': "the point's position", '\\bar x_{\\pi(i)}': "the centroid of its cell's points at x_0", 'o': '`offset`, along the axis', 'e_n': "the unit axis `axis` (default the sheet's normal)", 'r_n': "`radius`, or `radius_frac` times the cell's rms point radius"}
    REQUIRES_PARAMS = ["stiffness"]
    PARAM_ROLES = {"stiffness": "nucleus_youngs_over_cytoplasm", "radius": "nucleus_radius_world",
                   "radius_frac": "nucleus_radius_over_cell_rms_radius", "offset": "offset_along_axis_world",
                   "axis": "offset_axis", "label": "width_1_block_marking_nuclear_points", "cell_set": "parent_set"}
    MECHANISM_TAGS = ["material", "initial_condition", "nucleus", "heterogeneous_stiffness"]
    REFERENCE = "Guilak, F., Tedrow, J. R. & Burgkart, R. (2000). Biochem. Biophys. Res. Commun. 269:781-786."

    def __init__(self, params, device="cpu"):
        super().__init__({k: v for k, v in params.items()
                          if k not in ("stiffness", "radius", "radius_frac", "offset", "axis", "label", "cell_set")},
                         device)
        self.stiff = float(params["stiffness"])
        self.radius = params.get("radius")
        self.radius_frac = params.get("radius_frac")
        if (self.radius is None) == (self.radius_frac is None):
            raise ValueError("set_material[nucleus]: give exactly one of `radius` (world) or `radius_frac`")
        self.offset = self.tunable(params.get("offset"), 0.0, tensor=True)
        self.axis = int(params.get("axis", 2))
        self.label = str(params.get("label", "nuc"))
        self.cell_set = params.get("cell_set")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pname = self.cell_set or getattr(lvl, "parent_name", None)
        if pname is None:
            raise ValueError(f"set_material[nucleus]: {self.at!r} has no parent set; name it with `cell_set:`")
        X = lvl.get("pos").detach()
        idx = H.lift_index(self.at, pname).long()
        nc = int(H.level(pname).state.shape[0])
        live = (lvl.occ > 0.5).to(X.dtype)
        w = torch.zeros(nc, device=X.device, dtype=X.dtype).index_add_(0, idx, live)
        xb = torch.zeros(nc, X.shape[1], device=X.device, dtype=X.dtype).index_add_(0, idx, X * live[:, None])
        xb = xb / w.clamp_min(1.0)[:, None]
        c = xb[idx].clone()
        c[:, self.axis] = c[:, self.axis] + self.offset
        d = (X - c).norm(dim=1)
        if self.radius is not None:
            r = torch.full_like(d, float(self.radius))
        else:
            r2 = torch.zeros(nc, device=X.device, dtype=X.dtype).index_add_(0, idx, ((X - xb[idx]) ** 2).sum(1) * live)
            r = float(self.radius_frac) * (r2 / w.clamp_min(1.0)).sqrt()[idx]
        nuc = (d < r) & (live > 0)
        lvl.mu.copy_(torch.where(nuc, lvl.mu * self.stiff, lvl.mu))
        lvl.la.copy_(torch.where(nuc, lvl.la * self.stiff, lvl.la))
        if self.label in lvl.state_schema:
            a, b = lvl.state_schema[self.label]
            st = lvl.state.clone()
            st[:, a] = nuc.to(st.dtype)
            lvl.state = st
        n_nuc = int(nuc.sum()); n_live = int(live.sum())
        per = n_nuc / max(int((w > 0).sum()), 1)
        print(f"[set_material[nucleus]] {lvl.name}: {n_nuc:,} of {n_live:,} points nuclear ({per:.1f} per cell), "
              f"x{self.stiff:g} stiffer", flush=True)
        return {}


@register_operator("cell_nucleus", model="innermost", family="population", set="particle", kind="structural",
                   equation=r"""$$\nu_i=\mathbf 1\big[\mathrm{rank}_c\lvert\mathbf x_i-\bar{\mathbf x}_c\rvert<\lceil\phi\,n_c\rceil\big],\qquad(\mu_i,\lambda_i)\leftarrow s_c^{\,\nu_i^{\rm new}-\nu_i^{\rm old}}(\mu_i,\lambda_i)$$""",
                   title="keep each cell's nucleus the innermost share of its points")
class CellNucleusInnermost(Structural):
    r"""every cell holds one nucleus: the innermost share of its material points, `stiffness` times stiffer than the cytoplasm around them, marked in a point block; re-formed once per frame, so the nucleus grows with its cell (the share is of the points the cell has) and each daughter of a division gets its own nucleus at its own centre.
    particle -[containment]-> cell: reads the points' positions and parent cells (and, with `stiffness_block`, the cell's stiffness); rewrites `mu` and `la` of the points that join or leave the nucleus and writes 1 / 0 into the width-1 `label` block of every point.

        \nu_i=\mathbf 1\big[\mathrm{rank}_c\lvert\mathbf x_i-\bar{\mathbf x}_c\rvert<\lceil\phi\,n_c\rceil\big],\qquad(\mu_i,\lambda_i)\leftarrow s_c^{\,\nu_i^{\rm new}-\nu_i^{\rm old}}(\mu_i,\lambda_i)
        \nu_i          1 if point i is in its cell's nucleus, else 0
        \mathrm{rank}_c the rank of the point's distance to its cell's centroid among the cell's points, 0 the nearest
        \bar x_c       the centroid of cell c's live points
        \phi           `fraction`, the nucleus's share of its cell's points
        n_c            the number of live points of cell c
        s_c            `stiffness` (or the cell's value of `stiffness_block`), the nucleus-to-cytoplasm stiffness ratio
        \mu_i, \lambda_i the Lame parameters of point i

    Reference: Guilak, F., Tedrow, J. R. & Burgkart, R. (2000). Viscoelastic properties of the cell nucleus. Biochem. Biophys. Res. Commun. 269:781-786."""
    EXPLAIN = (r"Keeps one nucleus in every cell, made of material points. Each frame the points of a cell are ranked by their distance to the cell's centroid and the nearest $\lceil\phi n_c\rceil$ are the nucleus, $s_c$ times stiffer. "
               r"A cell that grows has more points and so a bigger nucleus; a cell that divides hands each daughter half the points, and each daughter's nucleus forms at its own centre.")
    PARAM_UNITS = {'fraction': 'fraction', 'stiffness': 'dimensionless'}
    SYMBOLS = {'\\nu_i': '1 if point i is in its cell\'s nucleus, else 0', '\\mathrm{rank}_c': "the rank of the point's distance to its cell's centroid among the cell's points, 0 the nearest", '\\bar x_c': "the centroid of cell c's live points", '\\phi': "`fraction`, the nucleus's share of its cell's points", 'n_c': 'the number of live points of cell c', 's_c': "`stiffness` (or the cell's value of `stiffness_block`), the nucleus-to-cytoplasm stiffness ratio", '\\mu_i, \\lambda_i': 'the Lame parameters of point i'}
    REQUIRES_PARAMS = []
    SUPPORTED_DIMS = [3]
    MECHANISM_TAGS = ["material", "nucleus", "heterogeneous_stiffness", "division"]
    PARAM_ROLES = {"fraction": "nucleus_share_of_the_cells_points", "fraction_block": "cell_block_with_each_cells_nucleus_share", "stiffness": "nucleus_youngs_over_cytoplasm", "stiffness_block": "cell_block_with_each_cells_stiffness_ratio",
                   "label": "width_1_point_block_marking_nuclear_points", "cell_set": "parent_set"}
    REFERENCE = "Guilak, F., Tedrow, J. R. & Burgkart, R. (2000). Biochem. Biophys. Res. Commun. 269:781-786."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "pt")
        self.frac = float(params.get("fraction", 0.4))
        self.stiff = float(params.get("stiffness", 4.0))
        self.stiff_block = params.get("stiffness_block")
        self.frac_block = params.get("fraction_block")                     # a cell block with each cell's own share, read instead of `fraction`
        self.label = str(params.get("label", "nuc"))
        self.cell_set = params.get("cell_set")
        self._calls = 0

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pname = self.cell_set or getattr(lvl, "parent_name", None)
        if pname is None:
            raise ValueError(f"cell_nucleus[innermost]: {self.at!r} has no parent set; name it with `cell_set:`")
        if self.label not in lvl.state_schema:
            raise ValueError(f"cell_nucleus[innermost]: the point set {self.at!r} needs a width-1 state block {self.label!r}")
        C = H.level(pname)
        X = lvl.get("pos").detach()
        nc = int(C.state.shape[0]); N = X.shape[0]
        idx = H.lift_index(self.at, pname).long().clamp(0, nc - 1)
        live = lvl.occ > 0.5
        ie = torch.where(live, idx, torch.full_like(idx, nc))                          # a dormant point belongs to no cell
        lf = live.to(X.dtype)
        w = torch.zeros(nc, device=X.device, dtype=X.dtype).index_add_(0, idx, lf)
        xb = torch.zeros(nc, X.shape[1], device=X.device, dtype=X.dtype).index_add_(0, idx, X * lf[:, None]) / w.clamp_min(1.0)[:, None]
        d = torch.where(live, (X - xb[idx]).norm(dim=1), torch.full((N,), float("inf"), device=X.device, dtype=X.dtype))
        o1 = torch.argsort(d)
        order = o1[torch.argsort(ie[o1], stable=True)]                                 # grouped by cell, the nearest to its centroid first
        cnt = torch.bincount(ie, minlength=nc + 1)
        start = torch.cumsum(cnt, 0) - cnt
        rank = torch.empty(N, dtype=torch.long, device=X.device)
        rank[order] = torch.arange(N, device=X.device) - start[ie[order]]
        frac = C.state[:, C.state_schema[self.frac_block][0]].detach().to(X.dtype) if self.frac_block else self.frac
        n_nuc = torch.ceil(frac * w).long()
        new = live & (rank < n_nuc[idx])
        a, _b = lvl.state_schema[self.label]
        old = lvl.state[:, a] > 0.5
        on, off = new & ~old, old & ~new & live
        if self.stiff_block:
            ca = C.state_schema[self.stiff_block][0]
            s_pt = C.state[:, ca].detach().to(X.dtype)[idx]
        else:
            s_pt = torch.full((N,), self.stiff, device=X.device, dtype=X.dtype)
        f = torch.ones(N, device=X.device, dtype=lvl.mu.dtype)
        f = torch.where(on, s_pt.to(f.dtype), f)
        f = torch.where(off, 1.0 / s_pt.to(f.dtype), f)
        lvl.mu = lvl.mu * f                                                            # new tensors, not in place: cell_populate rebinds these buffers the same way
        lvl.la = lvl.la * f
        st = lvl.state.clone(); st[:, a] = new.to(st.dtype); lvl.state = st
        if self._calls % 50 == 0:
            print(f"[cell_nucleus[innermost]] {lvl.name}: {int(new.sum()):,} of {int(live.sum()):,} points nuclear in {int((w > 0).sum())} cells ({int(on.sum())} joined, {int(off.sum())} left), " + (f"stiffness from the cell block {self.stiff_block!r}" if self.stiff_block else f"x{self.stiff:g} stiffer"), flush=True)
        self._calls += 1
        return {}


# ==========================================================================================================
# `implementation: differentiable` -- the same MPM cycle, written so autograd can keep the tape.
#
# WHY IT IS AN IMPLEMENTATION AND NOT A MODE. The physics is identical: every number these four
# compute is the number the default bodies compute, and the impl gate says so. What differs is where
# the results are PUT. The default bodies write in place, which is what a captured CUDA graph needs
# -- it holds the addresses it saw at capture -- and is exactly what autograd cannot have, since the
# tensor a later substep's backward needs has already been overwritten. Making that a `mode` (an
# ambient `torch.is_grad_enabled()`) would mean a run's numerics depended on the context it happened
# to be called in; making it an implementation means a specification asks for it BY NAME and a run
# can say which one it used, which is this repo's rule for every other alternative body.
#
# WHAT IT COSTS: one grid allocation and a handful of vector clones per substep. Measured on 12,500
# points over 12 frames, forward and backward, with `compile: true` on the substep block: 0.51 s an
# optimiser iteration against 1.01 s uncompiled, and 3.20 s for the same work at 50,000 points.
#
# WHAT IT IS FOR: deformation-gradient control, where the per-particle F is the control variable of
# an optimisation and the loss is taken on the grid's own NODAL MASS rather than on particle
# positions -- a target shape has no particle correspondence, so a position loss has nothing to
# match against exactly where the two shapes differ. `plexus.morph` is that loop: the optimiser,
# the two verified speed wins, and the renderer that makes a result judgeable.
#
# Reference: Xu, M., Song, C. Y., Levin, D. I. W. & Hyde, D. (2025). A differentiable material
# point method framework for shape morphing. IEEE TVCG 31(10):9140-9153 (arXiv:2409.15746);
# Xu, M. & Levin, D. I. W. (2023). Deformation gradient control of physically simulated amorphous
# solids. SCA '23, doi:10.1145/3606037.3606840. The render-guided variant is
# papers/Song_2025_physmorph_gs.pdf.


@register_operator("mpm_scatter", "p2g", implementation="differentiable", family="mpm",
                   set="particle", kind="exchange", title="spread point mass, differentiable", equation=r"""$$\boldsymbol\sigma = 2\mu(\mathbf F-\mathbf R)\mathbf F^{\!\top} + \lambda(J-1)J\,\mathbf I,
\qquad
\mathbf Q = -\frac{4\,\Delta t}{\Delta x^{2}}\,V\,\boldsymbol\sigma + m\,\mathbf C$$""")
class MPMScatterDiff(MPMScatter):
    r"""mpm_scatter with the grid rebuilt as new tensors rather than zeroed and re-accumulated in place, so each substep's grid survives for the backward (one grid allocation a substep); same arithmetic, same order.
    particle -> mpm_grid: as mpm_scatter.

        \boldsymbol\sigma = 2\mu(\mathbf F-\mathbf R)\mathbf F^{\!\top} + \lambda(J-1)J\,\mathbf I, \qquad \mathbf Q = -\frac{4\,\Delta t}{\Delta x^{2}}\,V\,\boldsymbol\sigma + m\,\mathbf C
        \sigma         the Kirchhoff stress of the fixed-corotated law
        \mu, \lambda   the particle's Lame parameters, stress
        F              the deformation gradient
        R              its rotation (polar decomposition F = R S; `polar`, `polar_iters`)
        J              det F, the volume ratio
        I              the identity
        Q              the affine momentum matrix deposited with the particle: the stress impulse plus the APIC term
        \Delta t       `dt_sub`, the substep (stable below dx / sqrt(E/rho))
        \Delta x       the grid spacing
        V              the particle's rest volume
        m              its mass
        C              its affine velocity gradient (APIC)

    Reference: as mpm_scatter."""
    PARAM_UNITS = {'a_max': 'acceleration', 'drag': 'rate', 'dt_sub': 'time', 'polar_iters': 'count', 'body_force': 'acceleration'}
    SYMBOLS = {'\\sigma': 'the Kirchhoff stress of the fixed-corotated law', '\\mu, \\lambda': "the particle's Lame parameters, stress", 'F': 'the deformation gradient', 'R': 'its rotation (polar decomposition F = R S; `polar`, `polar_iters`)', 'J': 'det F, the volume ratio', 'I': 'the identity', 'Q': 'the affine momentum matrix deposited with the particle: the stress impulse plus the APIC term', '\\Delta t': '`dt_sub`, the substep (stable below dx / sqrt(E/rho))', '\\Delta x': 'the grid spacing', 'V': "the particle's rest volume", 'm': 'its mass', 'C': 'its affine velocity gradient (APIC)'}
    FUNCTIONAL = True
    DIFFERENTIABLE = True

    def _const_read(self, t):
        # a seed may rewrite these buffers; a version counter that moves after this forward read is
        # reported by the backward against the line that READ it, which is not where the write was
        return t.clone()

    def _deposit(self, g, gm, gmv, gc, flat, weight, mass, mom, D, fresh, lw):
        gm = torch.zeros_like(g.m) if fresh else g.m
        gmv = torch.zeros_like(g.mv) if fresh else g.mv
        gc = torch.zeros_like(g.c) if fresh else g.c
        gm = gm.index_add(0, flat, (weight * mass[:, None]).reshape(-1))
        gmv = gmv.index_add(0, flat, (weight[..., None] * mom).reshape(-1, D))
        if lw is not None:
            gc = gc.index_add(0, flat, lw)
        g.m, g.mv, g.c = gm, gmv, gc


@register_operator("mpm_grid_update", implementation="differentiable", family="mpm", set="field",
                   kind="field", title="solve grid velocities, differentiable", equation=r"""$$\mathbf v_g = \frac{\mathbf p_g}{m_g},
\qquad
\mathbf v_g \mathrel{+}= \Delta t\,\mathbf f^{\,\text{CSF}}_g,
\qquad
\mathbf v_g \leftarrow \mathrm{BC}(\mathbf v_g)$$""")
class MPMGridUpdateDiff(MPMGridUpdate):
    r"""mpm_grid_update with the wall written as a stack rather than into a view, which autograd refuses; same arithmetic.
    mpm_grid -> mpm_grid: as mpm_grid_update.

        \mathbf v_g = \frac{\mathbf p_g}{m_g}, \qquad \mathbf v_g \mathrel{+}= \Delta t\,\mathbf f^{\,\text{CSF}}_g, \qquad \mathbf v_g \leftarrow \mathrm{BC}(\mathbf v_g)
        v_g            the velocity of grid node g
        p_g            its momentum
        m_g            its mass (nodes below `mass_floor`, a mass that scales with the particle mass, get zero)
        \Delta t       `dt_sub`, the substep
        f^{CSF}_g      the continuum surface force sigma kappa n from the mollified colour field (`cell_surface_tension`, `csf_*`)
        BC             the walls (`wall_damp` the restitution, `wall_friction`), the closing plates (`plate_*`), the obstacles and buoyancy (`buoyancy`, `rho_ref`)

    Reference: as mpm_grid_update."""
    PARAM_UNITS = {'buoyancy': 'acceleration', 'csf_band': 'fraction', 'csf_eps': 'dimensionless', 'csf_mass_floor': 'mass', 'csf_rho': 'density', 'csf_smooth': 'count', 'dt_sub': 'time', 'mass_floor': 'mass', 'plate_axis': 'count', 'plate_centre': 'length', 'plate_close_from': 'frames', 'plate_close_to': 'frames', 'plate_gap_half': 'length', 'plate_gap_half_end': 'length', 'rho_ref': 'density', 'cell_surface_tension': 'energy_per_area', 'wall_damp': 'fraction', 'wall_friction': 'fraction'}
    SYMBOLS = {'v_g': 'the velocity of grid node g', 'p_g': 'its momentum', 'm_g': 'its mass (nodes below `mass_floor`, a mass that scales with the particle mass, get zero)', '\\Delta t': '`dt_sub`, the substep', 'f^{CSF}_g': 'the continuum surface force sigma kappa n from the mollified colour field (`cell_surface_tension`, `csf_*`)', 'BC': 'the walls (`wall_damp` the restitution, `wall_friction`), the closing plates (`plate_*`), the obstacles and buoyancy (`buoyancy`, `rho_ref`)'}

    # `wall_friction` is imposed by the default path's wall loop only; this implementation has its
    # own wall code and does not read it yet. The base __init__ warns once if a spec asks.
    HONOURS_WALL_FRICTION = False
    FUNCTIONAL = True
    DIFFERENTIABLE = True

    def _const_read(self, t):
        return t.clone()

    def _write_v(self, g, gv):
        g.v = gv


@register_operator("mpm_strain", implementation="differentiable", family="mpm", set="particle",
                   kind="lateral", title="update material strain, differentiable", equation=r"""$$\mathbf F \;\leftarrow\; (\mathbf I + \Delta t\,\mathbf C)\,\mathbf F
\qquad\text{(liquid: } \mathbf F\leftarrow J^{1/d}\mathbf I,\ J=\det\mathbf F\text{)}$$""")
class MPMStrainDiff(MPMStrain):
    r"""mpm_strain with F (and Jp) rebound rather than written in place, so the control variable of an inverse run carries a graph; identical values.
    particle -> particle: as mpm_strain.

        \mathbf F \;\leftarrow\; (\mathbf I + \Delta t\,\mathbf C)\,\mathbf F \qquad\text{(liquid: } \mathbf F\leftarrow J^{1/d}\mathbf I,\ J=\det\mathbf F\text{)}
        F              the deformation gradient, the identity for unstrained material
        I              the identity
        \Delta t       `dt_sub`, the substep
        C              the affine velocity gradient
        J              det F, the ratio of current to rest volume (`liquid_volume`: carried through F, or integrated as dJ/dt = J tr C)
        d              the dimension

    Reference: as mpm_strain."""
    PARAM_UNITS = {'dt_sub': 'time'}
    SYMBOLS = {'F': 'the deformation gradient, the identity for unstrained material', 'I': 'the identity', '\\Delta t': '`dt_sub`, the substep', 'C': 'the affine velocity gradient', 'J': 'det F, the ratio of current to rest volume (`liquid_volume`: carried through F, or integrated as dJ/dt = J tr C)', 'd': 'the dimension'}
    FUNCTIONAL = True
    DIFFERENTIABLE = True

    def _const_read(self, t):
        return t.clone()

    def _write_F(self, p, F):
        p.F = F

    def _write_Jp(self, p, Jp):
        p.Jp = Jp

    def _visco_relax(self, Fv, a, D):
        """The same relaxation WITHOUT the SVD, whose backward is undefined when two stretches coincide -- and the relaxation
        drives them to coincide (exp30 smoke 4, 2026-10-23: the first NaN gradient came at the 8th window, once the cells had
        relaxed). With F = U diag(sig) Vh and R = U Vh the polar rotation,
            U diag(Jl + a (sig - Jl)) Vh  =  a F + (1 - a) Jl R,        Jl = det(F)^(1/D),
        an identity, so the values are the SVD form's; R comes from the Newton polar iteration (`_polar_higham`), matmuls only."""
        J = torch.linalg.det(Fv).clamp(min=1e-6)
        Jl = J.pow(1.0 / D)
        R = _polar_higham(Fv, int(getattr(self, "polar_iters", 6))) if D == 3 else Fv / Fv.det().clamp(min=1e-6).sqrt()[:, None, None]
        return a[:, None, None] * Fv + ((1.0 - a) * Jl)[:, None, None] * R


@register_operator("mpm_gather", "g2p", implementation="differentiable", family="mpm",
                   set="particle", kind="exchange", title="move points with the grid, differentiable", equation=r"""$$\mathbf v_i = \sum_g w_{ig}\,\mathbf v_g,
\qquad
\mathbf C_i = \frac{4}{\Delta x^{2}}\sum_g w_{ig}\,\mathbf v_g\,(\mathbf x_g-\mathbf x_i)^{\!\top},
\qquad
\mathbf x_i \mathrel{+}= \Delta t\,\mathbf v_i$$""")
class MPMGatherDiff(MPMGather):
    r"""mpm_gather writing a fresh state table instead of advecting in place: one clone of the particle state a substep, the price of keeping positions and velocities on the tape.
    mpm_grid -> particle: as mpm_gather.

        \mathbf v_i = \sum_g w_{ig}\,\mathbf v_g, \qquad \mathbf C_i = \frac{4}{\Delta x^{2}}\sum_g w_{ig}\,\mathbf v_g\,(\mathbf x_g-\mathbf x_i)^{\!\top}, \qquad \mathbf x_i \mathrel{+}= \Delta t\,\mathbf v_i
        v_i            the velocity of particle i
        w_{ig}         the quadratic B-spline weight of node g at particle i
        v_g            the velocity of node g
        C_i            the APIC affine velocity matrix (local velocity gradient) carried to the next scatter and to mpm_strain
        \Delta x       the grid spacing
        x_g, x_i       the node and particle positions
        \Delta t       `dt_sub`, the substep

    Reference: as mpm_gather."""
    PARAM_UNITS = {'dt_sub': 'time', 'vmax': 'velocity', 'wall_contact': 'length', 'wall_contact_cells': 'count', 'wall_damp': 'fraction'}
    SYMBOLS = {'v_i': 'the velocity of particle i', 'w_{ig}': 'the quadratic B-spline weight of node g at particle i', 'v_g': 'the velocity of node g', 'C_i': 'the APIC affine velocity matrix (local velocity gradient) carried to the next scatter and to mpm_strain', '\\Delta x': 'the grid spacing', 'x_g, x_i': 'the node and particle positions', '\\Delta t': '`dt_sub`, the substep'}
    FUNCTIONAL = True
    DIFFERENTIABLE = True

    def _const_read(self, t):
        return t.clone()

    def _write_particles(self, p, pa, pb, va, vb, Xn, new_V, new_C):
        S = p.state.clone()
        S[:, pa:pb] = Xn
        S[:, va:vb] = new_V
        p.state = S
        p.C = new_C


# ------------------------------------------------------------------------------------
# seeding the matrix
# ------------------------------------------------------------------------------------
# The matrix seed lives here rather than with the ECM forces: it lays out MPM material points, so
# it belongs beside the solver that integrates them, not beside the forces that read them.
@register_operator("ecm_seed", "seed_ecm", family="seed", set="particle", kind="seed", title="lay out the fibre matrix", equation=r"""$$\Omega=\text{box}\setminus\{\,r<r_c,\ |z-z_c|<h_c\,\},\qquad \mathbf x_{f,k}=\mathbf o_f+\frac{k}{K-1}\,\ell\,\hat{\mathbf d}_f+\boldsymbol\epsilon,\quad f=1\ldots N_f$$""")
class ECMSeed(Structural):
    r"""lays the matrix out once, at frame 0: the box minus a cavity, filled with aligned fibres of material points, never placed in the cavity rather than deleted from it.
    particle -> particle: rewrites every position in the set and the per-particle material at the opening of the trajectory.

        \Omega=\text{box}\setminus\{\,r<r_c,\ |z-z_c|<h_c\,\},\qquad \mathbf x_{f,k}=\mathbf o_f+\frac{k}{K-1}\,\ell\,\hat{\mathbf d}_f+\boldsymbol\epsilon,\quad f=1\ldots N_f
        \Omega         the occupied region: the box minus the cylindrical cavity
        r_c, h_c       `cavity_r`, `cavity_h`, the cavity's radius and half-height (or `cavity_sphere`)
        x_{f,k}        point k of fibre f
        o_f            the fibre's start, uniform in Omega
        \ell           `fibre_len`
        \hat d_f       the fibre's direction (`align`, `align_dir`, a cone of `dense_cone_deg`)
        \epsilon       `jitter`
        N_f            `n_fibres`

    Reference: Sulsky, D., Chen, Z. & Schreyer, H. L. (1994). A particle method for history-dependent materials. Comput. Methods Appl. Mech. Eng. 118:179-196."""
    PARAM_UNITS = {'cavity_h': 'length', 'cavity_r': 'length', 'cavity_sphere': 'length', 'centre': 'length', 'dense_boost': 'dimensionless', 'dense_cone_deg': 'angle', 'fibre_len': 'length', 'jitter': 'length', 'margin': 'length', 'n_fibres': 'count', 'plate_half': 'length', 'seed': 'count', 'shell_r': 'length', 'align': 'fraction', 'axis': 'count', 'dense_axis': 'count'}
    SYMBOLS = {'\\Omega': 'the occupied region: the box minus the cylindrical cavity', 'r_c, h_c': "`cavity_r`, `cavity_h`, the cavity's radius and half-height (or `cavity_sphere`)", 'x_{f,k}': 'point k of fibre f', 'o_f': "the fibre's start, uniform in Omega", '\\ell': '`fibre_len`', '\\hat d_f': "the fibre's direction (`align`, `align_dir`, a cone of `dense_cone_deg`)", '\\epsilon': '`jitter`', 'N_f': '`n_fibres`'}
    EMIT = None                       # rewrites positions in place at frame 0; no integrable delta
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["matrix_seeding", "fibre_alignment", "cavity"]
    PARAM_ROLES = {"cavity_r": "cavity_radius", "cavity_h": "cavity_half_height",
                   "fibre_len": "fibre_length", "n_fibres": "fibre_count",
                   "align": "fibre_alignment_strength", 'align_dir': 'direction_of_alignment', 'axis': 'axis_of_the_cavity', 'block': 'point_block_that_receives_the_fibre_id', 'cavity_sphere': 'radius_of_a_spherical_cavity', 'centre': 'centre_of_the_cavity', 'dense_axis': 'axis_of_the_dense_cone', 'dense_boost': 'density_boost_in_the_cone', 'dense_cone_deg': 'half_angle_of_the_dense_cone', 'jitter': 'point_jitter', 'margin': 'margin_kept_from_the_box_walls', 'plate_half': 'half_thickness_of_a_plate_layout', 'seed': 'random_seed', 'shell_r': 'radius_of_a_shell_layout'}
    REFERENCE = ("Sulsky, D., Chen, Z. & Schreyer, H. L. (1994). A particle method for "
                 "history-dependent materials. Comput. Methods Appl. Mech. Eng. 118:179-196.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        # THE CAVITY IS A DISC, and its two radii are separate numbers on purpose. A sphere would
        # confine the ball equally in every direction, which is the one case where the matrix
        # cannot tell us anything the ball's own mechanics does not: it would stay a ball. A disc
        # is anisotropic confinement -- free in two axes, pinched in the third -- so the shape the
        # tissue takes is a fact about the coupling rather than about the ball.
        self.cavity_r = float(params.get("cavity_r", 0.22))     # in-plane radius
        self.cavity_h = float(params.get("cavity_h", 0.07))     # half-thickness along `axis`
        self.axis = int(params.get("axis", 1))                  # the pinched axis
        self.margin = float(params.get("margin", 0.03))         # keep clear of the domain walls
        # SOLID BLOCKS, top and bottom: no matrix is seeded inside them. `None` = no blocks. The
        # blocks are the SAME object `plate_confine` holds the matrix out of during the run, so the
        # two numbers have to agree -- pass the same `gap_half` to both, from one place in the spec.
        # Seeding matrix into a solid and then relying on the confinement operator to evict it would
        # start the run with a shock the material never recovers from.
        self.plate_half = params.get("plate_half", None)
        self.plate_half = None if self.plate_half is None else float(self.plate_half)
        # A BLOCK REGION, in the framework's own vocabulary. `MPMParticle.provision` already accepts
        # `block: [x0,y0,z0,x1,y1,z1]` for a type that FILLS a box instead of a disc, and the stock MPM
        # demos build their falling cubes with it. Fibres cannot use that path -- the provision places
        # points and this operator places STRANDS -- so the same six numbers are accepted here and the
        # fibre CENTRES are drawn inside the box, inset by half a fibre so a strand stays in the cube it
        # belongs to. Without it the only regions this operator could seed were "the whole domain" and
        # "the whole domain minus a cavity", which is why a cube of matrix had to be faked with plates.
        self.block = params.get("block", None)
        self.block = None if self.block is None else [float(v) for v in self.block]
        self.cavity_sphere = bool(params.get("cavity_sphere", False))
        # A SHELL, WHICH IS THE ONLY AFFORDABLE GEOMETRY FOR A SPHEROID IN A MATRIX. Filling the box
        # at a usable particle density costs a million particles (ppc 4 at dx = 1/64 is 4*64^3 per
        # unit volume) and spends nearly all of them on a far field that never moves. `shell_r` caps
        # the radius a fibre CENTRE may be drawn at, so the matrix is the shell between the cavity
        # and it -- a strand may still stick out by half its length, which is why this is a cap on
        # centres and not a clamp on particles: clamping would pile mass on the outer surface.
        self.shell_r = params.get("shell_r", None)
        self.shell_r = None if self.shell_r is None else float(self.shell_r)
        # A DENSER REGION, so the matrix's ARCHITECTURE is anisotropic and not just its orientation.
        # Fibre alignment alone gave a 1.50x directional pressure difference (measured: 1448 along the
        # alignment axis against 2123 / 2176 across it) -- real, but small, because MPM interpolates
        # every particle onto a continuum grid and a fibrous ARRANGEMENT of an isotropic material
        # responds nearly isotropically. Putting MORE FIBRES in a region changes the mass and stiffness
        # the grid actually sees, which is a difference the continuum cannot average away.
        #
        # The region is the pair of cones about `dense_axis` within `dense_cone_deg` of it: dense at the
        # poles, sparse around the equator. Chosen so the suppression is AXISYMMETRIC and therefore
        # measurable by the semi-axes already recorded -- an off-centre dense blob makes a more striking
        # picture and needs a metric nobody has written.
        self.dense_axis = int(params.get("dense_axis", 2))
        self.dense_cone = float(params.get("dense_cone_deg", 0.0))     # 0 = uniform
        self.dense_boost = float(params.get("dense_boost", 1.0))
        self.n_fibres = int(params.get("n_fibres", 900))
        self.fibre_len = float(params.get("fibre_len", 0.16))
        # 0 = isotropic directions, 1 = every fibre parallel to `align_dir`. Anything between is a
        # von-Mises-like spread about it, which is what a real matrix looks like near a boundary.
        self.align = float(params.get("align", 0.0))
        self.align_dir = [float(v) for v in params.get("align_dir", [1.0, 0.0, 0.0])]
        self.jitter = float(params.get("jitter", 0.004))        # across-fibre thickness
        self.seed = int(params.get("seed", 0))
        self._done = False

    def _outside_cavity(self, pos):
        """True where a point is IN the matrix -- outside the cavity AND outside the solid blocks."""
        c = torch.tensor(self.centre, device=pos.device, dtype=pos.dtype)
        d = pos - c
        ax = self.axis
        along = d[:, ax].abs()
        radial = torch.sqrt((d ** 2).sum(1) - d[:, ax] ** 2 + 1e-12)
        # A DISC BY DEFAULT, A SPHERE ON REQUEST. The disc is two numbers because anisotropic
        # confinement is the experiment the cavity was written for; a hole that a SPHEROID will occupy
        # is a sphere, and setting r = h on the disc gives a cylinder with flat ends, which is a
        # different shape wearing the right radius.
        if self.cavity_sphere:
            ok = ~(d.norm(dim=1) < self.cavity_r)
        else:
            ok = ~((radial < self.cavity_r) & (along < self.cavity_h))
        if self.plate_half is not None:
            ok = ok & (d[:, ax].abs() < self.plate_half)
        return ok

    def forward(self, H, mask=None):
        if self._done:
            return {}
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        n, D = pos.shape
        dev, dt_ = pos.device, pos.dtype
        g = torch.Generator(device="cpu").manual_seed(self.seed)

        # FIBRE DIRECTIONS. Isotropic by default; `align` mixes toward a chosen axis. Normalised
        # AFTER mixing so a partially aligned fibre is still a unit vector.
        d = torch.randn(self.n_fibres, D, generator=g)
        d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-9)
        if self.align > 0:
            a = torch.tensor(self.align_dir, dtype=d.dtype)
            a = a / a.norm().clamp_min(1e-9)
            d = (1.0 - self.align) * d + self.align * a
            d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-9)

        # FIBRE CENTRES, REJECTION-SAMPLED OUT OF THE CAVITY. Rejection rather than a closed-form
        # placement because the cavity is a parameter: a formula would have to be rewritten for
        # every new cavity shape, and a rejection loop would not.
        # THE BOX IS THE WORLD, NOT [0, 1]. Every other length here -- the cavity centre, shell_r,
        # a block -- is already in world units, but the sampling region was written as the unit
        # cube. A tissue seeded in cvd2_adder_tension's world of 50 units with a margin of 1.0 then
        # got the region [1.0, 0.0]: empty, zero fibre centres, and a shape error two lines on.
        W = torch.as_tensor([float(w) for w in H.world_size], dtype=torch.float32)[:D]
        lo = torch.full((D,), float(self.margin))
        hi = W - float(self.margin)
        if self.block is not None:
            b = torch.tensor(self.block, dtype=torch.float32).reshape(2, D)
            inset = 0.5 * self.fibre_len
            blo = torch.maximum(torch.minimum(b[0] + inset, hi), lo)
            bhi = torch.maximum(torch.minimum(b[1] - inset, hi), lo)
            blo, bhi = torch.minimum(blo, bhi), torch.maximum(blo, bhi)
            # AND THE CAVITY STILL APPLIES INSIDE THE BLOCK. The hole is what the spheroid will
            # occupy, so it has to be absent from the matrix from frame 0 -- a fibre seeded where the
            # ball is going to be is in contact before the run starts, which is the defect the cavity
            # exists to prevent. Rejection, as everywhere else here, because the cavity's shape is a
            # parameter and a closed form would have to be rewritten for the next one.
            got = []
            for _ in range(64):
                c = torch.rand(self.n_fibres * 2, D, generator=g) * (bhi - blo) + blo
                c = c[self._outside_cavity(c.to(dev)).cpu()]
                got.append(c)
                if sum(x.shape[0] for x in got) >= self.n_fibres:
                    break
            centres = torch.cat(got)[: self.n_fibres]
            print(f"[ecm_seed] block region {self.block}: {centres.shape[0]} fibre centres inside it, "
                  f"inset by half a fibre ({inset:.3f}), cavity r={self.cavity_r:g} h={self.cavity_h:g} "
                  f"left empty", flush=True)
        else:
            centres = None
        keep = []
        for _ in range(64):
            c = torch.rand(self.n_fibres * 2, D, generator=g) * (hi - lo) + lo
            c = c[self._outside_cavity(c.to(dev)).cpu()]
            if self.shell_r is not None:
                c0 = torch.tensor(self.centre, dtype=c.dtype)
                c = c[(c - c0).norm(dim=1) < self.shell_r - 0.5 * self.fibre_len]
            keep.append(c)
            need = self.n_fibres * (max(self.dense_boost, 1.0) if self.dense_cone > 0 else 1.0)

            if sum(x.shape[0] for x in keep) >= need:
                break
        centres = torch.cat(keep) if centres is None else centres
        if self.block is None and self.dense_cone > 0 and self.dense_boost != 1.0:
            # IMPORTANCE SAMPLING, not a second sampling pass: keep every candidate in the cone and
            # thin the rest by 1/boost, so the ACCEPTED set has `boost` times the areal density inside
            # the cone. The particle count is unchanged -- `per = n // n_fibres` -- so this redistributes
            # material rather than adding it, and the sparse region really does get sparser.
            c0 = torch.tensor(self.centre, dtype=centres.dtype)
            v = centres - c0
            vn = v.norm(dim=1).clamp_min(1e-9)
            cosang = (v[:, self.dense_axis] / vn).abs()
            in_cone = cosang > math.cos(math.radians(self.dense_cone))
            u01 = torch.rand(centres.shape[0], generator=g)
            centres = centres[in_cone | (u01 < 1.0 / self.dense_boost)]
        centres = centres[: self.n_fibres]
        if centres.shape[0] < self.n_fibres:                  # cavity larger than the box
            centres = centres.repeat((self.n_fibres // max(centres.shape[0], 1)) + 1, 1)
            centres = centres[: self.n_fibres]

        # A STRAND CLEARS THE CAVITY AS A WHOLE, OR IT IS NOT SEEDED THERE. The per-particle eviction
        # below is a backstop and was doing this job: a strand whose CENTRE was outside the cavity but
        # whose ends crossed it had those ends teleported onto the rim while the rest stayed put, so
        # the strand was born already torn. Measured on `04c` before this: 64 of 10,000 strands had an
        # internal gap of up to seventeen times their own particle spacing at frame 0, and they are
        # the long straight streaks in the movie. The test is the distance from the cavity centre to
        # the SEGMENT, not to its midpoint, and a strand that fails it is pushed out along its own
        # centre's radius until it passes -- direction and length untouched, so the seeded fibre
        # architecture is unchanged.
        if self.cavity_sphere and self.cavity_r > 0:
            c0 = torch.tensor(self.centre, dtype=centres.dtype)
            half = 0.5 * self.fibre_len
            need_r = self.cavity_r + 2 * self.jitter
            for _ in range(8):
                w = c0 - centres
                t = (w * d).sum(1, keepdim=True).clamp(-half, half)
                dist = (w - t * d).norm(dim=1)
                bad = dist < need_r
                if not bool(bad.any()):
                    break
                u = centres[bad] - c0
                un = u / u.norm(dim=1, keepdim=True).clamp_min(1e-9)
                centres[bad] = centres[bad] + un * (need_r - dist[bad] + 1e-4)[:, None]
            print(f"[ecm_seed] {int(bad.sum())} strand(s) still crossing the cavity after "
                  f"whole-strand eviction", flush=True)

        # PARTICLES ALONG THE FIBRES, evenly spaced with a little across-fibre thickness so a
        # fibre reads as a strand rather than a line of dots.
        per = max(1, n // self.n_fibres)
        t = (torch.arange(per, dtype=torch.float32) / max(per - 1, 1) - 0.5) * self.fibre_len
        p = (centres[:, None, :] + t[None, :, None] * d[:, None, :]).reshape(-1, D)
        f = d[:, None, :].expand(-1, per, -1).reshape(-1, D)
        if p.shape[0] < n:                                    # top up with whole fibres
            k = n - p.shape[0]
            p = torch.cat([p, p[:k]])
            f = torch.cat([f, f[:k]])
        p, f = p[:n], f[:n]
        p = p + torch.randn(p.shape, generator=g) * self.jitter
        p = p.clamp(lo, hi)

        # A FIBRE THAT CROSSES THE CAVITY IS PULLED OUT OF IT. Seeding by centres alone leaves
        # particles inside the cavity whenever a fibre straddles the boundary -- and a single
        # matrix particle sitting where the cell ball is about to be is not a small error: it is
        # in contact from frame 0, so "the moment of first contact", which is the one event this
        # experiment exists to observe, would be frame 0 for the whole run.
        # INTO THE SLAB FIRST, THEN OUT OF THE CAVITY. A particle beyond a plate has to come back
        # along the plate normal; a particle in the cavity has to go out through its rim. Doing the
        # cavity push first and clamping afterwards can drop a particle straight back into the cavity.
        if self.plate_half is not None:
            lim = self.plate_half - self.jitter * 2
            p[:, self.axis] = (p[:, self.axis] - self.centre[self.axis]).clamp(-lim, lim) \
                + self.centre[self.axis]
        inside = ~self._outside_cavity(p.to(dev)).cpu()
        if inside.any():
            c = torch.tensor(self.centre, dtype=p.dtype)
            v = p[inside] - c
            rad = v.clone(); rad[:, self.axis] = 0
            rn = rad.norm(dim=1, keepdim=True).clamp_min(1e-9)
            push = torch.zeros_like(v)
            push += rad / rn * (self.cavity_r + self.jitter * 2)        # out through the rim
            push[:, self.axis] = torch.sign(v[:, self.axis]) * (self.cavity_h + self.jitter * 2)
            p[inside] = (c + push).clamp(lo, hi)

        # WRITTEN THROUGH THE VIEW. `Level.get` returns a slice of the state matrix, so an
        # in-place assignment IS the write; there is no setter, and `lvl.set(...)` does not exist.
        lvl.get("pos")[:] = p.to(dev, dt_)
        # THE FIBRE DIRECTIONS LIVE ON THE OPERATOR, not in the state matrix: the schema is fixed
        # at construction and has no `fibre` block. Keeping them here means an anisotropic term
        # can read them without a schema migration, and means they cannot be silently zeroed by
        # something that rewrites state.
        self.fibre = f.to(dev, dt_)
        self._done = True
        n_in = int((~self._outside_cavity(lvl.get("pos"))).sum())
        print(f"[ecm_seed] {n} particles on {self.n_fibres} fibres; cavity r={self.cavity_r} "
              f"h={self.cavity_h} about axis {self.axis}"
              + ("" if self.plate_half is None
                 else f"; solid blocks beyond +/-{self.plate_half:.3f} "
                      f"({100 * (1 - 2 * self.plate_half):.0f}% of the box)")
              + ("" if self.dense_cone <= 0 or self.dense_boost == 1.0
                 else f"; density x{self.dense_boost:g} within {self.dense_cone:g} deg of axis "
                      f"{self.dense_axis}")
              + f"; {n_in} left inside the cavity or a block", flush=True)
        return {}


# ============================================================================================
#  MPM BODIES AS CELLS: waking, sleeping and dividing a cell made of material points (exp 11)
# ============================================================================================
# A cell of `cell_set` owns a CONTIGUOUS block of `per` points of `point_set` (the containment map,
# built in parent order; a dormant cell's block is dormant). These three routines are the topology of
# such bodies: `wake_mpm_body` gives a free cell slot a fresh ball of material (a surface cell that
# dives into the lumen, `cell_die[implementation: t2]`'s `to_children`), `sleep_mpm_body` returns a
# cell's material to the pool (a daughter that re-enters the surface layer, `cell_divide[model:
# reinsert]`'s `from_children`), and `cell_divide[model: mpm]` cuts one body into two.

def mpm_blocks(H, cell_set, point_set):
    """[n_cells, per] indices of every cell's point block, and `per`."""
    nc = int(H.level(cell_set).state.shape[0])
    npnt = int(H.level(point_set).state.shape[0])
    if nc == 0 or npnt % nc:
        raise ValueError(f"{point_set!r} ({npnt} points) is not made of equal blocks for the {nc} slots of {cell_set!r}")
    per = npnt // nc
    return torch.arange(nc * per, device=H.level(point_set).state.device).view(nc, per), per


def _mpm_reset_points(P, idx, pos, vol):
    """Fresh material at `pos` [k, D]: zero velocity, F = I, C = 0, Jp = 1, p_vol `vol` [k], mass density x vol."""
    sch = P.state_schema
    st = P.state.clone()
    p0, p1 = sch["pos"]
    st[idx, p0:p1] = pos.to(st.dtype)
    if "vel" in sch:
        v0, v1 = sch["vel"]
        st[idx, v0:v1] = 0.0
    P.state = st
    D = p1 - p0
    if getattr(P, "F", None) is not None:
        P.F[idx] = torch.eye(D, device=P.F.device, dtype=P.F.dtype)
    if getattr(P, "C", None) is not None:
        P.C[idx] = 0.0
    if getattr(P, "Jp", None) is not None:
        P.Jp[idx] = 1.0
    if getattr(P, "p_vol", None) is not None:
        rho = (P.mass / P.p_vol.clamp_min(1e-30)) if getattr(P, "mass", None) is not None else None
        r = rho[idx] if rho is not None else None
        P.p_vol[idx] = vol.to(P.p_vol.dtype)
        if r is not None:
            P.mass[idx] = (r * vol.to(r.dtype)).to(P.mass.dtype)


def _mpm_label_points(P, idx, slot, pid_block, label_block):
    st = P.state.clone()
    if pid_block and pid_block in P.state_schema:
        st[idx, P.state_schema[pid_block][0]] = float(slot)
    if label_block and label_block in P.state_schema:
        st[idx, P.state_schema[label_block][0]] = float(int(slot) % 20)
    P.state = st


def wake_mpm_body(H, cell_set, point_set, slots, centres, volume, gen=None, pid_block="pid", label_block="cellc"):
    """Wake dormant cells `slots` of `cell_set` at `centres` [k, D], each with its whole point block as a
    uniform ball of `volume` (world units^3) about its centre. Returns the slots woken."""
    C, P = H.level(cell_set), H.level(point_set)
    blocks, per = mpm_blocks(H, cell_set, point_set)
    dev, dt = P.state.device, P.state.dtype
    D = P.state_schema["pos"][1] - P.state_schema["pos"][0]
    slots = [int(s) for s in slots]
    if not slots:
        return []
    cen = torch.as_tensor(np.asarray(centres, np.float64), device=dev, dtype=dt).reshape(len(slots), D)
    vol = torch.as_tensor(np.broadcast_to(np.asarray(volume, np.float64), (len(slots),)).copy(), device=dev, dtype=dt)
    R = (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)
    g = gen if gen is not None else torch.Generator(device="cpu").manual_seed(0)
    for j, s in enumerate(slots):
        u = torch.randn(per, D, generator=g, dtype=torch.float64)
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
        r = torch.rand(per, 1, generator=g, dtype=torch.float64) ** (1.0 / D)
        pos = cen[j] + (u * r).to(device=dev, dtype=dt) * R[j]
        idx = blocks[s]
        _mpm_reset_points(P, idx, pos, vol[j].expand(per) / per)
        _mpm_label_points(P, idx, s, pid_block, label_block)
        oc = P.occ.clone(); oc[idx] = 1.0; P.occ = oc
    sch = C.state_schema
    st, oc = C.state.clone(), C.occ.clone()
    c0, c1 = sch["pos"]
    st[slots, c0:c1] = cen
    if "vel" in sch:
        v0, v1 = sch["vel"]
        st[slots, v0:v1] = 0.0
    oc[slots] = 1.0
    C.state, C.occ = st, oc
    return slots


def sleep_mpm_body(H, cell_set, point_set, slots):
    """Return cells `slots`' point blocks to the dormant pool (occ 0, parked at the cell's position,
    velocity zero). The cell slots themselves are put to sleep by the caller (e.g. `Level.kill`)."""
    slots = [int(s) for s in slots]
    if not slots:
        return
    C, P = H.level(cell_set), H.level(point_set)
    blocks, per = mpm_blocks(H, cell_set, point_set)
    idx = blocks[slots].reshape(-1)
    c0, c1 = C.state_schema["pos"]
    park = C.state[slots, c0:c1].repeat_interleave(per, dim=0)
    st, oc = P.state.clone(), P.occ.clone()
    p0, p1 = P.state_schema["pos"]
    st[idx, p0:p1] = park.to(st.dtype)
    if "vel" in P.state_schema:
        v0, v1 = P.state_schema["vel"]
        st[idx, v0:v1] = 0.0
    oc[idx] = 0.0
    P.state, P.occ = st, oc


# ==========================================================================================================
# mpm_tear -- a solid that loses its stiffness, for good, where it is stretched past its breaking stretch.
# ==========================================================================================================
def _sym3_max_eig(A: torch.Tensor) -> torch.Tensor:
    """Largest eigenvalue of a batch of symmetric 3x3 matrices, in closed form (Smith 1961).

    Closed form rather than `torch.linalg.eigvalsh` because that is a cuSOLVER call: it cannot be
    captured into the CUDA graph the MPM substep runs as, and it costs a kernel per batch where this
    is a handful of fused element-wise ops."""
    a00, a11, a22 = A[:, 0, 0], A[:, 1, 1], A[:, 2, 2]
    a01, a02, a12 = A[:, 0, 1], A[:, 0, 2], A[:, 1, 2]
    p1 = a01 * a01 + a02 * a02 + a12 * a12
    q = (a00 + a11 + a22) / 3.0
    p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2.0 * p1
    p = torch.sqrt(p2 / 6.0).clamp_min(1e-12)
    b00, b11, b22 = (a00 - q) / p, (a11 - q) / p, (a22 - q) / p
    b01, b02, b12 = a01 / p, a02 / p, a12 / p
    detB = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02)
    r = (0.5 * detB).clamp(-1.0, 1.0)
    phi = torch.acos(r) / 3.0
    return q + 2.0 * p * torch.cos(phi)


@register_operator("mpm_tear", family="mpm", set="particle", kind="lateral", title="tear material past a breaking stretch",
                   equation=r"""$$d_p \leftarrow \max\!\big(d_p,\ [\lambda_{\max}(\mathbf F_p) > \lambda_b]\big),\qquad
\mu_p = (1-d_p)\,\mu_p^0,\quad \lambda_p = (1-d_p)\,\lambda_p^0$$""")
class MPMTear(Lateral):
    r"""a solid that tears: a point whose largest principal stretch passes the material's breaking stretch loses its stiffness for good, and the material on either side is free to pull apart (a pointwise criterion, a stated limitation against a tearing-energy one).
    particle -> particle: reads F; writes the Lame parameters mu, la, a per-point `torn` flag and, for a torn point, F = I; capturable, every write in place.

        d_p \leftarrow \max\!\big(d_p,\ [\lambda_{\max}(\mathbf F_p) > \lambda_b]\big),\qquad \mu_p = (1-d_p)\,\mu_p^0,\quad \lambda_p = (1-d_p)\,\lambda_p^0
        d_p            the damage flag of point p, 0 intact, 1 torn (it stays torn)
        \lambda_{\max}(F_p) the largest principal stretch, sqrt of the largest eigenvalue of F^T F
        \lambda_b      `stretch_break`, the stretch at which the material fails (latex: about 8)
        \mu_p, \lambda_p the point's Lame parameters
        \mu^0_p, \lambda^0_p their intact values

    Reference: Rivlin, R. S. & Thomas, A. G. (1953). Rupture of rubber. I. J. Polym. Sci. 10:291-318; Moulinet, S. & Adda-Bedia, M. (2015). Popping balloons: a case study of dynamical fragmentation. Phys. Rev. Lett. 115:184301."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"A solid that tears. A point whose largest principal stretch $\lambda_{\max}(\mathbf F)$ (the largest ratio of "
        r"current to rest length of the material around it, from the deformation gradient $\mathbf F$) passes the breaking "
        r"stretch $\lambda_b$ (`stretch_break`) is marked torn for good: both its Lamé stiffness parameters go to zero and "
        r"its $\mathbf F$ resets to the identity, so the material on either side of it is free to pull apart. The "
        r"criterion is pointwise, a stated limitation against a tearing-energy one."
    )
    PARAM_UNITS = {'stretch_break': 'dimensionless'}
    SYMBOLS = {'d_p': 'the damage flag of point p, 0 intact, 1 torn (it stays torn)', '\\lambda_{\\max}(F_p)': 'the largest principal stretch, sqrt of the largest eigenvalue of F^T F', '\\lambda_b': '`stretch_break`, the stretch at which the material fails (latex: about 8)', '\\mu_p, \\lambda_p': "the point's Lame parameters", '\\mu^0_p, \\lambda^0_p': 'their intact values'}

    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["stretch_break"]
    MECHANISM_TAGS = ["fracture", "damage", "tearing"]
    PARAM_ROLES = {"stretch_break": "principal_stretch_at_failure_relative_to_rest"}
    REFERENCE = ("Rivlin, R. S. & Thomas, A. G. (1953). J. Polym. Sci. 10:291-318; Moulinet, S. & "
                 "Adda-Bedia, M. (2015). Phys. Rev. Lett. 115:184301.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.stretch_break = float(params["stretch_break"])
        if self.stretch_break <= 1.0:
            raise ValueError(f"mpm_tear: stretch_break must exceed 1 (it is a stretch), got {self.stretch_break}")

    def forward(self, H, mask=None):
        p = H.level(self.at)
        if getattr(p, "torn", None) is None:
            p.register_buffer("torn", torch.zeros(p.n, device=p.F.device, dtype=p.F.dtype))
            p.register_buffer("mu_intact", p.mu.detach().clone())
            p.register_buffer("la_intact", p.la.detach().clone())
        F = p.F
        lmax2 = _sym3_max_eig(F.transpose(-2, -1) @ F)
        new = (lmax2 > self.stretch_break * self.stretch_break).to(F.dtype)
        occ = getattr(p, "occ", None)
        if occ is not None:
            new = new * (occ > 0).to(F.dtype)
        return self._damage(p, new)

    def _damage(self, p, new):
        """The damage write, shared with the `cut` model: `new` [N] in {0, 1} joins the torn set, the torn points lose
        their moduli and their strain (F reset to the identity: a torn point carries no memory of the load it broke under)."""
        F = p.F
        torn = torch.maximum(p.torn, new.to(F.dtype))
        p.torn.copy_(torn)
        keep = 1.0 - torn
        p.mu.copy_(p.mu_intact * keep.to(p.mu.dtype))
        p.la.copy_(p.la_intact * keep.to(p.la.dtype))
        eye = torch.eye(3, device=F.device, dtype=F.dtype).expand_as(F)
        p.F.copy_(torch.where(torn[:, None, None] > 0, eye, F))
        return {}


@register_operator("mpm_tear", model="cut", family="mpm", set="particle", kind="lateral", title="cut material along a plane",
                   equation=r"""$$d_p \leftarrow \max\!\big(d_p,\ [\,|x_{p,a}-c| \le w/2\,]\big),\qquad \mu_p=(1-d_p)\,\mu_p^0,\ \lambda_p=(1-d_p)\,\lambda_p^0$$""")
class MPMCut(MPMTear):
    r"""`cut` model of mpm_tear: the same damage written where a plane says, not where the material has stretched.
    particle -> particle: reads `pos`, writes `torn`, `mu`, `la`, `F` in place.

        d_p \leftarrow \max\!\big(d_p,\ [\,|x_{p,a}-c| \le w/2\,]\big),\qquad \mu_p=(1-d_p)\,\mu_p^0,\ \lambda_p=(1-d_p)\,\lambda_p^0
        d_p            the particle's damage, 0 or 1, never lowered
        x_{p,a}        its position along `axis`
        c              `plane_at`, the plane's position along that axis, in world units
        w              `thickness`, the slab torn around the plane, in world units
        \mu_p, \lambda_p the particle's Lame parameters, scaled to zero by the damage
        \mu^0_p, \lambda^0_p their undamaged values

    Reference: Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150 (damage in MLS-MPM); a prescribed plane as a blade. Plexus exp01 (this work)."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"A cut along a plane. Every point whose coordinate along `axis` lies within half of `thickness` of the plane at "
        r"`plane_at` (optionally only inside the box `within`) is marked torn for good, exactly as `mpm_tear` marks a "
        r"point past its breaking stretch: both its Lamé stiffness parameters go to zero and its deformation gradient "
        r"resets to the identity. A `thickness` of zero cuts a slab two grid cells wide."
    )
    PARAM_UNITS = {'axis': 'count', 'plane_at': 'length', 'thickness': 'length'}
    SYMBOLS = {'d_p': "the particle's damage, 0 or 1, never lowered", 'x_{p,a}': 'its position along `axis`', 'c': "`plane_at`, the plane's position along that axis, in world units", 'w': '`thickness`, the slab torn around the plane, in world units', '\\mu_p, \\lambda_p': "the particle's Lame parameters, scaled to zero by the damage", '\\mu^0_p, \\lambda^0_p': 'their undamaged values'}
    REQUIRES_PARAMS = ["axis", "plane_at"]                  # `at` is the set, as on every operator; the plane is `plane_at`
    PARAM_ROLES = {"axis": "plane_normal_axis", "plane_at": "plane_position", "thickness": "slab_width",
                   "within": "box_the_cut_is_limited_to"}

    def __init__(self, params, device="cpu"):
        params = dict(params); params.setdefault("stretch_break", 1e9)   # the tear's trigger, never reached here
        super().__init__(params, device)
        self.axis = int(params["axis"])
        self.at_plane = float(params["plane_at"])
        self.thickness = float(params.get("thickness", 0.0))
        self.within = [float(v) for v in params["within"]] if params.get("within") is not None else None
        if self.axis not in (0, 1, 2) or self.thickness < 0:
            raise ValueError("mpm_tear[cut]: axis in 0..2 and thickness >= 0")

    def forward(self, H, mask=None):
        p = H.level(self.at)
        if getattr(p, "torn", None) is None:
            p.register_buffer("torn", torch.zeros(p.n, device=p.F.device, dtype=p.F.dtype))
            p.register_buffer("mu_intact", p.mu.detach().clone())
            p.register_buffer("la_intact", p.la.detach().clone())
        x = p.get("pos")
        w = self.thickness
        if w <= 0.0:                                             # a cut thinner than a cell is a cell wide
            _g = getattr(H, "fields", None)
            w = 2.0 * float(_g["mpm_grid"].dx) if (_g is not None and "mpm_grid" in _g) else 0.0
        new = (x[:, self.axis] - self.at_plane).abs() <= 0.5 * w
        if self.within is not None:
            b = self.within
            new = new & (x[:, 0] >= b[0]) & (x[:, 1] >= b[1]) & (x[:, 2] >= b[2]) & (x[:, 0] <= b[3]) & (x[:, 1] <= b[4]) & (x[:, 2] <= b[5])
        occ = getattr(p, "occ", None)
        if occ is not None:
            new = new & (occ > 0)
        if mask is not None:
            new = new & mask.bool()
        print(f"[mpm_tear.cut] axis {self.axis} at {self.at_plane:g} width {w:g}: {int(new.sum()):,} points cut "
              f"({int(p.torn.sum()) if p.torn is not None else 0:,} torn before)", flush=True)
        return self._damage(p, new)


# ==========================================================================================================
# mpm_rigid_body -- a rigid projectile moving through the grid, coupled to the continuum both ways.
# ==========================================================================================================
@register_operator("mpm_rigid_body", family="mpm", set="particle", kind="exchange", title="move a rigid projectile through material",
                   equation=r"""$$\mathbf v_n \leftarrow \mathbf V + (\mathbf v_n-\mathbf V) - \min\!\big(0,(\mathbf v_n-\mathbf V)\cdot\hat{\mathbf n}\big)\hat{\mathbf n}\quad(\phi(\mathbf x_n)<0),
\qquad M\,\Delta\mathbf V = -\sum_n m_n\,\Delta\mathbf v_n$$""")
class MPMRigidBody(MPMWrites, Exchange):
    r"""a rigid body (a bullet) moving through the MPM grid: it pushes the continuum out of its way, lets it slide past and fall away behind, and loses exactly the momentum it hands over; it translates, it does not rotate.
    particle set (the body's own points) <-> mpm_grid: reads and writes the grid velocity in place; writes the body's points' `pos` and `vel` (a rigid translation); runs between mpm_grid_update and mpm_gather.

        \mathbf v_n \leftarrow \mathbf V + (\mathbf v_n-\mathbf V) - \min\!\big(0,(\mathbf v_n-\mathbf V)\cdot\hat{\mathbf n}\big)\hat{\mathbf n}\quad(\phi(\mathbf x_n)<0), \qquad M\,\Delta\mathbf V = -\sum_n m_n\,\Delta\mathbf v_n
        v_n            the velocity of grid node n inside the body
        V              the body's velocity
        \hat n         the capsule's outward normal at the node
        \phi(x_n)      the signed distance to the body (a capsule of radius `contact` and half-length along `axis`)
        M              `mass`, the body's mass
        \Delta V       the body's velocity change
        m_n, \Delta v_n the node's mass and velocity change

    Reference: Stomakhin, A. et al. (2013). A material point method for snow simulation. ACM Trans. Graph. 32(4):102 (collision objects on the grid); Hu, Y. et al. (2018). A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. ACM Trans. Graph. 37(4):150 (two-way rigid coupling)."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"A rigid body moving through the material. The body is a capsule (a sphere when its half-length is 0) of velocity "
        r"$\mathbf V$ that translates and does not rotate (`g`, when set, pulls it down). Each grid node inside it keeps "
        r"its velocity relative to the body except the part pointing into the body, so the material is pushed out of the "
        r"way and slides past (with `contact: stick`, and for nodes more than one cell deep, the node takes the body's "
        r"velocity outright). The body loses exactly the momentum it hands over, $M\,\Delta\mathbf V=-\sum_n "
        r"m_n\,\Delta\mathbf v_n$ ($M$ the body's mass, $m_n$ and $\Delta\mathbf v_n$ the mass and the velocity change of "
        r"grid node $n$)."
    )
    PARAM_UNITS = {'g': 'acceleration', 'mass': 'mass', 'velocity': 'velocity', 'dt_sub': 'time', 'axis': 'count', 'contact': 'length'}
    SYMBOLS = {'v_n': 'the velocity of grid node n inside the body', 'V': "the body's velocity", '\\hat n': "the capsule's outward normal at the node", '\\phi(x_n)': 'the signed distance to the body (a capsule of radius `contact` and half-length along `axis`)', 'M': "`mass`, the body's mass", '\\Delta V': "the body's velocity change", 'm_n, \\Delta v_n': "the node's mass and velocity change"}

    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["velocity"]
    MECHANISM_TAGS = ["rigid_body", "collider", "two_way_coupling", "projectile"]
    PARAM_ROLES = {"velocity": "launch_velocity", "mass": "body_mass_default_sum_of_point_masses",
                   "contact": "slip_or_stick", "axis": "capsule_axis_default_principal",
                   "g": "gravity_on_the_body_default_0", "from": "grid_field_or_list_of_them_the_body_pushes", 'dt_sub': 'substep'}
    REFERENCE = ("Stomakhin, A. et al. (2013). ACM Trans. Graph. 32(4):102; Hu, Y. et al. (2018). "
                 "ACM Trans. Graph. 37(4):150.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "bullet")
        # ONE BODY, SEVERAL GRIDS. With a material on its own grid (a membrane beside a liquid, coupled
        # by `mpm_grid_contact`) the bullet must push both, and its momentum pays for both: `from:` is
        # one grid name or a list of them.
        _g = params.get("from") or "mpm_grid"
        self.grids = [str(g) for g in (_g if isinstance(_g, list) else [_g])]
        self.v0 = [float(v) for v in params["velocity"]]
        self.mass = params.get("mass")
        self.contact = str(params.get("contact", "slip")).lower()
        if self.contact not in ("slip", "stick"):
            raise ValueError(f"mpm_rigid_body: contact must be slip or stick, got {self.contact!r}")
        self.axis = params.get("axis")
        self.g = float(params.get("g", 0.0))
        self.dt_sub = float(params.get("dt_sub", 2e-4))

    def _init(self, p, g, dev):
        a, b = p.state_schema["pos"]
        live = p.occ > 0
        X = p.state[:, a:b].double()
        c = X[live].mean(0)
        off = (X - c)
        if self.axis is not None:
            ax = torch.as_tensor([float(v) for v in self.axis], device=dev, dtype=torch.float64)
        else:
            cov = off[live].T @ off[live] / max(int(live.sum()), 1)
            ax = torch.linalg.eigh(cov).eigenvectors[:, -1]
        ax = ax / ax.norm()
        al = off[live] @ ax
        rad = (off[live] - al[:, None] * ax).norm(dim=1).max()
        half = (al.abs().max() - rad).clamp_min(0.0)
        M = float(self.mass) if self.mass is not None else float(p.mass[live].sum()) if hasattr(p, "mass") else None
        if not M or M <= 0:
            raise ValueError("mpm_rigid_body: the body has no mass -- give `mass` or a particle_mass")
        f = torch.float32
        p.register_buffer("rb_x", c.to(f).clone())
        p.register_buffer("rb_v", torch.as_tensor(self.v0, device=dev, dtype=f).clone())
        p.register_buffer("rb_off", off.to(f).contiguous())
        p.register_buffer("rb_impulse", torch.zeros(3, device=dev, dtype=f))
        self._ax = ax.to(f)
        self._r, self._h, self._M = float(rad), float(half), M
        dx = float(g.dx)
        self._ext = self._h + self._r + 2.0 * dx
        K = int(math.ceil(2.0 * self._ext / dx)) + 2
        rng = torch.arange(K, device=dev)
        self._stencil = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
        self._shape = torch.as_tensor(list(g.shape), device=dev)
        self._box = torch.as_tensor([s * dx for s in g.shape], device=dev, dtype=f)
        print(f"[mpm_rigid_body] {self.at}: capsule radius {self._r:.4g}, half-length {self._h:.4g}, axis "
              f"{[round(float(v), 3) for v in ax]}, mass {M:.4g}, launch {self.v0}, contact {self.contact}, "
              f"{K ** 3:,} grid nodes in its stencil", flush=True)

    def forward(self, H, mask=None):
        p = H.level(self.at); grids = [H.field(t) for t in self.grids]; g = grids[0]; dev = g.v.device
        for _g in grids:
            _no_origin(self, _g)
            if tuple(_g.shape) != tuple(g.shape) or float(_g.dx) != float(g.dx):
                raise ValueError(f"mpm_rigid_body: grids {self.grids} differ in shape or cell size")
        if getattr(p, "rb_x", None) is None:
            self._init(p, g, dev)
        dt = sub_dt(H, self.dt_sub)
        dx = float(g.dx)
        X, V = p.rb_x, p.rb_v
        base = torch.floor((X - self._ext) / dx).long()
        ijk = base[None, :] + self._stencil
        valid = ((ijk >= 0) & (ijk < self._shape[None, :])).all(1)
        ijkc = torch.minimum(ijk.clamp_min(0), self._shape[None, :] - 1)
        flat = (ijkc[:, 0] * self._shape[1] + ijkc[:, 1]) * self._shape[2] + ijkc[:, 2]
        xn = ijk.to(X.dtype) * dx
        rel = xn - X[None, :]
        al = (rel * self._ax).sum(1)
        alc = al.clamp(-self._h, self._h)
        radial = rel - alc[:, None] * self._ax[None, :]
        dist = radial.norm(dim=1)
        phi = dist - self._r
        nhat = radial / dist.clamp_min(1e-6 * dx)[:, None]
        active = ((X > self._ext) & (X < self._box - self._ext)).all().to(X.dtype)
        imp = torch.zeros(3, device=dev, dtype=X.dtype)
        for g in grids:
            mn = g.m.index_select(0, flat)
            vn = g.v.index_select(0, flat)
            inside = (valid & (phi < 0) & (mn > 0)).to(X.dtype) * active
            u = vn - V[None, :]
            if self.contact == "stick":
                u_new = torch.zeros_like(u)
            else:
                un = (u * nhat).sum(1, keepdim=True)
                u_new = u - un.clamp(max=0.0) * nhat
                u_new = torch.where((phi < -dx)[:, None], torch.zeros_like(u_new), u_new)
            dv = (V[None, :] + u_new - vn) * inside[:, None]
            g.v.index_add_(0, flat, dv)
            imp = imp + (mn[:, None] * dv).sum(0)
        p.rb_impulse.add_(imp)
        V.sub_(imp / self._M)
        if self.g != 0.0:
            V[1:2].sub_(self.g * dt)
        X.add_(V * dt)
        a, b = p.state_schema["pos"]
        p.state[:, a:b].copy_((X[None, :] + p.rb_off).to(p.state.dtype))
        if "vel" in p.state_schema:
            v0, v1 = p.state_schema["vel"]
            p.state[:, v0:v1].copy_(V[None, :].expand(p.state.shape[0], 3).to(p.state.dtype))
        return {}


# ==========================================================================================================
# mpm_sponge -- an absorbing band at chosen walls: material that reaches it stops instead of bouncing back.
# ==========================================================================================================
@register_operator("mpm_sponge", family="mpm", set="field", kind="field", title="absorb waves at the walls",
                   equation=r"""$$\mathbf v_n \leftarrow \mathbf v_n\,\min\!\Big(1,\ \min_{\text{faces}} \frac{d_n}{w}\Big)$$""")
class MPMSponge(MPMWrites, FieldUpdate):
    r"""an open boundary for the grid: within `width` of each named face the grid velocity is scaled down in proportion to the distance from the wall, so material that flies into the band comes to rest instead of rebounding into the scene.
    mpm_grid -> mpm_grid: reads and writes the grid velocity in place (the factor is built once; capturable).

        \mathbf v_n \leftarrow \mathbf v_n\,\min\!\Big(1,\ \min_{\text{faces}} \frac{d_n}{w}\Big)
        v_n            the velocity of grid node n (`normal_only`: only the component into the face)
        d_n            the node's distance to the nearest named face
        w              `width`, the band's width, world length

    Reference: Israeli, M. & Orszag, S. A. (1981). Approximation of radiation boundary conditions. J. Comput. Phys. 41:115-135 (the absorbing sponge layer)."""
    PARAM_UNITS = {'width': 'length'}
    SYMBOLS = {'v_n': 'the velocity of grid node n (`normal_only`: only the component into the face)', 'd_n': "the node's distance to the nearest named face", 'w': "`width`, the band's width, world length"}

    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["faces", "width"]
    MECHANISM_TAGS = ["open_boundary", "absorbing_layer"]
    PARAM_ROLES = {"faces": "which_walls_absorb", "width": "band_width",
                   "normal_only": "damp_only_the_component_into_the_wall"}
    REFERENCE = "Israeli, M. & Orszag, S. A. (1981). J. Comput. Phys. 41:115-135."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_grid")
        self.faces = [str(f).strip().lower() for f in params["faces"]]
        for f in self.faces:
            if f.lstrip("+-") not in ("x", "y", "z") or f[0] not in "+-":
                raise ValueError(f"mpm_sponge: face {f!r} must be one of +x -x +y -y +z -z")
        self.width = float(params["width"])
        self.normal_only = bool(params.get("normal_only", False))
        self._fac = None
        self._per = None

    def forward(self, H, mask=None):
        g = H.field(self.at)
        _no_origin(self, g)
        if self._fac is None or self._fac.shape[0] != g.n_cells:
            dev = g.v.device
            D = g.dim
            idx = [torch.arange(s, device=dev, dtype=torch.float32) * g.dx for s in g.shape]
            grids = torch.meshgrid(*idx, indexing="ij")
            fac = torch.ones(g.shape, device=dev)
            for f in self.faces:
                ax = "xyz".index(f[1])
                if ax >= D:
                    continue
                L = (g.shape[ax] - 1) * g.dx
                d = (L - grids[ax]) if f[0] == "+" else grids[ax]
                fac = torch.minimum(fac, (d / self.width).clamp(0.0, 1.0))
            self._fac = fac.reshape(-1, 1).contiguous()
            per = []
            for f in self.faces:                                  # one band per face, for normal_only
                ax = "xyz".index(f[1])
                if ax >= D:
                    continue
                L = (g.shape[ax] - 1) * g.dx
                d = (L - grids[ax]) if f[0] == "+" else grids[ax]
                per.append((ax, 1.0 if f[0] == "+" else -1.0, (d / self.width).clamp(0.0, 1.0).reshape(-1).contiguous()))
            self._per = per
        if not self.normal_only:
            g.v.mul_(self._fac)
            return {}
        for ax, sgn, fac in self._per:
            c = g.v[:, ax]
            g.v[:, ax] = torch.where(c * sgn > 0, c * fac, c)
        return {}



# ==========================================================================================================
# mpm_grid_contact -- two materials on two grids, touching without sticking.
# ==========================================================================================================
@register_operator("mpm_grid_contact", family="mpm", set="field", kind="field", title="let two materials push without sticking",
                   equation=r"""$$\delta=(\mathbf v_2-\mathbf v_1)\cdot\hat{\mathbf n},\ \ \hat{\mathbf n}=-\frac{\nabla m_1}{|\nabla m_1|};\quad
\delta<0:\ \ \mathbf v_1 \mathrel{+}= \frac{m_2}{m_1+m_2}\,\delta\,\hat{\mathbf n},\ \ \mathbf v_2 \mathrel{-}= \frac{m_1}{m_1+m_2}\,\delta\,\hat{\mathbf n}$$""")
class MPMGridContact(MPMWrites, FieldUpdate):
    r"""two materials, each on its own MPM grid, that push on each other where they meet but are free to slide along and separate from each other: frictionless multi-velocity-field contact, momentum conserved.
    mpm_grid (body 1) + mpm_grid (body 2) -> both: reads both grids' mass and velocity; writes both velocities in place; runs after both grid solves and before the gathers.

        \delta=(\mathbf v_2-\mathbf v_1)\cdot\hat{\mathbf n},\ \ \hat{\mathbf n}=-\frac{\nabla m_1}{|\nabla m_1|};\quad \delta<0:\ \ \mathbf v_1 \mathrel{+}= \frac{m_2}{m_1+m_2}\,\delta\,\hat{\mathbf n},\ \ \mathbf v_2 \mathrel{-}= \frac{m_1}{m_1+m_2}\,\delta\,\hat{\mathbf n}
        \delta         the speed at which body 2 closes on body 1 along the normal (acts only where negative)
        v_1, v_2       the two grids' node velocities
        \hat n         body 1's outward normal, from its own mass field (raw, or blurred over `normal_blur` cells with `normal: smoothed`)
        m_1, m_2       the two grids' node masses (nodes below `mass_eps` are empty)

    Reference: Bardenhagen, S. G., Brackbill, J. U. & Sulsky, D. (2000). The material-point method for granular materials. Comput. Methods Appl. Mech. Eng. 187:529-541."""
    PARAM_UNITS = {'mass_eps': 'mass', 'normal_blur': 'count'}
    SYMBOLS = {'\\delta': 'the speed at which body 2 closes on body 1 along the normal (acts only where negative)', 'v_1, v_2': "the two grids' node velocities", '\\hat n': "body 1's outward normal, from its own mass field (raw, or blurred over `normal_blur` cells with `normal: smoothed`)", 'm_1, m_2': "the two grids' node masses (nodes below `mass_eps` are empty)"}

    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["other"]
    MECHANISM_TAGS = ["contact", "multi_field", "frictionless", "separation"]
    PARAM_ROLES = {"other": "second_grid_body_2", "mass_eps": "least_node_mass_counted_as_present",
                   "normal": "body1_or_smoothed_contact_normal",
                   "normal_blur": "box_filter_passes_for_the_smoothed_normal"}
    REFERENCE = "Bardenhagen, S. G., Brackbill, J. U. & Sulsky, D. (2000). Comput. Methods Appl. Mech. Eng. 187:529-541."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_grid")
        self.other = str(params["other"])
        self.mass_eps = float(params.get("mass_eps", 0.0))
        self.normal = str(params.get("normal", "body1")).lower()
        if self.normal not in ("body1", "smoothed"):
            raise ValueError(f"mpm_grid_contact: normal must be body1 or smoothed, got {self.normal!r}")
        self.normal_blur = int(params.get("normal_blur", 3))

    @staticmethod
    def _grad(m):
        nx, ny, nz = m.shape
        gr = torch.zeros(nx, ny, nz, 3, device=m.device, dtype=m.dtype)
        gr[1:-1, :, :, 0] = m[2:] - m[:-2]
        gr[:, 1:-1, :, 1] = m[:, 2:] - m[:, :-2]
        gr[:, :, 1:-1, 2] = m[:, :, 2:] - m[:, :, :-2]
        return gr.view(-1, 3)

    def forward(self, H, mask=None):
        g1, g2 = H.field(self.at), H.field(self.other)
        _no_origin(self, g1)
        if tuple(g1.shape) != tuple(g2.shape):
            raise ValueError(f"mpm_grid_contact: {self.at} {tuple(g1.shape)} and {self.other} {tuple(g2.shape)} differ")
        nx, ny, nz = g1.shape
        m1 = g1.m[: nx * ny * nz].view(nx, ny, nz)
        if self.normal == "smoothed":
            mb = m1[None, None]
            for _ in range(self.normal_blur):
                mb = torch.nn.functional.avg_pool3d(mb, 3, stride=1, padding=1, count_include_pad=False)
            gr = -self._grad(mb[0, 0])
        else:
            gr = -self._grad(m1)
        n = gr / gr.norm(dim=1, keepdim=True).clamp_min(1e-30)
        a, b = g1.m.view(-1)[: n.shape[0]], g2.m.view(-1)[: n.shape[0]]
        v1, v2 = g1.v[: n.shape[0]], g2.v[: n.shape[0]]
        delta = ((v2 - v1) * n).sum(1)
        tot = (a + b).clamp_min(1e-30)
        hit = (a > self.mass_eps) & (b > self.mass_eps) & (delta < 0)
        d = torch.where(hit, delta, torch.zeros_like(delta))[:, None] * n
        v1.add_((b / tot)[:, None] * d)
        v2.sub_((a / tot)[:, None] * d)
        return {}


# ==========================================================================================================
# mpm_grid_noise -- a coherent (Perlin) body force on the grid, per region its own frequency.
# ==========================================================================================================
# Ken Perlin's permutation table (SIGGRAPH 1985; the reference implementation of 2002), the hash of a lattice
# point to one of the gradient directions. Any permutation of 0..255 works; this one is kept so a seed means the
# same field here as in every other implementation.
_PERLIN_PERM = np.array([
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21,
    10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237,
    149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111,
    229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73,
    209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52,
    217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189,
    28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39,
    253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144,
    12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204,
    176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66,
    215, 61, 156, 180], dtype=np.int64)
# The 32 gradient directions of 4-D Perlin noise: every (+-1, +-1, +-1, 0) with the zero in each of the four slots.
_PERLIN_GRAD4 = np.array([[a, b, c, 0] for a in (1, -1) for b in (1, -1) for c in (1, -1)]
                         + [[a, b, 0, c] for a in (1, -1) for b in (1, -1) for c in (1, -1)]
                         + [[a, 0, b, c] for a in (1, -1) for b in (1, -1) for c in (1, -1)]
                         + [[0, a, b, c] for a in (1, -1) for b in (1, -1) for c in (1, -1)], dtype=np.float32)


_PERLIN_CORNERS = np.array([[(c >> 0) & 1, (c >> 1) & 1, (c >> 2) & 1, (c >> 3) & 1] for c in range(16)], dtype=np.int64)


def perlin4(p, perm, grad, corners=None):
    """Classic 4-D Perlin noise at the points `p` [N, 4] (lattice units: the lattice spacing is 1), in about
    [-1, 1], zero at every lattice point. `perm` is the permutation table repeated twice [512], `grad` the 32
    gradients [32, 4] and `corners` the 16 hypercell corner offsets [16, 4] (built here when not given), all
    tensors on `p`'s device -- NOTHING IS CREATED FROM HOST MEMORY INSIDE: a `torch.tensor(...)` per corner is a
    host-to-device copy, which invalidates a CUDA graph capture (the first run of exp01 step 0004 died with
    cudaErrorStreamCaptureInvalidated at frame 0). The 16 corners each contribute the dot product of their gradient
    with the offset to the point, blended by Perlin's quintic fade 6t^5 - 15t^4 + 10t^3 (zero first and second
    derivative at the corners)."""
    if corners is None:
        corners = torch.as_tensor(_PERLIN_CORNERS, device=p.device)
    i0 = torch.floor(p)
    f = p - i0
    i0 = i0.to(torch.long) & 255
    u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
    n = None
    acc = {}
    for c in range(16):
        d = corners[c]
        ii = (i0 + d) & 255
        h = perm[perm[perm[perm[ii[:, 0]] + ii[:, 1]] + ii[:, 2]] + ii[:, 3]] & 31
        g = grad[h]                                              # [N, 4]
        acc[c] = (g * (f - d.to(f.dtype))).sum(-1)
    # blend the 16 corners: x first, then y, z, w
    def _lerp(a, b, t):
        return a + t * (b - a)
    x = {c: _lerp(acc[c], acc[c | 1], u[:, 0]) for c in range(16) if not c & 1}
    y = {c: _lerp(x[c], x[c | 2], u[:, 1]) for c in x if not c & 2}
    z = {c: _lerp(y[c], y[c | 4], u[:, 2]) for c in y if not c & 4}
    n = _lerp(z[0], z[8], u[:, 3])
    return n


@register_operator("mpm_grid_noise", family="mpm", set="field", kind="field", title="stir the grid with coherent noise",
                   equation=r"""$$\mathbf v_n \leftarrow \mathbf v_n + \Delta t\,A\,\mathbf N\!\big(f\,\mathbf x_n,\ s\,t\big)\qquad \mathbf N=(N_1,N_2,N_3),\ N_k\ \text{Perlin noise}$$""")
class MPMGridNoise(MPMWrites, FieldUpdate):
    r"""a body force that varies smoothly in space and time, as Perlin noise does.
    mpm_grid -> mpm_grid: reads and writes the grid velocity in place, on loaded nodes only (m_n > 0).

        \mathbf v_n \leftarrow \mathbf v_n + \Delta t\,A\,\mathbf N\!\big(f\,\mathbf x_n,\ s\,t\big)\qquad \mathbf N=(N_1,N_2,N_3),\ N_k\ \text{Perlin noise}
        v_n            the grid node's velocity
        \Delta t       the substep length
        A              `amplitude`, an acceleration in length per time squared
        N_k            three independent Perlin noise fields, k = 1..3
        f              `frequency`, cycles per world length (a list per region with `regions`)
        x_n            the node position
        s              `speed`, the noise's own time rate
        t              the time

    Reference: Perlin, K. (1985). An image synthesizer. ACM SIGGRAPH Computer Graphics 19(3):287-296; Perlin, K. (2002). Improving noise. ACM Trans. Graph. 21(3):681-682."""
    PARAM_UNITS = {'amplitude': 'acceleration', 'frequency': 'dimensionless', 'frequencies': 'dimensionless', 'speed': 'dimensionless', 'samples': 'count', 'seed': 'count'}
    SYMBOLS = {'v_n': "the grid node's velocity", '\\Delta t': 'the substep length', 'A': '`amplitude`, an acceleration in length per time squared', 'N_k': 'three independent Perlin noise fields, k = 1..3', 'f': '`frequency`, cycles per world length (a list per region with `regions`)', 'x_n': 'the node position', 's': "`speed`, the noise's own time rate", 't': 'the time'}

    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["amplitude"]
    MECHANISM_TAGS = ["body_force", "coherent_noise", "stirring"]
    PARAM_ROLES = {"amplitude": "acceleration_scale", "frequency": "noise_cycles_per_length", "zero_mean": "no_net_force_on_a_region",
                   "frequencies": "noise_cycles_per_length_per_region", "regions": "boxes_the_noise_acts_in",
                   "speed": "pattern_change_rate", "samples": "evaluations_per_noise_cell", "seed": "lattice_offset"}
    REFERENCE = "Perlin, K. (1985). ACM SIGGRAPH Computer Graphics 19(3):287-296; Perlin, K. (2002). ACM Trans. Graph. 21(3):681-682."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_grid")
        self.amplitude = float(params["amplitude"])
        self.speed = float(params.get("speed", 1.0))
        self.samples = int(params.get("samples", 4))
        self.seed = float(params.get("seed", 0.0))
        # `zero_mean: true` -- NO NET PUSH ON A REGION: the mean of the noise acceleration over the region's loaded
        # nodes is subtracted every substep, so a body is only DEFORMED. Without it a body under a low frequency
        # (a noise cell wider than the body) sees a nearly uniform force and is carried as a whole: exp01 step 0005
        # (2026-10-27) sent three of five balls drifting into each other within 2 s.
        self.zero_mean = bool(params.get("zero_mean", False))
        if params.get("regions") is not None:
            self.regions = [[float(v) for v in r] for r in params["regions"]]
            fr = params.get("frequencies", params.get("frequency"))
            fr = list(fr) if isinstance(fr, (list, tuple)) else [fr] * len(self.regions)
            if len(fr) != len(self.regions):
                raise ValueError(f"mpm_grid_noise: {len(self.regions)} regions but {len(fr)} frequencies")
            self.frequencies = [float(v) for v in fr]
        else:
            self.regions = None
            self.frequencies = [float(params["frequency"])]
        if self.amplitude < 0 or any(f <= 0 for f in self.frequencies):
            raise ValueError("mpm_grid_noise: amplitude must be >= 0 and every frequency > 0")
        # THE TWO KNOBS AS TENSORS, so the field is a function of them: the lattice is built in world units and
        # scaled by the frequency at every call, the increment multiplied by the amplitude -- the `differentiable`
        # implementation turns these two into parameters a fit can own.
        self.amp = torch.tensor(self.amplitude, dtype=torch.float32, device=device)
        self.freq = torch.tensor(self.frequencies, dtype=torch.float32, device=device)
        self._plan = None

    def _build(self, g):
        """Once per grid: for each region its index box on the grid, the lattice points (in noise cells) the noise
        is evaluated at, and the permutation and gradient tables on the device."""
        dev = g.v.device
        perm = torch.as_tensor(np.concatenate([_PERLIN_PERM, _PERLIN_PERM]), device=dev)
        grad = torch.as_tensor(_PERLIN_GRAD4, device=dev)
        corners = torch.as_tensor(_PERLIN_CORNERS, device=dev)
        D = g.dim
        box = [(g.shape[k] - 1) * g.dx for k in range(D)]
        regions = self.regions or [[0.0] * D + box]
        plan = []
        for r, f in zip(regions, self.frequencies):
            lo = [max(0, int(math.floor(r[k] / g.dx))) for k in range(D)]
            hi = [min(g.shape[k], int(math.ceil(r[D + k] / g.dx)) + 1) for k in range(D)]
            if any(hi[k] <= lo[k] for k in range(D)):
                continue
            # the noise lattice over this box: `samples` points per noise cell, never finer than the grid
            n_s = [max(2, min(hi[k] - lo[k], int(math.ceil((hi[k] - lo[k]) * g.dx * f * self.samples)) + 1))
                   for k in range(D)]
            ax = [torch.linspace(lo[k] * g.dx, (hi[k] - 1) * g.dx, n_s[k], device=dev) for k in range(D)]
            P = torch.stack(torch.meshgrid(*ax, indexing="ij"), -1).reshape(-1, D)   # world units; x frequency per call
            plan.append({"lo": lo, "hi": hi, "n_s": n_s, "P": P, "i": len(plan)})
        self._plan = {"perm": perm, "grad": grad, "corners": corners, "regions": plan}

    def forward(self, H, mask=None):
        g = H.field(self.at)
        _no_origin(self, g)
        if self._plan is None:
            self._build(g)
        from plexus.operators.mpm_ops import sub_dt
        dt = sub_dt(H, 2e-4)
        D = g.dim
        t = getattr(H, "frame_t", None)
        if t is None:                                            # outside the engine (a test): a frozen pattern
            if "t0" not in self._plan:
                self._plan["t0"] = torch.zeros((), device=g.v.device)
            t = self._plan["t0"]
        tw = t.to(g.v.dtype) * (float(getattr(H, "dt", 1.0) or 1.0) * self.speed)
        V = g.v.view(*g.shape, D)
        M = g.m.view(*g.shape)
        amp = self._const_read(self.amp) if self.FUNCTIONAL else self.amp
        freq = self._const_read(self.freq) if self.FUNCTIONAL else self.freq
        for R in self._plan["regions"]:
            P = R["P"] * freq[R["i"]] + self.seed; n_s = R["n_s"]
            comps = []
            for k in range(D):                                    # three independent fields: a shifted 4th axis
                p4 = torch.cat([P, (tw + 17.31 * (k + 1)).expand(P.shape[0], 1)], 1)
                comps.append(perlin4(p4, self._plan["perm"], self._plan["grad"], self._plan["corners"]).reshape(1, 1, *n_s))
            field = torch.cat(comps, 1)                            # [1, 3, nsx, nsy, nsz]
            lo, hi = R["lo"], R["hi"]
            size = [hi[k] - lo[k] for k in range(D)]
            acc = torch.nn.functional.interpolate(field, size=size, mode="trilinear", align_corners=True)[0]
            acc = acc.permute(1, 2, 3, 0)                         # [nx, ny, nz, 3]
            sl = tuple(slice(lo[k], hi[k]) for k in range(D))
            loaded = (M[sl] > 0).to(g.v.dtype)[..., None]
            if self.zero_mean:
                acc = acc - (acc * loaded).sum((0, 1, 2)) / loaded.sum().clamp(min=1.0)
            self._write_v(g, sl, acc * (dt * amp) * loaded)
        return {}

    def _write_v(self, g, sl, inc):
        """The increment into the grid velocity IN PLACE (a captured graph keeps the storage); the `differentiable`
        implementation rebinds instead."""
        g.v.view(*g.shape, g.dim)[sl].add_(inc)


@register_operator("mpm_grid_noise", implementation="differentiable", family="mpm", set="field", kind="field",
                   title="stir the grid with coherent noise, differentiable", equation=r"""$$\mathbf v_n \leftarrow \mathbf v_n + \Delta t\,A\,\mathbf N\!\big(f\,\mathbf x_n,\ s\,t\big)\qquad \mathbf N=(N_1,N_2,N_3),\ N_k\ \text{Perlin noise}$$""")
class MPMGridNoiseDiff(MPMGridNoise):
    r"""`differentiable` implementation of mpm_grid_noise: the grid velocity rebound rather than added in place, amplitude and frequencies as parameters, so a fit can own them.
    mpm_grid -> mpm_grid: as mpm_grid_noise, writing a new tensor into the grid velocity so a tape through the substep survives the backward.

        \mathbf v_n \leftarrow \mathbf v_n + \Delta t\,A\,\mathbf N\!\big(f\,\mathbf x_n,\ s\,t\big)\qquad \mathbf N=(N_1,N_2,N_3),\ N_k\ \text{Perlin noise}
        v_n            the grid node's velocity
        \Delta t       the substep length
        A              `amplitude`, an acceleration in length per time squared
        N_k            three independent Perlin noise fields, k = 1..3
        f              `frequency`, cycles per world length (a list per region with `regions`)
        x_n            the node position
        s              `speed`, the noise's own time rate
        t              the time

    Reference: Perlin, K. (1985). An image synthesizer. ACM SIGGRAPH Computer Graphics 19(3):287-296; Perlin, K. (2002). Improving noise. ACM Trans. Graph. 21(3):681-682."""
    PARAM_UNITS = {'amplitude': 'acceleration', 'frequency': 'dimensionless', 'frequencies': 'dimensionless', 'speed': 'dimensionless', 'samples': 'count', 'seed': 'count'}
    SYMBOLS = {'v_n': "the grid node's velocity", '\\Delta t': 'the substep length', 'A': '`amplitude`, an acceleration in length per time squared', 'N_k': 'three independent Perlin noise fields, k = 1..3', 'f': '`frequency`, cycles per world length (a list per region with `regions`)', 'x_n': 'the node position', 's': "`speed`, the noise's own time rate", 't': 'the time'}
    DIFFERENTIABLE = True
    FUNCTIONAL = True

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.amp = torch.nn.Parameter(self.amp.clone())
        self.freq = torch.nn.Parameter(self.freq.clone())

    def _const_read(self, t):
        return t

    def _write_v(self, g, sl, inc):
        D = g.dim
        delta = torch.zeros(*g.shape, D, dtype=g.v.dtype, device=g.v.device)
        delta[sl] = inc
        g.v = g.v + delta.reshape(-1, D)
