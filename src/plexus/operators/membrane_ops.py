"""The basement membrane: a thin crosslinked sheet between the epithelium and the stroma, and
everything that holds it there.

The sheet is a set of particles joined by breakable springs. Its mechanics come from those
crosslinks, its position from the anchors, and its fragmentation from bonds failing -- so the
interesting quantity throughout is a bond's STRAIN, the fractional extension it is carrying.

In the order they appear below:

    basement_membrane_particle  entity      a material point of the sheet (MPM state)
    basement_membrane_node      entity      the same point without the continuum state
    bm_seed                     seed        the shell, laid down once just outside the epithelium
    bm_bond                     lateral     the crosslinks: springs, each with its own rest length
    bm_unbond                   rewire      break the over-strained ones -- fragmentation
    integrin_adhesion           lateral     anchor each particle to the angular position it was
                                            seeded on, so growth stretches its bonds
    bm_remodel                  structural  rest lengths creep, so the sheet can grow not just stretch
    bm_contact                  lateral     it does not pass through what it rests on
    adhesion_seed / _pull / _turnover       hemidesmosomes: the grip, its force, and its renewal
    bm_repel                    lateral     excluded volume between unbonded nodes
    bm_secrete                  structural  new material as the surface it sits on grows
    bm_crosslink                rewire      new bonds between nodes that have drifted into range
    integrin_particle           entity      a material point of an integrin fibre
    integrin_fibre_seed / _track / _pull    fibres from the surface outward, bound at the tip
                                            (`integrin_*` names the receptor clusters, in
                                            integrin_ops.py -- a different thing)

THE TWO INTEGRIN FAMILIES ARE ONE HOP APART IN THE SAME CHAIN AND ARE NOT THE SAME THING.
`integrin_adhesion` anchors the MEMBRANE to the EPITHELIUM. `integrin_seed`, `integrin_track` and
`integrin_pull` run fibres from the epithelium out through the MATRIX to the membrane. The shared
prefix is what invites the confusion, so they are documented together rather than in two files
where nobody compares them.

A RESOLUTION LIMIT TRAVELS WITH ANY GRID COUPLING, and it is worth stating once at the top. The
sheet is far thinner than a grid cell, so a single cell holds many membrane particles: anything
that couples the sheet to another body THROUGH THE GRID has its strength set by the grid
resolution rather than by an adhesion parameter. That is why the anchoring, the contact and the
integrin pull are all per-particle forces here, and not boundary conditions on grid nodes.
"""
from __future__ import annotations
import math
import torch
from plexus.models.base import FieldUpdate, Lateral, Rewire, Structural
from plexus.models.entities import MPMParticle
from plexus.models.registry import register_entity, register_operator
from plexus.models.state import spatial_schema
from plexus.models.base import Lateral, Structural


BOND_TRACE: list = []
# (frame, |dp| total, radial dp, n nodes): what the matrix and membrane push back on the tissue
BOUNDARY_REACTION: list = []
# (frame, deepest penetration, n particles inside) for the per-particle contact
CONTACT_TRACE: list = []
# (frame, mean adhesion stretch, n adhesions or n ruptured)
ADHESION_TRACE: list = []
MEMBRANE_STRAIN: list = []       # per-particle bond strain, for the renderer


@register_entity(
    # `spatial_schema` (a `dim -> StateSchema` callable), NOT the legacy 2D dict this used to
    # carry: the registry is now consulted, and the basement-membrane specs are `dim: 3`.
    # See `models/entities.py` for the contract.
    # `membrane` is the biological name; the long one stays first and stays valid.
    "basement_membrane_particle", "membrane", depth=0,
    state_schema=spatial_schema,
    render={"color_by": "node_type", "arrows": None},
)
class BasementMembraneParticle:
    """A material point of the basement membrane, carrying the same continuum state as any other
    MPM particle -- the deformation gradient F, the affine velocity C, mass, the Lame parameters mu
    and lambda, and the rest volume p_vol -- through the stock provision. The crosslinks are not
    part of it; they live in `bm_bond`.

    Registered because the entity is resolved BY SET NAME. An unregistered set name falls back
    silently to a bare position/velocity schema, and the run then dies inside `mpm_strain` on a
    missing F -- which reads like a bug in the MPM operator and is a missing three-line class.
    """
    provision = MPMParticle.provision


@register_entity("basement_membrane_node")
class BasementMembraneNode:
    """A node of the SPRING-GRAPH membrane: position and velocity, and nothing else.

    The alternative, `BasementMembraneParticle`, carries a full continuum state -- deformation
    gradient, affine momentum, Lame parameters, particle volume -- and none of it is read by
    anything that decides the sheet's behaviour. The mechanics come from the crosslinks, the
    position from the integrin springs, the fragmentation from bonds breaking, and the figures
    colour it by crosslink strain rather than by MPM stress. The one thing the continuum state
    buys is momentum exchange with the interstitial matrix through the shared grid.

    And that one thing is resolution-limited rather than physical. The grid cell is an order of
    magnitude thicker than the sheet, so a single cell holds on the order of ten membrane
    particles: the coupling strength is then set by the grid resolution rather than by any
    parameter of the model. A spring-graph node states that plainly instead of carrying state that
    implies a continuum coupling the discretisation cannot deliver.
    """
    @staticmethod
    def provision(lvl, parent, s, H, device):
        # `alive` replaces the mass-zero trick the MPM version used to park its unsecreted reserve:
        # without a mass buffer there is nothing to zero, and a boolean is what was meant all along.
        n = lvl.get("pos").shape[0]
        lvl.register_buffer("alive", torch.zeros(n, dtype=torch.bool, device=device))


@register_operator("bm_seed", family="seed", set="particle", kind="seed",
                   equation=r"""$$\mathbf x_i=\mathbf c+\mathbf u_i\big(S\,R(\theta_i,\phi_i)+\text{offset}\big)$$""", title="lay down the basement membrane")
class BasementMembraneSeed(Structural):
    r"""lays the membrane down once as a monolayer shell just outside the epithelium's surface, at the recorded angular radius of the tissue plus a standoff.
    basement_membrane -> basement_membrane: writes every position at the opening of the trajectory (dormant `reserve` slots parked).

        \mathbf x_i=\mathbf c+\mathbf u_i\big(S\,R(\theta_i,\phi_i)+\text{offset}\big)
        x_i            the position of node i
        c              `centre`
        u_i            a quasi-uniform unit direction
        S              `scale`, the map's scale
        R(\theta_i, \phi_i) the recorded angular radius map of the epithelium (`surface`) in that direction
        offset         `offset`, the standoff outside the surface

    Reference: Topfer, U. et al. (2022). Basement membrane stiffness determines organ shape. Development 149:dev200456 (collagen IV sets basement-membrane stiffness)."""
    PARAM_UNITS = {'centre': 'length', 'jitter': 'fraction', 'offset': 'length', 'overdamped_gamma': 'mass_per_time', 'relax_iters': 'count', 'reserve': 'fraction', 'scale': 'dimensionless', 'seed': 'count', 'thickness': 'length'}
    SYMBOLS = {'x_i': 'the position of node i', 'c': '`centre`', 'u_i': 'a quasi-uniform unit direction', 'S': "`scale`, the map's scale", 'R(\\theta_i, \\phi_i)': 'the recorded angular radius map of the epithelium (`surface`) in that direction', 'offset': '`offset`, the standoff outside the surface'}

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["surface"]
    MECHANISM_TAGS = ["basement_membrane", "material_seeding", "epithelial_polarity"]
    PARAM_ROLES = {"offset": "shell_offset_outward", "thickness": "shell_thickness",
                   "jitter": "seed_lattice_disorder",
                   "scale": "surface_rescale", 'centre': 'centre_of_the_shell', 'implementation': 'ignored_here', 'overdamped_gamma': 'friction_of_the_relaxation', 'park': 'where_dormant_slots_are_parked', 'relax_iters': 'relaxation_sweeps_of_the_directions', 'reserve': 'dormant_slots_per_sheet_slot', 'seed': 'random_seed', 'surface': 'the_recorded_surface_map', 'surface_set': 'the_live_surface_set'}
    REFERENCE = ("Topfer, U. et al. (2022) Development 149:dev200456 "
                 "(collagen IV sets basement-membrane stiffness).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        import numpy as np
        self.at = params.get("_at", "basement_membrane_particle")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self.scale = float(params.get("scale", 1.0))
        self.offset = float(params.get("offset", 0.004))     # sits just outside the apical surface
        # A MONOLAYER, NOT A SLAB. At 0.010 the shell was ~5.7 particle spacings thick, so the k-nearest
        # search found radial neighbours and built chains through the thickness instead of a connected
        # sheet -- the seeded membrane came out with a largest component of 0.846, 15% of it already in
        # pieces. A basement membrane IS a sheet; the thickness should be about one spacing.
        self.thickness = float(params.get("thickness", 0.002))
        self.jitter = float(params.get("jitter", 0.35))   # tangential noise, in units of local spacing
        # MODEL vs IMPLEMENTATION. The model is "uniform areal density on a sphere". Fibonacci is one way
        # to compute it and `relaxed` is another; they are not different models and the parameter is named
        # so. Fibonacci is exactly uniform in density and badly non-uniform in ARRANGEMENT: consecutive
        # indices land angularly adjacent, so the points queue into spiral arms about one spacing apart,
        # which jitter at 0.35 of a spacing cannot erase. At 30k that is invisible; at the 3.3k a
        # secreting run starts with, the sheet renders as a spiral and the eye reads structure that is
        # not in the physics.
        self.implementation = str(params.get("implementation", "relaxed")).lower()
        self.relax_iters = int(params.get("relax_iters", 24))
        # THE RESERVE IS NOT SPARE CAPACITY, IT IS UNSECRETED MEMBRANE. A sheet pinned at fixed angular
        # positions on a surface whose radius triples must cover NINE TIMES the area with the particles
        # it started with, so it can only slip or thin or tear. `reserve` is the fraction of the set held
        # back at mass 0 for `bm_secrete` to lay down as the surface grows.
        self.reserve = float(params.get("reserve", 0.0))
        # where dormant particles wait: outside the box, not at the tissue centre (see `forward`)
        self.park = [float(v) for v in params.get("park", (-0.25, -0.25, -0.25))]
        self.surface_set = params.get("surface_set", None)
        self.gamma = float(params.get("overdamped_gamma", 0.0))
        self.seed = int(params.get("seed", 0))
        z = np.load(str(params["surface"]))
        self.smap0 = np.asarray(z["smap"], np.float32)[0] * self.scale
        self._done = False

    def forward(self, H, mask=None):
        if self._done:
            return {}
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        n = pos.shape[0]
        dev, dt_ = pos.device, pos.dtype
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        M = torch.as_tensor(self.smap0, dtype=torch.float32)
        nth, nph = M.shape

        # A FIBONACCI SPHERE, not uniform (theta, phi) sampling: equirectangular sampling piles points at
        # the poles, and a sheet with four times the areal density at its poles has four times the mass
        # and stiffness there -- an anisotropy nobody asked for, in the axis these experiments measure.
        # RELAX THE COUNT THAT WILL BE LAID DOWN, NOT THE WHOLE RESERVOIR. The relaxation used to run
        # over all `n` particles and the sheet then took every k-th of them -- and a SUBSET of a blue
        # noise set is not blue noise. Thinning randomises it straight back to Poisson:
        #
        #     relax 13.5k then keep 1 in 4     d/hex = 0.546   cv = 0.311
        #     uniform random                   d/hex = 0.461   cv = 0.535
        #     relax 3.4k directly              d/hex = 0.877   cv = 0.047
        #
        # So the seeded sheet was 8% better than random rather than packed, which is why holes are
        # present at frame 0 and grow from there -- an initialisation defect, not a resolution one.
        n_lay = n if self.reserve <= 0 else max(1, int(round(n / (1.0 + self.reserve))))
        if self.implementation == "relaxed":
            # BLUE NOISE BY REPULSION. Start from a uniform random sample -- which has the right density
            # and no arms, but clumps and holes -- and let each point drift away from its nearest
            # neighbours, renormalising to the sphere each step. That equalises SPACING without imposing
            # an order on the points, which is exactly the property a spiral lacks.
            gg = torch.Generator().manual_seed(self.seed)
            # the laid-down set is relaxed on its own; the reserve is placed afterwards and does not
            # participate, since it is parked at the centre until secreted
            u = torch.randn(n_lay, 3, generator=gg)
            u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
            u = u.to(dev)
            sp = math.sqrt(4.0 * math.pi / max(n_lay, 1))
            kk = 7
            for _ in range(self.relax_iters):
                push = torch.zeros_like(u)
                blk = 2048
                for a0 in range(0, n_lay, blk):
                    b0 = min(n_lay, a0 + blk)
                    dd = 1.0 - (u[a0:b0] @ u.T).clamp(-1, 1)          # 1 - cos, monotone in angle
                    dd[torch.arange(b0 - a0, device=dev), torch.arange(a0, b0, device=dev)] = 1e9
                    nb = torch.topk(-dd, kk, dim=1).indices
                    diff = u[a0:b0, None, :] - u[nb]                   # away from each neighbour
                    dist = diff.norm(dim=-1).clamp_min(1e-9)
                    w = (sp - dist).clamp_min(0.0) / sp                # only push if closer than target
                    push[a0:b0] = (diff / dist[..., None] * w[..., None]).sum(1)
                u = u + push * (0.35 * sp)
                u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
            # the reserve: random directions, parked at the centre anyway, so their arrangement is
            # irrelevant until secretion places them beside a live node
            if n_lay < n:
                extra = torch.randn(n - n_lay, 3, generator=gg)
                extra = extra / extra.norm(dim=1, keepdim=True).clamp_min(1e-12)
                u = torch.cat([u.cpu(), extra]).to(dev)
            u = u.to(torch.float32).cpu()
            print(f"[bm_seed] implementation=relaxed: {self.relax_iters} repulsion "
                  f"iterations from a uniform random sample -- same density as Fibonacci, no spiral "
                  f"arms", flush=True)
        else:
            i = torch.arange(n, dtype=torch.float64) + 0.5
            ct = 1.0 - 2.0 * i / n
            st = torch.sqrt((1.0 - ct * ct).clamp_min(0.0))
            phi = (math.pi * (1.0 + 5.0 ** 0.5) * i) % (2 * math.pi)
            u = torch.stack([st * torch.cos(phi), st * torch.sin(phi), ct], dim=1).to(torch.float32)

        # ...BUT NOT A PERFECT ONE. A pure Fibonacci spiral is a crystal: the rendered sheet shows obvious
        # lattice rows, and worse, the crosslink network inherits that regularity, so the strain field
        # carries a smooth large-scale modulation that is a property of the lattice and not of the
        # mechanics. (Measured on the un-jittered run: the end-state strain pattern is spatially organised
        # at 6x the shuffled null, yet growth, local coordination and nearby fibre density together
        # explain only 4% of it.) A real basement membrane is not crystalline. Displace each point
        # tangentially by a fraction of the local spacing, which breaks the lattice without opening holes.
        if self.jitter > 0.0 and self.implementation != "relaxed":
            spacing = math.sqrt(4.0 * math.pi / max(n, 1))          # mean angular separation, radians
            e1 = torch.stack([-u[:, 1], u[:, 0], torch.zeros_like(u[:, 0])], dim=1)
            nrm = e1.norm(dim=1, keepdim=True)
            e1 = torch.where(nrm > 1e-6, e1 / nrm.clamp_min(1e-12),
                             torch.tensor([[1.0, 0.0, 0.0]]).expand_as(e1))
            e2 = torch.cross(u, e1, dim=1)
            d1 = torch.randn(n, generator=g) * (self.jitter * spacing)
            d2 = torch.randn(n, generator=g) * (self.jitter * spacing)
            u = u + e1 * d1[:, None] + e2 * d2[:, None]
            u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)

        th = torch.acos(u[:, 2].clamp(-1, 1))
        ph = torch.atan2(u[:, 1], u[:, 0]) % (2 * math.pi)
        R = M[(th / math.pi * nth).long().clamp(0, nth - 1),
              (ph / (2 * math.pi) * nph).long().clamp(0, nph - 1)]
        c = torch.tensor(self.centre, dtype=torch.float32)
        # SEEDED ON THE SURFACE LEVEL IF THERE IS ONE, and this is not a refinement -- it is the
        # difference between a sheet that starts at rest and one that starts torn. The table lookup below
        # takes R from whichever 32x64 cell a direction falls in; `surface_track` interpolates the same
        # map smoothly. The two disagree by the quantisation error, ~0.0016 at the starting radius,
        # against a bond rest length of 0.0021. Seed with one and anchor with the other and EVERY
        # particle begins about a full bond length away from its anchor: the run does not crash, it
        # returns 141 surviving bonds out of 100,801 and an ECM stress of 57,792.
        sset = getattr(self, "surface_set", None)
        if sset is not None:
            sl = H.level(sset)
            sp = sl.get("pos")
            if sp.shape[0] != n:
                raise RuntimeError(
                    f"bm_seed: `surface_set` has {sp.shape[0]} elements against {n} "
                    f"membrane particles; the 1:1 seeding needs them equal.")
            su = sl.u if hasattr(sl, "u") else \
                (sp - c.to(sp)) / (sp - c.to(sp)).norm(dim=1, keepdim=True).clamp_min(1e-12)
            u = su.detach().cpu()
            thick = (torch.rand(n, generator=g) - 0.5) * self.thickness
            P = (sp.detach().cpu() + su.detach().cpu() * (self.offset + thick)[:, None])
        else:
            r = R + self.offset + (torch.rand(n, generator=g) - 0.5) * self.thickness
            P = c + u * r[:, None]
        # LOUD IF IT LANDS OUTSIDE THE BOX. The engine's wall boundary would silently clamp the shell
        # onto the cube faces, and the only symptom would be a bond count of zero several operators
        # later -- which reads as a bond bug rather than a units mistake.
        if float(P.min()) < 0.0 or float(P.max()) > 1.0:
            raise RuntimeError(
                f"bm_seed: shell radius "
                f"{float((P - c).norm(dim=1).mean()):.4g} puts particles outside "
                f"the unit box (range {float(P.min()):.3g}..{float(P.max()):.3g}). `scale` is almost "
                f"certainly wrong: the surface map is in TISSUE units and must be multiplied by the "
                f"tissue-to-box scale, which only `combine.build` knows.")
        lvl.get("pos")[:] = P.to(dev, dt_)
        n0 = n if self.reserve <= 0 else max(1, int(round(n / (1.0 + self.reserve))))
        alive = torch.zeros(n, dtype=torch.bool, device=dev)
        # STRIDED, NOT THE FIRST n0. The Fibonacci lattice is generated with ct = 1 - 2i/n, so its index
        # runs monotonically from the north pole to the south: `alive[:n0]` is a POLAR CAP, not a sparse
        # shell. Taking the first n0 covers a POLAR CAP whose extent is n0/n of the sphere, and the
        # sheet then grows downward as it secretes, which reads as the membrane migrating. Every
        # k-th point of a Fibonacci spiral is itself a coarser Fibonacci spiral,
        # so a stride gives a uniform sparse shell and the reserve fills in between.
        if self.implementation == "relaxed":
            # THE FIRST n0, because for `relaxed` those ARE the set that was relaxed -- the reserve
            # after them is random and parked. Striding here would interleave the two and hand the sheet
            # a mixture, undoing the packing the relaxation just produced.
            alive[:n0] = True
        else:
            step = max(1, int(round(n / max(n0, 1))))
            alive[::step] = True
        if int(alive.sum()) > n0:                       # trim the overshoot from the far end
            extra = int(alive.sum()) - n0
            idx = alive.nonzero(as_tuple=True)[0][-extra:]
            alive[idx] = False
        if n0 < n:
            # PARKED OUTSIDE THE BOX, MASSLESS. Parking the reserve at the tissue CENTRE is fine
            # for a spring membrane, which MPM never touches, and wrong for a continuum one: the
            # dormant particles then reach the live sheet through the shared grid and freeze it.
            # Massless is not inert either: `mpm_scatter` still deposits the
            # stress term, which carries p_vol and not m.
            #
            # Outside the box does not mean off the grid -- `bspline` clamps out-of-range indices to the
            # boundary cells, so the reserve lands on one corner node. That is the point: the corner is
            # ~0.87 from the tissue, far outside the surface band the boundary condition acts on, so
            # whatever it deposits cannot reach the membrane.
            lvl.get("pos")[~alive] = torch.tensor(self.park, device=dev, dtype=dt_)
            m = getattr(lvl, "mass", None)
            if m is None:
                raise RuntimeError(
                    "bm_seed: `reserve` needs a `mass` buffer to park the dormant "
                    "particles massless, and this level has none. Without it the reserve would scatter "
                    "into the grid from the tissue centre.")
            self._mass0 = float(m.reshape(-1)[0])
            m[~alive] = 0.0
            # AND `occ`, WHICH IS THE FLAG THE MPM OPERATORS ACTUALLY CHECK. Mass 0 was this file's own
            # convention for "dormant" and it is not the framework's: `mpm_scatter` masks its weights by
            # occupancy, `mpm_gather` freezes occ==0 particles instead of advecting them, and Level
            # already carries occ with a spawn/retire API. Mass alone does not stop a particle
            # scattering, because the scatter's STRESS term is weighted by p_vol, not by m -- so a
            # massless reserve parked at the tissue centre still deposits stress into the shared grid,
            # right where the boundary condition is looking. That is what froze the live sheet.
            oc = getattr(lvl, "occ", None)
            if oc is not None:
                oc[~alive] = 0.0
        H.membrane_alive = alive
        # Published so `surface_track` can pair element i with particle i without reproducing an RNG
        # sequence. (Rebuilding the lattice does in fact reproduce it -- the jitter draws come first in
        # both -- but a shared lattice that depends on two functions drawing in the same order is a
        # coincidence, not an invariant.)
        H.membrane_u0 = u.to(dev, dt_).clone()
        self._done = True
        print(f"[bm_seed] {n} particles on a shell at r_surface + {self.offset:.4g} "
              f"(thickness {self.thickness:.4g}), Fibonacci + {self.jitter:.2g}-spacing jitter; "
              f"{int(alive.sum())} laid down, {n - int(alive.sum())} held in reserve",
              flush=True)
        return {}


# ---- the membrane against the LIVE tissue (exp05 / exp11; roadmap M1, in its smallest form) ------
#
# `bm_seed` / `bm_contact` above read the epithelium as a RECORDED radius map; the `live`
# implementations below read the apico-basal tissue's BASAL surface (`pos - sep`; the mid-surface
# `pos` of a tissue without a separation) at the frame they run, and the contact returns its reaction
# to that surface (the load helpers in contact_ops, the relaxation in vertex_ops).
#
# THE CONTACT FINDS EACH NODE'S NEAREST POINT ON THE SURFACE, NOT THE SURFACE ON ITS RAY.
# `mesh_contact` rays from the tissue centroid and takes the outermost face, which assumes a
# star-shaped surface (FUSED_09.md, open defect 2). Measured on the first live hole run (60 cells
# grown to 700): once a bud grew, the deepest membrane node read 1.4 units "inside" the tissue, more
# than a whole cell thickness (0.88), because a node beside the bud's neck had its ray hit the bud
# wall; with the nearest-point lookup the deepest reads 0.26. A membrane is a sheet lying ON the
# surface, so the question is local: which face is nearest, and on which side of it the node sits.

def _basal_lookup():
    """A `mesh_contact[model: spatial_hash]` instance used ONLY for its surface index. Building one
    registers it as the contact `mesh_inside` counts against (`_LIVE`); that is put back."""
    from plexus.operators.contact_ops import MeshContactHash, _LIVE
    prev = _LIVE.get("contact")
    geo = MeshContactHash({"surface": "_lookup_only_", "verbose": False})
    geo._said_hash = True
    if prev is None:
        _LIVE.pop("contact", None)
    else:
        _LIVE["contact"] = prev
    return geo


def _live_basal(H, surface, sep_block, geo, dev, dt_, side="basal"):
    """(mesh table, centroid c, lookup M) of the tissue's basal surface `pos - sep` this frame,
    about its own centroid -- or None before the tissue is seeded. `side="apical"`: the other ring,
    `pos + sep` (the lumen-facing one of an `apical: in` shell)."""
    tl = H.level(surface)
    m = getattr(tl, "_mesh", None)
    if m is None or not int(m.get("Nv", 0)):
        return None
    nv = int(m["Nv"])
    B = tl.get("pos")[:nv].to(device=dev, dtype=dt_)
    # A MID-SURFACE TISSUE (the default `cell_mechanics` shape energy) HAS NO SEPARATION: its one
    # surface is the one the membrane rests on, and the standoff stands for half the cell thickness.
    if sep_block in tl.state_schema:
        if side == "apical":
            B = B + tl.get(sep_block)[:nv].to(device=dev, dtype=dt_)
        else:
            B = B - tl.get(sep_block)[:nv].to(device=dev, dtype=dt_)
    c = B.mean(0)
    V = B - c
    M = geo._build_from(V, m["E_srce"].to(dev), m["E_trgt"].to(dev), m["E_face"].to(dev),
                        torch.zeros_like(V), dev, dt_)
    return m, c, M


def _closest_point_barycentric(P, A, B, C):
    """Barycentric (b0, b1, b2) of the point of triangle ABC nearest P, batched over [..., 3].

    Ericson, C. (2005). Real-Time Collision Detection, sec. 5.1.5 (ClosestPtPointTriangle): the
    Voronoi regions of the three vertices, three edges and the face, the earlier region winning."""
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = P - B
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = P - C
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    def safe(x):
        return torch.where(x.abs() < 1e-30, torch.full_like(x, 1e-30), x)
    den = safe(va + vb + vc)
    v, w = vb / den, vc / den
    b = torch.stack([1.0 - v - w, v, w], -1)                                   # the face interior
    z, o = torch.zeros_like(d1), torch.ones_like(d1)
    regions = []
    t = (d4 - d3) / safe((d4 - d3) + (d5 - d6))
    regions.append(((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), torch.stack([z, 1 - t, t], -1)))
    t = d2 / safe(d2 - d6)
    regions.append(((vb <= 0) & (d2 >= 0) & (d6 <= 0), torch.stack([1 - t, z, t], -1)))
    regions.append(((d6 >= 0) & (d5 <= d6), torch.stack([z, z, o], -1)))
    t = d1 / safe(d1 - d3)
    regions.append(((vc <= 0) & (d1 >= 0) & (d3 <= 0), torch.stack([1 - t, t, z], -1)))
    regions.append(((d3 >= 0) & (d4 <= d3), torch.stack([z, o, z], -1)))
    regions.append(((d1 <= 0) & (d2 <= 0), torch.stack([o, z, z], -1)))
    for cond, val in regions:                          # later in the list = earlier in Ericson
        b = torch.where(cond[..., None], val, b)
    return b


def _nearest_faces(M, c, y):
    """For points y [n, 3] (world coordinates), the nearest fan triangle of the surface in M (built
    about the centroid c): (found [n], tri [n], w [n, 3], n_hat [n, 3] outward, gap [n] the height
    of y above the triangle along n_hat).

    OUTWARD BY THE MESH'S OWN WINDING, one sign for the whole surface -- the majority sign of
    n . (triangle centre), area-weighted -- so a face on a bud's side wall or under an overhang keeps
    its true outward side instead of the one its position from the centroid suggests."""
    dev, dt_ = y.device, y.dtype
    n = y.shape[0]
    found = torch.zeros(n, dtype=torch.bool, device=dev)
    tri = torch.zeros(n, dtype=torch.long, device=dev)
    w = torch.zeros(n, 3, device=dev, dtype=dt_)
    nh = torch.zeros(n, 3, device=dev, dtype=dt_)
    gap = torch.zeros(n, device=dev, dtype=dt_)
    if n == 0:
        return found, tri, w, nh, gap
    A, B, C = M["A"], M["B"], M["C"]
    if "nrm_wind" not in M:
        raw = torch.cross(B - A, C - A, dim=1)
        sgn = torch.sign((raw * (A + B + C)).sum())
        sgn = sgn if float(sgn) != 0.0 else torch.ones_like(sgn)
        M["nrm_wind"] = sgn * raw / raw.norm(dim=1).clamp_min(1e-20)[:, None]
    x = y - c
    nx, ny, nz = M["hdims"]
    h, lo = M["h"], M["lo"]
    cc = ((x - lo) / h).long()
    cc[:, 0].clamp_(0, nx - 1); cc[:, 1].clamp_(0, ny - 1); cc[:, 2].clamp_(0, nz - 1)
    o = torch.tensor([-1, 0, 1], device=dev)
    off = torch.stack(torch.meshgrid(o, o, o, indexing="ij"), -1).reshape(-1, 3)
    nb = cc[:, None, :] + off[None]
    nb[..., 0].clamp_(0, nx - 1); nb[..., 1].clamp_(0, ny - 1); nb[..., 2].clamp_(0, nz - 1)
    bi = (nb[..., 0] * ny + nb[..., 1]) * nz + nb[..., 2]
    sel = torch.nonzero(M["hcounts"][bi].sum(1) > 0).squeeze(1)
    for s0 in range(0, sel.numel(), 4096):
        ss = sel[s0:s0 + 4096]
        cand = M["htable"][bi[ss]].reshape(ss.numel(), -1)
        ok = cand >= 0
        ci = cand.clamp_min(0)
        P = x[ss][:, None, :].expand(-1, ci.shape[1], -1)
        bary = _closest_point_barycentric(P, A[ci], B[ci], C[ci])
        Q = bary[..., 0:1] * A[ci] + bary[..., 1:2] * B[ci] + bary[..., 2:3] * C[ci]
        d2 = torch.where(ok, ((P - Q) ** 2).sum(-1), torch.full_like(ok, float("inf"), dtype=dt_))
        best = d2.argmin(1)
        ar = torch.arange(ss.numel(), device=dev)
        t_ = ci[ar, best]
        found[ss] = torch.isfinite(d2[ar, best])
        tri[ss] = t_
        w[ss] = bary[ar, best]
        nh[ss] = M["nrm_wind"][t_]
        gap[ss] = ((x[ss] - Q[ar, best]) * M["nrm_wind"][t_]).sum(1)
    return found, tri, w, nh, gap


def _relaxed_directions(n, seed, iters, dev):
    """n blue-noise unit directions: `bm_seed`'s `relaxed` sampling (a uniform random sample whose
    points repel their seven nearest neighbours, renormalised to the sphere each step)."""
    gg = torch.Generator().manual_seed(int(seed))
    u = torch.randn(n, 3, generator=gg)
    u = (u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(dev)
    sp = math.sqrt(4.0 * math.pi / max(n, 1))
    for _ in range(int(iters)):
        push = torch.zeros_like(u)
        for a0 in range(0, n, 2048):
            b0 = min(n, a0 + 2048)
            dd = 1.0 - (u[a0:b0] @ u.T).clamp(-1, 1)
            dd[torch.arange(b0 - a0, device=dev), torch.arange(a0, b0, device=dev)] = 1e9
            nb = torch.topk(-dd, 7, dim=1).indices
            diff = u[a0:b0, None, :] - u[nb]
            dist = diff.norm(dim=-1).clamp_min(1e-9)
            wgt = (sp - dist).clamp_min(0.0) / sp
            push[a0:b0] = (diff / dist[..., None] * wgt[..., None]).sum(1)
        u = u + push * (0.35 * sp)
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return u


@register_operator("bm_seed", implementation="live", family="seed", set="particle", kind="seed",
                   equation=r"""$$\mathbf x_i=\mathbf c+\mathbf u_i\big(t(\mathbf u_i)+\text{offset}\big),\qquad \text{alive}_i=[\mathbf u_i\cdot\mathbf a<\cos\alpha]$$""", title="lay the basement membrane on the tissue")
class BasementMembraneSeedLive(BasementMembraneSeed):
    r"""lays the membrane on the live tissue's basal surface, blue-noise directions at a declared standoff, with an optional declared hole and a parked reserve for later secretion.
    basement_membrane_node -> basement_membrane_node: writes every position and `alive` once, at the opening; publishes the reserve as `H.membrane_reserve`.

        \mathbf x_i=\mathbf c+\mathbf u_i\big(t(\mathbf u_i)+\text{offset}\big),\qquad \text{alive}_i=[\mathbf u_i\cdot\mathbf a<\cos\alpha]
        x_i            the position of node i
        c              the centroid of the basal surface (pos - sep)
        u_i            a blue-noise unit direction
        t(u_i)         the basal surface's radius along u_i (the contact's ray cast)
        offset         `offset`, the standoff
        alive_i        0 inside the declared hole
        a, \alpha      `hole`'s axis and half-angle

    Reference: Harunaga, J. S., Doyle, A. D. & Yamada, K. M. (2014). Dev. Biol. 394:197-205 (perforations in the basement membrane at the tips of branching salivary-gland buds)."""
    PARAM_UNITS = {'offset': 'length', 'relax_iters': 'count', 'reserve': 'fraction', 'seed': 'count'}
    SYMBOLS = {'x_i': 'the position of node i', 'c': 'the centroid of the basal surface (pos - sep)', 'u_i': 'a blue-noise unit direction', 't(u_i)': "the basal surface's radius along u_i (the contact's ray cast)", 'offset': '`offset`, the standoff', 'alive_i': '0 inside the declared hole', 'a, \\alpha': "`hole`'s axis and half-angle"}
    REQUIRES_PARAMS = ["tissue"]
    PARAM_ROLES = {"offset": "standoff_outside_the_basal_surface", "hole": "declared_hole",
                   "tissue": "the_live_vertex_set", "reserve": "dormant_slots_per_sheet_slot", 'relax_iters': 'relaxation_sweeps_of_the_directions', 'seed': 'random_seed', 'sep_block': 'vertex_block_of_the_apico_basal_separation'}

    def __init__(self, params, device="cpu"):
        Structural.__init__(self, params, device)
        self.at = params.get("_at", "bm_node")
        self.tissue = str(params["tissue"])
        self.sep_block = str(params.get("sep_block", "sep"))
        self.offset = float(params.get("offset", 0.0))
        self.relax_iters = int(params.get("relax_iters", 24))
        self.seed = int(params.get("seed", 0))
        # `reserve` (default 0 = every slot is laid or left in the hole, bit-identical to before): the
        # base `bm_seed`'s own convention, reserve slots PER SHEET SLOT -- the sheet takes the first
        # round(n / (1 + reserve)) slots (laid + hole), the rest are parked dormant at c for
        # `bm_secrete[live]`. The sheet is relaxed ON ITS OWN (a thinned blue-noise set is Poisson), so
        # n = n_sheet (1 + reserve) lays the same sheet, node for node, as n_sheet with no reserve.
        self.reserve = float(params.get("reserve", 0.0))
        hole = params.get("hole", None)
        self.hole = None
        if hole:
            a = torch.tensor([float(v) for v in hole["axis"]], dtype=torch.float32)
            if float(a.norm()) <= 0:
                raise ValueError("bm_seed[live]: hole.axis must be a non-zero vector")
            self.hole = (a / a.norm(), float(hole["half_angle_deg"]))
        self._geo = _basal_lookup()
        self._done = False

    def forward(self, H, mask=None):
        if self._done:
            return {}
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        n = pos.shape[0]
        dev, dt_ = pos.device, pos.dtype
        got = _live_basal(H, self.tissue, self.sep_block, self._geo, dev, dt_)
        if got is None:
            raise RuntimeError(f"bm_seed[live]: the tissue set {self.tissue!r} has no mesh yet. List "
                               f"its `seed_mesh` BEFORE `bm_seed` in `seed:`.")
        m, c, M = got
        n_all = n
        if self.reserve > 0:
            n = max(1, int(round(n_all / (1.0 + self.reserve))))
        u = _relaxed_directions(n, self.seed, self.relax_iters, dev).to(dt_)
        from plexus.operators.contact_ops import MeshContact
        hit, tri, t, w = MeshContact._query(self._geo, M, u, u, torch.ones(n, device=dev, dtype=dt_))
        if not bool(hit.any()):
            raise RuntimeError("bm_seed[live]: no direction hit the tissue surface -- is it closed?")
        t = torch.where(hit, t, t[hit].median())
        P = c + u * (t + self.offset)[:, None]
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        if self.hole is not None:
            a, half = self.hole
            alive = (u @ a.to(dev, dt_)) < math.cos(math.radians(half))
            P[~alive] = c
        reserve = torch.zeros(n_all, dtype=torch.bool, device=dev)
        if n < n_all:
            # THE RESERVE: parked at c, dormant, unoccupied, with no direction (u0 = 0) -- distinct from
            # the hole's left-out nodes, which keep theirs, so `bm_secrete[live]` never draws on the hole
            k = n_all - n
            P = torch.cat([P, c.expand(k, 3)])
            alive = torch.cat([alive, torch.zeros(k, dtype=torch.bool, device=dev)])
            u = torch.cat([u, torch.zeros(k, 3, device=dev, dtype=dt_)])
            reserve[n:] = True
        lvl.get("pos")[:] = P
        oc = getattr(lvl, "occ", None)
        if oc is not None:
            oc[~alive] = 0.0
        H.membrane_alive = alive
        H.membrane_u0 = u.clone()
        # published for `bm_secrete[live]`: the declared hole (unit axis, half-angle in degrees) or None,
        # and which dormant slots are reserve
        H.membrane_hole = self.hole
        H.membrane_reserve = reserve
        self._done = True
        hs = ("no hole" if self.hole is None else
              f"hole of half-angle {self.hole[1]:g} deg about {[round(float(v), 3) for v in self.hole[0]]}")
        print(f"[bm_seed/live] {n} nodes on the live basal surface of `{self.tissue}` + {self.offset:g} "
              f"(radius {float(t.median()):.4g} about its centroid); {hs}: {int(alive.sum())} laid "
              f"down, {n - int(alive[:n].sum())} left out"
              + (f", {n_all - n} held in reserve" if n < n_all else ""), flush=True)
        return {}


@register_operator("bm_bond", family="mechanics", set="particle", kind="lateral",
                   equation=r"""$$L_e=\lVert\mathbf x_j-\mathbf x_i\rVert,\qquad \mathbf f_e=k\,(L_e-L_e^{0})\,\frac{\mathbf x_j-\mathbf x_i}{L_e}$$""", title="crosslink the basement membrane")
class BasementMembraneBond(Lateral):
    r"""crosslinks: springs between neighbouring membrane nodes, built once and thereafter breakable, Hookean in the extension; what makes the sheet a membrane rather than a cloud of stiff dust.
    basement_membrane -> basement_membrane: reads pos, emits an acceleration (f_e on i, -f_e on j); reports each bond's strain.

        L_e=\lVert\mathbf x_j-\mathbf x_i\rVert,\qquad \mathbf f_e=k\,(L_e-L_e^{0})\,\frac{\mathbf x_j-\mathbf x_i}{L_e}
        L_e            the current length of bond e
        x_i, x_j       the two nodes' positions
        f_e            the force on node i (per unit node mass)
        k              `k`, the crosslink stiffness, per time squared (`aniso` stiffens the hoop direction)
        L^0_e          the bond's rest length, recorded when it formed

    Reference: Topfer, U. et al. (2022). Development 149:dev200456 (collagen IV is the principal contributor to basement-membrane stiffness)."""
    PARAM_UNITS = {'aniso': 'dimensionless', 'centre': 'length', 'cutoff': 'dimensionless', 'damp': 'rate', 'k': 'acceleration', 'k_adhesion_hint': 'acceleration', 'k_block_cap': 'dimensionless', 'max_neighbours': 'count', 'overdamped_gamma': 'mass_per_time', 'rebond_every': 'frames', 'snapshot_every': 'frames', 'guard_dt': 'fraction'}
    SYMBOLS = {'L_e': 'the current length of bond e', 'x_i, x_j': "the two nodes' positions", 'f_e': 'the force on node i (per unit node mass)', 'k': '`k`, the crosslink stiffness, per time squared (`aniso` stiffens the hoop direction)', 'L^0_e': "the bond's rest length, recorded when it formed"}

    # `mpm_acceleration` routes the force into the MPM substep as an external body force; `acceleration`
    # hands it to the engine, which integrates v += dt*a; x += dt*v directly. The spring graph needs the
    # second, and the spec sets `emit:` per run -- `_resolve_emit` reads the spec ahead of this default.
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["crosslink", "basement_membrane", "elastic_network"]
    PARAM_ROLES = {"k": "bond_stiffness", "cutoff": "neighbour_search_radius",
                   "max_neighbours": "bonds_per_particle", "damp": "bond_damping", 'aniso': 'hoop_over_meridional_stiffness', 'centre': 'centre_that_defines_the_hoop_direction', 'graph_mode': 'how_the_bond_graph_is_built', 'guard_dt': 'stability_guard_on_the_frame_step', 'k_adhesion_hint': 'adhesion_stiffness_the_guard_accounts_for', 'k_block': 'node_block_that_scales_k', 'k_block_cap': 'cap_on_that_scale', 'overdamped_gamma': 'friction_of_the_overdamped_update', 'rebond_every': 'frames_between_rebondings_of_new_nodes', 'record_hoop': 'record_the_hoop_strain', 'snapshot_every': 'frames_between_bond_snapshots'}
    REFERENCE = "Topfer, U. et al. (2022) Development 149:dev200456."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "basement_membrane_particle")
        # STABILITY BOUND, stated because it is the thing that limits this number: the bond is a
        # spring-mass oscillator with omega = sqrt(k), and explicit integration needs
        # dt_sub < 2/omega. At dt_sub = 2e-4 that is k < 1e8; 2e5 gives ~70 substeps per period, which
        # is stable with room to spare.
        self.k = float(params.get("k", 2.0e5))
        # ...AND THAT ARGUMENT ONLY HOLDS FOR THE MPM PATH. `emit: acceleration` hands the force to the
        # ENGINE, which integrates once per FRAME at dt = 4e-3 -- twenty times the MPM substep. The limit
        # is dt < 2/omega with omega = sqrt(k_eff), and k_eff is not k but the SUM over a node's bonds,
        # about 4.6 of them, so the ceiling drops by another factor of 4.6:
        #
        #     k_max  =  4 / (dt^2 * bonds_per_node)   ~  5.4e4  at dt = 4e-3
        #
        # Above it the sheet does not wobble, it diverges: the first graph-mode run returned a mean
        # crosslink strain of INFINITY at k = 2e5, a value that is obviously wrong and would have been
        # obviously wrong to average into a sweep. Checked here rather than discovered there.
        self.graph_mode = bool(params.get("graph_mode", False))
        self.snapshot_every = int(params.get("snapshot_every", 20))
        self.rebond_every = int(params.get("rebond_every", 20))   # ongoing crosslinking
        self.aniso = float(params.get("aniso", 1.0))          # circumferential : meridional stiffness
        # `k_block` (default none): each bond's stiffness scaled by the mean of a width-1 block of its two
        # nodes -- the membrane's own mass `bm_M` (exp 11 Phase 3 B3, 2026-09-29), so a thicker membrane is a
        # stiffer one (collagen IV is its stiffness, Topfer 2022) -- capped at `k_block_cap` (default 3.0)
        # times k, and the explicit-integration guard below is checked at the cap
        self.k_block = params.get("k_block", None)
        self.k_block_cap = float(params.get("k_block_cap", 3.0))
        self.record_hoop = bool(params.get("record_hoop", False))
        self.centre_t = torch.tensor([float(v) for v in params.get("centre", [0.5, 0.5, 0.5])])
        # the adhesion stiffness acting on the same nodes; the spec passes it so the ceiling can see it
        self.k_adh_hint = float(params.get("k_adhesion_hint", 0.0))
        self.gamma = float(params.get("overdamped_gamma", 0.0))
        self.cutoff = float(params.get("cutoff", 0.020))
        self.max_nb = int(params.get("max_neighbours", 6))
        self.damp = float(params.get("damp", 0.0))
        # `guard_dt: spec` -- THE GUARD READS THE RUN'S OWN TIME STEP (exp 11, 2026-09-27). Called without
        # one, `_check_stability` assumes the archive's 0.004 per frame, so on a live-coupling spec at dt 1
        # its ceiling is 250x too generous: exp 11 batch 2 (overdamped_gamma 5, k 5, z 6.4, k·z·dt/gamma
        # 6.4 > 2) passed it and every node went NaN by frame 40 (exp11 Finding 38). Default: unchanged.
        self.guard_dt = str(params.get("guard_dt", "archive")).lower()
        self.i = self.j = self.rest = self.alive = None
        self._said = False

    def _check_stability(self, bonds_per_node, dt_frame=4.0e-3):
        """Refuse a stiffness the frame-rate integration cannot carry.

        CALIBRATED AGAINST A SWEEP, not derived and left. The textbook dt < 2/omega using the crosslinks
        alone gives 4.4e4; a measured sweep over k = 200..40,000 put the real transition between 5,000
        and 10,000, six times lower:

          k    200-5,000   bonds ~50,700   strain 0.101-0.106   stable
          k       10,000   bonds  33,421   strain 0.114         transition
          k   20,000+      bonds  63,615   strain 2.02          diverged

        Two corrections to the textbook form. The INTEGRIN spring pulls on the same node, so it belongs
        in the effective stiffness; and the usable criterion with drag present is dt*omega < 1, not the
        marginal 2. (The diverged runs also over-secrete -- 29k nodes against 11k -- because the
        secretion setpoint keys on mean radius and an exploding sheet has a large one. A numerical
        failure that reads as biology.)
        """
        if not self.graph_mode:
            return                                   # MPM path: integrated at the substep, ~70 per period
        # THE CRITERION DEPENDS ON WHICH EQUATION OF MOTION IS BEING INTEGRATED, and using the wrong one
        # refuses exactly the runs that motivated the change. Inertial (v += dt*a) is a spring-mass
        # oscillator, dt*sqrt(k_eff) < 1. Overdamped (x += dt*F/gamma) is first order, dt*k_eff/gamma < 2.
        # Either way k_eff is the sum over the node's OWN bonds plus its tether, not k alone.
        #
        #     inertial     k_max = (1/dt^2 - kappa)/z   ~ 9.0e3
        #     overdamped   k_max = (2*gamma/dt - kappa)/z ~ 1.7e5   at gamma = 2000
        #
        # 19x, not the 120x I first claimed -- that figure dropped the coordination factor z, which is
        # the same term that made the first version of this guard 13x too generous.
        #
        # Worth stating because it changes what a stiffness sweep MEANS: overdamped, k and gamma enter
        # only as k/gamma, a relaxation RATE. So the ceiling is not a property of the sheet, it can be
        # moved by raising gamma -- at the cost of a sheet that responds more slowly than the tissue grows.
        z = max(bonds_per_node, 1.0)
        if self.gamma > 0:
            k_max = (2.0 * self.gamma / dt_frame - self.k_adh_hint) / z
        else:
            k_max = (1.0 / dt_frame ** 2 - self.k_adh_hint) / z
        _k_top = self.k * (self.k_block_cap if self.k_block else 1.0)
        if _k_top > k_max:
            raise RuntimeError(
                f"bm_bond: k = {_k_top:.3g} (k x the k_block cap when a block scales it) exceeds the explicit-integration ceiling "
                f"{k_max:.3g} for graph mode at dt = {dt_frame:g} with {bonds_per_node:.1f} bonds per "
                f"node. The spring graph is integrated once per FRAME, not at the MPM substep, so it "
                f"cannot carry the stiffness the MPM path could: this run would return an infinite "
                f"strain rather than a soft sheet. Either lower k, or give the membrane its own substep "
                f"loop (which is what moving off MPM gave up).")

    def _neighbours_celllist(self, pos, k):
        """k nearest within `cutoff`, in O(N) instead of O(N^2), via a uniform cell list.

        WHY THIS REPLACED THE CHUNKED PAIRWISE SEARCH. The old comment here said a grid hash was "more
        code for a cost paid a single time", which was true at 40k nodes and false the moment secretion
        started re-bonding and the deposition term began measuring local sparsity every frame. At 45k the
        pairwise form is 2e9 distances; at 500k it is 2.5e11, and it is no longer paid once.

        Cell size IS the cutoff, so every neighbour within range lies in the particle's own cell or one
        of the 26 around it. On a shell at this density that is order 100 candidates per node rather than
        N, and the total work becomes linear.
        """
        n = pos.shape[0]
        dev = pos.device
        c = torch.div(pos, self.cutoff, rounding_mode="floor").long()
        c = c - c.min(0).values                                  # non-negative cell coords
        G = int(c.max().item()) + 2
        cid = (c[:, 0] * G + c[:, 1]) * G + c[:, 2]
        order = torch.argsort(cid)
        cid_s = cid[order]
        uniq, counts = torch.unique_consecutive(cid_s, return_counts=True)
        starts = torch.cumsum(counts, 0) - counts

        off = torch.tensor([[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
                           device=dev, dtype=torch.long)
        best_d = torch.full((n, k), 1e9, device=dev, dtype=pos.dtype)
        best_j = torch.zeros((n, k), device=dev, dtype=torch.long)
        for o in off:
            nid = ((c[:, 0] + o[0]) * G + (c[:, 1] + o[1])) * G + (c[:, 2] + o[2])
            slot = torch.searchsorted(uniq, nid).clamp(max=uniq.numel() - 1)
            ok = uniq[slot] == nid
            cnt = torch.where(ok, counts[slot], torch.zeros_like(counts[slot]))
            if int(cnt.max()) == 0:
                continue
            m = int(cnt.max())
            # a fixed (n, m) candidate block per offset: m is the fullest cell, single digits here
            ar = torch.arange(m, device=dev)[None, :]
            valid = ar < cnt[:, None]
            idx = (starts[slot][:, None] + ar).clamp(max=n - 1)
            cand = order[idx]
            d = (pos[:, None, :] - pos[cand]).norm(dim=-1)
            d = torch.where(valid & (cand != torch.arange(n, device=dev)[:, None]), d,
                            torch.full_like(d, 1e9))
            allj = torch.cat([best_j, cand], 1)
            alld = torch.cat([best_d, d], 1)
            sel = torch.topk(-alld, k, dim=1).indices
            best_d = torch.gather(alld, 1, sel)
            best_j = torch.gather(allj, 1, sel)
        keep = best_d <= self.cutoff
        rows = torch.arange(n, device=dev)[:, None].expand_as(best_j)
        return rows[keep], best_j[keep]

    def _build(self, pos, live=None):
        # O(N^2) ONCE, at frame 0, in chunks. 20-40k particles is 1.6e9 pairs if done in one tensor, so
        # it is chunked; it runs once, and the alternative (a grid hash) is more code for a cost paid
        # a single time.
        n = pos.shape[0]
        # COMPACT TO THE LIVE NODES FIRST, then map back. Filtering afterwards is what the pairwise path
        # did and it was affordable there; for the cell list it is fatal. The unsecreted reserve is parked
        # at a SINGLE POINT (the tissue centre), so with a cell the size of the cutoff that one cell holds
        # every dormant particle -- 41,667 of them at reserve 12.5. The candidate block is sized by the
        # fullest cell, so it became (45000, 41667, 3): a 14 GiB allocation that killed six jobs on a 22
        # GiB card. Bonds can only ever form between live nodes, so they are the only ones to search.
        idx = None
        if live is not None:
            idx = live.nonzero(as_tuple=True)[0]
            sub = pos[idx]
        else:
            sub = pos
        m = sub.shape[0]
        k = min(self.max_nb, max(m - 1, 1))
        if m > 20000:
            i, j = self._neighbours_celllist(sub, k)
        else:
            i, j = self._build_pairwise(sub, k)
        if idx is not None:
            i, j = idx[i], idx[j]
        n2 = n + 1
        uk = torch.unique(torch.minimum(i, j) * n2 + torch.maximum(i, j))
        return (uk // n2).long(), (uk % n2).long()

    def _build_pairwise(self, pos, k):
        """The original chunked O(N^2) search, kept as the reference the cell list is checked against."""
        n = pos.shape[0]
        I, J = [], []
        step = max(1, 4096 ** 2 // max(n, 1))
        for a in range(0, n, step):
            b = min(n, a + step)
            d = (pos[a:b, None, :] - pos[None, :, :]).norm(dim=-1)
            d[:, :] = torch.where(d > 0, d, torch.full_like(d, 1e9))
            near = d <= self.cutoff
            k = min(self.max_nb, n - 1)
            idx = torch.topk(torch.where(near, -d, torch.full_like(d, -1e9)), k, dim=1).indices
            keep = torch.gather(near, 1, idx)
            rows = (torch.arange(a, b, device=pos.device)[:, None]).expand_as(idx)
            I.append(rows[keep]); J.append(idx[keep])
        return torch.cat(I), torch.cat(J)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        # BONDED INCREMENTALLY, NOT REBUILT. The first version recomputed the whole network whenever
        # secretion fired, and the run stalled at frame 1: `_build` is an O(n^2) neighbour search over
        # 30k particles, which is affordable once at seeding and not once per frame. Secretion adds a few
        # hundred particles at a time, so only THEY need neighbours -- and bonding only the new material
        # carries every existing rest length forward untouched, which the rebuild had to reconstruct by
        # key just to avoid erasing the remodelling history.
        # CROSSLINKS FORMED BY THE REWIRE OPERATOR, appended here. `bm_crosslink` decides
        # WHICH pairs bond -- that is a change to the edge set and belongs in a `rewire` operator, beside
        # `bm_unbond` -- while this operator stays the owner of the bond arrays. Same
        # split as secretion: the structural operator chooses, this one wires in.
        _rb = getattr(H, "membrane_rebond", None)
        if _rb is not None and self.i is not None and _rb[0].numel():
            H.membrane_rebond = None
            ai, aj = _rb
            self.i = torch.cat([self.i, ai]); self.j = torch.cat([self.j, aj])
            self.rest = torch.cat([self.rest, (pos[aj] - pos[ai]).norm(dim=-1).detach().clamp_min(1e-9)])
            self.alive = torch.cat([self.alive, torch.ones(ai.numel(), dtype=torch.bool, device=dev)])

        new = getattr(H, "membrane_new", None)
        if new is not None and self.i is not None and new.numel():
            H.membrane_new = None
            live = getattr(H, "membrane_alive", None)
            p = pos.detach()
            d = (p[new][:, None, :] - p[None, :, :]).norm(dim=-1)
            d[torch.arange(new.numel(), device=p.device), new] = 1e9
            if live is not None:
                d = torch.where(live[None, :], d, torch.full_like(d, 1e9))
            k = min(self.max_nb, p.shape[0] - 1)
            idx = torch.topk(-d, k, dim=1).indices
            keep = torch.gather(d, 1, idx) <= self.cutoff
            rows = new[:, None].expand_as(idx)
            ni, nj = rows[keep], idx[keep]
            n2 = p.shape[0] + 1
            key_new = torch.minimum(ni, nj) * n2 + torch.maximum(ni, nj)
            key_old = torch.minimum(self.i, self.j) * n2 + torch.maximum(self.i, self.j)
            fresh = torch.unique(key_new)
            fresh = fresh[~torch.isin(fresh, key_old)]
            if fresh.numel():
                ai, aj = (fresh // n2).long(), (fresh % n2).long()
                self.i = torch.cat([self.i, ai]); self.j = torch.cat([self.j, aj])
                self.rest = torch.cat([self.rest,
                                       (p[aj] - p[ai]).norm(dim=-1).clamp_min(1e-9)])
                self.alive = torch.cat([self.alive,
                                        torch.ones(ai.numel(), dtype=torch.bool, device=p.device)])
        if self.i is None:
            self.i, self.j = self._build(pos.detach(), getattr(H, "membrane_alive", None))
            self.rest = (pos[self.j] - pos[self.i]).norm(dim=-1).detach().clamp_min(1e-9)
            self.alive = torch.ones_like(self.rest, dtype=torch.bool)
            if self.i.numel() == 0:
                raise RuntimeError(
                    f"bm_bond: ZERO bonds among {pos.shape[0]} particles at cutoff "
                    f"{self.cutoff:g}. A membrane with no crosslinks is not a membrane -- it is stiff "
                    f"dust, and every downstream fragmentation number would be vacuous. Either the "
                    f"shell was seeded at the wrong scale or the cutoff is below the particle spacing "
                    f"(~sqrt(4*pi*R^2/N)).")
            # LIVE nodes, not the whole set. Dividing by all 45,000 -- of which 41,000 are unsecreted
            # reserve parked at the centre -- gave 0.42 bonds per node instead of 5.6, a ceiling 13x too
            # high, and the guard passed a run that returned an infinite strain. A denominator that
            # includes material which does not exist yet is not a coordination number.
            _live = getattr(H, "membrane_alive", None)
            _n = int(_live.sum()) if _live is not None else pos.shape[0]
            if self.guard_dt == "spec":
                self._check_stability(self.i.numel() / max(_n, 1) * 2.0,
                                      dt_frame=float(getattr(H, "dt", 1.0) or 1.0))
            else:
                self._check_stability(self.i.numel() / max(_n, 1) * 2.0)
            print(f"[bm_bond] {self.i.numel()} bonds on {pos.shape[0]} particles "
                  f"({self.i.numel() / max(pos.shape[0], 1):.1f} per particle), k={self.k:g}, "
                  f"cutoff={self.cutoff:g}", flush=True)
        d = pos[self.j] - pos[self.i]
        L = d.norm(dim=-1).clamp_min(1e-9)
        strain = (L - self.rest) / self.rest
        # HOOKEAN IN THE EXTENSION, NOT IN THE STRAIN. `k * strain` is `k * (L - rest) / rest`, and
        # `rest` is the particle spacing -- a small number -- so writing the force that way scales
        # the declared stiffness by 1/rest, a factor of several hundred at a typical spacing. One
        # percent of strain then produces an acceleration that moves a particle several times its
        # own spacing in a frame, and the sheet overshoots, oscillates and tears itself apart
        # before the tissue has grown into it. The BREAKING criterion stays relative, since strain
        # is the right dimensionless failure measure; only the force is extension-based.
        # THE CORSET. `aniso` makes crosslinks stiffer around the girth than along the long axis, which
        # is the "molecular corset" idea: a sheet that resists circumferential expansion more than
        # meridional expansion should squeeze the middle and push growth into the ends.
        #
        # A bond's orientation is measured against the local parallel (the circle of latitude): a bond
        # lying along it is circumferential and gets the full factor, one lying along the meridian gets
        # none. `aniso = 1` is isotropic and reproduces every run to date bit-for-bit.
        kk = self.k
        if self.aniso != 1.0:
            mid = 0.5 * (pos[self.i] + pos[self.j]) - self.centre_t.to(pos)
            rad = mid / mid.norm(dim=1, keepdim=True).clamp_min(1e-12)
            axis = torch.tensor([0.0, 0.0, 1.0], device=pos.device, dtype=dt_)
            par = torch.cross(axis.expand_as(rad), rad, dim=1)          # local circle of latitude
            # ITS LENGTH IS sin(colatitude), AND THAT IS PHYSICS, NOT A NORMALISATION NUISANCE. The
            # circumferential direction is undefined at the poles, where the parallels shrink to a point.
            # Normalising with a clamp turned that near-zero vector into a unit vector of NOISE, so polar
            # bonds drew a random stiffness boost instead of none -- and the recorded hoop tension came
            # out HIGHER at the poles than the equator (8.63 against 7.01), which is a corset backwards.
            #
            # Keeping the length as a weight makes the anisotropy vanish smoothly toward the poles, which
            # is what a corset does: it grips the girth and nothing at the ends.
            sin_th = par.norm(dim=1, keepdim=True)
            par = par / sin_th.clamp_min(1e-12)
            circ = ((d / L[:, None]) * par).sum(1).abs() * sin_th[:, 0]
            kk = self.k * (1.0 + (self.aniso - 1.0) * circ)
        if self.k_block:
            if self.k_block not in lvl.state_schema:
                raise ValueError(f"bm_bond: k_block {self.k_block!r} is not a block of {self.at!r}")
            _mb = lvl.get(self.k_block)[:, 0]
            kk_base = kk
            kk = kk * (0.5 * (_mb[self.i] + _mb[self.j])).clamp(0.0, self.k_block_cap)
            # THE NODE'S OWN CEILING (exp 11 Finding 149): an overdamped explicit node is stable while
            # deg_n k + k_contact < 2 gamma / dt, and the DEGREE is the node's, not the mean -- secretion keeps
            # bonding new nodes to old ones, and at 2x stiffness a node with 12 bonds (12 x 0.2 + 0.5 = 2.9) flung
            # itself to 1e19 at frame 156 of BB-94 while the mean (6.4) passed the guard. The scaled stiffness
            # of a bond is held to 0.9 of what BOTH its nodes can carry; never below the base k, so a run at mass 1
            # is unchanged; the share of bonds held is `H.bm_k_capped_frac`.
            if self.gamma > 0:
                _dtf = float(getattr(H, "dt", 1.0) or 1.0)
                _ones = torch.ones(int(self.alive.sum()), device=dev, dtype=dt_)
                _deg = torch.zeros(pos.shape[0], device=dev, dtype=dt_)
                _deg.index_add_(0, self.i[self.alive], _ones).index_add_(0, self.j[self.alive], _ones)
                _kn = 0.9 * (2.0 * self.gamma / _dtf - self.k_adh_hint) / _deg.clamp_min(1.0)
                _lim = torch.maximum(torch.minimum(_kn[self.i], _kn[self.j]),
                                     kk_base if torch.is_tensor(kk_base) else torch.full_like(kk, float(kk_base)))
                H.bm_k_capped_frac = float(((kk > _lim) & self.alive).sum()) / max(int(self.alive.sum()), 1)
                kk = torch.minimum(kk, _lim)
        f = (kk * (L - self.rest) * self.alive.to(dt_))[:, None] * (d / L[:, None])
        # the sheet's own hoop tension, by direction -- this is what a corset would press with, and it is
        # what the growth gate can read in the next pass. Without it the corset cannot reach the tissue.
        if self.record_hoop:
            th = torch.acos((mid_u := (0.5 * (pos[self.i] + pos[self.j]) - self.centre_t.to(pos)))[:, 2]
                            / mid_u.norm(dim=1).clamp_min(1e-12)).clamp(0, math.pi)
            nb_ = 32
            bin_ = (th / math.pi * nb_).long().clamp(0, nb_ - 1)
            ten = (kk * (L - self.rest)).clamp_min(0.0) * self.alive.to(dt_)
            acc_ = torch.zeros(nb_, device=pos.device, dtype=dt_).index_add_(0, bin_, ten)
            cnt_ = torch.zeros(nb_, device=pos.device, dtype=dt_).index_add_(0, bin_, self.alive.to(dt_))
            HOOP_TRACE.append((acc_ / cnt_.clamp_min(1.0)).detach().cpu().numpy())
        acc = torch.zeros_like(pos)
        acc.index_add_(0, self.i, f)
        acc.index_add_(0, self.j, -f)
        # F/gamma, so the emitted quantity is a VELOCITY, not an acceleration. gamma = 0 leaves the
        # inertial path untouched, so the two can be compared at the same operating point.
        if self.gamma > 0:
            acc = acc / self.gamma
        # per-particle strain for the renderer: mean |strain| over its live bonds
        s_abs = (strain.abs() * self.alive.to(dt_))
        cnt = torch.zeros(pos.shape[0], device=dev, dtype=dt_)
        tot = torch.zeros(pos.shape[0], device=dev, dtype=dt_)
        for a, b in ((self.i, self.j), (self.j, self.i)):
            cnt.index_add_(0, a, self.alive.to(dt_))
            tot.index_add_(0, a, s_abs)
        MEMBRANE_STRAIN.append((tot / cnt.clamp_min(1.0)).detach().to("cpu", torch.float16).numpy())
        H.membrane_bonds = (self.i, self.j, self.rest, self.alive)     # for the break operator
        H.__dict__["_bm_bond_op"] = self         # the crosslink operator reuses this neighbour search
        _f = int(getattr(H, "frame", -1) or -1)
        if self.snapshot_every > 0 and _f >= 0 and _f % self.snapshot_every == 0:
            _k = self.alive
            BOND_SNAPSHOTS.append((_f,
                                   self.i[_k].detach().cpu().numpy().astype("int32"),
                                   self.j[_k].detach().cpu().numpy().astype("int32"),
                                   strain[_k].detach().cpu().numpy().astype("float16")))
        return {lvl.name: acc}


@register_operator("bm_unbond", family="topology", set="particle", kind="rewire", title="break overstretched crosslinks", equation=r"""$$\text{bond } e \text{ dies when } \frac{L_e-L^0_e}{L^0_e}>\varepsilon_b$$""")
class BasementMembraneBondBreak(Structural):
    r"""breaks over-strained crosslinks every `components_every` frames, so fragmentation is an emergent, measurable event; what is reported is whether the sheet is still one piece.
    basement_membrane -> basement_membrane (rewire): reads pos and the bond list, marks bonds dead; records the connected components.

        \text{bond } e \text{ dies when } \frac{L_e-L^0_e}{L^0_e}>\varepsilon_b
        L_e, L^0_e     the bond's length and rest length
        \varepsilon_b  `break_strain`, the fractional extension a crosslink survives

    Reference: Plexus (this work); failure of a crosslinked network at a critical extension."""
    PARAM_UNITS = {'break_strain': 'fraction', 'components_every': 'frames'}
    SYMBOLS = {'L_e, L^0_e': "the bond's length and rest length", '\\varepsilon_b': '`break_strain`, the fractional extension a crosslink survives'}

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["fragmentation", "crosslink_failure", "basement_membrane"]
    PARAM_ROLES = {"break_strain": "bond_failure_strain", "every": "check_period", 'components_every': 'frames_between_checks'}
    REFERENCE = "Plexus (this work); failure criterion in the spirit of discrete-element crosslink models."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "basement_membrane_particle")
        self.break_strain = float(params.get("break_strain", 0.35))
        self.components_every = int(params.get("components_every", 40))
        self._k = 0

    def forward(self, H, mask=None):
        b = getattr(H, "membrane_bonds", None)
        if b is None:
            return {}
        i, j, rest, alive = b
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        L = (pos[j] - pos[i]).norm(dim=-1).clamp_min(1e-9)
        strain = (L - rest) / rest
        _lv = getattr(H, "membrane_alive", None)
        self._n_live = int(_lv.sum()) if _lv is not None else pos.shape[0]
        broke = alive & (strain > self.break_strain)
        n_broke = int(broke.sum())
        if n_broke:
            alive &= ~broke
        self._k += 1
        # COMPUTED PERIODICALLY, BUT ALSO CARRIED FORWARD -- and that second half is the whole point.
        # Computing it only on the frames that are multiples of `components_every` and leaving the
        # rest NaN would make the LAST row -- which is what an analysis reads -- NaN unless the
        # frame count happens to be a multiple of the period. The connectivity would then be
        # silently dropped from every conclusion that claimed to rest on it. A metric that
        # is NaN wherever it is read is not a metric.
        _last = self._k >= int(getattr(H, "n_frames", 0) or 0) - 1
        if self.components_every > 0 and (self._k % self.components_every == 0 or _last):
            # NORMALISED BY THE LIVE SHEET, not by the whole set. Dividing by pos.shape[0] counted the
            # unsecreted reserve as membrane that had failed to connect, so `lcc` simply tracked
            # n0/N_total: 0.769 / 0.275 / 0.141 at reservoirs of 45k / 135k / 270k is 0.769 x 45/N, and
            # says nothing about whether the sheet is in one piece.
            _n = int(getattr(self, "_n_live", pos.shape[0])) or pos.shape[0]
            self._frac = self._largest_component(i[alive], j[alive], pos.shape[0], denom=_n)
        frac = getattr(self, "_frac", float("nan"))
        # MEAN DEGREE, the quantity that says whether this is a sheet at all. Central-force rigidity
        # percolation in 2D needs z of about 4, so a network below that is not a sheet at all --
        # however many bonds it still has. Reporting z makes an under-connected network unmissable
        # where a bond count alone reads as an intact sheet whose taut bonds happen to have broken.
        n_live = int(getattr(self, "_n_live", pos.shape[0])) or pos.shape[0]
        z = 2.0 * int(alive.sum()) / max(n_live, 1)
        BOND_TRACE.append((int(alive.sum()), n_broke, float(strain[alive].abs().mean())
                           if bool(alive.any()) else 0.0, frac, z))
        return {}

    @staticmethod
    def _largest_component(i, j, n, iters=64, denom=None):
        """Largest connected component as a fraction of the sheet, by label propagation.

        Iterative rather than a union-find: it is a diagnostic run every 40 frames on the GPU where the
        data already is, and 64 min-scatter sweeps reach the diameter of a shell mesh. Under-converged
        would REPORT MORE components than exist, so the failure direction is conservative -- it cannot
        make a fragmented sheet look intact.
        """
        if i.numel() == 0:
            return 0.0 if n else float("nan")
        lab = torch.arange(n, device=i.device)
        for _ in range(iters):
            prev = lab
            lab = lab.clone().scatter_reduce_(0, i, lab[j], reduce="amin")
            lab = lab.scatter_reduce_(0, j, lab[i], reduce="amin")
            if torch.equal(lab, prev):
                break
        return float(torch.bincount(lab).max().item() / max(denom or n, 1))


RELEASE_TRACE: list = []                # (frame, nodes released this frame, live nodes left)


def release_nodes(node_alive, bi, bj, bond_alive, dose, c_node, c_face, leader_face, rate, min_degree=0):
    """One frame of `bm_unbond[model: release]`, in place, on torch tensors. Returns the released mask.

        dose_n  += rate         for every live node n in contact with a LEADER cell's basal face this frame
        release n  when  dose_n >= 1,  or when it keeps fewer than `min_degree` live bonds (min_degree > 0)

    A released node is what `bm_seed[live]`'s declared hole is: not alive (so no contact acts through it
    and no crosslink reforms to it) and its bonds dead. `c_node` / `c_face` are the frame's contact list
    (`basal_contacts[set]` of `bm_contact[live]`): membrane node, and the tissue face it presses on."""
    dev = node_alive.device
    released = torch.zeros_like(node_alive)
    if rate > 0 and c_node.numel():
        lf = leader_face.to(dev)[c_face.long()]
        hit = c_node.long()[lf]
        if hit.numel():
            touched = torch.zeros_like(node_alive)
            touched[hit] = True
            dose[touched & node_alive] += float(rate)
        released |= node_alive & (dose >= 1.0)
    if min_degree > 0:
        deg = torch.zeros(node_alive.shape[0], dtype=torch.long, device=dev)
        deg.index_add_(0, bi[bond_alive].long(), torch.ones_like(bi[bond_alive], dtype=torch.long))
        deg.index_add_(0, bj[bond_alive].long(), torch.ones_like(bj[bond_alive], dtype=torch.long))
        released |= node_alive & (deg < int(min_degree))
    if bool(released.any()):
        node_alive &= ~released
        bond_alive &= ~(released[bi.long()] | released[bj.long()])
    return released


@register_operator("bm_unbond", model="release", family="topology", set="particle", kind="rewire", title="break crosslinks and release membrane", equation=r"""$$\mathrm{dose}_n\mathrel{+}=r\ \ \text{while node } n \text{ touches a leader cell},\qquad \text{released: } \mathrm{dose}_n\ge 1\ \ \text{or}\ \ \deg(n)<d_{\min}$$""")
class BasementMembraneRelease(BasementMembraneBondBreak):
    r"""the `release` model of bm_unbond: the parent's strain failure, and then nodes leave the sheet, degraded where leader cells press on them or stranded with too few crosslinks, so a tear becomes a hole during the run.
    basement_membrane -> basement_membrane (rewire): reads the frame's contact list (after bm_contact[live]) and the cells' `leader_block`; marks released nodes dead.

        \mathrm{dose}_n\mathrel{+}=r\ \ \text{while node } n \text{ touches a leader cell},\qquad \text{released: } \mathrm{dose}_n\ge 1\ \ \text{or}\ \ \deg(n)<d_{\min}
        dose_n         node n's accumulated protease dose
        r              `rate`, the dose per frame under a leader cell
        \deg(n)        the node's live bonds
        d_{\min}       `min_degree` (0 = off)

    Reference: Cheung, K. J. et al. (2013). Collective invasion in breast cancer requires a conserved basal epithelial program. Cell 155:1639-1651."""
    PARAM_UNITS = {'min_degree': 'count', 'rate': 'fraction'}
    SYMBOLS = {'dose_n': "node n's accumulated protease dose", 'r': '`rate`, the dose per frame under a leader cell', '\\deg(n)': "the node's live bonds", 'd_{\\min}': '`min_degree` (0 = off)'}
    MECHANISM_TAGS = BasementMembraneBondBreak.MECHANISM_TAGS + ["membrane_degradation", "leader_cells"]
    PARAM_ROLES = {**BasementMembraneBondBreak.PARAM_ROLES, "rate": "degradation_dose_per_contact_frame",
                   "min_degree": "fewest_live_bonds_a_node_keeps", "leader_block": "leader_label_block",
                   "surface": "tissue_vertex_set", "cell_set": "tissue_cell_set"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.rate = float(params.get("rate", 0.0))
        self.min_degree = int(params.get("min_degree", 0))
        self.leader_block = str(params.get("leader_block", "mutant"))
        self.surface = str(params.get("surface", "vertex"))
        self.cell_set = str(params.get("cell_set", "cell"))

    def forward(self, H, mask=None):
        out = super().forward(H, mask)
        b = getattr(H, "membrane_bonds", None)
        alive = getattr(H, "membrane_alive", None)
        if b is None or alive is None:
            return out
        bi, bj, _rest, bond_alive = b
        dose = H.__dict__.get("_bm_release_dose")
        if dose is None or dose.shape[0] != alive.shape[0]:
            dose = H._bm_release_dose = torch.zeros(alive.shape[0], dtype=torch.float32, device=alive.device)
        c_node = c_face = torch.zeros(0, dtype=torch.long, device=alive.device)
        leader = torch.zeros(0, dtype=torch.bool, device=alive.device)
        m = getattr(H.level(self.surface), "_mesh", None)
        if m is not None and self.rate > 0:
            C = (m.get("basal_contacts") or {}).get(self.at)
            nF = int(m["nF"])
            from plexus.operators.vertex_ops import cell_block
            lab = cell_block(H, self.cell_set, self.leader_block, nF)
            if C is not None and lab is not None:
                c_node, c_face = C["node"], C["face"]
                leader = torch.as_tensor(lab > 0.5, device=alive.device)
        rel = release_nodes(alive, bi, bj, bond_alive, dose, c_node, c_face,
                            leader if leader.numel() else torch.zeros(1, dtype=torch.bool, device=alive.device),
                            self.rate if leader.numel() else 0.0, self.min_degree)
        n_rel = int(rel.sum())
        if n_rel:
            oc = getattr(H.level(self.at), "occ", None)
            if oc is not None:
                oc[rel.to(oc.device)] = 0.0
        RELEASE_TRACE.append((int(getattr(H, "frame", -1) or -1), n_rel, int(alive.sum())))
        return out


@register_operator("bm_remodel", family="population", set="particle", kind="lateral",
                   equation=r"""$$L_e^{0}\leftarrow L_e^{0}+\big(L_e-L_e^{0}\big)\frac{\Delta t}{\tau}$$""", title="remodel membrane crosslinks")
class BasementMembraneRemodel(Lateral):
    r"""crosslink turnover: the rest lengths creep toward the current ones over tau, a Maxwell-like relaxation of the reference state, so the sheet can grow rather than only stretch; fragmentation becomes a race of tau against growth.
    basement_membrane -> basement_membrane: rewrites the bonds' reference state in place; no delta.

        L_e^{0}\leftarrow L_e^{0}+\big(L_e-L_e^{0}\big)\frac{\Delta t}{\tau}
        L^0_e          the bond's rest length
        L_e            its current length
        \Delta t       the frame step
        \tau           `tau`, the turnover time, frames (modulated by `tau_block`, `tau_power`, between `tau_min_frac` and `tau_max_frac` of tau); `cap` bounds the change per frame

    Reference: Ku, H.-Y. et al. (2023). Basement membrane mechanics regulates matrix metalloproteinase expression. Dev. Cell 58:211-223; Villeneuve, C. et al. (2024). Proteolytic softening releases pressure and permits local division. Nature."""
    PARAM_UNITS = {'cap': 'fraction', 'l_star': 'length', 'mesh_w': 'length', 'target': 'dimensionless', 'tau': 'frames', 'tau_max_frac': 'fraction', 'tau_min_frac': 'fraction', 'tau_power': 'dimensionless'}
    SYMBOLS = {'L^0_e': "the bond's rest length", 'L_e': 'its current length', '\\Delta t': 'the frame step', '\\tau': '`tau`, the turnover time, frames (modulated by `tau_block`, `tau_power`, between `tau_min_frac` and `tau_max_frac` of tau); `cap` bounds the change per frame'}

    EMIT = None                     # rewrites the bonds' reference state; no delta
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["basement_membrane", "crosslink_turnover", "stress_relaxation",
                      "matrix_remodelling"]
    PARAM_ROLES = {"tau": "turnover_timescale_in_frames", "cap": "max_rest_growth_per_frame", 'l_star': 'reference_length_of_the_mesh_term', 'mesh_w': 'width_of_the_mesh_term', 'target': 'target_strain_the_rest_length_relaxes_to', 'tau_block': 'node_block_that_modulates_tau', 'tau_max_frac': 'upper_bound_of_the_modulated_tau', 'tau_min_frac': 'lower_bound_of_the_modulated_tau', 'tau_power': 'exponent_of_the_modulation'}
    REFERENCE = ("Ku, H.-Y. et al. (2023) Dev. Cell 58:211 (BM mechanics regulates MMP, MMP remodels "
                 "BM); Villeneuve, C. et al. (2024) Nat. Cell Biol. 26:207 (proteolytic softening "
                 "releases pressure and permits local division).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "basement_membrane_particle")
        self.tau = float(params.get("tau", 60.0))
        self.cap = float(params.get("cap", 0.02))
        self.target = str(params.get("target", "own")).lower()    # "own" | "mesh" | "fixed"
        self.mesh_w = float(params.get("mesh_w", 1.0))            # how far toward the common spacing
        # `tau_block` (default none; exp 11 Phase 3 B6e, 2026-09-29): each bond's turnover time scaled by the mean of
        # a width-1 block of its two nodes -- the membrane's mass `bm_M` -- raised to `tau_power` (default 1),
        # clipped to [`tau_min_frac`, `tau_max_frac`] (0.1, 10) of tau. A membrane the protease has thinned FORGETS
        # ITS STRAIN FASTER: under a long base tau the sheet holds the tissue, and yields where it is cut
        # (Villeneuve 2024: proteolytic softening releases pressure; Harunaga 2014: the tip's distensibility needs
        # protease). Absent, every bond keeps the one tau.
        self.tau_block = params.get("tau_block", None)
        self.tau_power = float(params.get("tau_power", 1.0))
        self.tau_min_frac = float(params.get("tau_min_frac", 0.1))
        self.tau_max_frac = float(params.get("tau_max_frac", 10.0))
        lst = params.get("l_star", 0.0)
        self._l_star = float(lst) if lst else None                # None = freeze it from frame 0
        self._said = False

    def forward(self, H, mask=None):
        b = getattr(H, "membrane_bonds", None)
        if b is None:
            return {}
        i, j, rest, alive = b
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        L = (pos[j] - pos[i]).norm(dim=-1).clamp_min(1e-9)
        # CAPPED PER FRAME. Without it a single violent frame -- a contact transient, an instability --
        # is written permanently into the reference state, and the sheet would remember a shock it should
        # have forgotten. The cap is a fraction of the current rest length.
        # TOWARD WHAT? `L` -- each bond's OWN current length -- is what this used to relax to, and that
        # turns out to be why the sheet never evens out. If every spring remembers its own length, the
        # network has no common preferred spacing, so a disordered configuration IS its equilibrium and
        # relaxation cannot improve on it. Measured at the end of a run: cv of the rest lengths is 0.521.
        # Anchors, crosslinking and deposition were all tested against the holes and none of them
        # mattered, because none of them touches where the disorder is stored.
        #
        # A real collagen IV network has a characteristic mesh size -- protomers are a defined length --
        # so `mesh` relaxes rest lengths toward the spacing the sheet's own density implies, which gives
        # the network something uniform to relax TO. `own` is the previous behaviour, kept because every
        # run to date used it.
        # `fixed` GOES ONE STEP FURTHER, AND THE STEP MATTERS. `mesh` relaxes toward the sheet's OWN
        # current mean, which is self-referential: if secretion under-delivers, the mean spacing grows
        # and the springs simply accept the sparser sheet. The holes get ratified rather than closed.
        # A collagen IV protomer has a defined length, so the mesh size is set by the MOLECULE, not by
        # whatever density happens to exist. Holding l* fixed makes N = 4*pi*R^2/l*^2 the number of nodes
        # the sheet needs, secretion the only thing that can supply it, and a shortfall visible as
        # tension and holes -- a result, not an artefact. l* is frozen from the frame-0 sheet, which is
        # the one configuration that was seeded evenly on purpose.
        if self.target == "fixed":
            if self._l_star is None:
                self._l_star = float(L[alive].mean()) if float(alive.sum()) > 0 else float(rest.mean())
                print(f"[bm_remodel] target l* = {self._l_star:.5g} (frozen from frame 0); "
                      f"the sheet must SECRETE to hold it as it grows", flush=True)
            H.membrane_l_star = self._l_star
            tgt = torch.full_like(L, self._l_star)
        elif self.target == "mesh":
            live_n = float(alive.to(rest.dtype).sum())
            l_star = L[alive].mean() if live_n > 0 else rest.mean()
            tgt = (1.0 - self.mesh_w) * L + self.mesh_w * l_star
        else:
            tgt = L
        tau = max(self.tau, 1e-9)
        if self.tau_block:
            if self.tau_block not in lvl.state_schema:
                raise ValueError(f"bm_remodel: tau_block {self.tau_block!r} is not a block of {self.at!r}")
            _mb = lvl.get(self.tau_block)[:, 0].to(rest.dtype)
            _m = (0.5 * (_mb[i] + _mb[j])).clamp_min(1e-6)
            tau = (self.tau * _m ** self.tau_power).clamp(self.tau_min_frac * self.tau, self.tau_max_frac * self.tau)
        d = ((tgt - rest) / tau).clamp(-self.cap * rest, self.cap * rest)
        rest += d * alive.to(rest.dtype)
        if not self._said:
            print(f"[bm_remodel] crosslink turnover tau={self.tau} frames "
                  f"(cap {self.cap:g} of rest per frame): the sheet forgets strain over tau, so "
                  f"fragmentation is a race between turnover and growth", flush=True)
            self._said = True
        return {}



# Two operators are deliberately absent. A grid boundary condition on the tissue -- overwriting
# grid-node velocity -- is kinematic, does not conserve momentum, discards the reaction, and leaves
# a standoff set by the B-spline stencil width rather than by anything physical; `integrin_track`
# is the constraint it should have been. A continuum strain readout on the sheet is a measurement
# of MPM state that nothing in this model reads, the mechanics being carried by the crosslinks.



@register_operator("bm_contact", family="boundary", set="particle", kind="lateral",
                   equation=r"""$$d_i=S\,R(\theta_i,\phi_i)+\text{standoff}-\lVert\mathbf x_i-\mathbf c\rVert,\qquad \mathbf a_i=k\,d_i\,\mathbf u_i$$""", title="keep the membrane out of the tissue")
class BasementMembraneContact(Lateral):
    r"""non-penetration between the sheet and the epithelium, imposed as a force on each node rather than as a boundary condition on the grid, against the recorded surface at a held standoff.
    basement_membrane -> basement_membrane: reads pos and the recorded surface, emits an acceleration (only where the node is inside the standoff).

        d_i=S\,R(\theta_i,\phi_i)+\text{standoff}-\lVert\mathbf x_i-\mathbf c\rVert,\qquad \mathbf a_i=k\,d_i\,\mathbf u_i
        d_i            how far node i is inside the standoff
        S              `scale`
        R(\theta_i, \phi_i) the recorded surface's radius in the node's direction
        standoff       `offset`, the clearance held
        x_i            the node's position
        c              `centre`
        a_i            the acceleration
        k              `k`, the contact stiffness, per time squared
        u_i            the node's unit direction

    Reference: Plexus (this work); the surface is Okuda, S. et al. (2018). Sci. Rep. 8:2386."""
    PARAM_UNITS = {'centre': 'length', 'damp': 'rate', 'k': 'acceleration', 'offset': 'length', 'overdamped_gamma': 'mass_per_time', 'scale': 'dimensionless'}
    SYMBOLS = {'d_i': 'how far node i is inside the standoff', 'S': '`scale`', 'R(\\theta_i, \\phi_i)': "the recorded surface's radius in the node's direction", 'standoff': '`offset`, the clearance held', 'x_i': "the node's position", 'c': '`centre`', 'a_i': 'the acceleration', 'k': '`k`, the contact stiffness, per time squared', 'u_i': "the node's unit direction"}

    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["surface"]
    MECHANISM_TAGS = ["basement_membrane", "contact", "non_penetration"]
    PARAM_ROLES = {"k": "contact_stiffness", "offset": "standoff_the_sheet_is_held_at",
                   "surface": "tissue_surface_radius_map", "scale": "surface_rescale", 'centre': 'centre_of_the_surface', 'damp': 'contact_dashpot_rate', 'overdamped_gamma': 'friction_of_the_overdamped_update'}
    REFERENCE = ("Yurchenco, P.D. (2011) Cold Spring Harb. Perspect. Biol. 3:a004911 (the basement "
                 "membrane is directly apposed to the basal cell surface).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "basement_membrane_particle")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self.k = float(params.get("k", 1.0e4))
        self.offset = float(params.get("offset", 0.0))
        # critical by default, as integrin_adhesion has always been: an undamped one-sided spring on a
        # particle with mass oscillates about the surface instead of resting on it.
        self.damp = float(params.get("damp", 2.0 * math.sqrt(max(float(params.get("k", 1.0e4)), 1e-12))))
        # overdamped: emit F/gamma as a VELOCITY. Then there is no oscillation and no dashpot -- the
        # dashpot only exists to damp an inertia the sheet should not have at Re ~ 1e-10.
        self.gamma = float(params.get("overdamped_gamma", 0.0))
        self.scale = float(params.get("scale", 1.0))
        import numpy as _np
        z = _np.load(str(params["surface"]))
        self.smap = torch.as_tensor(_np.asarray(z["smap"], _np.float32)) * self.scale
        self.T = int(self.smap.shape[0])
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        c = torch.tensor(self.centre, device=dev, dtype=dt_)
        f = int(getattr(H, "frame", 0) or 0)
        t = min(self.T - 1, max(0, f))
        M = self.smap[t].to(dev, dt_)
        nth, nph = M.shape
        d = pos - c
        r = d.norm(dim=1).clamp_min(1e-9)
        u = d / r[:, None]
        th = torch.acos(u[:, 2].clamp(-1.0, 1.0))
        ph = torch.atan2(u[:, 1], u[:, 0])
        # bilinear, so the surface the sheet rests on is smooth rather than a 32x64 staircase
        fth = (th / math.pi * nth - 0.5).clamp(0, nth - 1)
        fph = ((ph + math.pi) / (2 * math.pi) * nph - 0.5) % nph
        t0 = fth.floor().long().clamp(0, nth - 1); t1 = (t0 + 1).clamp(0, nth - 1)
        p0 = fph.floor().long() % nph; p1 = (p0 + 1) % nph
        wt = fth - t0.to(fth.dtype); wp = fph - p0.to(fph.dtype)
        R = ((M[t0, p0] * (1 - wp) + M[t0, p1] * wp) * (1 - wt)
             + (M[t1, p0] * (1 - wp) + M[t1, p1] * wp) * wt)
        # ONE-SIDED: only material that has entered the epithelium is pushed, and only outward.
        pen = (R + self.offset - r).clamp_min(0.0)
        acc = (self.k * pen)[:, None] * u
        # A ONE-SIDED SPRING ON A MASSIVE PARTICLE RINGS, AND ON THE OUTWARD SWING IT SEPARATES. Runs
        # 110/111 did exactly that: the sheet oscillated and left the surface (+0.1238 standoff, strain
        # collapsed to 0.05 from 2.37), because this operator emitted an acceleration with no dashpot
        # while `integrin_adhesion` beside it has defaulted to critical damping, 2*sqrt(k), all along.
        # At Re ~ 1e-10 the inertia has no physical basis in the first place, so the damping is not a
        # numerical patch: it is the term that makes the contact overdamped, which is the real regime.
        vel = lvl.get("vel") if "vel" in lvl.state_schema else None   # same test integrin uses
        if self.gamma > 0:
            acc = acc / self.gamma          # overdamped: the emitted quantity is a velocity
        elif vel is not None and self.damp > 0:
            acc = acc - self.damp * vel     # inertial path: critical damping
        alive = getattr(H, "membrane_alive", None)
        if alive is not None:
            acc = torch.where(alive[:, None], acc, torch.zeros_like(acc))
        CONTACT_TRACE.append((f, float(pen.max()), float((pen > 0).sum())))
        if not self._said:
            print(f"[bm_contact] per-particle non-penetration, k={self.k:.3g}, "
                  f"offset={self.offset:.4g}: {int((pen > 0).sum())} particles inside the surface at "
                  f"frame {f}", flush=True)
            self._said = True
        return {lvl.name: acc}



@register_operator("bm_contact", implementation="live", family="boundary", set="particle",
                   kind="lateral",
                   equation=r"""$$p_i=\max\!\big(0,\ \text{offset}-(\mathbf x_i-\mathbf x^{s}_i)\cdot\mathbf n\big),\qquad \mathbf F_i=k\,p_i\,\mathbf n,\qquad \dot{\mathbf x}_i=\mathbf F_i/\gamma,\qquad \mathbf f_v=-\textstyle\sum_i w_{iv}\mathbf F_i$$""", title="keep membrane out of the live tissue")
class BasementMembraneContactLive(Lateral):
    r"""non-penetration between the membrane and the live tissue's basal surface, both ways, normal only: the frame's contact list, the standoff force on the nodes, its reaction on the tissue's vertices, an optional tether, and the per-cell ligand density the tissue senses.
    basement_membrane_node -> the tissue mesh: builds the frame's contact list (the forces are evaluated inside cell_mechanics[apicobasal_contact]); publishes m["bm_ligand"] per cell.

        p_i=\max\!\big(0,\ \text{offset}-(\mathbf x_i-\mathbf x^{s}_i)\cdot\mathbf n\big),\qquad \mathbf F_i=k\,p_i\,\mathbf n,\qquad \dot{\mathbf x}_i=\mathbf F_i/\gamma,\qquad \mathbf f_v=-\textstyle\sum_i w_{iv}\mathbf F_i
        p_i            how far inside the standoff node i is
        x_i            the node's position
        x^s_i          the nearest point of the basal surface (a fan triangle, weights w_iv)
        n              the outward normal
        F_i            the force on the node
        k              `k`, force per unit penetration per node (`k_adh` the tether)
        \dot x_i       the node's overdamped velocity
        \gamma         `overdamped_gamma`, the node's drag
        f_v            the reaction on vertex v
        w_{iv}         the barycentric weights

    Reference: Chen, Z. et al. (2015). Comput. Methods Appl. Mech. Engrg. 293:1-19 (the particle-to-surface contact and its barycentric reaction); Yurchenco, P. D. (2011). Cold Spring Harb. Perspect. Biol. 3:a004911; Haigo, S. L. & Bilder, D. (2011). Science 331:1071-1074 (the follicle rotates inside its membrane)."""
    PARAM_UNITS = {'band': 'length', 'k': 'acceleration', 'k_adh': 'acceleration', 'margin_contact': 'length', 'offset': 'length', 'overdamped_gamma': 'mass_per_time'}
    SYMBOLS = {'p_i': 'how far inside the standoff node i is', 'x_i': "the node's position", 'x^s_i': 'the nearest point of the basal surface (a fan triangle, weights w_iv)', 'n': 'the outward normal', 'F_i': 'the force on the node', 'k': '`k`, force per unit penetration per node (`k_adh` the tether)', '\\dot x_i': "the node's overdamped velocity", '\\gamma': "`overdamped_gamma`, the node's drag", 'f_v': 'the reaction on vertex v', 'w_{iv}': 'the barycentric weights'}
    EMIT = "velocity"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["surface"]
    MECHANISM_TAGS = ["basement_membrane", "contact", "non_penetration", "two_way_coupling",
                      "anchorage_sensing"]
    PARAM_ROLES = {"k": "contact_stiffness_per_node", "offset": "standoff_the_sheet_is_held_at",
                   "band": "ligand_reach", "surface": "the_live_vertex_set",
                   "k_adh": "cell_matrix_tether_stiffness",
                   "side": "basal_membrane_outside_or_apical_inner_cells_inside", 'margin_contact': 'how_far_outside_the_standoff_a_node_still_counts', 'overdamped_gamma': 'node_drag', 'sep_block': 'vertex_block_of_the_apico_basal_separation'}
    REFERENCE = ("Chen, Z., Qiu, X., Zhang, X. & Lian, Y. (2015). Comput. Methods Appl. Mech. Engrg. "
                 "293:1-19; Yurchenco, P.D. (2011) Cold Spring Harb. Perspect. Biol. 3:a004911.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "bm_node")
        self.surface = str(params["surface"])
        self.sep_block = str(params.get("sep_block", "sep"))
        self.k = float(params.get("k", 0.5))
        _g = params.get("overdamped_gamma", None)
        self.gamma = None if _g is None else float(_g)
        self.offset = float(params.get("offset", 0.0))
        self.band = float(params.get("band", 0.3))
        self.margin_contact = float(params.get("margin_contact", 0.5))
        self.k_adh = float(params.get("k_adh", 0.0))
        self.side = str(params.get("side", "basal")).lower()
        if self.side not in ("basal", "apical"):
            raise ValueError(f"bm_contact[live]: side is basal or apical, got {self.side!r}")
        if self.side == "apical" and self.gamma is None:
            raise ValueError("bm_contact[live]: side apical needs `overdamped_gamma` -- the drag of the "
                             "set it holds inside the tissue (no `bm_bond` acts on it)")
        if self.k_adh > 0 and self.band > self.margin_contact:
            raise ValueError("bm_contact[live]: the tether's reach `band` must lie inside "
                             "`margin_contact`, or a tethered node could miss the frame's list")
        self._geo = _basal_lookup()
        self._said = False

    def _node_gamma(self, H):
        if self.side == "apical":                   # its own drag; bm_bond's belongs to the membrane
            return self.gamma
        bb = H.__dict__.get("_bm_bond_op", None)
        g_b = float(getattr(bb, "gamma", 0.0) or 0.0) if bb is not None else 0.0
        if self.gamma is None:
            if g_b <= 0:
                raise ValueError("bm_contact[live]: no node drag -- give `overdamped_gamma`, or "
                                 "schedule an overdamped `bm_bond` BEFORE this operator")
            self.gamma = g_b
        elif g_b > 0 and abs(g_b - self.gamma) > 1e-12 * max(g_b, 1.0):
            raise ValueError(f"bm_contact[live]: overdamped_gamma {self.gamma:g} differs from "
                             f"bm_bond's {g_b:g}; one node has one drag")
        return self.gamma

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        f = getattr(H, "frame", None)
        f = -1 if f is None else int(f)
        vel = torch.zeros_like(pos)
        apical = self.side == "apical"
        got = (_live_basal(H, self.surface, self.sep_block, self._geo, dev, dt_, side="apical") if apical
               else _live_basal(H, self.surface, self.sep_block, self._geo, dev, dt_))
        if got is None:
            return {self.at: vel}
        m, c, M = got
        nv, nF = M["nv"], int(M["nF"])
        gamma = self._node_gamma(H)
        alive = getattr(H, "membrane_alive", None) if not apical else (lvl.occ > 0)
        alive = (torch.ones(pos.shape[0], dtype=torch.bool, device=dev) if alive is None
                 else alive.to(dev))
        if mask is not None:
            alive = alive & (mask > 0)
        idx = alive.nonzero(as_tuple=True)[0]
        lig = torch.zeros(nF, device=dev, dtype=dt_)
        f64 = torch.float64
        C = dict(frame=f, set=self.at, gamma=gamma, k=self.k, offset=self.offset, nv=nv, nF=nF,
                 k_adh=self.k_adh, band_adh=self.band,
                 es=M["es"], ef=M["ef"], cnt=M["cnt"], node=idx[:0],
                 src=M["es"][:0], trgt=M["et"][:0], face=M["ef"][:0],
                 w=torch.zeros(0, 3, dtype=f64, device=dev), n=torch.zeros(0, 3, dtype=f64, device=dev),
                 y=torch.zeros(0, 3, dtype=f64, device=dev))
        pmax, n_list = 0.0, 0
        if idx.numel():
            hit, tri, w, n_hat, gap = _nearest_faces(M, c, pos[idx])
            idx, tri, w, n_hat, gap = idx[hit], tri[hit], w[hit], n_hat[hit], gap[hit]
            if apical:                              # the normal INTO the lumen, the height along it
                n_hat, gap = -n_hat, -gap
            bound = gap <= (self.offset + self.band)
            lig.index_add_(0, M["ef"][tri][bound], torch.ones_like(gap[bound]))
            if gap.numel():
                pmax = float((self.offset - gap).clamp_min(0.0).max())
            keep = gap <= (self.offset + self.margin_contact)
            idx, tri, w, n_hat = idx[keep], tri[keep], w[keep], n_hat[keep]
            w64 = w.to(f64)
            w64 = torch.cat([1.0 - w64[:, 1:2] - w64[:, 2:3], w64[:, 1:3]], 1)
            C.update(node=idx, src=M["es"][tri], trgt=M["et"][tri], face=M["ef"][tri], w=w64,
                     n=n_hat.to(f64), y=pos[idx].detach().to(f64))
            n_list = int(idx.numel())
        bc = m.get("basal_contacts")
        if not isinstance(bc, dict):
            bc = m["basal_contacts"] = {}
        bc[self.at] = C
        if apical:                                  # the ligand density and the trace are the membrane's
            C["side"] = "apical"
            if not self._said:
                self._said = True
                print(f"[bm_contact/live] set `{self.at}` held INSIDE the apical surface of "
                      f"`{self.surface}`: k={self.k:g}, gamma={gamma:g}, offset={self.offset:g}; frame {f}: "
                      f"{n_list} nodes in the contact list, deepest {pmax:.3g} past the standoff", flush=True)
            return {self.at: vel}
        # THE LIGAND DENSITY: bound nodes per unit BASAL area of each cell, so a big cell is not
        # read as a well-anchored one.
        area = 0.5 * torch.cross(M["B"] - M["A"], M["C"] - M["A"], dim=1).norm(dim=1)
        farea = torch.zeros(nF, device=dev, dtype=dt_).index_add_(0, M["ef"], area)
        m["bm_ligand"] = lig / farea.clamp_min(1e-12)
        m["bm_ligand_frame"] = f
        CONTACT_TRACE.append((f, pmax, float(n_list)))
        if not self._said:
            self._said = True
            print(f"[bm_contact/live] membrane `{self.at}` against the basal surface of "
                  f"`{self.surface}`: k={self.k:g}, k_adh={self.k_adh:g}, gamma={gamma:g}, "
                  f"offset={self.offset:g}, band={self.band:g}; frame {f}: {n_list} nodes in the contact list, deepest "
                  f"{pmax:.3g} inside the standoff", flush=True)
        return {self.at: vel}


# ---------------------------------------------------------------------------------------------------
# HEMIDESMOSOMES AS ENTITIES, not as a boolean field on the membrane.
#
# A tether on every membrane particle is the approximation; the biology is punctate. Integrin a6b4
# clusters into discrete hemidesmosome plaques on the basal surface, with basement membrane spanning
# freely between them, and each plaque has a LIFETIME -- it forms, bears load, ruptures when the load
# exceeds what its bonds hold, and re-forms elsewhere over minutes. None of that is expressible as a
# mask: a mask has no age, no load, and nothing to break. As a set it is all three, and the biology
# becomes reachable -- an integrin b1 knockout is FEWER ADHESIONS, not a smaller spring constant.
#
# The set carries, per adhesion: the direction on the surface where it sits (frozen at formation, so it
# rides the growing surface), the index of the membrane particle it binds, and its age. `adhesion_pull`
# is the force, `adhesion_turnover` is the rewire that breaks and re-forms.
# ---------------------------------------------------------------------------------------------------

@register_operator("bm_secrete", family="population", set="particle", kind="structural", title="secrete new membrane as tissue grows", equation=r"""$$n_{new}(t)\propto A(t)-A(t-1),\qquad \mathbf x_{new}\ \text{in the sparsest directions of the sheet}$$""")
class BasementMembraneSecrete(Structural):
    r"""lays down new membrane as the surface it sits on grows, activating dormant nodes in the sparsest directions of the current sheet, so the sheet gains material rather than only thinning (the slip / tear / apparent-hold trilemma is about material).
    basement_membrane -> basement_membrane: activates dormant nodes at the surface, gives them positions, and lets bm_bond bond them in.

        n_{new}(t)\propto A(t)-A(t-1),\qquad \mathbf x_{new}\ \text{in the sparsest directions of the sheet}
        n_{new}(t)     the nodes activated this frame
        A(t)           the surface's area
        x_{new}        their positions

    Reference: Plexus (this work); continuous secretion of basement-membrane components by the epithelium they underlie."""
    PARAM_UNITS = {'cand_mult': 'dimensionless', 'centre': 'length', 'deposit': 'fraction', 'rate': 'fraction', 'relax_every': 'frames', 'relax_new': 'count', 'relax_sweeps': 'count', 'targeted': 'dimensionless'}
    SYMBOLS = {'n_{new}(t)': 'the nodes activated this frame', 'A(t)': "the surface's area", 'x_{new}': 'their positions'}
    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["secretion", "material_addition", "load_directed_deposition"]
    PARAM_ROLES = {"rate": "max_fraction_secreted_per_frame", "targeted": "load_directed_vs_uniform",
                   "centre": "tissue_centre", 'cand_mult': 'candidate_directions_per_new_node', 'deposit': 'fraction_of_the_area_gain_covered', 'relax_every': 'frames_between_relaxations', 'relax_new': 'relaxation_sweeps_of_a_new_node', 'relax_sweeps': 'relaxation_sweeps'}
    REFERENCE = ("Plexus (this work). Continuous basement-membrane deposition during epithelial growth: "
                 "Ku & Bilder (2023) Dev. Cell 58:522; Toepfer et al. (2022) Development 149:dev200456.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "basement_membrane_particle")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self.rate = float(params.get("rate", 0.02))
        self.targeted = self.tunable(params.get("targeted"), 1.0, tensor=True)
        self.relax_new = int(params.get("relax_new", 4))
        self.deposit = str(params.get("deposit", "uniform")).lower()  # uniform | gaps | parent
        self.cand_mult = int(params.get("cand_mult", 12))
        self.relax_every = int(params.get("relax_every", 20))     # sweep the whole sheet this often
        self.relax_sweeps = int(params.get("relax_sweeps", 3))
        self._r0 = None
        self._n0 = None

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        live = getattr(H, "membrane_alive", None)
        if live is None:
            return {}
        n_live = int(live.sum())
        n_tot = pos.shape[0]
        if n_live >= n_tot:
            return {}                                  # the reserve is spent; nothing left to lay down

        c = torch.tensor(self.centre, device=dev, dtype=dt_)
        r = (pos[live] - c).norm(dim=1)
        R = float(r.mean())
        if self._r0 is None:
            self._r0, self._n0 = R, n_live
            return {}

        # AREAL DENSITY IS THE SETPOINT, not particle count: the sheet should be as thick per unit area
        # at the end as at the start, and area goes as R^2.
        want = min(n_tot, int(round(self._n0 * (R / max(self._r0, 1e-9)) ** 2)))
        add = want - n_live
        add = min(add, max(1, int(self.rate * n_live)))
        if add <= 0:
            return {}

        # BONDS ARE ONLY NEEDED BY `parent`, which weights sites by the load on each particle's own
        # crosslinks. `uniform` and `gaps` never look at them -- so returning early when there is no
        # network would disable secretion for the entire continuum membrane, silently. The live
        # particle count then stays frozen at its seeded value, which is what a rate parameter looks
        # like when nothing reads it: the sheet holds its radius while the tissue grows past it,
        # leaving the membrane INSIDE the spheroid.
        bonds = getattr(H, "membrane_bonds", None)
        if bonds is None and self.deposit not in ("uniform", "gaps"):
            return {}
        if bonds is not None:
            bi, bj, brest, balive = bonds
            d = (pos[bj] - pos[bi]).norm(dim=1).clamp_min(1e-9)
            strain = ((d - brest) / brest) * balive.to(dt_)

        # DEPOSIT AGAINST PARTICLES, NOT AGAINST BONDS -- and this is the correction that matters.
        # Two versions failed before this one. Taking the globally most strained BONDS is winner-take-all:
        # this matrix has a dense polar cone, so every new particle landed there and the rest of the
        # sphere stayed bare. Sampling bonds in proportion to strain fixed the aim and not the outcome,
        # because the deeper fault is the same in both: a site is a MIDPOINT OF AN EXISTING BOND. A patch
        # that has lost its crosslinks therefore cannot receive new material, so it dilutes further and
        # loses more -- a one-way ratchet into bare, whose end state is a strain of exactly zero
        # away from the pole: not a small number, zero, meaning no particle there has a bond left.
        #
        # Cells are what secrete, and cells are everywhere on the surface. So the site is a LIVE PARTICLE,
        # chosen with weight (local sparsity x load), and the new material goes beside it, tangentially,
        # about one target spacing away. An isolated particle with no bonds at all is then still a place
        # the membrane can be repaired, which is what makes coverage recoverable instead of a ratchet.
        idx_live = live.nonzero(as_tuple=True)[0]
        sub = pos[idx_live]

        if self.deposit in ("uniform", "gaps"):
            # WHERE THE CELLS ARE, NOT WHERE THE HOLES ARE. Cells secrete basement membrane basally,
            # into the patch of surface directly beneath themselves: laminin polymerises on the cell
            # surface where integrin and dystroglycan nucleate it, collagen IV assembles onto that and is
            # crosslinked in place. So deposition is LOCAL TO EACH CELL and uniform per unit basal area.
            #
            # That rules out both of the rules tried here. Placing new material beside an existing node
            # is a random walk and clumps. Placing it in the largest gap is worse as biology, however
            # well it packs: it needs a global view of where every hole is, and a cell has none -- it
            # cannot know whether there is a gap two cells away.
            #
            # Uniform-per-area is what a sheet of secreting cells does, and it fills a hole at exactly
            # the rate it adds to anywhere else, which is the honest mechanism. Evening out is then the
            # job of the network itself -- crosslink turnover and rearrangement -- not of aimed
            # deposition. `deposit="gaps"` is kept as the non-biological upper bound on how well
            # targeted placement could do.
            n_cand = max(add * self.cand_mult, add)
            cd = torch.randn(n_cand, 3, generator=None, device=dev, dtype=dt_)
            cd = cd / cd.norm(dim=1, keepdim=True).clamp_min(1e-12)
            best = torch.full((n_cand,), 1e9, device=dev, dtype=dt_)
            near = torch.zeros(n_cand, dtype=torch.long, device=dev)
            u_live = (sub - c) / (sub - c).norm(dim=1, keepdim=True).clamp_min(1e-12)
            blk = 4096
            for a0 in range(0, n_cand, blk):
                b0 = min(n_cand, a0 + blk)
                dd = 1.0 - (cd[a0:b0] @ u_live.T).clamp(-1, 1)     # 1 - cos, monotone in angle
                v, i_ = dd.min(dim=1)
                best[a0:b0] = v
                near[a0:b0] = i_
            if self.deposit == "uniform":
                # every candidate direction is equally acceptable: a cell secretes under itself, and the
                # cells tile the surface, so the deposition field is flat per unit area
                pick_c = torch.randperm(n_cand, device=dev)[:min(add, n_cand)]
            else:
                # the non-biological bound: purely how far a candidate is from anything already there
                pick_c = torch.topk(best, min(add, n_cand)).indices
            add = int(pick_c.numel())
            slot = (~live).nonzero(as_tuple=True)[0][:add]
            add = int(slot.numel()); pick_c = pick_c[:add]
            rad = (sub - c).norm(dim=1)[near[pick_c]][:, None]
            pos[slot] = c + cd[pick_c] * rad
            v = lvl.get("vel") if "vel" in lvl.state_schema else None
            if v is not None:
                v[slot] = v[idx_live][near[pick_c]]
            m = getattr(lvl, "mass", None)
            if m is not None:
                m[slot] = m[idx_live][near[pick_c]]
            F = getattr(lvl, "F", None)
            if F is not None:
                F[slot] = torch.eye(F.shape[-1], device=dev, dtype=F.dtype)
            Cb = getattr(lvl, "C", None)
            if Cb is not None:
                Cb[slot] = 0.0
            live = live.clone(); live[slot] = True
            H.membrane_alive = live
            H.membrane_new = slot
            SECRETE_TRACE.append((n_live, add, int(live.sum()), R))
            return {}

        # ---- deposit == "parent": the original rule, kept as the contrast ------------------------------
        # Its parent selection lived in the code the uniform/gaps branch replaced, so it has to be
        # restated here. Weighted by local sparsity and by load, exactly as before -- 79 exists to say
        # what changes when only the ANCHORS move, so its deposition has to be the old one unchanged.
        want_sp = math.sqrt(4.0 * math.pi * R * R / max(n_live, 1))
        pl = torch.zeros(n_tot, device=dev, dtype=dt_)
        pc = torch.zeros(n_tot, device=dev, dtype=dt_)
        for a_, b_ in ((bi, bj), (bj, bi)):
            pl.index_add_(0, a_, strain.clamp_min(0.0))
            pc.index_add_(0, a_, balive.to(dt_))
        pl = pl / pc.clamp_min(1.0)
        w = 1.0 + self.targeted * pl[idx_live]
        k_ = min(add, idx_live.numel())
        if float(w.sum()) <= 0:
            pick = torch.randperm(idx_live.numel(), device=dev)[:k_]
        else:
            pick = torch.multinomial(w, k_, replacement=False)
        src = idx_live[pick]

        slot = (~live).nonzero(as_tuple=True)[0][:add]
        add = int(slot.numel()); src = src[:add]

        # beside the chosen particle, tangentially, about one target spacing out, then back onto the shell
        u0 = (pos[src] - c)
        r0 = u0.norm(dim=1, keepdim=True).clamp_min(1e-9)
        u0 = u0 / r0
        rnd = torch.randn(add, 3, device=dev, dtype=dt_)
        tang = rnd - (rnd * u0).sum(1, keepdim=True) * u0
        tang = tang / tang.norm(dim=1, keepdim=True).clamp_min(1e-12)
        newp = c + u0 * r0 + tang * want_sp
        newp = c + (newp - c) / (newp - c).norm(dim=1, keepdim=True).clamp_min(1e-12) * r0
        pos[slot] = newp
        v = lvl.get("vel") if "vel" in lvl.state_schema else None
        if v is not None:
            v[slot] = v[src]
        m = getattr(lvl, "mass", None)
        if m is not None:
            m[slot] = m[src]
        # NEWLY SECRETED MATERIAL IS UNSTRAINED, and this is not a detail. Marking a parked particle
        # dormant by mass alone does NOT stop it scattering, the stress term being weighted by
        # p_vol rather than m -- and `mpm_gather` still hands it a velocity every frame while
        # `mpm_strain` still integrates its deformation gradient from it. Sitting at the tissue centre for
        # hundreds of frames, F drifts arbitrarily far from identity. Activating it then promotes that
        # accumulated garbage into real material with real mass, and a single such particle can carry a
        # stress orders of magnitude above the rest, which renders as a few blazing pixels against a
        # black field and puts the colour scale of the whole run at the mercy of one particle.
        F = getattr(lvl, "F", None)
        if F is not None:
            F[slot] = torch.eye(F.shape[-1], device=dev, dtype=F.dtype)
        Cb = getattr(lvl, "C", None)
        if Cb is not None:
            Cb[slot] = 0.0
        oc = getattr(lvl, "occ", None)
        if oc is not None:
            oc[slot] = 1.0          # spawn: the flag mpm_scatter/mpm_gather key off
        live = live.clone(); live[slot] = True
        H.membrane_alive = live
        H.membrane_new = slot

        # RELAX THE NEW MATERIAL INTO THE GAPS. Placing a node beside a randomly chosen parent, one
        # spacing away in a random tangential direction, is a random walk -- and random walks clump. The
        # parent is weighted toward sparse regions but the PLACEMENT is not, so the node lands next to
        # the parent rather than in the hole. Measured: the seeded sheet has a p99.9 clearance of 1.10
        # spacings (better than uniform random at 1.53), and by the end of a run it has degraded to 3.08.
        # The holes an external reviewer called too big are made here, not at initialisation.
        #
        # A few repulsion sweeps over the newly placed nodes push them off their neighbours and into the
        # space that is actually free. Only the new ones move: relaxing the whole sheet every frame would
        # fight the integrin anchors, which are the thing holding it in place.
        # THE WHOLE SHEET, PERIODICALLY. Relaxing only the nodes placed this frame is a placement
        # correction, not a repair: once a gap has opened, nothing ever revisits it. Measured over a full
        # run that bought 20% (clearance 3.08 -> 2.47 spacings) against a well-packed 1.10. Every
        # `relax_every` frames the live sheet is swept as a whole so existing gaps are closed too.
        _f = int(getattr(H, "frame", -1) or -1)
        if self.relax_every > 0 and _f > 0 and _f % self.relax_every == 0:
            idx_all = live.nonzero(as_tuple=True)[0]
            sp_a = math.sqrt(4.0 * math.pi * R * R / max(int(live.sum()), 1))
            for _ in range(self.relax_sweeps):
                pa = pos[idx_all]
                blk = 4096
                push_all = torch.zeros_like(pa)
                for a0 in range(0, pa.shape[0], blk):
                    b0 = min(pa.shape[0], a0 + blk)
                    dd = (pa[a0:b0, None, :] - pa[None, :, :]).norm(dim=-1)
                    dd[torch.arange(b0 - a0, device=dev), torch.arange(a0, b0, device=dev)] = 1e9
                    nd, ni = torch.topk(-dd, min(7, pa.shape[0]), dim=1)
                    nd = -nd
                    diff = pa[a0:b0, None, :] - pa[ni]
                    w_ = (sp_a - nd).clamp_min(0.0) / sp_a
                    push_all[a0:b0] = (diff / nd[..., None].clamp_min(1e-9) * w_[..., None]).sum(1)
                pn = pa + push_all * (0.3 * sp_a)
                rad = (pa - c).norm(dim=1, keepdim=True)
                pos[idx_all] = c + (pn - c) / (pn - c).norm(dim=1, keepdim=True).clamp_min(1e-12) * rad
        if self.relax_new > 0 and add:
            sp_t = want_sp
            for _ in range(self.relax_new):
                pn = pos[slot]
                d2 = (pn[:, None, :] - pos[live][None, :, :]).norm(dim=-1)
                d2 = torch.where(d2 > 1e-9, d2, torch.full_like(d2, 1e9))
                k_ = min(7, int(live.sum()))
                nd, ni = torch.topk(-d2, k_, dim=1)
                nd = -nd
                diff = pn[:, None, :] - pos[live][ni]
                w_ = (sp_t - nd).clamp_min(0.0) / sp_t
                push = (diff / nd[..., None].clamp_min(1e-9) * w_[..., None]).sum(1)
                pn = pn + push * (0.3 * sp_t)
                # back onto the shell: relaxation must not change the radius the anchors set
                rad = (pos[slot] - c).norm(dim=1, keepdim=True)
                dirn = (pn - c) / (pn - c).norm(dim=1, keepdim=True).clamp_min(1e-12)
                pos[slot] = c + dirn * rad
        SECRETE_TRACE.append((n_live, add, int(live.sum()), R))
        return {}


@register_operator("bm_secrete", implementation="live", family="population", set="particle",
                   kind="structural",
                   equation=r"""$$n_{\mathrm{want}}(t)=n_0\,\frac{A_b(t)}{A_b(0)},\qquad \Delta n=\min\!\big(n_{\mathrm{want}}(t)-n_{\mathrm{live}}(t),\ \lceil r\,n_{\mathrm{live}}(t)\rceil\big)$$""", title="secrete membrane to keep its density")
class BasementMembraneSecreteLive(BasementMembraneSecrete):
    r"""new membrane from the cells that hold it, keeping the sheet's areal node density at its first-frame value as the live tissue's basal surface grows, from the reserve only and never inside a declared hole.
    basement_membrane_node -> basement_membrane_node: activates reserve slots under the cells that grew and hold membrane (between bm_contact[live] and cell_mechanics); bm_bond crosslinks them at its next call.

        n_{\mathrm{want}}(t)=n_0\,\frac{A_b(t)}{A_b(0)},\qquad \Delta n=\min\!\big(n_{\mathrm{want}}(t)-n_{\mathrm{live}}(t),\ \lceil r\,n_{\mathrm{live}}(t)\rceil\big)
        n_{want}(t)    the node count that keeps the first-frame density
        n_0            the live nodes at the first frame
        A_b(t)         the basal surface's area this frame
        \Delta n       the nodes added this frame
        n_{live}(t)    the live nodes
        r              `rate`, the cap per frame as a fraction of the live nodes

    Reference: Ku, K. & Bilder, D. (2023). Dev. Cell 58:522 (basement membrane is deposited basally by the epithelium it underlies); Harunaga, J. S., Doyle, A. D. & Yamada, K. M. (2014). Dev. Biol. 394:197-205."""
    PARAM_UNITS = {'cand_mult': 'dimensionless', 'jitter': 'fraction', 'offset': 'length', 'rate': 'fraction', 'relax_new': 'count', 'seed': 'count'}
    SYMBOLS = {'n_{want}(t)': 'the node count that keeps the first-frame density', 'n_0': 'the live nodes at the first frame', 'A_b(t)': "the basal surface's area this frame", '\\Delta n': 'the nodes added this frame', 'n_{live}(t)': 'the live nodes', 'r': '`rate`, the cap per frame as a fraction of the live nodes'}
    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    # `exclude_block` (default None = unchanged): NOT UNDER THE CELLS THAT DIGEST IT. A candidate whose nearest basal
    # face belongs to a cell whose width-1 block `exclude_block` (on `cell_set`) is above 0.5 is rejected, as a
    # candidate in the declared cone is. Without it the sparsest-first rule refills a hole CUT BY CELLS
    # (`bm_unbond[model: release]`, whose leaders carry `mutant`) as fast as they digest it: exp 11 Finding 70 --
    # the cell-made hole's bud 0.11 / 0.18 with secretion, 0.41 / 0.38 without. Harunaga 2014: the gland's holes are
    # dynamic and stay open only while proteases act.
    REQUIRES_PARAMS = ["surface"]
    MECHANISM_TAGS = ["secretion", "material_addition", "areal_density_setpoint", "basal_deposition"]
    PARAM_ROLES = {"rate": "max_fraction_secreted_per_frame", "offset": "standoff_the_sheet_is_held_at",
                   "surface": "the_live_vertex_set", "relax_new": "repulsion_sweeps_on_new_nodes",
                   "jitter": "tangential_jitter_in_spacings", "cand_mult": "candidates_per_new_node", 'cell_set': 'the_cell_set', 'exclude_block': 'cell_block_that_excludes_a_cell_from_secreting', 'seed': 'random_seed', 'sep_block': 'vertex_block_of_the_apico_basal_separation'}

    def __init__(self, params, device="cpu"):
        Structural.__init__(self, params, device)
        self.at = params.get("_at", "bm_node")
        self.surface = str(params["surface"])
        self.sep_block = str(params.get("sep_block", "sep"))
        self.rate = float(params.get("rate", 0.02))
        self.offset = float(params.get("offset", 0.05))
        self.relax_new = int(params.get("relax_new", 4))
        self.jitter = float(params.get("jitter", 0.25))
        self.cand_mult = int(params.get("cand_mult", 12))
        self.exclude_block = params.get("exclude_block", None)
        self.cell_set = str(params.get("cell_set", "cell"))
        self.seed = int(params.get("seed", 0))
        self._gen = torch.Generator().manual_seed(self.seed)
        self._geo = _basal_lookup()
        self._n0 = self._a0 = self._lstar = self._farea = self._spent = None
        self._said_spent = False

    def _count(self, want, n_live, lig, farea):
        """How many nodes to lay this call: up to the areal setpoint `want`, at most `rate` x n_live."""
        return min(want - n_live, int(math.ceil(self.rate * n_live)))

    def _weights(self, gain, lig, farea):
        """Per cell, its share of the new nodes: the basal area it gained, if it holds membrane (else its area)."""
        bound = (lig > 0).to(farea.dtype)
        w = gain * bound
        if float(w.sum()) <= 0:
            w = farea * bound
        return w

    def _onto(self, M, c, y):
        """y moved along the nearest face's outward normal to the basal surface + offset; found [n]."""
        found, _, _, nh, gap = _nearest_faces(M, c, y)
        return torch.where(found[:, None], y - (gap - self.offset)[:, None] * nh, y), found, nh

    def _in_excluded(self, M, c, y, excl):
        """[n] True where y's nearest basal face belongs to an excluded cell (`exclude_block`)."""
        if excl is None or y.shape[0] == 0:
            return torch.zeros(y.shape[0], dtype=torch.bool, device=y.device)
        found, tri, _, _, _ = _nearest_faces(M, c, y)
        cell = M["ef"][tri]
        return found & excl[cell.clamp(max=excl.numel() - 1)]

    @staticmethod
    def _in_cone(hole, c, y):
        if hole is None or y.shape[0] == 0:
            return torch.zeros(y.shape[0], dtype=torch.bool, device=y.device)
        a, half = hole
        d = y - c
        cs = (d @ a.to(y.device, y.dtype)) / d.norm(dim=1).clamp_min(1e-12)
        return cs >= math.cos(math.radians(float(half)))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        live = getattr(H, "membrane_alive", None)
        if live is None:
            return {}
        got = _live_basal(H, self.surface, self.sep_block, self._geo, dev, dt_)
        if got is None:
            return {}
        m, c, M = got
        f = getattr(H, "frame", None)
        f = -1 if f is None else int(f)
        nF = int(M["nF"])
        lig = m.get("bm_ligand")
        if lig is None or lig.numel() != nF or m.get("bm_ligand_frame") != f:
            raise RuntimeError("bm_secrete[live]: needs this frame's per-cell bound density "
                               "m['bm_ligand'] from `bm_contact[live]`, counted on the current cells -- "
                               "schedule it AFTER bm_contact and BEFORE cell_mechanics / cell_divide")
        ef = M["ef"]
        tri_a = 0.5 * torch.cross(M["B"] - M["A"], M["C"] - M["A"], dim=1).norm(dim=1)
        farea = torch.zeros(nF, device=dev, dtype=dt_).index_add_(0, ef, tri_a)
        A = float(tri_a.sum())
        n_live = int(live.sum())
        n = pos.shape[0]
        if self._n0 is None:
            self._n0, self._a0 = n_live, A
            self._lstar = math.sqrt(A / max(n_live, 1))
            self._farea = farea.clone()
            self._spent = torch.zeros(n, dtype=torch.bool, device=dev)
            SECRETE_LIVE_TRACE.append((f, n_live, 0, n_live, A))
            print(f"[bm_secrete/live] holding {n_live} live nodes per {A:.4g} of basal area "
                  f"(spacing {self._lstar:.3g}); rate cap {self.rate:g} per frame", flush=True)
            return {}
        prev, self._farea = self._farea, farea.clone()
        gain = torch.zeros(nF, device=dev, dtype=dt_)
        k_ = min(prev.numel(), nF)
        gain[:k_] = (farea[:k_] - prev[:k_]).clamp_min(0.0)
        want = int(round(self._n0 * A / max(self._a0, 1e-30)))
        add = self._count(want, n_live, lig.to(dev), farea)
        if add <= 0:
            SECRETE_LIVE_TRACE.append((f, n_live, 0, want, A))
            return {}

        # THE SLOTS: reserve only, each used once; never a hole node
        elig = ~live.to(dev) & ~self._spent
        res = getattr(H, "membrane_reserve", None)
        hole = getattr(H, "membrane_hole", None)
        if res is not None:
            elig = elig & res.to(dev)
        else:
            u0 = getattr(H, "membrane_u0", None)
            if hole is not None and u0 is not None:
                elig = elig & ~((u0.norm(dim=1) > 0) & self._in_cone(hole, torch.zeros_like(c),
                                                                      u0.to(dev, dt_)))
        slots = elig.nonzero(as_tuple=True)[0][:add]
        if slots.numel() == 0:
            if not self._said_spent:
                self._said_spent = True
                print(f"[bm_secrete/live] frame {f}: the reserve is spent -- {n_live} live nodes "
                      f"against {want} wanted", flush=True)
            SECRETE_LIVE_TRACE.append((f, n_live, 0, want, A))
            return {}
        add = int(slots.numel())

        # WHO: cells that grew and hold membrane (`_weights`; a variant may choose otherwise)
        w = self._weights(gain, lig.to(dev), farea)
        if float(w.sum()) <= 0:
            SECRETE_LIVE_TRACE.append((f, n_live, 0, want, A))
            return {}
        wt = (w[ef] * tri_a / farea[ef].clamp_min(1e-20)).detach().to("cpu", torch.float64)
        n_cand = max(add * self.cand_mult, add)
        ti = torch.multinomial(wt, n_cand, replacement=True, generator=self._gen).to(dev)
        r = torch.rand(n_cand, 2, generator=self._gen, dtype=torch.float64)
        fl = r.sum(1) > 1.0
        r[fl] = 1.0 - r[fl]
        r = r.to(dev, dt_)
        Aa = M["A"][ti]
        Q = c + Aa + r[:, 0:1] * (M["B"][ti] - Aa) + r[:, 1:2] * (M["C"][ti] - Aa)
        Y, found, _ = self._onto(M, c, Q)
        excl = None
        if self.exclude_block:
            _cl = H.level(self.cell_set)
            if self.exclude_block not in _cl.state_schema:
                raise ValueError(f"bm_secrete[live]: exclude_block {self.exclude_block!r} is not a block of "
                                 f"the {self.cell_set!r} set")
            excl = (_cl.get(self.exclude_block)[:nF, 0] > 0.5).to(dev)
        Y = Y[found & ~self._in_cone(hole, c, Y) & ~self._in_excluded(M, c, Y, excl)]
        if Y.shape[0] == 0:
            SECRETE_LIVE_TRACE.append((f, n_live, 0, want, A))
            return {}

        # WHERE: the sparsest candidates, greedily (each pick is a node for the next)
        P_live = pos[live].detach()
        dmin = torch.empty(Y.shape[0], device=dev, dtype=dt_)
        for a0 in range(0, Y.shape[0], 1024):
            dmin[a0:a0 + 1024] = torch.cdist(Y[a0:a0 + 1024], P_live).min(1).values
        pick = []
        for _ in range(min(add, Y.shape[0])):
            j = int(dmin.argmax())
            pick.append(j)
            dmin = torch.minimum(dmin, (Y - Y[j]).norm(dim=1))
        newp = Y[torch.tensor(pick, device=dev)]
        add = newp.shape[0]
        slots = slots[:add]
        ls = self._lstar
        if self.jitter > 0:
            _, _, nh = self._onto(M, c, newp)
            g = torch.randn(add, 3, generator=self._gen, dtype=torch.float64).to(dev, dt_)
            g = g - (g * nh).sum(1, keepdim=True) * nh
            cand, ok, _ = self._onto(M, c, newp + g * (self.jitter * ls))
            ok = ok & ~self._in_cone(hole, c, cand) & ~self._in_excluded(M, c, cand, excl)
            newp = torch.where(ok[:, None], cand, newp)
        for _ in range(self.relax_new):
            allp = torch.cat([P_live, newp])
            d = torch.cdist(newp, allp)
            d[torch.arange(add, device=dev), P_live.shape[0] + torch.arange(add, device=dev)] = 1e9
            kk = min(7, allp.shape[0] - 1)
            if kk <= 0:
                break
            nd, ni = torch.topk(-d, kk, dim=1)
            nd = -nd
            diff = newp[:, None, :] - allp[ni]
            w_ = (ls - nd).clamp_min(0.0) / ls
            push = (diff / nd[..., None].clamp_min(1e-9) * w_[..., None]).sum(1)
            cand, ok, _ = self._onto(M, c, newp + push * (0.3 * ls))
            ok = ok & ~self._in_cone(hole, c, cand) & ~self._in_excluded(M, c, cand, excl)
            newp = torch.where(ok[:, None], cand, newp)

        pos[slots] = newp
        if "vel" in lvl.state_schema:
            lvl.get("vel")[slots] = 0.0
        oc = getattr(lvl, "occ", None)
        if oc is not None:
            oc[slots] = 1.0
        live = live.clone()
        live[slots] = True
        H.membrane_alive = live
        self._spent[slots] = True
        pn = getattr(H, "membrane_new", None)
        H.membrane_new = slots if pn is None or pn.numel() == 0 else torch.cat([pn, slots])
        SECRETE_LIVE_TRACE.append((f, n_live, add, want, A))
        return {}


# Per call of `bm_secrete[live]`: (frame, live nodes before, nodes added, n_want, basal area A_b).
SECRETE_LIVE_TRACE: list = []


@register_operator("bm_secrete", implementation="heal", family="population", set="particle",
                   kind="structural",
                   equation=r"""$$d_f=\max\big(0,\ \rho^\ast-\rho_f\big)\,a_f,\qquad \Delta n=\min\Big(\big\lceil \textstyle\sum_f d_f/\tau_h\big\rceil,\ \lceil r\,n_{\rm live}\rceil\Big)$$""", title="secrete membrane to heal gaps")
class BasementMembraneSecreteHeal(BasementMembraneSecreteLive):
    r"""bm_secrete[live] with a local set point: the membrane re-covers the cells it left within `heal_tau` frames, who secretes and how much following each cell's own deficit.
    basement_membrane_node -> basement_membrane_node: as bm_secrete[live] (reserve slots only, sparsest first, never in a hole), the total bounded by `max_over` and `global_floor`.

        d_f=\max\big(0,\ \rho^\ast-\rho_f\big)\,a_f,\qquad \Delta n=\min\Big(\big\lceil \textstyle\sum_f d_f/\tau_h\big\rceil,\ \lceil r\,n_{\rm live}\rceil\Big)
        d_f            cell f's deficit of bound nodes, in nodes
        \rho^*         the median bound density at the first call
        \rho_f         the cell's bound density
        a_f            its basal area
        \Delta n       the nodes added this frame
        \tau_h         `heal_tau`, frames to close a deficit
        r, n_{live}    the per-frame cap

    Reference: as bm_secrete[live]; Plexus exp11 Phase 3 (this work)."""
    PARAM_UNITS = {'global_floor': 'fraction', 'heal_frac': 'fraction', 'heal_tau': 'frames', 'max_over': 'fraction'}
    SYMBOLS = {'d_f': "cell f's deficit of bound nodes, in nodes", '\\rho^*': 'the median bound density at the first call', '\\rho_f': "the cell's bound density", 'a_f': 'its basal area', '\\Delta n': 'the nodes added this frame', '\\tau_h': '`heal_tau`, frames to close a deficit', 'r, n_{live}': 'the per-frame cap'}
    MECHANISM_TAGS = ["secretion", "material_addition", "local_density_setpoint", "basal_deposition", "repair"]
    PARAM_ROLES = {**BasementMembraneSecreteLive.PARAM_ROLES, "heal_tau": "frames_to_close_a_local_deficit", 'global_floor': 'floor_of_the_total_as_a_fraction_of_the_live_setpoint', 'heal_frac': 'fraction_of_the_deficit_healed', 'max_over': 'cap_on_the_total_over_the_live_setpoint'}

    # `max_over` (default none = unbounded): the live count never exceeds max_over x the global setpoint n0 A / A0.
    # Unbounded, the local rule kept adding: R1 (exp 11, 2026-09-30) climbed to 19-25 nodes per unit area near a cell
    # (seeded 15) and used all 120,000 slots by row 200, after which nothing healed (Finding 187).
    # `global_floor` (default false): lay at least what `bm_secrete[live]` would (want - n_live) when the local
    # deficits ask for less, so the sheet's density never falls below its setpoint as the tissue grows.
    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.heal_tau = float(params.get("heal_tau", 6.0))
        _mo = params.get("max_over", None)
        self.max_over = None if _mo is None else float(_mo)
        self.global_floor = bool(params.get("global_floor", False))
        # `heal_frac` (default 1.0): a cell is in deficit only below heal_frac x rho*. At 1.0 the setpoint is the median,
        # so half the cells are always "in deficit" and draw nodes for ever -- the overshoot of R1; at 0.5 only cells
        # clearly thinner than the sheet are healed, up to half its density, and the rest is the global rule's.
        self.heal_frac = float(params.get("heal_frac", 1.0))
        self._rho_star = None

    def _deficit(self, lig, farea):
        if self._rho_star is None:
            self._rho_star = float(torch.median(lig[farea > 0])) if bool((farea > 0).any()) else 0.0
            print(f"[bm_secrete/heal] local setpoint {self._rho_star:.4g} bound nodes per unit basal area; "
                  f"deficits close in {self.heal_tau:g} frames", flush=True)
        return (self.heal_frac * self._rho_star - lig).clamp_min(0.0) * farea

    def _count(self, want, n_live, lig, farea):
        d = self._deficit(lig, farea)
        add = int(math.ceil(float(d.sum()) / max(self.heal_tau, 1e-9)))
        if self.global_floor:
            add = max(add, want - n_live)
        if self.max_over is not None:
            add = min(add, int(math.ceil(self.max_over * want)) - n_live)
        return min(add, int(math.ceil(self.rate * n_live)))

    def _weights(self, gain, lig, farea):
        d = self._deficit(lig, farea)
        if float(d.sum()) <= 0 and self.global_floor:
            return BasementMembraneSecreteLive._weights(self, gain, lig, farea)
        return d



SECRETE_TRACE = []

# Periodic snapshots of the crosslink NETWORK: (frame, i, j, strain). Per-particle strain cannot show
# topology -- a node with three taut bonds and a node with one look identical -- and "structured vs
# broken" is a statement about edges, so the edges have to be recorded. Every `snapshot_every` frames,
# because storing 140k bonds x 400 frames is not worth it and the network changes slowly.
BOND_SNAPSHOTS: list = []

# Per frame: mean crosslink tension by latitude band. What a corset would press with, and the only
# route by which the membrane can reach the epithelium while the coupling is one-way.
HOOP_TRACE: list = []

# Per rebonding event: (frame, crosslinks formed, live crosslinks after).
REBOND_TRACE: list = []

# Published by `run_ecm.run`/`rerender`: which particles are membrane and which are unsecreted reserve.
MEMBRANE_ALIVE = None


@register_entity(
    # `spatial_schema`, not the legacy 2D dict -- see `basement_membrane_particle` above.
    # `receptor` is the biological name; the long one stays first and stays valid.
    "integrin_particle", "receptor", depth=0,
    state_schema=spatial_schema,
    render={"color_by": "node_type", "arrows": None},
)
class IntegrinParticle:
    """A material point of an integrin fibre, carrying the stock MPM continuum state: the
    deformation gradient F, the affine velocity C, mass, the Lame parameters mu and lambda, and the
    rest volume p_vol.

    Registered because the entity is resolved BY SET NAME. An unregistered name falls back silently
    to a bare position/velocity schema, and the run then dies inside `mpm_strain` on a missing F --
    which reads like a bug in the MPM operator and is a missing three-line class.
    """

    provision = MPMParticle.provision


def _sphere_dirs(n, seed=0):
    """A Fibonacci sphere: exactly uniform in areal density, which is what a plaque distribution needs.

    The membrane's own seed prefers a relaxed sample because a spiral is visible in a RENDERED sheet of
    3,000 particles. Fibres are drawn as segments and there are far fewer of them, so the arms do not
    read; density uniformity is the property that matters and Fibonacci has it exactly.
    """
    i = torch.arange(n, dtype=torch.float64) + 0.5
    ct = 1.0 - 2.0 * i / n
    st = torch.sqrt((1.0 - ct * ct).clamp_min(0.0))
    ph = (math.pi * (1.0 + 5.0 ** 0.5) * i) % (2 * math.pi)
    return torch.stack([st * torch.cos(ph), st * torch.sin(ph), ct], dim=1).to(torch.float32)


def _radius(M, u):
    nth, nph = M.shape
    th = torch.acos(u[:, 2].clamp(-1, 1))
    ph = torch.atan2(u[:, 1], u[:, 0]) % (2 * math.pi)
    return M[(th / math.pi * nth).long().clamp(0, nth - 1),
             (ph / (2 * math.pi) * nph).long().clamp(0, nph - 1)]


# ==========================================================================================================
# bm_bond[fem_membrane] -- a rubber sheet as a triangulated neo-Hookean membrane over MPM points:
# pre-stretched, tearing triangle by triangle.
# ==========================================================================================================
@register_operator("bm_bond", model="fem_membrane", family="mechanics", set="particle", kind="lateral",
                   title="stretch and tear a rubber membrane",
                   equation=r"""$$\psi=\tfrac{\mu t_0}{2}\Big(\operatorname{tr}\mathbf C+\frac{1}{\det\mathbf C}-3\Big),\quad
\mathbf P=\mu t_0\big(\mathbf F-\mathbf F\mathbf C^{-1}/\det\mathbf C\big),\quad \mathbf f=-A_0\,\mathbf P\,\mathbf D_m^{-\top}$$""")
class FEMMembraneBond(Lateral):
    r"""a thin rubber sheet whose elasticity lives on a triangulation of its own points: every triangle a piece of incompressible neo-Hookean (or Gent) membrane stretched from a smaller rest triangle, wrinkling without compressive stress, failing for good past its breaking stretch.
    particle -> particle: reads pos and mass; emits an acceleration the MPM substep consumes (the convex hull of the live points, built once).

        \psi=\tfrac{\mu t_0}{2}\Big(\operatorname{tr}\mathbf C+\frac{1}{\det\mathbf C}-3\Big),\quad \mathbf P=\mu t_0\big(\mathbf F-\mathbf F\mathbf C^{-1}/\det\mathbf C\big),\quad \mathbf f=-A_0\,\mathbf P\,\mathbf D_m^{-\top}
        \psi           the membrane energy per rest area
        \mu            `shear_modulus`, the rubber's shear modulus
        t_0            `thickness`, the unstretched thickness
        C              the in-plane right Cauchy-Green tensor F^T F
        F              the triangle's deformation gradient, current edges over rest edges (shrunk by `prestretch`)
        P              the first Piola-Kirchhoff stress
        f              the force on the triangle's vertices
        A_0            the rest area
        D_m            the rest edge matrix

    Reference: Jiang, C. et al. (2017). Anisotropic elastoplasticity for cloth, knit and hair frictional contact. ACM Trans. Graph. 36(4):152; Mueller, I. & Strehlow, P. (2004). Rubber and Rubber Balloons. Lecture Notes in Physics 637; Gent, A. N. (1996). Rubber Chem. Technol. 69:59-61; Pipkin, A. C. (1986). IMA J. Appl. Math. 36:85-99 (tension-field theory)."""
    PARAM_UNITS = {'det_floor': 'dimensionless', 'jm': 'dimensionless', 'prestretch': 'dimensionless', 'shear_modulus': 'stress', 'stretch_break': 'dimensionless', 'thickness': 'length'}
    SYMBOLS = {'\\psi': 'the membrane energy per rest area', '\\mu': "`shear_modulus`, the rubber's shear modulus", 't_0': '`thickness`, the unstretched thickness', 'C': 'the in-plane right Cauchy-Green tensor F^T F', 'F': "the triangle's deformation gradient, current edges over rest edges (shrunk by `prestretch`)", 'P': 'the first Piola-Kirchhoff stress', 'f': "the force on the triangle's vertices", 'A_0': 'the rest area', 'D_m': 'the rest edge matrix'}

    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["shear_modulus", "thickness", "stretch_break"]
    MECHANISM_TAGS = ["membrane", "neo_hookean", "prestretch", "tearing", "codimensional"]
    PARAM_ROLES = {"shear_modulus": "rubber_shear_modulus",
                   "thickness": "wall_thickness_in_the_seeded_stretched_state",
                   "prestretch": "in_plane_stretch_of_the_seeded_sheet_relative_to_rest",
                   "stretch_break": "principal_stretch_at_failure_relative_to_rest",
                   "mesh": "triangulation_rule", "det_floor": "smallest_area_ratio_squared_in_the_law",
                   "law": "neo_hookean_or_gent", "jm": "gent_limit_on_I1_minus_3",
                   "wrinkle": "no_compressive_stress_tension_field"}
    REFERENCE = ("Jiang, C. et al. (2017). ACM Trans. Graph. 36(4):152; Mueller, I. & Strehlow, P. (2004). "
                 "Rubber and Rubber Balloons. LNP 637; Rivlin, R. S. & Thomas, A. G. (1953). J. Polym. Sci. "
                 "10:291-318.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.mu = float(params["shear_modulus"])
        self.lam = float(params.get("prestretch", 1.0))
        # the wall is incompressible: seeded at thickness t after a lam x lam stretch, it was lam^2 t at rest
        self.t0 = float(params["thickness"]) * self.lam * self.lam
        self.lam_b = float(params["stretch_break"])
        self.mesh = str(params.get("mesh", "convex_hull")).lower()
        self.det_floor = float(params.get("det_floor", 1e-2))
        self.wrinkle = bool(params.get("wrinkle", True))
        self.law = str(params.get("law", "neo_hookean")).lower()
        self.jm = float(params.get("jm", 0.0))
        if self.law not in ("neo_hookean", "gent"):
            raise ValueError(f"bm_bond[fem_membrane]: law {self.law!r} unknown (neo_hookean | gent)")
        if self.law == "gent":
            if self.jm <= 0:
                raise ValueError("bm_bond[fem_membrane]: law gent needs jm > 0")
            lim = lambda l: 2 * l * l + l ** -4 - 3.0
            if lim(self.lam) >= self.jm:
                raise ValueError(f"bm_bond[fem_membrane]: prestretch {self.lam} is past the Gent limit for jm {self.jm}")
        if self.mesh != "convex_hull":
            raise ValueError(f"bm_bond[fem_membrane]: mesh {self.mesh!r} unknown (convex_hull)")
        if self.lam_b <= self.lam:
            raise ValueError(f"bm_bond[fem_membrane]: stretch_break {self.lam_b} must exceed the prestretch "
                             f"{self.lam}, or the sheet is torn before anything happens")

    def _build(self, lvl, pos):
        import numpy as np
        from scipy.spatial import ConvexHull
        live = (lvl.occ > 0).nonzero(as_tuple=True)[0]
        P = pos[live].detach().double().cpu().numpy()
        tri = ConvexHull(P - P.mean(0)).simplices
        tri = torch.as_tensor(live.cpu().numpy()[tri], device=pos.device, dtype=torch.long)
        x = pos.detach().double()
        e1 = x[tri[:, 1]] - x[tri[:, 0]]
        e2 = x[tri[:, 2]] - x[tri[:, 0]]
        t1 = e1 / e1.norm(dim=1, keepdim=True)
        n = torch.linalg.cross(e1, e2)
        n = n / n.norm(dim=1, keepdim=True)
        t2 = torch.linalg.cross(n, t1)
        a, b, c = (e1 * t1).sum(1), (e2 * t1).sum(1), (e2 * t2).sum(1)
        a, b, c = a / self.lam, b / self.lam, c / self.lam              # the rest triangle
        det = a * c
        Dmi = torch.zeros(tri.shape[0], 2, 2, dtype=torch.float64, device=pos.device)
        Dmi[:, 0, 0] = 1.0 / a
        Dmi[:, 0, 1] = -b / det
        Dmi[:, 1, 1] = 1.0 / c
        f = torch.float32
        lvl.register_buffer("tri", tri)
        lvl.register_buffer("tri_Dmi", Dmi.to(f))
        lvl.register_buffer("tri_A0", (0.5 * det.abs()).to(f))
        lvl.register_buffer("tri_alive", torch.ones(tri.shape[0], device=pos.device, dtype=f))
        L = torch.cat([e1.norm(dim=1), e2.norm(dim=1)])
        T = self.mu * self.t0 * (1.0 - self.lam ** -6)
        if self.law == "gent":
            T = T / (1.0 - (2 * self.lam ** 2 + self.lam ** -4 - 3.0) / self.jm)
        print(f"[bm_bond fem_membrane] law {self.law}{' jm %g' % self.jm if self.law == 'gent' else ''}, {self.at}: {tri.shape[0]:,} triangles on {live.numel():,} points, edge "
              f"{float(L.median()):.4g} (max {float(L.max()):.4g}), rest thickness {self.t0:.4g}, prestretch "
              f"{self.lam:g} -> tension {T:.4g} (force per length), breaking stretch {self.lam_b:g}", flush=True)

    def forward(self, H, mask=None):
        # COMPONENT ARITHMETIC, NOT BATCHED MATMULS. The same algebra written as [T,3,2] @ [T,2,2]
        # products cost 17.4 ms a call on 327,680 triangles (A6000, captured or not): a batched GEMM
        # of 2x2 matrices is all launch and no work. Spelled out per component it is a few fused
        # element-wise passes. D_m^-1 is upper-triangular, [[a, b], [0, c]], so F's columns are
        # F0 = a e1 and F1 = b e1 + c e2.
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        if getattr(lvl, "tri", None) is None:
            self._build(lvl, pos)
        tri, Dmi = lvl.tri, lvl.tri_Dmi
        a, b, c = Dmi[:, 0, 0, None], Dmi[:, 0, 1, None], Dmi[:, 1, 1, None]
        x0 = pos.index_select(0, tri[:, 0])
        e1 = pos.index_select(0, tri[:, 1]) - x0
        e2 = pos.index_select(0, tri[:, 2]) - x0
        F0 = a * e1
        F1 = b * e1 + c * e2
        c11 = (F0 * F0).sum(1)
        c12 = (F0 * F1).sum(1)
        c22 = (F1 * F1).sum(1)
        detC = (c11 * c22 - c12 * c12).clamp_min(self.det_floor)
        tr = c11 + c22
        lmax2 = 0.5 * (tr + torch.sqrt(((c11 - c22) ** 2 + 4.0 * c12 * c12).clamp_min(0.0)))
        alive = lvl.tri_alive * (lmax2 <= self.lam_b * self.lam_b).to(pos.dtype)
        lvl.tri_alive.copy_(alive)
        k0 = self.mu * self.t0
        if self.wrinkle:
            # THE RELAXED (TENSION-FIELD) LAW, in C's principal frame: see the class docstring.
            m = 0.5 * tr
            r = torch.sqrt((0.5 * (c11 - c22)) ** 2 + c12 * c12)
            ca = m + r                                                  # l1^2
            cb = (m - r).clamp_min(0.0)                                 # l2^2
            l1 = torch.sqrt(ca)
            J2 = ca * cb                                                # (l1 l2)^2
            taut = (l1 > 1.0) & (cb * l1 >= 1.0)                       # l2 >= l1^(-1/2)
            wrink = (l1 > 1.0) & ~taut
            Jt = J2.clamp_min(1.0)
            S1 = torch.where(taut, k0 * (1.0 - 1.0 / (ca * Jt)),
                             torch.where(wrink, k0 * (1.0 - l1.clamp_min(1.0) ** -3), torch.zeros_like(ca)))
            S2 = torch.where(taut, k0 * (1.0 - 1.0 / (cb.clamp_min(1e-12) * Jt)), torch.zeros_like(ca))
            if self.law == "gent":
                I1 = torch.where(taut, ca + cb + 1.0 / Jt, ca + 2.0 / l1.clamp_min(1e-6))
                g = 1.0 / (1.0 - (I1 - 3.0) / self.jm).clamp_min(0.05)
                S1, S2 = S1 * g, S2 * g
            th = 0.5 * torch.atan2(2.0 * c12, c11 - c22)               # C's first eigenvector, rest frame
            cs, sn = torch.cos(th), torch.sin(th)
            s11 = S1 * cs * cs + S2 * sn * sn
            s22 = S1 * sn * sn + S2 * cs * cs
            s12 = (S1 - S2) * cs * sn
        else:
            # the full law, S = mu t0 g (I - C^-1/det C), det C floored
            k = torch.full_like(detC, k0)
            if self.law == "gent":
                k = k / (1.0 - (tr + 1.0 / detC - 3.0) / self.jm).clamp_min(0.05)
            d2 = detC * detC
            s11 = k * (1.0 - c22 / d2)
            s22 = k * (1.0 - c11 / d2)
            s12 = k * (c12 / d2)
        w = (lvl.tri_A0 * alive)[:, None]
        P0 = F0 * s11[:, None] + F1 * s12[:, None]
        P1 = F0 * s12[:, None] + F1 * s22[:, None]
        f1 = -w * (a * P0 + b * P1)                                     # -A0 (P D_m^-T), column 0
        f2 = -w * (c * P1)                                              # column 1
        force = torch.zeros_like(pos)
        force.index_add_(0, tri[:, 1], f1)
        force.index_add_(0, tri[:, 2], f2)
        force.index_add_(0, tri[:, 0], -(f1 + f2))
        m_ = getattr(lvl, "mass", None)
        if m_ is None:
            raise ValueError(f"bm_bond[fem_membrane]: {self.at!r} has no `mass` (it must be an MPM set)")
        acc = force / m_.clamp_min(1e-30)[:, None].to(force.dtype)
        if getattr(lvl, "occ", None) is not None:
            acc = acc * (lvl.occ > 0).to(acc.dtype)[:, None]
        return {lvl.name: acc}
