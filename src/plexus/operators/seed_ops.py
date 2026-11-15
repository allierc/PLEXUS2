# -*- coding: utf-8 -*-
"""seed_positions -- WHERE THE ENTITIES OF A SET START, as a seed operator.

Placement is not a property of the set. Declaring it there -- `sets.cell.spawn: sunflower` and
`spawn_radius`, read by the engine as it builds the level, before any operator exists -- puts the
one thing a seed is for, writing x_0, outside the operator algebra: a reader of the spec sees a
disc of cells appear with no operator responsible, and the vocabulary's own "seed: the initial
state, written once and never again" has nothing to point at. This operator is that seed. The set
declares how many entities it holds; the seed declares where they are.

The set-level `spawn*` keys are REFUSED by the engine (`build` raises, naming this operator), so
a spec cannot half-migrate.
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Seed
from plexus.models.registry import register_operator


# _spawn: Place a 2D self-propelled set -> (pos [n, 2], head [n, 2] unit vector).
def _spawn(mode: str, n: int, box, radius: float, rng, device: str):
    """Initial positions + headings of a 2D self-propelled set (Lague's SpawnMode).

    Placement is framed to the world box `[W, H]`: disc / point / ring centred at (W/2, H/2),
    clamped to [0,W] x [0,H]. Returns (pos [n, 2], head [n, 2] unit vector)."""
    box = box.float() if torch.is_tensor(box) else torch.as_tensor(box, dtype=torch.float32, device=device)
    W = float(box[0]); Hh = float(box[1]) if box.numel() > 1 else 1.0
    cx, cy = W / 2.0, Hh / 2.0
    if mode == "random":
        pos = torch.rand(n, 2, generator=rng, device=device); pos[:, 0] *= W; pos[:, 1] *= Hh
        a = torch.rand(n, generator=rng, device=device) * 2 * math.pi
    elif mode in ("point", "center"):
        pos = torch.stack([torch.full((n,), cx, device=device), torch.full((n,), cy, device=device)], 1)
        pos = pos + (torch.rand(n, 2, generator=rng, device=device) - 0.5) * 1e-3
        a = torch.rand(n, generator=rng, device=device) * 2 * math.pi
    elif mode == "disc":
        r = torch.sqrt(torch.rand(n, generator=rng, device=device)) * radius
        a = torch.rand(n, generator=rng, device=device) * 2 * math.pi
        pos = torch.stack([cx + r * torch.cos(a), cy + r * torch.sin(a)], 1)
        a = torch.rand(n, generator=rng, device=device) * 2 * math.pi
    elif mode in ("sunflower", "disc_even", "equidistant"):     # Vogel golden-angle spiral: even disc coverage
        idx = torch.arange(n, dtype=torch.float32, device=device) + 0.5
        r = torch.sqrt(idx / n) * radius
        theta = math.pi * (1.0 + 5.0 ** 0.5) * idx
        pos = torch.stack([cx + r * torch.cos(theta), cy + r * torch.sin(theta)], 1)
        a = torch.rand(n, generator=rng, device=device) * 2 * math.pi
    elif mode in ("ring_in", "ring_out"):
        a = torch.rand(n, generator=rng, device=device) * 2 * math.pi
        pos = torch.stack([cx + radius * torch.cos(a), cy + radius * torch.sin(a)], 1)
        a = (a + math.pi) if mode == "ring_in" else a
    else:
        raise ValueError(f"unknown spawn mode {mode!r}")
    pos[:, 0] = pos[:, 0].clamp(0, W - 1e-6); pos[:, 1] = pos[:, 1].clamp(0, Hh - 1e-6)
    head = torch.stack([torch.cos(a), torch.sin(a)], dim=1)        # [n, 2] unit heading
    return pos, head


# _spawn3d: Place a 3D self-propelled set -> (pos [n, 3], head [n, 3] unit vector).
def _spawn3d(mode: str, n: int, box, radius: float, rng, device: str, thickness: float = 0.0):
    """Initial positions + headings of a 3D self-propelled set: the 3D `_spawn`.

    Modes: `ball`/`sphere`/`disc` solid ball of `radius` about the box centre; `disk` a flat disc in
    xy with Gaussian `thickness` along z; `point`/`center` jittered centre; `random` fills the box.
    Returns (pos [n, 3], head [n, 3] random unit vector)."""
    box = torch.as_tensor(box, dtype=torch.float32, device=device)
    c = box * 0.5
    if mode == "random":
        pos = torch.rand(n, 3, generator=rng, device=device) * box
    elif mode in ("point", "center"):
        pos = c.expand(n, 3) + (torch.rand(n, 3, generator=rng, device=device) - 0.5) * 1e-3
    elif mode in ("disc", "ball", "sphere"):
        d = torch.randn(n, 3, generator=rng, device=device)
        d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-9)
        r = torch.rand(n, generator=rng, device=device).pow(1.0 / 3.0) * radius   # uniform in the ball
        pos = c.expand(n, 3) + d * r[:, None]
    elif mode in ("disk", "flat_disc"):
        r = torch.sqrt(torch.rand(n, generator=rng, device=device)) * radius       # uniform-area xy disc
        th = torch.rand(n, generator=rng, device=device) * 2 * math.pi
        pos = c.expand(n, 3).clone()
        pos[:, 0] = c[0] + r * torch.cos(th)
        pos[:, 1] = c[1] + r * torch.sin(th)
        if thickness > 0:
            pos[:, 2] = c[2] + thickness * torch.randn(n, generator=rng, device=device)
    else:
        raise ValueError(f"unknown 3D spawn mode {mode!r}")
    pos = torch.minimum(pos.clamp(min=0.0), box - 1e-6)
    head = torch.randn(n, 3, generator=rng, device=device)
    head = head / head.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return pos, head


# _spawn_pair3d: Place two flat discs for an encounter -> (pos [n, 3], head [n, 3], group id [n], rot [2, 3, 3]).
def _spawn_pair3d(n: int, box, radius: float, rng, device: str, thickness: float = 0.0,
                  separation: float = 0.0, offset: float = 0.0, tilt: float = 0.0,
                  arms: dict | None = None):
    """Two flat discs placed for an encounter: the `two_disks` 3D spawn mode.

    One set, so that mutual gravity (`squared_law all_pairs`) acts between the discs as well as
    within each. Disc 0 is centred at (cx - separation/2, cy - offset/2, cz), disc 1 at
    (cx + separation/2, cy + offset/2, cz): `separation` is the centre-to-centre distance along x,
    `offset` the impact parameter along y. Disc 1's plane is tilted by `tilt` radians about x.
    `radius` / `thickness` take one value or a pair (unequal discs). `arms` plants a logarithmic
    spiral of amplitude A, m arms, pitch angle `pitch` degrees, so the spiral does not fade with N.
    Returns (pos [n, 3], head [n, 3], group id [n] in {0, 1}, rot [2, 3, 3] each disc's plane
    rotation, which the velocity IC needs). Not clamped into the box: `boundary: free` is open."""
    def _pair(v, what):
        if isinstance(v, (list, tuple)):
            if len(v) != 2:
                raise ValueError(f"spawn_{what}: give one value for both discs or two, got {len(v)}")
            return [float(v[0]), float(v[1])]
        return [float(v), float(v)]

    radius = _pair(radius, "radius")
    thickness = _pair(thickness, "thickness")
    box = torch.as_tensor(box, dtype=torch.float32, device=device)
    c = box * 0.5
    ct, st = math.cos(tilt), math.sin(tilt)
    rot = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                        [[1., 0., 0.], [0., ct, -st], [0., st, ct]]], device=device)
    counts = (n // 2, n - n // 2)
    centres = torch.stack([
        c + torch.tensor([-0.5 * separation, -0.5 * offset, 0.0], device=device),
        c + torch.tensor([+0.5 * separation, +0.5 * offset, 0.0], device=device)])
    pos, gid = [], []
    for g, k in enumerate(counts):
        r = torch.sqrt(torch.rand(k, generator=rng, device=device)) * radius[g]  # uniform-area disc
        th = torch.rand(k, generator=rng, device=device) * 2 * math.pi
        # spiral arms: shift each star in azimuth by (A/m) sin(m phase), density ridges of amplitude A,
        # phase wound with ln(r) into a log spiral (shot-noise arms would fade as 1/sqrt(N))
        if arms and float(arms.get("amplitude", 0.0)) > 0.0:
            A = float(arms["amplitude"])
            mm = float(arms.get("m", 2))
            pitch = math.radians(float(arms.get("pitch", 20.0)))
            phase = th - torch.log(r.clamp(min=1e-6) / radius[g]) / math.tan(pitch)
            th = th + (A / mm) * torch.sin(mm * phase)
        z = (thickness[g] * torch.randn(k, generator=rng, device=device)
             if thickness[g] > 0 else torch.zeros(k, device=device))
        local = torch.stack([r * torch.cos(th), r * torch.sin(th), z], 1)       # [k, 3]
        pos.append(local @ rot[g].T + centres[g])                             # tilt, then translate
        gid.append(torch.full((k,), g, dtype=torch.long, device=device))
    pos = torch.cat(pos); gid = torch.cat(gid)
    head = torch.randn(n, 3, generator=rng, device=device)
    head = head / head.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return pos, head, gid, rot


# THE TITLE NAMES WHAT x_0 IS, NOT ONLY WHAT THIS CLASS WRITES. A reader of the `seed` block wants
# to know what the run STARTS FROM, and a set's opening state is more than its coordinates: the
# placement here, the initial velocity (`vel_init`, applied right after the seeds so it reads the
# placed positions), the partition into types that `sets.<name>.types` declares, the heading and
# the spawn group derived below. Naming only the positions left the pane saying "initial positions"
# beside a picture whose whole content was three colours -- the types -- and a reader then has
# nowhere to learn where those came from.
#
# The trailing "..." is doing work and is not filler: it says the list is open. Types are read by
# the engine at build (`_assign_types`) rather than written by this operator, and other seed
# operators establish other parts of x_0; the row is the SEED's claim, and the ellipsis keeps it
# from reading as an exhaustive one this class could not honour.
@register_operator("seed_positions", family="seed", set="cell", kind="seed",
                   title="set initial positions",
                   equation=r"""$$\mathbf x_i=\mathrm{placement}(\text{mode},\,r,\,i)$$""")
class SeedPositions(Seed):
    r"""writes every position of a set once, at the opening of the trajectory, by a named placement rule, with the unit heading per entity the engine used to derive from it.
    set -> set: writes `pos`; registers `heading` [n, D] and, for `two_disks`, `spawn_group` [n] and the two planes' rotations.

        \mathbf x_i=\mathrm{placement}(\text{mode},\,r,\,i)
        x_i            the position of entity i
        mode           the placement rule (`mode`: sunflower, ball, two_disks, ...), the `_spawn*` placement rules of this file
        r              `radius`, the rule's one length, world length, from the centre (`centre`: a point, a set's live centroid, or the world box's centre)
        i              the entity index, 1..n

    Reference: Vogel, H. (1979). A better way to construct the sunflower head. Math. Biosci. 44:179-189 (the golden-angle spiral); Toomre, A. & Toomre, J. (1972). Galactic bridges and tails. Astrophys. J. 178:623-666 (the inclined encounter); the rest is Plexus (this work)."""
    PARAM_UNITS = {'radius': 'length', 'centre': 'length', 'offset': 'length', 'separation': 'length', 'thickness': 'length', 'tilt': 'angle', 'arms': 'count', 'seed': 'count'}
    SYMBOLS = {'x_i': 'the position of entity i', 'mode': "the placement rule (`mode`: sunflower, ball, two_disks, ...), the `_spawn*` placement rules of this file", 'r': "`radius`, the rule's one length, world length, from the centre (`centre`: a point, a set's live centroid, or the world box's centre)", 'i': 'the entity index, 1..n'}
    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["mode"]
    MECHANISM_TAGS = ["initial_condition", "placement"]
    PARAM_ROLES = {"mode": "placement_rule", "radius": "placement_radius", "seed": "rng_seed",
                   "thickness": "out_of_plane_scatter", "separation": "disc_separation",
                   "offset": "impact_parameter", "tilt": "disc_inclination", "arms": "spiral_arms",
                   "centre": "placement_centre_point_or_set_centroid_default_box_centre"}
    REFERENCE = ("Vogel, H. (1979). Math. Biosci. 44:179-189; Toomre, A. & Toomre, J. (1972). "
                 "Astrophys. J. 178:623-666; Plexus (this work).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.mode = str(params["mode"]).lower()
        self.radius = params.get("radius", 0.3)                   # a number, or [r0, r1] for two discs
        self.thickness = params.get("thickness", 0.0)             # (3D disk / two_disks) a number or a pair
        self.separation = float(params.get("separation", 0.0))  # (two_disks) between the disc centres
        self.offset = float(params.get("offset", 0.0))  # (two_disks) impact parameter
        self.tilt = float(params.get("tilt", 0.0))  # (two_disks) radians about x
        self.arms = params.get("arms", None)                      # (two_disks) {amplitude, m, pitch}
        self.seed = int(params.get("seed", 0))
        self.centre = params.get("centre", None)                  # a point, or a set's name; None = box centre

    def _centre(self, H, D, dev):
        """The placement's centre in world units: the declared point, or the named set's live centroid."""
        if isinstance(self.centre, str):
            src = H.level(self.centre)
            live = src.occ > 0
            m = getattr(src, "_mesh", None)
            if m is not None and int(m.get("Nv", 0) or 0):          # a mesh set: its seeded vertices
                live = torch.zeros_like(live)
                live[: int(m["Nv"])] = True
            if not bool(live.any()):
                raise ValueError(f"seed_positions: centre set {self.centre!r} has nothing live to centre on -- "
                                 f"seed it earlier in the `seed:` list")
            return src.get("pos")[live].float().mean(0).to(dev)
        c = torch.as_tensor([float(v) for v in self.centre], device=dev)
        if c.numel() != D:
            raise ValueError(f"seed_positions: centre {self.centre} is not a {D}D point")
        return c

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        if "pos" not in lvl.state_schema:
            raise ValueError(f"seed_positions: set {self.at!r} has no `pos` block to place")
        p0, p1 = lvl.state_schema["pos"]
        n, D = int(lvl.occ.sum().item()), p1 - p0
        dev = lvl.state.device
        rng = torch.Generator(device=dev).manual_seed(self.seed)
        gid = rot = None
        if self.mode in ("two_disks", "disk_pair"):
            if D != 3:
                raise ValueError("seed_positions: `two_disks` is a 3D placement")
            pos, head, gid, rot = _spawn_pair3d(
                n, H.world_size, self.radius, rng, dev, thickness=self.thickness,
                separation=self.separation, offset=self.offset, tilt=self.tilt, arms=self.arms)
        elif D == 3:
            pos, head = _spawn3d(self.mode, n, H.world_size, float(self.radius), rng, dev,
                                 thickness=float(self.thickness))
        else:
            pos, head = _spawn(self.mode, n, H.world_size, float(self.radius), rng, dev)
        if self.centre is not None:                               # translated, the draws unchanged
            box = torch.as_tensor(H.world_size, dtype=pos.dtype, device=dev)[:D]
            pos = pos - 0.5 * box + self._centre(H, D, dev).to(pos.dtype)
        st = lvl.state.clone()
        st[:n, p0:p1] = pos.to(st.dtype)
        lvl.state = st
        buffer = lvl.state.shape[0]
        hbuf = torch.zeros(buffer, head.shape[1], device=dev); hbuf[:n] = head
        if getattr(lvl, "heading", None) is not None and lvl.heading.shape == hbuf.shape:
            lvl.heading = hbuf
        else:
            lvl.register_buffer("heading", hbuf)
        if gid is not None:
            gbuf = torch.zeros(buffer, dtype=torch.long, device=dev); gbuf[:n] = gid
            if getattr(lvl, "spawn_group", None) is not None:
                lvl.spawn_group = gbuf
            else:
                lvl.register_buffer("spawn_group", gbuf)
            lvl.spawn_group_rot = rot
        return {}


@register_operator("seed_positions", family="seed", set="cell", kind="seed", model="packed_ball",
                   title="pack cell centres in a ball",
                   equation=r"""$$\mathbf x_i=\mathbf c+a\,(\mathbf k_i+\boldsymbol\epsilon_i),\qquad \mathbf k_i\in\mathbb Z^3,\ \ |\mathbf k_i|\ \text{the }n\text{ smallest},\ \ \epsilon_{i,a}\sim U(-j,j)$$""")
class SeedPositionsPackedBall(SeedPositions):
    r"""the centres of a packed mass of cells: the n sites of a cubic lattice of pitch `spacing` closest to `centre`, each jittered by up to `jitter` of a pitch on every axis.
    set -> set: writes `pos` for the set's n live elements.

        \mathbf x_i=\mathbf c+a\,(\mathbf k_i+\boldsymbol\epsilon_i),\qquad \mathbf k_i\in\mathbb Z^3,\ \ |\mathbf k_i|\ \text{the }n\text{ smallest},\ \ \epsilon_{i,a}\sim U(-j,j)
        x_i            the position of cell i
        c              `centre` (a point, or a set whose live centroid is read)
        a              `spacing`, the centre-to-centre distance of the packing, world length
        k_i            the n integer lattice sites nearest the origin
        \epsilon_i     the jitter, each component uniform in (-j, j)
        j              `jitter`, a fraction of a pitch, below 0.5 so no two sites swap
        n              the set's live count

    Reference: none: an initial condition. Plexus (this work)."""
    PARAM_UNITS = {'spacing': 'length', 'jitter': 'fraction'}
    SYMBOLS = {'x_i': 'the position of cell i', 'c': '`centre` (a point, or a set whose live centroid is read)', 'a': '`spacing`, the centre-to-centre distance of the packing, world length', 'k_i': 'the n integer lattice sites nearest the origin', '\\epsilon_i': 'the jitter, each component uniform in (-j, j)', 'j': '`jitter`, a fraction of a pitch, below 0.5 so no two sites swap', 'n': "the set's live count"}
    REQUIRES_PARAMS = ["spacing"]
    MAY_MUTATE_INTEGRATED_STATE = True
    PARAM_ROLES = {"spacing": "centre_to_centre_distance", "jitter": "lattice_jitter_fraction",
                   "centre": "placement_centre_point_or_set_centroid", "seed": "rng_seed"}

    def __init__(self, params, device="cpu"):
        p2 = dict(params); p2.setdefault("mode", "ball")
        super().__init__(p2, device)
        self.spacing = float(params["spacing"])
        self.jitter = float(params.get("jitter", 0.2))
        if not 0.0 <= self.jitter < 0.5:
            raise ValueError("seed_positions[packed_ball]: jitter is a fraction of a pitch in [0, 0.5)")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        p0, p1 = lvl.state_schema["pos"]
        n, D = int(lvl.occ.sum().item()), p1 - p0
        if D != 3:
            raise ValueError("seed_positions[packed_ball] is a 3D placement")
        dev = lvl.state.device
        k = int(math.ceil((3.0 * n / (4.0 * math.pi)) ** (1.0 / 3.0))) + 2
        r = torch.arange(-k, k + 1, dtype=torch.float64)
        g = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
        g = g[torch.argsort((g * g).sum(1), stable=True)][:n]
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        eps = (torch.rand(n, 3, generator=gen, dtype=torch.float64) * 2.0 - 1.0) * self.jitter
        c = (self._centre(H, D, "cpu").double() if self.centre is not None else
             0.5 * torch.as_tensor(H.world_size, dtype=torch.float64)[:3])
        pos = c + self.spacing * (g + eps)
        st = lvl.state.clone()
        st[:n, p0:p1] = pos.to(device=dev, dtype=st.dtype)
        lvl.state = st
        return {}


@register_operator("seed_positions", family="seed", set="cell", kind="seed", model="packed_sheet",
                   title="pack cell centres in a sheet",
                   equation=r"""$$\mathbf x_i=\mathbf c+a\,\big(k_i\,\mathbf e_1+l_i\,\mathbf e_2+\boldsymbol\epsilon_i\big),\qquad (k_i,l_i)\ \text{the }n\text{ sites of a triangular lattice nearest }\mathbf c$$""")
class SeedPositionsPackedSheet(SeedPositionsPackedBall):
    r"""the centres of a monolayer: the n sites of a triangular lattice of pitch `spacing` in the plane through `centre` normal to `plane_axis`, each jittered in the plane (six neighbours, the epithelial mean).
    set -> set: writes `pos` for the set's n live elements.

        \mathbf x_i=\mathbf c+a\,\big(k_i\,\mathbf e_1+l_i\,\mathbf e_2+\boldsymbol\epsilon_i\big),\qquad (k_i,l_i)\ \text{the }n\text{ sites of a triangular lattice nearest }\mathbf c
        x_i            the position of cell i
        c              `centre` (a point, a set's live centroid, or the world box's centre)
        a              `spacing`, world length
        (k_i, l_i)     the triangular lattice coordinates (u + v/2, v sqrt 3 / 2) of the n sites nearest the centre (`shape: disc`) or of a strip
        e_1, e_2       the two axes other than `plane_axis`
        \epsilon_i     the in-plane jitter, uniform in (-j, j) on each axis
        n              the set's live count

    Reference: none: an initial condition. Plexus (this work). The six-neighbour packing is the epithelial mean: Lewis, F. T. (1928). Anat. Rec. 38:341; Gibson, M. C. et al. (2006). Nature 442:1038."""
    PARAM_UNITS = {'plane_axis': 'count'}
    SYMBOLS = {'x_i': 'the position of cell i', 'c': "`centre` (a point, a set's live centroid, or the world box's centre)", 'a': '`spacing`, world length', '(k_i, l_i)': 'the triangular lattice coordinates (u + v/2, v sqrt 3 / 2) of the n sites nearest the centre (`shape: disc`) or of a strip', 'e_1, e_2': 'the two axes other than `plane_axis`', '\\epsilon_i': 'the in-plane jitter, uniform in (-j, j) on each axis', 'n': "the set's live count"}
    PARAM_ROLES = dict(SeedPositionsPackedBall.PARAM_ROLES, plane_axis="sheet_normal_axis", shape="disc_or_square")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.plane_axis = int(params.get("plane_axis", 2))
        self.shape = str(params.get("shape", "disc"))
        if self.shape not in ("disc", "square"):
            raise ValueError("seed_positions[packed_sheet]: shape is disc or square")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        p0, p1 = lvl.state_schema["pos"]
        n, D = int(lvl.occ.sum().item()), p1 - p0
        if D != 3:
            raise ValueError("seed_positions[packed_sheet] is a 3D placement")
        k = int(math.ceil(math.sqrt(n))) + 3
        r = torch.arange(-k, k + 1, dtype=torch.float64)
        U, V = torch.meshgrid(r, r, indexing="ij")
        g2 = torch.stack([U + 0.5 * V, V * math.sqrt(3.0) / 2.0], -1).reshape(-1, 2)
        if self.shape == "square":
            # the max-norm, with a tie-break on the euclidean norm so a ring of equal max-norm fills from its middle
            key = g2.abs().max(1).values + 1e-6 * (g2 * g2).sum(1)
        else:
            key = (g2 * g2).sum(1)
        g2 = g2[torch.argsort(key, stable=True)][:n]
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        eps = (torch.rand(n, 2, generator=gen, dtype=torch.float64) * 2.0 - 1.0) * self.jitter
        ax = [a for a in range(3) if a != self.plane_axis]
        g = torch.zeros(n, 3, dtype=torch.float64)
        g[:, ax[0]] = g2[:, 0] + eps[:, 0]
        g[:, ax[1]] = g2[:, 1] + eps[:, 1]
        c = (self._centre(H, D, "cpu").double() if self.centre is not None else
             0.5 * torch.as_tensor(H.world_size, dtype=torch.float64)[:3])
        pos = c + self.spacing * g
        st = lvl.state.clone()
        st[:n, p0:p1] = pos.to(device=lvl.state.device, dtype=st.dtype)
        lvl.state = st
        return {}


@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="voronoi_parent",
                   title="fill each cell's own region with points",
                   equation=r"""$$\mathbf x_i\sim U\big(\{\mathbf x:\ \arg\min_{c}|\mathbf x-\mathbf x_c|=\pi(i),\ |\mathbf x-\mathbf o|<R\}\big),\qquad v_i=\frac{V_{\pi(i)}}{n_{\pi(i)}}$$""")
class SeedPositionsVoronoiParent(Seed):
    r"""every contained point placed inside its parent's Voronoi cell, so a mass of cells made of material points is space-filling and its cells are irregular polyhedra, not overlapping balls.
    particle -[containment]-> cell: reads the parent set's live centres; writes the points' `pos` (zero `vel`) and, on an MPM set, each point's `p_vol` and `mass` (the Voronoi cell's volume shared among its points).

        \mathbf x_i\sim U\big(\{\mathbf x:\ \arg\min_{c}|\mathbf x-\mathbf x_c|=\pi(i),\ |\mathbf x-\mathbf o|<R\}\big),\qquad v_i=\frac{V_{\pi(i)}}{n_{\pi(i)}}
        x_i            the position of point i, uniform over its parent's Voronoi region
        \pi(i)         point i's parent (the containment map)
        x_c            the centre of cell c
        o, R           the clip ball (`centre`, `clip_radius`); no clip without them
        v_i            the point's volume
        V_{\pi(i)}     the volume of the parent's region
        n_{\pi(i)}     the parent's point count

    Reference: Voronoi, G. (1908). J. Reine Angew. Math. 134:198-287 (the tessellation); Honda, H. (1978). J. Theor. Biol. 72:523-543 (cells of a packed tissue as the Voronoi cells of their centres)."""
    PARAM_UNITS = {'centre': 'length', 'clip_radius': 'length', 'reach': 'length', 'seed': 'count'}
    SYMBOLS = {'x_i': "the position of point i, uniform over its parent's Voronoi region", '\\pi(i)': "point i's parent (the containment map)", 'x_c': 'the centre of cell c', 'o, R': 'the clip ball (`centre`, `clip_radius`); no clip without them', 'v_i': "the point's volume", 'V_{\\pi(i)}': "the volume of the parent's region", 'n_{\\pi(i)}': "the parent's point count"}
    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["initial_condition", "voronoi", "space_filling", "encapsulation"]
    PARAM_ROLES = {"clip_radius": "clip_ball_radius", "centre": "clip_ball_centre_point_or_set",
                   "reach": "largest_distance_from_own_centre", "seed": "rng_seed", 'label': 'block_that_receives_the_cell_label', 'parent_id': 'block_that_receives_the_parent_index'}
    REFERENCE = "Voronoi, G. (1908). J. Reine Angew. Math. 134:198-287; Honda, H. (1978). J. Theor. Biol. 72:523-543."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.clip = params.get("clip_radius")
        self.centre = params.get("centre")
        self.reach = params.get("reach")
        self.seed = int(params.get("seed", 0))
        # `label: <block>` -- A COLOUR PER CELL: writes (the parent's rank mod 20) into that width-1
        # block of the points, for `plotting.color_field: <block>` with a 20-colour map. Opt-in.
        self.label = params.get("label")
        # `parent_id: <block>` -- THE CELL'S OWN INDEX on each of its points, for `pair_potential ...
        # exclude: <block>`, which then acts only BETWEEN cells (a contact force), never inside one.
        self.parent_id = params.get("parent_id")

    def forward(self, H, mask=None):
        import numpy as np
        from scipy.spatial import cKDTree
        lvl = H.level(self.at)
        pname = getattr(lvl, "parent_name", None)
        if pname is None or getattr(lvl, "parent", None) is None:
            raise ValueError(f"seed_positions[voronoi_parent]: {self.at!r} has no `parent:` set")
        par = H.level(pname)
        live_c = (par.occ > 0).cpu().numpy()
        C = par.get("pos").detach().double().cpu().numpy()
        cid = np.flatnonzero(live_c)
        tree = cKDTree(C[cid])
        pidx = lvl.parent.long().cpu().numpy()
        live_p = (lvl.occ > 0).cpu().numpy()
        reach = float(self.reach) if self.reach is not None else \
            float(np.median(tree.query(C[cid], k=2)[0][:, 1]))
        o = None
        if self.clip is not None:
            if isinstance(self.centre, str):
                src = H.level(self.centre); m = getattr(src, "_mesh", None)
                lv = (src.occ > 0).cpu().numpy()
                if m is not None and int(m.get("Nv", 0) or 0):
                    lv = np.zeros_like(lv); lv[: int(m["Nv"])] = True
                o = src.get("pos").detach().double().cpu().numpy()[lv].mean(0)
            else:
                o = np.asarray(self.centre if self.centre is not None else C[cid].mean(0), float)
        rng = np.random.default_rng(self.seed)
        X = lvl.get("pos").detach().double().cpu().numpy().copy()
        pv = getattr(lvl, "p_vol", None); ms = getattr(lvl, "mass", None)
        rho = None
        if pv is not None and ms is not None:
            _pv = pv.detach().double().cpu().numpy(); _ms = ms.detach().double().cpu().numpy()
            rho = float(np.median(_ms[live_p] / np.maximum(_pv[live_p], 1e-30)))
            new_pv = _pv.copy()
        ball = 4.0 / 3.0 * math.pi * reach ** 3
        slot = {int(c): np.flatnonzero((pidx == c) & live_p) for c in cid}
        for rank, c in enumerate(cid):
            need = slot[int(c)]
            if need.size == 0:
                continue
            got, tried, acc_n = [], 0, 0
            M = max(8 * need.size, 256)
            while len(got) < need.size and tried < 40 * need.size + 4096:
                u = rng.normal(size=(M, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
                cand = C[c] + u * reach * rng.random((M, 1)) ** (1.0 / 3.0)
                ok = tree.query(cand)[1] == rank
                if o is not None:
                    ok &= np.linalg.norm(cand - o, axis=1) < float(self.clip)
                got.extend(cand[ok][: need.size - len(got)]); acc_n += int(ok.sum()); tried += M
            if len(got) < need.size:
                raise ValueError(f"seed_positions[voronoi_parent]: cell {int(c)} holds too little room "
                                 f"for its {need.size} points (clip too tight or `reach` too small)")
            X[need] = np.asarray(got[: need.size])
            if rho is not None:
                new_pv[need] = (acc_n / max(tried, 1)) * ball / need.size
        p0, p1 = lvl.state_schema["pos"]
        st = lvl.state.clone()
        st[:, p0:p1] = torch.as_tensor(X, dtype=st.dtype, device=st.device)
        if "vel" in lvl.state_schema:
            v0, v1 = lvl.state_schema["vel"]
            st[:, v0:v1] = 0.0
        if self.label is not None:
            if str(self.label) not in lvl.state_schema:
                raise ValueError(f"seed_positions[voronoi_parent]: label block {self.label!r} is not declared "
                                 f"on {self.at!r} -- add `{self.label}: {{width: 1}}` to its state")
            l0, l1 = lvl.state_schema[str(self.label)]
            rank = np.full(C.shape[0], -1); rank[cid] = np.arange(cid.size)
            lab = np.where(rank[pidx] >= 0, rank[pidx] % 20, 0).astype(float)
            st[:, l0] = torch.as_tensor(lab, dtype=st.dtype, device=st.device)
        if self.parent_id is not None:
            if str(self.parent_id) not in lvl.state_schema:
                raise ValueError(f"seed_positions[voronoi_parent]: parent_id block {self.parent_id!r} is not "
                                 f"declared on {self.at!r} -- add `{self.parent_id}: {{width: 1}}` to its state")
            q0, _q1 = lvl.state_schema[str(self.parent_id)]
            st[:, q0] = torch.as_tensor(pidx.astype(float), dtype=st.dtype, device=st.device)
        lvl.state = st
        if rho is not None:
            pv.copy_(torch.as_tensor(new_pv, dtype=pv.dtype, device=pv.device))
            ms.copy_(torch.as_tensor(new_pv * rho, dtype=ms.dtype, device=ms.device))
        return {}


# ============================================================================================
#  a colony on a lattice of SITES (exp 15, rig 1; moved from colony_ops.py 2026-09-27, unchanged)
# ============================================================================================
# The set's buffer holds every SITE of the domain, placed once by `seed_positions`; a site is
# occupied when its slot is live (`occ = 1`) and empty when it is dormant, and the strain is two
# one-hot columns of `chem` at `chan`. A division wakes one specific empty slot beside the mother, so
# nothing moves and the colony's history is written into WHERE each strain sits. `Level.spawn` is not
# used: it wakes the FIRST free slots and copies the mother's position into them, right for an
# off-lattice body and wrong for a site. Momeni et al. 2013 (eLife 2:e00230): two cooperating strains
# intermix, competitors segregate -- their individual-based fitness model, every constant as printed.


def _strain_span(lvl, chan, who):
    if "chem" not in lvl.state_schema:
        raise ValueError(f"{who}: the set has no `chem` block; declare `chem: {{width: 2}}` for the two strains")
    h0, h1 = lvl.state_schema["chem"]
    if h0 + chan + 2 > h1:
        raise ValueError(f"{who}: chem is {h1 - h0} wide; the two strain columns at chan={chan} do not fit")
    return h0 + chan


@register_operator("cloud_seed", family="seed", set="particle", kind="seed",
                   equation=r"""$$\mathbf x_i=\mathbf x_{\mathrm{origin}}+\text{scale}\cdot\mathbf q_i\quad\text{or, contained,}\quad\mathbf x_i=\mathbf x_{\pi(i)}+\text{scale}\cdot\mathbf q_{k(i)}$$""", title="place a measured point cloud")
class CloudSeed(Seed):
    r"""writes every position of a set from a measured point cloud of the shape library, once: one copy per parent for a contained set, each copy turned by its parent's angles and inserted into a host when asked.
    set -> set: writes `pos` (and clears the host's `occ` under each copy with `displace`); the set's n must equal the count the file and the stride give.

        \mathbf x_i=\mathbf x_{\mathrm{origin}}+\text{scale}\cdot\mathbf q_i\quad\text{or, contained,}\quad\mathbf x_i=\mathbf x_{\pi(i)}+\text{scale}\cdot\mathbf q_{k(i)}
        x_i            the position of point i
        x_{origin}     `origin`, where the cloud's centre goes, world length
        scale          `scale`, metres of the cloud to world length
        q_i            the i-th point of the cloud (`cloud`: `<shape>/<part>`, metres, in the cloud's own frame)
        \pi(i)         the parent of point i (a contained set)
        q_{k(i)}       the cloud point of rank k(i) in its parent's block

    Reference: Plexus (this work); the shape library is src/plexus/shapes.py."""
    PARAM_UNITS = {'origin': 'length', 'scale': 'dimensionless', 'every': 'count', 'reach': 'length'}
    SYMBOLS = {'x_i': 'the position of point i', 'x_{origin}': "`origin`, where the cloud's centre goes, world length", 'scale': '`scale`, metres of the cloud to world length', 'q_i': "the i-th point of the cloud (`cloud`: `<shape>/<part>`, metres, in the cloud's own frame)", '\\pi(i)': 'the parent of point i (a contained set)', 'q_{k(i)}': "the cloud point of rank k(i) in its parent's block"}
    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["cloud"]
    MECHANISM_TAGS = ["initial_condition", "placement", "measured_anatomy"]
    PARAM_ROLES = {"cloud": "point_cloud", "origin": "placement_origin", "scale": "world_per_metre",
                   "every": "stride", "rotate": "parent_block_of_euler_degrees",
                   "displace": "host_set_cleared_under_the_cloud", "reach": "outline_margin_world"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.cloud = str(params["cloud"])
        self.origin_given = params.get("origin") is not None
        self.origin = [float(v) for v in (params.get("origin") or [0.5, 0.5, 0.5])]
        self.scale = float(params.get("scale", 1.0))
        self.every = max(1, int(params.get("every", 1)))
        self.rotate = params.get("rotate")
        self.displace = params.get("displace")
        self.reach = float(params.get("reach", 0.0))
        if "/" not in self.cloud:
            raise ValueError(f"cloud_seed: `cloud` names `<shape>/<part>`, got {self.cloud!r}")

    def forward(self, H, mask=None):
        import os
        import numpy as np
        from plexus import shapes
        shape, part = self.cloud.split("/", 1)
        path = next((os.path.join(r, shape, "points.npz") for r in shapes.roots()
                     if os.path.exists(os.path.join(r, shape, "points.npz"))), None)
        if path is None:
            raise FileNotFoundError(f"cloud_seed: no shapes/{shape}/points.npz under any of {shapes.roots()}")
        with np.load(path) as z:
            if part not in z:
                raise KeyError(f"cloud_seed: {path} has no part {part!r} (it has {', '.join(sorted(z.keys()))})")
            q = np.asarray(z[part], np.float64)[:: self.every]
        p = H.level(self.at)
        n = int(p.state.shape[0])
        dev, dt = p.state.device, p.state.dtype
        qw = torch.as_tensor(q, device=dev, dtype=dt) * self.scale
        pidx = p.parent
        if p.parent_name is not None and pidx.numel() == n:
            if self.origin_given:
                raise ValueError(f"cloud_seed: set {self.at!r} is contained in {p.parent_name!r}, so each copy "
                                 f"lands on its parent's position; drop `origin`")
            par = H.level(p.parent_name)
            counts = torch.bincount(pidx, minlength=par.n)
            if not bool((counts == len(q)).all()):
                raise ValueError(f"cloud_seed: set {self.at!r} holds {sorted(set(counts.tolist()))} points per "
                                 f"{p.parent_name!r} but {self.cloud} gives {len(q)} at every={self.every}; "
                                 f"declare per_parent: {len(q)}")
            pp = str((H.config.sets.get(self.at) or {}).get("parent_pos", "pos"))
            k = torch.arange(n, device=dev) - (torch.cumsum(counts, 0) - counts)[pidx]
            centre = par.get(pp)[:, :3].to(dt)
            qk = qw[k]
            where = f"{par.n} copies of {len(q)}, one per {p.parent_name!r}"
            if self.rotate:
                from plexus.models.entities import _rot_matrix
                if self.rotate not in par.state_schema:
                    raise ValueError(f"cloud_seed: `rotate: {self.rotate}` names no block of {p.parent_name!r}; "
                                     f"declare it under `sets.{p.parent_name}.state:` (width 1 = degrees about z, "
                                     f"3 = [x, y, z] degrees)")
                ang = par.get(self.rotate).tolist()
                ang = [[0.0, 0.0, a[0]] if len(a) == 1 else a for a in ang]
                R = torch.stack([_rot_matrix(a, 3, dev) for a in ang]).to(dt)
                qk = torch.einsum("nij,nj->ni", R[pidx], qk)
                where += f", each turned by its own {p.parent_name}.{self.rotate}"
            pos = centre[pidx] + qk
        else:
            if self.rotate:
                raise ValueError(f"cloud_seed: `rotate` reads a block of the PARENT, and {self.at!r} has none")
            if n != len(q):
                raise ValueError(f"cloud_seed: set {self.at!r} holds {n} entities but {self.cloud} gives "
                                 f"{len(q)} points at every={self.every}; declare n: {len(q)}")
            centre = torch.tensor([self.origin], device=dev, dtype=dt)
            pidx = torch.zeros(n, dtype=torch.long, device=dev)
            pos = qw + centre[0]
            where = f"{n} points"
        st = p.state.clone()
        a, b = p.state_schema["pos"]
        st[:, a:b] = pos
        p.state = st
        r = (pos[:, :2] - centre[pidx, :2]).norm(dim=1)
        print(f"[cloud_seed] {self.at}: {where} from {os.path.relpath(path)}::{part} (every {self.every}), "
              f"scale {self.scale:.4g} world/m, r {float(r.min()):.4f}-{float(r.max()):.4f} world from its centre, "
              f"z {float(pos[:, 2].min()):.4f}-{float(pos[:, 2].max()):.4f}", flush=True)
        if self.displace:
            self._displace(H, pos, pidx, centre)
        return {}

    def _displace(self, H, pos, pidx, centre, nb=72):
        host = H.level(self.displace)
        hx = host.get("pos")[:, :3].to(pos.dtype)
        live = host.occ > 0
        z0, z1 = hx[live, 2].min(), hx[live, 2].max()
        m = (pos[:, 2] >= z0) & (pos[:, 2] <= z1)
        d = pos[m, :2] - centre[pidx[m], :2]
        sector = ((torch.atan2(d[:, 1], d[:, 0]) + math.pi) / (2 * math.pi) * nb).long().clamp(0, nb - 1)
        P = centre.shape[0]
        R = torch.zeros(P * nb, device=pos.device, dtype=pos.dtype)
        R.scatter_reduce_(0, pidx[m] * nb + sector, d.norm(dim=1) + self.reach, reduce="amax")
        R = R.view(P, nb)
        e = hx[:, None, :2] - centre[None, :, :2]
        es = ((torch.atan2(e[..., 1], e[..., 0]) + math.pi) / (2 * math.pi) * nb).long().clamp(0, nb - 1)
        under = (e.norm(dim=2) < R[torch.arange(P, device=pos.device)[None, :], es]).any(1) & live
        host.occ[under] = 0.0
        print(f"[cloud_seed] {self.at} displaced {int(under.sum()):,} of {int(live.sum()):,} live "
              f"{self.displace} points, reach {self.reach:.4g} world", flush=True)


# ==========================================================================================================
# seed_positions[lattice_body] -- an MPM body laid out on a lattice (a ball) or in hole-free layers (a
# pre-stretched shell), its point volumes and masses set from the body's real geometry.
# ==========================================================================================================
def _corotated_free_normal_stretch(lam: float, nu: float) -> float:
    """The thickness stretch that leaves a fixed-corotated sheet, stretched by `lam` in its own plane,
    with NO stress across its thickness.

    For F = diag(lam, lam, s) (R = I) the Kirchhoff stress across the sheet is
        tau_nn = 2 mu (s - 1) s + la (J - 1) J,      J = lam^2 s,
    with mu = E / (2 (1 + nu)) and la = E nu / ((1 + nu)(1 - 2 nu)); E cancels. tau_nn rises with s
    from -2 mu ... at s -> 0 to positive at s = 1, so it has one root in (0, 1], found by bisection.
    """
    mu = 1.0 / (2.0 * (1.0 + nu))
    la = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def tau(s):
        J = lam * lam * s
        return 2.0 * mu * (s - 1.0) * s + la * (J - 1.0) * J

    lo, hi = 1e-4, 1.0
    if tau(hi) <= 0.0:                     # lam <= 1: a sheet in compression; no thinning
        return 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if tau(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="lattice_body",
                   title="lay out a solid body of points",
                   equation=r"""$$\text{ball: } \mathbf x_i \in h\,\mathbb Z^3,\ |\mathbf x_i-\mathbf o|\le R;\qquad
\text{shell: } \mathbf F_i=\lambda(\mathbf I-\hat{\mathbf n}_i\hat{\mathbf n}_i^{\!\top})+s\,\hat{\mathbf n}_i\hat{\mathbf n}_i^{\!\top},\ \ v_i=\frac{4\pi R^2 t}{n\,J}$$""")
class SeedPositionsLatticeBody(Seed):
    r"""lays an MPM set out as one body with no holes before anything moves: a solid ball on a cubic lattice, or a thin shell in concentric layers spread evenly over each sphere, each point given the volume and mass that body really has.
    particle -> particle: writes `pos` (zero `vel`), `p_vol`, `mass` and, for a pre-stretched shell, the deformation gradient `F`.

        \text{ball: } \mathbf x_i \in h\,\mathbb Z^3,\ |\mathbf x_i-\mathbf o|\le R;\qquad \text{shell: } \mathbf F_i=\lambda(\mathbf I-\hat{\mathbf n}_i\hat{\mathbf n}_i^{\!\top})+s\,\hat{\mathbf n}_i\hat{\mathbf n}_i^{\!\top},\ \ v_i=\frac{4\pi R^2 t}{n\,J}
        x_i            the position of point i (ball: the n lattice sites of pitch h nearest the centre o within R; shell: `layers` Fibonacci spheres between r_in and r_out)
        h              the lattice pitch
        o              `centre`
        R              `radius`
        F_i            the shell's pre-stretch as the initial deformation gradient: lambda in the plane, s through the thickness
        \lambda        `prestretch`, the in-plane stretch (1 = unstretched)
        \hat n_i       the outward normal at point i
        s              the thickness stretch the fixed-corotated law solves for at Poisson ratio `nu`
        v_i            the point's volume: the wall's real volume 4 pi R^2 t over n points, over J
        t              `thickness`, the wall's real current thickness, world length
        n              the point count
        J              det F

    Reference: Gonzalez, A. (2010). Measurement of areas on a sphere using Fibonacci and latitude-longitude lattices. Math. Geosci. 42:49-64; Stomakhin, A. et al. (2012). Energetically consistent invertible elasticity. SCA (the fixed-corotated law the thickness stretch solves)."""
    PARAM_UNITS = {'centre': 'length', 'radius': 'length', 'r_in': 'length', 'r_out': 'length', 'layers': 'count', 'thickness': 'length', 'prestretch': 'dimensionless', 'nu': 'dimensionless', 'seed': 'count'}
    SYMBOLS = {'x_i': 'the position of point i (ball: the n lattice sites of pitch h nearest the centre o within R; shell: `layers` Fibonacci spheres between r_in and r_out)', 'h': 'the lattice pitch', 'o': '`centre`', 'R': '`radius`', 'F_i': "the shell's pre-stretch as the initial deformation gradient: lambda in the plane, s through the thickness", '\\lambda': '`prestretch`, the in-plane stretch (1 = unstretched)', '\\hat n_i': 'the outward normal at point i', 's': 'the thickness stretch the fixed-corotated law solves for at Poisson ratio `nu`', 'v_i': "the point's volume: the wall's real volume 4 pi R^2 t over n points, over J", 't': "`thickness`, the wall's real current thickness, world length", 'n': 'the point count', 'J': 'det F'}
    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["shape", "centre"]
    MECHANISM_TAGS = ["initial_condition", "lattice", "prestretch", "thin_shell"]
    PARAM_ROLES = {"shape": "body_shape", "centre": "body_centre", "radius": "ball_radius",
                   "r_in": "shell_inner_placement_radius", "r_out": "shell_outer_placement_radius",
                   "layers": "shell_point_layers", "thickness": "shell_real_current_wall_thickness",
                   "prestretch": "in_plane_stretch_of_the_shell_relative_to_rest",
                   "nu": "poisson_ratio_for_the_free_thickness_stretch", "seed": "rng_seed", 'points': 'fibonacci_or_icosphere'}
    REFERENCE = ("Gonzalez, A. (2010). Math. Geosci. 42:49-64; Stomakhin, A. et al. (2012). SCA "
                 "(fixed-corotated elasticity).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.shape = str(params["shape"]).lower()
        if self.shape not in ("ball", "shell"):
            raise ValueError(f"seed_positions[lattice_body]: shape must be ball or shell, got {self.shape!r}")
        self.centre = [float(v) for v in params["centre"]]
        self.radius = params.get("radius")
        self.r_in = params.get("r_in")
        self.r_out = params.get("r_out")
        self.layers = int(params.get("layers", 2))
        self.points = str(params.get("points", "fibonacci")).lower()
        if self.points not in ("fibonacci", "icosphere"):
            raise ValueError(f"seed_positions[lattice_body]: points must be fibonacci or icosphere, got {self.points!r}")
        self.thickness = params.get("thickness")
        self.prestretch = float(params.get("prestretch", 1.0))
        self.nu = float(params.get("nu", 0.2))
        self.seed = int(params.get("seed", 0))
        if self.shape == "ball" and self.radius is None:
            raise ValueError("seed_positions[lattice_body]: a ball needs `radius`")
        if self.shape == "shell" and (self.r_in is None or self.r_out is None or self.thickness is None):
            raise ValueError("seed_positions[lattice_body]: a shell needs `r_in`, `r_out` and `thickness`")

    @staticmethod
    def _ball(n, R, dev):
        """The n points of a cubic lattice nearest the centre, spacing chosen so they fill radius R."""
        h = (4.0 / 3.0 * math.pi * R ** 3 / n) ** (1.0 / 3.0)
        m = int(math.ceil(1.15 * R / h)) + 1
        ax = (torch.arange(-m, m + 1, device=dev, dtype=torch.float64) + 0.5) * h
        g = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
        r = g.norm(dim=1)
        if g.shape[0] < n:
            raise ValueError(f"lattice_body: lattice too small ({g.shape[0]} < {n})")
        order = torch.argsort(r)[:n]
        return g[order], h

    @staticmethod
    def _icosphere(n, r, dev):
        """The vertices of an icosahedron subdivided k times (10 * 4^k + 2 of them), on radius r."""
        import numpy as np
        k = 0
        while 10 * 4 ** k + 2 < n:
            k += 1
        if 10 * 4 ** k + 2 != n:
            raise ValueError(f"lattice_body: an icosphere has 10*4^k+2 points (..., {10 * 4 ** max(k - 1, 0) + 2}, "
                             f"{10 * 4 ** k + 2}, ...); the set has {n}")
        t = (1.0 + 5.0 ** 0.5) / 2.0
        V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t],
                      [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=np.float64)
        Fc = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                       [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                       [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        for _ in range(k):
            E = np.concatenate([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]])
            Es = np.sort(E, 1)
            uniq, inv = np.unique(Es, axis=0, return_inverse=True)
            mid = V[uniq[:, 0]] + V[uniq[:, 1]]
            mid /= np.linalg.norm(mid, axis=1, keepdims=True)
            m = len(V) + inv.reshape(3, -1)
            a, b, c = Fc[:, 0], Fc[:, 1], Fc[:, 2]
            ab, bc, ca = m[0], m[1], m[2]
            Fc = np.concatenate([np.stack([a, ab, ca], 1), np.stack([b, bc, ab], 1),
                                 np.stack([c, ca, bc], 1), np.stack([ab, bc, ca], 1)])
            V = np.concatenate([V, mid])
        return torch.as_tensor(V * r, device=dev, dtype=torch.float64)

    @staticmethod
    def _fib(n, r, phase, dev):
        i = torch.arange(n, device=dev, dtype=torch.float64) + 0.5
        phi = torch.acos(1.0 - 2.0 * i / n)
        th = math.pi * (1.0 + 5.0 ** 0.5) * i + phase
        return torch.stack([r * torch.cos(th) * torch.sin(phi), r * torch.cos(phi),
                            r * torch.sin(th) * torch.sin(phi)], -1)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        live = lvl.occ > 0
        n = int(live.sum().item())
        idx = torch.nonzero(live).flatten()
        o = torch.as_tensor(self.centre, device=dev, dtype=torch.float64)
        pv = getattr(lvl, "p_vol", None); ms = getattr(lvl, "mass", None)
        if pv is None or ms is None:
            raise ValueError(f"seed_positions[lattice_body]: {self.at!r} is not an MPM set (no p_vol/mass)")
        rho = (ms[idx].double() / pv[idx].double().clamp_min(1e-30))      # the type's density, per point
        if self.shape == "ball":
            off, h = self._ball(n, float(self.radius), dev)
            vol = 4.0 / 3.0 * math.pi * float(self.radius) ** 3 / n
            note = f"lattice spacing {h:.4g}, radius {float(off.norm(dim=1).max()):.4g}"
        else:
            r_in, r_out, L = float(self.r_in), float(self.r_out), max(1, self.layers)
            radii = [r_in + (k + 0.5) * (r_out - r_in) / L for k in range(L)] if L > 1 else [0.5 * (r_in + r_out)]
            if self.points == "icosphere":
                if L != 1:
                    raise ValueError("seed_positions[lattice_body]: points: icosphere is one layer (layers: 1)")
                off = self._icosphere(n, radii[0], dev)
            else:
                w = torch.tensor([r * r for r in radii], dtype=torch.float64)
                cnt = torch.floor(w / w.sum() * n).long()
                cnt[-1] += n - int(cnt.sum())
                g = torch.Generator().manual_seed(self.seed)
                parts = [self._fib(int(c), r, float(torch.rand(1, generator=g)) * 2 * math.pi, dev)
                         for c, r in zip(cnt.tolist(), radii)]
                off = torch.cat(parts, 0)
            R = 0.5 * (r_in + r_out)
            lam = self.prestretch
            s = _corotated_free_normal_stretch(lam, self.nu)
            J = lam * lam * s
            vol = 4.0 * math.pi * R * R * float(self.thickness) / (n * J)
            nh = off / off.norm(dim=1, keepdim=True)
            eye = torch.eye(3, device=dev, dtype=torch.float64)
            nn = nh[:, :, None] * nh[:, None, :]
            F = lam * (eye - nn) + s * nn
            lvl.F[idx] = F.to(lvl.F.dtype)
            spacing = (4.0 * math.pi * R * R * L / n) ** 0.5
            note = (f"{L} layers r {r_in:.4g}-{r_out:.4g}, in-layer spacing {spacing:.4g}, prestretch {lam:g}, "
                    f"thickness stretch {s:.4f} (J {J:.4f}), wall {float(self.thickness):.3g} thick")
        pos = (o + off).to(lvl.state.dtype)
        a, b = lvl.state_schema["pos"]
        st = lvl.state.clone()
        st[idx, a:b] = pos
        if "vel" in lvl.state_schema:
            v0, v1 = lvl.state_schema["vel"]
            st[idx, v0:v1] = 0.0
        lvl.state = st
        pv[idx] = torch.full_like(pv[idx], vol)
        ms[idx] = (rho * vol).to(ms.dtype)
        print(f"[lattice_body] {self.at}: {self.shape} of {n:,} points at {self.centre}, {note}, "
              f"point volume {vol:.4g}, mass {float(ms[idx].sum()):.4g}", flush=True)
        return {}


# ==========================================================================================================
# seed_positions[cloud_filled] -- a body given as a point cloud that already carries each point's volume.
# PhysGaussian (Xie et al. 2024) simulates the splat's own Gaussian centres plus the particles it fills the
# inside with, and gives every point the volume of its MPM cell shared by the points in it, so the density
# is uniform although the points are not. A uniform `particle_mass` cannot say that.

@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="cloud_filled",
                   title="place a filled point cloud with a volume per point",
                   equation=r"""$$\mathbf x_i=\mathbf o+\mathbf q_i,\qquad v_i=\frac{\Delta x^3}{n_{c(i)}},\qquad m_i=\rho\,v_i$$""")
class SeedPositionsCloudFilled(Seed):
    r"""writes every position of an MPM set from a point cloud of the shape library, and gives each point the volume stored beside it and the mass that volume has at the type's density.
    particle -> particle: writes `pos`, `p_vol` and `mass` (the velocity the set's `vel_init` gave is kept); the set's n must equal the number of points of the cloud.

        \mathbf x_i=\mathbf o+\mathbf q_i,\qquad v_i=\frac{\Delta x^3}{n_{c(i)}},\qquad m_i=\rho\,v_i
        x_i            the position of point i
        o              `origin`, where the cloud's own origin goes, world length
        q_i            the i-th point of the cloud (`cloud`: `<shape>/<part>`, metres, in the cloud's own frame)
        v_i            the volume of point i, the entry `<part>_vol` of the cloud's points.npz: the volume of its MPM cell over the n_c points in that cell
        \Delta x       the MPM grid spacing the volumes were computed for
        n_{c(i)}       the number of points in the grid cell c(i) that holds point i
        m_i            the mass of point i
        \rho           the density of the point's type

    Reference: Xie, T. et al. (2024). PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics. CVPR (the filled points and their volumes, particle_filling/filling.py)."""
    PARAM_UNITS = {'origin': 'length'}
    SYMBOLS = {'x_i': 'the position of point i', 'o': "`origin`, where the cloud's own origin goes, world length",
               'q_i': "the i-th point of the cloud (`cloud`: `<shape>/<part>`, metres, in the cloud's own frame)",
               'v_i': "the volume of point i, the entry `<part>_vol` of the cloud's points.npz",
               '\\Delta x': 'the MPM grid spacing the volumes were computed for', 'n_{c(i)}': 'the number of points in the grid cell that holds point i',
               'm_i': 'the mass of point i', '\\rho': 'the density of the point\'s type'}
    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["cloud", "origin"]
    MECHANISM_TAGS = ["initial_condition", "measured_anatomy", "particle_volume"]
    PARAM_ROLES = {"cloud": "point_cloud", "origin": "placement_origin"}
    REFERENCE = "Xie, T. et al. (2024). PhysGaussian. CVPR."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.cloud = str(params["cloud"])
        self.origin = [float(v) for v in params["origin"]]
        if "/" not in self.cloud:
            raise ValueError(f"seed_positions[cloud_filled]: `cloud` names `<shape>/<part>`, got {self.cloud!r}")

    def forward(self, H, mask=None):
        import os
        import numpy as np
        from plexus import shapes
        shape, part = self.cloud.split("/", 1)
        path = next((os.path.join(r, shape, "points.npz") for r in shapes.roots()
                     if os.path.exists(os.path.join(r, shape, "points.npz"))), None)
        if path is None:
            raise FileNotFoundError(f"seed_positions[cloud_filled]: no shapes/{shape}/points.npz under any of {shapes.roots()}")
        with np.load(path) as z:
            for key in (part, part + "_vol"):
                if key not in z:
                    raise KeyError(f"seed_positions[cloud_filled]: {path} has no {key!r} (it has {', '.join(sorted(z.keys()))})")
            q = np.asarray(z[part], np.float64)
            v = np.asarray(z[part + "_vol"], np.float64)
        lvl = H.level(self.at)
        dev = lvl.state.device
        live = lvl.occ > 0
        idx = torch.nonzero(live).flatten()
        if len(idx) != len(q):
            raise ValueError(f"seed_positions[cloud_filled]: set {self.at!r} holds {len(idx)} entities but {self.cloud} gives "
                             f"{len(q)} points; declare n / per_parent so they agree")
        pv = getattr(lvl, "p_vol", None); ms = getattr(lvl, "mass", None)
        if pv is None or ms is None:
            raise ValueError(f"seed_positions[cloud_filled]: {self.at!r} is not an MPM set (no p_vol/mass)")
        rho = ms[idx].double() / pv[idx].double().clamp_min(1e-30)          # the type's density, per point
        pos = torch.as_tensor(q, device=dev, dtype=torch.float64) + torch.as_tensor(self.origin, device=dev, dtype=torch.float64)
        a, b = lvl.state_schema["pos"]
        st = lvl.state.clone()
        st[idx, a:b] = pos.to(st.dtype)
        lvl.state = st                                                      # `vel` stays: the set's own `vel_init` (a launch) was written at build
        vol = torch.as_tensor(v, device=dev, dtype=torch.float64)
        pv[idx] = vol.to(pv.dtype)
        ms[idx] = (rho * vol).to(ms.dtype)
        print(f"[cloud_filled] {self.at}: {len(q):,} points from {os.path.relpath(path)}::{part}, volume {float(vol.sum()):.4g} "
              f"(per point {float(vol.min()):.3g}-{float(vol.max()):.3g}), mass {float(ms[idx].sum()):.4g}", flush=True)
        return {}
