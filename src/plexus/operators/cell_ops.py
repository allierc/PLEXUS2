"""The cell as a COMPOSITION: one cell, many subcellular compartments, each compartment a
cloud of material points.

This module exists to exercise the claim of plexus2 sec. Hierarchy -- that a parent may hold
several distinct sets at once, and that a level is a heterogeneous collection rather than one
biological category -- on the object that claim is usually stated about, a eukaryotic cell.
The reference picture is `figures/cell_atlas.png`: an interactive cytology viewer that reports
a cell as 4,177 modelled PIECES distributed over 15 compartments (421 plasma-membrane pieces,
116 cytoskeletal filaments, 138 nuclear-envelope pieces, 31 chromatin strands, 2 nucleoli,
11 rough-ER cisternae, 3 smooth-ER cisternae, 14 Golgi cisternae, 77 mitochondria, and six
further compartments the screenshot's list scrolls past). A piece is not a cell and not a
material point; it is the intermediate level the atlas actually counts, and Plexus has to be
able to name it.

    cell            1 element              the body
      compartment   813 elements           a PIECE of an organelle -- one membrane patch, one
                                           mitochondrion, one Golgi cisterna. Carries `types`,
                                           one type per compartment of the atlas.
        mpm_particle  813,000 elements     material points, 1,000 per piece

Three levels, two containment maps, and the whole thing is declared in `sets:` -- the engine's
`build` already walks a chain of any depth, because pass 2 resolves children in declaration
order and a child's parent only has to exist by the time its own row is reached.

In the order they appear below:

    compartment      entity     a piece of an organelle: position, velocity, nothing else
    seed_cell_atlas  seed       lay the atlas out once, at x_0: piece centres, piece frames,
                                the material points inside each piece, and the per-particle
                                volume the piece's geometry implies
    aggregate_centroid aggregate  a parent's position as the mass-weighted mean of its
                                children's -- sum_pi, scheduled once per containment map

WHAT THE SEED DOES AND WHY IT IS NOT A `sets:` KEY. The engine's own placement scatters a
contained set uniformly in a ball about its parent, which is the right default and is exactly
wrong here: an atlas is not a fuzzball. A plasma-membrane piece belongs ON a sphere, tangent to
it, and its 1,000 points fill a curved patch; a mitochondrion is a capsule somewhere in the
cytoplasm but never inside the nucleus; a chromatin strand is a random walk confined to the
nucleus. That is a MECHANISM for establishing x_0 -- reusable, parameterised, shared between
specs -- which is what `seed:` is for (plexus2, "The seed window").

EVERY LENGTH IN THE ATLAS TABLE IS A FRACTION OF THE CELL RADIUS, never a world coordinate, so
one number (`cell_radius`, in world units) rescales the whole cell and the atlas stays a
statement about proportions. `size: 0.105` on the plasma membrane means "a patch whose radius
is 0.105 OF THE CELL'S RADIUS"; `r: 0.42` on the nuclear envelope means "a shell at 0.42 OF THE
CELL'S RADIUS".
"""
from __future__ import annotations

import math
import os

import torch

from plexus.models.base import Aggregate, Broadcast, Exchange, Lateral, Seed, Structural
from plexus.models.registry import register_entity, register_operator
from plexus.models.state import spatial_schema
from plexus.schema import split_ref


def _bnd(raw, lo, hi):
    """A positive value inside (lo, hi), smooth in `raw`: exp(log lo + (log hi - log lo) sigmoid(raw))."""
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.sigmoid(raw))


def _unbnd(x, lo, hi):
    """The raw value whose `_bnd` is x."""
    p = (math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo))
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _raw(values, n, lo, hi, device):
    """The unbounded learnable behind a positive value in (lo, hi): `values` a number or a list of n numbers -> [n] raw tensor, `_bnd(raw, lo, hi)` gives the values back."""
    v = list(values) if isinstance(values, (list, tuple)) else [float(values)] * n
    return torch.tensor([_unbnd(float(x), lo, hi) for x in v], dtype=torch.float32, device=device)


# --------------------------------------------------------------------------- the entity
@register_entity(
    "compartment", depth=1,
    state_schema=spatial_schema,                 # dim -> StateSchema (pos|vel, D-wide each)
    render={"color_by": "node_type", "arrows": None},
)
class Compartment:
    """One PIECE of a subcellular compartment: a single membrane patch, a single mitochondrion,
    a single Golgi cisterna.

    It is a set in its own right rather than a label on the material points, because the atlas
    counts pieces and because the two levels carry different state: a piece has a position, an
    orientation and an identity (`node_type` says which compartment it belongs to), while a
    material point has a deformation gradient and a mass. Collapsing them would make "how many
    mitochondria" unanswerable without clustering the point cloud.

    `depth=1` is the same scale hint `cell` carries, and that is not a contradiction: depth is a
    hint used only in `Level.__repr__`, while the containment chain that the engine and the
    operators actually traverse is `parent`/`parent_name`. Nothing dispatches on depth.

    Reference: the piece counts are those of the cell atlas in `figures/cell_atlas.png`; the
    level itself is plexus2 sec. Hierarchy, a parent holding several distinct sets at once.
    """


# --------------------------------------------------------------------------- the atlas table
#
# GEOMETRY ONLY. The COUNTS and the MATERIALS live on `sets.compartment.types` in the spec --
# the counts because `_assign_types` is what turns them into `node_type`, the materials because
# `MPMParticle.provision` reads `youngs`/`material` off the parent's types at build time, before
# any seed runs. Stating either of them here as well would be a second chance to disagree with
# the spec, which is the failure mode `bulk_modulus` beside `youngs` is refused for.
#
# Fields, all lengths as a fraction of `cell_radius`:
#   place  shell   piece centres on a sphere of radius `r`, laid out on a Fibonacci spiral
#          ball    piece centres uniform in the shell between radii `r_in` and `r_out`
#   axis   radial  the piece's `e3` points away from the cell centre
#          random  `e3` is an isotropic random direction
#   shape  patch     a curved cap of a sphere: a disc of radius `size` in the tangent plane,
#                    `thickness` deep radially. `e3` is the patch NORMAL.
#          sheet     a flat disc of radius `size`, `thickness` deep. `e3` is the sheet NORMAL.
#          rod       a cylinder of half-length `size` and radius `thickness`. `e3` is the AXIS.
#          ball      a solid ball of radius `size`.
#          filament  a persistent random walk of contour length `size` and radius `thickness`;
#                    `persistence` in [0, 1] is 0 for a freely-jointed coil and 1 for a straight
#                    rod. `e3` is the direction of the FIRST segment.
#          cisterna  a BOWED, FENESTRATED disc: radius `size`, `thickness` deep, sagging by
#                    `bow` x `size` along `e3`, with `holes` circular fenestrations of radius
#                    `hole_r` x `size` punched through it. Rough ER.
#          tubule    a BRANCHING TUBULAR NETWORK: a tree of `segments` tubes of radius
#                    `thickness`, each `size`/`segments` long, branching with probability
#                    `branch`. Smooth ER.
#          stack     `layers` bowed cisternae stacked along `e3`, separated by `gap` x `size`
#                    and tapering to `taper` of the first by the last. Golgi.
#          mito      a BENT capsule of half-length `size` and radius `thickness`, curved by
#                    `bend` (0 straight, 1 a half-circle), with `cristae` transverse inner
#                    lamellae taking `crista_frac` of the points. Mitochondrion.
#          barrel    `blades` rods on a circle of radius `size`, each `length` x `size` long and
#                    `thickness` thick, tilted by `skew` -- the 9 triplet microtubules of a
#                    centriole.
#
# WHY THE ORGANELLE SHAPES ARE NOT ALL DISCS AND CAPSULES. Drawn that way the picture says so: a
# Golgi drawn as one flat disc is indistinguishable from an ER cisterna drawn as one flat disc,
# and a mitochondrion drawn as a straight cylinder is indistinguishable from a piece of
# cytoskeleton. The shapes below are what makes the atlas an atlas rather than nine colours --
# a Golgi is a STACK, an ER is a bowed fenestrated sheet or a branching tube network, a
# mitochondrion is bent and has cristae, a centriole is a nine-fold barrel. Each is still a
# closed-form point sampler with a closed-form volume, so `p_vol` stays exact.
#
# The ten compartments below are the nine the screenshot's list shows, plus the centriole. The
# remaining five of the atlas's fifteen are not invented here -- add them to the spec's `types:`
# and to this table (or to the operator's `geometry:` override) when the list is known.
ATLAS: dict[str, dict] = {
    # the bounding shell: 421 patches on the cell surface. `size` is set just above the
    # equal-area spacing of a 421-point Fibonacci spiral (2/sqrt(421) = 0.0975 of the radius),
    # so neighbouring patches overlap by about 8% of their width and the shell is continuous
    # rather than perforated -- a perforated shell is not a membrane, it is a colander.
    "plasma_membrane": dict(place="shell", r=1.00, axis="radial",
                            shape="patch", size=0.105, thickness=0.05),
    # filaments spanning the cytoplasm, laid RADIALLY and nearly straight (persistence 0.88), each
    # centred between the nuclear envelope and the membrane. A contour length of 0.60 R centred in
    # the band 0.62-0.78 R reaches roughly 0.35 R to 1.02 R -- so the outer ends ANCHOR IN THE
    # MEMBRANE rather than stopping short of it, which is what a cytoskeleton does and what a
    # filament floating free in the cytoplasm does not. The inner ends stop outside the nuclear
    # envelope (0.42 R + its thickness), so the skeleton does not thread the nucleus.
    "cytoskeleton":    dict(place="ball", r_in=0.62, r_out=0.78, axis="radial",
                            shape="filament", size=0.60, thickness=0.006, persistence=0.88),
    # the nuclear ENVELOPE, not the nuclear volume: 138 patches on a sphere at 0.42 of the cell
    # radius. Same Fibonacci-spacing argument as the plasma membrane (2/sqrt(138) = 0.170 of
    # THAT sphere's radius = 0.0715 of the cell radius).
    "nucleus":         dict(place="shell", r=0.42, axis="radial",
                            shape="patch", size=0.078, thickness=0.040),
    # chromatin: coiled strands inside the envelope (persistence 0.25 -- a loose coil).
    "chromatin":       dict(place="ball", r_in=0.05, r_out=0.24, axis="random",
                            shape="filament", size=0.34, thickness=0.018, persistence=0.25),
    "nucleolus":       dict(place="ball", r_in=0.06, r_out=0.20, axis="random",
                            shape="ball", size=0.085),
    # ROUGH ER: broad bowed cisternae wrapped around the nucleus, perforated. The `axis: radial`
    # puts each sheet's NORMAL along the radius, so the sheet itself lies tangentially -- which is
    # how the reference figure shows the teal ER, as curved shells rather than as discs at
    # arbitrary angles. `holes` is what makes it read as ER and not as a plate.
    "rough_er":        dict(place="ball", r_in=0.52, r_out=0.70, axis="radial",
                            shape="cisterna", size=0.26, thickness=0.012,
                            bow=0.35, holes=9, hole_r=0.16),
    # SMOOTH ER: the tubular half of the same organelle, so a different SHAPE and not a smaller
    # sheet. A branching network of tubes is what distinguishes smooth from rough under EM.
    "smooth_er":       dict(place="ball", r_in=0.54, r_out=0.78, axis="random",
                            shape="tubule", size=0.90, thickness=0.010,
                            segments=16, branch=0.45, persistence=0.55),
    # GOLGI: a stack of cisternae, cis to trans, each smaller than the last. One disc was a plate;
    # five bowed discs 0.11 x size apart is the organelle.
    "golgi":           dict(place="ball", r_in=0.52, r_out=0.68, axis="radial",
                            shape="stack", size=0.075, thickness=0.008,
                            layers=6, gap=0.30, bow=0.40, taper=0.62),
    # MITOCHONDRIA: bent capsules with transverse cristae, anywhere in the cytoplasm and with no
    # preferred direction. `crista_frac` of the points go on the inner lamellae, which is what
    # makes a cut-open mitochondrion look like one.
    "mitochondria":    dict(place="ball", r_in=0.52, r_out=0.80, axis="random",
                            shape="mito", size=0.12, thickness=0.025,
                            bend=0.55, cristae=7, crista_frac=0.40),
    # CENTRIOLE: two of them, near the nucleus, as nine-fold barrels of triplet microtubules.
    # Placed close together because a centrosome is a PAIR at right angles; the right angle is not
    # imposed here (each barrel takes its own random axis), which is the one piece of centrosome
    # anatomy this shape does not yet carry.
    "centriole":       dict(place="ball", r_in=0.50, r_out=0.58, axis="random",
                            shape="barrel", size=0.011, thickness=0.0026,
                            blades=9, length=4.5, skew=0.30),
    # THE CROWD. Five protein species filling the whole cell -- one PIECE each, a ball of the
    # cell's own radius, drawn as dots. They are not an organelle and they are the reason a cell is
    # not hollow: a real cytoplasm is 20-30% protein by volume, and a shell with vacuum inside
    # pancakes on impact instead of bouncing, which is exactly what the first run did.
    #
    # ONE PIECE EACH, NOT A CLOUD OF PIECES, because a protein species is not counted in pieces --
    # it is a concentration. `volume_frac` is what keeps five co-located balls from claiming five
    # cell volumes of mass between them.
    **{f"protein_{k}": dict(place="ball", r_in=0.0, r_out=0.0, axis="random",
                            shape="ball", size=0.95, volume_frac=0.12)
       for k in ("a", "b", "c", "d", "e")},
}

_SHAPES = ("patch", "sheet", "rod", "ball", "filament",
           "cisterna", "tubule", "stack", "mito", "barrel")
_PLACES = ("shell", "ball")
_AXES = ("radial", "random")


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _ortho(e3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Any orthonormal (e1, e2) completing `e3` to a right-handed frame, per row.

    The seed vector is swapped away from `e3` when the two are nearly parallel, because a cross
    product of nearly parallel vectors is dominated by round-off and the frame it yields spins
    with the last bit of the input -- which would make a patch's orientation, and therefore the
    picture, non-reproducible for exactly the pieces sitting on the x axis.
    """
    a = torch.zeros_like(e3); a[:, 0] = 1.0
    b = torch.zeros_like(e3); b[:, 1] = 1.0
    a = torch.where((e3[:, 0].abs() > 0.9)[:, None], b, a)
    e1 = _unit(torch.cross(a, e3, dim=1))
    e2 = torch.cross(e3, e1, dim=1)
    return e1, e2


def _fibonacci(n: int, gen: torch.Generator) -> torch.Tensor:
    """`n` near-uniformly spaced unit vectors (the spherical Fibonacci spiral).

    NOT `randn`-normalised directions, and the difference is the whole appearance of the
    membrane: independent random directions clump, so at 421 points a random shell has bald
    patches several patch-widths across and a piece count that means nothing locally. The
    spiral's nearest-neighbour spacing is uniform to a few percent, which is what lets `size`
    be set from the count.
    """
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = 1.0 - 2.0 * i / n
    r = (1.0 - z * z).clamp_min(0.0).sqrt()
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    d = torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=1).float()
    # A RANDOM ROTATION PER RUN, so the spiral's own pole (a visible seam where the spacing is
    # least uniform) is not always at +z and the seam cannot be mistaken for a result.
    q = torch.randn(3, 3, generator=gen)
    q, _ = torch.linalg.qr(q)
    return _unit(d @ q)


def _ball_radii(n: int, r_in: float, r_out: float, gen: torch.Generator) -> torch.Tensor:
    """`n` radii uniform IN VOLUME between `r_in` and `r_out` (the 3-ball's r^2 dr weight)."""
    u = torch.rand(n, generator=gen)
    return (r_in ** 3 + u * (r_out ** 3 - r_in ** 3)) ** (1.0 / 3.0)


def _in_disc(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """`n` points uniform in the UNIT disc, as (a, b). sqrt(u) because the area element is r dr."""
    rho = torch.rand(n, generator=gen).sqrt()
    th = torch.rand(n, generator=gen) * (2.0 * math.pi)
    return rho * torch.cos(th), rho * torch.sin(th)


def _in_ball(n: int, gen: torch.Generator) -> torch.Tensor:
    """`n` points uniform in the UNIT ball [n, 3]."""
    d = _unit(torch.randn(n, 3, generator=gen))
    return d * torch.rand(n, 1, generator=gen).pow(1.0 / 3.0)


# --------------------------------------------------------------------------- the seed
# --------------------------------------------------------------------------- the aggregate
# -- the piece samplers of the former `seed_cell_atlas` (the atlas operator was dropped in the curation of 2026-10-27; `provision` and `cell_form` still draw MPM bodies
# of these shapes, so the geometry stays as module-level functions) --------------------------------------------------------------------------------------------------
def _piece_volume(row: dict, R: float) -> float:
    """The closed-form volume of ONE piece, in world units cubed (lengths are fractions of R).

    `volume_frac` scales the result, and it exists for the one case the closed forms cannot
    express: several bodies OCCUPYING THE SAME REGION, each holding a share of it. Five protein
    species crowding the cytoplasm are five balls of the cell's own radius, so taking each at
    its full geometric volume would give the cell five times its own mass. `volume_frac: 0.12`
    says this species fills 12% of the ball it is drawn in; the five together are a 60%
    crowded cytosol, which is roughly what a real one is. It changes only `p_vol` and hence
    the mass -- the points are still drawn throughout the region, because that is where the
    molecules are.
    """
    s = float(row.get("size", 0.05)) * R
    t = float(row.get("thickness", 0.02)) * R
    shape = row["shape"]
    vf = float(row.get("volume_frac", 1.0))
    if vf != 1.0:
        return vf * _piece_volume({**row, "volume_frac": 1.0}, R)
    if shape in ("patch", "sheet"):
        return math.pi * s * s * t                     # a disc of radius s, t deep
    if shape == "rod":
        return math.pi * t * t * (2.0 * s)             # a cylinder of half-length s, radius t
    if shape == "ball":
        return (4.0 / 3.0) * math.pi * s ** 3
    if shape == "filament":
        return math.pi * t * t * s                     # contour length s, radius t
    if shape == "cisterna":
        # a disc minus its fenestrations. `holes` discs of radius `hole_r` x size, and the
        # fraction they remove is capped below 1 because a sheet that is all hole has no volume
        # and would put p_vol at zero -- which is a division by zero in the density, not a thin
        # membrane.
        _h = min(0.8, int(row.get("holes", 0)) * float(row.get("hole_r", 0.0)) ** 2)
        return math.pi * s * s * t * (1.0 - _h)
    if shape == "tubule":
        return math.pi * t * t * s                     # total contour length s, radius t
    if shape == "stack":
        # `layers` discs tapering geometrically from radius s to `taper` x s
        n = int(row.get("layers", 5)); tap = float(row.get("taper", 1.0))
        fr = [1.0 - (1.0 - tap) * (i / max(n - 1, 1)) for i in range(n)]
        return math.pi * t * s * s * sum(f * f for f in fr)
    if shape == "mito":
        # the outer capsule; the cristae are INSIDE it, so they add no volume -- they are a
        # rearrangement of where the points sit, not extra material.
        return math.pi * t * t * (2.0 * s) + (4.0 / 3.0) * math.pi * t ** 3
    if shape == "barrel":
        # `blades` rods of radius t and length `length` x s, on a circle of radius s
        return (int(row.get("blades", 9)) * math.pi * t * t
                * float(row.get("length", 2.0)) * s)
    raise ValueError(f"unhandled shape {shape!r}")


def _fill(row: dict, n_pieces: int, per: int, R: float,
          centroid: torch.Tensor, e1, e2, e3, gen) -> torch.Tensor:
    """World positions [n_pieces * per, 3] of the material points of these pieces.

    Row-major: piece p owns rows [p*per, (p+1)*per), which is the layout `build` gives a
    contained set (`arange(n_parent).repeat_interleave(per)`), so the result lines up with
    `mpm_particle.parent` without a scatter.
    """
    N = n_pieces * per
    s = float(row.get("size", 0.05)) * R
    t = float(row.get("thickness", 0.02)) * R
    shape = row["shape"]
    rep = lambda v: v.repeat_interleave(per, dim=0)                       # noqa: E731

    if shape == "ball":
        return rep(centroid) + _in_ball(N, gen) * s

    if shape == "rod":
        a, b = _in_disc(N, gen)
        z = (torch.rand(N, generator=gen) * 2.0 - 1.0) * s
        return rep(centroid) + rep(e1) * (a * t)[:, None] + rep(e2) * (b * t)[:, None] \
            + rep(e3) * z[:, None]

    if shape == "sheet":
        a, b = _in_disc(N, gen)
        z = (torch.rand(N, generator=gen) - 0.5) * t
        return rep(centroid) + rep(e1) * (a * s)[:, None] + rep(e2) * (b * s)[:, None] \
            + rep(e3) * z[:, None]

    if shape == "patch":
        # A CURVED CAP, NOT A FLAT DISC. The piece is a patch OF A SPHERE, so a point is
        # placed in the tangent plane and then pushed back out to the shell radius: the 421
        # patches then meet edge to edge on the sphere instead of each cutting a chord
        # through it, which at 0.105 of the radius is a gap of 0.0055 R between neighbours --
        # a hole a grid cell wide, i.e. a leaking membrane.
        a, b = _in_disc(N, gen)
        tang = rep(e1) * (a * s)[:, None] + rep(e2) * (b * s)[:, None]
        depth = (torch.rand(N, generator=gen) - 0.5) * t
        rad = centroid.norm(dim=1, keepdim=True)                # the shell radius of each piece
        dirn = _unit(rep(centroid) + tang)
        return dirn * (rep(rad) + depth[:, None])

    if shape == "filament":
        # A PERSISTENT RANDOM WALK per piece, then points sprinkled along it. `persistence`
        # p mixes each new segment direction as p * previous + (1 - p) * random: p = 1 is a
        # straight rod (a microtubule), p = 0 a freely jointed coil (chromatin).
        n_seg = int(row.get("segments", 12))
        p = float(row.get("persistence", 0.5))
        step = s / n_seg
        d = e3.clone()                                     # the first segment: the piece axis
        nodes = [torch.zeros(n_pieces, 3)]
        for _ in range(n_seg):
            nd = _unit(p * d + (1.0 - p) * _unit(torch.randn(n_pieces, 3, generator=gen)))
            nodes.append(nodes[-1] + nd * step)
            d = nd
        nodes = torch.stack(nodes, dim=1)                  # [n_pieces, n_seg+1, 3]
        # CENTRED ON THE PIECE, NOT STARTED AT IT. The walk above begins at the origin of the
        # piece's frame and only ever moves away from it, so an uncentred filament is a piece
        # whose declared centre is one of its ENDS: a 0.70 R cytoskeletal filament placed at
        # 0.90 R reached 1.60 R and the cell had 116 spines sticking out of it (measured: the
        # particle bounding box ran to 0.873 in a cell whose surface is at 0.80). Subtracting
        # the polyline's own centroid makes `place` mean the same thing for a filament as it
        # does for a rod or a ball.
        nodes = nodes - nodes.mean(dim=1, keepdim=True)
        u = torch.rand(N, generator=gen) * n_seg
        i = u.floor().long().clamp(max=n_seg - 1)
        f = (u - i)[:, None]
        pid = torch.arange(n_pieces).repeat_interleave(per)
        a0 = nodes[pid, i]; a1 = nodes[pid, i + 1]
        return rep(centroid) + a0 + f * (a1 - a0) + _in_ball(N, gen) * t

    if shape == "cisterna":
        # A BOWED, FENESTRATED SHEET. The bow is a paraboloid in the sheet's own plane --
        # z += bow * s * (1 - rho^2) -- so the cisterna is a shallow dish rather than a plate,
        # which is what a stack of them wrapped round a nucleus has to be to nest.
        #
        # THE HOLES ARE REJECTION-SAMPLED, and rejection rather than a mask because a masked
        # point has to go somewhere: setting it to the centre piles material at r = 0, and
        # dropping it leaves the piece with fewer points than `p_vol` was divided by. Re-drawing
        # keeps the count exact and the density flat.
        bow = float(row.get("bow", 0.0))
        nh, hr = int(row.get("holes", 0)), float(row.get("hole_r", 0.0))
        hc = torch.zeros(n_pieces, max(nh, 1), 2)
        if nh:
            # hole centres inside the disc, kept off the rim so a fenestration is a hole and
            # not a bite out of the edge
            _hr = torch.rand(n_pieces, nh, generator=gen).sqrt() * (1.0 - hr)
            _ht = torch.rand(n_pieces, nh, generator=gen) * (2.0 * math.pi)
            hc = torch.stack([_hr * torch.cos(_ht), _hr * torch.sin(_ht)], dim=2)
        pid = torch.arange(n_pieces).repeat_interleave(per)
        a = torch.zeros(N); b = torch.zeros(N)
        todo = torch.arange(N)
        for _ in range(32):
            _a, _b = _in_disc(int(todo.numel()), gen)
            ok = torch.ones(int(todo.numel()), dtype=torch.bool)
            if nh:
                d2 = ((_a[:, None] - hc[pid[todo], :, 0]) ** 2
                      + (_b[:, None] - hc[pid[todo], :, 1]) ** 2)
                ok = (d2 > hr * hr).all(dim=1)
            a[todo[ok]] = _a[ok]; b[todo[ok]] = _b[ok]
            todo = todo[~ok]
            if todo.numel() == 0:
                break
        rho2 = a * a + b * b
        z = (torch.rand(N, generator=gen) - 0.5) * t + bow * s * (1.0 - rho2)
        return rep(centroid) + rep(e1) * (a * s)[:, None] + rep(e2) * (b * s)[:, None] \
            + rep(e3) * z[:, None]

    if shape == "tubule":
        # A BRANCHING TUBE NETWORK. A tree is grown per piece: node 0 at the centre, and each
        # new segment departs from a RANDOMLY CHOSEN existing node with probability `branch`
        # and from the previous tip otherwise. Points then pick a segment uniformly and sit
        # inside a tube of radius `thickness` about it -- so the point density follows contour
        # length, which is what `p_vol = pi t^2 s / per` assumes.
        n_seg = int(row.get("segments", 12))
        p = float(row.get("persistence", 0.5))
        br = float(row.get("branch", 0.0))
        step = s / n_seg
        nodes = [torch.zeros(n_pieces, 3)]
        dirs = [e3.clone()]
        segs = []                                       # (from_node, to_node)
        tip = torch.zeros(n_pieces, dtype=torch.long)
        for k in range(n_seg):
            # which node this segment leaves from: the tip, or an earlier node (a branch)
            if br > 0 and k > 1 and float(torch.rand(1, generator=gen)) < br:
                frm = torch.randint(0, len(nodes), (n_pieces,), generator=gen)
            else:
                frm = tip
            d0 = torch.stack(dirs, dim=1)[torch.arange(n_pieces), frm.clamp(max=len(dirs) - 1)]
            nd = _unit(p * d0 + (1.0 - p) * _unit(torch.randn(n_pieces, 3, generator=gen)))
            base = torch.stack(nodes, dim=1)[torch.arange(n_pieces), frm]
            nodes.append(base + nd * step)
            dirs.append(nd)
            segs.append((frm, torch.full((n_pieces,), len(nodes) - 1, dtype=torch.long)))
            tip = torch.full((n_pieces,), len(nodes) - 1, dtype=torch.long)
        NODES = torch.stack(nodes, dim=1)               # [n_pieces, n_seg+1, 3]
        NODES = NODES - NODES.mean(dim=1, keepdim=True)  # centred on the piece, as `filament`
        A = torch.stack([f for f, _ in segs], dim=1)     # [n_pieces, n_seg]
        B = torch.stack([t_ for _, t_ in segs], dim=1)
        pid = torch.arange(n_pieces).repeat_interleave(per)
        si = torch.randint(0, n_seg, (N,), generator=gen)
        f = torch.rand(N, 1, generator=gen)
        a0 = NODES[pid, A[pid, si]]; a1 = NODES[pid, B[pid, si]]
        return rep(centroid) + a0 + f * (a1 - a0) + _in_ball(N, gen) * t

    if shape == "stack":
        # `layers` BOWED CISTERNAE along e3, tapering cis -> trans. Each point picks a layer
        # uniformly, so every cisterna gets the same COUNT while the outer ones have more
        # volume -- which makes the trans face slightly denser, and is the price of keeping
        # this one closed-form draw instead of a per-layer loop.
        nl = int(row.get("layers", 5))
        gap = float(row.get("gap", 0.12)) * s
        bow = float(row.get("bow", 0.0))
        tap = float(row.get("taper", 1.0))
        li = torch.randint(0, nl, (N,), generator=gen)
        frac = 1.0 - (1.0 - tap) * (li.float() / max(nl - 1, 1))
        a, b = _in_disc(N, gen)
        rho2 = a * a + b * b
        z = (li.float() - (nl - 1) / 2.0) * gap + (torch.rand(N, generator=gen) - 0.5) * t \
            + bow * s * (1.0 - rho2)
        return rep(centroid) + rep(e1) * (a * s * frac)[:, None] \
            + rep(e2) * (b * s * frac)[:, None] + rep(e3) * z[:, None]

    if shape == "mito":
        # A BENT CAPSULE WITH CRISTAE. The centre line is an arc of angle `bend` * pi in the
        # (e3, e1) plane, so `bend: 0` is a straight rod and `bend: 1` is a
        # half-circle. `crista_frac` of the points are placed on transverse LAMELLAE -- discs
        # normal to the centre line at `cristae` evenly spaced stations, filling 0.8 of the
        # local radius -- and the rest fill the outer membrane shell. That split is why a
        # sectioned mitochondrion here shows internal structure instead of a solid sausage.
        bend = float(row.get("bend", 0.0))
        ncr = int(row.get("cristae", 0))
        cfr = float(row.get("crista_frac", 0.0)) if ncr else 0.0
        u = torch.rand(N, generator=gen) * 2.0 - 1.0            # -1..1 along the arc
        is_cr = torch.rand(N, generator=gen) < cfr
        if ncr:
            # snap the crista points onto their station, so the lamellae are discrete sheets
            st = torch.randint(0, ncr, (N,), generator=gen).float()
            u = torch.where(is_cr, (st / max(ncr - 1, 1)) * 1.8 - 0.9, u)
        ang = u * bend * math.pi * 0.5
        # arc of radius rc such that the arc length is 2s; bend -> 0 recovers the straight rod
        rc = (s / max(bend * math.pi * 0.5, 1e-6)) if bend > 1e-6 else 0.0
        if bend > 1e-6:
            along = rc * torch.sin(ang)
            lateral = rc * (1.0 - torch.cos(ang))
        else:
            along = u * s
            lateral = torch.zeros_like(u)
        # cross-section: an annulus for the outer membrane, a filled disc for a crista
        ca, cb = _in_disc(N, gen)
        crho = torch.sqrt(ca * ca + cb * cb).clamp_min(1e-9)
        shell = 0.72 + 0.28 * torch.rand(N, generator=gen)      # outer membrane band
        scale = torch.where(is_cr, torch.rand(N, generator=gen).sqrt() * 0.80, shell)
        ca, cb = ca / crho * scale, cb / crho * scale
        # e1 carries the bend, so the cross-section rides e1 (offset by `lateral`) and e2
        return rep(centroid) + rep(e3) * along[:, None] \
            + rep(e1) * ((lateral + ca * t))[:, None] + rep(e2) * (cb * t)[:, None]

    if shape == "barrel":
        # NINE TRIPLET MICROTUBULES ON A CIRCLE -- the centriole's defining nine-fold symmetry,
        # which no capsule or ball can express. Each blade is a rod of radius `thickness`
        # parallel to e3, tilted tangentially by `skew` so the barrel has the pinwheel the EM
        # cross-sections show.
        nb = int(row.get("blades", 9))
        L = float(row.get("length", 2.0)) * s
        skew = float(row.get("skew", 0.0))
        bi = torch.randint(0, nb, (N,), generator=gen).float()
        th = bi / nb * 2.0 * math.pi
        z = (torch.rand(N, generator=gen) - 0.5) * L
        th = th + skew * (z / max(L, 1e-9))                     # the pinwheel tilt
        ca, cb = _in_disc(N, gen)
        x = torch.cos(th) * s + ca * t
        y = torch.sin(th) * s + cb * t
        return rep(centroid) + rep(e1) * x[:, None] + rep(e2) * y[:, None] + rep(e3) * z[:, None]

    raise ValueError(f"unhandled shape {shape!r}")


def form_points(shape: str, n: int, params: dict, volume: float, copies: int = 1,
                device="cpu", seed: int = 0):
    """`n` points of an ATLAS FORM, centred on the origin and scaled so ONE piece has `volume`.

    THE TEN SHAPES WERE LOCKED INSIDE ONE OPERATOR. `seed_cell_atlas` owns closed-form samplers for
    a patch, a sheet, a rod, a ball, a filament, a bowed fenestrated cisterna, a branching tubule
    network, a Golgi stack, a bent mitochondrion with cristae and a nine-fold barrel -- each with a
    closed-form volume, which is exactly what the material seeder's contract needs (`V = per_parent
    * p_vol` fixes the size, the shape only arranges it). A spec that wanted a filament of jelly,
    or a stack for anything but a Golgi, could not have one: the repertoire was reachable only
    through a compartment hierarchy laid out about a cell centre.

    This is the door in. `plexus.models.entities.provision` calls it for `shape: <one of these>`,
    so the same sampler serves the atlas and any MPM body, and there is one implementation of a
    bent capsule rather than two. What stays behind in the operator is the ATLAS itself -- which
    organelle is which shape, at what radius, in what numbers -- because that is cell biology and
    not geometry.

    `copies` pieces are generated INDEPENDENTLY (a filament is a random walk; two copies of one
    walk would be a crystal) and interleaved to match `_place`, which sends point i to copy i % k.
    """
    import torch as _t
    row = {k: v for k, v in dict(params or {}).items() if k not in ("count", "fraction")}
    row["shape"] = str(shape)
    V1 = float(_piece_volume(row, 1.0))
    if not (V1 > 0):
        raise ValueError(f"form {shape!r}: its closed-form volume is {V1}; check `size`/`thickness`")
    k = max(int(copies), 1)
    per = max(int(n) // k, 1)
    gen = _t.Generator(device="cpu").manual_seed(int(seed))
    # THE SAMPLERS RUN ON THE CPU because their generator does; the points are moved once, after.
    e3 = _unit(_t.randn(k, 3, generator=gen))
    e1, e2 = _ortho(e3)
    pts = _fill(row, k, per, 1.0, _t.zeros(k, 3), e1, e2, e3, gen)
    pts = pts.reshape(k, per, 3)
    pts = pts - pts.mean(dim=1, keepdim=True)                    # each piece on its own centre
    pts = pts.transpose(0, 1).reshape(-1, 3)                     # interleave: point i -> copy i % k
    out = pts * (float(volume) / V1) ** (1.0 / 3.0)
    if out.shape[0] < n:                                         # a count that does not divide evenly
        out = _t.cat([out, out[: n - out.shape[0]]], 0)
    return out[:n].to(device)


ATLAS_FORMS = _SHAPES


@register_operator("aggregate_centroid", family="hierarchy", set="compartment", kind="aggregate",
                   equation=r"""$$\mathbf x_p=\frac{\sum_{c\in\pi^{-1}(p)} m_c\,\mathbf x_c}{\sum_{c\in\pi^{-1}(p)} m_c}$$""", title="track cell centres")
class AggregateCentroid(Aggregate):
    r"""a parent's position as the mass-weighted mean of its children's: sum over the fibre, and nothing else.
    child -> parent along the containment map `child.parent`: reads the children's `pos` (and `mass`, when the child set carries one); writes the parent's `pos`. Crosses any number of levels.

        \mathbf x_p=\frac{\sum_{c\in\pi^{-1}(p)} m_c\,\mathbf x_c}{\sum_{c\in\pi^{-1}(p)} m_c}
        x_p            the parent's position
        \pi^{-1}(p)    the children of parent p
        m_c, x_c       a child's mass (1 for a set with no mass) and position

    Reference: none -- the aggregate family of plexus2 sec. "The operator algebra". Plexus (this work)."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"A parent's position is the mass-weighted mean of its children's, $\mathbf x_p=\dfrac{\sum_{c}m_c\,\mathbf "
        r"x_c}{\sum_{c}m_c}$ over the children $c$ of $p$ (a child set with no mass counts 1 each). Here each cell's centre is "
        r"recomputed every frame from its material points. It reads the children's positions, writes the parent's, crosses any "
        r"number of levels and does nothing else."
    )
    PARAM_UNITS = {}
    SYMBOLS = {'x_p': "the parent's position", '\\pi^{-1}(p)': 'the children of parent p', 'm_c, x_c': "a child's mass (1 for a set with no mass) and position"}
    KIND = "aggregate"
    EMIT = None                                   # writes the parent's state; returns no delta
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["child"]
    INPUTS = ["child"]; OUTPUTS = ["parent"]; READS = ["pos"]; WRITES = ["pos"]
    MECHANISM_TAGS = ["aggregate", "centroid", "cross_scale"]
    PARAM_ROLES = {"child": "descendant_set", "weight": "mass_weighted_or_uniform"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        # `child:` AND NOT `from:`. `from:` is a reserved spec key that names a source FIELD, and
        # the schema rejects it when it names a set -- which is the right refusal: an Exchange
        # reads a field, an Aggregate reads a contained SET, and spelling both `from` would make
        # the two indistinguishable in a spec.
        self.child = params.get("child")
        self.weighted = bool(params.get("weight", True))

    def forward(self, H, mask=None):
        parent = H.level(self.at)
        child = H.level(self.child)
        if parent.name not in H.ancestors(child.name):
            raise ValueError(
                f"aggregate_centroid: {self.child!r} is not contained in {parent.name!r} at any "
                f"depth (its containment chain is "
                f"{' -> '.join(H.ancestors(child.name)) or '(none)'}). Aggregate travels the "
                f"containment map; without one there is no partition to sum over.")
        pidx = H.lift_index(child.name, parent.name)   # composed pi, however many hops away
        x = child.get("pos")
        w = child.occ if getattr(child, "occ", None) is not None else torch.ones_like(x[:, 0])
        if self.weighted and hasattr(child, "mass"):
            w = w * child.mass
        num = torch.zeros(parent.n, x.shape[1], device=x.device, dtype=x.dtype)
        den = torch.zeros(parent.n, device=x.device, dtype=x.dtype)
        num.index_add_(0, pidx, x * w[:, None])
        den.index_add_(0, pidx, w)
        centroid = num / den.clamp_min(1e-12)[:, None]
        px0, px1 = parent.state_schema["pos"]
        if torch.is_grad_enabled():
            # THE DIFFERENTIABLE PATH: clone, so the tape keeps the old state alive.
            st = parent.state.clone()
            st[:, px0:px1] = torch.where(den[:, None] > 0, centroid, st[:, px0:px1])
            parent.state = st
        else:
            # THE FORWARD PATH WRITES IN PLACE, AND MUST. Reassigning `parent.state` allocates a
            # new tensor, and the engine's graph signature is the set of buffer ADDRESSES the
            # captured substep baked in: one reassignment anywhere in the tick invalidates the
            # capture and the run finishes eager. The clone path drops the graph on the FIRST
            # tick, reporting a state buffer reallocated after the substep was captured -- for a
            # write to `compartment`, a set the substep never touches.
            old = parent.state[:, px0:px1]
            parent.state[:, px0:px1] = torch.where(den[:, None] > 0, centroid, old)
        return {}


# ============================================================================================
# THE ION ECONOMY: the membrane potential as an outcome, not a constant
# ============================================================================================
#
# `stator_step` above takes the free energy per transit, eps = q e psi, as a number. Here psi is a
# state of the CELL: a membrane capacitor charged by respiration and discharged by the stators
# and a leak, the zero mode of Poisson-Nernst-Planck at a scale where the Debye length is 1 nm
# and protons cross the box in 20 us against 380 us between one stator's steps. Three operators
# and the two hierarchy families the paper names: `membrane_potential` is the cell's own
# (Lateral) law, `aggregate` sums the stators' transits up the containment map (Sigma_pi) into
# the cell's motor current, `broadcast` hands psi down it (pi^*) so every stator steps against
# the potential the cell actually has. The stator set is therefore a CHILD of the cell.


# --------------------------------------------------------------------------- signalling
#
# TWO GENERIC OPERATORS ON A NAMED STATE BLOCK. Neither mentions a mitochondrion, and that is the
# point: what they need is a set with a scalar block and a relation over it, which is a shape many
# biological objects have. They exist because the alternative -- `cell_chem_diffuse` -- is welded
# to a two-species `chem` block (`N_SPECIES = 2`) and to the cell adjacency of a vertex mesh, so
# there was no way to diffuse ONE scalar over an ordinary `edge_index`.
@register_operator("seed_state_from_file", family="seed", set="compartment", kind="seed", title="load the initial state from a file")
class SeedStateFromFile(Seed):
    """Fill named state blocks from a `.npz`, once, at x_0 -- `seed_state_random` with a file.

    set -> set: writes the blocks named in `blocks:`, reads nothing.

        state[:, block]  <-  z[array]        one array per block, `[n]` or `[n, width]`

    THIS IS HOW A FITTED MODEL IS RUN AGAIN. A fit writes one number per element per quantity --
    the cardiomyocyte sheet's 472 cells carry an active strain, a fibre angle, a stiffness and an
    excitation delay -- and re-running that model means putting those numbers back where the
    forward pass reads them. There was no way to do it: `edges_file:` loads a RELATION (it defines
    the set's structure, so it is a set property), `load_run` loads a whole recorded trajectory,
    and `seed_state_random` draws values rather than reading them. A fitted per-element parameter
    is none of those three -- it is an initial condition with a provenance.

    NOT SPECIFIC TO ANY DOMAIN, and deliberately so. The values it loads are per-element numbers
    of a set; that the set's children happen to be material points, or neurons, or nothing at all,
    is not this operator's business. The cardiac sheet reads them from `cell` and its MPM points
    reach them through `H.lift_index(child, ancestor)`, which is the language's Broadcast and
    needs no help from here.

        seed:
          - op: seed_state_from_file
            at: cell
            file: cardio/healthy_allbeats.npz
            blocks: {phi: phi, g: g, g2: g2, logE: logE}     # block: array in the file

    `blocks:` maps a BLOCK NAME to an ARRAY NAME, because the two are allowed to differ -- a
    checkpoint calls the fitted leaf `logE` and the spec may call the block `stiffness`. Omit it
    and every array whose name matches a declared block is loaded, which is the common case.

    A LENGTH MISMATCH IS REFUSED RATHER THAN BROADCAST. A checkpoint fitted on 472 cells loaded
    into a set of 1,888 (which is what a default `grow_reserve` silently produces for a contained
    set) would otherwise tile or truncate, and the run would proceed with most cells holding
    another cell's parameters. That exact confusion cost the cardiac prototype every held-out
    score it had, 0.83 -> 0.36, and the spec now pins `grow_reserve: 0` because of it.

    Reference: none -- an initial condition, not a mechanism. Plexus (this work).
    """
    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["file"]
    MECHANISM_TAGS = ["initial_condition", "fitted_parameters", "checkpoint"]
    PARAM_ROLES = {"file": "npz_of_per_element_arrays", "blocks": "block_name_to_array_name"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "compartment")
        self.file = str(params["file"])
        self.blocks = dict(params.get("blocks") or {})

    def _path(self) -> str:
        from plexus.paths import graphs_data_path
        if os.path.isabs(self.file) and os.path.isfile(self.file):
            return self.file
        p = graphs_data_path(self.file)
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"seed_state_from_file: {self.file!r} -> {p} does not exist. `file:` resolves "
                f"through graphs_data_path exactly as `edges_file:` does, or may be absolute.")
        return p

    def forward(self, H, mask=None):
        import numpy as np
        lvl = H.level(self.at)
        z = np.load(self._path(), allow_pickle=False)
        declared = {b.name for b in lvl.state_schema.blocks}
        want = self.blocks or {k: k for k in z.files if k in declared}
        if not want:
            raise ValueError(
                f"seed_state_from_file: none of {sorted(z.files)} names a state block of "
                f"{lvl.name!r} (it has {sorted(declared)}). Give `blocks: {{<block>: <array>}}` "
                f"if the checkpoint's names differ from the spec's.")
        loaded = []
        for block, array in want.items():
            if block not in lvl.state_schema:
                raise ValueError(
                    f"seed_state_from_file: {lvl.name!r} has no state block {block!r} "
                    f"(has: {sorted(declared)}). Declare it under `sets.{lvl.name}.state:`.")
            if array not in z.files:
                raise ValueError(
                    f"seed_state_from_file: {self.file} holds no array {array!r} "
                    f"(has: {sorted(z.files)}).")
            b0, b1 = lvl.state_schema[block]
            v = np.asarray(z[array], dtype=np.float32)
            v = v[:, None] if v.ndim == 1 else v
            # ONE ROW PER CHILD OF ONE PARENT, ON A CONTAINED SET: the file describes one copy (a
            # channel's charged residues) and every parent carries that copy, in block order --
            # the same tiling `cloud_seed` places the copies' positions by.
            _par = getattr(lvl, "parent", None)
            if (v.shape[0] != lvl.n and lvl.parent_name is not None and _par is not None
                    and _par.numel() == lvl.n and v.shape[0] > 0 and lvl.n % v.shape[0] == 0
                    and bool((torch.bincount(_par) == v.shape[0]).all())):
                v = np.tile(v, (lvl.n // v.shape[0], 1))
            if v.shape[0] != lvl.n:
                raise ValueError(
                    f"seed_state_from_file: {array!r} has {v.shape[0]} rows but {lvl.name!r} has "
                    f"{lvl.n} elements. A fit belongs to the set it was fitted on; loading it "
                    f"into a set of a different size would give most elements another element's "
                    f"parameters. If the set is larger than expected, check `grow_reserve:` -- a "
                    f"contained set is given dormant slots unless the spec pins it to 0.")
            if v.shape[1] != b1 - b0:
                raise ValueError(
                    f"seed_state_from_file: {array!r} is {v.shape[1]} wide but block {block!r} is "
                    f"{b1 - b0}. Declare `state: {{{block}: {{width: {v.shape[1]}}}}}`.")
            lvl.state[:, b0:b1] = torch.as_tensor(v, device=lvl.state.device,
                                                  dtype=lvl.state.dtype)
            loaded.append(f"{block}({v.shape[1]})" if v.shape[1] > 1 else block)
        from plexus import utils as _eng
        if not _eng._QUIET:
            print(f"[seed_state_from_file] {lvl.name}: {', '.join(sorted(loaded))} "
                  f"<- {os.path.basename(self._path())} over {lvl.n:,} elements", flush=True)
        return {}


@register_operator("seed_state", "seed_state_random", family="seed", set="compartment",
                   kind="seed",
                   equation=r"""$$\mathbf v_i=A\sin\!\big(2\pi k\,\mathbf x_i\cdot\hat{\mathbf a}+\varphi\big)\,\hat{\mathbf d}$$""", title="set the initial state by a rule")
class SeedState(Seed):
    """Fill a named state block from a RULE, once, at x_0. `pattern` is which rule.

    set -> set: writes `block`; reads `pos` only when the pattern is spatial.

        pattern: random   v_i ~ U(lo, hi),  independently per element and per component
        pattern: wave     v_i = amplitude * sin(2 pi k x_i . axis + phase) * direction

    lo and hi are in the units of whatever `block` holds, so this operator has no units of its
    own; `seed` fixes the draw, on the CPU, so the same specification gives the same x_0 on any
    device.

    `pattern: wave` EXISTS FOR ONE MEASUREMENT and it is the standard one: a shear wave
    u = U0 sin(k z) x_hat decays as exp(-nu k^2 t), so fitting the decay of its amplitude returns
    the fluid's EFFECTIVE kinematic viscosity -- the physical one plus whatever the discretisation
    adds. In MLS-MPM the second is not small: the particle-to-grid-to-particle transfer smooths
    the velocity field every substep, contributing roughly dx^2 / (2 dt) or U dx / 2. On a run
    with eta = 1e-3 in a unit box at grid 64 that is 0.244 against a physical 1e-6, so the GRID
    sets the dissipation and the spec's viscosity does nothing at all. A Reynolds number quoted
    from the spec's eta is then wrong by however far apart those two are, and the only way to
    know is to measure.

    TWO NAMES, ONE OPERATOR. It was `seed_state_random`, which is what it did; a sinusoid is the
    same mechanism -- write an initial condition into a block from a rule -- so it is a `pattern`
    and not a second operator. The old name stays as an alias because about a thousand specs use
    it, and it defaults to `pattern: random`, so they are untouched.

    An initial condition with no structure is still an initial condition, and it has to be an
    operator rather than a `sets:` key for the reason `seed:` exists at all: `build` seeds
    POSITIONS from declarative placement rules, and every other block starts at zero. A network
    of identical zeros has no dynamics to show, so the first thing any relaxation on a graph
    needs is something to relax from.

    Reference: none -- an initial condition, not a mechanism. Plexus (this work).
    """
    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["block"]
    MECHANISM_TAGS = ["initial_condition", "random_state", "shear_wave", "benchmark"]
    PARAM_ROLES = {"block": "state_block", "lo": "lower_bound", "hi": "upper_bound",
                   "pattern": "random_or_wave", "amplitude": "peak_value_of_the_wave",
                   "k": "wavenumber_in_cycles_across_the_box",
                   "axis": "the_axis_the_wave_VARIES_along",
                   "direction": "the_vector_the_wave_POINTS_along",
                   "phase": "radians_of_offset"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "compartment")
        self.block = str(params["block"])
        self.lo = float(params.get("lo", 0.0))
        self.hi = float(params.get("hi", 1.0))
        self.seed = int(params.get("seed", 0))
        self.pattern = str(params.get("pattern", "random")).lower()
        if self.pattern not in ("random", "wave"):
            raise ValueError(f"seed_state: `pattern` is random or wave, got {self.pattern!r}")
        self.amplitude = float(params.get("amplitude", 1.0))
        self.k = float(params.get("k", 1.0))
        self.axis = int(params.get("axis", 2))
        self.direction = [float(v) for v in (params.get("direction") or [1.0, 0.0, 0.0])]
        self.phase = float(params.get("phase", 0.0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        if self.block not in lvl.state_schema:
            raise ValueError(
                f"seed_state: {lvl.name!r} has no state block {self.block!r} "
                f"(has: {', '.join(b.name for b in lvl.state_schema.blocks)}). Declare it under "
                f"`sets.{lvl.name}.state:`.")
        b0, b1 = lvl.state_schema[self.block]
        if self.pattern == "wave":
            # THE WAVE VARIES ALONG ONE AXIS AND POINTS ALONG ANOTHER, and keeping those separate
            # is the difference between a SHEAR wave and a compression wave. u = U0 sin(k z) x_hat
            # varies in z and points in x: the fluid slides over itself, which is pure shear and
            # decays at exactly nu k^2. Point it along the same axis it varies in and it is a
            # sound wave instead, which decays through the bulk modulus and measures nothing.
            X = lvl.get("pos")
            d = torch.tensor(self.direction[:b1 - b0], device=X.device, dtype=lvl.state.dtype)
            d = d / d.norm().clamp_min(1e-12)
            ph = 2.0 * math.pi * self.k * X[:, self.axis] + self.phase
            v = (self.amplitude * torch.sin(ph))[:, None] * d[None, :]
        else:
            g = torch.Generator(device="cpu").manual_seed(self.seed)
            v = (torch.rand(lvl.n, b1 - b0, generator=g) * (self.hi - self.lo) + self.lo).to(lvl.state.device)
        # A SELECTOR IS HONOURED (`at: 'neuron[type=AMN_L]'`): the engine hands the mask of the selected elements and only
        # those rows are written. It was ignored, so per-type seeds silently gave every element the LAST seed's value
        # (exp24, 2026-10-12: 16 per-type tau seeds left all 285 neurons at 0.1 s). No spec used a seed selector before.
        # A PLAIN `at: set` passes the live mask, and a seed must still fill EVERY slot, dormant ones included (a cell
        # born later by division wakes with the seeded value): only a mask NARROWER than the live set selects rows.
        act = getattr(lvl, "active", None)
        narrow = mask is not None and not (act is not None and torch.equal(mask.bool().cpu(), act.bool().cpu()))
        if narrow:
            m = mask.to(lvl.state.device).bool()
            lvl.state[m, b0:b1] = v[m].to(lvl.state.dtype)
        else:
            lvl.state[:, b0:b1] = v.to(lvl.state.dtype)
        # SILENT UNDER `engine.quiet`, because a trainer builds the world once per step and this
        # line would otherwise print thousands of times per fit. Imported here and not at module
        # scope: `engine` imports the operator registry, so the dependency only closes at call
        # time, by which point the module is loaded.
        from plexus import utils as _eng
        if not _eng._QUIET:
            _what = (f"{self.amplitude:g} sin(2 pi {self.k:g} x[{self.axis}]) along "
                     f"{self.direction}" if self.pattern == "wave"
                     else f"~ U({self.lo:g}, {self.hi:g})")
            print(f"[seed_state] {lvl.name}.{self.block} {_what} over {lvl.n:,} elements",
                  flush=True)
        return {}


@register_operator("laplacian_diffusion", family="signalling", set="compartment", kind="lateral",
                   equation=r"""$$\frac{dv_i}{dt}=D\sum_{j\sim i}\frac{v_j-v_i}{\deg(i)}$$""", title="diffuse a value between neighbours")
class LaplacianDiffusion(Lateral):
    """Diffusion of a scalar along a set's own relation -- the graph Laplacian, nothing more.

    set -[edge_index]-> set: reads `block` and the relation, emits d(block)/dt.

        dv_i/dt = D sum_{j ~ i} (v_j - v_i) / deg(i)      normalise: true (the default)
        dv_i/dt = D sum_{j ~ i} (v_j - v_i)               normalise: false

    v is the value of `block` at element i, the sum runs over its neighbours in `edge_index`, and
    D is a rate in inverse time -- the graph carries no length, so there is no length in the
    coefficient either. Dividing by the degree gives the NORMALISED Laplacian, whose eigenvalues
    lie in [-2, 0], so an explicit step is stable at any degree; without it an element that
    acquires many neighbours can overshoot in one tick, which on a relation rebuilt every frame by
    proximity is a thing that actually happens.

    THE RELATION IS NOT THIS OPERATOR'S BUSINESS. It reads whatever `edge_index` the set carries,
    so what the neighbours ARE is decided by the rewire scheduled before it: `radius_graph` makes
    them "everything within r", and a different rewire would make them something else without this
    operator changing. That separation is why the mechanism and the relation are two operators.

    Returns its delta under an explicit `(set, block)` key rather than relying on `INTEGRAND`,
    because the block is a PARAMETER here -- one class serving any scalar -- and `INTEGRAND` is
    read off the class by the integration-order check.

    Reference: Fick, A. (1855). Ueber Diffusion. Ann. Phys. 170:59-86 (diffusion); Chung, F.
    (1997). Spectral Graph Theory (the normalised Laplacian).
    """
    EMIT = "velocity"
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["block", "D"]
    INPUTS = ["set"]; OUTPUTS = ["set"]; READS = ["block"]; WRITES = ["block"]
    MECHANISM_TAGS = ["diffusion", "graph_laplacian", "signalling"]
    PARAM_ROLES = {"block": "state_block", "D": "diffusion_rate",
                   "normalise": "degree_normalised"}
    REFERENCE = ("Fick, A. (1855). Ueber Diffusion. Ann. Phys. 170:59-86; "
                 "Chung, F. (1997). Spectral Graph Theory.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "compartment")
        self.block = str(params["block"])
        self.D = float(params["D"])
        self.norm = bool(params.get("normalise", True))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        ei = getattr(lvl, "edge_index", None)
        b0, b1 = lvl.state_schema[self.block]
        v = lvl.state[:, b0:b1]
        if ei is None or ei.numel() == 0:
            return {(lvl.name, self.block): torch.zeros_like(v)}
        i, j = ei[0], ei[1]                                  # row 0 receives, row 1 sends
        occ = lvl.occ if getattr(lvl, "occ", None) is not None else torch.ones_like(v[:, 0])
        live = (occ[i] > 0) & (occ[j] > 0)
        i, j = i[live], j[live]
        d = torch.zeros_like(v)
        d.index_add_(0, i, v[j] - v[i])
        if self.norm:
            deg = torch.zeros(lvl.n, device=v.device, dtype=v.dtype)
            deg.index_add_(0, i, torch.ones_like(i, dtype=v.dtype))
            d = d / deg.clamp_min(1.0)[:, None]
        return {(lvl.name, self.block): self.D * d}


# --------------------------------------------------------------------------- motility
#
# HOW AN ADHERENT CELL MOVES: by gripping the substrate and pulling, not by being pushed.
#
#     seed_polarity        seed     a unit direction per cell, in the substrate plane
#     substrate_traction   lateral  a tangential force on the material that TOUCHES the floor
#     protrusion           lateral  an active push at the leading edge, pull at the rear
#
# WHY NOT A BODY FORCE ON THE WHOLE CELL. The cheap way to move a cell is `gravity` with a
# sideways vector, and it is wrong in a way that matters: every material point feels it equally,
# so the cell translates as a blob and its speed has no relationship to its shape, its contact
# area, or how well it adheres. A cell that spreads twice as far moves at the same speed, which
# makes the adhesion mechanics decorative.
#
# Both operators below tie the force to WHERE the material is -- near the floor for traction,
# along the polarity axis for protrusion -- so speed emerges from the mechanics instead of being
# set. They compose: traction is the grip, protrusion is the step.

@register_operator("protrusion", family="motility", set="particle", kind="lateral",
                   equation=r"""$$s_i=\frac{(\mathbf x_i-\mathbf x_c)\cdot\mathbf p_c}{R},\qquad \mathbf a_i=f\,\tanh\!\left(\frac{s_i}{w}\right)\mathbf p_c$$""", title="push the front and pull the rear")
class Protrusion(Lateral):
    r"""an active push at the leading edge and a pull at the rear: the step, not the grip.
    particle -[containment]-> particle: reads `pos`, its cell's centroid and its cell's `polarity`; emits an acceleration the MPM substep consumes as a body force. Sums to zero over a symmetric cell.

        s_i=\frac{(\mathbf x_i-\mathbf x_c)\cdot\mathbf p_c}{R},\qquad \mathbf a_i=f\,\tanh\!\left(\frac{s_i}{w}\right)\mathbf p_c
        s_i            where along the cell point i sits, in [-1, 1]
        x_i, x_c       the point and its cell's centroid (kept current by aggregate_centroid)
        p_c            the cell's unit polarity
        R              `radius`, the cell radius, in world units
        f              `f`, the drive per unit mass
        w              `width`, the softness of the front/rear split, a fraction of R

    Reference: Mogilner, A. (2009). Mathematics of cell motility. J. Math. Biol. 58:105-134 (the protrusion-adhesion-contraction picture); the implementation is not his."""
    PARAM_UNITS = {'f': 'acceleration', 'radius': 'length', 'width': 'fraction'}
    SYMBOLS = {'s_i': 'where along the cell point i sits, in [-1, 1]', 'x_i, x_c': "the point and its cell's centroid (kept current by aggregate_centroid)", 'p_c': "the cell's unit polarity", 'R': '`radius`, the cell radius, in world units', 'f': '`f`, the drive per unit mass', 'w': '`width`, the softness of the front/rear split, a fraction of R'}
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["f", "cell_set", "radius"]
    INPUTS = ["particle", "cell"]; OUTPUTS = ["particle"]
    READS = ["pos", "polarity"]; WRITES = []
    MECHANISM_TAGS = ["motility", "protrusion", "active_stress"]
    PARAM_ROLES = {"f": "drive_per_unit_mass", "radius": "cell_radius_world",
                   "gate": "phase_gate",
                   "width": "front_rear_softness", "cell_set": "polarity_owner"}
    REFERENCE = "Mogilner, A. (2009). Mathematics of cell motility. J. Math. Biol. 58:105-134."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.f = float(params["f"])
        self.cell_set = str(params["cell_set"])
        self.block = str(params.get("block", "polarity"))
        self.radius = float(params["radius"])
        self.width = float(params.get("width", 0.45))
        self.gate = params.get("gate")

    def forward(self, H, mask=None):
        p = H.level(self.at)
        X = p.get("pos")
        cl = H.level(self.cell_set)
        b0, b1 = cl.state_schema[self.block]
        pol = cl.state[:, b0:b1]
        cen = cl.get("pos")
        idx = H.lift_index(p.name, self.cell_set)
        pol_i, cen_i = pol[idx], cen[idx][:, : X.shape[1]]
        s = ((X - cen_i) * pol_i).sum(1) / max(self.radius, 1e-9)
        a = pol_i * (self.f * torch.tanh(s / max(self.width, 1e-6)))[:, None]
        gt = _gate(self, H, idx)
        if gt is not None:
            a = a * gt[:, None]
        if mask is not None:
            a = a * mask.float()[:, None]
        return {p.name: a}


LEADER_TRACE: list = []


def leader_traction(pos, es, ef, nF, leader, f, zero_net=True, et=None, stall=None, follow=0.0, stall_reach=0):
    """The traction load on a vertex tissue's vertices: f along each leader cell's outward direction,
    on every vertex of that cell; with `zero_net` the total is taken back evenly from every used
    vertex, so the tissue as a whole is not pushed. With `stall` (needs `et`), a leader whose
    perimeter P_c has reached `stall` times the tissue's median perimeter pulls no more:
    f_c = f clamp((stall - P_c/P_med) / (stall - 1), 0, 1). With `follow` > 0, a non-leader cell sharing a
    vertex with a leader pulls too, at `follow` x f along its own outward direction (the stall applies to it
    as well). With `stall_reach` 1, a pulled cell's stall factor is the SMALLEST among it and every cell
    sharing a vertex with it, so a leader stops pulling when a neighbour it drags is over the line. float64
    [nv, 3]."""
    dev = pos.device
    P = pos.to(torch.float64)
    live = ef < nF
    e_s, e_f = es[live].long(), ef[live].long()
    cnt = torch.zeros(nF, device=dev, dtype=torch.float64).index_add_(
        0, e_f, torch.ones_like(e_f, dtype=torch.float64))
    cen = torch.zeros(nF, 3, device=dev, dtype=torch.float64).index_add_(0, e_f, P[e_s])
    ok = cnt > 0
    cen[ok] /= cnt[ok, None]
    origin = cen[ok].mean(0) if ok.any() else torch.zeros(3, device=dev, dtype=torch.float64)
    u = cen - origin
    u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
    lead = leader[:nF].to(dev) & ok
    on = lead.to(torch.float64)
    if follow > 0.0:
        vl = torch.zeros(P.shape[0], dtype=torch.bool, device=dev)
        vl[e_s[lead[e_f]]] = True                          # the vertices of every leader
        touch = torch.zeros(nF, dtype=torch.bool, device=dev)
        touch[e_f[vl[e_s]]] = True                         # cells with one of those vertices
        on = on + float(follow) * (touch & ~lead & ok).to(torch.float64)
    if stall is not None and et is not None and float(stall) > 1.0:
        e_t = et[live].long()
        per = torch.zeros(nF, device=dev, dtype=torch.float64).index_add_(
            0, e_f, (P[e_t] - P[e_s]).norm(dim=1))
        pm = per[ok].median().clamp_min(1e-12) if ok.any() else per.new_tensor(1.0)
        sf = ((float(stall) - per / pm) / (float(stall) - 1.0)).clamp(0.0, 1.0)
        if int(stall_reach) > 0:
            for _ in range(int(stall_reach)):
                vmin = torch.ones(P.shape[0], device=dev, dtype=torch.float64).scatter_reduce(
                    0, e_s, sf[e_f], reduce="amin", include_self=True)   # per vertex: its most stretched cell
                sf = torch.ones_like(sf).scatter_reduce(0, e_f, vmin[e_s], reduce="amin", include_self=True)
        on = on * sf
    F = torch.zeros_like(P)
    # A VERTEX ONCE PER CELL IT BELONGS TO: each half-edge's source is one (vertex, cell) incidence.
    F.index_add_(0, e_s, float(f) * on[e_f, None] * u[e_f])
    if zero_net:
        used = torch.zeros(P.shape[0], dtype=torch.bool, device=dev)
        used[e_s] = True
        if used.any():
            F[used] -= F[used].sum(0) / used.sum()
    return F


@register_operator("protrusion", model="leader_traction", family="motility", set="vertex",
                   kind="lateral",
                   equation=r"""$$\mathbf u_c=\frac{\mathbf c_c-\bar{\mathbf c}}{\lVert\mathbf c_c-\bar{\mathbf c}\rVert},\qquad \mathbf f_v=f\sum_{c\ni v,\ c\ \mathrm{leader}}\mathbf u_c\;-\;\frac{1}{N_v}\sum_{w}f\sum_{c\ni w}\mathbf u_c$$""", title="pull leader cells out into the matrix")
class ProtrusionLeaderTraction(Lateral):
    r"""`leader_traction` model of protrusion: leader cells of a vertex tissue pull themselves outward into the matrix, invasion as migration.
    vertex -> vertex: reads the mesh and the cell set's leader block (`mutant`, set by seed_mesh[lineage]); adds a load to the frame's basal load buffer that cell_mechanics relaxes against.

        \mathbf u_c=\frac{\mathbf c_c-\bar{\mathbf c}}{\lVert\mathbf c_c-\bar{\mathbf c}\rVert},\qquad \mathbf f_v=f\sum_{c\ni v,\ c\ \mathrm{leader}}\mathbf u_c\;-\;\frac{1}{N_v}\sum_{w}f\sum_{c\ni w}\mathbf u_c
        u_c            leader cell c's outward direction from the tissue centre
        c_c, \bar c    the cell's centroid and the tissue's
        f              `f`, the traction per vertex
        f_v            the load on vertex v, summed over the leader cells it belongs to
        N_v            the vertex count (the `zero_net` subtraction takes the total back evenly)

    Reference: Cheung, K. J. et al. (2013). Collective invasion in breast cancer requires a conserved basal epithelial program. Cell 155:1639-1651."""
    PARAM_UNITS = {'f': 'force', 'follow': 'fraction', 'stall': 'dimensionless', 'stall_reach': 'count'}
    SYMBOLS = {'u_c': "leader cell c's outward direction from the tissue centre", 'c_c, \\bar c': "the cell's centroid and the tissue's", 'f': '`f`, the traction per vertex', 'f_v': 'the load on vertex v, summed over the leader cells it belongs to', 'N_v': 'the vertex count (the `zero_net` subtraction takes the total back evenly)'}
    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["f"]
    MECHANISM_TAGS = ["motility", "protrusion", "leader_cells", "collective_invasion"]
    PARAM_ROLES = {"f": "traction_per_leader_vertex", "leader_block": "leader_flag_block",
                   "cell_set": "leader_owner", "zero_net": "reaction_back_to_the_tissue",
                   "stall": "perimeter_over_median_at_which_traction_stops",
                   "follow": "follower_traction_fraction_of_f",
                   "stall_reach": "vertex_rings_over_which_the_stall_is_shared"}
    REFERENCE = ("Cheung, K. J., Gabrielson, E., Werb, Z. & Ewald, A. J. (2013). Collective "
                 "invasion in breast cancer requires a conserved basal epithelial program. Cell "
                 "155:1639-1651.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.f = float(params["f"])
        self.cell_set = str(params.get("cell_set", "cell"))
        self.leader_block = str(params.get("leader_block", "mutant"))
        self.zero_net = bool(params.get("zero_net", True))
        self.follow = float(params.get("follow", 0.0))
        self.stall_reach = int(params.get("stall_reach", 0))
        _st = params.get("stall")
        self.stall = None if _st is None else float(_st)
        if self.stall is not None and self.stall <= 1.0:
            raise ValueError(f"protrusion[leader_traction]: stall {self.stall:g} must exceed 1 -- it is a "
                             f"multiple of the median perimeter")
        self._said = False

    def forward(self, H, mask=None):
        from plexus.operators.contact_ops import basal_load
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        clvl = H.level(self.cell_set)
        fr = getattr(H, "frame", None)
        if (m is None or not int(m.get("Nv", 0)) or fr is None or clvl is None
                or self.leader_block not in getattr(clvl, "state_schema", {})):
            return {}
        nv, nF = int(m["Nv"]), int(m["nF"])
        pos = lvl.get("pos")[:nv]
        b0, _ = clvl.state_schema[self.leader_block]
        leader = clvl.state[:nF, b0] > 0.5
        F = leader_traction(pos, m["E_srce"], m["E_face"], nF, leader, self.f, self.zero_net,
                            et=m["E_trgt"], stall=self.stall, follow=self.follow,
                            stall_reach=self.stall_reach)
        buf = basal_load(m, int(fr), nv, pos.device)
        buf += F
        LEADER_TRACE.append((int(fr), int(leader.sum()), float(F.norm(dim=1).max()) if nv else 0.0))
        if not self._said:
            self._said = True
            print(f"[protrusion/leader_traction] {int(leader.sum())} leader cells of {nF} pull "
                  f"outward, f={self.f:g} per vertex, zero_net={self.zero_net}, stall={self.stall}", flush=True)
        return {}


# --------------------------------------------------------------------------- the crawl cycle
#
# A CLOCK PER CELL, AND TWO OPERATORS READING IT AT DIFFERENT PHASES.
#
#     phase_clock   lateral   dphi/dt = omega, per cell -- the frequency source, as STATE
#     `gate:` on protrusion and substrate_traction reads that phase
#
# WHY NET MOTION NEEDS TWO PHASES AND NOT ONE MECHANISM. `protrusion` is odd about the cell's
# centroid, so it extends the front and retracts the rear and sums to zero: driven by a symmetric
# cycle with a CONSTANT grip, the cell reaches out and is pulled straight back, and returns exactly
# where it started. That is not a tuning failure, it is reciprocity -- the same argument as the
# scallop theorem, and no amount of amplitude fixes it. What breaks it is gripping at one part of
# the cycle and releasing at another: extend while anchored, retract while free. So the adhesion
# has to be modulated OUT OF PHASE with the protrusion, which is why the clock is a separate thing
# both of them read rather than a parameter inside either.
#
# WHY THE PHASE IS PER-CELL STATE AND NOT THE GLOBAL `pacemaker`. `field_ops.pacemaker` publishes
# one periodic scalar per tick under `H.signals`, which is the right object when a whole tissue
# beats together and the wrong one here: every cell would extend and grip in lockstep, which is an
# artefact a reader has to be told to ignore. As a `phase` block on the cell set, each cell carries
# its own angle, the spread of angles is an initial condition, `omega` can vary from cell to cell,
# and the phase is recorded -- so "are they synchronised?" becomes a question about the data
# instead of about the code.
@register_operator("phase_clock", family="motility", set="cell", kind="lateral",
                   equation=r"""$$\frac{d\phi_i}{dt}=\omega_i,\qquad \omega_i=\omega\,(1+j\,u_i),\ u_i\sim\mathcal U(-1,1)$$""", title="advance each cell's phase clock")
class PhaseClock(Lateral):
    r"""advance a per-element phase at a fixed rate: the frequency source, as state.
    cell -> cell: reads `block`, emits d(block)/dt. The phase is not wrapped.

        \frac{d\phi_i}{dt}=\omega_i,\qquad \omega_i=\omega\,(1+j\,u_i),\ u_i\sim\mathcal U(-1,1)
        \phi_i         the element's phase, in radians
        \omega         `omega`, the rate, in radians per unit time (period 2 pi / omega)
        j              `jitter`, the spread of the rates, a fraction of omega
        u_i            a per-element uniform draw from `seed`

    Reference: none -- a linear phase is a modelling choice, not a published law. Plexus (this work)."""
    PARAM_UNITS = {'omega': 'rate', 'jitter': 'fraction', 'seed': 'count'}
    SYMBOLS = {'\\phi_i': "the element's phase, in radians", '\\omega': '`omega`, the rate, in radians per unit time (period 2 pi / omega)', 'j': '`jitter`, the spread of the rates, a fraction of omega', 'u_i': 'a per-element uniform draw from `seed`'}
    EMIT = "velocity"
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["omega"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["phase"]; WRITES = ["phase"]
    MECHANISM_TAGS = ["clock", "oscillator", "motility"]
    PARAM_ROLES = {"omega": "angular_frequency", "jitter": "rate_dispersion",
                   "block": "phase_block", 'seed': 'random_seed'}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.block = str(params.get("block", "phase"))
        self.omega = float(params["omega"])
        self.jitter = float(params.get("jitter", 0.0))
        self.seed = int(params.get("seed", 0))
        self._w = None

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        b0, b1 = lvl.state_schema[self.block]
        if self._w is None:
            g = torch.Generator(device="cpu").manual_seed(self.seed)
            u = (torch.rand(lvl.n, b1 - b0, generator=g) * 2.0 - 1.0)
            self._w = (self.omega * (1.0 + self.jitter * u)).to(lvl.state.device)
        return {(lvl.name, self.block): self._w}


def _gate(op, H, idx):
    """The cycle's amplitude for each point, from its cell's phase -- 1 + cos, in [0, 1].

    `gate: {block: phase, offset: <radians>, floor: <0..1>, signed: <bool>}` on any operator that
    carries a per-cell direction. `offset` is what puts two operators at different points of the
    SAME cycle; `floor` keeps a residual so a gated force never switches fully off, which matters
    for the adhesion (a cell that releases completely does not slide back, it falls over).

    `signed: true` GIVES cos(phi + offset) ITSELF, mean ZERO over a cycle, and it exists because
    `floor: 0` does not do what it sounds like. 1/2 (1 + cos) never goes negative: its average over
    a cycle is 1/2, so an operator gated that way applies half its amplitude STEADILY no matter what
    the floor is. A cell driven by it translates because it is being pushed, not because it is
    crawling, and the movie cannot tell the two apart. With `signed: true` the force reverses every
    half cycle and integrates to nothing on its own, so any net displacement has to come from the
    cell's shape changing in step with it -- which is the claim being tested.
    """
    cfg = getattr(op, "gate", None)
    if not cfg:
        return None
    cl = H.level(op.cell_set)
    b0, b1 = cl.state_schema[str(cfg.get("block", "phase"))]
    ph = cl.state[:, b0:b1][:, 0]
    c = torch.cos(ph + float(cfg.get("offset", 0.0)))
    if bool(cfg.get("signed", False)):
        return c[idx]
    fl = float(cfg.get("floor", 0.0))
    return (fl + (1.0 - fl) * 0.5 * (1.0 + c))[idx]


@register_operator("polar_active_stress", family="motility", set="particle", kind="lateral",
                   equation=r"""$$\boldsymbol\sigma_{\mathrm{act}}=A\cos(\phi_c+\delta)\left(\mathbf n\mathbf n^{\mathsf T}-\tfrac{1}{3}\mathbf I\right)$$""", title="extend and retract along the polarity")
class PolarActiveStress(Lateral):
    r"""the cytoskeleton extending and retracting along the cell's polarity, as a deviatoric stress that reverses sign over the cycle.
    particle -[containment]-> particle: reads its cell's `polarity` and `phase`; writes a per-particle active stress that mpm_scatter adds to the elastic Kirchhoff stress.

        \boldsymbol\sigma_{\mathrm{act}}=A\cos(\phi_c+\delta)\left(\mathbf n\mathbf n^{\mathsf T}-\tfrac{1}{3}\mathbf I\right)
        \sigma_{act}   the active stress
        A              `amplitude`, in the run's stress units (or `amplitude_frac` of the material's stiffness)
        \phi_c         the cell's phase
        \delta         `offset`, the phase offset, in radians
        n              the cell's unit polarity
        I              the identity; the traceless form keeps the volume to first order

    Reference: Simha, R. A. & Ramaswamy, S. (2002). Phys. Rev. Lett. 89:058101; Marchetti, M. C. et al. (2013). Rev. Mod. Phys. 85:1143-1189 (active stress in polar gels)."""
    PARAM_UNITS = {'amplitude': 'stress', 'amplitude_frac': 'fraction', 'offset': 'angle'}
    SYMBOLS = {'\\sigma_{act}': 'the active stress', 'A': "`amplitude`, in the run's stress units (or `amplitude_frac` of the material's stiffness)", '\\phi_c': "the cell's phase", '\\delta': '`offset`, the phase offset, in radians', 'n': "the cell's unit polarity", 'I': 'the identity; the traceless form keeps the volume to first order'}
    EMIT = None                       # a stress, consumed by the substep; no integrable delta
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["cell_set"]
    INPUTS = ["particle", "cell"]; OUTPUTS = ["particle"]
    READS = ["polarity", "phase"]; WRITES = []
    MECHANISM_TAGS = ["active_stress", "motility", "protrusion", "cytoskeleton"]
    PARAM_ROLES = {"amplitude": "active_stress", "amplitude_frac": "target_strain",
                   "offset": "cycle_phase_offset", "cell_set": "polarity_owner",
                   "deviatoric": "volume_preserving", 'phase_block': 'cell_block_the_phase_is_read_from'}
    REFERENCE = ("Simha, R. A. & Ramaswamy, S. (2002). Phys. Rev. Lett. 89:058101; "
                 "Marchetti, M. C. et al. (2013). Rev. Mod. Phys. 85:1143-1189.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.cell_set = str(params["cell_set"])
        self.block = str(params.get("block", "polarity"))
        self.phase_block = str(params.get("phase_block", "phase"))
        self.amplitude = params.get("amplitude")
        self.frac = params.get("amplitude_frac")
        self.offset = float(params.get("offset", 0.0))
        self.deviatoric = bool(params.get("deviatoric", True))
        if (self.amplitude is None) == (self.frac is None):
            raise ValueError("polar_active_stress: give `amplitude` (stress units) OR "
                             "`amplitude_frac` (of the set's own lambda + 2 mu), not both and "
                             "not neither -- the second is the one that states a target strain.")

    def forward(self, H, mask=None):
        p = H.level(self.at)
        X = p.get("pos")
        D = X.shape[1]
        cl = H.level(self.cell_set)
        b0, b1 = cl.state_schema[self.block]
        q0, q1 = cl.state_schema[self.phase_block]
        idx = H.lift_index(p.name, self.cell_set)
        n = cl.state[:, b0:b1][:, :D][idx]
        n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)
        ph = cl.state[:, q0:q1][:, 0][idx]
        A = (torch.as_tensor(float(self.amplitude), device=X.device, dtype=X.dtype)
             if self.amplitude is not None
             else float(self.frac) * (p.la + 2.0 * p.mu))
        g = A * torch.cos(ph + self.offset)
        if mask is not None:
            g = g * mask.float()
        M = n[:, :, None] * n[:, None, :]
        if self.deviatoric:
            M = M - torch.eye(D, device=X.device, dtype=X.dtype)[None] / float(D)
        sig = g[:, None, None] * M
        # ALLOCATED ONCE AND WRITTEN IN PLACE. The substep is captured as a CUDA graph, which bakes
        # in the addresses it saw; a fresh tensor per tick would leave the replay reading the one
        # from the tick it was captured on.
        buf = getattr(p, "act_stress", None)
        if buf is None or buf.shape != sig.shape:
            p.register_buffer("act_stress", torch.zeros_like(sig))
            buf = p.act_stress
        buf.copy_(sig)
        return {}


@register_operator("deform_control", family="motility", set="particle", kind="lateral",
                   equation=r"""$$\mathbf F\leftarrow\mathbf G\,\mathbf F,\qquad \mathbf G=\exp(-\mathbf A\,\Delta t),\quad \mathbf A=\mathbf A^{\mathsf T}$$""", title="change rest shape at a set rate")
class DeformControl(Lateral):
    r"""a declared rate of rest-shape change: the control variable of a shape-morphing optimisation.
    particle -> particle: multiplies the deformation gradient by G = expm(-A dt) every frame, A a symmetric 3x3 rate given as six numbers (or read from `field:`).

        \mathbf F\leftarrow\mathbf G\,\mathbf F,\qquad \mathbf G=\exp(-\mathbf A\,\Delta t),\quad \mathbf A=\mathbf A^{\mathsf T}
        F              the elastic deformation gradient
        G              the growth increment, the inverse growth F_g enters F
        A              `rate`, the symmetric rate of rest-shape change, per unit time (a_xx, a_yy, a_zz, a_xy, a_xz, a_yz); positive a_xx lengthens the material in x
        \Delta t       the frame length

    Reference: Xu, M., Song, C. Y., Levin, D. I. W. & Hyde, D. (2025). A Differentiable Material Point Method Framework for Shape Morphing. IEEE TVCG 31(10):9140-9153."""
    PARAM_UNITS = {'rate': 'rate', 'extent': 'length'}
    SYMBOLS = {'F': 'the elastic deformation gradient', 'G': 'the growth increment, the inverse growth F_g enters F', 'A': '`rate`, the symmetric rate of rest-shape change, per unit time (a_xx, a_yy, a_zz, a_xy, a_xz, a_yz); positive a_xx lengthens the material in x', '\\Delta t': 'the frame length'}
    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = True
    # NOT `REQUIRES_PARAMS = ["rate"]`. The schema checks that list against the raw YAML BEFORE the
    # operator is constructed, so a `field:` line -- which needs no rate at all -- was refused
    # unless it carried a dead `rate: [0, 0, 0, 0, 0, 0]` that nothing ever read. The requirement is
    # really "rate OR field", which only __init__ can see, and that is where it is enforced.
    REQUIRES_PARAMS = []
    REQUIRES_BUFFERS = ["F"]
    MECHANISM_TAGS = ["shape_control", "growth", "rest_shape", "inverse_design"]
    PARAM_ROLES = {"rate": "symmetric_3x3_rate_as_six_numbers", "over": "frames_of_action", 'extent': 'half_extent_of_the_rate_field_cube'}
    REFERENCE = ("Xu, M. et al. (2025). IEEE TVCG 31(10):9140-9153 (arXiv:2409.15746); "
                 "Rodriguez, E. K. et al. (1994). J. Biomech. 27:455-467.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        # A LEARNED FIELD INSTEAD OF SIX NUMBERS. `field:` names an .npz written by
        # `plexus.morph`: a K^3 x 6 cube of symmetric rates, read at each particle's own
        # MATERIAL coordinate -- its position the first time this operator runs, never its current
        # one. That is what makes the control Lagrangian, and it is why a control optimised on a
        # 50,000-point proxy can be applied to a 2.4 M-point cell: the field means the same thing
        # at any resolution because it is indexed by where the material STARTED.
        self.field = params.get("field")
        self.extent = params.get("extent")            # [cx, cy, cz, half_width] of the control cube
        self._W = None
        if self.field is None and params.get("rate") is None:
            raise ValueError("deform_control: give `rate` (six numbers, a symmetric 3x3) or "
                             "`field` (a .npz holding an optimised K^3 x 6 cube) -- one of them.")
        r = [float(v) for v in (params.get("rate") or [0.0] * 6)]
        if self.field is None and len(r) != 6:
            raise ValueError("deform_control: `rate` is six numbers -- the diagonal a_xx, a_yy, "
                             f"a_zz then the off-diagonals a_xy, a_xz, a_yz -- got {len(r)}")
        self.r = r
        # HOW LONG IT ACTS. Unset, forever; a morph that reached its target and then kept going
        # would be a different answer from the one that was optimised.
        self.over = int(params.get("over", 0))
        self._G = None

    def forward(self, H, mask=None):
        p = H.level(self.at)
        if self.over and int(getattr(H, "frame", 0) or 0) >= self.over:
            return {}
        if self.field is not None:
            return self._field_forward(H, p)
        if self._G is None:
            a = self.r
            A = torch.tensor([[a[0], a[3], a[4]], [a[3], a[1], a[5]], [a[4], a[5], a[2]]],
                             device=p.F.device, dtype=p.F.dtype)
            self._G = torch.matrix_exp(-A * float(getattr(H, "dt", 0.002)))
            print(f"[deform_control] {self.at}: rate {a}, acting for "
                  f"{self.over or 'the whole run'} frames", flush=True)
        F = torch.matmul(self._G, p.F)
        if torch.is_grad_enabled():
            p.F = F
        else:
            p.F.copy_(F)
        return {}

    def _field_forward(self, H, p):
        """The learned rate cube, applied per particle."""
        import numpy as np
        if self._W is None:
            z = np.load(self.field)
            A6 = torch.as_tensor(np.asarray(z["control"], np.float32), device=p.F.device)
            K = int(round(A6.shape[0] ** (1.0 / 3.0)))
            cx, cy, cz, half = [float(v) for v in (self.extent or [0.5, 0.5, 0.5, 0.05])]
            X0 = p.get("pos").detach()
            u = ((X0 - torch.tensor([cx, cy, cz], device=X0.device)) / (2 * half) + 0.5) \
                .clamp(0, 1) * (K - 1)
            b = u.floor().long().clamp(0, K - 1)
            f = (u - b).clamp(0, 1)
            a = torch.zeros(p.n, 6, device=X0.device)
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        w = (((1 - f[:, 0]) if dx == 0 else f[:, 0])
                             * ((1 - f[:, 1]) if dy == 0 else f[:, 1])
                             * ((1 - f[:, 2]) if dz == 0 else f[:, 2]))
                        i = (((b[:, 0] + dx).clamp(0, K - 1) * K
                              + (b[:, 1] + dy).clamp(0, K - 1)) * K
                             + (b[:, 2] + dz).clamp(0, K - 1))
                        a = a + w[:, None] * A6[i]
            dt = float(getattr(H, "dt", 0.002))
            A = torch.zeros(p.n, 3, 3, device=X0.device) + torch.diag_embed(a[:, :3])
            A[:, 0, 1] = A[:, 1, 0] = a[:, 3]
            A[:, 0, 2] = A[:, 2, 0] = a[:, 4]
            A[:, 1, 2] = A[:, 2, 1] = a[:, 5]
            Adt = -A * dt
            eye = torch.eye(3, device=X0.device)[None]
            self._W = eye + Adt + 0.5 * torch.bmm(Adt, Adt)   # 2nd order, as the optimiser used
            # SAY WHAT WAS CLAMPED. A point outside the cube silently takes the nearest edge's
            # rate, which is how a mismatched `extent:` applies the wrong deformation to a whole
            # body without anything looking wrong. The fraction is the number that catches it.
            out = float(((u < 0) | (u > (K - 1))).any(dim=1).to(torch.float32).mean()) \
                if u.numel() else 0.0
            print(f"[deform_control] {self.at}: a {K}^3 rate field from "
                  f"{os.path.basename(str(self.field))} over {p.n:,} points; "
                  f"{100 * out:.2f}% of them fall outside the control cube "
                  f"(centre {cx:g},{cy:g},{cz:g} half-width {half:g}) and take its edge rate",
                  flush=True)
            if out > 0.05:
                print(f"[deform_control] {self.at}: MORE THAN 5% OF THE MATERIAL IS OUTSIDE THE "
                      f"CONTROL CUBE. `extent:` almost certainly does not match where this body "
                      f"actually is -- check its bounding box against the cube.", flush=True)
        F = torch.bmm(self._W, p.F)
        if torch.is_grad_enabled():
            p.F = F
        else:
            p.F.copy_(F)
        return {}


@register_operator("deform_control", implementation="block", family="motility", set="particle",
                   kind="lateral",
                   equation=r"""$$\mathbf F_p\leftarrow\big(\mathbf I-\mathbf A_p\Delta t+\tfrac12(\mathbf A_p\Delta t)^2\big)\,\mathbf F_p,\quad \mathbf A_p=\mathrm{sym}(\mathbf a_p)$$""", title="change rest shape at per-point rates")
class DeformControlBlock(DeformControl):
    r"""`block` model of deform_control: the same rest-shape rate, read per particle from a state block of the set every call.
    particle -> particle: reads the six-column block `rate`, multiplies F by the second-order expansion of expm(-A_p dt) once a frame; caches nothing, so a trainer may write the block.

        \mathbf F_p\leftarrow\big(\mathbf I-\mathbf A_p\Delta t+\tfrac12(\mathbf A_p\Delta t)^2\big)\,\mathbf F_p,\quad \mathbf A_p=\mathrm{sym}(\mathbf a_p)
        F_p            the particle's elastic deformation gradient
        A_p            the symmetric rate built from the particle's six block values, per unit time
        \Delta t       the frame length

    Reference: as `deform_control`; Xu, M. et al. (2025). IEEE TVCG 31(10):9140-9153."""
    PARAM_UNITS = {}
    SYMBOLS = {'F_p': "the particle's elastic deformation gradient", 'A_p': "the symmetric rate built from the particle's six block values, per unit time", '\\Delta t': 'the frame length'}
    PARAM_ROLES = {'rate': 'state_block_holding_the_six_rate_components'}
    REQUIRES_PARAMS = []

    def __init__(self, params, device="cpu"):
        params = {**params, "rate": params.get("rate", [0.0] * 6)}   # the parent needs one of rate/field
        super().__init__(params, device)
        self.block = str(params.get("block", "rate"))

    def forward(self, H, mask=None):
        p = H.level(self.at)
        if self.over and int(getattr(H, "frame", 0) or 0) >= self.over:
            return {}
        a = p.get(self.block)
        if a.shape[-1] != 6:
            raise ValueError(f"deform_control[block]: block {self.block!r} of {self.at!r} has width "
                             f"{a.shape[-1]}; a symmetric rate is six numbers")
        dt = float(getattr(H, "dt", 0.002))
        A = torch.diag_embed(a[..., :3])
        off = torch.zeros_like(A)
        off[..., 0, 1] = off[..., 1, 0] = a[..., 3]
        off[..., 0, 2] = off[..., 2, 0] = a[..., 4]
        off[..., 1, 2] = off[..., 2, 1] = a[..., 5]
        Adt = -(A + off) * dt
        G = torch.eye(3, device=a.device, dtype=a.dtype) + Adt + 0.5 * (Adt @ Adt)
        F = G @ p.F
        if torch.is_grad_enabled():
            p.F = F
        else:
            p.F.copy_(F)
        return {}


# ---------------------------------------------------------------------------------------------
# `polar_active_stress [model: driven]` MOVED HERE from `cilia_ops.py`, which was deleted.
#
# THAT FILE WAS THE MPM CILIUM, AND THE MPM CILIUM IS A DEAD END. Its ten other operators built
# a cilium out of MLS-MPM material points, which carries one velocity per grid node: a body
# thinner than a grid cell cannot move relative to the fluid it shares nodes with, it is simply
# advected. A Platynereis cilium is 0.25 um thick against a 4.08 um grid cell, so the work kept
# widening the cilium until the grid could hold it -- a 6 um paddle, then a 25 x 10 um blade --
# and what beat was never a cilium. `rod_ops.py` replaces all of it: a filament whose thickness
# is a DRAG COEFFICIENT rather than a volume has no resolution floor at all.
#
# THIS ONE OPERATOR WAS NEVER PART OF THAT. It is a driven variant of `PolarActiveStress` just
# above, used by five `config/cell` adhesion-flow specs and by nothing ciliary, so it follows its
# parent class rather than the file it happened to sit in.
_RECTIFY = {
    "relu": lambda x: torch.clamp(x, min=0.0),
    "softplus": lambda x: torch.nn.functional.softplus(x),
    "abs": torch.abs,
    "identity": lambda x: x,
}


@register_operator("polar_active_stress", model="driven", family="motility", set="particle",
                   kind="lateral", title="extend and retract, amplitude from state", equation=r"""$$\boldsymbol\sigma_{\mathrm{act}}=A\,g(v)\cos(\phi_c+\delta)\left(\mathbf n\mathbf n^{\mathsf T}-\tfrac13\mathbf I\right),\qquad g(v)=\min\!\Big(g_{\max},\ \mathrm{relu}\frac{v-v_0}{v_s}\Big)$$""")
class PolarActiveStressDriven(PolarActiveStress):
    r"""`driven` model of polar_active_stress: the amplitude gated by a membrane state read from a block, excitation-contraction coupling in its simplest rectified-linear form.
    particle -[containment]-> particle: reads its cell's `polarity`, `phase` and `drive` block; writes the per-particle active stress mpm_scatter consumes.

        \boldsymbol\sigma_{\mathrm{act}}=A\,g(v)\cos(\phi_c+\delta)\left(\mathbf n\mathbf n^{\mathsf T}-\tfrac13\mathbf I\right),\qquad g(v)=\min\!\Big(g_{\max},\ \mathrm{relu}\frac{v-v_0}{v_s}\Big)
        A, \phi_c, \delta, n as polar_active_stress
        v              the cell's membrane state, the `drive` block of `drive_set`
        v_0            `v0`, the state below which the cell is silent
        v_s            `v_scale`, the state at which the cell beats at the declared amplitude
        g_{max}        `gain_max`, the cap on the gain

    Reference: as `polar_active_stress`; excitation-contraction coupling, Bers, D. M. (2002). Nature 415:198-205."""
    PARAM_UNITS = {'v0': 'dimensionless', 'v_scale': 'dimensionless', 'gain_max': 'dimensionless'}
    SYMBOLS = {'A, \\phi_c, \\delta, n': 'as polar_active_stress', 'v': "the cell's membrane state, the `drive` block of `drive_set`", 'v_0': '`v0`, the state below which the cell is silent', 'v_s': '`v_scale`, the state at which the cell beats at the declared amplitude', 'g_{max}': '`gain_max`, the cap on the gain'}

    READS = ["polarity", "phase", "drive"]
    MECHANISM_TAGS = ["active_stress", "cilia", "ciliary_beat", "neuromuscular",
                      "excitation_contraction_coupling"]
    PARAM_ROLES = dict(PolarActiveStress.PARAM_ROLES,
                       drive="the_cells_membrane_state_block",
                       drive_set="the_set_that_carries_the_membrane_state",
                       v0="membrane_state_below_which_the_cell_is_silent",
                       v_scale="membrane_state_giving_a_full_stroke",
                       rectify="relu_softplus_abs_or_identity",
                       gain_max="ceiling_on_the_gain")
    REFERENCE = ("Simha, R. A. & Ramaswamy, S. (2002). Phys. Rev. Lett. 89:058101; "
                 "Veraszto, C. et al. (2017). eLife 6:e26000.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.drive = str(params.get("drive", "voltage"))
        # `drive_set` names the set the membrane state lives on, which usually cannot be the
        # polarity set: `voltage` and `pos` are both coordinates, and one set has only one, so a
        # cell that is both a neuron and a lump of matter is two sets joined by containment.
        self.drive_set = str(params.get("drive_set", self.cell_set))
        self.v0 = float(params.get("v0", 0.0))
        self.v_scale = float(params.get("v_scale", 1.0))
        if self.v_scale == 0.0:
            raise ValueError("polar_active_stress[driven]: `v_scale` is the membrane state that "
                             "gives a full stroke and cannot be 0")
        self.rectify = str(params.get("rectify", "relu")).lower()
        if self.rectify not in _RECTIFY:
            raise ValueError(f"polar_active_stress[driven]: `rectify` must be one of "
                             f"{sorted(_RECTIFY)}, got {self.rectify!r}")
        # A CEILING, because the drive is not bounded. A membrane state is a real number,
        # and a cell that runs away would ask for an unbounded stress -- which in MLS-MPM is not a
        # big deformation, it is a velocity that breaks the Courant condition and a run that ends
        # in NaN several hundred frames after the actual fault.
        self.gain_max = float(params.get("gain_max", 4.0))

    def _drive_per_cell(self, H, cl, dtype, device):
        """The membrane state of each CELL, gathered from whichever set carries it. [n_cells]."""
        dl = H.level(self.drive_set)
        if self.drive not in dl.state_schema:
            raise KeyError(
                f"polar_active_stress[driven]: the set {dl.name!r} has no block {self.drive!r}. "
                f"This operator's whole purpose is to read the circuit's output, so the set it "
                f"points at must declare that block. Blocks present: "
                f"{sorted(dl.state_schema)}.")
        d0, d1 = dl.state_schema[self.drive]
        v = dl.state[:, d0:d1][:, 0].to(device=device, dtype=dtype)
        if dl.name == cl.name:
            return v
        # A CHILD SET CARRIES IT: invert the containment map. `lift_index` gives each driver its
        # cell; scattering the drivers back by that index gives each cell its driver. With one
        # driver per cell the scatter is a permutation and nothing is averaged away; with more
        # than one the LAST wins, which is why the spec is required to declare `per_parent: 1`.
        try:
            up = H.lift_index(dl.name, cl.name)
        except Exception:                                            # noqa: BLE001
            up = None
        if up is not None:
            out = torch.zeros(cl.n, device=device, dtype=dtype)
            out[up] = v
            return out
        # NO CONTAINMENT: FALL BACK TO ROW ORDER, AND CHECK IT RATHER THAN TRUST IT. A flat
        # neuron set beside the cells is the only way to name cell classes (type counts on a
        # child set are per parent), and then row i of one is row i of the other by construction
        # -- the kind of correspondence that breaks silently, so it is asserted.
        if dl.n != cl.n:
            raise ValueError(
                f"polar_active_stress[driven]: `drive_set` {dl.name!r} holds {dl.n:,} elements "
                f"and `cell_set` {cl.name!r} holds {cl.n:,}, and they are not joined by "
                f"containment -- so there is no correspondence between them. Either parent one "
                f"to the other with `per_parent: 1`, or declare both with the same count in the "
                f"same row order.")
        return v

    def gain(self, H, cl, idx, dtype, device):
        """g(v) per material point: its cell's membrane state, shifted, scaled, rectified, capped."""
        v = self._drive_per_cell(H, cl, dtype, device)
        g = _RECTIFY[self.rectify]((v - self.v0) / self.v_scale)
        return g.clamp(max=self.gain_max)[idx]

    def forward(self, H, mask=None):
        p = H.level(self.at)
        X = p.get("pos")
        D = X.shape[1]
        cl = H.level(self.cell_set)
        b0, b1 = cl.state_schema[self.block]
        q0, q1 = cl.state_schema[self.phase_block]
        idx = H.lift_index(p.name, self.cell_set)
        n = cl.state[:, b0:b1][:, :D][idx]
        # A CELL WITH NO POLARITY HAS NO STROKE AXIS, SO IT DOES NOT STROKE.
        #
        # This is how a spec says WHICH cells beat, and it needs no mask on this operator: seed
        # `polarity` on the ciliary band alone -- `at: 'cell[type=ciliary band]'` -- and every
        # other cell keeps the zero the entity provisioned it with. Normalising that zero, which
        # the parent operator does unconditionally, turns it into whatever
        # `clamp_min(1e-12)` leaves behind and gives 4,000 interior cells an arbitrary stroke
        # direction at full amplitude. Here a zero-length polarity zeroes the gain instead.
        nrm = n.norm(dim=1, keepdim=True)
        has_axis = (nrm[:, 0] > 1e-9).to(X.dtype)
        n = n / nrm.clamp_min(1e-12)
        ph = cl.state[:, q0:q1][:, 0][idx]
        A = (torch.as_tensor(float(self.amplitude), device=X.device, dtype=X.dtype)
             if self.amplitude is not None
             else float(self.frac) * (p.la + 2.0 * p.mu))
        g = A * has_axis * self.gain(H, cl, idx, X.dtype, X.device) * torch.cos(ph + self.offset)
        if mask is not None:
            g = g * mask.float()
        M = n[:, :, None] * n[:, None, :]
        if self.deviatoric:
            M = M - torch.eye(D, device=X.device, dtype=X.dtype)[None] / float(D)
        sig = g[:, None, None] * M
        # ALLOCATED ONCE AND WRITTEN IN PLACE, for the parent operator's reason: the substep is
        # captured as a CUDA graph, which bakes in the addresses it saw, so a fresh tensor per
        # tick would leave the replay reading the one from the tick it was captured on.
        buf = getattr(p, "act_stress", None)
        if buf is None or buf.shape != sig.shape:
            p.register_buffer("act_stress", torch.zeros_like(sig))
            buf = p.act_stress
        buf.copy_(sig)
        return {}


# =============================================================================================
# THE BACTERIAL FLAGELLAR MOTOR: stator units pushing a rotor (builder/exp_02_bacterium; moved here from
# flagellum_ops.py on 2026-09-27 -- INSTRUCTION.md: no module per experiment, an operator lands in its family's module).
#
# The motor is not an operator that says "rotate": the torque comes from stator units acting on a rotor, and the
# rotation rate has to COME OUT of that torque against whatever load the rotor carries.
#     stator_push               exchange  each unit pushes the rotor material in its reach, tangentially -- a constant
#                                         force per unit (the stated baseline)
#     stator_push[contact]      exchange  the elastic linkage between a unit and the ring: torque = kappa x
#     stator_step               lateral   the unit's ion-driven stepping, feeding the stretch x against its load
#     stator_membrane           exchange  the membrane as a place: each unit's current deposited where it stands
# The cell's side lives with its families: `aggregate` / `broadcast` (above, beside aggregate_centroid) carry the
# stators' transits up to the cell and its potential down; `membrane_potential` is in metabolism.py.
# =============================================================================================
# ============================================================================================
# THE STATOR AS A MACHINE: an ion-driven stepper on an elastic linkage, instead of a force
# ============================================================================================
#
# `stator_push` above puts a constant force on the rotor. These two operators replace it with
# what a stator unit is: a MotA5B2 complex whose MotA pentamer turns around MotB one step per ion
# transit (Santiveri et al. 2020; Deme et al. 2020), each step advancing its grip on the C-ring by
# one FliG spacing, through a linkage that is elastic (the MotB plug and peptidoglycan anchor, and
# here the PflA/PflB scaffold). Torque is then the linkage's stretch times its stiffness, and the
# rate at which the stretch is fed is set by ion kinetics against the work the stretch costs, so
# the torque-speed relation -- Berg's plateau, knee and fall -- is an OUTCOME of the two numbers a
# stator is made of (the free energy per ion and the step size) and not a curve written in.
#
# The split follows the language: `stator_contact` is the EXCHANGE between the stator set and the
# rotor's material points (the mechanical contact: force out, ring speed in), `stator_step` is the
# LATERAL dynamics of the stator set itself (the chemistry: how fast the stretch is fed). They
# meet on the stator's own state block `x`, the linkage stretch in radians, which `stator_step`
# integrates and `stator_contact` turns into force. Two buffers carry the contact's readings to
# the stepper (`stator_torque`, `stator_omega`), the way rod_ops passes its anchor.

