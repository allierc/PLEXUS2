"""exp09 molecular_landscape: movies of a crowded molecular landscape in the flat-ink look -- flat fills, every body outlined in its own colour darkened, far bodies
paler -- with every molecule a body of MPM material points (Cedric, 2026-11-14: "generate goodsell like movies with Plexus ... 1000 frames using MPM operators";
the look is inspired by D. Goodsell's cell cross-sections, https://ccsb.scripps.edu/goodsell/, and is not a copy of them: Cedric's own vivid palette, a grey ground).

    python tools/exp09_landscape.py list
    python tools/exp09_landscape.py write it01                # -> config/tissue/exp09_land_it01.yaml (+ its seed table in the record's data/)
    python tools/exp09_landscape.py measure it01              # the run's numbers, printed and saved in the record's data/
    python tools/exp09_landscape.py after it01                # what the after-job runs: measure, why text, the movie into the record
    python tools/exp09_landscape.py render it01 [--last-only] # the movie (or only the last stored row as a PNG) of a run or of a partial trajectory
    python tools/exp09_landscape.py look it01                 # a landed step read before the next job: six frames side by side (data/look_it01.png) and its numbers

The scene: a slab of cytoplasm 192 x 108 nm wide and 52 nm deep seen face on, about 400 bodies at a quarter of its volume. Sentence by sentence:

    the molecules        MPM bodies, elastic (they keep their shape), 27 to 343 material points at 2 nm per point: seed_positions[from_points_ragged] (NEW, adhesion_ops.py)
    they jostle          thermal_bath[langevin]: a drag -friction v and random kicks of thermal energy kT on every point, every MPM substep
    they cannot overlap  pair_potential[cell_contact]: the overlap penalty between bodies; its contact tension (type x type, negative = sticky) is 0 unless an iteration sets it
    they stay in view    active_force[box_slab] (NEW, flatten_ops.py): soft walls on the six faces of the slab
    the mechanics        the MPM substep block: mpm_strain, mpm_scatter, mpm_grid_update[warp_pp_balanced], mpm_gather, each body on its own grid block (per_parent)

From it10 (Cedric, 2026-11-15: "modify the geometries of the objects and their interaction", twelve reference pictures in the record's download/) a scene also has
many-lobed kinds (tetramers, dimers, pear-shaped heads, curved rods, rings), chains (`chains`: rows of block-shaped bodies seeded one point spacing apart, straight
filaments or meandering threads, held by the contact tension between their own kind alone) and binding (a contact tension between two chosen kinds). A body is at
most 343 points and 23 nm long (its own grid box is 30 nm): anything longer is a chain. `shapes` draws every kind's points locally (data/shapes.png).

Units: length 10 nm, time 0.2 us (a model choice: the free diffusion of the 12 nm body, kT / (friction x mass) = 0.002 length^2 per time, read as 1 um^2/s).
The movie is drawn by `render` below with the pieces of the record's renderer: every body the iso-surface tools/exp23_studio._cell_surface draws of an MPM cell, flat
material, the ink of gui/flat_ink.py; here full frame, no inset. Watcher: experiments/exp09_molecular_landscape at http://127.0.0.1:8843/watch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools"), os.path.join(ROOT, "discovery_okuda")]

LOG = os.path.join(ROOT, "log", "experiments", "exp09_land")
RECORD = os.path.join(ROOT, "experiments", "exp09_molecular_landscape")
DATA = os.path.join(RECORD, "data")
GD = os.path.join(ROOT, "graphs_data", "tissue")

NM = 10.0                      # nanometres per length unit
UM = NM / 1000.0               # micrometres per length unit (0.01): the unit of the seed table and of the spec's `units.length_um`
SPACING = 0.2                  # point spacing, length units (2 nm): the MPM grid's own cell size (cell_box 3.0 / n_grid 15)
PV = SPACING ** 3              # volume of one material point, length units^3 (8 nm^3)
TIME_US = 0.2                  # microseconds per time unit (see the docstring)

# ---- the bodies: points at start, shape, colour (Cedric's vivid set, gui/flat_ink.PALETTE), what it stands for at 10 nm per length unit ------------------------------
# shape: ("ellipsoid", (a, b, c)) the N lattice points nearest the centre in the metric of semi-axes a : b : c; ("lobes", [((x, y, z), r), ...]) the N lattice points
# nearest any of the lobe centres, each distance divided by its lobe's radius r (centres and radii in units of the first lobe's radius); ("block", (nx, ny, nz)) a
# full lattice block; ("torus", (R, r)) the N lattice points nearest a circle in the body's xy plane, a ring of centre-line radius R and tube radius r (their ratio);
# ("stack", (rho, ...)) layers of points along y from the top down, each the lattice points within rho point spacings of the body's axis (1.5: 9 points, 2: 13, 2.3: 21, 3: 29)
SPECIES = {
    "ribosome":   dict(n=343, shape=("lobes", [((0.0, 0.0, 0.0), 1.0), ((0.0, 0.0, 1.05), 0.78)]), colour="#9B5DE5", what="two unequal lobes, 22 x 15 nm (a small ribosome)"),
    "chaperonin": dict(n=125, shape=("ellipsoid", (1.0, 1.0, 1.15)), colour="#4D96FF", what="a barrel, 12 x 14 nm (a chaperonin-size complex)"),
    "rod":        dict(n=125, shape=("ellipsoid", (1.0, 1.0, 2.6)), colour="#F15BB5", what="a rod, 9 x 23 nm (an elongated complex)"),
    "trimer":     dict(n=125, shape=("lobes", [((0.75, 0.0, 0.0), 1.0), ((-0.375, 0.65, 0.0), 1.0), ((-0.375, -0.65, 0.0), 1.0)]), colour="#E9C46A", what="three lobes in a triangle, 15 nm across"),
    "enzyme":     dict(n=64, shape=("ellipsoid", (1.0, 1.0, 1.0)), colour="#2A9D8F", what="a globule, 10 nm (an enzyme of ~400 kDa)"),
    "small":      dict(n=27, shape=("ellipsoid", (1.0, 1.0, 1.0)), colour="#8AC926", what="a globule, 7 nm (an enzyme of ~170 kDa, the smallest body 2 nm points can carry)"),
    # the membrane scene (it02): a sheet of patches with proteins through it, and what floats on its far side
    "lipid":      dict(n=75, shape=("block", (5, 3, 5)), colour="#F4A261", what="a patch of membrane, 10 x 6 x 10 nm (a block of 5 x 3 x 5 points)"),
    "channel":    dict(n=125, shape=("ellipsoid", (0.8, 1.9, 0.8)), colour="#E63946", what="a protein through the membrane, 9 x 22 nm"),
    "antibody":   dict(n=125, shape=("lobes", [((0.0, -0.9, 0.0), 0.8), ((0.0, 0.0, 0.0), 0.8), ((0.8, 0.9, 0.0), 0.8), ((-0.8, 0.9, 0.0), 0.8)]), colour="#00BBF9", what="a Y of four lobes, 19 nm"),
    "serum":      dict(n=64, shape=("ellipsoid", (1.0, 1.0, 1.3)), colour="#43AA8B", what="a globule, 9 x 12 nm"),
    # chains (see chain_groups): blocks seeded in a row, one lattice spacing apart, and held to each other by a contact tension alone
    "actin":      dict(n=64, shape=("block", (4, 4, 4)), colour="#E63946", what="a filament subunit, 8 nm (a block of 4 x 4 x 4 points): in a straight row, a beaded filament 8 nm thick"),
    "bead":       dict(n=27, shape=("block", (3, 3, 3)), colour="#F15BB5", what="a thread element, 6 nm (a block of 3 x 3 x 3 points): in a meandering row, a thread 6 nm thick"),
    "myosin":     dict(n=125, shape=("block", (5, 5, 5)), colour="#9B5DE5", what="a thick-filament subunit, 10 nm (a block of 5 x 5 x 5 points): in a straight row, a filament 10 nm thick"),
    # the many-lobed kinds (Cedric, 2026-11-15: "modify the geometries of the objects and their interaction", with twelve reference pictures in the record's
    # download/): what those pictures are made of and it01-it09 lack -- tetramers, dimers, pear-shaped heads, curved rods, rings
    "clover":     dict(n=125, shape=("lobes", [((0.62, 0.62, 0.62), 1.0), ((0.62, -0.62, -0.62), 1.0), ((-0.62, 0.62, -0.62), 1.0), ((-0.62, -0.62, 0.62), 1.0)]),
                       colour="#4D96FF", what="four lobes at the corners of a tetrahedron (a tetramer)"),
    "dimer":      dict(n=64, shape=("lobes", [((0.0, 0.0, 0.66), 1.0), ((0.0, 0.0, -0.66), 1.0)]), colour="#00BBF9", what="two equal lobes (a dimer)"),
    "pear":       dict(n=64, shape=("lobes", [((0.0, 0.0, 0.0), 1.0), ((0.0, 0.0, 1.25), 0.6)]), colour="#F4A261", what="a large lobe and a small one (a motor head)"),
    "bean":       dict(n=125, shape=("lobes", [((-1.45, 0.55, 0.0), 0.8), ((-0.52, 0.0, 0.0), 0.9), ((0.52, 0.0, 0.0), 0.9), ((1.45, 0.55, 0.0), 0.8)]),
                       colour="#2A9D8F", what="a curved rod of four lobes"),
    "ring":       dict(n=133, shape=("torus", (2.0, 1.0)), colour="#43AA8B", what="a ring with a hole through it (a ring-shaped complex)"),
    # a membrane protein with a head: ten layers of points along y, from the top a head 14 nm wide (4 layers), a neck 6 nm wide (2), the part in the sheet, 10 nm wide
    # as a patch's site is (3, the sheet's own three layers), and a stub on the far side (1). `top`: where it sits in a sheet -- the sheet's middle plane this far
    # below the body's topmost point, length units (see membrane_bodies)
    "receptor":   dict(n=190, shape=("stack", (2.3, 3.0, 3.0, 2.3, 1.5, 1.5, 2.3, 2.3, 2.3, 1.5)), top=1.4, colour="#E9C46A", what="a head on a neck through the membrane, 14 x 20 nm (a receptor)"),
    "ligand":     dict(n=190, shape=("stack", (2.3, 3.0, 3.0, 2.3, 1.5, 1.5, 2.3, 2.3, 2.3, 1.5)), top=1.4, colour="#F15BB5", what="the receptor's partner in a facing membrane: the same head on a neck, 14 x 20 nm"),
    "replicator": dict(n=64, shape=("ellipsoid", (1.0, 1.0, 1.0)), colour="#E63946", what="a globule that grows and divides, 10 nm at birth (red, so that a lineage can be followed)"),
}
# mRNA AND PROTEINS (Cedric, 2026-11-19: "simulating a bunch of mRNA and proteins interacting ... mRNA as linear 1D nodes, proteins as a few nodes cluster, difference
# between length mRNA and protein x10 ... 10 mRNA types of different length and 10 proteins types ... a 20x20 affinity matrix"): an mRNA is a chain of 6 nm subunits
# (one kind per mRNA type, `chains`), a protein a body of 64 points about 10 nm across in one of ten shapes
RNA_COLOURS = ("#E63946", "#F3722C", "#F8961E", "#F9C74F", "#D62828", "#FF6B6B", "#C9184A", "#FFB703", "#E76F51", "#F4A3A8")
PROT_COLOURS = ("#4D96FF", "#00BBF9", "#43AA8B", "#2A9D8F", "#90BE6D", "#577590", "#9B5DE5", "#4361EE", "#06D6A0", "#118AB2")
PROT_SHAPES = (("ellipsoid", (1.0, 1.0, 1.0)), ("lobes", [((0.0, 0.0, 0.66), 1.0), ((0.0, 0.0, -0.66), 1.0)]),
               ("lobes", [((0.75, 0.0, 0.0), 1.0), ((-0.375, 0.65, 0.0), 1.0), ((-0.375, -0.65, 0.0), 1.0)]),
               ("lobes", [((0.62, 0.62, 0.62), 1.0), ((0.62, -0.62, -0.62), 1.0), ((-0.62, 0.62, -0.62), 1.0), ((-0.62, -0.62, 0.62), 1.0)]),
               ("lobes", [((0.0, 0.0, 0.0), 1.0), ((0.0, 0.0, 1.25), 0.6)]),
               ("lobes", [((0.0, -0.9, 0.0), 0.8), ((0.0, 0.0, 0.0), 0.8), ((0.8, 0.9, 0.0), 0.8), ((-0.8, 0.9, 0.0), 0.8)]),
               ("lobes", [((-1.45, 0.55, 0.0), 0.8), ((-0.52, 0.0, 0.0), 0.9), ((0.52, 0.0, 0.0), 0.9), ((1.45, 0.55, 0.0), 0.8)]),
               ("lobes", [((0.0, 0.0, 0.0), 1.0), ((0.0, 0.0, 1.05), 0.78)]), ("ellipsoid", (1.0, 0.6, 0.6)), ("ellipsoid", (1.0, 1.0, 0.45)))
PROT_WHAT = ("a globule", "two equal lobes", "three lobes in a triangle", "four lobes at a tetrahedron's corners", "a large lobe and a small one", "a Y of four lobes",
             "a curved rod of four lobes", "two unequal lobes", "a rod", "a disc")
for _k in range(10):
    SPECIES[f"m{_k}"] = dict(n=27, shape=("block", (3, 3, 3)), colour=RNA_COLOURS[_k], what=f"a subunit of mRNA type {_k}, 6 nm (a block of 3 x 3 x 3 points)")
    SPECIES[f"p{_k}"] = dict(n=64, shape=PROT_SHAPES[_k], colour=PROT_COLOURS[_k], what=f"protein type {_k}: {PROT_WHAT[_k]}, about 10 nm")


def rna_affinity(seed=19):
    """The 20 x 20 affinity matrix of the mRNA and protein kinds (m0..m9, p0..p9, in that order), symmetric: each protein type binds 3 mRNA types drawn at random, each at
    an affinity drawn between 0.4 and 1; 3 pairs of protein types bind each other at 0.5; no mRNA binds an mRNA. An affinity is bond_binding's type-matrix entry: the
    bound share a junction of that pair rises toward (0 = the pair never binds). Returns (names, matrix)."""
    rng = np.random.default_rng(seed)
    names = [f"m{k}" for k in range(10)] + [f"p{k}" for k in range(10)]
    A = np.zeros((20, 20))
    for j in range(10):
        for i in rng.choice(10, 3, replace=False):
            A[i, 10 + j] = A[10 + j, i] = round(float(rng.uniform(0.4, 1.0)), 2)
    pp = set()
    while len(pp) < 3:
        a, b = sorted(rng.choice(10, 2, replace=False))
        pp.add((int(a), int(b)))
    for a, b in pp:
        A[10 + a, 10 + b] = A[10 + b, 10 + a] = 0.5
    return names, A


def rna_delta_g(seed=19, strong=-8.0, weak=-2.0):
    """rna_affinity's pairs as binding free energies of a pair in contact, kT: an affinity of 1 is `strong` (-8 kT: bound for about 3,000 frames at one binding per
    frame), 0.4 is `weak` (-2 kT: a bound share of 0.88, let go every 7 frames or so), linear between; a pair of affinity 0 has none (it never binds).
    Returns (names, matrix with None)."""
    names, A = rna_affinity(seed)
    G = [[(None if A[i, j] <= 0.0 else round(weak + (strong - weak) * (A[i, j] - 0.4) / 0.6, 2)) for j in range(20)] for i in range(20)]
    return names, G


NMAX = max(s["n"] for s in SPECIES.values())             # point slots per body: the largest body's count

# ---- the base: every iteration is this plus its own changes --------------------------------------------------------------------------------------------------------
BASE = dict(
    frames=1000, every=4, seed=1,
    movie_fps=25,              # frames per second of the movie, which draws every stored row (frames / every + 1 of them): 251 rows at 25 per second = 10 s
    box=(19.2, 10.8, 5.2),     # the slab the walls hold, length units (192 x 108 x 52 nm): 16:9 seen along z, about four bodies deep
    margin=0.6,                # world = box + this on every side (the contact lattice clamps at the world's faces)
    counts=dict(ribosome=28, chaperonin=40, rod=40, trimer=40, enzyme=90, small=170),      # 280 length units^3 of bodies = 26 % of the slab's 1,078
    gap=0.3,                   # least distance between the points of two bodies at the start, length units (3 nm)
    youngs=20.0,               # Young's modulus of every body (elastic: the shape is kept)
    substep=0.01,              # MPM substep, time units: 100 per frame; elastic wave speed 4.7 x 0.01 / 0.2 (grid) = 0.24 of a grid cell per substep
    friction=0.05,             # Langevin drag rate, per time unit: a body's velocity is remembered for 1 / 0.05 = 20 frames (smooth paths, a movie's choice; a real molecule forgets in picoseconds)
    kT=1.0e-4,                 # thermal energy: free diffusion kT / (friction x mass) = 0.002 length^2 per time for the 125-point body of mass 1 (9e-3 for 27 points, 7e-4 for 343)
    rep=600.0, slack=0.2,      # pair_potential[cell_contact]: overlap stiffness, tolerated overlap (exp05/exp06's values)
    ctens={},                  # contact tension between two types, {"a-b": value} in the units of a cell surface tension (negative = sticky); every pair not named is 0
    wall=400.0,                # box wall stiffness, per time^2: sqrt(400) x substep 0.01 = 0.2, well under the explicit limit 2
    membrane=None,             # (or a list of them, lowest first) a sheet of `lipid` patches across the slab: dict(y = its height above the slab's floor, every = one site in this many holds a protein, proteins = the kinds
                               # in turn (default `channel`), heads = "above" | "below": the side of a `receptor`'s head); see membrane_bodies
    region={},                 # {species: "below" | "above" | "between"} the membrane(s): where the random placement may put that species (default anywhere free)
    youngs_of={},              # {species: Young's modulus} for the kinds that are not at `youngs`
    bonds=None,                # dict(k = spring stiffness, reach = bonding distance): apical_junction -- at the first substep every pair of points of two DIFFERENT bodies closer than
                               # `reach` is tied by a spring at its then length -- only between the membrane's bodies and the chains' subunits (the cell block `bonded`, 1 for them, 0
                               # for a body placed at random). With reach 0.24 that is the points seeded one spacing (0.2) apart, and up to 0.24 on the outside of a thread's turn:
                               # the consecutive subunits of a chain, the neighbours in a membrane sheet. What holds a chain for 1,000 frames (it10: a contact tension does not).
                               # k 4 = Young's modulus 20 x the point spacing 0.2, the stiffness of the material itself
    walk=None,                 # dict(types = {species: drive}, Dr): active_force[persistent_walk] -- every point of a body of those species is pushed along the body's own direction,
                               # which wanders (rotational diffusion Dr, per time unit: the direction is kept about 1 / (2 Dr) frames). `drive` is an acceleration: against the bath's
                               # drag a free body settles at drive / friction length units per frame (4e-4 / 0.0125 = 0.032: 320 nm in 1,000 frames, 6x its thermal speed 0.005)
    binding=None,              # dict(pairs = {"a-b": share} ("*": every pair), on, off, stiffness, strength, rest, range, min_area, on_block = None, valence = None): BONDS THAT FORM AND BREAK, three operators that only work
                               # together. contact_finder: every frame, a junction row for each pair of bodies with points closer than `range`, with its contact area.
                               # bond_binding: the junction's bound share rho rises at rate `on` toward `pairs[a-b]` (0 for a pair of kinds not named: no bond) while the contact
                               # area is above `min_area`, and falls at rate `off` x exp(load / `strength`) (Bell: a loaded bond lets go sooner). bond_springs: at every substep
                               # each pair of points of the two bodies closer than `range` is pulled together beyond `rest` with force `stiffness` x rho x the stretch. A tether
                               # only pulls when stretched: it does not make a body spread over its partner, as a negative contact tension does (it10, it12, it15).
                               # `on_block`: a cell block whose first column replaces `on`, the junction taking the mean of its two bodies' values -- with "chem" (the signal's
                               # excitation u, 0 to 1) two bodies bind only while one of them is excited
    signal=None,               # (or dict(model = "rps", a, rate, D, contact): three competitors in cyclic competition, see `build`)
                               # dict(pace = [kinds] or strip = width, times, duration, amp, rate, D, contact): A SIGNAL PASSED AT CONTACTS, three operators. Every body carries an excitation u
                               # and a recovery v (cell block `chem`). cell_chem_react[aliev_panfilov, paced]: u fires above its threshold 0.15, stays up, then v brings it down
                               # and holds it down for a while (refractory); the bodies of the kinds `pace` get a current `amp` for `duration` frames at each of `times`.
                               # cell_neighbours[point_contact_warp]: which bodies touch (points closer than `contact`) and by how many point pairs. cell_chem_diffuse
                               # [contact_weighted]: u passes between two touching bodies at rate `D`, weighted by their contact. `rate` scales the reaction's clock (0.15: an
                               # excitation lasts on the order of 100 frames). The movie shows u: a body's fill is mixed toward pale yellow by its u
    push=None,                 # dict(types = {species: drive}, Dr, gate_area): active_force[contact_push] -- a motor that moves only by pushing on what it touches. At each junction
                               # row of contact_finder (so `binding` must be on) a body of those species pushes its partner along its own slowly turning direction, in the plane
                               # of their contact, and is pushed back by as much: the forces add to zero, a body that touches nothing is not driven, and on a long bonded filament
                               # the motor is what moves. `gate_area`: the contact area, length units^2, at which a pair pushes with the full drive. `block` (in place of
                               # `types`): the drive of EVERY body is the first column of that cell block -- with "chem" its excitation u, 0 to 1, and `gate_area` then sets the
                               # scale: a contact of 0.4 pushes with 0.4 / gate_area x u
    unbonded=(),               # kinds that the static springs of `bonds` never tie, though they are seeded on a chain (riders that must be free to walk)
    grow=None,                 # dict(kinds = {species: rate}, factor, room): GROWTH AND DIVISION, four operators once per frame, as exp05 runs them on cells. cell_grow[mpm]: the
                               # mass of a body of those species (a number of points, cell block `mass`) grows as dm/dt = 3 x rate x m (it doubles in ln 2 / (3 rate) frames); a
                               # body of another species has rate 0. cell_populate[mpm]: wakes dormant point slots of the body until it holds its mass. cell_cycle[mpm, doubler_mass]:
                               # marks a body whose mass has reached `factor` x its mass at birth. cell_divide[cleave_typed]: cleaves a marked body in two of its own kind. `room`:
                               # the free body slots the cell set starts with (its `buffer` above the seeded count)
    vesicle=None,              # a closed membrane seen as a ring: dict(x, y = its centre from the slab's left and floor, r = its radius, every = one site in this many a protein with its head
                               # outward, proteins = their kinds in turn), `lipid` patches round a cylinder along z; see vesicle_bodies
    chains=None,               # [dict(kind = a block species, n = how many, m = subunits each, bend = 0, band = None, tilt = None)]: rows of subunits lying in the slab's plane, straight (bend 0) or
                               # meandering, anywhere, on their species' side of a membrane (`region`), or between the two heights `band` above the slab's floor; at any angle in
                               # the plane, or within `tilt` radians of the x axis; see chain_groups. Or `rows` = [(height above the floor, depth), ...] in place of n: a straight
                               # chain along x at each of those places (an ordered array; of one kind, the chains with `rows` are listed before those without). `riders` =
                               # dict(kind, every): a body of that kind seated on one subunit in `every` and tied to it by the bonds; see _riders
    # BIG SCENES (Cedric, 2026-11-17 15:40: "run A 100K molecules 1000 frames, run B 10M molecules 1000 frames ... reduce the number of particles per body to 7, increase world
    # scene accordingly ... we should not save at the end of the simulation but every 100 frames, generate temp movies every 100 frames and add them to the watcher")
    scale_to=None,             # the number of bodies: every count scaled by scale_to / (their sum), and the slab widened in x and y by the square root of that factor (its depth
                               # kept), so the bodies keep their share of the volume (_scaled)
    coarse=None,               # dict(points = 7, n_grid = 6, cell = 2): every body is 7 points of its kind's shape (body_points at that count) on a lattice of spacing s with 7 s^3 = the scene's
                               # mean body volume, each body on its own grid of n_grid^3 cells of side `cell` x s (two points per cell along an axis: the gather keeps a point within
                               # n_grid / 2 - 2 cells of its body's centre, 1 cell = 2 s here, and a 7-point body reaches 1.6 s); every length of the scene tied to the 2 nm spacing is
                               # scaled by s / 0.2 and every area by its square, the contact lattice kept at s (build)
    chunk=0,                   # general.save_chunk_frames: the rows recorded in every that many frames are written to their own file and no whole trajectory at the end
    record_points=True,        # False: the material points are not recorded (the set's `record: false`); the movie is drawn from the bodies' centres (render_big)
    record_junctions=True,     # False: nor the junction rows (bonds are not drawn)
    junction_buffer=16384,     # the junction set's rows: room for every touching pair (it33x4: at most 691 for 380 bodies)
    warmup=0,                  # general.warmup: frames run before frame 0 and not recorded (a coarse scene's lattice seed melts in them)
    sites=None,                # dict(k): binding sites along the chains -- every k consecutive subunits of a chain are ONE site, contact_finder[sites] counting all the point
                               # pairs between a body and a site in one junction (cell block `site`: each cell's site's first cell, from the seed table's `cell__site`)
    periodic=False,            # True: a PERIODIC BOX -- no walls, the world the box itself (margin 0), the contact finder (sites) and the Warp contact penalty wrapping every axis,
                               # bond_springs taking the nearest image; positions stay unwrapped (a body is never cut by a face), the movie draws each body shifted into the box
    matrix_panel=False,        # True: at the right of every movie frame the scene's binding matrix (free energies or affinities) and, live, the bonds held per pair of kinds
                               # with their total (matrix_hook)
    panels3d=False,            # True: the movie in two panels -- at left the whole box seen from 30 degrees above, at right the slice `section` from above (render_panels)
    trim_slots=False,          # True: each body's point slots are the scene's largest body's count, not NMAX (343): the same bodies on fewer slots, faster
    section=None,              # (z0, z1) above the slab's floor, length units: the movie draws only the bodies whose centre lies between (a deep box seen through a slice)
    seed_from=None,            # dict(run, row, kinds, per_bin, singles): the bodies seeded are bonded GROUPS taken whole from another run's row (its points as they are),
                               # up to `per_bin` groups of each size class (2, 3-4, 5-8, 9+) of each of `kinds` and `singles` unbonded bodies of each, spaced apart in an
                               # otherwise empty slab, z centred (_seed_groups)
    window=None,               # a big scene's window, in sizes of the scene before scaling (2: 384 x 216 nm): its bodies' points and bonds are recorded at every stored row
                               # (general.save_chunk_window) and drawn as step 21 is drawn (render), the whole slab as a dot per body in the inset
    fast=None,                 # dict(pp, strain, touched, capture): the operator variants tools/exp09_speed.py measured -- mpm_scatter/gather[warp_pp], mpm_strain[warp],
                               # pair_potential[cell_contact_touched_nodes], the substep block captured as one CUDA graph
)

ITERS = {
    "it01": dict(why="iteration 1, the crowded cytoplasm: 408 bodies of six kinds (28 two-lobed ribosomes of 343 points, 40 barrels, 40 rods and 40 three-lobed complexes of "
                     "125 points, 90 globules of 64 points, 170 of 27 points) fill 26 % of a slab 192 x 108 x 52 nm, jostled by the thermal bath and kept apart by the "
                     "overlap penalty alone (no adhesion): does the crowd read as a molecular landscape, and how far does each kind wander in 1,000 frames (0.2 ms)"),
    "it02": dict(parent="it01", membrane=dict(y=7.3, every=9),
                 counts=dict(ribosome=18, chaperonin=26, rod=26, trimer=26, enzyme=58, small=110, antibody=16, serum=30),
                 region=dict(ribosome="below", chaperonin="below", rod="below", trimer="below", enzyme="below", small="below", antibody="above", serum="above"),
                 ctens={"lipid-lipid": -0.2, "lipid-channel": -0.2},
                 why="iteration 2, from it01 ONE change: a membrane across the slab, two thirds of the way up -- a sheet of 95 sites, 10 x 6 x 10 nm patches of 75 points "
                     "with one site in nine a protein standing through it (125 points, 9 x 22 nm), the patches sticking to each other and to the proteins (contact tension "
                     "-0.2) -- with it01's crowd at the same 26 % below it and a sparse outside above it (16 Y-shaped bodies and 30 globules, 10 %): does a sheet of "
                     "sticky MPM patches hold as a membrane for 1,000 frames between a crowded and a sparse side"),
    "it03": dict(parent="it01", ctens={"trimer-trimer": -0.15},
                 why="iteration 3, from it01 ONE change: the three-lobed complexes (gold) stick to each other, contact tension -0.15 between two of them and 0 for every "
                     "other pair: do the 40 of them find each other in the crowd and gather into clusters within 1,000 frames (a condensate), or stay caged apart"),
    "it04": dict(parent="it01", box=(19.2, 10.8, 9.0), counts=dict(ribosome=48, chaperonin=69, rod=69, trimer=69, enzyme=156, small=294),
                 why="iteration 4, from it01 ONE change: the slab is 90 nm deep instead of 52 nm, at the same 26 % of its volume (705 bodies instead of 408): it01 leaves "
                     "about a sixth of the frame as bare ground between the bodies; does a deeper slab cover the frame edge to edge"),
    "it06": dict(parent="it02", youngs_of={"channel": 100.0},
                 why="iteration 6, from it02 ONE change: the proteins through the membrane are 5x stiffer (Young's modulus 100, every other body 20): in it02 the sheet held "
                     "for 1,000 frames and nothing crossed it, but the contact tension -0.2 that holds the proteins in the sheet smeared them along it (their radius of "
                     "gyration grew to 1.28-1.47 of its start); a strain set by tension over stiffness should fall about 5x, so the proteins stand through the sheet as seeded"),
    "it05": dict(parent="it01", kT=2.5e-5, friction=0.0125,
                 why="iteration 5, from it01 ONE change: a calmer bath with the same free diffusion -- thermal energy 1e-4 -> 2.5e-5 and drag 0.05 -> 0.0125 per time unit, "
                     "so kT / (drag x mass) is unchanged while every body's thermal speed is halved and its velocity is remembered 80 frames instead of 20: it01's bodies "
                     "step about 7 px per movie frame (the 12 nm ones) and turn every 5 movie frames, a shake; do slower, longer strides read as a drift"),
    # THE 20 SEQUENTIAL JOBS (Cedric, 2026-11-14 20:25: "do 20 more jobs, one after the other, look at the movie when it lands before running a new job"): it07-it26,
    # each written after the previous movie was read (`look`), ONE change from the parent it names
    "it07": dict(parent="it06", ctens={"lipid-lipid": -0.2, "lipid-channel": -0.05},
                 why="iteration 7, from it06 ONE change: the contact tension between a membrane patch and a protein through the membrane is -0.05 instead of -0.2 (patch to "
                     "patch stays -0.2, the sheet's own hold): in it02 and it06 that glue pulled the proteins' surface along the sheet and the patches over the proteins "
                     "(proteins' radius of gyration 1.18-1.47 of its start, whatever their stiffness); a quarter of the glue should leave them standing through the sheet, "
                     "if the patches closing round them do not squeeze them out"),
    "it08": dict(parent="it04", kT=2.5e-5, friction=0.0125,
                 why="iteration 8, from it04 ONE change: it05's calmer bath (thermal energy 2.5e-5, drag 0.0125 per time unit) in it04's 90 nm deep slab: the two "
                     "changes that each improved the picture -- the frame covered edge to edge (5 % bare ground) and 29 % less change per movie frame -- together, as the "
                     "base of what follows"),
    "it09": dict(parent="it08", membrane=dict(y=7.3, every=9),
                 counts=dict(ribosome=31, chaperonin=45, rod=45, trimer=45, enzyme=100, small=190, antibody=28, serum=52),
                 region=dict(ribosome="below", chaperonin="below", rod="below", trimer="below", enzyme="below", small="below", antibody="above", serum="above"),
                 ctens={"lipid-lipid": -0.2, "lipid-channel": -0.05},
                 why="iteration 9, from it08 ONE change: it07's settled membrane put into the 90 nm deep, calm slab -- a sheet of 171 sites (19 x 9 patches of 10 x 6 x 10 nm, "
                     "one site in nine a protein standing through it), patch-to-patch contact tension -0.2 and patch-to-protein -0.05, the crowd at 26 % below it and a "
                     "sparse outside at 10 % above it: does the sheet hold when it is 9 patches deep instead of 5, and does the membrane scene gain the deep slab's full frame"),
    # FROM HERE THE GEOMETRIES AND THE INTERACTIONS CHANGE (Cedric, 2026-11-15 07:05: "in the following run modify the geometries of the objects and their interaction see
    # example of goodsell in .../exp09_molecular_landscape/download"). The twelve pictures are made of what it01-it09 lack: many-lobed bodies, long beaded filaments,
    # thin threads, and bodies bound to them. So an iteration is no longer ONE change from its parent: it is a scene, named for what it shows
    "it10": dict(parent="it08",
                 counts=dict(ribosome=30, clover=50, antibody=40, bean=40, ring=30, dimer=70, pear=80, small=150),
                 chains=[dict(kind="actin", n=8, m=20), dict(kind="bead", n=5, m=30, bend=0.2)],
                 ctens={"actin-actin": -0.05, "bead-bead": -0.05, "pear-actin": -0.05, "ribosome-bead": -0.05},
                 why="iteration 10, the cytoplasm recast after the reference pictures (it08's 90 nm deep slab and calm bath): NEW GEOMETRIES -- 8 straight filaments of 20 "
                     "subunits (8 nm blocks in a row, 160 nm long), 5 meandering threads of 30 elements (6 nm blocks), four-lobed tetramers, two-lobed dimers, pear-shaped "
                     "heads, curved rods, rings with a hole, Y-shaped bodies, with the two-lobed ribosomes and the 6 nm globules kept; NEW INTERACTIONS -- a filament and "
                     "a thread are held together by the contact tension between their own subunits alone, the pear-shaped heads stick to the filaments and the ribosomes "
                     "to the threads, every one of these four tensions -0.05 (the value that held it07's membrane proteins without smearing them; at -0.2 the smoke's "
                     "filament subunits spread over each other, 26 of 160 past 1.10 of their radius of gyration in 60 frames): do the chains hold for 1,000 frames, do "
                     "the heads decorate the filaments and the ribosomes line the threads"),
    # it10's lesson (read 2026-11-15 09:25): a chain held by a contact tension degrades after frame 100-200 -- thread elements shredded (radius of gyration up to 3.5 of
    # the start), filament subunits to 1.7, bonds open to 1.9 -- so it11 goes WITHOUT the filaments and threads it was smoked with at 08:26 and 08:37; chains come
    # back when they are held by bonds, not by a tension. The docking tension is -0.03, not -0.05: it10's heads at -0.05 crept to 1.33
    "it11": dict(parent="it08", membrane=dict(y=6.2, every=6, proteins=["receptor", "channel"], heads="above"), gap=0.2,
                 counts=dict(antibody=45, serum=40, dimer=30, clover=125, ribosome=16, bean=34, small=130),
                 region=dict(antibody="above", serum="above", dimer="above", clover="below", ribosome="below", bean="below", small="below"),
                 ctens={"lipid-lipid": -0.2, "lipid-channel": -0.05, "lipid-receptor": -0.05, "antibody-receptor": -0.03},
                 why="iteration 11, a cell surface (it08's 90 nm deep slab and calm bath): NEW GEOMETRIES -- a membrane 62 nm above the floor whose proteins are of two kinds "
                     "in turn, one site in six: a receptor (a 14 nm head above the sheet on a neck through it) and the plain channel; inside, a dense crowd of "
                     "four-lobed tetramers with a few ribosomes, curved rods and globules; outside, Y-shaped bodies, dimers and globules; NEW INTERACTION -- the Y-shaped "
                     "bodies dock on the receptors' heads (contact tension -0.03; the sheet's own patch-to-patch tension stays -0.2, patch-to-protein -0.05): do the Y "
                     "bodies collect on the heads over 1,000 frames, do the receptors stay upright in the sheet, and does anything cross it"),
    "it12": dict(parent="it10", bonds=dict(k=4.0, reach=0.24), ctens={"pear-actin": -0.03, "ribosome-bead": -0.03},
                 why="iteration 12, it10's scene with its chains held by BONDS instead of a contact tension: at the first substep every pair of points of two different "
                     "chain subunits seeded within 2.4 nm of each other -- the facing points of two consecutive subunits of a filament or a thread, and nothing else -- is tied "
                     "by a spring at its then length, as stiff as the material itself (apical_junction, k 4); the tension between a chain's own subunits is gone (0), and "
                     "the heads and the ribosomes bind at -0.03 instead of -0.05. In it10 the chains held by a tension of -0.05 were whole at frame 100 and then "
                     "degraded (thread elements pulled into pieces, filament subunits to 1.7 of their radius of gyration, heads to 1.33): do bonded chains stay whole and "
                     "keep their subunits' shape for 1,000 frames"),
    "it13": dict(parent="it08", bonds=dict(k=4.0, reach=0.24),
                 # 160 nm chains in the 192 nm slab: a chain settles 3 to 7 % longer than it is seeded (the overlap penalty's rest distance), and the smoke of 08:59
                 # with 180 and 176 nm chains pressed them on the walls -- they buckled by up to 15 nm, bonds opened to 2.3 and thin subunits were crushed (rg 1.6)
                 chains=[dict(kind="myosin", m=16, rows=[(y, z) for y in (1.8, 5.4, 9.0) for z in (1.5, 4.5, 7.5)]),
                         dict(kind="actin", m=20, rows=[(y, z) for y in (3.6, 7.2) for z in (1.5, 4.5, 7.5)])],
                 counts=dict(pear=170, dimer=40, small=150),
                 ctens={"pear-myosin": -0.03},        # nothing sticks to the thin filaments: a tension on a 64-point subunit makes it creep (it15: 1.48 at -0.03; the smoke of 10:10: 1.23 at frame 250 at -0.015)
                 why="iteration 13, an ordered array of filaments (it08's 90 nm deep slab and calm bath): NEW GEOMETRY -- 9 thick filaments (16 subunits of 10 nm, "
                     "160 nm long) and 6 thin ones (20 subunits of 8 nm) laid along x in alternate layers 18 nm apart, at three depths, with 170 pear-shaped heads, "
                     "40 dimers and 150 globules between them; NEW INTERACTION -- the filaments are held by bonds (it12) and the heads hold on to the thick filaments "
                     "(contact tension -0.03; nothing sticks to the thin ones): do the heads gather on the thick filaments, and do "
                     "15 filaments that span the frame stay straight and parallel for 1,000 frames"),
    "it14": dict(parent="it08", bonds=dict(k=4.0, reach=0.24), membrane=dict(y=6.2, every=6, proteins=["receptor", "channel"], heads="above"),
                 counts=dict(antibody=45, serum=40, dimer=30, clover=90, ribosome=14, bean=30, small=120),
                 region=dict(antibody="above", serum="above", dimer="above", clover="below", ribosome="below", bean="below", small="below", actin="below", bead="below"),
                 chains=[dict(kind="actin", n=4, m=20, band=(4.0, 5.2), tilt=0.012), dict(kind="bead", n=3, m=30, bend=0.15, band=(1.0, 5.4), tilt=0.1)],
                 ctens={"antibody-receptor": -0.03, "actin-channel": -0.03, "actin-receptor": -0.03},      # nothing sticks to the threads: in it12 a 27-point thread element bound at -0.03 was pulled out of its chain after frame 500
                 why="iteration 14 (a candidate), the cell surface of it11 held by BONDS and with a skeleton under it: the membrane's patches and proteins are tied to "
                     "their neighbours in the sheet by springs at their seeded distance (apical_junction, as it12's chains), with NO contact tension between them (it11: "
                     "-0.2 patch to patch, -0.05 patch to protein, and a channel crept to 1.37 of its radius of gyration); under the sheet 4 bonded filaments and 3 "
                     "bonded threads lie along it, the filaments sticking to the inner ends of the membrane proteins (-0.03); the Y bodies dock on the receptors' heads (-0.03) as in "
                     "it11: does a membrane held by springs alone stay closed for 1,000 frames, do its proteins keep their shape, and does the skeleton stay under it"),
    "it15": dict(parent="it08", bonds=dict(k=4.0, reach=0.24),
                 membrane=[dict(y=3.6, every=6, proteins=["receptor"], heads="above"), dict(y=6.66, every=6, proteins=["ligand"], heads="below")],      # 6.66, not 6.6: the facing heads 2.6 nm apart, beyond the bonds' reach (2.4 nm), so what joins them is the contact tension
                 counts=dict(clover=45, ribosome=10, bean=20, enzyme=50, small=140, dimer=70, ring=24, pear=40),
                 region=dict(clover="below", ribosome="below", bean="below", enzyme="below", small="between", dimer="above", ring="above", pear="above", actin="above", bead="below"),
                 chains=[dict(kind="actin", n=3, m=20, band=(7.6, 9.2), tilt=0.012), dict(kind="bead", n=2, m=28, bend=0.15, band=(0.4, 2.8), tilt=0.1)],
                 ctens={"receptor-ligand": -0.03, "actin-ligand": -0.03, "bead-receptor": -0.03, "pear-actin": -0.03, "ribosome-bead": -0.03},
                 why="iteration 15, two cells meeting: TWO membranes held by springs, 30 nm apart, the lower one carrying receptors with their heads up, the "
                     "upper one their partners with their heads down, site against site, so that every receptor's head starts 2.6 nm from its partner's (no spring between them) and the two stick "
                     "(contact tension -0.03); below the lower membrane one cell's crowd (tetramers, ribosomes, curved rods, globules, 2 threads), above the upper one another's "
                     "(dimers, rings, heads, 3 filaments), globules in the cleft: do the pairs of heads hold the two membranes together across the cleft for 1,000 "
                     "frames, and does each crowd stay on its side"),
    # reach 0.29, not 0.24: round the ring two neighbours' points are 0.20 apart in the wall's inner layer, 0.27 in its middle layer and 0.35 in its outer one. At 0.24
    # only the inner layer is tied (5 springs a joint) and each joint is a hinge: in the smoke of 12:20 the ring lost its round (radius of the patches' centres
    # 2.91-2.96 -> 2.40-3.16 in 250 frames). At 0.29 the middle layer is tied too, and a joint resists bending
    "it16": dict(parent="it08", bonds=dict(k=4.0, reach=0.29), vesicle=dict(x=6.4, y=5.4, r=3.0, every=5, proteins=["receptor"]),
                 counts=dict(serum=60, antibody=45, ribosome=22, clover=45, bean=30, dimer=50, small=150),
                 region=dict(serum="lumen"),
                 chains=[dict(kind="actin", n=3, m=10, tilt=1.2)],
                 ctens={"antibody-receptor": -0.03},
                 why="iteration 16, a vesicle in the crowd: a CLOSED membrane held by springs -- 17 patches round a ring 60 nm across, 9 deep, one site "
                     "in five a receptor with its head outward -- full of 9 x 12 nm globules, in a cytoplasm of ribosomes, tetramers, curved rods, dimers, globules, "
                     "three short bonded filaments and 45 Y bodies that stick to the receptors' heads (contact tension -0.03): does the ring stay closed and round and keep "
                     "its cargo for 1,000 frames, and do the Y bodies coat it"),
    "it17": dict(parent="it08", bonds=dict(k=4.0, reach=0.24),
                 chains=[dict(kind="bead", n=7, m=30, bend=0.2, riders=dict(kind="ribosome", every=6))],
                 counts=dict(clover=60, bean=40, dimer=70, ring=30, enzyme=80, small=150),
                 why="iteration 17 (a candidate), threads that carry ribosomes: 7 bonded threads of 30 elements meander through the crowd, each with a ribosome "
                     "seated on one element in six, on alternate sides, and tied to it by springs (no contact tension anywhere: in it12 and it15 a thread element "
                     "under a tension of -0.03 was pulled out of its chain); around them tetramers, curved rods, dimers, rings and globules: does a thread with its "
                     "ribosomes move as one body for 1,000 frames, and does every element keep its shape"),
    # reach 0.31: the small ring has 12 patches, 30 degrees apart, so its wall's middle layers are 0.30 apart (0.27 in it16's ring of 17)
    "it18": dict(parent="it08", bonds=dict(k=4.0, reach=0.31),
                 membrane=dict(y=7.6, every=6, proteins=["receptor", "channel"], heads="above"),
                 vesicle=dict(x=4.6, y=3.4, r=2.2, every=4, proteins=["ligand"]),
                 counts=dict(antibody=22, dimer=24, serum=22, clover=60, bean=26, ring=20, enzyme=50, small=130),
                 region=dict(antibody="above", dimer="above", serum="lumen", clover="below", bean="below", ring="below", enzyme="below", small="below", actin="below", bead="below", ribosome="below"),
                 chains=[dict(kind="actin", n=2, m=12, band=(5.4, 6.6), tilt=0.012), dict(kind="bead", n=2, m=16, bend=0.15, band=(0.4, 6.4), tilt=0.15, riders=dict(kind="ribosome", every=5))],
                 ctens={"antibody-receptor": -0.03},
                 why="iteration 18 (a candidate), a cell's edge with everything so far in one frame: a spring-held membrane with receptors and channels, Y bodies and "
                     "dimers outside that dock on the receptors' heads (contact tension -0.03, the only tension in the scene); inside, a vesicle 44 nm across with its "
                     "own outward-pointing proteins and its cargo, two bonded filaments under the membrane, two threads that carry ribosomes, and a crowd of tetramers, "
                     "curved rods, rings and globules: do the membrane, the vesicle, the filaments and the threads each hold for 1,000 frames in one crowded slab"),
    # THE THIRD CAMPAIGN (Cedric, 2026-11-16 23:10: "run 20 jobs to be added to the watcher exp09, one after the other, add more and operator to get more and more
    # interactions between the proteins, see the results before generating a new job, because now it looks too much as jiggling proteins without too much interactions
    # and interesting phemomena"): it19 onward, each job adds ONE operator that makes the bodies DO something to each other, on the scene of the job before
    "it19": dict(parent="it05", counts=dict(pear=40, clover=40, dimer=60, ring=20, bean=30, enzyme=60, small=120),
                 walk=dict(types={"pear": 8.0e-4}, Dr=0.004),      # the smoke at 4e-4 / 0.002: motors' net displacement 12 nm in 150 frames against 6-8 nm for the passive 64-point bodies: active, but tame
                 why="iteration 19, the first of the interaction ladder -- ADDED OPERATOR: active_force[persistent_walk]. In a slab 52 nm deep, thinner and emptier than "
                     "before (370 bodies, 18 % of its volume, so that what a body does can be seen), the 40 pear-shaped bodies are motors: each is pushed along its own "
                     "direction, which wanders slowly (kept about 125 frames), with a drive that would carry a free body 640 nm in the 1,000 frames, 13 times its "
                     "thermal speed; the other 330 bodies are passive. Do the motors cross the frame, plough lanes through the passive crowd, and gather at the walls"),
    # it19 read (2026-11-17 00:45): the motors go 47 nm net in 1,000 frames (median) against 19-21 nm for passive bodies of their size: active, but in the movie
    # they do not yet look different enough from the crowd. So it20 also gives them 2.5x the drive (2e-3), with the bonds
    "it20": dict(parent="it19", walk=dict(types={"pear": 2.0e-3}, Dr=0.004), binding=dict(pairs={"clover-clover": 1.0, "pear-small": 1.0}, on=1.0, off=0.005, stiffness=1.0e-3, strength=1.0e-3, rest=0.22, range=0.45, min_area=0.12),
                 why="iteration 20 -- ADDED: BONDS THAT FORM AND BREAK (contact_finder, bond_binding, bond_springs), on it19's scene, its motors given 2.5x the drive (in it19 they "
                     "went 47 nm in the 1,000 frames against 20 nm for a passive body of their size: active but tame). Two pairs of kinds may "
                     "bind: tetramer to tetramer, and motor to 6 nm globule. A pair that touches is bound within a frame; a bond lets go by itself after about 200 frames "
                     "and sooner under load; a bound pair is held by tethers that pull only when stretched, ten of them about as strong as a motor's drive. Do the "
                     "tetramers gather into clusters that grow, merge and shed, and do the motors pick up globules and carry them"),
    # ONE source, the bodies within 15 nm of the left wall, instead of the 20 rings scattered through the slab that the smoke of 00:25 fired: twenty foci at once fill the
    # slab from everywhere; one strip sends a front across the frame
    # the strip smoke of 01:24 (D 0.05, rate 0.15): the front advances 0.27 nm per frame (24 nm from the wall at frame 90, 82 nm at frame 300): it would need 700 frames
    # to cross the 192 nm. So the run passes the excitation 3x faster between touching bodies (D 0.15) and runs the reaction's clock a third faster (rate 0.2)
    "it21": dict(parent="it20", signal=dict(strip=1.5, times=[40, 290, 540, 790], duration=6, amp=1.0, rate=0.2, D=0.15, contact=0.45),
                 why="iteration 21 -- ADDED: A SIGNAL PASSED AT CONTACTS (cell_neighbours, cell_chem_react[aliev_panfilov], cell_chem_diffuse[contact_weighted]), on "
                     "it20's scene with its motors and bonds. Every body carries an excitation: above a threshold it fires, stays up on the order of 100 frames, falls "
                     "and cannot fire again for a while. The bodies within 15 nm of the left wall are made to fire at frames 40, 290, 540 and 790. A body that fires "
                     "excites the bodies it touches, in proportion to their contact. The movie shows the excitation as a pale yellow fill. Does each firing cross the "
                     "slab from left to right as a front, from body to touching body, and how fast, in a crowd where contacts are made and lost"),
    "it22": dict(parent="it21", binding=dict(pairs={"*": 1.0}, on=1.0, on_block="chem", off=0.005, stiffness=1.0e-3, strength=1.0e-3, rest=0.22, range=0.45, min_area=0.12),
                 why="iteration 22 -- CHANGED: THE SIGNAL SETS THE BINDING (bond_binding's rate read from each body's excitation). Any two bodies that touch now bind, "
                     "but only while one of them is excited: the rate at which a junction binds is the mean of its two bodies' excitation (0 at rest, 1 when both "
                     "fire); a bond still lets go by itself in about 200 frames, sooner under load. So each front should leave a wake in which the "
                     "bodies it has excited are tied to their neighbours, and which loosens again once they have recovered: does the crowd clump behind the wave and "
                     "come apart after it"),
    "it23": dict(parent="it05", bonds=dict(k=4.0, reach=0.24), unbonded=("pear",),
                 chains=[dict(kind="actin", n=5, m=20, tilt=0.5, riders=dict(kind="pear", every=4))],
                 counts=dict(clover=30, dimer=40, enzyme=40, small=110),
                 # smoke 1 (02:04; release 0.02 per frame, Bell strength 5e-4, tether range 0.45, drive 3e-3): the motors are fast (57 nm net in 250 frames, median; 136 nm
                 # at most) and the filaments stay put (1 nm), but the motors do not stay on them: 13 of 25 within 12 nm of a subunit at frame 60, 8 at frame 250, and only
                 # about 10 motor-filament junctions bound at any time. So: bonds that hold 4x more load before they let go, a longer tether, a slower release, less drive
                 binding=dict(pairs={"pear-actin": 1.0, "pear-small": 1.0}, on=1.0, off=0.01, stiffness=1.0e-3, strength=2.0e-3, rest=0.22, range=0.6, min_area=0.08),
                 push=dict(types={"pear": 2.0e-3}, Dr=0.001, gate_area=0.4),
                 why="iteration 23 -- ADDED OPERATOR: active_force[contact_push], MOTORS ON TRACKS. Five bonded filaments 160 nm long cross the slab, each "
                     "with five motors seated on it. A motor binds the filament subunits it touches and the 6 nm globules (bonds that form within a frame and let go "
                     "under load), and it moves only by pushing on what it touches, along its own slowly turning direction: on a filament 20 subunits long the "
                     "motor is what moves. Do the motors run along their filaments, in both directions, carrying globules, and what happens where two meet"),
    "it24": dict(parent="it05", counts=dict(replicator=10, clover=30, dimer=40, ring=15, bean=20, small=100),
                 grow=dict(kinds={"replicator": 1.05e-3}, factor=2.0, room=300),
                 why="iteration 24 -- ADDED: GROWTH AND DIVISION (cell_grow[mpm], cell_populate[mpm], cell_cycle[mpm], cell_divide[cleave_typed]). Ten "
                     "red 10 nm globules are replicators: each gains material points at a rate that doubles it in 220 frames, and is cleaved in two when it has doubled; "
                     "the 205 other bodies are passive. Do the ten become about 200 in the 1,000 frames, each lineage a growing colony pushing the passive crowd aside"),
    "it25": dict(parent="it24", binding=dict(pairs={"replicator-replicator": 1.0}, on=1.0, off=0.005, stiffness=1.0e-3, strength=1.0e-3, rest=0.22, range=0.45, min_area=0.08),
                 why="iteration 25 (a candidate) -- ADDED to it24's growth and division: BONDS between replicators (contact_finder, bond_binding, bond_springs). Two "
                     "replicators that touch bind within a frame, so the two halves of a division, born in contact, stay together: does each of the ten lineages "
                     "become a compact colony that grows as a body and pushes the passive crowd aside, instead of a scatter of daughters"),
    "it26": dict(parent="it21", push=dict(block="chem", Dr=0.004, gate_area=130.0),
                 why="iteration 26 (a candidate) -- ADDED to it21's signal: THE SIGNAL DRIVES MOTION (active_force[contact_push] with each body's drive read from its "
                     "excitation). A body pushes on what it touches, along its own slowly turning direction, only while it is excited, about as hard as it20's motors "
                     "at a full contact. So a front should be a band of bodies that shove each other, in a crowd that is still before and after it: does the excitation "
                     "stir the crowd where it passes, and does that change how far the front gets"),
    "it27": dict(parent="it24", signal=dict(strip=1.5, times=[40, 290, 540, 790], duration=6, amp=1.0, rate=0.2, D=0.15, contact=0.45),
                 grow=dict(kinds={"replicator": 2.0e-3}, factor=2.0, room=300, on_signal=dict(a_sw=0.5, hill=4.0)),
                 why="iteration 27 (a candidate) -- THE SIGNAL TRIGGERS GROWTH (cell_grow[mpm_activator] on it24's replicators, with it21's signal). A replicator gains "
                     "material points only while it is excited, fast enough to double during one excitation; the bodies at the left wall are made to fire at frames "
                     "40, 290, 540 and 790 and the excitation crosses the slab from body to touching body. Does each front leave a step in the number of replicators "
                     "behind it, the divisions sweeping the slab from left to right"),
    "it28": dict(parent="it23", push=dict(types={"pear": 2.0e-3}, Dr=0.001, gate_area=0.4, track="actin"),
                 why="iteration 28 -- NEW OPERATOR VARIANT: active_force[track_push], on it23's scene. A motor that touches a filament takes the "
                     "filament's axis as its direction, one of its two senses (half the motors each); everything else is it23's: it pushes on what it touches, binds "
                     "the subunits and the 6 nm globules it touches. In it23 the motors' direction was their own and three quarters of them left their filament: do "
                     "they now run along the filaments to the ends, in both senses, carrying their globules"),
    "it29": dict(parent="it05", counts=dict(pear=40, clover=50, dimer=70, ring=24, bean=36, enzyme=80, small=140),
                 signal=dict(model="rps", a=2.0, rate=0.1, D=0.05, contact=0.45),
                 why="iteration 29 (a candidate) -- ADDED OPERATOR: cell_chem_react[rock_paper_scissor], THREE STATES THAT CHASE EACH OTHER. Every body carries three "
                     "competitors, purple, teal and yellow; each grows toward a shared ceiling and is suppressed by the next (purple by teal, teal by yellow, yellow "
                     "by purple), and all three pass between touching bodies. Each body starts mostly one of them, at random, and is drawn in the mix of the three "
                     "colours. 440 passive bodies at 22 % of the volume, no motors, no bonds. Do the three colours sort into domains that invade each other in turn, "
                     "purple into yellow, yellow into teal, teal into purple, in a crowd whose bodies keep changing neighbours"),
    # smoke 1 (08:01; rate 1.05e-3, ceiling 0.5): the crowding is computed (median 0.1 to 0.3 of a surrounded body's contact) but only 0 to 2 replicators are at the ceiling at
    # a time in this thin crowd: 43 replicators at frame 600 against 46 without the inhibition. To see the population level off within 1,000 frames: growth 1.5x faster
    # (doubling in 144 frames: 1,200 replicators at frame 1,000 if nothing stopped them) and a ceiling of 0.4
    "it30": dict(parent="it25", grow=dict(kinds={"replicator": 1.6e-3}, factor=2.0, room=300, inhibit=dict(ceiling=0.4, gate_area=2.0, tau=10.0)),
                 why="iteration 30 (a candidate) -- ADDED to it25's colonies: CONTACT INHIBITION OF GROWTH (cell_grow[mpm_adhesion] with a ceiling). A replicator stops "
                     "gaining material while it is crowded: while its contact area with other bodies, followed over 10 frames, is at least 0.4 of that of a surrounded "
                     "body; and it grows faster than in it24 (doubling in 144 frames: 1,200 replicators at the end if nothing stopped them). So a replicator inside its colony or hemmed in by the crowd stops, and one at the colony's edge goes on: do the colonies grow at their "
                     "edges only, and does the population level off instead of filling the slab"),
    "it31": dict(parent="it28", signal=dict(strip=1.5, times=[40, 290, 540, 790], duration=6, amp=1.0, rate=0.2, D=0.15, contact=0.45),
                 why="iteration 31 (a candidate) -- it21's SIGNAL on it28's filaments and motors: FILAMENTS AS WIRES. The bodies within 15 nm of the left wall are made "
                     "to fire at frames 40, 290, 540 and 790, and a firing body excites what it touches. The 20 subunits of a filament touch each other all the time, "
                     "over a full face, where two free bodies touch now and then at a point: does the excitation run along the filaments ahead of the front in the "
                     "loose crowd (245 free bodies), and light the motors and the cargo that ride on them"),
    "it32": dict(parent="it05", bonds=dict(k=4.0, reach=0.29), vesicle=dict(x=9.6, y=5.4, r=3.0),
                 counts=dict(replicator=14, clover=30, dimer=40, ring=15, bean=20, small=100), region=dict(replicator="lumen"),
                 binding=dict(pairs={}, on=1.0, off=0.005, stiffness=1.0e-3, strength=1.0e-3, rest=0.22, range=0.45, min_area=0.08),
                 grow=dict(kinds={"replicator": 1.6e-3}, factor=2.0, room=400, inhibit=dict(ceiling=0.8, gate_area=2.0, tau=10.0)),
                 why="iteration 32 (a candidate) -- GROWTH INSIDE A CLOSED MEMBRANE. A ring of 17 membrane patches 58 nm across, held by springs between neighbours, "
                     "encloses 14 replicators (doubling in 144 frames, cleaved in two when doubled, growth stopping only when a replicator is four fifths surrounded); "
                     "the passive crowd is outside. The lumen holds about 90 such bodies packed, which the 14 reach near frame 400: do they then stretch the ring, "
                     "push its patches apart, and get out"),
    # THE FOURTH CAMPAIGN (Cedric, 2026-11-17 08:12: "you could run a follow up of 15 with a rule adhesion only between given pairs of species an number of adhesion per
    # protein, protein dependent 2 3 4, run 5 more jobs"): it33 to it37, one job in flight, each read before the next. The rule is bond_binding[valence]
    "it33": dict(parent="it19", counts=dict(pear=40, clover=40, trimer=40, dimer=60, ring=20, enzyme=60, small=120),
                 binding=dict(pairs={"dimer-dimer": 1.0, "trimer-trimer": 1.0, "clover-clover": 1.0}, valence={"dimer": 2, "trimer": 3, "clover": 4},
                              on=1.0, off=0.005, stiffness=1.0e-3, strength=1.0e-3, rest=0.22, range=0.45, min_area=0.12),
                 why="iteration 33, a follow-up of step 15 with a RULE OF ADHESION (NEW operator variant bond_binding[valence]): bonds only between given pairs of kinds, "
                     "and each body holds at most its kind's number of them. Here a kind binds its own kind, and the number is the kind's number of lobes: two-lobed "
                     "dimers hold 2 bonds, three-lobed complexes 3, four-lobed tetramers 4; the motors of step 15 stir the crowd; a white line joins two bodies that "
                     "hold a bond. Do the dimers string into chains and rings, the three-lobed complexes into branched trees, the tetramers into networks"),
    "it34": dict(parent="it33", counts=dict(pear=40, clover=50, dimer=100, ring=20, enzyme=40, small=100),
                 binding=dict(pairs={"clover-dimer": 1.0}, valence={"clover": 4, "dimer": 2},
                              on=1.0, off=0.005, stiffness=1.0e-3, strength=1.0e-3, rest=0.22, range=0.45, min_area=0.12),
                 why="iteration 34 (job 2 of 5) -- the valence rule between TWO kinds: HUBS AND LINKERS. A four-lobed tetramer binds only two-lobed dimers, up to 4 of "
                     "them; a dimer binds only tetramers, up to 2: so a dimer is a bridge between two tetramers and nothing else binds. 50 tetramers and 100 dimers, the "
                     "ratio at which every bond can be used (50 x 4 = 100 x 2), among the motors and the passive crowd of step 15: does a network of tetramers joined by "
                     "dimer bridges grow, and how large do its pieces get"),
    "it35": dict(parent="it34", counts=dict(pear=40, clover=50, dimer=50, ring=20, enzyme=40, small=100),
                 binding=dict(pairs={"clover-dimer": 1.0, "clover-small": 1.0}, valence={"clover": 4, "dimer": 2, "small": 1},
                              on=1.0, off=0.005, stiffness=1.0e-3, strength=1.0e-3, rest=0.22, range=0.45, min_area=0.12),
                 why="iteration 35 (job 3 of 5) -- it34's hubs and linkers with CAPS: half the two-lobed linkers (valence 2) replaced by 6 nm globules of valence 1 that "
                     "also bind the tetramers. A globule on a tetramer takes one of its four bonds and bridges to nothing, so it ends a branch: do the caps hold the "
                     "network to small pieces, the valence of what competes for a hub deciding the size of what assembles"),
    "it36": dict(parent="it28", binding=dict(pairs={"pear-actin": 1.0, "pear-small": 1.0}, valence={"pear": 3, "actin": 2, "small": 1},
                                             on=1.0, off=0.01, stiffness=1.0e-3, strength=2.0e-3, rest=0.22, range=0.6, min_area=0.08),
                 why="iteration 36 (job 4 of 5) -- the valence rule on step 19's MOTORS ON TRACKS (with the motors that take the filament's axis): a motor holds at most "
                     "3 bonds, a filament subunit 2, a 6 nm globule 1. So a motor can grip its filament and carry at most one or two globules, a globule rides on one "
                     "motor only (no cargo shared between motors, which tied step 16's motors together), and a subunit is gripped by at most two motors: do the motors "
                     "run along their filaments each with its own cargo, and queue where they meet"),
    "it37": dict(parent="it22", binding=dict(pairs={"*": 1.0}, on=1.0, on_block="chem", off=0.005, stiffness=1.0e-3, strength=1.0e-3, rest=0.22, range=0.45, min_area=0.12,
                                             valence={k: 2 for k in ("pear", "clover", "dimer", "ring", "bean", "enzyme", "small")}),
                 why="iteration 37 (job 5 of 5) -- step 18 (bodies bind only while excited) WITH THE VALENCE RULE: every body holds at most 2 bonds. In step 18 each "
                     "front tied the bodies it excited into clumps that never came apart; with two bonds a body is a link in a chain, so a front should leave strands "
                     "and rings instead of clumps: does the crowd behind each front string into chains, and do the chains open again once the bodies recover"),
    # x4 (Cedric, 2026-11-17 09:50: "rerun simulation and follow up steps x4 longer, x4 fps"): the valence jobs run 4,000 frames (800 us) instead of 1,000, every 4th frame
    # stored (1,001 rows), and the movie plays them at 100 per second instead of 25: the same 10 s, four times the time. `x4` of a key is that key with those three numbers

    # THE BIG TWINS of step 21 (Cedric, 2026-11-17 15:40): it33's scene -- the valence rule, the motors, the passive crowd -- at 100,000 and at 10,000,000 bodies, 1,000 frames each,
    # on one H200; the rows of every 100 frames written as they come (`chunk`) and a movie of all of them so far put into the watcher after each (`live`)
    "a100k": dict(parent="it33", scale_to=100_000, chunk=250, junction_buffer=1 << 19, frames=4000, every=4, movie_fps=100, record_points=False, record_junctions=False,
                  window=2.0, fast=dict(pp=True, strain=True, capture=False),
                  why="TWIN A of step 21 at 100,000 molecules: it33's scene (the valence rule -- dimers 2 bonds, three-lobed complexes 3, tetramers 4, each kind binding its own; "
                      "the walking motors; the passive crowd) with every count x 263 and the slab widened 16.2 x in x and y to 3.1 x 1.75 um, its depth (52 nm) and its share of "
                      "the volume (19 %) kept, every body as in step 21; 1,000 frames (200 us) on one H200. At left the whole slab, every body a dot at its centre; at right a "
                      "192 x 108 nm window from its centre, the size of step 21's whole slab, every body drawn by its points. Do the chains, trees and networks of step 21 form alike when the crowd is 263 times larger"),
    # 1,000 frames, not A's 4,000 (2026-11-17 19:15): at an estimated 150 to 300 s per frame (A's 26 s scaled by B's 5 x the points and 6 x the grid nodes) 4,000 frames
    # would take 8 to 15 days, over the queue's 14
    # 2026-11-17 21:50: the warm-up of 400 frames and the tolerance 0.2 of bsmoke400 (0.5 nm per stored row after it, bonds 6.4 nm centre to centre), the Warp contact
    # (pair_potential[cell_contact_warp_pp]: the default's 85 GB of stencils at 70 million points). NOT no warm-up: the fast early motion comes from the contact penalty
    # pushing apart the seeded bodies whose points are within its reach (bsfinal, no warm-up and tolerance 0.15: 8.5 -> 4.3 nm per row over 40 frames, a decay time of about
    # 55 frames; bs_nocontact and bs_cgrid, calm, had in effect no contact force) -- a 400-frame warm-up runs it out (e^-7)
    # 9,900,000, not 10,000,000 (2026-11-17 22:00): Warp holds at most 2^31 entries along one axis, and the per-body MPM grid of 6^3 nodes for 9,998,592 bodies is
    # 2,159,695,872 of them (mpm_scatter / gather[warp_pp] take it as one axis): 9,899,776 bodies need 2,138,351,616
    "b10m": dict(parent="it33", scale_to=9_900_000, coarse=dict(points=7, n_grid=6, cell=2), chunk=250, record_points=False, record_junctions=False, junction_buffer=40_000_000, warmup=400,
                 frames=1000, every=4, movie_fps=25, window=2.0, fast=dict(pp=True, strain=True, capture=False, ccwarp=True),
                 why="TWIN B of step 21 at 9,900,000 molecules (Warp holds at most 2^31 entries along an axis: the per-body grid of 10 million bodies is 2.16 billion), 7 material points each: it33's kinds, bonds and motors with every count x 26,053 in a slab of 31.0 x 17.4 um (52 nm "
                     "deep, 19 % of its volume), every body coarsened to 7 points of its kind's shape (4.3 nm apart, the volume of step 21's mean body) on its own "
                     "grid of 6 x 6 x 6 cells of 8.6 nm; 1,000 frames (200 us) on one H200. Drawn from the bodies' centres: the whole slab at left, at right a 192 x 108 nm window. Does "
                     "the crowd of ten million stay a crowd, and do bonds still form between 7-point bodies"),
    "b1m": dict(parent="b10m", scale_to=1_000_000,
                why="B at 1,000,000 molecules, its smoke: memory, time per frame and the health of 7-point bodies, before the 10 million run"),
    # the smokes of the two (2026-11-17 18:00): the whole chain -- chunks, the window's points and bonds, the live drawing -- on a scene 4 times step 21's (A) and of 20,000 bodies (B)
    "asmoke": dict(parent="a100k", scale_to=1520, frames=40, every=4, chunk=20, movie_fps=25, why="A's smoke: 1,520 bodies, 40 frames, a chunk every 20"),
    "bsmoke": dict(parent="b10m", scale_to=20_000, frames=40, every=4, chunk=20, warmup=10, why="B's smoke: 20,000 bodies of 7 points, 40 frames, a chunk every 20"),
    "bsmoke400": dict(parent="bsmoke", warmup=400, why="B's smoke with b10m's warm-up of 400 frames: is the overlap's motion run out by frame 0"),
    # mRNA AND PROTEINS (Cedric, 2026-11-19): 10 mRNA types, chains of 5 to 28 subunits of 6 nm (30-170 nm, 100 nm on average: 10 x a protein), 2 of each, meandering, held by
    # static springs; 10 protein types of 64 points (about 10 nm), 15 of each; binding sites of 2 subunits (12 nm), each holding one protein, a protein holding two sites;
    # the 20 x 20 affinity matrix of rna_affinity as the type matrix; a bond 10 x sturdier against stretch than step 21's (the isolation test: step 21's let go when stretched)
    "rna1": dict(parent="it33", walk=None, counts={f"p{k}": 15 for k in range(10)},
                 chains=[dict(kind=f"m{k}", n=2, m=int(round(5 + 2.55 * k)), bend=0.2) for k in reversed(range(10))], bonds=dict(k=4.0, reach=0.24),     # longest first: placed while the slab is empty
                 sites=dict(k=2),
                 binding=dict(matrix=rna_affinity(), valence={**{f"m{k}": 1 for k in range(10)}, **{f"p{k}": 2 for k in range(10)}},
                              on=1.0, off=0.005, stiffness=1.0e-3, strength=1.0e-2, rest=0.22, range=0.45, min_area=0.12),
                 fast=dict(pp=True, strain=True, capture=False),
                 why="mRNA AND PROTEINS: 10 mRNA types (chains of 5 to 28 subunits of 6 nm, 30 to 170 nm long, 2 of each) and 10 protein types (about 10 nm, ten shapes, 15 of "
                     "each) in step 21's slab, bound by a 20 x 20 affinity matrix (each protein type binds 3 mRNA types, 3 protein pairs bind each other, no mRNA binds an mRNA). "
                     "A binding site is 2 consecutive subunits: every node pair between a protein and a site's subunits makes ONE bond (NEW operator variant "
                     "contact_finder[sites]), at least 3 pairs for it to bind; a site holds one protein, a protein two sites; the springs pull on every node pair of a bond. Do "
                     "the proteins find their mRNAs and decorate them, and do proteins that hold two sites bridge mRNAs"),
    # rna1 WITH BINDING FREE ENERGIES (Cedric, 2026-11-19: "add a variant of the operator to take DeltaG input"): the affinity matrix read as Delta G of a pair in contact,
    # -2 kT (affinity 0.4) to -8 kT (1), each junction ONE bond drawn bound or not every frame (stochastic); rna1's sites, valences, chains and bath
    "rna2": dict(parent="rna1", matrix_panel=True, binding=dict(delta_g=rna_delta_g(), delta_g_unit="kT", stochastic=True, valence={**{f"m{k}": 1 for k in range(10)}, **{f"p{k}": 2 for k in range(10)}},
                                             on=1.0, stiffness=1.0e-3, strength=1.0e-2, rest=0.22, range=0.45, min_area=0.12),
                 why="THE mRNA SCENE WITH BINDING FREE ENERGIES (NEW operator variant bond_binding[delta_g]): step 24's strands, proteins, sites and valences, each pair of kinds "
                     "given a Delta G of a pair in contact, -2 kT (the weakest affinity, 0.4) to -8 kT (the strongest, 1); a junction's unbinding rate is its binding rate x "
                     "exp(Delta G / kT), and each junction is one bond, bound or not, drawn every frame. Do the weak pairs flicker and the strong ones stay, and does the "
                     "occupancy of each mRNA kind follow its partners' free energies"),
    # THE 3D TWIN of rna1 (Cedric, 2026-11-19: "can you run a twin in 3D scene, I'm afraid that we are constrained in Z"): rna1's slab is 52 nm deep and its longest strands
    # 168 nm, so a strand cannot turn out of the plane; the box 341 x 192 x 192 nm holds the longest strand any way it turns, at rna1's crowding (every count x 11.66, the
    # volume's ratio); the movie draws the 52 nm slice at mid-depth, as rna1's slab
    "rna1_3d": dict(parent="rna1", box=(34.1, 19.2, 19.2), counts={f"p{k}": 175 for k in range(10)},
                    chains=[dict(kind=f"m{k}", n=23, m=int(round(5 + 2.55 * k)), bend=0.2) for k in reversed(range(10))], trim_slots=True, section=(7.0, 12.2), panels3d=True,
                    why="THE 3D TWIN of the mRNA scene (step 24): the same kinds, affinities, sites and valences in a box 341 x 192 x 192 nm instead of a slab 52 nm deep, every count "
                        "x 11.7 for the same 14.5 % of the volume (230 strands, 1,750 proteins); at left the whole box from 30 degrees above, at right the 52 nm slice at mid-depth seen from above, its bodies whole. Are the mRNA and the proteins held "
                        "in z by the slab: do they move and bind alike when they can turn in three dimensions"),
    # THE PERIODIC TWIN of rna1_3d (Cedric, 2026-11-19: "can we run a twin of last step but with no boundary"; "Periodic box"): no walls, every axis periodic, a box of whole
    # lattice spacings (340 x 192 x 192 nm), the same counts (0.3 % more crowded)
    "rna1_3dp": dict(parent="rna1_3d", box=(34.0, 19.2, 19.2), margin=0.0, periodic=True, fast=dict(pp=True, strain=True, capture=False, ccwarp=True),
                     why="THE PERIODIC TWIN of step 25: the same 5,545 bodies in a box 340 x 192 x 192 nm with NO WALLS -- every face joined to the opposite one (the contact finder, "
                         "the contact penalty and the bonds' springs wrapping every axis; a body is never cut by a face, the movie draws each one shifted into the box); at left "
                         "the whole box from 30 degrees above, at right the 52 nm slice at mid-depth. Do the walls change how the mRNA and the proteins move and bind"),
    # ARE BONDED GROUPS SLOWED BY THE CROWD OR BY THE WALLS (Cedric, 2026-11-17 21:20, of step 21: "bodies that get into contact are not moving with brownian motion
    # anymore"; measured: over 40 frames a bonded body moves 0.3 to 0.6 of a single one of its kind, 1/sqrt(N) predicts 0.4 to 0.7): groups taken whole from step 21 at
    # frame 2,000, alone in an empty slab 52 nm deep (as step 21) and 156 nm deep
    "iso52": dict(parent="it33", box=(57.6, 32.4, 5.2), counts={}, walk=None,
                  seed_from=dict(run="exp09_land_it33x4", row=500, kinds=["dimer", "trimer", "clover"], per_bin=6, singles=10),
                  why="isolation test of step 21's bonded groups: up to 6 groups of each size (2, 3-4, 5-8, 9+) of dimers, three-lobed complexes and tetramers taken whole from "
                      "step 21 at frame 2,000, with 10 single bodies of each kind, alone in an empty slab 576 x 324 x 52 nm (step 21's depth): do the groups move as 1/sqrt(N)"),
    "iso156": dict(parent="iso52", box=(57.6, 32.4, 15.6),
                   why="iso52 in a slab 156 nm deep instead of 52: are step 21's groups held by the slab's two walls"),
    # iso52 / iso156 (2026-11-17 21:51): the groups fell apart out of the crowd -- bonds 82 at frame 0, 18 at frame 200 (52 nm slab), 12 (156 nm): step 21's Bell scale
    # 1e-3 lets any stretched bond go, and only the crowd's pressure keeps them unstretched. To measure a group's own motion, its bonds made permanent (unbinding rate 0)
    "iso52k": dict(parent="iso52", binding=dict(pairs={"dimer-dimer": 1.0, "trimer-trimer": 1.0, "clover-clover": 1.0}, valence={"dimer": 2, "trimer": 3, "clover": 4},
                                                on=1.0, off=0.0, stiffness=1.0e-3, strength=1.0e-3, rest=0.22, range=0.45, min_area=0.12),
                   why="iso52 with permanent bonds (unbinding rate 0): do step 21's groups, kept whole, move as 1/sqrt(N) out of the crowd"),
    "iso156k": dict(parent="iso52k", box=(57.6, 32.4, 15.6), why="iso52k in a slab 156 nm deep: the walls' part"),
    "bsfinal": dict(parent="b10m", scale_to=20_000, frames=40, every=4, chunk=20, why="B's last smoke: b10m's settings exactly at 20,000 bodies, 40 frames"),
    "bs_cgrid": dict(parent="bsmoke", warmup=0, coarse=dict(points=7, n_grid=6, cell=2, contact_dx="grid"),
                     why="bsmoke with the contact lattice at the grid's cell (8.6 nm), no warm-up: does the penalty act on 7-point bodies, and what moves them"),
    "bs_nocontact": dict(parent="bsmoke", warmup=0, rep=0.0, why="bsmoke with no overlap penalty, no warm-up: what moves the bodies without it"),
    # B's smoke moved its bodies 3.1 nm per stored row (4 frames), A's 0.26 (2026-11-17 17:40): which piece heats the 7-point bodies -- each variant ONE change from bsmoke
    "bs_nobond": dict(parent="bsmoke", binding=None, why="bsmoke without the bonds (no contact_finder, bond_binding, bond_springs)"),
    "bs_soft": dict(parent="bsmoke", rep=600.0 / 2.1419, why="bsmoke with the overlap stiffness x L instead of x L^2 (L = 2.14, the spacing over 2 nm)"),
    "bs_fine": dict(parent="bsmoke", coarse=dict(points=7, n_grid=7, cell=1), why="bsmoke on grids of cells of one point spacing, 7 x 7 x 7, instead of two"),
    "bs_cold": dict(parent="bsmoke", kT=0.0, why="bsmoke with no thermal kicks: what moves the bodies then comes from the forces"),
}


X4 = dict(frames=4000, every=4, movie_fps=100)


def known(key):
    return key in ITERS or (key.endswith("x4") and key[:-2] in ITERS)


def why_of(key):
    """The iteration's why text; for a key ending in `x4`, its base's with what x4 changes."""
    if key in ITERS:
        return ITERS[key]["why"]
    return (ITERS[key[:-2]]["why"] + " -- RUN x4 (Cedric, 2026-11-17: \"x4 longer, x4 fps\"): 4,000 frames (800 us) instead of 1,000, every 4th stored, the movie at 100 frames "
            "per second instead of 25: the same 10 s, four times the time")


def _params_raw(key):
    if key.endswith("x4") and key[:-2] in ITERS:
        p = _params_raw(key[:-2])
        p.update(X4)
        return p
    it = ITERS[key]
    p = _params_raw(it["parent"]) if it.get("parent") else dict(BASE)
    p.update({kk: v for kk, v in it.items() if kk not in ("why", "parent")})
    return p


def _scaled(p):
    """A scene with `scale_to`: every count times f = scale_to / (their sum), rounded by largest remainder so that they add up to scale_to exactly, and the slab's x and y
    times the square root of f (its depth kept): the bodies keep their share of the volume. `base_counts` and `base_box` keep the scene before."""
    if not p.get("scale_to"):
        return p
    p = dict(p)
    c0 = {k: int(v) for k, v in p["counts"].items()}
    f = float(p["scale_to"]) / sum(c0.values())
    raw = {k: v * f for k, v in c0.items()}
    got = {k: int(math.floor(v)) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - got[k], reverse=True)[:int(p["scale_to"]) - sum(got.values())]:
        got[k] += 1
    p.update(base_counts=c0, base_box=tuple(p["box"]), counts=got, scale_factor=f,
             box=(float(p["box"][0]) * math.sqrt(f), float(p["box"][1]) * math.sqrt(f), float(p["box"][2])))
    return p


def params(key):
    """BASE, then every ancestor's changes from the root down, then the iteration's own: an iteration names only its ONE change from its parent. A key ending in `x4` is the
    key before it run four times longer, its movie at four times the frame rate (X4). A scene with `scale_to` is scaled last (_scaled)."""
    return _scaled(_params_raw(key))


def spacing_of(p):
    """The point spacing of a scene, length units: SPACING (2 nm), or for a `coarse` scene the s with points x s^3 = the scene's mean body volume (SPECIES' point counts
    times SPACING^3, over its own counts)."""
    if not p.get("coarse"):
        return SPACING
    c = p.get("base_counts") or p["counts"]
    mean = sum(SPECIES[k]["n"] * int(v) for k, v in c.items()) * PV / sum(int(v) for v in c.values())
    return (mean / float(p["coarse"]["points"])) ** (1.0 / 3.0)


def run_name(key):
    return f"exp09_land_{key}"           # NOT exp09_<key>: graphs_data/tissue/exp09_p2* are the runs of an earlier experiment 9 (TRAPS T-90)


# ---- the bodies' points and their packing --------------------------------------------------------------------------------------------------------------------------
def body_points(shape, n, spacing=SPACING):
    """[n, 3] points of one body about its own centroid, length units: the `n` points of a lattice of spacing `spacing` (SPACING, 2 nm) with the lowest score of the
    shape (see SPECIES)."""
    kind, par = shape
    PV = spacing ** 3                                                                              # this lattice's point volume (shadows the module's, for a coarse body)
    g = np.arange(-9, 10) * spacing
    lat = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    if kind == "ellipsoid":
        score = np.linalg.norm(lat / np.asarray(par, float)[None], axis=1)
    elif kind == "lobes":
        cen = np.array([c for c, _ in par], float)
        rad = np.array([r for _, r in par], float)
        cen = cen - (cen * rad[:, None] ** 3).sum(0) / (rad ** 3).sum()                         # the lobes' volume-weighted centre at the lattice's origin
        s = (n * PV / (4.0 / 3.0 * np.pi * (rad ** 3).sum())) ** (1.0 / 3.0)                    # the first lobe's radius if the lobes did not overlap; they do, so the n points fill them a little further out
        score = (np.linalg.norm(lat[:, None] - s * cen[None], axis=2) / (s * rad[None])).min(1)
    elif kind == "torus":
        R, r = (float(v) for v in par)
        s = (n * PV / (2.0 * np.pi ** 2 * R * r * r)) ** (1.0 / 3.0)                               # the tube radius at which the ring's volume 2 pi^2 R r^2 holds the n points
        score = np.hypot(np.hypot(lat[:, 0], lat[:, 1]) - s * R, lat[:, 2]) / (s * r)
    elif kind == "stack":
        g = np.arange(-5, 6)
        ij = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        P = np.concatenate([np.column_stack([q[:, 0] * SPACING, np.full(len(q), -k * SPACING), q[:, 1] * SPACING])
                            for k, rho in enumerate(par) for q in [ij[(ij ** 2).sum(1) <= float(rho) ** 2 + 1e-9]]])
        if len(P) != n:
            raise ValueError(f"a stack of {par} has {len(P)} points, not {n}")
        return P - P.mean(0)
    elif kind == "block":                                                                          # a full lattice block of nx x ny x nz points
        ax = [(np.arange(k) - 0.5 * (k - 1)) * SPACING for k in par]
        P = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
        if len(P) != n:
            raise ValueError(f"a block of {par} points has {len(P)} points, not {n}")
        return P
    else:
        raise ValueError(f"unknown shape {kind!r}")
    P = lat[np.argsort(score, kind="stable")[:n]]
    return P - P.mean(0)


def membrane_bodies(mem, lo, hi):
    """A flat sheet across the slab at height `mem['y']` above its floor, normal to y: `lipid` patches side by side along x and z on ONE lattice of spacing SPACING
    (the last point of a patch and the first of the next are one spacing apart: the sheet is continuous material cut into bodies), one site in `mem['every']`
    holding a protein instead, upright through the sheet: a `channel`, or the kinds of `mem['proteins']` in turn. A kind with a `top` (SPECIES) is not centred on
    the sheet: its topmost point sits `top` above the sheet's middle plane, on the side `mem['heads']` ("above", the default, or "below": the body turned over).
    Returns [(species, points [n, 3] world)]."""
    tx, _, tz = SPECIES["lipid"]["shape"][1]
    px, pz = tx * SPACING, tz * SPACING
    nx, nz = int((hi[0] - lo[0] + 1e-9) // px), int((hi[2] - lo[2] + 1e-9) // pz)
    x0 = lo[0] + 0.5 * ((hi[0] - lo[0]) - nx * px) + 0.5 * px
    z0 = lo[2] + 0.5 * ((hi[2] - lo[2]) - nz * pz) + 0.5 * pz
    tile = body_points(SPECIES["lipid"]["shape"], SPECIES["lipid"]["n"])
    kinds = list(mem.get("proteins") or ["channel"])
    sign = -1.0 if mem.get("heads") == "below" else 1.0
    prot = {}
    for nm in kinds:
        P = body_points(SPECIES[nm]["shape"], SPECIES[nm]["n"])
        if "top" in SPECIES[nm]:
            P = P - np.array([0.0, P[:, 1].max() - float(SPECIES[nm]["top"]), 0.0])
            P = P * np.array([1.0, sign, 1.0])
        prot[nm] = P
    ev = int(mem.get("every") or 0)
    out, k = [], 0
    for ix in range(nx):
        for iz in range(nz):
            c = np.array([x0 + ix * px, lo[1] + float(mem["y"]), z0 + iz * pz])
            if ev > 0 and (ix + 3 * iz) % ev == ev // 2:
                out.append((kinds[k % len(kinds)], c + prot[kinds[k % len(kinds)]]))
                k += 1
            else:
                out.append(("lipid", c + tile))
    return out


def vesicle_bodies(ves, lo, hi):
    """A closed membrane: `lipid` patches round a cylinder whose axis is z (the viewing direction), so the picture shows a ring. Centre `ves['x']`, `ves['y']` from
    the slab's left wall and floor, radius `ves['r']` to the patches' middle; each patch has its thin side across the wall; along z the patches continue one lattice
    from wall to wall (as the flat sheet does), round the ring they are as many as fit with one point spacing between the innermost points of two neighbours.
    With `ves['every']` one site in that many holds a protein instead, the kinds of `ves['proteins']` in turn (default `receptor`), set radially: a kind with a
    `top` (SPECIES) has its topmost point `top` outside the wall's middle, its head out of the ring. Returns [(species, points [n, 3] world)]."""
    tx, ty, tz = SPECIES["lipid"]["shape"][1]
    pz = tz * SPACING
    nz = int((hi[2] - lo[2] + 1e-9) // pz)
    z0 = lo[2] + 0.5 * ((hi[2] - lo[2]) - nz * pz) + 0.5 * pz
    R = float(ves["r"])
    r_in = R - 0.5 * (ty - 1) * SPACING                                  # the radius of the innermost layer of points
    n = int(math.pi / math.asin(tx * SPACING / (2.0 * r_in)))            # chord between two neighbours' centres at r_in >= a patch's width + one spacing
    r_in = tx * SPACING / (2.0 * math.sin(math.pi / n))                  # ... and the ring taken in to where that chord is EXACTLY a patch's width + one spacing: the innermost points
    R = r_in + 0.5 * (ty - 1) * SPACING                                  # of two neighbours are then one spacing apart, within the bonds' reach (at r 3.0: 17 patches, the ring's middle at 2.92)
    tile = body_points(SPECIES["lipid"]["shape"], SPECIES["lipid"]["n"])
    kinds, ev, prot = list(ves.get("proteins") or ["receptor"]), int(ves.get("every") or 0), {}
    for nm in kinds:
        P = body_points(SPECIES[nm]["shape"], SPECIES[nm]["n"])
        prot[nm] = P - np.array([0.0, P[:, 1].max() - float(SPECIES[nm]["top"]), 0.0]) if "top" in SPECIES[nm] else P
    out, k = [], 0
    for i in range(n):
        th = 2.0 * math.pi * i / n
        t, r = np.array([-math.sin(th), math.cos(th), 0.0]), np.array([math.cos(th), math.sin(th), 0.0])
        for iz in range(nz):
            c = np.array([lo[0] + float(ves["x"]) + R * r[0], lo[1] + float(ves["y"]) + R * r[1], z0 + iz * pz])
            nm, B = "lipid", tile
            if ev > 0 and (i + 3 * iz) % ev == ev // 2:
                nm, B = kinds[k % len(kinds)], prot[kinds[k % len(kinds)]]
                k += 1
            out.append((nm, c + B[:, :1] * t + B[:, 1:2] * r + B[:, 2:3] * np.array([0.0, 0.0, 1.0])))
    return out


def _riders(grp, ch):
    """`ch['riders']` = dict(kind, every): a body of species `kind` seated on one subunit in `every` of the row `grp`, on alternate sides in the row's plane, its
    lowest layer of points one spacing from the subunit's outer layer and on the subunit's own lattice, so that `bonds` ties the two (a ribosome on a thread: the
    pair then moves as one). A rider that would come within 0.7 of a spacing of another body of the row is left out. Returns the row with its riders appended."""
    rd = ch.get("riders")
    if not rd:
        return grp
    dims = SPECIES[ch["kind"]]["shape"][1]
    Q0 = body_points(SPECIES[rd["kind"]]["shape"], SPECIES[rd["kind"]]["n"])
    cen = np.array([P.mean(0) for _, P in grp])
    out, side = list(grp), 1.0
    for k in range(int(rd["every"]) // 2, len(grp), int(rd["every"])):
        t = cen[min(k + 1, len(grp) - 1)] - cen[max(k - 1, 0)]
        t = t / np.linalg.norm(t)
        nrm = side * np.array([-t[1], t[0], 0.0])
        Q = Q0.copy()
        for ax, nd in ((0, dims[0]), (2, dims[2])):                    # along the row and along z: on the subunit's lattice (its points sit at multiples of the spacing, or half-way for an even count)
            want = 0.0 if nd % 2 else 0.5 * SPACING
            Q[:, ax] += want - (Q[0, ax] - round((Q[0, ax] - want) / SPACING) * SPACING)
        Q[:, 1] += 0.5 * (dims[1] - 1) * SPACING + SPACING - Q[:, 1].min()          # its lowest layer one spacing outside the subunit's outer layer
        X = cen[k] + Q[:, :1] * t + Q[:, 1:2] * nrm + Q[:, 2:3] * np.array([0.0, 0.0, 1.0])
        others = np.concatenate([P for j, (_, P) in enumerate(out) if j != k])
        if np.linalg.norm(X[:, None] - others[None], axis=2).min() < 0.7 * SPACING:          # 0.7: on a turning thread the next element's corner is already 0.17 from its neighbour
            continue
        out.append((rd["kind"], X))
        side = -side
    return out


def chain_groups(chains, seed):
    """Chains of bodies for `pack` to place, each as one rigid piece: for every dict(kind, n, m, bend) of `chains`, `n` rows of `m` bodies of the block species `kind`,
    each one lattice spacing from the next, so a row is continuous material cut into bodies (what holds it afterwards is the contact tension kind-kind alone).
    `bend` 0 (the default): a straight row along x on ONE lattice -- a stiff filament. `bend` > 0: a meander -- from one subunit to the next the row's heading in the
    xy plane turns by a random angle of standard deviation `bend`, radians (cut at 1.5 x bend), each subunit turned to face along the row, and the step widened by half
    a point spacing x that angle: at the largest turn of bend 0.2 (0.3 rad) the two facing layers of points are 0.17 apart on the inside of the turn and 0.29 on the
    outside (0.2 on a straight row). A wider step leaves the outside out of contact: the smoke of 2026-11-15 07:25 widened by half a block's width x the angle and one
    bond of 145 never closed. A row that comes back within a block and a half of itself is drawn again, and so is one taller than 0.8 of its `band`; the row is returned turned
    so that it runs along x from its first subunit to its last. Returns [[(species, points)]]."""
    rng = np.random.default_rng(int(seed) + 7919)
    out = []
    for ch in chains:
        kind, m, bend = ch["kind"], int(ch["m"]), float(ch.get("bend") or 0.0)
        sub = body_points(SPECIES[kind]["shape"], SPECIES[kind]["n"])
        w = SPECIES[kind]["shape"][1][0] * SPACING
        for _ in range(int(ch["n"])):
            if bend <= 0.0:
                out.append(_riders([(kind, sub + np.array([(k - 0.5 * (m - 1)) * w, 0.0, 0.0])) for k in range(m)], ch))
                continue
            for _try in range(3000):
                phi, c, grp = rng.uniform(-0.5, 0.5) * bend * math.sqrt(m), np.zeros(3), []
                for k in range(m):
                    if k:
                        d = float(np.clip(rng.normal(0.0, bend), -1.5 * bend, 1.5 * bend))
                        step = w + 0.5 * SPACING * abs(d)
                        c = c + 0.5 * step * np.array([math.cos(phi), math.sin(phi), 0.0])
                        phi += d
                        c = c + 0.5 * step * np.array([math.cos(phi), math.sin(phi), 0.0])
                    R = np.array([[math.cos(phi), -math.sin(phi), 0.0], [math.sin(phi), math.cos(phi), 0.0], [0.0, 0.0, 1.0]])
                    grp.append((kind, c + sub @ R.T))
                cen = np.array([P.mean(0) for _, P in grp])
                far = np.abs(np.arange(m)[:, None] - np.arange(m)[None]) >= 3                      # two subunits at least three apart along the row
                if (np.linalg.norm(cen[:, None] - cen[None], axis=2)[far] < 1.5 * w).any():        # ... closer than a block and a half: the row has come back on itself
                    continue                                                                       # (until 2026-11-15 08:30 this compared subunits two apart with 1.34 and never passed: it10's threads are the last of 200 draws)
                a = -math.atan2(cen[-1, 1] - cen[0, 1], cen[-1, 0] - cen[0, 0])                   # turned so that the row runs along x from its first subunit to its last
                T = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
                grp = [(nm, P @ T.T) for nm, P in grp]
                if ch.get("band") and np.ptp(np.concatenate([P for _, P in grp])[:, 1]) + 2.0 * SPACING > 0.8 * (float(ch["band"][1]) - float(ch["band"][0])):
                    continue                                                                       # a meander that could not lie inside its band: drawn again
                break
            else:
                raise ValueError(f"chain_groups: no meander of {m} `{kind}` at bend {bend} fits {ch.get('band') or 'clear of itself'} in 3,000 draws; lower `bend` or `m`")
            grp = _riders(grp, ch)
            mid = np.concatenate([P for _, P in grp]).mean(0)
            out.append([(nm, P - mid) for nm, P in grp])
    return out


def _rotation(rng):
    """A rotation matrix drawn uniformly over all rotations (a uniform unit quaternion)."""
    w, x, y, z = (lambda q: q / np.linalg.norm(q))(rng.normal(size=4))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pack(counts, lo, hi, gap, seed, tries=4000, fixed=(), yrange=None, groups=(), where=None, group_band=None, group_tilt=None, shapes=None, spacing=SPACING):
    """Random sequential placement: every body at a random place and orientation inside the box [lo, hi], largest first, kept when none of its points is within `gap`
    of a point already placed (tested on a 1 nm occupancy lattice). `fixed`: bodies placed beforehand as given (a membrane); `yrange`: {species: (y_lo, y_hi)}, the
    part of the box that species may lie in; `where`: {species: rule(centre, half extents) -> bool}, a further rule on where a body of that species may sit (inside
    or outside a vesicle); `groups`: chains of bodies placed after the fixed ones, each as ONE rigid piece turned about z by a random angle (filaments and threads, lying in the
    slab's plane), inside its species' `yrange` or, when `group_band[i]` = (y0, y1) is given for chain i, between those two heights above the box's floor; with `group_tilt[i]`
    the random angle is within that many radians of the x axis (either way round) instead of anywhere in the turn.
    Returns ([(species, points [n, 3] world)], {species: bodies that found no place}, how many of the bodies are fixed ones or chain subunits: the first of the list)."""
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    h = 0.1
    shape = np.ceil((hi - lo) / h).astype(int) + 1
    taken = np.zeros(shape, bool)
    r = int(math.ceil(gap / h))
    o = np.arange(-r, r + 1)
    off = np.stack(np.meshgrid(o, o, o, indexing="ij"), -1).reshape(-1, 3)
    off = off[np.linalg.norm(off, axis=1) * h <= gap + 1e-9]
    bp = shapes or {name: body_points(SPECIES[name]["shape"], SPECIES[name]["n"]) for name in counts}             # the same for every body of a species: drawn once
    todo = [(name, bp[name]) for name, c in counts.items() for _ in range(int(c))]
    order = sorted(range(len(todo)), key=lambda i: (-len(todo[i][1]), rng.random()))
    placed, failed = [], {}
    for name, X in fixed:
        st = np.clip((np.floor((X - lo) / h).astype(int)[:, None, :] + off[None]).reshape(-1, 3), 0, shape - 1)
        taken[st[:, 0], st[:, 1], st[:, 2]] = True
        placed.append((name, X))
    for gi, grp in enumerate(groups):
        glo, ghi = lo.copy(), hi.copy()
        if group_band and group_band[gi]:
            glo[1], ghi[1] = lo[1] + float(group_band[gi][0]), lo[1] + float(group_band[gi][1])
        elif yrange and grp[0][0] in yrange:
            glo[1], ghi[1] = max(lo[1], yrange[grp[0][0]][0]), min(hi[1], yrange[grp[0][0]][1])
        P0 = np.concatenate([P for _, P in grp])
        cuts = np.cumsum([len(P) for _, P in grp])[:-1]
        for _ in range(tries):
            ang = rng.uniform(0.0, 2.0 * math.pi) if not (group_tilt and group_tilt[gi] is not None) else rng.uniform(-1.0, 1.0) * float(group_tilt[gi]) + math.pi * int(rng.integers(2))
            P = P0 @ np.array([[math.cos(ang), -math.sin(ang), 0.0], [math.sin(ang), math.cos(ang), 0.0], [0.0, 0.0, 1.0]]).T
            e = np.abs(P).max(0) + 0.5 * spacing
            if np.any(ghi - e <= glo + e):
                continue
            c = rng.uniform(glo + e, ghi - e)
            if where and grp[0][0] in where and not where[grp[0][0]](c, e):
                continue
            X = c + P
            ijk = np.floor((X - lo) / h).astype(int)
            if taken[ijk[:, 0], ijk[:, 1], ijk[:, 2]].any():
                continue
            st = np.clip((ijk[:, None, :] + off[None]).reshape(-1, 3), 0, shape - 1)
            taken[st[:, 0], st[:, 1], st[:, 2]] = True
            placed += [(nm, Xk) for (nm, _), Xk in zip(grp, np.split(X, cuts))]
            break
        else:
            failed[grp[0][0] + " chain"] = failed.get(grp[0][0] + " chain", 0) + 1
    n_struct = len(placed)                                                                         # the membrane's bodies and the chains' subunits: what `bonds` may tie
    for i in order:
        name, P0 = todo[i]
        blo, bhi = lo.copy(), hi.copy()
        if yrange and name in yrange:
            blo[1], bhi[1] = max(lo[1], yrange[name][0]), min(hi[1], yrange[name][1])
        for _ in range(tries):
            P = P0 @ _rotation(rng).T
            e = np.abs(P).max(0) + 0.5 * spacing
            if np.any(bhi - e <= blo + e):
                continue
            c = rng.uniform(blo + e, bhi - e)
            if where and name in where and not where[name](c, e):
                continue
            X = c + P
            ijk = np.floor((X - lo) / h).astype(int)
            if taken[ijk[:, 0], ijk[:, 1], ijk[:, 2]].any():
                continue
            st = np.clip((ijk[:, None, :] + off[None]).reshape(-1, 3), 0, shape - 1)
            taken[st[:, 0], st[:, 1], st[:, 2]] = True
            placed.append((name, X))
            break
        else:
            failed[name] = failed.get(name, 0) + 1
    return placed, failed, n_struct


RPS_COLOURS = ("#5B2A86", "#21908C", "#FDE725")          # the three competitors: purple, teal, yellow (Cedric's choice for three chemicals, exp05, 2026-11-15: not red, green, blue)


def _rps_start(n, seed):
    """[n, 3] starting levels of the three competitors of a `signal` of model "rps": each body mostly one of them, drawn at random (0.6 of it, 0.05 of each other)."""
    rng = np.random.default_rng(seed + 31)
    c = np.full((n, 3), 0.05)
    c[np.arange(n), rng.integers(0, 3, n)] = 0.6
    return c


def seed_table(key):
    """The seed table of `key` for seed_positions[from_points_ragged] (pos in micrometres, NaN rows where a body has fewer than NMAX points; bodies in SPECIES
    order, the cell set's type order). Returns (path relative to the repo, {species: bodies placed}, the bodies' share of the slab's volume)."""
    p = params(key)
    m = float(p["margin"])
    lo, hi = np.full(3, m), np.full(3, m) + np.asarray(p["box"], float)
    if p.get("coarse"):
        return _seed_tiles(key, p, lo, hi)
    if p.get("seed_from"):
        return _seed_groups(key, p, lo, hi)
    fixed, yrange, where, groups, band, tilt = [], None, None, (), None, None
    region = p.get("region") or {}
    if p.get("membrane"):                                                  # one sheet (a dict) or several (a list, lowest first): "below" = under the lowest, "above" = over the highest,
        mems = p["membrane"] if isinstance(p["membrane"], (list, tuple)) else [p["membrane"]]      # "between" = between the first two
        for mem in mems:
            fixed += membrane_bodies(mem, lo, hi)
        half = 0.5 * SPECIES["lipid"]["shape"][1][1] * SPACING
        ys = [lo[1] + float(mem["y"]) for mem in mems]
        sides = {"below": (lo[1], ys[0] - half), "above": (ys[-1] + half, hi[1]), **({"between": (ys[0] + half, ys[1] - half)} if len(ys) > 1 else {})}
        yrange = {nm: sides[side] for nm, side in region.items() if side in sides}
    if p.get("vesicle"):                                                   # a species of `region` "lumen" sits inside the ring, every other species outside it
        v = p["vesicle"]
        fixed += vesicle_bodies(v, lo, hi)
        cx, cy, half = lo[0] + float(v["x"]), lo[1] + float(v["y"]), 0.5 * SPECIES["lipid"]["shape"][1][1] * SPACING
        inside = lambda c, e: math.hypot(c[0] - cx, c[1] - cy) + max(e[0], e[1]) <= float(v["r"]) - half
        outside = lambda c, e: (math.hypot(c[0] - cx, c[1] - cy) - max(e[0], e[1]) >= float(v["r"]) + half) or abs(c[0] - cx) - e[0] >= float(v["r"]) + half       # a chain beside the ring: its box is clear of it in x
        where = {nm: (inside if region.get(nm) == "lumen" else outside) for nm in list(p["counts"]) + [c["kind"] for c in (p.get("chains") or [])]}
    if p.get("chains"):
        for ch in p["chains"]:                                             # a chain with `rows`: straight along x, centred in x, one at each (height above the floor, depth)
            for (y, zz), grp in zip(ch.get("rows") or [], chain_groups([dict(ch, n=len(ch.get("rows") or []), bend=0.0)], p["seed"])):
                fixed += [(nm, np.array([0.5 * (lo[0] + hi[0]), lo[1] + float(y), lo[2] + float(zz)]) + P) for nm, P in grp]
        free = [ch for ch in p["chains"] if not ch.get("rows")]
        groups = chain_groups(free, p["seed"])
        band = [ch.get("band") for ch in free for _ in range(int(ch["n"]))]
        tilt = [ch.get("tilt") for ch in free for _ in range(int(ch["n"]))]
    placed, failed, n_struct = pack(p["counts"], lo, hi, float(p["gap"]), int(p["seed"]), fixed=fixed, yrange=yrange, groups=groups, where=where, group_band=band, group_tilt=tilt)
    if failed:
        print(f"  seed table of {key}: NO PLACE FOUND for {failed} (the slab is full at this gap); they are left out")
    names = [nm for nm in SPECIES if any(q[0] == nm for q in placed)]
    bodies = [X for nm in names for q, X in placed if q == nm]
    struct = [float(i < n_struct and nm not in (p.get("unbonded") or ())) for nm in names for i, (q, _) in enumerate(placed) if q == nm]   # 1 for a membrane body or a chain subunit, 0 for a body placed at random or of an `unbonded` kind
    got = {nm: sum(1 for q, _ in placed if q == nm) for nm in names}
    pos = np.full((len(bodies), nmax_of(p), 3), np.nan)
    for k, X in enumerate(bodies):
        pos[k, :len(X)] = X * UM
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, f"seed_{key}.npz")
    np.savez(out, pos=pos, volume_um3=np.array([len(X) * PV * UM ** 3 for X in bodies]), value=np.array(struct),     # `value`: written to the cell block `bonded` when the scene has bonds
             **({"kind": np.array([names.index(q) for q, _ in [(nm, None) for nm in names for _ in range(got[nm])]], np.int16)} if p.get("scale_to") else {}),
             cell__adh=np.ones((len(bodies), 6)),
             cell__valence=np.array([float(((p.get("binding") or {}).get("valence") or {}).get(nm, 0.0)) for nm in names for q, _ in placed if q == nm]),      # the cell block `valence` of a scene whose binding has one
             **({"cell__site": _sites_of(p, names, got)} if p.get("sites") else {}),
             **({"cell__chem": _rps_start(len(bodies), int(p["seed"]))} if (p.get("signal") or {}).get("model") == "rps" else {}),
             cell__grate=np.array([float(((p.get("grow") or {}).get("kinds") or {}).get(nm, 0.0)) for nm in names for q, _ in placed if q == nm]),      # the cell block `grate` of a scene with `grow`
             cell__drive=np.array([float(((p.get("push") or {}).get("types") or {}).get(nm, 0.0)) for nm in names for q, _ in placed if q == nm]))     # the cell block `drive` of a scene with `push`                    # `cell__adh`: the cell block `adh` (bond_binding's ceiling per face class) when the scene has binding; 1 everywhere
    share = sum(len(X) for X in bodies) * PV / float(np.prod(p["box"]))
    return os.path.relpath(out, ROOT), got, share


def _sites_of(p, names, got):
    """Each body's site, as a cell index: for a subunit of a chain (its kind a chain kind, the chains of a kind one after the other, a chain's subunits in order), the first
    subunit of its run of `sites.k`; for any other body, itself. Bodies in the seed table's order (kind by kind, SPECIES order)."""
    k = int(p["sites"]["k"])
    m_of = {ch["kind"]: int(ch["m"]) for ch in (p.get("chains") or [])}
    out, base = [], 0
    for nm in names:
        n = int(got[nm])
        for q in range(n):
            if nm in m_of:
                m = m_of[nm]
                out.append(base + (q // m) * m + ((q % m) // k) * k)
            else:
                out.append(base + q)
        base += n
    return np.array(out, np.float64)


def nmax_of(p):
    """Point slots per body: NMAX (the largest species, 343), or for a scaled scene the largest of its own species (it33's ring, 133): the slots of a body are allocated
    whether its points fill them or not, so a 100,000-body scene would carry 34 million slots instead of 13 million; a coarse scene's 7."""
    if p.get("coarse"):
        return int(p["coarse"]["points"])
    if p.get("scale_to") or p.get("trim_slots"):
        kinds = [k for k, v in p["counts"].items() if int(v) > 0] + [ch["kind"] for ch in (p.get("chains") or [])]
        return max(SPECIES[k]["n"] for k in kinds)
    return NMAX


def _seed_lattice(key, p, lo, hi):
    """The seed table of a `coarse` scene, built at once instead of body by body (pack is a Python loop over the bodies: hours at 10 million). The slab is cut into nx x ny x nz
    boxes, nz = round(depth / 13 nm) layers and nx : ny the slab's own ratio, with at least as many boxes as bodies; each body takes one box at random, its kind at random
    in the scene's proportions, its centre jittered inside the room its box leaves (keeping `gap` to its neighbours). A body is its centre and its 6 lattice neighbours,
    s apart (spacing_of), turned at random about z. Same columns as seed_table's (float32)."""
    s = spacing_of(p)
    n = int(sum(int(v) for v in p["counts"].values()))
    rng = np.random.default_rng(int(p["seed"]))
    ext = hi - lo
    nz = max(1, int(round(ext[2] * NM / 13.0)))
    per = math.ceil(n / nz)
    nx = math.ceil(math.sqrt(per * ext[0] / ext[1]))
    ny = math.ceil(per / nx)
    cell = ext / np.array([nx, ny, nz], float)
    half = 1.0 * s + 0.5 * s                                         # a body's half extent: an arm and half a spacing
    room = np.maximum(0.5 * cell - half - 0.5 * float(p["gap"]), 0.0)
    pick = rng.choice(nx * ny * nz, size=n, replace=False)
    ijk = np.stack(np.unravel_index(pick, (nx, ny, nz)), 1).astype(np.float64)
    cen = lo + (ijk + 0.5) * cell + rng.uniform(-1.0, 1.0, (n, 3)) * room
    names = [nm for nm in SPECIES if int(p["counts"].get(nm, 0)) > 0]
    kind = np.repeat(np.arange(len(names)), [int(p["counts"][nm]) for nm in names])
    cen = cen[rng.permutation(n)]                                    # bodies listed kind by kind (the cell set's type order), each at a random box
    star = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float) * s
    a = rng.uniform(0.0, 2.0 * math.pi, n)
    ca, sa = np.cos(a), np.sin(a)
    X = np.empty((n, 7, 3), np.float32)
    X[..., 0] = cen[:, None, 0] + ca[:, None] * star[None, :, 0] - sa[:, None] * star[None, :, 1]
    X[..., 1] = cen[:, None, 1] + sa[:, None] * star[None, :, 0] + ca[:, None] * star[None, :, 1]
    X[..., 2] = cen[:, None, 2] + star[None, :, 2]
    X *= UM
    val = (p.get("binding") or {}).get("valence") or {}
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, f"seed_{key}.npz")
    np.savez(out, pos=X, volume_um3=np.full(n, 7 * s ** 3 * UM ** 3, np.float32), value=np.zeros(n, np.float32), cell__adh=np.ones((n, 6), np.float32),
             cell__valence=np.array([float(val.get(nm, 0.0)) for nm in names], np.float32)[kind],
             cell__grate=np.zeros(n, np.float32), cell__drive=np.zeros(n, np.float32))
    got = {nm: int(p["counts"][nm]) for nm in names}
    share = n * 7 * s ** 3 / float(np.prod(p["box"]))
    print(f"  seed table of {key}: {n:,} bodies of 7 points {s * NM:.2f} nm apart in {nx} x {ny} x {nz} boxes of {cell[0] * NM:.1f} x {cell[1] * NM:.1f} x {cell[2] * NM:.1f} nm "
          f"(jitter up to {room[0] * NM:.1f}, {room[1] * NM:.1f}, {room[2] * NM:.1f} nm); {share * 100:.1f} % of the slab")
    return os.path.relpath(out, ROOT), got, share


def _seed_groups(key, p, lo, hi):
    """The seed table of a `seed_from` scene: at row `row` of run `run` (its trajectory.npz, points recorded), the groups of bodies its bonds tie (bound share above 0.5),
    up to `per_bin` of each size class 2, 3-4, 5-8, 9 and more for each kind of `kinds`, and `singles` bodies of each of those kinds holding no bond; each group moved
    whole (its points as they are, shifted, not turned) to its own place in the slab, placed largest first on the free places of a 2 nm occupancy lattice, `gap` kept
    between any two groups' points, z centred on the slab's middle. Same columns as seed_table's; `kind` too."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    sf = p["seed_from"]
    z = _load(os.path.join(GD, str(sf["run"]), "trajectory.npz"))
    r = int(sf["row"])
    nt = z["cell__node_type"]; nt = np.rint(nt[-1] if nt.ndim == 2 else nt).astype(int)
    tn = [str(x) for x in z["cell__type_names"]]
    nc = z["cell__occ"].shape[1]
    m = z["junction__occ"][r].astype(bool) & (z["junction__rho"][r][:, 0] > 0.5)
    a, b = np.rint(z["junction__ca"][r][m, 0]).astype(int), np.rint(z["junction__cb"][r][m, 0]).astype(int)
    ng, lab = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(nc, nc)), directed=False)
    size = np.bincount(lab)
    occ = z["pt__occ"][r].astype(bool)
    pid = np.rint(z["pt__pid"][r][:, 0]).astype(int)
    X = z["pt__pos"][r].astype(np.float64)
    pts = {c: X[occ & (pid == c)] for c in range(nc)}
    rng = np.random.default_rng(int(p["seed"]))
    chosen = []                                                                  # (kind name, [body ids])
    for kind in sf["kinds"]:
        k = tn.index(kind)
        gl = [g for g in range(ng) if nt[np.flatnonzero(lab == g)[0]] == k]
        for lo_, hi_ in ((2, 2), (3, 4), (5, 8), (9, 10 ** 6)):
            cand = [g for g in gl if lo_ <= size[g] <= hi_]
            for g in rng.permutation(cand)[:int(sf["per_bin"])]:
                chosen.append((kind, list(np.flatnonzero(lab == g))))
        singles = [g for g in gl if size[g] == 1]
        for g in rng.permutation(singles)[:int(sf["singles"])]:
            chosen.append((kind, list(np.flatnonzero(lab == g))))
    chosen.sort(key=lambda q: -len(q[1]))
    h = 0.2
    shp = np.ceil((hi - lo) / h).astype(int) + 1
    taken = np.zeros(shp[:2], bool)                                              # x-y occupancy: groups never overlap in projection
    g2 = int(np.ceil(float(p["gap"]) / h)) + 1
    placed = []
    zc = 0.5 * (lo[2] + hi[2])
    for kind, ids in chosen:
        P = np.concatenate([pts[c] for c in ids])
        c0 = P.mean(0)
        e = np.abs(P[:, :2] - c0[:2]).max(0) + 0.5 * SPACING
        for _ in range(4000):
            cxy = rng.uniform(lo[:2] + e, hi[:2] - e)
            Q = P - c0 + np.array([cxy[0], cxy[1], zc])
            ij = np.floor((Q[:, :2] - lo[:2]) / h).astype(int)
            if taken[ij[:, 0], ij[:, 1]].any():
                continue
            for di in range(-g2, g2 + 1):
                for dj in range(-g2, g2 + 1):
                    taken[np.clip(ij[:, 0] + di, 0, shp[0] - 1), np.clip(ij[:, 1] + dj, 0, shp[1] - 1)] = True
            placed.append((kind, [Q[np.flatnonzero(np.isin(np.repeat(ids, [len(pts[c]) for c in ids]), [c]))] for c in ids], len(ids)))
            break
        else:
            print(f"  seed table of {key}: no place for a group of {len(ids)} {kind}")
    names = [nm for nm in SPECIES if any(q[0] == nm for q in placed)]
    bodies, kinds, gsize = [], [], []
    for nm in names:
        for q, Xs, n in placed:
            if q == nm:
                bodies += Xs; kinds += [names.index(nm)] * len(Xs); gsize += [n] * len(Xs)
    nmax = nmax_of(dict(p, counts={nm: 1 for nm in names}))
    pos = np.full((len(bodies), nmax, 3), np.nan)
    for k_, Xb in enumerate(bodies):
        pos[k_, :len(Xb)] = Xb * UM
    got = {nm: int(sum(1 for q in kinds if q == names.index(nm))) for nm in names}
    val = (p.get("binding") or {}).get("valence") or {}
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, f"seed_{key}.npz")
    np.savez(out, pos=pos, kind=np.array(kinds, np.int16), group0=np.array(gsize, np.int32), volume_um3=np.array([len(Xb) * PV * UM ** 3 for Xb in bodies]),
             value=np.zeros(len(bodies)), cell__adh=np.ones((len(bodies), 6)), cell__valence=np.array([float(val.get(names[k_], 0.0)) for k_ in kinds]),
             cell__grate=np.zeros(len(bodies)), cell__drive=np.zeros(len(bodies)))
    share = sum(len(Xb) for Xb in bodies) * PV / float(np.prod(p["box"]))
    print(f"  seed table of {key}: {len(placed)} groups, {len(bodies)} bodies {got} from {sf['run']} row {r}; {share * 100:.1f} % of the slab")
    return os.path.relpath(out, ROOT), got, share


TILES = (16, 16)                       # a coarse scene's seed tiles across x and y (_seed_tiles)


def _seed_tiles(key, p, lo, hi, nt=TILES, variants=4):
    """The seed table of a `coarse` scene, packed as a scene of step 21 is (pack: random places and turns, `gap` between the points of two bodies) but by tiles: the slab
    is cut into nt[0] x nt[1] tiles across x and y (the full depth), `variants` tiles are packed (each its own seed, every kind at its share of the scene's counts,
    kept `gap` / 2 inside the tile's sides so that two tiles side by side keep the gap), and every tile of the slab is one of them, mirrored in x and / or y at random.
    pack is a Python loop over the bodies: 10 million of them one by one would take about five hours, four tiles of 39,000 a few minutes. Every body is its kind's shape
    at 7 points (body_points with the spacing s of spacing_of). Same columns as seed_table's (float32). The count is nt[0] x nt[1] x the tile's, at most the scene's."""
    s = spacing_of(p)
    npts = int(p["coarse"]["points"])
    T = int(nt[0]) * int(nt[1])
    names = [nm for nm in SPECIES if int(p["counts"].get(nm, 0)) > 0]
    per = {nm: int(p["counts"][nm]) // T for nm in names}
    shapes = {nm: body_points(SPECIES[nm]["shape"], npts, spacing=s) for nm in names}
    ext = hi - lo
    tw = np.array([ext[0] / nt[0], ext[1] / nt[1], ext[2]])
    # the gap between two bodies' points is step 21's 3 nm, NOT scaled with the spacing: at 6.4 nm the tiles do not pack at 19 % in 30 minutes (2026-11-17 18:40). The bodies
    # start overlapping by part of a point's width and the overlap penalty drives them apart; that motion decays with the bath's friction (bsmoke: 4.3 -> 2.4 nm per stored
    # row of 4 frames over 40 frames, a decay time of about 70 frames, the same with no thermal kicks), so a coarse scene's warm-up runs it out before frame 0 (b10m `warmup`)
    gap = float(p["gap"])
    g2 = 0.5 * gap
    tiles = []
    for v in range(int(variants)):
        placed, failed, _ = pack(per, np.array([g2, g2, 0.0]), tw - np.array([g2, g2, 0.0]), gap, int(p["seed"]) + 1000 * (v + 1), shapes=shapes, spacing=s)
        if failed:
            print(f"  seed table of {key}: tile {v}: NO PLACE FOUND for {failed}")
        tiles.append({nm: np.stack([X for q, X in placed if q == nm]) if any(q == nm for q, _ in placed) else np.zeros((0, npts, 3)) for nm in names})
    rng = np.random.default_rng(int(p["seed"]))
    pick = rng.integers(0, int(variants), T)
    flip = rng.integers(0, 2, (T, 2))
    out_k = {nm: [] for nm in names}
    for t in range(T):
        ix, iy = t % nt[0], t // nt[0]
        o = lo + np.array([ix * tw[0], iy * tw[1], 0.0])
        for nm in names:
            X = tiles[pick[t]][nm].copy()
            if flip[t, 0]:
                X[..., 0] = tw[0] - X[..., 0]
            if flip[t, 1]:
                X[..., 1] = tw[1] - X[..., 1]
            out_k[nm].append((X + o).astype(np.float32))
    got = {nm: int(sum(len(a) for a in out_k[nm])) for nm in names}
    pos = np.concatenate([np.concatenate(out_k[nm]) for nm in names]) * UM                         # bodies kind by kind: the cell set's type order
    n = len(pos)
    kind = np.repeat(np.arange(len(names)), [got[nm] for nm in names])
    val = (p.get("binding") or {}).get("valence") or {}
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, f"seed_{key}.npz")
    np.savez(out, pos=pos.astype(np.float32), kind=kind.astype(np.int16), volume_um3=np.full(n, npts * s ** 3 * UM ** 3, np.float32), value=np.zeros(n, np.float32), cell__adh=np.ones((n, 6), np.float32),
             cell__valence=np.array([float(val.get(nm, 0.0)) for nm in names], np.float32)[kind], cell__grate=np.zeros(n, np.float32), cell__drive=np.zeros(n, np.float32))
    share = n * npts * s ** 3 / float(np.prod(p["box"]))
    print(f"  seed table of {key}: {n:,} bodies of {npts} points {s * NM:.2f} nm apart, {T} tiles of {tw[0] * NM:,.0f} x {tw[1] * NM:,.0f} nm from {variants} packed ones; {share * 100:.1f} % of the slab")
    return os.path.relpath(out, ROOT), got, share


def build(key):
    p = params(key)
    name = run_name(key)
    table, got, share = seed_table(key)
    names = list(got)
    n = int(sum(got.values()))
    m = float(p["margin"])
    box = [float(v) for v in p["box"]]
    W = [b + 2.0 * m for b in box]
    g = np.zeros((len(names), len(names)))
    for pair, v in (p.get("ctens") or {}).items():
        a, b = pair.split("-")
        if a in names and b in names:
            g[names.index(a), names.index(b)] = g[names.index(b), names.index(a)] = float(v)
    sub = float(p["substep"])
    sp = spacing_of(p)                                              # the point spacing: 0.2 (2 nm), or a coarse scene's own
    L = sp / SPACING                                                # every length tied to the spacing times L, every area times L^2 (1 for every scene but a coarse one)
    ngrid = int((p.get("coarse") or {}).get("n_grid", 15))
    gcell = sp * float((p.get("coarse") or {}).get("cell", 1))     # the grid's cell: the point spacing, or a coarse scene's `cell` of them
    # the contact lattice: the point spacing (L = sp / 0.2), or with coarse.contact_dx "grid" the grid's cell, where a 7-point body's volume fraction nears 1 (at the point
    # spacing it peaks at 0.42, so two overlapping bodies stay under the 0.2 the penalty tolerates: bsmoke, 2026-11-17 20:55); the overlap stiffness x (lattice / 0.2)^2
    cgrid = (p.get("coarse") or {}).get("contact_dx") == "grid"
    Lc = (gcell if cgrid else sp) / SPACING
    bd = p.get("binding")
    if bd and L != 1.0:                                             # a coarse body's junction: lengths x L, the contact area x L^2; the spring x L and its Bell scale x L^2, so the pull
        bd = dict(bd, range=float(bd["range"]) * L, rest=float(bd["rest"]) * L, min_area=float(bd["min_area"]) * L * L,     # between two bodies (point pairs ~ 1 / L^2, stretch ~ L) and
                  stiffness=float(bd["stiffness"]) * L, strength=float(bd["strength"]) * L * L)                             # the load per pair keep their size
    sg = p.get("signal")
    gw = p.get("grow")
    tm = np.zeros((len(names), len(names)))                         # bond_binding's type matrix: the share of the ceiling a pair of kinds may bind; 0 = that pair never binds
    if (bd or {}).get("matrix"):                                    # an affinity matrix over named kinds: its entries for the kinds the scene holds
        mn, mv = bd["matrix"]
        for a_ in names:
            for b_ in names:
                if a_ in mn and b_ in mn:
                    tm[names.index(a_), names.index(b_)] = float(mv[mn.index(a_)][mn.index(b_)])
    for pair, v in ((bd or {}).get("pairs") or {}).items():
        if pair == "*":                                             # every pair of kinds
            tm[:] = float(v)
            continue
        a, b = pair.split("-")
        if a in names and b in names:
            tm[names.index(a), names.index(b)] = tm[names.index(b), names.index(a)] = float(v)
    return {
        "general": {
            "name": name,
            "headline": f"exp09 {key}: a crowded molecular landscape of {n:,} MPM bodies ({share * 100:.0f} % of a slab {box[0] * NM:,.0f} x {box[1] * NM:,.0f} x {box[2] * NM:.0f} nm)",
            "title": why_of(key),
            "seed": int(p["seed"]), "n_frames": int(p["frames"]), "dt": 1.0, "record_cap": int(p["frames"]) // int(p["every"]) + 1,
            **({"warmup": int(p["warmup"])} if p.get("warmup") else {}),
            **({"save_chunk_window": {"set": "pt", "box": [[float(v) for v in big_window(key)[:2]], [float(v) for v in big_window(key)[2:]]], "margin": 1.5,
                                      **({"edges": {"set": "junction", "a": "ca", "b": "cb", "keep": ["rho"]}} if bd else {})}} if p.get("window") and p.get("chunk") else {}),
            **({"save_chunk_frames": int(p["chunk"])} if p.get("chunk") else
               {"save_every_min": 10.0}),  # the rows recorded so far written to trajectory.npz every 10 wall-clock minutes, so a running job can be previewed (Cedric, 2026-11-14)
            "boundary": "free", "dim": 3, "world": W, "units": {"length_um": UM, "time_s": TIME_US * 1e-6},
        },
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": ngrid, "per_parent": "cell", "cell_box": round(ngrid * gcell, 6)}},     # each body's own grid
        "sets": {
            "cell": {
                "n": n, **({"buffer": n + int(gw.get("room", 300))} if gw else {}), "type_layout": "ordered",
                "types": {nm: {"count": int(got[nm]), "shape": "ball", "material": "elastic", "youngs": float((p.get("youngs_of") or {}).get(nm, p["youngs"])), "density": 1.0} for nm in names},
                "state": {
                    "pos": {"width": 3, "role": "coordinate", "integration": "coordinate", "boundary": "world"},
                    "vel": {"width": 3, "role": "rate", "integration": "rate", "record": False},
                    **({"bonded": {"width": 1, "record": False}} if p.get("bonds") else {}),
                    **({"adh": {"width": 6, "record": False}} if p.get("binding") else {}),
                    **({"valence": {"width": 1, "record": False}} if (p.get("binding") or {}).get("valence") else {}),
                    **({"site": {"width": 1, "record": False}} if p.get("sites") else {}),
                    **({"chem": {"width": 3 if sg.get("model") == "rps" else 2, "integration": "first_order"}} if sg else {}),
                    **({"drive": {"width": 1, "record": False}} if p.get("push") and not p["push"].get("block") else {}),
                    **({"mass": {"width": 1}, "mass0": {"width": 1, "record": False}, "born": {"width": 1, "record": False}, "div_ready": {"width": 1, "record": False},
                        "grate": {"width": 1, "record": False}} if gw else {}),
                    **({"crowd": {"width": 1, "integration": "first_order"}} if gw and gw.get("inhibit") else {}),
                },
            },
            "pt": {
                "entity": "mpm_particle", "parent": "cell", "per_parent": nmax_of(p), "density": 1.0, "radius": 0.5, **({} if p.get("record_points", True) else {"record": False}),
                "state": {
                    "pos": {"width": 3, "role": "coordinate", "integration": "coordinate", "boundary": "world"},
                    "vel": {"width": 3, "role": "rate", "integration": "rate", "record": False},
                    "pid": {"width": 1}, "cellc": {"width": 1, "record": False},
                    **({"face": {"width": 1, "record": False}} if p.get("binding") else {}),
                },
            },
            **({"junction": {"n": 0, "buffer": int(p.get("junction_buffer", 16384)), **({} if p.get("record_junctions", True) else {"record": False}), "state": {"ca": {"width": 1, "integration": "none"}, "cb": {"width": 1, "integration": "none"}, "fg": {"width": 1, "integration": "none", "record": False},
                                                              "rho": {"width": 1, "integration": "first_order"}, "area": {"width": 1, "integration": "none"},
                                                              "load": {"width": 1, "integration": "none"}}}} if p.get("binding") else {}),
        },
        "operators": [
            {"op": "thermal_bath", "implementation": "langevin", "at": "pt", "friction": float(p["friction"]), "kT": float(p["kT"]), "substep_dt": sub, "seed": int(p["seed"])},
            *([] if p.get("periodic") else [{"op": "active_force", "model": "box_slab", "at": "pt", "wall_stiffness": float(p["wall"]), "lo": [m, m, m], "hi": [m + b for b in box]}]),
            # TWO operators named active_force, both inside the substep block, which therefore names `active_force` twice: at each substep the block's i-th entry of a
            # name runs the i-th declared instance (engine._run_scheduled; the block clears the count before every substep). Named once, only the first instance ran: in
            # the smoke of 2026-11-16 23:25 the walk was declared first, the walls never ran and every kind left the slab (up to 24 nm in 150 frames). The walk reads the
            # FRAME's dt (1.0) at every call, so its rotational diffusion is given per call: Dr x substep / dt, or the direction would be lost 100x too fast (that smoke
            # again: the motors moved no further than the passive bodies)
            *([{"op": "active_force", "model": "persistent_walk", "at": "pt", "cell_set": "cell", "f": 0.0, "Dr": float(p["walk"]["Dr"]) * sub, "seed": int(p["seed"]),
                "type_f": {nm: float(v) for nm, v in p["walk"]["types"].items() if nm in names}}] if p.get("walk") else []),
            # `push.track` = a chain kind: active_force[track_push] (NEW variant, mpm_ops.py, 2026-11-17) -- contact_push whose motor takes, while it touches that kind, the
            # chain's axis as its direction (one of its two senses, the motor's own): it runs along the filament instead of across it (it23: 7 of 25 motors left on a filament)
            *([{"op": "active_force", "model": "track_push" if p["push"].get("track") else "contact_push", "at": "pt", "cell_set": "cell", "f": 0.0, "f_block": str(p["push"].get("block") or "drive"),
                "Dr": float(p["push"]["Dr"]) * sub, "seed": int(p["seed"]), "contact_gate": "junction", "gate_area": float(p["push"]["gate_area"]),
                **({"track": str(p["push"]["track"]), "track_reach": 1.6 * SPECIES[p["push"]["track"]]["shape"][1][0] * SPACING} if p["push"].get("track") else {})}] if p.get("push") else []),
            # a coarse scene: the overlap stiffness x L^2 -- the penalty is on volume fractions over a lattice of the grid's cell, so for the same overlap in length its pull per
            # unit mass falls as 1 / L^2 (the tolerated overlap, a product of two volume fractions, has no length)
            # a periodic box: the Warp contact penalty, the only one whose lattice wraps on every axis
            {"op": "pair_potential", **({"implementation": "cell_contact_warp_pp", "periodic": True} if p.get("periodic") else {"model": "cell_contact"}), "at": "pt", "cells": "cell",
             "repulsion_strength": float(p["rep"]) * Lc * Lc, "contact_slack": float(p["slack"]),
             "contact_surface_tension": [[float(v) for v in row] for row in g], "contact_edge_width": 0.2, **({"dx": sp} if gcell != sp and not cgrid else {})},     # the lattice at the point spacing, or the grid's cell
            *([{"op": "apical_junction", "at": "pt", "cells": "cell", "k": float(p["bonds"]["k"]), "reach": float(p["bonds"]["reach"]), "shell": 1.0e9, "skip_block": "bonded"}] if p.get("bonds") else []),
            *([{"op": "contact_finder", "at": "junction", "points": "pt", "cells": "cell", "spacing": sp, "bond_range": float(bd["range"]),
                **({"implementation": "sites", "site_block": "site"} if p.get("sites") else {"implementation": "hashgrid"}), **({"periodic": True} if p.get("periodic") else {})},
               # `binding.delta_g` = (kind names, matrix of binding free energies, None = never binds): bond_binding[delta_g] (NEW variant, binding_delta_g_ops.py, 2026-11-19) --
               # the unbinding rate of a junction is binding_rate x exp(Delta G / kT) of its two kinds, in `delta_g_unit` (kcal/mol) at `temperature_K` (310.15); no `off`, no type matrix
               ({"op": "bond_binding", "implementation": "delta_g", "at": "junction", "cells": "cell", "binding_rate": float(bd["on"]), "bond_strength": float(bd["strength"]),
                 "bell": "area", "min_area": float(bd["min_area"]),
                 "delta_g": [[(None if (a_ not in bd["delta_g"][0] or b_ not in bd["delta_g"][0] or bd["delta_g"][1][bd["delta_g"][0].index(a_)][bd["delta_g"][0].index(b_)] is None)
                               else float(bd["delta_g"][1][bd["delta_g"][0].index(a_)][bd["delta_g"][0].index(b_)])) for b_ in names] for a_ in names],
                 "delta_g_unit": str(bd.get("delta_g_unit", "kcal/mol")), "temperature_K": float(bd.get("temperature_K", 310.15)),
                 **({"delta_g_reference": "standard", "binding_volume_nm3": float(bd["binding_volume_nm3"])} if bd.get("binding_volume_nm3") else {}),
                 **({"stochastic": True, "seed": int(p["seed"])} if bd.get("stochastic") else {}),
                 **({"valence_block": "valence"} if bd.get("valence") else {})} if bd.get("delta_g") else
               {"op": "bond_binding", "at": "junction", "cells": "cell", "binding_rate": float(bd["on"]), "unbinding_rate": float(bd["off"]), "bond_strength": float(bd["strength"]),
                "bell": "area", "min_area": float(bd["min_area"]), "type_matrix": [[float(v) for v in row] for row in tm],
                **({"binding_rate_block": str(bd["on_block"])} if bd.get("on_block") else {}),
                # `binding.valence` = {species: number}: bond_binding[valence] (NEW variant, adhesion_ops.py, 2026-11-17) -- a body holds at most that many bonds (a species not
                # named: none); the largest contacts are admitted first, a held bond is kept
                **({"implementation": "valence", "valence_block": "valence"} if bd.get("valence") else {})}),
               {"op": "bond_springs", "at": "pt", "junctions": "junction", "bond_stiffness": float(bd["stiffness"]), "bond_damping": 0.0, "spacing": sp,
                "bond_rest_length": float(bd["rest"]), "bond_range": float(bd["range"])}] if bd else []),
            *([{"op": "cell_neighbours", "model": "point_contact_warp", "at": "cell", "points": "pt", "contact": float(sg["contact"])},
               # the bodies that are made to fire: those of the kinds `pace` (the paced implementation), or, with `strip` = a width in length units, every body whose centre
               # is within that distance of the slab's left wall at the moment of a firing (the model's own region stimulus): one source, so the excitation crosses the frame as a front
               {"op": "cell_chem_react", **({"model": "aliev_panfilov"} if sg.get("strip") else {"implementation": "aliev_panfilov_paced"}), "at": "cell",
                "k": 8.0, "a": 0.15, "eps0": 0.002, "mu1": 0.2, "mu2": 0.3, "rate": float(sg["rate"]),
                "stim": {"times": [float(t) for t in sg["times"]], "duration": float(sg["duration"]), "amp": float(sg["amp"]),
                         **({"box": [[m, m, m], [m + float(sg["strip"]), m + box[1], m + box[2]]]} if sg.get("strip") else {"types": [nm for nm in sg["pace"] if nm in names]})}},
               {"op": "cell_chem_diffuse", "model": "contact_weighted", "at": "cell", "d": [float(sg["D"]), 0.0], "chi": 1.0}] if sg and sg.get("model") != "rps" else []),
            # `signal.model` "rps": cyclic competition instead of an excitation. Every body carries three competitors u, v, w (cell block `chem`, width 3; its start from the seed
            # table's `cell__chem`: mostly one of them); cell_chem_react[rock_paper_scissor] (May-Leonard): each grows toward a shared ceiling and is suppressed by the next,
            # u by v, v by w, w by u, with strength `a`; cell_chem_diffuse[contact_weighted] passes all three between touching bodies at rate `D`
            *([{"op": "cell_neighbours", "model": "point_contact_warp", "at": "cell", "points": "pt", "contact": float(sg["contact"])},
               {"op": "cell_chem_react", "model": "rock_paper_scissor", "at": "cell", "a": float(sg["a"]), "rate": float(sg["rate"])},
               {"op": "cell_chem_diffuse", "model": "contact_weighted", "at": "cell", "d": [float(sg["D"])] * 3, "chi": 1.0}] if sg and sg.get("model") == "rps" else []),
            {"op": "mpm_strain", "at": "pt", "implementation": "default"},
            {"op": "mpm_scatter", "at": "mpm_grid", "from": "pt", "drag": 0.0, "implementation": "default"},      # no grid drag: the bath's friction is the only drag, so the kicks and the drag balance at kT
            # NO cell_surface_tension here (smoke of 2026-11-14 21:05, tension 0.3 on every body with exp06's csf settings): the bodies shrink without end, the smallest
            # fastest (radius of gyration at frame 60: 27 points 0.73 of its start, 64 points 0.89, 125 points 0.94-0.96) -- at one point per grid cell the points of a
            # body can move together inside a cell without the grid's strain seeing it (TRAPS)
            {"op": "mpm_grid_update", "at": "mpm_grid", "implementation": "warp_pp_balanced"},
            {"op": "mpm_gather", "at": "pt", "from": "mpm_grid", "implementation": "default"},
            {"op": "aggregate_centroid", "at": "cell", "child": "pt"},
            *([{"op": "cell_cycle", "model": "mpm", "at": "cell", "rule": "doubler_mass", "factor": float(gw.get("factor", 2.0)), "factor_cv": 0.1, "mark": "div_ready", "born": "born", "seed": int(p["seed"])},
               # `grow.on_signal` = dict(a_sw, hill): cell_grow[mpm_activator] -- the same law times a Hill function of the body's excitation (half at a_sw): it grows only while excited.
               # `grow.inhibit` = dict(ceiling, gate_area, tau): cell_grow[mpm_adhesion] -- the same law, switched off while the body's CROWDING is at or above `ceiling`; the
               # crowding (cell block `crowd`) follows, with the time constant `tau` frames, the body's contact area with every other body (contact_finder's junction rows, so
               # `binding` must be on) over `gate_area`, the contact area of a body that is surrounded (2.0 length^2 for a 64-point body): contact inhibition of growth
               {"op": "cell_grow", "model": "mpm_activator" if gw.get("on_signal") else "mpm_adhesion" if gw.get("inhibit") else "mpm", "at": "cell", "points": "pt", "rate": 0.0, "rate_block": "grate",
                "dims": 3.0, "rho": 1.0,
                **({"chem_block": "chem", "channel": 0, "a_sw": float(gw["on_signal"]["a_sw"]), "hill": float(gw["on_signal"].get("hill", 4.0)), "rho_lam": 0.0, "lam_gain": 1.0} if gw.get("on_signal") else {}),
                **({"contact_gate": "junction", "gate_area": float(gw["inhibit"]["gate_area"]), "tau": float(gw["inhibit"]["tau"]), "threshold": 0.0, "ceiling": float(gw["inhibit"]["ceiling"]),
                    "adh_block": "crowd"} if gw.get("inhibit") else {})},
               # cell_populate[mpm_interior] (NEW variant, gastrulation_ops.py, 2026-11-17), not [mpm]: [mpm]'s `fill` seats new points on the outside of a body that is not
               # round, which with `swell` grows it twice -- it24's first run: the replicators at 0.46 of their density after four generations, drawn as fragments
               {"op": "cell_populate", "model": "mpm_interior", "at": "cell", "points": "pt", "jitter": 0.3, "swell": True, "seed": int(p["seed"])},
               {"op": "cell_divide", "model": "cleave_typed", "at": "cell", "points": "pt", "yolk_weight": 0.0, "p_div": 1.0, "max_per_call": 64, "mark": "div_ready", "daughter_mark": "born",
                "inherit": ["grate", *(["crowd"] if gw.get("inhibit") else []), *(["adh"] if bd else []), *(["chem"] if sg else []), *(["drive"] if p.get("push") else [])], "gap": 0.1, "seed": int(p["seed"])}] if gw else []),
        ],
        "schedule": [
            *(["contact_finder", "bond_binding"] if bd else []),
            *(["cell_neighbours", "cell_chem_react", "cell_chem_diffuse"] if sg else []),
            {"steps": ["thermal_bath", *([] if p.get("periodic") else ["active_force"]), *(["active_force"] if p.get("walk") else []), *(["active_force"] if p.get("push") else []), "pair_potential", *(["bond_springs"] if bd else []), *(["apical_junction"] if p.get("bonds") else []), "mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"],
             "substep_dt": sub, "capture": False},
            "aggregate_centroid",
            *(["cell_cycle", "cell_grow", "cell_populate", "cell_divide"] if gw else []),
        ],
        "seed": [{"op": "seed_positions", "model": "from_points_ragged_batched" if p.get("coarse") else "from_points_ragged", "at": "pt", "cell_set": "cell", "points": table, "um_per_unit": UM,
                  "centre_um": [0.0, 0.0, 0.0], "centre": [0.0, 0.0, 0.0], **({"value_block": "bonded"} if p.get("bonds") else {})}],
        "plotting": {
            "renderer": "vtk_points", "subject": "pt", "render_3d": "contour",
            "point_volume": round(sp ** 3, 8) if p.get("coarse") else PV,   # the seed names no point_spacing: without this the studio takes 0.125^3 and draws slivers (TRAPS T-100); a coarse scene's own
            # a coarse body's 7 points blurred (exp23_studio._cell_surface, 1.6 voxels of 0.55 point spacings) reach a volume fraction of 0.39 at most, below the 0.5 surface of a
            # dense body: at 0.2 the surface holds all 7 (an arm point is at 0.25)
            # and at 0.9 voxels instead of 1.6 each point stays a lobe (a sphere of 0.69 point spacings at 0.2), so a ring keeps its hole and a dimer its two lobes
            **({"surface_iso_frac": 0.2, "surface_blur": 0.9} if p.get("coarse") else {}),
            "cell_colours": "type", "colors": {nm: SPECIES[nm]["colour"] for nm in names},
            "surface_opacity": 1.0, "background": "#8d99a6", "foreground": "white", "box_frame": False,
            "camera": {"elev": 90, "azim": -90}, "zoom": 1.0, "duration_s": 10, "slow_motion": 1,
            # read by this tool's `render`: the fill of a body mixed toward one colour by its excitation, or (three channels) replaced by the mix of three colours in the shares of its three competitors
            **({"bond_lines": True} if (bd or {}).get("valence") else {}),          # read by this tool's `render`: a white line between the centres of every two bodies that hold a bond
            **({"section_z": [m + float(p["section"][0]), m + float(p["section"][1])]} if p.get("section") else {}),    # read by `render`: the bodies drawn lie in this z range
            **({"periodic": True} if p.get("periodic") else {}),          # read by `render`: each body drawn shifted into the box by whole periods (its centre inside)
            **({"glow": ({"block": "chem", "channels": [0, 1, 2], "colours": list(RPS_COLOURS)} if sg.get("model") == "rps" else {"block": "chem", "channel": 0, "colour": "#FFF6B0"})} if sg else {}),
        },
    }


def _fast(sp, fast):
    """The spec `sp` with the operator variants of `fast` (see BASE.fast), in place."""
    for o in sp["operators"]:
        if fast.get("pp") and o["op"] in ("mpm_scatter", "mpm_gather"):
            o["implementation"] = "warp_pp"
        if fast.get("strain") and o["op"] == "mpm_strain":
            o["implementation"] = "warp"
        if fast.get("touched") and o["op"] == "pair_potential" and o.get("model") == "cell_contact":
            o.pop("model")
            o["implementation"] = "cell_contact_touched_nodes"
        if fast.get("ccwarp") and o["op"] == "pair_potential" and o.get("model") == "cell_contact":
            o.pop("model")
            o["implementation"] = "cell_contact_warp_pp"
    for st in sp["schedule"]:
        if isinstance(st, dict) and "substep_dt" in st and "capture" in fast:
            st["capture"] = bool(fast["capture"])
    return sp


def write(key):
    out = os.path.join(ROOT, "config", "tissue", run_name(key) + ".yaml")
    sp = build(key)
    if params(key).get("fast"):
        _fast(sp, params(key)["fast"])
    yaml.safe_dump(sp, open(out, "w"), sort_keys=False, width=1000)
    print("wrote", os.path.relpath(out, ROOT))
    return out


# ---- the numbers of a run -----------------------------------------------------------------------------------------------------------------------------------------
def _load(path, tries=4, pause=30.0):
    """np.load of a trajectory that a running job may be rewriting (a partial save): retried when the file is caught half written."""
    for i in range(tries):
        try:
            z = np.load(path, allow_pickle=True)
            _ = z["pt__occ"].shape
            return z
        except Exception as e:                                  # noqa: BLE001  (BadZipFile, EOFError, OSError, KeyError: a file being written)
            if i == tries - 1:
                raise
            print(f"  {path}: not readable ({type(e).__name__}), a partial save in progress? retry in {pause:.0f} s", flush=True)
            time.sleep(pause)


def _box_of(spec):
    """(lo, hi) of the box a run's bodies live in, length units: its walls' (active_force[box_slab]), else, with no walls (a periodic box), the world's."""
    w = next((o for o in spec["operators"] if o["op"] == "active_force" and o.get("model") == "box_slab"), None)
    if w is not None:
        return np.asarray(w["lo"], float), np.asarray(w["hi"], float)
    return np.zeros(3), np.asarray(spec["general"]["world"], float)


def measure(key, save=True, run=None):
    """Per stored row of a run (or of a partial trajectory): are all positions finite; per kind of body the RMS distance of the bodies' centres from where they
    started, nm; every body's radius of gyration over its own at the start (1 = the shape is kept; the least, mean and largest over the bodies); how far the deepest
    point sits beyond a wall of the slab, nm; `rg_kind`, the least and largest of that ratio per kind; and, for a run with chains, `bond`: the distance between the
    centres of two consecutive subunits of a chain over its own at the start (least, mean, largest over all bonds). Health: `non_finite`, `torn` (a body's radius of
    gyration left 0.8-1.25 of its start), `escaped` (a point over 3 nm beyond a wall), `chain_broken` (a bond over 1.5 of its start)."""
    run = run or run_name(key)
    rd = os.path.join(GD, run)
    z = _load(os.path.join(rd, "trajectory.npz"))
    spec = yaml.safe_load(open(os.path.join(rd, "spec.yaml"))) if os.path.exists(os.path.join(rd, "spec.yaml")) else yaml.safe_load(open(os.path.join(ROOT, "config", "tissue", run + ".yaml")))
    g = spec["general"]
    stride = max(1, int(g["n_frames"]) // max(int(g["record_cap"]) - 1, 1))
    lo, hi = _box_of(spec)
    occ = z["pt__occ"].astype(bool)
    pid = np.rint(z["pt__pid"][..., 0]).astype(np.int64)
    X = z["pt__pos"]
    nt = z["cell__node_type"]
    nt = np.rint(nt[-1] if nt.ndim == 2 else nt).astype(np.int64)
    names = [str(x) for x in z["cell__type_names"]]
    ncell = z["cell__occ"].shape[1]
    ba, bb = [], []                                              # the two bodies of every chain bond: a kind's chains are its first n x m bodies, in seeding order (seed_table)
    used = {}
    for ch in (params(key).get("chains") or []) if known(key) else []:
        if ch["kind"] not in names:
            continue
        ids = np.flatnonzero(nt[:ncell] == names.index(ch["kind"]))[used.get(ch["kind"], 0):]
        k = min(len(ch["rows"]) if ch.get("rows") else int(ch["n"]), len(ids) // int(ch["m"]))
        used[ch["kind"]] = used.get(ch["kind"], 0) + k * int(ch["m"])
        ids = ids[:k * int(ch["m"])].reshape(k, int(ch["m"]))
        ba += list(ids[:, :-1].ravel())
        bb += list(ids[:, 1:].ravel())
    ba, bb, bl0 = np.asarray(ba, np.int64), np.asarray(bb, np.int64), None
    rows, cen0, rg0 = [], None, None
    for r in range(occ.shape[0]):
        L = occ[r]
        x = X[r][L].astype(np.float64)
        p = pid[r][L]
        fin = bool(np.isfinite(x).all())
        x = np.nan_to_num(x)
        cnt = np.maximum(np.bincount(p, minlength=ncell)[:ncell], 1)
        cen = np.stack([np.bincount(p, weights=x[:, k], minlength=ncell)[:ncell] for k in range(3)], 1) / cnt[:, None]
        rg = np.sqrt(np.bincount(p, weights=((x - cen[p]) ** 2).sum(1), minlength=ncell)[:ncell] / cnt)
        if cen0 is None:
            cen0, rg0, have = cen, np.maximum(rg, 1e-9), np.bincount(p, minlength=ncell)[:ncell] > 0
        ratio = (rg / rg0)[have]
        full = rg / rg0
        bl = np.linalg.norm(cen[bb] - cen[ba], axis=1) if len(ba) else None
        if bl is not None and bl0 is None:
            bl0 = np.maximum(bl, 1e-9)
        d = np.linalg.norm(cen - cen0, axis=1)
        beyond = float(np.maximum(np.maximum(lo[None] - x, x - hi[None]), 0.0).max()) if len(x) else 0.0
        rows.append(dict(frame=int(r * stride), t_us=float(r * stride * float(g.get("dt", 1.0)) * TIME_US), finite=fin, bodies=int(have.sum()), points=int(L.sum()),
                         rms_nm={nm: float(np.sqrt((d[have & (nt[:ncell] == i)] ** 2).mean()) * NM) for i, nm in enumerate(names) if (have & (nt[:ncell] == i)).any()},
                         rg_ratio=[float(ratio.min()), float(ratio.mean()), float(ratio.max())], beyond_wall_nm=beyond * NM,
                         rg_kind={nm: [float(full[have & (nt[:ncell] == i)].min()), float(full[have & (nt[:ncell] == i)].max())] for i, nm in enumerate(names) if (have & (nt[:ncell] == i)).any()},
                         **({} if bl is None else dict(bond=[float((bl / bl0).min()), float((bl / bl0).mean()), float((bl / bl0).max())]))))
    pick = lambda frac: rows[min(len(rows) - 1, int(round(frac * (len(rows) - 1))))]
    ms = np.asarray(z["frame_ms"], float) if "frame_ms" in z.files else None
    out = dict(run=f"tissue/{run}", rows=len(rows), frames_of=int(g["n_frames"]), start=rows[0], quarter=pick(0.25), half=pick(0.5), end=rows[-1],
               non_finite=not all(q["finite"] for q in rows),
               torn=any(q["rg_ratio"][0] < 0.8 or q["rg_ratio"][2] > 1.25 for q in rows),
               escaped=(not (spec.get("plotting") or {}).get("periodic")) and any(q["beyond_wall_nm"] > 3.0 for q in rows),      # a periodic box has no wall to escape
               chain_broken=any(q.get("bond", [1.0, 1.0, 1.0])[2] > 1.5 for q in rows),
               s_per_frame=None if ms is None or not len(ms) else float(np.median(ms) / 1000.0))
    if save and run == run_name(key):
        os.makedirs(DATA, exist_ok=True)
        json.dump(dict(summary=out, series=rows), open(os.path.join(DATA, f"measure_{key}.json"), "w"), indent=1)
    return out


def why_text(key, m):
    e, h = m["end"], m["half"]
    rms = lambda q: ", ".join(f"{nm} {v:.0f}" for nm, v in q["rms_nm"].items())
    bad = [w for w, on in (("NON-FINITE positions", m["non_finite"]), ("a TORN body", m["torn"]), ("points ESCAPED the slab", m["escaped"]),
                           ("a chain BROKEN", m.get("chain_broken", False))) if on]
    bond = (f"chains: the distance between two consecutive subunits ends between {e['bond'][0]:.2f} and {e['bond'][2]:.2f} of its start (mean {e['bond'][1]:.2f}); " if "bond" in e else "")
    fills = ", ".join(f"{nm} {SPECIES[nm]['colour']}" for nm in e["rms_nm"])
    return (f"EXP09 {key} (tissue/{run_name(key)}): {why_of(key)}. "
            f"RESULT at frame {e['frame']} ({e['t_us']:.0f} us): {e['bodies']} bodies, {e['points']:,} material points; {'HEALTH: ' + '; '.join(bad) if bad else 'healthy (finite, no body torn, none escaped)'}; "
            f"RMS distance of the bodies' centres from their start, nm, per kind: {rms(e)} (at frame {h['frame']}: {rms(h)}); "
            f"shape kept: each body's radius of gyration over its own at the start between {e['rg_ratio'][0]:.2f} and {e['rg_ratio'][2]:.2f} (mean {e['rg_ratio'][1]:.3f}); "
            f"{bond}deepest point beyond a wall of the slab {e['beyond_wall_nm']:.1f} nm. "
            f"Look: flat fills by kind ({fills}), each body outlined in its own colour darkened, far bodies paler.")


# ---- the movie ------------------------------------------------------------------------------------------------------------------------------------------------------
FRAME = (1920, 1088)           # px: 16:9 as the slab, both sides multiples of the encoder's 16-px block
MIN_PTS = 16                   # a body of fewer live points is not drawn (a dormant slot's leftovers); a coarse scene's 7-point bodies set it lower (big_first)
CLOCK = None                   # the clock's text in place of the drawn count and the time (big_first: the whole slab's count); or a function of the drawn row
FRAME_HOOK = None              # a function (row, frame) -> frame applied to every drawn frame before it is written (render_piece: the inset)
VIEW_ANGLE = 20.0              # degrees: a narrow lens, so the near face of the slab is drawn only 1.09x larger than its middle
ZOOM = 1.03                    # the slab's middle plane is 3 % taller than the frame: the picture bleeds off every edge
FADE = 0.25                    # share of the ground's colour mixed into a body at the far face of the slab (0 at the near face): far bodies paler
SHADE_OF_KIND = 0.10           # each body's fill is its kind's colour times 1 +- this, fixed per body: neighbours of one kind stay distinct


LABEL_PX = 44                  # height of the two labels' type, px of the 1920 x 1088 frame (Cedric, 2026-11-15 08:40: "increase font for 20 nm and timing"; it was 22 until step 9)


def _font():
    from PIL import ImageFont
    import matplotlib
    return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"), LABEL_PX)


def _bar_nm(frame, px_per_unit):
    """A white scale bar with its length in nm under it, bottom left: the round length (1, 2 or 5 x 10^k nm) nearest a tenth of the frame's width."""
    from PIL import Image, ImageDraw
    tgt = 0.1 * frame.shape[1] / px_per_unit * NM
    L = min((f * 10.0 ** k for k in range(-1, 4) for f in (1.0, 2.0, 5.0)), key=lambda v: abs(math.log(v / tgt)))
    im = Image.fromarray(frame)
    dr = ImageDraw.Draw(im)
    font, text = _font(), f"{L:g} nm"
    l, t, r, b = dr.textbbox((0, 0), text, font=font)
    x0, y = 24, frame.shape[0] - 24 - (b - t)                    # the label's top; its bottom 24 px above the frame's edge, as the clock's
    dr.rectangle([x0, y - 22, x0 + int(L / NM * px_per_unit), y - 14], fill=(255, 255, 255))
    dr.text((x0 - l, y - t), text, fill=(255, 255, 255), font=font)
    return np.array(im)


def _clock(frame, text):
    """The body count and the time, white, bottom right, in the scale bar's type (exp23_studio._clock draws it at 22 px, fixed)."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame)
    dr = ImageDraw.Draw(im)
    font = _font()
    l, t, r, b = dr.textbbox((0, 0), text, font=font)
    dr.text((frame.shape[1] - 24 - (r - l) - l, frame.shape[0] - 24 - (b - t) - t), text, fill=(255, 255, 255), font=font)
    return np.array(im)


def render(key, run=None, name=None, last_only=False, max_frames=300, fps=25, rows_sel=None, view=None, frame_wh=None, use_section=True, bar=True):
    """The movie of a run in the flat-ink look, full frame: per drawn row every body's iso-surface (exp23_studio._cell_surface, as the record's movies draw an MPM cell),
    filled flat in its kind's colour, then gui/flat_ink's ink -- folds and silhouettes from the depth buffer, and a line wherever two DIFFERENT bodies meet, found on a
    second render of the same surfaces with every body in its own id colour (flat_ink tells objects apart by hue, which cannot separate two bodies of one kind) -- and the
    far bodies mixed toward the ground. Writes data/<name>.mp4 and .png (the last frame); `last_only`: the last stored row alone, PNG only (the preview of a running job)."""
    import imageio.v2 as iio
    import pyvista as pv
    import exp23_studio as e23
    import exp29_studio as S
    from matplotlib.colors import to_rgb
    from plexus.gui import flat_ink
    e23._LOOK.clear(); e23._LOOK.update(e23.LOOK["ink"])
    run = run or run_name(key)
    name = name or run
    if not os.path.exists(os.path.join(GD, run, "trajectory.npz")):
        raise SystemExit(f"tissue/{run}: no trajectory.npz yet (a running job writes its first partial save 10 wall-clock minutes in)")
    d, kind = S.resolve(f"tissue/{run}")
    if kind != "mpm":
        raise SystemExit(f"{run}: not an MPM run")
    spec, _ = S.read_spec(d)
    for i in range(4):                                          # a partial save may be caught half written
        try:
            R = S.MPMRun(d, spec)
            R.positions(0)
            break
        except Exception as e:                                  # noqa: BLE001
            if i == 3:
                raise
            print(f"  trajectory of {run} not readable ({type(e).__name__}), a partial save in progress? retry in 30 s", flush=True)
            time.sleep(30.0)
    g = spec["general"]
    stride_f = max(1, int(g["n_frames"]) // max(int(g["record_cap"]) - 1, 1))             # frames between two stored rows (a partial trajectory holds the first rows of the run)
    lo, hi = _box_of(spec)
    c0 = 0.5 * (lo + hi)
    W, Hh = frame_wh or FRAME
    tn = math.tan(math.radians(VIEW_ANGLE / 2))
    if view is None:
        view, up = S.view_vectors(90.0, -90.0)                   # looking down z on the slab, y up
        span = (hi[1] - lo[1]) if frame_wh is None else max(hi[1] - lo[1], (hi[0] - lo[0]) * Hh / W)     # a panel (frame_wh): the box's x fits too
        dist = 0.5 * span / ZOOM / tn
        near, far = dist - 0.5 * (hi[2] - lo[2]), dist + 0.5 * (hi[2] - lo[2])
    else:                                                        # an oblique view (elevation, azimuth, degrees): the box's bounding sphere fits the panel
        view, up = S.view_vectors(float(view[0]), float(view[1]))
        Rb = 0.5 * float(np.linalg.norm(hi - lo))
        dist = Rb / tn / min(1.0, W / Hh)
        near, far = dist - Rb, dist + Rb
    px_per_unit = Hh / (2.0 * dist * tn)                        # at the box's centre
    wpp = 1.0 / px_per_unit
    ground = np.array([int(S.GROUND[i:i + 2], 16) for i in (1, 3, 5)], np.float64)
    pl = pv.Plotter(off_screen=True, window_size=(W, Hh))
    pl.set_background(S.GROUND)
    S._add_lights(pl, c0, float(np.linalg.norm(hi - lo)) / 2, view, up)
    pl.camera.position = tuple(c0 + dist * view)
    pl.camera.focal_point = tuple(c0)
    pl.camera.up = tuple(up)
    pl.camera.view_angle = VIEW_ANGLE
    # the colours: a body's fill = its kind's colour x a fixed factor in 1 +- SHADE_OF_KIND; its id colour = its slot number + 1 written in the three bytes
    ncell = R.live.shape[1]
    kinds = R.type_rgb if R.type_rgb is not None else np.tile(np.array(to_rgb("#4D96FF")) * 255.0, (ncell, 1))
    f_body = 1.0 + SHADE_OF_KIND * (2.0 * np.random.default_rng(9).random(ncell) - 1.0)
    fill = np.clip(np.rint(kinds[:ncell] * f_body[:, None]), 0, 255).astype(np.uint8)
    ids = np.arange(1, ncell + 1)
    idc = np.stack([(ids >> 16) & 255, (ids >> 8) & 255, ids & 255], 1).astype(np.uint8)
    glow = (spec.get("plotting") or {}).get("glow")                # a per-body value in 0..1 (a cell block's channel) shown as a mix of the fill toward one colour
    gval = gcol = None
    if glow:
        zz = _load(os.path.join(GD, run, "trajectory.npz"))
        key = "cell__" + str(glow["block"])
        if key in zz.files and glow.get("channels"):                 # three competitors: [rows, cells, 3] shares, sharpened so that the leading one shows (share^2, renormalised)
            c = np.clip(np.nan_to_num(np.asarray(zz[key], np.float64)[..., [int(v) for v in glow["channels"]]]), 0.0, None) ** 2
            gval = c / np.maximum(c.sum(-1, keepdims=True), 1e-12)
            gcol = np.array([to_rgb(v) for v in glow["colours"]]) * 255.0
        elif key in zz.files:
            gval = np.clip(np.nan_to_num(np.asarray(zz[key], np.float64)[..., int(glow.get("channel", 0))]), 0.0, 1.0)      # [rows, cells]
            gcol = np.array(to_rgb(glow["colour"])) * 255.0
    bonds_of = None                                                # per stored row, the pairs of bodies that hold a bond (bound density above 0.5), when the spec asks for their lines
    if (spec.get("plotting") or {}).get("bond_lines"):
        zb_ = _load(os.path.join(GD, run, "trajectory.npz"))
        if "junction__rho" in zb_.files:
            jo, jr = zb_["junction__occ"], zb_["junction__rho"][..., 0]
            ja, jb = np.rint(zb_["junction__ca"][..., 0]).astype(np.int64), np.rint(zb_["junction__cb"][..., 0]).astype(np.int64)
            bonds_of = lambda k_: (lambda m: np.stack([ja[k_][m], jb[k_][m]], 1))(jo[k_] & (jr[k_] > 0.5)) if k_ < jo.shape[0] else np.zeros((0, 2), np.int64)
    h_vox = 0.55 * (R.v ** (1.0 / 3.0))
    iso = float((spec.get("plotting") or {}).get("surface_iso_frac", 0.5))
    blur = float((spec.get("plotting") or {}).get("surface_blur", 1.6))             # the blur of a body's points before its surface, voxels (exp23_studio._cell_surface's 1.6)
    rows = R.rows
    ks = [rows - 1] if last_only else sorted(set(range(0, rows, max(1, math.ceil(rows / max_frames)))) | {rows - 1})
    if rows_sel is not None:                                     # a share of the rows (render_panels: one process of several)
        ks = [k_ for k_ in ks if k_ in set(rows_sel)]
    os.makedirs(DATA, exist_ok=True)
    out_mp4, out_png = os.path.join(DATA, name + ".mp4"), os.path.join(DATA, name + ".png")
    writer = None if last_only else iio.get_writer(out_mp4, fps=int(fps), quality=8, macro_block_size=16)
    frame = None
    for j, k in enumerate(ks):
        X = R.positions(k)
        idx = np.flatnonzero(R.occ[k])
        idx = idx[np.argsort(R.pid[k][idx], kind="stable")]
        cut = np.flatnonzero(np.diff(R.pid[k][idx])) + 1
        surfs = []
        centre = {}
        sec = (spec.get("plotting") or {}).get("section_z") if use_section else None
        for part in np.split(idx, cut):
            c = int(R.pid[k][part[0]])
            if len(part) < int((spec.get("plotting") or {}).get("min_points", MIN_PTS)) or c >= ncell:
                continue
            Xb = X[part]
            cz = Xb.mean(0)
            if (spec.get("plotting") or {}).get("periodic"):            # a periodic box: the body shifted by whole periods, its centre inside the box
                Lb_ = hi - lo
                sh_ = -Lb_ * np.floor((cz - lo) / Lb_)
                Xb, cz = Xb + sh_, cz + sh_
            if sec and not (float(sec[0]) <= cz[2] <= float(sec[1])):          # a deep box: only the bodies of the drawn slice
                continue
            centre[c] = cz
            s = e23._cell_surface(Xb, R.v, h_vox, iso=iso, blur=blur)
            if s is None:
                continue
            if gval is None or k >= gval.shape[0]:
                col = fill[c]
            elif gval.ndim == 3:                                         # the mix of the three colours, with the body's own fixed shade factor
                col = np.clip(np.rint((gval[k, c][:, None] * gcol).sum(0) * f_body[c]), 0, 255).astype(np.uint8)
            else:
                col = np.rint(fill[c] * (1.0 - gval[k, c]) + gcol * gval[k, c]).astype(np.uint8)
            s["rgb"] = np.tile(col, (s.n_points, 1))
            s["idrgb"] = np.tile(idc[c], (s.n_points, 1))
            surfs.append(s)
        merged = surfs[0] if len(surfs) == 1 else pv.merge(surfs, merge_points=False)
        # pass 1: the bodies in their id colours, unlit -- which body each pixel shows -- and the depth of the frame
        pl.add_mesh(merged, scalars="idrgb", rgb=True, lighting=False, name="cell_all", show_scalar_bar=False, opacity=1.0, reset_camera=False)
        pl.renderer.ResetCameraClippingRange()
        pl.render()
        idimg = np.ascontiguousarray(pl.screenshot(return_img=True)[..., :3]).astype(np.int64)
        zb = flat_ink.zbuffer(pl.render_window)
        # pass 2: the same surfaces in their fills, the record's flat material
        pl.add_mesh(merged, scalars="rgb", rgb=True, smooth_shading=True, pbr=False, specular=0.0, diffuse=0.15, ambient=0.85, name="cell_all", show_scalar_bar=False,
                    opacity=1.0, reset_camera=False)
        pl.renderer.ResetCameraClippingRange()
        pl.render()
        img = np.ascontiguousarray(pl.screenshot(return_img=True)[..., :3])
        if zb is not None and zb.shape == img.shape[:2]:
            D = flat_ink.depth(zb, pl.camera)
            scene = (zb < 1.0 - 1e-7) & (D > 0.2 * dist)
            # THE LINES ARE FOUND ON A FRAME PADDED BY ITS OWN EDGE PIXELS: flat_ink compares each pixel with its neighbours by np.roll, which wraps, and here bodies run off
            # the frame (the record's movies never touch it): the smoke's left edge was inked wherever the right edge showed ground (2026-11-14)
            pad = 3
            Dp, sp_ = np.pad(D, pad, mode="edge"), np.pad(scene, pad, mode="edge")
            _, cell, hard = flat_ink.ink_masks(np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge"), Dp, sp_, wpp, contacts=False)      # the silhouettes against the ground
            # flat_ink's THIN line (a depth step of over 1.6 px of world between two pixels, meant for folds) is left out: on these small round bodies every steep flank
            # passes it on some triangles and not on others, a stipple of ticks along each outline (smoke, 2026-11-14); a body in front of another is outlined below
            code = np.pad((idimg[..., 0] << 16) | (idimg[..., 1] << 8) | idimg[..., 2], pad, mode="edge")
            # ONE PIXEL ON EACH SIDE of the border between two bodies, two in all as the silhouette is. Not "two pixels on the nearer body": two bodies in contact are
            # drawn as surfaces that cross, their border is the crossing line where both depths are equal, and "nearer" flipped from pixel to pixel: a comb of ticks
            # round every body that touched another (preview of it01 at frame 320, 2026-11-14)
            for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
                hard |= sp_ & np.roll(sp_, sh, axis=ax) & (np.roll(code, sh, axis=ax) != code)
            crop = (slice(pad, -pad), slice(pad, -pad))
            img = flat_ink.apply(flat_ink.shade(img, D, scene, wpp), np.zeros_like(scene), cell[crop], hard[crop])
            mix = (FADE * np.clip((D - near) / max(far - near, 1e-9), 0.0, 1.0))[..., None]     # the far bodies paler, lines included
            img = np.where(scene[..., None], np.clip(img * (1.0 - mix) + ground[None, None] * mix, 0, 255), img).astype(np.uint8)
        if bonds_of is not None:                                     # the bonds, over the bodies: white lines 3 px wide, centre to centre, at the far face as faint as the bodies there
            from PIL import Image, ImageDraw
            pairs = [(a_, b_) for a_, b_ in bonds_of(k).tolist() if a_ in centre and b_ in centre]
            if (spec.get("plotting") or {}).get("periodic"):            # a bond across a face of the periodic box is not drawn (its two ends are on opposite sides)
                pairs = [(a_, b_) for a_, b_ in pairs if (np.abs(centre[a_] - centre[b_]) < 0.5 * (hi - lo)).all()]
            if pairs:
                ren = pl.renderer
                def _px(p_):
                    ren.SetWorldPoint(float(p_[0]), float(p_[1]), float(p_[2]), 1.0)
                    ren.WorldToDisplay()
                    d_ = ren.GetDisplayPoint()
                    return (d_[0], img.shape[0] - 1 - d_[1])
                over = Image.new("RGBA", (img.shape[1], img.shape[0]), (0, 0, 0, 0))
                dr = ImageDraw.Draw(over)
                for a_, b_ in pairs:
                    zf = 0.5 * (centre[a_][2] + centre[b_][2])                                  # the camera looks down z from above: high z is near
                    al = int(255 * (1.0 - FADE * float(np.clip((hi[2] - zf) / max(hi[2] - lo[2], 1e-9), 0.0, 1.0))) * 0.9)
                    dr.line([_px(centre[a_]), _px(centre[b_])], fill=(255, 255, 255, al), width=3)
                img = np.array(Image.alpha_composite(Image.fromarray(np.ascontiguousarray(img)).convert("RGBA"), over).convert("RGB"))
        frame = _bar_nm(img, px_per_unit) if bar else img
        frame = _clock(frame, (CLOCK(k) if callable(CLOCK) else CLOCK) or f"{len(surfs)} bodies   t = {k * stride_f * float(g.get('dt', 1.0)) * TIME_US:.0f} µs")
        if FRAME_HOOK is not None:                                 # big scenes: the whole slab's inset over the frame (render_piece)
            frame = FRAME_HOOK(k, frame)
        if writer is not None:
            writer.append_data(frame)
            if (j + 1) % 10 == 0:
                print(f"  frame {j + 1}/{len(ks)}", flush=True)
    if writer is not None:
        writer.close()
    iio.imwrite(out_png, frame)
    pl.close()
    info = dict(run=f"tissue/{run}", rows=rows, drawn=len(ks), last_frame=int(ks[-1] * stride_f), of_frames=int(g["n_frames"]), png=os.path.relpath(out_png, ROOT),
                mp4=None if last_only else os.path.relpath(out_mp4, ROOT))
    print(info)
    return info


MATRIX_W = 560                         # the matrix panel at the right of the movie, px wide (the frame 1920 + 560 = 2480, a multiple of 16)


def matrix_hook(key, run=None):
    """The hook `render` applies to every frame of a scene with `matrix_panel` (Cedric, 2026-11-19: "add on the right a panel with the Delta G matrix and another live
    matrix heatmap of binding with the total printed above"): at the right of the frame, top, the binding matrix of the scene's kinds -- its free energies (`binding.delta_g`)
    or its affinities (`binding.matrix`) -- and under it the bonds held at that frame (bound share above 0.5) counted per pair of kinds, their total printed above it; one
    colour scale each, the counts' fixed over the run (its largest count). Each row and column carries its kind's colour as in the movie. Returns the hook."""
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    from matplotlib import colormaps
    pk = params(key)
    bd = pk["binding"]
    if bd.get("delta_g"):
        mn, M = bd["delta_g"]
        title = "\u0394G, " + str(bd.get("delta_g_unit", "kcal/mol"))
    else:
        mn, M = bd["matrix"]
        title = "affinity"
    z = _load(os.path.join(GD, run or run_name(key), "trajectory.npz"))
    tn = [str(x) for x in z["cell__type_names"]]
    nt = z["cell__node_type"]
    nt = np.rint(nt[-1] if nt.ndim == 2 else nt).astype(np.int64)
    order = [nm for nm in mn if nm in tn]                               # the matrix's own order (mRNA kinds, then protein kinds), the kinds the run holds
    n = len(order)
    pos = np.full(len(tn), -1)
    for q, nm in enumerate(order):
        pos[tn.index(nm)] = q
    kc = pos[nt]                                                        # each cell's row in the matrix (-1: a kind outside it)
    G = np.array([[np.nan if M[mn.index(a)][mn.index(b)] is None else float(M[mn.index(a)][mn.index(b)]) for b in order] for a in order], float)
    if not bd.get("delta_g"):
        G[G == 0.0] = np.nan                                            # an affinity of 0: no entry
    jo, jr = z["junction__occ"], z["junction__rho"][..., 0]
    ja, jb = np.rint(z["junction__ca"][..., 0]).astype(np.int64), np.rint(z["junction__cb"][..., 0]).astype(np.int64)

    def counts(r):
        m = jo[r].astype(bool) & (jr[r] > 0.5)
        a, b = kc[ja[r][m]], kc[jb[r][m]]
        ok = (a >= 0) & (b >= 0)
        C = np.zeros((n, n))
        np.add.at(C, (a[ok], b[ok]), 1.0)
        return C + C.T - np.diag(np.diag(C)), int(m.sum())              # symmetric; a pair of one kind counted once
    cmax = max(1.0, max(float(counts(r)[0].max()) for r in range(jo.shape[0])))
    cmap = colormaps["viridis"]
    ground = tuple(int(v) for v in _rgb(GROUND))
    empty = (104, 114, 126)                                             # a cell with no entry, or no bond: darker than the ground
    ttf = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
    f_big, f_small = ImageFont.truetype(ttf, 36), ImageFont.truetype(ttf, 24)
    cell = max(4, 360 // n)
    side = cell * n
    x0 = (MATRIX_W - side) // 2 + 10

    def heat(V, lo, hi):
        t = np.clip((V - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
        rgb = (np.asarray(cmap(t))[..., :3] * 255).astype(np.uint8)
        rgb[~np.isfinite(V)] = empty
        return Image.fromarray(rgb).resize((side, side), Image.NEAREST)

    def block(img, y, V, lo, hi, head, lab_lo, lab_hi):
        dr = ImageDraw.Draw(img)
        dr.text((x0, y), head, fill=(255, 255, 255), font=f_big)
        y1 = y + 54
        img.paste(heat(V, lo, hi), (x0, y1))
        for q, nm in enumerate(order):                                  # the kinds' colours down the left and along the bottom
            c = tuple(int(v) for v in _rgb(SPECIES[nm]["colour"]))
            dr.rectangle([x0 - 16, y1 + q * cell + 1, x0 - 5, y1 + (q + 1) * cell - 2], fill=c)
            dr.rectangle([x0 + q * cell + 1, y1 + side + 5, x0 + (q + 1) * cell - 2, y1 + side + 16], fill=c)
        yb = y1 + side + 26
        bar = (np.asarray(cmap(np.linspace(0.0, 1.0, side)))[None, :, :3] * 255).astype(np.uint8)
        img.paste(Image.fromarray(np.repeat(bar, 12, 0)), (x0, yb))
        dr.text((x0, yb + 16), lab_lo, fill=(255, 255, 255), font=f_small)
        l, t, r, b = dr.textbbox((0, 0), lab_hi, font=f_small)
        dr.text((x0 + side - (r - l) - l, yb + 16), lab_hi, fill=(255, 255, 255), font=f_small)

    lo_g, hi_g = float(np.nanmin(G)), float(np.nanmax(G))
    static = Image.new("RGB", (MATRIX_W, FRAME[1]), ground)
    block(static, 30, G, lo_g, hi_g, title, f"{lo_g:g}", f"{hi_g:g}")

    def hook(k, frame):
        C, tot = counts(k)
        V = np.where(C > 0, C, np.nan)
        img = static.copy()
        block(img, 30 + 54 + side + 26 + 12 + 16 + 30 + 40, V, 1.0, max(cmax, 2.0), f"bonds {tot:,}", "1", f"{cmax:g}")
        return np.concatenate([frame, np.asarray(img)[:frame.shape[0]]], axis=1)
    return hook


PANEL = (960, 1088)                    # each half of a two-panel movie, px
VIEW_3D = (30.0, -60.0)                # the left panel's camera: 30 degrees above the box, turned 60 degrees about z


def render_part(key, i, n, fps):
    """Rows i/n of a run drawn twice by `render` -- at left the whole box seen from VIEW_3D (every body, no scale bar: it holds at one depth only), at right the slice
    `section` seen from above (every body whose centre lies in it, whole, with the scale bar) -- and the two put side by side, a white line between: data/panels_<key>_<i>.mp4."""
    import imageio_ffmpeg
    rd = os.path.join(GD, run_name(key))
    z = _load(os.path.join(rd, "trajectory.npz"))
    rows = int(z["cell__occ"].shape[0])
    part = list(range(rows))[i * rows // n:(i + 1) * rows // n]
    if not part:
        return None
    a = render(key, name=f"p3d_{key}_{i}", max_frames=10 ** 6, fps=fps, rows_sel=part, view=VIEW_3D, frame_wh=PANEL, use_section=False, bar=False)
    b = render(key, name=f"psl_{key}_{i}", max_frames=10 ** 6, fps=fps, rows_sel=part, frame_wh=PANEL, use_section=True)
    out = os.path.join(DATA, f"panels_{key}_{i}.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", os.path.join(ROOT, a["mp4"]), "-i", os.path.join(ROOT, b["mp4"]),
                    "-filter_complex", f"[0][1]hstack=inputs=2,drawbox=x={PANEL[0] - 1}:y=0:w=2:h=ih:color=white:t=fill", "-r", str(fps),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", out], check=True)
    for q in (a, b):
        for f in (q["mp4"], q["png"]):
            if f and os.path.exists(os.path.join(ROOT, f)):
                os.remove(os.path.join(ROOT, f))
    return out


def render_panels(key, fps, procs=8):
    """The two-panel movie of a run (render_part), its rows dealt to `procs` processes at once, the parts joined: data/<run>.mp4 and its last frame data/<run>.png.
    Returns render's info dict."""
    import imageio.v2 as iio
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]))
    logs = [open(os.path.join(DATA, f"panels_{key}_{i}.log"), "w") for i in range(procs)]
    jobs = [subprocess.Popen([sys.executable, "-u", __file__, "render_part", key, "--piece", f"{i}/{procs}", "--fps", str(fps)], env=env, cwd=ROOT, stdout=logs[i], stderr=logs[i])
            for i in range(procs)]
    bad = [i for i, j in enumerate(jobs) if j.wait() != 0]
    if bad:
        raise RuntimeError(f"render_part {key}: parts {bad} failed (see {DATA}/panels_{key}_<i>.log)")
    parts = [q for q in (os.path.join(DATA, f"panels_{key}_{i}.mp4") for i in range(procs)) if os.path.exists(q)]
    name = run_name(key)
    mp4, png = os.path.join(DATA, name + ".mp4"), os.path.join(DATA, name + ".png")
    _concat(parts, mp4)
    r = iio.get_reader(mp4)
    n = r.count_frames()
    iio.imwrite(png, np.asarray(r.get_data(n - 1))[..., :3])
    for q in parts:
        os.remove(q)
    g = yaml.safe_load(open(os.path.join(GD, name, "spec.yaml")))["general"] if os.path.exists(os.path.join(GD, name, "spec.yaml")) else params(key)
    nf = int(g.get("n_frames", params(key)["frames"]))
    return dict(run=f"tissue/{name}", rows=n, drawn=n, last_frame=nf, of_frames=nf, png=os.path.relpath(png, ROOT), mp4=os.path.relpath(mp4, ROOT))


def look(key):
    """What the session reads before the next job is submitted (Cedric, 2026-11-14: "look at the movie when it lands before running a new job"): six frames of the
    step's movie, evenly spaced, side by side at a third of their size -> data/look_<key>.png; the mean change of a pixel between consecutive movie frames, of 255
    (how much the picture shakes); the share of the last frame that is bare ground; and the run's health and wander from data/measure_<key>.json."""
    import imageio.v2 as iio
    steps = json.load(open(os.path.join(DATA, "steps.json")))
    mp4 = os.path.join(RECORD, "mp4", f"{int(steps[key]):04d}.mp4")
    r = iio.get_reader(mp4)
    n = r.count_frames()
    pick = [int(round(q * (n - 1))) for q in np.linspace(0.0, 1.0, 6)]
    fr = [np.asarray(r.get_data(i))[..., :3] for i in pick]
    small = [q[::3, ::3] for q in fr]
    sheet = np.concatenate([np.concatenate(small[:3], 1), np.concatenate(small[3:], 1)], 0)
    out = os.path.join(DATA, f"look_{key}.png")
    iio.imwrite(out, sheet)
    ground = np.array([0x8d, 0x99, 0xa6])
    bare = float((np.abs(fr[-1].astype(int) - ground).max(-1) <= 3).mean())
    idx = list(range(0, n - 1, max(1, (n - 1) // 20)))
    shake = float(np.mean([np.abs(np.asarray(r.get_data(i))[..., :3].astype(float) - np.asarray(r.get_data(i + 1))[..., :3].astype(float)).mean() for i in idx]))
    m = json.load(open(os.path.join(DATA, f"measure_{key}.json")))["summary"]
    e = m["end"]
    print(json.dumps(dict(key=key, step=int(steps[key]), sheet=os.path.relpath(out, ROOT), movie_frames=n, movie_s=round(n / r.get_meta_data()["fps"], 1), frames_shown=pick,
                          pixel_change_per_movie_frame_of_255=round(shake, 1), bare_ground_share_last_frame=round(bare, 3),
                          health=dict(non_finite=m["non_finite"], torn=m["torn"], escaped=m["escaped"], chain_broken=m.get("chain_broken", False)), bodies=e["bodies"],
                          rg_kind_end={k: [round(v, 2) for v in q] for k, q in e.get("rg_kind", {}).items()}, bond_end=[round(v, 2) for v in e.get("bond", [])],
                          rms_nm_end={k: round(v, 1) for k, v in e["rms_nm"].items()}, rg_ratio_end=[round(v, 2) for v in e["rg_ratio"]], s_per_frame=m["s_per_frame"]), indent=1))
    return out


def shapes(names=None):
    """data/shapes.png: the points of every kind (or of `names`) seen along z and along y, with its size in nm -- a local check of the lattice shapes, not a record step."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [nm for nm in (names or SPECIES) if nm in SPECIES]
    fig, axs = plt.subplots(2, len(names), figsize=(1.5 * len(names), 3.4), squeeze=False)
    for j, nm in enumerate(names):
        P = body_points(SPECIES[nm]["shape"], SPECIES[nm]["n"])
        size = (np.ptp(P, 0) + SPACING) * NM
        print(f"  {nm:<11s} {len(P):>3d} points  {size[0]:4.0f} x {size[1]:4.0f} x {size[2]:4.0f} nm")
        for i, (a, b, c) in enumerate(((0, 1, 2), (0, 2, 1))):
            o = np.argsort(P[:, c])
            axs[i, j].scatter(P[o, a], P[o, b], s=22, marker="s", c=SPECIES[nm]["colour"], edgecolors="none")
            axs[i, j].set_xlim(-1.5, 1.5)
            axs[i, j].set_ylim(-1.5, 1.5)
            axs[i, j].set_aspect("equal")
            axs[i, j].axis("off")
        axs[0, j].set_title(f"{nm} {len(P)}\n{size[0]:.0f}x{size[1]:.0f}x{size[2]:.0f} nm", fontsize=6)
    out = os.path.join(DATA, "shapes.png")
    fig.savefig(out, dpi=110, facecolor="#8d99a6")
    plt.close(fig)
    return out


def after(key, replace=None):
    """Measure, why text, the movie; then the movie as a NEW step of the record under its lock (or INTO step `replace`)."""
    import fcntl
    import shutil
    m = measure(key)
    why = why_text(key, m)
    open(os.path.join(DATA, f"why_{key}.txt"), "w").write(why + "\n")
    pk = params(key)
    global FRAME_HOOK
    if pk.get("matrix_panel") and not pk.get("panels3d"):                                # the binding matrix and the live bonds at the right of every frame
        FRAME_HOOK = matrix_hook(key)
    try:
        info = (render_panels(key, int(pk.get("movie_fps", 25))) if pk.get("panels3d") else
                render(key, max_frames=10 ** 6, fps=int(pk.get("movie_fps", 25))))          # every stored row, at the iteration's own frame rate
    finally:
        FRAME_HOOK = None
    os.environ["PLEXUS_BUILDER"] = RECORD
    import gui_drive as gd                                       # the record's own numbering and paths
    with open(os.path.join(DATA, ".render.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        i = int(replace) if replace else gd._next_index()
        shutil.copy(os.path.join(ROOT, info["png"]), gd._path("png", i))
        shutil.copy(os.path.join(ROOT, info["mp4"]), gd._path("mp4", i))
        sp = os.path.join(GD, run_name(key), "spec.yaml")
        shutil.copy(sp if os.path.exists(sp) else os.path.join(ROOT, "config", "tissue", run_name(key) + ".yaml"), gd._path("spec", i))
        with open(gd._path("why", i), "w") as f:
            f.write(f"time   {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"run    tissue/{run_name(key)} (mpm; flat-ink look, every body in its kind's colour, tools/exp09_landscape.py render)\n")
            f.write(f"drawn  {info['drawn']} of {info['rows']} stored rows, frame 0 to {info['last_frame']} of {info['of_frames']}\n")
            f.write(f"why    {why}\n")
        gd.note(f"  exp09_landscape {i:04d}  tissue/{run_name(key)}  -- {'REDRAWN IN PLACE: ' if replace else ''}{why_of(key)}")
        sj = os.path.join(DATA, "steps.json")
        json.dump(dict(json.load(open(sj)) if os.path.exists(sj) else {}, **{key: i}), open(sj, "w"), indent=1)
    print({"index": i, "key": key})
    return 0


# ---- the big scenes' movie: drawn from the bodies' centres (render_big), every 100 frames into the watcher while the run goes (live) -------------------------------------

BIG_W, BIG_H = 1920, 1080             # the frame: the window at movie 21's scale (192 x 108 nm at 10 px per nm), padded to 1088 for the encoder
INSET_W, INSET_H = 640, 360            # the whole slab, top right
GROUND = "#8d99a6"                     # the record's grey ground (exp29_studio.GROUND)
PT_R = 0.75                            # a point's disc in a silhouette, in point spacings: the discs overlap (1.5 spacings across) so a body's outline is smooth
INSET_PX = 30                          # the inset's scale-bar label, px


def _rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float64)


def _big_bar(img, x0, y_bottom, width_px, px_per_unit, size=None):
    """A white scale bar with its length under it, its left end at x0 and its label's bottom `y_bottom`: the round length (1, 2 or 5 x 10^k nm, in um from 1,000 nm)
    nearest a tenth of `width_px`."""
    from PIL import ImageDraw, ImageFont
    import matplotlib
    tgt = 0.1 * width_px / px_per_unit * NM
    Lnm = min((f * 10.0 ** k for k in range(-1, 6) for f in (1.0, 2.0, 5.0)), key=lambda v: abs(math.log(v / tgt)))
    text = f"{Lnm / 1000:g} \u00b5m" if Lnm >= 1000 else f"{Lnm:g} nm"
    dr = ImageDraw.Draw(img)
    font = _font() if size is None else ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"), int(size))
    l, t, r, b = dr.textbbox((0, 0), text, font=font)
    y = y_bottom - (b - t)
    th = 8 if size is None else 5
    dr.rectangle([x0, y - th - 14, x0 + max(2, int(round(Lnm / NM * px_per_unit))), y - 14], fill=(255, 255, 255))
    dr.text((x0 - l, y - t), text, fill=(255, 255, 255), font=font)


def big_frame(cen, kind, names, lo, hi, radius, win, bonds=None, text="", pts=None, pt_r=None):
    """One frame of a big scene, 1920 x 1088. The window `win` = (x0, y0, x1, y1), length units -- the size of the scene before it was scaled, at movie 21's 10 px per nm --
    seen from above: every body a disc of its own radius (`radius`, per kind) filled in its kind's colour and outlined in that colour darkened, or, for a body of `pts`
    = {body: its points [k, 3]}, its own silhouette (a disc of radius `pt_r` at every point over one 3 px wider in the outline's colour), so a lobed body shows its lobes;
    far bodies mixed toward the ground (FADE at the far face), near ones drawn over; the bonds (`bonds`, pairs of body indices) as white lines between the centres. Top
    right, the whole slab, every body a dot of its fill, a white frame round it and round the window in it. `text`: the count and the time, bottom right."""
    from PIL import Image, ImageDraw
    ground = _rgb(GROUND)
    cols = np.array([_rgb(SPECIES[nm]["colour"]) for nm in names])
    depth = np.clip((hi[2] - cen[:, 2]) / max(hi[2] - lo[2], 1e-9), 0.0, 1.0)            # 0 at the near (top) face, 1 at the far one: the camera looks down z
    fill = cols[kind] * (1.0 - FADE * depth[:, None]) + ground[None] * (FADE * depth[:, None])
    # the window
    x0, y0, x1, y1 = win
    sw = BIG_W / (x1 - x0)
    main = Image.new("RGB", (BIG_W, BIG_H), tuple(int(v) for v in ground))
    dr = ImageDraw.Draw(main)
    rk = np.asarray(radius, float)[kind]
    inw = np.flatnonzero((cen[:, 0] > x0 - 2 * rk) & (cen[:, 0] < x1 + 2 * rk) & (cen[:, 1] > y0 - 2 * rk) & (cen[:, 1] < y1 + 2 * rk))
    inw = inw[np.argsort(-depth[inw], kind="stable")]
    for i in inw:
        f = tuple(int(v) for v in np.clip(fill[i], 0, 255))
        o = tuple(int(0.6 * v) for v in f)
        if pts is not None and int(i) in pts:
            q = pts[int(i)]
            qx, qy, r = (q[:, 0] - x0) * sw, BIG_H - (q[:, 1] - y0) * sw, float(pt_r) * sw
            for cx, cy in zip(qx, qy):
                dr.ellipse([cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3], fill=o)
            for cx, cy in zip(qx, qy):
                dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=f)
            continue
        cx, cy, r = (cen[i, 0] - x0) * sw, BIG_H - (cen[i, 1] - y0) * sw, rk[i] * sw
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=f, outline=o, width=3)
    if bonds is not None and len(bonds):
        a, b = bonds[:, 0], bonds[:, 1]
        sel = np.isin(a, inw) & np.isin(b, inw)
        for i, j in zip(a[sel], b[sel]):
            dr.line([((cen[i, 0] - x0) * sw, BIG_H - (cen[i, 1] - y0) * sw), ((cen[j, 0] - x0) * sw, BIG_H - (cen[j, 1] - y0) * sw)], fill=(255, 255, 255), width=3)
    # the whole slab, a dot (a 1-3 px stamp) per body, near over far
    sx = INSET_W / (hi[0] - lo[0])
    inset = np.tile(ground.astype(np.uint8), (INSET_H, INSET_W, 1))
    order = np.argsort(-depth, kind="stable")
    rp = float(np.max(radius)) * sx
    k = int(np.ceil(rp - 0.5))
    px = ((cen[order, 0] - lo[0]) * sx).astype(np.int64)
    py = (INSET_H - 1 - ((cen[order, 1] - lo[1]) * sx)).astype(np.int64)
    fo = np.clip(fill[order], 0, 255).astype(np.uint8)
    for dx in range(-k, k + 1):
        for dy in range(-k, k + 1):
            if dx * dx + dy * dy > max(rp, 0.5) ** 2:
                continue
            xx, yy = px + dx, py + dy
            ok = (xx >= 0) & (xx < INSET_W) & (yy >= 0) & (yy < INSET_H)
            inset[yy[ok], xx[ok]] = fo[ok]
    frame = Image.new("RGB", (BIG_W, BIG_H + 8), tuple(int(v) for v in ground))
    frame.paste(main, (0, 4))
    ix, iy = BIG_W - INSET_W - 24, 24
    frame.paste(Image.fromarray(inset), (ix, iy))
    dr = ImageDraw.Draw(frame)
    dr.rectangle([ix - 2, iy - 2, ix + INSET_W + 1, iy + INSET_H + 1], outline=(255, 255, 255), width=2)
    wx0, wx1 = ix + (x0 - lo[0]) * sx, ix + (x1 - lo[0]) * sx
    wy0, wy1 = iy + INSET_H - 1 - (y1 - lo[1]) * sx, iy + INSET_H - 1 - (y0 - lo[1]) * sx
    dr.rectangle([wx0 - 1, wy0 - 1, max(wx1 + 1, wx0 + 3), max(wy1 + 1, wy0 + 3)], outline=(255, 255, 255), width=2)
    _big_bar(frame, ix + 14, iy + INSET_H - 10, INSET_W, sx, size=INSET_PX)
    _big_bar(frame, 24, frame.height - 24, BIG_W, sw)
    if text:
        font = _font()
        l, t, r, b = dr.textbbox((0, 0), text, font=font)
        dr.text((frame.width - 24 - (r - l) - l, frame.height - 24 - (b - t) - t), text, fill=(255, 255, 255), font=font)
    return np.array(frame)


def big_geometry(key):
    """(lo, hi) of the slab, length units; the radius of each kind's disc (the sphere of its volume: its points x the spacing^3), in the cell set's type order; that order;
    and the window, the size of the scene before it was scaled, at the slab's centre."""
    p = params(key)
    m = float(p["margin"])
    lo, hi = np.full(3, m), np.full(3, m) + np.asarray(p["box"], float)
    names = [nm for nm in SPECIES if int(p["counts"].get(nm, 0)) > 0]
    sp = spacing_of(p)
    pts = {nm: (int(p["coarse"]["points"]) if p.get("coarse") else SPECIES[nm]["n"]) for nm in names}
    radius = np.array([(3.0 * pts[nm] * sp ** 3 / (4.0 * math.pi)) ** (1.0 / 3.0) for nm in names])
    bx, by = (p.get("base_box") or p["box"])[:2]
    c = window_centre(p, lo, hi)
    return lo, hi, names, radius, (c[0] - 0.5 * bx, c[1] - 0.5 * by, c[0] + 0.5 * bx, c[1] + 0.5 * by)


def window_centre(p, lo, hi):
    """The window's centre: the slab's, or for a coarse scene the centre of the seed tile just above and right of it (the slab's centre is a corner of four tiles, where
    their seams cross until the warm-up has filled them)."""
    c = 0.5 * (lo + hi)
    if p.get("coarse"):
        c = c + 0.5 * np.array([(hi[0] - lo[0]) / TILES[0], (hi[1] - lo[1]) / TILES[1], 0.0])
    return c


def big_seed_png(key):
    """Frame 0 of a big scene drawn from its seed table (the bodies' centres before any frame is run): data/first_<key>.png."""
    p = params(key)
    path, got, share = seed_table(key)
    z = np.load(os.path.join(ROOT, path))
    X = np.asarray(z["pos"], np.float64) / UM
    cen = np.nanmean(X, axis=1)
    lo, hi, names, radius, win = big_geometry(key)
    kind = np.repeat(np.arange(len(names)), [int(got[nm]) for nm in names])
    pts = None
    if p.get("record_points", True):                                                     # the bodies in the window drawn by their points, as the movie will
        x0, y0, x1, y1 = win
        inw = np.flatnonzero((cen[:, 0] > x0 - 2) & (cen[:, 0] < x1 + 2) & (cen[:, 1] > y0 - 2) & (cen[:, 1] < y1 + 2))
        pts = {int(i): X[i][np.isfinite(X[i, :, 0])] for i in inw}
    img = big_frame(cen, kind, names, lo, hi, radius, win, text=f"{len(cen):,} bodies   t = 0 \u00b5s", pts=pts,
                    pt_r=PT_R * spacing_of(p))
    from PIL import Image
    out = os.path.join(DATA, f"first_{key}.png")
    Image.fromarray(img).save(out)
    print(f"  {key}: {len(cen):,} bodies {dict(got)}; {share * 100:.1f} % of a slab {p['box'][0] * NM:,.0f} x {p['box'][1] * NM:,.0f} x {p['box'][2] * NM:.0f} nm -> {os.path.relpath(out, ROOT)}")
    return out


def _chunk_files(run):
    d = os.path.join(GD, run, "chunks")
    return sorted(f for f in os.listdir(d) if f.startswith("rows_") and f.endswith(".npz") and ".partial" not in f) if os.path.isdir(d) else []


def render_chunk(key, fname, run=None):
    """The rows of one chunk file of a big scene drawn (big_frame) into its own movie part, data/big_<key>/<chunk>.mp4 at the iteration's frame rate, and the part's
    last frame as a PNG beside it. A body in the window is drawn by its points when the points are recorded; the bonds (bound share above 0.5) when the junctions are.
    Returns (part mp4, part png, numbers of the chunk's last row)."""
    import imageio.v2 as iio
    run = run or run_name(key)
    p = params(key)
    cd = os.path.join(GD, run, "chunks")
    meta = np.load(os.path.join(cd, "meta.npz"))
    z = np.load(os.path.join(cd, fname))
    lo, hi, names, radius, win = big_geometry(key)
    nt = np.rint(meta["cell__node_type"]).astype(np.int64)
    tn = [str(x) for x in meta["cell__type_names"]]
    kind = np.array([names.index(tn[t]) for t in range(len(tn))], np.int64)[nt]                # the cell set's type -> this tool's kind order (the same: SPECIES order)
    frames = z["frames"]
    od = os.path.join(DATA, f"big_{key}")
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, fname[:-4] + ".mp4")
    writer = iio.get_writer(out, fps=int(p.get("movie_fps", 25)), quality=8, macro_block_size=16)
    pt_r = PT_R * spacing_of(p)
    x0, y0, x1, y1 = win
    last = None
    for r in range(len(frames)):
        occ = z["cell__occ"][r].astype(bool)
        cen = z["cell__pos"][r].astype(np.float64)
        alive = np.flatnonzero(occ)
        pts = None
        if "pt__pos" in z.files:
            inw = alive[(cen[alive, 0] > x0 - 2) & (cen[alive, 0] < x1 + 2) & (cen[alive, 1] > y0 - 2) & (cen[alive, 1] < y1 + 2)]
            po = z["pt__occ"][r].astype(bool)
            pid = np.rint(z["pt__pid"][r][:, 0]).astype(np.int64)
            sel = np.flatnonzero(po & np.isin(pid, inw))
            sel = sel[np.argsort(pid[sel], kind="stable")]
            X = z["pt__pos"][r][sel].astype(np.float64)
            cut = np.flatnonzero(np.diff(pid[sel])) + 1
            pts = {int(pid[sel[g[0]]]): X[g] for g in np.split(np.arange(len(sel)), cut) if len(g)}       # body -> its live points, the rows of X in pid order
        bonds = None
        if "junction__rho" in z.files:
            jm = z["junction__occ"][r].astype(bool) & (z["junction__rho"][r][:, 0] > 0.5)
            bonds = np.stack([np.rint(z["junction__ca"][r][jm, 0]), np.rint(z["junction__cb"][r][jm, 0])], 1).astype(np.int64)
        idx = alive
        img = big_frame(cen[idx], kind[idx], names, lo, hi, radius, win, bonds=None if bonds is None else np.searchsorted(idx, bonds),
                        text=f"{len(idx):,} bodies   t = {int(frames[r]) * TIME_US:.0f} \u00b5s",
                        pts=None if pts is None else {int(np.searchsorted(idx, b)): q for b, q in pts.items()}, pt_r=pt_r)
        writer.append_data(img)
        last = img
    writer.close()
    from PIL import Image
    Image.fromarray(last).save(out[:-4] + ".png")
    ms = z["frame_ms"]
    num = dict(frame=int(frames[-1]), bodies=int(len(idx)), bonds=None if bonds is None else int(len(bonds)),
               s_per_frame=float(np.median(ms[-min(len(ms), 100):]) / 1000.0) if len(ms) else None)
    return out, out[:-4] + ".png", num


def _concat(parts, out):
    """The movie parts one after the other into `out`, without re-encoding (ffmpeg's concat demuxer: the parts share one encoder's settings)."""
    import imageio_ffmpeg
    lst = out + ".txt"
    open(lst, "w").write("".join(f"file '{os.path.abspath(q)}'\n" for q in parts))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", out], check=True)
    os.remove(lst)


def _put_step(key, png, mp4, why, drawn, i=None):
    """A record step for `key` (a NEW one, numbered by the record, when `i` is None; else INTO step i), under the record's lock: png, mp4, the spec, the why text.
    Returns the step's index."""
    import fcntl
    import shutil
    os.environ["PLEXUS_BUILDER"] = RECORD
    import gui_drive as gd
    with open(os.path.join(DATA, ".render.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        new = i is None
        i = gd._next_index() if new else int(i)
        shutil.copy(png, gd._path("png", i))
        shutil.copy(mp4, gd._path("mp4", i))
        sp = os.path.join(GD, run_name(key), "spec.yaml")
        shutil.copy(sp if os.path.exists(sp) else os.path.join(ROOT, "config", "tissue", run_name(key) + ".yaml"), gd._path("spec", i))
        with open(gd._path("why", i), "w") as f:
            f.write(f"time   {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"run    tissue/{run_name(key)} (mpm; drawn from the bodies' centres and, in the window, their points: tools/exp09_landscape.py render_chunk)\n")
            f.write(f"drawn  {drawn}\n")
            f.write(f"why    {why}\n")
        if new:
            gd.note(f"  exp09_landscape {i:04d}  tissue/{run_name(key)}  -- {why_of(key)}")
            sj = os.path.join(DATA, "steps.json")
            json.dump(dict(json.load(open(sj)) if os.path.exists(sj) else {}, **{key: i}), open(sj, "w"), indent=1)
    return i


def live(keys, hours=48.0, poll_s=60.0, record=True):
    """While the big runs of `keys` go: every new chunk file (100 frames) is drawn into a movie part, the parts so far joined into data/big_<key>.mp4, and that movie put
    into the watcher -- a new step at the first chunk, the same step redrawn at every later one. Ends when every run's job wrote `done` (or failed) and its last chunk is
    drawn, or after `hours`."""
    state = {k: {"done": [], "step": None} for k in keys}
    t_end = time.time() + 3600.0 * float(hours)
    while time.time() < t_end:
        busy = False
        for k in keys:
            run = run_name(k)
            for f in [f for f in _chunk_files(run) if f not in state[k]["done"]]:
                t0 = time.time()
                part, png, num = render_chunk_vtk(k, f) if params(k).get("window") else render_chunk(k, f)
                state[k]["done"].append(f)
                parts = [os.path.join(DATA, f"big_{k}", g[:-4] + ".mp4") for g in state[k]["done"]]
                mp4 = os.path.join(DATA, f"big_{k}.mp4")
                _concat(parts, mp4)
                p = params(k)
                stat = (f"RUNNING: frame {num['frame']:,} of {int(p['frames']):,}" if num["frame"] < int(p["frames"]) else f"frame {num['frame']:,} of {int(p['frames']):,}")
                why = (f"{why_of(k)}. {stat}; {num['bodies']:,} bodies" + (f", {num['bonds']:,} bonds held in the window" if num["bonds"] is not None else "")
                       + (f"; {num['s_per_frame']:.1f} s per frame" if num["s_per_frame"] else ""))
                if record:
                    state[k]["step"] = _put_step(k, png, mp4, why, f"chunks {len(state[k]['done'])}, frames 0 to {num['frame']} of {int(p['frames'])}", state[k]["step"])
                print(f"[live] {k}: {f} drawn in {time.time() - t0:.0f} s -> step {state[k]['step']} ({stat})", flush=True)
                busy = True
        ended = []
        for k in keys:
            d = os.path.join(LOG, run_name(k))
            failed = any(os.path.exists(os.path.join(d, q)) and re.search(r"Exited with exit code|TERM_[A-Z]+", open(os.path.join(d, q), errors="replace").read())
                         for q in ("run.out", "run.err"))
            ended.append(os.path.exists(os.path.join(d, "done")) or failed)
        if all(ended) and not busy:
            print("[live] every run ended and its chunks are drawn", flush=True)
            return 0
        time.sleep(poll_s)
    print("[live] out of time", flush=True)
    return 2


def big_first(key, factor=2.0, reseed=False):
    """Frame 0 of a big scene in movie 21's look: the window of `factor` x the scene's size before scaling (2: 384 x 216 nm), at the slab's centre, drawn by `render`
    (every body's iso-surface, lit, flat-ink outlines) from the seed's points, and the whole slab as big_frame's inset (a dot per body) over its top right corner;
    data/first_<key>.png. The window's bodies are written as a one-row run graphs_data/tissue/exp09_first_<key> (spec + trajectory.npz) for `render` to read."""
    global MIN_PTS, CLOCK
    import shutil
    from PIL import Image
    p = params(key)
    path = os.path.join(DATA, f"seed_{key}.npz")
    if reseed or not os.path.exists(path) or "kind" not in np.load(path).files:
        seed_table(key)
    z = np.load(path)
    X = np.asarray(z["pos"], np.float64) / UM
    kind = np.asarray(z["kind"], np.int64)
    cen = np.nanmean(X, axis=1)
    lo, hi, names, radius, _ = big_geometry(key)
    bx, by = (p.get("base_box") or p["box"])[:2]
    c = window_centre(p, lo, hi)
    win = (c[0] - 0.5 * factor * bx, c[1] - 0.5 * factor * by, c[0] + 0.5 * factor * bx, c[1] + 0.5 * factor * by)
    sel = np.flatnonzero((cen[:, 0] > win[0] - 1.5) & (cen[:, 0] < win[2] + 1.5) & (cen[:, 1] > win[1] - 1.5) & (cen[:, 1] < win[3] + 1.5))
    m = X.shape[1]
    Xs = X[sel]
    occ = np.isfinite(Xs[..., 0]).reshape(1, -1)
    run = f"exp09_first_{key}"
    d = os.path.join(GD, run)
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "trajectory.npz"), pt__pos=np.nan_to_num(Xs).reshape(1, -1, 3).astype(np.float32), pt__occ=occ,
             pt__pid=(np.arange(len(sel) * m) // m).astype(np.float32).reshape(1, -1, 1), cell__occ=np.ones((1, len(sel)), bool), cell__pos=cen[sel][None].astype(np.float32),
             cell__node_type=kind[sel].astype(np.int64), cell__type_names=np.array(names))
    real = seed_table
    globals()["seed_table"] = lambda k: (os.path.relpath(path, ROOT), {nm: int((kind == i).sum()) for i, nm in enumerate(names)}, 0.19)
    try:
        sp = build(key)
    finally:
        globals()["seed_table"] = real
    for o in sp["operators"]:
        if o["op"] == "active_force" and o.get("model") == "box_slab":
            o["lo"], o["hi"] = [float(win[0]), float(win[1]), float(lo[2])], [float(win[2]), float(win[3]), float(hi[2])]
    sp["general"].update(name=run, record_cap=2, n_frames=1)
    yaml.safe_dump(sp, open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False, width=1000)
    MIN_PTS, CLOCK = (min(16, int(p["coarse"]["points"])) if p.get("coarse") else 16), f"{len(cen):,} bodies   t = 0 \u00b5s"
    try:
        render(key, run=run, name=f"firstvtk_{key}", last_only=True)
    finally:
        MIN_PTS, CLOCK = 16, None
    main = Image.open(os.path.join(DATA, f"firstvtk_{key}.png")).convert("RGB")
    ins = big_frame(cen, kind, names, lo, hi, radius, win)                                 # only its inset is kept
    ix, iy = BIG_W - INSET_W - 24, 24
    frame = main.copy()
    frame.paste(Image.fromarray(ins).crop((ix - 2, iy - 2, ix + INSET_W + 2, iy + INSET_H + 2)), (ix - 2, iy - 2))
    out = os.path.join(DATA, f"first_{key}.png")
    frame.save(out)
    shutil.rmtree(d, ignore_errors=True)
    print(f"  {key}: window {factor * bx * NM:.0f} x {factor * by * NM:.0f} nm, {len(sel):,} of {len(cen):,} bodies drawn -> {os.path.relpath(out, ROOT)}")
    return out


def big_window(key, factor=None):
    """(x0, y0, x1, y1), length units: the window of a big scene, `factor` (default the scene's `window`, else 1) x the scene's size before scaling, at window_centre."""
    p = params(key)
    m = float(p["margin"])
    lo, hi = np.full(3, m), np.full(3, m) + np.asarray(p["box"], float)
    f = float(factor if factor is not None else (p.get("window") or 1.0))
    bx, by = (p.get("base_box") or p["box"])[:2]
    c = window_centre(p, lo, hi)
    return (float(c[0] - 0.5 * f * bx), float(c[1] - 0.5 * f * by), float(c[0] + 0.5 * f * bx), float(c[1] + 0.5 * f * by))


def big_inset(cen, kind, names, lo, hi, radius, win):
    """The whole slab, INSET_W x INSET_H: a dot (a 1-3 px stamp) per body in its kind's colour mixed toward the ground with depth, near over far, a white frame round
    the window -- the picture big_frame puts top right, with its frame and scale bar, cropped to its frame (INSET_W + 4 by INSET_H + 4 px)."""
    full = big_frame(cen, kind, names, lo, hi, radius, win)
    ix, iy = BIG_W - INSET_W - 24, 24
    return full[iy - 2:iy + INSET_H + 2, ix - 2:ix + INSET_W + 2]                  # big_frame pastes the inset at (ix, iy) of the frame, its white frame 2 px round it


def _window_run(key, fname, rows, run_dir):
    """The window's rows `rows` of one chunk written as a run `render` reads: trajectory.npz with the window's bodies (pt__pos / occ / pid, cell__occ / pos / node_type,
    junction__ca / cb / rho / occ in the window's own body numbers) and spec.yaml, the run's own with its walls moved to the window. Returns the chunk's frames of those rows."""
    p = params(key)
    cd = os.path.join(GD, run_name(key), "chunks")
    meta = np.load(os.path.join(cd, "meta.npz"))
    z = np.load(os.path.join(cd, fname))
    off = z["window__offsets"]
    par_all, pos_all = z["window__parent"], z["window__pos"]
    bods = np.unique(np.concatenate([par_all[off[r]:off[r + 1]] for r in rows]))
    nb = len(bods)
    K = max(int(off[r + 1] - off[r]) for r in rows)
    R = len(rows)
    P = np.zeros((R, K, 3), np.float32); O = np.zeros((R, K), bool); D = np.zeros((R, K, 1), np.float32)
    CO = np.zeros((R, nb), bool)
    CP = np.zeros((R, nb, 3), np.float32)
    cpos = z["cell__pos"]
    e_on = "window__edge_offsets" in z.files
    J = max((int(z["window__edge_offsets"][r + 1] - z["window__edge_offsets"][r]) for r in rows), default=0) if e_on else 0
    JA = np.zeros((R, max(J, 1), 1), np.float32); JB = np.zeros_like(JA); JR = np.zeros_like(JA); JO = np.zeros((R, max(J, 1)), bool)
    for i, r in enumerate(rows):
        par = np.searchsorted(bods, par_all[off[r]:off[r + 1]])
        k = len(par)
        P[i, :k] = pos_all[off[r]:off[r + 1]]; O[i, :k] = True; D[i, :k, 0] = par
        CO[i, np.unique(par)] = True
        CP[i] = cpos[r][bods]
        if e_on:
            eo = z["window__edge_offsets"]
            a, b, rho = z["window__edge_a"][eo[r]:eo[r + 1]], z["window__edge_b"][eo[r]:eo[r + 1]], z["window__edge_rho"][eo[r]:eo[r + 1]]
            n = len(a)
            JA[i, :n, 0], JB[i, :n, 0], JR[i, :n, 0], JO[i, :n] = np.searchsorted(bods, a), np.searchsorted(bods, b), rho[:, 0] if rho.ndim == 2 else rho, True
    nt = np.rint(meta["cell__node_type"]).astype(np.int64)[bods]
    os.makedirs(run_dir, exist_ok=True)
    np.savez(os.path.join(run_dir, "trajectory.npz"), pt__pos=P, pt__occ=O, pt__pid=D, cell__occ=CO, cell__pos=CP, cell__node_type=nt,
             cell__type_names=meta["cell__type_names"], **({"junction__ca": JA, "junction__cb": JB, "junction__rho": JR, "junction__occ": JO} if e_on else {}))
    sp = os.path.join(GD, run_name(key), "spec.yaml")
    spec = yaml.safe_load(open(sp if os.path.exists(sp) else os.path.join(ROOT, "config", "tissue", run_name(key) + ".yaml")))
    x0, y0, x1, y1 = big_window(key)
    for o in spec["operators"]:
        if o["op"] == "active_force" and o.get("model") == "box_slab":
            o["lo"], o["hi"] = [x0, y0, float(o["lo"][2])], [x1, y1, float(o["hi"][2])]
    spec["general"].update(name=os.path.basename(run_dir), n_frames=max(1, R - 1), record_cap=R)
    yaml.safe_dump(spec, open(os.path.join(run_dir, "spec.yaml"), "w"), sort_keys=False, width=1000)
    return z["frames"][list(rows)]


def render_piece(key, fname, i, n):
    """Piece i of n of one chunk (its rows dealt in order, piece i the i-th run of them) drawn by `render` -- the window's bodies as step 21's are drawn, the clock at the
    chunk's own frames, the whole slab's inset over each frame -- into data/big_<key>/<chunk>_<i>.mp4."""
    global CLOCK, FRAME_HOOK, MIN_PTS
    from PIL import Image
    import shutil
    p = params(key)
    z = np.load(os.path.join(GD, run_name(key), "chunks", fname))
    nr = len(z["frames"])
    rows = list(range(nr))[i * nr // n:(i + 1) * nr // n]
    if not rows:
        return None
    run = f"exp09_live_{key}_{fname[:-4]}_{i}"
    frames = _window_run(key, fname, rows, os.path.join(GD, run))
    meta = np.load(os.path.join(GD, run_name(key), "chunks", "meta.npz"))
    lo, hi, names, radius, _ = big_geometry(key)
    win = big_window(key)
    tn = [str(x) for x in meta["cell__type_names"]]
    kind_all = np.array([names.index(t) for t in tn], np.int64)[np.rint(meta["cell__node_type"]).astype(np.int64)]
    cpos, cocc = z["cell__pos"], z["cell__occ"]
    nall = int(cocc[rows[0]].sum())
    ix, iy = BIG_W - INSET_W - 24, 24

    def hook(k, frame):
        r = rows[k]
        alive = np.flatnonzero(cocc[r])
        ins = big_inset(cpos[r][alive].astype(np.float64), kind_all[alive], names, lo, hi, radius, win)
        im = Image.fromarray(frame)
        im.paste(Image.fromarray(ins), (ix - 2, iy - 2))
        return np.array(im)

    CLOCK = lambda k: f"{nall:,} bodies   t = {int(frames[k]) * TIME_US:.0f} \u00b5s"           # noqa: E731
    FRAME_HOOK = hook
    MIN_PTS = min(16, int(p["coarse"]["points"])) if p.get("coarse") else 16
    name = f"big_{key}_{fname[:-4]}_{i}"
    try:
        info = render(key, run=run, name=name, max_frames=10 ** 6, fps=int(p.get("movie_fps", 25)))
    finally:
        CLOCK, FRAME_HOOK, MIN_PTS = None, None, 16
        shutil.rmtree(os.path.join(GD, run), ignore_errors=True)
    od = os.path.join(DATA, f"big_{key}")
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, f"{fname[:-4]}_{i}.mp4")
    os.replace(os.path.join(ROOT, info["mp4"]), out)
    os.replace(os.path.join(ROOT, info["png"]), out[:-4] + ".png")
    return out


def render_chunk_vtk(key, fname, procs=4):
    """One chunk drawn by `procs` processes at once (render_piece, each its share of the rows), the pieces joined into data/big_<key>/<chunk>.mp4 and the last piece's
    last frame as its PNG. Returns (mp4, png, numbers of the chunk's last row)."""
    import shutil
    od = os.path.join(DATA, f"big_{key}")
    os.makedirs(od, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]))
    jobs = [subprocess.Popen([sys.executable, "-u", __file__, "render_piece", key, fname, "--piece", f"{i}/{procs}"], env=env, cwd=ROOT,
                             stdout=subprocess.DEVNULL, stderr=open(os.path.join(od, f"{fname[:-4]}_{i}.log"), "w")) for i in range(procs)]
    bad = [i for i, j in enumerate(jobs) if j.wait() != 0]
    if bad:
        raise RuntimeError(f"render_piece {key} {fname}: pieces {bad} failed (see {od}/{fname[:-4]}_<i>.log)")
    parts = [q for q in (os.path.join(od, f"{fname[:-4]}_{i}.mp4") for i in range(procs)) if os.path.exists(q)]
    out = os.path.join(od, fname[:-4] + ".mp4")
    _concat(parts, out)
    shutil.copy(parts[-1][:-4] + ".png", out[:-4] + ".png")
    z = np.load(os.path.join(GD, run_name(key), "chunks", fname))
    ms = z["frame_ms"]
    bonds = None
    if "window__edge_offsets" in z.files:
        eo = z["window__edge_offsets"]
        rho = z["window__edge_rho"][eo[-2]:eo[-1]]
        bonds = int(((rho[:, 0] if rho.ndim == 2 else rho) > 0.5).sum())
    num = dict(frame=int(z["frames"][-1]), bodies=int(z["cell__occ"][-1].sum()), bonds=bonds, s_per_frame=float(np.median(ms[-min(len(ms), 100):]) / 1000.0) if len(ms) else None)
    return out, out[:-4] + ".png", num


def big_motion(key):
    """How far a big scene's bodies move: per stored row (median, 99th percentile, largest, nm) and from the first row to the last, read from its chunk files."""
    d = os.path.join(GD, run_name(key), "chunks")
    fs = _chunk_files(run_name(key))
    c = np.concatenate([np.load(os.path.join(d, f))["cell__pos"] for f in fs]).astype(np.float64)
    fr = np.concatenate([np.load(os.path.join(d, f))["frames"] for f in fs])
    step = np.linalg.norm(np.diff(c, axis=0), axis=2) * NM
    tot = np.linalg.norm(c[-1] - c[0], axis=1) * NM
    ms = np.load(os.path.join(d, fs[-1]))["frame_ms"]
    print(f"{key}: per stored row ({int(fr[1] - fr[0])} frames) median {np.median(step):.2f} p99 {np.percentile(step, 99):.2f} max {step.max():.1f} nm | frame {int(fr[0])} -> {int(fr[-1])}: "
          f"median {np.median(tot):.1f} p99 {np.percentile(tot, 99):.1f} max {tot.max():.1f} nm | {np.median(ms[2:]) if len(ms) > 3 else float('nan'):.0f} ms per frame")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["list", "write", "measure", "after", "render", "look", "shapes", "first", "live", "render_chunk", "render_piece", "drawall", "motion", "render_part"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--run", default=None, help="measure, render: another run folder of graphs_data/tissue (a smoke)")
    ap.add_argument("--name", default=None, help="render: the output's name in the record's data/ (default the run's)")
    ap.add_argument("--last-only", action="store_true", help="render: only the last stored row, as a PNG (the preview of a running job)")
    ap.add_argument("--max-frames", type=int, default=300, help="render: at most this many of the stored rows are drawn (a 1,000-frame run stores 251: all of them, 10 s at 25 per second)")
    ap.add_argument("--fps", type=int, default=25, help="render: frames per second of the movie (250 rows at 25: 10 s)")
    ap.add_argument("--replace", type=int, default=None, help="after, after_job, chain: redraw (chain: rerun and redraw) INTO this step")
    ap.add_argument("--no-record", action="store_true", help="live: draw the chunks, do not put them into the watcher (a smoke)")
    ap.add_argument("--piece", default="0/1", help="render_piece: i/n, this piece of n")
    a = ap.parse_args()
    keys = a.names or list(ITERS)
    if a.cmd == "list":
        for k in ITERS:
            print(k, {kk: v for kk, v in params(k).items() if BASE.get(kk) != v} or "(the base)")
    elif a.cmd == "write":
        for k in keys:
            write(k)
    elif a.cmd == "measure":
        for k in keys:
            print(json.dumps(measure(k, run=a.run), indent=1))
    elif a.cmd == "render":
        render(keys[0], run=a.run, name=a.name, last_only=a.last_only, max_frames=a.max_frames, fps=a.fps)
    elif a.cmd == "shapes":
        print(os.path.relpath(shapes(a.names), ROOT))
    elif a.cmd == "look":
        for k in keys:
            look(k)
    elif a.cmd == "after":
        sys.exit(after(keys[0], a.replace))
    elif a.cmd == "first":
        for k in keys:
            big_first(k)
    elif a.cmd == "live":
        sys.exit(live(keys, record=not a.no_record))
    elif a.cmd == "render_part":
        i, n = (int(v) for v in a.piece.split("/"))
        print(render_part(keys[0], i, n, a.fps))
    elif a.cmd == "motion":
        for k in keys:
            big_motion(k)
    elif a.cmd == "drawall":                                       # every chunk of a run drawn and joined (a smoke: no watcher)
        for k in keys:
            parts = [render_chunk_vtk(k, f)[0] for f in _chunk_files(run_name(k))]
            _concat(parts, os.path.join(DATA, f"big_{k}.mp4"))
            print(k, len(parts), "chunks ->", os.path.relpath(os.path.join(DATA, f"big_{k}.mp4"), ROOT))
    elif a.cmd == "render_piece":
        i, n = (int(v) for v in a.piece.split("/"))
        print(render_piece(keys[0], keys[1], i, n))
    elif a.cmd == "render_chunk":
        print(render_chunk(keys[0], keys[1], run=a.run))
