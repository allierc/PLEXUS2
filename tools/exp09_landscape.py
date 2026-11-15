"""exp09 molecular_landscape: movies of a crowded molecular landscape in the flat-ink look -- flat fills, every body outlined in its own colour darkened, far bodies
paler -- with every molecule a body of MPM material points (Cedric, 2026-11-14: "generate goodsell like movies with Plexus ... 1000 frames using MPM operators";
the look is inspired by D. Goodsell's cell cross-sections, https://ccsb.scripps.edu/goodsell/, and is not a copy of them: Cedric's own vivid palette, a grey ground).

    python tools/exp09_landscape.py list
    python tools/exp09_landscape.py write it01                # -> config/tissue/exp09_land_it01.yaml (+ its seed table in the record's data/)
    python tools/exp09_landscape.py measure it01              # the run's numbers, printed and saved in the record's data/
    python tools/exp09_landscape.py after it01                # a landed run into the record: measure, why text, the movie as a new watcher step
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
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]

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
}
NMAX = max(s["n"] for s in SPECIES.values())             # point slots per body: the largest body's count

# ---- the base: every iteration is this plus its own changes --------------------------------------------------------------------------------------------------------
BASE = dict(
    frames=1000, every=4, seed=1,
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
    vesicle=None,              # a closed membrane seen as a ring: dict(x, y = its centre from the slab's left and floor, r = its radius, every = one site in this many a protein with its head
                               # outward, proteins = their kinds in turn), `lipid` patches round a cylinder along z; see vesicle_bodies
    chains=None,               # [dict(kind = a block species, n = how many, m = subunits each, bend = 0, band = None, tilt = None)]: rows of subunits lying in the slab's plane, straight (bend 0) or
                               # meandering, anywhere, on their species' side of a membrane (`region`), or between the two heights `band` above the slab's floor; at any angle in
                               # the plane, or within `tilt` radians of the x axis; see chain_groups. Or `rows` = [(height above the floor, depth), ...] in place of n: a straight
                               # chain along x at each of those places (an ordered array; of one kind, the chains with `rows` are listed before those without). `riders` =
                               # dict(kind, every): a body of that kind seated on one subunit in `every` and tied to it by the bonds; see _riders
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
}


def params(key):
    """BASE, then every ancestor's changes from the root down, then the iteration's own: an iteration names only its ONE change from its parent."""
    it = ITERS[key]
    p = params(it["parent"]) if it.get("parent") else dict(BASE)
    p.update({kk: v for kk, v in it.items() if kk not in ("why", "parent")})
    return p


def run_name(key):
    return f"exp09_land_{key}"           # NOT exp09_<key>: graphs_data/tissue/exp09_p2* are the runs of an earlier experiment 9 (TRAPS T-90)


# ---- the bodies' points and their packing --------------------------------------------------------------------------------------------------------------------------
def body_points(shape, n):
    """[n, 3] points of one body about its own centroid, length units: the `n` points of a lattice of spacing SPACING with the lowest score of the shape (see SPECIES)."""
    kind, par = shape
    g = np.arange(-9, 10) * SPACING
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


def pack(counts, lo, hi, gap, seed, tries=4000, fixed=(), yrange=None, groups=(), where=None, group_band=None, group_tilt=None):
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
    todo = [(name, body_points(SPECIES[name]["shape"], SPECIES[name]["n"])) for name, c in counts.items() for _ in range(int(c))]
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
            e = np.abs(P).max(0) + 0.5 * SPACING
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
            e = np.abs(P).max(0) + 0.5 * SPACING
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


def seed_table(key):
    """The seed table of `key` for seed_positions[from_points_ragged] (pos in micrometres, NaN rows where a body has fewer than NMAX points; bodies in SPECIES
    order, the cell set's type order). Returns (path relative to the repo, {species: bodies placed}, the bodies' share of the slab's volume)."""
    p = params(key)
    m = float(p["margin"])
    lo, hi = np.full(3, m), np.full(3, m) + np.asarray(p["box"], float)
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
    struct = [float(i < n_struct) for nm in names for i, (q, _) in enumerate(placed) if q == nm]   # 1 for a membrane body or a chain subunit, 0 for a body placed at random
    got = {nm: sum(1 for q, _ in placed if q == nm) for nm in names}
    pos = np.full((len(bodies), NMAX, 3), np.nan)
    for k, X in enumerate(bodies):
        pos[k, :len(X)] = X * UM
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, f"seed_{key}.npz")
    np.savez(out, pos=pos, volume_um3=np.array([len(X) * PV * UM ** 3 for X in bodies]), value=np.array(struct))     # `value`: written to the cell block `bonded` when the scene has bonds
    share = sum(len(X) for X in bodies) * PV / float(np.prod(p["box"]))
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
    return {
        "general": {
            "name": name,
            "headline": f"exp09 {key}: a crowded molecular landscape of {n} MPM bodies ({share * 100:.0f} % of a slab {box[0] * NM:.0f} x {box[1] * NM:.0f} x {box[2] * NM:.0f} nm)",
            "title": ITERS[key]["why"],
            "seed": int(p["seed"]), "n_frames": int(p["frames"]), "dt": 1.0, "record_cap": int(p["frames"]) // int(p["every"]) + 1,
            "save_every_min": 10.0,        # the rows recorded so far written to trajectory.npz every 10 wall-clock minutes, so a running job can be previewed (Cedric, 2026-11-14)
            "boundary": "free", "dim": 3, "world": W, "units": {"length_um": UM, "time_s": TIME_US * 1e-6},
        },
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": 15, "per_parent": "cell", "cell_box": 3.0}},
        "sets": {
            "cell": {
                "n": n, "type_layout": "ordered",
                "types": {nm: {"count": int(got[nm]), "shape": "ball", "material": "elastic", "youngs": float((p.get("youngs_of") or {}).get(nm, p["youngs"])), "density": 1.0} for nm in names},
                "state": {
                    "pos": {"width": 3, "role": "coordinate", "integration": "coordinate", "boundary": "world"},
                    "vel": {"width": 3, "role": "rate", "integration": "rate", "record": False},
                    **({"bonded": {"width": 1, "record": False}} if p.get("bonds") else {}),
                },
            },
            "pt": {
                "entity": "mpm_particle", "parent": "cell", "per_parent": NMAX, "density": 1.0, "radius": 0.5,
                "state": {
                    "pos": {"width": 3, "role": "coordinate", "integration": "coordinate", "boundary": "world"},
                    "vel": {"width": 3, "role": "rate", "integration": "rate", "record": False},
                    "pid": {"width": 1}, "cellc": {"width": 1, "record": False},
                },
            },
        },
        "operators": [
            {"op": "thermal_bath", "implementation": "langevin", "at": "pt", "friction": float(p["friction"]), "kT": float(p["kT"]), "substep_dt": sub, "seed": int(p["seed"])},
            {"op": "active_force", "model": "box_slab", "at": "pt", "wall_stiffness": float(p["wall"]), "lo": [m, m, m], "hi": [m + b for b in box]},
            {"op": "pair_potential", "model": "cell_contact", "at": "pt", "cells": "cell", "repulsion_strength": float(p["rep"]), "contact_slack": float(p["slack"]),
             "contact_surface_tension": [[float(v) for v in row] for row in g], "contact_edge_width": 0.2},
            *([{"op": "apical_junction", "at": "pt", "cells": "cell", "k": float(p["bonds"]["k"]), "reach": float(p["bonds"]["reach"]), "shell": 1.0e9, "skip_block": "bonded"}] if p.get("bonds") else []),
            {"op": "mpm_strain", "at": "pt", "implementation": "default"},
            {"op": "mpm_scatter", "at": "mpm_grid", "from": "pt", "drag": 0.0, "implementation": "default"},      # no grid drag: the bath's friction is the only drag, so the kicks and the drag balance at kT
            # NO cell_surface_tension here (smoke of 2026-11-14 21:05, tension 0.3 on every body with exp06's csf settings): the bodies shrink without end, the smallest
            # fastest (radius of gyration at frame 60: 27 points 0.73 of its start, 64 points 0.89, 125 points 0.94-0.96) -- at one point per grid cell the points of a
            # body can move together inside a cell without the grid's strain seeing it (TRAPS)
            {"op": "mpm_grid_update", "at": "mpm_grid", "implementation": "warp_pp_balanced"},
            {"op": "mpm_gather", "at": "pt", "from": "mpm_grid", "implementation": "default"},
            {"op": "aggregate_centroid", "at": "cell", "child": "pt"},
        ],
        "schedule": [
            {"steps": ["thermal_bath", "active_force", "pair_potential", *(["apical_junction"] if p.get("bonds") else []), "mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"],
             "substep_dt": sub, "capture": False},
            "aggregate_centroid",
        ],
        "seed": [{"op": "seed_positions", "model": "from_points_ragged", "at": "pt", "cell_set": "cell", "points": table, "um_per_unit": UM,
                  "centre_um": [0.0, 0.0, 0.0], "centre": [0.0, 0.0, 0.0], **({"value_block": "bonded"} if p.get("bonds") else {})}],
        "plotting": {
            "renderer": "vtk_points", "subject": "pt", "render_3d": "contour",
            "point_volume": PV,              # the seed names no point_spacing: without this the studio takes 0.125^3 and draws slivers (TRAPS T-100)
            "cell_colours": "type", "colors": {nm: SPECIES[nm]["colour"] for nm in names},
            "surface_opacity": 1.0, "background": "#8d99a6", "foreground": "white", "box_frame": False,
            "camera": {"elev": 90, "azim": -90}, "zoom": 1.0, "duration_s": 10, "slow_motion": 1,
        },
    }


def write(key):
    out = os.path.join(ROOT, "config", "tissue", run_name(key) + ".yaml")
    yaml.safe_dump(build(key), open(out, "w"), sort_keys=False, width=1000)
    print("wrote", os.path.relpath(out, ROOT))
    return out


# ---- reading a run ----------------------------------------------------------------------------------------------------------------------
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
    wall_op = next(o for o in spec["operators"] if o["op"] == "active_force")
    lo, hi = np.asarray(wall_op["lo"], float), np.asarray(wall_op["hi"], float)
    occ = z["pt__occ"].astype(bool)
    pid = np.rint(z["pt__pid"][..., 0]).astype(np.int64)
    X = z["pt__pos"]
    nt = z["cell__node_type"]
    nt = np.rint(nt[-1] if nt.ndim == 2 else nt).astype(np.int64)
    names = [str(x) for x in z["cell__type_names"]]
    ncell = z["cell__occ"].shape[1]
    ba, bb = [], []                                              # the two bodies of every chain bond: a kind's chains are its first n x m bodies, in seeding order (seed_table)
    used = {}
    for ch in (params(key).get("chains") or []) if key in ITERS else []:
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
               escaped=any(q["beyond_wall_nm"] > 3.0 for q in rows),
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
    return (f"EXP09 {key} (tissue/{run_name(key)}): {ITERS[key]['why']}. "
            f"RESULT at frame {e['frame']} ({e['t_us']:.0f} us): {e['bodies']} bodies, {e['points']:,} material points; {'HEALTH: ' + '; '.join(bad) if bad else 'healthy (finite, no body torn, none escaped)'}; "
            f"RMS distance of the bodies' centres from their start, nm, per kind: {rms(e)} (at frame {h['frame']}: {rms(h)}); "
            f"shape kept: each body's radius of gyration over its own at the start between {e['rg_ratio'][0]:.2f} and {e['rg_ratio'][2]:.2f} (mean {e['rg_ratio'][1]:.3f}); "
            f"{bond}deepest point beyond a wall of the slab {e['beyond_wall_nm']:.1f} nm. "
            f"Look: flat fills by kind ({fills}), each body outlined in its own colour darkened, far bodies paler.")


# ---- the movie ------------------------------------------------------------------------------------------------------------------------------------------------------
FRAME = (1920, 1088)           # px: 16:9 as the slab, both sides multiples of the encoder's 16-px block
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


def render(key, run=None, name=None, last_only=False, max_frames=300, fps=25):
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
    wall_op = next(o for o in spec["operators"] if o["op"] == "active_force")
    lo, hi = np.asarray(wall_op["lo"], float), np.asarray(wall_op["hi"], float)
    c0 = 0.5 * (lo + hi)
    W, Hh = FRAME
    view, up = S.view_vectors(90.0, -90.0)                       # looking down z on the slab, y up
    dist = 0.5 * (hi[1] - lo[1]) / ZOOM / math.tan(math.radians(VIEW_ANGLE / 2))
    px_per_unit = Hh / (2.0 * dist * math.tan(math.radians(VIEW_ANGLE / 2)))
    wpp = 1.0 / px_per_unit
    near, far = dist - 0.5 * (hi[2] - lo[2]), dist + 0.5 * (hi[2] - lo[2])
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
    h_vox = 0.55 * (R.v ** (1.0 / 3.0))
    iso = float((spec.get("plotting") or {}).get("surface_iso_frac", 0.5))
    rows = R.rows
    ks = [rows - 1] if last_only else sorted(set(range(0, rows, max(1, math.ceil(rows / max_frames)))) | {rows - 1})
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
        for part in np.split(idx, cut):
            c = int(R.pid[k][part[0]])
            if len(part) < 16 or c >= ncell:
                continue
            s = e23._cell_surface(X[part], R.v, h_vox, iso=iso)
            if s is None:
                continue
            s["rgb"] = np.tile(fill[c], (s.n_points, 1))
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
        frame = _bar_nm(img, px_per_unit)
        frame = _clock(frame, f"{len(surfs)} bodies   t = {k * stride_f * float(g.get('dt', 1.0)) * TIME_US:.0f} µs")
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
    info = render(key)
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
        gd.note(f"  exp09_landscape {i:04d}  tissue/{run_name(key)}  -- {'REDRAWN IN PLACE: ' if replace else ''}{ITERS[key]['why']}")
        sj = os.path.join(DATA, "steps.json")
        json.dump(dict(json.load(open(sj)) if os.path.exists(sj) else {}, **{key: i}), open(sj, "w"), indent=1)
    print({"index": i, "key": key})
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["list", "write", "measure", "after", "render", "look", "shapes"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--run", default=None, help="measure, render: another run folder of graphs_data/tissue")
    ap.add_argument("--name", default=None, help="render: the output's name in the record's data/ (default the run's)")
    ap.add_argument("--last-only", action="store_true", help="render: only the last stored row, as a PNG")
    ap.add_argument("--max-frames", type=int, default=300, help="render: at most this many of the stored rows are drawn (a 1,000-frame run stores 251: all of them, 10 s at 25 per second)")
    ap.add_argument("--fps", type=int, default=25, help="render: frames per second of the movie (250 rows at 25: 10 s)")
    ap.add_argument("--replace", type=int, default=None, help="after: redraw INTO this step")
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
