---
number: 5
name: spheroid_from_words
question: can a simulation described in a biologist's own words be run in Plexus as written, and what does each sentence of the description have to become for the movie to show it?
record: experiments/exp05_spheroid_from_words
watcher: http://127.0.0.1:8874/watch?rec=experiments/exp05_spheroid_from_words
---

# exp05 -- a spheroid from a biologist's words

Started 2026-11-14 for Cedric, to show that Plexus is usable by biologists: he gives a plain description, the session writes the
spec, runs it, looks at it, and iterates 10 times, each iteration a step in the watcher.

(The earlier exp05, `molecular_corset`, has no markdown left in this folder; its INDEX.md line is stale.)

## The description (Cedric, verbatim)

> 20 cells move randomly in space
> when cells touch each other they adhere one to another
> when adhesion goes above a threshold they can grow and divide leading to multicellular spheroids

## Sentence -> operator

| sentence | operator in the spec | what it does, in words |
|---|---|---|
| 20 cells | `sets.cell.n: 20`, `seed_positions` mode `ball` | 20 cells, each one point with a 10 um diameter, placed at random in a ball |
| move randomly in space | `thermal_bath` | every cell takes a random step each sub-step (a random walk) |
| when cells touch they adhere | `pair_potential[soft_adhesive]` | two cells closer than their diameter push apart; just beyond contact they pull together (an adhesive shoulder); a strong pull tears them apart |
| when adhesion goes above a threshold they grow and divide | `cell_divide[point_adhesion]` (new, `src/plexus/operators/spheroid_ops.py`) | a cell's adhesion = the number of cells it touches; at or above `min_contacts` it grows (size doubles in ln 2 / `rate`); at twice its birth size it divides in two; below the threshold it does not grow |

## Decisions

| when | decision | why |
|---|---|---|
| 2026-11-14 | point cells (one particle per cell, the centre-based model of Drasdo & Hoehme 2005), not deformable MPM cells | 10 iterations in one session: a point-cell run takes minutes; deformable cells (exp29) take hours each |
| 2026-11-14 | "adhesion" = the number of cells a cell touches; the threshold is a count | the plainest reading of "adhesion goes above a threshold"; a biologist can read `min_contacts: 2` |
| 2026-11-14 | growth = a size that doubles at an exponential rate while the cell adheres; division at twice the birth size | "grow and divide": the cell must grow before it divides, so division is a sizer, not a coin flip |
| 2026-11-14 | 1 length unit = 10 um (one cell diameter), 1 frame = 10 min, 72 h | a cell cycle of about 20 h; cells walk a few um per 10 min |
| 2026-11-14 | a spheroid = a cluster of at least 8 cells linked by touching | the measure in `tools/exp05_loop.py` |
| 2026-11-14 | iterations 11-20 (Cedric: "test the influence of cell surface tension and cell stiffness") use DEFORMABLE cells: 125 material points per cell (MPM), Young's modulus = stiffness, `mpm_grid_update.cell_surface_tension` = surface tension | a point cell has neither a surface nor a shape |
| 2026-11-14 | the MPM series reuses two operators the exp06 session registered today for the same description, `active_force[persistent_walk]` and `cell_grow[mpm_adhesion]` (growth only while at least 15% of the cell's surface touches others), with exp05's own spec and seed (`packed_ball` + `voronoi_parent`, 20 balls 25 um apart) | the registry already holds the rule; a second copy would be a third working point of the same thing |
| 2026-11-14 | MPM time is compressed: mass doubles in 100 min of contact, 400 min per run (about 4 s per frame on one rtx6000) | a 72-h MPM run would take a day per iteration |
| 2026-11-14 | the first MPM sweep runs as one batch of four (surface tension x1/3, x3; stiffness x1/4, x4 around iteration 11), filed in order | four sequential runs would take four times as long; each is still one change from 11 |
| 2026-11-14 | iterations 21-30 (signalling) go back to point cells (fast) on iteration 9's rules, adding an activator-inhibitor pair (Gierer-Meinhardt) exchanged between touching cells; a cell grows only where its activator is high; its fate (`cell_fate`, new) is read from the activator and colours it | growth localised by a Turing pattern is the classic route from a ball to buds and branches |

## Iterations

One row per watcher step (filled as they land). Measures by `tools/exp05_loop.py measure` from the trajectory.

| it | step | spec | change from the previous | result | next |
|---|---|---|---|---|---|
| 1 | 0001 | exp05_spheroid_v01 | first translation of the three sentences | 20 lone cells after 72 h: pairs met (closest centres 5.1 um) but parted within a frame; nothing adhered, nothing grew | the sub-step is too coarse for the 2 um adhesive shoulder |
| 2 | 0002 | exp05_spheroid_v02 | sub-steps 10x finer (0.1 min) | unchanged: 20 lone cells. The sub-steps never acted: a sub-step block keeps only its last sub-step for a non-MPM set (trap T-01, now added for positions) | drop the sub-step block, small frames |
| 2 | 0002 | exp05_spheroid_v02 | sub-steps 10x finer (0.1 min) | the same 20 lone cells: the sub-step block never moves the cells between sub-steps (TRAPS T-99), so its length changed nothing | drop the sub-step block, lower the frame time |
| 3 | 0003 | exp05_spheroid_v03 | no sub-step block; 12 s frames (21,600 frames for 72 h) | WORKS: 20 -> 89 cells; 3 spheroids of 31, 27 and 16 cells (roundness 0.88, 1 = round), first spheroid of 8+ cells at 30 h; 11 cells still alone and not growing, as the description says | show where each spheroid's cells come from: colour by founder |
| 4 | 0004 | exp05_spheroid_v04 | colour by founder (8 muted colours, inherited), camera 1.6x closer; dynamics identical | same numbers as 3 (89 cells, spheroids 31/27/16); every spheroid is TWO colours: each formed when two founders' small clusters met, then grew, an aggregate not a clone | raise the threshold to 2 touched cells |
| 5 | 0005 | exp05_spheroid_v05 | adhesion threshold 1 -> 2 touched cells | 20 -> 54 cells; ONE spheroid of 40 cells from about 5 founders (roundness 0.90), pairs stay pairs and do not grow: the threshold acts. But clustered cells sit 0.24 diameters from their nearest neighbour (2.4 um for 10 um cells): they interpenetrate, the spheroid is half the size it should be | stiffer cells |
| 6 | 0006 | exp05_spheroid_v06 | cells 20x stiffer against overlap (epsilon 1 -> 20) | cells now touch instead of interpenetrating (nearest neighbour 0.88 diameters, was 0.24); 20 -> 52 cells, one spheroid of 39 cells about 50 um across, but lumpy (roundness 0.72, was 0.90): stiff cells stack and do not slide into a ball | stronger adhesion so the aggregate compacts |
| 7 | 0007 | exp05_spheroid_v07 | adhesion twice as strong (well 0.3 -> 0.6) | slightly rounder spheroid (roundness 0.79, was 0.72), cells pressed closer (nearest neighbour 0.78 diameters); same counts (52 cells, one spheroid of 39); 10 cells never meet anyone | more motility so more cells meet |
Steps 1-6 were re-generated in place with white overlays (plotting.foreground: white) on 2026-11-14; same specs and seeds, numbers equal to within GPU noise (roundness of step 5: 0.90 -> 0.93).
| 8 | 0008 | exp05_spheroid_v08 | cells move twice as much (kT 0.045 -> 0.09) | WORSE: 35 cells, one spheroid of 16 (roundness 0.63), only 0.46 of the cells in a cluster: at 6.7x the random-motion energy the adhesion no longer holds pairs long enough to grow, so fewer cells adhere, not more | back to kT 0.045; give the run more time instead |
| 9 | 0009 | exp05_spheroid_v09 | back to kT 0.045; 120 h instead of 72 h | BEST: 20 -> 234 cells; ONE spheroid of 221 cells from 5 founders' lineages (roundness 0.79), 0.96 of the cells in a cluster; first spheroid of 8+ cells at 33 h; 5 cells never joined | same spec, another seed |
| 10 | 0010 | exp05_spheroid_v10 | same rules, every random seed 1 -> 2 | the outcome holds: 20 -> 276 cells, TWO spheroids of 163 and 99 cells, each from 3-4 founders' lineages (roundness 0.76), 0.96 of the cells in a cluster, first spheroid at 29 h (33 h on seed 1) | the ten iterations on cell surface tension and stiffness need cells with a surface: deformable MPM cells from 11 on |
| 11 | 0011 | exp05_mpm_v11 | DEFORMABLE cells (125 material points each, Young's modulus 20, surface tension 0.15, contact tension -0.1), persistent crawl, growth while at least 15% of the surface touches others; 400 min | works: cells flatten against each other at contacts; 20 -> 35 cells in 400 min; clusters of 9, 4, 3, 3, 2; the largest is a CHAIN, not a ball (roundness 0.21); 0.60 of the cells in a cluster | batch 12-15: surface tension x1/3, x3; stiffness x1/4, x4 |
| 21 | 0012 | exp05_signal_v21 | SIGNALLING: activator-inhibitor (Gierer-Meinhardt) in every cell, exchanged with the cells it touches (inhibitor 30x faster), growth only where the activator is at least 0.8 | NO PATTERN: every clustered cell sits at the uniform level 0.93 (lone cells 1.4-1.5), so the gate never acts; same spheroid as iteration 9 (219 cells, roundness 0.85). The exchange is averaged over a cell's neighbours, so the inhibitor must spread faster than 0.7 per 10 min for spots to form (linear stability of the uniform state); 0.3 was below it | inhibitor exchange 10x faster |
| 22 | 0013 | exp05_signal_v22 | inhibitor exchanged 10x faster (0.3 -> 3.0 per 10 min) | A PATTERN: in every pair one cell keeps the activator (up to 3.6), the other loses it (down to 0.03) -- lateral inhibition; in the 15-cell cluster the active cells sit on its surface. But growth nearly stops: 20 -> 30 cells, because an inactive cell does not grow and a pair never reaches the 2-contact threshold | adhesion threshold back to 1 touched cell, so active cells in pairs grow |
| 12 | 0015 | exp05_mpm_v12 | surface tension x1/3 (0.15 -> 0.05) | the STRONGEST lever: 20 -> 68 cells, a compact aggregate of 43 polyhedral cells (roundness 0.61) + one of 15; 0.90 of the cells growing: a soft outline spreads cells on each other, contacts pass the 15% gate | lower still; combine with stiffness |
| 13 | 0016 | exp05_mpm_v13 | surface tension x3 (0.15 -> 0.45) | growth stops: 20 -> 25 cells, nothing larger than 4; cells stay round, contacts stay small, only 0.08 of the cells pass the gate | find the tension where growth stops |
| 14 | 0017 | exp05_mpm_v14 | cells 4x softer (Young's modulus 20 -> 5) | 39 cells, largest 22 (roundness 0.46), 0.44 growing | stiffness matters little at this surface tension |
| 15 | 0018 | exp05_mpm_v15 | cells 4x stiffer (Young's modulus 20 -> 80) | 40 cells, largest 22 (roundness 0.38), 0.58 growing: as 14 -- a 16x range of stiffness moves the outcome less than a 3x step of surface tension | test stiffness where the tension is low |
| 23 | 0014 | exp05_signal_v23 | adhesion threshold 2 -> 1 touched cell | 20 -> 58 cells, four small irregular clusters (22, 10, 10, 8); the pattern is salt-and-pepper (a lone active cell among quiet ones, median activator 0.03), so growth stays sparse and slow | make the activator spread too, so the pattern is PATCHES of several cells, and small clusters stay uniform (and grow) |
| 24 | 0019 | exp05_signal_v24 | activator spreads too (0.01 -> 0.3 per 10 min) | NO PATTERN again: every clustered cell at the uniform 0.93, so all grow: 20 -> 419 cells, one aggregate of 322 (roundness 0.62) + 72 + 16. With the activator spreading at a tenth of the inhibitor's rate the fastest pattern grows e-fold only every 12 h (linear stability), slower than the cells divide and mix | inhibitor spreads 3x faster (3 -> 10): patches of several cells, e-fold 1.1 h |
| 25 | 0020 | exp05_signal_v25 | inhibitor spreads 3.3x faster (3 -> 10 per 10 min) | PATCHES: every cluster grows one active patch of 3-6 cells on one side (activator 90th percentile 2.3, median 0.55); only the patch grows, so clusters stay small (92 cells; 28, 22, 20, 12) and roundish (0.79) | differential adhesion: active cells stick weakly to quiet ones, so the growing patch can bulge out |
| 26 | 0021 | exp05_signal_v26 | DIFFERENTIAL ADHESION by fate (new pair_potential[soft_adhesive_typed]): quiet-quiet and active-active 0.6, quiet-active 0.15 | NOT SPHEROIDS ANY MORE: clusters become two-lobed and elongated (largest 23 cells, roundness 0.33, was 0.79), quiet lobes joined through active patches; 102 cells, 6 clusters of 8+ | let it grow longer: 200 h |
| 27 | 0022 | exp05_signal_v27 | 200 h instead of 120 h | the lobes keep growing: 20 -> 232 cells in four multi-lobed, elongated aggregates (68, 63, 56, 36 cells; largest roundness 0.37): quiet lobes linked and capped by active patches -- no aggregate rounded up into a spheroid | make the active fate irreversible, so active lobes keep their identity as they grow |
| 28 | 0023 | exp05_signal_v28 | active fate irreversible (cell_fate commit) | BACK TO SPHEROIDS: within 200 h every clustered cell has once been active, so all are active, adhesion is uniform again and the aggregates round up (71, 49, 48, 31 cells, roundness 0.76): the lobes of 26-27 need BOTH fates present at once | reversible fates again; the two fates stick to each other even less |
| 29 | 0024 | exp05_signal_v29 | the two fates stick 3x more weakly (quiet-active 0.15 -> 0.05) | BRANCHED AND BUDDING: the aggregates split where the fates meet; 20 -> 730 cells in 102 clusters (24 of 8+), the largest a branched, filamentous 142-cell body (roundness 0.33) of quiet strands and active knots, small buds shed all around; split pieces are small, stay uniform and grow, so growth runs away | between 27 and 29: quiet-active 0.09, camera further out |
| 16 | 0025 | exp05_mpm_v16 | surface tension x1/10 (0.015) | 20 -> 83 cells, aggregate of 47 (roundness 0.59) + 15 + 15; 0.93 growing: lower still is a little more growth than 0.05 (68) | |
| 17 | 0026 | exp05_mpm_v17 | surface tension 0.05 + cells 4x softer (Young's 5) | 76 cells, largest 44 (roundness 0.51) | |
| 18 | 0027 | exp05_mpm_v18 | surface tension 0.05 + cells 4x stiffer (Young's 80) | 73 cells, largest 42 (roundness 0.54): with a soft outline too, a 16x range of stiffness changes the count by 4% (76 vs 73) | |
| 19 | 0028 | exp05_mpm_v19 | surface tension 0.25 | 32 cells, largest 9, 0.34 growing -- as 0.15 (35 cells): growth is high below about 0.05, falls between 0.05 and 0.15, and stops near 0.45 | |

### Surface tension against stiffness (MPM iterations 11-19, 400 min, 20 cells at start)

| surface tension | Young's modulus | cells after 400 min | largest cluster |
|---|---|---|---|
| 0.015 | 20 | 83 | 47 |
| 0.05 | 20 | 68 | 43 |
| 0.05 | 5 | 76 | 44 |
| 0.05 | 80 | 73 | 42 |
| 0.15 | 20 | 35 | 9 |
| 0.15 | 5 | 39 | 22 |
| 0.15 | 80 | 40 | 22 |
| 0.25 | 20 | 32 | 9 |
| 0.45 | 20 | 25 | 4 |

Surface tension decides; stiffness barely matters. Growth here needs 15% of a cell's surface in contact. A low surface tension lets touching cells spread on each other and pass that gate. A high one keeps them round, so the contact stays small and growth stops near 0.45. A 16x range of Young's modulus moves the count by under 10% at either tension.
| 30 | 0029 | exp05_signal_v30 | quiet-active adhesion halfway (0.09); camera further out | THE MOST BRANCHED CONNECTED BODY: a 183-cell branched mass (roundness 0.20, the lowest of the experiment) of quiet strands joined at active knots, with buds of 12-80 cells around it; 20 -> 596 cells in 200 h | -- |
| 20 | 0030 | exp05_mpm_v20 | surface tension 0.05, 800 min instead of 400 | the deformable-cell spheroid: 297 polyhedral cells in one aggregate (roundness 0.58), 3 cells alone; growth 46 cells at 320 min -> 124 at 480 -> 209 at 560 -> 300 at about 640 min, where the cell set's buffer of 300 slots stopped divisions (a cap of the spec, not of the biology; raise `sets.cell.buffer` to continue) | -- |

Known defect of the MPM movies (steps 11, 15-18, 25-28, 30): the cell-count curve's x axis is the stored row (0-100, or 0-200 for step 30), not minutes, although steps 15-18 and 25-28 label it "time (min)"; one stored row = 4 min. The trajectories and the numbers above are unaffected.

## Restyle in the look of exp29 movie 1 (2026-11-14)

Cedric asked to re-render every step in the style of exp29 movie 1. All 30 movies are redrawn by `tools/exp29_studio.py` with no new simulation: the ink look on the grey ground, white insets, a scale bar, the "cells / time" clock, and a title top-left from `data/overlay_texts.yaml` (drawn by `tools/exp29_text_overlay.py`). Each step keeps its spec and its why text and gains a `relook` line. The old movies are kept in `data/pre_studio_backup_2026-11-14/`.

- Colours:
  - Point steps 1-3 and the MPM steps: each cell its own colour (by slot).
  - Steps 4-10: each founder lineage its own colour, in the vivid palette. Step 4's finding needs it: each spheroid holds two founders.
  - Signalling steps 12-14, 19-24 and 29: each cell coloured by its activity, the activator a. Blue is 0, yellow is the run's 99.5th percentile (Cedric: "from blue not active to yellow active").
- Point cells are drawn as spheres of 5 um radius (the spec's `dot_radius`). The cell-volume inset shows the model's size times that sphere's volume.
- The studio's new options:
  - `PointRun`, a point-cell kind.
  - `--act-cmap`, `--cell-colours type`, `--type-colours` and `--point-volume`.
  - A time axis for a 10-minute unit.
- The MPM curve-axis defect noted above is gone: the studio's axes are in minutes.

## Topology series, iterations 31-50 (2026-11-14)

Cedric asked to continue developing cell signalling that changes the tissue's topology, "from torus to tube to budding and tubing", with cell death as a sculptor, and with the activity of cells plotted from blue (not active) to yellow (active). All runs use point cells on iteration 9's rules, for 120 h unless stated.

| when | decision | why |
|---|---|---|
| 2026-11-14 | three new operators in `spheroid_ops.py`: `cell_die[point_signal]`, `cell_divide[point_oriented]` and `cell_chem_react[crowding]` | the registry had no death and no oriented division for point cells (`cell_die` covers vertex cells only), and no signal made by crowded cells |
| 2026-11-14 | the sculpting signal is a crowding signal c. Each cell makes 0.1 per touched cell per 10 min and loses it at 1 per 10 min, so c settles near 0.1 x the number of touched cells | inner cells of iteration 9's spheroid touch 9 cells (c about 0.9), surface cells 5 (about 0.5): one threshold separates the core from the surface |
| 2026-11-14 | topology is measured as tunnels b1 and cavities b2 of each body of 8+ cells (`tools/exp05_loop.py voxel_topology`). The body is where the cell density sum of exp(-d^2 / (2 x 0.5^2)) is at least 0.5 | checked on a ball (0 tunnels, 0 cavities), a shell (1 cavity), a ring (1 tunnel), an open tube (1 tunnel) and two rings (2 tunnels); the gaps between packed cells close, a lumen wider than about 1.5 cell diameters stays open |
| 2026-11-14 | the torus is grown in a slab one cell diameter thick (world z = 1 length unit, walls), which holds two layers of cells | a hole through a 3-D ball makes a cavity, not a tunnel; a disc's crowded centre dying makes a ring |
| 2026-11-14 | the movie colour is the signal that sculpts. c is channel 0 where only c acts; the activator a is channel 0 where a gates growth | the studio colours channel 0; the legend and the middle inset name it |

| it | step | spec | change from the previous | result | next |
|---|---|---|---|---|---|
| 31 | 0031 | exp05_topo_v31 | crowding signal c (0.1 per touched cell, loss 1 per 10 min); death at c 0.75 or more | SOLID: 83 cells, one 70-cell ball (roundness 0.83), no cavity; 57 deaths seen; the core died and the aggregate closed over it | a lumen pressure |
| 32 | 0032 | exp05_topo_v32 | 31 in a slab one cell diameter thick (two layers) | a flat disc of 681 cells (roundness 0.05), no hole and NO death: in two layers every cell touches 6-7 others, c sits at 0.68 inside and 0.62 at the rim, under 0.75 | threshold between the two, in-plane pressure |
| 33 | 0033 | exp05_topo_v33 | 31 with every division along z | SOLID: 84 cells, a 71-cell lump (roundness 0.74), 67 deaths | a lumen radial to z |
| 34 | 0034 | exp05_topo_v34 | Gierer-Meinhardt activator patches; growth only where active, divisions outward (first run crashed: GM gave 2 columns for a 3-column chem, fixed) | 254 cells in four separate balls of 41-48 (roundness 0.78-0.85), no buds, 32 deaths | a lumen first |
| 35 | 0035 | exp05_topo_v35 | 31 + lumen_pressure[point_cluster] 0.2 per 10 min on the non-crowded cells | SOLID: 78 cells, a 63-cell ball, no cavity; 238 deaths: death keeps the aggregate too small for a lumen | |
| 36 | 0036 | exp05_topo_v36 | the slab disc, death at c 0.66, the pressure in the disc's plane | a 153-cell disc, no hole; 645 deaths | |
| 37 | 0037 | exp05_topo_v37 | divisions along z, the pressure radial to z | SOLID: 87 cells, a 74-cell lump (roundness 0.80), 233 deaths | |
| 38 | 0038 | exp05_topo_v38 | 35's lumen + 34's growing active patches | four separate balls of 40-51 cells, no buds, 43 deaths | |
| 39 | 0039 | exp05_topo_v39 | slab disc, cells with activator 1.5 or more die after 33 h | 21 cells left: death took nearly the whole disc (155 deaths seen) | |

Why 31-39 kept their topology (a ball, or a disc): point cells packed at 0.75 cell diameters press on each other through their soft cores (a push of about 5 per pair), and dividing cells push into any gap, so the tissue flows into every hole death opens; a lumen pressure of 0.2 per cell loses to that. Epithelia hold a lumen open by apical-basal polarity: from iteration 40 the cells carry Nissen et al.'s (2018) polarities (`pair_potential[polar_nissen]`, new), under which cells attract only side by side and form one-cell-thick sheets.
| 40 | 0040 | exp05_topo_v40 | POLARISED CELLS: pair_potential[polar_nissen] (Nissen 2018) instead of soft_adhesive, apical-basal polarity only (lambda 1, 0, 0), kT 0.005, 20 cells in a ball of 4 | a one-cell-thick FOLDED SHEET of 2998 cells (roundness 0.17, nearest neighbour 0.98 diameters: a monolayer, no core pressure), not closed: the first polarities were random; 3000 cell slots filled at about 155 h | start the polarity facing outward |
| 41 | 0041 | exp05_topo_v41 | planar polarity too, Nissen's tube weights (0.41, 0.5, 0.09) on every cell | TUBING: a body of branching open tubes (2991 cells, 13 tunnels at the end, 86 tunnels and 10 cavities at 140 h): the first topology change of the series | q along one axis for a single tube |
| 42 | 0042 | exp05_topo_v42 | planar polarity only where the activator is 1.0 or more; growth only where it is 0.8 or more | a flat 156-cell sheet (roundness 0.11) and one of 33, no tubes: growth gated to the active cells stayed small | cyst first, everyone grows |
| 43 | 0043 | exp05_topo_v43 | apical-basal polarity; after 33 h cells with activator 1.5 or more die | a flat double sheet of 785 cells with dead spots, ONE tunnel at the end (461 deaths seen) | the same on a closed cyst |
| 44 | 0044 | exp05_topo_v44 | polarity facing outward at the start (p_init radial), 4000 cell slots | a large FLAT double plate (2359 + 1635 cells, roundness 0.11): the cyst formed and collapsed, nothing held the lumen | a lumen pressure along the polarity |
| 45 | 0045 | exp05_topo_v45 | outward p, planar polarity along z on every cell | a lumpy tubular body with two thin tubes and a closed loop (3995 cells, 8 tunnels) | inflate |
| 46 | 0046 | exp05_topo_v46 | outward cyst, planar polarity only in activator patches, all cells grow | a flat sheet carrying regular TURING SPOTS of active cells (3999 cells, 12 tunnels); the spots did not bud | inflate |
| 47 | 0047 | exp05_topo_v47 | outward cyst, its activator spots die after 33 h | a collapsed double plate whose dead spots open and close (1126 cells, 0 tunnels at the end, 529 deaths) | inflate |
| 48 | 0048 | exp05_topo_v48 | lumen_pressure[polarity] 0.05 per 10 min along each cell's polarity (new), 10000 cell slots | the cyst INFLATED and BURST: two huge open bowls (5909 + 3708 cells), 22 tunnels; slots full at 185 h | a weaker pressure, a shorter run |
| 49 | 0049 | exp05_topo_v49 | 45 (planar polarity along z) inflated | TUBING: inflated tube segments joined into a large closed loop, 9987 cells, 30 tunnels, 1 cavity | |
| 50 | 0050 | exp05_topo_v50 | 46 (planar polarity in activator patches) inflated | two open bowls covered in Turing spots, no buds (10 tunnels) | contract the spots instead |
| 51 | 0051 | exp05_topo_v51 | 47 (active spots die) inflated | small sheets of 662, 282, 28 cells, holed where spots died (7 tunnels), 549 deaths | |
| 52 | 0052 | exp05_topo_v52 | WAVES: Aliev-Panfilov excitation over the touch graph, paced every 400 min below z = 12; an excited cell rests at 0.7 of its spacing | the waves run: one excitation spike per 400 min (activity inset); a wave sweeps the whole lower bowl within one stored frame, and its wall carries CONCENTRIC RING FOLDS; the cyst burst into two bowls like 48, and the waves never reach the upper one | an intact cyst |

Why the inflated cysts burst (48-52): the lumen pushes every cell outward at a fixed 0.05 per 10 min, and the wall tension that balances it grows with the cyst's radius (Laplace), until the sheet tears; 10000 cells at 185 h make radii of about 30 cell diameters.
| 53 | 0053 | exp05_topo_v53 | lumen push 0.015 (a third of 48's), 150 h | a collapsed flat double plate (1744 + 1233 cells, roundness 0.07), no cavity at any time | the cysts never closed: start from a closed shell |
| 54 | 0054 | exp05_topo_v54 | waves on 53, excited cells rest at 0.6 | WAVE BANDS: each excitation band climbs the plate from the lower pole; ridges along the band edges; 1 tunnel | |
| 55 | 0055 | exp05_topo_v55 | planar polarity round the z axis (q_init azimuthal, new) | an upright tubular body with arms (2346 cells, 1 tunnel), not a ring | |
| 56 | 0056 | exp05_topo_v56 | active Turing spots rest at 0.6 | a flat plate with spots, no buds | |
| 57 | 0057 | exp05_topo_v57 | waves switch planar polarity on as they pass | a DOME over a flat skirt (cap-and-brim), the wave sweeping up it | |
| 58 | 0058 | exp05_topo_v58 | the 20 cells start as a CLOSED SHELL (seed_positions[shell], new), lumen push 0.03 | a closed cyst at 15 h (one cavity), then flattened into a lens-shaped plate (3148 cells) by 30 h | a self-regulating lumen |
| 59 | 0059 | exp05_topo_v59 | the shell, planar polarity round z | a tubular body with an arm (2072 cells), no ring | |
| 60 | 0060 | exp05_topo_v60 | the shell, active spots contract | a lens plate with Turing spots, no buds | |
| 61 | 0061 | exp05_topo_v61 | the shell, contraction waves | a cavity at 15 h, then a lens plate swept by waves | |
| 62 | 0062 | exp05_topo_v62 | the shell, planar polarity along z (Nissen's cyst-to-tube route) | THE TORUS: the shell opens at both poles (planar polarity along z cannot lie in the wall where z points out of it) and becomes a RING round the equator, ONE TUNNEL from 15 h to 135 h, widening into a broad band; at 150 h the band buckles into a crenellated belt (3 tunnels, 5 cavities, 3913 cells) | waves round the ring, buds and holes on it |
| 63 | 0063 | exp05_topo_v63 | lumen_pressure[volume] (new): a push only while the aggregate's mean radius is short of a closed monolayer's | a flattened lens again (3277 cells); a flat lens's mean radius is close to the shell's, so the push hardly acts | measure the flattest axis |
| 64 | 0064 | exp05_topo_v64 | WAVES ROUND THE RING: 62 + Aliev-Panfilov paced at one point of the ring, excited cells rest at 0.6 | THE FLOWER: the ring grows and buckles into a closed band with large inward lobes, 24 tunnels and 7 cavities at 150 h, where 62 without waves had 3 tunnels at 150 h: the wave-borne contractions drive the folding | waves more often; a tube |
| 65 | 0065 | exp05_topo_v65 | 62's ring, contracting Turing spots | a lobed ring dotted with spots (5 tunnels), no outward buds | |
| 66 | 0066 | exp05_topo_v66 | 62's ring, its active spots die | the deaths cut the ring open into a C-shaped arc (0 tunnels from 75 h, 278 deaths): death UNDOES the torus | |
| 67 | 0067 | exp05_topo_v67 | 62's ring to 200 h | a lobed flower ring with inward folds, 9986 cells, 35 tunnels at 200 h (1 tunnel until 140 h): growth alone also folds the band, later | |
| 68 | 0068 | exp05_topo_v68 | lumen_pressure[volume] reading the flattest axis (measure min_axis) | THE CYST: a round hollow sphere, roundness 0.98, one cavity at 150 h (3464 cells): the self-regulating lumen holds a growing cyst open without bursting it | buds, holes, waves, tubes on it |
| 69 | 0069 | exp05_topo_v69 | 62's ring with every division along z | a taller crenellated band, a short wide tube (1 tunnel), not a long one | |
| 70 | 0070 | exp05_topo_v70 | ring waves every 200 min | the flower folding of 64 again (12 tunnels at 150 h, 24 at 400 min): the waves fold the ring, their period matters less | |
| 71 | 0071 | exp05_topo_v71 | 69's band with waves along it | a crenellated band, the waves running up it (4 tunnels) | |
| 72 | 0072 | exp05_topo_v72 | ring waves, excited cells swell to 1.3 | the ring folds less, shallow lobes (1 tunnel): contraction, not swelling, makes the deep folds | |
| 73 | 0073 | exp05_topo_v73 | 68's cyst, Turing spots contract (spacing 0.6) | the cyst stays round (0.98, one cavity); the spots shrink IN the wall, no buds: an in-plane contraction does not bend a sheet | wedging, faster growth |
| 74 | 0074 | exp05_topo_v74 | 68's cyst, contraction waves from the lower pole | waves sweep a round cyst (one cavity), no folds: the self-regulating lumen holds it round | wedging waves |
| 75 | 0075 | exp05_topo_v75 | 68's cyst, its active spots die | the holes opened the cyst and it flattened into a 321-cell plate (269 deaths) | |
| 76 | 0076 | exp05_topo_v76 | 68's cyst, planar polarity along z | it still opens into a ring, then a tall crenellated, box-like cylinder (9 tunnels, 4 cavities) | |
| 77 | 0077 | exp05_topo_v77 | 68's cyst, planar polarity in active spots | the cyst flattened; the active spots aligned into STRIPES (planar polarity orders them) | |
| 78 | 0078 | exp05_topo_v78 | 68's cyst, active spots WEDGE (alpha_active 0.5, new: apical constriction in the polarity term) | BUDDING: every Turing spot bulges out of the round cyst as a bud, a knobbly sphere that stays closed (2480 cells, roundness 0.84, one cavity) | buds that elongate, buds that pinch off |
| 79 | 0079 | exp05_topo_v79 | the opposite wedging (alpha -0.5) | knobbly as 78 (3036 cells, one cavity): from outside the pits and the buds look alike in these stills | |
| 80 | 0080 | exp05_topo_v80 | active spots grow 3x faster (rate_active, new) | an irregular lobed cyst covered in spots, the 10000 slots full (6 tunnels) | |
| 81 | 0081 | exp05_topo_v81 | waves that wedge the excited cells | a round cyst (one cavity): a passing wave wedges too briefly to fold | |
| 82 | 0082 | exp05_topo_v82 | wedging AND 3x growth in the spots | a RASPBERRY-like body of many buds, 9982 cells, 33 tunnels | |
| 83 | 0083 | exp05_topo_v83 | wedged buds with planar polarity in them | the planar polarity broke the cyst open into an irregular cup (no cavity), no tubes | |
| 84 | 0084 | exp05_topo_v84 | the cells round each bud (activator 0.5-1.0) die after 50 h | one compact cyst with large active patches (583 cells, 259 deaths); the buds did not pinch off | |
| 85 | 0085 | exp05_topo_v85 | 78's budding cyst to 200 h | a CAULIFLOWER: a large lobed body whose every lobe carries its own spots (10000 cells, 17 tunnels) | |
| 86 | 0086 | exp05_topo_v86 | 62's ring with wedged Turing spots | the ring broke into a TANGLE of thin tubular strands and loops (52 pieces, 18 tunnels): a tubular network | |
| 87 | 0087 | exp05_topo_v87 | wedging twice as hard (alpha 1.0) | a compact deeply FOLDED cyst, brain-like (1161 cells, 17 tunnels) | |
| 88 | 0088 | exp05_topo_v88 | 78 with a faster inhibitor (30 per 10 min): bigger, fewer Turing spots | FEWER, BIGGER BUDS: five large knobs, on a cyst that opened (2828 cells, no cavity) | |
| 89 | 0089 | exp05_topo_v89 | deep wedging (alpha 1.0) and 3x growth in the spots | GYRIFICATION: a convoluted brain-like body (7329 cells, 113 tunnels) | |
| 90 | 0090 | exp05_topo_v90 | spots wedge out (0.5), the wall between them wedges in (new `alpha`, -0.5) | CLEAVING: the cyst divides into lobes separated by deep grooves, and lobes come off as separate cysts (pieces of 1346, 451, 206 cells): budding off, where 84's dying necks failed | stronger in-wedging |
| 91 | 0091 | exp05_topo_v91 | 62's ring, gentler wedged spots (alpha 0.3) | a flower ring with inward folds and branches (77 tunnels) | |
| 92 | 0092 | exp05_topo_v92 | the budding cyst with planar polarity along z | a knotted tangle of tubes (2015 cells, 12 tunnels) | |

## Audit and guiding sentence (2026-11-15)

All 92 movies were looked at again beside their insights. A first pass used the middle and last frame of each movie; Cedric pointed out that it misses the signalling (Turing patterns, waves), so the 50 signalling steps were audited over time: six frames across each movie with its activity-over-time inset, and numbers read from the trajectories (`/tmp/v4d/audit/signals.py`, `ring_waves.py`: per-wave reach, pacemaker cells, spot counts, the share of active cells that stay active). Twelve insights were corrected (46, 52, 54, 61, 64, 70-74, 81, 90). The deck is `presentation/plexus_exp05.pdf` (notes generated by `log/experiments/exp05/deck/make_notes.py`); its last slide sums up the five inputs and where they lead.

What the time course showed:
- Waves. Conduction is about 0.4-0.6 cell diameters per 10 min. On the round cyst each wave reaches every cell (step 74: 17 of 17, the farthest cell 15 diameters away after about 250 min) and the cyst stays round. With two bowls or plates only the piece that holds the pacemaker fires (52: 52-65% of the cells; 54: 42-56%).
- Waves on the ring. The pacemaker was a fixed box in space; the growing ring leaves it (2-5 cells in it until 68 h, none from 72-88 h). In step 64 the first 10 waves run round the whole ring and excitation ends at 82 h. In steps 70 and 72 one pulse keeps circulating with no pacemaker until the end (142 h), covering 22% falling to 10% of the growing ring per 400 min: a re-entrant wave, which a ring can carry and a closed cyst cannot.
- The ring's flower folding (64: 24 tunnels at 150 h) comes after 135 h, when no wave is left in step 64, and the wave-free ring of step 67 folds too: the folding is not shown to come from the waves. An earlier note here credited the waves; that was wrong.
- Turing spots. One spot per 50-110 cells (217 with the inhibitor at 30), 13-25% of the cells active, spots of 8-16 cells. New spots appear as the tissue grows (step 46: 2, 3, 7, 22, 55, 84 spots at 30, 60, 100, 140, 170, 200 h). Cells pass through the spots: of the cells active at one time, 10-60% (typically 30-50%) are still active 22-40 h later.
- Step 90's cleaving cyst has its spots (33 at 150 h); in the movie they lie in the grooves between the lobes. Step 73's cyst has 1-2 spots until 75 h (on the side away from the camera), 32 at 150 h.
- Lone cells in steps 14, 19-24 and 29 sit at the top of the activity scale: the old isolated-cell sink of the graph Laplacian (fixed by the exp06 session on 2026-11-14). The reruns of steps 12-13, made after the fix, show lone cells at the uniform level. Those eight steps still carry the pre-fix movies.

The landscape:
- Topology comes from polarity. Packed adhesive cells stay a ball (steps 1-39). Polarised cells make a sheet, and its closure, planar polarity and lumen choose cyst (68), tube (41, 49) or torus (62).
- Shape on that topology comes from a signal that sets the wall's curvature, the wedging: buds (78), folds (87, 89), cleavage (90). In-plane contraction changed no shape (73, 74); growth alone folds a free band (67).
- For a cell the signals are passing: a spot stays in place while cells move through it; a wave passes and leaves nothing.
- Cell death removed topology (66, 75) and made none.
- Not reached: one long tube; buds that lengthen into branching tubes; buds that pinch off by death.

Proposed guiding sentence for the next runs (not yet confirmed by Cedric):

> Cells keep a mark of the signal that passed through them and wedge by that mark: marks left by standing spots grow buds into branching tubes, marks left by travelling waves cut a tube into repeated segments.

- Route A, space (from step 78): a cell marks itself at the peak of a spot; marked cells wedge, their neighbours take planar polarity and lengthen the bud. Step 23 warns: a mark on every cell that was ever active reaches all cells.
- Route B, time (from steps 70-74): the wave is fired by cells (the cells of a spot), not from a fixed place; a cell marks itself when a wave passes; the marks form rings that constrict the tube.
- Shared: the mark (`cell_fate` commit), the wedging and the lumen; route A's spots are route B's pacemakers. Two runs at a time on gpu_rtx6000, one per route; each movie is watched over its whole length, with the signal's numbers, before the next pair.

## Routes A and B (from 2026-11-15, Cedric: "perfect go for it")

Two runs at a time on gpu_rtx6000, one per route. Each run is audited over its whole length before the next pair: a film strip of the movie, numbers from the trajectory (`/tmp/v4d/audit/route_audit.py`: cells per mark, patches, waves' reach, pacer cells) and cross-sections coloured by mark (`section.py`), because one camera hides what goes inside. New for the routes: `polar_nissen` takes wedging, polarity weights and spacing per cell TYPE (`alpha_types`, `lambda_types`, `spacing_types`); `cell_fate` has `only_types`; `cell_chem_react[aliev_panfilov_paced]` fires the wave from the cells of a type; the studio draws marked types over the activity colours (`--mark-colours`). The mark is a cell's type, taken by `cell_fate` with `commit` inside an `after_frame` / `before_frame` window.

| it | step | route | spec | change | result | next |
|---|---|---|---|---|---|---|
| 93 | 0093 | A | exp05_routeA_v01 | step 68's cyst; spots have no mechanical effect until 90 h; at 90-92 h a cell keeps a mark from its activator: tip at 1.5+, stalk at 0.5-1.5; tips wedge (alpha 0.5), stalks take planar polarity | the marks take: 76 tip cells in 11 patches, 66 stalk cells. The tip clones INVAGINATE: cross-sections show red arcs in the wall at 110 h, pits at 130 h, deep tip-lined pockets at 150 h and several tubes inside the cyst at 180 h, stalk cells at their mouths; the tip cells' mean distance from the centre falls from 1.00 to 0.54 of the body's radius; no tunnel until 144 h, 68 tunnels and 9 cavities at 180 h (9563 cells, outline still round, 0.97). From outside the movie shows little of it | the other wedging sign: do the clones bud outward? |
| - | (void) | B | exp05_routeB_v01 | step 62's ring, a pacer lineage fires the wave, cells excited at 105 h keep the mark `ring` | VOID, not filed: 0 pacer cells at every stimulus (the types were drawn at random from fractions, 1 of 20 gave none), so no wave and no mark | `type_layout: ordered` |
| 94 | 0094 | A | exp05_routeA_v02 | 93 with the tip clones wedged the other way (alpha -0.5) | the tip clones go OUTWARD: their mean distance from the centre rises from 1.00 to 1.31 of the body's radius (0.54 in 93); cross-sections show tip caps at 130 h, tip-capped lobes at 150 h and a star-shaped cyst at 180 h, tip cells at the apex and stalk cells at the neck of each arm; 9462 cells, 22 tunnels, 6 cavities. The wedging sign chooses invagination or evagination | tips that divide faster, to lengthen the arms |
| 95 | 0095 | B | exp05_routeB_v01 | step 62's ring; a pacer lineage fires the wave every 400 min; the cells excited at 105 h keep the mark `ring` and wedge (alpha 0.5) | the pacer lineage grows with the ring (2 cells at 20 h, 524 at 180 h) and each of the first 17 waves reaches 98-100% of the cells (81-90% after 120 h). The strobe marks 170 cells in SIX ARCS of about 30 cells, evenly spaced round the ring. Each arc folds inward into a pocket by 120 h: 6-7 regularly spaced, mark-lined lobes at 140 h; 4 tunnels at 126 h, 14 at 144 h, 45 at 180 h (the unmarked ring of 67: 1 tunnel until 140 h). The wave's spacing became the tissue's segments | the outward sign from route A |

Movies of 92-95 are two panels (full view and cross-section slab, no curves; `--section --section-slab 1.5 --no-insets`). For a ring the section panel takes its own camera from above (`--section-view 90,0`).
| 96 | 0096 | A | exp05_routeA_v03 | 94 with the tip cells dividing 3x faster (`rate_types`, new) | CLONAL OVERGROWTH INTO SACS: 86 tip cells at 92 h, 512 at 110 h, 8113 of 10000 at 140 h (the slots fill). Each tip clone inflates into a large hollow sac joined to a small core by a neck: four sacs at 145 h (an acinus-like body), no tube; then the body breaks into pieces (244 tunnels, 70 cavities at 144 h). A clone whose every cell wedges alike prefers a sphere | within the marked lineage only the cells in a current spot are tip, the others stalk |
| 97 | 0097 | B | exp05_routeB_v02 | 95 with the marked arcs wedged outward (alpha -0.5) | the same six arcs are marked at 105 h; the ring undulates with the arcs on the OUTWARD crests (112-133 h) instead of in inward pockets; the marked cells stay at 1.00 of the body's radius (0.73 in 95); 7 tunnels at 144 h, 97 at 180 h | the wave's period sets the number of segments? |
| 98 | 0098 | A | exp05_routeA_v04 | 96 with a mark that is re-read: the mark of 90-92 h makes a lineage competent (stalk or tip, kept); from 92 h a second `cell_fate` (`only_types: [stalk, tip]`) makes a competent cell tip while its activator is 1.0 or more and stalk otherwise | TIPS STAY SPOT-SIZED AND MULTIPLY: the largest tip patch never exceeds 29 cells, the number of tip patches grows with the tissue (11 at 95 h, 26 at 110 h, 92 at 140 h, 311 at 160 h), the stalk joins into one sheet (5064 cells at 160 h). Outward lobes capped by tip patches at 133 h, a lobed, folded wall at 159 h (the 10000 slots fill there); tip cells at 1.11-1.14 of the body's radius; 47 tunnels at 180 h. Lobes, not tubes: the stalk divides too, so the wall between tips widens as fast as the tips advance | only tips divide (stalk rate 0): does a tip leave a tube behind and split? |
| 99 | 0099 | B | exp05_routeB_v03 | 95 with the pacer firing every 200 min (400 in 95) | THE PERIOD DOES NOT SET THE NUMBER: still 6 arcs (156 cells) marked at 105 h, each folding inward (14 tunnels at 144 h, 65 at 180 h). Firing count per cell over 20 h: a median of 2 for 6 stimuli (min 1, max 6), against 3 for 3 at 400 min; a cell stays excited about 117 min per firing. The tissue's recovery blocks every second wave | pulses closer together need a slower wave: Laplacian coupling 1.0 -> 0.25 (half the speed) at 400-min pacing |

**Decision before pair 5 (2026-11-15): the wave is a ruler in cells.** Step 99's pacing was not followed, so before another cluster run the wave was checked on a chain of 160 cells (`/tmp/v4d/audit/ap_chain2.py`, same equations, coupling 0.33 per neighbour pair = the tissue's degree-normalised Laplacian at `d` 1.0). The chain reproduces the tissue: speed 0.64 cells per 10 min (0.5 measured), one firing per stimulus at 400-min pacing, one per two at 200 min. A scan of 64 settings (upstroke `k` 8 or 16, threshold `a` 0.15 or 0.05, recovery `mu1` 0.2 or 0.5, `mu2` 0.3 or 0.1, `eps0` 0.002 or 0.03, pacing 200 or 120 min) gave a shortest repeating wavelength of 24 cells against 27 for the present setting; a weaker coupling stops the wave (none at 0.3 of the present coupling). The kinetics cannot shorten the spacing, so the segment number can only come from the tissue's length over that ruler: run B4 moves the strobe from 105 h to 125 h. Route A's step 98 says the same of the spots (tip patches of one spot's size, their number growing with the tissue): in both routes the signal is a ruler in cells and growth sets the count.

| it | step | route | spec | change | result | next |
|---|---|---|---|---|---|---|
| 100 | 0100 | A | exp05_routeA_v05 | 98 with stalk cells that do not divide (`rate_types: {tip: 0.01734, stalk: 0.0}`): in the marked lineage only the tips grow | STILL LOBES: tip patches stay spot-sized (35 cells at most) and multiply (9 at 95 h, 26 at 110 h, 53 at 140 h, 118 at 160 h, 280 at 180 h); the stalk stays in separate collars (largest 569 cells at 180 h, against one sheet of 5064 in step 98); slower growth (10000 cells at 180 h, at 159 h in 98); tips at 1.10-1.15 of the body's radius; no tunnel until 162 h, 47 at 180 h. Cross-sections: a lobed wall with a tip patch on each lobe, no tube. A tip patch is a disc in a sheet and grows the same in every direction of the sheet: the stalk it leaves is a widening collar | hold the collar's girth: stalk planar polarity turned round the spot by the inhibitor's slope (new `polarity_guide[signal_slope]`) |
| 101 | 0101 | B | exp05_routeB_v04 | 95 with the strobe at 125 h (frame 37500) instead of 105 h | THE TISSUE'S LENGTH SETS THE NUMBER: the ring has 1590 cells at the mark (about 860 in 95) and the mark falls on 10 arcs (344 cells in 8 patches at 126 h, 10 at 130 h, about 40 cells each; 6 arcs of 30 in 95). Each folds inward into a pocket by 145 h, ten pockets at 160 h; 2 tunnels at 144 h, 23 at 162 h, 85 at 180 h. Waves reach 96% of the cells at 128 h, 74% from 148 h. Predicted from the ruler: 10-12 arcs | the mark as a cut line: marked cells die (`cell_die` `types`, new) |
| 102 | 0102 | A | exp05_routeA_v06 | 100 with the stalk's planar polarity turned along the level lines of the inhibitor, round the nearest spot (new `polarity_guide[signal_slope]`, stalk only, rate 2, slope of 0.1 or more) | THE POLARITY TURNS, THE SHAPE DOES NOT: stalk q along the level line 0.96 on average (1 = exact; 0.00 in step 100), 98% of stalk cells above 0.7. Shape as step 100: lobes capped by tips, tip cells at 1.11-1.14 of the body's radius in both runs, 39 tunnels at 180 h (47 in 100). Check on the tips (`/tmp/v4d/audit/tip_persist.py`): of the cells that are tip, 82% are still tip 2 h later, 45% after 10 h, 21% after 20 h, in both runs. Planar polarity round the spot does not hold the collar's girth at these weights (0.41, 0.5, 0.09) | make the wall bend round one axis: stalk wedging along q only (`alpha_q_types`, new) |
| 103 | 0103 | B | exp05_routeB_v05 | 95 with the marked cells dying instead of wedging (`cell_die` `types: [ring]`, rate 0.05 per 10 min from 105.5 h; no `alpha_types`) | SLOW DEATH HEALS: 160 marked cells at 105.6 h, 86 at 108 h, 46 at 110 h, 14 at 115 h, 0 at 130 h; the live cells form ONE body at every hour checked (104-130 h): the wall closes behind each loss and the ring is never cut. No trace of the six arcs remains; the ring later buckles into 8-9 lobes as an unmarked ring does (1 tunnel until 144 h, 9 at 162 h, 32 at 180 h; 8123 cells at 180 h) | the same death in 10 min instead of 10 h (rate 5): is the ring cut? |
| 104 | 0104 | A | exp05_routeA_v07 | 102 with stalk cells wedged along their planar polarity only (`alpha_q_types: {stalk: -0.5}`, new: the pair's wedging is alpha (r . q)^2), the bend of a tube wall round the spot | DEEPER FOLDS, NO TUBE: the stalk's polarity stays round the spots (0.94-0.97); stalk cells sit farther out, 1.12-1.18 of the body's radius (1.06-1.12 in 102), tips 1.11-1.18 (1.11-1.14); 12 tunnels at 144 h, 28 at 162 h, 60 and 4 cavities at 180 h (7, 13, 39 in 102). Cross-sections: a more folded wall with necked pouches at 165-180 h, tip patches all over it. Reading 100, 102 and 104 together: a competent stalk cell becomes tip again as soon as a new spot forms on it, and a collar wider than the spots' spacing always gets one (about 12 stalk cells per tip patch at 180 h in step 100): a cauliflower, not a tree | stalk is final: only a tip cell can change (`only_types: [tip]`), on step 100's simpler recipe |
| 105 | 0105 | B | exp05_routeB_v06 | 103 with the death 100x faster (rate 5 per 10 min): the marked cells are gone within 10 min | THE CUT HEALS IN FOUR HOURS: 164 marked cells at 105.5 h, 1 at 105.7 h. The ring is in SIX pieces of 144, 116, 112, 94, 79 and 77 cells from 105.7 h to 108 h (no tunnel at 108 h); the pieces lengthen as they grow, their ends meet and fuse: one body again at 110 h, a closed ring at 113 h. Then as step 103: 8-9 lobes at 150 h, 1 tunnel at 144 h, 5 at 162 h, 24 at 180 h (8122 cells). A gap of about 5 cells between two growing tube ends does not last | route A's growth zone on the wave's marks: the marked arcs divide 3x faster and wedge outward |
| 106 | 0106 | A | exp05_routeA_v08 | 100 with stalk final: the second `cell_fate` has `only_types: [tip]`, so a tip cell whose activator falls under 1.0 becomes stalk for good and no stalk cell turns tip | HALF THE TIPS, STILL LOBES: the tips do not die out (a tip cell is still tip after 10 h in 42-50% of cases, as before) and multiply by splitting alone: 20 patches at 95 h, 21 at 110 h, 46 at 140 h, 82 at 160 h, 151 at 180 h (280 in step 100), 22 cells at most each; stalk 2649 cells in 256 patches. The body stays rounder: no tunnel until 162 h, 27 at 180 h (roundness 0.91). A wall covered with small tip patches, no tube: one spot holds about 20 cells, too small a cap to pull a tube behind it | a longer ruler: the spots' reaction 4x slower (`rate` 0.4 -> 0.1), spots twice as wide |
| 107 | 0107 | B | exp05_routeB_v07 | 97 with the marked cells dividing 2x faster (`rate_types: {ring: 0.01156}`): route A's growth zone on the wave's marks | SIX SACS BUD FROM THE RING: 177 marked cells in 6 arcs of about 31 at 106 h; 483 at 120 h, 1754 at 140 h (the four largest sacs 283-335 cells), 6116 of 10000 at 160 h (995-1172 per sac), the unmarked ring between them; the 10000 slots fill at 160 h. Cross-sections: red domes on the ring at 121 h, round sacs joined by blue ring segments at 136-150 h. 2 tunnels at 126 h, 19 at 144 h, 23 at 162 h. Waves reach 93% of the cells at 122 h, 79-86% after 135 h | a second strobe at 125 h on the still unmarked cells: a second generation of buds between the first? |
| 108 | 0108 | A | exp05_routeA_v09 | 106 with the spots' reaction 4x slower (`gierer_meinhardt` `rate` 0.4 -> 0.1): the spots' spacing x2 | ONE WIDE TIP, STALK-WALLED ARMS: a single spot at the mark, one tip patch of 55-69 cells (6-13 patches of about 20 before) and one stalk patch of 61. The tip grows to 144 cells at 100 h, 359 at 120 h, then splits: 618 and 248 at 140 h, four patches of 181-365 at 160 h; the stalk it leaves is one sheet (2796 cells at 160 h). Cross-sections: a tip cap on one side at 115 h, a stalk-floored outgrowth rimmed by tips at 136 h, stalk-walled arms with a tip cap at each end at 156-180 h; the unmarked cyst stays a sphere. A tip cell stays tip longer: 59-68% after 10 h (42-50% before). 17 tunnels at 144 h, 150 at 180 h; the slots fill at 165 h. The closest to branching tubes of the route: the structure took the signal's size | (route A turned to rock-paper-scissors signals on Cedric's request) |
| 109 | 0109 | B | exp05_routeB_v08 | 107 with a second strobe at 125 h on the cells still unmarked (`cell_fate` -> type `ring2`, `only_types: [cell]`); `ring2` wedges and grows as `ring` | A SECOND GENERATION OF BUDS: the second strobe marks 276 cells in 12 patches (largest 66, 47, 39, 37) on the ring segments between the six first buds. They grow as the first did: 343 cells at 130 h, 667 at 140 h (130, 130, 91, 88), 2024 at 160 h, beside 4966 first-generation cells (798-910 per sac). A garland of large red and small orange buds at 140-165 h; the slots fill at 160 h | (route B turned to a tube along z with gastrulation on Cedric's request) |






## Rock-paper-scissors cysts and gastrulating tubes (from 2026-11-15 15:10, Cedric)

Cedric stopped the spots-and-waves routes after steps 108-109 and redirected both:

- Route A: "instead of Turing pattern inject rock paper scissor reaction diffusion chemical reaction see exp01-18 and make the growth-curvature dependent on the three chemicals concentration", "run 2 jobs with A rock paper scissor at a time, 16 jobs in total".
- Route B: "add growth and division directed to the Z dimension so that we obtained tube along Z and gastrulation in XY", "run 2 jobs with B gastrulated tubing at a time, 16 jobs in total".

Decisions:

- Route A (specs `exp05_rps_vNN`): the Gierer-Meinhardt pair and the tip/stalk marks are removed. Three chemicals u, v, w per cell, `cell_chem_react[rock_paper_scissor]` (May-Leonard: du/dt = r u (1 - p - a v), dv/dt = r v (1 - p - a w), dw/dt = r w (1 - p - a u), p = u + v + w; the cell-set twin of exp01 step 18's grid model) with `cell_chem_diffuse d: [D, D, D]`, started by `seed_cell_chem[simplex]`. No mark: from 90 h a cell's growth rate is a weighted sum of its three concentrations (`cell_divide` `rate_chem: {block, weights, after}`, new) and so is its wedging (`polar_nissen` `alpha_chem: {block, weights, after}`, new). First weights: growth 0.01156 u + 0.00578 v + 0 w per 10 min (a division every 10 h, 20 h, never), wedging -0.5 u + 0 v + 0.5 w.
- Before the first run the reaction was checked on a static periodic sheet of 2304 cells with the tissue's degree-normalised Laplacian (`/tmp/v4d/audit/rps_lattice.py`): rate 0.05 and diffusion 0.02 per 10 min (exp15's values, a 2.0) keep the three chemicals, in domains about 8 cells wide where a cell keeps one dominant chemical for about 55 h; rate 0.1 and diffusion 0.1 give 9 cells and 21 h; diffusion over rate of 3 or more lets one chemical take the whole sheet.
- Colours: Cedric does not want red, green, blue; from a montage of eight triplets (`log/experiments/exp05/rps/rps_palettes.png`) he chose C: u purple #5B2A86, v teal #21908C, w yellow #FDE725. The studio draws a cell in the blend of the three by the shares of its concentrations (`tools/exp29_studio.py --chem-rgb`, new; no activity legend).
- Route B (specs `exp05_gast_vNN`): a cyst (20-cell shell, `lambda: [1, 0, 0]`, `lumen_pressure[polarity]` as in the ring runs) grows round until 90 h, then divides along z only (`cell_divide[point_oriented]` `orient: axis`, `orient_after: 540`, new): a tube along z. The pacer lineage is drawn out into a stripe along z; a strobe marks the cells excited at one instant and the marked cells wedge inward: a furrow along z, an invagination in the xy slice (the movie's right panel, `--section-view 90,0`). Run length 150 h.
- Job cap: at most 5 exp05 jobs. While two MPM reruns (steps 28 and 30) hold two slots, three route jobs run at once: two of route A and one of route B; route B goes to two at a time when an MPM job ends.

| it | step | route | spec | change | result | next |
|---|---|---|---|---|---|---|
