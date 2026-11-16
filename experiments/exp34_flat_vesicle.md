---
number: 34
name: flat_vesicle
branch: simulation
title: 'flat_vesicle -- a UCB loop on the differentiable MPM sphere of movie 52, for a larger, thinner, rounder epithelial vesicle'
purpose: 'Cedric, 2026-10-29: "differentiable one straight away starting with 52 specs, 4 jobs in parallel on RTX6000, a score on larger diameter, thinner epilayer, rounder vesicle"'
status: 'launched 2026-10-29 21:35, round 1 of 4 parallel episodes; the loop resumes from its ledger'
time_scale: 'one rig frame = 1 time unit = 60 s = 100 substeps of 0.01; the optimisation frame is 0.1 time unit (10 substeps); 1 length unit = 10 um'
queue: gpu_rtx6000
wall: '240 per episode'
---

## Setup

Rig: `config/tissue/mpm_grid3x3_flatten_ves2n_slab_ov02_curv_rot.yaml` (movie 52: 100 MPM cells of 125 points on a vesicle, cell_flatten a 0.0035 per frame, slab half thickness 0.6 and stiffness 1000, contact overlap 0.2, contact tension -0.1, grid surface tension 0.15, mass growth 0.00693 per frame, the random-rotation wrapper `frame_rotation` round the substep block). Built on the differentiable bodies by `tools/exp29_diff_rig.py`; search `config/ucb_diff/exp33_ucb_flat.yaml`; driver `tools/exp34_ucb.py` with `EXP34_SEARCH=exp34_ucb_flat EXP34_RECORD=experiments/exp34_flat_vesicle`.

Levers the gradient moves inside an episode (one cell-block mean each; range in brackets, start = movie 52's value):
- `flat`, the flattening rate of `deform_control[cell_flatten]`, per frame [0, 0.03], 0.0035; step scale 20 (its effect piles up beyond a window)
- `half`, the half thickness of the slab, world length [0.15, 1.2], 0.6
- `contact_tension`, the adhesion per unit contact area [-1, 0.2], -0.1
- `tension`, the per-cell factor on the grid surface tension 0.15 [0.1, 6], 1

Levers the explorer chooses (no usable gradient or discrete): `overlap` {0.2, 0.25}, `slab_k` {300, 1000, 3000}, `division_cv` {0, 0.1}, `grow` (the mass growth rate, [1e-6, 0.0139], 0.00231), and the starts of the four above.

Loss, per window, a sum of failure fractions (zero when the gate is met):
- diameter of the cell set (2 x mean centroid distance): hinge below 1.3 x movie 52's diameter at the same frame (ramp to 1.3 at frame 200), band 0.05 of ln-ratio; curve `data/flat_reference.json`
- thin: `cell_radial_aspect` (radial over tangential rms extent of a cell, below 1 flat) ceiling 0.6, unit 0.3
- round: `set_roundness` (std over mean of the cell-centre distances from the vesicle centroid) ceiling 0.05, unit 0.03
- void fraction ceiling 0.02 and grid Gini ceiling 0.05, weight 10 each (no holes, uniform points)

Episode = a ladder of passes `50:3:2,100:2:1,150:2:1` (horizon in frames : passes : window in frames); value = minus the window-loss mean of the best pass at the last horizon. Windows of 1 to 2 frames because a 3-frame window already holds 18.7 GB on a 24 GB card at 128 cells.

## Baselines (movie 52, 230 frames, 100 to 383 cells)

diameter 6.25 to 11.61, roundness 0.008 to 0.077, wall spread 0.28 to 0.50. The radial aspect of the early rows could not be read from the trajectory (its parent index is final, stale after divisions); in the live stack it is about 0.9.

## Run

    EXP34_SEARCH=exp34_ucb_flat EXP34_RECORD=experiments/exp34_flat_vesicle PG_QUEUE=gpu_rtx6000 PYTHONPATH=src:tools python tools/exp34_ucb.py loop --n 4 --plan 50:3:2,100:2:1,150:2:1 --wall 240
    ... status | ingest | give-up EP | render-best EP...
    stop: touch experiments/exp34_flat_vesicle/data/ucb_stop_flat

Ledger `data/ledger_ucb_flat.jsonl`, pending `data/ucb_pending_flat.json`, log `data/ucb_loop_flat.log`.

## Findings

- Smoke (3 frames, one RTX6000): every knob gradient finite; the gradient of `flat` is about 25 times smaller than that of `half` per window.
- Round 1, first passes (50 frames): episodes 0 (movie 52's setup) and 3 (slab stiffness 300) descend (loss mean 0.75 to 0.64 and 0.92 to 0.80); episode 1 (flat 0.01, half 0.4) bursts (void gap +37, knobs on their bounds); episode 2 (overlap 0.25) drives `flat` to 0 and stalls.

## Decisions

- 2026-10-30 22:17 -- **Loop relaunched (flat2, same command, plan 50:5:2,100:2:1,150:2:1, wall 240).** The previous instance died at 18:57 after 4 rounds (22 episodes in the ledger, best still -1.016 = movie 52's own setup): its round-5 `bsub` over ssh was refused (`allierc@login1: Permission denied`, the VS Code agent socket of that shell had gone stale), so the four proposals of 18:57 were never submitted and the loop exited. The new instance re-proposed round 5 as episodes 22-25 (slab_k 3000 / 1000 / 300 / 300; jobs 155157016-9, all running on gpu_rtx6000 at 22:17); the log calls it "round 1" because the counter is per instance. Nothing else changed.

## exp33 deleted, its tools moved here (2026-10-31, Cedric: "delete exp033, move its tool to exp034 if some should not be deleted")

- Kept and renamed: the loop driver `tools/exp34_ucb.py`, the optimiser and step renderer `tools/exp34_optimize.py`, `exp34_progress.py`, `exp34_vertex_twin.py` (imported by two operator tests); the two searches `config/ucb_diff/exp34_ucb_flat.yaml` and `exp34_ucb_flat2.yaml` (names changed: the ledger and the run names `exp33_ucb_flat2_e0NN` already written stay as they are; new episodes are `exp34_ucb_flat2_e0NN`); env variables EXP33_* became EXP34_*; logs `log/experiments/exp34/`.
- Deleted: the exp33 record (47 steps), its specs and searches (77 + 6 configs), six exp33-only tools, the rulers `exp33.fit` and their test, and 116 GB of its own run folders; the 59 GB of exp34's runs (`graphs_data/tissue/exp33_ucb_flat*`) stay. Its written findings, audits and journal are in `data/from_exp33/` (the exp33 note with Findings 1-28, the UCB audit of 2026-10-28, the handoff and plan documents).
- The loop was stopped at Cedric's word (round 6 jobs killed); relaunch: `EXP34_SEARCH=exp34_ucb_flat2 EXP34_RECORD=experiments/exp34_flat_vesicle PG_QUEUE=gpu_rtx6000 PYTHONPATH=src:tools python tools/exp34_ucb.py loop --n 4 --plan 50:5:2,100:2:1,150:2:1 --wall 240`.
