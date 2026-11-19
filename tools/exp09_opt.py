"""exp09 speed optimisation of the run `rna1_3d` (step 25: 5,545 MPM bodies, 1.19 s per frame on one H200), 2026-11-19. A copy of the pattern of tools/exp09_speed.py
(one scene, a few frames, several configurations, each in its own process) with what that tool lacks: a per-operator profile, configurations named by the changes they
make, a health run (the bodies' size, the chain bonds, the held bonds, the motion per frame over 200 frames or more) and scenes with more bodies.

A configuration is `base` or tokens joined by `+`, each one change to the spec (see TOKENS): `cc+bath+wall` is the Warp contact penalty, the Warp bath and the Warp walls.
A scene is `rna1_3d` (the spec file as it is, read and never written) or one of SCENES (rna1_3d with more bodies in a wider box, built by `build` into tmp/exp09_opt/).

    python tools/exp09_opt.py profile <scene> <tag> --configs base,cc [--frames 6]       per-operator wall time (cuda-synchronised timers round every forward)
    python tools/exp09_opt.py kprofile <scene> <tag> --configs warp [--frames 6]         GPU time of every Warp kernel (CUDA events), per substep
    python tools/exp09_opt.py run <scene> <tag> --configs base,base2,cc [--frames 10]    each configuration in its own process; data/opt_<tag>.json: time per frame,
                                                                                        peak GPU memory, the final state against the FIRST configuration of the list
    python tools/exp09_opt.py health <scene> <tag> --configs base,... [--frames 200]     the same with the health measures sampled every 10 frames
    python tools/exp09_opt.py build <scene>                                             a scaled scene's seed table and spec, into tmp/exp09_opt/
    python tools/exp09_opt.py one <scene> <config> <out.npz> [--frames N] [--health]     one configuration (the child process of run and health)
    python tools/exp09_opt.py write <scene> <config> <name>                             the scene under a configuration as config/tissue/exp09_land_opt_<name>.yaml
    python tools/exp09_opt.py generate <name>                                           Plexus_Main -o generate of that spec, the new operators imported first
    python tools/exp09_opt.py trajhealth <run> [<run> ...]                              the health measures of recorded runs at frames 200, 500, 1,000

Everything that simulates runs on the GPU; `build` is numpy only. The new operator variants are in src/plexus/operators/substep_warp_ops.py and are
registered by importing that module here (plexus.operators.__init__ does not know them).
"""
import argparse
import collections
import copy
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools"), os.path.join(ROOT, "discovery_okuda")]
import yaml  # noqa: E402

import exp09_landscape as E  # noqa: E402

OPT = os.path.join(ROOT, "tmp", "exp09_opt")              # this tool's own seeds and specs (the shared file system: the cluster reads them)
NEW_OPS = "plexus.operators.substep_warp_ops"             # the new variants: imported by every process that loads a spec naming them

# ---- the scenes ---------------------------------------------------------------------------------------------------------------------------------------------------
# rna1_3d with f times the bodies: every protein count and every chain count x f, the box wider in x and y by sqrt(f) (its depth 192 nm kept), as E._scaled does for
# `scale_to` (which scales the counts of free bodies only, not the chains); the junction table's rows x f
SCENES = {
    "opt_s20k": dict(f=3.6, why="rna1_3d x 3.6: 19,995 bodies (83 strands of each mRNA type, 630 proteins of each type) in a box 647 x 364 x 192 nm"),
    "opt_s55k": dict(f=10.0, why="rna1_3d x 10: 55,450 bodies (230 strands of each mRNA type, 1,750 proteins of each type) in a box 1,078 x 607 x 192 nm"),
    "opt_s554k": dict(f=100.0, why="rna1_3d x 100: 554,500 bodies (2,300 strands of each mRNA type, 17,500 proteins of each type) in a box 3,410 x 1,920 x 192 nm"),
    # the periodic box rna1_3dp with f times the bodies: its sides whole multiples of 8 nm (0.8 length units: the contact lattice needs whole point spacings of 0.2, and
    # the periodic search a hash-grid cell that divides the three sides and is not smaller than its radius 0.55), the volume within 1.5 % of f times rna1_3dp's
    "opt_p20k": dict(f=3.6, parent="rna1_3dp", box=(64.8, 36.8, 19.2), why="rna1_3dp x 3.6: 19,995 bodies in a periodic box 648 x 368 x 192 nm"),
    "opt_p55k": dict(f=10.0, parent="rna1_3dp", box=(107.2, 60.8, 19.2), why="rna1_3dp x 10: 55,450 bodies in a periodic box 1,072 x 608 x 192 nm"),
}


def build(scene):
    """The seed table and the spec of a scaled scene, written to tmp/exp09_opt/ (E.DATA is pointed there while E.build runs: nothing is written into the record)."""
    f = float(SCENES[scene]["f"])
    parent = SCENES[scene].get("parent", "rna1_3d")
    base = E.params(parent)
    E.ITERS[scene] = dict(parent=parent, why=SCENES[scene]["why"],
                          box=SCENES[scene].get("box") or (base["box"][0] * math.sqrt(f), base["box"][1] * math.sqrt(f), base["box"][2]),
                          counts={k: int(round(v * f)) for k, v in base["counts"].items()},
                          chains=[dict(ch, n=int(round(ch["n"] * f))) for ch in base["chains"]],
                          junction_buffer=int(2 ** math.ceil(math.log2(16384 * f))), section=None, panels3d=False,
                          seed_batched=f > 5.0)       # from_points_ragged_batched above about 50,000 bodies: the default seed costs bodies x points (26 minutes at 554,500)
    os.makedirs(OPT, exist_ok=True)
    keep = E.DATA
    E.DATA = OPT
    try:
        t0 = time.time()
        sp = E.build(scene)
        E._fast(sp, E.params(scene)["fast"])
    finally:
        E.DATA = keep
    out = os.path.join(OPT, f"spec_{scene}.yaml")
    yaml.safe_dump(sp, open(out, "w"), sort_keys=False, width=1000)
    print(f"built {scene}: {sp['sets']['cell']['n']:,} bodies, world {sp['general']['world']}, seed {sp['seed'][0]['points']} ({time.time() - t0:.0f} s) -> {os.path.relpath(out, ROOT)}", flush=True)


def scene_spec(scene):
    """The spec of a scene: one of SCENES (built into tmp/exp09_opt/), else the written spec config/tissue/exp09_land_<scene>.yaml as it is (rna1_3d, rna1_3dp)."""
    if scene in SCENES:
        return yaml.safe_load(open(os.path.join(OPT, f"spec_{scene}.yaml")))
    return yaml.safe_load(open(os.path.join(ROOT, "config", "tissue", f"exp09_land_{scene}.yaml")))


# ---- the configurations -------------------------------------------------------------------------------------------------------------------------------------------
def _op(sp, name, model=None):
    return [o for o in sp["operators"] if o["op"] == name and (model is None or o.get("model") == model)]


def _impl(sp, name, impl, model=None, axis="implementation"):
    for o in _op(sp, name, model):
        o.pop("model", None)
        o.pop("implementation", None)
        o[axis] = impl


def _block(sp):
    return next(st for st in sp["schedule"] if isinstance(st, dict) and "substep_dt" in st)


def _frame_call(sp, name):
    """Name `name` once in the frame-level schedule too, right before the substep block: the i-th mention of a name in a tick runs its i-th instance, and the block
    clears the count before every substep, so the one instance runs once per frame (where it refreshes what its substep kernel reads) and once per substep."""
    i = next(k for k, st in enumerate(sp["schedule"]) if isinstance(st, dict) and "substep_dt" in st)
    if name not in sp["schedule"][:i]:
        sp["schedule"].insert(i, name)


def _substep(sp, dt):
    _block(sp)["substep_dt"] = float(dt)
    for o in _op(sp, "thermal_bath"):
        o["substep_dt"] = float(dt)                  # the bath's kicks are sized for the substep they are drawn in


def _grid(sp, n):
    g = sp["fields"]["mpm_grid"]
    dx = float(g["cell_box"]) / int(g["n_grid"])
    g["n_grid"], g["cell_box"] = int(n), round(int(n) * dx, 6)   # the same cell (2 nm), fewer of them


TOKENS = {
    "cc":    lambda sp: _impl(sp, "pair_potential", "cell_contact_warp_pp"),                    # the existing Warp contact penalty
    "ccf":   lambda sp: (_impl(sp, "pair_potential", "cell_contact_warp_pp_fixed"), _frame_call(sp, "pair_potential")),
    "bath":  lambda sp: _impl(sp, "thermal_bath", "langevin_warp"),
    "wall":  lambda sp: _impl(sp, "active_force", "box_slab_warp", model="box_slab", axis="model"),
    "bs":    lambda sp: (_impl(sp, "bond_springs", "warp"), _frame_call(sp, "bond_springs")),
    "aj":    lambda sp: _impl(sp, "apical_junction", "warp"),
    "sc":    lambda sp: _impl(sp, "mpm_scatter", "warp_pp_lean"),
    "gu":    lambda sp: _impl(sp, "mpm_grid_update", "warp_pp_plain"),
    "ccs":   lambda sp: (_impl(sp, "pair_potential", "cell_contact_warp_pp_sparse"), _frame_call(sp, "pair_potential")),
    "scs":   lambda sp: _impl(sp, "mpm_scatter", "warp_pp_sparse"),
    "gas":   lambda sp: (_impl(sp, "mpm_gather", "warp_pp_solve"), _impl(sp, "mpm_grid_update", "solved_in_gather")),
    "cfw":   lambda sp: _impl(sp, "contact_finder", "sites_wide"),
    "cfcpu": lambda sp: [o.__setitem__("search", "cpu") for o in _op(sp, "contact_finder")],        # a periodic box searched by the CPU kd-tree, as until 2026-11-19 (step 26)
    "bm2":   lambda sp: [o.__setitem__("block_margin", 2) for o in _op(sp, "pair_potential")],       # the contact blocks 2 lattice nodes wider than the widest body at the start (default 1): room for
                                                                                                    # a body that swells, so that the blocks are not enlarged mid-run and a captured graph dropped
    "cap":   lambda sp: _block(sp).__setitem__("capture", True),
    "dt02":  lambda sp: _substep(sp, 0.02),
    "dt0125": lambda sp: _substep(sp, 0.0125),                                                    # 80 substeps per frame
    "dt0167": lambda sp: _substep(sp, 1.0 / 60.0),                                                # 60
    "dt025": lambda sp: _substep(sp, 0.025),
    "dt04":  lambda sp: _substep(sp, 0.04),
    "dt05":  lambda sp: _substep(sp, 0.05),
    "s2":    lambda sp: [o.__setitem__("seed", 2) for o in _op(sp, "thermal_bath")],          # another noise stream: what two runs of one configuration differ by in the health measures
    "g14":   lambda sp: _grid(sp, 14),
    "g13":   lambda sp: _grid(sp, 13),
    "g12":   lambda sp: _grid(sp, 12),
    "g11":   lambda sp: _grid(sp, 11),
}
ALIAS = {"warp": "ccf+bath+wall+bs+aj+sc+gu", "nobath": "ccf+wall+bs+aj+sc+gu",                 # every substep operator a Warp variant (nobath: the torch bath, for an equality test)
         "sparse": "ccs+bath+wall+bs+aj+scs+gas", "sparsenb": "ccs+wall+bs+aj+scs+gas",             # ... with no pass over the grid's or the lattice's nodes (measured SLOWER: see the report)
         "best": "ccf+bath+wall+bs+aj+sc+gas"}                                                      # the fastest set measured: warp, with the grid solve done in the gather


def tokens_of(cfg):
    out = []
    for t in cfg.split("+"):
        if t in ("base", "base2", ""):
            continue
        out += tokens_of(ALIAS[t]) if t in ALIAS else [t]
    return out


def spec_of(scene, cfg, frames):
    sp = copy.deepcopy(scene_spec(scene))
    g = sp["general"]
    g.update(name=f"exp09_opt_{scene}_{cfg.replace('+', '_')}", n_frames=int(frames), record_cap=2)
    for k in ("save_chunk_frames", "save_every_min", "warmup", "save_chunk_window"):
        g.pop(k, None)
    for t in tokens_of(cfg):
        TOKENS[t](sp)
    return sp


def write(scene, cfg, name):
    """The scene's spec under configuration `cfg` as a spec file of its own, config/tissue/exp09_land_opt_<name>.yaml: every general setting of the scene kept (its
    1,000 frames, the rows it records, its partial saves), so that `generate` runs it as the scene's production job is run."""
    sp = copy.deepcopy(scene_spec(scene))
    run = f"exp09_land_opt_{name}"
    sp["general"]["name"] = run
    sp["general"]["headline"] = f"exp09 opt_{name}: {scene} under the speed configuration {cfg} (tools/exp09_opt.py)"
    for t in tokens_of(cfg):
        TOKENS[t](sp)
    out = os.path.join(ROOT, "config", "tissue", run + ".yaml")
    yaml.safe_dump(sp, open(out, "w"), sort_keys=False, width=1000)
    print("wrote", os.path.relpath(out, ROOT))
    return run


def generate(name):
    """`Plexus_Main.py -o generate tissue/exp09_land_opt_<name>` with the new operator module imported first (plexus.operators.__init__ does not import it, so the plain
    command refuses the spec's variants as unknown)."""
    import runpy
    import plexus.operators  # noqa: F401
    __import__(NEW_OPS)
    sys.argv = ["Plexus_Main.py", "-o", "generate", f"tissue/exp09_land_opt_{name}", "--device", "cuda:0", "--force", "--no-describe", "--no-viz"]
    t0 = time.time()
    try:
        runpy.run_path(os.path.join(ROOT, "Plexus_Main.py"), run_name="__main__")
    finally:
        print(f"[opt] generate exp09_land_opt_{name}: {time.time() - t0:.1f} s of wall time, start to end (seeding, the frames, the recording, the save)", flush=True)


def traj_health(run, rows=(50, 125, 250)):
    """The health measures of a recorded run (graphs_data/tissue/<run>/trajectory.npz, a row every 4 frames) at the given rows: every body's radius of gyration over its
    own at row 0, the chain bonds' length over their own at row 0, the junctions that hold a bond, the rms distance a body's centre moved since the row before (4 frames), nm."""
    rd = os.path.join(E.GD, run)
    z = np.load(os.path.join(rd, "trajectory.npz"), allow_pickle=True)
    sp = yaml.safe_load(open(os.path.join(rd, "spec.yaml")))
    a, b = _chain_pairs(sp)
    occ, X, pid = z["pt__occ"].astype(bool), z["pt__pos"], np.rint(z["pt__pid"][..., 0]).astype(np.int64)
    nC = int(z["cell__occ"].shape[1])

    def geom(r):
        L = occ[r]
        x, q = X[r][L].astype(np.float64), pid[r][L]
        cnt = np.maximum(np.bincount(q, minlength=nC), 1)
        cen = np.stack([np.bincount(q, weights=x[:, k], minlength=nC) for k in range(3)], 1) / cnt[:, None]
        return cen, np.sqrt(np.bincount(q, weights=((x - cen[q]) ** 2).sum(1), minlength=nC) / cnt)
    c0, rg0 = geom(0)
    bl0 = np.linalg.norm(c0[b] - c0[a], axis=1)
    npt = np.bincount(pid[0][occ[0]], minlength=nC)
    prot, sub = npt > 27, (npt > 0) & (npt <= 27)                 # the 64-point proteins, the 27-point mRNA subunits
    out = []
    # the motion over 40 frames (10 rows), from frame 200 on: the rms over the bodies of a class of the distance a body's centre moved, averaged over the windows
    # starting at rows 50, 60, ... (frames 200, 240, ...); a chain bond is centre to centre and a body is never cut by a periodic face, so nothing is unwrapped here
    last = occ.shape[0] - 1
    starts = [r for r in range(50, last - 9, 10)]
    if starts:
        cs = {r: geom(r)[0] for r in sorted(set(starts) | {r + 10 for r in starts})}
        mv = {nm: [float(np.sqrt(((cs[r + 10][m] - cs[r][m]) ** 2).sum(1).mean()) * E.NM) for r in starts] for nm, m in (("protein", prot), ("subunit", sub))}
        jo_l = z["junction__occ"][last] > 0.5
        held_l = int((z["junction__rho"][last][..., 0][jo_l] > 0.5).sum())
        q = dict(row=-1, frame=int(4 * last), protein_move40_nm=float(np.mean(mv["protein"])), protein_move40_first_nm=mv["protein"][0], subunit_move40_nm=float(np.mean(mv["subunit"])),
                 subunit_move40_first_nm=mv["subunit"][0], windows=len(starts), held=held_l, proteins=int(prot.sum()), bonds_per_protein=held_l / max(int(prot.sum()), 1))
        out.append(q)
        print(f"{run}: over 40 frames a protein moves {q['protein_move40_nm']:.2f} nm rms (mean of {len(starts)} windows from frame 200; the first window {q['protein_move40_first_nm']:.2f}), "
              f"an mRNA subunit {q['subunit_move40_nm']:.2f} nm (first window {q['subunit_move40_first_nm']:.2f}); held bonds {held_l} at frame {4 * last}, {q['bonds_per_protein']:.2f} per protein", flush=True)
    for r in rows:
        if r >= occ.shape[0]:
            continue
        c, rg = geom(r)
        cp, _ = geom(r - 1)
        ratio, bl = rg / np.maximum(rg0, 1e-9), np.linalg.norm(c[b] - c[a], axis=1) / bl0
        jo = z["junction__occ"][r] > 0.5
        out.append(dict(row=int(r), frame=int(4 * r), rg_min=float(ratio.min()), rg_max=float(ratio.max()), rg_out=int(((ratio < 0.8) | (ratio > 1.25)).sum()), bond_max=float(bl.max()),
                        bond_out=int((bl > 1.5).sum()), held=int((z["junction__rho"][r][..., 0][jo] > 0.5).sum()), rows=int(jo.sum()),
                        step4_nm=float(np.sqrt(((c - cp) ** 2).sum(1).mean()) * E.NM), finite=bool(np.isfinite(X[r][occ[r]]).all()),
                        # the swelling per size class: the 27-point mRNA subunits (they swell slowly, the more the longer the substep) and the 64-point proteins
                        rg_mean_small=float(ratio[sub].mean()), rg_p999_small=float(np.percentile(ratio[sub], 99.9)), rg_max_small=float(ratio[sub].max()),
                        n110_small=int((ratio[sub] > 1.10).sum()), rg_mean_large=float(ratio[prot].mean()), rg_max_large=float(ratio[prot].max()), bond_mean=float(bl.mean())))
        q = out[-1]
        print(f"{run} frame {q['frame']}: radius of gyration over start {q['rg_min']:.3f} to {q['rg_max']:.3f} (outside 0.8-1.25: {q['rg_out']}), chain bond over start at most {q['bond_max']:.3f} "
              f"(over 1.5: {q['bond_out']}), held bonds {q['held']} (rows {q['rows']}), motion over 4 frames {q['step4_nm']:.4f} nm, finite {q['finite']}; 27-point bodies mean {q['rg_mean_small']:.4f}, "
              f"99.9 % {q['rg_p999_small']:.4f}, largest {q['rg_max_small']:.4f}, over 1.10: {q['n110_small']}; 64-point bodies mean {q['rg_mean_large']:.4f}, largest {q['rg_max_large']:.4f}", flush=True)
    return out


# ---- one configuration ----------------------------------------------------------------------------------------------------------------------------------------------
def _chain_pairs(sp):
    """The two bodies of every chain bond of an rna scene: the bodies of an mRNA kind m<k> are its chains one after the other, each of int(round(5 + 2.55 k)) subunits
    (E.ITERS rna1); a bond joins two consecutive subunits of one chain."""
    a, b, base = [], [], 0
    for nm, t in sp["sets"]["cell"]["types"].items():
        n = int(t["count"])
        if nm.startswith("m") and nm[1:].isdigit():
            m = int(round(5 + 2.55 * int(nm[1:])))
            ids = base + np.arange(n).reshape(-1, m)
            a += list(ids[:, :-1].ravel())
            b += list(ids[:, 1:].ravel())
        base += n
    return np.asarray(a, np.int64), np.asarray(b, np.int64)


class Health:
    """Sampled on the device every `every` frames (and the frame after, for the motion per frame): every body's radius of gyration over its own at frame 0, every chain
    bond's length (centre to centre of two consecutive subunits) over its own at frame 0, the number of junctions that hold a bond (bound share above 0.5), the rms over
    the bodies of the distance a body's centre moved in one frame (nm), and the points' kinetic temperature m <v^2> / 3 over the bath's kT."""

    def __init__(self, sp, every=10):
        self.sp, self.every, self.rows, self.rg0, self.bl0, self.prev = sp, int(every), [], None, None, None
        self.kT = float(next(o for o in sp["operators"] if o["op"] == "thermal_bath")["kT"])

    def _geometry(self, H):
        import torch
        P = H.level("pt")
        X = P.get("pos").double()
        live = (P.occ > 0.5).double()
        cid = H.lift_index("pt", "cell").long()
        nC = int(H.level("cell").state.shape[0])
        cnt = torch.zeros(nC, dtype=torch.float64, device=X.device).index_add_(0, cid, live).clamp_min(1.0)
        cen = torch.zeros(nC, 3, dtype=torch.float64, device=X.device).index_add_(0, cid, X * live[:, None]) / cnt[:, None]
        rg = torch.sqrt(torch.zeros(nC, dtype=torch.float64, device=X.device).index_add_(0, cid, ((X - cen[cid]) ** 2).sum(1) * live) / cnt)
        return cen, rg

    def __call__(self, H, tick):
        import torch
        if tick % self.every not in (0, 1):
            return
        cen, rg = self._geometry(H)
        if tick % self.every == 0:
            self.prev = cen
            if self.rg0 is None:
                a, b = _chain_pairs(self.sp)
                self.ba = torch.as_tensor(a, device=cen.device)
                self.bb = torch.as_tensor(b, device=cen.device)
                self.rg0 = rg.clamp_min(1e-9)
                self.bl0 = (cen[self.bb] - cen[self.ba]).norm(dim=1).clamp_min(1e-9)
                P0 = H.level("pt")
                cnt = torch.zeros(cen.shape[0], device=cen.device).index_add_(0, H.lift_index("pt", "cell").long(), (P0.occ > 0.5).float())
                self.small = (cnt > 0) & (cnt <= 27)                 # the chain subunits (27 points); the others are the proteins (64)
                self.large = cnt > 27
            return
        P = H.level("pt")
        ratio = rg / self.rg0
        bl = (cen[self.bb] - cen[self.ba]).norm(dim=1) / self.bl0
        J = H.level("junction")
        held = int(((J.get("rho")[:, 0] > 0.5) & (J.occ > 0.5)).sum())
        live = P.occ > 0.5
        V = P.get("vel")[live].double()
        temp = float((P.mass[live].double() * (V * V).sum(1)).mean() / 3.0 / self.kT)
        step = float(((cen - self.prev) ** 2).sum(1).mean().sqrt()) * E.NM
        self.rows.append(dict(frame=int(tick), rg_min=float(ratio.min()), rg_max=float(ratio.max()), rg_out=int(((ratio < 0.8) | (ratio > 1.25)).sum()),
                              bond_max=float(bl.max()) if bl.numel() else 0.0, bond_out=int((bl > 1.5).sum()), held=held, rows=int((J.occ > 0.5).sum()),
                              step_nm=step, temp=temp, finite=bool(torch.isfinite(P.get("pos")[live]).all()),
                              # per size class: the mean, the 99.9th percentile and the largest of the radius of gyration over its start, and how many bodies are over 1.10
                              **{f"rg_{st}_{nm}": float(fn(ratio[m])) if bool(m.any()) else 1.0 for nm, m in (("small", self.small), ("large", self.large))
                                 for st, fn in (("mean", torch.mean), ("p999", lambda q: torch.quantile(q, 0.999)), ("max", torch.max))},
                              n110_small=int((ratio[self.small] > 1.10).sum()), n110_large=int((ratio[self.large] > 1.10).sum()),
                              bond_mean=float(bl.mean()) if bl.numel() else 0.0, bond_p999=float(torch.quantile(bl, 0.999)) if bl.numel() else 0.0))


def one(scene, cfg, out, frames, health, profile=False):
    """Run one configuration: frame times, peak memory, the state after the last frame (and the health rows, or the per-operator profile) to `out`."""
    import torch
    import plexus.operators  # noqa: F401
    __import__(NEW_OPS)
    from plexus import schema
    from plexus.engine import run
    sp = spec_of(scene, cfg, frames)
    path = os.path.join("/tmp", f"{sp['general']['name']}.yaml")
    yaml.safe_dump(sp, open(path, "w"), sort_keys=False, width=1000)
    sim = schema.load(path)
    hl = Health(sp) if health else None
    T, N, blk = collections.defaultdict(float), collections.Counter(), [0.0]
    in_block = set(_block(sp)["steps"])
    captured = bool(_block(sp).get("capture"))

    def on_ready(H):
        if not profile:
            return
        for nm, ob in zip(H.operator_names, H.operators):
            key = f"{nm}[{type(ob).__name__}]"
            f = ob.forward

            def timed(*a, _f=f, _k=key, **k):
                if captured and getattr(a[0], "sub_dt", None) is not None:      # inside a block that is captured: no synchronisation is allowed there, the block is timed as a whole
                    return _f(*a, **k)
                torch.cuda.synchronize(); t = time.perf_counter(); r = _f(*a, **k); torch.cuda.synchronize(); T[_k] += time.perf_counter() - t; N[_k] += 1
                return r
            ob.forward = timed

    if profile is True:                                                   # inside contact_finder: the neighbour search and the pieces of its bookkeeping
        import plexus.operators.adhesion_warp_ops as AW
        for fn in ("_partners", "_seg_sum3", "_known", "_lookup"):
            f0 = getattr(AW, fn)

            def timed_fn(*a, _f=f0, _k=f"  contact_finder.{fn}", **k):
                torch.cuda.synchronize(); t = time.perf_counter(); r = _f(*a, **k); torch.cuda.synchronize(); T[_k] += time.perf_counter() - t; N[_k] += 1
                return r
            setattr(AW, fn, timed_fn)
    if profile is True:                                                   # the substep block as a whole (its operators and the engine's bookkeeping round them)
        import plexus.engine as eng
        rsb = eng.run_substep_block

        def timed_block(*a, **k):
            torch.cuda.synchronize(); t = time.perf_counter(); r = rsb(*a, **k); torch.cuda.synchronize(); blk[0] += time.perf_counter() - t
            return r
        eng.run_substep_block = timed_block
    kern = {}
    on_frame = hl
    if profile == "kernels":                                              # GPU time of every Warp kernel launch (CUDA events), from the end of frame 2 to the end of the run
        import warp as wp

        def on_frame(H, tick):
            if tick == 2:
                wp.timing_begin(cuda_filter=wp.TIMING_KERNEL, synchronize=True)
            if tick == int(frames):
                for r in wp.timing_end(synchronize=True):
                    q = kern.setdefault(str(r.name), [0.0, 0])
                    q[0] += float(r.elapsed)
                    q[1] += 1
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    H, res = run(sim, out_path=None, device="cuda:0", progress=False, on_ready=on_ready if profile is True else None, on_frame=on_frame)
    torch.cuda.synchronize()
    wall = time.time() - t0
    ms = np.asarray(res["frame_ms"], np.float64)
    st = {"frame_ms": ms, "wall_s": wall, "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
          "gpu": torch.cuda.get_device_name(0), "n_cells": int(H.level("cell").state.shape[0]), "n_points": int((H.level("pt").occ > 0.5).sum())}
    try:
        import warp as wp
        st["warp_gb"] = float(wp.get_mempool_used_mem_high("cuda:0")) / 1e9      # Warp's own pool is not in torch's count
    except Exception:  # noqa: BLE001
        st["warp_gb"] = -1.0
    pt = H.level("pt")
    st["pt_pos"] = pt.get("pos").detach().cpu().numpy()
    st["pt_occ"] = pt.active.detach().cpu().numpy()
    st["cell_pos"] = H.level("cell").get("pos").detach().cpu().numpy()
    J = H.level("junction")
    st["j_rho"] = J.get("rho").detach().cpu().numpy()
    st["j_occ"] = J.active.detach().cpu().numpy()
    if hl is not None:
        st["health"] = json.dumps(hl.rows)
    if profile == "kernels":
        nsub = (int(frames) - 2) * max(1, round(float(sp["general"]["dt"]) / float(_block(sp)["substep_dt"])))
        st["kernels"] = json.dumps(dict(substeps=nsub, rows=[dict(kernel=k, ms_total=v[0], launches=v[1], ms_per_substep=v[0] / nsub) for k, v in kern.items()]))
    if profile is True:
        nfr = len(ms)
        rows = [dict(op=k, s_per_frame=v / nfr, calls_per_frame=N[k] / nfr, ms_per_call=1000 * v / max(N[k], 1), in_block=k.split("[")[0] in in_block) for k, v in T.items()]
        st["profile"] = json.dumps(dict(rows=rows, block_s_per_frame=blk[0] / nfr, frame_s=float(ms.sum()) / 1000 / nfr))
    np.savez(out, **st)


def _frame_time(ms):
    """Median over the frames after the first two (they carry the warm-up, the kernels' loading and the capture)."""
    return float(np.median(ms[2:])) if len(ms) > 3 else float(np.median(ms))


def run_all(scene, tag, configs, frames, health, profile=False):
    tmp = os.path.join("/tmp", f"exp09_opt_{tag}")
    os.makedirs(tmp, exist_ok=True)
    res = {"_scene": scene, "_frames": int(frames), "_ref": configs[0]}
    outp = os.path.join(E.DATA, f"opt_{tag}.json")
    for cfg in configs:
        out = os.path.join(tmp, f"{cfg.replace('+', '_')}.npz")
        t0 = time.time()
        r = subprocess.run([sys.executable, "-u", __file__, "one", scene, cfg, out, "--frames", str(frames)] + (["--health"] if health else [])
                           + (["--profile"] if profile is True else ["--kernels"] if profile == "kernels" else []), capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            tail = (r.stderr or r.stdout).strip().splitlines()[-12:]
            res[cfg] = {"error": " | ".join(tail)[:2400]}
            print(f"{cfg}: FAILED ({time.time() - t0:.0f} s): {res[cfg]['error']}", flush=True)
            json.dump(res, open(outp, "w"), indent=1)
            continue
        for ln in (r.stdout or "").splitlines():                         # what the engine and the operators said (capture, refusals, block sizes)
            if any(s in ln for s in ("captured", "capture refused", "not captured", "[cell_contact", "[bond_springs", "[apical_junction", "reallocated", "FULL", "[opt]")):
                print(f"  {cfg} | {ln[:300]}", flush=True)
        z = np.load(out)
        ms = z["frame_ms"]
        res[cfg] = {"s_per_frame": _frame_time(ms) / 1000.0, "first_two_ms": [float(v) for v in ms[:2]], "mean_s_per_frame": float(ms[2:].mean()) / 1000.0 if len(ms) > 3 else None,
                    "peak_gb": float(z["peak_gb"]), "reserved_gb": float(z["reserved_gb"]), "warp_gb": float(z["warp_gb"]), "wall_s": float(z["wall_s"]), "gpu": str(z["gpu"]),
                    "n_cells": int(z["n_cells"]), "n_points": int(z["n_points"])}
        print(f"{cfg}: {res[cfg]['s_per_frame']:.4f} s per frame (median of frames 2-{frames}) on {res[cfg]['gpu']}, first two {ms[0]:.0f} / {ms[1]:.0f} ms, peak GPU memory torch {res[cfg]['peak_gb']:.2f} GB "
              f"(reserved {res[cfg]['reserved_gb']:.2f}) + warp {res[cfg]['warp_gb']:.2f} GB, {res[cfg]['n_cells']:,} bodies, {res[cfg]['n_points']:,} points", flush=True)
        if "health" in z.files:
            rows = json.loads(str(z["health"]))
            res[cfg]["health"] = rows
            if rows:
                last, half = rows[-1], rows[len(rows) // 2:]
                print(f"  {cfg} health at frame {last['frame']}: radius of gyration over start {min(q['rg_min'] for q in rows):.3f} to {max(q['rg_max'] for q in rows):.3f} (bodies outside 0.8-1.25: {max(q['rg_out'] for q in rows)}), "
                      f"chain bond over start at most {max(q['bond_max'] for q in rows):.3f} (over 1.5: {max(q['bond_out'] for q in rows)}), held bonds {last['held']} (rows {last['rows']}), "
                      f"motion {np.mean([q['step_nm'] for q in half]):.4f} nm per frame, temperature {np.mean([q['temp'] for q in half]):.3f} kT (second half), finite {all(q['finite'] for q in rows)}", flush=True)
                if "rg_mean_small" in last:
                    print(f"  {cfg} at frame {last['frame']}: 27-point bodies' radius of gyration over start mean {last['rg_mean_small']:.4f}, 99.9 % {last['rg_p999_small']:.4f}, largest {last['rg_max_small']:.4f}, "
                          f"over 1.10: {last['n110_small']}; 64-point bodies mean {last['rg_mean_large']:.4f}, largest {last['rg_max_large']:.4f}, over 1.10: {last['n110_large']}; "
                          f"chain bonds mean {last['bond_mean']:.4f}, 99.9 % {last['bond_p999']:.4f}", flush=True)
        if "kernels" in z.files:
            kr = json.loads(str(z["kernels"]))
            res[cfg]["kernels"] = kr
            print(f"  KERNELS {cfg}: GPU time of the Warp kernels, {sum(q['ms_per_substep'] for q in kr['rows']):.4f} ms per substep in all (over {kr['substeps']} substeps; torch operations not counted)", flush=True)
            for q in sorted(kr["rows"], key=lambda q: -q["ms_total"]):
                print(f"  KERNELS {q['kernel']:<48s} {q['ms_per_substep']:8.4f} ms per substep  {q['launches']:7d} launches  {1000 * q['ms_total'] / max(q['launches'], 1):9.2f} us per launch", flush=True)
        if "profile" in z.files:
            pr = json.loads(str(z["profile"]))
            res[cfg]["profile"] = pr
            tot = sum(q["s_per_frame"] for q in pr["rows"] if not q["op"].startswith("  "))
            inb = sum(q["s_per_frame"] for q in pr["rows"] if q["in_block"])
            print(f"  PROFILE {cfg}: frame {pr['frame_s']:.4f} s (synchronised timers on), operators {tot:.4f} s, substep block {pr['block_s_per_frame']:.4f} s of which its operators {inb:.4f} s", flush=True)
            for q in sorted(pr["rows"], key=lambda q: -q["s_per_frame"]):
                print(f"  PROFILE {q['op']:<62s} {1000 * q['s_per_frame']:9.2f} ms per frame {100 * q['s_per_frame'] / pr['frame_s']:5.1f} %  {q['calls_per_frame']:6.1f} calls  {q['ms_per_call']:8.3f} ms per call  {'substep' if q['in_block'] else 'frame'}", flush=True)
        json.dump(res, open(outp, "w"), indent=1)
    ref = configs[0]
    if ref in res and "error" not in res[ref]:
        b = np.load(os.path.join(tmp, f"{ref.replace('+', '_')}.npz"))
        for cfg in configs[1:]:
            if "error" in res.get(cfg, {"error": 1}):
                continue
            z = np.load(os.path.join(tmp, f"{cfg.replace('+', '_')}.npz"))
            cmp = {}
            for k, unit in (("pt_pos", E.NM), ("cell_pos", E.NM), ("j_rho", 1.0)):
                m = b["pt_occ"].astype(bool) if k == "pt_pos" else (b["j_occ"].astype(bool) if k == "j_rho" else None)
                if b[k].shape != z[k].shape:
                    continue
                a1, a2 = (b[k], z[k]) if m is None else (b[k][m], z[k][m])
                d = np.abs(a1.astype(np.float64) - a2.astype(np.float64))
                cmp[k] = {"bit_identical": bool(np.array_equal(a1, a2)), "max_abs": float(d.max()) * unit if d.size else 0.0, "rms": float(np.sqrt((d ** 2).mean())) * unit if d.size else 0.0}
            res[cfg]["vs_ref"] = cmp
            print(f"{cfg} vs {ref}: " + "; ".join(f"{k} {'BIT-IDENTICAL' if v['bit_identical'] else 'differs'} (max {v['max_abs']:.3g}, rms {v['rms']:.3g}{' nm' if k != 'j_rho' else ''})" for k, v in cmp.items()), flush=True)
    json.dump(res, open(outp, "w"), indent=1)
    print("wrote", os.path.relpath(outp, ROOT), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "health", "profile", "kprofile", "one", "build", "write", "generate", "trajhealth"])
    ap.add_argument("scene")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--configs", default="base")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--health", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--kernels", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.scene)
    elif a.cmd == "write":                                   # write <scene> <config> <name>
        write(a.scene, a.rest[0], a.rest[1])
    elif a.cmd == "generate":                                # generate <name>
        generate(a.scene)
    elif a.cmd == "trajhealth":                              # trajhealth <tag> <run> [<run> ...]: data/opt_<tag>.json  (one argument: a run, data/opt_trajhealth.json)
        tag, runs = (a.scene, a.rest) if a.rest and not a.scene.startswith("exp09_land_") else ("trajhealth", [a.scene] + a.rest)
        res = {r: traj_health(r) for r in runs}
        json.dump(res, open(os.path.join(E.DATA, f"opt_{tag}.json"), "w"), indent=1)
    elif a.cmd == "one":
        one(a.scene, a.rest[0], a.rest[1], a.frames, a.health, "kernels" if a.kernels else a.profile)
    elif a.cmd in ("profile", "kprofile"):
        run_all(a.scene, a.rest[0], a.configs.split(","), a.frames, False, profile=True if a.cmd == "profile" else "kernels")
    else:
        run_all(a.scene, a.rest[0], a.configs.split(","), a.frames if a.cmd == "run" or a.frames != 10 else 200, a.cmd == "health")
