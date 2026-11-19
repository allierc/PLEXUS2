"""exp09 speed test (Cedric, 2026-11-17 16:20: "can you speed up these run use warp variant, torch compile give a few test over 10 frames check bite identical results").

For one scene, the same 10 frames run under several configurations, each in its own process (no compile or capture cache carried over), on one GPU:

    base        the spec as `exp09_landscape.build` writes it (torch scatter, gather and strain, the substep block neither captured nor compiled)
    base2       the same again: what two runs of one spec differ by (GPU atomics sum in any order unless PLEXUS_STRICT_DETERMINISM is set)
    pp          mpm_scatter[warp_pp] + mpm_gather[warp_pp]: Warp kernels on each body's own grid block
    pp_strain   pp + mpm_strain[warp]
    touched     pair_potential[cell_contact_touched_nodes]: the contact lattice stored only on the nodes a body touches
    capture     the substep block captured as one CUDA graph (`capture: true`)
    compile     the substep block under torch.compile (`compile: true`)
    fast        pp_strain + touched + capture

For each: the wall time of every frame (the median over frames 2 to 10, the first two carry the warm-up, the capture and the compile), the peak GPU memory, and the
state after frame 10 against `base`: the material points' and the bodies' positions and the junctions' bound share, bit for bit (torch.equal) and the largest
difference in nm (positions) or absolute (the share).

    python tools/exp09_speed.py run <key> [--configs base,base2,...] [--strict]    on the GPU, writes data/speed_<key>[_strict].json
    python tools/exp09_speed.py one <key> <config> <out.npz> [--strict]            one configuration (the child process)
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools"), os.path.join(ROOT, "discovery_okuda")]
import yaml  # noqa: E402

import exp09_landscape as E  # noqa: E402

FRAMES = 10
CONFIGS = ["base", "base2", "pp", "pp_strain", "touched", "capture", "compile", "fast", "ccwarp"]


def spec_of(key, cfg):
    """The 10-frame spec of `key` under configuration `cfg` (a dict, from the spec written once by `run`)."""
    sp = copy.deepcopy(yaml.safe_load(open(os.path.join(E.DATA, f"speed_spec_{key}.yaml"))))
    sp["general"].update(name=f"exp09_speed_{key}_{cfg}", n_frames=FRAMES, record_cap=2)
    sp["general"].pop("save_chunk_frames", None)
    sp["general"].pop("save_every_min", None)
    sp["general"].pop("warmup", None)
    sp["general"].pop("save_chunk_window", None)
    ops = sp["operators"]
    want = set()
    if cfg in ("pp", "pp_strain", "fast"):
        want |= {"pp"}
    if cfg in ("pp_strain", "fast"):
        want |= {"strain"}
    if cfg in ("touched", "fast"):
        want |= {"touched"}
    if cfg == "ccwarp":                                                  # pp_strain + pair_potential[cell_contact_warp_pp] (2026-11-17, for B)
        want |= {"pp", "strain", "ccwarp"}
    for o in ops:
        if "pp" in want and o["op"] in ("mpm_scatter", "mpm_gather"):
            o["implementation"] = "warp_pp"
        if "strain" in want and o["op"] == "mpm_strain":
            o["implementation"] = "warp"
        if "touched" in want and o["op"] == "pair_potential" and o.get("model") == "cell_contact":
            o.pop("model")
            o["implementation"] = "cell_contact_touched_nodes"
        if "ccwarp" in want and o["op"] == "pair_potential" and o.get("model") == "cell_contact":
            o.pop("model")
            o["implementation"] = "cell_contact_warp_pp"
    for st in sp["schedule"]:
        if isinstance(st, dict) and "substep_dt" in st:
            st["capture"] = cfg in ("capture", "fast")
            if cfg == "compile":
                st["compile"] = True
    return sp


def one(key, cfg, out, strict):
    """Run one configuration's 10 frames and save the frame times, the peak memory and the state after the last frame to `out`."""
    if strict:
        os.environ["PLEXUS_STRICT_DETERMINISM"] = "1"
    import torch
    import plexus.operators  # noqa: F401  the operator library and its entities register themselves (as Plexus_Main imports it)
    from plexus import schema
    from plexus.engine import run
    sp = spec_of(key, cfg)
    path = os.path.join("/tmp", f"exp09_speed_{key}_{cfg}.yaml")
    yaml.safe_dump(sp, open(path, "w"), sort_keys=False, width=1000)
    sim = schema.load(path)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    H, res = run(sim, out_path=None, device="cuda:0", progress=False)
    torch.cuda.synchronize()
    wall = time.time() - t0
    st = {"frame_ms": np.asarray(res["frame_ms"], np.float64), "wall_s": wall, "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
    pt = H.level("pt")
    st["pt_pos"] = pt.get("pos").detach().cpu().numpy()
    st["pt_occ"] = pt.active.detach().cpu().numpy()
    st["cell_pos"] = H.level("cell").get("pos").detach().cpu().numpy()
    if "junction" in H.levels:
        J = H.level("junction")
        st["j_rho"] = J.get("rho").detach().cpu().numpy()
        st["j_occ"] = J.active.detach().cpu().numpy()
    np.savez(out, **st)


def run_all(key, configs, strict):
    sp = E.build(key)                                                    # the scene's spec and seed, written once
    yaml.safe_dump(sp, open(os.path.join(E.DATA, f"speed_spec_{key}.yaml"), "w"), sort_keys=False, width=1000)
    tag = f"{key}{'_strict' if strict else ''}"
    tmp = os.path.join("/tmp", f"exp09_speed_{tag}")
    os.makedirs(tmp, exist_ok=True)
    res = {}
    for cfg in configs:
        out = os.path.join(tmp, f"{cfg}.npz")
        t0 = time.time()
        r = subprocess.run([sys.executable, "-u", __file__, "one", key, cfg, out] + (["--strict"] if strict else []), capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            tail = (r.stderr or r.stdout).strip().splitlines()[-6:]
            res[cfg] = {"error": " | ".join(tail)[:900]}
            print(f"{cfg}: FAILED ({time.time() - t0:.0f} s): {res[cfg]['error']}", flush=True)
            continue
        z = np.load(out)
        ms = z["frame_ms"]
        res[cfg] = {"ms_per_frame": float(np.median(ms[2:])) if len(ms) > 3 else float(np.median(ms)), "first_two_ms": [float(v) for v in ms[:2]],
                    "peak_gb": float(z["peak_gb"]), "wall_s": float(z["wall_s"])}
        print(f"{cfg}: {res[cfg]['ms_per_frame']:.0f} ms per frame (frames 2-{FRAMES}), first two {res[cfg]['first_two_ms'][0]:.0f} / {res[cfg]['first_two_ms'][1]:.0f} ms, "
              f"peak {res[cfg]['peak_gb']:.1f} GB", flush=True)
    if "base" in res and "error" not in res["base"]:
        b = np.load(os.path.join(tmp, "base.npz"))
        for cfg in configs:
            if cfg == "base" or "error" in res.get(cfg, {"error": 1}):
                continue
            z = np.load(os.path.join(tmp, f"{cfg}.npz"))
            cmp = {}
            for k, unit in (("pt_pos", E.NM), ("cell_pos", E.NM), ("j_rho", 1.0)):
                if k not in b.files or k not in z.files:
                    continue
                m = b["pt_occ"].astype(bool) if k == "pt_pos" else (b["j_occ"].astype(bool) if k == "j_rho" else None)
                a1, a2 = (b[k], z[k]) if m is None else (b[k][m], z[k][m])
                same = bool(a1.shape == a2.shape and np.array_equal(a1, a2))
                cmp[k] = {"bit_identical": same, "max_abs": float(np.max(np.abs(a1.astype(np.float64) - a2.astype(np.float64)))) * unit if a1.shape == a2.shape else None}
            res[cfg]["vs_base"] = cmp
            print(f"{cfg} vs base: " + "; ".join(f"{k} {'BIT-IDENTICAL' if v['bit_identical'] else 'differs'} (max {v['max_abs']:.3g}{' nm' if k != 'j_rho' else ''})"
                                                 for k, v in cmp.items() if v["max_abs"] is not None), flush=True)
    out = os.path.join(E.DATA, f"speed_{tag}.json")
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "one"])
    ap.add_argument("key")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    if a.cmd == "one":
        one(a.key, a.rest[0], a.rest[1], a.strict)
    else:
        run_all(a.key, a.configs.split(","), a.strict)
