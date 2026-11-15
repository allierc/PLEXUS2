"""exp29 cell-division bench WITH NUCLEI (Cedric, 2026-11-01): the nine isolated cells of the first steps (config/tissue/mpm_grid3x3_cycles.yaml, each grows and divides twice) with 1000 material points
per cell, 400 of them the cell's NUCLEUS (cell_nucleus[innermost], the innermost 0.4 of the cell's points, stiffer than the cytoplasm); the nucleus grows with the cell (the share is of the points the cell
has) and each division hands every daughter its own nucleus at its own centre. ONE parameter is swept per movie, one value per cell: the nucleus's stiffness over the cytoplasm's, the nucleus's share of the
cell's points, the cell's Young's modulus, the cell's Maxwell relaxation time. Cells are drawn translucent, nuclei blue.

  python tools/exp29_nucleus.py list
  python tools/exp29_nucleus.py write [--only stiffness fraction ...]       # -> config/tissue/mpm_nucleus_<key>.yaml
  python tools/exp29_nucleus.py post --only stiffness [--replace N]         # render the finished run into the exp29 record as a new step
"""
import argparse, os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "config/tissue/mpm_grid3x3_cycles.yaml"
POINTS = 1000                          # material points per cell at the start; a cube number (10^3), so the seed lays them on a lattice of spacing 0.1 inside the cell's 1.0-unit cube
FRACTION = 0.4                         # the nucleus's share of its cell's points: 400 of 1000 at the start
RATIO = 4.0                            # the nucleus's Young's modulus over the cytoplasm's when it is not the swept one (exp21's ratio; Guilak et al. 2000 measure a nucleus several times stiffer)
CELL_E, CELL_TAU = 20.0, 2.0           # the cell's Young's modulus and Maxwell relaxation time of the first steps
CELL_RGB, NUCLEUS_RGB, CELL_OPACITY = "#e63946", "#3b6fd4", 0.25
ISO = 0.3                              # the contour level of the cell surfaces (plotting.surface_iso_frac), the level Cedric chose for the red/blue movies (2026-11-01) and asked of this one

# one entry per movie: what is swept, nine values (one per cell, in the seed's layout order), how it reaches the run
#   op:stiffness_block / op:fraction_block  the value is read by cell_nucleus from the cell block `lab` (the label printed over the cell)
#   type:youngs / type:tau                   nine cell types, one per cell, each with its own value of that material parameter (a daughter keeps its mother's type)
SWEEPS = {
    "stiffness": dict(what="nucleus stiffness over the cytoplasm's (cell Young's modulus 20)", values=[1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0], how="op:stiffness_block",
                      fixed=f"nucleus {int(FRACTION * POINTS)} points", sub="the nucleus's Young's modulus over the cytoplasm's (the cell's is 20); 20 x is the largest ratio the outer-cell sweep already ran stably"),
    "fraction":  dict(what="nucleus share of the cell's points (0.4 = 400 of 1000)", values=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], how="op:fraction_block",
                      fixed=f"nucleus {RATIO:g}x stiffer", sub="the nucleus's share of its cell's points (the nucleus is the innermost share, 4 x stiffer than the cytoplasm)"),
    "youngs":    dict(what="cell Young's modulus (the nucleus is 4x stiffer)", values=[5.0, 10.0, 20.0, 40.0, 60.0, 100.0, 150.0, 250.0, 400.0], how="type:youngs",
                      fixed=f"nucleus {int(FRACTION * POINTS)} points, {RATIO:g}x stiffer", sub="the cell's Young's modulus (cytoplasm; the nucleus is 4 x that, 400 of the 1000 points); 400 is the outer-cell sweep's largest"),
    "tau":       dict(what="cell Maxwell relaxation time (min)", values=[0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 1000.0], how="type:tau",
                      fixed=f"nucleus {int(FRACTION * POINTS)} points, {RATIO:g}x stiffer", sub="the cell's Maxwell relaxation time, how fast its material forgets a deformation (the nucleus has the same, 4 x stiffer)"),
}


def name_of(key):
    return f"mpm_nucleus_{key}"


def build(key, points=POINTS):
    sw = SWEEPS[key]; vals = sw["values"]
    spec = yaml.safe_load(open(os.path.join(ROOT, BASE)))
    m = round(points ** (1.0 / 3.0))
    if m ** 3 != points:
        raise ValueError("points must be a cube number: the seed lays a cell's points on a cubic lattice")
    g = spec["general"]
    g.update(name=name_of(key), headline=f"9 cells with nuclei, growth and division over two cell cycles, sweep of the {sw['what']}",
             title=(f"Nine MPM cells 4 length units (40 um) apart, every cell {points} material points of which {int(FRACTION * points)} are its nucleus (the innermost {FRACTION:g} of its points, cell_nucleus[innermost]); "
                    f"each cell grows (mass doubles every 100 frames = 100 min) and divides twice; the nucleus grows with the cell and every division gives each daughter its own nucleus at its own centre. "
                    f"SWEPT, one value per cell printed over it: {sw['sub']}, {vals[0]:g} to {vals[-1]:g}. Cells translucent, nuclei blue"))
    spec["sets"]["cell"]["state"]["lab"] = {"width": 1}
    spec["sets"]["pt"].update(per_parent=points, grow_reserve=1000)
    spec["sets"]["pt"]["state"]["nuc"] = {"width": 1}
    nuc = {"op": "cell_nucleus", "model": "innermost", "at": "pt", "fraction": FRACTION, "stiffness": RATIO, "label": "nuc"}
    if sw["how"].startswith("op:"):
        opt = sw["how"][3:]
        nuc[opt] = "lab"
        nuc.pop(opt.replace("_block", ""))                                   # the block replaces the scalar it names
    spec["operators"].insert(0, nuc)
    spec["schedule"].insert(0, "cell_nucleus")
    seed = spec["seed"][0]
    seed.update(point_spacing=1.0 / m, live_per_cell=points, cell_values={"block": "lab", "values": vals})
    pl = spec["plotting"]
    names = ["epi"]
    if sw["how"].startswith("type:"):                                        # nine types, one per cell, in the order the seed lays the cells in (type_layout: ordered)
        par = sw["how"][5:]
        names = [f"epi_{a}" for a in range(len(vals))]
        base = {"count": 1, "shape": "ball", "material": "viscoelastic", "youngs": CELL_E, "tau": CELL_TAU, "density": 1.0}
        spec["sets"]["cell"]["types"] = {nm: {**base, par: float(vals[a])} for a, nm in enumerate(names)}
        for o in spec["operators"]:
            if o["op"] == "pair_potential":
                o["contact_surface_tension"] = [[0.0] * len(names) for _ in names]
    pl.update(cell_colours="type", colors={nm: CELL_RGB for nm in names}, type_opacity={nm: CELL_OPACITY for nm in names}, point_volume=round(1.0 / points, 6),
              cell_labels={"block": "lab", "value_only": True, "font": 20})
    pl["contour_bodies"].update(nucleus="nuc", nucleus_color=NUCLEUS_RGB)
    pl["surface_iso_frac"] = ISO
    return spec


def post(key, replace=None):
    """Render graphs_data/tissue/mpm_nucleus_<key> into the exp29 record as a new step (exp29_studio) and add its top-left text (what is swept, the range, what is drawn) to data/overlay_texts.yaml under the step's number."""
    import re, subprocess, sys
    sw = SWEEPS[key]; vals = sw["values"]; name = name_of(key)
    rec = os.path.join(ROOT, "experiments", "exp29_cell_tissue_2")
    env = {**os.environ, "PYTHONPATH": f"{ROOT}/src:{ROOT}/tools", "MPLBACKEND": "Agg", "PLEXUS_BUILDER": rec}
    spec = yaml.safe_load(open(os.path.join(ROOT, "config", "tissue", f"{name}.yaml")))
    why = f"CELL DIVISION WITH NUCLEI, SWEEP OF THE {sw['what'].upper()} (Cedric, 2026-11-01): " + spec["general"]["title"]
    sp_path = os.path.join(ROOT, "graphs_data", "tissue", name, "spec.yaml")     # the run's saved spec is what the studio reads
    saved = yaml.safe_load(open(sp_path))
    if saved["plotting"].get("surface_iso_frac") != ISO:
        saved["plotting"]["surface_iso_frac"] = ISO; yaml.safe_dump(saved, open(sp_path, "w"), sort_keys=False, width=1000)
    cmd = [sys.executable, os.path.join(ROOT, "tools", "exp29_studio.py"), f"tissue/{name}", "--why", why, "--no-activator"] + (["--replace", str(replace)] if replace else [])
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    m = re.findall(r"'index': (\d+)", r.stdout + r.stderr)
    if not m:
        print("render failed:", (r.stdout + r.stderr)[-600:]); return 1
    n = int(m[-1]); k = f"{n:04d}"
    tpath = os.path.join(rec, "data", "overlay_texts.yaml")
    t = yaml.safe_load(open(tpath))
    t["steps"][k] = {"lines": [sw["what"], f"{vals[0]:g} to {vals[-1]:g}, one value per cell (labels)", f"{POINTS} points per cell; {sw['fixed']} (blue); both grow and divide"]}
    yaml.safe_dump(t, open(tpath, "w"), sort_keys=False, width=1000)
    r2 = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "exp29_text_overlay.py"), str(n), "--from-current"], capture_output=True, text=True, env=env, cwd=ROOT)
    print(f"{key}: step {k} rendered; overlay rc {r2.returncode} {(r2.stdout + r2.stderr)[-200:].strip()}")
    return r2.returncode


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["list", "write", "post"]); ap.add_argument("--only", nargs="*"); ap.add_argument("--replace", type=int, default=None); a = ap.parse_args()
    keys = a.only or list(SWEEPS)
    if a.cmd == "list":
        for k, v in SWEEPS.items():
            print(f"{k:10s} {v['what']:62s} {v['values']}")
    elif a.cmd == "post":
        raise SystemExit(max(post(k, a.replace) for k in keys))
    else:
        for k in keys:
            out = os.path.join(ROOT, "config", "tissue", f"{name_of(k)}.yaml")
            yaml.safe_dump(build(k), open(out, "w"), sort_keys=False, width=1000); print("wrote", out)
