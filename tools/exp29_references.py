"""exp29: write `experiments/exp29_cell_tissue_2/data/vertex_references.json` from the vertex twins on disk, and print it.

    PYTHONPATH=src:tools python tools/exp29_references.py            # all clips, seeds 0 1 2
    PYTHONPATH=src:tools python tools/exp29_references.py --bank      # also the comparison bank's per-metric mean, sd and seed-to-seed spreads
    PYTHONPATH=src:tools python tools/exp29_references.py --bands     # also print the gate bands the references imply
    PYTHONPATH=src:tools python tools/exp29_references.py --write-bands   # and write them into gates.yaml where a band is still TBD

The file is written BY THE RULERS (`tools/exp_measures/exp29.py::build_references`) from `tissue/exp29_v_<clip>_s<seed>`: per clip and
per quantity the seed values, their mean and spread, and `own_gap`, the largest gap of one vertex seed to the mean of the seeds in
the units the `*_gap` measures use. The bands of `gates.yaml` that read these gaps are set from `own_gap` (full marks within the
vertex twins' own spread, zero at BAND_ZERO_MULTIPLE x that) -- see `bands()`.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import numpy as np                                                                   # noqa: E402
from exp_measures import exp29                                                       # noqa: E402

BAND_ZERO_MULTIPLE = 3.0         # zero marks at three times the vertex twins' own spread (a stated model choice)
BAND_FULL_FLOOR = 0.05           # a vertex spread of 0 would put the full line at 0: never tighter than 5 % of the reference


def bands(refs: dict) -> dict:
    """{gate id: (full, zero, how)} for the gates that read a `*_gap`: the mean over the clips the gate reads of the vertex twins'
    own_gap, floored. Every number comes from the references file."""
    def own(group, keys, clips):
        v = []
        for c in clips:
            for k in keys:
                x = ((refs["clips"].get(c) or {}).get(group) or {}).get(k)
                if x and x.get("own_gap") is not None:
                    v.append(x["own_gap"])
        return float(np.mean(v)) if v else None
    spec = {
        "G-growth": ("growth", ["doubling_norm"], ["grow_divide"]),
        "G-chem": ("chem", ["spots_per_100", "high_fraction"], ["r021_12", "r005_06", "r013_05", "r016_01", "r017_00_ctrl", "r020_01", "r020_00_ctrl", "r021_06"]),
        "G-gate": ("gate", ["div_contrast"], ["r005_06"]),
        "G-shape": ("shape", ["protr", "corr_act_rad"], ["r013_05", "r016_01", "r020_00_ctrl", "r021_06"]),
        "G-death": ("death", ["low_activator_fraction"], ["tsd_max", "sc_antiphase"]),
    }
    out = {}
    for gate, (group, keys, clips) in spec.items():
        o = own(group, keys, clips)
        if o is None:
            continue
        full = max(o, BAND_FULL_FLOOR)
        out[gate] = (round(full, 4), round(BAND_ZERO_MULTIPLE * full, 4), f"{group}.{'+'.join(keys)} over {','.join(clips)}")
    return out


GATES = os.path.join(ROOT, "experiments", "exp29_cell_tissue_2", "gates.yaml")
TBD_FULL, TBD_ZERO = "TBD from", "TBD from"


def apply_bands(refs: dict, path: str = GATES) -> list:
    """Replace each `TBD` band of gates.yaml that the references imply by its number, and say where it came from. Returns the
    gate ids written. Points, claims and values are untouched; a band is written ONCE (a gate with numbers is left alone)."""
    import re
    b = bands(refs)
    c1 = ((refs["clips"].get("grow_divide") or {}).get("shape") or {}).get("corr_act_rad")
    if c1 and c1.get("values"):                    # G-control: the vertex clip-1 twin's own |correlation| (no chemistry), floored
        full = max(max(abs(v) for v in c1["values"]), BAND_FULL_FLOOR)
        b["G-control"] = (round(full, 4), None, "shape.corr_act_rad of the vertex clip-1 twin, |max| over seeds")
    text = open(path).read()
    written = []
    for gate, (full, zero, how) in b.items():
        m = re.search(rf"(  - id: {re.escape(gate)}\n(?:    .*\n)*?)(?=  - id: |caps:)", text)
        if not m:
            continue
        block = m.group(1)
        if "TBD" not in block:
            continue
        nb = re.sub(r"    full: .*TBD.*\n", f"    full: {full}\n", block)
        if zero is not None:
            nb = re.sub(r"    zero: .*TBD.*\n", f"    zero: {zero}\n", nb)
        src = (f"; band from data/vertex_references.json ({how}): full = the vertex twins' own gap between seeds"
               f"{'' if gate == 'G-control' else f', zero = {BAND_ZERO_MULTIPLE:g} x that'}, floored at {BAND_FULL_FLOOR}")
        nb = re.sub(r'(    source: ".*?)"\n', lambda mm: mm.group(1) + src + '"\n', nb, count=1)
        text = text.replace(block, nb)
        written.append(gate)
    open(path, "w").write(text)
    return written


def main():
    refs = exp29.build_references()
    print(f"wrote {exp29.REFS}")
    for clip, d in refs["clips"].items():
        print(f"\n{clip}: runs {d['runs']}  cells at end {d['n_cells_end']}")
        for group in ("growth", "chem", "gate", "shape", "regime", "death"):
            for k, x in (d.get(group) or {}).items():
                sp = "-" if x.get("spread") is None else f"{x['spread']:.3g}"
                og = "-" if x.get("own_gap") is None else f"{x['own_gap']:.3g}"
                print(f"   {group}.{k:24s} mean {x['mean']:.4g}  spread {sp}  own_gap {og}  values {[round(v, 4) for v in x['values']]}")
    if "--bank" in sys.argv:
        b = exp29.bank_references()
        print(f"bank written for {sorted(b['bank'])}; errors: {b.get('bank_errors')}")
    if "--write-bands" in sys.argv:
        print("gates.yaml bands written for:", apply_bands(refs))
    if "--bands" in sys.argv:
        print("\nbands implied (full, zero):")
        for g, (f, z, how) in bands(refs).items():
            print(f"   {g}: full {f}, zero {z}   [{how}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
