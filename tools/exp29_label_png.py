"""The LAST FRAME of a run as a PNG with a label on every family: its id (the slot of its cell at frame 0, so the array order the spec's `cell_values` use) and the value of a per-cell block.

    PYTHONPATH=src:tools python tools/exp29_label_png.py tissue/mpm_grid3x3_width_sweep --block width --out /tmp/width_sweep_labels.png

The frame is drawn by tools/exp29_studio.py exactly as the record step is (the spec's own cell colours, no activator inset), with one white, unboxed, non-bold text label per family added at the centroid of the family's
points: `id 8  width 1`. A cell belongs to the family whose frame-0 centre is nearest to it in the x-y plane; the value is the block's value at frame 0 (a daughter keeps it). Nothing is archived.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
import exp29_studio as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--block", required=True, help="the per-cell block whose value goes on the label (e.g. width, overlap)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--font", type=int, default=26)
    ap.add_argument("--value-only", action="store_true", help="label each family with the block's value only (no id, no block name)")
    a = ap.parse_args()
    S.CELL_LABELS.update(block=a.block, value_only=a.value_only, font=a.font)      # the studio draws the labels itself (exp29_studio.draw_family_labels)
    info = S.render(a.run, "/tmp/_label_tmp.mp4", a.out, seconds=1.0, max_frames=1, no_activator=True)
    print("wrote", a.out, "| rows", info["rows"], "drawn", info["drawn"])


if __name__ == "__main__":
    main()
