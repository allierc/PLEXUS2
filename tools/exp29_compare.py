"""exp29: compare an MPM run with the vertex twin of its clip on a bank of metrics, and flag what differs.

    PYTHONPATH=src:tools python tools/exp29_compare.py <group>/<mpm run> [<clip>] [--refs data/vertex_references_n200.json] [--quiet]

Prints one table -- family, metric, the MPM run's value, the vertex twins' mean +- seed sd, the normalised gap, FLAG when the gap
exceeds 3 -- then the four family scores (mean of min(gap, 6) over the family's metrics) and the flags, and writes the same as JSON
(default `experiments/exp29_cell_tissue_2/data/compare/<run>.json`). <clip> is read from the run's name (`exp29_m_<clip>_s<k>`) unless
given; the pairing of a test run with a clip is the caller's (exp29_t_die against tsd_max is a rough pairing, and says so).

The metrics and their definitions are `tools/exp_measures/exp29.py` ("THE COMPARISON BANK"): NUMBER (count curve, growth, births and
deaths), DISTRIBUTION (volume, contact degree, spacing, activator: Wasserstein-1 against the vertex twins), GEOMETRY (radius curve,
protrusion, sphericity, HOLES in the sheet) and ORGANISATION (contact graph, Moran's I, spots, activator-radius correlation, division
contrast, death placement, hole recovery). The reference is `data/vertex_references.json` (`tools/exp29_references.py --bank`).
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

from exp_measures import exp29 as M, open_run                                       # noqa: E402

OUT = os.path.join(ROOT, "experiments", "exp29_cell_tissue_2", "data", "compare")


def table(run: str, clip: str | None = None, quiet=False, write=True, refs_path=None) -> dict:
    T = open_run(run)
    r = M.compare(T, clip=clip, refs_path=refs_path)
    cmp, b = r["compare"], r["bank"]
    if not quiet:
        meta = b["meta"]
        print(f"\n{run}  vs  vertex twin of {r['clip']}   ({meta['body']} body, {meta['rows']} rows, {meta['frames']:.0f} frames, "
              f"cells {meta['n_start']:.0f} -> {meta['n_end']:.0f}, chemistry active: {meta['chem_active']})")
        print(f"  {'family':13s} {'metric':20s} {'MPM':>10s} {'vertex mean':>12s} {'sd':>9s} {'gap':>7s}")
        for row in cmp["rows"]:
            f = lambda v: "-" if v is None else f"{v:.4g}"
            print(f"  {row['family']:13s} {row['metric']:20s} {f(row['mpm']):>10s} {f(row['vertex_mean']):>12s} {f(row['vertex_sd']):>9s} "
                  f"{row['gap']:7.2f}{'  FLAG' if row['flag'] else ''}{'  ' + row['note'] if row.get('note') else ''}")
        print("  family scores (mean of min(gap, 6); 1 = within one vertex seed spread, 4 = zero marks):",
              {k: (None if v is None else round(v, 2)) for k, v in cmp["family"].items()}, " hole_gap", round(cmp["hole_gap"], 2))
        print("  FLAGS:", ", ".join(cmp["flags"]) or "none")
    if write:
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, run.replace("/", "__") + ".json")
        json.dump({"run": run, "clip": r["clip"], "compare": cmp, "meta": b["meta"], "scalars": b["scalars"]}, open(path, "w"), indent=1, default=float)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("clip", nargs="?")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--refs", default=None, help="another references file, e.g. data/vertex_references_n200.json (vertex twins of 200 cells)")
    a = ap.parse_args()
    table(a.run, a.clip, a.quiet, refs_path=a.refs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
