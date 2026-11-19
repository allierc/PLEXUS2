"""exp10 oocyte_posterior: the mRNAs and proteins of the posterior pole of the fly oocyte as MPM bodies in a periodic box, bound by the binding free energies of the cast.

The scenes are built, run, measured and drawn by tools/exp09_landscape.py (exp09's steps 24 to 27: mRNA as chains of subunits, binding sites, contact_finder[sites],
bond_binding[delta_g]); this file gives it exp10's record, names, units, kinds and scenes and then hands it the command line.

    the kinds       experiments/exp10_oocyte_posterior/cast.yaml: 6 mRNAs (oskar, nanos, pgc, germ cell-less, Cyclin B, Hsp83) and 10 proteins (Short Oskar, Vasa, Tudor,
                    Aubergine, Bruno, Staufen, Hrp48, Cup, Glorund, Smaug), and the 31 pairs that bind with their standard binding free energies (5 measured, 1 from
                    homologs, 25 assumed)
    an mRNA         a chain of 3 nm subunits (a block of 3 x 3 x 3 points), one per 50 nucleotides, held by static springs; a binding site is 2 consecutive subunits
    a protein       ONE elastic MPM body on its own grid block: bodies.yaml -- a lobe per folded domain of its AlphaFold model and one per disordered stretch, each of its
                    own volume (0.133 nm^3 per residue), 56 points (Hrp48) to 334 (Tudor)
    binding         bond_binding[delta_g_molecule]: a junction's unbinding rate is its binding rate x exp(Delta G / kT) of its two kinds; two sites of ONE strand never bind

Units: 1 nm per material point, so the length unit (5 point spacings) is 5 nm, half of exp09's; the time unit is 0.025 us, exp09's 0.2 us x (5 / 10)^2 (the same number
of length units is a quarter of the distance squared) / 2 (a body half the size diffuses twice as fast): 1,000 frames are 25 us.

    python tools/exp10_oocyte.py shapes                 every protein body's points as the scene seeds them: count, extent, in one piece or not
    python tools/exp10_oocyte.py still oo1 --run RUN    the first and the last stored row of a run, drawn: data/still_<key>_first.png, _last.png
    any other command (write, measure, after, look, render_part, ...) is exp09_landscape's own, under exp10's names

Watcher: experiments/exp10_oocyte_posterior at http://127.0.0.1:8844/watch.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools"), os.path.join(ROOT, "discovery_okuda")]
import numpy as np  # noqa: E402
import yaml  # noqa: E402

import exp09_landscape as E  # noqa: E402

RECORD = os.path.join(ROOT, "experiments", "exp10_oocyte_posterior")
E.RECORD, E.DATA, E.LOG = RECORD, os.path.join(RECORD, "data"), os.path.join(ROOT, "log", "experiments", "exp10_oo")
E.SCRIPT, E.JOB, E.LABEL = "tools/exp10_oocyte.py", "exp10_", "exp10"
E.NM, E.UM, E.TIME_US = 5.0, 0.005, 0.025              # nm and um per length unit, us per time unit (the docstring)
E.run_name = lambda key: f"exp10_oo_{key}"             # noqa: E731  graphs_data/tissue/exp10* without `_oo_` are the runs of an earlier experiment 10

CAST = yaml.safe_load(open(os.path.join(RECORD, "cast.yaml")))
BODIES = {b["key"]: b for b in yaml.safe_load(open(os.path.join(RECORD, "bodies.yaml")))["bodies"]}
MRNA = [k for k in CAST["kinds"] if k["kind"] == "mRNA"]
PROT = [k for k in CAST["kinds"] if k["kind"] == "protein"]
NT_PER_SUBUNIT = 50                                     # nucleotides per 3 nm subunit (cast.yaml's 6 nm per 100)
SUBUNITS = {k["key"]: max(2, int(round(k["nt"] / NT_PER_SUBUNIT))) for k in MRNA}
for _i, _k in enumerate(MRNA):
    E.SPECIES[_k["key"]] = dict(n=27, shape=("block", (3, 3, 3)), colour=E.RNA_COLOURS[_i],
                                what=f"a subunit of {_k['name']} mRNA, 3 nm = {NT_PER_SUBUNIT} nucleotides (a block of 3 x 3 x 3 points); a strand is {SUBUNITS[_k['key']]} of them")
for _i, _k in enumerate(PROT):
    _b = BODIES[_k["key"]]
    _r0 = float(_b["radii_nm"][0])
    E.SPECIES[_k["key"]] = dict(n=int(_b["points"]), colour=E.PROT_COLOURS[_i],
                                shape=("lobes_wide", [(tuple(float(v) / _r0 for v in c), float(r) / _r0) for c, r in zip(_b["centres_nm"], _b["radii_nm"])]),
                                what=f"{_k['name']}, {_b['residues']:,} residues: {sum(l['folded'] for l in _b['lobes'])} folded domains and "
                                     f"{sum(not l['folded'] for l in _b['lobes'])} disordered stretches as lobes, {_b['points']} points of 1 nm")
NAMES = [k["key"] for k in MRNA] + [k["key"] for k in PROT]


def delta_g():
    """(kind names, the 16 x 16 matrix of standard binding free energies, kcal/mol; None = the pair does not bind) from cast.yaml's pairs."""
    G = [[None] * len(NAMES) for _ in NAMES]
    for p in CAST["pairs"]:
        a, b = NAMES.index(p["a"]), NAMES.index(p["b"])
        G[a][b] = G[b][a] = float(p["dg_kcal_mol"])
    return NAMES, G


# a body may hold as many bonds as it has folded domains, 2 at least; a binding site of an mRNA holds one
VALENCE = {**{k["key"]: 1 for k in MRNA}, **{k["key"]: max(2, sum(l["folded"] for l in BODIES[k["key"]]["lobes"])) for k in PROT}}
BIND_NM3 = round((0.45 * E.NM) ** 3, 2)                 # the volume a bound pair is held in: the cube of the bonding range (0.45 length units = 2.25 nm), 11.39 nm^3
N_STRANDS, N_PROTEINS = 32, 300                         # of each mRNA kind, of each protein kind (at 36 strands 2 nanos strands find no place at this gap, at 40, 4)

E.ITERS["oo1"] = dict(
    parent="rna1_3dp", box=(51.0, 28.8, 28.8), n_grid=24, seed_batched=True, junction_buffer=65536, matrix_panel=True, panels3d="matrix",
    counts={k["key"]: N_PROTEINS for k in PROT},
    chains=[dict(kind=k, n=N_STRANDS, m=m, bend=0.2) for k, m in sorted(SUBUNITS.items(), key=lambda q: -q[1])],             # longest first: placed while the box is empty
    binding=dict(delta_g=delta_g(), delta_g_unit="kcal/mol", temperature_K=298.15, binding_volume_nm3=BIND_NM3, stochastic=True, molecule=True, valence=VALENCE,
                 on=1.0, stiffness=1.0e-3, strength=1.0e-2, rest=0.22, range=0.45, min_area=0.12),
    fast=dict(pp=True, strain=True, ccwarp=True, tokens="ccf+bath+bs+aj+sc+gas", capture=True),
    why="THE TWIN OF exp09's STEP 26 WITH THE REAL CAST (Cedric, 2026-11-19: \"a twin of exp09-26 with the new cast only the 3D panel + the matrices\"): a periodic box "
        f"255 x 144 x 144 nm, no walls, holding {N_STRANDS} strands of each of 6 mRNAs (oskar, nanos, pgc, germ cell-less, Cyclin B, Hsp83: chains of 14 to 57 subunits of 3 nm, "
        f"one per 50 nucleotides) and {N_PROTEINS} of each of 10 proteins (Short Oskar, Vasa, Tudor, Aubergine, Bruno, Staufen, Hrp48, Cup, Glorund, Smaug), each protein ONE "
        "elastic body shaped from its AlphaFold model at 1 nm per point (a lobe per folded domain and per disordered stretch). The 31 pairs of the cast bind by their "
        "standard binding free energies (5 measured, 1 from homologs, 25 assumed), a strand never with itself (NEW operator variant bond_binding[delta_g_molecule]); "
        "the substep block runs the Warp variants of exp09's speed work as one captured graph. Do the two granule types appear: the germ granule's proteins "
        "(Oskar, Vasa, Tudor, Aubergine) gathering nanos, pgc, germ cell-less and Cyclin B, and oskar apart with Bruno, Staufen and Hrp48")
E.ITERS["oo1s"] = dict(parent="oo1", fast=dict(pp=True, strain=True, ccwarp=True, capture=False),
                       why="oo1 with the operator set of exp09's step 26 (the Warp scatter, gather, strain and contact penalty; the bath, the springs and the grid update "
                           "in torch, the block not captured): the fallback if the captured block fails on a periodic box with 24-cell grids, and what its speed-up is measured against")

# oo1 WITH WALLS (Cedric, 2026-11-19: "in parallel run a first exp10 with boundary to run faster"): what exp09 validated together -- the walled box and the captured Warp
# block (step 25's speed work) -- so the contact search runs on the GPU too (a periodic box sends it to the CPU kd-tree: 2.1 s of oo1's 2.6 s frame)
E.ITERS["oo1w"] = dict(parent="oo1", periodic=False, margin=0.6, fast=dict(pp=True, strain=True, ccwarp=True, tokens="ccf+bath+wall+bs+aj+sc+gas", capture=True),
                       why="THE FIRST exp10 SCENE WITH WALLS: oo1's cast, counts, shapes and binding free energies (32 strands of each of 6 mRNAs, 300 of each of 10 "
                           "proteins, 11,672 bodies) in the same box 255 x 144 x 144 nm, held by soft walls instead of joined face to face. With walls every operator of "
                           "a frame runs on the GPU, the contact search included, as validated on exp09's step 25. Do the same bonds form as in the periodic box, and how "
                           "much faster does the frame go")


def spec_of(key):
    sp = E.build(key)
    if E.params(key).get("fast"):
        E._fast(sp, E.params(key)["fast"])
    return sp


def still(key, run=None):
    """The first and the last stored row of a run as the movie draws them (the whole box from above at 30 degrees, the matrix panel at its right)."""
    run = run or E.run_name(key)
    z = E._load(os.path.join(E.GD, run, "trajectory.npz"))
    rows = int(z["cell__occ"].shape[0])
    out = []
    E.FRAME_HOOK = E.matrix_hook(key, run=run)
    try:
        for tag, r in (("first", 0), ("last", rows - 1)):
            info = E.render(key, run=run, name=f"still_{key}_{tag}", max_frames=10 ** 6, fps=5, rows_sel=[r], view=E.VIEW_3D, frame_wh=E.FRAME, use_section=False, bar=False, fit_box=True)
            out.append(info["png"])
            if info.get("mp4") and os.path.exists(os.path.join(ROOT, info["mp4"])):
                os.remove(os.path.join(ROOT, info["mp4"]))
    finally:
        E.FRAME_HOOK = None
    print("stills:", out)
    return out


def shapes():
    """Every protein body as the scene seeds it: points, extent, and whether its points are one piece through lattice neighbours."""
    for k in PROT:
        sp = E.SPECIES[k["key"]]
        P = E.body_points(sp["shape"], sp["n"])
        S = {tuple(v) for v in np.rint(P / E.SPACING - (P / E.SPACING).min(0)).astype(int)}
        seen, stack = set(), [next(iter(S))]
        seen.add(stack[0])
        while stack:
            q = stack.pop()
            for d in np.ndindex(3, 3, 3):
                t = (q[0] + d[0] - 1, q[1] + d[1] - 1, q[2] + d[2] - 1)
                if t in S and t not in seen:
                    seen.add(t)
                    stack.append(t)
        ext = (np.ptp(P, 0) / E.SPACING + 1).astype(int)
        print(f"{k['name']:12s} {len(P):4d} points (asked {sp['n']}), {ext[0]} x {ext[1]} x {ext[2]} points, valence {VALENCE[k['key']]}, {'one piece' if len(seen) == len(S) else 'IN PIECES'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    arg = sys.argv[2:]
    opt = {arg[i][2:]: arg[i + 1] for i in range(len(arg) - 1) if arg[i].startswith("--")}
    keys = [a for i, a in enumerate(arg) if not a.startswith("--") and not (i and arg[i - 1].startswith("--"))]
    if cmd == "shapes":
        shapes()
    elif cmd == "still":
        still(keys[0], opt.get("run"))
    else:
        E.main()
