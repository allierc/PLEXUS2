"""exp10: a body for each protein of the cast from its AlphaFold model -- one lobe per folded domain and one per disordered stretch, each of its own volume
(0.133 nm^3 per residue), on a lattice of 1 nm points -- written to bodies.yaml, and a montage of the ten at one scale. Usage: oo_shapes.py <record folder>"""
import json, math, os, sys, urllib.request
import numpy as np, yaml
REC = sys.argv[1]; ST = os.path.join(REC, "structures")
SP = 1.0                                                              # nm per lattice point
models = json.load(open(os.path.join(ST, "models.json")))
if models["Bru"]["accession"] != "Q960Z4":                            # the cast's Bruno is the 604-residue isoform, not the 810-residue one fetched first
    acc = "Q960Z4"
    get = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "research-client"}), timeout=40).read()
    meta = json.loads(get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"))[0]
    open(os.path.join(ST, f"Bru_{acc}.pdb"), "wb").write(get(meta["pdbUrl"])); open(os.path.join(ST, f"Bru_{acc}_pae.json"), "wb").write(get(meta["paeDocUrl"]))
    models["Bru"] = dict(models["Bru"], accession=acc, name=meta.get("uniprotDescription"), model=f"Bru_{acc}.pdb", url=meta["pdbUrl"])
    json.dump(models, open(os.path.join(ST, "models.json"), "w"), indent=1)
FIRST = {"Osk": 139}                                                  # Short Oskar: the model's residues 139 on
NAMES = dict(Osk="Short Oskar", Vas="Vasa", Tud="Tudor", Aub="Aubergine", Bru="Bruno", Stau="Staufen", Hrp48="Hrp48", Cup="Cup", Glo="Glorund", Smg="Smaug")

def domains(key):
    """(n residues, [(start, end, folded)] in sequence order, 0-based inclusive): the cut of oo_domains.py, then every stretch outside a domain."""
    m = models[key]
    L = [l for l in open(os.path.join(ST, m["model"])).read().splitlines() if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in L]) / 10.0
    b = np.array([float(l[60:66]) for l in L]); n = len(b)
    idx = np.flatnonzero(b >= 70); segs = []
    s = p = idx[0]
    for i in idx[1:]:
        if i - p > 9:
            segs.append((s, p)); s = i
        p = i
    segs.append((s, p)); segs = [(s, e) for s, e in segs if e - s + 1 >= 30]
    j = json.load(open(os.path.join(ST, m["model"].replace(".pdb", "_pae.json")))); j = j[0] if isinstance(j, list) else j
    pae = np.asarray(j["predicted_aligned_error"], np.float32)
    par = list(range(len(segs)))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for i in range(len(segs)):
        for k in range(i + 1, len(segs)):
            A = np.arange(segs[i][0], segs[i][1] + 1); B = np.arange(segs[k][0], segs[k][1] + 1)
            if (np.linalg.norm(xyz[A][:, None] - xyz[B][None], axis=2) < 0.8).sum() >= 10 and 0.5 * (pae[np.ix_(A, B)].mean() + pae[np.ix_(B, A)].mean()) <= 10.0:
                par[find(i)] = find(k)
    g = {}
    for i, sg in enumerate(segs):
        g.setdefault(find(i), []).append(sg)
    doms = sorted(((min(s for s, _ in v), sum(e - s + 1 for s, e in v)) for v in g.values()))          # (first residue, residues)
    lab = np.zeros(n, int)                                            # 0 disordered, k = domain k
    for k, v in enumerate(sorted(g.values(), key=lambda v: v[0][0]), 1):
        for s, e in v:
            lab[s:e + 1] = k
    return n, lab

def lobes_of(key):
    """The body's lobes in sequence order: (residues, folded). A domain is one lobe (at its first residue); a disordered stretch of 40 residues or more is one lobe;
    a shorter one is left out of the shape (its residues still count in the body's points)."""
    n, lab = domains(key)
    a = FIRST.get(key, 1) - 1
    lab = lab[a:]
    out, seen, i = [], set(), 0
    while i < len(lab):
        k = lab[i]; j = i
        while j < len(lab) and lab[j] == k:
            j += 1
        if k == 0:
            if j - i >= 40:
                out.append((j - i, False))
        elif k not in seen:
            seen.add(k); out.append((int((lab == k).sum()), True))
        i = j
    return len(lab), out

def layout(lobes, seed):
    """Lobe centres, nm: a string in sequence order, each lobe overlapping the one before (centres 0.8 x the sum of their radii apart), turning at random by at
    most 100 degrees, never within the sum of the radii of an earlier lobe that is not its neighbour."""
    rng = np.random.default_rng(seed)
    r = np.array([(3 * 0.133 * n_ / (4 * math.pi)) ** (1 / 3) for n_, _ in lobes])
    c = [np.zeros(3)]; d = np.array([1.0, 0.0, 0.0])
    for i in range(1, len(lobes)):
        for _ in range(4000):
            v = rng.normal(size=3); v /= np.linalg.norm(v)
            if v @ d < math.cos(math.radians(100)):
                continue
            x = c[-1] + 0.8 * (r[i - 1] + r[i]) * v
            if all(np.linalg.norm(x - c[k]) >= 1.0 * (r[i] + r[k]) for k in range(i - 1)):
                break
        c.append(x); d = v
    return np.array(c), r

def body(key, seed):
    nres, lobes = lobes_of(key)
    n = int(round(0.133 * nres / SP ** 3))
    cen, rad = layout(lobes, seed)
    lo, hi = (cen - rad[:, None]).min(0) - 2, (cen + rad[:, None]).max(0) + 2
    ax = [np.arange(math.floor(lo[k]), math.ceil(hi[k]) + 1) * SP for k in range(3)]
    lat = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    sc = np.linalg.norm(lat[:, None] - cen[None], axis=2) / rad[None]
    best = sc.min(1); pick = np.argsort(best, kind="stable")[:n]
    P = lat[pick]; which = sc[pick].argmin(1)
    # connected through lattice neighbours (faces, edges, corners)?
    S = {tuple(np.rint(p).astype(int)) for p in P}; start = next(iter(S)); seen = {start}; stack = [start]
    while stack:
        q = stack.pop()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    t = (q[0] + dx, q[1] + dy, q[2] + dz)
                    if t in S and t not in seen:
                        seen.add(t); stack.append(t)
    return dict(key=key, name=NAMES[key], residues=nres, points=n, lobes=[dict(residues=int(a), folded=bool(f), points=int((which == i).sum())) for i, (a, f) in enumerate(lobes)],
                centres_nm=np.round(cen, 2).tolist(), radii_nm=np.round(rad, 2).tolist(), one_piece=len(seen) == len(S)), P - P.mean(0), np.array([lobes[i][1] for i in which])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1a1a19", "#6b6a63"
keys = list(NAMES)
B = [body(k, seed=7 + i) for i, k in enumerate(keys)]
def view(P):
    w, V = np.linalg.eigh(np.cov(P.T))                                # seen along the body's thinnest axis, its longest axis across
    return P @ V[:, ::-1]
QS = [view(P) for _, P, _ in B]
hx = max(np.abs(Q[:, 0]).max() for Q in QS) + 1.5
hy = max(np.abs(Q[:, 1]).max() for Q in QS) + 1.5
fig, axs = plt.subplots(2, 5, figsize=(17, 17 * 2 * (hy / hx) / 5 + 3.4), facecolor="white")
for ax, (meta, P, folded), Q in zip(axs.ravel(), B, QS):
    order = np.argsort(Q[:, 2])
    z = (Q[:, 2] - Q[:, 2].min()) / max(float(np.ptp(Q[:, 2])), 1e-9)
    patches, cols = [], []
    for i in order:
        base = np.array(matplotlib.colors.to_rgb(BLUE if folded[i] else ORANGE))
        patches.append(Circle((Q[i, 0], Q[i, 1]), 0.62 * SP)); cols.append(tuple(base * (0.55 + 0.45 * z[i])))      # farther points darker
    ax.add_collection(PatchCollection(patches, facecolors=cols, edgecolors="white", linewidths=0.35))
    ax.set_xlim(-hx, hx); ax.set_ylim(-hy, hy); ax.set_aspect("equal"); ax.axis("off")
    nd = sum(l["folded"] for l in meta["lobes"]); dis = 1 - sum(l["residues"] for l in meta["lobes"] if l["folded"]) / meta["residues"]
    ax.text(0.5, 1.21, meta["name"], transform=ax.transAxes, ha="center", va="bottom", fontsize=17, color=INK)
    ax.text(0.5, 1.0, f"{meta['residues']:,} residues, {meta['points']} points\n{nd} folded domain{'s' if nd != 1 else ''}, {dis:.0%} disordered", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11.5, color=MUTED, linespacing=1.25)
half = hx
ax0 = axs[1, 0]
ax0.plot([-hx + 1, -hx + 6], [-hy + 0.6] * 2, color=INK, lw=2.5, solid_capstyle="butt")
ax0.text(-hx + 3.5, -hy + 1.1, "5 nm", ha="center", va="bottom", fontsize=12, color=INK)
fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=11, color=BLUE, label="folded domain (AlphaFold confidence 70 or more)"),
                    plt.Line2D([], [], marker="o", ls="", ms=11, color=ORANGE, label="disordered stretch, drawn at its compact volume")],
           loc="lower center", ncol=2, frameon=False, fontsize=13, labelcolor=INK)
fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.08, wspace=0.02, hspace=0.42)
out = os.path.join(REC, "shapes_montage.png")
fig.savefig(out, dpi=130, facecolor="white")
yaml.safe_dump(dict(point_spacing_nm=SP, volume_per_residue_nm3=0.133, bodies=[m for m, _, _ in B]), open(os.path.join(REC, "bodies.yaml"), "w"), sort_keys=False, width=1000)
for m, P, f in B:
    print(f"{m['name']:12s} {m['residues']:5d} residues {m['points']:4d} points, lobes " + " ".join(f"{'F' if l['folded'] else 'd'}{l['points']}" for l in m["lobes"])
          + f" | extent {np.ptp(P, 0).max() + SP:.0f} nm | {'one piece' if m['one_piece'] else 'IN PIECES'}")
print("wrote", out)
