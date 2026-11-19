"""Cut each AlphaFold model of exp10's cast into its folded domains and its disordered rest, and write shapes.yaml.

A residue is folded when its confidence pLDDT (the B-factor column) is >= 70; runs of folded residues (gaps of up to 8 bridged, 30 residues at least) are segments; two
segments are ONE domain when the model packs them against each other AND is confident of it: at least 10 pairs of C-alpha atoms within 0.8 nm, and a mean predicted
aligned error between them of at most 10 angstrom (a model places two domains joined by a disordered linker arbitrarily, so contact alone is not enough).
Per domain: residues, radius of gyration, and the diameter of a compact globule of that mass (0.133 nm^3 per residue). Usage: oo_domains.py <structures folder>"""
import json, math, os, sys
import numpy as np
D = sys.argv[1]
models = json.load(open(os.path.join(D, "models.json")))
def globule_nm(n):
    return (6 * 0.133 * n / math.pi) ** (1 / 3)
out = {}
for key, m in models.items():
    L = [l for l in open(os.path.join(D, m["model"])).read().splitlines() if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in L]) / 10.0          # nm
    b = np.array([float(l[60:66]) for l in L]); n = len(b)
    fold = b >= 70
    idx = np.flatnonzero(fold); segs = []
    if len(idx):
        s = p = idx[0]
        for i in idx[1:]:
            if i - p > 9:                                               # a gap of more than 8 residues ends the segment
                segs.append((s, p)); s = i
            p = i
        segs.append((s, p))
    segs = [(s, e) for s, e in segs if e - s + 1 >= 30]
    pae = None
    pf = os.path.join(D, m["model"].replace(".pdb", "_pae.json"))
    if os.path.exists(pf):
        j = json.load(open(pf)); j = j[0] if isinstance(j, list) else j
        pae = np.asarray(j["predicted_aligned_error"], np.float32)
    par = list(range(len(segs)))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for i in range(len(segs)):
        for k in range(i + 1, len(segs)):
            A = np.arange(segs[i][0], segs[i][1] + 1); B = np.arange(segs[k][0], segs[k][1] + 1)
            d = np.linalg.norm(xyz[A][:, None] - xyz[B][None], axis=2)
            ok = (d < 0.8).sum() >= 10
            if ok and pae is not None:
                ok = 0.5 * (pae[np.ix_(A, B)].mean() + pae[np.ix_(B, A)].mean()) <= 10.0
            if ok:
                par[find(i)] = find(k)
    doms = {}
    for i, sg in enumerate(segs):
        doms.setdefault(find(i), []).append(sg)
    rows = []
    for g in sorted(doms.values(), key=lambda g: g[0][0]):
        R = np.concatenate([np.arange(s, e + 1) for s, e in g]); c = xyz[R]
        rows.append(dict(residues=int(len(R)), ranges=[f"{s + 1}-{e + 1}" for s, e in g], rg_nm=round(float(np.sqrt(((c - c.mean(0)) ** 2).sum(1).mean())), 2),
                         globule_nm=round(globule_nm(len(R)), 1)))
    infold = sum(r["residues"] for r in rows)
    out[key] = dict(accession=m["accession"], residues=n, folded_residues=infold, disordered_residues=n - infold, domains=rows)
    print(f"{key} ({m['accession']}, {n} residues): {len(rows)} domains, {infold} residues folded, {n - infold} disordered ({(n - infold) / n:.0%})")
    for r in rows:
        print(f"     {r['residues']:4d} residues  {', '.join(r['ranges']):42s} radius of gyration {r['rg_nm']:.2f} nm, compact globule {r['globule_nm']} nm")
import yaml
yaml.safe_dump(out, open(os.path.join(D, "..", "shapes.yaml"), "w"), sort_keys=False, width=1000)
