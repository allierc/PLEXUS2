"""Write exp10's cast (16 kinds with their real sizes) and its pair table of binding free energies to cast.yaml, and print the two markdown tables.

Delta G0 = R T ln(K_d / 1 M) at 25 C (R T = 0.5925 kcal/mol), the standard binding free energy of a pair at 1 M; in kT it is ln(K_d / 1 M)."""
import math, sys, yaml
RT = 1.987204e-3 * 298.15                                             # kcal/mol at 25 C
def dg(kd_M):
    return round(RT * math.log(kd_M), 1)
def diam(aa):                                                         # nm: a compact globule of `aa` residues, 0.133 nm^3 per residue (1.21 nm^3 per kDa, 110 Da per residue)
    return round((6 * 0.133 * aa / math.pi) ** (1 / 3), 1)
MRNA = [  # key, name, nt taken, nt range over the RefSeq transcripts, where
    ("osk", "oskar", 2869, "2,869", "its own granules, not the germ granules"),
    ("nos", "nanos", 2292, "2,292 to 2,349", "germ granule, toward the centre"),
    ("pgc", "polar granule component", 686, "643 to 1,211", "germ granule, toward the periphery"),
    ("gcl", "germ cell-less", 2464, "2,464 to 4,238", "germ granule, toward the periphery"),
    ("CycB", "Cyclin B", 2548, "2,548 to 2,633", "germ granule, toward the centre"),
    ("Hsp83", "Hsp83", 2713, "2,713 to 2,960", "germ plasm"),
]
PROT = [  # key, name, aa taken, aa range, what
    ("Osk", "Short Oskar", 468, "468 (Long Oskar 606)", "nucleates the germ granules; LOTUS domain (dimer, binds Vasa) + OSK domain (binds RNA)"),
    ("Vas", "Vasa", 661, "661", "DEAD-box RNA helicase of the germ granule"),
    ("Tud", "Tudor", 2515, "2,512 to 2,515", "11 Tudor domains, each a site for a methylated arginine of Aubergine"),
    ("Aub", "Aubergine", 866, "795 to 866", "Piwi protein; its piRNAs hold mRNAs by partial base pairing"),
    ("Bru", "Bruno", 604, "573 to 810", "represses oskar; scaffold of oskar granules"),
    ("Stau", "Staufen", 1026, "914 to 1,026", "five double-stranded-RNA-binding domains; travels with oskar"),
    ("Hrp48", "Hrp48", 421, "421", "scaffold of oskar granules with Bruno"),
    ("Cup", "Cup", 1117, "1,117", "binds Bruno and the cap-binding factor; blocks oskar translation"),
    ("Glo", "Glorund", 586, "586", "keeps unlocalised nanos silent in the oocyte"),
    ("Smg", "Smaug", 999, "999", "keeps unlocalised nanos silent in the embryo; Oskar takes it off nanos"),
]
W = -5.0                                                              # assumed weak pair, kcal/mol: K_d 0.2 mM, weaker than every measured pair
M = lambda a, b, kd, what, src: dict(a=a, b=b, kd_M=kd, dg_kcal_mol=dg(kd), status="measured", what=what, source=src)
PAIRS = [
    M("Osk", "Vas", 1.0e-5, "LOTUS domain with Vasa's C-terminal domain, calorimetry at 25 C", "Jeske 2017, 10.1101/gad.297051.117"),
    M("Tud", "Aub", 4.8e-5, "ONE Tudor domain (the 11th) with ONE methylated arginine of Aubergine, calorimetry; 6 to 71 uM over three arginines", "Liu 2010, 10.1101/gad.1956010"),
    M("Bru", "osk", 1.24e-7, "Bruno's extended third RNA-recognition motif on a Bruno response element, half-saturation, Hill coefficient 1.4", "Lyon 2009, 10.1021/bi900624j"),
    M("Glo", "nos", 2.6e-7, "Glorund's first two RNA-recognition motifs on nanos' translational control element", "Tamayo 2017, 10.1016/j.celrep.2017.03.022"),
    M("osk", "osk", 9.0e-8, "two oskar 3' UTRs pairing by their palindromic loop, apparent constant (270 nM with the palindrome mutated)", "Ye 2026, 10.1091/mbc.E26-04-0176"),
    dict(a="Stau", b="osk", kd_M=1.0e-8, dg_kcal_mol=dg(1.0e-8), status="homolog", what="NOT the fly pair: human Staufen1 on its ARF1 site 6 nM, mouse Staufen2 1.3 nM and 185 nM; 10 nM taken",
         source="Lazzaretti 2018, 10.26508/lsa.201800187; Heber 2019, 10.1038/s41467-019-09655-3"),
]
PP = round((dg(1.0e-5) + dg(4.8e-5)) / 2, 1)                          # assumed protein-protein: the mean of the two measured ones
PR = sorted(dg(k) for k in (1.24e-7, 2.6e-7, 1.0e-8))[1]             # assumed specific protein-mRNA: the median of the three measured ones
A = lambda a, b, g, what, src: dict(a=a, b=b, kd_M=None, dg_kcal_mol=g, status="assumed", what=what, source=src)
PAIRS += [
    A("Osk", "Osk", PP, "the LOTUS domain forms dimers; no constant read", "Jeske 2015, 10.1016/j.celrep.2015.06.055"),
    A("Bru", "Bru", PP, "Bruno packs oskar into large particles; no constant read", "Chekulaeva 2006 (abstract); Bose 2022"),
    A("Bru", "Cup", PP, "Cup associates with Bruno; no constant read", "Nakamura 2004 (abstract)"),
    A("Smg", "Cup", PP, "Smaug and Cup act together on nanos; no constant read", "Pekovic 2023 (abstract)"),
    A("Smg", "Osk", PP, "Oskar binds Smaug; no constant in the text read", "Kubikova 2023, 10.1073/pnas.2304385120"),
    A("Hrp48", "osk", PR, "Hrp48 binds oskar and scaffolds its granules; no constant read", "Yano 2004 (abstract); Bose 2022"),
    A("Smg", "nos", PR, "Smaug's SAM domain on nanos' Smaug recognition elements; the papers' text not read", "Aviv 2003, 10.1038/nsb956; Green 2003"),
] + [A("Osk", m, PR, "Oskar crosslinks to this mRNA in vivo; no constant", "Jeske 2015, 10.1016/j.celrep.2015.06.055") for m in ("nos", "pgc", "gcl")] \
  + [A("Aub", m, W, "the piRNA trap, partial base pairing; no constant: the weak value", "Vourekas 2016") for m in ("nos", "pgc", "gcl", "CycB", "Hsp83")] \
  + [A("Vas", m, W, "Vasa keeps mRNAs in the granule; no constant: the weak value", "Tian 2026") for m in ("nos", "pgc", "gcl", "CycB", "Hsp83")] \
  + [A(m, m, W, "an mRNA clusters with its own kind, by no specific sequence; no constant: the weak value", "Trcek 2020; Niepielko 2018") for m in ("nos", "pgc", "gcl", "CycB", "Hsp83")]
kinds = [dict(key=k, name=n, kind="mRNA", nt=nt, nt_refseq=r, subunits=max(1, round(nt / 100)), where=w) for k, n, nt, r, w in MRNA] \
      + [dict(key=k, name=n, kind="protein", aa=aa, aa_refseq=r, globule_nm=diam(aa), what=w) for k, n, aa, r, w in PROT]
out = dict(temperature_C=25, delta_g="standard binding free energy R T ln(K_d / 1 M), kcal/mol; a pair not listed does not bind",
           subunit="one 6 nm subunit of an mRNA chain per 100 nucleotides", assumed=dict(protein_protein=PP, protein_mrna_specific=PR, weak=W), kinds=kinds, pairs=PAIRS)
yaml.safe_dump(out, open("cast.yaml", "w"), sort_keys=False, width=1000, allow_unicode=True)
names = [k["key"] for k in kinds]
assert all(p["a"] in names and p["b"] in names for p in PAIRS) and len({tuple(sorted((p["a"], p["b"]))) for p in PAIRS}) == len(PAIRS)
md = ["| kind | what | real size | in the scene |", "|---|---|---|---|"]
for k in kinds:
    if k["kind"] == "mRNA":
        md.append(f"| {k['name']} mRNA | {k['where']} | {k['nt']:,} nucleotides (RefSeq: {k['nt_refseq']}) | a chain of {k['subunits']} subunits |")
    else:
        md.append(f"| {k['name']} | {k['what']} | {k['aa']:,} amino acids (RefSeq: {k['aa_refseq']}) | one body; a compact globule of that mass is {k['globule_nm']} nm across |")
md += ["", "| pair | K_d | ΔG° at 25 °C, kcal/mol | ΔG°, kT | status | what was measured, or why assumed | source |", "|---|---|---|---|---|---|---|"]
for p in PAIRS:
    kd = p["kd_M"]
    kds = "none" if kd is None else (f"{kd * 1e9:.3g} nM" if kd < 1e-6 else f"{kd * 1e6:.3g} µM")
    md.append(f"| {p['a']} – {p['b']} | {kds} | {p['dg_kcal_mol']} | {p['dg_kcal_mol'] / RT:.1f} | {p['status']} | {p['what']} | {p['source']} |")
open("/tmp/oo_cast_tables.md", "w").write("\n".join(md) + "\n")
n = {s: sum(p["status"] == s for p in PAIRS) for s in ("measured", "homolog", "assumed")}
print(len(kinds), "kinds;", len(PAIRS), "pairs of", len(names) * (len(names) + 1) // 2, n, "| assumed PP", PP, "PR", PR, "weak", W)
print("\n".join(md))
