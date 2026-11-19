---
number: 10
name: oocyte_posterior
purpose: simulate, with Plexus's MPM bodies and binding operators, the mRNAs and proteins that assemble at the posterior pole of the Drosophila oocyte -- oskar's own granules and the germ granules Oskar protein nucleates -- from the pairwise binding free energies of their components
question: with one binding free energy per pair of kinds (Oskar, Vasa, Tudor, Aubergine; nanos, pgc, gcl, cycB and oskar mRNA), do granules form in which each mRNA gathers into clusters of its own kind while oskar stays in granules of its own, and which entries of that matrix does it take?
status: design -- papers downloaded, cast of 16 kinds and its 31 binding free energies written (cast.yaml); no scene run yet
record: experiments/exp10_oocyte_posterior
papers: experiments/exp10_oocyte_posterior/papers (a link to experiments/exp09_molecular_landscape/download, where Cedric asked for them; index oocyte_posterior_papers.md)
watcher: http://127.0.0.1:8844/watch (not started: no step yet)
tool: tools/exp09_landscape.py (the scene builder of exp09's mRNA runs, steps 24 to 26 there; exp10's own tool when the first scene is agreed)
queue: gpu_h100
started: 2026-11-19
---

# exp10 oocyte_posterior

(The slot's earlier experiment 10, organoid_crypt, has no markdown left in `experiments/`; its 169 runs are `graphs_data/tissue/exp10*`. This one's runs will be
named `exp10_oo_*`, so none of them is read here -- as exp09 did, TRAPS T-90.)

## The ask (Cedric, 2026-11-19)

> what can you tell me in the fly oocyte mRNA and protein involved in the posterior part like oskar
> yes I would need to know more of the other mRNA and proteins involved in the posterior pole
> download the paper into /workspace/Plexus/experiments/exp09_molecular_landscape/download
> start a new exp10 oocyte_posterior

It follows exp09's steps 24 to 26 (mRNA as chains of subunits, proteins as bodies, binding sites along the mRNA, a matrix of binding free energies between kinds:
`contact_finder[sites]`, `bond_binding[delta_g]`).

## At a glance

- **Question.** Can a matrix of binding free energies between a handful of kinds reproduce the two granule types of the posterior pole and the sorting of mRNAs inside them?
- **Time scale.** A run of exp09's kind is 200 microseconds of 10 nm bodies; the assembly it stands for takes hours (stage 9 of oogenesis to the early embryo). The runs show which structures the energies favour, not their real speed.
- **The movie a talk would show.** Proteins and mRNAs of the germ plasm condensing into granules, each mRNA kind in its own cluster, oskar apart.
- **Status / next step.** Design. Next: agree on the first scene (below), then give it its ΔG matrix and run it.

## The papers

Twenty with full text (`.xml` + `.txt`), three of them also as PDF; listed with DOIs in `papers/oocyte_posterior_papers.md`. Every statement below names the paper it was
read in; "review" is Lehmann 2016 (Curr Top Dev Biol, 10.1016/bs.ctdb.2015.11.024). Figures and panels are not yet read: a number is quoted from the text of the
paper named, and its panel is to be found before it becomes a gate.

## The cast of the posterior pole

### 1. oskar mRNA and what carries it (stages 8 to 10 of oogenesis)

| component | what it is | what it does | read in |
|---|---|---|---|
| oskar mRNA | the organiser's messenger; its 3' UTR and a splice-made element (SOLE) carry the localisation signals | made in the nurse cells, moved into the oocyte by dynein, then to the posterior by kinesin; translated only there; also acts as a non-coding RNA that holds Bruno | review |
| exon junction complex: Mago nashi, Y14, eIF4AIII, Barentsz | proteins left on the mRNA by splicing | with the SOLE, give the particle its preference for kinesin | review |
| Staufen | a protein with five double-stranded-RNA-binding domains | joins the transport particle; needed for kinesin transport and for translation at the posterior; competes with Egalitarian, which switches the particle from dynein to kinesin | review; Gáspár 2023 |
| Bruno | an RNA-recognition-motif protein that binds several regions of the oskar 3' UTR (Bruno response elements) | represses oskar translation during transport, with Cup; packs oligomerised oskar into large silencing particles; a scaffold of oskar granules | review; Bose 2022 |
| Cup | binds the cap-binding factor eIF4E | blocks translation initiation on Bruno-bound oskar | review |
| Hrp48, PTB | RNA-binding proteins | Hrp48 a scaffold of oskar granules with Bruno; PTB a client that partitions into them | Bose 2022 |
| oskar-oskar | a palindromic loop in the 3' UTR (dimerisation domain) | oskar mRNAs pair and travel together ("hitchhiking") | review (Jambor 2011) |
| kinesin-1, dynein, Tropomyosin 1, myosin-V | motors and adaptors | dynein into the oocyte, kinesin to the posterior, myosin-V competing at the cortex | review; Lu 2020 |

oskar granules are condensates with solid-like properties; made in vitro from oskar RNA with Bruno and Hrp48 they pass from liquid to solid, the liquid taking in more
RNA and the solid not (Bose 2022). In the embryo oskar sits in "founder granules" with Staufen, apart from the germ granules, where it is degraded (Eichler 2020).

### 2. Oskar protein and the germ granule's proteins

| component | what it is | what it does | read in |
|---|---|---|---|
| Short Oskar | 467 amino acids; a LOTUS domain (residues 139 to 240) and a C-terminal OSK domain | nucleates the germ granules; needed and sufficient for germ plasm; partners Vasa, Valois, Lasp | review |
| Long Oskar | 606 amino acids, Short Oskar with an N-terminal extension | at the posterior cortex with endosomes; anchors the germ plasm (actin, endocytosis) | review |
| Vasa | a DEAD-box RNA helicase | bound and stimulated by Oskar's LOTUS domain; keeps Oskar condensates exchanging; with Oskar, needed for mRNAs to stay in the granules | Jeske 2017; Tian 2026 |
| Tudor | 2,515 amino acids with 11 Tudor domains | required for germ plasm; its interaction with Aubergine is essential for granules | review; Vourekas 2016 |
| Aubergine | a Piwi protein loaded with piRNAs | its piRNAs hold mRNAs in the germ plasm by partial base pairing ("adhesion trap") | Vourekas 2016 |
| Valois, Lasp | a WD-repeat protein; an actin-binding protein | partners of Short Oskar | review |

The proteins (Vasa, Oskar, Tudor, Aubergine) are spread evenly through a granule (Trcek 2015). Proteins, not mRNAs, nucleate the granules and keep them; the mRNAs
set their size and content (Curnutte 2023).

### 3. The mRNAs the germ granules collect

| mRNA | what it encodes | where in the granule | translated | read in |
|---|---|---|---|---|
| nanos (nos) | a zinc-finger protein; with Pumilio represses hunchback (abdomen) and Cyclin B | toward the centre | as soon as it is localised | review; Trcek 2015 |
| Cyclin B (cycB) | a mitotic cyclin | toward the centre | not until the end of embryogenesis | review |
| polar granule component (pgc) | a small protein | toward the periphery | when the germ cells form | review |
| germ cell-less (gcl) | a BTB protein needed to form the germ cells | toward the periphery | as nuclei reach the cortex | review |
| oskar | Oskar | NOT in the germ granules: large clusters of its own | at the posterior, stages 9 on | review; Eichler 2020 |

Unlocalised nanos is kept silent by Glorund in the oocyte and by Smaug with Argonaute 1 in the embryo (review). More than 100 maternal mRNAs are reported in the germ
plasm (review; about 200 in Trcek 2015).

### 4. Numbers

| quantity | value | read in |
|---|---|---|
| diameter of a germ granule | 250 nm | Curnutte 2023 (its reference 46) |
| granules per area of posterior cortex | 1.6 to 1.8 per square micrometre (light), 1.7 (electron microscopy) | Curnutte 2023 |
| mRNAs of one gene in a homotypic cluster | 2 to more than 40 | Trcek & Lehmann 2019 |
| share of an effector mRNA that reaches the germ plasm | 2.5 to 4 % | review |
| effector mRNA concentration, germ plasm over the rest of the embryo | 8 to 12 times | review |
| granules that hold nanos | 56 % (51 % at half the nanos dose) | Niepielko 2018 |
| halving oskar's dose | 39 % fewer nanos and 28 % fewer pgc mRNAs per granule | Niepielko 2018 |
| halving vasa by knock-down | 48 % fewer granules, 15 % larger | Curnutte 2023 |

### 5. How the granules are thought to assemble

mRNAs arrive as single molecules by diffusion and are trapped (review; Vourekas 2016); a first transcript seeds a cluster that then recruits more of its own kind, the
clusters of two mRNAs in one granule growing independently of each other but both with the amount of Oskar (Niepielko 2018). Which granule an mRNA enters is set by
regions of the RNA; clustering with its own kind takes the whole mRNA and no specific sequence (Trcek 2020), and palindromes can do it in a model (Kimchi 2026).
Valentino et al. 2022 is a computational model of this seeding and self-recruitment: the prior model to compare with.

## A first scene to propose (not run)

Kinds: Short Oskar, Vasa, Tudor, Aubergine as bodies; nanos, pgc, gcl, cycB and oskar as chains of subunits; Bruno and Staufen for oskar. One ΔG per pair, read from
the tables above as signs first (binds / does not), magnitudes a model choice until a measured constant is found:

- Oskar-Oskar, Oskar-Vasa, Tudor-Aubergine: bind (the granule's protein scaffold);
- Aubergine and Oskar with nanos, pgc, gcl, cycB: bind (the trap); none with oskar mRNA;
- each of nanos, pgc, gcl, cycB with itself: bind weakly (homotypic clusters); between two different mRNAs: none;
- oskar with oskar, Bruno and Staufen: bind (its own granules).

Open before it runs: the scale. A 250 nm granule holds thousands of 10 nm bodies; either a granule is simulated at a tenth of its size, or a body stands for a complex
of several proteins. The measured constants are in the next section.

## The cast and its binding free energies (2026-11-19)

Cedric: "first I would like to figure out a better list of proteins and mRNA than the dummy one we have now and if possible collect DeltaG for these pairs", then "do
not go too far, lets just make a better list of bodies / DeltaG than a random one". The list below replaces the 10 + 10 kinds and the random matrix of exp09's steps 24
to 27. It is `experiments/exp10_oocyte_posterior/cast.yaml` (written by `data/session/oo_cast.py` of exp09's record); sizes are the reference transcripts and proteins
of NCBI (`cast_sizes_ncbi.json`), constants are from the search in `kd_search.json`.

**16 kinds: 6 mRNAs and 10 proteins.** An mRNA is a chain of one 6 nm subunit per 100 nucleotides (7 to 29 subunits, exp09's range was 5 to 28): 6 nm per 100
nucleotides is 5 to 10 times shorter than the stretched strand (0.3 to 0.6 nm per nucleotide), so the chain stands for a folded mRNA. A protein is one body; the last
column gives the diameter of a compact globule of its mass (0.133 nm³ per amino acid), 4.7 to 8.6 nm, against the 10 nm bodies of exp09 -- a lower bound, since several
of these proteins are largely disordered.

| kind | what | real size | in the scene |
|---|---|---|---|
| oskar mRNA | its own granules, not the germ granules | 2,869 nucleotides (RefSeq: 2,869) | a chain of 29 subunits |
| nanos mRNA | germ granule, toward the centre | 2,292 nucleotides (RefSeq: 2,292 to 2,349) | a chain of 23 subunits |
| polar granule component mRNA | germ granule, toward the periphery | 686 nucleotides (RefSeq: 643 to 1,211) | a chain of 7 subunits |
| germ cell-less mRNA | germ granule, toward the periphery | 2,464 nucleotides (RefSeq: 2,464 to 4,238) | a chain of 25 subunits |
| Cyclin B mRNA | germ granule, toward the centre | 2,548 nucleotides (RefSeq: 2,548 to 2,633) | a chain of 25 subunits |
| Hsp83 mRNA | germ plasm | 2,713 nucleotides (RefSeq: 2,713 to 2,960) | a chain of 27 subunits |
| Short Oskar | nucleates the germ granules; LOTUS domain (dimer, binds Vasa) + OSK domain (binds RNA) | 468 amino acids (RefSeq: 468 (Long Oskar 606)) | one body; a compact globule of that mass is 4.9 nm across |
| Vasa | DEAD-box RNA helicase of the germ granule | 661 amino acids (RefSeq: 661) | one body; a compact globule of that mass is 5.5 nm across |
| Tudor | 11 Tudor domains, each a site for a methylated arginine of Aubergine | 2,515 amino acids (RefSeq: 2,512 to 2,515) | one body; a compact globule of that mass is 8.6 nm across |
| Aubergine | Piwi protein; its piRNAs hold mRNAs by partial base pairing | 866 amino acids (RefSeq: 795 to 866) | one body; a compact globule of that mass is 6.0 nm across |
| Bruno | represses oskar; scaffold of oskar granules | 604 amino acids (RefSeq: 573 to 810) | one body; a compact globule of that mass is 5.4 nm across |
| Staufen | five double-stranded-RNA-binding domains; travels with oskar | 1,026 amino acids (RefSeq: 914 to 1,026) | one body; a compact globule of that mass is 6.4 nm across |
| Hrp48 | scaffold of oskar granules with Bruno | 421 amino acids (RefSeq: 421) | one body; a compact globule of that mass is 4.7 nm across |
| Cup | binds Bruno and the cap-binding factor; blocks oskar translation | 1,117 amino acids (RefSeq: 1,117) | one body; a compact globule of that mass is 6.6 nm across |
| Glorund | keeps unlocalised nanos silent in the oocyte | 586 amino acids (RefSeq: 586) | one body; a compact globule of that mass is 5.3 nm across |
| Smaug | keeps unlocalised nanos silent in the embryo; Oskar takes it off nanos | 999 amino acids (RefSeq: 999) | one body; a compact globule of that mass is 6.3 nm across |

**31 pairs bind, of the 136 pairs of 16 kinds: 5 with a measured constant for the fly pair, 1 from mammalian homologs, 25 assumed.** ΔG° = R T ln(K_d / 1 M) at 25 °C
(R T = 0.59 kcal/mol), the standard binding free energy; a pair not listed does not bind. Assumed values, where binding is reported and no constant was found: -6.3
kcal/mol between two proteins (the mean of the two measured protein pairs), -9.4 kcal/mol for a protein on its own mRNA (the median of the three measured ones), and
-5.0 kcal/mol (K_d 0.2 mM, weaker than every measured pair) for the three unspecific kinds of pair: Aubergine's piRNA trap, Vasa on the mRNAs, an mRNA with its own kind.

| pair | K_d | ΔG° at 25 °C, kcal/mol | ΔG°, kT | status | what was measured, or why assumed | source |
|---|---|---|---|---|---|---|
| Osk – Vas | 10 µM | -6.8 | -11.5 | measured | LOTUS domain with Vasa's C-terminal domain, calorimetry at 25 C | Jeske 2017, 10.1101/gad.297051.117 |
| Tud – Aub | 48 µM | -5.9 | -10.0 | measured | ONE Tudor domain (the 11th) with ONE methylated arginine of Aubergine, calorimetry; 6 to 71 uM over three arginines | Liu 2010, 10.1101/gad.1956010 |
| Bru – osk | 124 nM | -9.4 | -15.9 | measured | Bruno's extended third RNA-recognition motif on a Bruno response element, half-saturation, Hill coefficient 1.4 | Lyon 2009, 10.1021/bi900624j |
| Glo – nos | 260 nM | -9.0 | -15.2 | measured | Glorund's first two RNA-recognition motifs on nanos' translational control element | Tamayo 2017, 10.1016/j.celrep.2017.03.022 |
| osk – osk | 90 nM | -9.6 | -16.2 | measured | two oskar 3' UTRs pairing by their palindromic loop, apparent constant (270 nM with the palindrome mutated) | Ye 2026, 10.1091/mbc.E26-04-0176 |
| Stau – osk | 10 nM | -10.9 | -18.4 | homolog | NOT the fly pair: human Staufen1 on its ARF1 site 6 nM, mouse Staufen2 1.3 nM and 185 nM; 10 nM taken | Lazzaretti 2018, 10.26508/lsa.201800187; Heber 2019, 10.1038/s41467-019-09655-3 |
| Osk – Osk | none | -6.3 | -10.6 | assumed | the LOTUS domain forms dimers; no constant read | Jeske 2015, 10.1016/j.celrep.2015.06.055 |
| Bru – Bru | none | -6.3 | -10.6 | assumed | Bruno packs oskar into large particles; no constant read | Chekulaeva 2006 (abstract); Bose 2022 |
| Bru – Cup | none | -6.3 | -10.6 | assumed | Cup associates with Bruno; no constant read | Nakamura 2004 (abstract) |
| Smg – Cup | none | -6.3 | -10.6 | assumed | Smaug and Cup act together on nanos; no constant read | Pekovic 2023 (abstract) |
| Smg – Osk | none | -6.3 | -10.6 | assumed | Oskar binds Smaug; no constant in the text read | Kubikova 2023, 10.1073/pnas.2304385120 |
| Hrp48 – osk | none | -9.4 | -15.9 | assumed | Hrp48 binds oskar and scaffolds its granules; no constant read | Yano 2004 (abstract); Bose 2022 |
| Smg – nos | none | -9.4 | -15.9 | assumed | Smaug's SAM domain on nanos' Smaug recognition elements; the papers' text not read | Aviv 2003, 10.1038/nsb956; Green 2003 |
| Osk – nos | none | -9.4 | -15.9 | assumed | Oskar crosslinks to this mRNA in vivo; no constant | Jeske 2015, 10.1016/j.celrep.2015.06.055 |
| Osk – pgc | none | -9.4 | -15.9 | assumed | Oskar crosslinks to this mRNA in vivo; no constant | Jeske 2015, 10.1016/j.celrep.2015.06.055 |
| Osk – gcl | none | -9.4 | -15.9 | assumed | Oskar crosslinks to this mRNA in vivo; no constant | Jeske 2015, 10.1016/j.celrep.2015.06.055 |
| Aub – nos | none | -5.0 | -8.4 | assumed | the piRNA trap, partial base pairing; no constant: the weak value | Vourekas 2016 |
| Aub – pgc | none | -5.0 | -8.4 | assumed | the piRNA trap, partial base pairing; no constant: the weak value | Vourekas 2016 |
| Aub – gcl | none | -5.0 | -8.4 | assumed | the piRNA trap, partial base pairing; no constant: the weak value | Vourekas 2016 |
| Aub – CycB | none | -5.0 | -8.4 | assumed | the piRNA trap, partial base pairing; no constant: the weak value | Vourekas 2016 |
| Aub – Hsp83 | none | -5.0 | -8.4 | assumed | the piRNA trap, partial base pairing; no constant: the weak value | Vourekas 2016 |
| Vas – nos | none | -5.0 | -8.4 | assumed | Vasa keeps mRNAs in the granule; no constant: the weak value | Tian 2026 |
| Vas – pgc | none | -5.0 | -8.4 | assumed | Vasa keeps mRNAs in the granule; no constant: the weak value | Tian 2026 |
| Vas – gcl | none | -5.0 | -8.4 | assumed | Vasa keeps mRNAs in the granule; no constant: the weak value | Tian 2026 |
| Vas – CycB | none | -5.0 | -8.4 | assumed | Vasa keeps mRNAs in the granule; no constant: the weak value | Tian 2026 |
| Vas – Hsp83 | none | -5.0 | -8.4 | assumed | Vasa keeps mRNAs in the granule; no constant: the weak value | Tian 2026 |
| nos – nos | none | -5.0 | -8.4 | assumed | an mRNA clusters with its own kind, by no specific sequence; no constant: the weak value | Trcek 2020; Niepielko 2018 |
| pgc – pgc | none | -5.0 | -8.4 | assumed | an mRNA clusters with its own kind, by no specific sequence; no constant: the weak value | Trcek 2020; Niepielko 2018 |
| gcl – gcl | none | -5.0 | -8.4 | assumed | an mRNA clusters with its own kind, by no specific sequence; no constant: the weak value | Trcek 2020; Niepielko 2018 |
| CycB – CycB | none | -5.0 | -8.4 | assumed | an mRNA clusters with its own kind, by no specific sequence; no constant: the weak value | Trcek 2020; Niepielko 2018 |
| Hsp83 – Hsp83 | none | -5.0 | -8.4 | assumed | an mRNA clusters with its own kind, by no specific sequence; no constant: the weak value | Trcek 2020; Niepielko 2018 |

Left out on purpose: oskar mRNA with Oskar protein (binds in vitro, Yang 2015, 10.1073/pnas.1515568112, but oskar is kept out of the germ granules); two different
mRNAs of the germ granule (their clusters grow independently, Niepielko 2018). Measured and not in the cast: Pumilio on its site in hunchback 0.5 nM (Zamore 1999,
10.1021/bi982264s); the transport complex Egalitarian-Bicaudal D on four localisation signals 7 to 603 nM (Singh 2026, 10.1038/s41594-026-01794-8); the cap analogue
on eIF4E 726 nM, 372 nM with Cup bound (Kinkelin 2012, 10.1261/rna.033639.112). Not found: Vasa on RNA (Sengoku 2006), Smaug on nanos (Aviv 2003, Green 2003: text
not reachable), Cup on Bruno, Hrp48 on oskar, Aubergine's piRNA on a target.

**What these energies do in a run of exp09's length.** `bond_binding[delta_g]` with `delta_g_reference: standard` adds kT ln(0.6022 x the volume a bound pair is held
in, nm³) to ΔG°. With that volume at (4.5 nm)³ = 91 nm³ -- the cube of the scene's bonding range, a model choice -- the shift is +4.0 kT, and at one binding
attempt per frame a bond lasts exp(-ΔG / kT) frames: the weak assumed pairs 84 frames, Tudor-Aubergine 380, Oskar-Vasa 1,822, Bruno-oskar
146,957. In a 1,000-frame run the weak pairs come and go, the granule's protein scaffold exchanges a few times, and a protein on its own mRNA never
lets go.

## Directions

1. The germ granule alone: scaffold proteins plus two mRNAs (nanos, pgc); do homotypic clusters form, and where in the granule.
2. Two granule types in one scene: oskar's granules beside the germ granules; does oskar stay out.
3. Doses: half the Oskar, twice the nanos, against Niepielko 2018's numbers.

## Decisions

- 2026-11-19: papers go where Cedric asked (exp09's `download/`), and `papers/` of this record is a link to it, so the rule "the papers in the experiment's own folder" holds without a second copy.
- 2026-11-19: no scene is run before the first scene is agreed: the kinds and the scale are Cedric's to choose.
- 2026-11-19: shapes (Cedric: "so we have cast, what about the bodies shape ?"): proposed, not yet built -- one lobe per known domain, so that a body's size follows its mass (Tudor a string of 11 lobes, Staufen 5, Bruno and Glorund 3, Oskar, Vasa and Hrp48 2, Aubergine 4, Cup and Smaug loose strings: largely disordered). Real structures need the libraries, blocked from the devcontainer; Cedric: "add them to the devcontainer json list of allowed website and I shall rebuild" -- six hosts added to /workspace/.devcontainer/init-firewall.sh and, as WebFetch permissions, to .claude/settings.local.json (alphafold.ebi.ac.uk, rest.uniprot.org, files.rcsb.org, data.rcsb.org, search.rcsb.org, and www.ebi.ac.uk for Europe PMC's full texts and the domain annotations), in force after the rebuild.
- 2026-11-19: Cedric: "use h100 for exp10" -- every exp10 job goes to gpu_h100 (the front matter's queue).
- 2026-11-19: Cedric likes the montage: bodies at 1 nm per point (the length unit read as 5 nm), each protein ONE elastic MPM body on its own grid block (`per_parent: cell`, as every exp09 scene), the block widened from 15 to about 24 cells for Tudor, Staufen and Smaug (14 to 15 points long); uniform stiffness in the first run, a per-point stiffness (soft disordered stretches) a later variant of the strain operator.
- 2026-11-19: the literature search stops at the constants found (Cedric: "do not go too far"): 5 measured pairs, 1 from homologs; the other 25 pairs take three assumed values, each named in cast.yaml as assumed.

## Results

| step | key | what | bodies | health | result |
|---|---|---|---|---|---|

## Log

- 2026-11-19 14:30: experiment opened. PubMed searched (oskar germ plasm reviews; germ granule mRNA clusters; oskar granules; Oskar-Vasa); 24 papers chosen, 20 fetched
  with full text from PubMed Central by E-utilities, 4 abstract only (the publisher withholds the XML), 3 PDFs from eLife's CDN; PubMed Central's PDFs, Europe PMC and PLoS
  are blocked from the devcontainer. The cast above written from the Lehmann 2016 review and the abstracts and text of the others.
- 2026-11-19 15:10: the cast. Reference sizes of 30 genes from NCBI (`cast_sizes_ncbi.json`; the lists for nos and smg also hold other genes of those symbols, read by
  hand); PubMed searched for dissociation constants of the pairs (92 abstracts, 36 full texts, `kd_search.json`); Liu 2010 and Kinkelin 2012 read from PubMed Central's
  web page, which the E-utilities XML withholds. `cast.yaml`: 16 kinds (6 mRNAs, 10 proteins), 31 binding pairs (5 measured, 1 homolog, 25 assumed). Not yet a scene.
- 2026-11-19 15:50: Cedric: "are we ready to run in exp10 in 3D no boundary ?" NOT YET, three pieces, none built:
  (1) SHAPES of the 10 proteins (Decisions). After the rebuild: AlphaFold's model of each from https://alphafold.ebi.ac.uk/files/AF-<accession>-F1-model_v4.pdb, the accessions
  read from NCBI (`cast_uniprot_ncbi.json`): Oskar P25158 (the 606-residue Long form; Short Oskar is its residues 139 on), Vasa P09052, Tudor P25823, Aubergine O76922,
  Staufen P25159, Hrp48 P48809, Cup Q9VMA3, Smaug Q23972; Bruno and Glorund not found through NCBI (to look up at rest.uniprot.org: genes bru1 / aret, glo). A lobe per
  confident compact part of the model, a loose string for the rest.
  (2) A STRAND MUST NOT BIND ITSELF: the cast has mRNA-mRNA pairs (oskar-oskar, and each germ-granule mRNA with its own kind), and two neighbouring sites of one strand are
  always in contact (contact_finder[sites] makes a junction row for them; in exp09's steps 24 to 27 no mRNA bound an mRNA, so it never showed). Plan: a variant
  bond_binding[delta_g_molecule] (a subclass of bond_binding[delta_g] in operators/binding_delta_g_ops.py, with a cell block `molecule`: for a chain's subunit the chain's
  first cell, for any other body itself; a junction whose two cells share it never binds), the seed table's `cell__molecule` written beside `cell__site`
  (tools/exp09_landscape.py, `_sites_of`), and a unit test.
  (3) THE SCENE: cast.yaml is read by no scene yet; the periodic 3D box (exp09 step 26, rna1_3dp) and bond_binding[delta_g] (step 27, rna2) have each run alone. The scene =
  rna1_3dp's box with the cast's kinds, chains of `subunits`, the pair table as `delta_g` in kcal/mol with `delta_g_reference: standard` and `binding_volume_nm3: 91`, the
  matrix panel of step 27; counts at step 26's crowding unless Cedric gives numbers. Then a smoke test on gpu_rtx6000 and a first-frame picture of the 16 shapes BEFORE any
  full run.
- 2026-11-19 16:20: the container rebuilt; alphafold.ebi.ac.uk, rest.uniprot.org and files.rcsb.org answer (www.ebi.ac.uk still refuses). AlphaFold's model (version 6) of
  each of the 10 proteins is in `structures/` with its predicted aligned error (`models.json`: Oskar P25158, Vasa P09052, Tudor P25823, Aubergine O76922, Bruno Q960Z4 the
  604-residue isoform, Staufen P25159, Hrp48 P48809, Cup Q9VMA3, Glorund Q9VGH5, Smaug Q23972; Bruno and Glorund are unreviewed entries). Each model cut into folded domains
  (confidence pLDDT 70 or more, runs of 30 residues, two runs one domain when packed together with a predicted aligned error of 10 angstrom or less) and a disordered
  rest: `shapes.yaml` (data/session/oo_domains.py of exp09's record). READ: a folded domain is 2.2 to 5.8 nm across as a compact globule (Aubergine one block of 775
  residues, Vasa's helicase core 423, Tudor 10 domains by this cut against 11 Tudor domains in the literature, Staufen 5); 6 of the 10 proteins are more than half
  disordered (Bruno 62 %, Hrp48 63 %, Staufen 68 %, Smaug 69 %, Glorund 55 %, Cup 92 %). At exp09's 2 nm per point a domain is 1 to 3 points across: no shape.
  PROPOSED BODIES, `bodies.yaml` and `shapes_montage.png` (oo_shapes.py): 1 nm per point, 0.133 nm^3 per residue (56 points for Hrp48 to 334 for Tudor, under the 343
  slots), one lobe per folded domain and one per disordered stretch of 40 residues or more, in sequence order as a string, every body in one piece. A disordered stretch is
  drawn at its compact volume, smaller than the cloud it is. 1 nm per point needs no numerical change if the length unit is read as 5 nm instead of 10 (spacing 0.2 units):
  a 64-point body is then 5 nm, a 27-point mRNA subunit 3 nm (50 nucleotides at the cast's 6 nm per 100). Not decided: Cedric's.
