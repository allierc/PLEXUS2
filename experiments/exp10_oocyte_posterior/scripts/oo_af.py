"""Fetch AlphaFold's model of each protein of exp10's cast into the record's structures/ folder (the accession from UniProt's reviewed entry of the fly gene), and print
its length and the share of residues predicted with confidence (pLDDT >= 70, the model's B-factor column). Usage: oo_af.py <out folder>"""
import json, os, sys, time, urllib.parse, urllib.request
OUT = sys.argv[1]
UA = {"User-Agent": "research-client"}
def get(u, t=40):
    with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=t) as r:
        return r.read()
CAST = [("Osk", "osk", "P25158"), ("Vas", "vas", "P09052"), ("Tud", "tud", "P25823"), ("Aub", "aub", "O76922"), ("Bru", "bru1", None), ("Stau", "stau", "P25159"),
        ("Hrp48", "Hrb27C", "P48809"), ("Cup", "cup", "Q9VMA3"), ("Glo", "glo", None), ("Smg", "smg", "Q23972")]
res = {}
for key, gene, acc in CAST:
    if acc is None:                                                   # not found through NCBI: UniProt's own search, reviewed entries first
        q = urllib.parse.urlencode({"query": f"gene_exact:{gene} AND organism_id:7227", "fields": "accession,reviewed,protein_name,length", "format": "json", "size": 5})
        hits = json.loads(get("https://rest.uniprot.org/uniprotkb/search?" + q))["results"]
        hits.sort(key=lambda h: (h["entryType"].startswith("UniProtKB reviewed") is False, -h["sequence"]["length"]))
        for h in hits:
            print("   ", key, "candidate", h["primaryAccession"], h["entryType"], h["sequence"]["length"], h.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "?"))
        acc = hits[0]["primaryAccession"] if hits else None
        time.sleep(0.3)
    if acc is None:
        print(key, "no accession"); continue
    try:
        meta = json.loads(get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"))[0]
    except Exception as e:
        print(key, acc, "no AlphaFold entry:", str(e)[:80]); continue
    pdb = get(meta["pdbUrl"]); f = os.path.join(OUT, f"{key}_{acc}.pdb")
    open(f, "wb").write(pdb)
    try:
        open(os.path.join(OUT, f"{key}_{acc}_pae.json"), "wb").write(get(meta["paeDocUrl"]))
    except Exception as e:
        print("   ", key, "no PAE file:", str(e)[:60])
    b = [float(l[60:66]) for l in pdb.decode().splitlines() if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    res[key] = dict(accession=acc, gene=gene, name=meta.get("uniprotDescription"), residues=len(b), confident_share=round(sum(v >= 70 for v in b) / len(b), 2), model=os.path.basename(f),
                    version=meta.get("latestVersion"), url=meta["pdbUrl"])
    print(f"{key}: {acc} {meta.get('uniprotDescription')!s:.50} -- {len(b)} residues, {res[key]['confident_share']:.2f} of them at pLDDT >= 70, model v{meta.get('latestVersion')}")
    time.sleep(0.3)
json.dump(res, open(os.path.join(OUT, "models.json"), "w"), indent=1)
