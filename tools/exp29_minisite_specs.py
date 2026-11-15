"""exp29: pull the specs the minisite shows in its Okuda section out of `index.qmd`, as published.

The minisite section "Pattern and growth composed" (the experiment of Okuda et al. 2018, Sci. Rep. 8:2386, rebuilt as vertex-model
Plexus specs) embeds each clip's spec as an inline block. That text is the only record of what the clip was run with: the
specs on disk under `config/okuda/` and `config/promotion/` differ from it (other seeds and frame counts, defaults made explicit
or dropped, `r005_06` and `r021_06` present only in `_superseded_*`), and no run folder keeps a `spec.yaml`. So the published
text is read, not rebuilt: `experiments/exp29_cell_tissue_2/data/minisite_specs/<name>.yaml` per clip, and `manifest.json`.

    python tools/exp29_minisite_specs.py            # write the specs and the manifest
    python tools/exp29_minisite_specs.py --diff     # also print how each clip differs from its neighbour
"""
import argparse
import html
import json
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QMD = os.path.join(ROOT, "index.qmd")
OUT = os.path.join(ROOT, "experiments", "exp29_cell_tissue_2", "data", "minisite_specs")
HEADING = "<h3>Pattern and growth composed</h3>"
CANON = {"mesh_seed": "seed_mesh", "cell_chem_seed": "seed_cell_chem"}
NOT_A_PARAMETER = ("op", "at", "id", "model", "implementation", "cell_set", "vertex_set")


def clips():
    lines = open(QMD).read().split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.strip() == HEADING)
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("<h3>"))
    section = "\n".join(lines[start:end])
    out = []
    for card in re.findall(r'<figure class="sim-card">(.*?)</figure>', section, re.S):
        text = html.unescape(re.search(r"<pre>(.*?)</pre>", card, re.S).group(1))
        spec = yaml.safe_load(text)
        out.append(dict(
            name=spec["general"]["name"],
            label=re.search(r'class="sim-name"[^>]*>([^<]+)<', card).group(1),
            video=re.search(r'<video src="([^"]+)"', card).group(1),
            caption=re.sub(r"\s+", " ", re.search(r'class="sim-cap">(.*?)</span>', card, re.S).group(1)).strip(),
            comp_hash=(spec.get("_discovery") or {}).get("comp_hash"),
            text=text, spec=spec))
    return out


def operators(spec):
    """{`op[model]#k`: parameters} -- the k-th operator of that kind, canonical names, placement keys dropped."""
    out = {}
    for o in spec["operators"]:
        key = CANON.get(o["op"], o["op"])
        variant = o.get("model") or o.get("implementation")
        key += f"[{variant}]" if variant else ""
        k = 0
        while f"{key}#{k}" in out:
            k += 1
        out[f"{key}#{k}"] = {a: b for a, b in o.items() if a not in NOT_A_PARAMETER}
    return out


def differences(a, b):
    """The lines that separate spec `b` from spec `a`: operators added or removed, parameters changed, general keys."""
    oa, ob = operators(a), operators(b)
    lines = [f"-{k}" for k in oa if k not in ob] + [f"+{k}" for k in ob if k not in oa]
    for k in oa:
        if k in ob:
            lines += [f"{k}.{p}: {oa[k].get(p)} -> {ob[k].get(p)}"
                      for p in sorted(set(oa[k]) | set(ob[k])) if oa[k].get(p) != ob[k].get(p)]
    ga, gb = a["general"], b["general"]
    lines += [f"general.{k}: {ga.get(k)} -> {gb.get(k)}"
              for k in sorted(set(ga) | set(gb)) if k not in ("name", "title") and ga.get(k) != gb.get(k)]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", action="store_true")
    args = ap.parse_args()
    cl = clips()
    os.makedirs(OUT, exist_ok=True)
    for c in cl:
        with open(os.path.join(OUT, f"{c['name']}.yaml"), "w") as f:
            f.write(c["text"])
    manifest = [{k: c[k] for k in ("name", "label", "video", "caption", "comp_hash")} for c in cl]
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(dict(source="index.qmd, section 'Pattern and growth composed'", clips=manifest), f, indent=1)
    print(f"{len(cl)} specs written to {os.path.relpath(OUT, ROOT)}")
    if args.diff:
        for prev, cur in zip(cl, cl[1:]):
            d = differences(prev["spec"], cur["spec"])
            print(f"\n{prev['name']} -> {cur['name']} ({cur['label']}): {len(d)} lines")
            for ln in d:
                print("   ", ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
