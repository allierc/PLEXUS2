#!/usr/bin/env python
"""The containment hierarchy of a spec: a circular graph of its sets and a table of their numbers.

    PYTHONPATH=src python tools/spec_hierarchy.py <spec.yaml> [<spec.yaml> ...] -o montage.png

A set is an entity type; `parent` and `per_parent` say which sets live inside which and how many per parent, so the numbers are
products down the chain (1 cell x 500 mitochondria x 125 points). The graph puts the root in the centre and each level of
containment on a ring of its own, children at their parent's angle, a node's area following the square root of its count and its
colour the spec's `plotting.colors` for that set (the colours of the main view). The table lists each set's count and, for the
sets of material points (`entity: mpm_particle`), the points; the last line is the total of MPM points. Sets that declare no
parent and hold no children (a membrane mesh's vertices, faces and half-edges) are listed apart, drawn dashed.

The data (`hierarchy`) is separate from the drawing (`draw`), so the watcher's pane can use the same numbers.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import sys

import yaml

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MIN_ROWS = 5                      # a run of fewer stored rows is a still, not a movie: one panel, as the watcher plays none

BG, FG, DIM = "#0f1319", "#dfe5ea", "#7d8a96"
RING, FILL = "#3a4654", "#1a222b"   # a container's outline and the root disc; a black movie (`movie(grey=True)`) swaps these, BG and FG
TEXT_PT = None                      # a black movie sets every text of the circle-packing view to this one size, in pixels at its 72 dpi: 19.5 px of DejaVu Sans is the x-height of
#                                   the 3D movies' 11-point VTK text (Arial, 21 px) at their 900 pixels high


def _fs(default: float) -> float:
    return TEXT_PT or default

LOOKS = ("circular", "tree", "sunburst", "treemap", "circlepack", "icicle")
DEFAULT_LOOK = "circlepack"


def _summary():
    """`tools/spec_summary.py`, the module behind the watcher's `plexus` pane, imported by path and shared with the watcher's own copy
    (`plexus_spec_summary`): the sets, their counts and their containment are READ FROM IT, never recounted here, so the hierarchy and
    the pane cannot disagree on what exists."""
    m = sys.modules.get("plexus_spec_summary")
    if m is None:
        sp = importlib.util.spec_from_file_location("plexus_spec_summary", os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_summary.py"))
        m = importlib.util.module_from_spec(sp)
        sys.modules["plexus_spec_summary"] = m
        sp.loader.exec_module(m)
    return m


def dormant_of(raw: dict) -> set:
    """Sets a `park_reserve` seed parks whole (fraction 1.0): a reservoir of dormant points waiting to be polymerised, with no
    mass, no stress and no part in the cell, drawn nowhere. `plotting.hierarchy: {dormant: true}` shows them anyway."""
    out = set()
    for o in (raw.get("seed") or []) + (raw.get("operators") or []):
        if isinstance(o, dict) and o.get("op") == "park_reserve" and o.get("at") and float(o.get("fraction", 1.0)) >= 1.0:
            out.add(o["at"])
    return out


MPM_OPS = ("mpm_scatter", "mpm_gather", "mpm_strain", "mpm_viscosity")
UNFOLD_MAX = 64                                  # a set of at most this many instances is drawn instance by instance (9 cubes, not one circle)


def mpm_sets(raw: dict) -> set:
    """The sets of material points: those that declare `entity: mpm_particle`, and those an MPM operator (scatter, gather, strain,
    viscosity) acts `at` -- a spec need not declare the entity for its points to be points."""
    sets = raw.get("sets") or {}
    out = {nm for nm, s in sets.items() if isinstance(s, dict) and s.get("entity") == "mpm_particle"}
    out |= {o["at"] for o in (raw.get("operators") or []) if isinstance(o, dict) and o.get("op") in MPM_OPS and o.get("at") in sets}
    return out


def contained_points(raw: dict):
    """(the set that holds material points, the set of points it holds) for a spec with no composer: the first set with no parent that
    is the parent of another, and its points child; (None, None) when the spec nests no points."""
    sets = raw.get("sets") or {}
    parents = {s.get("parent") for s in sets.values() if isinstance(s, dict) and s.get("parent") in sets}
    root = next((nm for nm, s in sets.items() if isinstance(s, dict) and not s.get("parent") and nm in parents), None)
    pts = mpm_sets(raw)
    ps = next((nm for nm, s in sets.items() if isinstance(s, dict) and s.get("parent") == root and nm in pts), None) if root else None
    return (root, ps) if ps else (None, None)


def group_of(raw: dict) -> bool:
    """`plotting.hierarchy: {group: false}` keeps every set of a family (protein_a ... protein_e) as its own node; on by default."""
    v = (raw.get("plotting") or {}).get("hierarchy")
    return bool(v.get("group", True)) if isinstance(v, dict) else True


def look_of(raw: dict) -> str:
    """`plotting.hierarchy: <look>` (or `{look: <look>, group: bool, dormant: bool}`): how the hierarchy graph is drawn; circlepack when absent."""
    v = (raw.get("plotting") or {}).get("hierarchy", DEFAULT_LOOK)
    if isinstance(v, dict):
        v = v.get("look", DEFAULT_LOOK)
    v = str(v).lower()
    if v not in LOOKS:
        raise ValueError(f"plotting.hierarchy must be one of {LOOKS}, got {v!r}")
    return v



def _stem(name: str):
    """('protein', 'a') for protein_a and ('protein_node', 'a') for protein_a_node; (name, None) for a name with no one-letter or
    numeric suffix (cytoskeleton_new_node, rough_er, nuclear_envelope)."""
    m = re.fullmatch(r"(.+?)_([a-z]|\d+)(_node)?", name)
    return (m.group(1) + (m.group(3) or ""), m.group(2)) if m else (name, None)


def _group(levels: list, min_group: int = 3) -> list:
    """Collapse a family of sets -- the same stem, a one-letter or numeric suffix, the same parent, per-parent count and entity --
    into one node (`protein x5`), level by level, so a node's children are grouped under its group: five protein species and their
    five clouds of material points become two nodes whose counts and points are the sums."""
    remap, out = {}, [levels[0]]
    for d in range(1, len(levels)):
        buckets, order = {}, []
        for nd in levels[d]:
            par = remap.get(nd["parent"], nd["parent"])
            stem, suf = _stem(nd["name"])
            key = (stem, suf is not None, nd["per_parent"], nd["entity"], par)
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            buckets[key].append(nd)
        level = []
        for key in order:
            g = buckets[key]
            if key[1] and len(g) >= min_group:
                sufs = [_stem(x["name"])[1] for x in g]
                gname = f"{key[0]} x{len(g)}"
                node = dict(g[0])
                node.update(name=gname, parent=key[4], count=sum(x["count"] for x in g), points=sum(x["points"] for x in g),
                            members=[x["name"] for x in g], label=f"{key[0]} x{len(g)} ({sufs[0]}-{sufs[-1]})")
                for x in g:
                    remap[x["name"]] = gname
                level.append(node)
            else:
                for x in g:
                    x2 = dict(x)
                    x2["parent"] = key[4]
                    level.append(x2)
        out.append(level)
    return out


def _rosette(raw: dict, op: dict, live: dict | None = None) -> dict | None:
    """`seed_positions[rosette]` builds `arrays` rosettes of one INNER cell ringed by `n_outer` OUTER cells, from the cell set's
    types (`inner`, `outer_k`: the outer type of array k), each cell carrying the points set's `per_parent` material points. The
    spec declares 72 flat cells; the entity it means is 9 rosettes of 8 cells, so that is the hierarchy drawn."""
    sets = raw.get("sets") or {}
    cs, ps = op.get("cell_set"), op.get("at")
    if cs not in sets or ps not in sets:
        return None
    types = sets[cs].get("types") or {}
    arrays = op.get("arrays") or [1]
    na = 1
    for a in (arrays if isinstance(arrays, (list, tuple)) else [arrays]):
        na *= int(a)
    n_out = int(op.get("n_outer", 0))
    ppc = int(sets[ps].get("per_parent") or op.get("live_per_cell") or 0)
    labels = op.get("labels") or []
    nodes, levels = [], [[], [], []]

    def node(name, parent, entity, count, points, level, **kw):
        nd = {"name": name, "entity": entity, "parent": parent, "count": count, "declared": count, "per_parent": None,
              "points": points, "label": kw.pop("label", None), **kw}
        levels[level].append(nd)
        return nd

    total = na * (1 + n_out) * ppc
    ax_n, az_n = (int(arrays[0]), int(arrays[1])) if isinstance(arrays, (list, tuple)) and len(arrays) == 2 else (na, 1)
    asp = op.get("array_spacing") or [1.0, 1.0]
    if live is not None:
        return _rosette_at(raw, op, live, types, na, ax_n, az_n, asp, labels)
    node("array", None, "array", 1, total, 0, label=f"{'x'.join(str(int(a)) for a in arrays)} array", layout="grid", grid=(ax_n, az_n),
         grid_ratio=float(asp[1]) / float(asp[0]) if float(asp[0]) else 1.0)
    rows = [("array  " + "x".join(str(int(a)) for a in arrays), "1", "", fmt(total))]
    for j in range(na):
        ok = f"outer_{j}" if f"outer_{j}" in types else None
        e = (labels[j] if j < len(labels) else (types.get(ok or "", {}).get("youngs")))
        tag = f"rosette {j + 1}" + (f"  E {e:g}" if isinstance(e, (int, float)) else "")
        node(f"rosette {j + 1}", "array", "rosette", 1, (1 + n_out) * ppc, 1, label=tag, layout="rosette")
        node(f"inner {j + 1}", f"rosette {j + 1}", "cell", 1, ppc, 2, color_key="inner", label="inner cell")
        for m in range(n_out):
            node(f"outer {j + 1}.{m + 1}", f"rosette {j + 1}", "cell", 1, ppc, 2, color_key=ok or "outer_0", label="outer cell")
        rows.append(("  " + tag, "1", "", fmt((1 + n_out) * ppc)))
    rows.append(("  inner cell", fmt(na), "1 per rosette", fmt(na * ppc)))
    rows.append(("  outer cell", fmt(na * n_out), f"{n_out} per rosette", fmt(na * n_out * ppc)))
    rows.append((f"  {ps}  (points of one cell)", fmt(total // max(ppc, 1)), f"{ppc} per cell", fmt(total)))
    mesh = [{"name": nm, "count": int(s.get("n", 0)), "entity": s.get("entity") or "", "parent": None}
            for nm, s in sets.items() if nm not in (cs, ps) and isinstance(s, dict)]
    for m in mesh:
        rows.append((m["name"] + " (mesh)", fmt(m["count"]), "", ""))
    rows.append(("MPM points in all", "", "", fmt(total)))
    return {"root": "array", "levels": levels, "mesh": mesh, "points": total, "rows": rows,
            "legend": [("inner cell", "inner", f"{na} x {ppc}"), ("outer cell", "outer_0", f"{na * n_out} x {ppc}")],
            "colors": (raw.get("plotting") or {}).get("colors") or {}, "name": (raw.get("general") or {}).get("name", ""),
            "look": look_of(raw)}


def _vesicle(raw: dict, op: dict, live: dict | None = None) -> dict | None:
    """`seed_positions[vesicle]` closes `outer` cells into a shell around `inner` cells (a core), once per entry of `vesicles` (one when
    the spec lists none). The spec declares 21 flat cells; the entity it means is one vesicle: an inner cell with 20 outer cells around it."""
    import ast as _ast
    sets = raw.get("sets") or {}
    cs, ps = op.get("cell_set"), op.get("at")
    if cs not in sets or ps not in sets:
        return None
    types = sets[cs].get("types") or {}
    n_in = int((types.get("inner") or {}).get("count", 0))
    n_out = sum(int((t or {}).get("count", 0)) for k, t in types.items() if k != "inner")
    # `vesicles` lists one entry PER CELL TYPE (the inner type with `core: True`, the outer type as the shell), each with its own centre,
    # tension and polarity: together they are ONE vesicle -- a core cell in a closed shell -- unless the spec's centres differ.
    ves = op.get("vesicles")
    if isinstance(ves, str):
        try:
            ves = _ast.literal_eval(ves)
        except (ValueError, SyntaxError):
            ves = None
    centres = {tuple(e.get("centre") or ()) for e in (ves if isinstance(ves, (list, tuple)) else []) if isinstance(e, dict)}
    nv = max(len(centres), 1)
    if live is not None:
        return _vesicle_at(raw, op, live) if nv == 1 else None
    ni, no = n_in // nv, n_out // nv
    ppc = int(sets[ps].get("per_parent") or op.get("live_per_cell") or 0)
    levels = [[], [], []] if nv > 1 else [[], []]
    total = (n_in + n_out) * ppc

    def node(name, parent, entity, count, points, level, **kw):
        nd = {"name": name, "entity": entity, "parent": parent, "count": count, "declared": count, "per_parent": None,
              "points": points, "label": kw.pop("label", None), **kw}
        levels[level].append(nd)
        return nd

    rows = []
    if nv > 1:
        node("vesicles", None, "array", 1, total, 0, label=f"{nv} vesicles", layout="grid", grid=(nv, 1), grid_ratio=1.0)
        rows.append((f"vesicles  {nv}", "1", "", fmt(total)))
        for j in range(nv):
            node(f"vesicle {j + 1}", "vesicles", "vesicle", 1, (ni + no) * ppc, 1, label=f"vesicle {j + 1}", layout="rosette" if ni == 1 else None)
            rows.append((f"  vesicle {j + 1}", "1", "", fmt((ni + no) * ppc)))
        parents = [f"vesicle {j + 1}" for j in range(nv)]
        lvl = 2
    else:
        node("vesicle", None, "vesicle", 1, total, 0, label="vesicle", layout="rosette" if ni == 1 else None)
        rows.append(("vesicle", "1", "", fmt(total)))
        parents, lvl = ["vesicle"], 1
    for j, par in enumerate(parents):
        for m in range(ni):
            node(f"inner {j + 1}.{m + 1}", par, "cell", 1, ppc, lvl, color_key="inner", label="inner cell")
        for m in range(no):
            node(f"outer {j + 1}.{m + 1}", par, "cell", 1, ppc, lvl, color_key="outer", label="outer cell")
    rows.append(("  inner cell", fmt(n_in), f"{ni} per vesicle", fmt(n_in * ppc)))
    rows.append(("  outer cell", fmt(n_out), f"{no} per vesicle", fmt(n_out * ppc)))
    rows.append((f"  {ps}  (points of one cell)", fmt(n_in + n_out), f"{ppc} per cell", fmt(total)))
    mesh = [{"name": nm, "count": int(s_.get("n", 0)), "entity": s_.get("entity") or "", "parent": None}
            for nm, s_ in sets.items() if nm not in (cs, ps) and isinstance(s_, dict)]
    for m_ in mesh:
        rows.append((m_["name"] + " (mesh)", fmt(m_["count"]), "", ""))
    rows.append(("MPM points in all", "", "", fmt(total)))
    return {"root": levels[0][0]["name"], "levels": levels, "mesh": mesh, "points": total, "rows": rows,
            "legend": [("inner cell", "inner", f"{n_in} x {ppc}"), ("outer cell", "outer", f"{n_out} x {ppc}")],
            "colors": (raw.get("plotting") or {}).get("colors") or {}, "name": (raw.get("general") or {}).get("name", ""),
            "look": look_of(raw)}


def _mk(levels, name, parent, entity, count, points, level, **kw):
    nd = {"name": name, "entity": entity, "parent": parent, "count": count, "declared": count, "per_parent": None, "points": points,
          "label": kw.pop("label", None), **kw}
    levels[level].append(nd)
    return nd


def _cell_key(c: dict, colors: dict) -> str | None:
    """The `plotting.colors` key of a live cell: `inner` for the core type, else its own type (`outer_3`), else `outer`."""
    if c["type"] == "inner":
        return "inner"
    for k in (c["type"], "outer", "outer_0"):
        if k in colors:
            return k
    return None


def _entity_lines(raw: dict, live: dict) -> list:
    """(name, count) of every set the run counts, in the order and under the names of the plexus pane's entities table
    (`spec_summary.read_sets`: the set's title when the spec gives one, else its code name)."""
    out = []
    for r in _summary().read_sets(raw):
        if r["name"] in live["counts"]:
            out.append((r["title"] or r["name"], live["counts"][r["name"]]))
    return out


def _bound(live: dict, by_slot: dict) -> list:
    """The junctions between live cells as [(cell name, cell name, bound fraction rho)]."""
    return [(by_slot[a], by_slot[b], r) for a, b, r in live["junctions"] if a in by_slot and b in by_slot]


def _rosette_at(raw, op, live, types, na, ax_n, az_n, asp, labels):
    """The rosettes of `seed_positions[rosette]` as they are at one stored row: the cells that exist (a rosette's inner cells divide,
    so the single core cell of the start becomes a cluster of them), each carrying the material points it holds, and the junctions
    between them. The sets, their names and the totals are the plexus pane's (`_entity_lines`)."""
    colors = (raw.get("plotting") or {}).get("colors") or {}
    levels = [[], [], [], []]
    cells = live["cells"]
    by_arr = {j: [c for c in cells if c["arr"] == j] for j in range(na)}
    jn_arr = {}
    slot_arr = {c["slot"]: c["arr"] for c in cells}
    for a, _, _ in live["junctions"]:
        jn_arr[slot_arr.get(a)] = jn_arr.get(slot_arr.get(a), 0) + 1
    tot = sum(c["pts"] for c in cells)
    _mk(levels, "array", None, "array", len(cells), tot, 0, label=f"{'x'.join(str(int(a)) for a in ([ax_n, az_n] if az_n > 1 else [ax_n]))} array",
        layout="grid", grid=(ax_n, az_n), grid_ratio=float(asp[1]) / float(asp[0]) if float(asp[0]) else 1.0)
    by_slot = {}
    for j in range(na):
        cj = by_arr[j]
        ins, outs = [c for c in cj if c["type"] == "inner"], [c for c in cj if c["type"] != "inner"]
        e = labels[j] if j < len(labels) else (types.get(f"outer_{j}", {}) or {}).get("youngs")
        tag = f"rosette {j + 1}" + (f"  E {e:g}" if isinstance(e, (int, float)) else "")
        sub = f"{len(cj)} cells" + (f", {jn_arr.get(j, 0)} junctions" if live["has_junctions"] else "")
        rn = f"rosette {j + 1}"
        _mk(levels, rn, "array", "rosette", len(cj), sum(c["pts"] for c in cj), 1, label=tag, sub=sub, layout="rosette")
        if len(ins) > 1:                                    # a cluster of inner cells is ONE node of the rosette: its core
            gn = f"inner cells {j + 1}"
            _mk(levels, gn, rn, "cell_group", len(ins), sum(c["pts"] for c in ins), 2, label=f"{len(ins)} inner cells")
            inner_parent, inner_level = gn, 3
        else:
            inner_parent, inner_level = rn, 2
        for c in ins:
            nm = f"cell {c['slot']}"
            by_slot[c["slot"]] = nm
            _mk(levels, nm, inner_parent, "cell", 1, c["pts"], inner_level, color_key=_cell_key(c, colors), label="inner cell", slot=c["slot"])
        for c in outs:
            nm = f"cell {c['slot']}"
            by_slot[c["slot"]] = nm
            _mk(levels, nm, rn, "cell", 1, c["pts"], 2, color_key=_cell_key(c, colors), label="outer cell", slot=c["slot"])
    n_in = sum(1 for c in cells if c["type"] == "inner")
    return {"root": "array", "levels": [lv for lv in levels if lv], "mesh": [], "points": tot, "rows": None,
            "legend": [("inner cell", "inner", fmt(n_in)), ("outer cell", "outer_0", fmt(len(cells) - n_in))],
            "colors": colors, "name": (raw.get("general") or {}).get("name", ""), "look": look_of(raw),
            "junctions": _bound(live, by_slot), "entities": _entity_lines(raw, live)}


def _vesicle_at(raw, op, live):
    """The vesicle of `seed_positions[vesicle]` at one stored row: an inner core (a cell, or the cluster its descendants make) and the
    shell cells round it, each carrying the material points it holds."""
    colors = (raw.get("plotting") or {}).get("colors") or {}
    cells = live["cells"]
    ins, outs = [c for c in cells if c["type"] == "inner"], [c for c in cells if c["type"] != "inner"]
    levels = [[], [], []]
    tot = sum(c["pts"] for c in cells)
    _mk(levels, "vesicle", None, "vesicle", len(cells), tot, 0, label="vesicle", layout="rosette" if len(ins) >= 1 else None)
    by_slot = {}
    if len(ins) > 1:
        _mk(levels, "inner cells", "vesicle", "cell_group", len(ins), sum(c["pts"] for c in ins), 1, label=f"{len(ins)} inner cells")
    for c in ins:
        nm = f"cell {c['slot']}"
        by_slot[c["slot"]] = nm
        _mk(levels, nm, "inner cells" if len(ins) > 1 else "vesicle", "cell", 1, c["pts"], 2 if len(ins) > 1 else 1,
            color_key=_cell_key(c, colors), label="inner cell", slot=c["slot"])
    for c in outs:
        nm = f"cell {c['slot']}"
        by_slot[c["slot"]] = nm
        _mk(levels, nm, "vesicle", "cell", 1, c["pts"], 1, color_key=_cell_key(c, colors), label="outer cell", slot=c["slot"])
    return {"root": "vesicle", "levels": [lv for lv in levels if lv], "mesh": [], "points": tot, "rows": None,
            "legend": [("inner cell", "inner", fmt(len(ins))), ("shell cell", "outer", fmt(len(outs)))] if ins else [],
            "colors": colors, "name": (raw.get("general") or {}).get("name", ""), "look": look_of(raw),
            "junctions": _bound(live, by_slot), "entities": _entity_lines(raw, live)}


_COMPOSERS = {"rosette": _rosette, "vesicle": _vesicle}


def _composer_op(raw: dict) -> dict | None:
    for o in (raw.get("seed") or []):
        if isinstance(o, dict) and o.get("op") == "seed_positions" and o.get("model") in _COMPOSERS:
            return o
    return None


def _composed(raw: dict, live: dict | None = None) -> dict | None:
    o = _composer_op(raw)
    return _COMPOSERS[o["model"]](raw, o, live) if o is not None else None


def run_dir_of(why: str | None) -> str | None:
    """The folder of the run a step's note names (`run    <group>/<name> (...)`, as the `plexus` pane's `resolve` reads it), when
    it holds a trajectory; None for a step that records no run."""
    if not why or not os.path.exists(why):
        return None
    for ln in open(why, errors="replace"):
        head, _, rest = ln.partition(" ")
        tok = rest.strip().split(" ")[0] if rest.strip() else ""
        if head == "run" and tok:
            for base in ("", os.path.join(ROOT, "graphs_data"), ROOT):
                d = os.path.join(base, tok)
                if os.path.exists(os.path.join(d, "trajectory.npz")):
                    return d
    return None


def time_label(raw: dict, k: int, rows: int) -> str:
    """When stored row k of `rows` was, in the spec's own time unit (`general.units.time_s` 60: min, 3600: h), else in frames."""
    g = raw.get("general") or {}
    u = g.get("units") or raw.get("units") or {}
    ts = float(u.get("time_s", 0) or 0)
    unit, div = ("min", 1.0) if abs(ts - 60.0) < 1e-6 else (("h", 1.0) if abs(ts - 3600.0) < 1e-6 else ("frames", None))
    frame = k * float(g.get("n_frames", rows - 1)) / max(rows - 1, 1)
    t = frame * (float(g.get("dt", 1.0)) if div else 1.0)
    return f"t = {t:,.0f} {unit}"


def live_states(run_dir: str | None, raw: dict, rows=(0, -1)) -> list | None:
    """The entities of a run at its first and last stored row (`rows`: other row numbers, or None for every row), one dict per row (None when there is no run, it stores fewer than
    MIN_ROWS rows, or it counts none of the spec's sets).

    {'row', 'rows', 'time', 'counts': {set: how many of it exist}, 'cells': [{slot, type, arr, pts}], 'junctions': [(slot, slot, rho)],
    'has_junctions'}. A set's count is its occupancy flag summed (`<set>__occ`), the same sets the `plexus` pane lists."""
    if not run_dir:
        return None
    import numpy as np
    z = np.load(os.path.join(run_dir, "trajectory.npz"))
    files = set(z.files)
    sets = raw.get("sets") or {}
    occ = {nm: z[f"{nm}__occ"] for nm in sets if f"{nm}__occ" in files}
    if not occ:
        return None
    n_rows = next(iter(occ.values())).shape[0]
    if n_rows < MIN_ROWS:
        return None
    op = _composer_op(raw)
    cs, ps = (op.get("cell_set"), op.get("at")) if op else contained_points(raw)
    jset = next((nm for nm in sets if f"{nm}__ca" in files and f"{nm}__cb" in files and f"{nm}__occ" in files), None)
    types = list((sets.get(cs) or {}).get("types") or {}) if cs else []
    names = [str(x) for x in z[f"{cs}__type_names"]] if cs and f"{cs}__type_names" in files else types
    # each array is read ONCE, then sliced per row (an npz member is decompressed afresh at every access)
    pid_file = next((f"{ps}__{k}" for k in ("pid", "parent") if ps and f"{ps}__{k}" in files), None)     # the owner of each point: per row (pid) or fixed (parent)
    cells_ok = bool(cs and cs in occ and ps in occ and pid_file)
    if cells_ok:
        # a run records each cell's type per row (`node_type_t`) when types change, else once per slot (`node_type`)
        tt_all = z[f"{cs}__node_type_t"] if f"{cs}__node_type_t" in files else (z[f"{cs}__node_type"] if f"{cs}__node_type" in files else None)
        arr_all = z[f"{cs}__arr"] if f"{cs}__arr" in files else None
        pid_all = z[pid_file]
        nuc_b = nucleus_of(raw)
        nuc_all = z[f"{ps}__{nuc_b[0]}"] if nuc_b and f"{ps}__{nuc_b[0]}" in files else None      # 1 on each cell's nuclear points (cell_nucleus / set_material[nucleus])
    if jset:
        ca_all, cb_all = z[f"{jset}__ca"], z[f"{jset}__cb"]
        rho_all = z[f"{jset}__rho"] if f"{jset}__rho" in files else None
    out = []
    for r in (range(n_rows) if rows is None else rows):
        k = r % n_rows
        st = {"row": k, "rows": n_rows, "time": time_label(raw, k, n_rows), "counts": {nm: int(o[k].sum()) for nm, o in occ.items()},
              "cells": [], "junctions": [], "has_junctions": jset is not None}
        if cells_ok:
            ns = occ[cs].shape[1]
            slots = np.flatnonzero(occ[cs][k])
            tt = (tt_all[k] if tt_all is not None and tt_all.ndim == 2 else tt_all) if tt_all is not None else np.zeros(ns, int)
            arr = np.rint(arr_all[k].reshape(-1)).astype(int) if arr_all is not None else np.zeros(ns, int)
            pid = np.rint(pid_all if pid_all.ndim == 1 else pid_all[k].reshape(-1)).astype(int)[occ[ps][k]]
            npts = np.bincount(pid, minlength=ns)
            st["cells"] = [{"slot": int(i), "type": names[int(tt[i])] if int(tt[i]) < len(names) else "?", "arr": int(arr[i]), "pts": int(npts[i])}
                           for i in slots]
            if nuc_all is not None:
                nn = np.bincount(pid, weights=(nuc_all[k].reshape(-1)[occ[ps][k]] > 0.5).astype(float), minlength=ns)
                for c_ in st["cells"]:
                    c_["nuc"] = int(nn[c_["slot"]])
        if jset:
            m = occ[jset][k]
            ca, cb = ca_all[k].reshape(-1)[m], cb_all[k].reshape(-1)[m]
            rho = rho_all[k].reshape(-1)[m] if rho_all is not None else [None] * len(ca)   # None: this run stores no bound fraction
            st["junctions"] = [(int(a), int(b), None if x is None else float(x)) for a, b, x in zip(np.rint(ca), np.rint(cb), rho)]
        out.append(st)
    return out


def nucleus_of(raw: dict):
    """(point block marking the nuclear points, colour, nominal share of a cell's points or None) when the spec gives its cells nuclei, else None. Read from what the spec already says, never from a flag of
    its own: `plotting.contour_bodies.nucleus` (the block the 3-D views draw as a body inside the cell), else a `cell_nucleus` operator (its `label`, its `fraction`) or a `set_material[nucleus]` seed (its `label`)."""
    cb = ((raw.get("plotting") or {}).get("contour_bodies")) or {}
    blk, share = cb.get("nucleus"), None
    for o in (raw.get("operators") or []) + (raw.get("seed") or []):
        if isinstance(o, dict) and (o.get("op") == "cell_nucleus" or (o.get("op") == "set_material" and o.get("model") == "nucleus")):
            blk = blk or o.get("label", "nuc")
            if o.get("op") == "cell_nucleus" and o.get("fraction") is not None and not o.get("fraction_block"):
                share = float(o["fraction"])
    return None if not blk else (str(blk), str(cb.get("nucleus_color", "#3b6fd4")), share)


def hierarchy(raw: dict, live: dict | None = None) -> dict:
    """{'root', 'levels': [[node, ...], ...], 'mesh': [node, ...], 'points': int} for a loaded spec.

    A node is {name, entity, parent, count, per_parent, points}; `count` follows the containment (the parent's count times
    `per_parent`), `points` is the count of a set of material points, else the sum over its children."""
    composed = _composed(raw, live)
    if composed is not None:
        composed["_by_name"] = {nd["name"]: nd for level in composed["levels"] for nd in level}
        return composed
    sets = raw.get("sets") or {}
    # THE PANE'S OWN ROWS: `spec_summary.read_sets` gives each set its declared size (parent x per_parent, a size per parent type) and the
    # size its seed establishes (`n_cells`); `live` -- the sets' occupancy at one stored row of the run -- replaces both.
    rows = {r["name"]: r for r in _summary().read_sets(raw)}
    counts = (live or {}).get("counts", {})

    def count(name):
        return int(rows[name]["n"] or 0)

    parents = {s.get("parent") for s in sets.values() if isinstance(s, dict) and s.get("parent") in sets}
    v = (raw.get("plotting") or {}).get("hierarchy")
    drop = set() if (isinstance(v, dict) and v.get("dormant")) else dormant_of(raw)
    nodes = {}
    pts_sets = mpm_sets(raw)
    for nm, s in sets.items():
        if nm in drop:
            continue
        nodes[nm] = {"name": nm, "entity": s.get("entity") or ("mpm_particle" if nm in pts_sets else ""), "parent": s.get("parent") if s.get("parent") in sets else None,
                     "count": counts[nm] if nm in counts else (rows[nm]["seeded"] or count(nm)), "declared": count(nm),
                     "per_parent": s.get("per_parent"), "points": 0}
    for nd in nodes.values():
        if nd["entity"] == "mpm_particle":
            nd["points"] = nd["count"]
    for nd in nodes.values():
        if nd["entity"] != "mpm_particle":
            nd["points"] = sum(c["count"] for c in nodes.values() if c["parent"] == nd["name"] and c["entity"] == "mpm_particle")
    roots = [nm for nm, nd in nodes.items() if nd["parent"] is None and nm in parents]
    root = roots[0] if roots else next(iter(nodes), None)         # None: a spec with no sets at all (a field-only grid)

    def depth(nm):
        d = 0
        while nodes[nm]["parent"]:
            nm, d = nodes[nm]["parent"], d + 1
        return d

    contained = [nm for nm in nodes if nm == root or depth(nm) > 0]
    levels = []
    for nm in contained:
        d = depth(nm)
        while len(levels) <= d:
            levels.append([])
        levels[d].append(nodes[nm])
    if group_of(raw):
        levels = _group(levels)
    mesh = [nd for nm, nd in nodes.items() if nm not in contained]
    total = sum(nd["count"] for nd in nodes.values() if nd["entity"] == "mpm_particle")
    out = {"root": root, "levels": levels, "mesh": mesh, "points": total,
           "colors": (raw.get("plotting") or {}).get("colors") or {}, "name": (raw.get("general") or {}).get("name", ""),
           "look": look_of(raw)}
    if live:
        out["entities"] = _entity_lines(raw, live)
    h = _unfold(raw, out, live)
    if live and live.get("junctions") and not h.get("junctions"):         # the junction entity is shown whatever the layout (the rosette and the vesicle did, the generic tree did not): between the cells it joins
        by_slot = {nd["slot"]: nd["name"] for level in h["levels"] for nd in level if nd.get("slot") is not None}
        h["junctions"] = _bound(live, by_slot)
    return h


def _instance_types(raw: dict, set_name: str, n: int) -> list:
    """The type of each of a set's n instances in the order the seed lays them (`type_layout: ordered`: each type `count` times); None
    for each when the types do not add up to n."""
    types = (raw["sets"][set_name].get("types") or {})
    names = []
    if isinstance(types, dict):
        for k, t in types.items():
            names += [k] * int(t.get("count", 1) if isinstance(t, dict) else 1)
    return names if len(names) == n else [None] * n


def _unfold(raw: dict, h: dict, live: dict | None) -> dict:
    """A containment root of 2 to UNFOLD_MAX instances (`cell` n: 9, holding `mpm_particle` per_parent: 24000) is drawn as its instances:
    9 circles, each named by its type and holding its own points, under a root named for the run. A run's own per-cell counts
    (`live['cells']`) replace the declared per-parent count. Sets deeper in the tree are copied into every instance with the root's
    count divided out."""
    levels = h["levels"]
    root = levels[0][0] if levels else None
    one = root is not None and root["count"] == 1 and len((live or {}).get("cells") or []) == 1          # one live cell of a set with free slots (a cell before its division) is drawn as its one instance too
    if root is None or len(levels) < 2 or not (one or 1 < root["count"] <= UNFOLD_MAX):
        return h
    n = int(root["count"])
    kids = _kids(h)
    if root["name"] not in kids:
        return h
    cells = (live or {}).get("cells") or []
    cells = cells if len(cells) == n else []
    nuc_b = nucleus_of(raw)
    h["nucleus_color"] = nuc_b[1] if nuc_b else None
    labels = [c["type"] for c in cells] if cells else _instance_types(raw, root["name"], n)
    own = [c for c in kids[root["name"]] if c["entity"] == "mpm_particle"]
    title = h["name"] or root["name"]
    new = [[] for _ in range(len(levels) + 1)]
    _mk(new, title, None, "", 1, 0, 0, label=title)

    def copy(nd, parent, j, level):
        c = dict(nd)
        c.update(name=f"{nd['name']} {j + 1}", parent=parent, per_parent=None)
        c["count"] = c["declared"] = cells[j]["pts"] if cells and own == [nd] else max(1, round(nd["count"] / n))
        new[level].append(c)
        sub = [copy(k, c["name"], j, level + 1) for k in kids.get(nd["name"], [])]
        c["points"] = c["count"] if c["entity"] == "mpm_particle" else sum(k["points"] for k in sub)
        return c

    for j in range(n):
        lab = labels[j] or f"{root['name']} {j + 1}"
        nuc_pts = cells[j].get("nuc") if cells else None
        if nuc_pts is None and nuc_b and nuc_b[2]:                                    # no run to read: the spec's own share of the declared points
            nuc_pts = round(nuc_b[2] * sum(c["count"] for c in own) / n) if own else None
        inst = _mk(new, f"{root['name']} {j + 1}", title, root["entity"], 1, 0, 1, label=lab,
                   color_key=lab if lab in h["colors"] else None, slot=cells[j]["slot"] if cells else j, nuc=nuc_pts)
        inst["points"] = sum(copy(k, inst["name"], j, 2)["points"] for k in kids[root["name"]])
    new[0][0]["points"] = h["points"] = sum(i["points"] for i in new[1])
    h.update(root=title, levels=[lv for lv in new if lv])
    h["_by_name"] = {nd["name"]: nd for lv in h["levels"] for nd in lv}      # for `_colour`: an instance takes the colour of its type
    return h


def _colour(h, nm):
    from matplotlib.colors import to_rgb
    nd = h.get("_by_name", {}).get(nm)
    if nd is not None and nd.get("color_key") in h["colors"]:
        return tuple(to_rgb(h["colors"][nd["color_key"]]))
    c = h["colors"].get(nm)
    if c is None and nm.endswith("_node"):
        c = h["colors"].get(nm[:-5])
    if c is None:
        for k, v in h["colors"].items():
            if nm.startswith(k):
                c = v
    return tuple(to_rgb(c)) if c is not None else (0.62, 0.68, 0.74)


def fmt(n):
    return f"{int(n):,}"


def clean(name: str) -> str:
    """A set's name without the `_node` of a set of material points: its position in the graph says which level it is."""
    return name.replace("_node", "")


def scale_of(hs: list) -> dict:
    """The largest count among the sets of pieces and among the sets of material points, over every hierarchy drawn together,
    so that node areas and edge widths mean the same number in every panel of a montage."""
    nodes = [nd for h in hs for level in h["levels"] for nd in level]
    return {"pieces": max([nd["count"] for nd in nodes if nd["entity"] != "mpm_particle"] or [1]),
            "points": max([nd["count"] for nd in nodes if nd["entity"] == "mpm_particle"] or [1])}


def _node_area(nd, scale):
    cm = scale["points"] if nd["entity"] == "mpm_particle" else scale["pieces"]
    return 20 + 520 * math.sqrt(nd["count"] / cm)


def _edge_width(nd, scale):
    return 0.6 + 5.0 * math.sqrt(max(nd["points"], 0) / scale["points"])


def _order(h):
    """Depth-first: a set, then the sets inside it."""
    kids = {}
    for level in h["levels"][1:]:
        for nd in level:
            kids.setdefault(nd["parent"], []).append(nd)
    out = []

    def walk(nd, d):
        out.append((nd, d))
        for c in kids.get(nd["name"], []):
            walk(c, d + 1)

    walk(h["levels"][0][0], 0)
    return out


def _circular(h: dict, ax, title: str, scale: dict):
    import numpy as np
    lv = h["levels"]
    radii = [0.0] + [1.0 + 0.85 * i for i in range(len(lv) - 1)]
    pos = {}
    root = lv[0][0]["name"]
    pos[root] = (0.0, 0.0, 0.0)
    ring1 = lv[1] if len(lv) > 1 else []
    n1 = max(len(ring1), 1)
    ang = {nd["name"]: math.pi / 2 - 2 * math.pi * i / n1 for i, nd in enumerate(ring1)}
    for nd in ring1:
        a = ang[nd["name"]]
        pos[nd["name"]] = (radii[1] * math.cos(a), radii[1] * math.sin(a), a)
    for d in range(2, len(lv)):
        kids = {}
        for nd in lv[d]:
            kids.setdefault(nd["parent"], []).append(nd)
        for par, ks in kids.items():
            a0 = pos[par][2]
            for j, nd in enumerate(ks):
                a = a0 + (j - (len(ks) - 1) / 2) * 0.16
                pos[nd["name"]] = (radii[d] * math.cos(a), radii[d] * math.sin(a), a)
    for r in radii[1:]:
        t = np.linspace(0, 2 * math.pi, 200)
        ax.plot(r * np.cos(t), r * np.sin(t), color="#222b35", lw=0.8, zorder=0)
    allnodes = [nd for level in lv for nd in level]
    for nd in allnodes:
        if nd["parent"] in pos and nd["name"] in pos:
            x0, y0, _ = pos[nd["parent"]]
            x1, y1, _ = pos[nd["name"]]
            ax.plot([x0, x1], [y0, y1], color=_colour(h, nd["name"]), lw=_edge_width(nd, scale), alpha=0.55, zorder=1,
                    solid_capstyle="round")
    for nd in allnodes:
        x, y, a = pos[nd["name"]]
        ax.scatter([x], [y], s=_node_area(nd, scale), color=_colour(h, nd["name"]), edgecolor=FG, linewidths=0.4, zorder=3)
        if nd["name"] == root:
            ax.text(0, -0.17, f"{clean(nd['name'])} {fmt(nd['count'])}", color=FG, fontsize=12, ha="center", va="top", zorder=4)
            continue
        d = next(i for i, level in enumerate(lv) if nd in level)
        if d != len(lv) - 1:                    # the inner rings are named once, on the outer ring's label, so the centre stays clear
            continue
        lx, ly = (radii[d] + 0.30) * math.cos(a), (radii[d] + 0.30) * math.sin(a)
        rot, ha = math.degrees(a), "left"
        if math.cos(a) < 0:
            rot, ha = rot + 180, "right"
        ax.text(lx, ly, f"{clean(nd.get('label') or nd['name'])}  {fmt(nd['count'])}", color=FG, fontsize=6.0, rotation=rot,
                rotation_mode="anchor", ha=ha, va="center", zorder=4)
    lim = radii[-1] + 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    if title:
        ax.text(-lim, lim, title, color=FG, fontsize=8.5, ha="left", va="top")


def _tree(h: dict, ax, title: str, scale: dict):
    """Root on the left, one column per level of containment, leaves stacked top to bottom, parents at the mean of their children."""
    lv = h["levels"]
    kids = {}
    for level in lv[1:]:
        for nd in level:
            kids.setdefault(nd["parent"], []).append(nd["name"])
    pos, row = {}, [0]

    def place(nm, d):
        if nm not in kids:
            pos[nm] = (d, -row[0])
            row[0] += 1
        else:
            ys = [place(c, d + 1) for c in kids[nm]]
            pos[nm] = (d, sum(ys) / len(ys))
        return pos[nm][1]

    root = lv[0][0]["name"]
    place(root, 0)
    allnodes = [nd for level in lv for nd in level]
    for nd in allnodes:
        if nd["parent"] in pos:
            (x0, y0), (x1, y1) = pos[nd["parent"]], pos[nd["name"]]
            xm = (x0 + x1) / 2
            ax.plot([x0, xm, xm, x1], [y0, y0, y1, y1], color=_colour(h, nd["name"]), lw=_edge_width(nd, scale), alpha=0.55, zorder=1,
                    solid_capstyle="round")
    for nd in allnodes:
        x, y = pos[nd["name"]]
        ax.scatter([x], [y], s=_node_area(nd, scale), color=_colour(h, nd["name"]), edgecolor=FG, linewidths=0.4, zorder=3)
        if nd["name"] in kids and nd["name"] != root:
            continue
        if nd["name"] == root:
            ax.text(x - 0.1, y, f"{clean(nd['name'])} {fmt(nd['count'])}", color=FG, fontsize=12, ha="right", va="center")
        else:
            ax.text(x + 0.08, y, f"{clean(nd.get('label') or nd['name'])}  {fmt(nd['count'])}", color=FG, fontsize=6.0, ha="left", va="center")
    ax.set_xlim(-0.9, len(lv) - 1 + 1.3)
    ax.set_ylim(-row[0] - 0.2, 1.4)
    if title:
        ax.text(-0.9, 1.4, title, color=FG, fontsize=8.5, ha="left", va="top")


# ---- the area looks: a set's share is the square root of its points (so a 250-point centriole is not a hair beside 138,000)
def _w(nd):
    return math.sqrt(max(nd["points"] or nd["count"], 1))


def _kids(h):
    kids = {}
    for level in h["levels"][1:]:
        for nd in level:
            kids.setdefault(nd["parent"], []).append(nd)
    return kids


def _legend(ax, h, x, y0, step):
    """Name, colour and points of every set of pieces, for the looks whose small cells cannot carry a label."""
    for i, (nd, d) in enumerate(_order(h)):
        if d != 1:
            continue
        yy = y0 - (i * 0.0) - len([1 for n2, d2 in _order(h)[:i] if d2 == 1]) * step
        ax.scatter([x], [yy], s=22, color=_colour(h, nd["name"]), edgecolor=FG, linewidths=0.3, transform=ax.transData, zorder=5)
        ax.text(x + step * 0.5, yy, f"{clean(nd.get('label') or nd['name'])}  {fmt(nd['points'] or nd['count'])}", color=FG, fontsize=5.6,
                va="center", ha="left")


def _sunburst(h: dict, ax, title: str, scale: dict):
    from matplotlib.patches import Wedge
    kids = _kids(h)
    root = h["levels"][0][0]
    top = kids.get(root["name"], [])
    tot = sum(_w(nd) for nd in top) or 1.0
    a0 = 90.0
    R0, R1, R2 = 0.58, 1.1, 1.65
    ax.add_patch(Wedge((0, 0), R0, 0, 360, width=R0, facecolor="#3a4654", edgecolor=BG, lw=0.8))
    ax.text(0, 0, f"{clean(root['name'])}\n{fmt(h['points'])}", color=FG, fontsize=11, ha="center", va="center", linespacing=1.15)
    for nd in top:
        span = 360.0 * _w(nd) / tot
        t1, t0 = a0, a0 - span
        c = _colour(h, nd["name"])
        ax.add_patch(Wedge((0, 0), R1, t0, t1, width=R1 - R0 - 0.02, facecolor=c, edgecolor=BG, lw=0.6, alpha=0.95))
        sub = kids.get(nd["name"], [])
        sw = sum(_w(k) for k in sub) or 1.0
        u1 = t1
        for k in sub:
            ks = span * _w(k) / sw
            ax.add_patch(Wedge((0, 0), R2, u1 - ks, u1, width=R2 - R1 - 0.02, facecolor=_colour(h, k["name"]), edgecolor=BG, lw=0.6, alpha=0.6))
            u1 -= ks
        mid = math.radians((t0 + t1) / 2)
        if span >= 5.0:
            rot = math.degrees(mid)
            ha = "left"
            if math.cos(mid) < 0:
                rot, ha = rot + 180, "right"
            r = R2 + 0.06
            ax.text(r * math.cos(mid), r * math.sin(mid), f"{clean(nd['name'])}  {fmt(nd['points'] or nd['count'])}", color=FG, fontsize=5.8,
                    rotation=rot, rotation_mode="anchor", ha=ha, va="center")
        a0 = t0
    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    if title:
        ax.text(-2.7, 2.5, title, color=FG, fontsize=8.5, ha="left", va="top")


def _squarify(items, x, y, w, h):
    """Squarified treemap (Bruls et al.): {key: (x, y, w, h)} for [(key, weight), ...] sorted by decreasing weight."""
    out = {}
    tot = sum(wt for _, wt in items) or 1.0
    k = w * h / tot
    items = [(key, wt * k) for key, wt in items]

    def worst(row, side):
        sm = sum(a for _, a in row)
        return max(max(side * side * a / (sm * sm), sm * sm / (side * side * a)) for _, a in row)

    while items:
        side = min(w, h)
        row, i = [items[0]], 1
        while i < len(items) and worst(row + [items[i]], side) <= worst(row, side):
            row.append(items[i])
            i += 1
        sm = sum(a for _, a in row)
        if w >= h:
            cw, yy = sm / h, y
            for key, a in row:
                out[key] = (x, yy, cw, a / cw)
                yy += a / cw
            x, w = x + cw, w - cw
        else:
            rh, xx = sm / w, x
            for key, a in row:
                out[key] = (xx, y, a / rh, rh)
                xx += a / rh
            y, h = y + rh, h - rh
        items = items[i:]
    return out


def _treemap(h: dict, ax, title: str, scale: dict):
    from matplotlib.patches import Rectangle
    kids = _kids(h)
    root = h["levels"][0][0]
    top = sorted(kids.get(root["name"], []), key=_w, reverse=True)
    W, H = 1.6, 1.0
    rects = _squarify([(nd["name"], _w(nd)) for nd in top], 0, 0, W, H)
    for nd in top:
        x, y, w, hh = rects[nd["name"]]
        c = _colour(h, nd["name"])
        ax.add_patch(Rectangle((x, y), w, hh, facecolor=c, edgecolor=BG, lw=1.4, alpha=0.5))
        sub = kids.get(nd["name"], [])
        pad = min(w, hh) * 0.12
        if sub and w > 2.4 * pad and hh > 2.4 * pad:
            cr = _squarify([(k["name"], _w(k)) for k in sorted(sub, key=_w, reverse=True)], x + pad, y + pad, w - 2 * pad, hh - 2 * pad)
            for k in sub:
                kx, ky, kw, kh = cr[k["name"]]
                ax.add_patch(Rectangle((kx, ky), kw, kh, facecolor=_colour(h, k["name"]), edgecolor=BG, lw=0.8, alpha=0.95))
        if w * hh > 0.012:
            ax.text(x + 0.012, y + hh - 0.012, f"{clean(nd['name'])}\n{fmt(nd['points'] or nd['count'])}", color=FG, fontsize=5.8, ha="left", va="top")
    ax.set_xlim(-0.02, W + 0.02)
    ax.set_ylim(-0.02, H + 0.12)
    ax.set_aspect("equal")
    ax.text(0, H + 0.1, title, color=FG, fontsize=8.5, ha="left", va="top")
    ax.text(W, H + 0.1, f"{clean(root['name'])}  {fmt(h['points'])}", color=FG, fontsize=12, ha="right", va="top")


def _greedy(radii, pad=0.03):
    """Centres for circles of the given radii (largest first), each as near the middle as it can sit without touching another.
    The search outward starts a little inside where the previous circle was found: the cluster only grows."""
    import numpy as np
    th = np.linspace(0, 2 * math.pi, 90, endpoint=False)
    c, s_ = np.cos(th), np.sin(th)
    px, py, pr = [], [], []
    placed, last = [], 0.0
    for r in radii:
        if not placed:
            placed.append((0.0, 0.0, r))
            px.append(0.0), py.append(0.0), pr.append(r)
            continue
        ax_, ay_, ar_ = np.array(px), np.array(py), np.array(pr)
        for rho in np.arange(max(0.0, last - 2.0 * r), 60.0, 0.04):
            X, Y = rho * c, rho * s_
            ok = (np.hypot(X[:, None] - ax_[None, :], Y[:, None] - ay_[None, :]) >= (r + ar_ + pad)[None, :]).all(axis=1)
            if ok.any():
                i = int(np.argmax(ok))
                placed.append((float(X[i]), float(Y[i]), r))
                px.append(float(X[i])), py.append(float(Y[i])), pr.append(r)
                last = float(rho)
                break
    return placed


def _pack(nd, kids, wmax):
    """{'nd', 'r', 'kids': [(layout, x, y)]}: a leaf is a circle of area following its weight; a set with several children, or
    children that have children, is a circle enclosing them (a rosette: its first child in the middle, the others on a ring)."""
    ch = kids.get(nd["name"], [])
    inner = [c for c in ch if kids.get(c["name"]) or c["entity"] != "mpm_particle"]
    if not ch or (len(ch) == 1 and not inner):
        return {"nd": nd, "r": 0.12 + 0.88 * math.sqrt(_w(nd) / wmax), "kids": []}
    ordered = nd.get("layout") in ("rosette", "grid")                    # these keep the order the seed lays the cells in
    subs = [_pack(c, kids, wmax) for c in (ch if ordered else sorted(ch, key=_w, reverse=True))]
    if nd.get("layout") == "grid" and nd.get("grid"):
        # THE ARRAYS ON THEIR GRID, equidistant, as the seed lays them (array a at column a % ax_n, row a // ax_n, rows upward), so the
        # graph reads like the movie: one pitch from the widest rosette, the rows' pitch in the ratio of the spec's `array_spacing`
        ax_n, az_n = nd["grid"]
        Rm = max(s_["r"] for s_ in subs)
        px = 2 * Rm + 0.12
        py = (px + (0.045 * Rm + 0.35 if any(s_["nd"].get("sub") for s_ in subs) else 0.0)) * float(nd.get("grid_ratio") or 1.0)
        pos = [((a % ax_n - 0.5 * (ax_n - 1)) * px, (a // ax_n - 0.5 * (az_n - 1)) * py) for a in range(len(subs))]
        R = max(math.hypot(x, y) + sb["r"] for (x, y), sb in zip(pos, subs)) + 0.08
        return {"nd": nd, "r": R, "kids": [(sb, x, y) for sb, (x, y) in zip(subs, pos)]}
    if nd.get("layout") == "rosette" and len(subs) > 1:
        r0, ro = subs[0]["r"], max(s_["r"] for s_ in subs[1:])
        n = len(subs) - 1
        rho = max(r0 + ro + 0.04, (ro + 0.02) / math.sin(math.pi / n)) if n > 1 else r0 + ro + 0.04
        pos = [(0.0, 0.0)] + [(rho * math.cos(math.pi / 2 - 2 * math.pi * k / n), rho * math.sin(math.pi / 2 - 2 * math.pi * k / n)) for k in range(n)]
        R = rho + ro + 0.05
    else:
        placed = _greedy([s_["r"] for s_ in subs])
        pos = [(x, y) for x, y, _ in placed]
        R = max(math.hypot(x, y) + r for x, y, r in placed) + 0.06
    return {"nd": nd, "r": R, "kids": [(sb, x, y) for sb, (x, y) in zip(subs, pos)]}


def _draw_pack(h, ax, lay, x, y, depth, R_root, label_leaves=True, pos=None):
    from matplotlib.patches import Circle
    nd, r = lay["nd"], lay["r"]
    c = _colour(h, nd["name"])
    if not lay["kids"]:
        foc, alpha = h.get("focus"), 0.95
        if foc:                                         # the pair a close-up shows: the first in its colour, the second paler, the rest dimmed
            if nd["name"] == foc[1]:
                c = tuple(0.5 * v + 0.5 for v in c)
            elif nd["name"] != foc[0]:
                alpha = 0.2
        ax.add_patch(Circle((x, y), r, facecolor=c, edgecolor=BG, lw=0.6 if r < 0.5 else 0.8, alpha=alpha))
        nuc = nd.get("nuc")
        if nuc and (nd.get("points") or 0) > 0:                                 # its nucleus: a disc whose area is the cell's nuclear share of its points, in the middle
            ax.add_patch(Circle((x, y), r * math.sqrt(min(1.0, nuc / nd["points"])), facecolor=h.get("nucleus_color") or "#3b6fd4", edgecolor=BG, lw=0.5, alpha=alpha, zorder=3))
        if pos is not None:
            pos[nd["name"]] = (x, y)
        if foc and nd["name"] in foc:
            ax.text(x, y, f"{clean(nd['name'])}\n{fmt(nd['points'] or nd['count'])}", color="#0f1319", fontsize=_fs(7.5), ha="center", va="center", zorder=10)
        elif r > 0.3 and depth <= 1 and label_leaves and not foc:
            ax.text(x, y, f"{clean(nd.get('label') or nd['name'])}\n{fmt(nd['points'] or nd['count'])}" + (f"\nnucleus {fmt(nuc)}" if nuc else ""), color=FG if nuc else "#0f1319",
                    fontsize=_fs(5.4 if nuc else 5.8), ha="center", va="center", zorder=10)
        return
    ax.add_patch(Circle((x, y), r, facecolor=c, edgecolor=RING, lw=0.7, alpha=0.30 if depth else 0.5))
    for sb, dx, dy in lay["kids"]:
        _draw_pack(h, ax, sb, x + dx, y + dy, depth + 1, R_root, label_leaves, pos)
    if depth == 1:
        ax.text(x, y - r - 0.04, clean(nd.get("label") or nd["name"]) + (f"\n{nd['sub']}" if nd.get("sub") else ""), color=FG, fontsize=_fs(5.8),
                ha="center", va="top", linespacing=1.15)


def _wmax_of(h):
    """The weight of the biggest leaf: a leaf's radius is its share of it."""
    kids = _kids(h)
    top = kids.get(h["levels"][0][0]["name"], [])
    leaves = [nd for level in h["levels"] for nd in level if not kids.get(nd["name"])]
    wmax = max([_w(nd) for nd in leaves] or [1.0])
    return max(wmax, max([_w(nd) for nd in top] or [1.0])) if not any(kids.get(t["name"]) for t in top) else wmax


def _circlepack(h: dict, ax, title: str, scale: dict):
    """`scale` may carry `wmax` (the leaf weight both panels of a start/end pair take as their unit, so a circle's area means the same
    number of points in each) and `legend: False` (the pair draws one key under both panels, not a list beside each, and a grid of
    rosettes is drawn bare: its frame is the grid, not a circle round it)."""
    kids = _kids(h)
    root = h["levels"][0][0]
    top = kids.get(root["name"], [])
    lay = _pack(root, kids, scale.get("wmax") or _wmax_of(h))
    if not lay["kids"]:
        return
    R = lay["r"]
    pair = not scale.get("legend", True)
    bare = pair and root.get("layout") == "grid"
    if not bare:
        ax.add_patch(__import__("matplotlib").patches.Circle((0, 0), R, facecolor=FILL, edgecolor=RING, lw=0.8))
    pos = {}
    for sb, dx, dy in lay["kids"]:
        _draw_pack(h, ax, sb, dx, dy, 1, R, len(top) <= 14, pos)       # a ring of twenty cells is not named cell by cell
    jn = [j for j in (h.get("junctions") or []) if j[0] in pos and j[1] in pos]
    dense = min(1.0, 3.0 * len(pos) / max(len(jn), 1))                  # more than 3 junctions per cell on average: fade them so the net can be read
    foc = h.get("focus")
    for na, nb, rho in jn:                                              # the junctions: a line between the two cells, wider where more bound
        q = None if rho is None else min(max(rho, 0.0), 1.0)
        if foc and {na, nb} == set(foc):                                # the pair's own junction: yellow, the colour of its tubes in the close-up
            mid = ((pos[na][0] + pos[nb][0]) / 2.0, (pos[na][1] + pos[nb][1]) / 2.0); dist = math.hypot(pos[na][0] - pos[nb][0], pos[na][1] - pos[nb][1])
            frac = h.get("focus_frac", 0.3 if q is None else q)                                                                                          # the junction ENTITY: a yellow disc where the two cells touch, its area as the number of tethered point pairs
            ax.add_patch(__import__("matplotlib").patches.Circle(mid, 0.5 * dist * (0.24 + 0.20 * math.sqrt(max(frac, 0.0))), facecolor="#f2c230", edgecolor="none", zorder=9))
            if h.get("focus_n") is not None:
                ax.text(mid[0], mid[1], f"junction\n{fmt(h['focus_n'])}", color="#1a1a1a", fontsize=_fs(5.2), ha="center", va="center", zorder=10, linespacing=1.0)
            continue
        ax.plot([pos[na][0], pos[nb][0]], [pos[na][1], pos[nb][1]], color="#f4f6f8", lw=(0.5 if q is None else 0.3 + 1.4 * q) * dense ** 0.5 * (0.5 if foc else 1.0),
                alpha=(0.28 if q is None else 0.18 + 0.6 * q) * dense * (0.25 if foc else 1.0), solid_capstyle="round", zorder=6)
    if not pair:
        ax.text(0, R + 0.10, f"{clean(root.get('label') or root['name'])}  {fmt(h['points'])}", color=FG, fontsize=_fs(12), ha="center", va="bottom")
    if not pair:
        # a list beside the pack, compact: the sets of the first level, then the colour of the cells inside them
        lx = R + 0.12 * R + 0.3                 # a clear gap between the pack and its legend
        items = ([(clean(nd.get("label") or nd["name"]), _colour(h, nd["name"]), fmt(nd["points"] or nd["count"])) for nd in top]
                 if len(top) <= 14 else [])
        for nm, ck, txt in h.get("legend") or []:
            items.append((nm, __import__("matplotlib").colors.to_rgb(h["colors"].get(ck, "#9aa5b1")), txt))
        step = R / 9.5
        for i, (nm, col, txt) in enumerate(items):
            yy = R - 0.12 - i * step
            ax.scatter([lx], [yy], s=18, color=col, edgecolor=FG, linewidths=0.3)
            ax.text(lx + 0.03 * R + 0.06, yy, f"{nm}  {txt}", color=FG, fontsize=_fs(5.6), va="center", ha="left")
        ax.set_xlim(-R - 0.05, lx + 0.95 * R + 0.3)
        ax.set_ylim(-R - 0.1, R + 0.5)
        ax.set_aspect("equal")
        ax.text(-R, R + 0.42, title, color=FG, fontsize=_fs(8.5), ha="left", va="top")
        return
    ex = max(abs(dx) + sb["r"] for sb, dx, dy in lay["kids"]) + 0.1 if bare else R + 0.1
    ey = max(abs(dy) + sb["r"] for sb, dx, dy in lay["kids"]) + 0.1 if bare else R + 0.1
    if scale.get("extent"):                                             # a movie keeps one frame for every row
        ex = ey = scale["extent"]
    sub = any(nd.get("sub") for nd in top)
    ax.set_xlim(-ex, ex)
    ax.set_ylim(-ey - ((0.1 * ey + 0.5) if sub else 0.0), ey + 0.1 * ey + 0.3)
    ax.set_aspect("equal")
    ax.text(0.01, 0.995, title, color=FG, fontsize=_fs(8.5), ha="left", va="top", transform=ax.transAxes, linespacing=1.3)


def _icicle(h: dict, ax, title: str, scale: dict):
    from matplotlib.patches import Rectangle
    kids = _kids(h)
    root = h["levels"][0][0]
    top = kids.get(root["name"], [])
    tot = sum(_w(nd) for nd in top) or 1.0
    W, rh = 3.2, 0.5
    ax.add_patch(Rectangle((0, 0), W, rh, facecolor="#3a4654", edgecolor=BG, lw=0.8))
    ax.text(W / 2, rh / 2, f"{clean(root['name'])}   {fmt(h['points'])} MPM points", color=FG, fontsize=12, ha="center", va="center")
    x = 0.0
    for nd in top:
        w = W * _w(nd) / tot
        ax.add_patch(Rectangle((x, -rh - 0.02), w, rh, facecolor=_colour(h, nd["name"]), edgecolor=BG, lw=0.6))
        sub = kids.get(nd["name"], [])
        sw = sum(_w(k) for k in sub) or 1.0
        kx = x
        for k in sub:
            kw = w * _w(k) / sw
            ax.add_patch(Rectangle((kx, -2 * rh - 0.04), kw, rh, facecolor=_colour(h, k["name"]), edgecolor=BG, lw=0.6, alpha=0.6))
            kx += kw
        if w > 0.075:
            ax.text(x + w / 2, -rh / 2 - 0.02, f"{clean(nd['name'])}", color="#0f1319", fontsize=5.6, rotation=90, ha="center", va="center")
            ax.text(x + w / 2, -1.5 * rh - 0.04, fmt(nd["points"] or nd["count"]), color=FG, fontsize=5.4, rotation=90, ha="center", va="center")
        x += w
    ax.set_xlim(-0.05, W + 0.05)
    ax.set_ylim(-2 * rh - 0.1, rh + 0.35)
    ax.text(0, rh + 0.32, title, color=FG, fontsize=8.5, ha="left", va="top")


_DRAW = {"circular": _circular, "tree": _tree, "sunburst": _sunburst, "treemap": _treemap, "circlepack": _circlepack, "icicle": _icicle}


def draw(h: dict, ax_graph, ax_table, title: str = "", look: str | None = None, scale: dict | None = None):
    for ax in (ax_graph, ax_table):
        if ax is None:
            continue
        ax.set_facecolor(BG)
        ax.set_axis_off()
    look = look or h.get("look", DEFAULT_LOOK)
    if look not in LOOKS:
        raise ValueError(f"hierarchy look must be one of {LOOKS}, got {look!r}")
    scale = scale or scale_of([h])
    _DRAW[look](h, ax_graph, title, scale)
    if ax_table is None:
        return
    lv = h["levels"]
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    rows = [("", "count", "per parent", "MPM points")]
    if h.get("rows"):
        rows += h["rows"]
    else:
        for nd, d in _order(h):
            pp = "" if nd["per_parent"] is None else fmt(nd["per_parent"])
            rows.append(("  " * d + clean(nd.get("label") or nd["name"]), fmt(nd["count"]), pp, fmt(nd["points"]) if nd["points"] else ""))
        for nd in h["mesh"]:
            rows.append((nd["name"] + " (mesh)", fmt(nd["count"]), "", ""))
        rows.append(("MPM points in all", "", "", fmt(h["points"])))
    top, step = 0.985, min(0.034, 0.95 / (len(rows) + 1))
    xs = (0.0, 0.50, 0.70, 1.0)
    for i, row in enumerate(rows):
        y = top - i * step
        last = i == len(rows) - 1
        head = i == 0
        for j, txt in enumerate(row):
            ax_table.text(xs[j], y, txt, color=(FG if (j == 0 and not head) or last else DIM), fontsize=5.2,
                          fontweight="normal", ha="left" if j == 0 else "right", va="top",
                          family="monospace")
        if head or i == len(rows) - 2:
            ax_table.plot([0, 1], [y - step * 1.02, y - step * 1.02], color="#2f3945", lw=0.6)


def montage(specs: list, out: str, ncols: int = 2, look: str | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(specs)
    nrows = math.ceil(n / ncols)
    fig = plt.figure(figsize=(11.2 * ncols, 6.2 * nrows), facecolor=BG)
    gs = fig.add_gridspec(nrows, ncols, wspace=0.03, hspace=0.06, left=0.01, right=0.99, top=0.99, bottom=0.01)
    hs = [hierarchy(yaml.safe_load(open(path))) for _, path in specs]
    scale = scale_of(hs)
    for k, ((label, path), h) in enumerate(zip(specs, hs)):
        sub = gs[k // ncols, k % ncols].subgridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.02)
        axg, axt = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        draw(h, axg, axt, f"{label}  {h['name']}  --  {fmt(h['points'])} MPM points", look, scale)
    fig.savefig(out, dpi=110, facecolor=BG)
    plt.close(fig)


def gallery(path: str, out: str, ncols: int = 3):
    """Every look of one spec side by side (graph only: the table does not depend on the look), to choose between them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = hierarchy(yaml.safe_load(open(path)))
    scale = scale_of([h])
    nrows = math.ceil(len(LOOKS) / ncols)
    fig = plt.figure(figsize=(7.2 * ncols, 6.0 * nrows), facecolor=BG)
    gs = fig.add_gridspec(nrows, ncols, wspace=0.02, hspace=0.04, left=0.005, right=0.995, top=0.995, bottom=0.005)
    for k, look in enumerate(LOOKS):
        ax = fig.add_subplot(gs[k // ncols, k % ncols])
        ax.set_facecolor(BG)
        ax.set_axis_off()
        _DRAW[look](h, ax, f"plotting.hierarchy: {look}", scale)
    fig.savefig(out, dpi=110, facecolor=BG)
    plt.close(fig)


def table_of(raw: dict) -> bool:
    """`plotting.hierarchy: {table: true}` draws the table of counts beside the graph; the watcher's pane draws the graph alone."""
    v = (raw.get("plotting") or {}).get("hierarchy")
    return bool(v.get("table", False)) if isinstance(v, dict) else False


def _png(fig, dpi):
    import io
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=BG)
    return buf.getvalue()


def _signature(h: dict) -> tuple:
    """What a hierarchy panel shows, as a value: each node's name, count, points, nucleus and label, the entity lines and the junctions. Two
    panels with equal signatures are the same picture."""
    return (tuple((nd["name"], nd["count"], nd["points"], nd.get("nuc"), nd.get("label")) for lv in h["levels"] for nd in lv),
            tuple(map(tuple, h.get("entities") or ())),
            tuple((a, b, None if r is None else round(r, 3)) for a, b, r in (h.get("junctions") or ())))


def pair_png(raw: dict, states: list, look: str | None = None, width_in: float = 12.4, height_in: float = 6.6, dpi: int = 100):
    """A movie's hierarchy as TWO panels, its first stored row and its last (ONE panel, the first, when the two are the same picture), drawn from the run's own occupancy: the entities of the
    `plexus` pane (cells, material points, junctions) counted at each end under the pane's names. Both panels share one scale (a cell
    is the same size in each), so the growth reads as growth. A key under them names the colours and the junction lines."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import to_rgb
    from matplotlib.figure import Figure
    hs = [hierarchy(raw, st) for st in states]
    if len(hs[0]["levels"]) < 2:
        return None
    look = look or hs[0].get("look", DEFAULT_LOOK)
    wmax = max(_wmax_of(h) for h in hs)
    scale = dict(scale_of(hs), wmax=wmax, legend=False)
    fig = Figure(figsize=(width_in, height_in), facecolor=BG)
    FigureCanvasAgg(fig)
    # NO DIFFERENCE BETWEEN START AND END, ONE PANEL: two identical pictures say it twice (Cedric, 2026-11-05). The one panel sits in the middle at
    # the size of one of the pair, so a circle keeps its size across movies.
    same = _signature(hs[0]) == _signature(hs[-1])
    panels = list(zip(hs, states, ("start", "end")))[:1] if same else list(zip(hs, states, ("start", "end")))
    for i, (h, st, when) in enumerate(panels):
        ax = fig.add_axes([0.255 if same else 0.005 + 0.5 * i, 0.075, 0.49, 0.92])
        ax.set_facecolor(BG)
        ax.set_axis_off()
        line = "  \u00b7  ".join(f"{nm} {fmt(n)}" for nm, n in h["entities"]) if h.get("entities") else f"{fmt(h['points'])} MPM points"
        _DRAW[look](h, ax, f"{when}  {st['time']}" + ("  (the end is the same)" if same else "") + f"\n{line}", scale)
    key = fig.add_axes([0.005, 0.005, 0.99, 0.06])
    key.set_facecolor(BG)
    key.set_axis_off()
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    x = 0.01
    for nm, ck, _ in hs[-1].get("legend") or []:
        key.scatter([x], [0.5], s=26, color=to_rgb(hs[-1]["colors"].get(ck, "#9aa5b1")), edgecolor=FG, linewidths=0.3)
        key.text(x + 0.012, 0.5, nm, color=FG, fontsize=7, va="center")
        x += 0.012 + 0.0075 * len(nm) + 0.03
    if any(h.get("junctions") for h in hs):
        key.plot([x - 0.005, x + 0.02], [0.5, 0.5], color="#f4f6f8", lw=1.4, alpha=0.8)
        key.text(x + 0.027, 0.5, "junction between two cells" + (" (wider where more of it is bound)" if any(r is not None for h in hs for _, _, r in h["junctions"]) else ""), color=FG, fontsize=7, va="center")
    key.text(0.99, 0.5, "a circle's area follows the points the cell holds" + ("; the blue disc is its nucleus, its area the nuclear points" if any(nd.get("nuc") for h in hs for lv in h["levels"] for nd in lv) else "")
             + "; each panel is scaled to fill its frame", color=DIM, fontsize=6.5,
             va="center", ha="right")
    return _png(fig, dpi)


SPECIES_PALETTE = ("#4c8fd4", "#5fb85f", "#e08214", "#c25ba6", "#8d6cc0", "#d1b140", "#4fb3b0", "#c4584c")   # a type the spec gives no `plotting.colors` entry


def species_rows(raw: dict) -> list:
    """[(set name, n, [(type, count), ...])] for each set that declares two or more `types`: the species of a spec with no containment.
    A type's count is its `count`, else its `fraction` of the set's n, else an equal share of n."""
    rows = []
    for r in _summary().read_sets(raw):
        s = (raw.get("sets") or {}).get(r["name"]) or {}
        types = s.get("types")
        if r["relation"] or not isinstance(types, (dict, list)) or len(types) < 2:
            continue
        n = int(r["n"] or r["seeded"] or 0)
        out = []
        for k in types:
            t = types[k] if isinstance(types, dict) else {}
            t = t if isinstance(t, dict) else {}
            c = t.get("count")
            c = c if c is not None else (t["fraction"] * n if t.get("fraction") is not None else n / len(types))
            out.append((str(k), int(round(c))))
        rows.append((r["name"], n, out))
    return rows


def species_png(raw: dict, dpi: int = 100, width_in: float = 8.4):
    """The hierarchy pane for a spec with NO containment: each set's species SIDE BY SIDE (a row of circles per set, one circle per
    type, area proportional to the type's count and one area scale for the whole picture, the colour the spec's `plotting.colors`
    names for the type), because nothing is nested and a tree of one root would say nothing. PNG bytes, or None when no set has
    two types."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import to_rgb
    from matplotlib.figure import Figure
    rows = species_rows(raw)
    if not rows:
        return None
    colors = (raw.get("plotting") or {}).get("colors") or {}
    cmax = max(c for _, _, ts in rows for _, c in ts) or 1
    slots = max(len(ts) for _, _, ts in rows)
    row_in = 3.2
    fig = Figure(figsize=(max(width_in, 2.2 * slots), row_in * len(rows)), facecolor=BG)
    FigureCanvasAgg(fig)
    unit_in = fig.get_figwidth() * 0.98 / slots                  # inches per slot, so a circle is round in data units
    yh = row_in / 2 / unit_in
    for k, (name, n, ts) in enumerate(rows):
        ax = fig.add_axes([0.01, 1 - (k + 1) / len(rows), 0.98, 1 / len(rows)])
        ax.set_facecolor(BG)
        ax.set_axis_off()
        ax.set_xlim(0, slots)
        ax.set_ylim(-yh, yh)
        x0 = (slots - len(ts)) / 2                               # a row of fewer types is centred under the widest one
        for j, (tn, c) in enumerate(ts):
            col = to_rgb(colors[tn]) if tn in colors else to_rgb(SPECIES_PALETTE[j % len(SPECIES_PALETTE)])
            ax.add_patch(__import__("matplotlib").patches.Circle((x0 + j + 0.5, 0.05), 0.4 * math.sqrt(c / cmax), facecolor=col, edgecolor=FG, lw=0.6))
            ax.text(x0 + j + 0.5, -0.47, f"{clean(tn)}  {fmt(c)}", color=FG, fontsize=9, ha="center", va="top")
        ax.text(0.01, 0.99, f"{clean(name)}  {fmt(n)}", color=FG, fontsize=9.5, ha="left", va="top", transform=ax.transAxes)
        ax.text(0.99, 0.99, "a circle's area follows its type's count", color=DIM, fontsize=6.5, ha="right", va="top", transform=ax.transAxes)
    return _png(fig, dpi)


def png_bytes(path: str, look: str | None = None, title: str = "", width_in: float | None = None, height_in: float = 6.2, dpi: int = 100,
              why: str | None = None):
    """The watcher's `hierarchy` pane: the graph of one spec as PNG bytes (the species of a spec with no containment drawn side by side, `species_png`; None when no set has two types). The look is the
    spec's `plotting.hierarchy`, circlepack when absent. When the step is a MOVIE -- its note (`why`) names a run that stores at least
    MIN_ROWS rows -- the pane is `pair_png`: the entities at the start of the run beside the entities at its end.
    Drawn with the object API of matplotlib, no pyplot, so a server thread may call it."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    raw = yaml.safe_load(open(path))
    h = hierarchy(raw)
    if len(h["levels"]) < 2:
        return species_png(raw, dpi)                             # nothing nested: the species side by side, or None
    states = live_states(run_dir_of(why), raw)
    if states:
        png = pair_png(raw, states, look, dpi=dpi)
        if png is not None:
            return png
    with_table = table_of(raw)
    fig = Figure(figsize=(width_in or (11.2 if with_table else 8.4), height_in), facecolor=BG)
    FigureCanvasAgg(fig)
    if with_table:
        gs = fig.add_gridspec(1, 1, left=0.01, right=0.99, top=0.99, bottom=0.01).__getitem__(0)
        sub = gs.subgridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.02)
        axg, axt = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    else:
        axg, axt = fig.add_axes([0.005, 0.005, 0.99, 0.99]), None
    draw(h, axg, axt, title or h["name"], look)
    return _png(fig, dpi)


def movie(raw: dict, states: list, out: str, fps: float = 12.0, look: str | None = None, width_in: float = 8.0, height_in: float = 7.4,
          dpi: int = 100, focus: tuple | None = None, pairs: dict | None = None, grey: bool = False) -> str:
    """The hierarchy of a run OVER TIME, as an mp4: one frame per stored row, each drawn as a panel of `pair_png` (the entities of the
    `plexus` pane counted at that row). The leaf weight and the frame's extent are fixed from a sample of rows (first, quarters, last),
    so a circle is the same size throughout and the picture does not zoom. `focus` (two cell slots) highlights the pair a close-up movie
    shows -- the first cell in its colour, the second paler, their junction a yellow disc (an entity of its own, where the two cells touch) as large as the number of tethered point pairs
    between them (`pairs`, {stored row: that number}, the line's width 0.8 to 9.8 points for 0 to the most over the run), every other cell and
    junction dimmed -- and writes their points, those pairs and the junction's bound fraction in the title."""
    if grey:                                  # the 3D movies' frame (black, spec_inside.MOVIE_BG) and text size (11-point VTK text), 900 pixels high
        global BG, FG, RING, FILL, TEXT_PT
        saved = (BG, FG, RING, FILL, TEXT_PT)
        BG, FG, RING, FILL, TEXT_PT = "#000000", "white", "#3a4654", "#1a222b", 19.5
        try:
            return movie(raw, states, out, fps, look, 13.5, 12.5, 72, focus, pairs)
        finally:
            BG, FG, RING, FILL, TEXT_PT = saved
    import subprocess
    import imageio_ffmpeg
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import to_rgb
    from matplotlib.figure import Figure
    import numpy as np
    n = len(states)
    sample = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    hs = {i: hierarchy(raw, states[i]) for i in sample}
    look = look or hs[0].get("look", DEFAULT_LOOK)
    wmax = max(_wmax_of(h) for h in hs.values())
    ext = []
    for h in hs.values():
        lay = _pack(h["levels"][0][0], _kids(h), wmax)
        ext.append(max(max(abs(dx), abs(dy)) + sb["r"] for sb, dx, dy in lay["kids"]) + 0.1 if h["levels"][0][0].get("layout") == "grid" else lay["r"] + 0.1)
    fixed = 1.04 * max(ext)
    scale = dict(scale_of(list(hs.values())), wmax=wmax, legend=False, extent=fixed)
    W, H = int(width_in * dpi) // 2 * 2, int(height_in * dpi) // 2 * 2
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
                             "-framerate", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out],
                            stdin=subprocess.PIPE)
    for st in states:
        h = hierarchy(raw, st)
        extra = ""
        if focus:
            names = [next((nd["name"] for lv in h["levels"] for nd in lv if nd.get("slot") == c), None) for c in focus]
            if all(names):
                h["focus"] = tuple(names)
                pts = {c["slot"]: c["pts"] for c in st["cells"]}
                rho = [r for a_, b_, r in st["junctions"] if {a_, b_} == set(focus) and r is not None]
                tethers = (pairs or {}).get(st["row"], 0)
                h["focus_frac"] = tethers / max(max((pairs or {1: 1}).values()), 1); h["focus_n"] = tethers          # the junction disc: its area follows the tethered point pairs, over the run's most
                extra = (f"cell {focus[0]}: {fmt(pts.get(focus[0], 0))} points  \u00b7  cell {focus[1]}: {fmt(pts.get(focus[1], 0))} points\n"
                         f"junction: {fmt(tethers)} tethered point pairs" + (f"  \u00b7  bound fraction {max(rho):.2f}" if rho else ""))
        fig = Figure(figsize=(W / dpi, H / dpi), facecolor=BG, dpi=dpi)
        canvas = FigureCanvasAgg(fig)
        kh = 0.10 if TEXT_PT else 0.055                      # a black movie's key is two rows of its large text
        ax = fig.add_axes([0.005, kh + 0.015, 0.99, 0.985 - kh - 0.015])
        ax.set_facecolor(BG)
        ax.set_axis_off()
        line = extra or ("  \u00b7  ".join(f"{nm} {fmt(c)}" for nm, c in h["entities"]) if h.get("entities") else f"{fmt(h['points'])} MPM points")
        _DRAW[look](h, ax, f"{st['time']}\n{line}", scale)
        key = fig.add_axes([0.005, 0.005, 0.99, kh])
        y1, y2 = (0.72, 0.25) if TEXT_PT else (0.5, 0.5)
        key.set_facecolor(BG)
        key.set_axis_off()
        key.set_xlim(0, 1)
        key.set_ylim(0, 1)
        x = 0.02
        for nm, ck, _ in ([] if focus else hs[sample[-1]].get("legend") or []):
            key.scatter([x], [y1], s=26, color=to_rgb(h["colors"].get(ck, "#9aa5b1")), edgecolor=FG, linewidths=0.3)
            key.text(x + 0.015, y1, nm, color=FG, fontsize=_fs(7), va="center")
            x += 0.015 + (0.0113 if TEXT_PT else 0.0095) * len(nm) + 0.04
        if focus:
            key.text(0.02, y1, f"dark: cell {focus[0]}   pale: cell {focus[1]}   dim: the other cells", color=FG, fontsize=_fs(7), va="center")
            key.scatter([0.04], [y2], s=90, color="#f2c230", edgecolor="none")
            key.text(0.07, y2, "their junction: a disc as large as the number of tethered point pairs", color=FG, fontsize=_fs(7), va="center")
        elif any(h.get("junctions") for h in hs.values()):
            x = 0.02 if TEXT_PT else x
            key.plot([x - 0.005, x + 0.03], [y2, y2], color="#f4f6f8", lw=1.4, alpha=0.8)
            key.text(x + 0.04, y2, "junction between two cells", color=FG, fontsize=_fs(7), va="center")
        canvas.draw()
        proc.stdin.write(np.asarray(canvas.buffer_rgba()).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="spec.yaml files; `label=path` names a panel")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--gallery", action="store_true", help="draw every look of the first spec, graph only")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--look", choices=LOOKS, default=None, help="override the spec's `plotting.hierarchy`")
    ap.add_argument("--movie", metavar="WHY", default=None, help="the step's why file naming the run: write the hierarchy over time to -o (an .mp4)")
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--black", action="store_true", help="with --movie, the black frame and the text size of the 3D movies, 900 pixels high")
    ap.add_argument("--focus", metavar="A,B", default=None, help="with --movie, highlight the cell pair of these two slots (the pair a tools/spec_interactions.py close-up shows)")
    a = ap.parse_args()
    if a.movie:
        raw = yaml.safe_load(open(a.specs[0]))
        run = run_dir_of(a.movie)
        states = live_states(run, raw, rows=None)
        if not states:
            sys.exit("no run with a stored trajectory of 5 rows or more is named by " + a.movie)
        focus, pairs = (tuple(int(x) for x in a.focus.split(",")), {}) if a.focus else (None, None)
        if focus:                                                       # the close-up's own count: the tethered point pairs of the two cells at each row
            sp = importlib.util.spec_from_file_location("spec_interactions", os.path.join(ROOT, "tools", "spec_interactions.py"))
            SI = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(SI)
            p = SI.params_of(raw)
            for d in SI.read_rows(run, range(states[0]["rows"])):
                if d["live"][focus[0]] and d["live"][focus[1]]:
                    P = SI.pairs_of(d, p)
                    ci, cj = d["cid"][P["i"]], d["cid"][P["j"]]
                    sel = ((ci == focus[0]) & (cj == focus[1])) | ((ci == focus[1]) & (cj == focus[0]))
                    pairs[d["row"]] = int((sel & P["bonded"]).sum())
        print("wrote", movie(raw, states, a.out, a.fps, a.look, focus=focus, pairs=pairs, grey=a.black))
        return
    items = []
    for s in a.specs:
        label, path = (s.split("=", 1) if "=" in s else (os.path.splitext(os.path.basename(s))[0], s))
        items.append((label, path))
    if a.gallery:
        gallery(items[0][1], a.out, a.cols)
    else:
        montage(items, a.out, a.cols, a.look)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
