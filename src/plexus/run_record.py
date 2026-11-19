"""What a run records: the trajectory buffers, and the dict they are assembled into.

    _setup_recording   allocate the decimated buffers (set positions, state blocks, mesh, fields)
    _rec0              trial 0 of a batched block, the one trial that is recorded
    _type_rows         the per-row type column of a set whose types change during the run
    _assemble          the recorded trajectory dict, also used to salvage an interrupted run
                       (discovery_okuda/run_one.py calls it)
"""
from __future__ import annotations

import numpy as np
import torch

from plexus import utils
from plexus.render import _field_colors
from plexus.models.base import Hierarchy
from plexus.schema import Spec


# _setup_recording: Allocate the decimated trajectory buffers -> (rec_index, rec_sets, occ_sets, rec_state, rec_mesh, fstride, rec_fields).
# _unrecorded: The sets whose spec says `record: false` -> set of names.
def _unrecorded(sim) -> set:
    """Sets declared with `record: false`: none of their rows is kept (positions, live mask, state blocks, types). For a set too large to
    store per row and not needed after the run -- the 13 to 70 million material points of a run whose bodies are read from their centres
    alone (exp09 a100k, b10m: 100,000 and 10,000,000 bodies)."""
    return {n for n, s in (getattr(sim, "sets", None) or {}).items() if isinstance(s, dict) and s.get("record", True) is False}


def _setup_recording(sim: Spec, H: Hierarchy):
    """Allocate the DECIMATED trajectory buffers and announce the recording plan.

    The trajectory is strided, not stored every frame, to bound memory and disk: set frames keep
    <= `record_cap` (spec, default 10000), field frames (each a full [C, nx, ny(, nz)] grid) keep
    <= `field_record_cap` (default 256). The stride is 1 when n_frames <= the cap; the FINAL frame is
    always recorded. `save_data` wins over `record_cap`: True = every frame, False = the final frame only.
    Returns (rec_index, rec_sets, occ_sets, rec_state, rec_mesh, fstride, rec_fields)."""
    # `save_data` (True: every frame, False: final frame only) overrides the legacy `record_cap` striding;
    # the size is printed either way
    _sd = getattr(sim, "save_data", None)
    if _sd is False:
        set_cap = 1                                           # only the final frame, as a stub
    elif _sd is True:
        set_cap = sim.n_frames + 1                            # every frame, no striding
    else:
        set_cap = int(getattr(sim, "record_cap", 10000))      # legacy: max recorded SET frames
    sstride = max(1, (sim.n_frames + set_cap) // set_cap)     # 1 if n_frames <= cap, else sub-sample
    rec_ticks = sorted(set(range(0, sim.n_frames + 1, sstride)) | {sim.n_frames})   # ticks recorded (last always in)
    rec_index = {t: i for i, t in enumerate(rec_ticks)}      # tick -> row index in the recording arrays
    n_rec = len(rec_ticks)
    # pos [n_rec, N, D] for spatial sets; a non-spatial set (voltage, ...) records its blocks in `rec_state`
    skip = _unrecorded(sim)                                                                             # `record: false` sets: no buffers at all
    rec_sets = {name: np.zeros((n_rec, lvl.n, H.dim), np.float32)
                for name, lvl in H.levels.items() if "pos" in lvl.state_schema and name not in skip}
    occ_sets = {name: np.zeros((n_rec, lvl.n), bool) for name, lvl in H.levels.items() if name not in skip}   # live mask  [n_rec, N]
    H._type_hist = {}                                                                                   # per-row node_type, typed sets only
    # state/ group: every recorded block except `pos` -> [n_rec, N, width] (e.g. a neuron's voltage / calcium)
    rec_state: dict[str, dict] = {}
    for name, lvl in H.levels.items():
        if name in skip:
            continue
        blocks = [b for b in lvl.state_schema.recorded if b.name != "pos"]
        if blocks:
            rec_state[name] = {b.name: np.zeros((n_rec, lvl.n, b.width), np.float32) for b in blocks}
    field_cap = int(getattr(sim, "field_record_cap", 256))   # fields are large grids -> a tighter, spec-tunable cap
    fstride = max(1, (sim.n_frames + field_cap) // field_cap)
    # half-edge topology, one snapshot per recorded row (a list: nF / Nv change with division and death);
    # `topo_record` still writes its own `hist` until its readers move over
    rec_mesh: dict[str, list] = {name: [] for name, lvl in H.levels.items()
                                 if getattr(lvl, "mesh", None) is not None}
    rec_fields: dict[str, list] = {fn: [] for fn in H.fields}
    if not utils._QUIET:
        print(f"[engine] {sim.n_frames} sim frames -> recording {n_rec} set frames (stride "
              f"{sstride}), fields every {fstride} steps (<= {field_cap})"
              + (f"; NOT recorded (record: false): {sorted(skip)}" if skip else ""), flush=True)
    return rec_index, rec_sets, occ_sets, rec_state, rec_mesh, fstride, rec_fields


# _type_rows: The type column of every recorded row -> int16 [n_rec, n] | None when types never changed.
def _type_rows(hist, lvl):
    """[n_rec, n] int16 of the type column per recorded row, or None when it never differed from the
    final column (the common case: types are static, and a static column is already recorded once)."""
    if not hist or getattr(lvl, "node_type", None) is None:
        return None
    final = lvl.node_type.detach().cpu().numpy().astype(np.int16)
    if all(np.array_equal(v, final) for v in hist.values()):
        return None
    rows = max(hist) + 1
    out = np.tile(final, (rows, 1))
    for ri, v in hist.items():
        out[ri] = v
    return out


# _assemble: The recorded trajectory from the recording buffers -> {sets, fields, world, world_size, name}.
def _assemble(H, sim, rec_sets, occ_sets, rec_state, rec_fields, n_rows=None, rec_mesh=None):
    """The recorded trajectory dict from the recording buffers: per set pos [T, N, D], occ [T, N],
    state blocks, half-edge snapshots, types, parent; plus fields and the world.

    Split out of `run` so an interrupted run can be salvaged: `n_rows` truncates the preallocated
    [n_rec, ...] buffers to the rows actually written (the rest is uninitialised memory)."""
    def _cut(a):
        return a if (n_rows is None or a is None) else a[:max(1, int(n_rows))]

    return {"sets": {name: {"pos": _cut(rec_sets.get(name)), "occ": _cut(occ_sets[name]),
                           # non-spatial recorded blocks (voltage, calcium, ...); None for a pos/vel set
                           "state": ({k: _cut(v) for k, v in rec_state[name].items()}
                                     if rec_state.get(name) else None),
                           # half-edge table, one snapshot per row (a list: rows are ragged); None without a mesh
                           "mesh": _cut((rec_mesh or {}).get(name) or None),
                           "node_type": (H.level(name).node_type.cpu().numpy()
                                         if hasattr(H.level(name), "node_type") else None),
                           "type_names": getattr(H.level(name), "type_names", None),
                           # the per-row type column, only if some row differs from the final one
                           "node_type_t": _type_rows(getattr(H, "_type_hist", {}).get(name), H.level(name)),
                           # containment: parent set + per-node parent index (a plotter renders a container as its merged child cloud)
                           "parent_name": getattr(H.level(name), "parent_name", None),
                           "parent": (H.level(name).parent.cpu().numpy()
                                      if H.level(name).parent.numel() else None)}
                    for name in H.levels if name in occ_sets},      # a `record: false` set is left out
           "fields": {fn: {"grid": np.stack(fr[:n_rows] if n_rows else fr), "colors": _field_colors(H, sim, H.fields[fn]),
                           "world": H.world_width}
                      for fn, fr in rec_fields.items() if fr},
           "world": H.world_width,
           "world_size": H.world_size.cpu().numpy(),   # per-axis box [w0..w_{D-1}] (3D plotter reads it)
           "name": sim.name}


# _rec0: Trial 0 of a batched block -> [N, W].
def _rec0(t: torch.Tensor) -> torch.Tensor:
    """Trial 0 of a batched block [B, N, W]; an unbatched [N, W] unchanged.

    The record is one trial, not all B: every reader (renderers, `test_plot`, fingerprints) expects
    [T, N, dim]. A batch is a training device; its product is the loss, which reaches the caller through `H`."""
    return t[0] if t.dim() == 3 else t


# record_levels: Save this tick's positions, occupancy, extra state blocks, node types and mesh snapshots into row `ri` of the recording buffers -> None.
def record_levels(H, ri: int, rec_sets: dict, occ_sets: dict, rec_state: dict, rec_mesh: dict) -> None:
    """`rec_sets` and `occ_sets` hold one array per recorded set (positions and live mask), `rec_state` the extra recorded blocks,
    `rec_mesh` the mesh snapshots; all are filled in place, and `H._type_hist` keeps the type column per row."""
    for name, lvl in H.levels.items():
        if name not in occ_sets:                          # a `record: false` set (`_unrecorded`)
            continue
        # detach: under grad=True the trajectory is a record, never a gradient path
        if name in rec_sets:                              # spatial: the pos trajectory
            rec_sets[name][ri] = _rec0(lvl.get("pos")).detach().cpu().numpy()
        occ_sets[name][ri] = lvl.active.detach().cpu().numpy()
        # type column per row, kept only when it changed (see the assembly below)
        if getattr(lvl, "node_type", None) is not None and getattr(lvl, "_type_table", None) is not None:
            H._type_hist.setdefault(name, {})[ri] = lvl.node_type.detach().cpu().numpy().astype(np.int16)
        if name in rec_state:                            # non-pos recorded state blocks (voltage, ...)
            for bname, arr in rec_state[name].items():
                arr[ri] = _rec0(lvl.get(bname)).detach().cpu().numpy()
        # record the mesh whenever one exists, even if the spec did not declare `mesh: half_edge`
        # (legacy specs build `_mesh` anyway and used to lose their topology silently)
        _m = getattr(lvl, "mesh", None) or getattr(lvl, "_mesh", None)
        if _m is not None and getattr(_m, "snapshot", None) is not None:
            if name not in rec_mesh:
                rec_mesh[name] = []
                print(f"[engine] set {name!r} has a half-edge mesh but does not declare "
                      f"`mesh: half_edge`; recording it anyway. Add the declaration to "
                      f"the spec -- an undeclared mesh used to be dropped silently.",
                      flush=True)
            rec_mesh[name].append(_m.snapshot())


# record_fields: Append every recorded field's grid to its frame list -> None.
def record_fields(H, rec_fields: dict) -> None:
    """Fields flagged `RECORD = False` (transient scratch such as `mpm_grid`) are skipped."""
    for fn, fld in H.fields.items():
        if not getattr(fld, "RECORD", True):     # transient scratch fields (e.g. mpm_grid) are not recorded
            continue
        rec_fields[fn].append(fld.grid.detach().to("cpu", torch.float32).numpy().copy())
