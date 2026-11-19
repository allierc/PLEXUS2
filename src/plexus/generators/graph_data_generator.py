"""Ground-truth data generation: run a spec's forward simulation, save it.

This is the Plexus analogue of connectome-gnn's `data_generate`: given a validated
`Spec`, it builds the Hierarchy, rolls the schedule forward through the
engine, and writes the trajectory dataset (+ a preview) under

    {data_root}/graphs_data/<pre_folder>/<name>/

The saved trajectory is the dataset a future inverse GNN will train on (recover
the operators/parameters from the dynamics). For now generation == the forward
run; the engine stays the interpreter, the generator owns the data layout + viz.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import torch

from plexus.schema import Spec
from plexus.engine import run
from plexus.paths import graphs_data_path
from plexus.generators.live_hooks import build_frame_hooks
from plexus.generators.save_outputs import save_trajectory_and_contact, finish_render


def _with_partial_saves(sim, data_dir, save, on_frame):
    """`general.save_every_min` (wall-clock minutes, 0 = off): every that many minutes write the rows recorded so far to `trajectory.npz` (the end of the run writes the whole file over it),
    with `frame_ms` the wall time between consecutive frames in ms, so a long run can be scored and timed while it goes. Returns `on_frame` unchanged when off."""
    every = float(getattr(sim, "save_every_min", 0.0) or 0.0)
    if not save or every <= 0.0:
        return on_frame
    import time
    from plexus.run_record import _assemble
    from plexus.generators.save_outputs import save_trajectory_and_contact
    st = {"last_save": time.time(), "last_frame": time.time(), "ms": []}

    def hook(H, tick):
        if on_frame is not None:
            on_frame(H, tick)
        now = time.time(); st["ms"].append((now - st["last_frame"]) * 1000.0); st["last_frame"] = now
        if now - st["last_save"] < every * 60.0:
            return
        rows = 1 + max((r for t, r in H._rec_index.items() if t <= H.frame), default=0)
        out = _assemble(H, *H._rec, n_rows=rows, rec_mesh=getattr(H, "_rec_mesh", None))
        out["frame_ms"] = np.asarray(st["ms"], np.float32)
        save_trajectory_and_contact(out, data_dir, True)
        st["last_save"] = time.time()
        print(f"[generate] partial save: {rows} recorded rows (frame {int(tick)} of {sim.n_frames}) -> {data_dir}/trajectory.npz", flush=True)
    return hook


def _chunk_columns(out: dict, r0: int, r1: int) -> dict:
    """Rows r0 .. r1 - 1 of an assembled record, flattened as `trajectory.npz` names them: `<set>__pos`, `<set>__occ` and `<set>__<block>`."""
    flat = {}
    for sname, d in out["sets"].items():
        if d["pos"] is not None:
            flat[f"{sname}__pos"] = d["pos"][r0:r1]
        flat[f"{sname}__occ"] = d["occ"][r0:r1]
        for bname, arr in (d.get("state") or {}).items():
            flat[f"{sname}__{bname}"] = arr[r0:r1]
    return flat


def _chunk_meta(out: dict, sim) -> dict:
    """What does not change from row to row, written once to `chunks/meta.npz`: the world, and per set its types, type names and containment."""
    flat = {"world": out["world"], "world_size": out["world_size"], "n_frames": np.int64(sim.n_frames)}
    for sname, d in out["sets"].items():
        if d.get("node_type") is not None:
            flat[f"{sname}__node_type"] = d["node_type"]
        if d.get("type_names"):
            flat[f"{sname}__type_names"] = np.asarray(list(d["type_names"]))
        if d.get("parent") is not None:
            flat[f"{sname}__parent"] = d["parent"]
            flat[f"{sname}__parent_name"] = np.asarray(d["parent_name"])
    return flat


def _window_rows(H, win: dict):  # noqa: C901
    """The live members of set `win["set"]` whose parent's centre lies within `margin` of the box `win["box"]` = [[x0, y0], [x1, y1]] in x and y: (positions [k, 3] float32,
    parent index [k] int32), this tick."""
    from plexus.run_record import _rec0
    L = H.level(str(win["set"]))
    P = H.level(L.parent_name)
    x = _rec0(L.get("pos"))
    par = L.parent.long()
    pc = _rec0(P.get("pos"))[par]
    (b0x, b0y), (b1x, b1y) = win["box"]
    m = float(win.get("margin", 0.0))
    keep = (L.active > 0.5) & (pc[:, 0] > b0x - m) & (pc[:, 0] < b1x + m) & (pc[:, 1] > b0y - m) & (pc[:, 1] < b1y + m)
    out = (x[keep].detach().cpu().numpy().astype(np.float32), par[keep].cpu().numpy().astype(np.int32))
    e = win.get("edges")                                  # and the live rows of an edge-like set whose two parents are both inside: {set, a, b, keep: [blocks]}
    if not e:
        return out + (None,)
    pin = torch.zeros(P.n, dtype=torch.bool, device=par.device)
    pin[par[keep]] = True
    E = H.level(str(e["set"]))
    a = torch.round(_rec0(E.get(str(e["a"])))[:, 0]).long().clamp(0, P.n - 1)
    b = torch.round(_rec0(E.get(str(e["b"])))[:, 0]).long().clamp(0, P.n - 1)
    ek = (E.active > 0.5) & pin[a] & pin[b]
    cols = {"a": a[ek].cpu().numpy().astype(np.int32), "b": b[ek].cpu().numpy().astype(np.int32)}
    for k in e.get("keep", []):
        cols[k] = _rec0(E.get(str(k)))[ek].detach().cpu().numpy().astype(np.float32)
    return out + (cols,)


def _with_chunk_saves(sim, data_dir, save, on_frame):
    """`general.save_chunk_frames` = N (0 = off): every N frames (and at the last) the rows recorded since the previous chunk are written to
    `<data_dir>/chunks/rows_<first>_<last>.npz` (columns as `trajectory.npz` names them, plus `frames`, the frame of each row, and `frame_ms`, the wall time of every
    frame so far), the world and the sets' types once to `chunks/meta.npz`; `data_generate` then writes no whole trajectory at the end. Each file is written under a
    temporary name and renamed, so a reader never sees half of one. Returns (hook, state) -- state["row0"] is the first row not yet written."""
    chunk = int(getattr(sim, "save_chunk_frames", 0) or 0)
    win = getattr(sim, "save_chunk_window", None)
    st = {"row0": 0, "next": chunk, "ms": [], "last": None, "meta": False, "win": []}
    if not save or chunk <= 0:
        return on_frame, None
    import time
    from plexus.run_record import _assemble
    cdir = os.path.join(data_dir, "chunks")
    os.makedirs(cdir, exist_ok=True)

    def write(H, upto_frame):
        W = int(getattr(H, "warmup", 0) or 0)                 # the record's keys are ticks, frame 0 = tick W (engine.run's warm-up)
        rows = 1 + max((r for t, r in H._rec_index.items() if t <= upto_frame + W), default=-1)
        if rows <= st["row0"]:
            return
        out = _assemble(H, *H._rec, n_rows=rows, rec_mesh=None)
        if not st["meta"]:
            tmp = os.path.join(cdir, "meta.partial.npz")
            np.savez(tmp, **_chunk_meta(out, sim))
            os.replace(tmp, os.path.join(cdir, "meta.npz"))
            st["meta"] = True
        tick_of = sorted(H._rec_index, key=H._rec_index.get)
        r0 = st["row0"]
        name = f"rows_{r0:05d}_{rows - 1:05d}.npz"
        tmp = os.path.join(cdir, name[:-4] + ".partial.npz")
        wcols = {}
        if win:                                          # the window's rows since the last chunk, one per recorded row (`hook` keeps them), concatenated with their offsets
            wr = st["win"][:rows - r0]
            st["win"] = st["win"][rows - r0:]
            wcols = {"window__pos": np.concatenate([q for q, _, _ in wr]) if wr else np.zeros((0, 3), np.float32),
                     "window__parent": np.concatenate([b for _, b, _ in wr]) if wr else np.zeros(0, np.int32),
                     "window__offsets": np.cumsum([0] + [len(b) for _, b, _ in wr]).astype(np.int64), "window__box": np.asarray(win["box"], np.float64)}
            if win.get("edges") and wr:                  # the window's edge rows: window__edge_<column>, window__edge_offsets
                for k in wr[0][2]:
                    wcols[f"window__edge_{k}"] = np.concatenate([c[k] for _, _, c in wr])
                wcols["window__edge_offsets"] = np.cumsum([0] + [len(c["a"]) for _, _, c in wr]).astype(np.int64)
        np.savez(tmp, frames=np.asarray(tick_of[r0:rows], np.int64) - W, frame_ms=np.asarray(st["ms"], np.float32), **_chunk_columns(out, r0, rows), **wcols)
        os.replace(tmp, os.path.join(cdir, name))
        st["row0"] = rows
        print(f"[generate] chunk {name}: rows {r0} to {rows - 1} (frame {int(upto_frame)} of {sim.n_frames}) -> {cdir}", flush=True)

    def hook(H, tick):
        if on_frame is not None:
            on_frame(H, tick)
        now = time.time()
        if st["last"] is not None:
            st["ms"].append((now - st["last"]) * 1000.0)
        st["last"] = now
        if win and (int(tick) + int(getattr(H, "warmup", 0) or 0)) in H._rec_index:         # a recorded row: its window too
            st["win"].append(_window_rows(H, win))
        if int(tick) >= st["next"] or int(tick) >= sim.n_frames:
            write(H, int(tick))
            while st["next"] <= int(tick):
                st["next"] += chunk
    st["write"] = write
    return hook, st


def data_generate(
    sim: Spec,
    pre_folder: str,
    device: str = "cpu",
    erase: bool = False,
    save: bool = True,
    live_every_frac: float | None = 0.05,
    live_movie: dict | None = None,
    on_frame=None,
) -> tuple[str, dict]:
    """forward-simulate `sim` and write its trajectory under
    graphs_data/<pre_folder>/<sim.name>/. Returns (data_dir, out).

    generation writes DATA ONLY -- the trajectory + metadata. Visualization is a
    separate, external concern (plexus.plot, run as `Plexus_Main -o plot`); the
    generator never imports matplotlib, so adding simulations never grows a plot
    switch in here (the ParticleGraph anti-pattern).

    `live_movie` is the ONE EXCEPTION, and it earns it by being the case the separate
    concern cannot serve: at 100 M particles a single recorded frame is 1.2 GB, so
    `plot_dataset` has nothing to read back from. It is an `on_frame` hook -- the same
    extension point `live_every_frac` already uses -- and its pyvista import lives inside
    `plexus.live_movie`, below the branch that decides whether to build one, so this module
    still imports no rendering stack. Pass `None` (what `--no-viz` does) and nothing is
    built, which is how a throughput measurement is taken.

    `on_frame(H, tick)` is a caller's own hook, composed AFTER the movie's: the material page runs
    this very function and reads the hierarchy off it to draw at its own camera and to count what
    is alive. Raising from it aborts the run (the movie is still closed and the frames so far
    kept); `plexus.pipeline.StopRun` is the exception a stop button raises."""
    folder = pre_folder.rstrip("/")
    data_dir = graphs_data_path(folder, sim.name)
    if erase and os.path.isdir(data_dir):
        for entry in os.scandir(data_dir):
            if entry.name.startswith(".nfs"):
                continue
            try:
                shutil.rmtree(entry.path) if entry.is_dir() else os.unlink(entry.path)
            except OSError as e:
                print(f"[generate] --force could not remove {entry.name}: {e}", flush=True)
    os.makedirs(data_dir, exist_ok=True)

    out_path = os.path.join(data_dir, "simulation.zarr") if save else None
    print(f"[generate] {folder}/{sim.name}: {sim.n_frames} frames, "
          f"sets={ {k: int(v.get('n', 0)) for k, v in sim.sets.items() if 'n' in v} } -> {data_dir}",
          flush=True)
    _want = str(((sim.plotting or {}).get("renderer", "") or "")).strip().lower()
    on_frame, movs, live_movie = build_frame_hooks(sim, data_dir, live_movie, live_every_frac,
                                                   on_frame, _want)
    on_frame, _chunks = _with_chunk_saves(sim, data_dir, save, on_frame)
    if _chunks is None:
        on_frame = _with_partial_saves(sim, data_dir, save, on_frame)
    else:                                        # chunked: no zarr and no trajectory.npz at the end (general.save_chunk_frames)
        out_path = None

    # WHICH ENGINE. `general.engine` selects the driver; the default is plexus.engine.run and every
    # existing spec omits the key, so nothing changes for them. A named engine is a MODE OF RUNNING,
    # not a different physics -- `continuous` still delegates to the same loop, and exists to refuse
    # a spec that its operators alone cannot check.
    _engine = str(getattr(sim, "engine", "default") or "default").lower()
    _run = run
    if _engine not in ("", "default"):
        try:
            import importlib
            _run = importlib.import_module(f"plexus.{_engine}_engine").run
            print(f"[engine] using the {_engine!r} engine (plexus.{_engine}_engine.run)", flush=True)
        except Exception as e:                                       # noqa: BLE001
            raise ValueError(f"general.engine: {_engine!r} -- no such engine "
                             f"(expected plexus/{_engine}_engine.py with a run(); {e})") from e
    try:
        H, out = _run(sim, out_path=out_path, device=device, progress=True, on_frame=on_frame)
    finally:
        for _m in movs:
            _m.close()

    if _chunks is not None:
        _chunks["write"](H, sim.n_frames)            # whatever the last chunk did not hold
    save_trajectory_and_contact(out, data_dir, save and _chunks is None)

    nrec = (next(iter(out["sets"].values()))["occ"].shape[0] if out.get("sets")
            else next(iter(out["fields"].values()))["grid"].shape[0])          # a spec of fields alone has no set to count
    print(f"[generate] done: {nrec} recorded frames -> {data_dir}", flush=True)

    finish_render(sim, out, data_dir, live_movie, _want, save and _chunks is None)
    return data_dir, out
