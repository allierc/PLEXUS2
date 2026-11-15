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
    on_frame = _with_partial_saves(sim, data_dir, save, on_frame)

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

    save_trajectory_and_contact(out, data_dir, save)

    nrec = (next(iter(out["sets"].values()))["occ"].shape[0] if out.get("sets")
            else next(iter(out["fields"].values()))["grid"].shape[0])          # a spec of fields alone has no set to count
    print(f"[generate] done: {nrec} recorded frames -> {data_dir}", flush=True)

    finish_render(sim, out, data_dir, live_movie, _want, save)
    return data_dir, out
