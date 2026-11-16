"""The `-o generate` pipeline as ONE function, so every entry point runs the same run.

`Plexus_Main.py` parses a command line and calls `generate`; the web page's RUN button calls
`generate` in a thread with an `on_frame` hook. Before this module existed the page had its own
loop around `engine.run` -- its own substep, its own wall model, its own frame keeping, no movie,
nothing in `graphs_data/` -- and the same three bodies ran at 202 ms/frame on the page against 68
from the command line. That was not two engines; it was two callers. This is the one caller.

What `generate` does, in order, is exactly what `Plexus_Main.main` did for `-o generate`:

  1. resolve the spec (a config name or a yaml path) and its pre-folder;
  2. the MPM guards that REWRITE the spec file -- the grid CFL (substep against the elastic and
     capillary wave speeds) and particles-per-cell;
  3. `schema.load`, the gatekeeper;
  4. a copy of the spec into `log/<type>/<name>/spec.yaml`;
  5. the VRAM projection warning for captured runs;
  6. `data_generate` with the live movie shaped by the spec's `plotting:` block (`max_frames`,
     `stills`, `keep_stills`) and the CLI defaults where the spec says nothing;
  7. a copy of the spec next to the data, the plot pass and the completion markers.

`StopRun` is what a stop button raises inside `on_frame`; `generate` catches it, closes the movie
and returns what was written so far with `stopped=True`.
"""
from __future__ import annotations

import os
import shutil

# REPO: The repository root.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# StopRun: Raised from an `on_frame` hook to end a run early; not an error.
class StopRun(Exception):
    """Raised from an `on_frame` hook to end a run early; not an error."""


# generate: Run one spec through the generate pipeline -> {name, pre_folder, data_dir, run_log_dir, spec_path, stopped, frame_ms}.
def generate(config_name: str, *, device: str = "cuda:0", force: bool = False, viz: bool = True,
             render_n="500000000", render_max_frames: int = 300, real_time: bool = True,
             keep_stills: bool = False, render_stills: int = 10, render_dot=None,
             plot_after: bool = True, grid: bool = False,
             movie: bool = False, plot: bool = False, on_frame=None, argv_line: str | None = None) -> dict:
    """Run one spec through the generate pipeline. Returns {name, pre_folder, data_dir, run_log_dir,
    spec_path, stopped, frame_ms} -- `frame_ms` is the engine's own per-tick wall clock. `plot_after`: the plot
    pass after the run (the recorded trajectory drawn into its movies and figures); the page's RUN button turns it off."""
    import plexus.operators  # noqa: F401  self-register the operator library
    from plexus.schema import load
    from plexus.paths import resolve_config, validate_pre_folder, log_path
    from plexus.generators.graph_data_generator import data_generate

    if device.startswith("cuda"):                    # fall back to CPU if no GPU is present
        import torch
        if not torch.cuda.is_available():
            print(f"[device] {device} unavailable -> falling back to cpu", flush=True)
            device = "cpu"

    yaml_file, pre_folder, name = resolve_config(config_name)
    validate_pre_folder(pre_folder)
    if not os.path.isfile(yaml_file):
        raise FileNotFoundError(f"config not found: {yaml_file}")
    print(f"task=generate  type={pre_folder.rstrip('/')}  config={name}  ({yaml_file})", flush=True)
    # MPM grid-dt CFL: auto-correct the SPEC (not the engine) so dt_sub respects the Courant condition; idempotent for non-MPM and already-stable specs
    from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition, particles_per_grid_cell
    Courant_Friedrichs_Lewy_condition(yaml_file)
    # the grid's other discretisation constraint, particles per cell (the CFL check bounds the time step; a spec raised to n_grid 192 at a fixed particle count had its snow quietly collapse)
    particles_per_grid_cell(yaml_file)
    # load the whole simulation description read from the YAML: sets, fields, operators, schedule, and the general settings such as name, dimension and world size
    sim = load(yaml_file)

    # self-describing run dir: snapshot the spec into log/<type>/<name>/
    run_log_dir = log_path(pre_folder.rstrip("/"), name)
    os.makedirs(run_log_dir, exist_ok=True)
    shutil.copy2(yaml_file, os.path.join(run_log_dir, "spec.yaml"))

    plot_after = plot_after and viz
    # the mp4 is written by the run itself (a live `on_frame` movie), for runs where reading the trajectory back is impossible (one 100 M-particle frame is 1.2 GB); `plot_dataset` below still renders from recorded data; `viz=False` turns off every renderer
    _dot = (None if render_dot is None else render_dot if render_dot == "auto" else float(render_dot))
    if viz and device.startswith("cuda"):
        _vram_warning(sim, device)
    _rn = [int(x) for x in str(render_n).split(",") if x.strip()]
    # the spec may set the movie's own shape (`max_frames`, `stills`, `keep_stills`), so a benchmark says it once in the file, not on every command line and job script; the CLI values stay the defaults
    _pl = getattr(sim, "plotting", None) or {}
    _units = getattr(sim, "units", None)
    _declared = bool(getattr(_units, "declared", False))
    lm = None if not viz else {"render_n": (_rn if len(_rn) > 1 else _rn[0]),
                               "max_frames": int(_pl.get("max_frames", render_max_frames)),
                               "dot": _dot,
                               "stills": int(_pl.get("stills", render_stills)),
                               "keep_stills": bool(_pl.get("keep_stills", keep_stills)),
                               # the movie can only be timed if the run has a clock
                               "dt": getattr(sim, "dt", None),
                               "time_s": (_units.time_s if _declared else None),
                               "real_time": real_time,
                               "length_um": (_units.length_um if _declared else None),
                               "force_nN": (getattr(_units, "force_nN", None) if _declared else None)}
    stopped = False
    frame_ms = None
    replayed = False
    try:
        data_dir, out = data_generate(sim, pre_folder, device=device, erase=force, save=True,
                                      live_every_frac=(None if not viz else 0.05),
                                      live_movie=lm, on_frame=on_frame)
        frame_ms = out.get("frame_ms") if isinstance(out, dict) else None
        replayed = bool(out.get("replayed")) if isinstance(out, dict) else False
    except StopRun:
        # the run was ended on purpose: the movie was closed by `data_generate`; the trajectory and the markers are not written, as there is no complete run to mark
        from plexus.paths import graphs_data_path
        stopped = True
        data_dir = graphs_data_path(pre_folder.rstrip("/"), sim.name)
        print(f"[generate] stopped early -> {data_dir}", flush=True)
    if not stopped:
        shutil.copy2(yaml_file, os.path.join(data_dir, "spec.yaml"))   # co-locate the spec with its data
        _mark(run_log_dir, "_completed_generate", data_dir)

    # the plot pass: when plotting was asked, or after every run unless `data_generate` just replayed the trajectory into movie.mp4 (the `plotting.curve` re-render)
    if viz and not stopped and (plot or (plot_after and not replayed)):
        from plexus.plot import plot_dataset
        data_dir = plot_dataset(sim, pre_folder, movie=(movie or plot_after))
        if plot:
            _mark(run_log_dir, "_completed_plot", data_dir)

    # optional MLS-MPM grid-diagnostic movie (re-runs the sim to capture F/C/Jp/stress/grid)
    if grid and viz and not stopped:
        from plexus.generators.mpm_grid_diag import generate_grid_movie
        generate_grid_movie(sim, data_dir, device=device)

    if not stopped:
        _mark(run_log_dir, "_complete", argv_line or f"plexus.pipeline.generate({config_name!r})")
    return {"name": name, "pre_folder": pre_folder.rstrip("/"), "data_dir": data_dir,
            "run_log_dir": run_log_dir, "spec_path": yaml_file, "stopped": stopped,
            "frame_ms": frame_ms}


# _vram_warning: Warn when a captured CUDA graph plus the renderer would not fit this card's memory -> None.
def _vram_warning(sim, device: str) -> None:
    """A captured graph and a renderer compete for the same card, and the failure is silent (the allocator retries rather than raising: 100% CPU, no output; a 100 M render never printed `[engine] substep captured as a CUDA graph`).

    The test is the footprint against THIS card, not a particle count or a card name (20 M stalls a 48 GiB A6000, not an 80 GiB H100). Coefficients measured on `warp` over 500 k -> 100 M (paper/mpm_warp.pdf 5.2-5.3): 0.309 GiB per million particles eager, 0.42 with capture (its private pool stays resident, ~39%)."""
    # `per_parent` may be a mapping from the parent's type name to a count (a membrane patch 50 points, a nuclear-envelope patch 50,000), so the total is the sum over types of count x per-type budget; it feeds the VRAM projection
    def _child_total(v):
        pp = v.get("per_parent", 0)
        par = sim.sets.get(v.get("parent"), {}) or {}
        ptypes = par.get("types") or {}
        if isinstance(pp, dict):
            npar = int(par.get("n", par.get("per_parent", 1)) or 1)
            return sum(int(pp.get(tn, 0)) * int(t.get("count", 0) or
                                                round(float(t.get("fraction", 0.0)) * npar))
                       for tn, t in ptypes.items())
        return int(pp) * int(par.get("n", 1))
    _npart = sum(_child_total(v) for v in sim.sets.values() if isinstance(v, dict) and "per_parent" in v)
    _cap_on = any(isinstance(x, dict) and x.get("capture") for x in sim.schedule)
    if _npart and _cap_on:
        import torch
        _tot = torch.cuda.get_device_properties(device).total_memory / 2 ** 30
        _proj = 0.42 * _npart / 1e6            # GiB, captured
        if _proj > 0.80 * _tot:                # the renderer and the recorder want the rest
            print(f"[capture] WARNING: {_npart:,} particles with capture ON projects "
                  f"~{_proj:.0f} GiB on a {_tot:.0f} GiB "
                  f"{torch.cuda.get_device_properties(device).name}, and a live "
                  f"renderer needs room too. Without capture it is ~{0.309 * _npart / 1e6:.0f} "
                  f"GiB. If the run stalls at 100% CPU with no output and never prints "
                  f"'substep captured as a CUDA graph', that is why -- set "
                  f"`capture: false` on the spec's substep block.", flush=True)


# _mark: Write a marker file (`_complete`, `_completed_plot`) into a run folder.
def _mark(run_log_dir: str, marker: str, info: str) -> None:
    with open(os.path.join(run_log_dir, marker), "w") as f:
        f.write(f"{info}\n")
