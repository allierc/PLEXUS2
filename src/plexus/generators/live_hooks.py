"""The per-frame hooks a generated run carries into the engine: the live PNG snapshot, the live
movie, the caller's own `on_frame`, composed into the ONE `on_frame` slot the engine offers.

Split out of `data_generate` so that function reads as: make the directory, build the hooks, run
the engine, save. Nothing here imports a rendering stack at module level -- matplotlib, pyvista
and the panel renderers are imported inside the branch that builds them, so a `--no-viz` run
(throughput measurement) and `plexus.generators.graph_data_generator` itself still import none.
"""
from __future__ import annotations

import os


def build_frame_hooks(sim, data_dir: str, live_movie: dict | None, live_every_frac: float | None,
                      on_frame, want: str):
    """Build what runs after every frame. Returns `(on_frame, movs, live_movie)`:

    `on_frame(H, tick)` -- the single hook to hand the engine (None when there is nothing to run),
    `movs` -- the movie writers to `.close()` in a `finally` once the engine returns,
    `live_movie` -- the request AS IT STANDS AFTER THIS: None when the spec's own renderer
    (`want`, its `plotting.renderer`, lower-cased) writes the movie instead, or when a panel
    renderer took it over. The caller's later "was a live movie made" tests read this value.

    `live_movie` is the dict from `--render-n` / the pipeline (None = no movie); a movie turns the
    PNG snapshot off. `on_frame` is the caller's own hook, composed after the movie's."""
    # THE LIVE SNAPSHOT, ON BY DEFAULT. A 1,800-frame run writes nothing anyone can look at for
    # half an hour; this rewrites `2d.png`/`3d.png` in the data directory every 5% of the frames --
    # TWENTY pictures over a run rather than ten. At 10% a 6,000-frame Turing run showed nothing for
    # the first ten minutes and then jumped 600 frames at a time, which is too coarse to catch a
    # pattern forming or a run going non-finite while there is still time to kill it. The cost is a
    # matplotlib figure per snapshot, a fraction of a second against the minutes between them.
    # frame number on it, so the run can be watched rather than only waited for. Off with
    # `live_every_frac=None`. The matplotlib import is inside `plexus.live.snapshot`, so this module
    # still imports no plotting stack -- see its own docstring on why that matters.
    hooks = []
    # THE VTK MOVIE SUPERSEDES THE MATPLOTLIB PNG SNAPSHOT -- do not run both. They answer the same
    # question ("what is this run doing right now") and the movie answers it more often and far more
    # cheaply: 0.1 s a frame against seconds-to-minutes for a matplotlib scatter of the full set.
    # Running both is how a 100 M render came to spend its time in `snapshot` while the movie it was
    # asked for sat idle. `--no-viz` still turns off both; asking for the movie now turns off the
    # stills, and a run that wants the stills can pass `live_movie=None`.
    if live_movie is not None:
        live_every_frac = None
    if live_every_frac:
        from plexus.live import every_n, snapshot
        _stride = every_n(sim.n_frames, live_every_frac)

        def _snap(H, tick, _d=data_dir, _n=sim.n_frames, _name=sim.name, _s=_stride,
                  _st=(sim.plotting or {})):
            if tick % _s == 0 or tick == _n:
                # THE SPEC'S OWN COLOUR TABLE. Which chem column is drawn in which colour is a
                # property of the model (a Gray-Scott substrate is usually not drawn; May-Leonard's
                # three species are a partition and want RGB), so it travels with the spec rather
                # than being guessed by the renderer.
                snapshot(H, tick, _n, _d, name=_name, style=_st)
        hooks.append(_snap)

    movs = []
    # A MESH SPEC GETS NO POINT MOVIE. `live_movie` draws the largest set that carries positions,
    # and for a vertex-model run that is the VERTEX set -- a few hundred points scattered on a
    # surface, in the corner of a world box sized for the grown tissue. The result is a mostly empty
    # frame that is nonetheless called `movie.mp4`, so the run's headline artefact was the one
    # picture of it that shows nothing. When the spec has declared `renderer: vtk_mesh` it has said
    # which renderer it wants; `render_vtk` then writes `movie.mp4` itself.
    if live_movie is not None and want == "vtk_mesh":
        print("[live-movie] skipped: plotting.renderer is vtk_mesh, so the mesh renderer writes "
              "movie.mp4 -- a point cloud of a mesh set's vertices is not this run's picture",
              flush=True)
        live_movie = None
    if live_movie is not None:
        from plexus.live_movie import LiveMovie
        # SEVERAL MOVIES FROM ONE SIMULATION. At 200 M particles the trajectory is not stored --
        # 1000 frames would be 2.4 TB -- so a movie cannot be re-rendered at a different draw count
        # afterwards; the only way to see the same run at 10 M, 50 M and 100 M drawn is to write all
        # three DURING it. Each gets its own subsample and its own file; they share the simulation.
        _cfg = dict(live_movie)
        _ns = _cfg.pop("render_n")
        _ns = [int(x) for x in (_ns if isinstance(_ns, (list, tuple)) else [_ns])]
        for _n in _ns:
            _tag = "" if len(_ns) == 1 else f"_{_n // 1000000}M" if _n >= 1000000 else f"_{_n // 1000}k"
            movs.append(LiveMovie(out=os.path.join(data_dir, f"movie{_tag}.mp4"),
                                  world=list(sim.world_size), n_frames=sim.n_frames,
                                  up=int((sim.plotting or {}).get("up_axis", 2)),
                                  name=sim.name, sim=sim, style=(sim.plotting or {}),
                                  # A FREE BOUNDARY PUTS NOTHING AT [0, world]: this spec's box is a
                                  # camera hint, and the content is about the origin.
                                  # `free` SAYS NOTHING ABOUT WHERE THE CONTENT IS. A vesicle is
                                  # built about the ORIGIN and an MPM block is seeded inside
                                  # [0, world]; a spec with both has one of each, and framing on
                                  # +-world/2 then drew the gel outside the box it was in. When
                                  # there are material points the box is [0, world] and is right.
                                  centred=(str(getattr(sim, "boundary", "") or "").lower() == "free"
                                           and not any(k in (sim.sets or {}) for k in
                                                       ("mpm_particle", "cytosol", "nucleus"))),
                                  can_curve=False,      # the live path has only the current frame
                                  render_n=_n, stills=(_cfg.get("stills", 10) if _n == _ns[0] else 0),
                                  **{k: v for k, v in _cfg.items() if k != "stills"}))
        hooks.extend(movs)

    # COMPOSED, not replaced. The live PNG snapshot and the live movie are independent answers to
    # "what is this run doing right now" and a run may want both; `on_frame` is a single slot, so
    # the composition happens here rather than by one hook knowing about the other.
    if on_frame is not None:
        hooks.append(on_frame)
    on_frame = None
    if hooks:
        def on_frame(H, tick, _hs=tuple(hooks)):
            for h in _hs:
                h(H, tick)

    return on_frame, movs, live_movie
