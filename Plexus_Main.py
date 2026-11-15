"""Plexus entry point.

    python Plexus_Main.py -o <task> <config_name> [--output_root ROOT] [--force]

`<task>` is generate and/or plot, optionally chained (e.g. `generate_plot`). `<config_name>`
selects a spec under config/: `tissue/exp05_spheroid_v01_f1000` loads
config/tissue/exp05_spheroid_v01_f1000.yaml and writes the trajectory, the movie and the
stills to {data_root}/graphs_data/tissue/exp05_spheroid_v01_f1000/. A bare name (no slash) has
its folder inferred from the name (plexus.paths); an absolute .yaml path names its folder
explicitly. There is no default data root: pass --output_root or set $PLEXUS_OUTPUT_ROOT /
$GNN_OUTPUT_ROOT.
"""
from __future__ import annotations

import os
import sys
import shutil
import argparse

# ensure src/ is importable when run from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import plexus.operators  # noqa: F401  self-register the operator library
from plexus.schema import load
from plexus.paths import resolve_config, validate_pre_folder, set_data_root, log_path


def main():
    # --- manual / debug entry --------------------------------------------------- #
    # Run Plexus_Main.py with NO CLI args (e.g. the IDE "Debug" button) and it falls
    # back to these hardcoded values. Edit the list to debug a different task/config.
    # Ignored as soon as any real CLI argument is passed.
    if len(sys.argv) == 1:
        # The first step of experiments/exp05_spheroid_from_words: 20 point cells that move, adhere, grow and divide
        # (`--no-viz` keeps the VTK plotter from firing on every frame while stopped on a breakpoint).
        sys.argv += ["-o", "generate", "tissue/exp05_spheroid_v01_f1000",
                     "--device", "cpu", "--no-viz", "--no-describe", "--force"]

    parser = argparse.ArgumentParser(description="plexus")
    parser.add_argument("-o", "--option", nargs="+", required=True,
                        help="<task> <config_name>, e.g. -o generate tissue/exp05_spheroid_v01_f1000")
    parser.add_argument("--output_root", default=None,
                        help="root for graphs_data/ and log/ (default: $PLEXUS_OUTPUT_ROOT / $GNN_OUTPUT_ROOT)")
    parser.add_argument("--device", default="cuda:0", help="cuda:N (default) or cpu")
    parser.add_argument("--force", action="store_true",
                        help="erase + regenerate data even if it already exists")
    parser.add_argument("--movie", action="store_true",
                        help="on -o plot, also render a gif movie per set")
    parser.add_argument("--grid", action="store_true",
                        help="render the MLS-MPM 6-panel grid-diagnostic movie (objects/C/F/Jp/stress/grid)")
    parser.add_argument("--no-viz", action="store_true",
                        help="NO RENDERING AT ALL: no live mp4, no live png snapshots, no plot pass, "
                             "no captioning. This is how a throughput measurement is taken -- the "
                             "ms/frame a render is folded into is not the simulation's")
    parser.add_argument("--render-n", default="500000000",
                        help="particles DRAWN in the live mp4 (default: all, capped at 500 M); "
                             "the run still simulates all of them. "
                             "COMMA-SEPARATED writes one movie per value (movie_10M.mp4, "
                             "movie_50M.mp4, ...) from the SAME simulation -- the only way to see a "
                             "run at several draw counts when its trajectory is too big to store")
    parser.add_argument("--render-max-frames", type=int, default=300,
                        help="cap on rendered frames; longer runs are strided down to this")
    parser.add_argument("--no-real-time", action="store_true",
                        help="draw every frame instead of choosing the stride that makes 60 fps "
                             "playback equal real time. Only has an effect when the spec declares "
                             "`general.units`, since without them a second means nothing")
    parser.add_argument("--keep-stills", action="store_true",
                        help="keep the numbered still_NN_*.png after the run. They exist to let a "
                             "run be WATCHED while it runs; once the mp4 is written they are "
                             "redundant copies of frames it already holds, so they are deleted by "
                             "default and only 3d.png (the final frame) is kept")
    parser.add_argument("--render-stills", type=int, default=10,
                        help="how many PNG stills to drop through the run, copied from the movie's "
                             "own rendered frames (no extra render). 0 disables. The newest is "
                             "always also written as 3d.png so a long run can be watched")
    parser.add_argument("--render-dot", default=None,
                        help="dot size in px, or 'auto' to size it to the drawn particles' median "
                             "nearest-neighbour spacing. DEFAULT: the spec's `plotting.dot_size`, "
                             "then auto -- so the size lives in the config, not in the command")
    parser.add_argument("--no-describe", action="store_true",
                        help="skip the automatic VLM video description that -o generate runs by default")
    parser.add_argument("--describe-out", default=None,
                        help="aggregate description file (default: graphs_data/video_descriptions.txt)")
    args = parser.parse_args()

    if args.device.startswith("cuda"):               # fall back to CPU if no GPU is present
        import torch
        if not torch.cuda.is_available():
            print(f"[device] {args.device} unavailable -> falling back to cpu", flush=True)
            args.device = "cpu"

    if args.output_root:
        assert os.path.isdir(args.output_root), f"--output_root does not exist: {args.output_root}"
        assert os.access(args.output_root, os.W_OK), f"--output_root not writable: {args.output_root}"
        set_data_root(args.output_root)

    task = args.option[0]
    config_name = args.option[1] if len(args.option) > 1 else None
    if config_name is None:
        parser.error("a config name is required: -o <task> <config_name>")

    # THE PIPELINE IS ONE FUNCTION AND THIS IS ITS COMMAND LINE. `plexus.pipeline.generate` is what
    # runs here and what the web page's RUN button runs, so a run started from a terminal and one
    # started from the page are the same run -- same CFL rewrite, same schema gate, same movie,
    # same folder. Nothing below this line is a second implementation of any of it.
    from plexus.pipeline import generate
    if "generate" in task:
        generate(config_name, device=args.device, force=args.force, viz=not args.no_viz,
                 render_n=args.render_n, render_max_frames=args.render_max_frames,
                 real_time=not args.no_real_time, keep_stills=args.keep_stills,
                 render_stills=args.render_stills, render_dot=args.render_dot,
                 describe=not args.no_describe, describe_out=args.describe_out,
                 grid=args.grid, movie=args.movie, plot=("plot" in task),
                 argv_line=" ".join(sys.argv))
        return
    if "plot" in task:
        from plexus.paths import log_path
        from plexus.utils import _mark
        yaml_file, pre_folder, name = resolve_config(config_name)
        validate_pre_folder(pre_folder)
        if not os.path.isfile(yaml_file):
            parser.error(f"config not found: {yaml_file}")
        print(f"task={task}  type={pre_folder.rstrip('/')}  config={name}  ({yaml_file})")
        sim = load(yaml_file)
        run_log_dir = log_path(pre_folder.rstrip("/"), name)
        os.makedirs(run_log_dir, exist_ok=True)
        shutil.copy2(yaml_file, os.path.join(run_log_dir, "spec.yaml"))
        data_dir = None
        if not args.no_viz:
            from plexus.plot import plot_dataset
            data_dir = plot_dataset(sim, pre_folder, movie=args.movie)
            _mark(run_log_dir, "_completed_plot", data_dir)
        if args.grid and data_dir is None:
            from plexus.paths import graphs_data_path
            data_dir = os.path.join(graphs_data_path(), pre_folder.rstrip("/"), name)
        if args.grid and data_dir and not args.no_viz:
            from plexus.generators.mpm_grid_diag import generate_grid_movie
            generate_grid_movie(sim, data_dir, device=args.device)
        _mark(run_log_dir, "_complete", " ".join(sys.argv))



if __name__ == "__main__":
    main()


# python Plexus_Main.py -o generate tissue/exp05_spheroid_v01_f1000 --output_root <dir>
# python Plexus_Main.py -o generate si_material/exp01_mpm88_r2 --force --output_root <dir>
