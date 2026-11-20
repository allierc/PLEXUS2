# PLEXUS

A differentiable cell tissue computation

![exp34_flat_vesicle](exp34_flat_vesicle.gif)

## Installation

```bash
conda env create -f envs/environment.linux.yaml   # or envs/environment.mac.yaml
conda activate plexus
```

`environment.linux.yaml` is the environment the runs were made in (NVIDIA GPU, CUDA 12.8); `environment.mac.yaml` has the same packages unpinned and runs the watcher and the specs, not the Warp scenes of exp09 and exp10.

## Run

```bash
python Plexus_Main.py -o generate tissue/exp33_ucb_flat_e000_h150_e07 --output_root <data root>   # run one spec of config/
```

Open the watcher in a browser with `python Plexus_gui.py` (it opens http://127.0.0.1:8799/watch); it shows the records of `experiments/`, one at a time (`?rec=experiments/<name>`).

![watcher](watcher.png)

To start a new experiment, follow [`experiments/INSTRUCTION.md`](experiments/INSTRUCTION.md).
