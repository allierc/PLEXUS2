# PLEXUS

A differentiable language for tissue mechanisms: hierarchical sets, fields and operators.

```bash
python Plexus_Main.py -o generate tissue/exp05_spheroid_v01_f1000 --output_root <data root>   # run one spec of config/
python Plexus_gui.py                                                                       # the watcher, http://127.0.0.1:8799/watch
```

The watcher shows the records of `experiments/`, one at a time (`?rec=experiments/<name>`); their movies are not in the repository.

To start a new experiment, follow `Plexus/experiments/INSTRUCTION.md` in the Plexus repository.
