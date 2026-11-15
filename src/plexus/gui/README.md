# plexus.gui -- the watcher and the scene server

```bash
PYTHONPATH=src python Plexus_gui.py                  # http://127.0.0.1:8799/watch
PYTHONPATH=src python Plexus_gui.py --tab material   # the scene's starting tab: bio | material
```

`/watch` (`watch.py`) is the read-only build history and live views. The scene routes are what build sessions drive (`tools/gui_drive.py`); a tab (`tabs/`) turns a form dict into a spec the way a person would write it (`config/studio/<name>.yaml`, validated by `plexus.schema.load`; the CFL guard on an MPM spec), then the seeded scene is shown through the movie renderer (`bio_view.py`) and run by `plexus.pipeline.generate` on the server's VTK thread.

Each tab's default form, and the specs it shows the assistant (`gui/corpus.py`):

| tab | reference | dynamics |
|---|---|---|
| bio | `config/tissue/exp05_spheroid_v01_f1000.yaml` | point cells, vertex model, proteins, organelles |
| material | `config/si_material/exp01_three_materials_r2.yaml` | MPM bodies in a box (`mpm_ops.py`); render (dots x3, surface, surface specular, glassy), light and colour (particles, deformation, stress, velocities) menus rewrite the spec's plotting and re-seed, keeping the last run's frames for PLAY |

Routes (`server.py`, a table): `/watch` and `/api/watch/*`; `/api/tab/<tab>/build`; `/api/scene/{state, spec, seed, render, pick, info, view, run, frames, artefacts, ls, open, counts, save, refine, visible, claude, reset}`; `/api/bio/*` are the same handlers under the older name.
