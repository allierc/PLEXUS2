#!/usr/bin/env python
"""The Plexus GUI server: the watcher at /watch and the scene API that build sessions drive.

    python Plexus_gui.py                     # http://127.0.0.1:8799/watch
    python Plexus_gui.py --tab bio           # the scene's starting tab: bio | material
    python Plexus_gui.py --port 8790 --no-browser

`/watch` is the read-only build history and live views (`src/plexus/gui/watch.py`); `/api/scene/*` and `/api/tab/<tab>/build` seed, view and run scenes through `plexus.pipeline.generate`, the body of `Plexus_Main.py -o generate` (`tools/gui_drive.py` drives them). The old `--bio` / `--material` / `--studio` flags are `--tab`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from plexus.gui.__main__ import main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    for old in ("--bio", "--material", "--studio"):          # the flags the bookmarks still use
        if old in argv:
            argv.remove(old)
            if old != "--studio":
                argv += ["--tab", old[2:]]
    main(argv)
