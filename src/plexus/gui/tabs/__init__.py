"""The tabs of the scene API: each turns a form dict into a spec (`/api/tab/<tab>/build`).

A tab module declares

    NAME, TITLE          the tab's name and heading
    PICK_DIR             where OPEN starts browsing
    CORPUS_MODE          which priming corpus a Claude session gets (`gui/corpus.py`)
    BRIEF                Claude's system prompt for driving this tab
    DEFAULT_FORM         the scene the tab opens on
    build_spec(form)     -> spec dict, the way a person would write it
    form_from_spec(spec) -> form dict, so an opened spec refills the form
    normalise(spec)      optional: derived consequences of the form (widths, fractions) -- bio
    write_spec(spec, path) -> raw text: dump + whatever guard the pipeline would run (CFL)

Everything after the form -- seed, view, run, play, Claude -- is the shared `/api/scene/*` routes,
and the run is `plexus.pipeline.generate`, the body of `Plexus_Main.py -o generate`.
"""
from __future__ import annotations

from importlib import import_module

ORDER = ("bio", "material")


def get(name: str):
    if name not in ORDER:
        raise KeyError(f"no tab {name!r}; tabs are {', '.join(ORDER)}")
    return import_module(f"plexus.gui.tabs.{name}")


def write_spec(tab, spec: dict, path: str) -> str:
    """Write through the tab's own writer when it has one, else dump and run the CFL guard."""
    w = getattr(tab, "write_spec", None)
    if w is not None:
        return w(spec, path)
    import os
    from plexus.gui.server import _dump_yaml
    from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(_dump_yaml(spec))
    Courant_Friedrichs_Lewy_condition(path)
    return open(path).read()
