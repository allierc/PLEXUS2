"""Display colours of a run's data: the colour of each channel of a field.

Used by `engine._assemble` to store `colors` beside each recorded field grid, so a renderer
(`plexus.plot`, `live_movie`) draws a field without asking the spec again.
"""
from __future__ import annotations

import numpy as np

from plexus.models.base import Hierarchy
from plexus.schema import Spec


# _DEFAULT_FIELD_COLORS: Per-species RGBA when the spec's plotting block names none (Lague's).
_DEFAULT_FIELD_COLORS = [
    (0.20, 0.95, 0.65), (1.00, 0.35, 0.45), (0.40, 0.65, 1.00), (1.00, 0.85, 0.30),
]






# _field_colors: Per-channel colours of a field -> RGB [C, 3].
def _field_colors(H: Hierarchy, sim: Spec, fld) -> np.ndarray:
    """Per-channel RGB [C, 3] of a field: the types' colours of the set it couples to, white if
    uncoupled / prescribed (e.g. a video). `plotting.field_colors` / `plotting.colors` override."""
    fcol = ((sim.plotting or {}).get("field_colors") or {}).get(getattr(fld, "name", None))
    if fcol:                                       # `plotting.field_colors: {<field>: [[r, g, b], ...]}`: one colour per channel, for a field no set colours
        return np.array([[float(x) for x in col[:3]] for col in fcol[:fld.C]], np.float32)
    couples = getattr(fld, "couples_to", None)
    lvl = H.levels[couples] if couples in H.levels else None
    names = list(getattr(lvl, "type_names", []) or []) if lvl is not None else []
    pcolors = (sim.plotting or {}).get("colors", {})
    cols = []
    for c in range(fld.C):
        nm = names[c] if c < len(names) else None
        default = _DEFAULT_FIELD_COLORS[c % len(_DEFAULT_FIELD_COLORS)] if names else (1.0, 1.0, 1.0)
        cols.append(pcolors.get(nm, default))
    return np.array([[float(x) for x in col[:3]] for col in cols], np.float32)
