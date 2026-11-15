"""Small run-wide helpers: the per-run banner and the switch that silences it, and the run-folder markers.

`quiet(True)` is called by every trainer and many tools, because a training job runs the engine
thousands of times and one banner per rollout buries the loss. The flag lives here and is read
as `utils._QUIET` (a copy imported by name would never see `quiet()` change it).
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from plexus.models.registry import get_operator

if TYPE_CHECKING:                  # annotations only
    from plexus.models.base import Hierarchy
    from plexus.schema import Spec


# _QUIET: Set by quiet(); silences the per-run banner and the recording line (not errors or warnings).
_QUIET = False


# quiet: Silence the per-run banner -> the previous setting.
def quiet(on: bool = True) -> bool:
    """Silence the per-run banner. Returns the previous value, so a caller can restore it."""
    global _QUIET
    was, _QUIET = _QUIET, bool(on)
    return was


# _print_run_summary: Print the run banner: world, sets with live counts, fields, operators.
def _print_run_summary(sim: Spec, H: Hierarchy) -> None:
    """One-time neat banner: the world, the sets (with live counts), the fields, and the
    operators (grouped by family) of this run -- so a glance shows what is being simulated."""
    ws = "[" + ", ".join(f"{float(w):g}" for w in H.world_size.tolist()) + "]"
    if _QUIET:
        return
    print(f"[engine] === {sim.name} ===  dim={H.dim}  world={ws}  boundary={sim.boundary}  "
          f"dt={sim.dt:g}  frames={sim.n_frames}", flush=True)
    # the physical scale, printed with the run: a spec without `general.units:` says it is dimensionless
    print(f"[build] {sim.units.describe()}", flush=True)
    sets = "  ".join(f"{n}({int((l.occ > 0).sum().item())})" for n, l in H.levels.items())
    print(f"[engine] sets:      {sets or '—'}", flush=True)
    print(f"[engine] fields:    {'  '.join(H.fields) or '—'}", flush=True)
    # operators grouped by their registry family (motion / interaction / mpm / ...), in spec order
    by_fam: dict[str, list[str]] = {}
    for o in sim.operators:
        fam = getattr(get_operator(o.op), "FAMILY", "?")
        by_fam.setdefault(fam, []).append(o.op)
    groups = "   ".join(f"{fam}: {', '.join(ops)}" for fam, ops in by_fam.items())
    print(f"[engine] operators: {groups}", flush=True)


# _mark: Write a marker file (`_complete`, `_completed_plot`) into a run folder.
def _mark(run_log_dir: str, marker: str, info: str) -> None:
    with open(os.path.join(run_log_dir, marker), "w") as f:
        f.write(f"{info}\n")
