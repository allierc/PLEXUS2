"""What a finished run writes beside its trajectory, and the render steps that follow it.

Split out of `data_generate` so that function reads as: make the directory, build the hooks, run
the engine, save, finish. Rendering imports stay inside the branches that need them.
"""
from __future__ import annotations

import os

import numpy as np


def save_trajectory_and_contact(out: dict, data_dir: str, save: bool) -> None:
    """`trajectory.npz` (when `save`: every set's columns flattened to `<set>__<column>`, plus
    fields and per-frame timings), then `contact.json` / `pressure.npz` (the contact operators'
    per-frame record; skipped silently on a run with no contact)."""
    if save:
        flat = {}
        for sname, d in out["sets"].items():
            if d["pos"] is not None:
                flat[f"{sname}__pos"] = d["pos"]
            for bname, arr in (d.get("state") or {}).items():
                flat[f"{sname}__{bname}"] = arr
            flat[f"{sname}__occ"] = d["occ"]
            ms = d.get("mesh")
            if ms:
                off = np.cumsum([0] + [len(m["E_srce"]) for m in ms]).astype(np.int64)
                flat[f"{sname}__mesh_offsets"] = off
                flat[f"{sname}__mesh_nF"] = np.asarray([m["nF"] for m in ms], np.int64)
                flat[f"{sname}__mesh_Nv"] = np.asarray([m["Nv"] for m in ms], np.int64)
                foff = np.cumsum([0] + [int(m["nF"]) for m in ms]).astype(np.int64)
                flat[f"{sname}__mesh_face_offsets"] = foff
                for col in ("E_srce", "E_trgt", "E_face"):
                    flat[f"{sname}__mesh_{col}"] = (np.concatenate([m[col] for m in ms])
                                                    .astype(np.int64))
                from plexus.models.mesh import mesh_row_columns
                scal, edge, face, fill = mesh_row_columns(ms)
                for col in scal:
                    flat[f"{sname}__mesh_{col}"] = np.asarray([fill(m, col) for m in ms], np.float64)
                for col in edge:
                    vals = [fill(m, col) for m in ms]
                    flat[f"{sname}__mesh_{col}"] = np.concatenate(vals).astype(np.float32)
                    flat[f"{sname}__mesh_{col}_offsets"] = (
                        np.cumsum([0] + [len(v) for v in vals]).astype(np.int64))
                for col in face:
                    flat[f"{sname}__mesh_{col}"] = (np.concatenate([fill(m, col) for m in ms])
                                                    .astype(np.float32))
            if d.get("node_type") is not None:
                flat[f"{sname}__node_type"] = d["node_type"]
            if d.get("type_names"):
                flat[f"{sname}__type_names"] = np.asarray(list(d["type_names"]))
            if d.get("node_type_t") is not None:               # the type column per row, when it changed
                flat[f"{sname}__node_type_t"] = d["node_type_t"]
            if d.get("parent") is not None:                  # containment: child -> parent index
                flat[f"{sname}__parent"] = d["parent"]
                flat[f"{sname}__parent_name"] = np.asarray(d["parent_name"])
        for fname, fd in out.get("fields", {}).items():     # continuum fields (heatmap movies)
            flat[f"{fname}__grid"] = fd["grid"]
            flat[f"{fname}__colors"] = fd["colors"]
        if out.get("frame_ms") is not None:
            flat["frame_ms"] = np.asarray(out["frame_ms"], np.float32)
        tmp = os.path.join(data_dir, "trajectory.partial.npz")      # written whole, then renamed: a reader never sees half a file
        np.savez(tmp, world=out["world"], world_size=out["world_size"], **flat)
        os.replace(tmp, os.path.join(data_dir, "trajectory.npz"))

    try:
        from plexus.operators import contact_ops as _co
        _rec = {}
        if _co.CONTACT_HISTORY:
            _rec["contact"] = [{k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                                for k, v in row.items()} for row in _co.CONTACT_HISTORY]
        if _co.INSIDE_HISTORY:
            _rec["inside"] = [{k: float(v) for k, v in row.items()} for row in _co.INSIDE_HISTORY]
        if _rec:
            import json as _json
            _json.dump(_rec, open(os.path.join(data_dir, "contact.json"), "w"))
            _pm = [pm for pm in _co.PRESSURE_MAP if pm is not None]
            if _pm:
                np.savez_compressed(os.path.join(data_dir, "pressure.npz"),
                                    pressure=np.stack(_pm).astype(np.float32))
            _c = _rec.get("contact") or []
            _i = _rec.get("inside") or []
            print(f"[generate] contact.json: {len(_c)} contact rows"
                  + (f", momentum residual max {max(r['momentum_residual'] for r in _c):.2e}, "
                     f"penetration max {max(r['depth_max'] for r in _c):.4g} (world units)" if _c else "")
                  + (f"; {len(_i)} inside rows, max behind the surface {int(max(r['n_inside'] for r in _i))}" if _i else "")
                  + (f"; pressure.npz {len(_pm)} maps" if _pm else ""), flush=True)
    except Exception as _e:                                       # noqa: BLE001 -- a record, never the run
        print(f"[generate] contact record not written ({type(_e).__name__}: {_e})", flush=True)


def finish_render(sim, out: dict, data_dir: str, live_movie: dict | None, want: str, save: bool) -> None:
    """After the run: keep or re-render the live movie, draw the mesh movie of a `vtk_mesh` spec,
    and delete the leftover stills. `want` is `plotting.renderer`, lower-cased; `live_movie` is
    the request as `build_frame_hooks` returned it (None when no live movie was made).
    Sets `out["replayed"]` when the movie was re-rendered, so the caption pass does not redo it."""
    _cf = str(((sim.plotting or {}).get("color_field", "") or "")).lower()
    if _cf in ("deformation", "strain", "volume", "pressure", "vorticity"):
        if (sim.plotting or {}).get("curve"):
            print(f"[render] plotting.curve declared but `color_field: {_cf}` cannot be recomputed "
                  f"from a trajectory -- keeping the movie this run just made, panels and all "
                  f"omitted. Use `color_field: speed` if the panels matter more.", flush=True)
    elif save and (sim.plotting or {}).get("curve") and live_movie is not None \
            and not (sim.plotting or {}).get("replay_curves", True):
        print("[render] plotting.replay_curves: false -- keeping the live movie (its panels were drawn live "
              "on declared axes; a replay would drop the grid fields)", flush=True)
        if isinstance(out, dict):
            out["replayed"] = True
    elif save and (sim.plotting or {}).get("curve") and live_movie is not None:
        try:
            from plexus import live_movie as _lm, render_vtk as _rv
            if _rv.available():
                print("[render] plotting.curve declared -- re-rendering the movie off the "
                      "trajectory, which is the first point the panel's axes can be fixed",
                      flush=True)
                _lm.replay(data_dir, sim, name=sim.name, stills=10)
                if isinstance(out, dict):
                    out["replayed"] = True
        except Exception as e:                              # noqa: BLE001
            print(f"[render] curve re-render unavailable ({type(e).__name__}: {e})", flush=True)
    if save and want == "vtk_mesh" and live_movie is None:
        try:
            from plexus import render_vtk
            if render_vtk.available():
                render_vtk.use_plotting(sim.plotting or {})
                render_vtk.still(data_dir, style="flat",
                                 out=os.path.join(data_dir, "3d.png"), name=sim.name)
                render_vtk.render_all(data_dir, seq=int((sim.plotting or {}).get("vtk_seq", 2)),
                                      name=sim.name)
        except Exception as e:                              # noqa: BLE001 -- never lose a finished run
            print(f"[render] mesh movie unavailable ({type(e).__name__}: {e})", flush=True)

    if save and not (live_movie or {}).get("keep_stills", False):
        import glob as _glob
        _left = _glob.glob(os.path.join(data_dir, "still_*.png"))
        for _p in _left:
            try:
                os.remove(_p)
            except OSError:
                pass
        if _left:
            print(f"[generate] removed {len(_left)} leftover still(s); pass --keep-stills to keep "
                  f"the strip", flush=True)
