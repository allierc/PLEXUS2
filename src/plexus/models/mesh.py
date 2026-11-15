"""The half-edge mesh as an engine concept: a closed cellular surface stored as a flat table.

Three parallel int64 arrays `E_srce`, `E_trgt`, `E_face` plus counts `nF` (faces = cells) and `Nv`
(vertices). Face f's ring is recovered by walking the table and appending `E_srce[k]` for every k with
`E_face[k] == f`, IN TABLE ORDER: that ordering IS the geometry (`topology_ops.rings_from_flat_3d` never
reads `E_trgt` and never sorts), so nothing here may sort, compact or scatter the table.

`MeshTable` subclasses `dict` on purpose: `discovery_okuda/instrument.py` decides whether an operator
DID ANYTHING with `isinstance(m, dict)` inside a bare `except`, and a non-dict mapping would silently
score every mesh-only operator zero acts.

The namespace is open: `RESERVED` are the five engine-owned keys; everything else (per-face targets,
myosin, caches, a probe's published field) is an extension namespace, because three mechanisms depend
on a name being ABSENT (`cell_shape_probe` publishes then deletes its field, `ecm_gate_growth` tests
`'mg_scale' in m`, the apoptosis discriminator resolves a spec-given name)."""
from __future__ import annotations

import numpy as np
import numpy as _np

try:
    import torch
except Exception:                                     # the schema is readable without torch
    torch = None


# RESERVED: The five keys the engine owns; every other key is the open extension namespace.
RESERVED = ("E_srce", "E_trgt", "E_face", "nF", "Nv")

# SEED_KEYS: What `mesh_seed` lays down (RESERVED + A0, P0, alive, V0f, V0, v_ref, R0, verts0), so a checkpoint loader can be checked against it; `divjit` and `Vbirth` are blocks on the cell set now.
SEED_KEYS = RESERVED + ("A0", "P0", "alive", "V0f", "V0", "v_ref", "R0", "verts0")

# MESH_KINDS: The recognised values of the legacy `mesh:` key.
MESH_KINDS = ("half_edge",)


# MeshTable: The half-edge table: a dict (see the module docstring) that is also a typed object.
class MeshTable(dict):
    """The half-edge table: a `dict` first, a typed object second. Constructed where the bare dict was, with the same keys, so every existing reader is unaffected and the change is checkable against a bit-for-bit twin run."""

    KIND = "half_edge"

    # a name in `_proxied` is not stored here: it is a width-1 block on the CELL SET, served as a view (reads slice `cell.state[:nF, c]`, writes go into it), so `reindex_faces` cannot carry it, `snapshot` cannot record it, and `Hierarchy.renumber_set` is the only thing that moves it
    # a view rather than 130 edited call sites: the arithmetic stays untouched and the move is testable by byte-identity (the spelling `m["A0"]` remains for now)
    _proxied: tuple = ()
    _cell = None                      # the cell set's Level, bound by `bind_cell_state`

    def bind_cell_state(self, cell_level, names):
        """Serve `names` from `cell_level`'s declared blocks instead of storing them here."""
        object.__setattr__(self, "_cell", cell_level)
        object.__setattr__(self, "_proxied", tuple(n for n in names
                                                   if n in cell_level.state_schema))
        for n in self._proxied:
            dict.pop(self, n, None)   # a stale local copy would shadow the set silently
        return self._proxied

    def _px(self, k):
        """A CONTIGUOUS copy of the cell set's column, not the strided view: `Level.get(k)` slices `state[:, c0:c1]`, whose stride is the full state width, `warp.from_torch` rejects it and an `index_add` over it reduces in another order (measured: 1.09e-04 drift in `pos` by frame 1). Safe because every write goes through `__setitem__`, into the column itself."""
        c = self._cell
        if c is None:
            return None
        return c.get(k)[:int(dict.get(self, "nF", 0)), 0].contiguous()

    def __getitem__(self, k):
        return self._px(k) if k in self._proxied else dict.__getitem__(self, k)

    def get(self, k, default=None):
        if k in self._proxied:
            v = self._px(k)
            return default if v is None else v
        return dict.get(self, k, default)

    def __setitem__(self, k, v):
        if k not in self._proxied:
            return dict.__setitem__(self, k, v)
        import torch as _t
        col = self._cell.get(k)
        w = v if _t.is_tensor(v) else _t.as_tensor(_np.asarray(v))
        # the array's own length is the authority, not `nF`: they disagree by design mid topology edit (`cell_die` writes the rebuilt `apop_flag`, one entry per SURVIVING face, before it sets `m["nF"]`)
        n = min(int(w.shape[0]), int(col.shape[0]))
        col[:n, 0] = w[:n].to(dtype=col.dtype, device=col.device)

    def __contains__(self, k):
        return (k in self._proxied and self._cell is not None) or dict.__contains__(self, k)

    def update(self, *a, **kw):
        """`dict.update` bypasses `__setitem__`, and `seed_mesh` seeds through it."""
        for src in a:
            for k, v in (src.items() if hasattr(src, "items") else src):
                self[k] = v
        for k, v in kw.items():
            self[k] = v

    # introspection
    @property
    def n_faces(self) -> int:
        return int(self.get("nF", 0))

    @property
    def n_vertices(self) -> int:
        """The vertex HIGH-WATER MARK, not the live count: `cell_die` never touches `Nv`, so vertices orphaned by a death stay inside every `state[:Nv]` slice (today's behaviour; fixing it would move the energy and the gate is bit-equality)."""
        return int(self.get("Nv", 0))

    def describe(self) -> str:
        extra = sorted(k for k in self if k not in SEED_KEYS)
        return (f"half_edge: {self.n_faces} faces, {self.n_vertices} vertices, "
                f"{len(self.get('E_srce', ())) if 'E_srce' in self else 0} half-edges"
                + (f"; + {len(extra)} operator key(s): {', '.join(extra[:8])}" if extra else ""))

    # the one carry
    def reindex_faces(self, keep, dt=None, dev=None, names=None):
        """Permute every per-face array through `keep` (new face -> old face): the one carry, shared by `cell_divide`, `cell_die`, `_carry_face_state` and `edge_flip`.

        `keep` COPIES the parent's value onto both daughters, so only INTENSIVE quantities (a density, a concentration, an age) may travel this way; an extensive one doubles at every division. A short array is clamped, not raised on (as `_carry_face_state` does; `medioapical_myosin` raises, and changing that is its own step)."""
        if torch is None:
            raise RuntimeError("reindex_faces needs torch")
        idx = torch.as_tensor(np.asarray(keep, np.int64), device=dev)
        for nm in sorted(set(names or ()) | set(self.get("face_carry") or ())):
            a = self.get(nm)
            if a is None:
                continue
            a = a if torch.is_tensor(a) else torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
            self[nm] = a.to(dev)[idx.clamp(max=max(a.shape[0] - 1, 0))].to(dt)

    def carry_vertices(self, births, dt=None, dev=None, names=None, level=None):
        """Give every vertex BORN this tick a value, by averaging its parents': the per-vertex half of `reindex_faces`.

        `births` is `[(new_vertex, (parent, ...)), ...]`, as appended by `divide_face_3d` (two parents, the split edge's endpoints) and `face_collapse_3d` (three: the survivor and the two it absorbed). Applied in order, so a vertex that is the parent of a later birth contributes its NEW value. The blend is a mean, so intensive quantities only (a thickness, a density); extensive ones need their own operator.

        `level`: a per-vertex quantity is either a mesh column (`self[name]`) or a state block on the Level (`level.state[:, c0:c1]`, e.g. the apico-basal separation). Without the second branch `declare_vertex_carry(m, "sep")` succeeded and every vertex born by division held zero (66 vertices with |sep| 0.0 against a seeded 0.2 on `ab_sphere`)."""
        if torch is None:
            raise RuntimeError("carry_vertices needs torch")
        want = sorted(set(names or ()) | set(self.get("vertex_carry") or ()))
        if not want or not births:
            return
        for nm in want:
            a = self.get(nm)
            sl = None
            if a is None and level is not None and nm in getattr(level, "state_schema", ()):
                sl = level.state_schema[nm]                # (c0, c1) into the level's state
                a = level.state[:, sl[0]:sl[1]]
            if a is None:
                continue
            a = a if torch.is_tensor(a) else torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
            a = a.to(dev) if dev is not None else a
            for new_i, parents in births:
                ps = [int(q) for q in parents if 0 <= int(q) < a.shape[0]]
                if not ps or int(new_i) >= a.shape[0]:
                    # a vertex past the end is not an error: like `reindex_faces`, clamp (an operator may not have grown its own array yet)
                    continue
                a[int(new_i)] = a[ps].mean(dim=0)
            if sl is None:
                self[nm] = a.to(dt) if dt is not None else a
            else:
                st = level.state.clone()
                st[:, sl[0]:sl[1]] = a
                level.state = st

    # FACE_RECORD: Per-face names recorded beside the topology (`myo_med`); the colours the renderers read (activator, age / ndiv, apop, inhib) live as `cell__*` blocks on the cell set. Every name is optional: an absent one is simply not drawn.
    FACE_RECORD = ("myo_med",)

    # FACE_ALIAS: Recorded name -> live name bridge for per-face arrays; empty since the columns became declared blocks on the cell set (kept, because `snapshot` and offline readers still consult it).
    FACE_ALIAS: dict = {}

    # SCALAR_RECORD: The operators' own scalar counters, recorded per row so a gate reads what an operator knows instead of inferring it: n_t1 (accepted reconnections, edge_flip), n_apop (extrusions, cell_die), div_blocked (divisions refused for want of buffer), apop_spill (material a death could not bequeath, must stay ~0), renumber_failed (must be 0), mono_h (uniform monolayer thickness; a scalar because a per-face array goes stale), interface_force_sum / interface_force_max (contact ledger of `contact_ops.record_interface`).
    SCALAR_RECORD = ("n_t1", "n_apop", "div_blocked", "apop_spill", "renumber_failed", "mono_h", "n_div", "n_reinsert", "n_type2",
                     "interface_force_sum", "interface_force_max", "itg_fresh", "itg_carry_err", "itg_bound_frac",
                     "itg_bound_amount", "bm_mass_mean", "bm_dead",
                     "ecad_surface_mean", "ecad_interior_mean", "mt2_surface_mean", "mt2_interior_mean")

    # EDGE_RECORD: Per-half-edge state (`myo`, `myo_amount`, `fz`, `vang`: the planar-polarity complexes of `junction_pcp`): a third ragged length with its own offsets. Recorded at the end of the tick, after `junction_sync`, so the myosin array is re-keyed to the edges (`myosin_array_aligned_with_half_edges` checks it).
    EDGE_RECORD = ("myo", "myo_amount", "fz", "vang")

    def snapshot(self, face_record=None):
        """One recorded frame: the three half-edge arrays, the two counts and the per-face state chosen by `FACE_RECORD` (not "whatever is in the table", whose open namespace would grow without bound). The topology part is what `topo_record` appends to `hist`, so offline readers are unaffected."""
        def _np(x):
            if torch is not None and torch.is_tensor(x):
                return x.detach().cpu().numpy()
            return np.asarray(x)
        out = dict(E_srce=_np(self["E_srce"]), E_trgt=_np(self["E_trgt"]),
                   E_face=_np(self["E_face"]), nF=int(self["nF"]), Nv=int(self["Nv"]))
        nF = out["nF"]
        for nm in (face_record if face_record is not None else self.FACE_RECORD):
            v = self.get(self.FACE_ALIAS.get(nm, nm), self.get(nm))
            if v is None:
                continue
            a = _np(v).ravel()
            if a.shape[0] >= nF:                    # a stale short array is dropped, not padded
                out[nm] = a[:nF].astype(np.float32)
        nE = len(out["E_srce"])
        for nm in (self.EDGE_RECORD):
            v = self.get(nm)
            if v is None:
                continue
            a = _np(v).ravel()
            # a short array is recorded as short, not padded: padding would make the alignment row pass on the frames it exists to catch
            out["e_" + nm] = a.astype(np.float32) if a.shape[0] == nE else a.astype(np.float32)
        for nm in self.SCALAR_RECORD:
            v = self.get(nm)
            if v is not None and np.ndim(v) == 0:
                out["scalar_" + nm] = float(v)
        return out


# mesh_row_columns: Every mesh column any recorded row carries, with a zero filler -> (scalars, edge_cols, face_cols, fill).
def mesh_row_columns(ms):
    """Every mesh column ANY recorded row carries, with a filler for the rows that do not.

    Returns `(scalars, edge_cols, face_cols, fill)`: three sorted name lists and `fill(m, col)`, a row's values, zero-filled when the row lacks the name.

    The UNION, not the intersection: almost nothing exists at row 0 (`age` / `ndiv` appear when `cell_divide` starts, `apop` at the first sentence, `renumber_failed` only when a renumber fails), so an intersection deleted those columns from the whole trajectory, blinded the gates (a missing key reads 0.0) and left the movies plain. Zero is the truthful fill: before the first division a row has age 0.

    Per-face columns are filled to `nF` because they share `mesh_face_offsets`; per-half-edge columns keep their own offsets and a missing row contributes nothing, so a short myosin array stays visible to the alignment check."""
    names = set()
    for m in ms:
        names |= {k for k in m if k not in RESERVED}
    scal = sorted(c for c in names if c.startswith("scalar_"))
    edge = sorted(c for c in names if c.startswith("e_"))
    face = sorted(c for c in names if not c.startswith(("scalar_", "e_")))

    def fill(m, col):
        v = m.get(col)
        if col.startswith("scalar_"):
            return 0.0 if v is None else float(v)
        if v is not None:
            return np.asarray(v).ravel()
        if col.startswith("e_"):
            return np.zeros(0, np.float32)
        return np.zeros(int(m["nF"]), np.float32)

    return scal, edge, face, fill


# declare_vertex_carry: Ask the topology operators to give a per-vertex array a value at every new vertex.
def declare_vertex_carry(m, name):
    """Ask the topology operators to give this per-vertex array a value at every new vertex. The per-vertex twin of `junction_ops._face_carry`, open for the same reason: a literal tuple of carried names dropped any state added by a new operator."""
    m.setdefault("vertex_carry", set()).add(name)


# is_mesh: Whether an object is a mesh table, or the bare dict that predates it -> bool.
def is_mesh(obj) -> bool:
    """A mesh table, or the bare dict that predates it. Both are legal for now: archived runs and
    four operator self-tests build the table as a plain dict, and they must keep working."""
    return isinstance(obj, dict) and "E_face" in obj
