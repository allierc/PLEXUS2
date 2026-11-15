"""A substep block of the schedule (`{substep_dt: X, steps: [...]}`), run once per frame, by CUDA graph or eagerly.

`run_substep_block` records the block once into a torch.cuda.CUDAGraph and replays it as one launch per substep (or runs it
eagerly where capture is refused); `SubstepState` holds what it keeps between ticks. The functions below decide whether a
capture may happen and detect when a captured graph went stale:
    _capture_refusals  why a block cannot be captured (empty list = it can)
    _block_sets        the sets the block's operators touch (the checks look only at these)
    _capture_state     every buffer a substep writes, to snapshot and restore around the capture
    _graph_sig         (address, shape, dtype) of those buffers, compared once per tick
"""
from __future__ import annotations

import torch

from plexus.paths import warn
from plexus.schema import Spec


# _capture_state: Every buffer a substep can write, for snapshot and restore around a graph capture -> list of tensors.
def _capture_state(H, sets=None) -> list:
    """Every buffer (levels and fields) a substep can write, de-duplicated by address, restricted to
    `sets` if given. Snapshot / restore around a graph capture: warm-up and capture run the substep
    and would otherwise silently advance the simulation."""
    seen, out = set(), []
    for _nm, lvl in H.levels.items():
        if sets is not None and _nm not in sets:
            continue                       # a set the captured block never touches cannot go stale
        for _n, t in lvl.named_buffers(recurse=False):
            if torch.is_tensor(t) and t.data_ptr() not in seen:
                seen.add(t.data_ptr()); out.append(t)
    for fld in H.fields.values():
        for _n, t in fld.named_buffers(recurse=False):
            if torch.is_tensor(t) and t.data_ptr() not in seen:
                seen.add(t.data_ptr()); out.append(t)
    return out


# _block_sets: Sets the operators of one substep block act on, with their ancestors -> set of names.
def _block_sets(step: dict, inst: list) -> set:
    """Sets the operators INSIDE one substep block act on, plus every ancestor they reach by containment
    (an MPM scatter reads its ancestors' deltas). Capture checks use this, not every set: operators
    outside the block (e.g. a `radius_graph` rebuilding `edge_index`) must not invalidate the graph."""
    names = set()
    op_names = set(step.get("steps", []) or [])
    for nm, ob, sel, _g in inst:
        if nm in op_names:
            names.add(sel.set)
            names.add(getattr(ob, "at", None))
    names.discard(None)
    return names


# _graph_sig: Identity of every buffer a captured graph baked in -> tuple of (address, shape, dtype).
def _graph_sig(H, sets=None) -> tuple:
    """Identity (address, shape, dtype) of every buffer a captured graph baked in. A graph replays
    against the addresses it saw; a reallocated buffer (a set resized, `lvl.state` rebound, an
    assignment instead of a copy) makes the replay write into memory nothing reads, silently.
    Compared once per tick over ~30 tensors."""
    return tuple((t.data_ptr(), tuple(t.shape), t.dtype) for t in _capture_state(H, sets))


# _capture_refusals: Reasons a substep block cannot be CUDA-graph captured -> list of strings, empty if it can.
def _capture_refusals(sim: Spec, H, step: dict, inst: list) -> list[str]:
    """Why this substep block cannot be captured (empty list = it can). Each item was measured to
    break capture: a wrongly accepted capture gives a silently wrong result, a refusal only an eager run."""
    why = []
    if int(getattr(H, "dim", 2)) != 3:
        why.append("dim != 3 (the 2D wall BC uses boolean-mask index_put and the 2D polar branch "
                   "calls cuSOLVER; both are uncapturable and neither is rewritten yet)")
    op_names = set(step.get("steps", []))
    for nm, ob, _sel, _g in inst:
        if nm not in op_names:
            continue
        if nm in ("mpm_scatter", "p2g") and str(getattr(ob, "polar", "svd")).lower() != "higham":
            why.append(f"{nm} at {getattr(ob, 'at', '?')} uses polar={getattr(ob, 'polar', 'svd')!r}; "
                       f"torch.linalg.svd is a cuSOLVER call and cannot be captured -- "
                       f"set `polar: higham` on it")
    # snow / viscoelastic depend on the implementation: the default `mpm_strain` uses cuSOLVER SVD and
    # boolean-mask indexing (uncapturable); the warp one declares `CAPTURABLE_MATERIAL_BRANCHES`
    _strain_ok = any(getattr(ob, "CAPTURABLE_MATERIAL_BRANCHES", False)
                     for nm, ob, _s, _g in inst if nm in op_names and nm == "mpm_strain")
    if not _strain_ok:
        for name, lvl in H.levels.items():
            for attr, label in (("is_snow", "snow"), ("is_visco", "viscoelastic")):
                t = getattr(lvl, attr, None)
                if t is not None and bool(t.any()):
                    why.append(f"set {name!r} contains {label} particles; that branch of "
                               f"mpm_strain is SVD plus boolean-mask indexing and is uncapturable "
                               f"-- `implementation: warp` on mpm_strain removes this")
    _touched = _block_sets(step, inst)
    _ints = [n for n in (getattr(H, "emit_order", None) or {}) if n in _touched]
    if _ints:
        why.append(f"engine-integrated sets {_ints} -- `_integrate` rebinds lvl.state "
                   f"every tick, so a captured graph would read a stale buffer")
    # `mesh_contact` rebuilds a shape-varying direction-bin table on the host every frame (host
    # tensors, a quantile sync): uncapturable while the surface moves
    if {"mesh_contact"} & op_names:
        why.append("mesh_contact rebuilds its direction-bin table from the live surface every "
                   "frame (host tensor construction, a quantile sync, a variable bucket depth), "
                   "so the block is uncapturable while the surface moves")
    # no static check for "structural" operators: what invalidates a graph is a REALLOCATED buffer,
    # which `_graph_sig` checks directly once per tick
    return why


# SubstepState: What a run keeps between ticks for its substep blocks: the persistent delta buffers and the graph bookkeeping.
class SubstepState:
    """Created once per run and passed to every `run_substep_block` call. The delta buffers are allocated once and refilled in
    place, because a captured CUDA graph bakes in tensor addresses; the other fields remember what was already said or captured."""

    def __init__(self):
        self.static_d: dict = {}      # set -> persistent copy of the frame-level coordinate delta
        self.static_b: dict = {}      # set -> {block: persistent copy of the frame-level extra-block delta}
        self.graph = {}               # (id(step), live ids) -> torch.cuda.CUDAGraph, or False once refused
        self.graph_sigs = {}          # the same key -> the buffer identities the graph was captured against
        self.refused_said = set()     # id(step) whose refusal has been printed already
        self.gate_said = set()        # (id(step), live ids) already checked for a step the frame gate emptied


# run_substep_block: Run one substep block for the current frame: `count` substeps, each from the frame-level baseline, by graph replay or eagerly -> None.
def run_substep_block(H, sim: Spec, step: dict, inst: list, tick: int, device, live_ids, run_scheduled, seen_this_tick, state: SubstepState) -> None:
    """`step` is the block `{substep_dt, steps: [...]}`; `live_ids` the instance indices whose frame gate is open at this tick;
    `run_scheduled(op_name, tick)` runs one named operator; `seen_this_tick` is the occurrence counter that run_scheduled reads.

    Stages: 2a snapshot the frame-level deltas as the baseline; 2b pick the graph for this frame's live instances; 2c capture it
    once (or refuse and stay eager); 2d drop it if a state buffer moved since the capture; 2e run the substeps."""
    _static_d, _static_b = state.static_d, state.static_b
    _graph, _graph_sigs = state.graph, state.graph_sigs
    _refused_said, _gate_said = state.refused_said, state.gate_said
    H.sub_dt = float(step["substep_dt"])
    count = max(1, round(H.dt / H.sub_dt))   # H.dt == sim.dt unless dt_schedule
    # ---- 2a. BASELINE: snapshot the frame-level deltas, the starting point of every substep
    # the frame-level delta is each substep's baseline: snapshot it (_d0, _b0), restore
    # before every substep, so an operator inside the block does not stack on its own
    # previous substep (a 20/40/60/80 body force instead of 20/20/20/20)
    # the persistent buffers (_static_*) keep tensor addresses stable for a CUDA graph; under
    # grad the clones are installed instead (an in-place copy_ breaks a later backward)
    _static_ok = not torch.is_grad_enabled()
    _d0 = {k: v.clone() for k, v in H._delta.items()}
    _b0 = {k: {b: t.clone() for b, t in d.items()}
           for k, d in H._delta_blocks.items()}
    # install the persistent buffers, refilled from this tick's snapshot
    for _k, _v in (_d0.items() if _static_ok else ()):
        _t = _static_d.get(_k)
        if _t is None or _t.shape != _v.shape or _t.device != _v.device:
            _t = _static_d[_k] = torch.empty_like(_v)
        _t.copy_(_v)
    for _k in [k for k in _static_d if k not in _d0]:
        del _static_d[_k]                       # a set that stopped emitting
    for _k, _d in (_b0.items() if _static_ok else ()):
        _sb = _static_b.setdefault(_k, {})
        for _b, _t2 in _d.items():
            _c = _sb.get(_b)
            if _c is None or _c.shape != _t2.shape or _c.device != _t2.device:
                _c = _sb[_b] = torch.empty_like(_t2)
            _c.copy_(_t2)
    # drop block buffers this tick does not emit: `zero_delta` does not reach them, and a
    # stale entry made a later operator add onto every earlier tick's sum (TRAPS T-81)
    for _k in [k for k in _static_b if k not in _b0]:
        del _static_b[_k]
    for _k, _d in _b0.items():
        for _b in [b for b in _static_b.get(_k, {}) if b not in _d]:
            del _static_b[_k][_b]
    if _static_ok:
        H._delta = _static_d
        H._delta_blocks = _static_b
    # ---- 2b. GRAPH KEY: which graph this frame's live instances select (and a warning for a step with none)
    # ---- CUDA graph capture of the substep (default on, bit-identical to eager).
    # The substep is ~400 tiny kernel launches and the GPU idles ~60% waiting on the
    # CPU; one captured graph replays it as one launch (2.36x measured on a two-set cell)
    # Manual capture, not torch.compile, which cannot chain operators into one graph.
    # `capture: true` in the spec gets the full refusal reason; otherwise one line
    # one graph per gate configuration: the frame gate is evaluated at capture time, so
    # the key holds the live instances of this block
    _blk_names = set(step.get("steps", []))
    _cap_key = (id(step), tuple(_j for _j in live_ids
                                if inst[_j][0] in _blk_names))
    _cap_asked = "capture" in step
    # warn once per configuration when a gate leaves the cycle without a step
    # (e.g. the only `mpm_grid_update` out of window = no grid solve at all)
    if _cap_key not in _gate_said:
        _gate_said.add(_cap_key)
        _dead = [t for t in step.get("steps", [])
                 if t not in {inst[_j][0] for _j in _cap_key[1]}]
        if _dead:
            warn(f"at frame {tick} the substep block has NO live instance of "
                 f"{_dead} -- every declared one is outside its "
                 f"after_frame/before_frame window, so that step is absent from "
                 f"the cycle from here on. If this is a staged run, the stage that "
                 f"takes over is missing or its window does not start at {tick}.")
    # ---- 2c. CAPTURE: once per key, either refuse (run eager) or warm up, capture and restore the state
    _blk_sets = _block_sets(step, inst)
    if step.get("capture", True) and _graph.get(_cap_key) is None and tick >= 1:
        _why = _capture_refusals(sim, H, step, inst)
        if _why or not str(device).startswith("cuda"):
            _graph[_cap_key] = False
            _msg = _why or ["device is not cuda"]
            _said = id(step) in _refused_said       # say it once per BLOCK, not
            _refused_said.add(id(step))             # once per gate configuration
            if _said:
                pass
            elif _cap_asked:
                warn("capture refused for this substep block; running it eager:\n"
                     + "\n".join(f"    - {w}" for w in _msg))
            else:
                print(f"[engine] substep not captured ({_msg[0].split(';')[0]})",
                      flush=True)
        else:
            # capture on the tensors' device (torch.cuda.graph binds to current_device)
            _dev_ctx = torch.cuda.device(device)
            _dev_ctx.__enter__()
            _snap = _capture_state(H, _blk_sets)
            _keep = [t.clone() for t in _snap]
            # 3 warm-ups on a side stream (a cold capture hits lazy cuBLAS init), each
            # restoring the state snapshot and refilling the delta buffers: operators
            # inside the block rebind `_delta[k]`, so unrestored warm-ups would sum into it
            _pd = dict(_static_d)
            _pb = {k: dict(d) for k, d in _static_b.items()}

            def _refill_deltas():
                for _k, _t in _pd.items():
                    _static_d[_k] = _t
                    _t.copy_(_d0[_k])
                for _k, _d in _pb.items():
                    for _b, _t in _d.items():
                        _static_b[_k][_b] = _t
                        _t.copy_(_b0[_k][_b])
                H._delta = _static_d
                H._delta_blocks = _static_b
            _side = torch.cuda.Stream(device=device)
            _side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(_side):
                for _w in range(3):
                    _refill_deltas()
                    seen_this_tick.clear()
                    for op_name in step["steps"]:
                        run_scheduled(op_name, tick)
            torch.cuda.current_stream(device).wait_stream(_side)
            for _t, _k in zip(_snap, _keep):
                _t.copy_(_k)                    # undo the warm-up
            _refill_deltas()
            _gr = torch.cuda.CUDAGraph()
            seen_this_tick.clear()
            with torch.cuda.graph(_gr):
                for op_name in step["steps"]:
                    run_scheduled(op_name, tick)
            for _t, _k in zip(_snap, _keep):
                _t.copy_(_k)                    # and undo the capture pass
            _refill_deltas()                    # the replay reads THESE, refilled each substep
            _graph[_cap_key] = _gr
            _graph_sigs[_cap_key] = _graph_sig(H, _blk_sets)
            _dev_ctx.__exit__(None, None, None)
            print(f"[engine] substep captured as a CUDA graph "
                  f"({len(step['steps'])} operators, {count} replays/frame)",
                  flush=True)
    # ---- 2d. CHECK: drop the graph if a state buffer moved since the capture
    _gr = _graph.get(_cap_key)
    if _gr and _graph_sig(H, _blk_sets) != _graph_sigs[_cap_key]:
        # a buffer moved since capture: replaying would write into memory nothing reads
        warn("a state buffer was reallocated after the substep was captured "
             "(a set resized, or an operator assigned instead of copying); "
             "dropping the CUDA graph and continuing eager for the rest of the run")
        for _k in [k for k in _graph if k[0] == id(step)]:    # every configuration of this block
            _graph[_k] = False
        _refused_said.add(id(step))
        _gr = None

    # ---- 2e. SUBSTEPS: run the block `count` times, each from the baseline, by graph replay or eagerly
    for _s in range(count):
        # the occurrence counter resets every SUBSTEP: the i-th mention takes the i-th
        # instance, and without this a block naming `mpm_scatter` once per particle set
        # scattered every set at every occurrence after substep 0
        seen_this_tick.clear()
        H.micro = H.micro + 1        # a new solve: the next scatter clears the grid
        if _s:
            # restore the frame-level baseline: refill the persistent buffers in place
            if not _static_ok:
                # under grad: rebind the clones (a copy_ would break a later backward)
                H._delta = {k: v.clone() for k, v in _d0.items()}
                H._delta_blocks = {k: {b: t.clone() for b, t in d.items()}
                                   for k, d in _b0.items()}
            for _k, _v in (_d0.items() if _static_ok else ()):
                _static_d[_k].copy_(_v)
            for _k, _d in (_b0.items() if _static_ok else ()):
                for _b, _t2 in _d.items():
                    _static_b[_k][_b].copy_(_t2)
            if _static_ok:
                H._delta = _static_d
                H._delta_blocks = _static_b
        if _gr:
            _gr.replay()        # the whole substep, one launch
        else:
            for op_name in step["steps"]:
                run_scheduled(op_name, tick)
    H.sub_dt = None
