"""Where a set's entities start and how fast: the `start:`, `frame:` and `vel_init:` keys of a set.

Read by `engine.build` while it places each top-level set, before any operator exists
(`seed_positions` in operators/seed_ops.py is the seed-operator route to the same x_0):
    _start_centers      `start:`    explicit points or a box -> positions [n, D]
    _measured_centers   `frame:`    body centres read from a scanned shape -> [n, 3]
    _init_velocity      `vel_init:` rest / random / circular_orbit / solid_body / radial -> [n, D]
    _orbit_groups       `circular_orbit` for a set placed as two discs
"""
from __future__ import annotations

import torch


# _orbit_groups: Circular-orbit velocities of a set placed as groups (two_disks) -> vel [n, D].
def _orbit_groups(vinit, lvl, n, D, rng, device):
    """`circular_orbit` for a set placed as groups (`two_disks`): returns vel [n, D].

    Each group orbits its own centre in its own plane (`spawn_group_rot`), with M(<r) summed over
    that group only; `central_mass` parks the group's first particle at its centre as a core.
    The encounter adds two half-speeds to every star of a disc: `approach` along x (closing) and
    `swirl` along y (orbiting). The pair's energy follows from |v_rel| = 2 sqrt(approach^2 + swirl^2)
    against the parabolic sqrt(2 G M_tot / d); its angular momentum L = 2 x separation x swirl sets
    the pericentre (swirl 0 = head-on plunge). `spins: [s0, s1]` overrides `spin` per group."""
    G_const = float(vinit.get("G", 1.0)); spin = float(vinit.get("spin", 1.0))
    soft = float(vinit.get("softening", 0.05)); jitter = float(vinit.get("jitter", 0.0))
    cmass = float(vinit.get("central_mass", 0.0)); approach = float(vinit.get("approach", 0.0))
    swirl = float(vinit.get("swirl", 0.0))
    spins = list(vinit.get("spins", []) or [])
    gid = lvl.spawn_group[:n]; rot = lvl.spawn_group_rot
    ngroups = int(gid.max().item()) + 1
    mbuf = getattr(lvl, "mass", None)
    m = mbuf[:n].clone() if mbuf is not None else torch.ones(n, device=device)
    st = lvl.state.clone(); px0, px1 = lvl.state_schema["pos"]
    idxs = [torch.nonzero(gid == g, as_tuple=False).flatten() for g in range(ngroups)]
    if cmass > 0.0:                                  # a core per group, parked at the group's centre
        for g, idx in enumerate(idxs):
            i0 = int(idx[0])
            m[i0] = cmass
            if mbuf is not None:
                lvl.mass[i0] = cmass
            st[i0, px0:px1] = st[idx, px0:px1].mean(0)
        lvl.state = st
    pos = lvl.get("pos")[:n]
    centres = torch.stack([pos[idx].mean(0) for idx in idxs])
    vel = torch.zeros(n, D, device=device)
    for g, idx in enumerate(idxs):
        R = pos[idx] - centres[g]
        nrm = rot[g][:, 2]                            # the disc's normal (its own spin axis)
        Rp = R - (R @ nrm)[:, None] * nrm[None, :]    # in-plane part: the orbital radius
        rr = Rp.norm(dim=-1).clamp(min=1e-9)
        order = torch.argsort(rr)
        m_cum = torch.zeros(idx.numel(), device=device)
        m_cum[order] = torch.cumsum(m[idx][order], 0)          # M(<r) inside THIS disc
        v_circ = (spins[g] if g < len(spins) else spin) * \
            torch.sqrt((G_const * m_cum.clamp(min=0)) / rr.clamp(min=soft))
        tang = torch.linalg.cross(nrm.expand_as(Rp), Rp)       # CCW about the disc normal
        tang = tang / tang.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        v = v_circ[:, None] * tang
        bulk = torch.zeros(D, device=device)
        side = torch.sign((centres[g] - centres.mean(0))[0])       # -1 = the low-x disc, +1 = high-x
        if approach > 0.0:
            bulk[0] = -side * approach                # radial (closing) half-speed, along the separation axis x
        if swirl != 0.0:
            bulk[1] = side * swirl                    # tangential half-speed along y; > 0 = prograde (L along +z)
        if approach > 0.0 or swirl != 0.0:
            v = v + bulk[None, :]
        if jitter > 0.0:
            v = v + jitter * torch.randn(idx.numel(), D, generator=rng, device=device)
        vel[idx] = v
        if cmass > 0.0:                               # the core rides with the disc, without spin
            vel[int(idx[0])] = bulk
    return vel


# _init_velocity: Initial velocity of a top-level set from its vel_init dict -> vel [n, D].
def _init_velocity(vinit, lvl, world_size, rng, device):
    """Initial velocity [n, D] (n = live count) of a top-level set from its `vel_init` dict.

    A physics-aware IC that may read all positions and masses, so it lives here, not in `spawn`.
    Mass-dependent modes read `lvl.mass`, so this runs after `_assign_types`. Modes:
      rest            v = 0
      random          v ~ U(-speed, speed) per axis
      circular_orbit  v = spin * sqrt(G M(<r) / r), tangential in xy; optional `central_mass` at node 0
      solid_body      rigid rotation omega x r in xy
      radial          v = speed * r_hat (outward > 0)"""
    n = int((lvl.occ > 0).sum().item())                # top-level: the first n slots are live
    pos = lvl.get("pos")[:n]; D = pos.shape[-1]
    mode = str(vinit.get("mode", "random"))
    c = 0.5 * world_size[:D]
    R = pos - c; r = R.norm(dim=-1, keepdim=True).clamp(min=1e-9); rr = r.squeeze(-1)
    if mode == "rest":
        return torch.zeros(n, D, device=device)
    if mode == "random":
        v = float(vinit.get("speed", 0.0))
        return (torch.rand(n, D, generator=rng, device=device) - 0.5) * (2 * v)
    if mode == "circular_orbit":
        # a `two_disks` placement orbits per group (each disc about its own centre and plane)
        if getattr(lvl, "spawn_group", None) is not None:
            return _orbit_groups(vinit, lvl, n, D, rng, device)
        G = float(vinit.get("G", 1.0)); spin = float(vinit.get("spin", 1.0))
        soft = float(vinit.get("softening", 0.05)); jitter = float(vinit.get("jitter", 0.0))
        mbuf = getattr(lvl, "mass", None)
        m = mbuf[:n].clone() if mbuf is not None else torch.ones(n, device=device)
        cmass = float(vinit.get("central_mass", 0.0))  # optional central point mass at node 0, parked at centre
        if cmass > 0.0:
            m[0] = cmass
            if mbuf is not None:
                lvl.mass[0] = cmass
            st = lvl.state.clone(); st[0, :D] = c; lvl.state = st
            pos = lvl.get("pos")[:n]; R = pos - c
            r = R.norm(dim=-1, keepdim=True).clamp(min=1e-9); rr = r.squeeze(-1)
        order = torch.argsort(rr)                       # enclosed mass M(<r) per particle
        m_cum = torch.zeros(n, device=device); m_cum[order] = torch.cumsum(m[order], 0)
        v_circ = spin * torch.sqrt((G * m_cum.clamp(min=0)) / rr.clamp(min=soft))
        tang = torch.zeros(n, D, device=device)         # CCW tangent in the xy-plane (axes 0,1)
        tang[:, 0] = -R[:, 1] / rr; tang[:, 1] = R[:, 0] / rr
        vel = v_circ[:, None] * tang
        if jitter > 0.0:
            vel = vel + jitter * torch.randn(n, D, generator=rng, device=device)
        if cmass > 0.0:
            vel[0] = 0.0
        return vel
    if mode == "solid_body":
        omega = float(vinit.get("omega", 1.0))
        tang = torch.zeros(n, D, device=device)
        tang[:, 0] = -R[:, 1]; tang[:, 1] = R[:, 0]     # omega x r in the xy-plane
        return omega * tang
    if mode == "radial":
        speed = float(vinit.get("speed", 1.0))          # outward >0 / inward <0
        return speed * (R / r)
    raise ValueError(f"unknown vel_init mode {mode!r}")


# _measured_centers: Body centres read from the scan (frame:, place: measured) -> (centres [n, 3], {scale, volume}).
def _measured_centers(sname, s, n, device):
    """Body centres read from the scan: `frame:` on a set, `place: measured` on its types.

        frame: {shape: eye, origin: globe, at: [0.25, 0.25, 0.25], scale: {part: globe, radius: 0.08}}

      shape   the library shape whose parts are placed
      origin  the part pinned to `at` (world units)
      scale   a number, or {part, radius}: the factor making that part's equivalent-sphere radius `radius`

    A type with `place: measured` goes to `at + scale * (its centroid - the origin's)`; a type
    without it keeps whatever `start` gives it."""
    import numpy as _np
    import torch as _t
    from plexus import shapes as _sh
    fr = dict(s["frame"] or {})
    shape = str(fr.get("shape") or "")
    if not shape:
        raise ValueError(f"sets.{sname}.frame: needs `shape: <a shape in the library>`")
    folder, _ = _sh.resolve(shape)
    alias = _sh.aliases(folder)
    P = _sh.parts(shape)

    def _part(nm):
        nm = alias.get(str(nm), str(nm))
        if nm not in P:
            raise ValueError(f"sets.{sname}.frame: shape {shape!r} has no part {nm!r} "
                             f"(it has {', '.join(sorted(P))})")
        return P[nm]

    o_name = fr.get("origin")
    if o_name is None:
        raise ValueError(f"sets.{sname}.frame: needs `origin: <the part the frame is pinned to>`")
    o_cen = _part(o_name)[0].mean(0)
    at = _np.asarray(fr.get("at", [0.0, 0.0, 0.0]), float)
    sc = fr.get("scale", 1.0)
    if isinstance(sc, dict):
        V, F = _part(sc.get("part", o_name))
        r_now = (3.0 * _sh._volume(V, F) / (4.0 * _np.pi)) ** (1.0 / 3.0)
        sc = float(sc["radius"]) / max(r_now, 1e-30)
    sc = float(sc)
    rows = []
    for tname, t in (s.get("types") or {}).items():
        t = t or {}
        cnt = int(t.get("count", 1))
        shp = str(t.get("shape", "") or "")
        if str(t.get("place", "")).lower() == "measured":
            if not shp.startswith("mesh:"):
                raise ValueError(f"sets.{sname}.{tname}: `place: measured` reads the part's own "
                                 f"centroid, so the type must name one (`shape: mesh:{shape}/...`)")
            c = at + sc * (_part(shp.split("/", 1)[-1] if "/" in shp else tname)[0].mean(0) - o_cen)
        else:
            c = at
        rows += [c] * cnt
    if len(rows) < n:
        rows += [at] * (n - len(rows))
    # the scan also gives each body's size: part volume x scale^3 (`provision` reads it off the parent level)
    vols = {}
    for tname, t in (s.get("types") or {}).items():
        t = t or {}
        shp = str(t.get("shape", "") or "")
        if str(t.get("place", "")).lower() == "measured" and shp.startswith("mesh:"):
            V, F = _part(shp.split("/", 1)[-1] if "/" in shp else tname)
            vols[tname] = float(_sh._volume(V, F)) * sc ** 3
    print(f"[frame] {sname}: {shape} scaled by {sc:.5g}, {len(rows)} bodies placed from the scan "
          f"about {o_name!r} at {_np.round(at, 4).tolist()}"
          + (f"; volumes " + ", ".join(f"{k} {v:.4g}" for k, v in vols.items()) if vols else ""),
          flush=True)
    return _t.tensor(_np.asarray(rows[:n], _np.float32), device=device), {"scale": sc, "volume": vols}


# _start_centers: Explicit placement from a spec start: (point list or box) -> positions [n, D].
def _start_centers(start, n: int, rng, device: str) -> torch.Tensor:
    """Explicit top-level placement from a spec `start`: either a list of D-point
    coords (deterministic, tiled to n) or a flat region box [lo..., hi...] (2*D values,
    uniform sample). Dimension-generic. Used by sets that seed at known locations
    (e.g. an MPM water blob / falling cube)."""
    if isinstance(start[0], (list, tuple)):
        pts = torch.tensor([[float(x) for x in p] for p in start], device=device)
        return pts[torch.arange(n, device=device) % pts.shape[0]]
    v = [float(x) for x in start]; D = len(v) // 2
    lo = torch.tensor(v[:D], device=device); hi = torch.tensor(v[D:2 * D], device=device)
    return lo + torch.rand(n, D, generator=rng, device=device) * (hi - lo)
