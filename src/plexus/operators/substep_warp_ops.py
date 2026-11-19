"""exp09 speed (2026-11-19): the substep block of a crowd of small MPM bodies as Warp kernels that one CUDA graph can replay.

WHY. exp09 rna1_3d (5,545 bodies, 214,465 material points, 100 substeps per frame) spent 1.25 s per frame on one H200: 0.92 s in the torch contact penalty
(pair_potential[cell_contact]), the rest in nine small operators whose GPU work is under a millisecond each but which are launched one by one from Python, 100 times per
frame. The engine can record a substep block once into a CUDA graph and replay it as one launch (plexus/CUDA_graph_capture.py), but only if no operator of the block
talks to the host while it runs. Each default below does: a private torch.Generator (thermal_bath[langevin]), a Python list turned into a device tensor (the walls, the
scatter's block frame), a boolean-mask index (bond_springs), a `float()` or `int()` of a device value (the contact penalty). These variants do the same arithmetic with
none of that, so the whole block is capturable; run eagerly they are faster too (one kernel each).

    thermal_bath[langevin_warp]                   the Langevin bath, its normal draws made inside the kernel from (seed, substep count, point): no generator on the host
    active_force[box_slab_warp]                   the soft walls of a fixed box
    pair_potential[cell_contact_warp_pp_fixed]    pair_potential[cell_contact_warp_pp] with the block origins found by a kernel and the block size fixed between frames
    bond_springs[warp]                            the tethers of bound junctions; the pair list and each pair's bound share copied once per frame into fixed buffers
    apical_junction[warp]                         the static springs between the bodies seeded in contact (the chains)
    mpm_scatter[warp_pp_lean]                     mpm_scatter[warp_pp] with the block origins from a kernel (no host tensor), the unused colour channel not cleared
    mpm_grid_update[warp_pp_plain]                the grid solve with no surface tension as one in-place kernel (warp_pp_balanced falls back to torch for it)

A second group removes passes over NODES from the substep, so that its cost follows the points (a body of at most 64 points touches a tenth of its 15^3 grid block
and of its 13^3 contact block). MEASURED on an H200 (exp09 rna1_3d, 5,545 and 19,995 bodies): only the gather that solves is faster (5 to 8 % of a frame); the sparse
scatter and the sparse contact are 12 to 18 % SLOWER than the dense ones -- a kernel that walks the 27 nodes of every point costs 0.1 to 0.25 ms per substep whatever
it does there, more than zeroing or sweeping the whole blocks. They are kept as variants for grids too large to sweep, and are not part of the recommended set:

    mpm_scatter[warp_pp_sparse]                   warp_pp_lean that clears only the nodes the points deposited on at the last substep (each point records them)
    mpm_gather[warp_pp_solve]                     mpm_gather[warp_pp] dividing momentum by mass at the nodes it reads, with
    mpm_grid_update[solved_in_gather]             the grid solve left to that gather (it does nothing; the schedule keeps the step)
    pair_potential[cell_contact_warp_pp_sparse]   cell_contact_warp_pp_fixed with the sum of squares built from the points and the touched nodes cleared after the force
    contact_finder[sites_wide]                    contact_finder[sites] (once per frame) with its search grid sized to the world instead of 64 cells per axis

WHAT A CAPTURED BLOCK NEEDS FROM THE SPEC. A replayed graph runs no Python, so what changes once per frame must be handed over once per frame, outside the block:
`bond_springs[warp]` and `pair_potential[cell_contact_warp_pp_fixed]` are named in the schedule TWICE, once before the substep block and once inside it. The engine runs
the i-th mention of a name on its i-th instance and clears that count before every substep, so the one instance is called once per frame (H.sub_dt is None: it refreshes
its buffers and returns nothing) and once per substep (its kernels). Named only inside the block they refresh at the first substep of each frame, which is right when the
block runs eagerly; under a capture they refuse at capture time (loudly) instead of replaying a stale pair list.

State that a substep writes and that must not be advanced by the capture's warm-up passes (the bath's substep count, the junctions' load sum, the contact blocks) is
registered as a buffer of the point set, where the engine's capture snapshot restores it and where a reallocation makes the engine drop the graph and run eagerly.

Equal to the defaults to float round-off (atomic adds in any order), not bit-identical; the bath draws another noise stream than torch's generator (same law). CUDA, 3D, float32.
"""
import math

import torch

from plexus.models.registry import register_operator
from plexus.operators.adhesion_ops import CellAdhesion
from plexus.operators.chain_ops import ThermalBathLangevin
from plexus.operators.contact_warp_pp_ops import CellContactWarpPP
from plexus.operators.flatten_ops import ActiveForceBoxSlab
from plexus.operators.gastrulation_ops import ApicalJunction
from plexus.operators.mpm_ops import HAVE_WARP, _ancestor_accel, _const_any, _wp_launch, sub_dt
from plexus.operators.mpm_warp_pp_grid_ops import MPMGridUpdateWarpPPBalanced
from plexus.operators.mpm_warp_pp_ops import FIXED_SCALE, MPMGatherWarpPP, MPMScatterWarpPP, _pp_refuse

if HAVE_WARP:
    import warp as wp

    from plexus.operators.contact_warp_pp_ops import _cc_square
    from plexus.operators.mpm_warp_pp_grid_ops import pp_final
    from plexus.operators.mpm_warp_pp_ops import fixed_to_float, p2g_pp

    # ---- the bath ---------------------------------------------------------------------------------------------------------------------------------------------------
    @wp.kernel
    def _tick(CNT: wp.array(dtype=wp.int32)):
        """One thread: the substep count the bath draws from, advanced by one. A launch of its own, so that a replayed CUDA graph advances it."""
        CNT[0] = CNT[0] + 1

    @wp.kernel
    def _langevin(STATE: wp.array2d(dtype=float), va: int, OCC: wp.array(dtype=float), M: wp.array(dtype=float), CNT: wp.array(dtype=wp.int32),
                  seed: int, gamma: float, amp: float, drift: wp.vec3, ACC: wp.array(dtype=wp.vec3)):
        """One thread per point: a = -gamma v + sqrt(amp / m) xi + drift, amp = 2 gamma kT / dt_s; xi three standard normals from a generator seeded by (seed, substep count, point)."""
        i = wp.tid()
        if OCC[i] <= 0.5:
            ACC[i] = wp.vec3(0.0, 0.0, 0.0)
            return
        st = wp.rand_init(int(wp.rand_init(seed, CNT[0])), i)
        z0 = wp.randn(st)
        z1 = wp.randn(st)
        z2 = wp.randn(st)
        s = wp.sqrt(amp / wp.max(M[i], 1.0e-30))
        ACC[i] = wp.vec3(-gamma * STATE[i, va + 0] + s * z0 + drift[0], -gamma * STATE[i, va + 1] + s * z1 + drift[1], -gamma * STATE[i, va + 2] + s * z2 + drift[2])

    # ---- the walls --------------------------------------------------------------------------------------------------------------------------------------------------
    @wp.kernel
    def _box_walls(STATE: wp.array2d(dtype=float), pa: int, OCC: wp.array(dtype=float), k: float, lo: wp.vec3, hi: wp.vec3, ACC: wp.array(dtype=wp.vec3)):
        """One thread per point: a_d = k [(lo_d - x_d)^+ - (x_d - hi_d)^+] on each axis d, 0 for an empty slot."""
        i = wp.tid()
        if OCC[i] <= 0.5:
            ACC[i] = wp.vec3(0.0, 0.0, 0.0)
            return
        a = wp.vec3(0.0, 0.0, 0.0)
        for d in range(3):
            x = STATE[i, pa + d]
            a[d] = k * (wp.max(lo[d] - x, 0.0) - wp.max(x - hi[d], 0.0))
        ACC[i] = a

    # ---- the springs ------------------------------------------------------------------------------------------------------------------------------------------------
    @wp.kernel
    def _pair_tether(STATE: wp.array2d(dtype=float), pa: int, M: wp.array(dtype=float), PI: wp.array(dtype=wp.int32), PJ: wp.array(dtype=wp.int32),
                     ROW: wp.array(dtype=wp.int32), RHO: wp.array(dtype=float), kappa: float, r_eq: float, reach: float, per: wp.vec3,
                     ACC: wp.array(dtype=wp.vec3), LOAD: wp.array(dtype=float)):
        """One thread per slot of the pair list (ROW < 0: an unused slot or a pair with no junction row): the tether tension kappa rho (r - r_eq)^+ for r < reach,
        as an acceleration on both points (equal and opposite forces) and added to the junction row's load."""
        t = wp.tid()
        row = ROW[t]
        if row < 0:
            return
        i = PI[t]
        j = PJ[t]
        d = wp.vec3(STATE[j, pa + 0] - STATE[i, pa + 0], STATE[j, pa + 1] - STATE[i, pa + 1], STATE[j, pa + 2] - STATE[i, pa + 2])
        for a in range(3):
            if per[a] > 0.0:                                   # a periodic axis: the nearest image
                d[a] = d[a] - per[a] * wp.rint(d[a] / per[a])
        r = wp.max(wp.length(d), 1.0e-9)
        if r >= reach:
            return
        ten = kappa * RHO[t] * wp.max(r - r_eq, 0.0)
        if ten <= 0.0:
            return
        F = d * (ten / r)
        wp.atomic_add(ACC, i, F / wp.max(M[i], 1.0e-30))
        wp.atomic_add(ACC, j, F * (-1.0 / wp.max(M[j], 1.0e-30)))
        wp.atomic_add(LOAD, row, ten)

    @wp.kernel
    def _fixed_spring(STATE: wp.array2d(dtype=float), pa: int, M: wp.array(dtype=float), BI: wp.array(dtype=wp.int32), BJ: wp.array(dtype=wp.int32),
                      L0: wp.array(dtype=float), k: float, ACC: wp.array(dtype=wp.vec3)):
        """One thread per bond: F = k (L - L0) d / L on point i toward j, the opposite on j, each over its point's mass."""
        t = wp.tid()
        i = BI[t]
        j = BJ[t]
        d = wp.vec3(STATE[j, pa + 0] - STATE[i, pa + 0], STATE[j, pa + 1] - STATE[i, pa + 1], STATE[j, pa + 2] - STATE[i, pa + 2])
        L = wp.max(wp.length(d), 1.0e-9)
        F = d * (k * (L - L0[t]) / L)
        wp.atomic_add(ACC, i, F / M[i])
        wp.atomic_add(ACC, j, F * (-1.0 / M[j]))

    # ---- the contact penalty ------------------------------------------------------------------------------------------------------------------------------------------
    @wp.func
    def _bw(f: float, k: int) -> float:
        """The quadratic B-spline weight of stencil node k (0, 1, 2) at offset f = x / dx - base (as contact_warp_pp_ops._w)."""
        if k == 0:
            return 0.5 * (1.5 - f) * (1.5 - f)
        if k == 1:
            return 0.75 - (f - 1.0) * (f - 1.0)
        return 0.5 * (f - 0.5) * (f - 0.5)

    @wp.func
    def _bdw(f: float, k: int, dx: float) -> float:
        """Its derivative along the axis, per length."""
        if k == 0:
            return (f - 1.5) / dx
        if k == 1:
            return (-2.0 * (f - 1.0)) / dx
        return (f - 0.5) / dx

    @wp.func
    def _nd(u: int, n: int, per: int) -> int:
        """A node index on an axis of the world lattice: wrapped onto [0, n) on a periodic axis, else clamped to it."""
        if per != 0:
            return ((u % n) + n) % n
        return wp.clamp(u, 0, n - 1)

    @wp.kernel
    def _cc_origin(STATE: wp.array2d(dtype=float), pa: int, OCC: wp.array(dtype=float), CID: wp.array(dtype=wp.int32),
                   nx: int, ny: int, nz: int, dx: float, per: int, ORG: wp.array2d(dtype=wp.int32)):
        """One thread per point: its lowest stencil node (clamped to the lattice unless periodic) taken into its cell's block origin, the minimum per axis."""
        i = wp.tid()
        if OCC[i] <= 0.0:
            return
        bx = int(wp.floor(STATE[i, pa + 0] / dx - 0.5))
        by = int(wp.floor(STATE[i, pa + 1] / dx - 0.5))
        bz = int(wp.floor(STATE[i, pa + 2] / dx - 0.5))
        if per == 0:
            bx = wp.clamp(bx, 0, nx - 1)
            by = wp.clamp(by, 0, ny - 1)
            bz = wp.clamp(bz, 0, nz - 1)
        c = CID[i]
        wp.atomic_min(ORG, c, 0, bx)
        wp.atomic_min(ORG, c, 1, by)
        wp.atomic_min(ORG, c, 2, bz)

    @wp.kernel
    def _cc_deposit_g(STATE: wp.array2d(dtype=float), pa: int, OCC: wp.array(dtype=float), PV: wp.array(dtype=float), CID: wp.array(dtype=wp.int32),
                      ORG: wp.array(dtype=wp.vec3i), nx: int, ny: int, nz: int, B: int, dx: float, dx3: float, per: int,
                      PHIB: wp.array2d(dtype=float), S1: wp.array(dtype=float), SPAN: wp.array(dtype=wp.int32)):
        """One thread per point: its 27 weights times V / dx^3 onto its cell's block (PHIB [cells, B^3]) and onto the world's S1. A node beyond the block (a cell wider
        than B nodes) is left out of both and the width the cell needs is recorded in SPAN[0], read once per frame on the host."""
        i = wp.tid()
        if OCC[i] <= 0.0:
            return
        gx = STATE[i, pa + 0] / dx
        gy = STATE[i, pa + 1] / dx
        gz = STATE[i, pa + 2] / dx
        bx = int(wp.floor(gx - 0.5))
        by = int(wp.floor(gy - 0.5))
        bz = int(wp.floor(gz - 0.5))
        fx = gx - float(bx)
        fy = gy - float(by)
        fz = gz - float(bz)
        vd = PV[i] / dx3
        c = CID[i]
        o = ORG[c]
        top = int(0)
        for a in range(3):
            ix = _nd(bx + a, nx, per)
            lx = ix - o[0]
            if per != 0:
                lx = bx + a - o[0]
            wa = _bw(fx, a)
            for b in range(3):
                iy = _nd(by + b, ny, per)
                ly = iy - o[1]
                if per != 0:
                    ly = by + b - o[1]
                wb = _bw(fy, b)
                for d in range(3):
                    iz = _nd(bz + d, nz, per)
                    lz = iz - o[2]
                    if per != 0:
                        lz = bz + d - o[2]
                    if lx < B and ly < B and lz < B:
                        dep = wa * wb * _bw(fz, d) * vd
                        wp.atomic_add(PHIB, c, (lx * B + ly) * B + lz, dep)
                        wp.atomic_add(S1, (ix * ny + iy) * nz + iz, dep)
                    else:
                        top = wp.max(top, wp.max(lx, wp.max(ly, lz)) + 1)
        if top > 0:
            wp.atomic_max(SPAN, 0, top)

    @wp.kernel
    def _cc_force_g(STATE: wp.array2d(dtype=float), pa: int, OCC: wp.array(dtype=float), PV: wp.array(dtype=float), M: wp.array(dtype=float),
                    CID: wp.array(dtype=wp.int32), ORG: wp.array(dtype=wp.vec3i), nx: int, ny: int, nz: int, B: int, dx: float, per: int, k: float, c0: float,
                    PHIB: wp.array2d(dtype=float), S1: wp.array(dtype=float), S2: wp.array(dtype=float), ACC: wp.array(dtype=wp.vec3)):
        """One thread per point: a_i = -V_i sum_n grad w_in dU(n) (S1(n) - phi_A(n)) / m_i, dU(n) = k ((S1(n)^2 - S2(n)) / 2 - c0)^+ (0 for a dormant point)."""
        i = wp.tid()
        if OCC[i] <= 0.0:
            ACC[i] = wp.vec3(0.0, 0.0, 0.0)
            return
        gx = STATE[i, pa + 0] / dx
        gy = STATE[i, pa + 1] / dx
        gz = STATE[i, pa + 2] / dx
        bx = int(wp.floor(gx - 0.5))
        by = int(wp.floor(gy - 0.5))
        bz = int(wp.floor(gz - 0.5))
        fx = gx - float(bx)
        fy = gy - float(by)
        fz = gz - float(bz)
        c = CID[i]
        o = ORG[c]
        F = wp.vec3(0.0, 0.0, 0.0)
        for a in range(3):
            ix = _nd(bx + a, nx, per)
            lx = ix - o[0]
            if per != 0:
                lx = bx + a - o[0]
            wa = _bw(fx, a)
            da = _bdw(fx, a, dx)
            for b in range(3):
                iy = _nd(by + b, ny, per)
                ly = iy - o[1]
                if per != 0:
                    ly = by + b - o[1]
                wb = _bw(fy, b)
                db = _bdw(fy, b, dx)
                for d in range(3):
                    iz = _nd(bz + d, nz, per)
                    lz = iz - o[2]
                    if per != 0:
                        lz = bz + d - o[2]
                    if lx < B and ly < B and lz < B:
                        n = (ix * ny + iy) * nz + iz
                        s1 = S1[n]
                        du = k * wp.max(0.5 * (s1 * s1 - S2[n]) - c0, 0.0)
                        if du > 0.0:
                            wd = _bw(fz, d)
                            dd = _bdw(fz, d, dx)
                            e = du * (s1 - PHIB[c, (lx * B + ly) * B + lz])
                            F = F + wp.vec3(da * wb * wd, wa * db * wd, wa * wb * dd) * e
        ACC[i] = F * (-PV[i] / wp.max(M[i], 1.0e-30))

    # ---- the per-body grid blocks -------------------------------------------------------------------------------------------------------------------------------------
    @wp.kernel
    def _pp_origin(STATE: wp.array2d(dtype=float), pa: int, OCC: wp.array(dtype=float), per: int, half: float, ORG: wp.array(dtype=wp.vec3)):
        """One thread per body: its block origin = the centroid of its live points less half the block's side. A body's points are the `per` consecutive rows of its slot
        block, added in row order (the same sum on every run)."""
        b = wp.tid()
        s = wp.vec3(0.0, 0.0, 0.0)
        n = float(0.0)
        for q in range(per):
            p = b * per + q
            if OCC[p] > 0.0:
                s = s + wp.vec3(STATE[p, pa + 0], STATE[p, pa + 1], STATE[p, pa + 2])
                n = n + 1.0
        n = wp.max(n, 1.0)
        ORG[b] = wp.vec3(s[0] / n - half, s[1] / n - half, s[2] / n - half)


    # ---- the same, with every pass over the points and none over the nodes ----------------------------------------------------------------------------------------------
    from plexus.operators.mpm_warp_pp_ops import bspline_w, polar_cofactor

    @wp.kernel
    def p2g_pp_b(STATE: wp.array2d(dtype=float), pa: int, va: int,
                 C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
                 MASS: wp.array(dtype=float), MU: wp.array(dtype=float), LA: wp.array(dtype=float), PVOL: wp.array(dtype=float),
                 AEXT: wp.array(dtype=wp.vec3), JP: wp.array(dtype=float), SNW: wp.array(dtype=float), LIQ: wp.array(dtype=float),
                 OCC: wp.array(dtype=float), XTR: wp.array(dtype=wp.mat33),
                 PID: wp.array(dtype=wp.int64), ORG: wp.array(dtype=wp.vec3),
                 GM: wp.array(dtype=float), GMV: wp.array(dtype=wp.vec3), GC: wp.array(dtype=float),
                 ngx: int, ngy: int, ngz: int, n_local: int,
                 dx: float, inv_dx: float, dt: float, drag: float, iters: int,
                 has_snw: int, has_liq: int, has_xtr: int,
                 BASE: wp.array(dtype=wp.vec3i), BLK: wp.array(dtype=wp.int32)):
        """mpm_warp_pp_ops.p2g_pp (float atomics, the same arithmetic line by line) that also records where each point deposited: its block (BLK, -1 for a dormant point)
        and the lowest node of its 27 (BASE, before the clamp to the block), which `_pp_clear` reads to clear exactly those nodes before the next deposit."""
        p = wp.tid()
        if OCC[p] <= 0.0:
            BLK[p] = -1
            return
        blk = int(PID[p])
        o = ORG[blk]
        x = wp.vec3(STATE[p, pa + 0] - o[0], STATE[p, pa + 1] - o[1], STATE[p, pa + 2] - o[2])
        v0 = wp.vec3(STATE[p, va + 0], STATE[p, va + 1], STATE[p, va + 2])
        v = v0 + dt * (AEXT[p] - drag * v0)
        mass = MASS[p]
        Fp = F[p]
        Cp = C[p]
        J = wp.determinant(Fp)
        R = polar_cofactor(Fp, iters)
        mu_p = MU[p]
        la_p = LA[p]
        if has_snw == 1 and SNW[p] > 0.0:
            h = wp.exp(wp.clamp(10.0 * (1.0 - JP[p]), -6.0, 6.0))
            mu_p = mu_p * h
            la_p = la_p * h
        S = 2.0 * mu_p * ((Fp - R) * wp.transpose(Fp)) + wp.identity(n=3, dtype=float) * (la_p * J * (J - 1.0))
        if has_xtr == 1:
            S = S + XTR[p]
        S = S * ((-dt * 4.0 * inv_dx * inv_dx) * PVOL[p])
        affine = S + Cp * mass
        base = wp.vec3(wp.floor(x[0] * inv_dx - 0.5), wp.floor(x[1] * inv_dx - 0.5), wp.floor(x[2] * inv_dx - 0.5))
        fx = wp.vec3(x[0] * inv_dx - base[0], x[1] * inv_dx - base[1], x[2] * inv_dx - base[2])
        node0 = blk * n_local
        BASE[p] = wp.vec3i(int(base[0]), int(base[1]), int(base[2]))
        BLK[p] = blk
        for i in range(3):
            wi = bspline_w(fx[0], i)
            gi = wp.clamp(int(base[0]) + i, 0, ngx - 1)
            for j in range(3):
                wj = bspline_w(fx[1], j)
                gj = wp.clamp(int(base[1]) + j, 0, ngy - 1)
                for k in range(3):
                    wk = bspline_w(fx[2], k)
                    gk = wp.clamp(int(base[2]) + k, 0, ngz - 1)
                    w = wi * wj * wk
                    idx = node0 + (gi * ngy + gj) * ngz + gk
                    dpos = wp.vec3((float(i) - fx[0]) * dx, (float(j) - fx[1]) * dx, (float(k) - fx[2]) * dx)
                    mom = mass * v + affine * dpos
                    wp.atomic_add(GM, idx, w * mass)
                    wp.atomic_add(GMV, idx, w * mom)
                    if has_liq == 1:
                        wp.atomic_add(GC, idx, w * mass * LIQ[p])

    @wp.kernel
    def _pp_clear(BASE: wp.array(dtype=wp.vec3i), BLK: wp.array(dtype=wp.int32), ngx: int, ngy: int, ngz: int, n_local: int, has_liq: int,
                  GM: wp.array(dtype=float), GMV: wp.array(dtype=wp.vec3), GC: wp.array(dtype=float)):
        """One thread per point: the 27 nodes it deposited on at the last scatter set back to zero (points of one body write the same zeros)."""
        p = wp.tid()
        blk = BLK[p]
        if blk < 0:
            return
        b = BASE[p]
        node0 = blk * n_local
        for i in range(3):
            gi = wp.clamp(b[0] + i, 0, ngx - 1)
            for j in range(3):
                gj = wp.clamp(b[1] + j, 0, ngy - 1)
                for k in range(3):
                    gk = wp.clamp(b[2] + k, 0, ngz - 1)
                    idx = node0 + (gi * ngy + gj) * ngz + gk
                    GM[idx] = 0.0
                    GMV[idx] = wp.vec3(0.0, 0.0, 0.0)
                    if has_liq == 1:
                        GC[idx] = 0.0

    @wp.kernel
    def g2p_pp_mv(STATE: wp.array2d(dtype=float), pa: int, va: int,
                  C: wp.array(dtype=wp.mat33), GM: wp.array(dtype=float), GMV: wp.array(dtype=wp.vec3), OCC: wp.array(dtype=float),
                  PID: wp.array(dtype=wp.int64), ORG: wp.array(dtype=wp.vec3),
                  ngx: int, ngy: int, ngz: int, n_local: int,
                  inv_dx: float, dt: float, vmax: float, lo: float, hi: float, floor: float):
        """mpm_warp_pp_ops.g2p_pp reading each node's momentum and mass instead of a velocity solved beforehand: v_g = mv_g / m_g where m_g > floor, else 0 (the grid
        solve without surface tension, done at the 27 nodes a point reads)."""
        p = wp.tid()
        blk = int(PID[p])
        o = ORG[blk]
        x = wp.vec3(STATE[p, pa + 0] - o[0], STATE[p, pa + 1] - o[1], STATE[p, pa + 2] - o[2])
        if OCC[p] <= 0.0:
            STATE[p, pa + 0] = x[0] + o[0]
            STATE[p, pa + 1] = x[1] + o[1]
            STATE[p, pa + 2] = x[2] + o[2]
            return
        base = wp.vec3(wp.floor(x[0] * inv_dx - 0.5), wp.floor(x[1] * inv_dx - 0.5), wp.floor(x[2] * inv_dx - 0.5))
        fx = wp.vec3(x[0] * inv_dx - base[0], x[1] * inv_dx - base[1], x[2] * inv_dx - base[2])
        newv = wp.vec3(0.0, 0.0, 0.0)
        newC = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        node0 = blk * n_local
        for i in range(3):
            wi = bspline_w(fx[0], i)
            gi = wp.clamp(int(base[0]) + i, 0, ngx - 1)
            for j in range(3):
                wj = bspline_w(fx[1], j)
                gj = wp.clamp(int(base[1]) + j, 0, ngy - 1)
                for k in range(3):
                    wk = bspline_w(fx[2], k)
                    gk = wp.clamp(int(base[2]) + k, 0, ngz - 1)
                    w = wi * wj * wk
                    idx = node0 + (gi * ngy + gj) * ngz + gk
                    m = GM[idx]
                    gv = wp.vec3(0.0, 0.0, 0.0)
                    if m > floor:
                        mv = GMV[idx]
                        gv = wp.vec3(mv[0] / m, mv[1] / m, mv[2] / m)
                    dpos = wp.vec3(float(i) - fx[0], float(j) - fx[1], float(k) - fx[2])
                    newv = newv + w * gv
                    newC = newC + w * wp.outer(gv, dpos)
        newC = newC * (4.0 * inv_dx)
        for a in range(3):
            if newv[a] != newv[a]:
                newv[a] = 0.0
            for b in range(3):
                if newC[a, b] != newC[a, b]:
                    newC[a, b] = 0.0
        sp = wp.max(wp.length(newv), 1.0e-9)
        newv = newv * (wp.min(sp, vmax) / sp)
        xn = x + dt * newv
        for a in range(3):
            if xn[a] != xn[a]:
                xn[a] = 0.5
            xn[a] = wp.clamp(xn[a], lo, hi)
        STATE[p, pa + 0] = xn[0] + o[0]
        STATE[p, pa + 1] = xn[1] + o[1]
        STATE[p, pa + 2] = xn[2] + o[2]
        STATE[p, va + 0] = newv[0]
        STATE[p, va + 1] = newv[1]
        STATE[p, va + 2] = newv[2]
        C[p] = newC

    @wp.kernel
    def _cc_square_p(STATE: wp.array2d(dtype=float), pa: int, OCC: wp.array(dtype=float), PV: wp.array(dtype=float), CID: wp.array(dtype=wp.int32),
                     ORG: wp.array(dtype=wp.vec3i), nx: int, ny: int, nz: int, B: int, dx: float, dx3: float, per: int,
                     PHIB: wp.array2d(dtype=float), S2: wp.array(dtype=float)):
        """One thread per point: S2(n) += dep_p(n) phi_A(n) at its 27 nodes. Summed over the points of cell A that is phi_A(n)^2 (their deposits at n add up to phi_A(n)),
        so the world's sum of squares is built from the points, with no pass over every node of every block."""
        i = wp.tid()
        if OCC[i] <= 0.0:
            return
        gx = STATE[i, pa + 0] / dx
        gy = STATE[i, pa + 1] / dx
        gz = STATE[i, pa + 2] / dx
        bx = int(wp.floor(gx - 0.5))
        by = int(wp.floor(gy - 0.5))
        bz = int(wp.floor(gz - 0.5))
        fx = gx - float(bx)
        fy = gy - float(by)
        fz = gz - float(bz)
        vd = PV[i] / dx3
        c = CID[i]
        o = ORG[c]
        for a in range(3):
            ix = _nd(bx + a, nx, per)
            lx = ix - o[0]
            if per != 0:
                lx = bx + a - o[0]
            wa = _bw(fx, a)
            for b in range(3):
                iy = _nd(by + b, ny, per)
                ly = iy - o[1]
                if per != 0:
                    ly = by + b - o[1]
                wb = _bw(fy, b)
                for d in range(3):
                    iz = _nd(bz + d, nz, per)
                    lz = iz - o[2]
                    if per != 0:
                        lz = bz + d - o[2]
                    if lx < B and ly < B and lz < B:
                        dep = wa * wb * _bw(fz, d) * vd
                        wp.atomic_add(S2, (ix * ny + iy) * nz + iz, dep * PHIB[c, (lx * B + ly) * B + lz])

    @wp.kernel
    def _cc_clear(STATE: wp.array2d(dtype=float), pa: int, OCC: wp.array(dtype=float), CID: wp.array(dtype=wp.int32),
                  ORG: wp.array(dtype=wp.vec3i), nx: int, ny: int, nz: int, B: int, dx: float, per: int,
                  PHIB: wp.array2d(dtype=float), S1: wp.array(dtype=float), S2: wp.array(dtype=float)):
        """One thread per point: its 27 nodes set back to zero, on its cell's block and on the world's two sums (the positions have not moved since the deposit)."""
        i = wp.tid()
        if OCC[i] <= 0.0:
            return
        bx = int(wp.floor(STATE[i, pa + 0] / dx - 0.5))
        by = int(wp.floor(STATE[i, pa + 1] / dx - 0.5))
        bz = int(wp.floor(STATE[i, pa + 2] / dx - 0.5))
        c = CID[i]
        o = ORG[c]
        for a in range(3):
            ix = _nd(bx + a, nx, per)
            lx = ix - o[0]
            if per != 0:
                lx = bx + a - o[0]
            for b in range(3):
                iy = _nd(by + b, ny, per)
                ly = iy - o[1]
                if per != 0:
                    ly = by + b - o[1]
                for d in range(3):
                    iz = _nd(bz + d, nz, per)
                    lz = iz - o[2]
                    if per != 0:
                        lz = bz + d - o[2]
                    if lx < B and ly < B and lz < B:
                        PHIB[c, (lx * B + ly) * B + lz] = 0.0
                        n = (ix * ny + iy) * nz + iz
                        S1[n] = 0.0
                        S2[n] = 0.0

    @wp.kernel
    def _cc_overlap_max(STATE: wp.array2d(dtype=float), pa: int, OCC: wp.array(dtype=float), nx: int, ny: int, nz: int, dx: float, per: int,
                        S1: wp.array(dtype=float), S2: wp.array(dtype=float), PMAX: wp.array(dtype=float)):
        """One thread per point: the largest overlap (S1^2 - S2) / 2 over the nodes the points touch, kept in PMAX[0] (read once per frame)."""
        i = wp.tid()
        if OCC[i] <= 0.0:
            return
        bx = int(wp.floor(STATE[i, pa + 0] / dx - 0.5))
        by = int(wp.floor(STATE[i, pa + 1] / dx - 0.5))
        bz = int(wp.floor(STATE[i, pa + 2] / dx - 0.5))
        top = float(0.0)
        for a in range(3):
            ix = _nd(bx + a, nx, per)
            for b in range(3):
                iy = _nd(by + b, ny, per)
                for d in range(3):
                    n = (ix * ny + iy) * nz + _nd(bz + d, nz, per)
                    top = wp.max(top, 0.5 * (S1[n] * S1[n] - S2[n]))
        wp.atomic_max(PMAX, 0, top)


def _capturing():
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def _in_block(H):
    """True inside a substep block (the engine sets H.sub_dt there and clears it after), False at a frame-level call."""
    return getattr(H, "sub_dt", None) is not None


def _substeps_per_frame(H, op_name):
    """How many times per frame the substep block that names `op_name` runs it (None when the spec is not at hand: a bare test rig)."""
    for st in (getattr(getattr(H, "config", None), "schedule", None) or []):
        if isinstance(st, dict) and "substep_dt" in st and op_name in (st.get("steps") or []):
            return max(1, round(float(H.dt) / float(st["substep_dt"])))
    return None


def _need_cuda32(name, X):
    if not HAVE_WARP or X.device.type != "cuda" or X.dtype != torch.float32:
        raise ValueError(f"{name} is Warp on CUDA, float32 (got {X.device.type}, {X.dtype}); use the default implementation")


def _own_buffer(P, name, t):
    """Register `t` as buffer `name` of the level `P` (replacing an older one): the engine's capture snapshot restores it after its warm-up passes, and a new address
    makes the engine drop a captured graph and run the block eagerly (CUDA_graph_capture._graph_sig)."""
    if name in dict(P.named_buffers(recurse=False)):
        setattr(P, name, t)
    else:
        P.register_buffer(name, t, persistent=False)
    return t


@register_operator("thermal_bath", implementation="langevin_warp", family="mechanics", set="particle", kind="lateral",
                   title="the thermal bath of an MPM body, the kicks drawn inside one Warp kernel",
                   equation=r"""$$\mathbf a_i=-\gamma\,\mathbf v_i+\sqrt{2\gamma\,k_BT/(m_i\,\Delta t_s)}\;\boldsymbol\xi_i$$""")
class ThermalBathLangevinWarp(ThermalBathLangevin):
    r"""thermal_bath[langevin] as one Warp kernel that draws its own normals from (seed, substep count, point): no torch.Generator, so a CUDA graph can replay it.
    pt -> pt: emits `mpm_acceleration`; same parameters as thermal_bath[langevin]. Another noise stream than the default's (the same law: independent standard normals per point, axis and substep).

        \mathbf a_i=-\gamma\,\mathbf v_i+\sqrt{2\gamma\,k_BT/(m_i\,\Delta t_s)}\;\boldsymbol\xi_i
        a_i            the acceleration of point i
        \gamma         `friction`, the drag rate per time
        k_BT           `kT`, the thermal energy
        m_i            the point's mass
        \Delta t_s     `substep_dt`, the MPM substep the kicks are sized for
        \xi_i          a standard normal vector, new at every substep

    The substep count is one int32 on the device, advanced by a kernel of its own at every call and registered on the point set (see the module docstring).

    Reference: as thermal_bath[langevin]; Plexus exp09 speed (this work, 2026-11-19)."""
    EXPLAIN = r"The bath of a body of material points, as thermal_bath[langevin]: a drag $-\gamma\mathbf v$ and a random kick sized so that the body fluctuates with energy $k_BT$. The kicks are drawn on the GPU inside the kernel."
    MECHANISM_TAGS = ["langevin", "thermal_noise", "fused_kernel"]
    DIFFERENTIABLE = False

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self._cnt = None
        self._acc = None

    def forward(self, H, mask=None):
        P = H.level(self.at)
        X = P.get("pos")
        _need_cuda32("thermal_bath[langevin_warp]", X)
        dev, n = X.device, int(P.n)
        if self._cnt is None:
            if _capturing():
                raise RuntimeError("thermal_bath[langevin_warp]: first call inside a CUDA-graph capture (its buffers are allocated at the first, eager, call)")
            self._cnt = _own_buffer(P, f"_bath_substeps_{id(self)}", torch.zeros(1, dtype=torch.int32, device=dev))
            self._acc = torch.zeros(n, 3, dtype=torch.float32, device=dev)
            self._m = (P.mass.to(torch.float32) if getattr(P, "mass", None) is not None else torch.ones(n, dtype=torch.float32, device=dev)).contiguous()
        va = int(P.state_schema["vel"][0])
        m = P.mass if (getattr(P, "mass", None) is not None and P.mass.dtype == torch.float32 and P.mass.is_contiguous()) else self._m
        _wp_launch(_tick, 1, dev, [wp.from_torch(self._cnt)])
        _wp_launch(_langevin, n, dev, [wp.from_torch(P.state), va, wp.from_torch(P.occ.contiguous()), wp.from_torch(m), wp.from_torch(self._cnt),
                                       int(self.seed), float(self.gamma), float(2.0 * self.gamma * self.kT / self.dts),
                                       wp.vec3(float(self.drift[0]), float(self.drift[1]), float(self.drift[2])), wp.from_torch(self._acc, dtype=wp.vec3)])
        return {self.at: self._acc if mask is None else self._acc * mask[:, None].to(self._acc.dtype)}


@register_operator("active_force", model="box_slab_warp", family="mechanics", set="particle", kind="exchange",
                   equation=r"""$$\mathbf a_p=k\sum_{d=1}^{3}\Big[\big(l_d-x_{p,d}\big)^{+}-\big(x_{p,d}-u_d\big)^{+}\Big]\,\hat{\mathbf e}_d$$""",
                   title="hold every point inside a fixed box with soft walls, one Warp kernel")
class ActiveForceBoxSlabWarp(ActiveForceBoxSlab):
    r"""active_force[box_slab] as one Warp kernel: the box's faces are kernel constants, not a device tensor built from a Python list at every call, so a CUDA graph can replay it.
    particle -[containment]-> particle: as active_force[box_slab] (same parameters, the same acceleration to the last bit of each product).

        \mathbf a_p=k\sum_{d=1}^{3}\Big[\big(l_d-x_{p,d}\big)^{+}-\big(x_{p,d}-u_d\big)^{+}\Big]\,\hat{\mathbf e}_d
        a_p            the acceleration of point p
        k              `wall_stiffness`, the stiffness of the walls, per time squared (keep sqrt(k) x substep_dt well under 2)
        x_{p,d}        the coordinate of point p along axis d
        l_d, u_d       `lo`, `hi`: the lower and upper face of the box on axis d (default 0 and the world size)
        e_d            the unit vector of axis d
        (.)^+          the positive part: zero for a point inside the box

    Reference: as active_force[box_slab]; Plexus exp09 speed (this work, 2026-11-19)."""
    MECHANISM_TAGS = ["confinement", "soft_wall", "box", "fused_kernel"]
    DIFFERENTIABLE = False

    def forward(self, H, mask=None):
        P = H.level(self.at)
        X = P.get("pos")
        _need_cuda32("active_force[box_slab_warp]", X)
        dev, n = X.device, int(P.n)
        if getattr(self, "_acc", None) is None:
            self._acc = torch.zeros(n, 3, dtype=torch.float32, device=dev)
            self._lo = [float(v) for v in (self.lo if self.lo is not None else [0.0, 0.0, 0.0])]
            self._hi = [float(v) for v in (self.hi if self.hi is not None else [float(w) for w in H.world_size][:3])]
        _wp_launch(_box_walls, n, dev, [wp.from_torch(P.state), int(P.state_schema["pos"][0]), wp.from_torch(P.occ.contiguous()), float(self.k),
                                        wp.vec3(*self._lo), wp.vec3(*self._hi), wp.from_torch(self._acc, dtype=wp.vec3)])
        return {self.at: self._acc if mask is None else self._acc * mask[:, None].to(self._acc.dtype)}


@register_operator("bond_springs", implementation="warp", family="interaction", set="particle", kind="exchange",
                   title="pull adhered cells together, one Warp kernel over a pair list refreshed once per frame",
                   equation=r"""$$\mathbf F_{ij}=\kappa\,\rho_J\,(r-r_{eq})^+\,\hat{\mathbf r}_{ij}\,[r<r_c],\qquad \mathbf F_{ji}=-\mathbf F_{ij}$$""")
class CellAdhesionWarp(CellAdhesion):
    r"""bond_springs as one Warp kernel: the point pairs of contact_finder and the bound share of each pair's junction are copied once per frame into buffers of a fixed size
    (one slot per point: a point has at most one partner), which the kernel reads at every substep; the junctions' load is summed on the device.
    pt -> pt (reads junction): as bond_springs with no damping. Refused: `bond_damping`, the per-cell blocks (`bond_stiffness_block`, `bond_rest_length_block`, `bond_range_block`, `bond_damping_block`).

        \mathbf F_{ij}=\kappa\,\rho_J\,(r-r_{eq})^+\,\hat{\mathbf r}_{ij}\,[r<r_c],\qquad \mathbf F_{ji}=-\mathbf F_{ij}
        F_{ij}         the force on point i from point j (and -F_ij on j)
        \kappa         `bond_stiffness`, the tether stiffness per pair
        \rho_J         the bound share of their junction J, as it stands after contact_finder at the start of the frame
        r              the pair distance (the nearest image on a periodic axis)
        r_{eq}         `bond_rest_length`
        \hat r_{ij}    the unit vector from i to j
        r_c            `bond_range`, beyond which a pair does not pull

    In the schedule: once before the substep block (the refresh) and once inside it (see the module docstring).

    Reference: as bond_springs; Plexus exp09 speed (this work, 2026-11-19)."""
    MECHANISM_TAGS = ["cadherin_adhesion", "cell_cell_junction", "fused_kernel"]
    DIFFERENTIABLE = False

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self._bufs = None
        self._frame = None
        self._at_frame_level = False
        self._calls = 0

    def _setup(self, H, P):
        X = P.get("pos")
        _need_cuda32("bond_springs[warp]", X)
        why = [nm for nm, blk in (("bond_stiffness_block", self.kappa_block), ("bond_rest_length_block", self.r_eq_block), ("bond_range_block", self.bond_reach_block),
                                  ("bond_damping_block", self.zeta_block)) if blk]
        if self.zeta != 0.0:
            why.append("bond_damping")
        if why:
            raise ValueError("bond_springs[warp] does not do: " + ", ".join(why) + " -- use the default bond_springs")
        dev, n = X.device, int(P.n)
        J = H.level(self.junctions)
        z = lambda dt: torch.zeros(n, dtype=dt, device=dev)                   # noqa: E731
        self._bufs = dict(pi=z(torch.int32), pj=z(torch.int32), row=torch.full((n,), -1, dtype=torch.int32, device=dev), rho=z(torch.float32),
                          acc=torch.zeros(n, 3, dtype=torch.float32, device=dev),
                          load=_own_buffer(P, f"_bond_load_{id(self)}", torch.zeros(int(J.n), dtype=torch.float32, device=dev)))
        self._k = (float(self.kappa), float(self.r_eq), float(self.bond_reach))    # read once: a `float()` of a device tensor inside a capture is illegal
        self._nsub = _substeps_per_frame(H, "bond_springs")

    def _refresh(self, H, P):
        """Once per frame: contact_finder's pair list, each pair's junction row and bound share into the fixed buffers; the load sum cleared and handed to contact_finder."""
        b = self._bufs
        J = H.level(self.junctions)
        pairs = getattr(H, "_adh_pairs", None)
        n = 0 if pairs is None else int(pairs[0].numel())
        if n > b["row"].numel():
            raise RuntimeError(f"bond_springs[warp]: {n:,} point pairs but {b['row'].numel():,} point slots (a point has at most one partner)")
        b["row"].fill_(-1)
        if n:
            i, j, row = pairs
            b["pi"][:n] = i.to(torch.int32)
            b["pj"][:n] = j.to(torch.int32)
            b["row"][:n] = row.to(torch.int32)
            rho = J.get("rho")[:, 0].clamp(min=0.0) * (J.occ > 0.5).to(torch.float32)
            b["rho"][:n] = torch.where(row >= 0, rho[row.clamp(min=0)], torch.zeros((), dtype=torch.float32, device=rho.device))
        Lb = getattr(H, "_adh_box", None)                                   # every axis's period as the contact finder searched (zeros: none)
        if Lb is None:
            Lb = [float(getattr(H, "_adh_period", 0.0)), 0.0, 0.0]
        per = tuple(float(v) for v in Lb)
        if getattr(self, "_per", per) != per and _capturing():
            raise RuntimeError("bond_springs[warp]: the periods changed inside a capture")
        self._per = per
        b["load"].zero_()
        if b["load"].numel() == int(J.n):
            H._adh_load_acc = b["load"]                                      # what contact_finder divides by the substep count at the next frame
            if self._nsub is not None:
                H._adh_load_n = int(self._nsub)

    def forward(self, H, mask=None):
        P = H.level(self.at)
        if self._bufs is None:
            if _capturing():
                raise RuntimeError("bond_springs[warp]: first call inside a CUDA-graph capture")
            self._setup(H, P)
        frame = getattr(H, "frame", None)
        inb = _in_block(H)
        if not inb or frame is None or frame != self._frame:
            if _capturing():
                raise RuntimeError("bond_springs[warp]: the substep block is being captured, but the pair list was not refreshed at the frame level. Name `bond_springs` "
                                   "in the schedule once BEFORE the substep block as well as inside it (a replayed graph runs no Python: the pairs would never change)")
            self._refresh(H, P)
            self._frame, self._at_frame_level = frame, not inb
        if not inb:
            return {}
        if _capturing() and not self._at_frame_level:
            raise RuntimeError("bond_springs[warp]: the substep block is being captured, but `bond_springs` is not named in the schedule before the block (see the module docstring)")
        b = self._bufs
        dev = b["acc"].device
        m = P.mass if (P.mass.dtype == torch.float32 and P.mass.is_contiguous()) else P.mass.to(torch.float32).contiguous()
        b["acc"].zero_()
        _wp_launch(_pair_tether, int(b["row"].numel()), dev,
                   [wp.from_torch(P.state), int(P.state_schema["pos"][0]), wp.from_torch(m), wp.from_torch(b["pi"]), wp.from_torch(b["pj"]), wp.from_torch(b["row"]),
                    wp.from_torch(b["rho"]), self._k[0], self._k[1], self._k[2], wp.vec3(*self._per), wp.from_torch(b["acc"], dtype=wp.vec3), wp.from_torch(b["load"])])
        if self._nsub is None and not _capturing():                         # no spec at hand: count the calls, as the default does
            H._adh_load_n = int(getattr(H, "_adh_load_n", 0)) + 1
        return {self.at: b["acc"] if mask is None else b["acc"] * mask[:, None].to(torch.float32)}


@register_operator("apical_junction", implementation="warp", family="mechanics", set="particle", kind="lateral",
                   title="bond surface cells into a sheet, the springs as one Warp kernel",
                   equation=r"""$$\mathbf F_{ij}=k_j\,(\lVert\mathbf x_j-\mathbf x_i\rVert-L^0_{ij})\,\frac{\mathbf x_j-\mathbf x_i}{\lVert\mathbf x_j-\mathbf x_i\rVert},\quad \mathbf a_i=\sum_j\mathbf F_{ij}/m_i$$""")
class ApicalJunctionWarp(ApicalJunction):
    r"""apical_junction with the springs' forces as one Warp kernel over the bond list (built at the first call by the default's own search, on the host).
    particle -> particle: as apical_junction (same bonds, same parameters); an acceleration F / m on both ends of every bond, equal and opposite.

        \mathbf F_{ij}=k_j\,(\lVert\mathbf x_j-\mathbf x_i\rVert-L^0_{ij})\,\frac{\mathbf x_j-\mathbf x_i}{\lVert\mathbf x_j-\mathbf x_i\rVert},\quad \mathbf a_i=\sum_j\mathbf F_{ij}/m_i
        F_{ij}         the force on point i from its bond to j
        k_j            `k`, the junction stiffness
        x_i, x_j       the two positions
        L^0_{ij}       the bond's rest length, the distance at the first call
        a_i            the acceleration
        m_i            the point's mass

    Reference: as apical_junction; Plexus exp09 speed (this work, 2026-11-19)."""
    MECHANISM_TAGS = ["adherens_junction", "epithelial_sheet", "bond_network", "fused_kernel"]
    DIFFERENTIABLE = False

    def forward(self, H, mask=None):
        P = H.level(self.at)
        X = P.get("pos")
        _need_cuda32("apical_junction[warp]", X)
        dev, n = X.device, int(P.n)
        if self.bonds is None:
            if _capturing():
                raise RuntimeError("apical_junction[warp]: first call inside a CUDA-graph capture (the bonds are found on the host at the first, eager, call)")
            self._build(H, P, X)
            i, j, L0 = self.bonds
            self._w = (i.to(torch.int32).contiguous(), j.to(torch.int32).contiguous(), L0.to(torch.float32).contiguous())
            self._acc = torch.zeros(n, 3, dtype=torch.float32, device=dev)
        bi, bj, l0 = self._w
        self._acc.zero_()
        if int(bi.numel()):
            m = P.mass if (P.mass.dtype == torch.float32 and P.mass.is_contiguous()) else P.mass.to(torch.float32).contiguous()
            _wp_launch(_fixed_spring, int(bi.numel()), dev, [wp.from_torch(P.state), int(P.state_schema["pos"][0]), wp.from_torch(m), wp.from_torch(bi), wp.from_torch(bj),
                                                             wp.from_torch(l0), float(self.k), wp.from_torch(self._acc, dtype=wp.vec3)])
        return {self.at: self._acc if mask is None else self._acc * mask[:, None].to(torch.float32)}


@register_operator("pair_potential", implementation="cell_contact_warp_pp_fixed", family="interaction", set="particle", kind="exchange",
                   title="keep cells apart, the overlap penalty on each cell's own block of lattice nodes, Warp kernels with no host round-trip",
                   equation=r"""$$E_{\rm overlap}=\Delta x^3\sum_n\tfrac{k}{2}\,(P_n-s_0)_+^2,\quad P_n=\tfrac12\Big[\big(\textstyle\sum_A\phi_A(n)\big)^2-\sum_A\phi_A(n)^2\Big],\qquad \mathbf F_i=-\frac{\partial E}{\partial\mathbf x_i}$$""")
class CellContactWarpPPFixed(CellContactWarpPP):
    r"""pair_potential[cell_contact_warp_pp] with nothing read back to the host during a substep, so a CUDA graph can replay it: each cell's block origin is found by a
    kernel (an atomic minimum over its points), the penalty dU is formed inside the force kernel, and the block's size is fixed between frames.
    particle -> particle: as cell_contact_warp_pp (same parameters, same refusals, and `repulsion_strength_block` refused too); `block_margin` (default 1) lattice nodes are
    added to the widest cell's span when the blocks are sized; `track_max: true` publishes H.cell_contact_max (the largest overlap of the frame's last substep) once per frame.

        E_{\rm overlap}=\Delta x^3\sum_n\tfrac{k}{2}\,(P_n-s_0)_+^2,\quad P_n=\tfrac12\Big[\big(\textstyle\sum_A\phi_A(n)\big)^2-\sum_A\phi_A(n)^2\Big],\qquad \mathbf F_i=-\frac{\partial E}{\partial\mathbf x_i}
        E_{overlap}    the overlap penalty: zero until two cells overlap by more than s_0 at a lattice node, then quadratic in the excess
        P_n            the overlap at lattice node n
        \phi_A(n)      cell A's volume fraction at node n
        k              `repulsion_strength`
        s_0            `contact_slack`
        \Delta x       the lattice spacing (`dx`, else the grid's)
        F_i, x_i       the force on material point i and its position; F_i / m_i is the acceleration handed to the MPM substep

    THE BLOCK SIZE. Sized at the first call from the widest cell (as cell_contact_warp_pp does at every call) plus `block_margin`. During the substeps a cell that has
    grown wider than its block leaves the nodes beyond it out (of its own fraction and of the world's sums alike) and the width it needs is recorded on the device;
    once per frame, at the frame-level call, that record is read: the blocks are then enlarged, which reallocates them, and the engine drops a captured graph and runs
    the block eagerly from there on (it says so). In the schedule: once before the substep block and once inside it (see the module docstring).

    Reference: as cell_contact_warp_pp; Plexus exp09 speed (this work, 2026-11-19)."""
    PARAM_UNITS = {"block": "count", "block_margin": "count"}
    DIFFERENTIABLE = False

    def __init__(self, params, device="cpu"):
        super().__init__({k: v for k, v in params.items() if k not in ("block_margin", "track_max")}, device)
        self.block_margin = int(params.get("block_margin", 1))
        self.track_max = bool(params.get("track_max", False))    # True: H.cell_contact_max, the largest overlap, published once per frame (a pass over the lattice and a host read; nothing in the library reads it)
        self._b = None
        self._frame = None
        self._at_frame_level = False

    def _shape(self, H, dx):
        if self.periodic:
            shape = [int(round(float(w) / dx)) for w in H.world_size]
            if any(abs(float(w) / dx - n_) > 1e-4 for w, n_ in zip(H.world_size, shape)):
                raise ValueError(f"pair_potential[cell_contact_warp_pp_fixed]: periodic, but the world {[float(w) for w in H.world_size]} is not a whole number of lattice spacings {dx}")
            return shape
        return [int(math.ceil(float(w) / dx)) + 3 for w in H.world_size]

    def _span(self, H, P, dx, shape, nC):
        """The widest cell's span in lattice nodes along an axis, on the host (the first call only): cell_contact_warp_pp's own measure."""
        X = P.get("pos").detach()
        live = P.occ > 0
        cid = H.lift_index(self.at, self.cells).long()
        base = torch.floor(X / dx - 0.5).long()
        if self.periodic:
            lo, hi = base, base + 2
        else:
            lo = torch.stack([base[:, k].clamp(0, shape[k] - 1) for k in range(3)], 1)
            hi = torch.stack([(base[:, k] + 2).clamp(0, shape[k] - 1) for k in range(3)], 1)
        big = torch.iinfo(torch.int64).max
        org = torch.full((nC, 3), big, dtype=torch.long, device=X.device)
        lc, ll, lh = cid[live], lo[live], hi[live]
        org.scatter_reduce_(0, lc[:, None].expand(-1, 3), ll, reduce="amin", include_self=True)
        return int((lh - org[lc]).max()) + 1 if lc.numel() else 0

    def _alloc(self, P, B):
        b = self._b
        b["B"] = int(B)
        b["phib"] = _own_buffer(P, f"_contact_blocks_{id(self)}", torch.zeros((b["nC"], int(B) ** 3), dtype=torch.float32, device=b["dev"]))

    def _setup(self, H, P):
        X = P.get("pos")
        self._refuse(X.dtype, X.device)
        if self.stiffness_block:
            raise ValueError("pair_potential[cell_contact_warp_pp_fixed] does not do `repulsion_strength_block` -- use pair_potential[cell_contact_warp_pp]")
        dev, n = X.device, int(P.n)
        C = H.level(self.cells)
        nC = int(C.state.shape[0])
        dx = float(self.dx) if self.dx is not None else float(H.field(self.grid).dx)
        shape = self._shape(H, dx)
        Nw = shape[0] * shape[1] * shape[2]
        span = self._span(H, P, dx, shape, nC)
        self._b = dict(dev=dev, n=n, nC=nC, dx=dx, shape=shape, dx3=float(torch.tensor(dx, dtype=torch.float32) ** 3),
                       s1=torch.zeros(Nw, dtype=torch.float32, device=dev), s2=torch.zeros(Nw, dtype=torch.float32, device=dev),
                       org=torch.zeros((nC, 3), dtype=torch.int32, device=dev), span=torch.zeros(1, dtype=torch.int32, device=dev),
                       acc=torch.zeros((n, 3), dtype=torch.float32, device=dev), cid=H.lift_index(self.at, self.cells).to(torch.int32).contiguous(),
                       k=float(self.k), c0=float(self.c0))
        self._alloc(P, max(self.block, span + self.block_margin))
        print(f"[cell_contact_warp_pp_fixed] blocks of {self._b['B']}^3 lattice nodes (the widest cell touches {span} along an axis, margin {self.block_margin}), "
              f"{nC:,} cells, lattice {shape[0]} x {shape[1]} x {shape[2]}", flush=True)

    def _frame_check(self, H, P):
        """Once per frame, on the host: did a cell outgrow its block during the last frame's substeps (then enlarge the blocks); the largest overlap of the last substep."""
        b = self._b
        need = int(b["span"][0])
        if need > b["B"]:
            new = need + self.block_margin
            print(f"[cell_contact_warp_pp_fixed] frame {getattr(H, 'frame', '?')}: a cell's points touch {need} lattice nodes along an axis, more than its block of {b['B']}: "
                  f"blocks of {new}^3 from here (the nodes beyond the block were left out of the penalty during the last frame; a captured graph is dropped)", flush=True)
            self._alloc(P, new)
        b["span"].zero_()
        if self.track_max:
            H.cell_contact_max = float((0.5 * (b["s1"] * b["s1"] - b["s2"])).max()) if b["s1"].numel() else 0.0

    def forward(self, H, mask=None):
        P = H.level(self.at)
        if self._b is None:
            if _capturing():
                raise RuntimeError("pair_potential[cell_contact_warp_pp_fixed]: first call inside a CUDA-graph capture")
            self._setup(H, P)
        frame = getattr(H, "frame", None)
        inb = _in_block(H)
        if not inb or frame is None or frame != self._frame:
            if _capturing():
                raise RuntimeError("pair_potential[cell_contact_warp_pp_fixed]: the substep block is being captured, but the operator was not called at the frame level. Name "
                                   "`pair_potential` in the schedule once BEFORE the substep block as well as inside it (the block-size check runs there)")
            self._frame_check(H, P)
            self._frame, self._at_frame_level = frame, not inb
        if not inb:
            return {}
        if _capturing() and not self._at_frame_level:
            raise RuntimeError("pair_potential[cell_contact_warp_pp_fixed]: the substep block is being captured, but `pair_potential` is not named in the schedule before the block")
        b = self._b
        dev, n, nC, B = b["dev"], b["n"], b["nC"], b["B"]
        nx, ny, nz = b["shape"]
        per = int(self.periodic)
        pa = int(P.state_schema["pos"][0])
        st = wp.from_torch(P.state)
        occ = wp.from_torch(P.occ.contiguous())
        pv = wp.from_torch(P.p_vol if (P.p_vol.dtype == torch.float32 and P.p_vol.is_contiguous()) else P.p_vol.to(torch.float32).contiguous())
        m = wp.from_torch(P.mass if (P.mass.dtype == torch.float32 and P.mass.is_contiguous()) else P.mass.to(torch.float32).contiguous())
        cid = wp.from_torch(b["cid"])
        org3 = wp.from_torch(b["org"], dtype=wp.vec3i)
        phib, s1, s2 = wp.from_torch(b["phib"]), wp.from_torch(b["s1"]), wp.from_torch(b["s2"])
        b["org"].fill_(2 ** 30)                                              # above any node: a cell with no live point keeps it and is never read
        b["phib"].zero_()
        b["s1"].zero_()
        b["s2"].zero_()
        _wp_launch(_cc_origin, n, dev, [st, pa, occ, cid, nx, ny, nz, b["dx"], per, wp.from_torch(b["org"])])
        _wp_launch(_cc_deposit_g, n, dev, [st, pa, occ, pv, cid, org3, nx, ny, nz, B, b["dx"], b["dx3"], per, phib, s1, wp.from_torch(b["span"])])
        _wp_launch(_cc_square, (nC, B ** 3), dev, [phib, org3, nx, ny, nz, B, per, s2])
        _wp_launch(_cc_force_g, n, dev, [st, pa, occ, pv, m, cid, org3, nx, ny, nz, B, b["dx"], per, b["k"], b["c0"], phib, s1, s2, wp.from_torch(b["acc"], dtype=wp.vec3)])
        return {self.at: b["acc"] if mask is None else b["acc"] * mask[:, None].to(torch.float32)}


@register_operator("mpm_scatter", implementation="warp_pp_lean", family="mpm", set="particle", kind="exchange",
                   title="spread point mass onto the per-cell grid blocks, the block origins from a kernel", equation=r"""$$\boldsymbol\sigma = 2\mu(\mathbf F-\mathbf R)\mathbf F^{\!\top} + \lambda(J-1)J\,\mathbf I,
\qquad
\mathbf Q = -\frac{4\,\Delta t}{\Delta x^{2}}\,V\,\boldsymbol\sigma + m\,\mathbf C$$""")
class MPMScatterWarpPPLean(MPMScatterWarpPP):
    r"""mpm_scatter[warp_pp] (the same deposit kernel) with each block's origin, its body's centroid less half a block, found by one kernel per body instead of torch sums
    and a device tensor built from a Python list (which a CUDA-graph capture refuses), and with the liquid-colour channel cleared only when the set holds liquid.
    particle -> field: as mpm_scatter[warp_pp]. The centroid is summed in row order, so it is the same on every run (the default's index_add_ is not).

        \boldsymbol\sigma = 2\mu(\mathbf F-\mathbf R)\mathbf F^{\!\top} + \lambda(J-1)J\,\mathbf I, \qquad \mathbf Q = -\frac{4\,\Delta t}{\Delta x^{2}}\,V\,\boldsymbol\sigma + m\,\mathbf C
        \sigma         the Kirchhoff stress of the fixed-corotated law
        \mu, \lambda   the particle's Lame parameters
        F              the deformation gradient
        R              its rotation
        J              det F
        Q              the affine momentum matrix deposited with the particle
        \Delta t       the substep
        \Delta x       the grid spacing
        V              the particle's rest volume
        m              its mass
        C              its affine velocity gradient (APIC)

    Needs every body's points in equal consecutive slot blocks (`per_parent` an integer); otherwise the origins are the default's torch sums with the half-block cached.

    Reference: as mpm_scatter[warp_pp]; Plexus exp09 speed (this work, 2026-11-19)."""
    MECHANISM_TAGS = ["particle_to_grid", "fixed_corotated_stress", "per_parent_blocks", "fused_kernel"]

    def _origins(self, H, g, p, recentre):
        """(the block index of every point, the block origins [n_blocks, 3]) as mpm_ops._pp_frame gives them, in a buffer of this operator that the gather reads through g.pp_origin."""
        pidx = H.lift_index(p.name, g.per_parent)
        nb = int(H.level(g.per_parent).state.shape[0])
        g.ensure_blocks(nb)
        dev = p.state.device
        c = getattr(self, "_oc", None)
        if c is None or c["nb"] != nb:
            if _capturing():
                raise RuntimeError("mpm_scatter[warp_pp_lean]: the number of blocks changed inside a CUDA-graph capture")
            per = int(p.n) // max(nb, 1)
            even = per * nb == int(p.n) and bool((pidx == torch.arange(nb, device=dev).repeat_interleave(per)).all())
            c = self._oc = dict(nb=nb, per=per, even=even, org=torch.zeros(nb, 3, dtype=torch.float32, device=dev), half=0.5 * float(g.world_size[0]),
                                half_t=0.5 * torch.as_tensor(g.world_size, device=dev, dtype=torch.float32), pid=pidx.contiguous())
            if any(abs(float(w) - float(g.world_size[0])) > 0 for w in g.world_size):
                c["even"] = False                                            # a block that is not a cube: the per-axis half, in torch
            recentre = True
        if recentre or g.pp_origin is None or g.pp_origin.shape[0] != nb:
            if c["even"]:
                _wp_launch(_pp_origin, nb, dev, [wp.from_torch(p.state), int(p.state_schema["pos"][0]), wp.from_torch(p.occ.contiguous()), c["per"], c["half"],
                                                 wp.from_torch(c["org"], dtype=wp.vec3)])
            else:
                X = p.get("pos")
                w = (p.occ > 0).to(X.dtype)
                cnt = torch.zeros(nb, device=dev, dtype=X.dtype).index_add_(0, pidx, w)
                cen = torch.zeros(nb, 3, device=dev, dtype=X.dtype).index_add_(0, pidx, X * w[:, None])
                c["org"].copy_(cen / cnt.clamp_min(1.0)[:, None] - c["half_t"])
            g.pp_origin = c["org"]
        return c["pid"], g.pp_origin

    def _clear_grid(self, H, g, p, recentre, has_liq, fixed):
        """The first scatter of a substep into this grid clears it: every node's mass and momentum (the colour only when the set holds liquid: nothing else writes it)."""
        if recentre:
            g.m.zero_(); g.mv.zero_()
            if has_liq:
                g.c.zero_()

    def _deposit(self, p, dev, args):
        _wp_launch(p2g_pp, int(p.n), dev, args)

    def forward(self, H, mask=None):
        p = H.level(self.at); g = H.field(self.grid); dev = p.state.device
        _pp_refuse(self, g, p, H, "mpm_scatter[warp_pp_lean]")
        if getattr(self, "store_stress", False) or getattr(p, "F_res_inv", None) is not None or self.polar != "higham" or self.body_force != "particle":
            raise ValueError("mpm_scatter[warp_pp_lean]: `store_stress`, a rest tensor `F_res_inv`, `polar` other than higham and `body_force: grid` are not implemented; use the default body")
        dt = sub_dt(H, self.dt_sub)
        pa, _ = p.state_schema["pos"]; va, _ = p.state_schema["vel"]
        a_ext = _ancestor_accel(H, p, self.a_max, 3, dev)
        part_accel = getattr(H, "part_accel", None)
        if part_accel is not None:
            a_ext = a_ext + part_accel
        a_ext = (a_ext + torch.nan_to_num(H.delta(p.name))).contiguous()
        has_snw = _const_any(self, "_c_snow", getattr(p, "is_snow", None))
        has_liq = _const_any(self, "_c_liquid", getattr(p, "is_liquid", None))
        if getattr(self, "_sbuf", None) is None:                         # run-constant side arrays, built once
            z = torch.zeros(p.n, device=dev)

            def _mf(t):
                return t.float().contiguous() if t is not None else z
            self._sbuf = (_mf(getattr(p, "is_snow", None)), _mf(getattr(p, "is_liquid", None)), torch.zeros(1, 3, 3, device=dev), torch.zeros(1, dtype=torch.int64, device=dev))
        snw, liq, no_xtr, no_fixed = self._sbuf
        if getattr(self, "colour_from_face", None) is not None:
            liq, has_liq = self._face_colour(p, H).float().contiguous(), True
        # the additive Kirchhoff stresses, as the parent: active (per set or global), viscous, turgor
        xtr = getattr(p, "act_stress", None)
        gact = getattr(H, "active_stress", None)
        if gact is not None and gact.shape[0] == p.n:
            xtr = gact if xtr is None else xtr + gact
        ext = getattr(p, "extra_stress", None)
        if ext is None:
            gx = getattr(H, "extra_stress", None)
            ext = gx if (gx is not None and gx.shape[0] == p.n) else None
        if ext is not None:
            xtr = ext if xtr is None else xtr + ext
        turg = getattr(p, "turgor", None)
        if turg is not None:
            eye = torch.eye(3, device=dev)
            xtr = -eye * turg[:, None, None] if xtr is None else xtr - eye * turg[:, None, None]
        has_xtr = xtr is not None
        xtr = xtr.contiguous() if has_xtr else no_xtr
        _recentre = getattr(self, "_zeroes_grid", True)
        g.ensure_blocks(int(H.level(g.per_parent).state.shape[0]))           # the blocks' arrays, before anything takes them
        fixed = bool(torch.are_deterministic_algorithms_enabled())
        self._clear_grid(H, g, p, _recentre, has_liq, fixed)
        pidx, org = self._origins(H, g, p, _recentre)
        gm, gmv, gc = g.m, g.mv, g.c
        if fixed:
            if has_liq or not _recentre:
                raise ValueError("mpm_scatter[warp_pp_lean]: strict determinism with a liquid colour field or a second scatter into the same grid is not implemented")
            nn_ = g.m.numel()
            if getattr(self, "_fixed_buf", None) is None or self._fixed_buf[0].numel() != nn_:
                self._fixed_buf = (torch.zeros(nn_, dtype=torch.int64, device=dev), torch.zeros(3 * nn_, dtype=torch.int64, device=dev))
            gmi, gmvi = self._fixed_buf
            gmi.zero_(); gmvi.zero_()
        else:
            gmi = gmvi = no_fixed
        self._deposit(p, dev, [wp.from_torch(p.state), int(pa), int(va),
                               wp.from_torch(p.C.contiguous(), dtype=wp.mat33), wp.from_torch(p.F.contiguous(), dtype=wp.mat33),
                               wp.from_torch(p.mass.contiguous()), wp.from_torch(p.mu.contiguous()), wp.from_torch(p.la.contiguous()),
                               wp.from_torch(p.p_vol.contiguous()), wp.from_torch(a_ext, dtype=wp.vec3),
                               wp.from_torch(p.Jp.contiguous()), wp.from_torch(snw), wp.from_torch(liq), wp.from_torch(p.occ.contiguous()),
                               wp.from_torch(xtr, dtype=wp.mat33),
                               wp.from_torch(pidx), wp.from_torch(org, dtype=wp.vec3),
                               wp.from_torch(gm), wp.from_torch(gmv.view(-1, 3), dtype=wp.vec3), wp.from_torch(gc),
                               wp.from_torch(gmi), wp.from_torch(gmvi), float(FIXED_SCALE),
                               int(g.shape[0]), int(g.shape[1]), int(g.shape[2]), int(g.n_local),
                               float(g.dx), float(g.inv_dx), float(dt), float(self.drag), int(self.polar_iters),
                               int(has_snw), int(has_liq), int(has_xtr), int(fixed)])
        if fixed:
            _wp_launch(fixed_to_float, int(g.m.numel()), dev,
                       [wp.from_torch(gmi), wp.from_torch(gmvi), float(1.0 / FIXED_SCALE),
                        wp.from_torch(gm), wp.from_torch(gmv.view(-1, 3), dtype=wp.vec3)])
        return {}


@register_operator("mpm_grid_update", implementation="warp_pp_plain", family="mpm", set="field", kind="field",
                   title="solve grid velocities on the per-cell blocks, one in-place kernel when there is no surface tension",
                   equation=r"""$$\mathbf v_g=\frac{\mathbf p_g}{m_g}\ \ (m_g>m_{\rm floor}),\qquad \mathbf v_g=0\ \ \text{otherwise}$$""")
class MPMGridUpdateWarpPPPlain(MPMGridUpdateWarpPPBalanced):
    r"""mpm_grid_update[warp_pp_balanced] whose case WITHOUT surface tension (`cell_surface_tension` 0, where warp_pp_balanced falls back to the torch body: four passes over
    every node and a new velocity array per call) is one Warp kernel writing the velocity in place. With a surface tension it is warp_pp_balanced unchanged.
    field -> field: reads the blocks' mass `m` and momentum `mv`, writes their velocity `v` in place; the same float32 numbers as the torch body (one division per component).

        \mathbf v_g=\frac{\mathbf p_g}{m_g}\ \ (m_g>m_{\rm floor}),\qquad \mathbf v_g=0\ \ \text{otherwise}
        v_g            the velocity of grid node g
        p_g            its momentum
        m_g            its mass
        m_{floor}      `mass_floor`, below which a node is empty

    Reference: as mpm_grid_update[warp_pp_balanced]; Plexus exp09 speed (this work, 2026-11-19)."""
    MECHANISM_TAGS = ["grid_solve", "per_body_blocks", "fused_kernel"]

    def forward(self, H, mask=None):
        g = H.field(self.at)
        dev = g.m.device
        if self.surface_tension > 0.0 or not HAVE_WARP or not getattr(g, "per_parent", None) or g.dim != 3 or dev.type != "cuda":
            return super().forward(H, mask)
        dt = sub_dt(H, self.dt_sub)
        floor = self._const("mass_floor", g)
        if getattr(self, "_none", None) is None:
            self._none = torch.zeros(1, 3, device=dev)                   # stands for the arrays the kernel does not read when there is no tension
        bf = getattr(H, "_mpm_body_accel", None)
        none3 = wp.from_torch(self._none, dtype=wp.vec3)
        _wp_launch(pp_final, int(g.m.numel()), dev, [wp.from_torch(g.m), wp.from_torch(g.mv, dtype=wp.vec3),
                                                     wp.from_torch(bf.view(-1, 3), dtype=wp.vec3) if bf is not None else none3, none3, none3,
                                                     wp.from_torch(g.v, dtype=wp.vec3), int(g.n_local), float(floor), float(dt), int(bf is not None), 0])
        return {}


@register_operator("mpm_scatter", implementation="warp_pp_sparse", family="mpm", set="particle", kind="exchange",
                   title="spread point mass onto the per-cell grid blocks, clearing only the nodes the points touched", equation=r"""$$\boldsymbol\sigma = 2\mu(\mathbf F-\mathbf R)\mathbf F^{\!\top} + \lambda(J-1)J\,\mathbf I,
\qquad
\mathbf Q = -\frac{4\,\Delta t}{\Delta x^{2}}\,V\,\boldsymbol\sigma + m\,\mathbf C$$""")
class MPMScatterWarpPPSparse(MPMScatterWarpPPLean):
    r"""mpm_scatter[warp_pp_lean] that does not zero every node of every block at each substep (16 bytes x 3,375 nodes x bodies: 300 MB for 5,545 bodies of at most 64
    points, which touch a tenth of their block): each point records the 27 nodes it deposited on, and the next scatter clears those, one kernel over the points.
    particle -> field: as mpm_scatter[warp_pp] (the deposit kernel is its copy plus that record). The cost of a substep then follows the points, not `n_grid`^3.
    MEASURED SLOWER than warp_pp_lean (exp09 rna1_3d on an H200: 0.110 against 0.093 s per frame at 5,545 bodies, 0.315 against 0.270 at 19,995): clearing 27 nodes per
    point (0.23 ms per substep) costs more than zeroing all 18.7 million nodes of the blocks. Kept for a grid too large to sweep.
    Refused: strict determinism (the fixed-point deposit), a second scatter into the same grid.

        \boldsymbol\sigma = 2\mu(\mathbf F-\mathbf R)\mathbf F^{\!\top} + \lambda(J-1)J\,\mathbf I, \qquad \mathbf Q = -\frac{4\,\Delta t}{\Delta x^{2}}\,V\,\boldsymbol\sigma + m\,\mathbf C
        \sigma, \mu, \lambda, F, R, J, Q, \Delta t, \Delta x, V, m, C   as mpm_scatter[warp_pp]

    The record (two per-point buffers) is registered on the point set: the engine's capture snapshot restores it with the grid it describes.

    Reference: as mpm_scatter[warp_pp]; Plexus exp09 speed (this work, 2026-11-19)."""
    MECHANISM_TAGS = ["particle_to_grid", "fixed_corotated_stress", "per_parent_blocks", "fused_kernel", "sparse_clear"]

    def _clear_grid(self, H, g, p, recentre, has_liq, fixed):
        if fixed or not recentre:
            raise ValueError("mpm_scatter[warp_pp_sparse]: strict determinism and a second scatter into the same grid are not implemented; use mpm_scatter[warp_pp_lean]")
        dev = p.state.device
        r = getattr(self, "_rec", None)
        if r is None or r["n"] != int(p.n) or r["nodes"] != int(g.m.numel()):
            if _capturing():
                raise RuntimeError("mpm_scatter[warp_pp_sparse]: first call (or a resized set) inside a CUDA-graph capture")
            self._rec = dict(n=int(p.n), nodes=int(g.m.numel()),
                             base=_own_buffer(p, f"_scatter_base_{id(self)}", torch.zeros(int(p.n), 3, dtype=torch.int32, device=dev)),
                             blk=_own_buffer(p, f"_scatter_block_{id(self)}", torch.full((int(p.n),), -1, dtype=torch.int32, device=dev)))
            g.m.zero_(); g.mv.zero_(); g.c.zero_()                          # once: from here on the grid is cleared node by node
            return
        r = self._rec
        _wp_launch(_pp_clear, int(p.n), dev, [wp.from_torch(r["base"], dtype=wp.vec3i), wp.from_torch(r["blk"]), int(g.shape[0]), int(g.shape[1]), int(g.shape[2]),
                                              int(g.n_local), int(has_liq), wp.from_torch(g.m), wp.from_torch(g.mv.view(-1, 3), dtype=wp.vec3), wp.from_torch(g.c)])

    def _deposit(self, p, dev, args):
        r = self._rec
        _wp_launch(p2g_pp_b, int(p.n), dev, args[:20] + args[23:35] + [wp.from_torch(r["base"], dtype=wp.vec3i), wp.from_torch(r["blk"])])    # without the fixed-point arrays, their scale and flag


@register_operator("mpm_gather", implementation="warp_pp_solve", family="mpm", set="particle", kind="exchange",
                   title="move points with the per-cell grid blocks, the node velocities solved where they are read", equation=r"""$$\mathbf v_g=\frac{\mathbf p_g}{m_g},\qquad \mathbf v_i = \sum_g w_{ig}\,\mathbf v_g,
\qquad
\mathbf C_i = \frac{4}{\Delta x^{2}}\sum_g w_{ig}\,\mathbf v_g\,(\mathbf x_g-\mathbf x_i)^{\!\top},
\qquad
\mathbf x_i \mathrel{+}= \Delta t\,\mathbf v_i$$""")
class MPMGatherWarpPPSolve(MPMGatherWarpPP):
    r"""mpm_gather[warp_pp] that reads each node's momentum and mass and divides them itself, at the 27 nodes a point reads, instead of a velocity that a grid solve wrote
    on every node of every block beforehand. With mpm_grid_update[solved_in_gather] no pass over the nodes is left in the substep.
    field -> particle: as mpm_gather[warp_pp]; the grid solve it stands for is the one WITHOUT surface tension, body force on the grid or walls (v = p / m above `mass_floor`).

        \mathbf v_g=\frac{\mathbf p_g}{m_g},\qquad \mathbf v_i = \sum_g w_{ig}\,\mathbf v_g, \qquad \mathbf C_i = \frac{4}{\Delta x^{2}}\sum_g w_{ig}\,\mathbf v_g\,(\mathbf x_g-\mathbf x_i)^{\!\top}, \qquad \mathbf x_i \mathrel{+}= \Delta t\,\mathbf v_i
        v_g, p_g, m_g  the velocity, momentum and mass of node g (v_g = 0 where m_g is not above `mass_floor`, the grid solve's own floor)
        v_i, C_i, x_i  the particle's velocity, affine matrix and position
        w_{ig}         the quadratic B-spline weight of node g at particle i
        \Delta x, \Delta t   the grid spacing and the substep

    Reference: as mpm_gather[warp_pp]; Plexus exp09 speed (this work, 2026-11-19)."""
    MECHANISM_TAGS = ["grid_to_particle", "advection", "per_parent_blocks", "fused_kernel", "grid_solve"]

    def _floor(self, H, g):
        """The grid solve's `mass_floor` for this grid: the scheduled mpm_grid_update's own (it must have no surface tension), read once."""
        f = getattr(self, "_fl", None)
        if f is None:
            from plexus.operators.mpm_ops import _scale_constant
            f = _scale_constant("mass_floor", float(g.dx), 1.0)
            for nm, ob in zip(getattr(H, "operator_names", []) or [], getattr(H, "operators", []) or []):
                if nm == "mpm_grid_update" and getattr(ob, "at", None) == self.frm:
                    if float(getattr(ob, "surface_tension", 0.0)) > 0.0:
                        raise ValueError("mpm_gather[warp_pp_solve]: the grid solve has a surface tension, which this gather does not apply; use mpm_gather[warp_pp]")
                    f = float(ob._const("mass_floor", g))
            self._fl = f
        return f

    def forward(self, H, mask=None):
        p = H.level(self.at); g = H.field(self.frm); dev = p.state.device
        _pp_refuse(self, g, p, H, "mpm_gather[warp_pp_solve]")
        if self.wall_damp != 1.0 or getattr(H, "_mpm_body_accel", None) is not None:
            raise ValueError("mpm_gather[warp_pp_solve]: `wall_damp` != 1 and a body force on the grid are not implemented; use mpm_gather[warp_pp]")
        dt = sub_dt(H, self.dt_sub)
        pa, _ = p.state_schema["pos"]; va, _ = p.state_schema["vel"]
        nb = int(H.level(g.per_parent).state.shape[0])
        if g.pp_origin is None or g.pp_origin.shape[0] != nb:
            raise RuntimeError("mpm_gather[warp_pp_solve]: no block origins yet (the scatter of this substep sets them)")
        if getattr(self, "_pid", None) is None or self._pid.shape[0] != int(p.n):
            self._pid = H.lift_index(p.name, g.per_parent).contiguous()
        dx = float(g.dx)
        vmax = min(self.vmax, 0.4 * dx / dt)
        _wp_launch(g2p_pp_mv, int(p.n), dev,
                   [wp.from_torch(p.state), int(pa), int(va), wp.from_torch(p.C, dtype=wp.mat33),
                    wp.from_torch(g.m), wp.from_torch(g.mv.view(-1, 3), dtype=wp.vec3), wp.from_torch(p.occ.contiguous()),
                    wp.from_torch(self._pid), wp.from_torch(g.pp_origin.contiguous(), dtype=wp.vec3),
                    int(g.shape[0]), int(g.shape[1]), int(g.shape[2]), int(g.n_local),
                    float(g.inv_dx), float(dt), float(vmax), float(2 * dx), float(float(g.world_size[0]) - 2 * dx), float(self._floor(H, g))])
        return {}


@register_operator("mpm_grid_update", implementation="solved_in_gather", family="mpm", set="field", kind="field",
                   title="the grid solve left to the gather: no pass over the nodes",
                   equation=r"""$$\mathbf v_g=\frac{\mathbf p_g}{m_g}\ \ (m_g>m_{\rm floor})\quad\text{at the nodes mpm\_gather[warp\_pp\_solve] reads}$$""")
class MPMGridUpdateSolvedInGather(MPMGridUpdateWarpPPPlain):
    r"""the grid solve of a per-body grid with no surface tension, left to mpm_gather[warp_pp_solve]: that gather divides momentum by mass at the nodes it reads, so
    this operator does nothing, and the schedule keeps the step the MPM cycle is checked for (strain, scatter, grid update, gather).
    field -> field: writes nothing (the grid's velocity `v` is not kept up to date). Refused at its first call unless every mpm_gather that reads this grid is
    mpm_gather[warp_pp_solve]; refused with a surface tension.

        \mathbf v_g=\frac{\mathbf p_g}{m_g}\ \ (m_g>m_{\rm floor})\quad\text{at the nodes mpm\_gather[warp\_pp\_solve] reads}
        v_g, p_g, m_g  the velocity, momentum and mass of node g
        m_{floor}      `mass_floor`, below which a node is empty

    Reference: as mpm_grid_update; Plexus exp09 speed (this work, 2026-11-19)."""
    MECHANISM_TAGS = ["grid_solve", "per_body_blocks", "deferred"]

    def forward(self, H, mask=None):
        if not getattr(self, "_ok", False):
            if self.surface_tension > 0.0:
                raise ValueError("mpm_grid_update[solved_in_gather]: a surface tension needs a grid solve; use mpm_grid_update[warp_pp_balanced]")
            bad = [type(ob).__name__ for nm, ob in zip(getattr(H, "operator_names", []) or [], getattr(H, "operators", []) or [])
                   if nm in ("mpm_gather", "g2p") and getattr(ob, "frm", None) == self.at and not isinstance(ob, MPMGatherWarpPPSolve)]
            if bad:
                raise ValueError(f"mpm_grid_update[solved_in_gather]: the grid {self.at!r} is read by {bad}, which need the solved velocity; give every mpm_gather of this grid "
                                 "`implementation: warp_pp_solve`, or use mpm_grid_update[warp_pp_plain]")
            self._ok = True
        return {}


@register_operator("pair_potential", implementation="cell_contact_warp_pp_sparse", family="interaction", set="particle", kind="exchange",
                   title="keep cells apart, the overlap penalty with every pass over the points and none over the lattice",
                   equation=r"""$$E_{\rm overlap}=\Delta x^3\sum_n\tfrac{k}{2}\,(P_n-s_0)_+^2,\quad P_n=\tfrac12\Big[\big(\textstyle\sum_A\phi_A(n)\big)^2-\sum_A\phi_A(n)^2\Big],\qquad \mathbf F_i=-\frac{\partial E}{\partial\mathbf x_i}$$""")
class CellContactWarpPPSparse(CellContactWarpPPFixed):
    r"""pair_potential[cell_contact_warp_pp_fixed] with no pass over every node of every cell's block or of the world's lattice: the sum of squares is built from the
    points (S2(n) += dep_p(n) phi_A(n), which over the points of cell A is phi_A(n)^2), and after the force the points set the nodes they touched back to zero, so the
    blocks and the two world sums are zero between calls. Five kernels over the points: block origins, deposit, squares, force, clear.
    particle -> particle: as cell_contact_warp_pp_fixed (same parameters, same block-size rule; `track_max: true` costs one more kernel per substep here and gives the
    largest overlap over all the frame's substeps). The cost of a call follows the points, not cells x block^3 + lattice nodes. MEASURED SLOWER than
    cell_contact_warp_pp_fixed (exp09 rna1_3d on an H200: 0.104 against 0.093 s per frame at 5,545 bodies, 0.302 against 0.270 at 19,995): a pass over the points' 27
    nodes costs more than the dense passes it replaces. Kept for a lattice too large to sweep.

        E_{\rm overlap}=\Delta x^3\sum_n\tfrac{k}{2}\,(P_n-s_0)_+^2,\quad P_n=\tfrac12\Big[\big(\textstyle\sum_A\phi_A(n)\big)^2-\sum_A\phi_A(n)^2\Big],\qquad \mathbf F_i=-\frac{\partial E}{\partial\mathbf x_i}
        E_{overlap}, P_n, \phi_A(n), k, s_0, \Delta x, F_i, x_i   as cell_contact_warp_pp

    Reference: as cell_contact_warp_pp; Plexus exp09 speed (this work, 2026-11-19)."""
    DIFFERENTIABLE = False

    def _frame_check(self, H, P):
        b = self._b
        need = int(b["span"][0])
        if need > b["B"]:
            new = need + self.block_margin
            print(f"[cell_contact_warp_pp_sparse] frame {getattr(H, 'frame', '?')}: a cell's points touch {need} lattice nodes along an axis, more than its block of {b['B']}: "
                  f"blocks of {new}^3 from here (the nodes beyond the block were left out of the penalty during the last frame; a captured graph is dropped)", flush=True)
            self._alloc(P, new)
        b["span"].zero_()
        if "pmax" not in b:
            b["pmax"] = torch.zeros(1, dtype=torch.float32, device=b["dev"])
        if self.track_max:
            H.cell_contact_max = float(b["pmax"][0])                         # the largest overlap over the last frame's substeps
            b["pmax"].zero_()

    def forward(self, H, mask=None):
        P = H.level(self.at)
        if self._b is None:
            if _capturing():
                raise RuntimeError("pair_potential[cell_contact_warp_pp_sparse]: first call inside a CUDA-graph capture")
            self._setup(H, P)
        frame = getattr(H, "frame", None)
        inb = _in_block(H)
        if not inb or frame is None or frame != self._frame:
            if _capturing():
                raise RuntimeError("pair_potential[cell_contact_warp_pp_sparse]: the substep block is being captured, but the operator was not called at the frame level. Name "
                                   "`pair_potential` in the schedule once BEFORE the substep block as well as inside it (the block-size check runs there)")
            self._frame_check(H, P)
            self._frame, self._at_frame_level = frame, not inb
        if not inb:
            return {}
        if _capturing() and not self._at_frame_level:
            raise RuntimeError("pair_potential[cell_contact_warp_pp_sparse]: the substep block is being captured, but `pair_potential` is not named in the schedule before the block")
        b = self._b
        dev, n, B = b["dev"], b["n"], b["B"]
        nx, ny, nz = b["shape"]
        per = int(self.periodic)
        pa = int(P.state_schema["pos"][0])
        st = wp.from_torch(P.state)
        occ = wp.from_torch(P.occ.contiguous())
        pv = wp.from_torch(P.p_vol if (P.p_vol.dtype == torch.float32 and P.p_vol.is_contiguous()) else P.p_vol.to(torch.float32).contiguous())
        m = wp.from_torch(P.mass if (P.mass.dtype == torch.float32 and P.mass.is_contiguous()) else P.mass.to(torch.float32).contiguous())
        cid = wp.from_torch(b["cid"])
        org3 = wp.from_torch(b["org"], dtype=wp.vec3i)
        phib, s1, s2 = wp.from_torch(b["phib"]), wp.from_torch(b["s1"]), wp.from_torch(b["s2"])
        b["org"].fill_(2 ** 30)
        _wp_launch(_cc_origin, n, dev, [st, pa, occ, cid, nx, ny, nz, b["dx"], per, wp.from_torch(b["org"])])
        _wp_launch(_cc_deposit_g, n, dev, [st, pa, occ, pv, cid, org3, nx, ny, nz, B, b["dx"], b["dx3"], per, phib, s1, wp.from_torch(b["span"])])
        _wp_launch(_cc_square_p, n, dev, [st, pa, occ, pv, cid, org3, nx, ny, nz, B, b["dx"], b["dx3"], per, phib, s2])
        _wp_launch(_cc_force_g, n, dev, [st, pa, occ, pv, m, cid, org3, nx, ny, nz, B, b["dx"], per, b["k"], b["c0"], phib, s1, s2, wp.from_torch(b["acc"], dtype=wp.vec3)])
        if self.track_max:
            _wp_launch(_cc_overlap_max, n, dev, [st, pa, occ, nx, ny, nz, b["dx"], per, s1, s2, wp.from_torch(b["pmax"])])
        _wp_launch(_cc_clear, n, dev, [st, pa, occ, cid, org3, nx, ny, nz, B, b["dx"], per, phib, s1, s2])
        return {self.at: b["acc"] if mask is None else b["acc"] * mask[:, None].to(torch.float32)}


from plexus.operators.adhesion_warp_ops import JunctionContactSites      # noqa: E402


@register_operator("contact_finder", implementation="sites_wide", family="topology", set="junction", kind="rewire",
                   title="detect contacts, each point counted in its site, the search grid sized to the world",
                   equation=r"""$$A_{a\,S(b),fg}=s^{2}\,\big\lvert\{\,i\in a:\ \exists j\in S(b),\ \lvert\mathbf x_i-\mathbf x_j\rvert<r_b,\ (f_i,f_j)=fg\}\big\rvert$$""")
class JunctionContactSitesWide(JunctionContactSites):
    r"""contact_finder[sites] with the hash grid of its neighbour search sized to the world instead of 64 cells per axis: 64 cells of the search radius (0.55 length units
    for exp09) cover 35 units, and a wider world wraps onto them, so that each query also walks the points of every other period (rejected by the exact distance test:
    the answer is the same, the search slower by the number of periods, about 5 for a world of 109 x 62 units).
    particle <-> particle: as contact_finder[sites], to the last bit (the same candidates pass the same test); `grid_max` (default 512) caps the cells per axis.

        A_{a\,S(b),fg}=s^{2}\,\big\lvert\{\,i\in a:\ \exists j\in S(b),\ \lvert\mathbf x_i-\mathbf x_j\rvert<r_b,\ (f_i,f_j)=fg\}\big\rvert
        A_{a S(b),fg}, S(b), s, r_b   as contact_finder[sites]

    Reference: as contact_finder[sites]; Plexus exp09 speed (this work, 2026-11-19)."""

    def __init__(self, params, device="cpu"):
        super().__init__({k: v for k, v in params.items() if k != "grid_max"}, device)
        self.grid_max = int(params.get("grid_max", 512))

    def forward(self, H, mask=None):
        dev = H.level(self.points).state.device
        if getattr(self, "_grid", None) is None and HAVE_WARP and dev.type == "cuda":
            rq = float(self.bond_reach + self.skin) * (1.0 + 1.0e-4) + 1.0e-5          # the search radius of adhesion_warp_ops._partners: the hash grid's cell
            dims = [int(min(max(math.ceil(float(w) / rq) + 1, 64), self.grid_max)) for w in H.world_size][:3]
            self._grid = wp.HashGrid(dims[0], dims[1], dims[2], device=f"cuda:{dev.index or 0}")
            print(f"[contact_finder:sites_wide] hash grid of {dims[0]} x {dims[1]} x {dims[2]} cells of {rq:.3f} (the default: 64 x 64 x 64)", flush=True)
        return super().forward(H, mask)
