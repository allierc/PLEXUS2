"""exp27 speed audit (2026-10-21), test 5c: `mpm_grid_update[implementation: warp_pp_balanced]`.

`mpm_grid_update[body_csf_balanced]` (the continuum surface force of each cell's own grid block, with each block's net push removed) as
Warp kernels on the per-cell blocks. The torch body does it as ~18 `torch.roll` and ~100 elementwise operations over all 625 blocks of
24^3 nodes (8.64M nodes), ~208 kernel launches per call and a fresh 104 MB velocity array per call (RTX A6000, 14.2 ms per call); here it
is 8 launches into scratch buffers allocated once, and the velocity is written into `g.v`'s own storage.

The arithmetic is the torch body's, operation by operation and in its order, so that the result is the same float32 numbers and not just
the same equations:
  * every kernel of this module is compiled with `fuse_fp: False` (no fused multiply-add: the torch body rounds every product);
  * `x / scalar` as the torch CUDA kernel does it, `x * (1 / scalar)`; `scalar / x` as `(1 / x) * scalar` (Tensor.__rtruediv__);
  * the smoothing is three axis passes in the torch order (axis 1, 2, 3), the wrap-around at the block faces is the `torch.roll` one
    (a block holds its cell with a two-node empty margin, so it reads zeros, but the wrap is kept so that the result does not depend on it);
  * the curvature sum is ((t1 + t2) + t3) and the gradient magnitude sqrt((q1^2 + q2^2) + q3^2), as Python's `sum` over the generator.
The per-block sums of the net-force removal, `sum_n m_n dv_n` and `sum_n m_n` over a block's 13,824 nodes, are the torch body's own
calls (`(m dv).sum(1)` on a [blocks, nodes, 3] array and `m.sum(1)`, from products a kernel rounds as the torch body does): a float sum
has no order of its own, and the torch reduction is deterministic, so using it is what makes the velocity the same float32 numbers.

What is not ported falls back to the parent's torch body: no surface tension, `csf_band: 0` (its interface test is a reduction over the
whole block), `csf_smooth: 0`, a grid that is not 3D / not per-parent / not on CUDA.
"""
from __future__ import annotations

import numpy as np
import torch

from plexus.models.registry import register_operator
from plexus.operators.adhesion_ops import MPMGridUpdateBodyCSFBalanced
from plexus.operators.mpm_ops import HAVE_WARP, _wp_launch, sub_dt

if HAVE_WARP:
    import warp as wp

    wp.set_module_options({"fuse_fp": False})

    @wp.func
    def _node(b: int, i: int, j: int, k: int, n_local: int, ny: int, nz: int) -> int:
        return b * n_local + (i * ny + j) * nz + k

    @wp.kernel
    def pp_smooth(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32), scale: float, axis: int,
                  nx: int, ny: int, nz: int, n_local: int):
        """dst = 0.25 roll(c, 1) + 0.5 c + 0.25 roll(c, -1) along `axis` (0, 1, 2) inside each block, c = src * scale."""
        t = wp.tid()
        b = t // n_local
        r = t - b * n_local
        i = r // (ny * nz)
        j = (r // nz) % ny
        k = r % nz
        im = i
        ip = i
        jm = j
        jp = j
        km = k
        kp = k
        if axis == 0:
            im = (i - 1 + nx) % nx
            ip = (i + 1) % nx
        elif axis == 1:
            jm = (j - 1 + ny) % ny
            jp = (j + 1) % ny
        else:
            km = (k - 1 + nz) % nz
            kp = (k + 1) % nz
        cm = src[_node(b, im, jm, km, n_local, ny, nz)] * scale
        c0 = src[t] * scale
        cp = src[_node(b, ip, jp, kp, n_local, ny, nz)] * scale
        dst[t] = 0.25 * cm + 0.5 * c0 + 0.25 * cp

    @wp.kernel
    def pp_normal(c: wp.array(dtype=wp.float32), nrm: wp.array(dtype=wp.vec3), eps: float, h: float,
                  nx: int, ny: int, nz: int, n_local: int):
        """n = grad c / (|grad c| + eps), central differences with the block's wrap-around."""
        t = wp.tid()
        b = t // n_local
        r = t - b * n_local
        i = r // (ny * nz)
        j = (r // nz) % ny
        k = r % nz
        ip = (i + 1) % nx
        im = (i - 1 + nx) % nx
        jp = (j + 1) % ny
        jm = (j - 1 + ny) % ny
        kp = (k + 1) % nz
        km = (k - 1 + nz) % nz
        g0 = (c[_node(b, ip, j, k, n_local, ny, nz)] - c[_node(b, im, j, k, n_local, ny, nz)]) * h
        g1 = (c[_node(b, i, jp, k, n_local, ny, nz)] - c[_node(b, i, jm, k, n_local, ny, nz)]) * h
        g2 = (c[_node(b, i, j, kp, n_local, ny, nz)] - c[_node(b, i, j, km, n_local, ny, nz)]) * h
        gmag = wp.sqrt(g0 * g0 + g1 * g1 + g2 * g2)
        d = gmag + eps
        nrm[t] = wp.vec3(g0 / d, g1 / d, g2 / d)

    @wp.kernel
    def pp_dv(gm: wp.array(dtype=wp.float32), c: wp.array(dtype=wp.float32), nrm: wp.array(dtype=wp.vec3),
              fac: wp.array(dtype=wp.float32), nfac: wp.array(dtype=wp.float32), dv: wp.array(dtype=wp.vec3),
              nx: int, ny: int, nz: int, n_local: int,
              dt: float, h: float, surf: float, band: float, band_hi: float, band_mass: float, gain: float,
              csf_floor: float, cell_vol: float, has_fac: int, has_nfac: int):
        """dv = dt * (surf fac[block] nfac[node] kappa grad c fmask) * (cell_vol / max(m, csf_floor)), kappa = -div n."""
        t = wp.tid()
        b = t // n_local
        r = t - b * n_local
        i = r // (ny * nz)
        j = (r // nz) % ny
        k = r % nz
        ip = (i + 1) % nx
        im = (i - 1 + nx) % nx
        jp = (j + 1) % ny
        jm = (j - 1 + ny) % ny
        kp = (k + 1) % nz
        km = (k - 1 + nz) % nz
        nxp = nrm[_node(b, ip, j, k, n_local, ny, nz)]
        nxm = nrm[_node(b, im, j, k, n_local, ny, nz)]
        nyp = nrm[_node(b, i, jp, k, n_local, ny, nz)]
        nym = nrm[_node(b, i, jm, k, n_local, ny, nz)]
        nzp = nrm[_node(b, i, j, kp, n_local, ny, nz)]
        nzm = nrm[_node(b, i, j, km, n_local, ny, nz)]
        kap = -(((nxp[0] - nxm[0]) * h + (nyp[1] - nym[1]) * h) + (nzp[2] - nzm[2]) * h)
        g0 = (c[_node(b, ip, j, k, n_local, ny, nz)] - c[_node(b, im, j, k, n_local, ny, nz)]) * h
        g1 = (c[_node(b, i, jp, k, n_local, ny, nz)] - c[_node(b, i, jm, k, n_local, ny, nz)]) * h
        g2 = (c[_node(b, i, j, kp, n_local, ny, nz)] - c[_node(b, i, j, km, n_local, ny, nz)]) * h
        m = gm[t]
        cc = c[t]
        fm = float(0.0)
        if cc > band and cc < band_hi and m > band_mass:
            fm = gain
        inv_m = (1.0 / wp.max(m, csf_floor)) * cell_vol
        sf = surf
        if has_fac == 1:
            sf = surf * fac[b]
        if has_nfac == 1:
            sf = sf * nfac[t]
        dv[t] = wp.vec3(dt * (((sf * kap) * g0 * fm) * inv_m),
                        dt * (((sf * kap) * g1 * fm) * inv_m),
                        dt * (((sf * kap) * g2 * fm) * inv_m))

    @wp.kernel
    def pp_prod(gm: wp.array(dtype=wp.float32), dv: wp.array(dtype=wp.vec3), prod: wp.array(dtype=wp.vec3),
                mb: wp.array(dtype=wp.float32), floor: float):
        """mb = m for the massive nodes (m > floor), else 0; prod = mb * dv: the two arrays the block sums are taken over."""
        t = wp.tid()
        m = gm[t]
        w = float(0.0)
        if m > floor:
            w = m
        d = dv[t]
        mb[t] = w
        prod[t] = wp.vec3(w * d[0], w * d[1], w * d[2])

    @wp.kernel
    def pp_shift(psum: wp.array(dtype=wp.vec3), msum: wp.array(dtype=wp.float32), shift: wp.array(dtype=wp.vec3)):
        """shift[b] = (sum m dv) / max(sum m, 1e-30), the mass-weighted mean of the block's dv."""
        b = wp.tid()
        d = wp.max(msum[b], 1.0e-30)
        p = psum[b]
        shift[b] = wp.vec3(p[0] / d, p[1] / d, p[2] / d)

    @wp.kernel
    def pp_final(gm: wp.array(dtype=wp.float32), gmv: wp.array(dtype=wp.vec3), bf: wp.array(dtype=wp.vec3),
                 dv: wp.array(dtype=wp.vec3), shift: wp.array(dtype=wp.vec3), gv: wp.array(dtype=wp.vec3),
                 n_local: int, floor: float, dt: float, has_bf: int, has_csf: int):
        """v = (m > floor) ? mv / max(m, floor) [+ dt bf] [+ (dv - shift) * has] : 0, written into the grid's own velocity."""
        t = wp.tid()
        b = t // n_local
        m = gm[t]
        if m > floor:
            mv = gmv[t]
            mc = wp.max(m, floor)
            vx = mv[0] / mc
            vy = mv[1] / mc
            vz = mv[2] / mc
            if has_bf == 1:
                a = bf[0]
                vx = vx + dt * a[0]
                vy = vy + dt * a[1]
                vz = vz + dt * a[2]
            if has_csf == 1:
                d = dv[t]
                s = shift[b]
                vx = vx + (d[0] - s[0]) * 1.0
                vy = vy + (d[1] - s[1]) * 1.0
                vz = vz + (d[2] - s[2]) * 1.0
            gv[t] = wp.vec3(vx, vy, vz)
        else:
            gv[t] = wp.vec3(0.0, 0.0, 0.0)


@register_operator("mpm_grid_update", implementation="warp_pp_balanced", family="mpm", set="field", kind="field",
                   title="solve grid velocities, tension without net push, Warp kernels on the per-cell blocks", equation=r"""$$\mathbf v_g=\frac{\mathbf p_g}{m_g},\qquad \mathbf v_g\mathrel{+}=\Delta t\,\frac{\sigma\,\kappa_b\,\nabla c_b}{m_g/\Delta x^3},\qquad \kappa_b=-\nabla\cdot\frac{\nabla c_b}{\lvert\nabla c_b\rvert}\ \text{ on body } b\text{'s block}$$""")
class MPMGridUpdateWarpPPBalanced(MPMGridUpdateBodyCSFBalanced):
    r"""`warp_pp_balanced` implementation of mpm_grid_update: `mpm_grid_update[body_csf_balanced]` as nine Warp launches over the per-cell grid blocks, the block sums in torch.
    field -> field: reads the blocks' mass `m`, momentum `mv` (and liquid colour `c` for `tension_on: liquid`), writes the blocks' velocity `v` in place. Same parameters as `body_csf_balanced`. CUDA, 3D.

        \mathbf v_g=\frac{\mathbf p_g}{m_g},\qquad \mathbf v_g\mathrel{+}=\Delta t\,\frac{\sigma\,\kappa_b\,\nabla c_b}{m_g/\Delta x^3},\qquad \kappa_b=-\nabla\cdot\frac{\nabla c_b}{\lvert\nabla c_b\rvert}\ \text{ on body } b\text{'s block}
        v_g            the velocity of grid node g, in world length per time
        p_g            its momentum
        m_g            its mass
        \Delta t       the substep
        \sigma         the surface tension (`cell_surface_tension`, scaled per cell by `cell_surface_tension_block`), force per length
        \kappa_b       the curvature of body b's colour interface at the node, per world length
        c_b            body b's liquid colour on its own block, normalised to a volume fraction by csf_rho dx^3
        \Delta x       the grid spacing

    Reference: as `mpm_grid_update[body_csf_balanced]`."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"Solves the grid of each cell. A node's velocity is its momentum over its mass, $\mathbf v_g=\mathbf p_g/m_g$; then the "
        r"cell's surface tension is added, $\mathbf v_g\mathrel{+}=\Delta t\,\dfrac{\sigma_s\,\kappa_b\,\nabla c_b}{m_g/\Delta "
        r"x^3}$, where $c_b$ is the cell's smoothed volume indicator, $\kappa_b=-\nabla\cdot\dfrac{\nabla c_b}{\lvert\nabla "
        r"c_b\rvert}$ the curvature of its surface and $\sigma_s$ the surface tension (`cell_surface_tension`). The velocities "
        r"are written in place on the blocks. Nine Warp launches over the per-cell grid blocks, the block sums in torch."
    )
    PARAM_UNITS = {}
    SYMBOLS = {'v_g': 'the velocity of grid node g, in world length per time', 'p_g': 'its momentum', 'm_g': 'its mass', '\\Delta t': 'the substep', '\\sigma': 'the surface tension (`cell_surface_tension`, scaled per cell by `cell_surface_tension_block`), force per length', '\\kappa_b': "the curvature of body b's colour interface at the node, per world length", 'c_b': "body b's liquid colour on its own block, normalised to a volume fraction by csf_rho dx^3", '\\Delta x': 'the grid spacing'}

    MECHANISM_TAGS = ["grid_solve", "surface_tension", "per_body_blocks", "fused_kernel"]
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False

    def _scratch(self, g, dev):
        n, nb = int(g.m.numel()), int(g.n_blocks)
        s = getattr(self, "_sc", None)
        if s is None or s["n"] != n:
            z = torch.zeros
            s = {"n": n, "ca": z(n, device=dev), "cb": z(n, device=dev), "nrm": z(n, 3, device=dev), "dv": z(n, 3, device=dev),
                 "prod": z(n, 3, device=dev), "mb": z(n, device=dev), "shift": z(nb, 3, device=dev),
                 "zero3": z(1, 3, device=dev), "one": z(1, device=dev)}
            self._sc = s
        return s

    def forward(self, H, mask=None):
        g = H.field(self.at)
        dev = g.m.device
        if (not HAVE_WARP or not getattr(g, "per_parent", None) or g.dim != 3 or dev.type != "cuda"
                or self.surface_tension <= 0.0 or self.csf_band <= 0.0 or self.csf_smooth <= 0):
            return super().forward(H, mask)
        dt = sub_dt(H, self.dt_sub)
        floor = self._const("mass_floor", g)
        D, dx, inv_dx = g.dim, float(g.dx), float(g.inv_dx)
        nx, ny, nz = g.shape
        nl, nb, n = int(g.n_local), int(g.n_blocks), int(g.m.numel())
        sc = self._scratch(g, dev)
        cell_mass = self.csf_rho * dx ** D
        cell_vol = dx ** D
        h = 0.5 * inv_dx
        gain = 1.0 / max(1.0 - 2.0 * self.csf_band, 1e-6)
        src = g.c if self.tension_on == "liquid" else g.m
        gm_w = wp.from_torch(g.m)
        # the colour: x / cell_mass is x * (1 / cell_mass) in the torch CUDA kernel (a Python scalar divisor)
        scale = float(np.float32(1.0) / np.float32(cell_mass))
        bufs = (sc["ca"], sc["cb"])
        cur, k = src, 0
        for _ in range(self.csf_smooth):
            for axis in range(3):
                dst = bufs[k % 2]
                _wp_launch(pp_smooth, n, dev, [wp.from_torch(cur), wp.from_torch(dst), float(scale if cur is src else 1.0), int(axis),
                                               int(nx), int(ny), int(nz), nl])
                cur, k = dst, k + 1
        c = cur
        _wp_launch(pp_normal, n, dev, [wp.from_torch(c), wp.from_torch(sc["nrm"], dtype=wp.vec3), float(self._const("csf_eps", g)),
                                       float(h), int(nx), int(ny), int(nz), nl])
        if self.tension_block:
            Cl = H.level(g.per_parent)
            fac = Cl.state[:, Cl.state_schema[self.tension_block][0]].to(torch.float32).contiguous()
            if fac.shape[0] != nb:
                raise ValueError(f"mpm_grid_update[warp_pp_balanced]: {nb} blocks but {fac.shape[0]} cells")
            has_fac = 1
        else:
            fac, has_fac = sc["one"], 0
        if self.tension_on == "face":                       # the per-node tension factor: the face-weighted colour over the mass
            if float(g.c.abs().max()) == 0.0:
                raise ValueError("mpm_grid_update[warp_pp_balanced]: tension_on face needs `mpm_scatter` with `colour_from_face`")
            nfac, has_nfac = (g.c / g.m.clamp(min=float(self._const("csf_mass_floor", g)))).contiguous(), 1
        else:
            nfac, has_nfac = sc["one"], 0
        _wp_launch(pp_dv, n, dev, [gm_w, wp.from_torch(c), wp.from_torch(sc["nrm"], dtype=wp.vec3), wp.from_torch(fac), wp.from_torch(nfac),
                                   wp.from_torch(sc["dv"], dtype=wp.vec3), int(nx), int(ny), int(nz), nl,
                                   float(dt), float(h), float(self.surface_tension), float(self.csf_band), float(1.0 - self.csf_band),
                                   float(self.csf_band * (self.csf_rho * dx ** D)), float(gain),
                                   float(self._const("csf_mass_floor", g)), float(cell_vol), int(has_fac), int(has_nfac)])
        _wp_launch(pp_prod, n, dev, [gm_w, wp.from_torch(sc["dv"], dtype=wp.vec3), wp.from_torch(sc["prod"], dtype=wp.vec3),
                                     wp.from_torch(sc["mb"]), float(floor)])
        psum = sc["prod"].view(nb, nl, 3).sum(1)         # the torch body's own block sums (see the module docstring)
        msum = sc["mb"].view(nb, nl).sum(1)
        _wp_launch(pp_shift, nb, dev, [wp.from_torch(psum, dtype=wp.vec3), wp.from_torch(msum), wp.from_torch(sc["shift"], dtype=wp.vec3)])
        bf = getattr(H, "_mpm_body_accel", None)
        _wp_launch(pp_final, n, dev, [gm_w, wp.from_torch(g.mv, dtype=wp.vec3),
                                      wp.from_torch((bf if bf is not None else sc["zero3"]).view(-1, 3), dtype=wp.vec3),
                                      wp.from_torch(sc["dv"], dtype=wp.vec3), wp.from_torch(sc["shift"], dtype=wp.vec3),
                                      wp.from_torch(g.v, dtype=wp.vec3), nl, float(floor), float(dt), int(bf is not None), 1])
        return {}
