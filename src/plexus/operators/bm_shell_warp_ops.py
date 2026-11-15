"""`bm_cell_contact[implementation: warp]` -- the membrane-to-cell contact as ONE Warp kernel (exp27 speed audit, test 5d, 2026-10-21).

The default (`bm_shell_ops.BMCellContact`) finds the point-node pairs with `radius_pairs`: the nodes binned and sorted, a Python loop over the
27 neighbouring bins with a binary search, `int(cnt.sum())` and three `repeat_interleave` per bin (each a host synchronisation), boolean masks to
drop the far pairs, `index_add_` for the two sides, a `float(acc.abs().max())` for the diagnostic. At 320,000 points and 24,800 nodes that is 2,716
kernel launches and about 395 synchronisations per call, 16.9 ms on an RTX A6000, and the call runs once per MPM substep, 50 times per frame.

Here the same law runs in one kernel, one thread per point:

    the node hash grid        the live nodes, compacted, bucketed in cubes of side `adhesion_reach` (`wp.HashGrid`). The nodes do not move during the MPM
                              substeps (the membrane is advanced once per frame, `bm_shell`, and the engine integrates it after the substep block), so
                              the grid is built ONCE PER FRAME and reused by the 50 calls; it is rebuilt when `H.frame` changes, or when the node
                              position block or the `alive` mask is written to (the tensor's version counter) or reallocated (its address)
    per point                 loop over the nodes within `adhesion_reach`; each pair gives the force of `pair_mag` (repulsion k_rep (r0 - d)^+, the tether
                              k_adh A (d - r0)^+ (1 - s)^2 for the adherent types, s = (d - r0) / (adhesion_reach - r0)); the point sums its pairs in registers
                              and writes F / m, the node side is `atomic_add`ed into `bm_f` (equal and opposite)
    diagnostics               the pair count and the largest point acceleration go to device counters, read when asked (a property), never in the
                              call

The arithmetic of a pair follows the torch body operation for operation in float32 (d = x - n; the norm clamped from below at 1e-9; the unit vector by
division; the tether strength k_adh A as one product made in double precision; s multiplied by the reciprocal of (adhesion_reach - r0), which is what torch does
for a division by a host scalar). What differs from the torch body is only the ORDER of the sums: a point adds its pairs in the hash grid's order and
the node buffer is summed by atomics (the torch body's `index_add_` is atomic too, so neither is deterministic to the last bit). A pair at a
distance within one float32 step of `adhesion_reach` may be counted by one and not the other; its force there is zero (the tether is (1 - s)^2 -> 0).

CPU, float64 or a missing Warp fall back to the default body, as `radius_graph[warp]` does.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from plexus.models.registry import register_operator
from plexus.operators.bm_shell_ops import BMCellContact
from plexus.operators.mpm_ops import HAVE_WARP

if HAVE_WARP:
    import warp as wp

    @wp.kernel
    def _bm_contact(STATE: wp.array2d(dtype=float), pa: int, OCC: wp.array(dtype=float), MASS: wp.array(dtype=float),
                    CID: wp.array(dtype=wp.int64), NTYPE: wp.array(dtype=wp.int64), adh_mask: int, use_adh: int,
                    grid: wp.uint64, XN: wp.array(dtype=wp.vec3), NID: wp.array(dtype=wp.int32),
                    adhesion_reach: float, r0: float, k_rep: float, k_adh_a: float, inv_span: float,
                    ACC: wp.array(dtype=wp.vec3), BMF: wp.array(dtype=wp.vec3),
                    NPAIR: wp.array(dtype=wp.int32), AMAX: wp.array(dtype=float)):
        i = wp.tid()
        if OCC[i] <= 0.5:
            ACC[i] = wp.vec3(0.0, 0.0, 0.0)
            return
        x = wp.vec3(STATE[i, pa], STATE[i, pa + 1], STATE[i, pa + 2])
        adherent = int(0)
        if use_adh == 1:
            ci = int(CID[i])
            t = int(NTYPE[ci])
            if t >= 0 and t < 31:
                adherent = (adh_mask >> t) & 1
        fx = float(0.0)
        fy = float(0.0)
        fz = float(0.0)
        npair = int(0)
        q = wp.hash_grid_query(grid, x, adhesion_reach)
        j = int(0)
        while wp.hash_grid_query_next(q, j):
            d = x - XN[j]
            dist = wp.length(d)
            if dist < adhesion_reach:
                dc = wp.max(dist, 1.0e-9)
                mag = k_rep * wp.max(r0 - dc, 0.0)
                if adherent == 1:
                    s = wp.clamp((dc - r0) * inv_span, 0.0, 1.0)
                    om = 1.0 - s
                    mag = mag - (k_adh_a * wp.max(dc - r0, 0.0)) * (om * om)
                f = wp.vec3(mag * (d[0] / dc), mag * (d[1] / dc), mag * (d[2] / dc))
                fx = fx + f[0]
                fy = fy + f[1]
                fz = fz + f[2]
                wp.atomic_add(BMF, NID[j], -f)
                npair = npair + 1
        m = wp.max(MASS[i], 1.0e-30)
        a = wp.vec3(fx / m, fy / m, fz / m)
        ACC[i] = a
        if npair > 0:
            wp.atomic_add(NPAIR, 0, npair)
            wp.atomic_max(AMAX, 0, wp.max(wp.abs(a[0]), wp.max(wp.abs(a[1]), wp.abs(a[2]))))


@register_operator("bm_cell_contact", implementation="warp", family="boundary", set="particle", kind="exchange",
                   title="a basement membrane and the cells inside it, one Warp kernel", equation=r"""$$\mathbf F_{ij}=\Big[k_{\rm rep}\,(r_0-d)^+ -k_{\rm adh}A\,(d-r_0)\,(1-s)^2\,[r_0<d<adhesion_reach]\Big]\hat{\mathbf d}_{ij},\quad s=\tfrac{d-r_0}{adhesion_reach-r_0},\ \ \mathbf F_{ji}=-\mathbf F_{ij}$$""")
class BMCellContactWarp(BMCellContact):
    r"""`warp` implementation of bm_cell_contact: the membrane-cell contact as one Warp kernel on a node hash grid built once per frame.
    particle <-> particle: as bm_cell_contact (same parameters, same outputs: the points' acceleration and the opposite force on the nodes). CUDA; not bit-reproducible (float atomics).

        \mathbf F_{ij}=\Big[k_{\rm rep}\,(r_0-d)^+ -k_{\rm adh}A\,(d-r_0)\,(1-s)^2\,[r_0<d<adhesion_reach]\Big]\hat{\mathbf d}_{ij},\quad s=\tfrac{d-r_0}{adhesion_reach-r_0},\ \ \mathbf F_{ji}=-\mathbf F_{ij}
        F_{ij}         the force on point i from node j, per unit mass
        k_{rep}        `k_rep`, the repulsion stiffness, acceleration per length
        r_0            `r0`, the contact distance
        d              the point-node distance
        k_{adh}        `k_adh`, the adhesion stiffness, acceleration per length
        A              the cell's adhesion level (block `adh`)
        s              the adhesion's fade, (d - r0)/(adhesion_reach - r0)
        adhesion\_reach `adhesion_reach`, how far a node reaches to adhere
        \hat d_{ij}    the unit vector from node j to point i
        F_{ji}         the equal and opposite on the node

    Reference: as `bm_cell_contact`; NVIDIA Warp hash grid."""
    PARAM_UNITS = {'adhesion_reach': 'length', 'k_adh': 'acceleration', 'k_rep': 'acceleration', 'r0': 'length'}
    SYMBOLS = {'F_{ij}': 'the force on point i from node j, per unit mass', 'k_{rep}': '`k_rep`, the repulsion stiffness, acceleration per length', 'r_0': '`r0`, the contact distance', 'd': 'the point-node distance', 'k_{adh}': '`k_adh`, the adhesion stiffness, acceleration per length', 'A': "the cell's adhesion level (block `adh`)", 's': "the adhesion's fade, (d - r0)/(adhesion_reach - r0)", 'adhesion\\_reach': '`adhesion_reach`, how far a node reaches to adhere', '\\hat d_{ij}': 'the unit vector from node j to point i', 'F_{ji}': 'the equal and opposite on the node'}

    MECHANISM_TAGS = ["contact", "adhesion", "fused_kernel"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self._cache = None                      # {key, grid, xn, nid, n}: the node hash grid of the current frame
        self._grid = None                       # the wp.HashGrid object, reused while its dimensions stay the same
        self._grid_dim = None
        self._dg_pairs = None                   # device counters of the last call: pairs, largest acceleration component
        self._dg_amax = None
        self._ones = None

    @property
    def last(self):
        """{"pairs": the number of point-node pairs of the last call} -- read from the device counter here, in the reader's own time."""
        if self._dg_pairs is None:
            return self.__dict__.get("_last", {})
        return {"pairs": int(self._dg_pairs[0])}

    @last.setter
    def last(self, v):
        self.__dict__["_last"] = v

    def _node_grid(self, H, Nd, dev):
        """The compacted live nodes and their hash grid for this frame; rebuilt when the frame, the node positions or the alive mask changed."""
        pos = Nd.get("pos")
        key = (getattr(H, "frame", None), pos.data_ptr(), pos._version, Nd.alive.data_ptr(), Nd.alive._version, self.adhesion_reach)
        c = self._cache
        if c is not None and c["key"] == key:
            return c
        idx = Nd.alive.nonzero().squeeze(1)
        if idx.numel() == 0:
            self._cache = {"key": key, "n": 0}
            return self._cache
        xn = pos.detach()[idx].contiguous()
        nid = idx.to(torch.int32).contiguous()
        lo, hi = xn.min(0).values.tolist(), xn.max(0).values.tolist()        # one synchronisation per frame
        dim = tuple(int(min(256, max(4, math.ceil((hi[a] - lo[a]) / self.adhesion_reach) + 3))) for a in range(3))
        if self._grid is None or self._grid_dim != dim:
            self._grid = wp.HashGrid(dim[0], dim[1], dim[2], device=wp.device_from_torch(dev))
            self._grid_dim = dim
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(dev)), sync_enter=False, sync_exit=False):
            self._grid.build(points=wp.from_torch(xn, dtype=wp.vec3), radius=float(self.adhesion_reach))
        self._cache = {"key": key, "n": int(xn.shape[0]), "xn": xn, "nid": nid}
        return self._cache

    def forward(self, H, mask=None):
        P, Nd = H.level(self.at), H.level(self.nodes)
        X = P.get("pos")
        dev = X.device
        if not HAVE_WARP or not str(dev).startswith("cuda") or X.dtype != torch.float32 or X.shape[-1] != 3 or P.state.dim() != 2:
            return super().forward(H, mask)
        C = H.level(self.cells)
        n = int(X.shape[0])
        if not hasattr(Nd, "bm_f"):
            Nd.register_buffer("bm_f", torch.zeros(Nd.get("pos").shape[0], 3, dtype=X.dtype, device=dev))
            Nd.register_buffer("bm_n", torch.zeros(1, dtype=X.dtype, device=dev))
        if self._dg_pairs is None:
            self._dg_pairs = torch.zeros(1, dtype=torch.int32, device=dev)
            self._dg_amax = torch.zeros(1, dtype=X.dtype, device=dev)
            self._ones = torch.ones(n, dtype=X.dtype, device=dev)
        g = self._node_grid(H, Nd, dev)
        acc = torch.zeros_like(X) if g["n"] == 0 else torch.empty(n, 3, dtype=X.dtype, device=dev)
        self._dg_pairs.zero_()
        self._dg_amax.zero_()
        if g["n"] > 0:
            pa = int(P.state_schema["pos"][0])
            occ = P.occ.to(torch.float32).contiguous()
            m = P.mass.to(X.dtype).contiguous() if getattr(P, "mass", None) is not None else self._ones
            use_adh = 1 if self.adh > 0.0 else 0
            if use_adh:
                cid = H.lift_index(self.at, self.cells).to(torch.int64).contiguous()
                ntype = C.node_type.to(dev).to(torch.int64).contiguous()
            else:
                cid = ntype = torch.zeros(1, dtype=torch.int64, device=dev)
            mask_bits = int(sum(1 << t for t in self.adh_types if 0 <= int(t) < 31))
            inv_span = float(np.float32(1.0) / np.float32(max(self.adhesion_reach - self.r0, 1e-9)))
            with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(dev)), sync_enter=False, sync_exit=False):
                wp.launch(_bm_contact, dim=n, device=wp.device_from_torch(dev), inputs=[
                    wp.from_torch(P.state), pa, wp.from_torch(occ), wp.from_torch(m), wp.from_torch(cid), wp.from_torch(ntype),
                    mask_bits, use_adh, self._grid.id, wp.from_torch(g["xn"], dtype=wp.vec3), wp.from_torch(g["nid"]),
                    float(self.adhesion_reach), float(self.r0), float(self.k_rep), float(self.k_adh * self.adh), inv_span,
                    wp.from_torch(acc, dtype=wp.vec3), wp.from_torch(Nd.bm_f, dtype=wp.vec3),
                    wp.from_torch(self._dg_pairs), wp.from_torch(self._dg_amax)])
        Nd.bm_n += 1.0
        H.bm_contact_calls = int(getattr(H, "bm_contact_calls", 0)) + 1
        H.bm_contact_pairs = self._dg_pairs[0]
        H.bm_contact_amax = self._dg_amax[0]
        return {self.at: acc}
