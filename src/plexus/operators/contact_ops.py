"""Where a triangulated surface meets a continuum, and what each tells the other.

Two representations meet here: a vertex mesh, which is a surface, and an MPM (material point
method) continuum, which is a volume. Every operator below is one direction of that meeting --
the surface pushing the material out of itself, or the material pushing back on the surface.

In the order they appear below:

    mesh_contact     lateral     the surface pushes particles out, and collects the reaction
    mesh_inside      lateral     how many particles are behind the surface, and how deep
    bm_sense         structural  the epithelium reads the membrane it is resting on
    plate_confine    structural  a rigid half-space, as a projection
    seed_plate       seed        an open planar patch: sheet, disc, disc with a hole, or grid
    surface_drive    structural  its prescribed rigid descent, so the surface is a piston
    surface          entity      one patch of a tracked surface: a direction and a radius
    surface_track    structural  the surface's own moving frame, kept across division and death
    ecm_load         structural  the load the matrix puts back on the tissue
    ecm_gate_growth  structural  and what that load does to the cell cycle

The contact is particle-to-surface, not grid-based. Resolving contact by comparing the two
bodies' velocities at shared grid nodes requires the mesh and the grid cells to be comparable in
size. Here they are not -- a cell and the basement membrane are both a fraction of one grid cell
across -- so both bodies would live inside a single cell and the grid would hand them one shared
velocity, welding them together.
"""
from __future__ import annotations
import math
import os
import numpy as np
import torch
from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.models.base import Structural
from plexus.models.registry import register_entity, register_operator


CONTACT_HISTORY: list = []          # per frame: the dict of scalars below
VERTEX_FORCE: list = []             # per frame: the reaction on the tissue, one row per vertex
PRESSURE_MAP: list = []             # per frame: that reaction as a pressure on a (theta, phi) grid
INSIDE_HISTORY: list = []           # per frame: how many particles ended up behind the surface
_LIVE: dict = {}                    # the live contact operator, so the counter can share its bins

# ---- the tissue's side of every contact: ONE load buffer, ONE ledger (exp05 / exp11) -----------
#
# WHY THE TISSUE IS IMPLICIT AND THE MEMBRANE EXPLICIT. About fifteen membrane nodes press on each
# tissue vertex, so the contact stiffness per vertex is ~15 k, while one frame's relaxation of the
# tissue (30 iterations x eta 0.08 x mu 1) moves a free mode by up to 2.4 force-units: a load
# computed at the frame's start and held through the relaxation feeds back with a gain of ~36 k and
# is unstable above k ~ 0.05 -- a membrane softer than the cells it is meant to hold. So
# `bm_contact[live]` (membrane_ops) freezes WHICH node presses on WHICH fan triangle (weights,
# normal, the node's position) and `cell_mechanics[apicobasal_contact]` / `[shape_contact]`
# (vertex_ops) re-evaluate the penetration at every iteration (`linear_contact_forces`); the tissue
# settles where its energy balances the contact at the positions it reaches. Each node then moves by
# F_i / gamma for the force the tissue settled against, an explicit overdamped step whose gain
# k / gamma is small.
#
# WHY A LEDGER. "The bodies push each other and create no force" is a number: per interface and
# frame, |sum of the forces on the other body + sum of the reactions on the tissue| in float64, over
# that interface's largest single contact force. Two per-row mesh scalars carry it,
# `interface_force_sum` and `interface_force_max` (recorded by `MeshTable.SCALAR_RECORD`), read by
# `tools/exp_measures/exp11.balance`.

def basal_load(m, frame, nv, dev):
    """This frame's float64 [nv, 3] load on the tissue's basal points, stored on the mesh `m`, and
    the frame's ledger -- zeroed together the first time any contact asks in a new frame."""
    buf = m.get("f_basal")
    if (m.get("f_basal_frame") != frame or not torch.is_tensor(buf) or buf.shape[0] != int(nv)
            or buf.device != torch.device(dev)):
        buf = torch.zeros(int(nv), 3, dtype=torch.float64, device=dev)
        m["f_basal"] = buf
        m["f_basal_frame"] = frame
        m["_iface_ratio"] = 0.0
        m["interface_force_max"] = 0.0
        m["interface_force_sum"] = 0.0
    return buf


def record_interface(m, f_other, f_tissue):
    """Book one interface's pair into the frame's ledger (open it with `basal_load` first).

    f_other   [k, 3] float64, the contact forces on the other body's points
    f_tissue  [nv, 3] float64, the reactions put on the tissue's vertices

    The frame keeps the LARGEST |sum f_other + sum f_tissue| / max_i |f_other_i| over interfaces
    and substeps, and writes `interface_force_sum` as that ratio times `interface_force_max`, so the
    ruler's sum / max is the worst interface's own ratio."""
    if f_other.shape[0] == 0:
        return
    fmax = float(f_other.norm(dim=1).max())
    if fmax <= 0.0:
        return
    net = float((f_other.sum(0) + f_tissue.sum(0)).norm())
    m["_iface_ratio"] = max(float(m.get("_iface_ratio", 0.0)), net / fmax)
    m["interface_force_max"] = max(float(m.get("interface_force_max", 0.0)), fmax)
    m["interface_force_sum"] = m["_iface_ratio"] * m["interface_force_max"]


def barycentric_reaction(M, tri, w, F, nv):
    """The reaction -F of contacts on fan triangles `tri` (weights `w`) as a float64 [nv, 3] force
    on the surface's vertices; the fan's virtual centroid hands its share to the face's vertices in
    equal parts, `mesh_contact`'s rule. w0 is recomputed as 1 - w1 - w2 IN FLOAT64, so the shares
    sum to the force to double round-off and the ledger measures the coupling, not the weights."""
    dev = F.device
    Fd = F.to(torch.float64)
    w1 = w[:, 1:2].to(torch.float64)
    w2 = w[:, 2:3].to(torch.float64)
    w0 = 1.0 - w1 - w2
    fv = torch.zeros(int(nv), 3, dtype=torch.float64, device=dev)
    fv.index_add_(0, M["es"][tri], -w1 * Fd)
    fv.index_add_(0, M["et"][tri], -w2 * Fd)
    fface = torch.zeros(int(M["nF"]), 3, dtype=torch.float64, device=dev)
    fface.index_add_(0, M["ef"][tri], -w0 * Fd)
    fv.index_add_(0, M["es"], fface[M["ef"]] / M["cnt"][M["ef"], None].to(torch.float64))
    return fv


def linear_contact_forces(C, x, s, nF):
    """The forces of a frozen contact list at the tissue's CURRENT positions -- the implicit half.

    With the basal points b = x - s and the face centroids c_f = mean of the face's b,

        xs_i = w0 c_f + w1 b_src + w2 b_trgt       the contact point, moving with the tissue
        g_i  = (y_i - xs_i) . n_i                   y_i the node, frozen for the frame
        p_i  = max(0, offset - g_i)                 penetration of the standoff
        a_i  = g_i - offset  if 0 < g_i - offset <= band_adh, else 0     the stretched tether
        F_i  = (k p_i - k_adh a_i) n_i              on the node
        f_v  = - sum_i w_iv F_i                     on the tissue

    Returns (F [K, 3], f_v [Nv, 3]), both float64. With k_adh 0 (a list without the key) the
    tether term is absent and the forces are the repulsion alone.

    A list built on the APICAL surface (`C["side"] == "apical"`, `bm_contact[live]` with `side:
    apical`) reads the points a = x + s instead, and its n_i is the normal INTO the lumen; f_v is
    then the reaction on the apical points, which the relaxation turns into -f_v on x AND -f_v on s
    (for the basal points: -f_v on x, +f_v on s). A list without the key is basal, as it always was."""
    b = ((x + s) if C.get("side") == "apical" else (x - s)).to(torch.float64)
    es, ef = C["es"], C["ef"]
    cnt = C["cnt"].to(torch.float64)
    cent = torch.zeros(int(nF), 3, dtype=torch.float64, device=b.device).index_add_(0, ef, b[es])
    cent = cent / cnt[:, None].clamp_min(1.0)
    w = C["w"]
    xs = w[:, 0:1] * cent[C["face"]] + w[:, 1:2] * b[C["src"]] + w[:, 2:3] * b[C["trgt"]]
    g = ((C["y"] - xs) * C["n"]).sum(1)
    p = (C["offset"] - g).clamp_min(0.0)
    F = (C["k"] * p)[:, None] * C["n"]
    ka = float(C.get("k_adh", 0.0) or 0.0)
    if ka > 0.0:
        a = g - C["offset"]
        a = torch.where((a > 0) & (a <= C["band_adh"]), a, torch.zeros_like(a))
        F = F - (ka * a)[:, None] * C["n"]
    fv = torch.zeros(b.shape[0], 3, dtype=torch.float64, device=b.device)
    fv.index_add_(0, C["src"], -w[:, 1:2] * F)
    fv.index_add_(0, C["trgt"], -w[:, 2:3] * F)
    fface = torch.zeros(int(nF), 3, dtype=torch.float64, device=b.device)
    fface.index_add_(0, C["face"], -w[:, 0:1] * F)
    fv.index_add_(0, es, fface[ef] / cnt[ef, None].clamp_min(1.0))
    return F, fv



def _grid(nrow, dev):
    """A REDUCED grid: rows uniform in theta, and each row given as many phi bins as it can hold
    SQUARE ones -- `n_phi(row) = 2 pi sin(theta) / d(theta)`.

    A plain (theta, phi) lattice is square only at the equator. At theta = 0.1 rad a phi bin is a
    tenth of the row's own height, so a triangle spans ten of them and a lookup that tests a
    particle's own bin and its eight neighbours misses the face it is standing on -- silently, and
    only near the poles, which is the worst way for a contact to fail. Here every bin is d(theta)
    on a side wherever it sits, so the 3x3 query is exact everywhere, and the pole row collapses to
    three bins by the same formula rather than by a special case.
    """
    dth = math.pi / nrow
    th_c = (torch.arange(nrow, device=dev, dtype=torch.float32) + 0.5) * dth
    npr = (2 * math.pi * th_c.sin() / dth).round().long().clamp_min(1)
    off = torch.zeros(nrow + 1, dtype=torch.long, device=dev)
    off[1:] = torch.cumsum(npr, 0)
    return dict(nrow=nrow, dth=dth, npr=npr, off=off, nbin=int(off[-1]))


def _bins_of(u, G):
    """Direction -> (row, bucket)."""
    th = torch.acos(u[:, 2].clamp(-1, 1))
    ph = torch.atan2(u[:, 1], u[:, 0]) % (2 * math.pi)
    it = (th / G["dth"]).long().clamp(0, G["nrow"] - 1)
    npr = G["npr"][it]
    ip = (ph / (2 * math.pi) * npr).long().clamp(torch.zeros_like(npr), npr - 1)
    return it, ph, G["off"][it] + ip


def _neighbours(it, ph, G):
    """The 3x3 neighbourhood, as bucket indices. Each of the three rows has its own phi count, so
    the phi bin is recomputed per row rather than carried across -- carrying it is what a uniform
    lattice lets you do and a reduced one does not."""
    dev = it.device
    rows = (it[:, None] + torch.tensor([-1, 0, 1], device=dev)[None, :]).clamp(0, G["nrow"] - 1)
    npr = G["npr"][rows]                                              # [n,3]
    ip = (ph[:, None] / (2 * math.pi) * npr).long()
    out = []
    for d in (-1, 0, 1):
        out.append(G["off"][rows] + (ip + d) % npr)
    return torch.stack(out, 2).reshape(it.shape[0], 9)                # [n,9]


@register_operator("mesh_contact", family="boundary", set="particle", kind="lateral", title="push material off a moving surface",
                   equation=r"""$$\mathbf a_n=k\,d_i\,\mathbf n,\qquad \mathbf v_t=(\mathbf v_i-\mathbf v_{\mathrm{face}})-\big((\mathbf v_i-\mathbf v_{\mathrm{face}})\cdot\mathbf n\big)\mathbf n,\qquad \mathbf a_t=-\mu\lVert\mathbf a_n\rVert\frac{\mathbf v_t}{\lVert\mathbf v_t\rVert+\varepsilon_v}$$""")
class MeshContact(Lateral):
    r"""particle-to-surface contact against a live triangulated surface: the surface pushes material points out of itself, drags on them with Coulomb friction, and feels the equal and opposite reaction on its vertices.
    particle -> particle (and, through VERTEX_FORCE, back onto the surface's vertices by the barycentric weights of the contact point): reads pos and the surface set's mesh, emits an external acceleration the MPM substep consumes.

        \mathbf a_n=k\,d_i\,\mathbf n,\qquad \mathbf v_t=(\mathbf v_i-\mathbf v_{\mathrm{face}})-\big((\mathbf v_i-\mathbf v_{\mathrm{face}})\cdot\mathbf n\big)\mathbf n,\qquad \mathbf a_t=-\mu\lVert\mathbf a_n\rVert\frac{\mathbf v_t}{\lVert\mathbf v_t\rVert+\varepsilon_v}
        a_n            the penalty acceleration, normal to the face
        k              the penalty stiffness, `k_frac` of the largest the substep carries stably
        d_i            the depth of particle i behind its nearest face
        n              the face normal
        v_t            the relative tangential velocity of particle and face
        v_i, v_{face}  the particle's and the face's velocities
        a_t            the friction acceleration
        \mu            `mu`, the Coulomb friction coefficient
        \varepsilon_v  `eps_v`, the velocity that regularises the friction's direction

    Reference: Chen, Z., Qiu, X., Zhang, X. & Lian, Y. (2015). Improved coupling of finite element method with material point method based on a particle-to-surface contact algorithm. Comput. Methods Appl. Mech. Engrg. 293:1-19."""
    PARAM_UNITS = {'a_max': 'acceleration', 'band_cells': 'count', 'cap_rows': 'count', 'centre': 'length', 'centre_offset': 'length', 'dt': 'time', 'eps_v': 'velocity', 'k_frac': 'fraction', 'map_phi': 'count', 'map_theta': 'count', 'mu': 'dimensionless', 'n_grid': 'count', 'scale': 'dimensionless'}
    SYMBOLS = {'a_n': 'the penalty acceleration, normal to the face', 'k': 'the penalty stiffness, `k_frac` of the largest the substep carries stably', 'd_i': 'the depth of particle i behind its nearest face', 'n': 'the face normal', 'v_t': 'the relative tangential velocity of particle and face', 'v_i, v_{face}': "the particle's and the face's velocities", 'a_t': 'the friction acceleration', '\\mu': '`mu`, the Coulomb friction coefficient', '\\varepsilon_v': "`eps_v`, the velocity that regularises the friction's direction"}

    EMIT = "mpm_acceleration"          # consumed by `mpm_scatter` as a_ext, like `ecm_from_cell`
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["surface"]      # a live vertex set; there is no archive path any more
    MECHANISM_TAGS = ["cell_matrix_contact", "particle_to_surface", "friction", "moving_boundary"]
    PARAM_ROLES = {"k_frac": "penalty_fraction_of_ceiling", "mu": "friction_coefficient",
                   "scale": "tissue_to_box_scale", "eps_v": "slip_regularisation_velocity", 'a_max': 'cap_on_the_contact_acceleration', 'band_cells': 'thickness_of_the_search_band_in_cells', 'cap_rows': 'cap_on_contact_rows_per_frame', 'centre': 'centre_the_directions_are_binned_about', 'centre_offset': 'offset_of_that_centre', 'dt': 'substep', 'map_phi': 'direction_bins_in_phi', 'map_theta': 'direction_bins_in_theta', 'n_grid': 'grid_resolution_of_the_lookup', 'side': 'which_side_of_the_surface_is_outside', 'surface': 'the_surface_vertex_set', 'tissue': 'the_tissue_set', 'verbose': 'print_contact_diagnostics'}
    REFERENCE = ("Chen, Z., Qiu, X., Zhang, X. & Lian, Y. (2015). Improved coupling of finite "
                 "element method with material point method based on a particle-to-surface "
                 "contact algorithm. Comput. Methods Appl. Mech. Engrg. 293:1-19 -- the scheme "
                 "this implements. The grid-node alternative is Lian, Y. P., Zhang, X. & Liu, Y. "
                 "(2011). Comput. Methods Appl. Mech. Engrg. 200:3482-3494. The surface is "
                 "Okuda, S. et al. (2018). Sci. Rep. 8:2386.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        # THE STAR-SHAPE REFERENCE, AND IT MAY MOVE WITH THE SURFACE.
        #
        #   centre: [x, y, z]   a fixed point in the box, as before.
        #   centre: centroid    the LIVE surface's own centroid, re-read every frame.
        #
        # A fixed point is right for a surface that stays put and wrong for one that travels: the
        # forbidden region is the side of the surface the reference sits on, so a sphere driven
        # 0.15 of a box downward leaves its own reference behind and the ray cast starts pointing
        # at the wrong face. `centroid` also removes the drift the plate had -- its declared
        # standoff of 0.30 grew to 0.50 over the descent, which is a 90 -> 150 walk in the contact's
        # bin rows for a geometry that never changed.
        #
        # `centre_offset` is added to whichever, so a plate says `centre: centroid` with an offset
        # into the piston body and the standoff is then constant by construction.
        _c = params.get("centre", [0.5, 0.5, 0.5])
        self.centre_track = isinstance(_c, str) and str(_c).lower() in ("centroid", "surface")
        self.centre = [0.0, 0.0, 0.0] if self.centre_track else [float(v) for v in _c]
        self.centre_offset = [float(v) for v in params.get("centre_offset", [0.0, 0.0, 0.0])]
        self.scale = float(params.get("scale", 1.0))
        # THE PENALTY AS A FRACTION OF THE EXPLICIT CEILING. In acceleration form the contact is
        # a = k*d, integrated at the substep, so it is stable only while dt_sub*sqrt(k) < 1, i.e.
        # k < 1/dt_sub^2. Written as a fraction so the number cannot be quoted as an absolute
        # stiffness that happens to be three hundred times past its own limit, which is how the
        # first version of the flat rig and `flat_mpm` both failed.
        self.k_frac = float(params.get("k_frac", 0.15))
        self.mu = float(params.get("mu", 0.4))
        self.eps_v = float(params.get("eps_v", 1.0e-3))
        self.a_max = float(params.get("a_max", 3.0e4))
        self.dt_frame = float(params.get("dt", 3.2e-3))
        self.band_cells = self.tunable(params.get("band_cells"), 3.0, tensor=True)  # prefilter width, in grid cells
        self.n_grid = int(params.get("n_grid", 64))
        self.cap_rows = int(params.get("cap_rows", 2))
        # WHICH SIDE OF THE SURFACE IS FORBIDDEN. `outside` (the default, and what every indenter
        # spec means) treats the surface's INTERIOR as solid and pushes material out of it. `inside`
        # is the membrane's reading: the material belongs in the interior and the surface pushes
        # back anything that has left. The two differ by the sign of the penetration and of the
        # direction it is corrected along, and by nothing else -- but running a cell with the wrong
        # one expels its own cytoplasm, which is what it did: 26 um of cytoplasm became 62 in five
        # frames while the membrane it had been pushed through collapsed from 13 um to 6.
        self.side = str(params.get("side", "outside"))
        self.map_theta = int(params.get("map_theta", 32))        # the pressure map's own resolution
        self.map_phi = int(params.get("map_phi", 64))
        self.verbose = bool(params.get("verbose", True))

        # THE SURFACE IS A SET, NOT A FILE. See the class docstring for why the archive path is gone.
        self.surface = str(params["surface"])
        if "tissue" in params:
            raise ValueError("mesh_contact: `tissue:` (the replayed mesh archive) is deleted. Seed "
                             "the surface as a vertex set -- `seed_plate` or `seed_mesh` -- and "
                             f"name it with `surface: <set>` instead of {params['tissue']!r}")
        if "mesh_stride" in params:
            raise ValueError("mesh_contact: `mesh_stride:` counted pass-2 frames per KEPT mesh in "
                             "the archive. A live surface has one mesh per frame and no cadence to "
                             "reconcile, so the parameter has no meaning; remove it")
        # `scale` is how the two subsystems keep their own coordinates. The tissue must stay in
        # its own world about the ORIGIN, because `cell_mechanics` penalises (|pos| - R0)^2 from
        # the world origin -- translating the vesicle into the matrix's box would tear it apart.
        # So the mapping happens at the point of use and on a COPY: the surface is centred on its
        # own centroid, scaled by `scale`, and placed at `centre`. Nothing moves the tissue.
        self._frame = -2
        self._built = None
        self._dom = None
        self._newframe = False
        # THE COUNTER SHARES THIS OPERATOR'S BINS rather than rebuilding them: two bin structures
        # that disagree would have the diagnostic measuring a different surface from the one the
        # contact acted on, which is the one way a non-penetration count can be wrong and look right.
        _LIVE["contact"] = self

    # ---- the mesh of one frame ------------------------------------------------------------
    def _mesh_live(self, H, dev, dt_):
        """Vertices, half-edges and the surface's own velocity, read off the live vertex set.

        IN THE STAR FRAME, WHICH IS THE ONE CONVERSION THIS OWES. Everything below assumes the
        surface is given about the ray origin, and a live set is in box coordinates -- so `centre`
        is subtracted here and added back in `forward` when the hit point is reconstructed. Getting
        this wrong does not crash: it reports zero contacts, which reads as "the surface never
        touched anything".

        `Vv` comes off the mesh table because the surface KNOWS its velocity -- `plate_drive` writes
        it. The archive path had to recover it by differencing two cached meshes and dividing by a
        stride, which is one place a cadence could disagree with the positions it was handed.
        """
        lvl = H.level(self.surface)
        m = getattr(lvl, "_mesh", None)
        if m is None or not int(m.get("Nv", 0)):
            return None
        nv = int(m["Nv"])
        P = lvl.get("pos")[:nv].to(dt_)
        if self.centre_track:
            self.centre = [a + b for a, b in zip(P.mean(0).detach().to("cpu").tolist(),
                                                 self.centre_offset)]
        # THE SURFACE'S OWN FRAME, SCALED. `V` is what everything below works in: the surface about
        # the ray origin. With `scale: 1` and a set already in box units this is the old behaviour
        # exactly (`P - centre`); with a scale it is the archive's mapping done live -- centre on
        # the surface's OWN centroid, scale, and let `centre` place that origin in the box.
        if self.scale != 1.0:
            V = (P - P.mean(0)) * self.scale
        else:
            V = P - torch.tensor(self.centre, device=dev, dtype=dt_)
        vv = m.get("Vv", None)
        # A COPY, AND SCALED WITH THE GEOMETRY. `.to()` with a matching dtype returns THE SAME
        # TENSOR, so this aliased the live mesh table's velocity buffer -- harmless while the
        # contact only read it, and a trap waiting for the first line that does not.
        Vv = (torch.zeros_like(V) if vv is None
              else vv[:nv].to(device=dev, dtype=dt_).clone() * self.scale)
        return V, m["E_srce"].to(dev), m["E_trgt"].to(dev), m["E_face"].to(dev), Vv

    def _build(self, dev, dt_, H):
        """Everything about the surface that changes once per frame: triangles, their bins, and the
        vertex velocity the friction law needs. Rebuilt every frame, because a live surface has
        moved by definition.
        """
        got = self._mesh_live(H, dev, dt_)
        return None if got is None else self._build_from(*got, dev=dev, dt_=dt_)

    def _build_from(self, V, es, et, ef, Vv, dev, dt_):
        """The geometry, given the four arrays -- split out so `selftest` can certify the lookup on
        a surface it builds in memory instead of one a run has to produce first."""
        nv, nF = V.shape[0], int(ef.max()) + 1
        # FACE CENTROIDS. Every face contributes each of its vertices exactly once as a half-edge
        # source, so a scatter-mean over `E_srce` is the polygon's centroid with no ring ordering
        # required -- and the ordering is exactly what a re-meshed tissue does not hand over.
        cnt = torch.bincount(ef, minlength=nF).clamp_min(1)
        centroid = torch.zeros(nF, 3, device=dev, dtype=dt_)
        centroid.index_add_(0, ef, V[es])
        centroid = centroid / cnt[:, None].to(dt_)
        # ONE TRIANGLE PER HALF-EDGE: (face centroid, source vertex, target vertex). A fan, but
        # built from the half-edge list rather than from a ring, so it is correct for a polygon of
        # any size and needs no ordering. The centroid is a VIRTUAL vertex: whatever reaction it
        # receives is handed to the face's real vertices, equally, at the end.
        A, B, C = centroid[ef], V[es], V[et]
        nrm = torch.cross(B - A, C - A, dim=1)
        nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp_min(1e-20)
        # OUTWARD, DECIDED BY THE STAR-SHAPE. The mesh is centroid-referenced, so a face's own
        # position is its outward direction and the winding never has to be trusted.
        flip = ((nrm * (A + B + C)).sum(1) < 0)
        nrm = torch.where(flip[:, None], -nrm, nrm)

        # THE BIN GRID, SIZED BY THIS FRAME'S LARGEST TRIANGLE. The 3x3 query below is exact only
        # while a triangle fits inside a bin, and the faces shrink sixfold in angular size between
        # frame 0 (edge 0.17 rad) and the last (0.028 rad); a fixed grid is wrong at one end or a
        # hundred times more expensive than it needs to be at the other. The 99.9th percentile
        # rather than the maximum, because one sliver of a just-divided cell should not set the
        # resolution of the whole lookup -- and the largest triangles are the ones a 3x3 query has
        # the most slack for, since they are found from any of their three corners.
        rA, rB, rC = (A.norm(dim=1), B.norm(dim=1), C.norm(dim=1))
        rmin = torch.minimum(torch.minimum(rA, rB), rC).clamp_min(1e-9)
        # BY THE FACE'S OWN RADIUS, NOT ITS NEAREST CORNER'S. `rmin` is the minimum corner radius, so
        # one vertex near the centroid sends a face's apparent angular size to infinity -- and a
        # budded tissue has them: measured on 08b_s2_big's final mesh, rmin reaches 0.14 against a
        # median radius of 14.7, which put the 99.9th percentile of `ang` at 6.19 RADIANS (355 deg)
        # and collapsed `nrow` to its floor of 4. That is ~32 bins for 184,974 triangles, ~5,800 to a
        # bin, which destroys this structure's own premise ("a triangle spans at most one bin") and
        # makes the query allocate [n, 9K] with K ~ 5,800. On a sphere rmin ~ rmed so it never showed:
        # on the reference tissue both forms give 0.0586 rad and the same 53 rows.
        rmed = torch.maximum(torch.minimum(torch.maximum(rA, rB), torch.maximum(rB, rC)),
                             torch.minimum(rA, rC)).clamp_min(1e-20)     # the MEDIAN corner radius
        ang = (torch.maximum((B - A).norm(dim=1),
                             torch.maximum((C - A).norm(dim=1), (C - B).norm(dim=1))) / rmed)
        a999 = float(torch.quantile(ang.float(), 0.999))
        nrow = int(min(200, max(4, math.floor(math.pi / max(a999, 1e-3)))))
        G = _grid(nrow, dev)
        nb = G["nbin"]

        # EACH TRIANGLE INTO THE BUCKETS OF ITS THREE CORNERS. With the grid sized as above a
        # triangle spans at most one bin, so a particle whose direction lies inside the triangle has
        # one of its corners in its own bin or in a neighbouring one -- which is what the 3x3 query
        # covers. Registered three times rather than once so a triangle straddling a bin edge is
        # found from either side.
        u3 = torch.cat([A, B, C]) / torch.cat([rA, rB, rC]).clamp_min(1e-20)[:, None]
        _, _, b3 = _bins_of(u3, G)
        tri3 = torch.arange(A.shape[0], device=dev).repeat(3)
        order = torch.argsort(b3 * (A.shape[0] + 1) + tri3)
        b3, tri3 = b3[order], tri3[order]
        counts = torch.bincount(b3, minlength=nb)
        K = int(counts.max())
        first = torch.zeros(nb + 1, dtype=torch.long, device=dev)
        first[1:] = torch.cumsum(counts, 0)
        slot = torch.arange(b3.shape[0], device=dev) - first[b3]
        table = torch.full((nb, K), -1, dtype=torch.long, device=dev)
        table[b3, slot] = tri3
        # AND THE OUTERMOST RADIUS IN EACH BUCKET, which is the prefilter: a particle further out
        # than any corner of any triangle the bucket holds, plus a margin, cannot be in contact and
        # is never tested. This is what keeps the query over ~1% of the matrix instead of all of it.
        # Taken through the table rather than by a scatter-max, which has no deterministic
        # implementation and would make a re-run non-reproducible for a number that is a maximum.
        # A COARSE SURFACE RAILS THIS GRID, and the cost of that is checked where it is
        # actually incurred -- see `_query`. `K` alone is the wrong yardstick: the query allocates
        # [n_candidates, 9K], so 127 per bucket is gigabytes against 500,000 candidates and a few
        # megabytes against the 20,000 that a small shell's prefilter actually keeps. Budgeting on
        # K refused a run that fits comfortably, which is a guard doing the damage it exists to
        # prevent.
        rtri = torch.maximum(torch.maximum(rA, rB), rC)
        rmax = torch.where(table >= 0, rtri[table.clamp_min(0)],
                           torch.zeros_like(rtri[0]).expand(nb, K)).max(dim=1).values

        # VERTEX VELOCITY comes off the surface itself (`_mesh_live`), because the operator that
        # moves it knows it exactly. Nothing is differenced here.
        return dict(V=V, Vv=Vv, es=es, et=et, ef=ef, cnt=cnt, A=A, B=B, C=C, nrm=nrm,
                    table=table, rmax=rmax, G=G, nv=nv, nF=nF, K=K, n_tri=int(A.shape[0]))

    # ---- the contact ----------------------------------------------------------------------
    def _query(self, M, x, u, r):
        """For each candidate particle, the outermost triangle its own direction hits -- in slices
        of candidates sized to a fixed scratch budget.

        THE ALLOCATION IS [n, 9K] PER CANDIDATE, six such tensors of 3 floats, and n grows with the
        matrix around the surface while K grows with the surface's faces: spheroid_ecm_04 at
        frame 50 (9,588 sub-triangles, 78 per bucket, 74,035 candidates) asked for 5.2e7 entries,
        3.7 GB, and the guard that used to stand here refused the frame on an 80 GB card. Nothing
        in the query couples two candidates -- each row takes its own argmax -- so it runs in
        slices whose n*9K stays under `PLEXUS_CONTACT_MAX_ENTRIES` (default 2e7 entries, ~1.4 GB
        of scratch) and the slices are concatenated. Bit-identical to the one-shot answer.
        """
        n = x.shape[0]
        budget = float(os.environ.get("PLEXUS_CONTACT_MAX_ENTRIES", "2.0e7"))
        step = max(1, int(budget // max(9.0 * float(M["K"]), 1.0)))
        if n <= step:
            return self._query_chunk(M, x, u, r)
        outs = [self._query_chunk(M, x[i:i + step], u[i:i + step], r[i:i + step])
                for i in range(0, n, step)]
        return tuple(torch.cat([o[j] for o in outs], dim=0) for j in range(4))

    def _query_chunk(self, M, x, u, r):
        """One slice of `_query`: the outermost triangle each candidate's direction hits.

        Returns (hit, tri, t, w) -- whether a face was found, which one, the radius of the surface
        along the particle's ray, and the three barycentric weights.
        """
        dev = x.device
        it, ph, _ = _bins_of(u, M["G"])
        bn = _neighbours(it, ph, M["G"])                                       # [n,9]
        # THE BUDGET, WHERE THE ALLOCATION IS. Six [n, 9K] tensors of 3 floats each are built below,
        # so the peak is about n*9K*72 bytes; the cap is in ENTRIES and is compared against what
        # this call will really ask for rather than against a bucket depth that says nothing on its
        # own. It fires only on a surface that is BOTH angularly coarse and near a lot of material,
        # which is the case that ran a run out of memory at frame 230 of 400 with an error naming an
        # allocation and not a cause.
        # 5e7 ENTRIES, NOT 4e8. The first number came from costing an entry at 72 bytes -- six
        # [n, 9K, 3] tensors -- but each is float32, so an entry is 12 bytes in ONE of them and the
        # peak is a handful of those plus the intermediates. A single [n, 9K, 3] of 2.75e8 entries
        # is 3.3 GB, which is what the run actually asked for and the guard waved through. At 5e7
        # the largest single allocation is ~600 MB and the peak a few GB.
        cand = M["table"][bn].reshape(x.shape[0], -1)                          # [n, 9K]
        ok = cand >= 0
        ci = cand.clamp_min(0)
        A = M["A"][ci]; B = M["B"][ci]; C = M["C"][ci]                         # [n, 9K, 3]
        # MOLLER-TRUMBORE, with the ray from the tissue centroid along the particle's own
        # direction. `t` is then the radius of the surface in that direction -- the curved-surface
        # form of the flat rig's interpolated `z_face` -- and the weights fall out of the same
        # solve rather than being a second, separate projection.
        e1, e2 = B - A, C - A
        uu = u[:, None, :]
        p = torch.cross(uu.expand_as(e2), e2, dim=2)
        det = (e1 * p).sum(2)
        inv = 1.0 / torch.where(det.abs() < 1e-20, torch.full_like(det, 1e-20), det)
        s = -A                                              # ray origin is the centroid = 0
        w1 = (s * p).sum(2) * inv
        q = torch.cross(s, e1, dim=2)
        w2 = (uu * q).sum(2) * inv
        t = (e2 * q).sum(2) * inv
        tol = 1e-6
        good = ok & (det.abs() > 1e-20) & (w1 >= -tol) & (w2 >= -tol) & (w1 + w2 <= 1 + tol) & (t > 0)
        tt = torch.where(good, t, torch.full_like(t, -1.0))
        best = tt.argmax(dim=1)                             # the OUTERMOST face on the ray
        ar = torch.arange(x.shape[0], device=dev)
        hit = good[ar, best]
        tri = ci[ar, best]
        tb = t[ar, best]
        w = torch.stack([1.0 - w1[ar, best] - w2[ar, best], w1[ar, best], w2[ar, best]], 1)
        return hit, tri, tb, w

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        # `or -1` WOULD MAKE FRAME 0 READ AS -1, because 0 is falsy -- and frame 0 is the frame
        # that opens the history, so the aggregation below would index an empty list.
        f = getattr(H, "frame", None)
        f = -1 if f is None else int(f)
        # REBUILT ONCE PER FRAME AND NOT PER SUBSTEP. The surface's position is a frame-level fact --
        # `plate_drive` writes it once, before the substep block -- so rebuilding inside the block
        # would cost eight bin tables to describe one geometry. The build is a few milliseconds
        # against a frame's eight substeps.
        if f != self._frame:
            self._frame, self._newframe = f, True
            self._built = self._build(dev, dt_, H)
        elif self._built is None:
            # the seed ran between this frame's substeps; pick the surface up without opening a
            # second history row for a frame that already has one
            self._built = self._build(dev, dt_, H)
        M = self._built
        if M is None:
            # THE SEED HAS NOT RUN. `seed` operators are confined to the opening frames and the
            # contact may be scheduled inside a substep block that runs on the same frame, so an
            # empty surface is an ordering fact and not an error -- but a silent zero would make a
            # mis-ordered schedule look like a run in which nothing ever touched.
            if not getattr(self, "_said_empty", False):
                print(f"[mesh_contact] surface `{self.surface}` is empty at frame {f}; no contact "
                      f"until its seed has run", flush=True)
                self._said_empty = True
            return {self.at: torch.zeros_like(pos)}
        dt_sub = float(getattr(H, "sub_dt", None) or self.dt_frame)
        k = (self.k_frac / dt_sub) ** 2
        # THE GRID CELL IS world/n_grid, NOT 1/n_grid. In a 117-unit box the unit-box value made
        # the prefilter band 117x too thin and reported a 0.18-unit penetration as "11 cells".
        _W = getattr(H, "world_size", None)
        try:
            _Wmin = float(min(float(v) for v in _W)) if _W is not None else 1.0
        except TypeError:
            _Wmin = float(_W) if _W else 1.0
        dx = _Wmin / self.n_grid
        self._dx = dx

        c = torch.tensor(self.centre, device=dev, dtype=dt_)
        d = pos - c
        r = d.norm(dim=1).clamp_min(1e-12)
        u = d / r[:, None]
        # THE PREFILTER: only particles inside the outermost corner their own bucket holds, plus a
        # margin. Everything else is beyond any face this direction can offer and is never tested.
        _, _, b = _bins_of(u, M["G"])
        near = r < (M["rmax"][b] + self.band_cells * dx)
        acc = torch.zeros_like(pos)
        if not bool(near.any()):
            self._record(f, 0, 0.0, 0.0, 0.0, acc, None, None, M, dev, dt_)
            return {self.at: acc}
        idx = torch.nonzero(near).squeeze(1)
        hit, tri, t, w = self._query(M, pos[idx], u[idx], r[idx])
        if not bool(hit.any()):
            self._record(f, 0, 0.0, 0.0, 0.0, acc, None, None, M, dev, dt_)
            return {self.at: acc}
        idx, tri, t, w = idx[hit], tri[hit], t[hit], w[hit]
        n_hat = M["nrm"][tri]
        xs = c + t[:, None] * u[idx]                       # the point on the surface, on the ray
        _sgn = -1.0 if self.side == "inside" else 1.0
        depth = (_sgn * ((xs - pos[idx]) * n_hat).sum(1)).clamp_min(0.0)
        inside = depth > 0
        idx, tri, w, n_hat, depth = idx[inside], tri[inside], w[inside], n_hat[inside], depth[inside]
        if idx.numel() == 0:
            self._record(f, 0, 0.0, 0.0, 0.0, acc, None, None, M, dev, dt_)
            return {self.at: acc}
        a_n = k * depth
        # THE FACE'S OWN VELOCITY at the contact point: the two real vertices by their weights, and
        # the virtual centroid by the mean of the face's vertices. Friction against the GRID's
        # velocity instead would be friction against whatever else is in the cell, which is the weld
        # this scheme exists to avoid.
        ef = M["ef"][tri]
        vface = torch.zeros(M["nF"], 3, device=dev, dtype=dt_)
        vface.index_add_(0, M["ef"], M["Vv"][M["es"]])
        vface = vface / M["cnt"][:, None].to(dt_)
        v_surf = (w[:, 0:1] * vface[ef] + w[:, 1:2] * M["Vv"][M["es"][tri]]
                  + w[:, 2:3] * M["Vv"][M["et"][tri]])
        v_p = lvl.get("vel")[idx] if "vel" in lvl.state_schema else torch.zeros_like(v_surf)
        dv = v_p - v_surf
        dv_t = dv - (dv * n_hat).sum(1)[:, None] * n_hat
        speed = dv_t.norm(dim=1)
        # REGULARISED COULOMB: saturates at mu*a_n, linear below `eps_v`, so a resting contact does
        # not chatter. `eps_v` is a slip velocity and not a fudge factor.
        a_t = torch.minimum(self.mu * a_n, self.mu * a_n * speed / self.eps_v)
        a_par = (_sgn * a_n[:, None]) * n_hat - a_t[:, None] * dv_t / speed.clamp_min(1e-12)[:, None]
        # THE CLAMP IS HERE, not in the scatter, so what is recorded as the reaction is what acts.
        mag = a_par.norm(dim=1)
        a_par = a_par * (self.a_max / mag.clamp_min(1e-20)).clamp(max=1.0)[:, None]
        acc.index_add_(0, idx, a_par)

        # THE REACTION. Mass is uniform across the matrix, so force and acceleration differ by one
        # constant and the residual below is the same number either way; it is written in force
        # units because that is what a vertex would receive.
        m = float(getattr(lvl, "mass", torch.ones(1, device=dev)).reshape(-1)[0])
        if self.verbose and not getattr(self, "_m_printed", False):
            print(f"[mesh_contact] particle mass {m:.4g}; penalty k = {k:.4g} in acceleration "
                  f"units = ({self.k_frac}/dt_sub)^2 against the explicit ceiling "
                  f"1/dt_sub^2 = {1.0 / dt_sub ** 2:.4g}; own clamp at a = {self.a_max:.4g}",
                  flush=True)
            self._m_printed = True
        fvert = torch.zeros(M["nv"], 3, device=dev, dtype=dt_)
        fvert.index_add_(0, M["es"][tri], (-m * w[:, 1:2]) * a_par)
        fvert.index_add_(0, M["et"][tri], (-m * w[:, 2:3]) * a_par)
        # the virtual centroid's share, handed to the face's own vertices in equal parts
        fface = torch.zeros(M["nF"], 3, device=dev, dtype=dt_)
        fface.index_add_(0, ef, (-m * w[:, 0:1]) * a_par)
        fvert.index_add_(0, M["es"], fface[M["ef"]] / M["cnt"][M["ef"], None].to(dt_))

        self._record(f, int(idx.numel()), float(depth.max()), float(speed.mean()),
                     float(a_n.max()), acc * m, fvert, (idx, tri, depth, a_par, m), M, dev, dt_)
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc * lvl.occ[:, None].float()}

    # ---- what it writes down --------------------------------------------------------------
    def _record(self, f, n_con, dmax, slip, a_max_seen, fpart, fvert, detail, M, dev, dt_):
        """ONE ROW PER FRAME, AGGREGATED OVER THE FRAME'S OWN SUBSTEPS -- not the first substep's
        values wearing the frame's name.

        The first version kept whichever substep had the most contacts, which reported the frame's
        pressure map from a single instant and its residual from another; the peak quantities
        (contacts, penetration, residual) are maxima over the substeps and the continuous ones
        (slip, pressure) are means, so each is aggregated the way its own question asks.
        """
        resid = 0.0
        if fvert is not None:
            resid = float((fpart.sum(0) + fvert.sum(0)).norm()
                          / fpart.norm(dim=1).sum().clamp_min(1e-30))
        if self._newframe:
            CONTACT_HISTORY.append(dict(frame=f, n_contact=0, depth_max=0.0, slip=0.0, a_max=0.0,
                                        momentum_residual=0.0, n_tri=M["n_tri"],
                                        nrow=M["G"]["nrow"], K=M["K"], n_sub=0))
            PRESSURE_MAP.append(np.zeros((self.map_theta, self.map_phi), np.float32))
            VERTEX_FORCE.append(None)
            self._newframe = False
            if self.verbose and (f < 2 or f % 50 == 0):
                print(f"[mesh_contact] frame {f}: {M['n_tri']} triangles, {M['G']['nbin']} bins in "
                      f"{M['G']['nrow']} rows (max {M['K']} per bucket), {n_con} contacts, "
                      f"penetration {dmax / getattr(self, '_dx', 1.0 / self.n_grid):.2f} cells, momentum residual "
                      f"{resid:.2e}", flush=True)
        row = CONTACT_HISTORY[-1]
        row["n_sub"] += 1
        row["n_contact"] = max(row["n_contact"], n_con)
        row["depth_max"] = max(row["depth_max"], dmax)
        row["a_max"] = max(row["a_max"], a_max_seen)
        row["momentum_residual"] = max(row["momentum_residual"], resid)
        row["slip"] += (slip - row["slip"]) / row["n_sub"]
        PRESSURE_MAP[-1] += (self._pressure(detail, M, dev, dt_) - PRESSURE_MAP[-1]) / row["n_sub"]
        if fvert is not None:
            VERTEX_FORCE[-1] = fvert.detach().to("cpu", torch.float32).numpy()

    def _pressure(self, detail, M, dev, dt_):
        """The reaction as a pressure by direction, on the same (theta, phi) grid `apical_map` uses.

        Binned by direction and divided by the area each bin covers -- R^2 dOmega, with dOmega
        exact per row rather than sin(theta) d(theta), which diverges from it precisely at the poles
        where the bins are slivers and the error would read as a polar pressure.
        """
        nth, nph = self.map_theta, self.map_phi
        P = torch.zeros(nth * nph, device=dev, dtype=dt_)
        Rm = torch.zeros(nth * nph, device=dev, dtype=dt_)
        cntm = torch.zeros(nth * nph, device=dev, dtype=dt_)
        if self._dom is None:
            e = torch.linspace(0, math.pi, nth + 1, device=dev, dtype=dt_)
            self._dom = (e[:-1].cos() - e[1:].cos()) * (2 * math.pi / nph)
        # the surface radius per bin, from the mesh's own vertices
        Vv = M["V"]
        rv = Vv.norm(dim=1).clamp_min(1e-12)
        uv = Vv / rv[:, None]
        thv = torch.acos(uv[:, 2].clamp(-1, 1)); phv = torch.atan2(uv[:, 1], uv[:, 0]) % (2 * math.pi)
        bv = ((thv / math.pi * nth).long().clamp(0, nth - 1) * nph
              + (phv / (2 * math.pi) * nph).long().clamp(0, nph - 1))
        Rm.index_add_(0, bv, rv); cntm.index_add_(0, bv, torch.ones_like(rv))
        Rm = Rm / cntm.clamp_min(1.0)
        if detail is not None:
            idx, tri, depth, a_par, m = detail
            xs = M["A"][tri]
            r_ = xs.norm(dim=1).clamp_min(1e-12); u_ = xs / r_[:, None]
            th = torch.acos(u_[:, 2].clamp(-1, 1)); ph = torch.atan2(u_[:, 1], u_[:, 0]) % (2 * math.pi)
            b = ((th / math.pi * nth).long().clamp(0, nth - 1) * nph
                 + (ph / (2 * math.pi) * nph).long().clamp(0, nph - 1))
            P.index_add_(0, b, m * a_par.norm(dim=1))
        area = (Rm.reshape(nth, nph) ** 2) * self._dom[:, None]
        return (P.reshape(nth, nph) / area.clamp_min(1e-12)).detach().to("cpu",
                                                                        torch.float32).numpy()


@register_operator("mesh_inside", family="hierarchy", set="particle", kind="lateral", title="measure matrix inside the surface", probe=True, equation=r"""$$N_{in}=\big\lvert\{i:\ d_i>0\}\big\rvert,\qquad d_{\max}=\max_i d_i$$""")
class MeshInsideCount(Lateral):
    r"""a measurement as an operator: how many matrix particles are behind the surface and how deep; it counts and does not correct, once per frame.
    particle -> particle: reads pos and the surface, writes nothing; records the count and the maximum depth per frame (contact.json beside the trajectory).

        N_{in}=\big\lvert\{i:\ d_i>0\}\big\rvert,\qquad d_{\max}=\max_i d_i
        N_{in}         the number of particles behind the surface this frame
        d_i            the depth of particle i behind its nearest face (0 outside)
        d_{\max}       the deepest penetration

    Reference: none: a measurement, not a mechanism. Plexus (this work)."""
    PARAM_UNITS = {'n_grid': 'count'}
    SYMBOLS = {'N_{in}': 'the number of particles behind the surface this frame', 'd_i': 'the depth of particle i behind its nearest face (0 outside)', 'd_{\\max}': 'the deepest penetration'}
    PARAM_ROLES = {'n_grid': 'grid_resolution_of_the_lookup'}

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["diagnostic", "non_penetration"]
    REFERENCE = "Plexus (this work); a measurement, not a mechanism."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.n_grid = int(params.get("n_grid", 64))

    def forward(self, H, mask=None):
        op = _LIVE.get("contact")
        if op is None or op._built is None:
            return {}
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        c = torch.tensor(op.centre, device=dev, dtype=dt_)
        d = pos - c
        r = d.norm(dim=1).clamp_min(1e-12)
        u = d / r[:, None]
        M = op._built
        _, _, b = _bins_of(u, M["G"])
        near = r < M["rmax"][b] + 1e-9
        n_in, dmax = 0, 0.0
        if bool(near.any()):
            idx = torch.nonzero(near).squeeze(1)
            hit, tri, t, w = op._query(M, pos[idx], u[idx], r[idx])
            deep = hit & (t > r[idx])
            n_in = int(deep.sum())
            if n_in:
                dmax = float((t[deep] - r[idx][deep]).max())
        _f = getattr(H, "frame", None)
        INSIDE_HISTORY.append(dict(frame=-1 if _f is None else int(_f), n_inside=n_in,
                                   depth_max_cells=dmax * self.n_grid))
        return {}


def reset():
    """Clear every module-level record. Called by the run script before the engine starts, because
    a second run in the same process would otherwise append to the first one's history."""
    for L in (CONTACT_HISTORY, VERTEX_FORCE, PRESSURE_MAP, INSIDE_HISTORY):
        L.clear()
    _LIVE.clear()


# --------------------------------------------------------------------------- the self-test
def selftest(surface="sphere", dev="cuda:0", n=40000, n_brute=400, **kw):
    """THE LOOKUP IS THE ONE THING HERE THAT FAILS SILENTLY, so it is certified against brute force
    before any run is trusted to it.

    A miss is not an error and not a NaN: the particle simply feels no contact, ends up behind the
    surface, and the movie shows matrix where the solid is -- which is exactly the artefact
    `cell_exclude` was written to sweep up rather than to explain. Two questions, both of which can
    come back wrong:

      COVERAGE  a ray from inside a CLOSED star-shaped surface must hit it, so the fraction of
                random directions that find a face is 1 and anything less is the bin structure
                losing faces -- which is what a plain (theta, phi) lattice does near the poles. An
                OPEN surface (a plate) covers only its own solid angle, so the number is reported
                and not asserted; there the agreement test is the whole check.
      AGREEMENT for a subsample, the face and the radius the bins return must be the ones a test
                against EVERY triangle returns. Coverage alone would pass on a lookup that
                confidently returns the wrong face.

    THE SURFACE IS BUILT HERE, NOT LOADED, so the one check that certifies the lookup needs no
    artefact -- a check that loads a mesh archive certifies the lookup against whatever that file
    happens to hold.

    RUN IT AS AN IMPORT, NOT AS `python -m`: this module registers its operators at import, and
    `-m` executes it a SECOND time under the name `__main__`, so the decorators raise
    "operator 'mesh_contact' already has variant 'default'" before the check starts.

        python -c "import plexus.operators as _; from plexus.operators.contact_ops import selftest;\
                   raise SystemExit(0 if selftest('sphere') else 1)"
        python -c "import plexus.operators as _; from plexus.operators.contact_ops import selftest;\
                   raise SystemExit(0 if selftest('plate', shape='disc_hole') else 1)"
    """
    if surface == "sphere":
        from plexus.operators.vertex_ops import build_sphere_mesh
        verts, es, et, ef, _nF = build_sphere_mesh(int(kw.get("n_cells", 400)),
                                                   float(kw.get("radius", 5.0)),
                                                   float(kw.get("jitter", 0.15)), 0)
        V = np.asarray(verts, np.float64)
        closed = True
    elif surface == "plate":
        # THE PLATE IN ITS STAR FRAME: the patch sits `standoff` below the ray origin, which is
        # where `seed_plate` + a `centre` above it put it. Open, so coverage is a fraction.
        xy, quads = _plate_lattice(int(kw.get("nq", 48)), float(kw.get("half_width", 0.25)),
                                   str(kw.get("shape", "sheet")))
        es, et, ef = _plate_half_edges(quads)
        h = float(kw.get("standoff", 0.30))
        V = np.stack([xy[:, 0], np.full(len(xy), -h), xy[:, 1]], -1)
        closed = False
    else:
        raise ValueError(f"selftest: surface must be 'sphere' or 'plate', got {surface!r}")

    dv = torch.device(dev)
    op = MeshContact({"surface": "vertex", "verbose": False}, device=dev)
    Vt = torch.as_tensor(V, device=dv, dtype=torch.float32)
    M = op._build_from(Vt, torch.as_tensor(np.asarray(es, np.int64), device=dv),
                       torch.as_tensor(np.asarray(et, np.int64), device=dv),
                       torch.as_tensor(np.asarray(ef, np.int64), device=dv),
                       torch.zeros_like(Vt), dev=dv, dt_=torch.float32)
    g = torch.Generator(device="cpu").manual_seed(0)
    u = torch.randn(n, 3, generator=g).to(dv)
    u = u / u.norm(dim=1, keepdim=True)
    r = torch.full((n,), 1e-6, device=dv)
    hit, tri, t, w = op._query(M, u * r[:, None], u, r)
    cov = float(hit.float().mean())

    # BRUTE FORCE OVER EVERY TRIANGLE, for a subsample, and only where the lookup found one: on an
    # open surface most rays legitimately hit nothing, and comparing those would measure the
    # geometry rather than the bins.
    pool = torch.nonzero(hit).squeeze(1)
    s_ = pool[torch.randperm(pool.numel(), generator=g).to(dv)[:n_brute]] if pool.numel() else pool
    nb = int(s_.numel())
    bad, dmax = 0, 0.0
    if nb:
        A, B, C = M["A"], M["B"], M["C"]
        e1, e2 = B - A, C - A
        us = u[s_][:, None, :]
        pp = torch.cross(us.expand(-1, A.shape[0], -1), e2[None], dim=2)
        det = (e1[None] * pp).sum(2)
        inv = 1.0 / torch.where(det.abs() < 1e-20, torch.full_like(det, 1e-20), det)
        sv = -A[None]
        w1 = (sv * pp).sum(2) * inv
        q = torch.cross(sv.expand(nb, -1, -1), e1[None].expand(nb, -1, -1), dim=2)
        w2 = (us * q).sum(2) * inv
        tb = (e2[None] * q).sum(2) * inv
        good = (det.abs() > 1e-20) & (w1 >= -1e-6) & (w2 >= -1e-6) & (w1 + w2 <= 1 + 1e-6) & (tb > 0)
        tb = torch.where(good, tb, torch.full_like(tb, -1.0))
        t_ref = tb.max(dim=1).values
        d = (t[s_] - t_ref).abs() / t_ref.clamp_min(1e-9)
        bad, dmax = int((d > 1e-4).sum()), float(d.max())
    ok = (bad == 0) and (cov > 0.9999 if closed else cov > 0.0)
    print(f"[selftest] {surface}: {M['n_tri']:6d} triangles, {M['G']['nbin']:6d} bins in "
          f"{M['G']['nrow']:3d} rows, max {M['K']:3d} per bucket | coverage {cov:.5f}"
          f"{' (closed: must be 1)' if closed else ' (open: reported, not asserted)'} | "
          f"brute-force disagreement {bad}/{nb} (max relative {dmax:.2e})", flush=True)
    print(f"[selftest] {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


SENSE_TRACE: list = []


@register_operator("bm_sense", family="signalling", set="vertex", kind="structural",
                   equation=r"""$$\mathbf u_f=\frac{\mathbf c_f-\mathbf c}{\lVert\mathbf c_f-\mathbf c\rVert},\qquad \mathrm{def}_f=\mathrm{clamp}\!\left(1-\frac{L_f}{p_{\mathrm{ref}}},0,1\right)^{\text{sharp}}$$""", title="sense the basement membrane below")
class BMSense3D(Structural):
    r"""the epithelium reads the membrane it is resting on: each cell senses how much basement membrane is under it from a recorded membrane map, and a shortfall becomes a chemical signal the rest of the model can act on.
    vertex -> cell: reads the mesh geometry and a recorded membrane map, writes one channel (`chan`) of the cell set's `chem` block in place.

        \mathbf u_f=\frac{\mathbf c_f-\mathbf c}{\lVert\mathbf c_f-\mathbf c\rVert},\qquad \mathrm{def}_f=\mathrm{clamp}\!\left(1-\frac{L_f}{p_{\mathrm{ref}}},0,1\right)^{\text{sharp}}
        u_f            cell f's own direction from the tissue centroid c
        c_f            the cell's centroid
        def_f          the cell's deficit, 0 fully anchored to 1 no membrane
        L_f            the membrane level in the cell's direction, read from the map M(theta, phi)
        p_{ref}        `p_ref`, the level counted as fully supported, in the map's units
        sharp          `sharp`, the exponent of the deficit

    Reference: Streuli, C. H. (2009). Integrins and cell-fate determination. Curr. Opin. Cell Biol. 21:194-198; Frantz, C., Stewart, K. M. & Weaver, V. M. (2010). The extracellular matrix at a glance. J. Cell Sci. 123:4195-4200."""
    PARAM_UNITS = {'p_ref': 'dimensionless', 'sharp': 'dimensionless', 'chan': 'count'}
    SYMBOLS = {'u_f': "cell f's own direction from the tissue centroid c", 'c_f': "the cell's centroid", 'def_f': "the cell's deficit, 0 fully anchored to 1 no membrane", 'L_f': "the membrane level in the cell's direction, read from the map M(theta, phi)", 'p_{ref}': "`p_ref`, the level counted as fully supported, in the map's units", 'sharp': '`sharp`, the exponent of the deficit'}

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["map"]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["mechanosensing", "integrin_signalling", "anchorage_dependence",
                      "matrix_to_cell_feedback", "morphogen_source"]
    PARAM_ROLES = {"p_ref": "membrane_reference_level", "sharp": "deficit_sharpness",
                   "chan": "chem_channel_written", 'cell_set': 'the_cell_set', 'map': 'the_recorded_membrane_map'}
    REFERENCE = ("Streuli, C. H. (2009). Integrins and cell-fate determination. Curr. Opin. Cell "
                 "Biol. 21:194-198; Frantz, C., Stewart, K. M. & Weaver, V. M. (2010). The "
                 "extracellular matrix at a glance. J. Cell Sci. 123:4195-4200.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.cat = params.get("cell_set", "cell")
        z = np.load(str(params["map"]))
        self.P = torch.as_tensor(np.asarray(z["pmap"], np.float32))
        self.T = int(self.P.shape[0])
        self.p_ref = float(params.get("p_ref", 1.0))
        self.sharp = float(params.get("sharp", 1.0))
        self.chan = int(params.get("chan", 0))
        self._frame, self._t, self._said = -1, 0, False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        clvl = H.level(self.cat)
        if m is None or clvl is None or "chem" not in getattr(clvl, "state_schema", {}):
            return {}
        nF = int(m["nF"])
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        f = int(getattr(H, "frame", -1) or -1)
        if f != self._frame:
            self._frame = f
            self._t = min(self.T - 1, max(0, f))

        # PER-CELL DIRECTION, centroid-referenced -- the same construction the map was binned with,
        # and the same one `ecm_gate_growth` uses, so the two cannot drift apart.
        es, ef = m["E_srce"], m["E_face"]
        live = ef < nF
        e_s, e_f = es[live].long(), ef[live].long()
        cnt = torch.zeros(nF, device=dev, dtype=dt_).index_add_(
            0, e_f, torch.ones_like(e_f, dtype=dt_))
        centroid = torch.zeros(nF, 3, device=dev, dtype=dt_).index_add_(0, e_f, pos[e_s].to(dt_))
        ok = cnt > 0
        centroid[ok] /= cnt[ok, None]
        origin = centroid[ok].mean(0) if ok.any() else torch.zeros(3, device=dev, dtype=dt_)
        d = centroid - origin
        u = d / d.norm(dim=1).clamp_min(1e-9)[:, None]
        M = self.P[self._t].to(dev, dt_)
        nth, nph = M.shape
        th = torch.acos(u[:, 2].clamp(-1, 1))
        ph = torch.atan2(u[:, 1], u[:, 0]) % (2 * math.pi)
        lig = M[(th / math.pi * nth).long().clamp(0, nth - 1),
                (ph / (2 * math.pi) * nph).long().clamp(0, nph - 1)]
        # A CELL WITH NO LIVE HALF-EDGE HAS NO DIRECTION, so it is given no deficit rather than the
        # deficit of whatever direction the origin happens to point in.
        def_ = (1.0 - lig / max(self.p_ref, 1e-12)).clamp(0.0, 1.0) ** self.sharp
        def_ = torch.where(ok, def_, torch.zeros_like(def_))

        ci, _ = clvl.state_schema["chem"]
        clvl.state[:nF, ci + self.chan] = def_.to(clvl.state.dtype)
        SENSE_TRACE.append((int((def_ > 0.5).sum()), float(def_.max()), float(def_.mean())))
        if not self._said:
            print(f"[bm_sense] writing the membrane deficit into {self.cat}.chem[:, {self.chan}] "
                  f"({self.T} frames, p_ref {self.p_ref}, sharp {self.sharp}); frame {f}: "
                  f"{int((def_ > 0.5).sum())} of {nF} cells above 0.5, max {float(def_.max()):.3f}",
                  flush=True)
            self._said = True
        return {}


@register_operator("bm_sense", implementation="live", family="signalling", set="vertex",
                   kind="structural",
                   equation=r"""$$\mathrm{def}_f=\mathrm{clamp}\!\left(1-\frac{\rho_f}{p_{\mathrm{ref}}\,\mathrm{median}_g\,\rho_g},0,1\right)^{\text{sharp}}$$""", title="sense the live basement membrane below")
class BMSenseLive(BMSense3D):
    r"""bm_sense reading the live membrane instead of a recorded map: the deficit from the per-cell ligand density bm_contact[live] publishes, relative to the median cell, scaled by the cells' integrin level.
    vertex -> cell: reads m["bm_ligand"], writes the deficit into one channel of the cell set's `chem` (a stale or mis-sized density writes nothing and says so once).

        \mathrm{def}_f=\mathrm{clamp}\!\left(1-\frac{\rho_f}{p_{\mathrm{ref}}\,\mathrm{median}_g\,\rho_g},0,1\right)^{\text{sharp}}
        def_f          cell f's deficit, 0 to 1
        \rho_f         the membrane nodes bound per unit basal area of cell f (scaled by `receptor` R)
        p_{ref}        `p_ref`, the fraction of the median counted as full support
        \rho_g         the other cells' densities, their median the per-frame reference
        sharp          `sharp`, the exponent

    Reference: Streuli, C. H. (2009). Curr. Opin. Cell Biol. 21:194-198; the archive's chain, discovery_okuda/BUDDING_08.md."""
    PARAM_UNITS = {'p_ref': 'dimensionless', 'receptor': 'fraction', 'sharp': 'dimensionless', 'chan': 'count'}
    SYMBOLS = {'def_f': "cell f's deficit, 0 to 1", '\\rho_f': 'the membrane nodes bound per unit basal area of cell f (scaled by `receptor` R)', 'p_{ref}': '`p_ref`, the fraction of the median counted as full support', '\\rho_g': "the other cells' densities, their median the per-frame reference", 'sharp': '`sharp`, the exponent'}
    REQUIRES_PARAMS = []
    PARAM_ROLES = {"p_ref": "fraction_of_median_density_counted_as_anchored",
                   "sharp": "deficit_sharpness", "chan": "chem_channel_written",
                   "receptor": "integrin_level_fraction_of_normal", 'cell_set': 'the_cell_set', 'source': 'which_published_density_is_read'}

    def __init__(self, params, device="cpu"):
        Structural.__init__(self, params, device)
        self.at = params.get("_at", "vertex")
        self.cat = params.get("cell_set", "cell")
        self.p_ref = float(params.get("p_ref", 0.5))
        self.sharp = float(params.get("sharp", 1.0))
        self.chan = int(params.get("chan", 0))
        self.receptor = float(params.get("receptor", 1.0))
        # `source: matrix` (default `nodes`, the membrane's node density `bm_ligand`; exp 11 Phase 3 B6, 2026-09-29):
        # read the MATTER a cell stands on instead -- `bm_clutch`'s covered fraction x the mass of the nodes it
        # touches (`m["clutch_matrix"]`) -- so a membrane the protease has thinned reads as a deficit even where
        # `bm_secrete` has kept its node density even. Needs `bm_clutch` earlier in the same frame.
        self.source = str(params.get("source", "nodes")).lower()
        if self.source not in ("nodes", "matrix"):
            raise ValueError(f"bm_sense[live]: source is nodes or matrix, got {self.source!r}")
        self._said = False
        self._warned = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        clvl = H.level(self.cat)
        if m is None or clvl is None or "chem" not in getattr(clvl, "state_schema", {}):
            return {}
        nF = int(m["nF"])
        f = getattr(H, "frame", None)
        f = -1 if f is None else int(f)
        rho, fr = ((m.get("clutch_matrix"), m.get("clutch_matrix_frame")) if self.source == "matrix"
                   else (m.get("bm_ligand"), m.get("bm_ligand_frame")))
        if not torch.is_tensor(rho) or fr is None or f - int(fr) not in (0, 1) or rho.shape[0] != nF:
            if not self._warned and f > 1:
                self._warned = True
                print(f"[bm_sense/live] no current ligand density on the tissue mesh at frame {f} "
                      f"(published frame {fr}); no deficit written. Schedule "
                      f"`bm_contact[implementation: live]` every frame.", flush=True)
            return {}
        rho = rho.to(clvl.state.dtype)
        med = rho.median().clamp_min(1e-12)
        def_ = (1.0 - self.receptor * rho / (self.p_ref * med)).clamp(0.0, 1.0) ** self.sharp
        ci, _ = clvl.state_schema["chem"]
        clvl.state[:nF, ci + self.chan] = def_.to(clvl.state.dtype)
        SENSE_TRACE.append((int((def_ > 0.5).sum()), float(def_.max()), float(def_.mean())))
        if not self._said:
            self._said = True
            print(f"[bm_sense/live] membrane deficit into {self.cat}.chem[:, {self.chan}] from the "
                  f"live contact's ligand density (receptor {self.receptor:g}, p_ref {self.p_ref} of the median "
                  f"{float(med):.4g}, sharp {self.sharp}); frame {f}: "
                  f"{int((def_ > 0.5).sum())} of {nF} cells above 0.5", flush=True)
        return {}



# contact_retype -- a juxtacrine type switch (experiment 9, Toda et al. 2018's synNotch circuits).
from plexus.operators.channel_ops import _type_index  # noqa: E402  the type-name lookup pair_potential[typed] uses


PLATE_CONTACT: list = []


@register_operator("plate_confine", family="boundary", set="vertex", kind="structural",
                   equation=r"""$$g(t)=g_0+(g_1-g_0)\,\mathrm{clamp}\!\left(\frac{t-t_0}{t_1-t_0},0,1\right),\qquad x_a\leftarrow c_a+\mathrm{sign}(x_a-c_a)\min\big(\lvert x_a-c_a\rvert,\,g(t)\big)$$""", title="confine between two plates")
class PlateConfine3D(Structural):
    r"""confines a set between two rigid plates, a hard boundary imposed as a projection rather than a force, so nothing can be pushed through it however hard it is pressed; the gap closes linearly between two frames (a compression assay).
    vertex -> vertex: reads pos, writes pos in place (and kills the normal velocity on contact when the set carries one).

        g(t)=g_0+(g_1-g_0)\,\mathrm{clamp}\!\left(\frac{t-t_0}{t_1-t_0},0,1\right),\qquad x_a\leftarrow c_a+\mathrm{sign}(x_a-c_a)\min\big(\lvert x_a-c_a\rvert,\,g(t)\big)
        g(t)           the free half-gap at frame t
        g_0, g_1       `gap_half`, `gap_half_end`, world length
        t_0, t_1       `close_from`, `close_to`, frames
        x_a            the coordinate along the confined axis `axis`
        c_a            `centre`, the gap's centre on it

    Reference: Plexus (this work); the confinement geometry of Okuda, S. et al. (2018). Sci. Rep. 8:2386."""
    PARAM_UNITS = {'axis': 'count', 'centre': 'length', 'close_from': 'frames', 'close_to': 'frames', 'damp_normal': 'fraction', 'gap_half': 'length', 'gap_half_end': 'length', 'stiff': 'dimensionless'}
    SYMBOLS = {'g(t)': 'the free half-gap at frame t', 'g_0, g_1': '`gap_half`, `gap_half_end`, world length', 't_0, t_1': '`close_from`, `close_to`, frames', 'x_a': 'the coordinate along the confined axis `axis`', 'c_a': "`centre`, the gap's centre on it"}

    EMIT = None                        # moves positions in place; no integrable delta
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["gap_half"]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["rigid_confinement", "anisotropic_boundary", "solid_obstacle"]
    PARAM_ROLES = {"gap_half": "free_half_gap", "gap_half_end": "final_free_half_gap",
                   "close_from": "frame_closing_starts", "close_to": "frame_closing_ends", "stiff": "projection_fraction",
                   "axis": "confined_axis", "centre": "gap_centre_on_axis", 'damp_normal': 'fraction_of_the_normal_velocity_killed'}
    REFERENCE = ("Plexus (this work); the confinement geometry of Okuda, S. et al. (2018). "
                 "Sci. Rep. 8:2386.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.axis = int(params.get("axis", 2))
        self.centre = float(params.get("centre", 0.0))
        self.gap_half = float(params["gap_half"])
        self.stiff = float(params.get("stiff", 0.6))
        self.damp_normal = bool(params.get("damp_normal", True))
        # A MOVING PLATEN. `gap_half` alone is a gap that was always there, which answers "how does
        # a cell behave when confined" but not "what does confining it DO" -- and the second is the
        # question a squashing experiment asks. With `gap_half_end` the plates close linearly from
        # `close_from` to `close_to` (frames), so one run shows the whole sequence: sphere, ovoid,
        # then the flattened, mutually-pressed shape an epithelium has.
        #
        # WHY THE PLATEN MUST BE THIS OPERATOR AND NOT AN `obstacles:` BOX. The obstacle path
        # rasterises solid cells onto the MPM grid and zeroes their velocity, which stops a particle
        # entering but cannot EXPEL one already inside: a cell seeded across a plate keeps its shape
        # and simply freezes the overlapping shell (measured on cell_10 -- z/x aspect 1.00 against a
        # gap 13% narrower than the cell). This operator PROJECTS positions, so a plate that arrives
        # where matter already is pushes that matter out, which is what squashing means.
        self.gap_half_end = params.get("gap_half_end", None)
        self.gap_half_end = None if self.gap_half_end is None else float(self.gap_half_end)
        self.close_from = int(params.get("close_from", 0))
        self.close_to = int(params.get("close_to", 0))
        self._gap0 = self.gap_half
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        ax = self.axis
        # LIVE ENTITIES ONLY, when the level knows how many it has. A vertex buffer is mostly empty
        # slots; projecting them is harmless but it makes the contact COUNT meaningless, and that
        # count is the only evidence of when confinement began.
        m = getattr(lvl, "_mesh", None)
        n = int(m["Nv"]) if (m is not None and "Nv" in m) else pos.shape[0]

        if self.gap_half_end is not None and self.close_to > self.close_from:
            u = (int(getattr(H, "frame", 0)) - self.close_from) / (self.close_to - self.close_from)
            u = min(1.0, max(0.0, u))
            self.gap_half = self._gap0 + u * (self.gap_half_end - self._gap0)
        z = pos[:n, ax] - self.centre
        over = z.abs() - self.gap_half
        hit = over > 0
        n_hit = int(hit.sum())
        if n_hit:
            pos[:n, ax] = torch.where(
                hit, pos[:n, ax] - self.stiff * torch.sign(z) * over.clamp_min(0.0), pos[:n, ax])
            # `lvl.state` IS THE TENSOR; `lvl.state_schema` is what knows the block names. Probing
            # the tensor calls Tensor.__contains__ with a string, which raises -- so the vertex set,
            # which has no `vel` block at all, took down the whole run at frame 1.
            if self.damp_normal and "vel" in lvl.state_schema:
                v = lvl.get("vel")
                # KILL ONLY THE INTO-PLATE COMPONENT. Zeroing the whole velocity would apply a
                # friction the plate does not have, and a matrix particle sliding ALONG a plate is
                # exactly what the free directions are for.
                vn = v[:n, ax]
                v[:n, ax] = torch.where(hit & (vn * torch.sign(z) > 0),
                                        torch.zeros_like(vn), vn)
        PLATE_CONTACT.append((n_hit, float(over[hit].max()) if n_hit else 0.0))
        if not self._said:
            print(f"[plate_confine] {self.at}: rigid plates at {self.centre:+.4g} "
                  f"+/- {self.gap_half:.4g} along axis {ax}, stiff={self.stiff}; "
                  f"{n_hit} of {n} in contact at frame 0", flush=True)
            self._said = True
        return {}


# =========================================================== plate_confine[sphere]: a spherical capsule
# Folded here from its own module `sphere_confine.py` on 2026-09-27 (INSTRUCTION.md: every variant
# lives beside its base operator). What it was written for, verbatim:
#
# A spherical capsule -- `plate_confine[model: sphere]`: a closed tissue held inside a sphere of radius R.
#
#     plate_confine[sphere]   x <- c + (x - c) (R + (1 - stiff)(|x - c| - R)) / |x - c|   for |x - c| > R
#
# WHY A MODEL OF `plate_confine` AND NOT A NEW OPERATOR. It is the same contract -- a hard boundary imposed
# as a PROJECTION of positions, so nothing is pushed through it however hard it is pressed, with `stiff` the
# fraction of the violation corrected each frame -- on a different geometry. The two plates bound one axis;
# the capsule bounds the radius. Registered here, on the contract, so no existing operator is edited.
#
# WHY EXPERIMENT 13 NEEDS IT. Aegerter-Wilmsen et al. 2012 (Development 139:3221, p. 3223) build their
# size arrest on a condition the free shell of `config/tissue/exp03_base.yaml` does not meet: "As long as
# the disc stays flat, this will lead to a build up of compression in the center". A free shell relieves
# compression by bulging outward -- exp 13's fast clone rose out of the shell as a cap and folded it, and
# Shraiman 2005 Fig. 2 draws the same buckling -- so no compression builds with size and the crowding
# readout of `cell_grow[stretch]` slowed the tissue without stopping it (exp 13 findings 30, 35). The
# capsule is the wing disc's surroundings (peripodial epithelium, basement membrane, the larval body) in
# the one form a closed shell allows: it forbids growth OUTWARD past R, so a tissue that fills it is
# compressed in the plane, and the compression grows with every further increase of target volume.
#
# THE CENTRE is the live vertices' centroid by default (`centre: tissue`), so the capsule travels with the
# tissue and exerts no net force; `centre: [x, y, z]` pins it in world units. R is the capsule radius in
# world units, measured to the vertex positions (the apico-basal mid-surface on an apico-basal run).
#
# Reference: Aegerter-Wilmsen, T. et al. (2012). Development 139:3221 (the flat-disc condition); Plexus
# (this work) for the capsule itself.

SPHERE_CONTACT: list = []           # (vertices in contact, largest overshoot) per call, like PLATE_CONTACT


def project_into_sphere(pos, centre, radius, stiff):
    """(new positions, contact mask, overshoot): every point beyond `radius` of `centre` moved back by
    `stiff` of its overshoot along the radius; points inside are untouched."""
    d = pos - centre
    r = d.norm(dim=1)
    over = r - float(radius)
    hit = over > 0
    if bool(hit.any()):
        scale = torch.where(hit, (r - float(stiff) * over) / r.clamp(min=1e-12), torch.ones_like(r))
        pos = centre + d * scale[:, None]
    return pos, hit, over


@register_operator("plate_confine", model="sphere", family="boundary", set="vertex", kind="structural",
                   equation=r"""$$\mathbf x\leftarrow\mathbf c+(\mathbf x-\mathbf c)\,\frac{R+(1-s)(\lvert\mathbf x-\mathbf c\rvert-R)}{\lvert\mathbf x-\mathbf c\rvert}\quad\text{for }\lvert\mathbf x-\mathbf c\rvert>R$$""", title="confine inside a sphere")
class SphereConfine3D(Structural):
    r"""confines a set inside a sphere by projection, with the outward velocity killed on contact, as the plates do for their normal.
    vertex -> vertex: reads pos, writes pos in place; kills the outward component of `vel` on contact when the set carries one.

        \mathbf x\leftarrow\mathbf c+(\mathbf x-\mathbf c)\,\frac{R+(1-s)(\lvert\mathbf x-\mathbf c\rvert-R)}{\lvert\mathbf x-\mathbf c\rvert}\quad\text{for }\lvert\mathbf x-\mathbf c\rvert>R
        x              a vertex position
        c              `centre`
        R              `radius`
        s              `stiff`, the projection's share (1 = hard)

    Reference: as plate_confine; Plexus (this work)."""
    PARAM_UNITS = {'centre': 'length', 'damp_normal': 'fraction', 'radius': 'length', 'stiff': 'dimensionless'}
    SYMBOLS = {'x': 'a vertex position', 'c': '`centre`', 'R': '`radius`', 's': "`stiff`, the projection's share (1 = hard)"}

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["radius"]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["rigid_confinement", "capsule", "solid_obstacle"]
    PARAM_ROLES = {"radius": "capsule_radius_world_units", "stiff": "projection_fraction",
                   "centre": "tissue_or_world_point", 'damp_normal': 'fraction_of_the_outward_velocity_killed'}
    REFERENCE = "Aegerter-Wilmsen, T. et al. (2012). Development 139:3221; Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.radius = float(params["radius"])
        self.stiff = float(params.get("stiff", 0.6))
        c = params.get("centre", "tissue")
        self.centre = None if (c is None or str(c).lower() == "tissue") else [float(v) for v in c]
        self.damp_normal = bool(params.get("damp_normal", True))
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        m = getattr(lvl, "_mesh", None)
        n = int(m["Nv"]) if (m is not None and "Nv" in m) else pos.shape[0]
        p = pos[:n]
        c = (p.mean(0, keepdim=True) if self.centre is None
             else torch.as_tensor(self.centre, dtype=p.dtype, device=p.device)[None, :])
        new, hit, over = project_into_sphere(p, c, self.radius, self.stiff)
        n_hit = int(hit.sum())
        if n_hit:
            pos[:n] = new
            if self.damp_normal and "vel" in lvl.state_schema:
                v = lvl.get("vel")
                d = new - c
                u = d / d.norm(dim=1, keepdim=True).clamp(min=1e-12)
                vr = (v[:n] * u).sum(1)
                out = hit & (vr > 0)
                v[:n] = torch.where(out[:, None], v[:n] - vr[:, None] * u, v[:n])
        SPHERE_CONTACT.append((n_hit, float(over[hit].max()) if n_hit else 0.0))
        if not self._said:
            print(f"[plate_confine:sphere] {self.at}: capsule of radius {self.radius:.4g} around "
                  f"{'the tissue centroid' if self.centre is None else self.centre}, stiff={self.stiff}; "
                  f"{n_hit} of {n} vertices in contact at the first call", flush=True)
            self._said = True
        return {}


def block_fraction(gap_half, half_extent):
    """Fraction of the domain's volume the two blocks occupy, for a box of half-width `half_extent`.

    Reported rather than specified, because the two things a caller wants to control -- how squashed
    the tissue is, and how much of the box is solid -- are the SAME number seen twice, and only one of
    them can be set. See `PLATES.md` for the arithmetic that pins them together.
    """
    free = min(1.0, max(0.0, gap_half / half_extent))
    return 1.0 - free


# ==========================================================================================================
# THE PISTON: a surface this repository can SEED, so the interaction has a second geometry.
#
# `mesh_contact` has only ever met one surface -- a growing sphere, replayed from a cache -- and every
# gate row about it is a row about that sphere. What the operator claims is more general than that:
# a triangulated surface, of any shape, pushing a continuum. A claim exercised on one geometry is a
# claim about that geometry.
#
# SO THE SURFACE BECOMES A SEED OPERATOR RATHER THAN A FILE. `seed_plate` builds the patch at frame 0
# the way `seed_mesh` builds the vesicle, and `plate_drive` prescribes its motion, so a spec declares
# the piston instead of pointing at an npz somebody generated. The four shapes are one lattice under
# four masks, which is what makes them comparable: same resolution, same edge length, same bin count
# per unit area -- geometry is the only thing that differs across the ladder.
# ==========================================================================================================
def _plate_lattice(nq, half_width, shape, hole_frac=0.35, bars=4, bar_frac=0.5):
    """One `nq` x `nq` quad lattice minus whatever `shape` removes: (xy of kept vertices, quads).

    THE MASK IS ON THE QUAD CENTRES, not on the vertices, so a removed quad takes no vertex its
    neighbours still need and the kept set is exactly the boundary of what survives.
    """
    g = np.linspace(-half_width, half_width, nq + 1)
    X, Y = np.meshgrid(g, g, indexing="ij")
    vid = np.arange((nq + 1) ** 2).reshape(nq + 1, nq + 1)
    i, j = np.meshgrid(np.arange(nq), np.arange(nq), indexing="ij")
    quads = np.stack([vid[i, j], vid[i + 1, j], vid[i + 1, j + 1], vid[i, j + 1]], -1).reshape(-1, 4)
    cx = 0.5 * (g[:-1] + g[1:])
    CX, CZ = (a.reshape(-1) for a in np.meshgrid(cx, cx, indexing="ij"))
    r = np.hypot(CX, CZ)
    if shape == "sheet":
        keep = np.ones(len(quads), bool)
    elif shape == "disc":
        keep = r <= half_width
    elif shape == "disc_hole":
        keep = (r <= half_width) & (r >= hole_frac * half_width)
    elif shape == "grid":
        # A LATTICE OF BARS: solid strips with square windows between them, so the material is
        # loaded in patches and can extrude through the gaps. `bar_frac` is the fraction of each
        # period the strip occupies, so 1.0 recovers `sheet` and the shape is a continuum.
        per = 2 * half_width / bars
        u = ((CX + half_width) % per) / per
        v = ((CZ + half_width) % per) / per
        w = 0.5 * bar_frac
        keep = (np.minimum(u, 1 - u) < w) | (np.minimum(v, 1 - v) < w)
    else:
        raise ValueError(f"seed_plate: shape must be sheet|disc|disc_hole|grid, got {shape!r}")
    quads = quads[keep]
    if not len(quads):
        raise ValueError(f"seed_plate: shape {shape!r} kept no quads at nq={nq}")
    used = np.unique(quads)
    remap = np.full(vid.size, -1, np.int64)
    remap[used] = np.arange(len(used))
    return np.stack([X.reshape(-1)[used], Y.reshape(-1)[used]], -1), remap[quads]


def _plate_half_edges(quads):
    """(E_srce, E_trgt, E_face) -- four half-edges per quad, in ring order, grouped by face.

    QUADS AND NOT TRIANGLES because `mesh_contact` fans every face from its own centroid, one
    sub-triangle per half-edge, and needs no ring ordering to do it. A quad therefore costs four
    sub-triangles and buys a lattice that can have holes punched in it by dropping whole faces.
    """
    nf = len(quads)
    es = np.concatenate([quads[:, k] for k in (0, 1, 2, 3)])
    et = np.concatenate([quads[:, k] for k in (1, 2, 3, 0)])
    ef = np.tile(np.arange(nf), 4)
    o = np.argsort(ef, kind="stable")
    return es[o], et[o], ef[o]


@register_operator("seed_plate", family="seed", set="vertex", kind="seed", title="lay out the plate", equation=r"""$$\mathbf x_{uv}=\mathbf c+h\,\hat{\mathbf e}_{axis}+W\Big(\frac{2u}{n_q}-1\Big)\hat{\mathbf e}_1+W\Big(\frac{2v}{n_q}-1\Big)\hat{\mathbf e}_2,\quad u,v=0\ldots n_q$$""")
class SeedPlate(Structural):
    r"""an open planar half-edge patch laid out once at frame 0: the rigid tool a parallel-plate or nanoindentation assay presses with (sheet, disc, annulus, bars).
    vertex -> vertex: writes every position and the mesh's half-edge table at the opening of the trajectory; the standoff from the contact's centre is reported at frame 0.

        \mathbf x_{uv}=\mathbf c+h\,\hat{\mathbf e}_{axis}+W\Big(\frac{2u}{n_q}-1\Big)\hat{\mathbf e}_1+W\Big(\frac{2v}{n_q}-1\Big)\hat{\mathbf e}_2,\quad u,v=0\ldots n_q
        x_{uv}         the vertex at grid index (u, v)
        c              `centre`
        h              `height`, the patch's position along `axis`
        W              `half_width`
        n_q            `nq`, quads across
        \hat e_1, \hat e_2 the in-plane axes

    Reference: Plexus (this work); the indenter geometry of a nanoindentation or parallel-plate compression assay; the half-edge layout of Okuda, S. et al. (2013). Biomech. Model. Mechanobiol. 12:627-644."""
    PARAM_UNITS = {'axis': 'count', 'bar_frac': 'fraction', 'bars': 'count', 'centre': 'length', 'half_width': 'length', 'height': 'length', 'hole_frac': 'fraction', 'nq': 'count', 'profile_depth': 'length', 'standoff': 'length'}
    SYMBOLS = {'x_{uv}': 'the vertex at grid index (u, v)', 'c': '`centre`', 'h': "`height`, the patch's position along `axis`", 'W': '`half_width`', 'n_q': '`nq`, quads across', '\\hat e_1, \\hat e_2': 'the in-plane axes'}

    EMIT = None                        # writes positions and the mesh table; no integrable delta
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["planar_patch", "half_edge_mesh", "rigid_indenter", "initial_condition"]
    PARAM_ROLES = {"shape": "patch_topology", "nq": "quads_across", "half_width": "patch_half_extent",
                   "height": "position_on_axis", "axis": "patch_normal_axis",
                   "hole_frac": "inner_over_outer_radius", "bars": "strips_per_side",
                   "bar_frac": "strip_width_over_period", 'centre': 'centre_of_the_patch', 'profile': 'height_profile_of_the_patch', 'profile_depth': 'depth_of_that_profile', 'standoff': 'distance_from_the_contacts_centre_reported'}
    REFERENCE = ("Plexus (this work); the indenter geometry of a nanoindentation or "
                 "parallel-plate compression assay. The half-edge layout is Okuda, S. et al. "
                 "(2013). Biomech. Model. Mechanobiol. 12:627-644.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.shape = str(params.get("shape", "sheet"))
        self.nq = int(params.get("nq", 48))
        self.half_width = float(params.get("half_width", 0.25))
        self.axis = int(params.get("axis", 1))
        self.height = float(params.get("height", 0.55))
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5])]   # the two IN-PLANE axes
        # THE PROFILE IS ORTHOGONAL TO THE SHAPE, and keeping them apart is the point. `shape`
        # decides which quads of the lattice survive (sheet / disc / disc with a hole / grid) and
        # `profile` displaces the survivors along the normal -- so `disc` + `cone` is a conical
        # indenter and `sheet` + `pyramid` a Vickers-like one, from one seeder and one lattice.
        # Fold them into a single `shape:` list and every new tip costs a new mask.
        #
        #   flat     the plane, and the default: 461 specs' worth of nothing changes.
        #   pyramid  apex DOWN at the centre, four faces rising to the rim -- Chebyshev distance.
        #   cone     the same with a radial distance, i.e. the axisymmetric tip.
        #
        # `profile_depth` is the APEX-TO-RIM rise. It is declared rather than an angle because the
        # angle also depends on `half_width`, and two numbers that pin one geometry is one too many;
        # the half-angle it implies is printed instead, since that is what an indenter is specified
        # by (Berkovich 70.3 deg, Vickers 68 deg, both as equivalent cones).
        self.profile = str(params.get("profile", "flat"))
        self.profile_depth = float(params.get("profile_depth", 0.0))
        self.hole_frac = float(params.get("hole_frac", 0.35))
        self.bars = int(params.get("bars", 4))
        self.bar_frac = float(params.get("bar_frac", 0.5))
        self.standoff = params.get("standoff", None)
        self.standoff = None if self.standoff is None else float(self.standoff)

    def forward(self, H, mask=None):
        from plexus.models.mesh import MeshTable
        lvl = H.level(self.at)
        dev, dt_ = lvl.state.device, lvl.state.dtype
        xy, quads = _plate_lattice(self.nq, self.half_width, self.shape,
                                   self.hole_frac, self.bars, self.bar_frac)
        es, et, ef = _plate_half_edges(quads)
        Nv, nF = len(xy), len(quads)
        Nbuf = lvl.state.shape[0]
        if Nv > Nbuf:
            raise ValueError(f"seed_plate: the {self.shape} patch has {Nv} vertices but the "
                             f"`{self.at}` set declares n={Nbuf}")
        ip = [a for a in (0, 1, 2) if a != self.axis]              # the two in-plane axes
        pos = torch.zeros(Nbuf, 3, dtype=dt_, device=dev)
        pos[:Nv, ip[0]] = torch.as_tensor(xy[:, 0] + self.centre[0], dtype=dt_, device=dev)
        pos[:Nv, ip[1]] = torch.as_tensor(xy[:, 1] + self.centre[1], dtype=dt_, device=dev)
        pos[:Nv, self.axis] = self.height
        if self.profile != "flat":
            if self.profile == "pyramid":
                d = np.maximum(np.abs(xy[:, 0]), np.abs(xy[:, 1]))       # square, apex-down
            elif self.profile == "cone":
                d = np.hypot(xy[:, 0], xy[:, 1])                         # axisymmetric
            else:
                raise ValueError(f"seed_plate: profile must be flat|pyramid|cone, "
                                 f"got {self.profile!r}")
            # THE APEX SITS AT `height` AND THE RIM RISES ABOVE IT, so `height` keeps meaning "the
            # lowest point of the tool" across the whole ladder -- the number a descent is measured
            # against. Defining it at the rim instead would make the flat plate and the pyramid
            # start their contact at different depths for the same declared height.
            pos[:Nv, self.axis] += torch.as_tensor(
                self.profile_depth * d / max(self.half_width, 1e-12), dtype=dt_, device=dev)
        p0, p1 = lvl.state_schema["pos"]
        st = lvl.state.clone(); st[:, p0:p1] = pos; lvl.state = st
        if getattr(lvl, "occ", None) is not None:
            occ = torch.zeros(Nbuf, device=dev); occ[:Nv] = 1.0; lvl.occ = occ
        seeded = dict(E_srce=torch.as_tensor(es, device=dev),
                      E_trgt=torch.as_tensor(et, device=dev),
                      E_face=torch.as_tensor(ef, device=dev), nF=nF, Nv=Nv,
                      # THE SURFACE VELOCITY LIVES ON THE TABLE, not in a `vel` state block, because
                      # it is a property of the SURFACE and `mesh_contact` reads it as one -- the
                      # replay path derives exactly this by differencing consecutive cached meshes.
                      # A vertex set that declared `vel` would also have the engine integrate it,
                      # and a prescribed piston must not be integrated.
                      Vv=torch.zeros(Nv, 3, dtype=dt_, device=dev),
                      plate_axis=self.axis, plate_height=self.height,
                      verts0=pos[:Nv].detach().to("cpu").numpy())
        m = getattr(lvl, "_mesh", None)
        if isinstance(m, MeshTable):
            m.clear(); m.update(seeded)
        else:
            lvl._mesh = MeshTable(**seeded)
        edge = 2.0 * self.half_width / self.nq
        so = "" if self.standoff is None else (
            f"; standoff {self.standoff:.4g} from the declared contact centre -> a face subtends "
            f"~{edge / self.standoff:.4f} rad, i.e. ~{int(math.pi / max(edge / self.standoff, 1e-3))} "
            f"bin rows (the contact caps at 200 and floors at 4)")
        pr = ""
        if self.profile != "flat":
            pr = (f", {self.profile} apex-down: rim {self.profile_depth:.4g} above the apex, "
                  f"half-angle {math.degrees(math.atan2(self.half_width, max(self.profile_depth, 1e-12))):.1f} "
                  f"deg from the axis (Berkovich 70.3, Vickers 68 as equivalent cones)")
        print(f"[seed_plate] {self.shape}: {nF} quads ({4 * nF} sub-triangles), {Nv} vertices, "
              f"edge {edge:.4f}, half-width {self.half_width} at axis-{self.axis} = "
              f"{self.height}{pr}{so}", flush=True)
        return {}


@register_operator("surface_drive", "plate_drive", family="mechanics", set="vertex",
                   kind="structural", title="move the surface at a set rate",
                   equation=r"""$$x_a(t)\leftarrow x_a(0)+b_y\,\mathrm{clamp}\!\left(\frac{t-t_{\mathrm{hold}}}{t_{\mathrm{over}}},0,1\right),\qquad v_a=\frac{b_y}{t_{\mathrm{over}}}$$""")
class SurfaceDrive(Structural):
    r"""moves a seeded surface along one axis at a prescribed rate and publishes its velocity: the loading protocol of a displacement-controlled indentation, kinematic not dynamic.
    vertex -> vertex: reads pos, writes pos and the surface's velocity buffer in place.

        x_a(t)\leftarrow x_a(0)+b_y\,\mathrm{clamp}\!\left(\frac{t-t_{\mathrm{hold}}}{t_{\mathrm{over}}},0,1\right),\qquad v_a=\frac{b_y}{t_{\mathrm{over}}}
        x_a(t)         the surface's coordinate along `axis` at frame t
        x_a(0)         its seeded coordinate
        b_y            `by`, the signed displacement (or `target`, an absolute final centroid coordinate)
        t_{hold}       `hold`, frames of rest before the travel
        t_{over}       `over`, frames of travel
        v_a            the published velocity

    Reference: Plexus (this work); the loading protocol of a displacement-controlled indentation."""
    PARAM_UNITS = {'axis': 'count', 'by': 'length', 'hold': 'frames', 'over': 'frames', 'target': 'length'}
    SYMBOLS = {'x_a(t)': "the surface's coordinate along `axis` at frame t", 'x_a(0)': 'its seeded coordinate', 'b_y': '`by`, the signed displacement (or `target`, an absolute final centroid coordinate)', 't_{hold}': '`hold`, frames of rest before the travel', 't_{over}': '`over`, frames of travel', 'v_a': 'the published velocity'}

    EMIT = None                        # moves positions in place; no integrable delta
    SUPPORTED_DIMS = [3]
    # `target`, not `to`: the schema reserves `to` and `from` for FIELD references, so a scalar
    # there would be resolved as a field name and the specification would fail to load.
    REQUIRES_PARAMS = []
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["prescribed_kinematics", "rigid_indenter", "moving_boundary"]
    PARAM_ROLES = {"by": "signed_displacement_along_axis",
                   "target": "final_centroid_position_on_axis", "over": "frames_of_travel",
                   "rigid": "reset_from_seed_or_translate_live",
                   "axis": "motion_axis", "hold": "frames_held_before_moving", 'grow': 'whether_the_surface_also_scales'}
    REFERENCE = "Plexus (this work); the loading protocol of a displacement-controlled indentation."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.axis = int(params.get("axis", 1))
        # Growth as a prescription: the third way a tool can move, and the one a spheroid needs.
        # `by` and `target` translate the surface; `grow` SCALES it about its own centroid by a
        # declared factor over `over` frames, so a shell can indent a gel by inflating instead of by
        # descending -- and it stays a kinematic tool while doing it.
        #
        # Not `cell_grow` plus `cell_mechanics`: those SOLVE a tissue, and a solved tissue sharing
        # a schedule with an MPM continuum is a different experiment, since the two have timesteps
        # orders of magnitude apart. Prescribing the growth keeps the coupling one-way on purpose:
        # the gel feels the shell and the shell feels nothing.
        self.grow = params.get("grow", None)
        self.grow = None if self.grow is None else float(self.grow)
        if self.grow is not None:
            if "target" in params or "by" in params:
                raise ValueError("surface_drive: `grow` scales the surface and `by`/`target` "
                                 "translate it -- declare one motion, not two")
        elif ("target" in params) == ("by" in params):
            raise ValueError("surface_drive: declare exactly one of `grow:` (a scale factor "
                             "about the centroid), `by:` (a signed displacement "
                             "along the axis, the same for every tool shape) or `target:` (an "
                             "absolute coordinate of the surface's centroid)")
        self.target = float(params["target"]) if "target" in params else None
        self.by = float(params["by"]) if "by" in params else None
        self.over = int(params.get("over", 0))          # 0 -> the whole run, resolved at frame 0
        self.hold = int(params.get("hold", 0))
        # RIGID BY DEFAULT: a tool's shape is its own, and resetting from the seeded positions is
        # exact at every frame instead of accumulating one increment's rounding per step.
        self.rigid = bool(params.get("rigid", True))
        self._from = None
        self._v0 = None
        self._y_prev = None
        self._f_prev = 1.0
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        if m is None or not int(m.get("Nv", 0)):
            return {}                                   # the seed has not run yet
        Nv = int(m["Nv"])
        pos = lvl.get("pos")
        if self._from is None:
            v0 = m.get("verts0", None)
            self._v0 = (torch.as_tensor(v0[:Nv, self.axis], device=pos.device, dtype=pos.dtype)
                        if v0 is not None else pos[:Nv, self.axis].clone())
            self._from = float(self._v0.mean())      # the surface's own centroid on that axis
            _p0 = m.get("verts0", None)
            self._p0 = (torch.as_tensor(np.asarray(_p0)[:Nv], device=pos.device, dtype=pos.dtype)
                        if _p0 is not None else pos[:Nv].clone())
            self._c0 = self._p0.mean(0, keepdim=True)
            if self.grow is not None:
                self.target = self._from
            if self.target is None:
                self.target = self._from + self.by
            self._y_prev = self._from
            self.over = self.over or max(1, int(getattr(H, "n_frames", 0) or 1) - self.hold)
        f = int(getattr(H, "frame", 0) or 0)
        u = min(1.0, max(0.0, (f - self.hold) / float(self.over)))
        y = self._from + u * (self.target - self._from)
        # ZERO BEFORE AND AFTER THE TRAVEL. Hoisted above the branches because BOTH need it and the
        # growth branch returns before the translation branch's copy was ever reached.
        moving = (self.hold <= f < self.hold + self.over)
        # A TRANSLATION OF THE SEEDED SHAPE, NOT AN ASSIGNMENT. `pos[:Nv, axis] = y` is the same
        # thing for a plate, whose vertices all share one coordinate on that axis -- and it FLATTENS
        # a sphere into a disc on the first frame. The offset is measured from the seeded positions,
        # so the shape is exact at every frame rather than drifting with whatever the last one did.
        if self.grow is not None:
            # ABOUT THE SEEDED CENTROID, from the SEEDED positions -- exact at every frame rather
            # than a product of 400 incremental multiplications.
            f = 1.0 + u * (self.grow - 1.0)
            if self.rigid:
                # FROM THE SEEDED SHAPE: exact at every frame, and right for a tool.
                pos[:Nv] = self._c0 + (self._p0 - self._c0) * f
            else:
                # INCREMENTAL, ABOUT THE LIVE CENTROID, so whatever else shaped the surface this
                # frame survives. Resetting from the seed is what a rigid tool wants and what
                # DESTROYS a solved one: `cell_mechanics` would run every frame and be discarded,
                # and the shell would read as rigid while a schedule full of operators said
                # otherwise. The ratio is per-frame, so 400 of them compose to `grow`.
                _c = pos[:Nv].mean(0, keepdim=True)
                _r = f / max(self._f_prev, 1e-12)
                pos[:Nv] = _c + (pos[:Nv] - _c) * _r
            self._f_prev = f
            v = ((self.grow - 1.0) / (self.over * float(getattr(H, "dt", 1.0)))) if moving else 0.0
            vv = m.get("Vv", None)
            if vv is None or vv.shape[0] < Nv:
                vv = torch.zeros(pos.shape[0], 3, device=pos.device, dtype=pos.dtype)
                m["Vv"] = vv
            # THE SURFACE VELOCITY OF AN INFLATING SHELL IS RADIAL and proportional to the radius,
            # which is what the friction law needs and what a single axis-aligned number cannot say.
            #
            # FROM THE LIVE POSITIONS, NOT THE SEEDED ONES. `_p0` is the frame-0 vertex array, and
            # `cell_divide` APPENDS vertices -- so the moment a cell split this was 796 rows being
            # written into 1,160 and the run died on a shape mismatch. The live array is the right
            # one anyway: after a division the new vertices have a radius too, and it is theirs.
            _cv = pos[:Nv].mean(0, keepdim=True)
            vv[:Nv] = (pos[:Nv] - _cv) * v
            self._y_prev = y
            self._say(v)
            return {}
        if self.rigid:
            pos[:Nv, self.axis] = self._v0 + (y - self._from)
        else:
            # INCREMENTAL, so whatever else moved the surface this frame SURVIVES. Resetting from
            # the seeded positions is exact for a rigid tool and destroys a solved one: a spheroid
            # under `cell_mechanics` would have its shape overwritten on the drive axis every frame,
            # so the mechanics would run and be discarded and the surface would read as rigid while
            # a schedule full of operators said otherwise.
            pos[:Nv, self.axis] = pos[:Nv, self.axis] + (y - self._y_prev)
        self._y_prev = y
        # `moving` IS HOISTED ABOVE: not the mean rate, because a friction law told the tool is
        # still moving while it is parked would shear the material it is resting on for the whole
        # hold, and the hold exists precisely to show the material at rest under a static load.
        v = (self.target - self._from) / (self.over * float(getattr(H, "dt", 1.0))) if moving else 0.0
        # THE VELOCITY BUFFER IS THIS OPERATOR'S, NOT THE SEED'S. `seed_plate` allocated it and
        # `seed_mesh` -- which predates all of this and serves 461 specs -- does not, so a surface
        # built by the second and driven by this died on `KeyError: 'Vv'`. The operator that
        # publishes a quantity is the one that should create the place to put it.
        vv = m.get("Vv", None)
        if vv is None or vv.shape[0] < Nv:
            vv = torch.zeros(pos.shape[0], 3, device=pos.device, dtype=pos.dtype)
            m["Vv"] = vv
        vv[:Nv] = 0.0
        vv[:Nv, self.axis] = v
        m["plate_height"] = y
        self._say(v)
        return {}

    def _say(self, v):
        if not self._said:
            if self.grow is not None:
                print(f"[surface_drive] {self.at}: scaling x{self.grow:g} about its centroid over "
                      f"{self.over} frames after a {self.hold}-frame hold", flush=True)
                self._said = True
                return
            print(f"[surface_drive] {self.at}: axis {self.axis}, {self._from:.4g} -> {self.target:.4g} "
                  f"over {self.over} frames after a {self.hold}-frame hold; "
                  f"speed {abs(v):.5g} box units per unit time", flush=True)
            self._said = True
        return {}


@register_entity("surface")            # registered under the SET's name, not the class's
class SurfaceElement:
    """One patch of a tracked epithelial surface: a direction u, the radius R along it, and the
    buffers those live in.

    Registered under the name `surface` because entities resolve BY SET NAME. An unregistered name
    falls back silently to a bare position/velocity schema, and the operator then dies on a
    missing `u` buffer -- which reads like a bug in the operator and is a missing registration.
    """
    @staticmethod
    def provision(lvl, parent, s, H, device):
        n = lvl.get("pos").shape[0]
        lvl.register_buffer("u", torch.zeros(n, 3, device=device))
        lvl.register_buffer("R", torch.zeros(n, device=device))


LOAD_TRACE: list = []



@register_operator("mesh_contact", model="centre", family="boundary", set="particle",
                   kind="lateral", title="push material off a surface, by direction", equation=r"""$$\mathbf a_n=k\,d_i\,\mathbf n,\qquad \mathbf v_t=(\mathbf v_i-\mathbf v_{\mathrm{face}})-\big((\mathbf v_i-\mathbf v_{\mathrm{face}})\cdot\mathbf n\big)\mathbf n,\qquad \mathbf a_t=-\mu\lVert\mathbf a_n\rVert\frac{\mathbf v_t}{\lVert\mathbf v_t\rVert+\varepsilon_v}$$""")
class MeshContactCentre(MeshContact):
    r"""mesh_contact as it has always worked, named: the surface indexed by direction from a centre (a particle's direction bin and its eight neighbours), exact while a face spans at most one bin.
    particle -> particle: as mesh_contact.

        \mathbf a_n=k\,d_i\,\mathbf n,\qquad \mathbf v_t=(\mathbf v_i-\mathbf v_{\mathrm{face}})-\big((\mathbf v_i-\mathbf v_{\mathrm{face}})\cdot\mathbf n\big)\mathbf n,\qquad \mathbf a_t=-\mu\lVert\mathbf a_n\rVert\frac{\mathbf v_t}{\lVert\mathbf v_t\rVert+\varepsilon_v}
        a_n            the penalty acceleration, normal to the face
        k              the penalty stiffness, `k_frac` of the largest the substep carries stably
        d_i            the depth of particle i behind its nearest face
        n              the face normal
        v_t            the relative tangential velocity of particle and face
        v_i, v_{face}  the particle's and the face's velocities
        a_t            the friction acceleration
        \mu            `mu`, the Coulomb friction coefficient
        \varepsilon_v  `eps_v`, the velocity that regularises the friction's direction

    Reference: Chen, Z., Qiu, X., Zhang, X. & Lian, Y. (2015). Improved coupling of finite element method with material point method based on a particle-to-surface contact algorithm. Comput. Methods Appl. Mech. Engrg. 293:1-19."""
    PARAM_UNITS = {}
    SYMBOLS = {'a_n': 'the penalty acceleration, normal to the face', 'k': 'the penalty stiffness, `k_frac` of the largest the substep carries stably', 'd_i': 'the depth of particle i behind its nearest face', 'n': 'the face normal', 'v_t': 'the relative tangential velocity of particle and face', 'v_i, v_{face}': "the particle's and the face's velocities", 'a_t': 'the friction acceleration', '\\mu': '`mu`, the Coulomb friction coefficient', '\\varepsilon_v': "`eps_v`, the velocity that regularises the friction's direction"}


@register_operator("mesh_contact", model="adhesive_plane", family="boundary", set="particle",
                   kind="lateral", title="stick material to a flat substrate",
                   equation=r"""$$a_n=k\,(z_0-z_i)^+\;-\;a_{\rm adh}\sin\!\big(\pi\,u_i\big)\,[0\le u_i\le 1],\quad u_i=\frac{z_i-z_0}{b},\qquad \mathbf a_t=-\mu\,|a_n|\,\frac{\mathbf v_t}{\lVert\mathbf v_t\rVert+\varepsilon_v}$$""")
class MeshContactAdhesivePlane(Lateral):
    r"""mesh_contact against a flat substrate the material sticks to, an epithelium's basement membrane as a plane: points below it are pushed out, points within a thin band above it are held down, and both slide on it with friction.
    particle -> particle: reads pos and vel; emits an external acceleration the MPM substep consumes.

        a_n=k\,(z_0-z_i)^+\;-\;a_{\rm adh}\sin\!\big(\pi\,u_i\big)\,[0\le u_i\le 1],\quad u_i=\frac{z_i-z_0}{b},\qquad \mathbf a_t=-\mu\,|a_n|\,\frac{\mathbf v_t}{\lVert\mathbf v_t\rVert+\varepsilon_v}
        a_n            the normal acceleration along `axis`
        k              `k`, the penalty stiffness, per time squared
        z_0            `z`, the plane's coordinate
        z_i            the particle's coordinate along the axis
        a_{adh}        `adhesion`, the hold per unit mass
        u_i            the height above the plane in bands
        b              `band`, the adhesive band's thickness
        a_t            the friction acceleration
        \mu            `mu`
        v_t            the tangential velocity
        \varepsilon_v  `eps_v`

    Reference: Chen, Z., Qiu, X., Zhang, X. & Lian, Y. (2015). Improved coupling of finite element method with material point method based on a particle-to-surface contact algorithm. Comput. Methods Appl. Mech. Engrg. 293:1-19; Wang, S. et al. (2021). Cell 184:3702-3716 (the salivary epithelium on its membrane)."""
    PARAM_UNITS = {'adhesion': 'acceleration', 'axis': 'count', 'band': 'length', 'eps_v': 'velocity', 'k': 'acceleration', 'mu': 'dimensionless', 'z': 'length'}
    SYMBOLS = {'a_n': 'the normal acceleration along `axis`', 'k': '`k`, the penalty stiffness, per time squared', 'z_0': "`z`, the plane's coordinate", 'z_i': "the particle's coordinate along the axis", 'a_{adh}': '`adhesion`, the hold per unit mass', 'u_i': 'the height above the plane in bands', 'b': "`band`, the adhesive band's thickness", 'a_t': 'the friction acceleration', '\\mu': '`mu`', 'v_t': 'the tangential velocity', '\\varepsilon_v': '`eps_v`'}
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["z"]
    MECHANISM_TAGS = ["cell_matrix_contact", "adhesion", "friction", "substrate", "basement_membrane"]
    PARAM_ROLES = {"z": "substrate_plane_world", "axis": "plane_normal_axis", "side": "which_side_the_material_is_on",
                   "k": "penalty_per_depth",
                   "adhesion": "hold_per_unit_mass", "band": "adhesive_band_height_world", "mu": "friction_coefficient",
                   "eps_v": "slip_regularisation_velocity"}
    REFERENCE = ("Chen, Z. et al. (2015). Comput. Methods Appl. Mech. Engrg. 293:1-19; Wang, S. et al. (2021). "
                 "Cell 184:3702-3716.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.z0 = self.tunable(params["z"], None, tensor=True)
        self.axis = int(params.get("axis", 2))
        self.k = float(params.get("k", 100.0))
        self.a_adh = float(params.get("adhesion", 0.0))
        self.band = float(params.get("band", 0.1))
        self.mu = float(params.get("mu", 0.0))
        self.eps_v = float(params.get("eps_v", 0.05))
        self.side = float(params.get("side", 1.0))
        if self.side not in (1.0, -1.0):
            raise ValueError("mesh_contact[adhesive_plane]: side is +1 (material above the plane) or -1 (below)")
        if self.band <= 0.0 or self.k < 0.0 or self.a_adh < 0.0 or self.mu < 0.0:
            raise ValueError("mesh_contact[adhesive_plane]: band > 0 and k, adhesion, mu >= 0")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        X = lvl.get("pos")
        h = self.side * (X[:, self.axis] - self.z0)                    # > 0 on the material's side
        an = torch.where(h < 0.0, -self.k * h, torch.zeros_like(h))
        if self.a_adh > 0.0:
            u = h / self.band
            an = an + torch.where((u >= 0.0) & (u <= 1.0), -self.a_adh * torch.sin(math.pi * u), torch.zeros_like(h))
        a = torch.zeros_like(X)
        a[:, self.axis] = self.side * an
        if self.mu > 0.0 and "vel" in lvl.state_schema:
            v = lvl.get("vel")
            vt = v.clone()
            vt[:, self.axis] = 0.0
            a = a - (self.mu * an.abs() / (vt.norm(dim=1) + self.eps_v))[:, None] * vt
        a = a * lvl.occ[:, None].to(X.dtype)
        if mask is not None:
            a = a * mask[:, None].to(X.dtype)
        return {self.at: a}


@register_operator("mesh_contact", model="spatial_hash", family="boundary", set="particle",
                   kind="lateral", title="push material off a surface, by position", equation=r"""$$\mathbf a_n=k\,d_i\,\mathbf n,\qquad \mathbf v_t=(\mathbf v_i-\mathbf v_{\mathrm{face}})-\big((\mathbf v_i-\mathbf v_{\mathrm{face}})\cdot\mathbf n\big)\mathbf n,\qquad \mathbf a_t=-\mu\lVert\mathbf a_n\rVert\frac{\mathbf v_t}{\lVert\mathbf v_t\rVert+\varepsilon_v}$$""")
class MeshContactHash(MeshContact):
    r"""the same contact with candidates found by position instead of direction, a uniform spatial grid sized by the largest triangle edge: the nearest face, not the outermost one on the ray from the centre.
    particle -> particle: as mesh_contact.

        \mathbf a_n=k\,d_i\,\mathbf n,\qquad \mathbf v_t=(\mathbf v_i-\mathbf v_{\mathrm{face}})-\big((\mathbf v_i-\mathbf v_{\mathrm{face}})\cdot\mathbf n\big)\mathbf n,\qquad \mathbf a_t=-\mu\lVert\mathbf a_n\rVert\frac{\mathbf v_t}{\lVert\mathbf v_t\rVert+\varepsilon_v}
        a_n            the penalty acceleration, normal to the face
        k              the penalty stiffness, `k_frac` of the largest the substep carries stably
        d_i            the depth of particle i behind its nearest face
        n              the face normal
        v_t            the relative tangential velocity of particle and face
        v_i, v_{face}  the particle's and the face's velocities
        a_t            the friction acceleration
        \mu            `mu`, the Coulomb friction coefficient
        \varepsilon_v  `eps_v`, the velocity that regularises the friction's direction

    Reference: Ericson, C. (2005). Real-Time Collision Detection, ch. 7; Teschner, M. et al. (2003). Optimized spatial hashing for collision detection of deformable objects. VMV 2003:47-54; Chen, Z., Qiu, X., Zhang, X. & Lian, Y. (2015). Improved coupling of finite element method with material point method based on a particle-to-surface contact algorithm. Comput. Methods Appl. Mech. Engrg. 293:1-19."""
    PARAM_UNITS = {}
    SYMBOLS = {'a_n': 'the penalty acceleration, normal to the face', 'k': 'the penalty stiffness, `k_frac` of the largest the substep carries stably', 'd_i': 'the depth of particle i behind its nearest face', 'n': 'the face normal', 'v_t': 'the relative tangential velocity of particle and face', 'v_i, v_{face}': "the particle's and the face's velocities", 'a_t': 'the friction acceleration', '\\mu': '`mu`, the Coulomb friction coefficient', '\\varepsilon_v': "`eps_v`, the velocity that regularises the friction's direction"}

    def _build_from(self, V, es, et, ef, Vv, dev, dt_):
        M = super()._build_from(V, es, et, ef, Vv, dev, dt_)
        A, B, C = M["A"], M["B"], M["C"]
        # THE CELL IS THE TYPICAL TRIANGLE AND EACH TRIANGLE GOES INTO EVERY CELL ITS BOUNDING BOX
        # TOUCHES, which is the pair of choices that makes this index work where the direction grid
        # does not. Sizing the cell by the LARGEST triangle -- the 99.9th percentile edge, as the
        # direction grid sizes its rows -- gave 4.4 um cells on a 28 um cell, an 11x7x11 grid and
        # 2,551 triangles in one of them: a coarse index is a coarse index whatever it is indexed
        # by. The median edge sizes the grid to the mesh's own resolution, and inserting by bounding
        # box rather than by the three corners is what keeps that legal for the few triangles that
        # are much larger than the median -- a corner registration would leave the middle of such a
        # triangle in no cell at all.
        edge = torch.maximum((B - A).norm(dim=1),
                             torch.maximum((C - A).norm(dim=1), (C - B).norm(dim=1)))
        h = max(float(edge.median()), 1e-6)
        P3 = torch.cat([A, B, C])
        lo = P3.min(0).values - 2.0 * h
        hi = P3.max(0).values + 2.0 * h
        tlo = torch.minimum(torch.minimum(A, B), C)
        thi = torch.maximum(torch.maximum(A, B), C)
        for _ in range(8):                       # coarsen until the insert list is affordable
            dims = ((hi - lo) / h).ceil().long().clamp_min(1)
            ilo = ((tlo - lo) / h).long().clamp_min(0)
            ihi = ((thi - lo) / h).long()
            ihi = torch.minimum(ihi, (dims - 1)[None, :])
            span = (ihi - ilo + 1).clamp_min(1)
            if int(span.prod(1).sum()) <= 20_000_000:
                break
            h *= 1.6
        nx, ny, nz = (int(dims[0]), int(dims[1]), int(dims[2]))
        nb = nx * ny * nz
        sx, sy, sz = span[:, 0], span[:, 1], span[:, 2]
        cnt3 = sx * sy * sz
        tri3 = torch.repeat_interleave(torch.arange(A.shape[0], device=dev), cnt3)
        base = torch.cumsum(cnt3, 0) - cnt3
        k = torch.arange(int(cnt3.sum()), device=dev) - base[tri3]
        kz = k % sz[tri3]
        ky = (k // sz[tri3]) % sy[tri3]
        kx = k // (sz[tri3] * sy[tri3])
        cx = ilo[tri3, 0] + kx
        cy = ilo[tri3, 1] + ky
        cz = ilo[tri3, 2] + kz
        b3 = (cx * ny + cy) * nz + cz
        order = torch.argsort(b3 * (A.shape[0] + 1) + tri3)     # deterministic, as in the parent
        b3, tri3 = b3[order], tri3[order]
        counts = torch.bincount(b3, minlength=nb)
        Kh = int(counts.max())
        first = torch.zeros(nb + 1, dtype=torch.long, device=dev)
        first[1:] = torch.cumsum(counts, 0)
        slot = torch.arange(b3.shape[0], device=dev) - first[b3]
        htable = torch.full((nb, Kh), -1, dtype=torch.long, device=dev)
        htable[b3, slot] = tri3
        # THE PREFILTER IS THE GRID ITSELF, so the inherited direction-bin prefilter must let
        # everything through: `rmax` is what `forward` compares a particle's radius against, and a
        # spatial model has no opinion about radius.
        M["rmax"] = torch.full_like(M["rmax"], float("inf"))
        M.update(h=h, lo=lo, hdims=(nx, ny, nz), htable=htable, hcounts=counts, Kh=Kh)
        if not getattr(self, "_said_hash", False):
            self._said_hash = True
            print(f"[mesh_contact/spatial_hash] {M['n_tri']:,} sub-triangles in a "
                  f"{nx}x{ny}x{nz} grid of {h:.4g} cells, up to {Kh} per cell -- "
                  f"{27 * Kh} candidates a particle", flush=True)
        return M

    def _query(self, M, x, u, r):
        dev = x.device
        n = x.shape[0]
        nx, ny, nz = M["hdims"]
        h, lo = M["h"], M["lo"]
        c = ((x - lo) / h).long()
        c[:, 0].clamp_(0, nx - 1); c[:, 1].clamp_(0, ny - 1); c[:, 2].clamp_(0, nz - 1)
        o = torch.tensor([-1, 0, 1], device=dev)
        off = torch.stack(torch.meshgrid(o, o, o, indexing="ij"), -1).reshape(-1, 3)   # [27,3]
        nb = c[:, None, :] + off[None]                                                 # [n,27,3]
        nb[..., 0].clamp_(0, nx - 1); nb[..., 1].clamp_(0, ny - 1); nb[..., 2].clamp_(0, nz - 1)
        bi = (nb[..., 0] * ny + nb[..., 1]) * nz + nb[..., 2]                          # [n,27]
        # ONLY THE PARTICLES WITH A FACE NEAR THEM. Everything else is interior (or outside the
        # band) and is never gathered, which is what keeps the [m, 27K] tensor small.
        have = M["hcounts"][bi].sum(1) > 0
        hit = torch.zeros(n, dtype=torch.bool, device=dev)
        tri = torch.zeros(n, dtype=torch.long, device=dev)
        tb = torch.zeros(n, device=dev, dtype=x.dtype)
        w = torch.zeros(n, 3, device=dev, dtype=x.dtype)
        if not bool(have.any()):
            return hit, tri, tb, w
        sel = torch.nonzero(have).squeeze(1)
        _cap = float(os.environ.get("PLEXUS_CONTACT_MAX_ENTRIES", "5.0e7"))
        _need = float(sel.numel()) * 27.0 * float(M["Kh"])
        if _need > _cap:
            raise ValueError(
                f"mesh_contact[spatial_hash]: {sel.numel():,} particles are within a cell of the "
                f"surface and the grid holds up to {M['Kh']} triangles a cell, so the lookup would "
                f"allocate {_need:.3g} entries. Either the surface is far finer than the material "
                f"it touches, or the band is thick with particles. "
                f"(PLEXUS_CONTACT_MAX_ENTRIES overrides; expect the memory.)")
        cand = M["htable"][bi[sel]].reshape(sel.numel(), -1)                           # [m, 27Kh]
        ok = cand >= 0
        ci = cand.clamp_min(0)
        A = M["A"][ci]; B = M["B"][ci]; C = M["C"][ci]
        # THE SAME MOLLER-TRUMBORE AS THE CENTRE MODEL, so `t` keeps its meaning -- the radius of
        # the surface along the particle's own ray -- and everything downstream is untouched. Only
        # the set of triangles it is asked about is different.
        e1, e2 = B - A, C - A
        uu = u[sel][:, None, :]
        p = torch.cross(uu.expand_as(e2), e2, dim=2)
        det = (e1 * p).sum(2)
        inv = 1.0 / torch.where(det.abs() < 1e-20, torch.full_like(det, 1e-20), det)
        s = -A
        w1 = (s * p).sum(2) * inv
        q = torch.cross(s, e1, dim=2)
        w2 = (uu * q).sum(2) * inv
        t = (e2 * q).sum(2) * inv
        tol = 1e-6
        good = ok & (det.abs() > 1e-20) & (w1 >= -tol) & (w2 >= -tol) & (w1 + w2 <= 1 + tol) & (t > 0)
        tt = torch.where(good, t, torch.full_like(t, -1.0))
        best = tt.argmax(dim=1)
        ar = torch.arange(sel.numel(), device=dev)
        hit[sel] = good[ar, best]
        tri[sel] = ci[ar, best]
        tb[sel] = t[ar, best]
        w[sel] = torch.stack([1.0 - w1[ar, best] - w2[ar, best], w1[ar, best], w2[ar, best]], 1)
        return hit, tri, tb, w


@register_operator("mesh_contact", implementation="basal_reaction", family="boundary",
                   set="particle", kind="lateral", title="push against the basal surface, both ways", equation=r"""$$\mathbf a_n=k\,d_i\,\mathbf n,\qquad \mathbf v_t=(\mathbf v_i-\mathbf v_{\mathrm{face}})-\big((\mathbf v_i-\mathbf v_{\mathrm{face}})\cdot\mathbf n\big)\mathbf n,\qquad \mathbf a_t=-\mu\lVert\mathbf a_n\rVert\frac{\mathbf v_t}{\lVert\mathbf v_t\rVert+\varepsilon_v}$$""")
class MeshContactBasalReaction(MeshContact):
    r"""mesh_contact (the centre model) against a live apico-basal tissue's basal surface (pos - sep, plus `shift`), with its reaction returned to the tissue, explicitly through the relaxation.
    particle -> particle and the tissue's vertices: as mesh_contact, the surface being `pos - sep` of the vertex set.

        \mathbf a_n=k\,d_i\,\mathbf n,\qquad \mathbf v_t=(\mathbf v_i-\mathbf v_{\mathrm{face}})-\big((\mathbf v_i-\mathbf v_{\mathrm{face}})\cdot\mathbf n\big)\mathbf n,\qquad \mathbf a_t=-\mu\lVert\mathbf a_n\rVert\frac{\mathbf v_t}{\lVert\mathbf v_t\rVert+\varepsilon_v}
        a_n            the penalty acceleration, normal to the face
        k              the penalty stiffness, `k_frac` of the largest the substep carries stably
        d_i            the depth of particle i behind its nearest face
        n              the face normal
        v_t            the relative tangential velocity of particle and face
        v_i, v_{face}  the particle's and the face's velocities
        a_t            the friction acceleration
        \mu            `mu`, the Coulomb friction coefficient
        \varepsilon_v  `eps_v`, the velocity that regularises the friction's direction

    Reference: Chen, Z., Qiu, X., Zhang, X. & Lian, Y. (2015). Improved coupling of finite element method with material point method based on a particle-to-surface contact algorithm. Comput. Methods Appl. Mech. Engrg. 293:1-19."""
    PARAM_UNITS = {'shift': 'length'}
    SYMBOLS = {'a_n': 'the penalty acceleration, normal to the face', 'k': 'the penalty stiffness, `k_frac` of the largest the substep carries stably', 'd_i': 'the depth of particle i behind its nearest face', 'n': 'the face normal', 'v_t': 'the relative tangential velocity of particle and face', 'v_i, v_{face}': "the particle's and the face's velocities", 'a_t': 'the friction acceleration', '\\mu': '`mu`, the Coulomb friction coefficient', '\\varepsilon_v': "`eps_v`, the velocity that regularises the friction's direction"}
    PARAM_ROLES = {'sep_block': 'vertex_block_of_the_apico_basal_separation', 'shift': 'constant_translation_of_the_surface'}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.shift = [float(v) for v in params.get("shift", [0.0, 0.0, 0.0])]
        self.sep_block = str(params.get("sep_block", "sep"))
        if self.scale != 1.0:
            raise ValueError("mesh_contact[basal_reaction]: `scale` must be 1 -- a rescaled surface "
                             "would return a reaction in the wrong units; translate with `shift`")
        self._H = None
        self._mask = None

    def _mesh_live(self, H, dev, dt_):
        lvl = H.level(self.surface)
        m = getattr(lvl, "_mesh", None)
        if m is None or not int(m.get("Nv", 0)):
            return None
        nv = int(m["Nv"])
        P = (lvl.get("pos")[:nv].to(dt_) - lvl.get(self.sep_block)[:nv].to(dt_)
             + torch.tensor(self.shift, device=dev, dtype=dt_))
        if self.centre_track:
            self.centre = [a + b for a, b in zip(P.mean(0).detach().to("cpu").tolist(),
                                                 self.centre_offset)]
        V = P - torch.tensor(self.centre, device=dev, dtype=dt_)
        return V, m["E_srce"].to(dev), m["E_trgt"].to(dev), m["E_face"].to(dev), torch.zeros_like(V)

    def forward(self, H, mask=None):
        self._H, self._mask = H, mask
        tm = getattr(H.level(self.surface), "_mesh", None)
        f = getattr(H, "frame", None)
        if tm is not None and int(tm.get("Nv", 0)) and f is not None:
            basal_load(tm, int(f), int(tm["Nv"]), H.level(self.at).get("pos").device)
        return super().forward(H, mask)

    def _record(self, f, n_con, dmax, slip, a_max_seen, fpart, fvert, detail, M, dev, dt_):
        if detail is not None and self._H is not None:
            H = self._H
            idx, tri, depth, a_par, mass = detail
            lvl = H.level(self.at)
            on = lvl.occ[idx] > 0
            if self._mask is not None:
                on = on & (self._mask[idx] > 0)
            idx, tri, a_par = idx[on], tri[on], a_par[on]
            if idx.numel():
                c = torch.tensor(self.centre, device=dev, dtype=torch.float64)
                d = lvl.get("pos")[idx].to(torch.float64) - c
                u = d / d.norm(dim=1, keepdim=True).clamp_min(1e-30)
                A, B, Cc = (M[k][tri].to(torch.float64) for k in ("A", "B", "C"))
                e1, e2 = B - A, Cc - A
                p = torch.cross(u, e2, dim=1)
                inv = 1.0 / (e1 * p).sum(1)
                q = torch.cross(-A, e1, dim=1)
                w1 = ((-A) * p).sum(1) * inv
                w2 = (u * q).sum(1) * inv
                w = torch.stack([1.0 - w1 - w2, w1, w2], 1)
                F_p = (mass * a_par).to(torch.float64)
                fv64 = barycentric_reaction(M, tri, w, F_p, M["nv"])
                _sd = getattr(H, "sub_dt", None)
                count = max(1, int(round(float(H.dt) / float(_sd)))) if _sd else 1
                tm = H.level(self.surface)._mesh
                buf = basal_load(tm, f, M["nv"], dev)
                buf += fv64 / count
                record_interface(tm, F_p, fv64)
        return super()._record(f, n_con, dmax, slip, a_max_seen, fpart, fvert, detail, M, dev, dt_)



AFFINE_TRACE: list = []


@register_operator("surface_drive", model="affine", family="mechanics", set="vertex",
                   kind="structural", title="shear and rotate the surface",
                   equation=r"""$$x_a(t)=c_0+e^{u(t)\,\mathsf L}\,(X_a-c_0),\quad \mathsf L=\begin{pmatrix}\varepsilon&-\theta\\ \theta&-\varepsilon\end{pmatrix},\quad u=\mathrm{clamp}\!\left(\frac{t-t_{\mathrm{hold}}}{t_{\mathrm{over}}},0,1\right)$$""")
class SurfaceDriveAffine(Structural):
    r"""an imposed homogeneous flow in the sheet's plane, pure shear plus rotation, applied kinematically to a seeded surface: the tissue flow of a planar-polarity experiment, prescribed.
    vertex -> vertex: reads the seeded positions, writes pos (and vel, if declared) in place; `incremental` applies it as a background velocity, `return_over` undoes it.

        x_a(t)=c_0+e^{u(t)\,\mathsf L}\,(X_a-c_0),\quad \mathsf L=\begin{pmatrix}\varepsilon&-\theta\\ \theta&-\varepsilon\end{pmatrix},\quad u=\mathrm{clamp}\!\left(\frac{t-t_{\mathrm{hold}}}{t_{\mathrm{over}}},0,1\right)
        x_a(t)         a vertex's position at frame t
        c_0            the seeded centroid
        u(t)           the travel's progress, 0 to 1 between `hold` and `hold + over`
        L              the flow generator in the plane
        X_a            the seeded position
        \varepsilon    `strain`, the total pure shear (stretch e^eps along `extend_axis`)
        \theta         `rotate_deg`, the total rotation

    Reference: Aigouy, B. et al. (2010). Cell flow reorients the axis of planar polarity in the wing epithelium of Drosophila. Cell 142:773-786 (the flow decomposition); Aw, W. Y. et al. (2016). Curr. Biol. 26:2090-2100."""
    PARAM_UNITS = {'extend_deg': 'angle', 'hold': 'frames', 'over': 'frames', 'plane_axis': 'count', 'return_over': 'frames', 'rotate_deg': 'angle', 'strain': 'dimensionless', 'extend_axis': 'count'}
    SYMBOLS = {'x_a(t)': "a vertex's position at frame t", 'c_0': 'the seeded centroid', 'u(t)': "the travel's progress, 0 to 1 between `hold` and `hold + over`", 'L': 'the flow generator in the plane', 'X_a': 'the seeded position', '\\varepsilon': '`strain`, the total pure shear (stretch e^eps along `extend_axis`)', '\\theta': '`rotate_deg`, the total rotation'}

    EMIT = None                        # moves positions in place; no integrable delta
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["prescribed_kinematics", "tissue_flow", "pure_shear", "moving_boundary"]
    PARAM_ROLES = {"strain": "total_pure_shear_log_stretch", "rotate_deg": "total_local_rotation_degrees",
                   "extend_deg": "angle_of_the_extension_axis_in_the_plane_degrees",
                   "return_over": "frames_to_undo_the_flow_after_its_travel",
                   "extend_axis": "extension_axis_in_the_plane", "plane_axis": "sheet_normal_axis",
                   "hold": "frames_held_before_the_flow", "over": "frames_of_flow", 'incremental': 'impose_as_a_background_velocity'}
    REFERENCE = ("Aigouy, B. et al. (2010). Cell flow reorients the axis of planar polarity in the wing "
                 "epithelium of Drosophila. Cell 142:773-786.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.strain = float(params.get("strain", 0.0))
        self.rotate = math.radians(float(params.get("rotate_deg", 0.0)))
        self.plane_axis = int(params.get("plane_axis", 2))
        self.extend_axis = int(params.get("extend_axis", 0))
        self.extend_deg = math.radians(float(params.get("extend_deg", 0.0)))
        self.hold = int(params.get("hold", 0))
        self.over = int(params.get("over", 0)) or None
        self.incremental = bool(params.get("incremental", False))
        self.return_over = int(params.get("return_over", 0))
        self._u_prev = 0.0
        self._p0 = None
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        if m is None or not int(m.get("Nv", 0)):
            return {}                                   # the seed has not run yet
        Nv = int(m["Nv"])
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        ax = [i for i in range(pos.shape[1]) if i != self.plane_axis]
        if self.extend_axis not in ax:
            raise ValueError(f"surface_drive[affine]: extend_axis {self.extend_axis} is the sheet normal")
        a0, a1 = self.extend_axis, [i for i in ax if i != self.extend_axis][0]
        if self.incremental:
            return self._step(H, lvl, pos, Nv, a0, a1, dev, dt_)
        if self._p0 is None:
            self._p0 = pos[:Nv].clone()
            self._c0 = self._p0.mean(0, keepdim=True)
            self.over = self.over or max(1, int(getattr(H, "n_frames", 0) or 1) - self.hold)
        if pos.shape[0] < Nv or self._p0.shape[0] != Nv:
            raise RuntimeError("surface_drive[affine]: the vertex count changed; an affine drive of the "
                               "seeded shape needs a fixed surface")
        f = int(getattr(H, "frame", 0) or 0)
        u = self._u(f)
        c2, s2 = math.cos(2 * self.extend_deg), math.sin(2 * self.extend_deg)
        L = torch.tensor([[self.strain * c2, self.strain * s2 - self.rotate],
                          [self.strain * s2 + self.rotate, -self.strain * c2]], dtype=torch.float64)
        F = torch.linalg.matrix_exp(u * L).to(dev, dt_)
        d = self._p0 - self._c0
        new = pos[:Nv].clone()
        new[:, a0] = self._c0[0, a0] + F[0, 0] * d[:, a0] + F[0, 1] * d[:, a1]
        new[:, a1] = self._c0[0, a1] + F[1, 0] * d[:, a0] + F[1, 1] * d[:, a1]
        if "vel" in lvl.state_schema:
            lvl.get("vel")[:Nv] = new - pos[:Nv]        # per frame, this Level's time unit
        pos[:Nv] = new
        AFFINE_TRACE.append((f, u, float(F[0, 0]), float(F[1, 0])))
        self._say(a0)
        return {}

    def _u(self, f):
        """The fraction of the map applied at frame f: 0 -> 1 over `over`, then back to 0 over `return_over`."""
        u = min(1.0, max(0.0, (f - self.hold) / float(self.over)))
        if self.return_over > 0 and f >= self.hold + self.over:
            u = max(0.0, 1.0 - (f - self.hold - self.over) / float(self.return_over))
        return u

    def _L(self):
        c2, s2 = math.cos(2 * self.extend_deg), math.sin(2 * self.extend_deg)
        return torch.tensor([[self.strain * c2, self.strain * s2 - self.rotate],
                             [self.strain * s2 + self.rotate, -self.strain * c2]], dtype=torch.float64)

    def _step(self, H, lvl, pos, Nv, a0, a1, dev, dt_):
        """The incremental form: one frame's share of the flow on the CURRENT positions."""
        if self.over is None:
            self.over = max(1, int(getattr(H, "n_frames", 0) or 1) - self.hold)
        f = int(getattr(H, "frame", 0) or 0)
        if self.return_over > 0:                           # transient: step by the change of u
            u = self._u(f)
            du, self._u_prev = u - self._u_prev, u
            if du == 0.0:
                return {}
            F = torch.linalg.matrix_exp(du * self._L()).to(dev, dt_)
        elif not (self.hold <= f < self.hold + self.over):
            return {}
        else:
            F = torch.linalg.matrix_exp(self._L() / float(self.over)).to(dev, dt_)
        p = pos[:Nv]
        c = p.mean(0, keepdim=True)
        d = p - c
        new = p.clone()
        new[:, a0] = c[0, a0] + F[0, 0] * d[:, a0] + F[0, 1] * d[:, a1]
        new[:, a1] = c[0, a1] + F[1, 0] * d[:, a0] + F[1, 1] * d[:, a1]
        pos[:Nv] = new
        AFFINE_TRACE.append((f, 1.0 / self.over, float(F[0, 0]), float(F[1, 0])))
        self._say(a0)
        return {}

    def _say(self, a0):
        if not self._said:
            print(f"[surface_drive:affine] total shear {self.strain} (stretch x{math.exp(self.strain):.3f} "
                  f"along {math.degrees(self.extend_deg):.1f} deg from axis {a0}), rotation {math.degrees(self.rotate):.1f} deg, frames "
                  f"{self.hold}-{self.hold + self.over}" + (", incremental" if self.incremental else ""), flush=True)
            self._said = True
