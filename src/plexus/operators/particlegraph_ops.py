"""exp01 demos (2026-10-30, Cedric: "add a simple step-movie to exp01 based on ParticleGraph/config/..."): the four ParticleGraph models Plexus had no operator for.

    diffusion, FitzHugh-Nagumo, rock-paper-scissors  ->  `react_diffuse[model: heat | fitzhugh_nagumo | rock_paper_scissors]`, one field, a coefficient per region,
                                                          its start from `seed_field[model: image | uniform | simplex]`
    springs_matrix_N100                              ->  `spring_matrix`, particles tied in pairs by a random symmetric matrix of spring constants

ParticleGraph runs the first three on a 100 x 100 Delaunay mesh whose cotangent Laplacian is, on that regular mesh, the unit-weight five-point stencil; the Plexus field is
that grid, and `L u = u_E + u_W + u_N + u_S - 4 u` is the same operator (no 1/h^2: the coefficients c of the configs are per node and per unit time, as there).
Their per-node coefficient comes from a label image (256 x 256, labels 0..4) sampled under each node centre, `img[int(y 255), int(x 255)]` after a vertical flip.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as Fnn

from plexus.models.base import FieldUpdate, Lateral, Seed
from plexus.models.registry import register_operator
from plexus.paths import graphs_data_path


def _image(path):
    """the TIFF at `path` (relative to graphs_data), flipped so that the image's top row is the domain's top (y = 1)."""
    import tifffile
    p = path if os.path.isabs(path) else graphs_data_path(path)
    return np.flipud(tifffile.imread(p))


def _under_nodes(img, nx, ny):
    """the image value under each node centre of an nx x ny grid over [0, 1]^2: img[int(y (H-1)), int(x (W-1))], as ParticleGraph samples it. [nx, ny]."""
    h, w = img.shape[:2]
    ix = (((np.arange(nx) + 0.5) / nx) * (w - 1)).astype(int)
    iy = (((np.arange(ny) + 0.5) / ny) * (h - 1)).astype(int)
    return img[iy[None, :], ix[:, None]]


def _diffusion(g, c, periodic):
    """sum over the four neighbours j of k_ij (g_j - g_i) on a [C, nx, ny] grid, with k_ij the harmonic mean 2 c_i c_j / (c_i + c_j) of the two nodes' coefficients c [1, nx, ny]:
    symmetric, so what leaves one node enters the next and the total is kept, and zero when either is zero, so a coefficient-0 region is a wall. A periodic grid wraps; any other has no flux."""
    mode = "circular" if periodic else "replicate"
    pg = Fnn.pad(g.unsqueeze(0), (1, 1, 1, 1), mode=mode)[0]
    pc = Fnn.pad(c.unsqueeze(0), (1, 1, 1, 1), mode=mode)[0]
    out = torch.zeros_like(g)
    for sl in ((slice(None), slice(2, None), slice(1, -1)), (slice(None), slice(None, -2), slice(1, -1)),
               (slice(None), slice(1, -1), slice(2, None)), (slice(None), slice(1, -1), slice(None, -2))):
        cn = pc[sl]
        out = out + (2.0 * c * cn / (c + cn).clamp_min(1e-12)) * (pg[sl] - g)
    return out


@register_operator("seed_field", family="seed", set="field", kind="seed", title="set the initial field from an image",
                   equation=r"""$$c_k(\mathbf x_n)=o+s\,I\big(\lfloor y_n (H-1)\rfloor,\ \lfloor x_n (W-1)\rfloor\big)$$""")
class SeedFieldImage(Seed):
    r"""writes one channel of a grid field from an image once: the value under each node centre, times a scale.
    field -> field: reads the TIFF named by `image`, writes channel `channel` of the field `at:`; nothing else is touched.

        c_k(\mathbf x_n)=o+s\,I\big(\lfloor y_n (H-1)\rfloor,\ \lfloor x_n (W-1)\rfloor\big)
        c_k            channel k of the field, `channel`
        \mathbf x_n    the centre of node n, on [0, 1]^2
        I              the image, flipped so its top row is y = 1
        H, W           its height and width in pixels
        s              `scale`, the field unit per image unit (1/255 for an 8-bit picture read as 0..1)
        o              `offset`, the field value where the image is 0

    Reference: Plexus (this work); the sampling is ParticleGraph's `init_mesh` (Allier, C. et al.)."""
    PARAM_UNITS = {'scale': 'dimensionless', 'offset': 'dimensionless', 'channel': 'count'}
    SYMBOLS = {'c_k': 'channel k of the field, `channel`', '\\mathbf x_n': 'the centre of node n, on [0, 1]^2', 'I': 'the image, flipped so its top row is y = 1',
               'H, W': 'its height and width in pixels', 's': '`scale`, the field unit per image unit (1/255 for an 8-bit picture read as 0..1)', 'o': '`offset`, the field value where the image is 0'}
    EMIT = None
    SUPPORTED_DIMS = [2]
    REQUIRES_PARAMS = ["image"]
    MECHANISM_TAGS = ["initial_condition", "image"]
    PARAM_ROLES = {"image": "tiff_under_the_nodes", "scale": "field_unit_per_image_unit", "offset": "field_value_at_image_zero", "channel": "channel_written"}
    REFERENCE = "Plexus (this work); ParticleGraph init_mesh."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params["_at"]
        self.image = str(params["image"])
        self.scale = float(params.get("scale", 1.0 / 255.0))
        self.offset = float(params.get("offset", 0.0))
        self.channel = int(params.get("channel", 0))

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        g = fld.grid.clone()
        v = _under_nodes(_image(self.image), g.shape[-2], g.shape[-1]).astype(np.float32)
        g[self.channel] = self.offset + torch.as_tensor(v, device=g.device) * self.scale
        fld.grid = g
        print(f"[seed_field] {self.field_name}[{self.channel}] from {self.image}: {tuple(g.shape[-2:])} nodes, range {float(g[self.channel].min()):.4g}..{float(g[self.channel].max()):.4g}", flush=True)
        return {}


@register_operator("seed_field", family="seed", set="field", kind="seed", model="uniform", title="set the initial field to small random values",
                   equation=r"""$$c_k(\mathbf x_n)=a\,\xi_{kn},\qquad \xi_{kn}\sim U[0,1)$$""")
class SeedFieldUniform(SeedFieldImage):
    r"""writes every channel of a grid field once with independent uniform random values from 0 to an amplitude: ParticleGraph's start of FitzHugh-Nagumo, `torch.rand * 0.1`.
    field -> field: writes all channels of the field `at:`.

        c_k(\mathbf x_n)=a\,\xi_{kn},\qquad \xi_{kn}\sim U[0,1)
        c_k            channel k of the field
        \mathbf x_n    node n
        a              `amplitude`, the largest value a node can start with
        \xi_{kn}       an independent uniform draw per channel and node (seeded by `seed`)

    Reference: Plexus (this work); ParticleGraph init_mesh (RD_FitzHugh_Nagumo_Mesh)."""
    PARAM_UNITS = {'amplitude': 'dimensionless', 'seed': 'count'}
    SYMBOLS = {'c_k': 'channel k of the field', '\\mathbf x_n': 'node n', 'a': '`amplitude`, the largest value a node can start with',
               '\\xi_{kn}': 'an independent uniform draw per channel and node (seeded by `seed`)'}
    REQUIRES_PARAMS = []
    PARAM_ROLES = {"amplitude": "largest_start_value", "seed": "rng_seed"}

    def __init__(self, params, device="cpu"):
        Seed.__init__(self, params, device)
        self.field_name = params["_at"]
        self.amplitude = float(params.get("amplitude", 0.1))
        self.seed = int(params.get("seed", 0))

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        rng = torch.Generator(device=fld.grid.device).manual_seed(self.seed)
        fld.grid = self.amplitude * torch.rand(fld.grid.shape, generator=rng, device=fld.grid.device)
        return {}


@register_operator("seed_field", family="seed", set="field", kind="seed", model="von_karman", title="set the initial field to a random tissue index",
                   equation=r"""$$\hat n(\mathbf k)=\hat\xi(\mathbf k)\,(1+k^2L_0^2)^{-m/2}\,e^{-k^2 l_0^2/2},\qquad n=\sigma\,\frac{\mathcal F^{-1}[\hat n]}{\mathrm{std}(\mathcal F^{-1}[\hat n])}$$""")
class SeedFieldVonKarman(SeedFieldImage):
    r"""writes the first channel of a grid field once with a zero-mean random field of a von Karman power spectrum: the refractive-index texture of soft tissue (Schmitt and Kumar 1996), the static landscape a beam crosses.
    field -> field: writes channel 0 of the field `at:`; every other channel is left alone.

        \hat n(k) = \hat\xi(k) (1 + k^2 L_0^2)^(-m/2) exp(-k^2 l_0^2 / 2),     n = rms * IFFT[\hat n] / std(IFFT[\hat n])
        \hat\xi(k)   the Fourier transform of white Gaussian noise (seeded by `seed`)
        k             the angular wavenumber, 2 pi times the spatial frequency in 1/um
        L_0           `outer_scale`, the scale below which the spectrum is a power law, in um
        l_0           `inner_scale`, the scale below which it is cut off, in um (without it the field reaches down to the grid)
        m             `exponent`, the amplitude's power-law exponent; 1.9 for a 3D field whose 2D projection measures 1.4 (a projection flattens the spectrum by half a power)
        rms           `rms`, the root-mean-square index deviation of the result

    Reference: Schmitt, J. M. and Kumar, G. (1996) Optics Letters 21:1310; Fluo-BPM report.pdf section 6."""
    PARAM_UNITS = {'rms': 'dimensionless', 'outer_scale': 'length', 'inner_scale': 'length', 'exponent': 'dimensionless', 'seed': 'count'}
    SYMBOLS = {'\\hat n': 'the field in Fourier space', 'k': 'angular wavenumber', 'L_0': '`outer_scale`', 'l_0': '`inner_scale`', 'm': '`exponent`',
               'rms': '`rms`', '\\hat\\xi': 'Fourier transform of white Gaussian noise'}
    SUPPORTED_DIMS = [2, 3]                       # the spectrum is built on the grid's own axes, whatever their number
    REQUIRES_PARAMS = []
    PARAM_ROLES = {"rms": "index_rms", "outer_scale": "outer_scale", "inner_scale": "inner_scale", "exponent": "spectrum_exponent", "seed": "rng_seed"}

    def __init__(self, params, device="cpu"):
        Seed.__init__(self, params, device)
        self.field_name = params["_at"]
        self.rms = float(params.get("rms", 0.01))
        self.L0 = float(params.get("outer_scale", 6.0))
        self.l0 = float(params.get("inner_scale", 1.0))
        self.m = float(params.get("exponent", 1.9))
        self.seed = int(params.get("seed", 0))

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        shape = tuple(fld.grid.shape[1:])
        dev = fld.grid.device
        rng = torch.Generator(device=dev).manual_seed(self.seed)
        xi = torch.randn(shape, generator=rng, device=dev)
        dx = 1.0 / float(fld.R)
        k2 = sum((2.0 * np.pi * torch.fft.fftfreq(n, d=dx, device=dev).reshape([-1 if a == i else 1 for a in range(len(shape))])) ** 2 for i, n in enumerate(shape))
        amp = (1.0 + k2 * self.L0 ** 2) ** (-self.m / 2.0) * torch.exp(-k2 * self.l0 ** 2 / 2.0)
        n = torch.fft.ifftn(torch.fft.fftn(xi) * amp).real
        n = n - n.mean()
        g = fld.grid.clone()
        g[0] = self.rms * n / n.std().clamp_min(1e-12)
        fld.grid = g
        return {}


@register_operator("seed_field", family="seed", set="field", kind="seed", model="volume", title="set the initial field to a recorded 3D volume",
                   equation=r"""$$c_0(i,j,k)=o+s\,V[k,j,i]$$""")
class SeedFieldVolume(SeedFieldImage):
    r"""writes the first channel of a 3D grid field once from a TIFF stack: the volume as it was recorded or simulated, one voxel per voxel.
    field -> field: reads the stack named by `image` (z, y, x order, as ImageJ and Fluo-BPM write it), writes channel `channel` of the field `at:`.

        c_0(i, j, k) = o + s V[k, j, i]
        V            the stack, indexed [plane k, row j, column i]
        i, j, k      the field's voxel along x, y and z (the axis order of every 3D grid field)
        s            `scale`, the field unit per stack unit
        o            `offset`, the field value where the stack is 0

    The stack must have the field's own shape (nx, ny, nz); nothing is resampled.

    Reference: Plexus (this work); Fluo-BPM data/*.tif."""
    PARAM_UNITS = {'scale': 'dimensionless', 'offset': 'dimensionless', 'channel': 'count'}
    SYMBOLS = {'c_0': 'channel `channel` of the field', 'V': 'the stack, [plane, row, column]', 'i, j, k': 'voxel along x, y, z', 's': '`scale`', 'o': '`offset`'}
    SUPPORTED_DIMS = [3]
    PARAM_ROLES = {"image": "tiff_stack", "scale": "field_unit_per_stack_unit", "offset": "field_value_at_stack_zero", "channel": "channel_written"}

    def __init__(self, params, device="cpu"):
        super().__init__({**params, "scale": params.get("scale", 1.0)}, device)

    def forward(self, H, mask=None):
        import tifffile
        fld = H.fields[self.field_name]
        p = self.image if os.path.isabs(self.image) else graphs_data_path(self.image)
        v = np.asarray(tifffile.imread(p), dtype=np.float32).transpose(2, 1, 0)       # [z, y, x] -> [x, y, z]
        if tuple(v.shape) != tuple(fld.grid.shape[1:]):
            raise ValueError(f"seed_field[volume]: {p} holds {tuple(v.shape)} voxels (x, y, z) but field {self.field_name} is {tuple(fld.grid.shape[1:])}")
        g = fld.grid.clone()
        g[self.channel] = self.offset + torch.as_tensor(v, device=g.device) * self.scale
        fld.grid = g
        print(f"[seed_field] {self.field_name}[{self.channel}] from {p}: {tuple(v.shape)} voxels, range {float(g[self.channel].min()):.4g}..{float(g[self.channel].max()):.4g}", flush=True)
        return {}


@register_operator("seed_field", family="seed", set="field", kind="seed", model="simplex", title="set the initial field to random fractions",
                   equation=r"""$$c_k(\mathbf x_n)=\frac{\xi_{kn}}{\sum_m \xi_{mn}},\qquad \xi_{kn}\sim U[0,1)$$""")
class SeedFieldSimplex(SeedFieldUniform):
    r"""writes the channels of a grid field once as random fractions that add up to 1 at every node: ParticleGraph's start of rock-paper-scissors, three random numbers over their sum.
    field -> field: writes all channels of the field `at:`.

        c_k(\mathbf x_n)=\frac{\xi_{kn}}{\sum_m \xi_{mn}},\qquad \xi_{kn}\sim U[0,1)
        c_k            channel k of the field
        \mathbf x_n    node n
        \xi_{kn}       an independent uniform draw per channel and node (seeded by `seed`)
        m              the channel index of the sum

    Reference: Plexus (this work); ParticleGraph init_mesh (RD_Mesh)."""
    PARAM_UNITS = {'seed': 'count'}
    SYMBOLS = {'c_k': 'channel k of the field', '\\mathbf x_n': 'node n', '\\xi_{kn}': 'an independent uniform draw per channel and node (seeded by `seed`)',
               'm': 'the channel index of the sum'}
    PARAM_ROLES = {"seed": "rng_seed"}

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        rng = torch.Generator(device=fld.grid.device).manual_seed(self.seed)
        u = torch.rand(fld.grid.shape, generator=rng, device=fld.grid.device)
        fld.grid = u / u.sum(0, keepdim=True)
        return {}


@register_operator("seed_field", family="seed", set="field", kind="seed", model="mask", title="mark where an image is zero",
                   equation=r"""$$c_k(\mathbf x_n)=\begin{cases}1 & I(\mathbf x_n)\le\theta\\ 0 & \text{otherwise}\end{cases}$$""")
class SeedFieldMask(SeedFieldImage):
    r"""writes one channel of a grid field once as a mask: 1 where the image under the node is at most a threshold, 0 elsewhere -- the walls of a picture whose walls are 0.
    field -> field: reads the TIFF named by `image`, writes channel `channel` of the field `at:`.

        c_k(\mathbf x_n)=\begin{cases}1 & I(\mathbf x_n)\le\theta\\ 0 & \text{otherwise}\end{cases}
        c_k            channel k of the field, `channel`
        \mathbf x_n    the centre of node n, on [0, 1]^2
        I              the image under the node, sampled as `seed_field` samples it
        \theta         `at_most`, the largest image value that counts as marked

    Reference: Plexus (this work)."""
    PARAM_UNITS = {'at_most': 'dimensionless', 'channel': 'count'}
    SYMBOLS = {'c_k': 'channel k of the field, `channel`', '\\mathbf x_n': 'the centre of node n, on [0, 1]^2', 'I': "the image under the node, sampled as `seed_field` samples it",
               '\\theta': '`at_most`, the largest image value that counts as marked'}
    PARAM_ROLES = {"image": "tiff_under_the_nodes", "at_most": "largest_marked_value", "channel": "channel_written"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at_most = float(params.get("at_most", 0.0))

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        g = fld.grid.clone()
        v = _under_nodes(_image(self.image), g.shape[-2], g.shape[-1]).astype(np.float32)
        g[self.channel] = torch.as_tensor(v <= self.at_most, device=g.device).to(g.dtype)
        fld.grid = g
        print(f"[seed_field] {self.field_name}[{self.channel}] marks {int((v <= self.at_most).sum())} of {v.size} nodes where {self.image} <= {self.at_most:g}", flush=True)
        return {}


@register_operator("react_diffuse", family="fields", set="field", kind="field", title="diffuse a field with a coefficient per region",
                   equation=r"""$$\frac{\partial u_i}{\partial t}=\beta\sum_{j\in N(i)}k_{ij}\,(u_j-u_i),\qquad k_{ij}=\frac{2\,c_i c_j}{c_i+c_j}$$""")
class ReactDiffuse(FieldUpdate):
    r"""diffusion with its own coefficient in each region of a grid: every channel spreads between neighbouring nodes at the harmonic mean of their two coefficients, read from a label image, one number per label; a region of coefficient 0 is a wall and the total is kept.
    field -> field: acts on the field named by `at:`, writing its grid in place, `substeps` explicit Euler steps per tick; a periodic field wraps, any other has no flux.

        \frac{\partial u_i}{\partial t}=\beta\sum_{j\in N(i)}k_{ij}\,(u_j-u_i),\qquad k_{ij}=\frac{2\,c_i c_j}{c_i+c_j}
        u_i            a channel of the field at node i
        c_i            `coefficients`[label under node i], from the image `type_map`: the coefficient of the region the node is in
        k_{ij}         the exchange between neighbours i and j, the harmonic mean of their coefficients (0 if either is 0)
        \beta          `beta`, the rate scale shared by all regions, per unit time
        N(i)           the four grid neighbours of node i (unit weights, no division by the squared node spacing)

    Reference: Fick, A. (1855). Ann. Phys. 170:59-86; the per-node coefficient and the unit-weight Laplacian are ParticleGraph's (`PDE_Laplacian`, `DiffMesh`)."""
    PARAM_UNITS = {'beta': 'rate', 'substeps': 'count'}
    SYMBOLS = {'u_i': 'a channel of the field at node i', 'c_i': '`coefficients`[label under node i], from the image `type_map`: the coefficient of the region the node is in', 'k_{ij}': 'the exchange between neighbours i and j, the harmonic mean of their coefficients (0 if either is 0)', 'N(i)': 'the four grid neighbours of node i (unit weights, no division by the squared node spacing)',
               '\\beta': '`beta`, the rate scale shared by all regions, per unit time',
               }
    EMIT = None
    SUPPORTED_DIMS = [2]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["diffusion", "heterogeneous_medium", "laplacian"]
    PARAM_ROLES = {"coefficients": "diffusion_coefficient_per_label", "type_map": "label_image_under_the_nodes",
                   "coefficient_image": "image_whose_values_are_the_coefficients", "beta": "rate_scale",
                   "substeps": "euler_steps_per_tick", "noise": "noise_on_the_first_channel_per_step"}
    REFERENCE = "Fick, A. (1855). Ann. Phys. 170:59-86; ParticleGraph (DiffMesh)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params["_at"]
        self.coefficients = [float(v) for v in params["coefficients"]] if params.get("coefficients") is not None else None
        self.coefficient_image = params.get("coefficient_image")        # an image whose values ARE the coefficients, instead of labels
        self.type_map = params.get("type_map")
        self.beta = float(params.get("beta", 1.0))
        self.substeps = int(params.get("substeps", 1))
        self.noise = float(params.get("noise", 0.0))
        self._c = None

    def _coeff(self, g):
        if self._c is None:
            nx, ny = g.shape[-2:]
            if self.coefficient_image:
                c = _under_nodes(_image(str(self.coefficient_image)), nx, ny).astype(np.float32)
                print(f"[react_diffuse] {self.field_name}: {nx}x{ny} nodes, coefficient from {self.coefficient_image}: "
                      f"values {sorted(set(np.round(c.ravel(), 3).tolist()))[:8]}, {int((c <= 0).sum())} nodes of coefficient 0", flush=True)
            else:
                lab = (np.clip(_under_nodes(_image(str(self.type_map)), nx, ny).astype(int), 0, len(self.coefficients) - 1)
                       if self.type_map else np.zeros((nx, ny), int))
                c = np.asarray(self.coefficients, np.float32)[lab]
                print(f"[react_diffuse] {self.field_name}: {nx}x{ny} nodes, coefficient per label {self.coefficients}, "
                      f"nodes per label {np.bincount(lab.ravel(), minlength=len(self.coefficients)).tolist()}", flush=True)
            self._c = torch.as_tensor(c, device=g.device)[None]
        return self._c

    def rhs(self, g, c, periodic, gen):
        return self.beta * _diffusion(g, c, periodic)

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        g = fld.grid
        c = self._coeff(g)
        h = float(getattr(H.config, "dt", 1.0)) / self.substeps
        gen = getattr(H, "rng", None)
        for _ in range(self.substeps):
            g = g + h * self.rhs(g, c, bool(getattr(fld, "periodic", False)), gen)
        fld.grid = g
        return {}


@register_operator("react_diffuse", family="fields", set="field", kind="field", model="fitzhugh_nagumo", title="run FitzHugh-Nagumo on the field",
                   equation=r"""$$\dot u=\beta\,\mathcal D[u]+u-\tfrac{u^3}{3}-v+I,\qquad \dot v=\varepsilon\,(u+a-b\,v)$$""")
class ReactDiffuseFitzHughNagumo(ReactDiffuse):
    r"""an excitable medium: a fast activator u that spreads and a slow recovery v, so a stimulus sends a wave through the rest state and the regions of different diffusivity set how fast it goes.
    field -> field: acts on the two channels (u, v) of the field `at:`, in place, `substeps` explicit Euler steps per tick.

        \dot u=\beta\,\mathcal D[u]+u-\tfrac{u^3}{3}-v+I,\qquad \dot v=\varepsilon\,(u+a-b\,v)
        u, v           channel 0 and channel 1 of the field: the activator and the recovery variable
        \mathcal D[u]  the exchange of u with the four grid neighbours at the harmonic mean of the regions' coefficients `coefficients`[label of `type_map`], as in `react_diffuse`
        \beta          `beta`, the diffusion scale
        I              `I`, a constant current (0)
        a, b           `a`, `b`: where the recovery nullcline sits and how steep it is (0.7 and 0.8)
        \varepsilon    `eps`, how slow the recovery is next to the activator (0.08)

    Reference: FitzHugh, R. (1961). Biophys. J. 1:445-466; Nagumo, J., Arimoto, S. & Yoshizawa, S. (1962). Proc. IRE 50:2061-2070. ParticleGraph's own constants (`RD_FitzHugh_Nagumo`: a1 5e-3, a2 -2.8e-3, a3 5e-3, a4 0.02, a5 0.125) sit at a stable focus and give noise, not waves."""
    PARAM_UNITS = {'beta': 'rate', 'substeps': 'count', 'a': 'dimensionless', 'b': 'dimensionless', 'eps': 'rate', 'I': 'dimensionless'}
    SYMBOLS = {'u, v': 'channel 0 and channel 1 of the field: the activator and the recovery variable', '\\mathcal D[u]': "the exchange of u with the four grid neighbours at the harmonic mean of the regions' coefficients `coefficients`[label of `type_map`], as in `react_diffuse`",
               '\\beta': '`beta`, the diffusion scale', 'I': '`I`, a constant current (0)',
               'a, b': '`a`, `b`: where the recovery nullcline sits and how steep it is (0.7 and 0.8)', '\\varepsilon': '`eps`, how slow the recovery is next to the activator (0.08)'}
    PARAM_ROLES = {**ReactDiffuse.PARAM_ROLES, "a": "recovery_offset", "b": "recovery_slope", "eps": "recovery_rate", "I": "constant_current"}
    REFERENCE = "FitzHugh, R. (1961). Biophys. J. 1:445-466; Nagumo, J. et al. (1962). Proc. IRE 50:2061-2070."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.a, self.b = float(params.get("a", 0.7)), float(params.get("b", 0.8))
        self.eps, self.I = float(params.get("eps", 0.08)), float(params.get("I", 0.0))

    def rhs(self, g, c, periodic, gen):
        u, v = g[0:1], g[1:2]
        du = self.beta * _diffusion(u, c, periodic) + u - u ** 3 / 3.0 - v + self.I
        dv = self.eps * (u + self.a - self.b * v)
        return torch.cat([du, dv], 0)


@register_operator("react_diffuse", family="fields", set="field", kind="field", model="rock_paper_scissors", title="run rock-paper-scissors on the field",
                   equation=r"""$$\dot u_k=\mathcal D[u_k]+u_k\Big(1-\sum_m u_m-a\,u_{k+1}\Big)$$""")
class ReactDiffuseRockPaperScissors(ReactDiffuse):
    r"""three species that each beat the next: every channel spreads and grows into the room the three leave, and is eaten by the one after it in the cycle, so spirals form.
    field -> field: acts on the three channels of the field `at:`, in place, `substeps` explicit Euler steps per tick.

        \dot u_k=\mathcal D[u_k]+u_k\Big(1-\sum_m u_m-a\,u_{k+1}\Big)
        u_k            channel k (0, 1, 2) of the field; k+1 is taken modulo 3
        \mathcal D[u_k] the exchange of channel k with the four grid neighbours at the harmonic mean of the regions' diffusion coefficients `coefficients`[label of `type_map`], as in `react_diffuse`
        a              `a`, how hard species k+1 eats species k (0.6 in the paper's config)
        m              the species index of the sum

    Reference: May, R. M. & Leonard, W. J. (1975). SIAM J. Appl. Math. 29:243-253; Reichenbach, T., Mobilia, M. & Frey, E. (2007). Nature 448:1046-1049; ParticleGraph's `RD_RPS`."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"Three species that each beat the next: spirals form. Channel $k$ ($k=0,1,2$; $k+1$ taken modulo 3) spreads to "
        r"the four grid neighbours, grows into the room the three leave, $u_k\big(1-\sum_m u_m\big)$, and is eaten by "
        r"species $k+1$ at strength $a$ (`a`): $\dot u_k=\mathcal D[u_k]+u_k\big(1-\sum_m u_m-a\,u_{k+1}\big)$, with "
        r"$\mathcal D$ the exchange with the neighbours. Explicit Euler, `substeps` steps per tick."
    )
    PARAM_UNITS = {'a': 'dimensionless', 'substeps': 'count', 'beta': 'rate'}
    SYMBOLS = {'u_k': 'channel k (0, 1, 2) of the field; k+1 is taken modulo 3', '\\mathcal D[u_k]': "the exchange of channel k with the four grid neighbours at the harmonic mean of the regions' diffusion coefficients `coefficients`[label of `type_map`], as in `react_diffuse`", 'a': '`a`, how hard species k+1 eats species k (0.6 in the paper\'s config)',
               'm': 'the species index of the sum'}
    PARAM_ROLES = {**ReactDiffuse.PARAM_ROLES, "a": "predation_strength"}
    REFERENCE = "May, R. M. & Leonard, W. J. (1975). SIAM J. Appl. Math. 29:243-253; Reichenbach, T. et al. (2007). Nature 448:1046-1049; ParticleGraph (RD_RPS)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.pred = float(params.get("a", 0.6))

    def rhs(self, g, c, periodic, gen):
        return _diffusion(g, c, periodic) + g * (1.0 - g.sum(0, keepdim=True) - self.pred * g[[1, 2, 0]])


@register_operator("react_diffuse", family="fields", set="field", kind="field", model="wave", title="propagate a wave on the field",
                   equation=r"""$$\dot v_i=\beta\sum_{j\in N(i)}k_{ij}\,(u_j-u_i),\qquad \dot u_i=v_i$$""")
class ReactDiffuseWave(ReactDiffuse):
    r"""the wave equation on a grid whose speed differs from region to region and is 0 in a wall: channel 0 is the displacement u, channel 1 its rate v; any further channel is left alone (a mask to draw).
    field -> field: acts on channels 0 and 1 of the field `at:`, in place; each of the `substeps` steps first moves v by the force, then u by the new v (ParticleGraph's `WaveMesh` stepping).

        \dot v_i=\beta\sum_{j\in N(i)}k_{ij}\,(u_j-u_i),\qquad \dot u_i=v_i,\qquad k_{ij}=\frac{2\,c_i c_j}{c_i+c_j}
        u_i, v_i       the displacement of node i and its rate of change (channels 0 and 1)
        \beta          `beta`, the force scale: the wave speed squared is beta c in grid units
        k_{ij}         the exchange between neighbours i and j, the harmonic mean of their coefficients c (0 if either is 0: a wall reflects)
        N(i)           the four grid neighbours of node i

    Reference: ParticleGraph's `WaveMesh` and `PDE_Laplacian` (Allier, C. et al.); the finite-volume form of a wave in a medium of piecewise constant speed."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"The wave equation on a grid whose speed differs from region to region and is 0 in a wall, so a wall reflects. "
        r"Channel 0 is the displacement $u$ and channel 1 its rate $v$; each of the `substeps` steps first moves $v$ by "
        r"the force $\beta\sum_{j\in N(i)}k_{ij}(u_j-u_i)$ over the four grid neighbours $N(i)$, then moves $u$ by the new "
        r"$v$. $k_{ij}$ is the harmonic mean of the two nodes' coefficients and $\beta$ (`beta`) scales the force; any "
        r"further channel is left alone, as a mask to draw."
    )
    PARAM_UNITS = {'beta': 'rate', 'substeps': 'count'}
    SYMBOLS = {'u_i, v_i': 'the displacement of node i and its rate of change (channels 0 and 1)', '\\beta': '`beta`, the force scale: the wave speed squared is beta c in grid units',
               'k_{ij}': 'the exchange between neighbours i and j, the harmonic mean of their coefficients c (0 if either is 0: a wall reflects)', 'N(i)': 'the four grid neighbours of node i'}
    REFERENCE = "ParticleGraph WaveMesh / PDE_Laplacian."

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        g = fld.grid
        c = self._coeff(g)
        h = float(getattr(H.config, "dt", 1.0)) / self.substeps
        per = bool(getattr(fld, "periodic", False))
        u, v = g[0:1], g[1:2]
        for _ in range(self.substeps):
            v = v + h * self.beta * _diffusion(u, c, per)
            u = u + h * v
        fld.grid = torch.cat([u, v, g[2:]], 0)
        return {}


@register_operator("spring_matrix", family="interaction", set="particle", kind="lateral", title="tie particles in pairs with random springs",
                   equation=r"""$$\ddot{\mathbf x}_i=k\sum_j C_{ij}\,(\mathbf x_j-\mathbf x_i),\qquad C_{ij}=C_{ji}\sim p(c)$$""")
class SpringMatrix(Lateral):
    r"""a spring of zero rest length between every pair of particles, each with its own constant drawn once from a short list of values: the coupling of a random harmonic network.
    particle -[all pairs]-> particle: reads pos, emits an acceleration; the matrix C is drawn at the first tick, symmetric, with a zero diagonal.

        \ddot{\mathbf x}_i=k\sum_j C_{ij}\,(\mathbf x_j-\mathbf x_i),\qquad C_{ij}=C_{ji}\sim p(c)
        x_i            the position of particle i
        C_{ij}         the spring constant between i and j, per time squared: one of `values`, drawn with the weights `probabilities`
        k              `k`, a common factor on the whole matrix
        p(c)           the draw: value `values[m]` with probability `probabilities[m]`

    Reference: Plexus (this work); ParticleGraph `PDE_K` and the `connection_matrix` of graph_data_generator."""
    # EXPLAIN: what the operator does, in plain English, for a click on its equation in the maths and plexus panes
    EXPLAIN = (
        r"A random harmonic network. Every pair of particles is tied by a spring of zero rest length whose constant "
        r"$C_{ij}$ is drawn once, at the first step, from the list `values` with the weights `probabilities` (symmetric, "
        r"zero on the diagonal). Particle $i$ then accelerates as $\ddot{\mathbf x}_i=k\sum_j C_{ij}(\mathbf x_j-\mathbf "
        r"x_i)$, with $\mathbf x_i$ its position and $k$ a common factor `k` on the whole matrix."
    )
    PARAM_UNITS = {'k': 'dimensionless', 'seed': 'count'}
    SYMBOLS = {'x_i': 'the position of particle i', 'C_{ij}': 'the spring constant between i and j, per time squared: one of `values`, drawn with the weights `probabilities`',
               'k': '`k`, a common factor on the whole matrix', 'p(c)': 'the draw: value `values[m]` with probability `probabilities[m]`'}
    EMIT = "acceleration"
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["values", "probabilities"]
    MECHANISM_TAGS = ["harmonic_network", "random_coupling", "normal_modes"]
    PARAM_ROLES = {"values": "spring_constants_to_draw_from", "probabilities": "their_weights", "k": "common_factor", "seed": "rng_seed"}
    REFERENCE = "Plexus (this work); ParticleGraph PDE_K."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.values = [float(v) for v in params["values"]]
        self.prob = [float(v) for v in params["probabilities"]]
        self.k = float(params.get("k", 1.0))
        self.seed = int(params.get("seed", 0))
        self._C = None

    def _matrix(self, n, dev, dt):
        if self._C is None:
            e = np.random.default_rng(self.seed).choice(self.values, size=(n, n), p=self.prob)
            e = np.tril(e) + np.tril(e, -1).T
            np.fill_diagonal(e, 0.0)
            self._C = torch.as_tensor(e, device=dev, dtype=dt)
            print(f"[spring_matrix] {self.at}: {n}x{n}, constants {self.values}, mean {float(self._C.sum()) / (n * (n - 1)):.4g}", flush=True)
        return self._C

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        occ = lvl.occ.to(pos.dtype)
        C = self._matrix(pos.shape[0], pos.device, pos.dtype) * occ[None, :]
        acc = self.k * (C @ pos - C.sum(1, keepdim=True) * pos) * occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].to(pos.dtype)
        return {self.at: acc}
