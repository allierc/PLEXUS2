"""Morph a ball into a named shape by optimising a FIELD of deformation-gradient rates, and render the result as glass.

One file, on purpose: of seven scripts from a round of four agents, three things survived (the optimiser, the two verified speed wins, the renderer that made shapes judgeable); the rest was measurement scaffolding.

    A_p = sum_i w_i(X0_p) A_i           w = trilinear weights on a `--ctrl` cube of control nodes
    F_p <- expm(-A_p dt) F_p            once a frame

Lagrangian on purpose: the trilinear weights are computed once from each particle's position at t = 0, so a point carries its own instruction as it moves (an Eulerian field would instruct whatever passes through a place, a much harder thing to optimise), and a coarse-to-fine schedule works because the parameters mean the same at 5,000 particles as at 50,000.

    loss    log nodal mass on the grid, the Eulerian loss the paper argues for: a target shape has no particle correspondence, so a position loss has nothing to match exactly where the shapes differ; `log` because the two mass fields differ by orders of magnitude between "material" and "none"
    target  points sampled INSIDE a mesh by ray parity against its surface, scaled into the box

What worked, measured and merged below: torch.compile on the substep block (1.85x wall clock, 1.7x lower peak memory); warp for the final gradient-free rollout (3.6x at 12,500 particles on an L4, 5.9x at 50,000 on an A100, against the ALREADY-COMPILED torch path, positions agreeing to 1.5e-07); TF32 (free, 0.015% change in final loss; kept, not a lever).

What did NOT work (each is still runnable, since a negative result is credible only while its code is here):
  a spatial smoothness penalty on the control field   monotonically harmful (final loss 0.003442 at weight 0 up to 0.055271 at 0.1; total-variation the same); the paper's stabiliser is a TEMPORAL low-pass on F between control layers, and its control is per-particle
  --control-dof 9 (control may rotate, not only stretch)   a tie on every target at identical cost (teapot 0.002060 vs 0.002049, armadillo 0.007383 vs 0.007391, spot 0.002587 vs 0.002533); six components store a symmetric rate, so expm(-A dt) is a pure stretch, but that is not why appendages fail
  --trace-free (preserve volume by construction)   made volume WORSE, 0.316x of initial against 0.817x (loss 0.008465 vs 0.007014): det(expm(-A dt)) = 1 constrains the GROWTH tensor, while realised volume is sum(p_vol * det F) and the elastic part is free to compress; volume is set by the MPM solve
  --control ngp (multiresolution hash encoding instead of the grid)   diverged, not under-trained

Before comparing anything: scatter atomics make training non-deterministic, so repeat runs of one configuration move the final loss by 18 to 44 per cent (a sweep that ranked six configurations inside a 1.9 per cent spread was noise, and one arm of an apparent 28 per cent "winner" scored worse than the baseline it beat). Run repeat seeds or do not claim an ordering.

    PYTHONPATH=src python -m plexus.morph --targets cow,armadillo --iters 60"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import numpy as np
import yaml

# ROOT, MODELS: The repository root and the folder of the target meshes.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(ROOT, "papers", "morph_models")


# spec: The smallest model that can be asked this question (one material, one grid, no forces) -> spec dict.

def spec(n_pts, box, n_grid, frames, dt, sub, differentiable=True):
    """The smallest model that can be asked this question: one material, one grid, no forces. `differentiable` names the operator bodies: the default bodies write in place (what a captured CUDA graph needs and autograd cannot have), the differentiable ones compute the same numbers and rebind. Training needs them; the final rollout does not, and warp, which the engine selects on its own once the pin is dropped, is several times faster."""
    impl = dict(implementation="differentiable") if differentiable else {}
    return dict(
        general=dict(name="morph", seed=0, n_frames=frames, dt=dt, record_cap=3,
                     boundary="wall", dim=3, world=[box, box, box], units=dict(length_um=100.0)),
        sets={"mpm_particle": dict(n=n_pts, types={"cyto": dict(
            fraction=1.0, youngs=90.0, density=1.0,
            block=[0.4 * box, 0.4 * box, 0.4 * box, 0.6 * box, 0.6 * box, 0.6 * box])})},
        fields=dict(mpm_grid=dict(frame="mpm_grid", n_grid=n_grid)),
        operators=[dict(op="mpm_strain", at="mpm_particle", **impl),
                   dict(op="mpm_scatter", at="mpm_grid", **{"from": "mpm_particle"}, drag=0.5,
                        a_max=200.0, **impl),
                   dict(op="mpm_grid_update", at="mpm_grid", wall_damp=0.9, **impl),
                   dict(op="mpm_gather", at="mpm_particle", **{"from": "mpm_grid"},
                        wall_damp=0.9, vmax=1.0e9, **impl)],
        # `compile: true` on the substep block is worth 1.85x under grad for ~35 s of one-time compilation and cuts peak memory 1.7x (the substep is a few hundred small kernels, what a fusing compiler wants); `max-autotune` and `reduce-overhead` both FAIL (they conflict with the engine's own CUDA-graph capture), so `default` only
        schedule=[dict(substep_dt=sub, compile=True,
                       steps=["mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"])],
        plotting={},
    )


# ball: n points uniform in a ball, the body this starts as -> [n, 3].
def ball(n, c, r, rng):
    """n points uniform in a ball -- the body this starts as."""
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return c + u * (r * rng.random((n, 1)) ** (1.0 / 3.0))


# sample_inside: n points uniform inside a mesh, centred at `box_c` and scaled to `size` across its longest axis -> [n, 3].
def sample_inside(path, n, box_c, size, rng):
    """`n` points uniform inside a mesh, centred at `box_c` and scaled to `size` across its longest
    axis. Rejection against the surface itself, so the sample is the SOLID and not the shell."""
    import pyvista as pv
    m = pv.read(path)
    if m.n_points == 0:
        raise ValueError(f"{path}: empty mesh")
    m = m.clean().triangulate()
    # a hole makes `inside` meaningless (the Stanford bunny is open at the base, and a ray from an interior point escapes through it, so parity says "outside" for half the body)
    m = m.fill_holes(1e9).clean()
    m = m.compute_normals(auto_orient_normals=True, consistent_normals=True)
    lo, hi = np.array(m.bounds).reshape(3, 2).T
    ctr = 0.5 * (lo + hi)
    scale = size / float((hi - lo).max())
    out = []
    while sum(len(o) for o in out) < n:
        q = rng.random((4 * n, 3)) * (hi - lo) + lo
        sel = pv.PolyData(q).select_enclosed_points(m, tolerance=0.0, check_surface=False)
        keep = q[np.asarray(sel["SelectedPoints"]) > 0]
        if len(keep) == 0:
            raise ValueError(f"{path}: no interior points found -- is the surface closed?")
        out.append(keep)
    return (np.concatenate(out)[:n] - ctr) * scale + box_c


# mass_grid: The grid's own nodal particle count by the solver's trilinear weights -> grid.
def mass_grid(X, m, box, n_grid, dev, torch):
    """The grid's own nodal mass, by the trilinear weights the solver would use. Unit weights, not the particle mass, at every call site: a material point weighs ~1e-8, so log1p of its nodal mass is 1e-8 too and the loss underflowed (0.000000 with a gradient of 6e-17, which looks like convergence and is arithmetic); counting particles per node makes the field O(1) and lets the log do its job."""
    h = box / n_grid
    g = (X / h).clamp(0, n_grid - 1.001)
    b = g.floor().long()
    f = g - b.float()
    acc = torch.zeros(n_grid ** 3, device=dev, dtype=X.dtype)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (((1 - f[:, 0]) if dx == 0 else f[:, 0])
                     * ((1 - f[:, 1]) if dy == 0 else f[:, 1])
                     * ((1 - f[:, 2]) if dz == 0 else f[:, 2]))
                i = ((b[:, 0] + dx).clamp(0, n_grid - 1) * n_grid
                     + (b[:, 1] + dy).clamp(0, n_grid - 1)) * n_grid \
                    + (b[:, 2] + dz).clamp(0, n_grid - 1)
                acc = acc.index_add(0, i, w * m)
    return acc


# trilinear: Each particle's eight corner nodes in the control cube and its weight on each, from its MATERIAL coordinate -> (indices, weights).
def trilinear(X0n, c, R, K, dev, torch):
    """Each particle's eight corner nodes in the control cube, and its weight on each, from its
    MATERIAL coordinate. Returns (indices, weights), eight of each."""
    u = np.clip((X0n - (c - R)) / (2 * R) * (K - 1), 0, K - 1.001)
    b = np.floor(u).astype(int)
    f = u - b
    idx, wts = [], []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (((1 - f[:, 0]) if dx == 0 else f[:, 0])
                     * ((1 - f[:, 1]) if dy == 0 else f[:, 1])
                     * ((1 - f[:, 2]) if dz == 0 else f[:, 2]))
                i = ((b[:, 0] + dx).clip(0, K - 1) * K + (b[:, 1] + dy).clip(0, K - 1)) * K \
                    + (b[:, 2] + dz).clip(0, K - 1)
                idx.append(torch.as_tensor(i, device=dev))
                wts.append(torch.as_tensor(w, dtype=torch.float32, device=dev))
    return idx, wts


# _ENV_CACHE: Cache of the synthetic environment texture. The render is VTK's physically-based path, not path tracing (OSPRay is unavailable in the devcontainer and on the cluster: `import vtkmodules.vtkRenderingRayTracing` raises ModuleNotFoundError); 50,000 opaque dots would hide the silhouette a morph is judged on, so a surface is always reconstructed.

_ENV_CACHE = {}


# default_settings: The glass render settings: a cold saturated blue dielectric -> dict.
def default_settings():
    """A cold, saturated blue dielectric: opacity ~0.5, roughness low but not mirror-flat, sixteen depth-peeling layers so the far side of a concave body (an arm crossing a torso) is not clipped, and a fourth BACKLIGHT, since glass with nothing behind it looks like dark plastic."""
    return dict(
        ngrid=176, sigma=0.95, iso_frac=0.3, smooth_iter=35,
        color="#1a4fc0", target_color="#4a5a6a", ground_color="#04050a", bg="#05070a",
        roughness=0.07, opacity=0.5, diffuse=0.42, ambient=0.06,
        ior=1.4, coat_strength=0.85, coat_roughness=0.05, coat_ior=1.5, metallic=0.0,
        ssao_radius_frac=0.12, ssao_bias_frac=0.002, ssao_kernel=192, peels=16,
        env_bright=2.3, key_intensity=1.2, fill_intensity=0.32, rim_intensity=1.1,
        rim_color=(0.75, 0.85, 1.0), back_intensity=1.0, back_color=(0.6, 0.8, 1.0),
        # (elevation_deg, azimuth_deg) for pv.Light.set_direction_angle: key and fill model the form, rim and back separate the glass silhouette from the black background
        key_angle=(55, -35), fill_angle=(35, 150), rim_angle=(-35, 20), back_angle=(15, 175),
    )


# reconstruct_contour: The body as a contoured, gaussian-blurred density -> PolyData.
def reconstruct_contour(X, box, ngrid=176, sigma=0.95, iso_frac=0.3, smooth_iter=35, w=None):
    """The body as a contoured, gaussian-blurred density. The two numbers fight: a finer grid resolves thinner features but fewer particles land per voxel, so the density is noisier (measured: 96^3 with one 3-voxel box blur is clean but over-fused, legs merge into a wedge; 224^3 at the same blur is pockmarked). 176^3 with a single gaussian of about one voxel is the balance for these 50,000-particle clouds; change the particle count and this number should move with it."""
    from scipy.ndimage import gaussian_filter
    import pyvista as pv
    dx = box / ngrid
    idx = np.clip((X / dx).astype(int), 0, ngrid - 1)
    dens = np.zeros((ngrid, ngrid, ngrid), np.float32)
    np.add.at(dens, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0 if w is None else np.asarray(w, np.float32))      # w: a weight per point (its volume) when the points are not of one size, e.g. a dense skin on a coarse bulk
    dens = gaussian_filter(dens, sigma=sigma)
    g = pv.ImageData(dimensions=(ngrid + 1,) * 3, spacing=(dx, dx, dx), origin=(0.0, 0.0, 0.0))
    g["v"] = np.pad(dens, ((0, 1), (0, 1), (0, 1))).flatten(order="F")
    iso = max(iso_frac * float(dens[dens > 0].mean()), 1e-6)
    surf = g.contour([iso], scalars="v")
    if surf.n_points == 0:
        return None
    surf = surf.smooth(n_iter=smooth_iter, relaxation_factor=0.15)
    return surf.compute_normals(auto_orient_normals=True, non_manifold_traversal=False)


# env_texture: A synthetic equirectangular sky for image-based lighting -> texture.
def env_texture(bright=2.3):
    """A synthetic equirectangular sky for image-based lighting (the cluster node has no internet to fetch an HDRI): a gradient from horizon navy to a lighter blue-teal zenith plus one soft bright patch, which is what a physically-based dielectric picks up as highlight and reflected colour. `bright` matters: at 1.0 a translucent, low-diffuse glass has nothing to catch and reads near-black with white flecks, not blue."""
    import pyvista as pv
    key = round(float(bright), 3)
    if key not in _ENV_CACHE:
        H, W = 128, 256
        yy = np.linspace(0, 1, H)[:, None] * np.ones((1, W))
        xx = np.linspace(0, 1, W)[None, :] * np.ones((H, 1))
        sky = np.stack([0.05 + 0.10 * yy, 0.07 + 0.14 * yy, 0.12 + 0.22 * yy], axis=-1) * bright
        sun = np.exp(-(((xx - 0.7) * W) ** 2 + ((yy - 0.15) * H) ** 2) / (2 * 22 ** 2))
        sky = sky + sun[..., None] * np.array([0.6, 0.6, 0.65]) * bright
        tex = pv.Texture((np.clip(sky, 0, 1) * 255).astype(np.uint8))
        # mipmapping is not cosmetic: without it every marching-cubes facet catches a raw, aliased texel of the bright patch and the glass reads as covered in tiny white measles
        tex.SetMipmap(True)
        tex.SetInterpolate(True)
        _ENV_CACHE[key] = tex
    return _ENV_CACHE[key]


# build_lights: Key, fill, rim and back lights -> list of lights.
def build_lights(pl, s):
    """Key, fill, rim, plus a fourth, a BACKLIGHT, that a glass render needs and an opaque one does not: light from behind is what a translucent dielectric forwards through itself, and without it there is nothing to be transparent to except the dim environment map."""
    import pyvista as pv
    for angle, intensity, color in ((s["key_angle"], s["key_intensity"], None),
                                    (s["fill_angle"], s["fill_intensity"], None),
                                    (s["rim_angle"], s["rim_intensity"], s["rim_color"]),
                                    (s["back_angle"], s["back_intensity"], s["back_color"])):
        light = pv.Light(light_type="scene light")
        light.set_direction_angle(*angle)
        light.intensity = intensity
        if color is not None:
            light.SetColor(*color)
        pl.add_light(light)


# draw_scene: Everything but the plotter and the camera: glass body, faint target cloud, lights, image-based lighting, ground plane, domain wireframe -> None.
def draw_scene(pl, surf, target, box, X, s):
    """Everything but the plotter and the camera: the glass body, the faint target cloud behind it,
    four lights, image-based lighting, a small dark ground plane, and the domain wireframe."""
    import pyvista as pv
    pl.set_background(s["bg"])
    pl.set_environment_texture(env_texture(s["env_bright"]))
    pl.renderer.TexturedBackgroundOff()   # image-based lighting for reflections only; bg stays black

    pl.add_mesh(pv.PolyData(target.astype("float32")), color=s["target_color"],
                opacity=0.08, point_size=1.6)

    if surf is not None and surf.n_points > 0:
        pl.add_mesh(surf, color=s["color"], pbr=True, metallic=s["metallic"],
                    roughness=s["roughness"], opacity=s["opacity"], diffuse=s["diffuse"],
                    ambient=s["ambient"], smooth_shading=True)
        # the dielectric itself: an index of refraction and a thin sharp clear-coat over the base layer, which gives a low-roughness surface its glassy double highlight instead of one soft plastic one
        prop = list(pl.renderer.actors.values())[-1].GetProperty()
        prop.SetBaseIOR(s["ior"])
        prop.SetCoatStrength(s["coat_strength"])
        prop.SetCoatRoughness(s["coat_roughness"])
        prop.SetCoatIOR(s["coat_ior"])
    else:
        pl.add_mesh(pv.PolyData(X.astype("float32")), color=s["color"], point_size=2.2)

    build_lights(pl, s)

    # the ground is plain Lambertian, not PBR: a pbr plane under the bright environment texture that lights the glass becomes a lit grey studio floor filling half the frame, wrong for the black-background convention
    ymin = float(X[:, 1].min())
    ground = pv.Plane(center=(box / 2, ymin - 0.002, box / 2), direction=(0, 1, 0),
                      i_size=box * 1.05, j_size=box * 1.05)
    pl.add_mesh(ground, color=s["ground_color"], pbr=False, ambient=0.02, diffuse=0.5, specular=0.0)
    pl.add_mesh(pv.Box((0, box, 0, box, 0, box)), style="wireframe", color="#131a24",
                line_width=1.0, lighting=False)

    pl.enable_ssao(radius=box * s["ssao_radius_frac"], bias=box * s["ssao_bias_frac"],
                   kernel_size=s["ssao_kernel"])
    pl.enable_depth_peeling(number_of_peels=s["peels"], occlusion_ratio=0.0)
    pl.enable_anti_aliasing("ssaa")


# camera_for: The camera position for a cubic domain of side `box` -> camera tuple.
def camera_for(box):
    # asymmetric on purpose: equal x and z offsets from a cubic domain put the box's far vertical edge exactly behind the body's centreline, a bright seam down every frame that has nothing to do with the body
    return [(box * 2.5, box * 1.4, box * 1.65), (box / 2,) * 3, (0, 1, 0)]


# render_movie: The morph as movie.mp4 plus `n_stills` evenly spaced pngs -> None.
def render_movie(out_dir, frames, target, box, name, settings=None, px=1100, fps=20, n_stills=7):
    """The morph as movie.mp4 plus `n_stills` evenly spaced pngs."""
    import pyvista as pv
    import imageio.v3 as iio
    s = settings or default_settings()
    pv.OFF_SCREEN = True
    os.makedirs(out_dir, exist_ok=True)
    still_at = {int(round(v)) for v in np.linspace(0, len(frames) - 1, n_stills)}
    imgs, t_recon, t_draw = [], 0.0, 0.0
    for i, Xf in enumerate(frames):
        X = Xf.astype(np.float64)
        t0 = time.time()
        surf = reconstruct_contour(X, box, s["ngrid"], s["sigma"], s["iso_frac"], s["smooth_iter"])
        t_recon += time.time() - t0
        pl = pv.Plotter(off_screen=True, window_size=(px, px), lighting="none")
        draw_scene(pl, surf, target, box, X, s)
        pl.camera_position = camera_for(box)
        pl.add_text(f"{name}   frame {i}/{len(frames) - 1}   {len(X):,} material points",
                    position="upper_left", font_size=9, color="#c8d4e0")
        t0 = time.time()
        img = pl.screenshot(return_img=True)
        t_draw += time.time() - t0
        pl.close()
        imgs.append(img)
        if i in still_at:
            iio.imwrite(os.path.join(out_dir, f"still_{i:04d}.png"), img)
    iio.imwrite(os.path.join(out_dir, "movie.mp4"), imgs, fps=fps, codec="libx264",
                macro_block_size=None)
    n = max(len(frames), 1)
    return dict(n_frames=len(frames), px=px, fps=fps, s_per_frame_reconstruct=t_recon / n,
                s_per_frame_draw=t_draw / n, s_per_frame_total=(t_recon + t_draw) / n)


# main: The command line: optimise the control field per target and render the morph (coarse-to-fine stages, final gradient-free rollout) -> None.

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default="cow,armadillo",
                    help="comma list of .obj basenames in papers/morph_models")
    ap.add_argument("--n-pts", type=int, default=20000)
    ap.add_argument("--frames", type=int, default=20, help="frames the control is OPTIMISED over")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--ctrl", type=int, default=12, help="control nodes per axis")
    ap.add_argument("--lr", type=float, default=3.0)
    ap.add_argument("--n-grid", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(ROOT, "graphs_data", "si_material"))
    ap.add_argument("--name", default=None, help="output folder name; defaults to morph_<target>")
    ap.add_argument("--px", type=int, default=1100)
    ap.add_argument("--render-frames", type=int, default=0,
                    help="frames in the FINAL rollout, the one that becomes the movie. The rate is "
                         "divided by the same factor, so the total deformation is identical and "
                         "only the sampling changes -- a morph optimised over 20 frames and "
                         "rendered at 400 is the same trajectory seen twenty times more finely, "
                         "not twenty times more of it.")
    ap.add_argument("--stages", default="",
                    help="coarse-to-fine, as `pts:grid:iters,pts:grid:iters,...`. The control is a "
                         "function of MATERIAL coordinates, so it transfers between stages exactly "
                         "-- a cheap stage buys the parameters a dear one would have.")
    ap.add_argument("--vol-weight", type=float, default=0.0,
                    help="penalise the REALISED volume of the rollout, sum(p_vol * det F), against "
                         "its value at the start. Not the control's trace: constraining that "
                         "kinematically makes volume worse, because the loss is lost through the "
                         "dynamics the kinematic proxy cannot see. Differentiated end to end, "
                         "which is the channel the mass loss is blind to.")
    ap.add_argument("--trace-free", action="store_true",
                    help="project the rate onto its deviatoric part. Measured to make realised "
                         "volume WORSE, not better -- see the module docstring. Kept to be rerun, "
                         "not to be used.")
    ap.add_argument("--control-dof", type=int, default=6, choices=[6, 9],
                    help="components per control node: 6 a symmetric rate (pure stretch), 9 the "
                         "full 3x3 (stretch plus rotation). Measured to be a tie on every target.")
    ap.add_argument("--no-render", action="store_true",
                    help="skip the glass movie: morph.npz and meta.json only. The record's own renderer (the 3D "
                         "view, tools/view_movie.py) draws the frames; the glass reconstruction costs seconds a "
                         "frame at 10^5 points and more.")
    ap.add_argument("--control", default="grid", choices=["grid", "ngp"],
                    help="`grid` is a cube of rates read trilinearly. `ngp` is the multiresolution "
                         "hash encoding plus a small head, A(X0) = MLP(hash(X0)); it diverged.")
    args = ap.parse_args()

    import torch
    # TF32 is free and changes nothing: 1.002 -> 0.990 s an iteration at 12,500 particles on an L4, inside that sweep's own noise, final loss moving 0.015%
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from . import engine, operators  # noqa: F401  -- importing operators registers them
    from .schema import load

    BOX, C, R = 0.35, 0.175, 0.055
    DOF = args.control_dof
    dev = torch.device(args.device)
    rng = np.random.default_rng(0)
    c = np.array([C, C, C])
    os.makedirs(args.out, exist_ok=True)

    def build(raw):
        f = os.path.join(tempfile.mkdtemp(prefix="morph_"), "spec.yaml")
        yaml.safe_dump(raw, open(f, "w"), sort_keys=False)
        return load(f)

    def ngp_control():
        from .models.hashgrid import MultiResHashGrid
        enc = MultiResHashGrid(n_input_dims=3, n_levels=8, n_features_per_level=2,
                               log2_hashmap_size=15, base_resolution=4,
                               per_level_scale=1.6).to(dev)
        head = torch.nn.Sequential(torch.nn.Linear(8 * 2, 32), torch.nn.GELU(),
                                   torch.nn.Linear(32, DOF)).to(dev)
        with torch.no_grad():                        # start from no deformation at all
            head[-1].weight.mul_(0.0); head[-1].bias.mul_(0.0)
        return enc, head

    stages = ([tuple(int(v) for v in st.split(":")) for st in args.stages.split(",")]
              if args.stages else [(args.n_pts, args.n_grid, args.iters)])

    for name in [t.strip() for t in args.targets.split(",") if t.strip()]:
        path = os.path.join(MODELS, f"{name}.obj")
        theta = enc = head = None
        t_target = time.time()
        print(f"\n  {name}: {len(stages)} stage(s), control {args.control}, dof {DOF}", flush=True)

        for si, (n_pts, n_grid, iters) in enumerate(stages):
            sim = build(spec(n_pts, BOX, n_grid, args.frames, 0.002, 3.4e-4))
            X0n = ball(n_pts, c, R, rng)
            X0 = torch.as_tensor(X0n, dtype=torch.float32, device=dev)
            one = torch.ones(n_pts, device=dev)
            tgt_np = sample_inside(path, n_pts, c, 2.6 * R, rng)
            tgt = torch.as_tensor(tgt_np, dtype=torch.float32, device=dev)
            with torch.no_grad():
                rho_t = mass_grid(tgt, one, BOX, n_grid, dev, torch)

            if args.control == "ngp":
                Xm = torch.as_tensor(np.clip((X0n - (c - R)) / (2 * R), 0, 1),
                                     dtype=torch.float32, device=dev)
                if enc is None:
                    enc, head = ngp_control()
                params = list(enc.parameters()) + list(head.parameters())
            else:
                idx, wts = trilinear(X0n, c, R, args.ctrl, dev, torch)
                if theta is None:
                    theta = torch.zeros(args.ctrl ** 3, DOF, device=dev, requires_grad=True)
                params = [theta]

            opt = torch.optim.Adam(params, lr=args.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(iters, 1),
                                                              eta_min=args.lr / 20.0)
            eye = torch.eye(3, device=dev)
            dt = float(sim.dt)
            t_stage = time.time()

            def rollout(keep=None):
                a = (head(enc(Xm)) if args.control == "ngp"
                     else sum(w[:, None] * theta[i] for i, w in zip(idx, wts)))
                if DOF == 9:
                    A = a.reshape(-1, 3, 3)
                else:
                    A = torch.zeros(a.shape[0], 3, 3, device=dev) + torch.diag_embed(a[:, :3])
                    A[:, 0, 1] = A[:, 1, 0] = a[:, 3]
                    A[:, 0, 2] = A[:, 2, 0] = a[:, 4]
                    A[:, 1, 2] = A[:, 2, 1] = a[:, 5]
                if args.trace_free:
                    A = A - torch.diagonal(A, dim1=1, dim2=2).mean(1)[:, None, None] * eye
                Adt = -A * dt
                G = eye + Adt + 0.5 * (Adt @ Adt)   # 2nd order: A dt is small, and 40x cheaper
                # the initial condition goes in through `on_frame`, because `engine.run` builds and seeds its own hierarchy: a ball written into a hierarchy built here is discarded (measured once as a 7 um cube where an 11 um ball was intended, with a plausible-looking loss)
                def cb(Hh, tick, G=G):
                    q = Hh.level("mpm_particle")
                    if tick == 0:
                        q0, q1 = q.state_schema["pos"]
                        with torch.no_grad():
                            q.state[:, q0:q1] = X0
                    else:
                        q.F = torch.bmm(G, q.F)     # functionally, so the tape survives
                    if keep is not None:
                        keep.append(q.get("pos").detach().cpu().numpy().copy())
                Hh, _ = engine.run(sim, device=args.device, on_frame=cb, progress=False,
                                   grad=(keep is None))
                q = Hh.level("mpm_particle")
                return q.get("pos"), (q.p_vol * torch.linalg.det(q.F)).sum()

            vol0 = None
            for it in range(iters):
                opt.zero_grad()
                Xs, vol = rollout()
                if vol0 is None:
                    vol0 = float(vol.detach())
                loss = ((torch.log1p(mass_grid(Xs, one, BOX, n_grid, dev, torch))
                         - torch.log1p(rho_t)) ** 2).mean()
                # the mass loss alone is gameable: it supervises where material IS, not how much, so the cheapest way to match a dense target is to implode (max|J - 1| = 2.33, J crossing zero into inverted elements)
                if args.vol_weight > 0.0:
                    loss = loss + args.vol_weight * (vol / vol0 - 1.0) ** 2
                loss.backward()
                opt.step(); sched.step()
                if it % 20 == 0 or it == iters - 1:
                    e = ((Xs.max(0).values - Xs.min(0).values) * 100).tolist()
                    et = ((tgt.max(0).values - tgt.min(0).values) * 100).tolist()
                    # volume is printed beside the extent because a 55% collapse was invisible in every number this line carried: the shape can approach the target while its body quietly disappears
                    print(f"    stage {si} ({n_pts:,} pts, grid {n_grid}) iter {it:3d}  "
                          f"loss {float(loss):.6f}  vol {float(vol) / vol0:.3f}x  "
                          f"shape {e[0]:5.1f} x {e[1]:5.1f} x {e[2]:5.1f}  "
                          f"target {et[0]:5.1f} x {et[1]:5.1f} x {et[2]:5.1f} um", flush=True)
            print(f"    stage {si}: {iters} iterations in {(time.time()-t_stage)/60:.1f} min "
                  f"({(time.time()-t_stage)/max(iters,1):.2f} s an iteration)", flush=True)

        # the final rollout needs no tape: `differentiable=False` lets the engine select warp (3.6x at 12,500 particles on an L4, 5.9x at 50,000 on an A100 against the already-compiled torch path, positions agreeing to 1.5e-07)
        with torch.no_grad():
            frames = []
            rf = args.render_frames or args.frames
            # the same physical time as the fit, sampled rf / frames times finer: the frame dt shrinks with the frame count (0.002 s x 20 frames = 0.04 s whether 20 or 200 are drawn) and so does the control increment per frame (until 2026-10-27 only the control was divided, so a 200-frame rollout ran the elastic body 10x longer between growth increments and the shape relaxed away: armadillo fitted at log-mass loss 0.0089 rolled out at 0.049, exp01 minisite v6)
            dt_f = 0.002 * args.frames / rf
            sim = build(spec(n_pts, BOX, n_grid, rf, dt_f, min(3.4e-4, dt_f), differentiable=False))
            dt = float(sim.dt)                        # the control increment per frame = the frame's own dt
            rollout(keep=frames)
        frames = np.stack(frames)
        # a rollout-only final stage (`...,500000:64:0`: zero iterations) runs the control optimised on the stage before it on more points and a finer grid: the control is a function of MATERIAL coordinates, so it transfers, but the MPM dynamics are not grid-independent, so the loss is printed on the final stage's own grid and the transfer is judged by a number (Cedric, 2026-10-27, exp01 minisite)
        with torch.no_grad():
            rho_f = mass_grid(torch.as_tensor(frames[-1], dtype=torch.float32, device=dev), one, BOX, n_grid, dev, torch)
            print(f"    final rollout: {len(frames)} frames of {n_pts:,} points on grid {n_grid}; log-mass loss of the "
                  f"last frame against the target, on this grid: "
                  f"{float(((torch.log1p(rho_f) - torch.log1p(rho_t)) ** 2).mean()):.6f}", flush=True)

        d = os.path.join(args.out, args.name or f"morph_{name}")
        os.makedirs(d, exist_ok=True)
        ctrl = (theta.detach().cpu().numpy() if theta is not None
                else np.concatenate([q.detach().cpu().numpy().ravel()
                                     for q in list(enc.parameters()) + list(head.parameters())]))
        # the control without its shape is not a control: the array does not say how many nodes per axis it has nor which family it came from, so nothing downstream can rebuild the rollout (two runs lacking this block were compared as if they differed in one respect)
        meta = dict(control_kind=args.control, ctrl_K=(args.ctrl if theta is not None else 0),
                    control_dof=DOF, vol_weight=args.vol_weight, trace_free=bool(args.trace_free),
                    iters=args.iters, lr=args.lr, stages=args.stages,
                    extent=[C, C, C, R], opt_frames=args.frames, render_frames=len(frames),
                    dt=0.002, n_pts=n_pts, n_grid=n_grid, target=name,
                    seconds_optimise=time.time() - t_target)
        np.savez_compressed(os.path.join(d, "morph.npz"), frames=frames, target=tgt_np,
                            box=BOX, control=ctrl, meta=np.array(json.dumps(meta)))
        t0 = time.time()
        stats = {} if args.no_render else render_movie(d, frames, tgt_np, BOX, name, px=args.px)
        meta["render"] = stats
        meta["seconds_total"] = time.time() - t_target
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2, sort_keys=True)
        print(f"    -> {os.path.relpath(d, ROOT)}  ({len(frames)} frames, "
              + (f"{stats['s_per_frame_total']:.2f} s a frame to render, " if stats else "not rendered, ")
              + f"{(time.time() - t_target) / 60:.1f} min total)", flush=True)


if __name__ == "__main__":
    main()
