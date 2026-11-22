# Audit of the Plexus operators for LATERAL GROWTH of an MPM cell sheet (2026-10-27)

Written for: Cedric and the next exp33 session. Question: which existing operator could make the MPM vesicle grow in DIAMETER (cells widening in the sheet plane) instead of in THICKNESS
(the radial columns of exp29 movies 32-35 and of UCB round 1's best episode, 103.6 um at 200 frames with the wall thickening)?

Source: the 431 registrations of `register_operator` in the 35 files of `src/plexus/operators/` (61,257 lines; listing by family in `/tmp/op_registry.txt`, rebuilt by the script in the session
transcript). Families: mechanics 71, population 63, signalling 52, seed 49, fields 43, mpm 39, interaction 20, boundary 19, motility 18, topology 15, hierarchy 14, motion 9, metabolism 4, polarity 3,
harness 3, cell 3, relation 2, io 2, observation 1, coupling 1. Signalling (neurons), fields (morphogens), seed, io, harness, metabolism, relation, motion, observation carry nothing that moves
material and are not listed below.

## 1. The one operator written for exactly this: `active_force[apical_area]` (adhesion_ops.py:1196)

A FORCE that spreads each cell's points in the tangent plane until the cell reaches a target footprint A0_c = a_0 N_c^exponent (N_c its point count; exponent 1 = a sheet of constant height,
2/3 = isotropic); force-free per cell (the cell's mean acceleration is taken back), the deficit clamped. Knob hooks exist: `k_block` (the `kdrive` knob of the exp30-33 vertex fits),
`exponent_block` (`kexp`). It is NOT in movie 51's rig (`exp29_polv200_p250_ov05_g0m`: no `active_force` at all).

What happened with it (exp29 md, 2026-10-22/23; exp33 md Finding 11):
- k 2 outside the substep block: nothing (the contact penalty 600 and the swell pressure win). k >= 100 outside the block: the cells implode within one frame. k 20-100 INSIDE the substep block:
  stable; before the first division the footprint follows its target (x1.29 at k 100) and the diameter leads (x1.07).
- At k 100 over 300 frames with divisions (movie 35, 4/10): every cell sits AT its footprint target (0.99) and the sphere still does not widen (x1.05): the packing absorbs the spreading by
  INTERLEAVING (contact degree 15.7 against 12; each cell's radial span 1.8 x the height its volume needs). The leak was the contact's overlap tolerance 0.25.
- In the vertex fits (exp33, overlap 0.05, thin seed) the drive's gradient fell to 0 once the cells reached the target: a target-reached ceiling, not a bound. Those fits opened holes (void 0.25)
  for other reasons (tension / lumen pushed to walls).
- NEVER TRIED: the drive on the current rig (250 points, overlap 0.05, mass rule, growth from frame 0, no lumen) with k and the exponent as gradient knobs. That is the cheapest next step: the
  operator, both hooks and the builder code (`exp30_problem.py`) exist.

Its own docstring records the finding that disqualifies the swell route (section 3): "A one-shot in-plane compression of F would not do: the cells' Maxwell time (tau 2 frames) turns any anisotropic
compression back into an isotropic one within two frames, so the drive has to be re-applied every frame".

## 2. Kinematic growth = a rate on the deformation gradient (the route not yet taken on this rig)

These change the material's REST SHAPE (F <- G F, the multiplicative growth split of Rodriguez 1994) instead of pushing with a force. Growth as a declared rate does not ask the material how stiff it is.

- `deform_control` / `deform_control{block}` (cell_ops.py:7901 / 8044): F <- expm(-A dt) F once a frame, A a symmetric 3x3 rate; the `block` variant reads six numbers PER PARTICLE from a state
  block every call, differentiable, "the control variable of a shape-morphing optimisation" (plexus.morph). The general in-plane growth tensor is this with A = -a (I - n n^T), n the cell's radial
  normal: no new mathematics, only the per-point A from the cell's normal and its mass rate.
- `polar_growth` (cell_ops.py:6863): the same split constrained to an elongation along the cell's polarity at constant volume, G = I + (r - 1) n n^T + (r^-1/2 - 1)(I - n n^T), with a cap of 5 %
  of the length per frame. The in-plane growth is the same formula with the axes exchanged and det G > 1 (the mass is added).
- `active_strain` (contractile_ops.py:83): the cardio rest-length increment per cell along a fibre direction f and across it, from per-cell blocks `g`, `g2`; negative g = growth. Same split, per
  cell, differentiable, but with the cardio clock attached.
- `cell_populate[mpm]` `swell` / `swell: inplane` / `lateral_block` (gastrulation_ops.py; built 2026-10-26): a ONE-SHOT compression of F when points are added. The lateral scan (lateral 0 / 0.5 / 1,
  3 passes at horizon 50) moved the loss by 3 % and the diameter by 0.3 um: the Maxwell relaxation of `mpm_strain{polar}` (F pulled to J^(1/3) R with memory exp(-dt / tau), tau 2) erases the
  anisotropy within 2 frames, exactly as the apical_area docstring says. A one-shot cannot work on a viscoelastic cell; a rate re-applied every frame can, because the relaxation removes the STRESS
  of the last increment while the rest shape keeps the accumulated growth.

## 3. Volume and pressure (isotropic: they set HOW MUCH, not WHERE)

- `mpm_density_pressure` (mpm_ops.py:5335): P = k max(rho / rho_0 - 1, 0) from the grid density into the `turgor` buffer, "so that material added anywhere has to make room" -- the force that point
  addition lacks (a point's stress depends on its own F only; two points at one place do not feel each other). Isotropic; the relief goes where the resistance is least, radially on a free shell.
- `mpm_turgor[cycle]` (mpm_ops.py:2047): each cell's rest volume J* = s_c 2^(cycle progress) through the `turgor` buffer; isotropic growth through the cycle, per cell.
- `lumen_pressure` / `[outward]` (adhesion_ops.py:2464 / 2613): each cell's inner face pushed out to its share of the lumen volume (knob `klumen`). Widens the sphere by STRETCHING the sheet: movie
  20's holes (void 0.26 at frame 290) are this.

## 4. Constraints on the thickness (make radial the direction of most resistance, so an isotropic push goes lateral)

- `bm_shell` + `bm_cell_contact{warp}` (bm_shell_ops.py:164, 341): a stretchable, bendable basement membrane on a node graph outside the cells (adhesion_reach, repulsion, tether), with `bm_secrete`,
  `bm_remodel`, `bm_crosslink`, `bm_unbond` to let it grow and turn over. An outer membrane that resists the wall thickening is the biological constraint (Chan 2019's cyst: a footprint or a
  lumen pressure sets the growth). Cost: a second set, Warp, exp27's seeding (`seed_positions[bm_sphere]`).
- `mesh_contact[centre]` (contact_ops.py:2377): contact against a surface indexed by direction from a centre -- two concentric spherical surfaces, one inside and one outside the shell, moved at the
  target diameter's rate (`surface_drive`, contact_ops.py:1724, moves a seeded surface at a set rate; `ecm_from_cell{sphere}` is a prescribed growing ball), would clamp the wall's thickness
  kinematically. A test rig, not a mechanism.
- `plate_confine[sphere]` (contact_ops.py:1440): the same clamp, vertex sets only.
- `cortical_tension` (cell_ops.py:7489): an in-plane tension gamma (I - m m^T) on a cell's membrane -- the SIGN that pulls a cell round, i.e. the opposite of lateral growth; `polar_active_stress`
  (cell_ops.py:6769) is an active stress along the polarity (negative = extension). The apical_area docstring and `polar_growth`'s argue against stresses for growth: a stiff cell balances a stress
  it can afford, and the strain won is sigma / (lambda + 2 mu).

## 5. Division geometry and the vertex laws (no help on their own)

`cell_divide[cleave_tangent]` (in the rig: the spindle in the layer, the daughters side by side) and `cell_divide[mpm_planar]`; `cell_grow[planar]` and `[timer_planar]` (diffusion_reaction.py:3425,
4027) are the VERTEX laws "grow in area at constant height" -- the target this audit wants on MPM cells, written only on the vertex mesh; `cell_grow[mpm]` is their mass-only twin.

## 6. Hole closers (lateral MOTION, not growth)

`active_force[sphere_crawl]` / `[sphere_crawl_hashgrid]` / `[sphere_crawl_seek]` (adhesion_ops.py:1437, 1979; exp27): cells crawl over the sphere and reach into the gaps around them -- the free
space to the side of each point is read along its tangential outward direction. Closes holes by crawling; does not change a cell's rest shape.

## Conclusion

No registered operator produces lateral growth of an MPM cell as a kinematic law. The nearest is the footprint force `active_force[apical_area]`, tried in exp29 and the vertex fits with two
defects (it stalls at its own target; the packing absorbed it by interleaving at overlap 0.25) and never run on the overlap-0.05, 250-point rig. The swell route (one-shot F compression, the
`lateral` knob) is refuted by the Maxwell relaxation, which the apical_area docstring had recorded on 2026-10-22.

Proposal, assembled from the pieces above (no new mathematics): `cell_grow[mpm_planar]` -- a RATE on F, per cell, every frame, as `deform_control{block}` / `polar_growth` do it:

    G_c = exp( e_c * g_c * dt ) on the tangent plane,  1 on the normal:   G = I + (exp(e_c g_c dt) - 1) (I - n_c n_c^T)
    F_i <- G_c F_i   for every point i of cell c, once a frame, before the substeps

with n_c the cell's radial unit vector (as `apical_area` computes it: centroid minus the vesicle centre, or `centre_block`), g_c = d ln m_c / dt the cell's own mass growth rate (what `cell_grow[mpm]`
already integrates and `cell_populate[mpm]` follows), and e_c the share of the growth taken in the plane: e = 1/2 is area ~ mass at constant height (the vertex `cell_grow[planar]`), e = 1/3 is
isotropic (then the normal gets the remaining third). `cell_populate[mpm]` then adds its points with `swell: false`: the rest volume the tensor opened is what the new points fill, so no one-shot
compression and no density pressure are needed. Differentiable in e_c (F is rebound, as `deform_control{block}`); e_c is the `lateral` knob with a meaning that survives the relaxation.
It is `polar_growth` with the axes exchanged and det G > 1, or `deform_control{block}` with A_i = -e g (I - n n^T) computed per point from its cell: about 40 lines, registered as a model of
`cell_grow` (the law) or as `cell_populate[mpm]`'s companion. Risk: a rate the relaxation cannot keep up with tears the cell (polar_growth caps |ln r| at 5 % per frame; here g dt = 0.0057 per rig
frame, 0.0003 per optimisation frame: far below).

Second lever, independent: put `active_force[apical_area]` back on the current rig with `kdrive` and `kexp` as gradient knobs (one builder flag), and let the UCB tree tell the two apart.
