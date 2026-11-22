# Growth and division as one vocabulary on both sets -- audit and design (exp33, 2026-10-25)

Status: build order steps 1 (vertex planar growth) and the vertex half of 3-4 (currency option on cell_divide, doubler on mass) DONE and tested on branch growth-cycle (see exp33 md Finding 23); the noise blocks, the MPM half and the naming below are not built.

## 1. What exists (read from the code)

Vertex set (`vertex_ops.py`, `diffusion_reaction.py`):
- `cell_grow` (what a cell ASKS for): linear scale s, ds/dt = s r (rho + Hill(a)); raises the TARGETS A0 ~ s^2, P0 ~ s, V0f ~ s^GROWTH_DIMS (3; `timer_planar` sets 2). Models: default, `sizer`, `balance`, `timer` (d ln V/dt = ln(V_target/V)/T), `stretch`, `timer_planar`.
- `cell_divide` (what the topology DOES): trigger `_trigger(v_now, v_birth, jit, age, v_ref)`: default `sizer` v >= factor * jit * v_ref, v = the REALISED wedge (or polyhedron) volume read by `cell_size()`; `doubler` v >= factor * v_birth; `cycle` reads the phase `cell_cycle` owns. `jit` = a per-cell threshold multiplier drawn with numpy at (re)birth (`_fresh_djit`, Gaussian cv clipped 0.4-1.8). Daughters: targets halved.
- `cell_cycle` (when a cell is READY): per-cell state p (progress), phases G1/S/G2/M; G1 exit rules sizer, adder, doubler, timer, transition_probability; S, G2, M are clocks. Divides nothing; `cell_divide[cycle]` does.
- Known defects recorded in notes/size_cycle/SIZE_CYCLE_PLAN.md: the growth rate on the LINEAR scale is a volume rate 3 x r; three size readers in two volume conventions; adder arms behaving as sizers.

MPM set (`gastrulation_ops.py`, `cleavage_ops.py`):
- `colony_grow[mpm_points*]` FUSES growth and trigger: dN/dt = dims r (rho + Hill) N on the point count N; when N >= `target` (jittered by `target_jitter`, numpy) it sets the cell block `mark`; `cell_divide[cleave | cleave_tangent]` halves the points. No cycle operator, no timer, no doubler/adder.

Randomness today: numpy draws on the host at (re)birth, realised values stored (threshold, target). Not differentiable in their spread; only the seed varies.

Mismatch for the requested scheme (division at x2 mass, fixed mass growth rate, fixed density, optional fixed-duration cycle, errors):
- no MASS currency on the vertex set (it reads realised volume or targets);
- no VOLUME trigger on the MPM set (it reads N);
- growth rate error and cycle-duration error exist nowhere;
- growth + trigger fused on the MPM set, split on the vertex set.

## 2. Design (Plexus rules: variants beside their base, no new *_ops.py, subtract before adding)

Keep the split already drawn: grow = the rate, cycle = when ready, divide = topology.

1. CURRENCY. One reader `cell_size(..., currency: mass | volume)`, default mass. Vertex: mass = target area A0 under planar growth (V0f ~ A0), volume = realised wedge volume (today's reading). MPM: mass = N points, volume = hull / J-corrected volume. Fixed density means mass and volume coincide; the choice matters only when density varies.
2. GROWTH LAWS (in the currency), as models of `cell_grow` (vertex) and of `colony_grow` (MPM, rate only, trigger removed): `exponential` d ln m / dt = g_j (fixed rate); `homeostatic` g_j = ln(m_target / m_0) / T_j (the rate that reaches the target in the cycle time, from the initial and target values); existing sizer / balance / timer unchanged. `cell_grow[planar]` = GROWTH_DIMS 2 (three lines: the else-branch already exists).
3. DIVISION RULES as rules of `cell_cycle`, shared by both sets (the MPM set gains a cell_cycle model that reads N and writes the `ready` mark `cleave_tangent` already consumes; `colony_grow`'s embedded trigger is deleted): `doubler_mass` m >= f * m_birth; `doubler_volume`; `timer` age >= T_j. Default: doubler_mass, f = 2.
4. ERRORS as per-cell parameters, stored as NOISE not as realisations: each cell carries standard-normal draws eps_f (threshold), eps_g (rate), eps_T (duration) in width-1 cell blocks, drawn at seeding and at birth from a counter-based generator keyed by (seed, lineage id, birth counter), detached and non-learnable; the realised parameter is computed at use time: f_j = f exp(cv_f eps_f), g_j = g exp(cv_g eps_g), T_j = T exp(cv_T eps_T). Log-normal keeps them positive with no clipping.

## 3. Differentiability of the error (the trick)

Reparameterisation: x_j = mu + sigma eps_j (or mu exp(sigma eps_j)) with eps_j drawn once and frozen. The gradient flows to mu and sigma (the learnables) and never to eps; the randomness is an INPUT, not a differentiable function. Any distribution works through the inverse CDF of a frozen uniform u_j: x_j = F^-1(u_j; params). Frozen per-lineage noise also makes the loss landscape deterministic across iterations (common random numbers), so a fit's variation is the parameters', not the sampling's; claims about spread are made over several noise seeds, as the paper already requires of repeats (plexus2.tex, training limits).

What stays non-differentiable: the division EVENT (a threshold crossing, a whole-number jump), exactly as whole-point additions are today. The continuous growth between events carries the gradient to g, f, T and the cv's through the realised per-cell parameters; the event count and times get none. Under the fixed-duration scheme the event time is known in advance (T_j), so the loss still sees it through the states, not through the timing.

## 4. Build order (each step tested before the next)

1. `cell_grow[planar]` (vertex). Test: A0 and V0f grow at the same ln rate.
2. Noise blocks + counter-based draws (one helper, used by both sets); tests: reproducible per (seed, lineage), identical across two rollouts, gradient reaches sigma and not eps.
3. `cell_size` currency (mass default); test: identity against today's reading where currency = volume.
4. `cell_cycle` rules doubler_mass / timer on the vertex set; `cell_divide[cycle]` unchanged.
5. MPM: `colony_grow` rate-only models (exponential, homeostatic); `cell_cycle` MPM model; remove the embedded trigger behind a model switch (default behaviour of existing specs unchanged).
6. Twin: re-run the vertex twin (3 seeds, arm 100) with planar + doubler_mass; MPM with the same g, f; the mass is identical by construction, the controller is dropped; rebuild the reference curves; then relaunch.

## 5. Naming (Cedric: 'consistency between vertex and MPM')

One vocabulary, the operator names the registry already shares: `cell_grow`, `cell_cycle`, `cell_divide` on both sets (`cell_divide` already has vertex and MPM models under one name, so there is precedent). Models name the mechanism, a `currency: mass | volume` parameter names what is read. The registry demands one KIND per operator name, so the MPM growth cannot be a model of `cell_grow` while it adds points (`colony_grow` is `structural`, `cell_grow` is `lateral`): split it. `cell_grow[points]` (lateral) integrates a continuous per-cell mass state, differentiable and the same law as the vertex's (exponential at g_j, or homeostatic); a separate structural operator, proposed name `cell_populate`, adds material points so each cell holds the mass it has. `colony_grow` stays registered, unchanged, for the existing specs, marked legacy.
