# How this folder works

Read this first in any new session that touches `experiments/`.

An experiment is **one markdown file**. It states the question, names the specs
that answer it, and carries the results back. A tool, `tools/exp.py`, reads that
file to launch the jobs, to refresh the numbers, and to build the report.
Nothing else knows a run name.

**The experiment NUMBER is the only handle.** `exp launch 0`, `exp poll 0`,
`exp report` — never `exp00_gen_l4_r1`.

```
experiments/
  INDEX.md              number, name, purpose, landed/total   (written by the tool)
  INSTRUCTION.md        this file                             (written by hand)
  TEMPLATE.md           the SHAPE of one experiment markdown, and the audit of exp00-05 it came from
  expNN_<name>.md       one per experiment, started from TEMPLATE.md (both)
  expNN_<name>/         its RECORD FOLDER, the builder's layout (written by tools/exp_record.py)
  expNN_<name>/papers/  the REFERENCE PAPER(S), downloaded before the design (see below)
  expNN_<name>/data/    the experimental data the experiment compares against (or a path in `data:`)
  expNN_<name>/AGENT_*.md  the ROLE SPEC of the experiment's judge or auditor (written by hand, lands HERE)
  expNN_<name>/gates.yaml  the GATES: points, full/zero bands, basis, source, caps (by hand; the ONE place bands live)
  expNN_<name>/relevant.yaml       the RULES that pick what Plexus already has for this experiment (by hand)
  expNN_<name>/relevant_in_plexus/ SYMLINKS to those specs and outputs + INDEX.md (DERIVED by tools/plexus_catalog.py)
  expNN_<name>/LAUNCH_PROMPT.txt   the prompt that starts a session running the experiment (4 L4 jobs a batch,
                                   a 5-minute judge); watcher port 8840 + NN - 6 for exp06-15
  expNN_<name>/landings/  one report per landed batch: health, card, check (written by tools/exp_land.py)
  TRAPS.md              engine / renderer / tool traps, SHARED by every session (read at start, add when found)
  EPITHELIUM_BM.md      the page exp11 and exp12 share for their one epithelium + basement-membrane rig
  specs/expNN/          the specs the tool generates          (written by the tool)
  specs/catalog/        plexus_catalog.jsonl: every spec of config/ log/ builder/ experiments/ (DERIVED)
  figs/                 figures an experiment's findings cite  (written by tools)
  report.md             one section per experiment            (written by the tool)
```

**An agent's role spec lands in its experiment's own folder, `experiments/expNN_<name>/AGENT_<role>.md`,
never loose in `experiments/`**, and the markdown's `agent:` names that path. An experiment that reuses
another one's judge points at it where it lives (exp 6-15 use exp 3's growth auditor,
`exp03_size_hypotheses/AGENT_growth_audit.md`) instead of copying it.

---

## A new experiment, in order -- the process every experiment follows from its first hour

Learned on exp06-15 (2026-09-26/27); each step names the section that details it.

1. **Read.** This file; `TRAPS.md` whole (`python tools/exp_trap.py list`), then `list <op>` for every
   operator the rig will use; the pool's `relevant_in_plexus/INDEX.md` ("What Plexus already has");
   and any shared page with another experiment on the same rig (`EPITHELIUM_BM.md`). Build on what
   exists; a trap already listed is never rediscovered.
2. **Papers and data** into the experiment's folder ("The reference paper"); numbers read from the
   figures, figure and panel for each.
3. **The markdown** from `TEMPLATE.md`, with its `question:` (the ONE main question, "The main question"),
   `## The question`, `## Directions`, `## Decisions`, LANDING and STATUS blocks. The talk or claim it serves, the main rig and at least two exploration directions.
4. **Gates, rulers, judge** ("Gates, the judge and its tools"): `gates.yaml` (10 points, bands from
   figures or `TBD from <paper, figure>`), rulers in `tools/exp_measures/expNN.py` with planted tests,
   the judge's `AGENT_*_judge.md` (report ends with exploit AND explore). Rig code as registered
   variants ("Code: reuse the operator ...").
5. **DRY RUN CLEAN -- nothing is submitted as a batch until every line below is clean:**

   | check | command | clean means |
   |---|---|---|
   | the specs | `tools/exp.py launch NN --dry-run` | every run's spec writes, and differs from its base only as `differs_by` declares |
   | the gates | `tools/exp_gate_score.py NN --dry` | points sum to 10; every unset band is a `TBD from <paper, figure>` you cannot read yet |
   | the rulers | `pytest tests/test_exp_measures_expNN.py` | the planted inputs read what they plant |
   | the rig runs, healthy, and the card can read it | `tools/exp_smoke.py <group>/<base> --frames 100 --exp NN` | no NON-FINITE, no WRECKED, the movie moves, "every gate key found" |
   | a long rig lasts | `tools/exp_smoke.py <group>/<base> --full` (background) | the same, at full length -- whenever a batch runs past ~2,000 frames or an hour |

   A failure here costs two minutes; the same failure found after a batch cost the sessions hours
   (exp13's UNKNOWN KEY, exp11's NaN membrane, exp14's late mesh collapse).
6. **The first batch**: at most 4 L4 jobs, ONE background waiter (`tools/exp_wait.py NN`, which lands
   it: health, card, `--check`, record, report), at least one exploration arm.
7. **The loop** ("The main question", "The pace", "No questions to the human", "Rungs and the score",
   "Explore within the framework"): at every landing, answer the main question, and let the next batch follow from it; from a
   landing to the next batch at most 10 minutes; nothing over an hour without a submitted batch; open
   choices decided and logged under `## Decisions`, never asked; every new trap added to `TRAPS.md`
   the minute it is found; three stuck batches mean a pivot, never a stop.

**A session's LAUNCH_PROMPT.txt carries this order**: read (1), set up (2-4), dry run clean (5), then
batches (6-7); the watcher port; and "do not commit".

---

## What Plexus already has -- `relevant_in_plexus/`, read BEFORE designing anything

**Plexus has run a great deal already. An experiment starts from what exists, not from nothing.**
On 2026-09-26 the codebase held 2,865 config specs (802 with a run in graphs_data), 773 specs with
their output under `log/`, 637 builder steps and 226 experiment specs. A new experiment that writes
its own sheet, shell, membrane or reaction without looking is how the same working point gets
rebuilt three times with three different defects.

**The catalog.** `tools/plexus_catalog.py build` reads every spec OF THE CODEBASE -- `config/`,
`log/**/spec.yaml`, `builder/*/spec/NNNN.yaml`, `experiments/*/spec/NNNN.yaml` and
`experiments/specs/expNN/*.yaml` -- once, and writes one JSON line per spec to
`experiments/specs/catalog/plexus_catalog.jsonl`: the operators it uses (`op` and `op[model]`), its
sets, the output folder it produced if one exists (`graphs_data/<group>/<name>`, or its own log
folder), and the words written about it (the builder's `why` and caption, the spec's header
comments). graphs_data is never walked: each spec's own folder is looked up, nothing else.

**The rules.** `experiments/expNN_<name>/relevant.yaml`, written by hand for the experiment:

    key_ops: [...]        the experiment's DEFINING operators, `op` or `op[model]`   +6 each
    ops_any: [...]        other operators that matter                                +3 each
    keywords: [...]       regex over the spec's name, path and text                  +1 each
    exclude: [...]        regex over the path (worktree copies, _archive, _superseded)
    score_min, max_per_kind, max_per_family (3), reserve_per_key_op (3)

A spec with an output earns +1. Each defining operator's best entries are kept first, over every
cap, so twenty variants of a richer spec cannot hide the one spec that runs it; at most three
entries of one family of variants are kept.

**The pool.** `tools/plexus_catalog.py pool NN` (or `pool --all`) rebuilds
`experiments/expNN_<name>/relevant_in_plexus/`: SYMLINKS ONLY, relative, through the repo's
`graphs_data` link, so nothing is copied --

    INDEX.md            the table, most relevant first: score, kind, name, operators and keywords
                        matched, outputs, and what the spec is (caption, why, header)
    index.json          the same, for an agent
    runs/<group>__<run>             -> graphs_data/<group>/<run>/   the spec as run, trajectory, movie
    specs/<group>__<name>.yaml      -> config/...                    the declared spec
    log/<path__joined>              -> log/<path>/                   an archived run with its metrics
    builder/<exp>__<NNNN>.<ext>     -> a builder step: spec, png, mp4, why, caption
    experiments/<exp>__<NNNN>.<ext> -> an experiment record step

The folder is DERIVED: a pool rebuild deletes it, so nothing is added to it by hand -- change
`relevant.yaml` and rebuild. Rebuild the catalog (`build`) after new runs land, then the pools.

**How it is used.** Stage 0's audit (`## The model -- the audit`) starts from the pool's INDEX.md:
every spec the experiment will run is either an entry of the pool, extended, or a new spec whose
markdown names the pool entries it could NOT be built from and why. The judge and the rulers read
pool entries as references and controls where they fit (exp 15's `atlas/turing2d_rps` IS the
rock-paper-scissors model the experiment measures). A pool entry is a candidate, not a verdict: the
score says it matched the rules, never that it is right -- it is audited like everything else.

---

## Explore within the framework -- coarse to fine, park a gate, pivot instead of stopping

**An experiment explores the framework its papers define; it does not grind one rig against one gate.** The
papers of an experiment describe a FIELD -- several phenomena, several models, several tissues -- and the gates
are a compass through it, not a finish line. The failure this section exists to prevent has happened several
times already: a session spends six to fifteen batches pushing one gate of one rig from 0.75 to 0.8 of its
points, never looks elsewhere, and ends "stopped, card below 8". The balance is explicit:

- **The gates stay honest.** Bands are never moved to pass, targets are never edited, numbers come from the
  paper's figure. Exploring changes WHERE the session looks, never what counts as a pass.
- **Coarse to fine.** First bring EVERY gate to a coarse level -- the phenomenon present and measured, at least
  half its points -- before refining any one of them. A card of 6 with every gate showing something beats a 7
  with one gate perfect and two never tried.
- **Park a gate.** A gate at or above 75 % of its points, or unmoved beyond its seed spread for two batches, is
  PARKED: recorded with its best value and why it is short, and no batch is spent on it alone until the others
  catch up. "Shown, not passed" is a result; write it as a finding and move on.
- **One exploration arm in every batch.** Of the (at most) four jobs of a batch, at least ONE is an exploration
  arm: a different mechanism for the same claim, a different rig or geometry, or a different phenomenon of the
  same papers. It carries its own single change from its own parent, so the one-change rule still holds. Every
  third batch may be entirely exploratory.
- **Stuck means pivot, not stop.** Three batches in a row that move the lowest gate by less than its seed spread
  do not end the experiment: they PARK that gate and open a new direction -- another mechanism, another rig,
  another claim of the same papers -- written as a new row of the md's direction log (`## Directions`: what was
  tried, the best card on it, why it was parked or kept). The experiment ends when the human says so, or when
  every direction the papers suggest has been tried coarsely.
- **Zoom out regularly.** Every three batches the session writes three lines under `## Directions`: what the
  papers' field asks, what the model shows so far, and at least two other ways the same biology could be shown
  that have not been tried.
- **The judge scores and points both ways.** Its report ends with TWO next steps: the one change most likely to
  raise the lowest gate (exploit) and one direction not yet tried (explore). The session runs both.

---

## Code: reuse the operator, add a variant, land it where it belongs

**No experiment duplicates an operator or edits one.** Before a line of code, in this order:

1. **Does an existing operator do it as is** -- a parameter, a mode, or a `model:` already registered?
   Look in the registry, not in memory: `plexus.models.registry.catalog_summary()` (every operator with
   its kind, family, models and implementations), `operators_by_family(...)`, `operators_of_kind(...)`,
   `get_contract(name)`; `library_operators.qmd`; and the experiment's pool (`relevant_in_plexus/`) for
   the specs that already use it.
2. **Else, a variant of an existing operator**, registered BESIDE it with `@register_operator("<op>", ...)`
   (`src/plexus/models/registry.py`) and selected in the spec:
   `model="<name>"` -- different biology under the same contract (`{op: <op>, model: <name>}`);
   `implementation="<name>"` -- the same biology computed differently (`{op: <op>, implementation: <name>}`).
   The default, and every spec already using it, stays untouched.
3. **Only if no existing contract fits, a new operator** -- with its `kind` (one of
   `plexus.models.base.KINDS`) and its `family`
   (`OPERATOR_FAMILIES`) declared, and a docstring that names the existing operators it could not be a
   variant of, and why.

**A variant before a new operator, always.** A new process is first read as a new `model:` or
`implementation:` of the operator whose contract it shares -- an excitable kinetics is a model of
`cell_chem_react`, a periodic lattice is a model of `radius_graph`, a typed potential is a model of
`pair_potential`. A brand-new operator name is the exception, justified in its docstring by the contract
no existing operator offers.

**NO NEW `*_ops.py` FILE. Every operator and every variant lands in the EXISTING module of the operator
it belongs to** -- the module where its base operator is registered (`grep -l '@register_operator("<op>"'
src/plexus/operators/*.py`); a genuinely new operator, in the existing module of its family. The folder
already has enough modules; one per operator or one per experiment is exactly the sprawl this rule
prevents. A new module needs the human's explicit approval. Never in `experiments/`, `tools/` or a
prototype folder, never a copy of an existing operator under a new name: one operator, many variants,
one place.

**Already sprawled, to fold back (found 2026-09-27)** -- nine modules the experiment sessions created,
nearly all holding correct VARIANTS but in new files. Each moves into the module of its base operator,
then its file is deleted (tests re-run before and after):

| new file | holds | goes into |
|---|---|---|
| `bm_coupling_ops.py` | `bm_seed`, `bm_contact` `[live]` / `bm_sense[live]` / `mesh_contact[basal_reaction]` / `cell_mechanics[apicobasal_contact, shape_contact]` | `membrane_ops.py` / `contact_ops.py` / `contact_ops.py` / `vertex_ops.py` |
| `excitable_ops.py` | `cell_chem_react[aliev_panfilov]`, `cell_chem_diffuse[closed_junctions]` | `diffusion_reaction.py` |
| `nutrient_ops.py` | `cell_chem_diffuse[steady_uptake]` | `diffusion_reaction.py` |
| `growth_feedback.py` | `cell_grow[stretch]` | `diffusion_reaction.py` |
| `crypt_ops.py` | `cell_mechanics[apicobasal_region]` | `vertex_ops.py` |
| `sheet_ops.py` | `cell_divide[adder_sheet]` | `vertex_ops.py` |
| `sorting_ops.py` | `pair_potential[typed]` / `junction_myosin[type_pair]` / `contact_retype` (new) | `channel_ops.py` / `junction_ops.py` / the family module of `contact_retype` |
| `sphere_confine.py` | `plate_confine[sphere]` | `contact_ops.py` |
| `colony_ops.py` | `seed_colony` (new) / `colony_grow` (new) | `seed_ops.py` (or a mode of `seed_positions`) / a model of `cell_divide` if its contract fits, else `vertex_ops.py` |

**Before writing a variant, check how the engine combines it** with the other operators writing the same
block -- the schedule's order, and whether their deltas add or one overwrites the other. A variant that
assumes the wrong combination is silently wrong.

**Every variant carries** a test in `tests/` -- an identity case where it must agree with the default and a
planted case where it must differ -- its reason in its docstring, and a line in the experiment markdown
naming it. **Never** change the default implementation's behaviour or a default parameter value, rename an
operator, or edit another experiment's variant.

---

## Gates, the judge and its tools -- one set per experiment

**What the judge scores lives in ONE file, `experiments/expNN_<name>/gates.yaml`**: every gate's
points (they sum to 10), the value it reads, its full-marks and zero lines, their `basis`
(identity, analytic, reference, literature, model choice) and `source`, and the caps. The
experiment markdown's `## The goal` and the judge's `AGENT_<role>_judge.md` QUOTE it; neither
restates a band of its own. **A band that must come from a paper and has not been read from its
figure is `TBD from <paper, figure>`; a TBD gate is reported as unset and a round with an unset gate
cannot pass.** The pass line is above 8 of 10 with no gate unset.

The tools, and every experiment's two markdowns (the experiment's and its judge's) name them:

| tool | what it does |
|---|---|
| `tools/exp_measures/expNN.py` | the experiment's RULERS: whole-run measures `expNN.<name>`, registered in `plexus.measures` (kind "run") and reading runs through its trajectory facade -- one module per experiment, each with its planted-input tests in `tests/test_exp_measures_expNN.py` |
| `tools/exp_measures/exp_shared.py` | measures every experiment reads: `shared.growth_audit` (exp 3's auditor, for the "wrecked" cap), `shared.cell_count` |
| `tools/exp_gate_score.py NN` | measures every run on disk (cached in `specs/expNN/measures.jsonl`), computes each gate's value, points (linear between the lines, rounded down to 0.25) and caps, appends the card to `specs/expNN/gate_scores.jsonl`; `--dry` lists the gates, which bands are unset, and which runs are on disk |
| `tools/exp_wait.py NN` | the ONE background waiter per batch; lands the batch when it exits (below) |
| `tools/exp_land.py NN` | a batch's landing: health of every run, card, `--check`, record, report; `--health <group>/<run>` for one run |
| `tools/exp_smoke.py <group>/<spec>` | a GPU smoke (100 frames, ~1-2 min) or `--full` long-run canary, with the run's health |
| `tools/exp_trap.py` | `list [word]` / `add` / `fix` on the shared `experiments/TRAPS.md` |

**A card is checked before it is believed.** A gate that reads a key its ruler does not return used to
score "no value", exactly like a run that has not landed -- exp13 lost two gates for hours to
`tension_ratio_rim` in gates.yaml against `s_tt_over_rr_rim` in the ruler, and its sessions could not tell.
The scorer now separates the two:

    PYTHONPATH=src:tools python tools/exp_gate_score.py <n> --check     after EVERY scoring; exit 1 on a problem

| check | means | the fix |
|---|---|---|
| UNKNOWN KEY | a gate or cap reads a key no run's output of that measure contains; the gate is INVALID and the card is marked CARD INVALID (it cannot pass) | correct the KEY NAME in gates.yaml or the ruler -- never the band; the check lists the ruler's closest keys |
| UNKNOWN MEASURE / UNKNOWN ARM | a key whose measure, or a value whose arm, gates.yaml does not declare | declare it, or correct the name |
| NOT ON ARM | the key exists on other arms' runs but on none of this arm's | usually cache entries from before the ruler gained the key: `--recompute`; else declare the measure on this arm |
| RULER ERROR | a measure raised on a run; the cache holds the error | fix the ruler, re-measure |
| STALE CARD | runs of the experiment landed after the last card | score them when their batch's waiter exits (expected while a batch is still open) |

The waiter (`tools/exp_wait.py`) LANDS the batch itself when it exits (`tools/exp_land.py`: health,
card, `--check`, record), so a batch is scored the minute it lands. A value is re-measured whenever its
run's trajectory is newer than the cached value or its ruler changed since (`CHANGED` in
`tools/exp_measures/common.py`) -- `--recompute` is no longer needed after a re-run. The judge runs `--check` before it reads the card and reports every problem under `checks`; a card
with an INVALID gate is fixed before it is judged, not judged as it stands.

`exp11_bm_hole_budding` is the worked example: `gates.yaml`, `AGENT_budding_judge.md`,
`tools/exp_measures/exp11.py` (which imports the archive's own bud metric rather than rewriting it),
`tests/test_exp_measures_exp11.py`.

## The reference paper -- downloaded FIRST, into the experiment's own folder

**An experiment that reproduces or tests a paper starts by downloading that paper into
`experiments/expNN_<name>/papers/`, before a single arm, spec, operator or judge is written.** A
figure cannot be reproduced from memory of it: exp 5 was designed and run for six rounds without
the PDF in the folder, put its anisotropy in the cells' junction tension, and the paper's own laser
ablation (Crest et al. 2017, Figure 1--figure supplement 1G) shows that tension is isotropic. All six
rounds were deleted.

What goes in the folder:

- the **PDF**, named `<FirstAuthor>_<year>_<journal>_<volume>_<id>_<short_title>.pdf`, and its
  **text** beside it (`.txt`, extracted with `fitz`; `pdftotext` is not installed);
- the **figure supplements and supplementary data** -- they are often where the control is (exp 5's
  ablation and no-epithelium controls are only in the supplements). eLife serves them at
  `https://cdn.elifesciences.org/articles/<id>/elife-<id>-fig<N>-figsupp<M>-v2.jpg`;
- the **figure pages rendered** to `papers/figs/` (`page.get_pixmap(dpi=...)`), zoomed crops of any
  panel whose numbers the experiment uses;
- when the publisher blocks the PDF, the full text from NCBI E-utilities
  (`efetch.fcgi?db=pmc&id=<PMCID>&rettype=xml`) -- Europe PMC and the NCBI PDF link are blocked
  from the devcontainer.

**Every number the markdown takes from a paper names its figure and panel**, read from the files in
`papers/` (a table of them opens the markdown). A number no panel backs is marked `model choice`.
The judge reads its targets from that table, never from the session's memory of the paper.

---

## The results table -- every experiment has one, and it is kept current

**Every experiment markdown has a `## Results` section with a table, one row per iteration (a
search) or per arm (a grid), appended as soon as the row's run is scored.** It is the first thing a
reader opens and the one place the whole campaign can be read at a glance; a result that lives only
in a JSON file, a chat or a finding paragraph has not been recorded. The shape is exp01's:

    | v | spec | agent | ruler <quantity> | verdict | what changed | what the result said |

- **Search experiments** (`branch: simulation`): step 4 of the loop writes the row by hand, after
  EVERY iteration -- the agent's score, the ruler's number, the verdict, the one change, and the
  diagnosis that decided the next change.
- **Grid experiments** (`branch: grid`): the table is DERIVED, never typed. Declare what it shows in
  the front matter and regenerate it after every scored run:

  ```yaml
  results:
    source: elongation            # experiments/specs/expNN/<source>.jsonl, one JSON line per run
    columns: {AR_late: "AR, late window (seeds)", angle_late_deg: "axis angle, deg (seeds)"}
  rounds:                         # optional: arm-name prefix per round, and the judge's round score
    4: {prefix: hp_, judge: 5.0}
  ```

  ```bash
  PYTHONPATH=src python tools/exp_results_table.py <n>     # rewrites the block between
                                                           # <!-- RESULTS --> and <!-- /RESULTS -->
  ```

  One row per arm, rounds in declaration order: runs, the agent (the round judge's score and the
  growth auditor per seed), each ruler column as the seed mean with the seeds listed, the verdict
  (`wrecked` if any seed scored below 2 on the growth auditor, else the round's judged score), and
  the arm's change. A scorer that lands runs must call it (exp 3's `signature_table.py` is the same
  rule for its signatures).

Prose goes under `## Findings`: what the rows MEAN, numbered, with the numbers. The table says what
happened; the findings say why.

---

## Render after the full run -- every colour scale from the whole simulation or rollout

**A movie is drawn only after the whole simulation, or the trained model's whole rollout, has run, and every colour
scale, axis range and LUT limit in it is taken from that complete run before the first frame is drawn.** A limit
fixed in the code (a drive bar topped at 1, an input box divided by 1.5) or read from the first frames saturates the
panel as soon as the run leaves that range, and a saturated panel shows nothing: exp24 step 0014 (2026-10-13) drew
muscle drives of 1.3-1.7 against a fixed top of 1 and rates near +-1 through a LUT clipped at the 90th percentile, so
every panel read as solid colour. The rule:

- run first, render second: the full trajectory or rollout is in memory (or on disk) when the renderer starts;
- each scale is a statistic of the WHOLE run (its max, or a high percentile such as p99.5 when a few outliers would
  flatten the rest), computed once and fixed for all frames, so two frames are comparable;
- two movies meant to be compared side by side (a run and its twin) are rendered with the same rule, and placed as
  consecutive steps of the record so the watcher compares them in one click.

## The record folder, and the watcher

Every experiment has a folder beside its markdown, laid out exactly like a builder record
(`builder/exp_NN_<subject>/`), so the session watcher shows it unchanged:

```
experiments/expNN_<name>/
  png/NNNN.png      -> ../../../graphs_data/<group>/<run>/3d.png      (symlink, the run's still)
  mp4/NNNN.mp4      -> ../../../graphs_data/<group>/<run>/movie.mp4   (symlink)
  spec/NNNN.yaml    -> ../../../graphs_data/<group>/<run>/spec.yaml   (symlink, the spec AS RUN)
  why/NNNN.txt      the results-table row of that iteration, one `column: value` per line
  caption/NNNN.txt  the run's VLM description (graphs_data/video_descriptions.txt), or a local one
  journal.txt       one line per step: iteration, run, verdict, what changed
```

**Step NNNN is the Nth row of the results table**, in table order -- `v7a` and `v7b` are two steps.
For a **grid** experiment the steps are the launch list instead -- every arm x grid point, in the
order `tools/exp.py launch` derives them -- and each step's `why` carries the arm's patch and its row
of the markdown's prediction table (the table whose first column names arms in backticks). Its
`caption` is the trajectory readout, `specs/expNN/audit.jsonl` (the growth auditor,
`experiments/exp03_size_hypotheses/AGENT_growth_audit.md`) and `specs/expNN/signatures.jsonl`, once computed: no VLM is asked.
The links go through the repo's `graphs_data` link and are relative, so nothing is copied and the
folder reads the same on any mount. The folder is **derived**: rebuild it after every new table row
(it removes steps no row backs any more, and keeps captions already written):

```bash
PYTHONPATH=src python tools/exp_record.py <n> [<n> ...]
PYTHONPATH=src python tools/exp_record.py <n> --caption-missing   # local VLM, one run at a time
```

Runs on `gpu_l4` do not always leave a description in `video_descriptions.txt`; `--caption-missing`
fills those with the local VLM. It is the one step here that uses a local GPU.

**The watcher** is the GUI server's `/watch` page, pointed at the folder by `PLEXUS_BUILDER`, one
server per folder, on a port no other session uses (8799, 8830 and 8831 are taken by the builder
sessions):

```bash
PLEXUS_BUILDER=/workspace/Plexus/experiments/expNN_<name> \
  python Plexus_gui.py --host 0.0.0.0 --port <port> --no-browser --tab material
"$BROWSER" http://127.0.0.1:<port>/watch      # VS Code must forward the port to the browser
```

The page re-reads the folder, so a rebuild after a new iteration shows up without a restart; a
change to the watcher's own code (`src/plexus/gui/watch.py`, `tools/spec_summary.py`) needs one.

---

## What is different here from connectome-gnn's `experiments/`

That folder is the ancestor of this one and the design is taken from it whole:
one markdown per experiment, the number as the handle, run names DERIVED from a
pattern so the prose and the disk cannot drift, a STATUS block the tool owns,
`--dry-run` before thirty jobs, and the one rule below. Three things genuinely
differ, and they are the reason this is not a copy:

**PLEXUS HAS TWO VERBS.** connectome-gnn only trains. Here an arm is either a
SIMULATION -- `Plexus_Main.py -o generate <group>/<spec>`, landing in
`graphs_data/<group>/<name>/` -- or a FIT -- `plexus.tasks.spec_trainer` or
`.trainer` on a run spec, landing in `log/task/<run>/`. An arm says which with
`kind: generate` or `kind: fit`, and the two are polled from different trees.

**THREE SPEC LANGUAGES, not one.** A forward spec (`config/<group>/`) says what
the world is; a task spec (`config/task/`) says what the data is; a run spec
(`config/run/`) pairs a model with a task and carries the hyperparameters. An
arm's `differs_by` names which of the three it patches.

**LOCAL IS A REAL OPTION.** The devcontainer has two A6000s, so "submit it" is a
question and not a given. `where: local` runs in the devcontainer and `where: l4`
submits to `gpu_l4`; an experiment can put both on one axis and measure the
difference rather than assume it.

---

## Two branches

An experiment is one of two kinds, and the markdown says which with `branch:`.

**`branch: grid`** — a fixed grid of specs launched at once and read back: the
question is known, the arms are known, the job is to measure. Cost-and-repeat
(exp 0) and the multi-function ladder are this kind. `tools/exp.py` launches,
polls and reports it; the verbs are below.

**`branch: simulation`** — a SEARCH. There is a target the simulation should
reach and no spec yet that reaches it, so the experiment is a sequence of specs,
each written after reading the last one's result. It cannot be launched as a
grid because spec N+1 does not exist until spec N has been judged.

### The simulation loop

```
0  once per search           python VLLM/vlm_server.py --device cuda:1 &
1  write a new spec          config/<group>/expNN_vK.yaml   (one change from vK-1)
2  run the engine            python Plexus_Main.py -o generate <group>/expNN_vK
3  judge it against target   python tools/judge.py <group>/expNN_vK --target "..." --measure ...
4  record it in the md       a new row in the RESULTS table, and why
5  rebuild the record folder  python tools/exp_record.py NN     (the watcher shows it)
6  go back to 1
```

**Step 0 keeps the VLM loaded.** Both the caption pass inside step 2 and the judge in
step 3 ask the same 12B model; without the server each loads its own copy, and the
two loads were most of an iteration. Both find the server on its own
(`VLLM/vlm_client.py`, `PLEXUS_VLM_URL`, default port 8766) and fall back to loading
when it is not running, so it is a speed-up and never a dependency. With it, exp 1
v2 captioned in 13 s and was scored in 3 s.

**The score** is the external agent's (the local VLM, `tools/judge.py`), 0-10:

| score | meaning |
|---|---|
| 0-4 | out of scope -- the target's subject is not in the picture |
| 5-8 | the right direction, not acceptable |
| 9-10 | acceptable |

**The markdown COLLECTS THE RESULTS.** Every iteration is a row: the spec name,
the agent's score, the ruler's measured value, the verdict, WHAT CHANGED from the
previous spec, and WHAT THE RESULT SAID -- the diagnosis that decided the next
change. Nothing lives only in the session: a search whose record is a chat log
cannot be resumed, audited, or shown. The table is written after EVERY iteration,
not at the end.

**The agent scores, the ruler decides.** A VLM score alone is soft -- it reads a
picture, and a picture can be right for the wrong reason. `--measure` computes the
quantity the target is actually about from the trajectory, and the verdict is the
RULER'S. Where they disagree, that is recorded as a finding, not resolved in the
agent's favour. Exp 1 v4 is the example: agent 10 ("large coarse circles on the
left"), ruler 1.34x against a 2x target -- the agent was reading sparser spacing
as bigger cells, and the spec was not done.

**One change per iteration.** Two changes at once make a better score unreadable:
it is not known which of them earned it, and the next change is a guess. The
"what changed" column is how this is checked.

**The target is written ONCE, at the top, before the first spec, and never edited.**
Moving it to meet the result is the one way this loop can lie.

**Stop only at a pass or on the human's word.** When three consecutive iterations move the ruler by less than
its own run-to-run spread the search is stuck on that path: PARK it and pivot to another mechanism or rig, as
"Explore within the framework" says -- a stuck path is a finding, not the end of the experiment.

---

## The main question -- asked at every landing, answered before the next batch

**Each experiment declares ONE main question in its front matter (`question:`)**: how its phenomenon emerges
in the best run so far, and how that differs from the paper's mechanism -- e.g. exp11: "How does the bud
emerge in the best run, and how does that differ from Wang 2021's mechanism?". Asked exactly that, exp11's
session answered with the causal chain of its own rig (the inner mass's volume clock pushes, the membrane
holds everywhere but the hole, so the growth escapes there), saw that this is not Wang's mechanism (a surface
layer that outgrows its interior by re-insertion and folds), and wrote a five-change structural plan, each
change named at the coarse level and sketched in YAML, that it had never proposed on its own.

- **At every landing** the report ends with the question (`tools/exp_land.py`).
- **Before the next batch** the session answers it in the md, as a new dated line under `## The question`:
  `- YYYY-MM-DD HH:MM -- <the answer, >= 25 words>`, written from the best run's movie and numbers: first how
  the phenomenon happens in the model (the causal chain), then how that differs from the paper's mechanism.
- **The next batch follows from the answer**: its structural or exploration arm is the change the answer
  points to, named in its row's "what changed".
- `tools/exp.py launch` refuses a batch when the md declares a question and has no answer newer than the
  last landing -- so the question cannot be skipped, but answering it takes a minute.

## The pace -- ten minutes between batches, one hour at most on anything

**The GPUs wait on the session, never the reverse.** The cluster is not the bottleneck (queue ~0, runs
1-5 min); the sessions' own gaps between batches were (exp10: 629 min, exp14: 345 min). The rules:

- **At most 10 minutes from a batch landing to the next batch submitted.** The landing report prints
  the time it is due and, from the second landing on, the gap the previous batch waited
  ("Gap between batches ... OVER the 10-minute limit"). The next batch is PREPARED while the current one
  runs (its specs, its one change, its exploration arm), so landing -> submit is: read the report, write
  the rows, adjust the one change, submit.
- **No rabbit hole over one hour.** A diagnosis, a build or a paper hunt that has not produced a
  submitted batch within an hour stops: what is known is written as a finding, the question becomes an
  ARM of the next batch (the run answers it) or a `## Directions` row, and the best available arms are
  submitted. A long build runs in a background subagent WHILE a batch of the existing rig runs.
- **A question to a run beats a question to the code.** "Does gamma 5 go unstable?" is a 2-minute GPU
  smoke, not 30 minutes of reading the integrator.

## No questions to the human -- decide, log, continue

**A session never waits on the human.** Where the papers, the gates or the code leave a choice open (a
band's definition, a ruler's side, a waiver, which direction next, a missing paper number), the session
DECIDES: it takes the option best supported by the papers and the model's physics, writes it in the
md's `## Decisions` (date, the choice, the options it had, why -- one line each) and carries on the same
minute. The human reads that list and overrules when they disagree; an overruled decision is a new row,
not an edit. Things that are not the session's to decide are not asked either: they are simply not done
(moving a band to pass, a new `*_ops.py` module, committing, touching another experiment's files).
A paper or dataset that is not on disk is replaced by the nearest one that is, and the gate says so.

**One exception: a rung** (next section but one). A rung the human has put under a score is walked by the session alone, step after step, and the session asks the human ONCE, at the end of the rung, for the go / no-go that closes it.

## Rungs and the score -- one step at a time, scored 0-10 by the session, closed by the human

When an experiment has lost its way (a gate that failed and was never re-established; features added on a layer that does not work; a null result nobody looked at), the human puts it on a
LADDER of rungs. (First used for exp27, 2026-10-21; Cedric's rules.)

- **A rung is named by what it must achieve**, not by a counter: "R1 = an intact outer layer under the membrane". Its steps are R1a, R1b, R1c ... until the rung is closed; the next rung opens only
  when the human closes this one. ONE run per step. Every step is archived to the watcher with: the **watcher number** (the watcher counts steps by POSITION, so a deleted step moves every later
  number by one: give the number the watcher shows, never the file name), the **score**, and a **short list of what to look at**.
- **ONE job in flight at a time, inside a rung.** The next run is submitted only after the previous one has landed, its snapshot has been read and scored, and the single next change has been argued from
  that reading (Cedric: running one job at a time forces the right choice and better thinking). A step is one run plus its render; no parallel arms, no waiting batch.
- **Before the first step** the session writes the rung's SOUNDNESS CHECKS in the md: each a ruler quantity with a band (for example detached point clusters per cell under 0.1) plus what the
  snapshot must show and must not show. The checks are not edited after a step has been scored with them.
- **The score of a step, 0 to 10, is the session's:**
  1. If any check fails, or anything in the snapshot is unsatisfying, the score is **below 6** (how far below: how many checks fail and how badly).
  2. If every check passes and the snapshot looks sound, the score is **6**. Six is "sound", not "good".
  3. Above 6, the score is the **progress toward the experiment's gate** that the rung serves (a ruler quantity with a band, in the md's gate table): 6 + 4 x the fraction of the way from the
     sound baseline to the gate; **10 means the gate is met**.
  4. **Anything weird sends the score back to 6** (or below, if a check fails): a feature the session does not understand, a ruler that disagrees with the snapshot, an artefact. The weird thing
     is the subject of the next step. A score above 6 is lost the moment something weird shows.
  5. The session READS THE SNAPSHOT ITSELF (the png of the step) before it scores, and the first line of the step's text is its own verdict ("my score 2 of 10: ragged layer, ring of empty
     space under it"). A score is never the ruler's alone, and the human is never the one who finds the defect: a defect the session saw and did not name is a bug in the session.
- **Six is only "looks good": it is not the gate.** Every rung states its GATE in the md before its first step (a ruler quantity or a staircase of them, with bands) and the score above 6 is the
  distance walked toward it. A rung with no gate beyond "looks sound" is not a rung.
- **The session continues alone**, one step after the other, choosing the next single change from the score and the diagnosis, and asks the human ONLY when the gate is passed: **9 or more on two
  consecutive steps**. Then it asks ONCE: the watcher number to look at and what to check. The human gives the go / no-go; **only the human closes a rung**. Six, however many times in a row, never
  triggers a question.
- **The session stops and asks earlier** when the score falls on two consecutive steps, after 8 steps without readiness, or when a step needs something outside the rung (a new module, a changed
  gate, a changed check).
- The score and the reason for it are a column of the md's results table. Tool for exp27: `tools/exp27_score.py`.

## Smoke on the GPU -- `tools/exp_smoke.py`, never a CPU smoke of more than a few frames

```bash
PYTHONPATH=src:tools python tools/exp_smoke.py <group>/<spec> [--frames 100] [--set "operators[op].key=v"] [--exp NN]
PYTHONPATH=src:tools python tools/exp_smoke.py <group>/<spec> --full        # the long-run canary, in the background
```

Copies the spec as `smoke_<name>` (`canary_<name>` with `--full`), cuts it to `--frames`, applies `--set`
overrides (the `differs_by` addressing), submits it to the L4 queue (wall 30 min), waits and prints the
run's health; `--exp NN` also runs the experiment's rulers on it and checks every key its gates read.
20 frames of exp11's rig: 33 s from submission to health. A 200-frame CPU smoke took 20-30
min (exp13) and runs another code path than CUDA (`TRAPS.md` T-20). `--full` runs the spec at its own
length before a long batch: defects that appear late (T-09) cost one canary, not a round. Smoke and
canary outputs are kept out of the record and the scoring by their prefix.

## Waiting for a batch -- ONE background waiter, which also LANDS it

**After a batch is submitted, the session starts ONE waiter and stops looking:**

```bash
PYTHONPATH=src:tools python tools/exp_wait.py <n>            # in the background (the shell's run_in_background)
PYTHONPATH=src:tools python tools/exp_wait.py <n> --spec tissue/exp10_c2_s1 ...    # runs outside the md's arms
```

`tools/exp_wait.py` sleeps between checks without the model, reads only the batch's own run folders
(landed: the trajectory reads and the movie exists; died: `cluster.out` carries "Exited with exit code
N" or an LSF `TERM_` reason), and exits ONCE when every run has landed or died, or at its timeout (6 h):
exit 0 all landed, 1 a run died, 2 timeout. It never touches the cluster or a login node.

**Before it exits it LANDS the batch** (`tools/exp_land.py`, skipped with `--no-land`):

| step | what | why |
|---|---|---|
| health | per run: every set's count and extent first -> last; NON-FINITE values in EVERY float block of EVERY set, with the first frame; the growth audit (jump and finiteness on every frame); STATIC MOVIE when the movie's pixels do not move while the state does | a NaN membrane under a finite tissue, a budding runaway, a movie drawn in the wrong colour -- each went unseen for hours |
| card | `tools/exp_gate_score.py <n>`, then `--check` | the batch arrives scored and checked |
| record | `tools/exp_record.py <n>` | the watcher shows the batch |
| report | `expNN_<name>/landings/<date>_<time>.md`, the same text between `<!-- LANDING -->` and `<!-- /LANDING -->` in the md (tool-owned, like STATUS), and the waiter's output | the session wakes to one report, and the time the next batch is due |

The session then: looks at every FLAGGED run's movie (and the others' stills), writes the results rows and
findings from the report, adjusts the prepared next batch, submits -- within ten minutes.
`tools/exp_land.py --health <group>/<run>` prints one run's health without scoring or writing anything.

**What is NOT done while a batch runs:** `exp.py poll` in a loop, a `sleep` loop in the foreground, a
scheduled wake-up every few minutes, or "is it done?" checks by hand. Between the submission and the
waiter's exit the session PREPARES THE NEXT BATCH (its specs, a ruler, a GPU smoke) or ends its turn.

---

## The verbs (grid branch)

```bash
PYTHONPATH=src python tools/exp.py launch  <n> [--arm A] [--where AXIS=V1,V2] [--dry-run]
PYTHONPATH=src python tools/exp.py poll    [n...]      # rewrite the STATUS blocks
PYTHONPATH=src python tools/exp.py report  [n...]      # experiments/report.md
```

`launch` writes one spec per grid point into `experiments/specs/expNN/`, then
submits one job each -- `bsub` for `where: l4`, a detached local process for
`where: local` -- and records the job ids into the markdown's front matter.

`poll` reads the output trees and rewrites the `<!-- STATUS -->` block inside
each markdown, plus `INDEX.md`. Run it before reading any table.

Always `--dry-run` first.

---

## The experiment markdown

YAML front matter the tool reads, then prose for humans, then a STATUS block the
tool owns.

```yaml
number: 0
name: cost_and_repeat
title: What a run costs, and whether it repeats
purpose: ...                                # the QUESTION, printed under the table
queue: gpu_l4
wall: '240'
papers:                                     # the reference paper(s), in expNN_<name>/papers/
- exp00_cost_and_repeat/papers/<FirstAuthor>_<year>_<journal>_<id>_<title>.pdf
axes:                                       # the grid
  where:  [local, l4]
  repeat: [r0, r1]
arms:
- id: generate
  label: one forward spec through the engine
  kind: generate                            # or: fit
  spec: demo/a2_4000_cells_flat             # config/<group>/<name>
  run_pattern: exp00_gen_{where}_{repeat}
  differs_by: {}                            # dotted keys patched into the spec
- id: fit
  kind: fit
  spec: m1_integrate_ctrnn64                # config/run/<name>
  run_pattern: exp00_fit_{where}_{repeat}
  differs_by: {training.epochs: 20}
job_ids: {}                                 # written by launch
```

Run names are derived from `run_pattern` and the grid point, never stored.

---

## The one rule

**Nothing in the report is typed. Every number and every label is derived from
the run that produced it.**

This is inherited rather than invented, and it was not free: in connectome-gnn it
broke three times and each break produced a wrong conclusion that survived until
someone happened to look -- two column headers typed the wrong way round so every
table read backwards, a speed table typed from memory whose ranking was inverted,
and a mean taken across folds read at whatever iteration each had reached.

If a label can differ between runs, derive it and put a test on the derivation.

---

## Reading a table without being fooled

- **A constant across a sweep is a rail, not a result.** If a column is the same
  to three digits across independent runs, look for the clamp.
- **A wall-clock is LSF's or the engine's, never a stopwatch.** `frame_ms` in the
  trajectory is one reading per tick; `cluster.out` carries LSF's own `Run time`.
  Where they disagree the job's is the outer number and includes the queue.
- **Two repeats of one spec are not expected to be bit-identical on CUDA.** MPM's
  scatter is `index_add`, whose CUDA form is atomic and reorders; the cardiac
  prototype measures itself disagreeing with itself by 6e-07 in float32 and
  1.3e-15 in float64. A repeat axis measures THAT floor, and any arm difference
  smaller than it is not a difference.
- A pooled normalised error can move the wrong way when a louder condition joins
  the pool. Per-cell against its own variance is the honest ratio.

---

## The environment

**Never run a process on a login node.** `python`, `conda`, `mamba`, `cp` and
`rsync` are watched there; admins kill them and email Cedric. Everything on the
cluster is a `bsub` job. `/groups` is mounted in the devcontainer, so staging a
spec is a local copy and nothing of ours runs on `login1`.

**IMPORTANT -- THE CLUSTER IS FOR TRAINING, TESTING AND PLOTTING RESULTS, NOTHING ELSE** (Cedric,
2026-09-30). A training experiment sends to the cluster ONLY

    python Plexus_Main.py -o train_test_plot <training spec>

No export, baseline, statistics, download, render, ruler or slide job goes to the cluster: those run in
the devcontainer (check the local A6000s' free memory first). **A paper, code or dataset that is not on
disk is ASKED FOR from Cedric, never downloaded by the session.** (A forward-simulation experiment keeps
its own verb, `-o generate`, under the same rule: the cluster runs the model, nothing around it.) The GRAPH PHASE of a
trace run (`-o graph <run>`, `-o test_plot_graph <run>`; `src/plexus/tasks/graph_analysis.py`, 2026-10-10: is the
learned graph doing something, what and how -- results/<run>_graph.json and results/graph/) is an analysis: it runs
locally, after the cluster's train_test_plot.

| | |
|---|---|
| python (devcontainer) | `/workspace/.conda_envs/neural-graph-linux/bin/python` — plain `python` is /opt/conda base with no torch |
| python (cluster job) | relative, after a `cd`: `python Plexus_Main.py -o generate <group>/<spec>` |
| path translation | `discovery_okuda/cluster.py::cpath` — `/workspace/Plexus` is a different string there |
| ssh | `ssh allierc@login1` — bare `login1` resolves to `node@` and is denied |
| generate output | `GraphData/graphs_data/<group>/<name>/` |
| fit output | `GraphData/log/task/<run>/` |
| queues | `gpu_l4`, `gpu_rtx6000`, `gpu_a100` |
| git push | `--no-verify`, always — git-lfs is missing in the devcontainer |

---

## Traps

**The shared list is `experiments/TRAPS.md`**: engine, renderer and tool behaviours that cost a session
time, written by every session. Read it when a session starts and before a new rig
(`python tools/exp_trap.py list [word]`); ADD a trap the minute one is found
(`python tools/exp_trap.py add --by expNN --title ... --symptom ... --cause ... --avoid ...`, appended under
a lock, so ten sessions can write at once). The lines below are this folder's own tool traps.

- **LSF memory is 20 GB per slot**, so `n_cpus` is how an experiment asks for
  RAM and it has nothing to do with the GPU.
- **`launch` clears a run directory before submitting.** Read the failure reason
  BEFORE relaunching, or it is gone.
- `launch --arm X` submits a subset; the job-id record is MERGED, not replaced.
- A killed job and a slow one look identical from the training logs. Check
  `cluster.out` for the LSF reason.
- `Plexus_Main.py` needs `<group>/<name>`, not a bare name: `add_pre_folder`
  classifies a bare name by keyword and an unrecognised one is refused.
- A generate job with `--no-describe` lands with a trajectory and no movie.
  Plexus generation ALWAYS captions; do not pass it.


## The watcher holds movies of the cell systems -- no figures, no cards (Cedric, 2026-11-07)

A step of an experiment's record is a MOVIE OF THE CELL SYSTEM it is about (the render of the cells, the junction, the hierarchy, a vesicle cut at its equator): the thing the run simulated, moving. **Never file a video of a figure or of a card (a 2-second clip of a still picture), and never file a figure or a card (a gate card, a curve plot, a bar chart, a table picture) as a step, unless Cedric asks for it.** Gate values, curves and tables go into the step's why text, the experiment's markdown and its `data/` folder, where they can be read without being played. On 2026-11-07 exp38 filed eight such steps (a gate card four times, three figures of curves, a fit picture); Cedric: "I expected different movies of the cell systems along the progress and I get curves that are not what I wanted"; all eight were deleted. When a result has no movie yet, render one (tools/exp38_junction_movie.py for a two-cell or doublet run, tools/exp38_vesicle_movie.py for a vesicle) rather than file a picture of its numbers.
