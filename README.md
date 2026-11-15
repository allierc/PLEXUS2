# Plexus

**A language for saying what a biological mechanism *is*, precisely enough to run it.**

A biologist explains a tissue with parts, arrows, and boxes inside boxes: these cells, pulling
on each other, in this signal, dividing at this rate. That sketch is a hypothesis. It is also,
usually, where the explanation stops — because turning it into something that runs means
writing a simulator, and the simulator is a different artefact from the claim it was meant to
test. Plexus removes that gap. The sketch *is* the program.

## The idea

A model states a hypothesis in two parts, and both are declared rather than coded.

**What exists** — the entities, the state each carries, the relations between them, the
continuous fields they live in.

**What happens** — the activities, one per mechanism: adhesion, diffusion, division, death,
contraction, a beating cilium. Each names the entities, state and relations it needs, and
nothing else.

The *organization* is declared too, and that is the part a list of equations loses: where the
parts sit, what order the activities run in and at what rates, and how a level is made of the
level beneath it — with information carried up to the whole and influence pushed back down.

Everything then falls into four kinds of object. **Entities**, grouped into sets by what kind
of thing they are. **Fields**, the continua they move through. **Operators**, the activities,
each belonging to one of a handful of structural kinds according to *what it moves and where
to*. And a **schedule**, which is the organization in time.

One interpreter runs all of it. The same machine runs an epithelium as a vertex model, a
continuum as a material-point method, a tissue growing into a gel, a reaction–diffusion
pattern, a swimming larva, a beating cilium, and a pair of colliding galaxies — because none
of those is a program here. Each is a spec.

## Why it is built this way

**A vocabulary is a cost.** Every operator is a word, and two words that mean almost the same
thing make a reader compare their definitions to find out which one a model is getting. So the
library grows by *merging*, not by adding variants — and shrinks when it can. Perfection is
when there is nothing left to remove.

**It is differentiable.** A hypothesis you can run is useful; a hypothesis you can *fit* is a
different thing. Because the forward model is built from differentiable operators, observations
can be turned back into the parameters, the geometry, or the schedule that would produce them —
and a residual that will not go away points at a missing activity rather than a wrong number.

**Nothing is claimed that is not measured.** A model that renders beautifully and conserves
nothing is not a weaker result, it is not a result. Invariants — momentum, mass, the sampling a
numerical method actually requires — are checked and reported, because every one of them has
already been found broken behind a picture that looked fine.

## What is in this repository

The engine, the operator library, the record viewer and the experiments of the talk
*PLEXUS: differentiable tissue computing* (November 2026):

| experiment | what it shows | slides |
|---|---|---|
| `experiments/composition` | Gray-Scott Turing pattern, cell growth and division, and both composed on a growing shell (3D vertex model) | 11-14 |
| `experiments/exp01_minisite` | material point method (MPM) demos: elastic, plastic and liquid bodies, a 100-million-point bench | 18 |
| `experiments/exp29_cell_tissue_2` | cells made of material points held by junctions; a sweep of the junction parameters | 19-22 |
| `experiments/exp09_molecular_landscape` | a crowded molecular landscape, every molecule an MPM body | 23 |
| `experiments/exp05_spheroid_from_words` | a biologist's three-sentence description run step by step: spheroids, signalling, tissue topology | 25-33 |

Each experiment is a record: one numbered step per run, with its spec (`spec/`), the reason it
was run (`why/`), its movie (`mp4/`) and still (`png/`), and the panes the viewer draws
(`story/`). The notes `experiments/<name>.md` hold each experiment's decisions and findings.
Every spec of a record is also in `config/` under the run's name.

| folder | content |
|---|---|
| `src/plexus/` | the language: schema, engine, operators, renderers, the viewer (`gui/`) |
| `src/plexus/operators/` | the operator library, its table in `README.md` |
| `config/` | the specs of every recorded run (`si_material/`, `tissue/`) |
| `tools/` | the record tools (`gui_drive`, `archive_run`, `view_movie`, `watcher_deck`), the studio renderers (`exp23_studio`, `exp29_studio`, `cell_panels`), and the experiment tools of exp05, exp09 and exp29 |
| `paper/` | the paper describing the language, `plexus2.pdf` |
| `VLLM/` | the movie captioning (needs the Gemma weights, not included; a run without them simply has no caption) |

## Running one

```bash
python Plexus_Main.py -o generate tissue/exp05_spheroid_v01_f1000 --output_root .
python Plexus_Main.py -o generate si_material/exp01_three_materials_r2 --output_root .
```

A run writes its trajectory, its movie (`movie.mp4`) and its stills to
`<output_root>/graphs_data/<folder>/<name>/`, and a log to `<output_root>/log/<folder>/<name>/`;
both folders are ignored by git. `--device cuda:0` is the default, `--device cpu` works for the
small runs. The environment needs Python 3.10+, PyTorch with CUDA, NVIDIA Warp, PyVista / VTK,
NumPy, SciPy and PyYAML.

## Watching a record

```bash
PLEXUS_BUILDER=experiments/exp05_spheroid_from_words python Plexus_gui.py --port 8799
# then open http://127.0.0.1:8799/watch?rec=experiments/exp05_spheroid_from_words
```

The watcher plays each step's movie beside its spec, the operators it composes (the `plexus`
pane), their equations (`math`) and the reason the step was run.
