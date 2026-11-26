# PLEXUS

A differentiable cell tissue computation

![exp34_flat_vesicle](exp34_flat_vesicle.gif)

## Installation

```bash
conda env create -f envs/environment.linux.yaml   # or envs/environment.mac.yaml
conda activate plexus
```

`environment.linux.yaml` is the environment the runs were made in (NVIDIA GPU, CUDA 12.8); `environment.mac.yaml` has the same packages unpinned and runs the watcher and the specs, not the Warp scenes of exp09 and exp10.

## Run

```bash
python Plexus_Main.py -o generate tissue/exp34_ucb_flat_e000_h150_e07 --output_root <data root>   # run one spec of config/
```

Open the watcher in a browser with `python Plexus_gui.py` (it opens http://127.0.0.1:8799/watch); it shows the records of `experiments/`, one at a time (`?rec=experiments/<name>`).

![watcher](watcher.png)

To start a new experiment, follow [`experiments/INSTRUCTION.md`](experiments/INSTRUCTION.md).

PLEXUS

A differentiable cell tissue computation

!exp34_flat_vesicle

Mathematical formalism

PLEXUS represents a cell tissue as a dynamical system composed of graphs, states, coupling relationships, and differentiable functions.

We define the system at time t as:

D(t)=(G(t),X(t),F,C(t))

where:

G(t)={G1(t),…,Gn(t)} is a collection of graphs.
X(t) is the collection of states associated with graph elements.
F={f1,…,fm} is a collection of functions governing the system dynamics.
C(t) describes the coupling relationships between graphs.
Graph coupling

Graphs can be coupled through three types of relationships:

Cross-graph edges: Edges connecting nodes belonging to different graphs.
Shared nodes: Nodes belonging to multiple graphs.
Supernodes: Nodes containing an entire graph or subgraph.

We can therefore write:

C(t)=(CE(t),CV(t),CH(t))

where the three components represent cross-graph edges, shared nodes, and hierarchical relationships, respectively.

Differentiable functions

Each function acts on subsets of states, graphs, and coupling relationships, and produces a state increment:

ΔXi=fi(Xi,Gi,Ci;θi)

where:

Xi⊆X is the subset of states accessed by the function.
Gi⊆G is the subset of graphs involved.
Ci⊆C is the subset of coupling relationships involved.
θi denotes the learnable parameters of the function.

The global state increment is obtained by summing the contributions of all functions:

ΔX=∑i=1mΔXi

The system evolves through Euler integration:

X(t+Δt)=X(t)+Δt∑i=1mfi(Xi,Gi,Ci;θi)

If the individual functions are differentiable, their sum and the resulting Euler evolution map are also differentiable, provided the graph topology and coupling structure are fixed during the differentiation step.

This allows the learnable parameters of individual functions to be optimized jointly against target trajectories or energy functions using gradient-based optimization.

Dynamic topology and cell lifecycle

Graph structures and coupling relationships can evolve during simulation.

In particular, cells are represented as supernodes containing graphs of material points or vertices. Cell growth, division, and death can change graph membership and topology without changing the dimensions of the underlying state tensors.

This is achieved using a fixed reservoir of material points or vertices, with active elements changing over time.

Cell division, for example, partitions the parent cell's material points between its two daughter cells, preserving the underlying state tensor dimensions.

Thus, PLEXUS combines differentiable continuous dynamics with discrete structural changes, enabling the simulation and learning of evolving cell tissues.

Installation
bash
conda env create -f envs/environment.linux.yaml   # or envs/environment.mac.yaml
conda activate plexus


environment.linux.yaml is the environment the runs were made in (NVIDIA GPU, CUDA 12.8); environment.mac.yaml has the same packages unpinned and runs the watcher and the specs, not the Warp scenes of exp09 and exp10.

Run
bash
python Plexus_Main.py -o generate tissue/exp34_ucb_flat_e000_h150_e07 --output_root <data root>   # run one spec of config/


Open the watcher in a browser with python Plexus_gui.py (it opens http://127.0.0.1:8799/watch); it shows the records of experiments/, one at a time (?rec=experiments/<name>).

!watcher

To start a new experiment, follow experiments/INSTRUCTION.md.
