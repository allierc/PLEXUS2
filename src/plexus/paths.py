"""Data-root, config, and pre-folder resolution (mirrors connectome-gnn's utils).

Plexus keeps three roots:
  * the **repo** holds `config/<pre_folder>/<name>.yaml` (the specs, version-controlled);
  * the **data root** holds `graphs_data/<pre_folder>/<name>/` (generated trajectories)
    and `log/<pre_folder>/<name>/` (run logs);  set with `--output_root` /
    `$PLEXUS_OUTPUT_ROOT` / `$GNN_OUTPUT_ROOT` (no default in code: its location is local).

The **pre-folder** is the simulation *type* (interaction / boids / mpm / divide ...).
It is inferred from the config name -- the way the prototype scenarios were named
("attraction_repulsion", "tissue_two_types_mpm", "divide_cell") -- so a bare
`-o generate attraction_repulsion` routes to `config/interaction/...` and writes to
`graphs_data/interaction/...`. A name containing a `/` names its folder explicitly.
"""
from __future__ import annotations

import os

# _data_root: Root of `graphs_data/` and `log/` (`--output_root`, `$PLEXUS_OUTPUT_ROOT` or `$GNN_OUTPUT_ROOT`); no default in code, the shared data area is a local setting (this repo is public).
_data_root = os.environ.get("PLEXUS_OUTPUT_ROOT") or os.environ.get("GNN_OUTPUT_ROOT") or ""


# warn: Print a `[warn]` line in yellow on a terminal, so it is not read as ordinary progress.
def warn(msg: str) -> None:
    """Print a `[warn]` line in yellow, so it is not read as ordinary progress (a correct "property read by no operator" warning scrolled past in the banner's white for weeks while a child set silently built as the wrong material). Colour only on a terminal: `sys.stdout.isatty()` is False for a cluster job's redirected output, and an escape sequence in a log is noise."""
    import sys
    on = "\033[33m" if sys.stdout.isatty() else ""
    off = "\033[0m" if sys.stdout.isatty() else ""
    print(f"{on}{msg}{off}", flush=True)


# get_data_root: The root under which graphs_data/ and log/ live -> str.
def get_data_root() -> str:
    if not _data_root:
        raise RuntimeError("no data root: pass --output_root or set PLEXUS_OUTPUT_ROOT / GNN_OUTPUT_ROOT")
    return _data_root


# set_data_root: Set the root under which graphs_data/ and log/ live (call early in main).
def set_data_root(path: str) -> None:
    """Set the root under which graphs_data/ and log/ live (call early in main)."""
    global _data_root
    _data_root = path


# graphs_data_path: {data_root}/graphs_data/<parts> -> str.
def graphs_data_path(*parts: str) -> str:
    """`graphs_data_path('interaction', 'ar') -> {data_root}/graphs_data/interaction/ar`"""
    return os.path.join(get_data_root(), "graphs_data", *parts)


# log_path: {data_root}/log/<parts> -> str.
def log_path(*parts: str) -> str:
    """`log_path('interaction', 'ar') -> {data_root}/log/interaction/ar`"""
    return os.path.join(get_data_root(), "log", *parts)


# get_repo_root: The repository root -> str.
def get_repo_root() -> str:
    # this file is src/plexus/paths.py -> repo root is three levels up
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


# config_path: {repo}/config/<parts> -> str; always the repo, never the data root.
def config_path(*parts: str) -> str:
    """`config_path('interaction', 'ar.yaml') -> {repo}/config/interaction/ar.yaml`.
    Configs always load from the repo (version-controlled), never the data root;
    use an absolute path to load a spec from elsewhere."""
    return os.path.join(get_repo_root(), "config", *parts)


# _PRE_FOLDER_RULES: Ordered (folder, trigger substrings) rules inferring a spec's simulation type from its name: the leftmost trigger in the name wins, a trigger starting with `^` is anchored at position 0, and a folder with no triggers is addressed explicitly as `<folder>/<name>`.
_PRE_FOLDER_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("attraction_repulsion", ("attraction_repulsion", "attract", "repuls", "arbitrary")),
    ("inverse_square", ("inverse_square", "coulomb", "electro", "charge", "nbody", "galaxy", "spiral")),
    ("attractors",  ("attractor", "lorenz", "halvorsen", "aizawa", "sprott", "thomas",
                     "rossler", "dadras", "chen", "chua", "rabinovich")),
    ("boids",       ("boids", "flock", "swarm")),
    ("slime",       ("slime", "physarum", "trail")),
    ("material",    ("material",)),
    # the SI family lives apart from `material/`: an si_material spec declares metres, seconds and pascals via `general.units`, a material one declares nothing
    ("si_material", ("si_material", "si_water", "si_fill", "si_splash", "si_gate")),
    # where the studio puts what it writes: drafts generated and edited in the browser, kept apart from curated families (`validate_pre_folder` admits only a known simulation type)
    ("studio",      ("studio",)),
    ("active_matter", ("active_matter", "vicsek", "flock_am")),
    ("mpm",         ("mpm", "tissue", "elastic", "soft")),
    ("divide",      ("divide", "grow", "mitosis", "morula")),
    ("forage",      ("forage", "maze", "race", "graze")),
    ("gravity",     ("gravity",)),
    # the mesh-MPM interaction ladder: a prescribed vertex surface pressed into an MPM continuum through `mesh_contact`; the ANCHORED trigger `^mesh_mpm` claims exactly its own names (a bare `mesh` or `mpm` substring would sweep unrelated specs in, and `mpm` at position 5 once routed `mesh_mpm_sheet_elastic` to a folder that does not exist)
    ("mesh_mpm",    ("^mesh_mpm",)),
    # the myosin heterogeneity ladder: a spheroid whose halves get different myosin drive (`junction_myosin.activity_split`); anchored, since three specs already carry `myosin` in their names
    ("myosin_mesh", ("^myosin_mesh",)),
    # the vertex-model energy comparison, named for the question and not for one of its answers (it was `marinari`, which asserted a conclusion the comparison exists to test); anchored
    ("vertex_energy", ("^vertex_energy",)),
    # the composed-cell ladder (a cell built from a nucleus, cytosol and membrane, each its own contained set); the ladder is gone, the rule stays so a `cell_...` spec still lands somewhere; no substring trigger, `cell` is in half the corpus
    ("cell",        ("^cell_",)),
    # the tissue folder: the specs of the cell and tissue experiments (exp05, exp09, exp29, composition), addressed explicitly as `tissue/<name>`
    ("tissue",      ()),
]
# _VALID_PRE_FOLDERS: The simulation types `validate_pre_folder` admits (the folders of the rules above).
_VALID_PRE_FOLDERS = {f for f, _ in _PRE_FOLDER_RULES}


# validate_pre_folder: Check that a spec's folder names a known simulation type; subfolders are allowed -> None (raises otherwise).
def validate_pre_folder(pre_folder: str) -> None:
    """The folder a spec sits in must name a simulation type, and may then say more: `config/si_material/seeder/s0_a.yaml` is a si_material spec in a drawer. The FIRST segment is the type (it decides the output tree and the loaders); the rest is filing and travels with the run, so `graphs_data/si_material/seeder/s0_a` mirrors where the spec lives."""
    if not pre_folder:
        return
    name = pre_folder.rstrip("/").split("/", 1)[0]
    assert name in _VALID_PRE_FOLDERS, (
        f"pre_folder {pre_folder!r} is not a known simulation type "
        f"(expected one of: {', '.join(sorted(_VALID_PRE_FOLDERS))})")


# add_pre_folder: Map a bare config name to (`<pre_folder>/<name>`, `<pre_folder>/`) -> tuple.
def add_pre_folder(config_name: str) -> tuple[str, str]:
    """Map a bare config name to (`<pre_folder>/<name>`, `<pre_folder>/`).

    Idempotent: a name that already contains `/` is taken to name its folder
    explicitly (the parent), so re-prefixing is avoided."""
    if "/" in config_name:
        head = config_name.split("/", 1)[0]
        return config_name, head + "/"
    low = config_name.lower()
    # leftmost trigger wins: the type keyword leads the name (`slime_two_attract` is slime, `slime` at 0, not attraction_repulsion, `attract` at 10); an `^` trigger matches only at position 0 (a bare `cell_` also matched `saturating_cell_growth` at 11 and beat its real trigger `grow` at 16)
    best = None                                    # (position, folder)
    for folder, triggers in _PRE_FOLDER_RULES:
        hits = [0 for t in triggers if t.startswith("^") and low.startswith(t[1:])]
        hits += [low.find(t) for t in triggers if not t.startswith("^") and t in low]
        if hits and (best is None or min(hits) < best[0]):
            best = (min(hits), folder)
    if best is not None:
        return os.path.join(best[1], config_name), best[1] + "/"
    raise ValueError(
        f"cannot infer a simulation type (pre-folder) from config name {config_name!r}; "
        f"known types: {', '.join(sorted(_VALID_PRE_FOLDERS))}. "
        f"Either rename the config to contain a type keyword, pass it as "
        f"'<type>/{config_name}', or add a rule in plexus.paths._PRE_FOLDER_RULES.")


# resolve_run: Where a finished run's outputs live (`<type>/<name>`, a bare name, or a directory) -> its graphs_data directory.
def resolve_run(name: str) -> str:
    """Where a FINISHED run's outputs live: `cell/adh_jelly_x10` -> its `graphs_data` directory. The mirror of `resolve_config`: a spec names a run as `<type>/<name>`, not by an absolute path true on one machine. Accepts a directory (taken as-is), `<pre_folder>/<name>`, or a bare `<name>` (folder inferred). Raises rather than returning a missing directory, so a spec loading a run that is not there fails at load, not hundreds of frames in."""
    if os.path.isdir(name):
        return os.path.abspath(name)
    rel, _pre = add_pre_folder(name) if "/" not in name else (name, None)
    path = graphs_data_path(*rel.split("/"))
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"run {name!r} is not at {path}. `resolve_run` looks under "
            f"{get_data_root()}/graphs_data; pass an absolute directory, or set --output_root / "
            f"$PLEXUS_OUTPUT_ROOT to the root that holds it.")
    return path


# resolve_config: Resolve a CLI config argument -> (yaml_file, pre_folder, name).
def resolve_config(config_name: str) -> tuple[str, str, str]:
    """Resolve a CLI config argument to (yaml_file, pre_folder, name).

    Accepts an absolute/relative filesystem path (folder = parent dir) or a
    repo-relative bare/`type/`-prefixed name (folder inferred / explicit)."""
    if os.path.isfile(config_name) or os.path.isabs(config_name) or config_name.endswith(".yaml"):
        yaml_file = config_name if config_name.endswith(".yaml") else config_name + ".yaml"
        # the folder is everything under `config/`, not the last segment: `config/si_material/seeder/` is a si_material spec in a drawer
        _d = os.path.dirname(os.path.abspath(yaml_file))
        _root = os.path.abspath(config_path(""))
        pre_folder = (os.path.relpath(_d, _root) if _d.startswith(_root + os.sep) or _d == _root
                      else os.path.basename(_d)) + "/"
        name = os.path.splitext(os.path.basename(yaml_file))[0]
        return yaml_file, pre_folder, name
    rel, pre_folder = add_pre_folder(config_name)
    yaml_file = config_path(f"{rel}.yaml")
    name = os.path.basename(rel)
    return yaml_file, pre_folder, name
