"""The bio tab: a tissue -- an apico-basal sheet, disc or spheroid of cells -- with its protein
species and organelles, as a form over `gui/bio.py`'s spec builder (the `spheroid_proteins`
template's mechanics, growth and division; the form decides geometry, polarity and the species).
"""
from __future__ import annotations

import os

from plexus.gui import bio, studio

NAME, TITLE = "bio", "Plexus bio objects"
PICK_DIR = os.path.join(studio.REPO, "config", "tissue")
CORPUS_MODE = "bio"
BRIEF = bio.BIO_BRIEF.replace("/api/bio/build", "/api/tab/bio/build").replace("/api/bio/", "/api/scene/")
# THE TAB OPENS A FILE, NOT THE FORM. `ms4_round_prism_shell` is a run's own spec (copied from
# graphs_data/tissue/ms4_round_prism_shell/spec.yaml): a 200-cell apico-basal shell with the
# growth, mechanics, division and edge-flip a tissue run actually uses, which the form -- whose
# template is `spheroid_proteins` -- cannot write. BUILD + SEED still writes the form's spec.
DEFAULT_SPEC = os.path.join(studio.REPO, "config", "tissue", "ms4_round_prism_shell.yaml")
DEFAULT_FORM = {"name": "bio_scene", "shape": "sphere", "n_cells": 200, "radius": 5.0, "h0": 1.2, "apical": "in",
                "world": 50.0, "n_frames": 801,
                "species": [{"name": "integrin", "region": "basal", "density": 3.0, "s": 0.02, "tau": 300.0}],
                "organelles": [{"name": "nucleus", "count": 1, "radius": 0.3, "region": "basal_side",
                                "on_divide": "duplicate", "tau": None}]}


def build_spec(form: dict) -> dict:
    return bio.normalise(bio.build_spec(form))


normalise = bio.normalise
form_from_spec = bio.form_from_spec
