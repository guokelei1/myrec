"""Load the byte-locked C73 generator as C74's fixed information problem."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
C73_SOURCE = (
    REPO_ROOT
    / "systems/73_counterfactual_query_relay_transformer/probe/synthetic.py"
)
SPEC = importlib.util.spec_from_file_location("c73_locked_synthetic_for_c74", C73_SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("C74 cannot load locked C73 generator")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

SyntheticData = MODULE.SyntheticData
make_dataset = MODULE.make_dataset
wrong_history = MODULE.wrong_history
shuffled_history = MODULE.shuffled_history
coarse_history = MODULE.coarse_history
query_masked = MODULE.query_masked

__all__ = [
    "SyntheticData",
    "make_dataset",
    "wrong_history",
    "shuffled_history",
    "coarse_history",
    "query_masked",
]
