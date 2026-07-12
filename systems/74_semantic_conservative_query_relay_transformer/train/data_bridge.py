"""Use C64's label-staged token store under C74's compatible config."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
SOURCE = REPO_ROOT / "systems/64_end_to_end_lm_representation_probe/train/data.py"
SPEC = importlib.util.spec_from_file_location("c64_locked_data_for_c74", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("C74 cannot load C64 token store")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class C74Store(MODULE.C64Store):
    def split_manifest(self) -> dict[str, Any]:
        value = super().split_manifest()
        value["candidate_id"] = "c74"
        value["source_boundary"] = "c64_exact_exposed_fit_split"
        return value


iter_training_batches = MODULE.iter_training_batches
iter_validation_batches = MODULE.iter_validation_batches
to_device = MODULE.to_device
rankings = MODULE.rankings

__all__ = [
    "C74Store",
    "iter_training_batches",
    "iter_validation_batches",
    "to_device",
    "rankings",
]
