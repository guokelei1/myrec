"""C75 view of the locked C74/C64 exposed-fit token store."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
SOURCE = (
    REPO_ROOT
    / "systems/74_semantic_conservative_query_relay_transformer/train/data_bridge.py"
)
SPEC = importlib.util.spec_from_file_location("c74_data_bridge_for_c75", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("C75 cannot load C74 token store")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class C75Store(MODULE.C74Store):
    def split_manifest(self) -> dict[str, Any]:
        value = super().split_manifest()
        value["candidate_id"] = "c75"
        value["source_boundary"] = "c64_exact_exposed_fit_split_c75"
        return value


iter_training_batches = MODULE.iter_training_batches
iter_validation_batches = MODULE.iter_validation_batches
to_device = MODULE.to_device

__all__ = ["C75Store", "iter_training_batches", "iter_validation_batches", "to_device"]
