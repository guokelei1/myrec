"""C66 read-only bridge plus stable item-key construction."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
SOURCE = REPO_ROOT / "systems/64_end_to_end_lm_representation_probe/train/data.py"


def _load() -> Any:
    name = "c66_runtime_c64_data"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError("C66 cannot load the frozen C64 data interface")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DATA = _load()
C64Store = DATA.C64Store
iter_training_batches = DATA.iter_training_batches
iter_validation_batches = DATA.iter_validation_batches
to_device = DATA.to_device


def stable_item_key(item_id: str) -> int:
    return int.from_bytes(hashlib.sha256(str(item_id).encode()).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def candidate_keys(batch: Mapping[str, Any]) -> np.ndarray:
    mask = np.asarray(batch["candidate_mask"], dtype=bool)
    items = np.asarray(batch["candidate_item_ids"], dtype=object)
    output = np.zeros(mask.shape, dtype=np.int64)
    for row, column in zip(*np.where(mask)):
        output[row, column] = stable_item_key(str(items[row, column]))
    for row in range(len(output)):
        valid = output[row, mask[row]]
        if len(np.unique(valid)) != len(valid):
            raise ValueError("C66 stable item-key collision")
    return output


__all__ = [
    "C64Store",
    "candidate_keys",
    "iter_training_batches",
    "iter_validation_batches",
    "stable_item_key",
    "to_device",
]
