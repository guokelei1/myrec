"""Read-only bridge to C64's hash-locked token data interface."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
SOURCE = REPO_ROOT / "systems/64_end_to_end_lm_representation_probe/train/data.py"


def _load() -> Any:
    name = "c65_runtime_c64_data"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError("C65 cannot load the frozen C64 data interface")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DATA = _load()
C64Store = DATA.C64Store
iter_training_batches = DATA.iter_training_batches
iter_validation_batches = DATA.iter_validation_batches
to_device = DATA.to_device

__all__ = [
    "C64Store",
    "iter_training_batches",
    "iter_validation_batches",
    "to_device",
]
