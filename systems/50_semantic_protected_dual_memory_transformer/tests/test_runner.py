from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "probe"
sys.path.insert(0, str(PROBE))
SPEC = importlib.util.spec_from_file_location("c50_runner", PROBE / "run_formulation_gate.py")
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_flatten_round_trip():
    rows = [np.asarray([1.0, 2.0], np.float32), np.asarray([3.0], np.float32)]
    offsets, values = MOD.flatten(rows)
    restored = MOD.unflatten(offsets, values)
    assert all(np.array_equal(a, b) for a, b in zip(rows, restored))
