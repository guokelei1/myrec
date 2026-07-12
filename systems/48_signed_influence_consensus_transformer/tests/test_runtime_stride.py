from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
PROBE_ROOT = SYSTEM_ROOT / "probe"
sys.path.insert(0, str(PROBE_ROOT))
SPEC = importlib.util.spec_from_file_location("c48_runtime_v2", PROBE_ROOT / "run_formulation_gate_v2.py")
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_negative_stride_candidates_and_history_are_accepted():
    config = yaml.safe_load((SYSTEM_ROOT / "configs/formulation_gate.yaml").read_text(encoding="utf-8"))
    query = np.asarray([1.0, 0.2, -0.1], dtype=np.float32)
    history = np.eye(3, dtype=np.float32)[::-1]
    candidates = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)[::-1]
    output = MOD.score_one(query, history, candidates, config)
    assert all(value.flags.c_contiguous for value in output.values())
    assert all(np.isfinite(value).all() for value in output.values())
