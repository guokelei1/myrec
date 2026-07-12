from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
PROBE_ROOT = SYSTEM_ROOT / "probe"
sys.path.insert(0, str(PROBE_ROOT))
SPEC = importlib.util.spec_from_file_location("c48_runtime_v3", PROBE_ROOT / "run_formulation_gate_v3.py")
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_length_one_negative_stride_is_unconditionally_copied():
    config = yaml.safe_load((SYSTEM_ROOT / "configs/formulation_gate.yaml").read_text(encoding="utf-8"))
    query = np.asarray([1.0, 0.2, -0.1], dtype=np.float32)
    history = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)[::-1]
    assert history.strides[0] < 0 and history.flags.c_contiguous
    candidates = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    copied = MOD._copy(history)
    assert copied.strides[0] > 0 and copied.flags.c_contiguous
    output = MOD.score_one(query, history, candidates, config)
    assert all(np.isfinite(value).all() for value in output.values())
