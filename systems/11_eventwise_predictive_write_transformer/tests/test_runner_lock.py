from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_synthetic_gpu_gate as runner


def test_runner_refuses_execution_before_independent_approval():
    assert not runner.LOCK_PATH.exists()
    with pytest.raises(RuntimeError, match="no approved frozen_manifest"):
        runner.verify_approved_lock()
