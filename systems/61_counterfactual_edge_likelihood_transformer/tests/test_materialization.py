from __future__ import annotations

from pathlib import Path
import sys


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from execution.locking import load_config, verify_proposal  # noqa: E402
from probe.materialize_contextual import required_indices  # noqa: E402


def test_proposal_and_required_surface() -> None:
    config = load_config(SYSTEM_ROOT / "configs/train_gate.yaml")
    lock, digest = verify_proposal(config)
    assert lock["candidate_id"] == "c61"
    assert len(digest) == 64
    requests, items = required_indices(config)
    assert len(requests) == 6000 + 1200 + 512 + 512
    assert len(items) > 0
    assert len(set(map(int, requests))) == len(requests)
