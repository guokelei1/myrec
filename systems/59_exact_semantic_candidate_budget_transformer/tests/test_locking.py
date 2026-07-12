from __future__ import annotations

from pathlib import Path
import sys

import pytest


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from execution.locking import load_config, verify_proposal, write_once  # noqa: E402


def test_registered_proposal_lock_verifies() -> None:
    config = load_config(SYSTEM_ROOT / "configs/formulation_gate.yaml")
    lock, digest = verify_proposal(config)
    assert lock["candidate_id"] == "c59"
    assert len(digest) == 64


def test_write_once_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    write_once(path, {"ok": True})
    with pytest.raises(FileExistsError):
        write_once(path, {"ok": False})
