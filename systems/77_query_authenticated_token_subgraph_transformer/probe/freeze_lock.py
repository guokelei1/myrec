#!/usr/bin/env python
"""Freeze and verify the C77 data-free implementation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
C76_GENERATOR = REPO / "systems/76_counterfactual_layer_trajectory_transformer/probe/synthetic.py"
SOURCES = (
    "model/qats.py",
    "model/__init__.py",
    "probe/c76_surface.py",
    "probe/freeze_lock.py",
    "probe/run_design_gate.py",
    "probe/summarize_design_gate.py",
    "tests/test_model.py",
    "tests/test_surface.py",
    "configs/design_gate.yaml",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("C77 config must be a mapping")
    if any(bool(value["authorization"][key]) for key in ("repository_data", "train_labels", "dev", "test", "qrels")):
        raise PermissionError("C77 data-free gate requests unauthorized access")
    return value


def source_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in SOURCES}


def verify_proposal() -> str:
    path = ROOT / "notes/proposal_lock.json"
    lock = json.loads(path.read_text(encoding="utf-8"))
    for name, expected in lock["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"C77 proposal source changed: {name}")
    for name, expected in lock["evidence_sha256"].items():
        if sha256(REPO / name) != expected:
            raise RuntimeError(f"C77 evidence changed: {name}")
    return sha256(path)


def verify_execution_lock() -> tuple[dict[str, Any], str]:
    proposal_hash = verify_proposal()
    path = ROOT / "notes/execution_lock.json"
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock["proposal_lock_sha256"] != proposal_hash:
        raise RuntimeError("C77 proposal lock differs")
    if lock["source_sha256"] != source_hashes():
        raise RuntimeError("C77 implementation changed")
    if lock["c76_generator_sha256"] != sha256(C76_GENERATOR):
        raise RuntimeError("C77 inherited generator changed")
    return lock, sha256(path)


def freeze() -> None:
    path = ROOT / "notes/execution_lock.json"
    if path.exists():
        raise FileExistsError(path)
    lock = {
        "candidate_id": "c77",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "authorize_three_seed_data_free_gate",
        "proposal_lock_sha256": verify_proposal(),
        "source_sha256": source_hashes(),
        "c76_generator_sha256": sha256(C76_GENERATOR),
        "boundary": {
            "repository_data": False,
            "labels": False,
            "dev_test_qrels": False,
            "attempts": 1,
        },
    }
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "sha256": sha256(path)}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        _, digest = verify_execution_lock()
        print(json.dumps({"verified": True, "sha256": digest}, sort_keys=True))
    else:
        freeze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
