#!/usr/bin/env python
"""Run the one authorized C02 continuation under its additive lock."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
for entry in (SYSTEM_ROOT, REPO_ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from train import train_screen
from train.runtime import read_json, sha256_file


ALLOWED_REPLACEMENTS = {
    "train/train_screen.py",
    "tests/test_chht.py",
}


def assert_continuation_lock(config: dict[str, Any]) -> dict[str, Any]:
    source_root = Path(config["paths"]["candidate_source_root"])
    proposal_path = source_root / "notes/proposal_lock.json"
    continuation_path = source_root / "notes/mechanical_continuation_lock.json"
    proposal = read_json(proposal_path)
    continuation = read_json(continuation_path)

    if proposal.get("status") != "locked_before_c02_gpu_outcome":
        raise ValueError("original C02 proposal lock is invalid")
    if continuation.get("status") != "frozen_before_continuation_gpu_outcome":
        raise ValueError("C02 continuation is not frozen before its GPU outcome")
    if continuation["execution"]["candidate_manifest_sha256"] != config["integrity"][
        "candidate_manifest_sha256"
    ]:
        raise ValueError("continuation/config candidate manifest mismatch")
    if int(continuation["execution"]["seed"]) != int(config["seed"]):
        raise ValueError("continuation/config seed mismatch")

    for relative, expected in proposal["file_sha256"].items():
        if relative in ALLOWED_REPLACEMENTS:
            continue
        actual = sha256_file(source_root / relative)
        if actual != expected:
            raise ValueError(f"post-proposal mutation outside continuation: {relative}")

    for path_string, expected in continuation["locked_files"].items():
        actual = sha256_file(REPO_ROOT / path_string)
        if actual != expected:
            raise ValueError(f"post-continuation-lock mutation: {path_string}")
    return proposal


if __name__ == "__main__":
    train_screen.assert_proposal_lock = assert_continuation_lock
    raise SystemExit(train_screen.main())
