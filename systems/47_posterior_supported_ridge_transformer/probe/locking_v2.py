"""Supplemental lock for the C47 physical-line reader repair."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from probe.locking import REPO_ROOT, SYSTEM_ROOT, read_json, sha256_file


FILES = (
    "notes/proposal_lock.json",
    "notes/preselection_execution_abort.md",
    "probe/locking_v2.py",
    "probe/materialize_selection_v2.py",
)


def freeze_v2(config: Mapping[str, Any]) -> dict[str, Any]:
    target = SYSTEM_ROOT / "notes/proposal_lock_v2.json"
    if target.exists():
        raise FileExistsError(target)
    selection = REPO_ROOT / config["paths"]["selection"]
    if selection.exists():
        raise RuntimeError("C47 selection exists before v2 lock")
    hashes = {
        str((SYSTEM_ROOT / relative).relative_to(REPO_ROOT)): sha256_file(SYSTEM_ROOT / relative)
        for relative in FILES
    }
    value = {
        "candidate_id": "c47",
        "status": "locked_v2_physical_line_reader_only",
        "files": hashes,
        "checks": {
            "v1_selection_absent": True,
            "labels_features_scores_opened": False,
            "scientific_settings_changed": False,
            "repair": "read physical JSONL lines instead of str.splitlines",
        },
    }
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def verify_v2(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    target = SYSTEM_ROOT / "notes/proposal_lock_v2.json"
    value = read_json(target)
    if value.get("status") != "locked_v2_physical_line_reader_only":
        raise RuntimeError("C47 v2 lock status differs")
    for relative, expected in value["files"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"C47 v2 locked input changed: {relative}")
    return value, sha256_file(target)
