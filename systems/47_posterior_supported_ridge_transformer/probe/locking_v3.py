"""Supplemental lock for the C47 boolean-polarity repair."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from probe.locking import REPO_ROOT, SYSTEM_ROOT, read_json, sha256_file


FILES = (
    "notes/proposal_lock.json",
    "notes/proposal_lock_v2.json",
    "notes/preselection_v2_execution_abort.md",
    "probe/locking_v3.py",
    "probe/materialize_selection_v3.py",
)


def freeze_v3(config: Mapping[str, Any]) -> dict[str, Any]:
    target = SYSTEM_ROOT / "notes/proposal_lock_v3.json"
    if target.exists():
        raise FileExistsError(target)
    selection = REPO_ROOT / config["paths"]["selection"]
    if selection.exists():
        raise RuntimeError("C47 selection exists before v3 lock")
    value = {
        "candidate_id": "c47",
        "status": "locked_v3_boolean_polarity_only",
        "files": {
            str((SYSTEM_ROOT / relative).relative_to(REPO_ROOT)): sha256_file(SYSTEM_ROOT / relative)
            for relative in FILES
        },
        "checks": {
            "v1_v2_selection_absent": True,
            "labels_features_scores_opened": False,
            "scientific_settings_changed": False,
            "repair": "rename negative access facts to positive closed checks",
        },
    }
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def verify_v3(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    target = SYSTEM_ROOT / "notes/proposal_lock_v3.json"
    value = read_json(target)
    if value.get("status") != "locked_v3_boolean_polarity_only":
        raise RuntimeError("C47 v3 lock status differs")
    for relative, expected in value["files"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"C47 v3 locked input changed: {relative}")
    return value, sha256_file(target)
