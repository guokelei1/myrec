"""Hash locks and immutable JSON helpers for C56."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("candidate_id") != "c56":
        raise ValueError("unexpected C56 config")
    if value.get("gate_id") != "c56_query_complement_token_competition_v3":
        raise ValueError("unexpected C56 gate")
    forbidden = ("qrels", "records_dev", "records_test", "metrics.json")
    for name, raw in value.get("paths", {}).items():
        if any(token in str(raw).lower() for token in forbidden):
            raise PermissionError(f"forbidden C56 path {name}: {raw}")
    return value


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_once(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _verify_lock(config: Mapping[str, Any], name: str) -> tuple[dict[str, Any], str]:
    path = REPO_ROOT / config["paths"][name]
    lock = read_json(path)
    for relative, expected in lock["locked_files"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"C56 {name} source changed: {relative}")
    return lock, sha256_file(path)


def verify_proposal(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    return _verify_lock(config, "proposal_lock")


def verify_execution(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    verify_proposal(config)
    return _verify_lock(config, "execution_lock")
