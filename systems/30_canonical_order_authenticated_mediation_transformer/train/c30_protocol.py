"""C30 config and immutable continuation-lock utilities."""

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


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(target)


def write_json_once(path: str | Path, value: Mapping[str, Any]) -> None:
    if Path(path).exists():
        raise FileExistsError(f"immutable C30 output exists: {path}")
    atomic_json(path, value)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or value.get("candidate_id") != "c30":
        raise ValueError("unexpected C30 config")
    if value.get("gate_id") != "c30_canonical_order_continuation_v1":
        raise ValueError("unexpected C30 gate")
    if [int(seed) for seed in value["training"]["seeds"]] != [20260831, 20260832, 20260833]:
        raise ValueError("C30 seed registration differs")
    if value["training"]["retraining"] is not False or int(
        value["training"]["optimizer_steps"]
    ) != 0:
        raise PermissionError("C30 retraining is forbidden")
    for name, raw in value["paths"].items():
        if name == "train_candidate_labels":
            continue
        if any(token in str(raw).lower() for token in ("qrels", "records_dev", "records_test")):
            raise ValueError(f"forbidden C30 path: {name}")
    return value


def candidate_hashes() -> dict[str, str]:
    excluded = {"notes/continuation_lock.json", "notes/continuation_outcome.md"}
    output: dict[str, str] = {}
    for path in sorted(SYSTEM_ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = str(path.relative_to(SYSTEM_ROOT))
        if relative not in excluded:
            output[relative] = sha256_file(path)
    return output


def verify_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = Path(config["paths"]["continuation_lock"])
    lock = read_json(path)
    if lock.get("status") != "locked_before_c30_canonical_score_or_A_label":
        raise ValueError("unexpected C30 lock status")
    failures: list[str] = []
    lines: list[str] = []
    for relative, expected in sorted(lock["files_sha256"].items()):
        source = SYSTEM_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            failures.append(f"candidate:{relative}")
        lines.append(f"{expected}  {relative}\n")
    if hashlib.sha256("".join(lines).encode()).hexdigest() != lock["aggregate_sha256"]:
        failures.append("aggregate")
    for relative, expected in lock["external_inputs_sha256"].items():
        source = REPO_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            failures.append(f"external:{relative}")
    if failures:
        raise RuntimeError(f"C30 lock mismatch: {failures}")
    return lock, sha256_file(path)
