"""C65 hash and write-once guards."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_proposal_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = REPO_ROOT / config["paths"]["proposal_lock"]
    lock = json.loads(path.read_text(encoding="utf-8"))
    design = {
        "config": SYSTEM_ROOT / "configs/train_gate.yaml",
        "nearest_neighbors": REPO_ROOT / config["paths"]["nearest_neighbors"],
        "preimplementation_review": REPO_ROOT / config["paths"]["preimplementation_review"],
        "proposal": REPO_ROOT / config["paths"]["proposal"],
        "readme": SYSTEM_ROOT / "README.md",
    }
    for name, source in design.items():
        if sha256_file(source) != lock["design_sha256"][name]:
            raise RuntimeError(f"C65 proposal source changed: {name}")
    if sha256_file(REPO_ROOT / config["paths"]["c64_report"]) != lock[
        "predecessor_sha256"
    ]["c64_report"]:
        raise RuntimeError("C65 C64 report changed")
    selections = {
        "c26": REPO_ROOT / config["paths"]["c26_selection"],
        "c64_split": REPO_ROOT / config["paths"]["c64_split_manifest"],
    }
    for name, source in selections.items():
        if sha256_file(source) != lock["selection_sha256"][name]:
            raise RuntimeError(f"C65 selection changed: {name}")
    return lock, sha256_file(path)


def verify_g0_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    _, proposal_hash = verify_proposal_lock(config)
    path = REPO_ROOT / config["paths"]["g0_lock"]
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock["proposal_lock_sha256"] != proposal_hash:
        raise RuntimeError("C65 G0 lock points to another proposal")
    for relative, expected in lock["source_sha256"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"C65 G0 source changed: {relative}")
    return lock, sha256_file(path)


def verify_execution_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    _, g0_hash = verify_g0_lock(config)
    path = REPO_ROOT / config["paths"]["execution_lock"]
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock["g0_lock_sha256"] != g0_hash:
        raise RuntimeError("C65 execution lock points to another G0")
    for relative, expected in lock["source_sha256"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"C65 execution source changed: {relative}")
    for relative, expected in lock["data_sha256"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"C65 data changed: {relative}")
    return lock, sha256_file(path)
