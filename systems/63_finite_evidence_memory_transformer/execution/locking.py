"""C63 hash and write-once guards."""

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
    sources = {
        "config": SYSTEM_ROOT / "configs/g0.yaml",
        "nearest_neighbors": REPO_ROOT / config["paths"]["nearest_neighbors"],
        "preimplementation_review": REPO_ROOT / config["paths"]["preimplementation_review"],
        "proposal": REPO_ROOT / config["paths"]["proposal"],
        "readme": SYSTEM_ROOT / "README.md",
    }
    for name, source in sources.items():
        if sha256_file(source) != lock["design_sha256"][name]:
            raise RuntimeError(f"C63 proposal source changed: {name}")
    predecessor = REPO_ROOT / config["paths"]["c62_report"]
    if sha256_file(predecessor) != lock["predecessor"]["c62_report_sha256"]:
        raise RuntimeError("C63 predecessor report changed")
    return lock, sha256_file(path)


def verify_g0_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    _, proposal_hash = verify_proposal_lock(config)
    path = REPO_ROOT / config["paths"]["g0_lock"]
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock["proposal_lock_sha256"] != proposal_hash:
        raise RuntimeError("C63 G0 lock points to another proposal")
    for relative, expected in lock["source_sha256"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"C63 G0 source changed: {relative}")
    return lock, sha256_file(path)
