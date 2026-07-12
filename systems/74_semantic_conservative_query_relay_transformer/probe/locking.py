"""C74 immutable-source and external-generator verification."""

from __future__ import annotations

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


def load_config() -> dict[str, Any]:
    return yaml.safe_load(
        (SYSTEM_ROOT / "configs/design_gate.yaml").read_text(encoding="utf-8")
    )


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = REPO_ROOT / config["paths"]["proposal_lock"]
    lock = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in lock["files"].items():
        if sha256_file(SYSTEM_ROOT / relative) != expected:
            raise RuntimeError(f"C74 locked source changed: {relative}")
    for relative, expected in lock["external_files"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"C74 external source changed: {relative}")
    return lock, sha256_file(path)
