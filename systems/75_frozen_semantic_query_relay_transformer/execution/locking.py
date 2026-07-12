"""C75 lock helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config() -> dict[str, Any]:
    return yaml.safe_load((SYSTEM_ROOT / "configs/kuai_probe.yaml").read_text())


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _verify(rows: Mapping[str, str], label: str) -> None:
    for relative, expected in rows.items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"C75 {label} changed: {relative}")


def verify_g0_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = REPO_ROOT / config["paths"]["g0_lock"]
    value = json.loads(path.read_text())
    _verify(value["source_sha256"], "G0 source")
    _verify(value["authority_sha256"], "G0 authority")
    return value, sha256_file(path)


def verify_execution_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = REPO_ROOT / config["paths"]["execution_lock"]
    value = json.loads(path.read_text())
    _verify(value["source_sha256"], "execution source")
    _verify(value["data_sha256"], "execution data")
    return value, sha256_file(path)
