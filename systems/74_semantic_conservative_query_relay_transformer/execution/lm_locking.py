"""C74 pretrained-LM lock verification helpers."""

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


def load_config(path: Path | None = None) -> dict[str, Any]:
    target = path or SYSTEM_ROOT / "configs/kuai_lm_probe.yaml"
    return yaml.safe_load(target.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _verify_hashes(rows: Mapping[str, str], *, label: str) -> None:
    for relative, expected in rows.items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"C74 {label} changed: {relative}")


def verify_g0_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = REPO_ROOT / config["paths"]["lm_probe_g0_lock"]
    value = json.loads(path.read_text(encoding="utf-8"))
    _verify_hashes(value["source_sha256"], label="G0 source")
    _verify_hashes(value["authority_sha256"], label="G0 authority")
    return value, sha256_file(path)


def verify_execution_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = REPO_ROOT / config["paths"]["lm_probe_execution_lock"]
    value = json.loads(path.read_text(encoding="utf-8"))
    _verify_hashes(value["source_sha256"], label="execution source")
    _verify_hashes(value["data_sha256"], label="execution data")
    return value, sha256_file(path)
