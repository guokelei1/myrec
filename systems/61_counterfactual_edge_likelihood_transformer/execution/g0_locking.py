"""C61 G0-specific immutable source lock."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from execution.locking import (
    REPO_ROOT,
    SYSTEM_ROOT,
    read_json,
    sha256_file,
    verify_materialization,
)


G0_LOCK = SYSTEM_ROOT / "notes/g0_lock.json"


def verify_g0(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    verify_materialization(config)
    lock = read_json(G0_LOCK)
    for relative, expected in lock["locked_files"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"C61 G0 source changed: {relative}")
    manifest = REPO_ROOT / config["paths"]["contextual_manifest"]
    if sha256_file(manifest) != lock["contextual_manifest_sha256"]:
        raise RuntimeError("C61 G0 contextual manifest changed")
    return lock, sha256_file(G0_LOCK)
