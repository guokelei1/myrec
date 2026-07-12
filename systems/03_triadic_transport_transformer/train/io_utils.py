"""I/O and integrity helpers restricted to the C03 candidate boundary."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CANDIDATE_ROOT.parents[1]


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    parent_name = config.pop("extends", None)
    if parent_name:
        parent_path = config_path.parent / str(parent_name)
        parent, _ = load_config(parent_path)
        config = deep_merge(parent, config)
    return config, config_path


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_u64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def stable_i63(*parts: object) -> int:
    return stable_u64(*parts) & ((1 << 63) - 1)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield row


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def assert_safe_input(path: str | Path, reject_tokens: list[str]) -> Path:
    resolved = repo_path(path).resolve()
    lowered_name = resolved.name.lower()
    for token in reject_tokens:
        if str(token).lower() in lowered_name:
            raise ValueError(f"forbidden input path token {token!r}: {resolved}")
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def assert_manifest(config: dict[str, Any]) -> Path:
    manifest = assert_safe_input(
        config["paths"]["candidate_manifest"],
        config["integrity"]["reject_path_tokens"],
    )
    actual = sha256_file(manifest)
    expected = config["integrity"]["candidate_manifest_sha256"]
    if actual != expected:
        raise ValueError(f"candidate manifest hash mismatch: {actual} != {expected}")
    return manifest


def assert_run_id(config: dict[str, Any]) -> str:
    run_id = str(config["candidate"]["run_id"])
    prefix = str(config["integrity"]["allowed_run_prefix"])
    if not run_id.startswith(prefix):
        raise ValueError(f"run ID {run_id!r} does not use C03 prefix {prefix!r}")
    return run_id


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def candidate_source_hash() -> str:
    digest = hashlib.sha256()
    excluded_parts = {
        "outputs",
        "checkpoints",
        "runs",
        "logs",
        ".cache",
        "__pycache__",
    }
    excluded_names = {"proposal_lock.json", "final_report.md"}
    for path in sorted(CANDIDATE_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(CANDIDATE_ROOT)
        if any(part in excluded_parts for part in relative.parts):
            continue
        if path.name in excluded_names or path.suffix == ".pyc":
            continue
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
