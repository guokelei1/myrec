"""Candidate-local I/O and integrity helpers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping config: {path}")
    return value


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_candidate_manifest(path: str | Path, expected_sha256: str) -> str:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"candidate manifest hash mismatch: expected={expected_sha256} actual={actual}"
        )
    return actual


def assert_train_only_path(path: str | Path) -> None:
    lowered = str(path).lower()
    forbidden = ("qrels", "records_dev", "records_test", "metrics.json")
    if any(token in lowered for token in forbidden):
        raise ValueError(f"training path crosses the frozen label/split boundary: {path}")


def assert_label_free_record(record: dict[str, Any]) -> None:
    for candidate in record.get("candidates", []):
        leaked = {"clicked", "purchased"}.intersection(candidate)
        if leaked:
            raise ValueError(
                f"label-bearing candidate in label-free scoring record: {sorted(leaked)}"
            )


def finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite score: {value!r}")
    return result
