"""Reusable qrels-blind storage contract for scalar intervention bundles."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.utils.hashing import sha256_file, sha256_text


@dataclass
class PreparedScalarBundle:
    metadata: dict[str, Any]
    progress: dict[str, Any]
    partial_hasher: Any


def prepare_scalar_bundle(
    run_dir: str | Path,
    *,
    metadata: Mapping[str, Any],
    contract_sha256: str,
    records: Sequence[ModelRecord],
    conditions: Sequence[str],
    resume: bool,
) -> PreparedScalarBundle:
    """Create or audit a resumable bundle before any model forward."""

    run_dir = Path(run_dir)
    conditions = _validate_conditions(conditions)
    partial_path = run_dir / "scores.partial.jsonl"
    if not resume:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"scalar bundle directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        partial_path.touch(exist_ok=False)
        stored_metadata = {
            **dict(metadata),
            "elapsed_seconds": 0.0,
            "resumable": True,
            "resume_lineage": [],
        }
        progress = {
            "schema_version": 1,
            "run_contract_sha256": str(contract_sha256),
            "completed_requests": 0,
            "completed_score_rows": 0,
            "last_request_id": None,
            "partial_sha256": sha256_file(partial_path),
            "status": "initializing",
            "updated_at": _utc_now(),
        }
        _write_json(run_dir / "metadata.json", stored_metadata)
        _write_json(run_dir / "progress.json", progress)
        return PreparedScalarBundle(stored_metadata, progress, hashlib.sha256())

    stored_metadata = _read_json(run_dir / "metadata.json")
    progress = _read_json(run_dir / "progress.json")
    if stored_metadata.get("run_contract_sha256") != contract_sha256 or progress.get(
        "run_contract_sha256"
    ) != contract_sha256:
        raise ValueError("scalar bundle resume contract drift")
    observed = audit_scalar_partial(partial_path, records, conditions)
    for key in (
        "completed_requests",
        "completed_score_rows",
        "last_request_id",
        "partial_sha256",
    ):
        if progress.get(key) != observed[key]:
            raise ValueError(f"scalar bundle progress differs from partial: {key}")
    hasher = hashlib.sha256()
    with partial_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    lineage = list(stored_metadata.get("resume_lineage", []))
    lineage.append(
        {
            "resumed_at": _utc_now(),
            "completed_requests": observed["completed_requests"],
            "partial_sha256": observed["partial_sha256"],
        }
    )
    stored_metadata["resume_lineage"] = lineage
    _write_json(run_dir / "metadata.json", stored_metadata)
    return PreparedScalarBundle(stored_metadata, progress, hasher)


def append_scalar_request(
    run_dir: str | Path,
    block_row: Mapping[str, Any],
    prepared: PreparedScalarBundle,
) -> None:
    """Durably append one complete request and then advance progress."""

    run_dir = Path(run_dir)
    line = _canonical_json(block_row) + "\n"
    with (run_dir / "scores.partial.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    prepared.partial_hasher.update(line.encode("utf-8"))
    rows = list(block_row["rows"])
    prepared.progress.update(
        {
            "completed_requests": int(block_row["ordinal"]) + 1,
            "completed_score_rows": int(
                prepared.progress["completed_score_rows"]
            )
            + len(rows),
            "last_request_id": str(block_row["request_id"]),
            "partial_sha256": prepared.partial_hasher.hexdigest(),
            "status": "running",
            "updated_at": _utc_now(),
        }
    )
    _write_json(run_dir / "progress.json", prepared.progress)


def finalize_scalar_bundle(
    run_dir: str | Path,
    prepared: PreparedScalarBundle,
    records: Sequence[ModelRecord],
    conditions: Sequence[str],
    *,
    maximum_identity_delta: float,
    identity_tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    """Audit complete finite coverage and atomically publish scores.jsonl."""

    run_dir = Path(run_dir)
    partial_path = run_dir / "scores.partial.jsonl"
    observed = audit_scalar_partial(partial_path, records, conditions)
    if observed["completed_requests"] != len(records):
        raise ValueError("cannot finalize incomplete scalar condition bundle")
    maximum_identity_delta = float(maximum_identity_delta)
    identity_tolerance = float(identity_tolerance)
    if not math.isfinite(maximum_identity_delta):
        raise FloatingPointError("scalar identity delta is non-finite")
    identity_passed = maximum_identity_delta <= identity_tolerance
    if not identity_passed:
        raise ValueError(
            "scalar condition identity gate failed: "
            f"{maximum_identity_delta} > {identity_tolerance}"
        )
    scores_path = run_dir / "scores.jsonl"
    if scores_path.exists():
        raise FileExistsError(f"published scalar scores already exist: {scores_path}")
    os.replace(partial_path, scores_path)
    prepared.progress.update({"status": "completed", "updated_at": _utc_now()})
    _write_json(run_dir / "progress.json", prepared.progress)
    prepared.metadata.update(
        {
            "status": "completed",
            "resumable": False,
            "request_count": observed["completed_requests"],
            "score_rows": observed["completed_score_rows"],
            "scores_path": str(scores_path),
            "scores_sha256": sha256_file(scores_path),
            "complete_finite_score_coverage": True,
            "identity_passed": True,
            "identity_tolerance": identity_tolerance,
            "maximum_identity_delta": maximum_identity_delta,
        }
    )
    _write_json(run_dir / "metadata.json", prepared.metadata)
    return prepared.metadata


def audit_scalar_partial(
    path: str | Path,
    records: Sequence[ModelRecord],
    conditions: Sequence[str],
) -> dict[str, Any]:
    """Reconstruct coverage from durable bytes without reading qrels."""

    path = Path(path)
    conditions = _validate_conditions(conditions)
    condition_set = set(conditions)
    completed_rows = 0
    last_request_id = None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for ordinal, raw_line in enumerate(handle):
            hasher.update(raw_line)
            if ordinal >= len(records):
                raise ValueError("scalar bundle has more requests than target records")
            try:
                block = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError("scalar bundle contains a partial JSON line") from exc
            record = records[ordinal]
            if block.get("ordinal") != ordinal or block.get("request_id") != record.request_id:
                raise ValueError("scalar bundle request order/identity drift")
            rows = block.get("rows")
            if not isinstance(rows, list) or len(rows) != len(record.candidates):
                raise ValueError("scalar bundle candidate count drift")
            if block.get("rows_sha256") != _canonical_sha256(rows):
                raise ValueError("scalar bundle row digest drift")
            for candidate_ordinal, (row, candidate) in enumerate(
                zip(rows, record.candidates)
            ):
                if (
                    row.get("request_id") != record.request_id
                    or row.get("candidate_ordinal") != candidate_ordinal
                    or row.get("candidate_item_id") != str(candidate["item_id"])
                ):
                    raise ValueError("scalar bundle candidate identity/order drift")
                values = row.get("conditions")
                if not isinstance(values, dict) or set(values) != condition_set:
                    raise ValueError("scalar bundle condition coverage drift")
                if any(
                    not isinstance(values[name], (int, float))
                    or not math.isfinite(float(values[name]))
                    for name in conditions
                ):
                    raise FloatingPointError("scalar bundle contains a non-finite score")
            completed_rows += len(rows)
            last_request_id = record.request_id
    completed_requests = ordinal + 1 if "ordinal" in locals() else 0
    return {
        "completed_requests": completed_requests,
        "completed_score_rows": completed_rows,
        "last_request_id": last_request_id,
        "partial_sha256": hasher.hexdigest(),
    }


def _validate_conditions(conditions: Sequence[str]) -> tuple[str, ...]:
    result = tuple(map(str, conditions))
    if not result or len(set(result)) != len(result) or any(not item for item in result):
        raise ValueError("scalar bundle conditions must be unique nonempty strings")
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_sha256(value: Any) -> str:
    return sha256_text(_canonical_json(value))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
