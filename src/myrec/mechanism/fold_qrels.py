"""Deterministically split frozen dev qrels before adaptive layer selection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from myrec.mechanism.representation_probe import (
    load_m2_probe_manifest,
    normalized_query_fold,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


def materialize_fold_qrels(
    standardized_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create immutable fold-specific qrels before any fold-0 score is inspected."""

    standardized_dir = Path(standardized_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"fold qrels output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = standardized_dir / "records_dev.jsonl"
    qrels_path = standardized_dir / "qrels_dev.jsonl"
    frozen = load_m2_probe_manifest()["frozen_inputs"]
    if sha256_file(records_path) != frozen["records_dev_sha256"]:
        raise ValueError("fold-qrels records differ from frozen dev records")
    if sha256_file(qrels_path) != frozen["qrels_dev_sha256"]:
        raise ValueError("fold-qrels source differs from frozen dev qrels")
    records = list(iter_jsonl(records_path))
    qrels = list(iter_jsonl(qrels_path))
    if len(records) != 8000 or len(qrels) != 8000:
        raise ValueError("fold-qrels materialization requires 8000 aligned requests")
    handles = {}
    temporary_paths = {}
    counts = {0: 0, 1: 0}
    try:
        for fold in (0, 1):
            path = output_dir / f"qrels_dev_fold{fold}.jsonl"
            temporary = output_dir / f".{path.name}.tmp-{os.getpid()}"
            handles[fold] = temporary.open("x", encoding="utf-8")
            temporary_paths[fold] = temporary
        for record, qrel in zip(records, qrels):
            request_id = str(record.get("request_id") or "")
            if not request_id or str(qrel.get("request_id") or "") != request_id:
                raise ValueError("fold-qrels records/qrels order or identity differs")
            fold = normalized_query_fold(str(record.get("query") or ""))
            handles[fold].write(_canonical_json(qrel) + "\n")
            counts[fold] += 1
    finally:
        for handle in handles.values():
            handle.close()
    outputs = {}
    for fold in (0, 1):
        path = output_dir / f"qrels_dev_fold{fold}.jsonl"
        os.replace(temporary_paths[fold], path)
        outputs[str(fold)] = {
            "path": str(path),
            "request_count": counts[fold],
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_frozen_fold_qrels",
        "split_rule": "sha256(normalized_query)_mod_2",
        "materialized_before_postblock_fold0_selection": True,
        "source_records_path": str(records_path),
        "source_records_sha256": sha256_file(records_path),
        "source_qrels_path": str(qrels_path),
        "source_qrels_sha256": sha256_file(qrels_path),
        "outputs": outputs,
        "status": "completed",
    }
    manifest_path = output_dir / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return manifest


def audit_fold_qrels(
    standardized_dir: str | Path,
    split_dir: str | Path,
    fold: int,
) -> tuple[Path, dict[str, Any]]:
    """Verify a frozen split without opening the other fold's qrels file."""

    standardized_dir = Path(standardized_dir)
    split_dir = Path(split_dir)
    fold = int(fold)
    if fold not in (0, 1):
        raise ValueError("qrels fold must be 0 or 1")
    manifest_path = split_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "analysis_type": "transformer_deep_dive_frozen_fold_qrels",
        "split_rule": "sha256(normalized_query)_mod_2",
        "materialized_before_postblock_fold0_selection": True,
        "status": "completed",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"fold-qrels manifest mismatch: {key}")
    records_path = standardized_dir / "records_dev.jsonl"
    source_qrels_path = standardized_dir / "qrels_dev.jsonl"
    if (
        manifest.get("source_records_sha256") != sha256_file(records_path)
        or manifest.get("source_qrels_sha256") != sha256_file(source_qrels_path)
    ):
        raise ValueError("fold-qrels source binding differs")
    entry = manifest.get("outputs", {}).get(str(fold), {})
    path = split_dir / f"qrels_dev_fold{fold}.jsonl"
    if entry.get("path") != str(path) or entry.get("sha256") != sha256_file(path):
        raise ValueError("fold-qrels output binding differs")
    return path, manifest


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
