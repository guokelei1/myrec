"""Label-free fixed QC plus history-residual score control."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any

from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


def write_fixed_residual_anchor_scores(
    qc_run_dir: str | Path,
    condition_run_dir: str | Path,
    null_run_dir: str | Path,
    output_run_dir: str | Path,
    candidate_manifest_path: str | Path,
    *,
    coefficient: float,
    history_condition: str,
    method_id: str,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Write ``QC + coefficient * (condition - FULL-null)`` scores.

    Source runs are score-only inputs. Candidate identities and the registered
    candidate-manifest hash are checked before any output is finalized.
    """

    if history_condition not in {"true", "null", "wrong"}:
        raise ValueError(f"unsupported history_condition={history_condition}")
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("coefficient must be finite and non-negative")
    qc_run_dir = Path(qc_run_dir)
    condition_run_dir = Path(condition_run_dir)
    null_run_dir = Path(null_run_dir)
    output_run_dir = Path(output_run_dir)
    candidate_manifest_path = Path(candidate_manifest_path)
    protocol_path = Path(protocol_path)
    if output_run_dir.exists() and any(output_run_dir.iterdir()):
        raise FileExistsError(f"output run directory is not empty: {output_run_dir}")
    output_run_dir.mkdir(parents=True, exist_ok=True)

    candidate_sha = sha256_file(candidate_manifest_path)
    source_dirs = {
        "qc": qc_run_dir,
        "condition": condition_run_dir,
        "full_null": null_run_dir,
    }
    source_metadata = {
        name: _load_metadata(path / "metadata.json")
        for name, path in source_dirs.items()
    }
    for name, metadata in source_metadata.items():
        observed = metadata.get("candidate_manifest_sha256")
        if observed != candidate_sha:
            raise ValueError(
                f"candidate manifest mismatch for {name}: {observed} != {candidate_sha}"
            )
    identity_keys = ("dataset_id", "dataset_version", "request_manifest_sha256", "split")
    for key in identity_keys:
        values = {str(metadata.get(key)) for metadata in source_metadata.values()}
        if len(values) != 1 or "None" in values:
            raise ValueError(f"source metadata identity mismatch for {key}: {values}")

    source_scores = {
        name: _load_scores(path / "scores.jsonl")
        for name, path in source_dirs.items()
    }
    key_sets = {name: set(rows) for name, rows in source_scores.items()}
    if len({frozenset(keys) for keys in key_sets.values()}) != 1:
        raise ValueError("source score runs have different candidate keys")

    scores_path = output_run_dir / "scores.jsonl"
    with scores_path.open("w", encoding="utf-8") as handle:
        for request_id, item_id in sorted(key_sets["qc"]):
            qc = source_scores["qc"][(request_id, item_id)]
            condition = source_scores["condition"][(request_id, item_id)]
            null = source_scores["full_null"][(request_id, item_id)]
            score = qc + coefficient * (condition - null)
            if not math.isfinite(score):
                raise ValueError(f"non-finite composed score for {request_id} {item_id}")
            handle.write(
                json.dumps(
                    {
                        "candidate_item_id": item_id,
                        "method_id": method_id,
                        "request_id": request_id,
                        "score": score,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    condition_metadata = source_metadata["condition"]
    null_metadata = source_metadata["full_null"]
    checkpoint_payload = json.dumps(
        {
            "coefficient": coefficient,
            "full_checkpoint": null_metadata.get("checkpoint_id"),
            "protocol_sha256": sha256_file(protocol_path),
            "qc_checkpoint": source_metadata["qc"].get("checkpoint_id"),
        },
        sort_keys=True,
    ).encode("utf-8")
    checkpoint_id = f"fixed-residual-anchor@{hashlib.sha256(checkpoint_payload).hexdigest()[:20]}"
    scoring_signature = {
        "coefficient": coefficient,
        "composition": "QC + coefficient * (FULL-condition - FULL-null)",
        "full_checkpoint_id": null_metadata.get("checkpoint_id"),
        "qc_checkpoint_id": source_metadata["qc"].get("checkpoint_id"),
        "source_full_scoring_signature": null_metadata.get("scoring_signature"),
        "source_qc_scoring_signature": source_metadata["qc"].get("scoring_signature"),
    }
    metadata = {
        "analysis_type": "fixed_qc_plus_history_residual_score_control",
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": candidate_sha,
        "checkpoint_id": checkpoint_id,
        "coefficient": coefficient,
        "dataset_id": condition_metadata["dataset_id"],
        "dataset_version": condition_metadata["dataset_version"],
        "history_assignment_sha256": condition_metadata.get(
            "history_assignment_sha256"
        ),
        "history_assignments_path": condition_metadata.get(
            "history_assignments_path"
        ),
        "history_condition": history_condition,
        "method_id": method_id,
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "qrels_read": False,
        "request_manifest_sha256": condition_metadata["request_manifest_sha256"],
        "score_definition": "QC + coefficient * (FULL-condition - FULL-null)",
        "score_rows": len(key_sets["qc"]),
        "scores_sha256": sha256_file(scores_path),
        "scoring_signature": scoring_signature,
        "split": condition_metadata["split"],
        "source_runs": {
            name: {
                "path": str(path),
                "metadata_sha256": sha256_file(path / "metadata.json"),
                "scores_sha256": sha256_file(path / "scores.jsonl"),
            }
            for name, path in source_dirs.items()
        },
        "test_or_confirmation": False,
        "training_performed": False,
        "tuning": {
            "class": "exploratory_fixed_arithmetic_simple_control",
            "dev_labels_used_during_scoring": False,
        },
    }
    write_json(output_run_dir / "metadata.json", metadata)
    return metadata


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_scores(path: Path) -> dict[tuple[str, str], float]:
    rows: dict[tuple[str, str], float] = {}
    for row in iter_jsonl(path):
        key = (str(row["request_id"]), str(row["candidate_item_id"]))
        if key in rows:
            raise ValueError(f"duplicate score key in {path}: {key}")
        value = float(row["score"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite source score in {path}: {key}")
        rows[key] = value
    if not rows:
        raise ValueError(f"empty score file: {path}")
    return rows
