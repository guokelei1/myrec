"""Qrels-blind equivalence gate for reusing frozen Q2 M2 patch scores."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.mechanism.attention_edge_runtime import _load_manifest, _write_json
from myrec.mechanism.patch_evaluator import _audit_score_bundle
from myrec.mechanism.postblock_sweep_scoring import POSTBLOCK_CONDITIONS
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


ANALYSIS_TYPE = "transformer_deep_dive_d2_q2_postblock_reuse_equivalence"
TOLERANCE = 1.0e-5
CONDITION_MAP = {
    "full_to_full_identity": "full_to_full_identity",
    "same_full_to_null": "same_request_full_to_null",
    "cross_full_to_null": "cross_request_same_layer",
}


def audit_q2_postblock_reuse_equivalence(
    standardized_dir: str | Path,
    smoke_bundle_dir: str | Path,
    identity_dir: str | Path,
    same_dir: str | Path,
    cross_dir: str | Path,
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    block: int,
    manifest_path: str | Path = "experiments/motivation/transformer_deep_dive_manifest.yaml",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare the new live implementation with the three reusable score paths."""

    block = int(block)
    if block not in (13, 27):
        raise ValueError("Q2 reuse equivalence is restricted to blocks 13/27")
    if not analysis_run_id or "/" in analysis_run_id or "\\" in analysis_run_id:
        raise ValueError("invalid Q2 reuse audit run id")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Q2 reuse audit output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    standardized_dir = Path(standardized_dir)
    records_path = standardized_dir / "records_dev.jsonl"
    manifest = _load_manifest(manifest_path)
    if sha256_file(records_path) != manifest["frozen_inputs"]["records_dev_sha256"]:
        raise ValueError("Q2 reuse audit records differ from frozen manifest")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("Q2 reuse audit requires all 8000 dev records for source audit")
    records_by_id = {record.request_id: record for record in records}

    smoke_root = Path(smoke_bundle_dir)
    smoke_metadata = _read_json(smoke_root / "metadata.json")
    smoke_scores_path = smoke_root / "scores.jsonl"
    if (
        smoke_metadata.get("analysis_stage")
        != "transformer_deep_dive_d2_postblock_sweep"
        or smoke_metadata.get("status") != "completed"
        or smoke_metadata.get("evidence_mode") != "mechanical_smoke_non_result"
        or smoke_metadata.get("result_eligible") is not False
        or smoke_metadata.get("method_id") != "q2_recranker_generalqwen"
        or int(smoke_metadata.get("block_zero_based", -1)) != block
        or int(smoke_metadata.get("request_count", -1)) > 32
        or smoke_metadata.get("qrels_read") is not False
        or smoke_metadata.get("source_test_opened") is not False
    ):
        raise ValueError("new Q2 post-block smoke bundle is not admissible")
    if smoke_metadata.get("scores_sha256") != sha256_file(smoke_scores_path):
        raise ValueError("new Q2 post-block smoke bytes changed after completion")
    smoke = _load_smoke_scores(smoke_scores_path, records_by_id)

    bundle_specs = {
        "full_to_full_identity": (identity_dir, "full_to_full_identity"),
        "same_request_full_to_null": (same_dir, "same_request_full_to_null"),
        "cross_request_same_layer": (cross_dir, "cross_request_same_layer"),
    }
    old = {}
    identities = {}
    for name, (root, kind) in bundle_specs.items():
        bundle = _audit_score_bundle(
            root,
            records,
            expected_condition=None,
            patch=(kind, block),
        )
        if (
            bundle.metadata.get("method_id") != smoke_metadata.get("method_id")
            or bundle.metadata.get("checkpoint_id")
            != smoke_metadata.get("checkpoint_id")
            or bundle.metadata.get("config_sha256")
            != smoke_metadata.get("config_sha256")
        ):
            raise ValueError("new/old Q2 reuse model binding differs")
        old[name] = bundle.scores
        identities[name] = {
            "path": str(bundle.root),
            "metadata_sha256": sha256_file(bundle.root / "metadata.json"),
            "scores_sha256": bundle.scores_sha256,
        }

    rows = {}
    all_deltas = []
    for new_name, old_name in CONDITION_MAP.items():
        deltas = []
        for key, conditions in smoke.items():
            request_id, item_id = key
            delta = abs(
                float(conditions[new_name]) - float(old[old_name][request_id][item_id])
            )
            if not math.isfinite(delta):
                raise FloatingPointError("Q2 reuse comparison is non-finite")
            deltas.append(delta)
        all_deltas.extend(deltas)
        rows[new_name] = {
            "old_patch_kind": old_name,
            "score_rows": len(deltas),
            "mean_abs_score_delta": float(np.mean(deltas)),
            "maximum_abs_score_delta": max(deltas, default=math.inf),
            "tolerance": TOLERANCE,
            "passed": max(deltas, default=math.inf) <= TOLERANCE,
        }
    maximum = max(all_deltas, default=math.inf)
    passed = maximum <= TOLERANCE and all(row["passed"] for row in rows.values())
    result = {
        "schema_version": 1,
        "analysis_type": ANALYSIS_TYPE,
        "analysis_run_id": analysis_run_id,
        "status": "passed" if passed else "failed",
        "method_id": smoke_metadata["method_id"],
        "checkpoint_id": smoke_metadata["checkpoint_id"],
        "config_sha256": smoke_metadata["config_sha256"],
        "block_zero_based": block,
        "comparison_scope": "new_live_postblock_vs_frozen_first_round_per_candidate",
        "score_rows_per_condition": len(smoke),
        "conditions": rows,
        "maximum_abs_score_delta": maximum,
        "tolerance": TOLERANCE,
        "smoke_bundle": {
            "path": str(smoke_root),
            "metadata_sha256": sha256_file(smoke_root / "metadata.json"),
            "scores_sha256": sha256_file(smoke_scores_path),
        },
        "first_round_bundles": identities,
        "formal_reuse_policy": {
            "reused": list(bundle_specs),
            "newly_computed": ["null_to_null_identity"],
            "baseline_scores": "frozen_first_round_full_and_null",
        },
        "qrels_read": False,
        "source_test_opened": False,
        "command": list(command or []),
    }
    _write_json(output_dir / "metrics.json", result)
    if not passed:
        raise ValueError(
            f"Q2 post-block reuse equivalence failed: {maximum} > {TOLERANCE}"
        )
    return result


def _load_smoke_scores(
    path: Path, records_by_id: Mapping[str, Any]
) -> dict[tuple[str, str], dict[str, float]]:
    result = {}
    for block_row in iter_jsonl(path):
        request_id = str(block_row.get("request_id") or "")
        record = records_by_id.get(request_id)
        if record is None:
            raise ValueError("Q2 smoke contains an unknown request")
        rows = block_row.get("rows")
        if not isinstance(rows, list) or len(rows) != len(record.candidates):
            raise ValueError("Q2 smoke candidate count differs from frozen record")
        for ordinal, (row, candidate) in enumerate(zip(rows, record.candidates)):
            item_id = str(candidate["item_id"])
            if (
                row.get("request_id") != request_id
                or row.get("candidate_item_id") != item_id
                or row.get("candidate_ordinal") != ordinal
            ):
                raise ValueError("Q2 smoke candidate identity/order drift")
            conditions = row.get("conditions")
            if not isinstance(conditions, dict) or set(conditions) != set(
                POSTBLOCK_CONDITIONS
            ):
                raise ValueError("Q2 smoke condition coverage drift")
            if any(not math.isfinite(float(value)) for value in conditions.values()):
                raise FloatingPointError("Q2 smoke score is non-finite")
            key = (request_id, item_id)
            if key in result:
                raise ValueError("Q2 smoke contains duplicate candidate identity")
            result[key] = {name: float(value) for name, value in conditions.items()}
    if not result:
        raise ValueError("Q2 smoke bundle is empty")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value
