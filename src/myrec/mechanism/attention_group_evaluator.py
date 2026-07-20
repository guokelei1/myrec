"""Qrels-blind aggregation for fixed GQA-group causal localization."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.mechanism.attention_edge_runtime import _canonical_sha256
from myrec.mechanism.attention_edge_evaluator import _append_jsonl, _write_json
from myrec.mechanism.attention_group_runtime import (
    CONTENT_MANIFEST,
    FIXED_BLOCKS,
    SUPPORTED_METHODS,
)
from myrec.mechanism.attention_group_scoring import (
    GROUP_CONDITIONS,
    SUPPLEMENTAL_CONDITIONS,
)
from myrec.mechanism.attention_observation_runtime import (
    SAMPLE_MANIFEST,
    SAMPLE_MANIFEST_SHA256,
)
from myrec.mechanism.patch_scorer import _cross_request_mapping
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


ACTIVE_GROUP_CONDITIONS = (
    "history_to_readout_logits_mask",
    "history_to_readout_value_zero",
    "neutral_history_kv",
)
ACTIVE_SUPPLEMENTAL_CONDITIONS = (
    "query_to_history_logits_mask",
    "query_to_history_value_zero",
    "cross_request_history_summary_kv",
)


def evaluate_attention_group_bundles(
    bundles: Mapping[str, Mapping[int, str | Path]],
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Report all groups and supplemental edges without selecting a winner."""

    if set(bundles) != set(SUPPORTED_METHODS) or any(
        set(map(int, blocks)) != set(FIXED_BLOCKS) for blocks in bundles.values()
    ):
        raise ValueError("attention GQA evaluator requires Q2/Q3 x blocks 13/20/27")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"attention GQA output is not empty: {output_dir}")
    sample_manifest = Path(SAMPLE_MANIFEST)
    if sha256_file(sample_manifest) != SAMPLE_MANIFEST_SHA256:
        raise ValueError("attention GQA evaluator sample manifest differs")
    frozen_inputs = {
        method_id: _load_frozen_sample_and_controls(method_id)
        for method_id in SUPPORTED_METHODS
    }
    audited = {
        method_id: {
            block: _audit_bundle(
                path,
                method_id,
                block,
                samples=frozen_inputs[method_id][0],
                content_eligibility=frozen_inputs[method_id][1],
                cross_donors=frozen_inputs[method_id][2],
                cross_mapping_sha256=frozen_inputs[method_id][3],
            )
            for block, path in blocks.items()
        }
        for method_id, blocks in bundles.items()
    }
    implementation_digest = _common_implementation_digest(audited)
    pre_qrels = {
        "schema_version": 1,
        "analysis_type": "d3_attention_gqa_qrels_blind_integrity",
        "analysis_run_id": analysis_run_id,
        "status": "passed",
        "qrels_read": False,
        "checks": {
            "all_six_fixed_bundles_present": True,
            "all_512_rows_each_complete_finite": True,
            "all_eight_groups_reported_without_selection": True,
            "group_and_formation_identity_at_most_1e-5": True,
            "native_baseline_identity_at_most_1e-5": True,
            "query_formation_and_history_transport_separated": True,
            "cross_request_history_summary_kv_present": True,
            "rows_match_bound_frozen_candidate_sample": True,
            "neutral_eligibility_matches_bound_frozen_controls": True,
            "ineligible_neutral_groups_equal_baseline": True,
            "cross_request_donors_match_frozen_mapping": True,
            "all_six_bundles_share_one_implementation_digest": True,
        },
        "implementation_digest": implementation_digest,
        "bundles": {
            method_id: {
                str(block): {
                    "path": str(root),
                    "metadata_sha256": sha256_file(root / "metadata.json"),
                    "groups_sha256": sha256_file(root / "groups.jsonl"),
                }
                for block, (root, _metadata, _rows) in blocks.items()
            }
            for method_id, blocks in audited.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    pre_qrels_path = output_dir / "pre_qrels_audit.json"
    _write_json(pre_qrels_path, pre_qrels)

    results: dict[str, Any] = {}
    for method_id, blocks in audited.items():
        results[method_id] = {}
        for block, (_root, metadata, rows) in blocks.items():
            group_results = []
            for group in range(8):
                group_rows = [row["result"]["groups"][group] for row in rows]
                if any(int(row["gqa_group"]) != group for row in group_rows):
                    raise ValueError("attention GQA group order differs")
                baseline = np.asarray(
                    [row["result"]["supplemental"]["baseline_full"] for row in rows],
                    dtype=np.float64,
                )
                group_results.append(
                    {
                        "gqa_group": group,
                        "query_heads": [2 * group, 2 * group + 1],
                        "effects_relative_to_baseline": {
                            condition: _summary(
                                np.asarray(
                                    [
                                        row["conditions"][condition]
                                        for row in group_rows
                                    ],
                                    dtype=np.float64,
                                )
                                - baseline
                            )
                            for condition in ACTIVE_GROUP_CONDITIONS
                        },
                    }
                )
            baseline = np.asarray(
                [row["result"]["supplemental"]["baseline_full"] for row in rows],
                dtype=np.float64,
            )
            supplemental = {
                condition: _summary(
                    np.asarray(
                        [row["result"]["supplemental"][condition] for row in rows],
                        dtype=np.float64,
                    )
                    - baseline
                )
                for condition in ACTIVE_SUPPLEMENTAL_CONDITIONS
            }
            results[method_id][str(block)] = {
                "sample_rows": len(rows),
                "groups": group_results,
                "supplemental_effects_relative_to_baseline": supplemental,
                "neutral_history_eligible_rows": int(
                    sum(
                        row["result"].get("neutral_history_eligible") is True
                        for row in rows
                    )
                ),
                "maximum_identity_delta": float(
                    metadata["maximum_identity_delta"]
                ),
                "maximum_baseline_delta": float(
                    metadata["maximum_baseline_delta"]
                ),
            }
    metrics = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d3_attention_gqa_causal_localization",
        "analysis_run_id": analysis_run_id,
        "methods": list(SUPPORTED_METHODS),
        "blocks": list(FIXED_BLOCKS),
        "gqa_groups": 8,
        "query_heads_per_group": 2,
        "sample_rows_per_bundle": 512,
        "evidence_mode": "exploratory_localization",
        "confirmatory_family_membership": False,
        "selection": "all groups reported; no outcome-selected head/group",
        "implementation_digest": implementation_digest,
        "results": results,
        "pre_qrels_audit_path": str(pre_qrels_path),
        "pre_qrels_audit_sha256": sha256_file(pre_qrels_path),
        "qrels_read": False,
        "source_test_opened": False,
        "command": list(command or []),
        "status": "completed",
    }
    metrics_path = output_dir / "metrics.json"
    _write_json(metrics_path, metrics)
    _append_jsonl(
        Path(dev_eval_log_path),
        {
            "analysis_type": metrics["analysis_type"],
            "run_id": analysis_run_id,
            "method_ids": list(SUPPORTED_METHODS),
            "split": "dev_qrels_blind_fixed_candidate_sample",
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
            "qrels_read": False,
        },
    )
    return metrics


def _audit_bundle(
    path,
    method_id,
    block,
    *,
    samples,
    content_eligibility,
    cross_donors,
    cross_mapping_sha256,
):
    root = Path(path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    sample_manifest_value = json.loads(
        Path(SAMPLE_MANIFEST).read_text(encoding="utf-8")
    )
    content_manifest_value = json.loads(
        Path(CONTENT_MANIFEST).read_text(encoding="utf-8")
    )
    content_method = content_manifest_value["methods"][method_id]
    expected = {
        "analysis_stage": "transformer_deep_dive_d3_attention_gqa_causal_localization",
        "method_id": method_id,
        "block_zero_based": block,
        "status": "completed",
        "result_eligible": True,
        "identity_passed": True,
        "complete_finite_group_coverage": True,
        "qrels_read": False,
        "source_test_opened": False,
        "gqa_groups": 8,
        "group_conditions": list(GROUP_CONDITIONS),
        "supplemental_conditions": list(SUPPLEMENTAL_CONDITIONS),
        "sample_manifest_sha256": SAMPLE_MANIFEST_SHA256,
        "sample_rows_sha256": sample_manifest_value["sha256"],
        "content_neutral_manifest_sha256": sha256_file(CONTENT_MANIFEST),
        "content_neutral_rows_sha256": content_method["sha256"],
        "cross_request_mapping_sha256": cross_mapping_sha256,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"attention GQA metadata differs: {key}")
    if float(metadata.get("maximum_identity_delta", math.inf)) > 1.0e-5:
        raise ValueError("attention GQA identity failed")
    if float(metadata.get("maximum_baseline_delta", math.inf)) > 1.0e-5:
        raise ValueError("attention GQA baseline identity failed")
    groups_path = root / "groups.jsonl"
    if metadata.get("groups_sha256") != sha256_file(groups_path):
        raise ValueError("attention GQA rows hash differs")
    rows = list(iter_jsonl(groups_path))
    if len(rows) != 512 or len(samples) != 512:
        raise ValueError("attention GQA fixed sample coverage differs")
    for index, (row, sample) in enumerate(zip(rows, samples)):
        groups = row.get("result", {}).get("groups", [])
        supplemental = row.get("result", {}).get("supplemental", {})
        if (
            row.get("row_index") != index
            or len(groups) != 8
            or set(supplemental) != set(SUPPLEMENTAL_CONDITIONS)
            or any(set(group.get("conditions", {})) != set(GROUP_CONDITIONS) for group in groups)
            or not _all_finite(row)
        ):
            raise ValueError("attention GQA row coverage/finite audit differs")
        _audit_frozen_sample_and_eligibility_row(
            row,
            sample,
            content_eligibility,
            cross_donors,
        )
    return root, metadata, rows


def _load_frozen_sample_and_controls(method_id):
    sample_manifest_path = Path(SAMPLE_MANIFEST)
    sample_manifest = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
    sample_path = Path(str(sample_manifest.get("path") or ""))
    if (
        sample_manifest.get("selected_candidate_rows") != 512
        or sample_manifest.get("qrels_read") is not False
        or sample_manifest.get("model_scores_read") is not False
        or not sample_path.is_file()
        or sample_manifest.get("sha256") != sha256_file(sample_path)
    ):
        raise ValueError("attention GQA frozen sample bytes differ")
    samples = list(iter_jsonl(sample_path))

    content_manifest_path = Path(CONTENT_MANIFEST)
    content_manifest = json.loads(
        content_manifest_path.read_text(encoding="utf-8")
    )
    method = content_manifest.get("methods", {}).get(method_id, {})
    content_path = Path(str(method.get("path") or ""))
    if (
        content_manifest.get("qrels_read") is not False
        or content_manifest.get("model_scores_read") is not False
        or not content_path.is_file()
        or method.get("sha256") != sha256_file(content_path)
        or int(method.get("requests", -1)) != 8000
        or int(method.get("eligible_requests", -1)) != 7254
    ):
        raise ValueError("attention GQA frozen content-control bytes differ")
    content_rows = list(iter_jsonl(content_path))
    if len(content_rows) != 8000:
        raise ValueError("attention GQA frozen content-control coverage differs")
    eligibility = {}
    for control in content_rows:
        eligible = control.get("eligible")
        if not isinstance(eligible, bool):
            raise ValueError("attention GQA frozen eligibility is not boolean")
        request_id = str(control.get("request_id"))
        if request_id in eligibility:
            raise ValueError("attention GQA frozen control request is duplicated")
        eligibility[request_id] = eligible
    if sum(eligibility.values()) != 7254:
        raise ValueError("attention GQA frozen eligible count differs")
    records_path = Path(str(content_manifest.get("target_records_path") or ""))
    if (
        not records_path.is_file()
        or content_manifest.get("target_records_sha256")
        != sha256_file(records_path)
        or sample_manifest.get("target_records_sha256")
        != sha256_file(records_path)
    ):
        raise ValueError("attention GQA frozen records bytes differ")
    records = [
        sanitize_record_for_model(row) for row in iter_jsonl(records_path)
    ]
    if len(records) != 8000:
        raise ValueError("attention GQA frozen record coverage differs")
    records_by_id = {record.request_id: record for record in records}
    cross_mapping = _cross_request_mapping(records)
    cross_donors = {}
    for sample in samples:
        request_id = str(sample["request_id"])
        donor_request_id = cross_mapping[request_id]
        donor_record = records_by_id[donor_request_id]
        donor_ordinal = int(sample["candidate_ordinal"]) % len(
            donor_record.candidates
        )
        selection = str(sample["selection_sha256"])
        if selection in cross_donors:
            raise ValueError("attention GQA frozen selection is duplicated")
        cross_donors[selection] = {
            "donor_request_id": donor_request_id,
            "donor_candidate_item_id": str(
                donor_record.candidates[donor_ordinal]["item_id"]
            ),
        }
    return samples, eligibility, cross_donors, _canonical_sha256(cross_mapping)


def _audit_frozen_sample_and_eligibility_row(
    row,
    sample,
    content_eligibility,
    cross_donors,
):
    for key in (
        "selection_sha256",
        "request_id",
        "candidate_item_id",
        "candidate_ordinal",
    ):
        if row.get(key) != sample.get(key):
            raise ValueError(f"attention GQA row differs from frozen sample: {key}")
    request_id = str(row["request_id"])
    if request_id not in content_eligibility:
        raise ValueError("attention GQA sample request is absent from frozen controls")
    selection = str(row["selection_sha256"])
    if selection not in cross_donors:
        raise ValueError("attention GQA selection is absent from frozen cross mapping")
    for key, value in cross_donors[selection].items():
        if row.get(key) != value:
            raise ValueError(f"attention GQA donor differs from frozen mapping: {key}")
    expected = content_eligibility[request_id]
    result = row.get("result", {})
    observed = result.get("neutral_history_eligible")
    if not isinstance(observed, bool) or observed is not expected:
        raise ValueError("attention GQA neutral eligibility differs from frozen control")
    if not expected:
        baseline = float(result["supplemental"]["baseline_full"])
        if any(
            float(group["conditions"]["neutral_history_kv"]) != baseline
            for group in result["groups"]
        ):
            raise ValueError(
                "attention GQA ineligible neutral group differs from baseline"
            )


def _common_implementation_digest(bundles):
    metadata_rows = [
        metadata
        for blocks in bundles.values()
        for _root, metadata, _rows in blocks.values()
    ]
    digests = {
        str(metadata.get("implementation_identity", {}).get("digest") or "")
        for metadata in metadata_rows
    }
    if len(digests) != 1 or not next(iter(digests), ""):
        raise ValueError(
            "attention GQA bundles use different implementation digests"
        )
    digest = next(iter(digests))
    if any(
        metadata.get("run_contract", {}).get("implementation_digest") != digest
        for metadata in metadata_rows
    ):
        raise ValueError("attention GQA implementation differs from run contract")
    return digest


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("attention GQA summary values are invalid")
    return {
        "rows": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "mean_absolute": float(np.abs(values).mean()),
    }


def _all_finite(value):
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True
