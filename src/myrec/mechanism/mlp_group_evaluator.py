"""Qrels-blind aggregation for the frozen D4 SwiGLU group localization."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.mechanism.attention_edge_runtime import (
    DEEP_DIVE_MANIFEST_PATH,
    FIXED_BLOCKS,
    SUPPORTED_METHODS,
    _canonical_sha256,
    _load_manifest,
)
from myrec.mechanism.mlp_group_interventions import MLP_GROUPS
from myrec.mechanism.patch_scorer import _cross_request_mapping
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


GROUP_METRICS = (
    "same_minus_null",
    "cross_minus_null",
    "same_minus_cross",
    "full_rms",
    "null_rms",
    "full_hoyer_sparsity",
    "null_hoyer_sparsity",
    "full_null_cosine",
)


def evaluate_mlp_group_bundles(
    bundle_dirs: Mapping[str, Mapping[int, str | Path]],
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Report every fixed group and residual geometry without best-group selection."""

    if set(bundle_dirs) != set(SUPPORTED_METHODS) or any(
        set(map(int, blocks)) != set(FIXED_BLOCKS) for blocks in bundle_dirs.values()
    ):
        raise ValueError("D4 MLP evaluator requires Q2/Q3 x blocks 13/20/27")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"D4 MLP output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, cross_donors, frozen_identity = _load_frozen_mlp_sample_lineage()
    bundles = {
        method_id: {
            block: _audit_bundle(
                bundle_dirs[method_id][block],
                method_id,
                block,
                samples=samples,
                cross_donors=cross_donors,
                frozen_identity=frozen_identity,
            )
            for block in FIXED_BLOCKS
        }
        for method_id in SUPPORTED_METHODS
    }
    implementation_digest = _common_implementation_digest(bundles)
    pre_qrels = {
        "schema_version": 1,
        "analysis_type": "d4_mlp_groups_qrels_blind_integrity",
        "analysis_run_id": analysis_run_id,
        "qrels_read": False,
        "status": "passed",
        "checks": {
            "all_six_registered_bundles_present": True,
            "all_512_sample_rows_complete_finite": True,
            "all_16_groups_reported_without_selection": True,
            "same_group_score_identity_at_most_1e-5": True,
            "permutation_recomposition_within_frozen_bound": True,
            "rows_match_bound_frozen_candidate_sample": True,
            "cross_request_donors_match_frozen_mapping": True,
            "all_six_bundles_share_one_implementation_digest": True,
        },
        "implementation_digest": implementation_digest,
        "bundles": {
            method_id: {
                str(block): {
                    "path": str(bundle[0]),
                    "metadata_sha256": sha256_file(bundle[0] / "metadata.json"),
                    "rows_sha256": sha256_file(bundle[0] / "rows.jsonl"),
                }
                for block, bundle in blocks.items()
            }
            for method_id, blocks in bundles.items()
        },
    }
    integrity_path = output_dir / "integrity.json"
    _write_json(integrity_path, pre_qrels)
    results = {}
    for method_id, blocks in bundles.items():
        results[method_id] = {}
        for block, (_root, metadata, rows) in blocks.items():
            groups = []
            for group_id in range(MLP_GROUPS):
                group_rows = [row["result"]["groups"][group_id] for row in rows]
                if any(int(row["group_id"]) != group_id for row in group_rows):
                    raise ValueError("D4 MLP group order differs")
                groups.append(
                    {
                        "group_id": group_id,
                        **{
                            metric: _summary(
                                np.asarray([row[metric] for row in group_rows], dtype=np.float64)
                            )
                            for metric in GROUP_METRICS
                        },
                    }
                )
            residual = {}
            for condition in ("full", "null", "cross"):
                names = sorted(rows[0]["result"]["residual_geometry"][condition])
                residual[condition] = {
                    name: _summary(
                        np.asarray(
                            [row["result"]["residual_geometry"][condition][name] for row in rows],
                            dtype=np.float64,
                        )
                    )
                    for name in names
                }
            results[method_id][str(block)] = {
                "block_zero_based": block,
                "sample_rows": 512,
                "groups": groups,
                "residual_geometry": residual,
                "maximum_same_group_identity_delta": metadata[
                    "maximum_same_group_identity_delta"
                ],
                "maximum_permutation_low_precision_ratio": metadata[
                    "maximum_permutation_low_precision_ratio"
                ],
            }
    metrics = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d4_mlp_groups",
        "analysis_run_id": analysis_run_id,
        "methods": list(SUPPORTED_METHODS),
        "blocks": list(FIXED_BLOCKS),
        "groups": MLP_GROUPS,
        "sample_rows": 512,
        "descriptive_only": True,
        "selection": "all fixed groups reported; no outcome-selected neuron/group",
        "implementation_digest": implementation_digest,
        "qrels_read": False,
        "source_test_opened": False,
        "integrity_path": str(integrity_path),
        "integrity_sha256": sha256_file(integrity_path),
        "results": results,
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
        },
    )
    return metrics


def _audit_bundle(
    root,
    method_id,
    block,
    *,
    samples,
    cross_donors,
    frozen_identity,
):
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    expected = {
        "analysis_stage": "transformer_deep_dive_d4_mlp_groups",
        "method_id": method_id,
        "block_zero_based": block,
        "status": "completed",
        "result_eligible": True,
        "qrels_read": False,
        "source_test_opened": False,
        "row_count": 512,
        "identity_passed": True,
        "permutation_recomposition_passed": True,
        "permutation_recomposition_dtype": "float32",
        "permutation_bound_reference_dtype": "native_swiglu_product_dtype",
        **frozen_identity,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"D4 MLP metadata differs: {key}")
    if float(metadata.get("maximum_same_group_identity_delta", math.inf)) > 1.0e-5:
        raise ValueError("D4 MLP identity failed")
    if float(metadata.get("maximum_permutation_low_precision_ratio", math.inf)) > 1.0:
        raise ValueError("D4 MLP permutation bound failed")
    path = root / "rows.jsonl"
    if metadata.get("rows_sha256") != sha256_file(path):
        raise ValueError("D4 MLP rows hash differs")
    rows = list(iter_jsonl(path))
    if len(rows) != 512 or len(samples) != 512 or len(cross_donors) != 512:
        raise ValueError("D4 MLP row coverage differs")
    for index, (row, sample, donor) in enumerate(
        zip(rows, samples, cross_donors)
    ):
        result = row.get("result", {})
        groups = result.get("groups", [])
        if (
            len(groups) != MLP_GROUPS
            or result.get("permutation_recomposition_dtype") != "float32"
            or result.get("permutation_bound_reference_dtype") != "bfloat16"
            or not _all_finite(row)
        ):
            raise ValueError("D4 MLP group/finite coverage differs")
        _audit_mlp_sample_and_donor_row(
            row,
            sample,
            donor,
            index=index,
            block=block,
        )
    return root, metadata, rows


def _load_frozen_mlp_sample_lineage():
    manifest_path = Path(DEEP_DIVE_MANIFEST_PATH)
    manifest = _load_manifest(manifest_path)
    sample_contract = manifest["frozen_qrels_blind_controls"][
        "fixed_high_dimensional_sample"
    ]
    sample_manifest_path = Path(sample_contract["manifest_path"])
    if sha256_file(sample_manifest_path) != sample_contract["manifest_sha256"]:
        raise ValueError("D4 MLP frozen sample manifest hash mismatch")
    sample_manifest = json.loads(
        sample_manifest_path.read_text(encoding="utf-8")
    )
    sample_path = Path(str(sample_manifest.get("path") or ""))
    if (
        sample_manifest.get("selected_candidate_rows") != 512
        or sample_manifest.get("qrels_read") is not False
        or sample_manifest.get("model_scores_read") is not False
        or not sample_path.is_file()
        or sample_contract["rows_sha256"] != sha256_file(sample_path)
        or sample_manifest.get("sha256") != sample_contract["rows_sha256"]
    ):
        raise ValueError("D4 MLP frozen sample bytes differ")
    samples = list(iter_jsonl(sample_path))
    if len(samples) != 512:
        raise ValueError("D4 MLP frozen sample coverage differs")

    records_path = (
        Path(manifest["frozen_inputs"]["standardized_dir"])
        / "records_dev.jsonl"
    )
    records_sha256 = sha256_file(records_path)
    if records_sha256 != manifest["frozen_inputs"]["records_dev_sha256"]:
        raise ValueError("D4 MLP frozen records hash mismatch")
    records = [
        sanitize_record_for_model(row) for row in iter_jsonl(records_path)
    ]
    if len(records) != 8000:
        raise ValueError("D4 MLP frozen record coverage differs")
    records_by_id = {record.request_id: record for record in records}
    if len(records_by_id) != len(records):
        raise ValueError("D4 MLP frozen request is duplicated")
    cross_mapping = _cross_request_mapping(records)
    cross_donors = []
    for index, sample in enumerate(samples):
        request_id = str(sample["request_id"])
        record = records_by_id.get(request_id)
        candidate_ordinal = int(sample["candidate_ordinal"])
        if (
            record is None
            or not 0 <= candidate_ordinal < len(record.candidates)
            or str(record.candidates[candidate_ordinal]["item_id"])
            != str(sample["candidate_item_id"])
        ):
            raise ValueError(
                f"D4 MLP frozen sample differs from dev records: {index}"
            )
        donor_request_id = cross_mapping[request_id]
        donor_record = records_by_id[donor_request_id]
        donor_ordinal = candidate_ordinal % len(donor_record.candidates)
        cross_donors.append(
            {
                "donor_request_id": donor_request_id,
                "donor_candidate_ordinal": donor_ordinal,
                "donor_candidate_item_id": str(
                    donor_record.candidates[donor_ordinal]["item_id"]
                ),
            }
        )
    return samples, cross_donors, {
        "records_sha256": records_sha256,
        "sample_manifest_sha256": sample_contract["manifest_sha256"],
        "sample_rows_sha256": sample_contract["rows_sha256"],
        "cross_mapping_sha256": _canonical_sha256(cross_mapping),
        "deep_dive_manifest_sha256": manifest["_sha256"],
    }


def _audit_mlp_sample_and_donor_row(
    row,
    sample,
    donor,
    *,
    index,
    block,
):
    expected = {
        "ordinal": index,
        "request_id": str(sample["request_id"]),
        "candidate_ordinal": int(sample["candidate_ordinal"]),
        "candidate_item_id": str(sample["candidate_item_id"]),
        "block_zero_based": block,
        **donor,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"D4 MLP row differs from frozen lineage: {key}")


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
        raise ValueError("D4 MLP bundles use different implementation digests")
    digest = next(iter(digests))
    if any(
        metadata.get("run_contract", {}).get("implementation_digest") != digest
        for metadata in metadata_rows
    ):
        raise ValueError("D4 MLP implementation differs from run contract")
    return digest


def _summary(values: np.ndarray) -> dict[str, float | int]:
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("D4 summary values differ")
    return {
        "rows": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
