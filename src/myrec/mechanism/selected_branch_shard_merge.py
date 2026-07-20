"""Qrels-blind deterministic merge for selected-branch request shards."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.baselines.motivation_v12_ranker import _validate_run_id
from myrec.mechanism.attention_edge_runtime import _canonical_sha256, _read_json
from myrec.mechanism.representation_probe import normalized_query_fold
from myrec.mechanism.scalar_condition_bundle import (
    append_scalar_request,
    audit_scalar_partial,
    finalize_scalar_bundle,
    prepare_scalar_bundle,
)
from myrec.mechanism.selected_branch_runtime import (
    _request_shard_records,
    selected_branch_implementation_identity,
)
from myrec.mechanism.selected_branch_scoring import (
    SELECTED_NODES,
    selected_branch_conditions,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


_COMMON_METADATA_FIELDS = (
    "method_id",
    "checkpoint_id",
    "checkpoint_files",
    "config_path",
    "config_sha256",
    "training_metadata_sha256",
    "selected_block",
    "selected_nodes",
    "branch_contract",
    "evidence_role",
    "normalized_query_fold",
    "full_population_request_count",
    "fold1_request_count",
    "records_sha256",
    "candidate_manifest_sha256",
    "request_manifest_sha256",
    "dataset_manifest_sha256",
    "deep_dive_manifest_sha256",
    "cross_request_mapping_sha256",
    "wrong_user_control",
    "frozen_full_baseline",
    "frozen_null_baseline",
    "score_conditions",
    "identity_tolerance",
    "random_direction_seed",
    "wrong_user_ineligible_scoring",
    "implementation_identity",
    "qrels_read",
    "source_test_opened",
)

_MAXIMUM_FIELDS = (
    "maximum_identity_delta",
    "maximum_full_baseline_delta",
    "maximum_null_baseline_delta",
    "maximum_baseline_low_precision_ratio",
    "maximum_direction_rms_reconstruction_error",
    "shared_prompt_path_max_abs_delta",
)


def merge_selected_branch_request_shards(
    standardized_dir: str | Path,
    shard_dirs: Sequence[str | Path],
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Merge a complete modulo partition into one evaluator-compatible bundle."""

    _validate_run_id(analysis_run_id)
    standardized_dir = Path(standardized_dir)
    output_dir = Path(output_dir)
    shard_dirs = tuple(Path(path) for path in shard_dirs)
    if len(shard_dirs) < 2:
        raise ValueError("selected-branch shard merge requires at least two shards")
    if len(set(map(str, shard_dirs))) != len(shard_dirs):
        raise ValueError("selected-branch shard paths must be unique")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"selected-branch merged output is not empty: {output_dir}"
        )

    records_path = standardized_dir / "records_dev.jsonl"
    all_records = [
        sanitize_record_for_model(row) for row in iter_jsonl(records_path)
    ]
    if len(all_records) != 8000:
        raise ValueError("selected-branch shard merge requires frozen 8000-request dev")
    fold1_records = [
        record for record in all_records if normalized_query_fold(record.query) == 1
    ]
    conditions = selected_branch_conditions()
    metadata_rows = [_read_json(path / "metadata.json") for path in shard_dirs]
    shard_count = len(shard_dirs)
    by_index: dict[int, tuple[Path, dict[str, Any], list[Any]]] = {}
    reference = metadata_rows[0]
    implementation = selected_branch_implementation_identity()

    for shard_dir, metadata in zip(shard_dirs, metadata_rows):
        expected = {
            "analysis_stage": "transformer_deep_dive_d2_selected_branch",
            "status": "completed",
            "result_eligible": False,
            "complete_finite_score_coverage": True,
            "identity_passed": True,
            "normalized_query_fold": 1,
            "selected_nodes": list(SELECTED_NODES),
            "score_conditions": list(conditions),
            "qrels_read": False,
            "source_test_opened": False,
            "evidence_mode": "registered_mechanism_diagnostic_request_shard",
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"selected-branch request shard metadata mismatch: {key}")
        for key in _COMMON_METADATA_FIELDS:
            if metadata.get(key) != reference.get(key):
                raise ValueError(f"selected-branch request shards differ: {key}")
        if metadata.get("implementation_identity") != implementation:
            raise ValueError("selected-branch request shard implementation bytes changed")
        request_shard = metadata.get("request_shard")
        if not isinstance(request_shard, Mapping):
            raise ValueError("selected-branch request shard declaration is missing")
        index = int(request_shard.get("index", -1))
        if (
            int(request_shard.get("count", -1)) != shard_count
            or request_shard.get("rule")
            != "fold1_ordinal_mod_request_shard_count"
            or index in by_index
            or not 0 <= index < shard_count
        ):
            raise ValueError("selected-branch request shard partition drift")
        shard_records = _request_shard_records(
            fold1_records,
            request_shard_index=index,
            request_shard_count=shard_count,
        )
        if request_shard.get("request_count") != len(shard_records):
            raise ValueError("selected-branch request shard count drift")
        contract = metadata.get("run_contract")
        if (
            not isinstance(contract, Mapping)
            or contract.get("request_shard") != request_shard
            or contract.get("implementation_digest") != implementation["digest"]
            or metadata.get("run_contract_sha256") != _canonical_sha256(contract)
        ):
            raise ValueError("selected-branch request shard run contract drift")
        scores_path = shard_dir / "scores.jsonl"
        if metadata.get("scores_sha256") != sha256_file(scores_path):
            raise ValueError("selected-branch request shard score bytes changed")
        observed = audit_scalar_partial(scores_path, shard_records, conditions)
        if (
            observed["completed_requests"] != len(shard_records)
            or observed["completed_score_rows"]
            != sum(len(record.candidates) for record in shard_records)
        ):
            raise ValueError("selected-branch request shard coverage drift")
        by_index[index] = (shard_dir, metadata, shard_records)

    if set(by_index) != set(range(shard_count)):
        raise ValueError("selected-branch request shard indices are incomplete")
    if sha256_file(records_path) != reference.get("records_sha256"):
        raise ValueError("selected-branch request shard records bytes changed")
    for field, filename in (
        ("candidate_manifest_sha256", "candidate_manifest.json"),
        ("request_manifest_sha256", "request_manifest.json"),
        ("dataset_manifest_sha256", "manifest.json"),
    ):
        if sha256_file(standardized_dir / filename) != reference.get(field):
            raise ValueError(f"selected-branch request shard input bytes changed: {filename}")

    source_shards = []
    block_rows: dict[str, dict[str, Any]] = {}
    for index in range(shard_count):
        shard_dir, metadata, shard_records = by_index[index]
        source_shards.append(
            {
                "index": index,
                "path": str(shard_dir),
                "metadata_sha256": sha256_file(shard_dir / "metadata.json"),
                "scores_sha256": sha256_file(shard_dir / "scores.jsonl"),
                "request_count": len(shard_records),
            }
        )
        for block in iter_jsonl(shard_dir / "scores.jsonl"):
            request_id = str(block["request_id"])
            if request_id in block_rows:
                raise ValueError("selected-branch request shards overlap")
            block_rows[request_id] = dict(block)
    if set(block_rows) != {record.request_id for record in fold1_records}:
        raise ValueError("selected-branch request shard union is incomplete")

    aggregate_shard = {
        "count": shard_count,
        "rule": "fold1_ordinal_mod_request_shard_count",
        "aggregation": "complete_disjoint_union",
        "request_count": len(fold1_records),
    }
    aggregate_contract = copy.deepcopy(reference["run_contract"])
    aggregate_contract.update(
        {
            "run_id": analysis_run_id,
            "target_requests": len(fold1_records),
            "request_shard": aggregate_shard,
            "device": "deterministic_request_shard_merge",
            "evidence_mode": "registered_mechanism_diagnostic",
            "source_shards": source_shards,
        }
    )
    aggregate_metadata = copy.deepcopy(reference)
    for key in (
        "scores_path",
        "scores_sha256",
        "request_count",
        "score_rows",
        "resume_lineage",
        "error",
    ):
        aggregate_metadata.pop(key, None)
    aggregate_metadata.update(
        {
            "run_id": analysis_run_id,
            "status": "initializing",
            "result_eligible": True,
            "evidence_mode": "registered_mechanism_diagnostic",
            "request_shard": aggregate_shard,
            "request_shard_sources": source_shards,
            "run_contract": aggregate_contract,
            "run_contract_sha256": _canonical_sha256(aggregate_contract),
            "command": list(command or ()),
            "source_gpu_seconds": sum(
                float(metadata.get("elapsed_seconds", 0.0))
                for metadata in metadata_rows
            ),
            "parallel_wall_seconds": max(
                float(metadata.get("elapsed_seconds", 0.0))
                for metadata in metadata_rows
            ),
            "wrong_user_eligible_requests": sum(
                int(metadata.get("wrong_user_eligible_requests", 0))
                for metadata in metadata_rows
            ),
        }
    )
    for field in _MAXIMUM_FIELDS:
        aggregate_metadata[field] = max(
            float(metadata.get(field, 0.0)) for metadata in metadata_rows
        )

    prepared = prepare_scalar_bundle(
        output_dir,
        metadata=aggregate_metadata,
        contract_sha256=aggregate_metadata["run_contract_sha256"],
        records=fold1_records,
        conditions=conditions,
        resume=False,
    )
    for ordinal, record in enumerate(fold1_records):
        block = block_rows[record.request_id]
        block["ordinal"] = ordinal
        append_scalar_request(output_dir, block, prepared)
    return finalize_scalar_bundle(
        output_dir,
        prepared,
        fold1_records,
        conditions,
        maximum_identity_delta=aggregate_metadata["maximum_identity_delta"],
    )
