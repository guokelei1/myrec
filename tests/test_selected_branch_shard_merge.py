from __future__ import annotations

import json
from pathlib import Path

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.mechanism.attention_edge_runtime import _canonical_sha256
from myrec.mechanism.representation_probe import normalized_query_fold
from myrec.mechanism.selected_branch_evaluator import _audit_selected_branch_bundle
from myrec.mechanism.selected_branch_runtime import (
    selected_branch_implementation_identity,
)
from myrec.mechanism.selected_branch_scoring import (
    SELECTED_NODES,
    selected_branch_conditions,
)
from myrec.mechanism.selected_branch_shard_merge import (
    merge_selected_branch_request_shards,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl


def _query_for_fold(target: int) -> str:
    for index in range(100):
        query = f"query-{index}"
        if normalized_query_fold(query) == target:
            return query
    raise AssertionError("could not construct a query fold fixture")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def test_selected_branch_request_shards_merge_to_exact_fold_order(tmp_path: Path):
    standardized = tmp_path / "standardized"
    standardized.mkdir()
    fold0_query = _query_for_fold(0)
    fold1_query = _query_for_fold(1)
    raw_records = []
    for index in range(8000):
        raw_records.append(
            {
                "request_id": f"r{index}",
                "query": fold1_query if index in (17, 7001) else fold0_query,
                "history": [],
                "candidates": [
                    {"item_id": f"i{index}-a"},
                    {"item_id": f"i{index}-b"},
                ],
            }
        )
    records_path = standardized / "records_dev.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in raw_records),
        encoding="utf-8",
    )
    for filename in ("candidate_manifest.json", "request_manifest.json", "manifest.json"):
        (standardized / filename).write_text("{}\n", encoding="utf-8")
    all_records = [sanitize_record_for_model(row) for row in raw_records]
    frozen_null_root = tmp_path / "frozen-null"
    frozen_null_root.mkdir()
    (frozen_null_root / "metadata.json").write_text(
        json.dumps(
            {
                "method_id": "q2_recranker_generalqwen",
                "checkpoint_id": "checkpoint",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (frozen_null_root / "scores.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "request_id": row["request_id"],
                    "candidate_item_id": candidate["item_id"],
                    "score": 1.0 if row["request_id"] == "r7001" else 0.0,
                },
                sort_keys=True,
            )
            + "\n"
            for row in raw_records
            for candidate in row["candidates"]
        ),
        encoding="utf-8",
    )
    frozen_null_identity = {
        "root": str(frozen_null_root),
        "metadata_sha256": sha256_file(frozen_null_root / "metadata.json"),
        "scores_sha256": sha256_file(frozen_null_root / "scores.jsonl"),
        "score_rows": 16000,
    }

    conditions = selected_branch_conditions()
    implementation = selected_branch_implementation_identity()
    postblock_digest = "postblock-digest"
    selection_path = tmp_path / "selection.json"
    selection = {
        "analysis_type": "transformer_deep_dive_d2_postblock_fold0_selection",
        "method_id": "q2_recranker_generalqwen",
        "checkpoint_id": "checkpoint",
        "selected_block": 20,
        "implementation_digest": postblock_digest,
    }
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    selection_identity = {
        "path": str(selection_path),
        "sha256": sha256_file(selection_path),
    }
    confirmation_path = tmp_path / "confirmation.json"
    confirmation = {
        "analysis_type": "transformer_deep_dive_d2_postblock_fold1_confirmation",
        "method_id": "q2_recranker_generalqwen",
        "checkpoint_id": "checkpoint",
        "selected_block": 20,
        "implementation_digest": postblock_digest,
        "selection": selection_identity,
    }
    confirmation_path.write_text(
        json.dumps(confirmation, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    confirmation_identity = {
        "path": str(confirmation_path),
        "sha256": sha256_file(confirmation_path),
    }
    contract_path = tmp_path / "contract.json"
    contract = {
        "contract_type": "transformer_deep_dive_d2_selected_branch_contract",
        "selected_block": 20,
        "method_id": "q2_recranker_generalqwen",
        "checkpoint_id": "checkpoint",
        "evidence_role": "registered_confirmatory",
        "qrels_values_exposed_to_scorer": False,
        "postblock_implementation_digest": postblock_digest,
        "selection": selection_identity,
        "confirmation": confirmation_identity,
    }
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    common = {
        "analysis_stage": "transformer_deep_dive_d2_selected_branch",
        "method_id": "q2_recranker_generalqwen",
        "checkpoint_id": "checkpoint",
        "checkpoint_files": [],
        "config_path": "config.yaml",
        "config_sha256": "config-sha",
        "training_metadata_sha256": "training-sha",
        "selected_block": 20,
        "selected_nodes": list(SELECTED_NODES),
        "branch_contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
            "evidence_role": "registered_confirmatory",
        },
        "evidence_role": "registered_confirmatory",
        "normalized_query_fold": 1,
        "full_population_request_count": 8000,
        "fold1_request_count": 2,
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized / "candidate_manifest.json"),
        "request_manifest_sha256": sha256_file(standardized / "request_manifest.json"),
        "dataset_manifest_sha256": sha256_file(standardized / "manifest.json"),
        "deep_dive_manifest_sha256": "manifest-sha",
        "cross_request_mapping_sha256": "cross-sha",
        "wrong_user_control": {"mapping_sha256": "wrong-sha"},
        "frozen_full_baseline": {"scores_sha256": "full-sha"},
        "frozen_null_baseline": frozen_null_identity,
        "score_conditions": list(conditions),
        "identity_tolerance": 1.0e-5,
        "random_direction_seed": 20_260_715,
        "wrong_user_ineligible_scoring": "copy_frozen_null_score",
        "implementation_identity": implementation,
        "qrels_read": False,
        "source_test_opened": False,
        "evidence_mode": "registered_mechanism_diagnostic_request_shard",
        "result_eligible": False,
        "complete_finite_score_coverage": True,
        "identity_passed": True,
        "status": "completed",
        "maximum_identity_delta": 0.0,
        "maximum_full_baseline_delta": 0.0,
        "maximum_null_baseline_delta": 0.0,
        "maximum_baseline_low_precision_ratio": 0.0,
        "maximum_direction_rms_reconstruction_error": 0.0,
        "shared_prompt_path_max_abs_delta": 0.0,
        "wrong_user_eligible_requests": 0,
        "elapsed_seconds": 5.0,
    }
    shard_dirs = []
    for shard_index, record_index in enumerate((17, 7001)):
        shard_dir = tmp_path / f"shard-{shard_index}"
        shard_dir.mkdir()
        shard_dirs.append(shard_dir)
        raw = raw_records[record_index]
        values = {condition: float(shard_index) for condition in conditions}
        rows = [
            {
                "request_id": raw["request_id"],
                "candidate_item_id": candidate["item_id"],
                "candidate_ordinal": candidate_ordinal,
                "wrong_user_eligible": False,
                "conditions": values,
            }
            for candidate_ordinal, candidate in enumerate(raw["candidates"])
        ]
        block = {
            "ordinal": 0,
            "request_id": raw["request_id"],
            "wrong_user_eligible": False,
            "rows": rows,
            "rows_sha256": sha256_text(_canonical(rows)),
        }
        scores_path = shard_dir / "scores.jsonl"
        scores_path.write_text(_canonical(block) + "\n", encoding="utf-8")
        request_shard = {
            "index": shard_index,
            "count": 2,
            "rule": "fold1_ordinal_mod_request_shard_count",
            "request_count": 1,
        }
        run_contract = {
            "schema_version": 1,
            "run_id": f"20260718_kuaisearch_mech_d2_q2_selected_shard{shard_index}_v1",
            "method_id": common["method_id"],
            "checkpoint_id": common["checkpoint_id"],
            "config_sha256": common["config_sha256"],
            "selected_block": common["selected_block"],
            "branch_contract": common["branch_contract"],
            "normalized_query_fold": 1,
            "target_requests": 1,
            "request_shard": request_shard,
            "score_conditions": list(conditions),
            "records_sha256": common["records_sha256"],
            "cross_request_mapping_sha256": common["cross_request_mapping_sha256"],
            "wrong_user_mapping_sha256": "wrong-sha",
            "full_scores_sha256": "full-sha",
                "null_scores_sha256": frozen_null_identity["scores_sha256"],
            "deep_dive_manifest_sha256": common["deep_dive_manifest_sha256"],
            "device": "cuda:0",
            "implementation_digest": implementation["digest"],
            "evidence_mode": common["evidence_mode"],
        }
        metadata = {
            **common,
            "run_id": run_contract["run_id"],
            "request_shard": request_shard,
            "run_contract": run_contract,
            "run_contract_sha256": _canonical_sha256(run_contract),
            "scores_path": str(scores_path),
            "scores_sha256": sha256_file(scores_path),
            "request_count": 1,
            "score_rows": 2,
        }
        (shard_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    output = tmp_path / "merged"
    result = merge_selected_branch_request_shards(
        standardized,
        shard_dirs,
        output,
        "20260718_kuaisearch_mech_d2_q2_selected_branch_fold1_v1",
    )
    assert result["status"] == "completed"
    assert result["result_eligible"] is True
    assert result["request_count"] == 2
    assert result["request_shard"]["aggregation"] == "complete_disjoint_union"
    assert [row["request_id"] for row in iter_jsonl(output / "scores.jsonl")] == [
        "r17",
        "r7001",
    ]
    fold1_records = [
        sanitize_record_for_model(row)
        for row in raw_records
        if normalized_query_fold(row["query"]) == 1
    ]
    audited = _audit_selected_branch_bundle(
        output,
        fold1_records,
        all_records=all_records,
    )
    assert audited.metadata["request_shard"]["aggregation"] == "complete_disjoint_union"
