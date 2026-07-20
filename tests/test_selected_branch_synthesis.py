from __future__ import annotations

import json

import pytest

from myrec.mechanism.selected_branch_evaluator import (
    ENDPOINTS,
    selected_branch_contrast_specs,
)
from myrec.mechanism.selected_branch_synthesis import (
    MODELS,
    NEGATIVE_EXPECTED_GROUPS,
    SELECTED_BRANCH_FOLD_SCOPE,
    synthesize_selected_branches,
)
from myrec.utils.hashing import sha256_file


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _valid_metrics(tmp_path, method_id=MODELS[0]):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_json(bundle_dir / "metadata.json", {"status": "completed"})
    (bundle_dir / "scores.jsonl").write_text(
        '{"request_id":"r1","candidate_id":"c1","score":0.0}\n',
        encoding="utf-8",
    )
    input_bundle = {
        "path": str(bundle_dir),
        "metadata_sha256": sha256_file(bundle_dir / "metadata.json"),
        "scores_sha256": sha256_file(bundle_dir / "scores.jsonl"),
    }
    per_request_path = tmp_path / "per_request_contrasts.npz"
    per_request_path.write_bytes(b"fixed-test-npz")
    implementation_digest = "selected-branch-test-implementation-v1"
    pre_qrels = {
        "schema_version": 1,
        "analysis_type": "d2_selected_branch_fold1_pre_qrels_integrity",
        "analysis_run_id": "eval-run",
        "status": "passed",
        "method_id": method_id,
        "selected_block": 13,
        "qrels_read": False,
        "checks": {
            "fold1_request_candidate_coverage_complete_finite": True,
            "all_14_identity_controls_at_most_1e-5": True,
            "frozen_baseline_recompute_within_path_local_bf16_bound": True,
            "wrong_user_ineligible_scores_equal_frozen_null": True,
            "candidate_and_request_manifests_reconstructed": True,
            "minimal_selected_branch_contract_bound": True,
            "selected_branch_implementation_digest_bound": True,
        },
        "implementation_digest": implementation_digest,
        "bundle": input_bundle,
    }
    pre_qrels_path = tmp_path / "pre_qrels_audit.json"
    _write_json(pre_qrels_path, pre_qrels)

    results = {}
    family_rows = []
    for contrast_id, spec in selected_branch_contrast_specs().items():
        endpoints = {}
        for endpoint in ENDPOINTS:
            inference = {
                "two_sided_p": 0.5,
                "mean": -0.01,
                "ci95": [-0.02, 0.01],
            }
            endpoints[endpoint] = dict(inference)
            family_rows.append(
                {
                    "contrast_id": contrast_id,
                    "group": spec["group"],
                    "endpoint": endpoint,
                    **inference,
                }
            )
        results[contrast_id] = {
            "group": spec["group"],
            "node": spec.get("node"),
            "left_node": spec.get("left_node"),
            "right_node": spec.get("right_node"),
            "control": spec.get("control"),
            "eligible_surface": (
                "strict_transfer_and_frozen_wrong_user_eligible"
                if spec["group"] == "same_minus_wrong_history"
                else "strict_transfer"
            ),
            "endpoints": endpoints,
        }
    metrics = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d2_selected_branch",
        "analysis_run_id": "eval-run",
        "status": "completed",
        "method_id": method_id,
        "selected_block": 13,
        "evidence_role": "registered_confirmatory_branch_localization",
        "implementation_digest": implementation_digest,
        "normalized_query_fold": 1,
        "fold_scope": dict(SELECTED_BRANCH_FOLD_SCOPE),
        "strict_transfer_requests": 100,
        "strict_transfer_wrong_user_eligible_requests": 80,
        "bootstrap": {
            "cluster": "normalized_query",
            "samples": 5000,
            "seed": 20260715,
        },
        "family_policy": {
            "BH_applied_only_in_two_model_synthesis": True,
            "per_endpoint_separate_families": True,
            "planned_two_model_units": {
                "same": 14,
                "same_minus_cross": 14,
                "same_minus_wrong_history": 14,
                "adjacent_node": 12,
                "direction_scale": 42,
            },
        },
        "results": results,
        "family_rows": family_rows,
        "input_bundle": input_bundle,
        "pre_qrels_audit_path": str(pre_qrels_path),
        "pre_qrels_audit_sha256": sha256_file(pre_qrels_path),
        "qrels_read": True,
        "qrels_fold_opened": 1,
        "other_fold_qrels_opened": False,
        "qrels_fold_sha256": "a" * 64,
        "qrels_split_manifest_sha256": "b" * 64,
        "qrels_source_sha256": "c" * 64,
        "per_request_contrasts_path": str(per_request_path),
        "per_request_contrasts_sha256": sha256_file(per_request_path),
    }
    metrics_path = tmp_path / "metrics.json"
    _write_json(metrics_path, metrics)
    return metrics_path, metrics, bundle_dir


def test_selected_branch_synthesis_freezes_models_and_sign_groups():
    assert MODELS == (
        "q2_recranker_generalqwen",
        "q3_tallrec_generalqwen",
    )
    assert "adjacent_node" not in NEGATIVE_EXPECTED_GROUPS
    assert "same_minus_wrong_history" in NEGATIVE_EXPECTED_GROUPS


def test_selected_branch_synthesis_keeps_all_missing_cells(tmp_path):
    result = synthesize_selected_branches({}, tmp_path / "out", "run")
    assert result["fold_scope"] == SELECTED_BRANCH_FOLD_SCOPE
    assert (
        result["fold_scope"]["node_effect_two_fold_replication_tested"]
        is False
    )
    assert len(result["families"]["same__target_margin"]["rows"]) == 14
    assert result["families"]["direction_scale__ndcg@10"][
        "missing_cells_fixed_p1"
    ] == 42


def test_selected_branch_synthesis_rejects_missing_implementation_digest(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "analysis_type": "transformer_deep_dive_d2_selected_branch",
                "status": "completed",
                "method_id": MODELS[0],
                "normalized_query_fold": 1,
                "fold_scope": dict(SELECTED_BRANCH_FOLD_SCOPE),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="implementation digest is missing"):
        synthesize_selected_branches(
            {MODELS[0]: metrics_path}, tmp_path / "out", "run"
        )


def test_selected_branch_synthesis_rejects_fold_scope_drift(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    drifted_scope = dict(SELECTED_BRANCH_FOLD_SCOPE)
    drifted_scope["node_effect_two_fold_replication_tested"] = True
    metrics_path.write_text(
        json.dumps(
            {
                "analysis_type": "transformer_deep_dive_d2_selected_branch",
                "status": "completed",
                "method_id": MODELS[0],
                "normalized_query_fold": 1,
                "fold_scope": drifted_scope,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fold scope differs"):
        synthesize_selected_branches(
            {MODELS[0]: metrics_path}, tmp_path / "out", "run"
        )


def test_selected_branch_synthesis_binds_evaluator_and_input_bytes(tmp_path):
    metrics_path, _, _ = _valid_metrics(tmp_path)
    result = synthesize_selected_branches(
        {MODELS[0]: metrics_path}, tmp_path / "out", "run"
    )
    identity = result["input_metrics"][MODELS[0]]
    assert identity["evaluator_tables_cross_checked"] is True
    assert identity["pre_qrels_audit_sha256"] == sha256_file(
        tmp_path / "pre_qrels_audit.json"
    )
    assert identity["per_request_contrasts_sha256"] == sha256_file(
        tmp_path / "per_request_contrasts.npz"
    )


def test_selected_branch_synthesis_rejects_result_family_row_drift(tmp_path):
    metrics_path, metrics, _ = _valid_metrics(tmp_path)
    metrics["family_rows"][0]["mean"] = -0.02
    _write_json(metrics_path, metrics)
    with pytest.raises(ValueError, match="result/family row drift"):
        synthesize_selected_branches(
            {MODELS[0]: metrics_path}, tmp_path / "out", "run"
        )


def test_selected_branch_synthesis_rejects_tampered_bundle_bytes(tmp_path):
    metrics_path, _, bundle_dir = _valid_metrics(tmp_path)
    (bundle_dir / "scores.jsonl").write_text(
        '{"request_id":"r1","candidate_id":"c1","score":1.0}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="input scores bytes differ"):
        synthesize_selected_branches(
            {MODELS[0]: metrics_path}, tmp_path / "out", "run"
        )
