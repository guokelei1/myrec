from __future__ import annotations

import json

import pytest

from myrec.mechanism.deep_dive_progress import audit_deep_dive_progress


def _metadata(root, short, block, fold, status="completed"):
    run_id = (
        f"20260718_kuaisearch_mech_d2_{short}_postblock_"
        f"b{block}_fold{fold}_v1"
    )
    path = root / "runs" / run_id / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": status,
                "result_eligible": True,
                "method_id": (
                    "q2_recranker_generalqwen"
                    if short == "q2"
                    else "q3_tallrec_generalqwen"
                ),
                "block_zero_based": block,
                "normalized_query_fold": fold,
                "evidence_mode": "registered_mechanism_diagnostic",
                "qrels_read": False,
                "source_test_opened": False,
                "run_contract_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )


def _contract(root, short, eligible):
    path = (
        root
        / "runs"
        / f"20260718_kuaisearch_mech_d2_{short}_selected_branch_contract_v1"
        / "contract.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "contract_type": "transformer_deep_dive_d2_selected_branch_contract",
                "status": "completed",
                "branch_scoring_eligible": eligible,
                "method_id": (
                    "q2_recranker_generalqwen"
                    if short == "q2"
                    else "q3_tallrec_generalqwen"
                ),
                "selected_block": 20 if eligible else None,
                "fold1_negative_transition_reproduced": bool(eligible),
                "evidence_role": (
                    "registered_confirmatory_branch_localization"
                    if eligible
                    else "stopped_no_negative_fold0_transition"
                ),
                "scoring_population": "normalized_query_fold_1",
                "selected_nodes": [
                    "block_input_residual",
                    "input_rmsnorm_output",
                    "attention_o_projection",
                    "post_attention_residual",
                    "post_attention_rmsnorm_output",
                    "mlp_down_projection",
                    "block_output_residual",
                ],
                "qrels_values_exposed_to_scorer": False,
                "source_test_opened": False,
            }
        ),
        encoding="utf-8",
    )


def _selected_shard(
    root,
    short,
    shard_index,
    *,
    completed_requests,
    status="running",
    fold_request_count=10,
):
    method_id = (
        "q2_recranker_generalqwen"
        if short == "q2"
        else "q3_tallrec_generalqwen"
    )
    run_id = (
        f"20260718_kuaisearch_mech_d2_{short}_selected_branch_"
        f"fold1_shard{shard_index}of2_v1"
    )
    path = root / "runs" / run_id / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    shard_request_count = (
        (fold_request_count + 1) // 2
        if shard_index == 0
        else fold_request_count // 2
    )
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": status,
                "method_id": method_id,
                "normalized_query_fold": 1,
                "evidence_mode": "registered_mechanism_diagnostic_request_shard",
                "result_eligible": False,
                "qrels_read": False,
                "source_test_opened": False,
                "selected_nodes": [
                    "block_input_residual",
                    "input_rmsnorm_output",
                    "attention_o_projection",
                    "post_attention_residual",
                    "post_attention_rmsnorm_output",
                    "mlp_down_projection",
                    "block_output_residual",
                ],
                "fold1_request_count": fold_request_count,
                "request_shard": {
                    "index": shard_index,
                    "count": 2,
                    "request_count": shard_request_count,
                    "rule": "fold1_ordinal_mod_request_shard_count",
                },
                "run_contract_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    if status != "completed":
        (path.parent / "progress.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "completed_requests": completed_requests,
                    "run_contract_sha256": "c" * 64,
                    "updated_at": "2026-07-19T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )


def test_progress_counts_fixed_and_conditional_bundles_without_effects(tmp_path):
    for block in range(13, 24):
        _metadata(tmp_path, "q2", block, 0)
    for block in range(13, 25):
        _metadata(tmp_path, "q3", block, 0)
    _metadata(tmp_path, "q2", 24, 0, status="running")
    run_dir = (
        tmp_path
        / "runs"
        / "20260718_kuaisearch_mech_d2_q2_postblock_b24_fold0_v1"
    )
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["fold_request_count"] = 4082
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "progress.json").write_text(
        json.dumps(
            {
                "status": "running",
                "completed_requests": 1021,
                "run_contract_sha256": "a" * 64,
                "updated_at": "2026-07-18T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    result = audit_deep_dive_progress(tmp_path)
    assert result["status"] == "ok"
    assert result["fixed"]["completed_bundles"] == 23
    assert result["fixed"]["remaining_mandatory_bundles"] == 37
    assert result["fixed"]["in_flight_bundles"] == 1
    in_flight_equivalent = 1021 / 4082
    executed_equivalents = 23 + in_flight_equivalent
    assert result["fixed"]["in_flight_bundle_equivalents"] == (
        in_flight_equivalent
    )
    assert result["fixed"]["request_weighted_executed_bundle_equivalents"] == (
        executed_equivalents
    )
    assert result["fixed"]["request_weighted_execution_fraction"] == (
        executed_equivalents / 60
    )
    assert result["fixed"]["request_weighted_remaining_bundle_equivalents"] == (
        60 - executed_equivalents
    )
    assert result["fixed"]["in_flight_progress"] == [
        {
            "method_id": "q2_recranker_generalqwen",
            "block_zero_based": 24,
            "fold": 0,
            "run_id": "20260718_kuaisearch_mech_d2_q2_postblock_b24_fold0_v1",
            "status": "running",
            "progress_status": "running",
            "completed_requests": 1021,
            "total_requests": 4082,
            "request_completion_fraction": 1021 / 4082,
            "updated_at": "2026-07-18T00:00:00+00:00",
        }
    ]
    assert result["completed_scientific_bundles"] == 23
    assert result["resolved_scientific_units"] == 23
    assert result["resolution_fraction"] == 23 / 62
    assert result["maximum_remaining_scientific_bundles"] == 39
    assert result["maximum_completion_fraction"] == 23 / 62
    assert result["maximum_request_weighted_executed_bundle_equivalents"] == (
        executed_equivalents
    )
    assert result["maximum_request_weighted_execution_fraction"] == (
        executed_equivalents / 62
    )
    assert result["scientific_effect_values_read"] is False
    assert result["qrels_read"] is False


def test_progress_resolves_gate_stopped_and_completed_selected_branches(tmp_path):
    _contract(tmp_path, "q2", False)
    _contract(tmp_path, "q3", True)
    eval_path = (
        tmp_path
        / "runs/20260718_kuaisearch_mech_d2_q3_selected_branch_eval_v1/metrics.json"
    )
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")

    result = audit_deep_dive_progress(tmp_path)
    conditional = result["conditional_selected_branches"]
    assert conditional["completed_bundles"] == 1
    assert conditional["gate_stopped_models"] == 1
    assert conditional["in_flight_bundle_equivalents"] == 0.0
    assert conditional["request_weighted_executed_bundle_equivalents"] == 1.0
    assert conditional["request_weighted_execution_fraction"] == 0.5
    assert conditional["maximum_remaining_bundles"] == 0
    assert result["completed_scientific_bundles"] == 1
    assert result["resolved_scientific_units"] == 2
    assert result["resolution_fraction"] == 2 / 62
    assert result["maximum_remaining_scientific_bundles"] == 60


def test_progress_counts_partial_selected_branch_request_shards(tmp_path):
    _contract(tmp_path, "q2", True)
    _selected_shard(tmp_path, "q2", 0, completed_requests=3)
    _selected_shard(
        tmp_path,
        "q2",
        1,
        completed_requests=5,
        status="completed",
    )

    result = audit_deep_dive_progress(tmp_path)
    assert result["status"] == "ok"
    conditional = result["conditional_selected_branches"]
    assert conditional["completed_bundles"] == 0
    assert conditional["in_flight_bundle_equivalents"] == 0.8
    assert conditional["request_weighted_executed_bundle_equivalents"] == 0.8
    assert conditional["request_weighted_execution_fraction"] == 0.4
    q2 = conditional["models"][0]
    assert q2["status"] == "pending_branch"
    assert q2["request_completion_fraction"] == 0.8
    assert q2["shards"][0]["fold_request_completion_fraction"] == 0.3
    assert q2["shards"][1]["fold_request_completion_fraction"] == 0.5
    assert result["maximum_request_weighted_executed_bundle_equivalents"] == 0.8
    assert result["maximum_request_weighted_execution_fraction"] == 0.8 / 62


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("method_id", "q3_tallrec_generalqwen", "method_id mismatch"),
        ("selected_block", 13, "selected_block mismatch"),
        ("qrels_values_exposed_to_scorer", True, "qrels_values"),
        ("selected_nodes", ["block_output_residual"], "selected_nodes"),
    ],
)
def test_progress_rejects_invalid_selected_branch_contract(
    tmp_path, field, bad_value, message
):
    _contract(tmp_path, "q2", True)
    path = (
        tmp_path
        / "runs/20260718_kuaisearch_mech_d2_q2_selected_branch_contract_v1"
        / "contract.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value[field] = bad_value
    path.write_text(json.dumps(value), encoding="utf-8")

    result = audit_deep_dive_progress(tmp_path)
    assert result["status"] == "failed"
    assert result["conditional_selected_branches"]["models"][0]["status"] == (
        "invalid_contract"
    )
    assert any(message in error for error in result["errors"])


def test_progress_fails_closed_on_run_identity_drift(tmp_path):
    _metadata(tmp_path, "q2", 13, 0)
    path = next((tmp_path / "runs").glob("*/metadata.json"))
    value = json.loads(path.read_text(encoding="utf-8"))
    value["run_id"] = "drifted"
    path.write_text(json.dumps(value), encoding="utf-8")

    result = audit_deep_dive_progress(tmp_path)
    assert result["status"] == "failed"
    assert any("run-id mismatch" in error for error in result["errors"])


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("method_id", "q3_tallrec_generalqwen"),
        ("block_zero_based", 27),
        ("normalized_query_fold", 1),
        ("evidence_mode", "exploratory"),
        ("qrels_read", True),
        ("source_test_opened", True),
    ],
)
def test_progress_fails_closed_on_identity_or_data_boundary_drift(
    tmp_path, field, bad_value
):
    _metadata(tmp_path, "q2", 13, 0)
    path = next((tmp_path / "runs").glob("*/metadata.json"))
    value = json.loads(path.read_text(encoding="utf-8"))
    value[field] = bad_value
    path.write_text(json.dumps(value), encoding="utf-8")

    result = audit_deep_dive_progress(tmp_path)
    assert result["status"] == "failed"
    assert any(f"{field} mismatch" in error for error in result["errors"])


def test_progress_fails_closed_when_partial_exceeds_registered_total(tmp_path):
    _metadata(tmp_path, "q2", 13, 0, status="running")
    run_dir = (
        tmp_path
        / "runs"
        / "20260718_kuaisearch_mech_d2_q2_postblock_b13_fold0_v1"
    )
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["fold_request_count"] = 10
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "progress.json").write_text(
        json.dumps(
            {
                "status": "running",
                "completed_requests": 11,
                "run_contract_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )

    result = audit_deep_dive_progress(tmp_path)
    assert result["status"] == "failed"
    assert any("progress exceeds request total" in error for error in result["errors"])


def test_progress_fails_closed_on_partial_contract_mismatch(tmp_path):
    _metadata(tmp_path, "q2", 13, 0, status="running")
    run_dir = (
        tmp_path
        / "runs"
        / "20260718_kuaisearch_mech_d2_q2_postblock_b13_fold0_v1"
    )
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["fold_request_count"] = 10
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "progress.json").write_text(
        json.dumps(
            {
                "status": "running",
                "completed_requests": 5,
                "run_contract_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )

    result = audit_deep_dive_progress(tmp_path)
    assert result["status"] == "failed"
    assert result["fixed"]["in_flight_progress"][0][
        "request_completion_fraction"
    ] is None
    assert result["fixed"]["in_flight_bundle_equivalents"] == 0.0
    assert any("progress contract mismatch" in error for error in result["errors"])


def test_progress_reaches_full_resolution_when_optional_branches_gate_stop(tmp_path):
    for short in ("q2", "q3"):
        for fold in (0, 1):
            for block in range(13, 28):
                _metadata(tmp_path, short, block, fold)
        _contract(tmp_path, short, False)

    result = audit_deep_dive_progress(tmp_path)
    assert result["completed_scientific_bundles"] == 60
    assert result["resolved_scientific_units"] == 62
    assert result["resolution_fraction"] == 1.0
    assert result["maximum_remaining_scientific_bundles"] == 0
