from __future__ import annotations

from myrec.mechanism.deep_dive_closeout_audit import EXPECTED_DELIVERABLES
from myrec.mechanism.deep_dive_overview import (
    build_deep_dive_overview,
    component_model_coverage_debt,
)
from myrec.mechanism.deep_dive_report_contract import COMPONENT_IDS


def _closeout(completed, *, failures=()):
    return {
        "status": "pending",
        "deliverables": {
            name: {
                "status": "completed" if name in completed else "pending",
                "path": path,
            }
            for name, path in EXPECTED_DELIVERABLES.items()
        },
        "run_declarations": [
            {"result_eligible": True, "status": "completed"},
            {"result_eligible": True, "status": "running"},
            {"result_eligible": False, "status": "completed"},
        ],
        "formal_completed_integrity_checked": 7,
        "failures": list(failures),
        "pending": ["missing deliverable"],
        "mechanical_failure_records": [{"run_id": "old_failure"}],
        "source_test_content_read_by_this_audit": False,
        "qrels_content_read_by_this_audit": False,
    }


def _d2(status="ok"):
    return {
        "status": status,
        "completed_scientific_bundles": 27,
        "maximum_total_scientific_bundles": 62,
        "scientific_effect_values_read": False,
    }


def test_overview_reports_stage_component_and_model_completion(monkeypatch):
    completed = {
        "d1_representation",
        "d2_q3_native_gate",
        "d6_q2_native_readout",
        "d7_q2_objective",
        "d7_q3_lora_path",
    }
    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_overview.audit_deep_dive_closeout",
        lambda _root: _closeout(completed),
    )
    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_overview.audit_deep_dive_progress",
        lambda _root: _d2(),
    )
    result = build_deep_dive_overview(".")
    assert result["status"] == "ok"
    assert result["completion_status"] == "pending"
    assert result["deliverable_summary"]["completed"] == 5
    assert result["deliverable_summary"]["registered"] == 19
    assert len(result["stage_completion"]) == 7
    assert len(result["component_completion"]) == len(COMPONENT_IDS) == 18
    assert result["component_model_cell_summary"] == {
        "component_count": 18,
        "model_count": 4,
        "cell_count": 72,
        "registered_cell_count": 55,
        "causal_support_capable_cell_count": 31,
        "registered_but_descriptive_only_cell_count": 24,
        "not_directly_registered_cell_count": 17,
        "disposition_counts": {
            "registered_causal_support_capable": 31,
            "registered_descriptive_only": 24,
            "not_directly_registered": 17,
        },
        "all_cells_have_outcome_independent_disposition": True,
        "scientific_support_inferred_from_coverage": False,
    }
    assert result["component_summary"][
        "registered_but_descriptive_only_components"
    ] == [
        "lora_parameterization",
        "mlp_feature_formation",
        "optimizer_effective_update",
        "token_embedding",
    ]
    q0_summary = result["model_component_summary"][
        "q0_qwen3_reranker_06b"
    ]
    assert "attention_query_key_routing" in q0_summary[
        "not_directly_registered_components"
    ]
    assert "attention_output" in q0_summary[
        "causal_support_capable_components"
    ]
    q3_summary = result["model_component_summary"][
        "q3_tallrec_generalqwen"
    ]
    assert "lora_parameterization" in q3_summary[
        "registered_but_descriptive_only_components"
    ]
    assert q3_summary["registered_component_count"] > q0_summary[
        "registered_component_count"
    ]
    assert result["formal_run_summary"] == {
        "declared": 2,
        "status_counts": {"completed": 1, "running": 1},
        "completed_integrity_checked": 7,
    }
    native_readout = next(
        row
        for row in result["component_completion"]
        if row["component_id"] == "native_readout"
    )
    assert native_readout["completed_registered_deliverables"] == [
        "d2_q3_native_gate",
        "d6_q2_native_readout",
    ]
    assert native_readout["completed_causal_capable_deliverables"] == [
        "d6_q2_native_readout"
    ]
    assert native_readout["all_causal_capable_outputs_closed"] is False
    q2 = native_readout["per_model"]["q2_recranker_generalqwen"]
    assert q2["pending_causal_capable_deliverables"] == []
    assert q2["completed_causal_capable_deliverables"] == [
        "d6_q2_native_readout"
    ]
    assert result["outcome_values_used_for_progress"] is False
    assert result["scientific_effect_values_summarized"] is False
    h0 = next(
        row
        for row in result["hypothesis_evidence_readiness"]
        if row["hypothesis_id"] == "H0"
    )
    assert h0["all_required_groups_have_completed_source"] is True
    assert h0["hypothesis_support_inferred_from_readiness"] is False
    h5 = next(
        row
        for row in result["hypothesis_evidence_readiness"]
        if row["hypothesis_id"] == "H5"
    )
    assert h5["support_authorized_in_this_stage"] is False
    assert h5["all_required_groups_have_completed_source"] is None
    h4_opportunity = next(
        row
        for row in result["opportunity_evidence_readiness"]
        if row["opportunity_id"] == "OP_H4_SURFACE_AWARE_GRADIENT_BUDGET"
    )
    assert h4_opportunity["all_global_groups_have_completed_source"] is True
    assert h4_opportunity["per_model_readiness"][
        "q2_recranker_generalqwen"
    ]["all_required_groups_have_completed_source"] is False
    assert h4_opportunity["opportunity_priority_inferred_from_readiness"] is False
    h2_opportunity = next(
        row
        for row in result["opportunity_evidence_readiness"]
        if row["opportunity_id"]
        == "OP_H2_ID_FREE_FACTORIZED_PREFERENCE_BOTTLENECK"
    )
    assert all(
        model["all_required_groups_have_completed_source"] is None
        for model in h2_opportunity["per_model_readiness"].values()
    )


def test_component_model_coverage_debt_is_outcome_independent():
    debt = component_model_coverage_debt()
    assert debt["component_count"] == 18
    assert debt["model_count"] == 4
    assert debt["component_model_cell_count"] == 72
    assert debt["registered_component_model_cell_count"] == 55
    assert debt["causal_support_capable_component_model_cell_count"] == 31
    assert debt[
        "registered_but_descriptive_only_component_model_cell_count"
    ] == 24
    assert debt["not_directly_registered_component_model_cell_count"] == 17
    assert debt["component_model_cell_disposition_counts"] == {
        "registered_causal_support_capable": 31,
        "registered_descriptive_only": 24,
        "not_directly_registered": 17,
    }
    assert debt[
        "all_component_model_cells_have_outcome_independent_disposition"
    ] is True
    assert debt["scientific_support_inferred_from_coverage"] is False
    assert debt["globally_registered_but_descriptive_only_components"] == [
        "lora_parameterization",
        "mlp_feature_formation",
        "optimizer_effective_update",
        "token_embedding",
    ]
    assert debt["per_model"]["q0_qwen3_reranker_06b"][
        "causal_support_capable_component_count"
    ] == 2
    assert debt["per_model"]["q3_tallrec_generalqwen"][
        "registered_component_count"
    ] == 18


def test_overview_does_not_treat_descriptive_completion_as_causal_support(
    monkeypatch,
):
    completed = {"d7_q3_lora_path"}
    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_overview.audit_deep_dive_closeout",
        lambda _root: _closeout(completed),
    )
    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_overview.audit_deep_dive_progress",
        lambda _root: _d2(),
    )
    result = build_deep_dive_overview(".")
    lora = next(
        row
        for row in result["component_completion"]
        if row["component_id"] == "lora_parameterization"
    )
    assert lora["completed_registered_deliverables"] == ["d7_q3_lora_path"]
    assert lora["causal_support_capable_deliverables"] == []
    assert lora["all_causal_capable_outputs_closed"] is None
    assert lora["scientific_support_inferred_from_completion"] is False


def test_overview_propagates_closeout_or_d2_audit_failure(monkeypatch):
    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_overview.audit_deep_dive_closeout",
        lambda _root: _closeout(set(), failures=("identity drift",)),
    )
    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_overview.audit_deep_dive_progress",
        lambda _root: _d2(status="failed"),
    )
    result = build_deep_dive_overview(".")
    assert result["status"] == "failed"
    assert result["closeout_summary"]["audit_failures"] == ["identity drift"]
