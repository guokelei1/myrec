from __future__ import annotations

import hashlib
import json

import pytest

from myrec.mechanism.deep_dive_report_builder import (
    DELIVERABLE_RESULT_STRUCTURE_SPECS,
    DELIVERABLE_MODEL_COVERAGE,
    MODEL_IDS,
    _attenuation_transition_profile,
    _component_probe_coverage,
    _component_practical_equivalence_gate_catalog,
    _component_practical_equivalence_gate_outcomes,
    _component_result_gate_catalog,
    _component_result_gate_outcomes,
    _decision_gate_catalog,
    _layerwise_attenuation_profile,
    _primary_attribution_evidence_census,
    _registered_result_structure_census,
    build_deep_dive_report,
    render_deep_dive_report_markdown,
)
from myrec.mechanism.deep_dive_progress import SELECTED_NODES
from myrec.mechanism.deep_dive_closeout_audit import EXPECTED_DELIVERABLES
from myrec.mechanism.deep_dive_report_contract import (
    COMPONENT_DELIVERABLE_MODEL_COVERAGE,
    COMPONENT_IDS,
    COMPONENT_PROBE_CLAIM_BOUNDARIES,
    CROSS_MODEL_ATTRIBUTION_BOUNDARY,
    D7_OBJECTIVE_FAMILY_ENDPOINTS,
    D7_OBJECTIVE_FAMILY_STATES,
    D7_OBJECTIVE_FAMILY_SURFACES,
    PRIMARY_ATTRIBUTION_ENDPOINT_SCOPE,
    PRIMARY_ATTRIBUTION_FOLD_SCOPE,
    PRIMARY_ATTRIBUTION_INTERVENTION_SCOPE,
    PRIMARY_ATTRIBUTION_LAYER_SELECTION_SCOPE,
    REQUIRED_ASSERTIONS,
    RESULT_LEVEL_EQUIVALENCE_COMPONENT_ROUTES,
    RESULT_LEVEL_SUPPORTED_COMPONENT_ROUTES,
    TRANSFER_EXPLANATION_LADDER_SCOPE,
    TRANSFER_EXPLANATION_LEVELS,
    DeepDiveReportContractError,
)
from myrec.mechanism.deep_dive_opportunity_catalog import (
    OPPORTUNITY_DESIGN_CATALOG_SHA256,
    OPPORTUNITY_IDS,
    OPPORTUNITY_STAGE_BOUNDARY,
)
from myrec.mechanism.deep_dive_overview import component_model_coverage_debt


def _decisions():
    return {
        "narratives": {
            "executive_summary": "Summary.",
            "primary_mechanism_diagnosis": "Diagnosis.",
            "signal_attenuation_answer": "Attenuation answer.",
            "cross_model_boundary": "Boundary.",
            "negative_evidence_summary": "Negative evidence.",
            "remaining_uncertainty": "Uncertainty.",
            "recommended_next_action": "Next action.",
        },
        "component_evidence_matrix": [
            {
                "component_id": "mlp_output",
                "model_scope": ["q2_recranker_generalqwen"],
                "status": "supported",
                "negative_evidence_basis": "not_applicable",
                "finding": "Finding.",
                "claim_boundary": COMPONENT_PROBE_CLAIM_BOUNDARIES["mlp_output"],
                "optimization_implication": "Implication.",
                "evidence_deliverables": ["d4_mlp_groups"],
                "mechanical_failure_records": [],
            }
        ],
        "primary_loss_attribution": [
            {
                "method_id": method_id,
                "primary_component": "unresolved",
                "evidence_strength": "unresolved",
                "fold1_transition_reproduced": False,
                "attention_branch_registered_support": False,
                "mlp_branch_registered_support": False,
                "postblock_registered_support": False,
                "residual_composition_criterion_met": False,
                "residual_norm_interaction_criterion_met": False,
                "selected_branch_node_inference_fold": 1,
                "node_effect_two_fold_replication_tested": False,
                "split_sample_component_localization": True,
                "selected_transition_interpretation": (
                    "largest_fold0_mean_negative_adjacent_postblock_step"
                ),
                "earliest_loss_layer_established": False,
                "global_unique_loss_layer_established": False,
                "layer_scan_role": (
                    "unbiased_localization_for_component_decomposition"
                ),
                "layer_scan_observed_state_scope": (
                    "native_candidate_scoring_positions_only"
                ),
                "history_effect_interpretation": (
                    "accumulated_state_sufficiency_not_token_path"
                ),
                "history_token_flow_directly_observed_by_layer_scan": False,
                "exact_layer_index_is_architecture_evidence": False,
                "cross_model_exact_layer_generalization_authorized": False,
                "design_implication_requires_component_or_distributed_pattern_evidence": True,
                "primary_attribution_endpoint": "target_margin",
                "strict_transfer_ndcg_component_corroborated": False,
                "strict_transfer_ndcg_beyond_equivalence_component_corroborated": False,
                "target_margin_component_is_not_automatically_ndcg_cause": True,
                "transfer_failure_causal_scope": "unresolved",
                "transfer_explanation_level": "unresolved_or_gate_stopped",
                "causal_intervention_role": "null_context_sufficiency",
                "primary_component_interpretation": (
                    "registered_candidate_bottleneck_not_unique_origin"
                ),
                "within_block_adjacent_change_role": (
                    "descriptive_only_without_registered_directional_gate"
                ),
                "component_erasure_boundary_established": False,
                "necessity_tested": False,
                "exclusive_component_origin_established": False,
                "additive_or_shapley_contribution_estimated": False,
                "descriptive_localization_used_as_primary_cause": False,
                "rationale": "Rationale.",
                "claim_boundary": PRIMARY_ATTRIBUTION_INTERVENTION_SCOPE[
                    "claim_boundary"
                ],
                "evidence_deliverables": [
                    "d2_postblock",
                    "d2_selected_branches",
                ],
            }
            for method_id in (
                "q2_recranker_generalqwen",
                "q3_tallrec_generalqwen",
            )
        ],
        "cross_model_primary_attribution": {
            "scope": "no_registered_component_resolution",
            "q2_primary_component": "unresolved",
            "q3_primary_component": "unresolved",
            "shared_primary_component": None,
            "both_models_resolved": False,
            "same_component_across_models": False,
            "generalization_beyond_q2_q3_authorized": False,
            "universal_llm4rec_mechanism_claim_authorized": False,
            "rationale": "Bounded cross-model rationale.",
            "claim_boundary": CROSS_MODEL_ATTRIBUTION_BOUNDARY[
                "claim_boundary"
            ],
        },
        "hypothesis_status_matrix": [
            {
                "hypothesis_id": "H3",
                "status": "weakened",
                "negative_evidence_basis": "mixed_registered_evidence",
                "rationale": "Rationale.",
                "remaining_uncertainty": "Uncertainty.",
                "evidence_deliverables": ["d4_mlp_groups"],
            }
        ],
        "architecture_opportunity_ranking": [
            {
                "opportunity_id": "OP_H3_CANDIDATE_CONDITIONED_SIGNED_PREFERENCE_RESIDUAL",
                "rank": 1,
                "status": "primary",
                "model_scope": ["q2_recranker_generalqwen"],
                "rationale": "Rationale.",
                "falsification_gate": "Gate.",
                "innovation_claim": "Innovation.",
                "required_modules": ["Module A"],
                "training_signal": "Signal.",
                "training_data_requirements": "Train-only data.",
                "exact_null_recovery_invariant": "Exact fallback.",
                "critical_ablations": ["A", "B", "C"],
                "prior_work_differences": {
                    "CoPPS": "Difference.",
                    "BATA": "Difference.",
                    "HMPPS": "Difference.",
                    "MemRerank": "Difference.",
                },
                "stage_boundary": "design_opportunity_only_not_implemented",
                "evidence_deliverables": ["d4_mlp_groups"],
            }
        ],
        "boundary_assertions": dict(REQUIRED_ASSERTIONS),
    }


def _closeout():
    return {
        "status": "completed",
        "frozen_assets": {"plan": "hash"},
        "deliverables": {
            f"d{index}": {
                "status": "completed",
                "sha256": f"hash-{index}",
                "path": f"runs/d{index}/metrics.json",
            }
            for index in range(19)
        },
        "mechanical_failure_records": [],
        "run_declarations": [
            {
                "analysis_stage": "attention",
                "method_id": "q2",
                "status": "completed",
                "result_eligible": True,
            },
            {
                "analysis_stage": "attention",
                "method_id": "q3",
                "status": "completed",
                "result_eligible": False,
            },
            {
                "analysis_stage": "mlp",
                "method_id": "q2",
                "status": "running",
                "result_eligible": True,
            },
        ],
        "formal_completed_integrity_checked": 41,
        "dev_eval_ledger": {"entry_count": 100, "sha256": "ledger"},
        "source_test_content_read_by_this_audit": False,
        "qrels_content_read_by_this_audit": False,
    }


def _sample_attenuation_profile():
    return {
        "source": {"path": "runs/selected/metrics.json", "sha256": "hash"},
        "row_count": 0,
        "directional_description_counts": {},
        "interpretation_boundary": "Attenuation is not literal sign reversal.",
        "rows": [],
    }


def _sample_layerwise_profile():
    return {
        "source": {"path": "runs/postblock/metrics.json", "sha256": "hash"},
        "all_layer_row_count": 0,
        "adjacent_layer_row_count": 0,
        "shape_summary": [],
        "shape_taxonomy": [],
        "interpretation_boundary": "Exact layer indices are not architecture evidence.",
        "all_layer_rows": [],
        "adjacent_layer_rows": [],
    }


def _sample_result_structure_census():
    return [
        {
            "deliverable": "d1_representation",
            "analysis_type": "transformer_deep_dive_d1_region_decoding_synthesis",
            "source": {"path": "runs/d1/metrics.json", "sha256": "hash"},
            "structures": [
                {
                    "path": "cells",
                    "mode": "list",
                    "matched_containers": 1,
                    "unit_count": 96,
                    "expected_unit_count": 96,
                    "exact_count_registered": True,
                    "role": "confirmatory_family",
                }
            ],
            "multiplicity_metadata": {
                "family": {"planned_size": 96, "observed_size": 96}
            },
            "outcome_values_used_for_structure_admission": False,
            "scientific_effect_values_summarized": False,
        }
    ]


def _sample_component_result_gate_outcomes():
    rows = [
        {
            "component_id": component_id,
            "method_id": method_id,
            "deliverable": deliverable,
            "gate_outcome": "registered_support_not_established",
            "registered_support": False,
            "source": {
                "path": f"runs/{deliverable}/metrics.json",
                "sha256": f"gate-source-sha-{deliverable}",
            },
        }
        for component_id, deliverable in sorted(
            RESULT_LEVEL_SUPPORTED_COMPONENT_ROUTES
        )
        for method_id in sorted(
            COMPONENT_DELIVERABLE_MODEL_COVERAGE[component_id][deliverable]
        )
    ]
    return {
        "route_count": 25,
        "model_route_count": len(rows),
        "registered_support_count": 0,
        "registered_support_not_established_count": len(rows),
        "all_registered_routes_evaluated": True,
        "raw_effect_values_emitted": False,
        "support_not_established_is_opposite_effect_evidence": False,
        "support_not_established_is_weakened_or_rejected_evidence": False,
        "rows": rows,
    }


def _sample_component_practical_equivalence_gate_outcomes():
    rows = [
        {
            "component_id": component_id,
            "method_id": method_id,
            "deliverable": deliverable,
            "gate_outcome": (
                "registered_practical_equivalence_not_established"
            ),
            "registered_practical_equivalence": False,
            "source": {
                "path": f"runs/{deliverable}/metrics.json",
                "sha256": f"equivalence-source-sha-{deliverable}",
            },
        }
        for component_id, deliverable in sorted(
            RESULT_LEVEL_EQUIVALENCE_COMPONENT_ROUTES
        )
        for method_id in sorted(
            COMPONENT_DELIVERABLE_MODEL_COVERAGE[component_id][deliverable]
        )
    ]
    return {
        "route_count": 4,
        "model_route_count": len(rows),
        "registered_practical_equivalence_count": 0,
        "registered_practical_equivalence_not_established_count": len(rows),
        "all_registered_routes_evaluated": True,
        "raw_effect_values_emitted": False,
        "not_established_is_non_equivalence_evidence": False,
        "rows": rows,
    }


def _sample_primary_attribution_evidence_census():
    rows = [
        {
            "method_id": method_id,
            "localization_status": "unspecified",
            "fold1_transition_reproduced": False,
            "postblock_registered_support": False,
            "incoming_block_state_registered_support": False,
            "incoming_state_confounds_residual_or_norm_attribution": False,
            "attention_branch_registered_support": False,
            "mlp_branch_registered_support": False,
            "residual_node_registered_support": False,
            "normalization_node_registered_support": False,
            "normalization_boundary_isolated_registered_support": False,
            "normalization_boundary_isolated_nodes": [],
            "normalization_state_support_without_boundary_isolation": False,
            "normalization_nodes_without_boundary_isolation": [],
            "residual_composition_criterion_met": False,
            "residual_norm_interaction_criterion_met": False,
            "derived_primary_component": "unresolved",
            "derived_evidence_strength": "unresolved",
            "derived_resolution": False,
            "selected_branch_node_inference_fold": 1,
            "node_effect_two_fold_replication_tested": False,
            "split_sample_component_localization": True,
            "selected_transition_interpretation": (
                "largest_fold0_mean_negative_adjacent_postblock_step"
            ),
            "earliest_loss_layer_established": False,
            "global_unique_loss_layer_established": False,
            "layer_scan_role": (
                "unbiased_localization_for_component_decomposition"
            ),
            "layer_scan_observed_state_scope": (
                "native_candidate_scoring_positions_only"
            ),
            "history_effect_interpretation": (
                "accumulated_state_sufficiency_not_token_path"
            ),
            "history_token_flow_directly_observed_by_layer_scan": False,
            "exact_layer_index_is_architecture_evidence": False,
            "cross_model_exact_layer_generalization_authorized": False,
            "design_implication_requires_component_or_distributed_pattern_evidence": True,
            "primary_attribution_endpoint": "target_margin",
            "strict_transfer_ndcg_component_corroborated": False,
            "strict_transfer_ndcg_beyond_equivalence_component_corroborated": False,
            "target_margin_component_is_not_automatically_ndcg_cause": True,
            "transfer_failure_causal_scope": "unresolved",
            "transfer_explanation_level": "unresolved_or_gate_stopped",
            "causal_intervention_role": "null_context_sufficiency",
            "primary_component_interpretation": (
                "registered_candidate_bottleneck_not_unique_origin"
            ),
            "within_block_adjacent_change_role": (
                "descriptive_only_without_registered_directional_gate"
            ),
            "component_erasure_boundary_established": False,
            "necessity_tested": False,
            "exclusive_component_origin_established": False,
            "additive_or_shapley_contribution_estimated": False,
            "descriptive_head_or_group_used_as_primary_cause": False,
            "raw_effect_values_emitted": False,
        }
        for method_id in MODEL_IDS[2:]
    ]
    return {
        "fold_scope": dict(PRIMARY_ATTRIBUTION_FOLD_SCOPE),
        "layer_selection_scope": dict(
            PRIMARY_ATTRIBUTION_LAYER_SELECTION_SCOPE
        ),
        "endpoint_scope": dict(PRIMARY_ATTRIBUTION_ENDPOINT_SCOPE),
        "intervention_scope": dict(PRIMARY_ATTRIBUTION_INTERVENTION_SCOPE),
        "transfer_explanation_ladder": dict(
            TRANSFER_EXPLANATION_LADDER_SCOPE
        ),
        "transfer_explanation_levels": list(TRANSFER_EXPLANATION_LEVELS),
        "cross_model_scope": {
            "scope": "no_registered_component_resolution",
            "q2_primary_component": "unresolved",
            "q3_primary_component": "unresolved",
            "shared_primary_component": None,
            "both_models_resolved": False,
            "same_component_across_models": False,
            "generalization_beyond_q2_q3_authorized": False,
            "universal_llm4rec_mechanism_claim_authorized": False,
            "claim_boundary": CROSS_MODEL_ATTRIBUTION_BOUNDARY[
                "claim_boundary"
            ],
        },
        "sources": {
            "d2_postblock": {
                "path": "runs/postblock/metrics.json",
                "sha256": "postblock-sha",
            },
            "d2_selected_branches": {
                "path": "runs/selected/metrics.json",
                "sha256": "selected-sha",
            },
        },
        "rows": rows,
        "model_count": 2,
        "deterministic_precedence": [
            "fold1_transition_required",
            "mixed_attention_mlp",
            "attention_output",
            "mlp",
            "residual_norm_interaction",
            "residual_composition",
            "unresolved",
        ],
        "descriptive_head_or_group_used_as_primary_cause": False,
        "raw_effect_values_emitted": False,
    }


def _d7_objective_family_rows():
    return [
        {
            "state": state,
            "surface": surface,
            "endpoint": endpoint,
            "conflict_beyond_sesoi": False,
            "bh_q_below_0.05": True,
        }
        for state in D7_OBJECTIVE_FAMILY_STATES
        for surface in D7_OBJECTIVE_FAMILY_SURFACES
        for endpoint in D7_OBJECTIVE_FAMILY_ENDPOINTS
    ]


def test_report_builder_writes_both_admission_bound_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_report_builder.validate_deep_dive_report_against_closeout",
        lambda _root, _decisions: _closeout(),
    )
    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_report_builder.audit_deep_dive_progress",
        lambda _root: {
            "status": "ok",
            "fixed": {"remaining_mandatory_bundles": 0},
            "conditional_selected_branches": {"gate_stopped_models": 0},
            "completed_scientific_bundles": 62,
            "maximum_total_scientific_bundles": 62,
            "maximum_remaining_scientific_bundles": 0,
        },
    )
    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_report_builder._layerwise_attenuation_profile",
        lambda _root, _closeout_value: _sample_layerwise_profile(),
    )
    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_report_builder._attenuation_transition_profile",
        lambda _root, _closeout_value: _sample_attenuation_profile(),
    )
    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_report_builder._registered_result_structure_census",
        lambda _root, _closeout_value: _sample_result_structure_census(),
    )
    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_report_builder._component_result_gate_outcomes",
        lambda _root, _closeout_value: _sample_component_result_gate_outcomes(),
    )
    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_report_builder._component_practical_equivalence_gate_outcomes",
        lambda _root, _closeout_value: (
            _sample_component_practical_equivalence_gate_outcomes()
        ),
    )
    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_report_builder._primary_attribution_evidence_census",
        lambda _root, _closeout_value: _sample_primary_attribution_evidence_census(),
    )
    payload = build_deep_dive_report(
        tmp_path,
        _decisions(),
        json_output="reports/result.json",
        markdown_output="reports/result.md",
    )
    loaded = json.loads((tmp_path / "reports/result.json").read_text())
    markdown = (tmp_path / "reports/result.md").read_text()
    assert loaded["analysis_type"] == "transformer_deep_dive_mechanism_closeout"
    assert loaded["evidence_admission"]["formal_completed_integrity_checked"] == 41
    assert loaded["opportunity_design_catalog"]["sha256"] == OPPORTUNITY_DESIGN_CATALOG_SHA256
    assert loaded["opportunity_design_catalog"][
        "design_fields_frozen_before_remaining_outcomes"
    ] is True
    assert "model_scope" in loaded["opportunity_design_catalog"][
        "permitted_result_interpretation_fields"
    ]
    assert loaded["execution_census"]["run_status_counts"] == {
        "completed": 2,
        "running": 1,
    }
    assert loaded["execution_census"]["completed_result_eligible_runs"] == 1
    assert loaded["d2_causal_core_progress"]["completed_scientific_bundles"] == 62
    assert loaded["execution_census"]["run_stage_counts"] == {
        "attention": 2,
        "mlp": 1,
    }
    assert loaded["execution_census"]["formal_run_stage_counts"] == {
        "attention": 1,
        "mlp": 1,
    }
    assert len(loaded["component_probe_coverage"]) == len(COMPONENT_IDS) == 18
    assert len(loaded["component_result_gate_catalog"]) == 25
    assert len(loaded["component_practical_equivalence_gate_catalog"]) == 4
    assert loaded["component_practical_equivalence_gate_outcomes"][
        "model_route_count"
    ] == 5
    assert loaded["component_practical_equivalence_gate_outcomes"][
        "raw_effect_values_emitted"
    ] is False
    assert loaded["component_result_gate_outcomes"][
        "all_registered_routes_evaluated"
    ] is True
    assert loaded["component_result_gate_outcomes"][
        "registered_support_not_established_count"
    ] == 45
    assert loaded["component_result_gate_outcomes"][
        "support_not_established_is_weakened_or_rejected_evidence"
    ] is False
    assert loaded["primary_attribution_evidence_census"]["model_count"] == 2
    assert loaded["primary_attribution_evidence_census"][
        "descriptive_head_or_group_used_as_primary_cause"
    ] is False
    assert loaded["primary_attribution_evidence_census"][
        "transfer_explanation_levels"
    ] == list(TRANSFER_EXPLANATION_LEVELS)
    assert loaded["primary_attribution_evidence_census"][
        "transfer_explanation_ladder"
    ]["highest_level_establishes_complete_transfer_failure_cause"] is False
    assert loaded["primary_attribution_evidence_census"]["cross_model_scope"][
        "universal_llm4rec_mechanism_claim_authorized"
    ] is False
    assert loaded["component_model_coverage_debt"]["per_model"][
        "q0_qwen3_reranker_06b"
    ]["registered_component_count"] == 10
    assert loaded["component_model_coverage_debt"][
        "component_model_cell_count"
    ] == 72
    assert loaded["component_model_coverage_debt"][
        "not_directly_registered_component_model_cell_count"
    ] == 17
    assert len(loaded["decision_gate_catalog"]["hypothesis_supported"]) == 6
    assert len(loaded["decision_gate_catalog"]["opportunity_primary"]) == 5
    assert len(loaded["registered_result_structure_census"]) == 1
    assert loaded["registered_result_structure_census"][0]["structures"][0][
        "unit_count"
    ] == 96
    optimizer = next(
        row
        for row in loaded["component_probe_coverage"]
        if row["component_id"] == "optimizer_effective_update"
    )
    assert optimizer["causal_support_deliverables"] == []
    assert set(DELIVERABLE_MODEL_COVERAGE) == set(EXPECTED_DELIVERABLES)
    attention_qk = next(
        row
        for row in loaded["component_probe_coverage"]
        if row["component_id"] == "attention_query_key_routing"
    )
    assert attention_qk["model_registered_deliverables"][MODEL_IDS[0]] == []
    assert attention_qk["model_registered_deliverables"][MODEL_IDS[1]] == []
    assert attention_qk["model_registered_deliverables"][MODEL_IDS[2]]
    assert attention_qk["model_registered_deliverables"][MODEL_IDS[3]]
    assert attention_qk["model_causal_support_deliverables"][MODEL_IDS[0]] == []
    assert attention_qk["model_causal_support_deliverables"][MODEL_IDS[1]] == []
    assert set(
        attention_qk["model_causal_support_deliverables"][MODEL_IDS[2]]
    ) == {"d3_attention_edges", "d5_rope"}
    assert set(
        attention_qk["model_causal_support_deliverables"][MODEL_IDS[3]]
    ) == {"d3_attention_edges", "d5_rope"}
    assert "not directly registered" in markdown
    assert "descriptive-only" in markdown
    assert "## Transformer component evidence matrix" in markdown
    assert "## Per-model Transformer coverage debt" in markdown
    assert "18x4 cells: `72` total; `55` registered" in markdown
    assert "`17` not directly registered" in markdown
    assert "### Registered result structures" in markdown
    assert "cells=96 (confirmatory_family)" in markdown
    assert "## Component result-level support gates" in markdown
    assert "### Registered component gate outcomes" in markdown
    assert "registered_support_not_established" in markdown
    assert "## Mechanically derived primary attribution evidence" in markdown
    assert "## Hypothesis and opportunity decision gates" in markdown
    assert "history_value_edge_zero only" in markdown
    assert "## Primary registered candidate bottleneck by model" in markdown
    assert "## Adjacent-node attenuation transition profile" in markdown
    assert "## Full layer-scan attenuation shape" in markdown
    assert "## Architecture opportunity design cards" in markdown
    assert "Its exact index is localization metadata, not architecture evidence" in markdown
    assert "Prior-work differences" in markdown
    assert "OP_H3_CANDIDATE_CONDITIONED_SIGNED_PREFERENCE_RESIDUAL" in markdown
    assert "D2 completed scientific bundles: `62/62`" in markdown
    assert payload["status"] == "completed"


def test_report_builder_refuses_nonterminal_d2_progress(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_report_builder.validate_deep_dive_report_against_closeout",
        lambda _root, _decisions: _closeout(),
    )
    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_report_builder.audit_deep_dive_progress",
        lambda _root: {
            "status": "ok",
            "fixed": {"remaining_mandatory_bundles": 1},
            "maximum_remaining_scientific_bundles": 1,
        },
    )
    with pytest.raises(ValueError, match="progress is not terminal"):
        build_deep_dive_report(
            tmp_path,
            _decisions(),
            json_output="reports/result.json",
            markdown_output="reports/result.md",
        )
    assert not (tmp_path / "reports/result.json").exists()
    assert not (tmp_path / "reports/result.md").exists()


def test_report_builder_refuses_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_report_builder.validate_deep_dive_report_against_closeout",
        lambda _root, _decisions: _closeout(),
    )
    json_path = tmp_path / "result.json"
    json_path.write_text("owned\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_deep_dive_report(
            tmp_path,
            _decisions(),
            json_output=json_path,
            markdown_output=tmp_path / "result.md",
        )
    assert json_path.read_text() == "owned\n"


def test_component_probe_coverage_is_explicit_per_model():
    rows = {row["component_id"]: row for row in _component_probe_coverage()}
    assert set(COMPONENT_PROBE_CLAIM_BOUNDARIES) == set(rows) == set(COMPONENT_IDS)
    assert "gate_proj" in rows["mlp_feature_formation"]["coverage_boundary"]
    assert "native-Q/K scoring no-op" in rows["positional_encoding_rope"][
        "coverage_boundary"
    ]
    assert "incoming block-state" in rows["residual_composition"][
        "coverage_boundary"
    ]
    token_registered = rows["token_embedding"]["model_registered_deliverables"]
    assert "d7_optimizer_replay" in token_registered[MODEL_IDS[2]]
    assert "d7_optimizer_replay" not in token_registered[MODEL_IDS[3]]
    lora_registered = rows["lora_parameterization"][
        "model_registered_deliverables"
    ]
    assert lora_registered[MODEL_IDS[0]] == []
    assert lora_registered[MODEL_IDS[1]] == []
    assert lora_registered[MODEL_IDS[2]] == []
    assert set(lora_registered[MODEL_IDS[3]]) == {
        "d7_optimizer_replay",
        "d7_q3_lora_path",
    }
    assert all(
        not deliverables
        for deliverables in rows["token_embedding"]
        ["model_causal_support_deliverables"].values()
    )
    rope_causal = rows["positional_encoding_rope"][
        "model_causal_support_deliverables"
    ]
    assert rope_causal[MODEL_IDS[0]] == []
    assert rope_causal[MODEL_IDS[1]] == []
    assert rope_causal[MODEL_IDS[2]] == ["d5_rope"]
    assert rope_causal[MODEL_IDS[3]] == ["d5_rope"]
    residual_causal = rows["residual_composition"][
        "model_causal_support_deliverables"
    ]
    assert residual_causal[MODEL_IDS[0]] == []
    assert residual_causal[MODEL_IDS[1]] == []
    assert residual_causal[MODEL_IDS[2]] == ["d2_selected_branches"]
    assert residual_causal[MODEL_IDS[3]] == ["d2_selected_branches"]
    normalization_causal = rows["normalization"][
        "model_causal_support_deliverables"
    ]
    assert normalization_causal[MODEL_IDS[0]] == []
    assert normalization_causal[MODEL_IDS[1]] == []
    assert normalization_causal[MODEL_IDS[2]] == ["d2_selected_branches"]
    assert normalization_causal[MODEL_IDS[3]] == ["d2_selected_branches"]
    layerwise = rows["layerwise_representation"]["model_registered_deliverables"]
    assert layerwise[MODEL_IDS[0]] == ["d6_q0_trajectory"]
    assert layerwise[MODEL_IDS[1]] == ["d6_q1_trajectory"]
    assert {"d1_representation", "d2_postblock"} <= set(layerwise[MODEL_IDS[2]])
    assert {"d1_representation", "d2_postblock"} <= set(layerwise[MODEL_IDS[3]])
    assert rows["attention_query_key_routing"][
        "causal_support_mechanical_dependencies"
    ] == {"d3_attention_edges": ["d3_attention_heads"]}
    assert rows["attention_value_transport"][
        "causal_support_mechanical_dependencies"
    ] == {"d3_attention_edges": ["d3_attention_heads"]}

    readout = rows["native_readout"]["model_registered_deliverables"]
    assert readout[MODEL_IDS[0]] == ["d6_q0_q1_readouts"]
    assert readout[MODEL_IDS[1]] == ["d6_q0_q1_readouts"]
    assert readout[MODEL_IDS[2]] == ["d6_q2_native_readout"]
    assert set(readout[MODEL_IDS[3]]) == {
        "d2_q3_native_gate",
        "d6_q3_native_readout",
    }
    readout_causal = rows["native_readout"][
        "model_causal_support_deliverables"
    ]
    assert readout_causal[MODEL_IDS[0]] == []
    assert readout_causal[MODEL_IDS[1]] == []
    assert readout_causal[MODEL_IDS[2]] == ["d6_q2_native_readout"]
    assert readout_causal[MODEL_IDS[3]] == ["d6_q3_native_readout"]

    assert rows["layerwise_representation"]["model_coverage_levels"] == {
        MODEL_IDS[0]: "descriptive-only",
        MODEL_IDS[1]: "descriptive-only",
        MODEL_IDS[2]: "causal-support-capable",
        MODEL_IDS[3]: "causal-support-capable",
    }
    assert set(
        rows["layerwise_representation"]["model_causal_support_deliverables"]
        [MODEL_IDS[2]]
    ) == {"d2_postblock"}
    assert all(
        not deliverables
        for deliverables in rows["optimizer_effective_update"]
        ["model_causal_support_deliverables"].values()
    )


def test_report_builder_leaves_no_outputs_when_closeout_is_pending(
    monkeypatch, tmp_path
):
    def refuse(_root, _decisions):
        raise DeepDiveReportContractError("closeout is pending")

    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_report_builder.validate_deep_dive_report_against_closeout",
        refuse,
    )
    json_path = tmp_path / "result.json"
    markdown_path = tmp_path / "result.md"
    with pytest.raises(DeepDiveReportContractError, match="pending"):
        build_deep_dive_report(
            tmp_path,
            _decisions(),
            json_output=json_path,
            markdown_output=markdown_path,
        )
    assert not json_path.exists()
    assert not markdown_path.exists()


def test_report_markdown_escapes_table_cells():
    decisions = _decisions()
    decisions["component_evidence_matrix"][0]["finding"] = "a|b\nc"
    payload = {
        "status": "completed",
        "generated_at": "2026-07-18T00:00:00+00:00",
        "evidence_admission": {
            "deliverables": _closeout()["deliverables"],
            "mechanical_failure_records": [
                {
                    "run_id": "mechanical-run",
                    "status": "mechanical_failure",
                    "path": "runs/mechanical-run/mechanical_failure_record.json",
                    "sha256": "mechanical-sha",
                }
            ],
            "formal_completed_integrity_checked": 41,
            "source_test_content_read_by_closeout_audit": False,
            "qrels_content_read_by_closeout_audit": False,
        },
        "opportunity_design_catalog": {
            "sha256": OPPORTUNITY_DESIGN_CATALOG_SHA256,
            "opportunity_ids": list(OPPORTUNITY_IDS),
            "stage_boundary": OPPORTUNITY_STAGE_BOUNDARY,
            "design_fields_frozen_before_remaining_outcomes": True,
            "permitted_result_interpretation_fields": [
                "rank",
                "status",
                "model_scope",
                "rationale",
                "falsification_gate",
                "evidence_deliverables",
            ],
            "allowed_model_scope": {},
        },
        "execution_census": {
            "run_declaration_count": 3,
            "run_status_counts": {"completed": 2, "running": 1},
            "run_stage_counts": {"attention": 2, "mlp": 1},
            "run_method_counts": {"q2": 2, "q3": 1},
            "formal_run_stage_counts": {"attention": 1, "mlp": 1},
            "result_eligible_run_declarations": 2,
            "completed_result_eligible_runs": 1,
            "formal_completed_integrity_checked": 41,
            "completed_deliverables": 19,
            "mechanical_failure_records": 1,
            "dev_eval_ledger_entries": 100,
        },
        "registered_result_structure_census": _sample_result_structure_census(),
        "d2_causal_core_progress": {
            "completed_scientific_bundles": 62,
            "maximum_total_scientific_bundles": 62,
            "conditional_selected_branches": {"gate_stopped_models": 0},
        },
        "layerwise_attenuation_profile": _sample_layerwise_profile(),
        "component_probe_coverage": _component_probe_coverage(),
        "component_model_coverage_debt": component_model_coverage_debt(),
        "component_result_gate_catalog": _component_result_gate_catalog(),
        "component_practical_equivalence_gate_catalog": (
            _component_practical_equivalence_gate_catalog()
        ),
        "component_practical_equivalence_gate_outcomes": (
            _sample_component_practical_equivalence_gate_outcomes()
        ),
        "component_result_gate_outcomes": (
            _sample_component_result_gate_outcomes()
        ),
        "decision_gate_catalog": _decision_gate_catalog(),
        "primary_attribution_evidence_census": (
            _sample_primary_attribution_evidence_census()
        ),
        "attenuation_transition_profile": _sample_attenuation_profile(),
        **decisions,
    }
    markdown = render_deep_dive_report_markdown(payload)
    assert "a\\|b<br>c" in markdown
    assert "### Mechanical failure record ledger" in markdown
    assert "mechanical-run" in markdown
    assert "runs/mechanical-run/mechanical_failure_record.json" in markdown
    assert "mechanical-sha" in markdown
    assert "Transfer explanation level" in markdown
    assert "unresolved_or_gate_stopped" in markdown
    assert "not necessity, exclusive causal origin" in markdown
    assert "Cross-model primary attribution boundary" in markdown
    assert "Universal LLM4Rec claim authorized" in markdown
    assert "Registered practical-equivalence outcomes" in markdown


def test_result_structure_specs_cover_all_closeout_deliverables():
    assert set(DELIVERABLE_RESULT_STRUCTURE_SPECS) == set(EXPECTED_DELIVERABLES)


def test_result_structure_census_counts_wildcard_leaves_and_binds_sha(tmp_path):
    metrics_path = tmp_path / "runs/example/metrics.json"
    metrics_path.parent.mkdir(parents=True)
    result = {
        "analysis_type": "example_registered_result",
        "status": "completed",
        "multiple_testing": {
            "method": "benjamini_hochberg",
            "family_size": 3,
        },
        "results": {
            "q2": {
                "13": {"groups": [{"effect": 1.0}, {"effect": -1.0}]},
                "20": {"groups": [{"effect": 0.0}]},
            }
        },
    }
    metrics_path.write_text(json.dumps(result), encoding="utf-8")
    digest = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    closeout = {
        "deliverables": {
            "example": {
                "status": "completed",
                "path": "runs/example/metrics.json",
                "sha256": digest,
            }
        }
    }
    specs = {
        "example": (
            ("results.*", "mapping", 2, "model_blocks"),
            ("results.*.*.groups", "list", 3, "registered_groups"),
        )
    }
    rows = _registered_result_structure_census(
        tmp_path, closeout, specs=specs
    )
    assert [row["unit_count"] for row in rows[0]["structures"]] == [2, 3]
    assert rows[0]["structures"][1]["matched_containers"] == 2
    assert rows[0]["multiplicity_metadata"] == {
        "multiple_testing": {
            "method": "benjamini_hochberg",
            "family_size": 3,
        }
    }
    assert rows[0]["outcome_values_used_for_structure_admission"] is False
    assert rows[0]["scientific_effect_values_summarized"] is False

    metrics_path.write_text(json.dumps({**result, "extra": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="source identity changed"):
        _registered_result_structure_census(tmp_path, closeout, specs=specs)


def test_result_structure_census_rejects_missing_registered_units(tmp_path):
    metrics_path = tmp_path / "runs/example/metrics.json"
    metrics_path.parent.mkdir(parents=True)
    result = {
        "analysis_type": "example_registered_result",
        "status": "completed",
        "rows": [{"effect": 0.0}],
    }
    metrics_path.write_text(json.dumps(result), encoding="utf-8")
    closeout = {
        "deliverables": {
            "example": {
                "status": "completed",
                "path": "runs/example/metrics.json",
                "sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
            }
        }
    }
    specs = {"example": (("rows", "list", 2, "confirmatory_family"),)}
    with pytest.raises(ValueError, match="unit count differs"):
        _registered_result_structure_census(tmp_path, closeout, specs=specs)


def test_component_result_gate_catalog_is_complete_and_specific():
    rows = _component_result_gate_catalog()
    assert len(rows) == 25
    assert len(
        {(row["component_id"], row["deliverable"]) for row in rows}
    ) == len(rows)
    value_gate = next(
        row
        for row in rows
        if row["component_id"] == "attention_value_transport"
    )
    assert value_gate["deliverable"] == "d3_attention_edges"
    assert "history_value_edge_zero only" in value_gate["registered_support_gate"]
    assert "logits" not in value_gate["registered_support_gate"]


def test_component_practical_equivalence_gate_catalog_is_complete_and_strict():
    rows = _component_practical_equivalence_gate_catalog()
    assert len(rows) == 4
    assert {(row["component_id"], row["deliverable"]) for row in rows} == {
        ("positional_encoding_rope", "d5_rope"),
        ("native_readout", "d6_q2_native_readout"),
        ("native_readout", "d6_q3_native_readout"),
        ("loss_gradient", "d7_q2_objective"),
    }
    assert all(
        "all" in row["registered_practical_equivalence_gate"]
        for row in rows
    )


def test_component_practical_equivalence_outcomes_bind_source_and_gate(tmp_path):
    metrics_path = tmp_path / "runs/readout/metrics.json"
    metrics_path.parent.mkdir(parents=True)

    def equivalent_rows():
        return [
            {
                "normalized_query_fold": fold,
                "mean": 0.0,
                "ci95": [-0.001, 0.001],
            }
            for fold in ("all", "0", "1")
        ]

    metrics = {
        "analysis_type": "transformer_deep_dive_d6_q2_native_readout",
        "status": "completed",
        "method_id": "q2_recranker_generalqwen",
        "results": {
            node: {
                comparison: {"ndcg@10": equivalent_rows()}
                for comparison in (
                    "same_minus_null",
                    "same_minus_full",
                    "same_minus_cross",
                )
            }
            for node in ("final_rmsnorm_input", "final_rmsnorm_output")
        },
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    digest = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    closeout = {
        "deliverables": {
            "d6_q2_native_readout": {
                "status": "completed",
                "path": "runs/readout/metrics.json",
                "sha256": digest,
            }
        }
    }
    routes = {("native_readout", "d6_q2_native_readout")}
    census = _component_practical_equivalence_gate_outcomes(
        tmp_path, closeout, routes=routes
    )
    assert census["route_count"] == 1
    assert census["model_route_count"] == 1
    assert census["registered_practical_equivalence_count"] == 1
    assert census["all_registered_routes_evaluated"] is False
    assert census["raw_effect_values_emitted"] is False
    assert census["not_established_is_non_equivalence_evidence"] is False
    assert census["rows"][0]["source"]["sha256"] == digest
    assert census["rows"][0]["gate_outcome"] == (
        "registered_practical_equivalence"
    )

    metrics["results"]["final_rmsnorm_input"]["same_minus_null"][
        "ndcg@10"
    ][0]["ci95"] = [-0.006, 0.001]
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    closeout["deliverables"]["d6_q2_native_readout"]["sha256"] = (
        hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    )
    census = _component_practical_equivalence_gate_outcomes(
        tmp_path, closeout, routes=routes
    )
    assert census["registered_practical_equivalence_count"] == 0
    assert census["registered_practical_equivalence_not_established_count"] == 1


def test_component_result_gate_outcomes_bind_source_and_preserve_fail_closed_state(
    tmp_path,
):
    metrics_path = tmp_path / "runs/d7/metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics = {
        "analysis_type": "transformer_deep_dive_d7_q2_objective_conflict",
        "status": "completed",
        "family_rows": _d7_objective_family_rows(),
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    digest = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    closeout = {
        "deliverables": {
            "d7_q2_objective": {
                "status": "completed",
                "path": "runs/d7/metrics.json",
                "sha256": digest,
            }
        }
    }
    routes = {("loss_gradient", "d7_q2_objective")}

    census = _component_result_gate_outcomes(
        tmp_path, closeout, routes=routes
    )
    assert census["route_count"] == 1
    assert census["model_route_count"] == 1
    assert census["registered_support_count"] == 0
    assert census["registered_support_not_established_count"] == 1
    assert census["all_registered_routes_evaluated"] is False
    assert census["raw_effect_values_emitted"] is False
    assert census["support_not_established_is_opposite_effect_evidence"] is False
    assert census["rows"][0]["gate_outcome"] == (
        "registered_support_not_established"
    )
    assert census["rows"][0]["source"]["sha256"] == digest

    active = next(
        row
        for row in metrics["family_rows"]
        if row["endpoint"] == "ranknet_listnet_cosine"
    )
    active["conflict_beyond_sesoi"] = True
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    closeout["deliverables"]["d7_q2_objective"]["sha256"] = hashlib.sha256(
        metrics_path.read_bytes()
    ).hexdigest()
    census = _component_result_gate_outcomes(
        tmp_path, closeout, routes=routes
    )
    assert census["registered_support_count"] == 1
    assert census["rows"][0]["gate_outcome"] == "registered_support"


def test_component_result_gate_outcomes_reject_source_sha_drift(tmp_path):
    metrics_path = tmp_path / "runs/d7/metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps(
            {
                "analysis_type": (
                    "transformer_deep_dive_d7_q2_objective_conflict"
                ),
                "status": "completed",
                "family_rows": [],
            }
        ),
        encoding="utf-8",
    )
    closeout = {
        "deliverables": {
            "d7_q2_objective": {
                "status": "completed",
                "path": "runs/d7/metrics.json",
                "sha256": "wrong",
            }
        }
    }
    with pytest.raises(ValueError, match="identity changed"):
        _component_result_gate_outcomes(
            tmp_path,
            closeout,
            routes={("loss_gradient", "d7_q2_objective")},
        )


def test_primary_attribution_evidence_census_derives_unique_labels_and_binds_sha(
    tmp_path,
):
    postblock_path = tmp_path / "runs/postblock/metrics.json"
    selected_path = tmp_path / "runs/selected/metrics.json"
    postblock_path.parent.mkdir(parents=True)
    selected_path.parent.mkdir(parents=True)
    postblock = {
        "analysis_type": "transformer_deep_dive_d2_postblock_synthesis",
        "status": "completed",
        "localization": {
            MODEL_IDS[2]: {"resolved": True},
            MODEL_IDS[3]: {"resolved": False},
        },
    }
    node = "attention_o_projection"
    selected = {
        "analysis_type": (
            "transformer_deep_dive_d2_selected_branch_synthesis"
        ),
        "status": "completed",
        "fold_scope": dict(PRIMARY_ATTRIBUTION_FOLD_SCOPE),
        "rows": [
            {
                "method_id": MODEL_IDS[2],
                "endpoint": "target_margin",
                "contrast_id": f"{prefix}__{node}",
                "registered_support": True,
                "missing": False,
                "evidence_role": (
                    "registered_confirmatory_branch_localization"
                ),
            }
            for prefix in (
                "same",
                "same_minus_cross",
                "same_minus_wrong",
                "norm",
                "direction",
                "random",
            )
        ],
    }
    postblock_path.write_text(json.dumps(postblock), encoding="utf-8")
    selected_path.write_text(json.dumps(selected), encoding="utf-8")
    closeout = {
        "deliverables": {
            "d2_postblock": {
                "status": "completed",
                "path": "runs/postblock/metrics.json",
                "sha256": hashlib.sha256(
                    postblock_path.read_bytes()
                ).hexdigest(),
            },
            "d2_selected_branches": {
                "status": "completed",
                "path": "runs/selected/metrics.json",
                "sha256": hashlib.sha256(
                    selected_path.read_bytes()
                ).hexdigest(),
            },
        }
    }

    census = _primary_attribution_evidence_census(tmp_path, closeout)
    by_model = {row["method_id"]: row for row in census["rows"]}
    assert census["model_count"] == 2
    assert by_model[MODEL_IDS[2]]["derived_primary_component"] == (
        "attention_output"
    )
    assert by_model[MODEL_IDS[2]]["derived_resolution"] is True
    assert by_model[MODEL_IDS[2]]["transfer_explanation_level"] == (
        "target_margin_component_sufficiency"
    )
    assert by_model[MODEL_IDS[3]]["derived_primary_component"] == "unresolved"
    assert by_model[MODEL_IDS[3]]["derived_resolution"] is False
    assert census["descriptive_head_or_group_used_as_primary_cause"] is False
    assert census["raw_effect_values_emitted"] is False
    assert census["transfer_explanation_levels"] == list(
        TRANSFER_EXPLANATION_LEVELS
    )
    assert census["cross_model_scope"]["scope"] == (
        "single_model_registered_component_only"
    )
    assert census["cross_model_scope"][
        "universal_llm4rec_mechanism_claim_authorized"
    ] is False
    assert census["sources"]["d2_postblock"]["sha256"] == closeout[
        "deliverables"
    ]["d2_postblock"]["sha256"]


def test_decision_gate_catalog_exposes_independence_and_scope_boundaries():
    catalog = _decision_gate_catalog()
    h5 = next(
        row
        for row in catalog["hypothesis_supported"]
        if row["hypothesis_id"] == "H5"
    )
    assert h5["support_authorized_in_this_stage"] is False
    assert "second seed" in h5["boundary"]
    h1 = next(
        row
        for row in catalog["hypothesis_supported"]
        if row["hypothesis_id"] == "H1"
    )
    assert h1["required_supported_components"] == [
        "attention_query_key_routing",
        "history_routing",
    ]
    h4 = next(
        row
        for row in catalog["opportunity_primary"]
        if row["opportunity_id"] == "OP_H4_SURFACE_AWARE_GRADIENT_BUDGET"
    )
    assert h4["allowed_model_scope"] == ["q2_recranker_generalqwen"]


def test_attenuation_profile_reports_all_registered_adjacent_rows(tmp_path):
    rows = []
    row_index = 0
    for model_id in MODEL_IDS[2:]:
        for endpoint in ("target_margin", "ndcg@10"):
            for left, right in zip(SELECTED_NODES[:-1], SELECTED_NODES[1:]):
                row_index += 1
                missing = row_index == 3
                significant = row_index in {1, 2}
                mean = None if missing else (-0.2 if row_index == 1 else 0.2 if row_index == 2 else 0.0)
                rows.append(
                    {
                        "group": "adjacent_node",
                        "method_id": model_id,
                        "endpoint": endpoint,
                        "contrast_id": f"adjacent__{left}__to__{right}",
                        "missing": missing,
                        "mean": mean,
                        "ci95": None if missing else [float(mean) - 0.1, float(mean) + 0.1],
                        "bh_q": 1.0 if missing else 0.01 if significant else 0.5,
                        "bh_significant": False if missing else significant,
                        "expected_sign": None,
                        "registered_support": False,
                        "evidence_role": (
                            "missing_or_gate_stopped"
                            if missing
                            else "registered_confirmatory_branch_localization"
                        ),
                    }
                )
    metrics_path = tmp_path / "runs/selected/metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps(
            {
                "analysis_type": "transformer_deep_dive_d2_selected_branch_synthesis",
                "status": "completed",
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    import hashlib

    identity = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    closeout = {
        "deliverables": {
            "d2_selected_branches": {
                "path": "runs/selected/metrics.json",
                "sha256": identity,
            }
        }
    }
    profile = _attenuation_transition_profile(tmp_path, closeout)
    assert profile["row_count"] == 24
    assert profile["directional_description_counts"] == {
        "gate_stopped_or_missing": 1,
        "significant_amplification": 1,
        "significant_attenuation": 1,
        "unresolved_no_directional_claim": 21,
    }
    assert all(
        row["literal_hidden_state_sign_reversal_claimed"] is False
        and row["used_as_primary_component_attribution"] is False
        for row in profile["rows"]
    )


def test_layerwise_profile_retains_all_cells_and_derives_shape_taxonomy(tmp_path):
    all_layer = {"target_margin": [], "ndcg@10": []}
    adjacent = {"target_margin": [], "ndcg@10": []}
    for method_id in MODEL_IDS[2:]:
        gate_stopped = method_id == MODEL_IDS[3]
        for endpoint in ("target_margin", "ndcg@10"):
            for block in range(13, 28):
                all_layer[endpoint].append(
                    {
                        "method_id": method_id,
                        "block_zero_based": block,
                        "mean": None if gate_stopped else 0.1,
                        "ci95": [None, None] if gate_stopped else [0.05, 0.15],
                        "bh_q": 1.0 if gate_stopped else 0.5,
                        "missing_cell": gate_stopped,
                    }
                )
            for block in range(14, 28):
                negative = endpoint == "target_margin" and block in {14, 15}
                positive = endpoint == "target_margin" and block == 16
                mean = -0.2 if negative else 0.2 if positive else 0.0
                adjacent[endpoint].append(
                    {
                        "method_id": method_id,
                        "block_zero_based": block,
                        "contrast": f"block_{block}_minus_block_{block - 1}",
                        "mean": None if gate_stopped else mean,
                        "ci95": (
                            [None, None]
                            if gate_stopped
                            else [mean - 0.05, mean + 0.05]
                        ),
                        "bh_q": (
                            1.0
                            if gate_stopped
                            else 0.01
                            if negative or positive
                            else 0.5
                        ),
                        "missing_cell": gate_stopped,
                    }
                )
    metrics_path = tmp_path / "runs/postblock/metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps(
            {
                "analysis_type": "transformer_deep_dive_d2_postblock_synthesis",
                "status": "completed",
                "endpoints": ["target_margin", "ndcg@10"],
                "all_layer": all_layer,
                "adjacent_transition": adjacent,
            }
        ),
        encoding="utf-8",
    )
    identity = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    profile = _layerwise_attenuation_profile(
        tmp_path,
        {
            "deliverables": {
                "d2_postblock": {
                    "path": "runs/postblock/metrics.json",
                    "sha256": identity,
                }
            }
        },
    )
    assert profile["all_layer_row_count"] == 60
    assert profile["adjacent_layer_row_count"] == 56
    by_key = {
        (row["method_id"], row["endpoint"]): row
        for row in profile["shape_summary"]
    }
    q2_margin = by_key[(MODEL_IDS[2], "target_margin")]
    assert q2_margin["shape"] == "mixed_attenuation_and_amplification"
    assert q2_margin["significant_attenuation_steps"] == 2
    assert q2_margin["significant_amplification_steps"] == 1
    assert q2_margin["distributed_attenuation_pattern_established"] is True
    assert q2_margin["layer_scan_alone_authorizes_design"] is False
    assert q2_margin["registered_followup"] == (
        "combine_selected_transition_decomposition_with_cross_layer_shape"
    )
    assert q2_margin["design_implication_scope"] == "distributed_pattern_candidate"
    q2_ndcg = by_key[(MODEL_IDS[2], "ndcg@10")]
    assert q2_ndcg["shape"] == (
        "no_registered_significant_adjacent_change"
    )
    assert q2_ndcg["registered_followup"] == (
        "do_not_force_layer_local_cause_use_fixed_breadth_and_readout_checks"
    )
    q3_margin = by_key[(MODEL_IDS[3], "target_margin")]
    assert q3_margin["shape"] == (
        "gate_stopped_or_missing"
    )
    assert q3_margin["registered_followup"] == "stop_no_layer_based_inference"
    assert all(
        row["exact_layer_index_is_architecture_evidence"] is False
        and row["used_as_primary_component_attribution"] is False
        for row in profile["all_layer_rows"] + profile["adjacent_layer_rows"]
    )
